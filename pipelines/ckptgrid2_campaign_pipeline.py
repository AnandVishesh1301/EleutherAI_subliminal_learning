"""Lean auto-advancing driver for the bear checkpoint-grid stage-2 campaign.

One `run(cfg)` = one advance pass over:

    preflight              -> prefetch + snapshot + prereg + symlinks
                              + disk + datasets + generated YAMLs
    submit_p1              -> new SDF teachers + teacher evals
    submit_wave1           -> students that need no new teacher + evals
    dose_gate              -> label + hard per-rung ``ok`` (report first)
    submit_dose            -> A / A' / G students + evals
    hp_control_escalation  -> extra gseeds on a NEW queue (I6)
    submit_backfills       -> E / F additive evals
    submit_wave2           -> optional; needs ``seed_audit.json`` passed
    finalize               -> results + coverage + CAMPAIGN_DONE
    finalize_partial       -> coverage only; deadline / drain path

`CampaignDriverConfig` MUST be built with `spec=ckptgrid2_plan.SPEC`.
After preflight, later passes rebind `code_sha` from `state["code_sha"]`
so teacher and student queues share one pin (I8 late-anchor gap).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

from pipelines import ckptgrid2_results_pipeline as res
from subliminal_learning.campaign import ckptgrid2_config_gen as cgen
from subliminal_learning.campaign import ckptgrid2_expected as exp2
from subliminal_learning.campaign import ckptgrid2_jobs as jobs
from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign import ckptgrid2_selection as sel
from subliminal_learning.campaign import control_io, driver_io
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

# Staging planners live in S8. If that module is not importable yet,
# degrade to the cold-symlink check (assembly links cannot be verified).
try:
    from scripts.stage_ckptgrid2_assets import (
        DEFAULT_COLD_ROOT,
        DEFAULT_SDF_EXPERIMENT,
        PREREG_PATH,
        assembly_plan,
        dataset_checks,
        disk_gate_readings,
        sdf_docs_root_from_experiment,
    )
except ImportError:  # pragma: no cover - S8 landing race
    from scripts.stage_ckptgrid_assets import (  # type: ignore[no-redef]
        DEFAULT_COLD_ROOT,
        DEFAULT_SDF_EXPERIMENT,
        dataset_checks,
        disk_gate_readings,
        sdf_docs_root_from_experiment,
    )

    PREREG_PATH = Path("docs/PREREG_CKPT_XGRID2.md")
    assembly_plan = None  # type: ignore[assignment]


_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
_WAVE1_TRAIN: tuple[str, ...] = (
    plan.Q_T1_STUDENTS,
    plan.Q_T1_CONTROLS,
    plan.Q_T15_STUDENTS,
    plan.Q_T2_STUDENTS,
)
_P1_QUEUES: tuple[str, ...] = (plan.Q_P1_TEACHERS, plan.Q_P1_TEACHER_EVALS)


class CampaignGateError(RuntimeError):
    """A phase gate failed hard; do not advance."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass.

    Attributes:
        fired: Transitions that ran on this pass.
        waiting: Human reasons a gated transition did not fire.
        done: True after ``finalize`` or ``finalize_partial``.
    """

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class Ckptgrid2DriverOptions:
    """Optional roots and autonomy knobs the driver reads.

    Attributes:
        cold_root: Cold-1 artifact root the staging symlinks target.
        sdf_experiment: SDF teacher experiment YAML (corpus root).
        prereg_path: Preregistration markdown that must exist at P0.
        wave: ``1`` or ``2`` (wave-2 docs and queues).
        deadline_hours: Open-window budget for this ``run`` / ``watch``.
        drain_timeout_s: Max wait for in-flight cells after a deadline pause.
        max_failed_frac: Failed/total above this aborts; at or below is a hole.
        dataset_cache_root: Override for the dataset gate (tests).
        hf_cache_root: Override for the disk gate (tests).
    """

    cold_root: Path = DEFAULT_COLD_ROOT
    sdf_experiment: Path = DEFAULT_SDF_EXPERIMENT
    prereg_path: Path = PREREG_PATH
    wave: int = 1
    deadline_hours: float = plan.DEADLINE_HOURS_DEFAULT
    drain_timeout_s: int = plan.DRAIN_TIMEOUT_S
    max_failed_frac: float = plan.MAX_FAILED_FRAC
    dataset_cache_root: Path | None = None
    hf_cache_root: Path | None = None


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the ckptgrid2 `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


def _parse_utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, _ISO_FMT).replace(tzinfo=timezone.utc)


def _audit_passed(cfg: driver_io.CampaignDriverConfig) -> bool:
    path = cfg.logs_dir / plan.SEED_AUDIT_REPORT
    if not path.is_file():
        return False
    doc = json.loads(path.read_text())
    return doc.get("passed") is True


def _dataset_cache_root(
    cfg: driver_io.CampaignDriverConfig,
    options: Ckptgrid2DriverOptions,
) -> Path:
    if options.dataset_cache_root is not None:
        return Path(options.dataset_cache_root)
    return Path(cfg.output_root) / "datasets"


def _bundle(
    opts: Ckptgrid2DriverOptions, audit_passed: bool
) -> tuple[dict[str, dict], dict[str, dict], dict[str, list]]:
    """Train docs, verify docs, and queue → jobs for this wave."""
    train = cgen.all_train_configs(wave=opts.wave, audit_passed=audit_passed)
    verify = cgen.all_verify_configs(train)
    by_q = jobs.all_jobs(train, verify, wave=opts.wave, audit_passed=audit_passed)
    return train, verify, by_q


def _failed_jobkeys(cfg: driver_io.CampaignDriverConfig, name: str) -> list[str]:
    if not store.queue_exists(cfg.queue_root, name):
        return []
    return [
        job.jobkey
        for job in store.load_jobs(cfg.queue_root, name)
        if store.job_state(cfg.queue_root, name, job) == store.FAILED
    ]


def _record_hole(holes: list[dict], record: dict) -> None:
    if any(h.get("queue") == record.get("queue") for h in holes):
        return
    holes.append(record)


def _queue_ready(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    opts: Ckptgrid2DriverOptions,
    holes: list[dict],
) -> bool:
    """Whether ``name`` is terminal, with bounded failure holes.

    `requeue_failed_under_cap` raising means a job hit the attempt cap.
    A failed/total fraction at or below `opts.max_failed_frac` is recorded
    as a hole and treated as complete-with-holes; above it is a hard gate.

    Args:
        cfg: Driver config (queue root + auto-requeue cap).
        name: Queue name.
        opts: Autonomy knobs (`max_failed_frac`).
        holes: Accumulator for tolerated-failure records.

    Returns:
        True when the queue exists and every job is terminal (done or a
        tolerated hole).

    Raises:
        CampaignGateError: Failed fraction exceeds `max_failed_frac`.
    """
    if not store.queue_exists(cfg.queue_root, name):
        return False
    try:
        driver_io.requeue_failed_under_cap(cfg, name)
    except RuntimeError as err:
        st = store.queue_status(cfg.queue_root, name)
        total = max(int(st["total"]), 1)
        frac = int(st["failed"]) / total
        if frac > opts.max_failed_frac:
            raise CampaignGateError(f"{name}: {frac:.1%} failed > cap; {err}") from err
        _record_hole(
            holes,
            {
                "queue": name,
                "failed": int(st["failed"]),
                "total": int(st["total"]),
                "jobkeys": _failed_jobkeys(cfg, name),
            },
        )
        logger.warning("tolerating {} failed cells in {}", st["failed"], name)
    if not store.queue_complete(cfg.queue_root, name):
        return False
    st = store.queue_status(cfg.queue_root, name)
    failed = int(st["failed"])
    if failed:
        total = max(int(st["total"]), 1)
        frac = failed / total
        if frac > opts.max_failed_frac:
            raise CampaignGateError(f"{name}: {frac:.1%} failed > cap after requeue")
        _record_hole(
            holes,
            {
                "queue": name,
                "failed": failed,
                "total": int(st["total"]),
                "jobkeys": _failed_jobkeys(cfg, name),
            },
        )
        logger.warning("tolerating {} failed cells in {}", failed, name)
    return True


def _phase_done(
    cfg: driver_io.CampaignDriverConfig,
    names: list[str],
    opts: Ckptgrid2DriverOptions,
    waiting: list[str],
    holes: list[dict],
    *,
    check_disk: bool,
) -> bool:
    """True when every named queue is ready and, optionally, disk is complete.

    Disk completion uses `ckptgrid2_expected` (I9), never
    `campaign.expected.treatment_leaf_dirs` (that helper overcounts the
    3×3 seed grid).
    """
    ready = True
    for name in names:
        if not _queue_ready(cfg, name, opts, holes):
            waiting.append(f"{name}: not ready")
            ready = False
    if not ready:
        return False
    if check_disk:
        audit = _audit_passed(cfg)
        miss = exp2.missing_cells(cfg.output_root, wave=opts.wave, audit_passed=audit)
        if miss:
            waiting.append(f"{len(miss)} leaves without a manifest")
            return False
        miss_e = exp2.missing_evals(
            cfg.output_root,
            required_prompt_sets=plan.EVAL_PROMPT_SETS,
            wave=opts.wave,
            audit_passed=audit,
        )
        if miss_e:
            waiting.append("eval summaries incomplete")
            return False
    return True


def _open_window(state: dict, opts: Ckptgrid2DriverOptions) -> dict:
    """Append a deadline window when the last one is closed (or none exists).

    Called once per `run()` pass. A resume therefore gets a fresh budget;
    a closed window is never mutated back to open.

    Args:
        state: Mutable driver state (``deadline_windows`` list).
        opts: Supplies ``deadline_hours``.

    Returns:
        The open window dict ``{start, hours, ended_at, reason}``.
    """
    windows = state.setdefault("deadline_windows", [])
    if windows and windows[-1].get("ended_at") is None:
        return windows[-1]
    window = {
        "start": utc_now_iso(),
        "hours": float(opts.deadline_hours),
        "ended_at": None,
        "reason": None,
    }
    windows.append(window)
    return window


def _deadline_passed(state: dict, opts: Ckptgrid2DriverOptions) -> bool:
    windows = state.get("deadline_windows") or []
    if not windows:
        return False
    window = windows[-1]
    if window.get("ended_at") is not None:
        return False
    start = _parse_utc(str(window["start"]))
    hours = float(window.get("hours", opts.deadline_hours))
    now = datetime.now(timezone.utc)
    return now - start > timedelta(hours=hours)


def _inflight(cfg: driver_io.CampaignDriverConfig) -> int:
    """Sum of ``running`` across this campaign's submitted queues."""
    total = 0
    for name in cfg.queue_priorities:
        if not store.queue_exists(cfg.queue_root, name):
            continue
        total += int(store.queue_status(cfg.queue_root, name)["running"])
    return total


def _close_window(state: dict, *, reason: str) -> None:
    windows = state.get("deadline_windows") or []
    if not windows:
        return
    windows[-1]["ended_at"] = utc_now_iso()
    windows[-1]["reason"] = reason


def _all_submitted_queues(state: dict) -> list[str]:
    """Queue names this campaign has submitted (by fired transition)."""
    tr = state["transitions"]
    names: list[str] = []
    if "submit_p1" in tr:
        names.extend(_P1_QUEUES)
    if "submit_wave1" in tr:
        names.extend(_WAVE1_TRAIN)
        names.extend(
            [
                plan.Q_EVALS_T1,
                plan.Q_EVALS_T15,
                plan.Q_EVALS_T2,
            ]
        )
    if "submit_dose" in tr:
        names.extend([plan.Q_DOSE_STUDENTS, plan.Q_EVALS_DOSE])
    if "hp_control_escalation" in tr:
        names.extend([plan.Q_T1_CONTROLS_ESC, plan.Q_EVALS_T1_ESC])
    if "submit_backfills" in tr:
        names.extend([plan.Q_E_BACKFILL, plan.Q_F_BACKFILL])
    if "submit_wave2" in tr:
        names.extend(
            [
                plan.Q_W2_TEACHERS,
                plan.Q_W2_TEACHER_EVALS,
                plan.Q_W2_STUDENTS,
                plan.Q_W2_CONTROLS,
                plan.Q_W2_EVALS,
            ]
        )
    return names


def _existing_submitted(cfg: driver_io.CampaignDriverConfig, state: dict) -> list[str]:
    return [
        name
        for name in _all_submitted_queues(state)
        if store.queue_exists(cfg.queue_root, name)
    ]


# ---- P0 requirements --------------------------------------------------------


def _require_prefetch_manifest(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Load this campaign's `prefetch_manifest.json` and demand every ref.

    Raises:
        CampaignGateError: The file is missing or omits a designed ref.
    """
    path = cfg.logs_dir / plan.PREFETCH_MANIFEST
    if not path.is_file():
        raise CampaignGateError(
            f"P0 missing {plan.PREFETCH_MANIFEST} at {path} — run "
            "`python -m scripts.stage_ckptgrid2_assets` first"
        )
    doc = json.loads(path.read_text())
    records = doc.get("refs") or []
    have = set()
    for rec in records:
        repo = str(rec.get("repo") or rec.get("repo_id") or "")
        have.add((repo, str(rec.get("revision") or "")))
    missing = [
        (ref.repo_id, ref.revision)
        for ref in jobs.prefetch_refs()
        if (ref.repo_id, ref.revision) not in have
    ]
    if missing:
        raise CampaignGateError(
            f"P0 {plan.PREFETCH_MANIFEST} missing {len(missing)} refs "
            f"(first: {missing[0]})"
        )
    return doc


def _require_reference_snapshot(cfg: driver_io.CampaignDriverConfig) -> Path:
    path = cfg.logs_dir / plan.REFERENCE_SNAPSHOT_DIR
    if not path.is_dir():
        raise CampaignGateError(
            f"P0 missing {plan.REFERENCE_SNAPSHOT_DIR} at {path} — run "
            "`python -m scripts.stage_ckptgrid2_assets` first"
        )
    return path


def _require_prereg(prereg_path: Path) -> None:
    if not Path(prereg_path).is_file():
        raise CampaignGateError(f"P0 missing prereg {prereg_path}")


def _require_cold_and_assembly_symlinks(
    cfg: driver_io.CampaignDriverConfig,
    opts: Ckptgrid2DriverOptions,
) -> None:
    """Demand every planned cold + assembly symlink is already a noop."""
    try:
        for link, target in jobs.cold_symlink_plan(cfg.output_root, opts.cold_root):
            action = jobs.symlink_action(link, target)
            if action != "noop":
                raise CampaignGateError(
                    f"P0 cold symlink {link} → {target} is {action!r}, "
                    "expected noop — run "
                    "`python -m scripts.stage_ckptgrid2_assets`"
                )
    except FileExistsError as err:
        raise CampaignGateError(f"P0 cold symlink FAILED: {err}") from err
    if assembly_plan is None:
        return
    try:
        items = assembly_plan(cfg.output_root)
    except FileNotFoundError as err:
        raise CampaignGateError(f"P0 assembly symlink FAILED: {err}") from err
    try:
        for item in items:
            action = jobs.symlink_action(item.link, item.target)
            if action != "noop":
                raise CampaignGateError(
                    f"P0 assembly symlink {item.link} → {item.target} is "
                    f"{action!r}, expected noop — run "
                    "`python -m scripts.stage_ckptgrid2_assets`"
                )
    except FileExistsError as err:
        raise CampaignGateError(f"P0 assembly symlink FAILED: {err}") from err


def _assert_control_coverage(
    train: dict[str, dict],
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> None:
    """Require every `controls_of` key is staged or present in ``train``.

    Raises:
        CampaignGateError: A required control is neither staged nor in
            the generated control YAMLs.
    """
    required = {
        ctrl.key()
        for ctrl in pairs.controls_of(
            pairs.all_cells(wave=wave, audit_passed=audit_passed)
        )
    }
    staged = pairs.staged_controls()
    covered: set[tuple[str, str, str]] = set()
    prefix = f"{cgen.STUDENT_PREFIX}{cgen.CTRL_INFIX}"
    for fname, doc in train.items():
        if not fname.startswith(prefix):
            continue
        tag = jobs._gkd_tag_from_doc(doc)
        for animal in doc["animals"]:
            for student in jobs._students_of(doc):
                covered.add((str(animal), str(student), tag))
    missing = []
    for key in sorted(required):
        animal, student, tag, _gseed = key
        if key in staged:
            continue
        if (animal, student, tag) not in covered:
            missing.append(key)
    if missing:
        raise CampaignGateError(
            f"P0 missing control coverage ({len(missing)}): {missing[0]!r}"
        )


def _require_valid_run_set_id(run_set_id: str) -> None:
    if not is_valid_run_set_id(run_set_id):
        raise CampaignGateError(
            f"run_set_id {run_set_id!r} is malformed — every sweep would "
            "reject it at startup, so nothing is submitted. Expected "
            "{YYYYMMDD}_{purpose-slug}_{size} with hyphens (not "
            "underscores) inside the slug."
        )


def _require_spec_is_ours(cfg: driver_io.CampaignDriverConfig) -> None:
    if cfg.spec is not plan.SPEC and cfg.spec.run_set_id != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"ckptgrid2 driver bound to wrong spec run_set_id="
            f"{cfg.spec.run_set_id!r}; expected {plan.RUN_SET_ID!r}. "
            "Pass spec=ckptgrid2_plan.SPEC via driver_config()."
        )


def _cfg_with_pin(
    cfg: driver_io.CampaignDriverConfig, state: dict
) -> driver_io.CampaignDriverConfig:
    sha = state.get("code_sha")
    if not sha:
        return cfg
    return replace(cfg, code_sha=str(sha))


# ---- transitions ------------------------------------------------------------


def _preflight(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
) -> None:
    """P0: refs, snapshot, prereg, symlinks, disk, datasets, YAMLs."""
    manifest = _require_prefetch_manifest(cfg)
    _require_reference_snapshot(cfg)
    _require_prereg(opts.prereg_path)
    _require_cold_and_assembly_symlinks(cfg, opts)
    try:
        sdf_root = sdf_docs_root_from_experiment(opts.sdf_experiment)
        datasets = dataset_checks(
            SimpleNamespace(dataset_cache_root=_dataset_cache_root(cfg, opts)),
            sdf_docs_root=sdf_root,
            animals=plan.ANIMALS,
        )
    except (FileNotFoundError, OSError) as err:
        raise CampaignGateError(f"P0 dataset gate FAILED: {err}") from err
    hf_root = opts.hf_cache_root or cfg.output_root
    try:
        disk = disk_gate_readings(
            cold_root=opts.cold_root,
            ssd_roots=(cfg.output_root, hf_root),
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
        )
    except RuntimeError as err:
        raise CampaignGateError(f"P0 disk gate FAILED: {err}") from err
    audit = _audit_passed(cfg)
    train = cgen.all_train_configs(wave=opts.wave, audit_passed=audit)
    verify = cgen.all_verify_configs(train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    _assert_control_coverage(train, wave=opts.wave, audit_passed=audit)
    try:
        state["code_sha"] = code_pin_io.resolve_commit_sha(
            cfg.repo_root, cfg.code_sha or "HEAD"
        )
    except ValueError as err:
        raise CampaignGateError(f"P0 cannot resolve code pin: {err}") from err
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "n_refs": len(jobs.prefetch_refs()),
            "manifest": str(cfg.logs_dir / plan.PREFETCH_MANIFEST),
            "n_manifest_refs": len(manifest.get("refs") or []),
            "datasets": datasets,
            "disk_free_gib": disk,
            "prereg": str(opts.prereg_path),
            "n_train_yamls": len(train),
            "n_verify_yamls": len(verify),
            "code_sha": state["code_sha"],
            "wave": opts.wave,
        },
    )
    state["transitions"]["preflight"] = {
        "at": utc_now_iso(),
        "train_yamls": sorted(train),
        "verify_yamls": sorted(verify),
        "code_sha": state["code_sha"],
    }


def _submit_named(
    cfg: driver_io.CampaignDriverConfig,
    by_q: dict[str, list],
    name: str,
    *,
    depends_on: list[str] | None = None,
) -> int:
    job_list = list(by_q.get(name) or [])
    if not job_list:
        return 0
    driver_io.submit_queue(cfg, name, job_list, depends_on=depends_on)
    return len(job_list)


def _submit_p1(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
) -> None:
    _train, _verify, by_q = _bundle(opts, _audit_passed(cfg))
    n_t = _submit_named(cfg, by_q, plan.Q_P1_TEACHERS)
    n_e = _submit_named(
        cfg,
        by_q,
        plan.Q_P1_TEACHER_EVALS,
        depends_on=[plan.Q_P1_TEACHERS],
    )
    if n_t == 0 or n_e == 0:
        raise CampaignGateError(f"P1 built an empty queue (teachers={n_t} evals={n_e})")
    state["transitions"]["submit_p1"] = {
        "at": utc_now_iso(),
        "n_teacher_jobs": n_t,
        "n_teacher_eval_jobs": n_e,
    }


def _submit_wave1_students(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
) -> None:
    """Submit students that need no new teacher, plus per-train evals."""
    _train, _verify, by_q = _bundle(opts, _audit_passed(cfg))
    counts: dict[str, int] = {}
    for name in _WAVE1_TRAIN:
        counts[name] = _submit_named(cfg, by_q, name)
        if counts[name] == 0:
            raise CampaignGateError(f"wave-1 queue {name} built zero jobs")
    eval_deps: dict[str, list[str]] = {}
    for train_q in _WAVE1_TRAIN:
        eval_q = cgen.eval_queue(train_q)
        eval_deps.setdefault(eval_q, []).append(train_q)
    for eval_q, deps in eval_deps.items():
        counts[eval_q] = _submit_named(cfg, by_q, eval_q, depends_on=list(deps))
        if counts[eval_q] == 0:
            raise CampaignGateError(f"wave-1 queue {eval_q} built zero jobs")
    state["transitions"]["submit_wave1"] = {
        "at": utc_now_iso(),
        "counts": counts,
    }


def _run_dose_gate(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
    *,
    teacher_df=None,
) -> None:
    """Screen dose, write the report, persist, then raise on failure."""
    del opts
    frame = (
        teacher_df
        if teacher_df is not None
        else res.assemble_teacher_frame(cfg.output_root)
    )
    try:
        rows = sel.screen_dose(frame)
    except ValueError as err:
        raise CampaignGateError(f"dose gate: {err}") from err
    result = sel.dose_gate(rows)
    driver_io.write_logs_report(
        cfg,
        plan.DOSE_GATE_REPORT,
        {"at": utc_now_iso(), **sel.dose_report(rows, result)},
    )
    state["transitions"]["dose_gate"] = {
        "at": utc_now_iso(),
        "passed": result.passed,
        "ok_rungs": list(result.ok_rungs),
        "reasons": list(result.reasons),
    }
    driver_io.save_state(cfg, state)
    if not result.passed:
        raise CampaignGateError("dose gate FAILED — " + "; ".join(result.reasons))


def _submit_dose_students(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
) -> None:
    _train, _verify, by_q = _bundle(opts, _audit_passed(cfg))
    n_s = _submit_named(
        cfg,
        by_q,
        plan.Q_DOSE_STUDENTS,
        depends_on=[plan.Q_P1_TEACHER_EVALS],
    )
    n_e = _submit_named(
        cfg,
        by_q,
        plan.Q_EVALS_DOSE,
        depends_on=[plan.Q_DOSE_STUDENTS],
    )
    if n_s == 0 or n_e == 0:
        raise CampaignGateError(f"dose queues empty (students={n_s} evals={n_e})")
    state["transitions"]["submit_dose"] = {
        "at": utc_now_iso(),
        "n_student_jobs": n_s,
        "n_eval_jobs": n_e,
    }


def _escalate_hp_controls(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
    *,
    control_df=None,
) -> None:
    """Submit extra gseeds on `Q_T1_CONTROLS_ESC` (I6 — never append)."""
    frame = (
        control_df
        if control_df is not None
        else res.assemble_control_frame(cfg.output_root)
    )
    esc_pairs = sel.hp_control_escalation(frame)
    n_jobs = 0
    if esc_pairs:
        audit = _audit_passed(cfg)
        train = cgen.all_train_configs(wave=opts.wave, audit_passed=audit)
        verify = cgen.all_verify_configs(train)
        split = jobs.split_train_configs(train)
        w1 = {
            k: v
            for k, v in split.controls.items()
            if v.get("queue") == plan.Q_T1_CONTROLS
        }
        wanted = set(esc_pairs)
        ctrl_jobs = [
            job
            for job in jobs.control_jobs(
                w1,
                gseeds=plan.HP_CONTROL_ESCALATION_GSEEDS,
                queue_tag="t1-esc",
                wave=opts.wave,
                audit_passed=audit,
            )
            if _control_job_pair(job) in wanted
        ]
        _t_v, students_v, _e, _f = jobs.split_verify_configs(verify)
        vdocs = {
            cgen.verify_name(fname): students_v[cgen.verify_name(fname)]
            for fname in w1
            if cgen.verify_name(fname) in students_v
        }
        eval_jobs = [
            job
            for job in jobs.student_eval_jobs(
                vdocs,
                w1,
                queue_tag="t1-esc",
                wave=opts.wave,
                audit_passed=audit,
            )
            if _eval_job_student(job) in {p[0] for p in wanted}
        ]
        # Trains and evals go to SEPARATE queues: a queue has no internal
        # ordering, so an eval claimed before its control finished would
        # skip as NOT-TRAINED and exit 0, silently leaving the escalated
        # control unscored.
        if ctrl_jobs:
            driver_io.submit_queue(cfg, plan.Q_T1_CONTROLS_ESC, ctrl_jobs)
            n_jobs += len(ctrl_jobs)
        if eval_jobs:
            driver_io.submit_queue(
                cfg,
                plan.Q_EVALS_T1_ESC,
                eval_jobs,
                depends_on=[plan.Q_T1_CONTROLS_ESC],
            )
            n_jobs += len(eval_jobs)
    driver_io.write_logs_report(
        cfg,
        plan.ESCALATION_REPORT,
        {
            "at": utc_now_iso(),
            **sel.escalation_report(esc_pairs, frame),
            "n_jobs": n_jobs,
            "queues": [plan.Q_T1_CONTROLS_ESC, plan.Q_EVALS_T1_ESC],
        },
    )
    state["transitions"]["hp_control_escalation"] = {
        "at": utc_now_iso(),
        "pairs": [list(p) for p in esc_pairs],
        "n": len(esc_pairs),
        "n_jobs": n_jobs,
    }


def _control_job_pair(job) -> tuple[str, str]:
    parts = job.key.split("|")
    if len(parts) < 4:
        return ("", "")
    return (parts[2], parts[3])


def _eval_job_student(job) -> str:
    parts = job.key.split("|")
    if len(parts) < 4:
        return ""
    return parts[3]


def _submit_backfills(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
) -> None:
    _train, _verify, by_q = _bundle(opts, _audit_passed(cfg))
    n_e = _submit_named(cfg, by_q, plan.Q_E_BACKFILL)
    n_f = _submit_named(cfg, by_q, plan.Q_F_BACKFILL)
    if n_e == 0 or n_f == 0:
        raise CampaignGateError(f"backfill queues empty (e={n_e} f={n_f})")
    state["transitions"]["submit_backfills"] = {
        "at": utc_now_iso(),
        "n_e_jobs": n_e,
        "n_f_jobs": n_f,
    }


def _submit_wave2(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
) -> None:
    if not _audit_passed(cfg):
        raise CampaignGateError("wave 2 needs seed_audit.json: passed")
    train = cgen.all_train_configs(wave=2, audit_passed=True)
    verify = cgen.all_verify_configs(train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    by_q = jobs.wave2_jobs(train, verify, audit_passed=True)
    counts: dict[str, int] = {}
    for name in (
        plan.Q_W2_TEACHERS,
        plan.Q_W2_TEACHER_EVALS,
        plan.Q_W2_STUDENTS,
        plan.Q_W2_CONTROLS,
    ):
        counts[name] = _submit_named(cfg, by_q, name)
    eval_deps = [q for q in (plan.Q_W2_STUDENTS, plan.Q_W2_CONTROLS) if counts.get(q)]
    counts[plan.Q_W2_EVALS] = _submit_named(
        cfg,
        by_q,
        plan.Q_W2_EVALS,
        depends_on=eval_deps or None,
    )
    state["transitions"]["submit_wave2"] = {
        "at": utc_now_iso(),
        "counts": counts,
    }


def _coverage_lists(
    cfg: driver_io.CampaignDriverConfig,
    opts: Ckptgrid2DriverOptions,
) -> tuple[list, list]:
    audit = _audit_passed(cfg)
    return (
        exp2.missing_cells(cfg.output_root, wave=opts.wave, audit_passed=audit),
        exp2.missing_evals(
            cfg.output_root,
            required_prompt_sets=plan.EVAL_PROMPT_SETS,
            wave=opts.wave,
            audit_passed=audit,
        ),
    )


def _finalize(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
    *,
    partial: bool,
    holes: list[dict],
) -> None:
    written = res.build_results(
        cfg.output_root,
        cfg.campaign_subdir,
        dry_run=cfg.dry_run,
        partial=partial,
    )
    miss_cells, miss_evals = _coverage_lists(cfg, opts)
    res.write_coverage(
        cfg.output_root,
        cfg.campaign_subdir,
        missing_cells=miss_cells,
        missing_evals=miss_evals,
        holes=holes,
        deadline_windows=state.get("deadline_windows") or [],
        dry_run=cfg.dry_run,
    )
    key = "finalize_partial" if partial else "finalize"
    if not partial and not cfg.dry_run:
        marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
        atomic_write_text(
            marker,
            f"campaign {plan.RUN_SET_ID} finished {utc_now_iso()}\n",
        )
        logger.info("Campaign {} COMPLETE — marker at {}", plan.RUN_SET_ID, marker)
    elif partial:
        logger.warning(
            "Campaign {} partial finalize — coverage written, no {}",
            plan.RUN_SET_ID,
            plan.CAMPAIGN_DONE_FILE,
        )
    state["transitions"][key] = {
        "at": utc_now_iso(),
        "partial": partial,
        "results": [str(p) for p in written],
        "n_holes": len(holes),
    }


def _deadline_stop(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ckptgrid2DriverOptions,
    holes: list[dict],
) -> None:
    """Pause only a campaign-owned pool (I13), drain, partial-finalize."""
    if state.get("pool_scope") == "campaign" and not cfg.dry_run:
        stop = store.stop_path(cfg.queue_root, store.this_host())
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.touch()
        snap = control_io.snapshot(
            queue_root=cfg.queue_root,
            logs_dir=cfg.logs_dir,
            spec=cfg.spec,
        )
        control_io.append_event(
            cfg.logs_dir,
            cfg.spec,
            action=control_io.PAUSE,
            reason="deadline",
            snap=snap,
        )
        waited = 0
        while _inflight(cfg) and waited < opts.drain_timeout_s:
            time.sleep(30)
            waited += 30
    else:
        logger.warning("deadline reached but the pool is shared — not pausing it")
    _finalize(cfg, state, opts, partial=True, holes=holes)
    _close_window(state, reason="deadline")


def run(
    cfg: driver_io.CampaignDriverConfig,
    options: Ckptgrid2DriverOptions | None = None,
) -> AdvanceResult:
    """One advance pass over the ckptgrid2 DAG.

    Args:
        cfg: Driver config bound to `ckptgrid2_plan.SPEC`.
        options: Autonomy knobs; defaults when omitted.

    Returns:
        Fired transitions, wait reasons, and whether a finalize ran.

    Raises:
        CampaignGateError: Malformed id, wrong spec, or a hard gate.
    """
    _require_valid_run_set_id(cfg.run_set_id)
    _require_spec_is_ours(cfg)
    opts = options or Ckptgrid2DriverOptions()
    state = driver_io.load_state(cfg)
    if state.get("code_sha"):
        cfg = _cfg_with_pin(cfg, state)
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []
    holes: list[dict] = list(state.get("holes") or [])
    _open_window(state, opts)
    if pin_note:
        fired.append(pin_note)

    if "preflight" not in transitions:
        _preflight(cfg, state, opts)
        cfg = _cfg_with_pin(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("preflight")

    if state.get("code_sha"):
        cfg = _cfg_with_pin(cfg, state)

    if "preflight" in transitions and "submit_p1" not in transitions:
        _submit_p1(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_p1")

    if "submit_p1" in transitions and "submit_wave1" not in transitions:
        _submit_wave1_students(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_wave1")

    if (
        "submit_p1" in transitions
        and "dose_gate" not in transitions
        and _phase_done(cfg, list(_P1_QUEUES), opts, waiting, holes, check_disk=False)
    ):
        _run_dose_gate(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("dose_gate")

    if "dose_gate" in transitions and not transitions["dose_gate"].get("passed", False):
        raise CampaignGateError(
            "dose gate FAILED — see "
            f"{cfg.logs_dir / plan.DOSE_GATE_REPORT}; not submitting dose"
        )

    if "dose_gate" in transitions and "submit_dose" not in transitions:
        _submit_dose_students(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_dose")

    if (
        "submit_wave1" in transitions
        and "hp_control_escalation" not in transitions
        and _phase_done(
            cfg,
            [plan.Q_T1_CONTROLS],
            opts,
            waiting,
            holes,
            check_disk=False,
        )
    ):
        _escalate_hp_controls(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("hp_control_escalation")

    if "preflight" in transitions and "submit_backfills" not in transitions:
        _submit_backfills(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_backfills")

    if opts.wave == 2 and "submit_wave2" not in transitions:
        _submit_wave2(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_wave2")

    state["holes"] = holes
    if "finalize" not in transitions:
        submitted = _existing_submitted(cfg, state)
        if submitted and _phase_done(
            cfg, submitted, opts, waiting, holes, check_disk=True
        ):
            state["holes"] = holes
            _finalize(cfg, state, opts, partial=False, holes=holes)
            driver_io.save_state(cfg, state)
            fired.append("finalize")
        elif _deadline_passed(state, opts) and "finalize_partial" not in transitions:
            state["holes"] = holes
            _deadline_stop(cfg, state, opts, holes)
            driver_io.save_state(cfg, state)
            fired.append("finalize_partial")

    state["holes"] = holes
    driver_io.save_state(cfg, state)
    tr = state["transitions"]
    return AdvanceResult(
        tuple(fired),
        tuple(waiting),
        done=("finalize" in tr) or ("finalize_partial" in tr),
    )


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: transitions, queues, holes, deadline windows."""
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in cfg.queue_priorities
        if store.queue_exists(cfg.queue_root, name)
    }
    return {
        "transitions": sorted(state["transitions"]),
        "queues": queues,
        "holes": list(state.get("holes") or []),
        "deadline_windows": list(state.get("deadline_windows") or []),
        "pool_scope": state.get("pool_scope"),
        "code_sha": state.get("code_sha"),
        "done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }
