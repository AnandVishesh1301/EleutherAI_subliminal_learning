"""Auto-advancing driver for the xfam wave-2 campaign (`xfam2`).

One `run(cfg)` = one advance pass over:

    preflight        -> prereg, staging report, disk, measured footprints, teacher YAMLs
    submit_teachers  -> per teacher label: its SDF train queue and eval queue
    submit_controls  -> every unit's control queue (a control needs no teacher)
    gate_{label}     -> once that label's eval queue is complete: its admission decision
                        merged into `teacher_gate_report.json` FIRST, then every unit it
                        emits submitted over its admitted ftseeds (void: nothing)
    rehome           -> claim-free queues moved to `xfam2_plan.queue_node`'s host
    finalize_core    -> results + CAMPAIGN_DONE once every core queue is complete
    results_tail     -> results again once the tail queues drain
    deadline_pause   -> when ACTIVE time reaches the deadline with work left: a stop
                        sentinel and a control-log pause event per xfam2 host, in the
                        xfam2 queue root only; no results are written

A persisted gate is read back, never recomputed: the admitted ftseeds set the jobkeys
of already-submitted queues. A job over the requeue cap is recorded as a hole and
never stops the driver; its queue stays incomplete until it is repaired. Active time =
wall time since the first submission minus every interval in which all xfam2 hosts
were paused (from the control log; persisted as ``deadline_windows``).

Pause / resume (run each command ON the node, from its campaign pin
``<clone>/.code_pins/<sha>``; clones: soar ``/home/vishesh/eleutherAI/
EleutherAI_subliminal_learning``, goncalo ``/home/angela/eleutherAI/
EleutherAI_subliminal_learning``; non-soar hosts via ``ssh <alias> 'cd <pin> && ...'``):

    drained pause  uv run python -m scripts.campaign_control pause --campaign xfam2 \\
                       --paths-config configs/paths/paths_xfam2.yaml --reason "..."
                   (writes stop.<host> in the xfam2 queue root; in-flight leaves finish)
    resume         same command with ``resume``; the tmux ``workers`` restart loop
                   re-spawns the pool within 60 s
    status         same command with ``status``
    hard stop      Ctrl-C the tmux ``xfam2:workers`` window; that host's orphaned claims
                   are stolen (dead pid) on its next pool start, and each killed leaf
                   resumes from its last ``checkpoint-*`` (<= 125 steps lost)

The tmux ``xfam2:driver`` window keeps running through a pause: it may submit queues,
requeue failures and re-home claim-free queues, but finalizes nothing until every core
queue is complete.

`CampaignDriverConfig` MUST be built with `driver_config` (xfam2 spec, shared YAML dir).
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import control_io, driver_io
from subliminal_learning.campaign import xfam2_config_gen as cgen
from subliminal_learning.campaign import xfam2_jobs as jobs
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.xfam_selection import (
    TeacherAdmission,
    TeacherGateDecision,
)
from subliminal_learning.campaign.xsize_selection import admit_teacher
from subliminal_learning.evaluation.sdf_results_io import (
    load_sdf_capability_results,
    load_sdf_preference_results,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import is_valid_run_set_id, run_set_subdir
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

_SWEEP_CONFIG_FLAG = "--sweep-config "


class CampaignGateError(RuntimeError):
    """A pre-registered gate refused to pass."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class Xfam2DriverOptions:
    """Roots and knobs the driver reads (overridable in tests).

    Attributes:
        prereg_path: Pre-registration document that must exist.
        campaign_root: Root the disk gate reads.
        expected_queue_root: The only queue root the driver may act on; ``None``
            skips the check (tests).
        deadline_hours: Wall budget after the first submission.
        now: Clock (tests).
    """

    prereg_path: Path = Path(plan.PREREG_PATH)
    campaign_root: Path = plan.CAMPAIGN_ROOT
    expected_queue_root: Path | None = Path(plan.paths_config_doc()["queue_root"])
    deadline_hours: float = plan.DEADLINE_HOURS
    now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))


def shared_config_dir(output_root: Path) -> Path:
    """The run set's shared generated-YAML dir (every node reads it)."""
    run_set_root = Path(output_root) / run_set_subdir(
        GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID
    )
    return run_set_root / plan.SHARED_GENERATED_DIRNAME


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the xfam2 spec and its shared YAML dir."""
    kwargs.setdefault(
        "generated_dir_override", shared_config_dir(kwargs["output_root"])
    )
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


# ---- pins, submission ----------------------------------------------------------------


def node_pin_defaults(cfg: driver_io.CampaignDriverConfig, host: str) -> dict:
    """jobq defaults binding a queue to `host`'s worktree of the campaign pin."""
    if cfg.dry_run:
        return {"cwd": str(plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME)}
    pin = code_pin_io.resolve_campaign_pin(
        cfg.run_set_root,
        repo_root=cfg.repo_root,
        requested_sha=cfg.code_sha,
        allow_advance=False,
    )
    if host != store.this_host():
        root = plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME / pin.sha
        pin = code_pin_io.CodePin(sha=pin.sha, root=root)
    return code_pin_io.pin_job_defaults(pin)


def _require_configs(
    cfg: driver_io.CampaignDriverConfig, name: str, queue_jobs: Sequence[Job]
) -> None:
    """Refuse a queue whose jobs read a YAML another node could not see."""
    for job in queue_jobs:
        path = Path(job.cmd.split(_SWEEP_CONFIG_FLAG, 1)[1].split(" ", 1)[0])
        if not path.is_absolute() or not (cfg.dry_run or path.is_file()):
            raise CampaignGateError(
                f"{name}: job {job.key} reads {path}, not an existing absolute YAML"
            )


def _submit(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    queue_jobs: Sequence[Job],
    *,
    depends_on: Sequence[str] = (),
    strict: bool = False,
) -> None:
    if not queue_jobs:
        raise CampaignGateError(f"queue {name} built zero jobs")
    _require_configs(cfg, name, queue_jobs)
    host = plan.queue_node(name)
    driver_io.submit_queue(
        cfg,
        name,
        list(queue_jobs),
        depends_on=list(depends_on) or None,
        node=host,
        defaults=node_pin_defaults(cfg, host),
        strict_deps=strict,
    )


def _write(cfg: driver_io.CampaignDriverConfig, docs: Mapping[str, dict]) -> None:
    try:
        driver_io.write_generated_configs(cfg, dict(docs))
    except RuntimeError as err:
        raise CampaignGateError(str(err)) from err


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in cfg.queue_priorities if store.queue_exists(cfg.queue_root, n)]


def _queue_done(
    cfg: driver_io.CampaignDriverConfig, name: str, waiting: list[str]
) -> bool:
    if cfg.dry_run:
        return True
    ok, msg = driver_io.queue_finished(cfg, name)
    if not ok:
        waiting.append(msg)
    return ok


# ---- preflight -------------------------------------------------------------------------


def _require_staging(cfg: driver_io.CampaignDriverConfig) -> dict:
    path = cfg.logs_dir / plan.STAGING_REPORT
    if not path.is_file():
        raise CampaignGateError(
            f"preflight: missing {path} -- run `uv run python -m "
            "scripts.stage_xfam2_assets` first"
        )
    doc = json.loads(path.read_text())
    if doc.get("passed") is not True:
        raise CampaignGateError(f"preflight: {plan.STAGING_REPORT} did not pass")
    if doc.get("run_set_id") != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"preflight: staging report is for run set {doc.get('run_set_id')!r}"
        )
    return doc


def check_footprints(doc: Mapping[str, object]) -> list[str]:
    """Measured shapes whose smoke peak exceeds the plan's stamp.

    `doc` is ``xfam2_footprints.json``: ``{"students": {"<teacher>--<student>":
    {"peak_mib": int}}, "teachers": {"<label>": {"peak_mib": int}}}``. A student
    stamp already carries the steady-state margin on its student side (the teacher's
    weights do not grow), so a student peak is compared as measured; a teacher peak
    gets the margin.
    """
    bad: list[str] = []
    for key, entry in (doc.get("students") or {}).items():  # type: ignore[union-attr]
        teacher, student = str(key).split("--", 1)
        need = math.ceil(float(entry["peak_mib"]))
        have = plan.cell_peak_mib(teacher, student)
        if need > have:
            bad.append(f"student {key}: needs {need} MiB, plan stamps {have}")
    for label, entry in (doc.get("teachers") or {}).items():  # type: ignore[union-attr]
        need = math.ceil(float(entry["peak_mib"]) * plan.STEADY_STATE_MARGIN)
        have = plan.TEACHER_FOOTPRINT[str(label)].peak_mib
        if need > have:
            bad.append(f"teacher {label}: needs {need} MiB, plan stamps {have}")
    return bad


def _require_footprints(cfg: driver_io.CampaignDriverConfig) -> dict:
    path = cfg.logs_dir / plan.FOOTPRINTS_FILE
    if not path.is_file():
        raise CampaignGateError(f"preflight: missing smoke footprints {path}")
    doc = json.loads(path.read_text())
    try:
        bad = check_footprints(doc)
    except (KeyError, ValueError) as err:
        raise CampaignGateError(f"preflight: unreadable footprints: {err}") from err
    if bad:
        raise CampaignGateError(f"preflight: footprints under-stamped: {bad}")
    return doc


def _preflight(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Xfam2DriverOptions
) -> None:
    if not Path(opts.prereg_path).is_file():
        if not cfg.dry_run:
            raise CampaignGateError(f"preflight: missing prereg {opts.prereg_path}")
        logger.warning("[dry-run] prereg {} not written yet", opts.prereg_path)
    free_gib: float | None = None
    if not cfg.dry_run:
        _require_staging(cfg)
        _require_footprints(cfg)
        free_gib = shutil.disk_usage(opts.campaign_root).free / 2**30
        if free_gib < plan.MIN_DISK_FREE_GIB:
            raise CampaignGateError(
                f"preflight: {opts.campaign_root} has {free_gib:.0f} GiB free "
                f"< {plan.MIN_DISK_FREE_GIB}"
            )
    try:
        cgen.assert_teacher_max_length()
    except ValueError as err:
        raise CampaignGateError(f"preflight: {err}") from err
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "passed": True,
            "run_set_id": plan.RUN_SET_ID,
            "prereg": str(opts.prereg_path),
            "disk_free_gib": free_gib,
            "n_units": len(plan.units()),
            "n_teacher_labels": len(plan.teacher_labels()),
            "config_dir": str(cfg.generated_dir),
        },
    )
    state["transitions"]["preflight"] = {"at": utc_now_iso()}


# ---- teachers + controls -----------------------------------------------------------------


def _submit_teachers(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Xfam2DriverOptions
) -> None:
    for label in plan.teacher_labels():
        train = cgen.teacher_config(label)
        docs = cgen.with_verify(train, cfg.generated_dir, plan.q_teacher_evals(label))
        _write(cfg, docs)
        _submit(
            cfg, plan.q_teachers(label), jobs.teacher_jobs(label, cfg.generated_dir)
        )
        _submit(
            cfg,
            plan.q_teacher_evals(label),
            jobs.teacher_eval_jobs(label, cfg.generated_dir),
            depends_on=[plan.q_teachers(label)],
            strict=True,
        )
    state.setdefault("deadline_started_at", opts.now().isoformat())
    state["transitions"]["submit_teachers"] = {
        "at": utc_now_iso(),
        "labels": list(plan.teacher_labels()),
    }


def _submit_controls(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    n = 0
    for unit in plan.units():
        docs = cgen.with_verify(
            cgen.control_config(unit),
            cfg.generated_dir,
            plan.q_unit(unit, plan.CLS_EVAL),
        )
        _write(cfg, docs)
        control_jobs = jobs.unit_control_jobs(unit, cfg.generated_dir)
        _submit(cfg, plan.q_unit(unit, plan.CLS_CTRL), control_jobs)
        n += len(control_jobs)
    state["transitions"]["submit_controls"] = {"at": utc_now_iso(), "n_jobs": n}


# ---- per-label teacher gate ----------------------------------------------------------


def _per_seed_items(raw: object) -> list[tuple[int, float]]:
    if not isinstance(raw, Mapping):
        return []
    out = [
        (int(seed), float(value))
        for seed, value in raw.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return sorted(out)


def teacher_admission_rows(
    cfg: driver_io.CampaignDriverConfig, label: str
) -> list[TeacherAdmission]:
    """Score every trained ftseed of one teacher label against the admission rule.

    Raises:
        ValueError: If a summary lacks ``per_seed_p_target`` (a multi-seed teacher
            would silently collapse into one leaf).
    """
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    pref = load_sdf_preference_results(
        cfg.output_root, subdir, animals=list(plan.ANIMALS)
    )
    if pref.empty:
        return []
    rows = pref[
        (pref["set"].astype(str) == plan.PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == plan.PRIMARY_SCORING)
        & (pref["model"].astype(str) == label)
        & (pref["param_tag"].astype(str) == plan.TEACHER_PARAM_TAG)
    ]
    cap = load_sdf_capability_results(
        cfg.output_root, subdir, animals=list(plan.ANIMALS)
    )
    ratio = None
    if not cap.empty and "perplexity_ratio" in cap.columns:
        match = cap[
            (cap["model"].astype(str) == label)
            & (cap["param_tag"].astype(str) == plan.TEACHER_PARAM_TAG)
        ]
        values = pd.to_numeric(match["perplexity_ratio"], errors="coerce").dropna()
        if len(values) > 1:
            raise ValueError(f"duplicate teacher capability rows for {label}")
        ratio = float(values.iloc[0]) if len(values) else None
    out: list[TeacherAdmission] = []
    for _, row in rows.iterrows():
        seeds = _per_seed_items(row.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(f"teacher summary for {label} lacks per_seed_p_target")
        control = row.get("base_p_target")
        control_mean = None if control is None or pd.isna(control) else float(control)
        for ftseed, p_target in seeds:
            admitted, reasons = admit_teacher(
                p_target=p_target,
                control_mean=control_mean,
                control_sd=None,
                ppl_ratio=ratio,
                sd_multiple=plan.ADMISSION_SD_MULTIPLE,
                abs_min=plan.ADMISSION_ABS_MIN,
                ppl_max=plan.TEACHER_PPL_RATIO_MAX,
            )
            out.append(
                TeacherAdmission(
                    label=label,
                    tag=plan.TEACHER_PARAM_TAG,
                    ftseed=ftseed,
                    admitted=admitted,
                    ppl_ratio=ratio,
                    p_target=p_target,
                    control_mean=control_mean,
                    control_sd=None,
                    reasons=reasons,
                )
            )
    return out


def label_decision(label: str, rows: Sequence[TeacherAdmission]) -> TeacherGateDecision:
    """Whether a teacher label emits: >= `MIN_ADMITTED_FTSEEDS` admitted ftseeds."""
    admitted = tuple(
        sorted(
            r.ftseed
            for r in rows
            if r.label == label and r.admitted and r.ftseed in plan.TEACHER_FTSEEDS
        )
    )
    usable = len(admitted) >= plan.MIN_ADMITTED_FTSEEDS
    if usable:
        reason = "admitted"
    elif not rows:
        reason = "no_teacher_rows"
    else:
        reason = f"below_quorum({len(admitted)}<{plan.MIN_ADMITTED_FTSEEDS})"
    return TeacherGateDecision(
        label=label,
        role="emitter",
        admitted_ftseeds=admitted,
        usable=usable,
        reason=reason,
    )


def _decision(cfg: driver_io.CampaignDriverConfig, label: str) -> TeacherGateDecision:
    """The persisted decision for `label`, computing and writing it first if absent."""
    path = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    doc = json.loads(path.read_text()) if path.is_file() else {}
    entry = (doc.get("decisions") or {}).get(label)
    if entry is not None:
        return TeacherGateDecision(
            label=label,
            role=str(entry.get("role", "emitter")),
            admitted_ftseeds=tuple(int(x) for x in entry.get("admitted_ftseeds") or ()),
            usable=bool(entry.get("usable")),
            reason=str(entry.get("reason", "")),
        )
    if cfg.dry_run:
        return TeacherGateDecision(
            label, "emitter", plan.TEACHER_FTSEEDS, True, "dry_run"
        )
    rows = teacher_admission_rows(cfg, label)
    decision = label_decision(label, rows)
    doc.setdefault("decisions", {})[label] = decision.to_dict()
    doc.setdefault("rows", {})[label] = [r.to_dict() for r in rows]
    doc["run_set_id"] = plan.RUN_SET_ID
    doc["updated_at"] = utc_now_iso()
    driver_io.write_logs_report(cfg, plan.TEACHER_GATE_REPORT, doc)
    return decision


def _submit_units_of(
    cfg: driver_io.CampaignDriverConfig, label: str, decision: TeacherGateDecision
) -> tuple[list[str], list[str]]:
    """``(submitted unit slugs, void unit slugs)`` for every unit `label` emits."""
    submitted: list[str] = []
    void: list[str] = []
    for unit in plan.units():
        if unit.cell.teacher != label:
            continue
        ftseeds = jobs.unit_ftseeds(unit, decision.admitted_ftseeds)
        if not decision.usable or not ftseeds:
            void.append(unit.slug)
            continue
        q_eval = plan.q_unit(unit, plan.CLS_EVAL)
        _write(
            cfg,
            cgen.with_verify(
                cgen.treatment_config(unit, ftseeds), cfg.generated_dir, q_eval
            ),
        )
        q_train = plan.q_unit(unit, plan.CLS_TRAIN)
        _submit(cfg, q_train, jobs.unit_train_jobs(unit, ftseeds, cfg.generated_dir))
        deps = [q_train, plan.q_unit(unit, plan.CLS_CTRL)]
        if unit.tier == plan.TIER_G2:
            deps.append(
                plan.q_unit(plan.Unit(unit.cell, plan.TIER_CORE), plan.CLS_EVAL)
            )
        _submit(
            cfg,
            q_eval,
            jobs.unit_eval_jobs(unit, cfg.generated_dir),
            depends_on=deps,
            strict=True,
        )
        submitted.append(unit.slug)
    return submitted, void


def _gate_labels(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    waiting: list[str],
    fired: list[str],
) -> None:
    for label in plan.teacher_labels():
        key = f"gate_{label}"
        if key in state["transitions"]:
            continue
        if not _queue_done(cfg, plan.q_teacher_evals(label), waiting):
            continue
        try:
            decision = _decision(cfg, label)
        except (ValueError, FileNotFoundError, OSError) as err:
            raise CampaignGateError(f"teacher gate {label}: {err}") from err
        submitted, void = _submit_units_of(cfg, label, decision)
        if not decision.usable:
            logger.warning(
                "teacher {} is void ({}): {} units", label, decision.reason, len(void)
            )
        state["transitions"][key] = {
            "at": utc_now_iso(),
            "usable": decision.usable,
            "admitted_ftseeds": list(decision.admitted_ftseeds),
            "submitted": submitted,
            "void": void,
        }
        driver_io.save_state(cfg, state)
        fired.append(key)


# ---- maintenance ----------------------------------------------------------------------


def _requeue(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    """Requeue failures under the cap; a job over it is a hole, not a driver stop."""
    holes = state.setdefault("holes", {})
    for name in _submitted(cfg):
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if name not in holes:
                logger.error("HOLE {}: {}", name, err)
                holes[name] = {"at": utc_now_iso(), "error": str(err)}


def _rehome(cfg: driver_io.CampaignDriverConfig, state: dict) -> str | None:
    """Move claim-free submitted queues whose plan host changed; defaults follow."""
    moves = []
    for name in _submitted(cfg):
        want = plan.queue_node(name)
        have = store.read_meta(cfg.queue_root, name).node
        if have != want:
            moves.append({"queue": name, "from": have, "to": want})
    if not moves:
        return None
    if cfg.dry_run:
        return f"would rehome {len(moves)} queues"
    moved = []
    for move in moves:
        try:
            store.update_queue_node(cfg.queue_root, move["queue"], move["to"])
        except RuntimeError as err:
            logger.warning("rehome {} deferred: {}", move["queue"], err)
            continue
        store.update_queue_defaults(
            cfg.queue_root, move["queue"], node_pin_defaults(cfg, move["to"])
        )
        moved.append(move)
    if not moved:
        return None
    state.setdefault("rehomes", []).append({"at": utc_now_iso(), "moves": moved})
    return f"rehome_queues({len(moved)})"


# ---- completion -----------------------------------------------------------------------


def expected_queues(state: dict, *, tail: bool) -> tuple[list[str], list[str]]:
    """``(queues that must finish, labels not yet gated)`` for the core or the tail.

    Only queues that are submitted or will be once their gate fires: the train/eval
    queues of a void unit are never expected.
    """
    names: list[str] = []
    if not tail:
        for label in plan.teacher_labels():
            names += [plan.q_teachers(label), plan.q_teacher_evals(label)]
    ungated: list[str] = []
    transitions = state.get("transitions", {})
    for unit in plan.units():
        if unit.is_tail != tail:
            continue
        names.append(plan.q_unit(unit, plan.CLS_CTRL))
        gate = transitions.get(f"gate_{unit.cell.teacher}")
        if gate is None:
            if unit.cell.teacher not in ungated:
                ungated.append(unit.cell.teacher)
            continue
        if unit.slug in (gate.get("submitted") or ()):
            names += [
                plan.q_unit(unit, plan.CLS_TRAIN),
                plan.q_unit(unit, plan.CLS_EVAL),
            ]
    return names, ungated


def _ready(
    cfg: driver_io.CampaignDriverConfig, state: dict, waiting: list[str], *, tail: bool
) -> bool:
    names, ungated = expected_queues(state, tail=tail)
    if ungated and not cfg.dry_run:
        waiting.append(f"{'tail' if tail else 'core'}: teacher gates pending {ungated}")
        return False
    pending = [n for n in names if not _queue_done(cfg, n, [])]
    if pending:
        waiting.append(
            f"{'tail' if tail else 'core'}: {len(pending)} queues unfinished"
        )
        return False
    return True


def _build_results(cfg: driver_io.CampaignDriverConfig, *, dry_run: bool) -> None:
    if dry_run and importlib.util.find_spec("pipelines.xfam2_results_pipeline") is None:
        logger.warning("[dry-run] pipelines.xfam2_results_pipeline not present; skip")
        return
    from pipelines.xfam2_results_pipeline import build_results

    build_results(cfg, dry_run=dry_run)


def _finalize_core(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _build_results(cfg, dry_run=cfg.dry_run)
    if not cfg.dry_run:
        atomic_write_text(
            cfg.logs_dir / plan.CAMPAIGN_DONE_FILE,
            f"campaign {plan.RUN_SET_ID} core finished {utc_now_iso()}\n",
        )
    state["transitions"]["finalize_core"] = {"at": utc_now_iso()}


# ---- deadline (active time) -------------------------------------------------------------

_TOGGLES = (control_io.PAUSE, control_io.RESUME)


def _parse_time(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _host_toggles(
    events: Sequence[Mapping[str, object]],
) -> list[tuple[datetime, str, str]]:
    """``(at, host, action)`` of every well-formed pause/resume event, by time."""
    out: list[tuple[datetime, str, str]] = []
    for event in events:
        action, host = event.get("action"), event.get("host")
        if action not in _TOGGLES:
            continue
        try:
            at = _parse_time(event["at"])
        except (KeyError, ValueError):
            logger.warning("control-log event without a valid time: {}", event)
            continue
        if not host:
            logger.warning("control-log event without a host: {}", event)
            continue
        out.append((at, str(host), str(action)))
    return sorted(out, key=lambda row: row[0])


def paused_intervals(
    events: Sequence[Mapping[str, object]],
    host_history: Sequence[Mapping[str, object]],
    start: datetime,
    now: datetime,
) -> list[tuple[datetime, datetime]]:
    """Intervals within ``[start, now]`` during which EVERY campaign host was paused.

    Args:
        events: Control-log records (``at``, ``host``, ``action``).
        host_history: ``[{"at": iso, "hosts": [...]}]``, ascending; the entry in force
            at ``t`` is the last one at or before ``t`` (the first before its own time).
        start: First submission.
        now: End of the accounting window.

    Returns:
        Disjoint, ascending ``(begin, end)`` pairs.
    """
    toggles = _host_toggles(events)
    history = [(_parse_time(h["at"]), set(h["hosts"])) for h in host_history]
    cuts = {start, now}
    cuts |= {t for t, _, _ in toggles if start < t < now}
    cuts |= {t for t, _ in history if start < t < now}
    ordered = sorted(cuts)
    out: list[tuple[datetime, datetime]] = []
    for begin, end in zip(ordered, ordered[1:], strict=False):
        paused: set[str] = set()
        for t, host, action in toggles:
            if t > begin:
                break
            if action == control_io.PAUSE:
                paused.add(host)
            else:
                paused.discard(host)
        hosts = next(
            (h for t, h in reversed(history) if t <= begin),
            history[0][1] if history else set(),
        )
        if not hosts or not hosts <= paused:
            continue
        if out and out[-1][1] == begin:
            out[-1] = (out[-1][0], end)
        else:
            out.append((begin, end))
    return out


def _record_hosts(
    cfg: driver_io.CampaignDriverConfig, state: dict, now: datetime
) -> None:
    history = state.setdefault("deadline_hosts", [])
    hosts = campaign_hosts(cfg)
    if not history or sorted(history[-1]["hosts"]) != hosts:
        history.append({"at": now.isoformat(), "hosts": hosts})


def active_hours(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Xfam2DriverOptions
) -> float | None:
    """Hours since the first submission minus every all-hosts-paused interval.

    Rewrites ``state["deadline_windows"]`` (the active windows) and
    ``state["deadline_accounting"]``. Falls back to wall time when the control log
    cannot be read.
    """
    started = state.get("deadline_started_at")
    if not started:
        return None
    start, now = _parse_time(started), opts.now()
    wall = (now - start) / timedelta(hours=1)
    try:
        events = control_io.read_events(cfg.logs_dir, cfg.spec)
        source = "control_log"
    except (OSError, UnicodeDecodeError) as err:
        logger.warning("control log unreadable ({}); deadline uses wall time", err)
        events, source = [], "wall_fallback"
    paused = paused_intervals(events, state.get("deadline_hosts") or [], start, now)
    paused_h = sum((e - b) / timedelta(hours=1) for b, e in paused)
    edges = [start, *(t for pair in paused for t in pair), now]
    state["deadline_windows"] = [
        {"started_at": b.isoformat(), "ended_at": e.isoformat() if e != now else None}
        for b, e in zip(edges[::2], edges[1::2], strict=True)
        if e > b
    ]
    state["deadline_accounting"] = {
        "at": now.isoformat(),
        "source": source,
        "wall_h": wall,
        "paused_h": paused_h,
        "active_h": wall - paused_h,
        "paused_intervals": [[b.isoformat(), e.isoformat()] for b, e in paused],
    }
    return wall - paused_h


def campaign_hosts(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Every host an xfam2 queue is (or is planned to be) pinned to."""
    hosts = {plan.queue_node(n) for n in plan.QUEUE_PRIORITIES}
    hosts |= {store.read_meta(cfg.queue_root, n).node for n in _submitted(cfg)}
    return sorted(h for h in hosts if h)


def _deadline(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Xfam2DriverOptions,
    fired: list[str],
) -> None:
    """Pause every xfam2 host once per deadline value when active time runs out.

    Writes each host's stop sentinel in the xfam2 queue root and one control-log
    pause event per host, so `campaign_control resume` on a host undoes it. After a
    manual resume the driver does not re-pause until it runs with a new deadline.
    """
    prior = state["transitions"].get("deadline_pause")
    if prior is not None and float(prior["hours"]) == float(opts.deadline_hours):
        return
    elapsed = active_hours(cfg, state, opts)
    if elapsed is None or elapsed < opts.deadline_hours:
        return
    remaining = [
        n
        for tail in (False, True)
        for n in expected_queues(state, tail=tail)[0]
        if not _queue_done(cfg, n, [])
    ]
    if not remaining and "finalize_core" in state["transitions"]:
        return
    hosts = campaign_hosts(cfg)
    reason = (
        f"xfam2 deadline: {elapsed:.2f} active h >= {opts.deadline_hours} h; "
        f"{len(remaining)} queues unfinished"
    )
    if not cfg.dry_run:
        root = Path(cfg.queue_root)
        snap = control_io.snapshot(
            queue_root=root, logs_dir=cfg.logs_dir, spec=cfg.spec
        )
        for host in hosts:
            path = store.stop_path(root, host)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            control_io.append_event(
                cfg.logs_dir,
                cfg.spec,
                action=control_io.PAUSE,
                reason=reason,
                snap={**snap, "host": host, "paused": True},
            )
    if prior is not None:
        state.setdefault("deadline_history", []).append(prior)
    state["transitions"]["deadline_pause"] = {
        "at": utc_now_iso(),
        "hours": float(opts.deadline_hours),
        "active_h": elapsed,
        "hosts": hosts,
        "n_truncated": len(remaining),
        "truncated": remaining,
    }
    fired.append("deadline_pause")
    logger.warning("deadline reached -- paused xfam2 pools on {}", hosts)


# ---- run / status ---------------------------------------------------------------------


def _check_identity(
    cfg: driver_io.CampaignDriverConfig, opts: Xfam2DriverOptions
) -> None:
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(f"run_set_id {cfg.run_set_id!r} is malformed")
    if cfg.spec.run_set_id != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"driver bound to spec {cfg.spec.run_set_id!r}; use driver_config()"
        )
    if (
        opts.expected_queue_root is not None
        and Path(cfg.queue_root).resolve() != Path(opts.expected_queue_root).resolve()
    ):
        raise CampaignGateError(
            f"queue root {cfg.queue_root} is not the xfam2 root {opts.expected_queue_root}"
        )


def run(
    cfg: driver_io.CampaignDriverConfig, options: Xfam2DriverOptions | None = None
) -> AdvanceResult:
    """One advance pass over the xfam2 DAG."""
    opts = options or Xfam2DriverOptions()
    _check_identity(cfg, opts)
    pin_note = None if cfg.dry_run else driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]
    fired: list[str] = [pin_note] if pin_note else []
    waiting: list[str] = []

    for key, step in (
        ("preflight", lambda: _preflight(cfg, state, opts)),
        ("submit_teachers", lambda: _submit_teachers(cfg, state, opts)),
        ("submit_controls", lambda: _submit_controls(cfg, state)),
    ):
        if key not in transitions:
            step()
            driver_io.save_state(cfg, state)
            fired.append(key)

    _requeue(cfg, state)
    _gate_labels(cfg, state, waiting, fired)
    note = _rehome(cfg, state)
    if note:
        fired.append(note)

    if "finalize_core" not in transitions and _ready(cfg, state, waiting, tail=False):
        _finalize_core(cfg, state)
        fired.append("finalize_core")
    if (
        "finalize_core" in transitions
        and "results_tail" not in transitions
        and _ready(cfg, state, waiting, tail=True)
    ):
        _build_results(cfg, dry_run=cfg.dry_run)
        transitions["results_tail"] = {"at": utc_now_iso()}
        fired.append("results_tail")
    if "deadline_started_at" in state:
        _record_hosts(cfg, state, opts.now())
    _deadline(cfg, state, opts, fired)
    driver_io.save_state(cfg, state)

    done = "finalize_core" in transitions and "results_tail" in transitions
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(
    cfg: driver_io.CampaignDriverConfig, options: Xfam2DriverOptions | None = None
) -> dict:
    """Read-only snapshot: transitions, per-node queue totals, gates, holes, deadline."""
    opts = options or Xfam2DriverOptions()
    state = driver_io.load_state(cfg)
    per_node: dict[str, dict[str, int]] = {}
    for name in _submitted(cfg):
        node = str(store.read_meta(cfg.queue_root, name).node)
        counts = per_node.setdefault(node, {"queues": 0})
        counts["queues"] += 1
        for k, v in store.queue_status(cfg.queue_root, name).items():
            if isinstance(v, int):
                counts[k] = counts.get(k, 0) + v
    elapsed = active_hours(cfg, state, opts)
    root = Path(cfg.queue_root)
    return {
        "run_set_id": plan.RUN_SET_ID,
        "transitions": sorted(state["transitions"]),
        "per_node": per_node,
        "gates": {
            k.removeprefix("gate_"): {
                "usable": v.get("usable"),
                "admitted_ftseeds": v.get("admitted_ftseeds"),
            }
            for k, v in state["transitions"].items()
            if k.startswith("gate_")
        },
        "holes": state.get("holes", {}),
        "paused_hosts": [
            h for h in campaign_hosts(cfg) if store.stop_path(root, h).is_file()
        ],
        "deadline": {
            "hours": opts.deadline_hours,
            "elapsed_h": elapsed,
            "remaining_h": None if elapsed is None else opts.deadline_hours - elapsed,
            "accounting": state.get("deadline_accounting"),
        },
        "core_done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }


__all__ = [
    "AdvanceResult",
    "CampaignGateError",
    "Xfam2DriverOptions",
    "active_hours",
    "campaign_hosts",
    "check_footprints",
    "driver_config",
    "expected_queues",
    "label_decision",
    "node_pin_defaults",
    "paused_intervals",
    "run",
    "shared_config_dir",
    "status",
    "teacher_admission_rows",
]
