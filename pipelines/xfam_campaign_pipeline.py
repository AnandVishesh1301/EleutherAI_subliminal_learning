"""Auto-advancing driver for the cross-family OPD limits campaign.

One `run(cfg)` = one advance pass over:

    preflight        -> staging report, prereg, cold links, disk, footprint stamp,
                        generated teacher YAMLs
    submit_teachers  -> the SDF teacher queue and its verify queue
    teacher_gate     -> `teacher_gate_report.json` (written FIRST), then a hard stop
                        if a cluster emitter failed admission
    submit_ladders   -> the six LR-ladder units (treatment + controls + evals)
    pick_{student}   -> when a ladder student's queues drain: `screen_picks.json` and
                        `replication_gate.json` are written once, atomically, and every
                        unit those decisions unlock is submitted with a frozen jobkey set
    finalize_core    -> results CSVs + claims.json + CAMPAIGN_DONE once every core
                        queue is complete
    results_tail     -> idempotent results re-run as tail units complete
    deadline_pause   -> at `DEADLINE_HOURS` with work left: pause the pool, re-run
                        results, record what was truncated

Every gate persists its decision before anything downstream is submitted and reads it
back on resume, never recomputing: an LR is part of the student's GKD tag, so a
re-derived pick would change the jobkey set of a submitted queue.

`CampaignDriverConfig` MUST be built with ``spec=xfam_plan.SPEC`` (use `driver_config`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.xfam_results_pipeline import build_results
from scripts.stage_ckptgrid_assets import DEFAULT_COLD_ROOT, disk_gate_readings
from subliminal_learning.campaign import control_io, driver_io
from subliminal_learning.campaign import xfam_config_gen as cgen
from subliminal_learning.campaign import xfam_jobs as jobs
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign import xfam_selection as selection
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_preference_results,
)
from subliminal_learning.evaluation.sdf_results_io import (
    load_sdf_capability_results,
    load_sdf_preference_results,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import is_valid_run_set_id, run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso


class CampaignGateError(RuntimeError):
    """A pre-registered gate refused to pass."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class XfamDriverOptions:
    """Roots and knobs the preflight reads (overridable in tests).

    Attributes:
        cold_root: Cold-1 artifact root (real dirs live here).
        prereg_path: Pre-registration document that must exist.
        hf_cache_root: Hub cache root for the disk gate.
        deadline_hours: Wall budget after the first submission.
    """

    cold_root: Path = DEFAULT_COLD_ROOT
    prereg_path: Path = Path(plan.PREREG_PATH)
    hf_cache_root: Path | None = None
    deadline_hours: float = plan.DEADLINE_HOURS


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the xfam `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


# ---- queue helpers -----------------------------------------------------------


def _requeue_submitted(cfg: driver_io.CampaignDriverConfig) -> None:
    if cfg.auto_requeue <= 0:
        return
    for name in cfg.queue_priorities:
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            raise CampaignGateError(str(err)) from err


def _queues_ready(
    cfg: driver_io.CampaignDriverConfig, names: Sequence[str], waiting: list[str]
) -> bool:
    if cfg.dry_run:
        return True
    ready = True
    for name in names:
        if not store.queue_exists(cfg.queue_root, name):
            waiting.append(f"{name}: not submitted")
            ready = False
            continue
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            raise CampaignGateError(str(err)) from err
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ready = False
        if "FAILED" in msg:
            raise CampaignGateError(msg)
    return ready


# ---- S0 ----------------------------------------------------------------------


def _require_staging(cfg: driver_io.CampaignDriverConfig) -> dict:
    path = cfg.logs_dir / plan.STAGING_REPORT
    if not path.is_file():
        raise CampaignGateError(
            f"S0 missing {plan.STAGING_REPORT} at {path} — run "
            "`uv run python -m scripts.stage_xfam_assets` first"
        )
    doc = json.loads(path.read_text())
    if doc.get("passed") is not True:
        raise CampaignGateError(f"S0 {plan.STAGING_REPORT} did not pass")
    if doc.get("run_set_id") != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"S0 staging report is for run set {doc.get('run_set_id')!r}"
        )
    return doc


def _require_prefetch_manifest(cfg: driver_io.CampaignDriverConfig) -> dict:
    path = cfg.logs_dir / plan.PREFETCH_MANIFEST
    if not path.is_file():
        raise CampaignGateError(f"S0 missing {plan.PREFETCH_MANIFEST} at {path}")
    doc = json.loads(path.read_text())
    have = {str(entry.get("commit_sha") or "") for entry in (doc.get("refs") or {}).values()}
    missing = [
        spec.label
        for spec in models.ROSTER.values()
        if str(spec.revision) not in have
    ]
    if missing:
        raise CampaignGateError(
            f"S0 {plan.PREFETCH_MANIFEST} has no snapshot for {missing}"
        )
    return doc


def _require_cold_symlinks(
    cfg: driver_io.CampaignDriverConfig, cold_root: Path
) -> None:
    for link, target, action in jobs.symlink_actions(cfg.output_root, cold_root):
        if action != "noop":
            raise CampaignGateError(
                f"S0 cold symlink {link} -> {target} is {action!r}, expected noop — "
                "run `uv run python -m scripts.stage_xfam_assets`"
            )


def _preflight(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: XfamDriverOptions
) -> None:
    if not Path(opts.prereg_path).is_file():
        raise CampaignGateError(f"S0 missing prereg {opts.prereg_path}")
    staging: dict = {"dry_run": True}
    disk: dict[str, float] | None = None
    if not cfg.dry_run:
        staging = _require_staging(cfg)
        _require_prefetch_manifest(cfg)
        _require_cold_symlinks(cfg, opts.cold_root)
        try:
            disk = disk_gate_readings(
                cold_root=opts.cold_root,
                ssd_roots=(cfg.output_root,),
                min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
                min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
            )
        except RuntimeError as err:
            raise CampaignGateError(f"S0 disk gate FAILED: {err}") from err
    teacher_docs = cgen.teacher_configs()
    verify_docs = cgen.teacher_verify_configs(teacher_docs)
    driver_io.write_generated_configs(cfg, {**teacher_docs, **verify_docs})
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "passed": True,
            "run_set_id": plan.RUN_SET_ID,
            "prereg": str(opts.prereg_path),
            "disk_free_gib": disk,
            "staging_at": staging.get("at"),
            "n_teacher_yamls": len(teacher_docs),
            "n_units": len(plan.all_units()),
        },
    )
    state["transitions"]["preflight"] = {
        "at": utc_now_iso(),
        "teacher_yamls": sorted(teacher_docs),
    }


# ---- teacher wave ------------------------------------------------------------


def _submit_teachers(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    docs = cgen.teacher_configs()
    verify = cgen.teacher_verify_configs(docs)
    driver_io.write_generated_configs(cfg, {**docs, **verify})
    t_jobs = jobs.teacher_jobs(docs)
    e_jobs = jobs.teacher_eval_jobs(docs)
    if not t_jobs or not e_jobs:
        raise CampaignGateError("teacher queues built zero jobs")
    driver_io.submit_queue(cfg, plan.Q_TEACHERS, t_jobs)
    driver_io.submit_queue(
        cfg, plan.Q_TEACHER_EVALS, e_jobs, depends_on=[plan.Q_TEACHERS]
    )
    _open_window(state)
    state["transitions"]["submit_teachers"] = {
        "at": utc_now_iso(),
        "n_teacher_jobs": len(t_jobs),
        "n_teacher_eval_jobs": len(e_jobs),
    }


def _teacher_frame(cfg: driver_io.CampaignDriverConfig) -> pd.DataFrame:
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    return load_sdf_preference_results(
        cfg.output_root, subdir, animals=list(plan.ANIMALS)
    )


def teacher_admission_rows(
    cfg: driver_io.CampaignDriverConfig,
) -> list[selection.TeacherAdmission]:
    """Score every trained SDF teacher leaf against the admission rule."""
    pref = _teacher_frame(cfg)
    if pref.empty:
        return []
    primary = pref[
        (pref["set"].astype(str) == plan.PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == plan.PRIMARY_SCORING)
    ]
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    try:
        cap = load_sdf_capability_results(
            cfg.output_root, subdir, animals=list(plan.ANIMALS)
        )
    except (FileNotFoundError, ValueError, OSError) as err:
        logger.warning("teacher capability summaries unreadable: {}", err)
        cap = pd.DataFrame()
    ppl: dict[tuple[str, str, int], float] = {}
    if not cap.empty:
        for _, row in cap.iterrows():
            ratio = row.get("perplexity_ratio")
            if ratio is None or pd.isna(ratio):
                continue
            ppl[
                (str(row.get("model")), str(row.get("param_tag")), int(row.get("ft_seed", 0)))
            ] = float(ratio)
    rows: list[selection.TeacherAdmission] = []
    for _, row in primary.iterrows():
        label = str(row.get("model"))
        tag = str(row.get("param_tag"))
        ftseed = int(row.get("ft_seed", 0))
        p_target = row.get("p_target")
        control = row.get("base_p_target")
        ratio = ppl.get((label, tag, ftseed))
        admitted, reasons = selection.admit_teacher(
            p_target=None if p_target is None or pd.isna(p_target) else float(p_target),
            control_mean=None if control is None or pd.isna(control) else float(control),
            control_sd=None,
            ppl_ratio=ratio,
            sd_multiple=plan.ADMISSION_SD_MULTIPLE,
            abs_min=plan.ADMISSION_ABS_MIN,
            ppl_max=plan.TEACHER_PPL_RATIO_MAX,
        )
        rows.append(
            selection.TeacherAdmission(
                label=label,
                tag=tag,
                ftseed=ftseed,
                admitted=admitted,
                ppl_ratio=ratio,
                p_target=None if p_target is None or pd.isna(p_target) else float(p_target),
                control_mean=None
                if control is None or pd.isna(control)
                else float(control),
                control_sd=None,
                reasons=reasons,
            )
        )
    return rows


def _teacher_gate(cfg: driver_io.CampaignDriverConfig, state: dict) -> dict:
    """Write `teacher_gate_report.json`, then raise if a cluster emitter failed."""
    path = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    if path.is_file():
        doc = json.loads(path.read_text())
    else:
        rows = teacher_admission_rows(cfg)
        decisions = selection.teacher_gate(rows)
        doc = selection.gate_report(decisions, rows)
        if not cfg.dry_run:
            driver_io.write_logs_report(cfg, plan.TEACHER_GATE_REPORT, doc)
    if cfg.dry_run:
        return doc
    unusable = [
        label
        for label, entry in (doc.get("decisions") or {}).items()
        if entry.get("role") == "cluster" and not entry.get("usable")
    ]
    if unusable:
        raise CampaignGateError(
            f"teacher gate: cluster emitters {unusable} have no admissible teacher; "
            f"see {path}"
        )
    state["transitions"]["teacher_gate"] = {
        "at": utc_now_iso(),
        "decisions": {
            label: entry.get("usable")
            for label, entry in (doc.get("decisions") or {}).items()
        },
    }
    return doc


def _teacher_usable(cfg: driver_io.CampaignDriverConfig, label: str) -> bool:
    path = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    if not path.is_file():
        return True
    entry = (json.loads(path.read_text()).get("decisions") or {}).get(label) or {}
    return bool(entry.get("usable", True))


# ---- units -------------------------------------------------------------------


def _submit_unit(
    cfg: driver_io.CampaignDriverConfig,
    arm_name: str,
    student: str,
    picks: Mapping[str, float],
) -> dict[str, int]:
    """Write one unit's YAMLs and submit its train, control and eval queues."""
    docs = cgen.unit_student_configs(arm_name, student, picks)
    verify = cgen.student_verify_configs(docs)
    driver_io.write_generated_configs(cfg, {**docs, **verify})
    treatments, controls = jobs.unit_train_jobs(docs, arm_name, student, picks)
    evals = jobs.unit_eval_jobs(docs, arm_name, student, picks)
    if not treatments or not evals:
        raise CampaignGateError(
            f"unit {arm_name}/{student} built an empty queue "
            f"(treatments={len(treatments)} evals={len(evals)})"
        )
    q_s = plan.q_students(arm_name, student)
    q_e = plan.q_evals(arm_name, student)
    driver_io.submit_queue(cfg, q_s, treatments)
    depends = [q_s]
    counts = {"treatments": len(treatments), "evals": len(evals)}
    if controls:
        q_c = plan.q_controls(arm_name, student)
        driver_io.submit_queue(cfg, q_c, controls)
        depends.append(q_c)
        counts["controls"] = len(controls)
    else:
        # Controls come from the ladder arm this unit reuses; its queue must drain
        # first or the eval would read a half-trained control.
        ladder = _ladder_unit_of(student)
        if ladder is not None:
            depends.append(plan.q_controls(*ladder))
    driver_io.submit_queue(cfg, q_e, evals, depends_on=depends)
    logger.info("submitted unit {}/{}: {}", arm_name, student, counts)
    return counts


def _ladder_unit_of(student: str) -> tuple[str, str] | None:
    """The ladder ``(arm, student)`` unit whose controls a derived unit reuses."""
    for a in plan.ladder_arms():
        if student in a.students:
            return a.name, student
    return None


def _submit_ladders(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    picks = dict(plan.LR_DEFAULTS)
    counts: dict[str, dict[str, int]] = {}
    for a in plan.ladder_arms():
        if not _teacher_usable(cfg, a.teacher):
            logger.warning("skipping ladder arm {}: teacher not usable", a.name)
            continue
        for student in a.students:
            counts[f"{a.name}/{student}"] = _submit_unit(cfg, a.name, student, picks)
    state["transitions"]["submit_ladders"] = {"at": utc_now_iso(), "units": counts}


# ---- LR picks ----------------------------------------------------------------


def _student_frame(cfg: driver_io.CampaignDriverConfig) -> pd.DataFrame:
    return load_gkd_preference_results(
        cfg.output_root,
        run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID),
        animals=list(plan.ANIMALS),
    )


def ladder_inputs(
    cfg: driver_io.CampaignDriverConfig, student: str
) -> tuple[dict[float, float], dict[float, float]]:
    """``(delta_by_lr, ppl_by_lr)`` for one ladder student's pick.

    Delta is the treatment mean minus its matched control mean at the same LR; PPL is
    the treatment mean ``perplexity_ratio``. A rung with no readable control gets a
    delta of 0.0 and a PPL of ``inf``, so it cannot win the pick.
    """
    from subliminal_learning.evaluation.gkd_results_io import (
        load_gkd_capability_results,
    )

    ladder = plan.LR_LADDERS[student]
    pref = _student_frame(cfg)
    delta = {lr: 0.0 for lr in ladder}
    ppl = {lr: float("inf") for lr in ladder}
    if pref.empty:
        return delta, ppl
    rows = pref[
        (pref["set"].astype(str) == plan.PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == plan.PRIMARY_SCORING)
        & (pref["model"].astype(str) == student)
    ]
    if rows.empty:
        return delta, ppl
    try:
        cap = load_gkd_capability_results(
            cfg.output_root,
            run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID),
            animals=list(plan.ANIMALS),
        )
    except (FileNotFoundError, ValueError, OSError) as err:
        logger.warning("student capability summaries unreadable: {}", err)
        cap = pd.DataFrame()
    for lr in ladder:
        tag = plan.student_gkd_tag(lr, plan.BASE_RECIPE, plan.BASE_EPOCHS)
        at_lr = rows[rows["gkd_tag"].astype(str) == tag]
        if at_lr.empty:
            continue
        treat = at_lr[~at_lr["is_control"].astype(bool)]
        ctrl = at_lr[at_lr["is_control"].astype(bool)]
        if treat.empty or ctrl.empty:
            continue
        delta[lr] = float(treat["p_target"].mean()) - float(ctrl["p_target"].mean())
        if not cap.empty:
            crows = cap[
                (cap["model"].astype(str) == student)
                & (cap["gkd_tag"].astype(str) == tag)
                & (~cap["is_control"].astype(bool))
            ]
            ratios = pd.to_numeric(crows.get("perplexity_ratio"), errors="coerce")
            if ratios is not None and ratios.notna().any():
                ppl[lr] = float(ratios.dropna().mean())
    return delta, ppl


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _persist_pick(
    cfg: driver_io.CampaignDriverConfig, student: str, pick: selection.LrPick
) -> dict[str, float]:
    """Merge one pick into `screen_picks.json`; an existing entry is authoritative."""
    path = cfg.logs_dir / plan.SCREEN_PICKS_FILE
    doc = _load_json(path)
    picks_raw = dict(doc.get("picks") or {})
    if student in picks_raw:
        return selection.picks_from_report(doc)
    existing = {
        s: selection.LrPick(
            student=s,
            lr=float(entry["lr"]),
            flagged=bool(entry.get("flagged")),
            reason=str(entry.get("reason", "")),
            table=tuple(entry.get("table") or ()),
        )
        for s, entry in picks_raw.items()
    }
    existing[student] = pick
    body = selection.picks_report(existing)
    if not cfg.dry_run:
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, body)
    logger.info("picked lr*({}) = {} ({})", student, pick.lr, pick.reason)
    return selection.picks_from_report(body)


def _persist_replication(
    cfg: driver_io.CampaignDriverConfig,
    student: str,
    decision: selection.ReplicationDecision,
) -> dict[str, bool]:
    path = cfg.logs_dir / plan.REPLICATION_GATE_FILE
    doc = _load_json(path)
    raw = dict(doc.get("decisions") or {})
    if student not in raw:
        existing = {
            s: selection.ReplicationDecision(
                student=s,
                replicate=bool(entry.get("replicate")),
                delta=entry.get("delta"),
                ci_low=entry.get("ci_low"),
                ci_high=entry.get("ci_high"),
                reason=str(entry.get("reason", "")),
            )
            for s, entry in raw.items()
        }
        existing[student] = decision
        body = selection.replication_report(existing)
        if not cfg.dry_run:
            cfg.logs_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, body)
        logger.info(
            "replication gate for {}: {} ({})",
            student,
            decision.replicate,
            decision.reason,
        )
        doc = body
    return selection.replications_from_report(doc)


def _replication_inputs(
    cfg: driver_io.CampaignDriverConfig, student: str, lr: float
) -> tuple[float | None, float | None, float | None]:
    """Cell mean and bootstrap CI of the student's treatment delta at `lr`.

    Resamples the ladder cell's leaves with `analysis.stats.cluster_bootstrap` over
    (ftseed, gseed), the crossing the cell actually has. Imported inside the function
    because `subliminal_learning.analysis` eagerly pulls the evaluation IO stack.
    """
    import numpy as np

    from subliminal_learning.analysis.stats import cluster_bootstrap

    pref = _student_frame(cfg)
    if pref.empty:
        return None, None, None
    tag = plan.student_gkd_tag(lr, plan.BASE_RECIPE, plan.BASE_EPOCHS)
    rows = pref[
        (pref["set"].astype(str) == plan.PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == plan.PRIMARY_SCORING)
        & (pref["model"].astype(str) == student)
        & (pref["gkd_tag"].astype(str) == tag)
    ]
    treat = rows[~rows["is_control"].astype(bool)]
    ctrl = rows[rows["is_control"].astype(bool)]
    if treat.empty or ctrl.empty:
        return None, None, None
    base = float(ctrl["p_target"].mean())
    rows = treat[treat["p_target"].notna()]
    if rows.empty:
        return None, None, None
    values = np.asarray([float(v) - base for v in rows["p_target"]], dtype=float)
    mean = float(values.mean())
    ftseeds = [str(v) for v in rows.get("teacher_ftseed", ["0"] * len(rows))]
    gseeds = [str(i) for i in range(len(rows))]
    try:
        est = cluster_bootstrap(
            lambda idx: float(values[idx].mean()),
            ftseeds,
            gseeds,
            n_boot=plan.BOOTSTRAP_N,
            seed=plan.BOOTSTRAP_SEED,
        )
    except (ValueError, ZeroDivisionError):
        return mean, None, None
    return mean, float(est.ci_low), float(est.ci_high)


def _unit_queue_names(arm_name: str, student: str) -> list[str]:
    names = [plan.q_students(arm_name, student), plan.q_evals(arm_name, student)]
    if plan.arm(arm_name).emits_controls:
        names.insert(1, plan.q_controls(arm_name, student))
    return names


def _pick_gates(
    cfg: driver_io.CampaignDriverConfig, state: dict, waiting: list[str], fired: list[str]
) -> None:
    """Fire each ladder student's pick once its ladder queues have drained."""
    for a in plan.ladder_arms():
        for student in a.students:
            key = f"pick_{student}"
            if key in state["transitions"]:
                continue
            if not _teacher_usable(cfg, a.teacher):
                continue
            if not _queues_ready(cfg, _unit_queue_names(a.name, student), waiting):
                continue
            delta_by_lr, ppl_by_lr = ladder_inputs(cfg, student)
            pick = selection.pick_lr(
                student=student, delta_by_lr=delta_by_lr, ppl_by_lr=ppl_by_lr
            )
            picks = _persist_pick(cfg, student, pick)
            mean, low, high = _replication_inputs(cfg, student, picks[student])
            replications = _persist_replication(
                cfg,
                student,
                selection.replication_decision(
                    student=student, delta=mean, ci_low=low, ci_high=high
                ),
            )
            submitted: list[str] = []
            for arm_name, unit_student in plan.units_unlocked_by(student):
                unit_arm = plan.arm(arm_name)
                if not _teacher_usable(cfg, unit_arm.teacher):
                    logger.warning(
                        "skipping {}/{}: teacher {} not usable",
                        arm_name,
                        unit_student,
                        unit_arm.teacher,
                    )
                    continue
                if unit_arm.replication_gated and not replications.get(
                    unit_student, False
                ):
                    logger.info(
                        "skipping {}/{}: replication gate did not fire",
                        arm_name,
                        unit_student,
                    )
                    continue
                _submit_unit(cfg, arm_name, unit_student, picks)
                submitted.append(f"{arm_name}/{unit_student}")
            state["transitions"][key] = {
                "at": utc_now_iso(),
                "lr": picks[student],
                "flagged": pick.flagged,
                "reason": pick.reason,
                "submitted": submitted,
            }
            driver_io.save_state(cfg, state)
            fired.append(key)


# ---- completion --------------------------------------------------------------


def _submitted_queue_names(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [
        name
        for name in cfg.queue_priorities
        if store.queue_exists(cfg.queue_root, name)
    ]


def _core_queue_names(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    tail = plan.tail_queue_names()
    return [n for n in _submitted_queue_names(cfg) if n not in tail]


def _all_picks_fired(state: dict) -> bool:
    return all(
        f"pick_{student}" in state["transitions"] for student in plan.LADDER_STUDENTS
    )


def _core_ready(
    cfg: driver_io.CampaignDriverConfig, state: dict, waiting: list[str]
) -> bool:
    transitions = state["transitions"]
    if "submit_ladders" not in transitions or "finalize_core" in transitions:
        return False
    if not _all_picks_fired(state):
        waiting.append("LR picks pending")
        return False
    names = [plan.Q_TEACHERS, plan.Q_TEACHER_EVALS, *_core_queue_names(cfg)]
    return _queues_ready(cfg, list(dict.fromkeys(names)), waiting)


def _finalize_core(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    build_results(cfg, dry_run=cfg.dry_run)
    if not cfg.dry_run:
        atomic_write_text(
            cfg.logs_dir / plan.CAMPAIGN_DONE_FILE,
            f"campaign {plan.RUN_SET_ID} core finished {utc_now_iso()}\n",
        )
    state["transitions"]["finalize_core"] = {"at": utc_now_iso()}


def _tail_incomplete(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    if cfg.dry_run:
        return []
    out: list[str] = []
    for name in plan.tail_queue_names():
        if not store.queue_exists(cfg.queue_root, name):
            out.append(name)
        elif not driver_io.queue_finished(cfg, name)[0]:
            out.append(name)
    return sorted(out)


def _tail_results(
    cfg: driver_io.CampaignDriverConfig, state: dict, fired: list[str]
) -> None:
    if "finalize_core" not in state["transitions"]:
        return
    if "results_tail" in state["transitions"] or _tail_incomplete(cfg):
        return
    build_results(cfg, dry_run=cfg.dry_run)
    state["transitions"]["results_tail"] = {"at": utc_now_iso()}
    fired.append("results_tail")


# ---- deadline ----------------------------------------------------------------


def _open_window(state: dict) -> None:
    windows = state.setdefault("deadline_windows", [])
    if not windows or windows[-1].get("ended_at"):
        windows.append({"started_at": utc_now_iso()})


def _elapsed_hours(state: dict) -> float | None:
    windows = state.get("deadline_windows") or []
    if not windows:
        return None
    started = datetime.fromisoformat(str(windows[0]["started_at"]).replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - started) / timedelta(hours=1)


def _deadline(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: XfamDriverOptions,
    fired: list[str],
) -> None:
    if "deadline_pause" in state["transitions"]:
        return
    elapsed = _elapsed_hours(state)
    if elapsed is None or elapsed < opts.deadline_hours:
        return
    remaining = _tail_incomplete(cfg)
    if not remaining and "finalize_core" in state["transitions"]:
        return
    if not cfg.dry_run:
        control_io.pause(
            queue_root=cfg.queue_root,
            logs_dir=cfg.logs_dir,
            spec=cfg.spec,
            reason=(
                f"deadline {opts.deadline_hours} h reached with work left: {remaining}"
            ),
        )
        build_results(cfg, dry_run=False)
    windows = state.get("deadline_windows") or []
    if windows:
        windows[-1]["ended_at"] = utc_now_iso()
        windows[-1]["reason"] = "deadline"
    state["transitions"]["deadline_pause"] = {
        "at": utc_now_iso(),
        "truncated": remaining,
    }
    fired.append("deadline_pause")
    logger.warning("deadline reached — paused the pool; truncated: {}", remaining)


# ---- run / status ------------------------------------------------------------


def run(
    cfg: driver_io.CampaignDriverConfig, options: XfamDriverOptions | None = None
) -> AdvanceResult:
    """One advance pass over the xfam DAG."""
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(f"run_set_id {cfg.run_set_id!r} is malformed")
    if cfg.spec.run_set_id != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"xfam driver bound to wrong spec run_set_id={cfg.spec.run_set_id!r}; "
            "pass spec=xfam_plan.SPEC via driver_config()"
        )
    opts = options or XfamDriverOptions()
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    state.setdefault("deadline_windows", [])
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []
    if pin_note:
        fired.append(pin_note)

    if "preflight" not in transitions:
        _preflight(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("preflight")

    _requeue_submitted(cfg)

    if "submit_teachers" not in transitions:
        _submit_teachers(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_teachers")

    if "teacher_gate" not in transitions:
        if _queues_ready(cfg, [plan.Q_TEACHERS, plan.Q_TEACHER_EVALS], waiting):
            _teacher_gate(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("teacher_gate")
        else:
            return AdvanceResult(tuple(fired), tuple(waiting), False)

    if "submit_ladders" not in transitions:
        _submit_ladders(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_ladders")

    _pick_gates(cfg, state, waiting, fired)

    if _core_ready(cfg, state, waiting):
        _finalize_core(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("finalize_core")

    _tail_results(cfg, state, fired)
    _deadline(cfg, state, opts, fired)
    driver_io.save_state(cfg, state)

    done = "finalize_core" in transitions and (
        "deadline_pause" in transitions or "results_tail" in transitions
    )
    if "finalize_core" in transitions and not done:
        waiting.append(f"tail draining: {_tail_incomplete(cfg)}")
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(
    cfg: driver_io.CampaignDriverConfig, options: XfamDriverOptions | None = None
) -> dict:
    """Read-only snapshot: transitions, queues, gates, deadline."""
    opts = options or XfamDriverOptions()
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in cfg.queue_priorities
        if store.queue_exists(cfg.queue_root, name)
    }
    elapsed = _elapsed_hours(state)
    picks = _load_json(cfg.logs_dir / plan.SCREEN_PICKS_FILE).get("picks") or {}
    return {
        "run_set_id": plan.RUN_SET_ID,
        "transitions": sorted(state["transitions"]),
        "queues": queues,
        "n_queues_submitted": len(queues),
        "picks": {s: entry.get("lr") for s, entry in picks.items()},
        "deadline": {
            "hours": opts.deadline_hours,
            "elapsed_h": elapsed,
            "remaining_h": (opts.deadline_hours - elapsed) if elapsed is not None else None,
        },
        "tail_incomplete": _tail_incomplete(cfg),
        "core_done": "finalize_core" in state["transitions"],
        "done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }


__all__ = [
    "AdvanceResult",
    "CampaignGateError",
    "XfamDriverOptions",
    "driver_config",
    "ladder_inputs",
    "run",
    "status",
    "teacher_admission_rows",
]
