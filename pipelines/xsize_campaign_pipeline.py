"""Lean auto-advancing driver for the cross-size bear GKD campaign.

One `run(cfg)` = one advance pass over:

    S0 preflight   -> staged reports + cold links + disk + P1/P2 YAMLs
    P1             -> SDF teachers + teacher evals
    teacher_gate   -> write teacher_gate_report.json, then act
    P1 fallback    -> tfs 1/2 at the chosen fallback tag (failure path)
    P2             -> LR ladder treatments + sb controls + evals
    P3 / P3b       -> per-size headline (+ LR robustness) after pick_lr
    P5 finalize    -> results CSVs + claims.json + CAMPAIGN_DONE

P3b is the optional lowest-priority tail: it is excluded from the finalize gate
(`plan.p3b_queue_names`), so an overrun self-truncates from the tail and X3
reports whatever P3b leaves exist. `scripts.run_xsize_campaign results` re-folds
late ones in.

`CampaignDriverConfig` MUST be built with `spec=xsize_plan.SPEC`. Code pin
is `driver_io.ensure_campaign_code_pin` (workspace HEAD may move).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pipelines.xsize_results_pipeline import (
    _admissions_from_frame,
    _decisions_from_report,
    assemble_student_frame,
    assemble_teacher_frame,
    build_results,
)
from scripts.stage_ckptgrid_assets import DEFAULT_COLD_ROOT, disk_gate_readings
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import expected
from subliminal_learning.campaign import xsize_config_gen as cgen
from subliminal_learning.campaign import xsize_jobs as jobs
from subliminal_learning.campaign import xsize_metrics as xm
from subliminal_learning.campaign import xsize_plan as plan
from subliminal_learning.campaign.xsize_jobs import prefetch_refs, symlink_action
from subliminal_learning.campaign.xsize_selection import (
    CampaignGateError,
    LrPick,
    TeacherGateDecision,
    gate_report,
    pick_lr,
    picks_from_report,
    picks_report,
    teacher_gate,
)
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

Q_P1_FALLBACK = plan.Q_P1_FALLBACK
Q_P1_FALLBACK_EVALS = plan.Q_P1_FALLBACK_EVALS


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class XsizeDriverOptions:
    """Optional roots the preflight reads (overridable in tests)."""

    cold_root: Path = DEFAULT_COLD_ROOT
    prereg_path: Path = Path(plan.PREREG_PATH)
    hf_cache_root: Path | None = None


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the cross-size `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


def _requeue_submitted(cfg: driver_io.CampaignDriverConfig) -> None:
    if cfg.auto_requeue <= 0:
        return
    for name in cfg.queue_priorities:
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            raise CampaignGateError(str(err)) from err


def _queues_ready(
    cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]
) -> bool:
    if cfg.dry_run:
        return True
    ready = True
    for name in names:
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


def _submitted_queue_names(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [
        name
        for name in cfg.queue_priorities
        if store.queue_exists(cfg.queue_root, name)
    ]


def _teacher_depends(state: dict) -> list[str]:
    deps = [plan.Q_P1_TEACHERS]
    if "submit_p1_fallback" in state["transitions"] or state.get("fallback_pending"):
        deps.append(Q_P1_FALLBACK)
    return deps


def _voided(state: dict) -> frozenset[str]:
    return frozenset(str(s) for s in state.get("voided_sizes") or [])


def _teacher_tags(cfg: driver_io.CampaignDriverConfig, state: dict) -> dict[str, str]:
    """Admitted teacher size → SDF tag, from the persisted gate report.

    Voided sizes are absent: they teach no P3/P3b cell but still run as
    students (`config_gen` substitutes the frozen tag for their anchor ladder).
    Read from disk (not recomputed) so every student YAML / jobkey built after
    the gate is identical on re-advance. Before the gate (dry-run S0 only) the
    frozen tag stands in.
    """
    path = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    if not path.is_file():
        if cfg.dry_run or "teacher_gate" not in state["transitions"]:
            return cgen.frozen_teacher_tags()
        raise CampaignGateError(
            f"{plan.TEACHER_GATE_REPORT} missing at {path} after the teacher gate fired"
        )
    decisions = _decisions_from_report(json.loads(path.read_text()))
    voided = _voided(state)
    tags = {
        size: str(dec.tag)
        for size, dec in decisions.items()
        if dec.tag and dec.status != "no_admissible_teacher" and size not in voided
    }
    if not tags:
        raise CampaignGateError("teacher gate admitted no size — nothing to distill")
    return tags


def _teacher_docs() -> tuple[dict[str, dict], dict[str, dict]]:
    train = cgen.teacher_configs()
    return train, cgen.teacher_verify_configs(train)


def _ladder_docs(cfg: driver_io.CampaignDriverConfig, state: dict) -> dict[str, dict]:
    return cgen.ladder_student_configs(teacher_tags=_teacher_tags(cfg, state))


def _p3_docs(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    picks: dict[str, float],
    size: str,
    *,
    p3b: bool,
) -> dict[str, dict]:
    tags = _teacher_tags(cfg, state)
    train: dict[str, dict] = {}
    train.update(cgen.headline_student_configs(picks, size=size, teacher_tags=tags))
    train.update(cgen.headline_control_configs(picks, size=size, teacher_tags=tags))
    if p3b:
        train.update(
            cgen.lr_robustness_student_configs(picks, size=size, teacher_tags=tags)
        )
        train.update(
            cgen.lr_robustness_control_configs(picks, size=size, teacher_tags=tags)
        )
    return train


def _require_prefetch_manifest(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Load `prefetch_manifest.json` and demand every designed Hub ref."""
    path = cfg.logs_dir / plan.PREFETCH_MANIFEST
    if not path.is_file():
        raise CampaignGateError(
            f"S0 missing {plan.PREFETCH_MANIFEST} at {path} — run "
            "`python -m scripts.stage_xsize_assets` first"
        )
    doc = json.loads(path.read_text())
    records = doc.get("refs") or []
    have = set()
    for rec in records:
        repo = str(rec.get("repo") or rec.get("repo_id") or "")
        have.add((repo, str(rec.get("revision") or "")))
    missing = [
        (ref.repo_id, ref.revision)
        for ref in prefetch_refs()
        if (ref.repo_id, ref.revision) not in have
    ]
    if missing:
        raise CampaignGateError(
            f"S0 {plan.PREFETCH_MANIFEST} missing {len(missing)} refs "
            f"(first: {missing[0]})"
        )
    return doc


def _require_passed_report(path: Path, filename: str, *, require_true: bool) -> dict:
    if not path.is_file():
        raise CampaignGateError(
            f"S0 missing {filename} at {path} — run "
            "`python -m scripts.stage_xsize_assets` first"
        )
    doc = json.loads(path.read_text())
    if require_true and doc.get("passed") is not True:
        raise CampaignGateError(f"S0 {filename} did not pass")
    if doc.get("passed") is False:
        raise CampaignGateError(f"S0 {filename} did not pass")
    if doc.get("dataset_error"):
        raise CampaignGateError(f"S0 {filename} dataset_error: {doc['dataset_error']}")
    if doc.get("disk_error"):
        raise CampaignGateError(f"S0 {filename} disk_error: {doc['disk_error']}")
    return doc


def _require_cold_symlinks(
    cfg: driver_io.CampaignDriverConfig, cold_root: Path
) -> None:
    pairs = jobs.cold_symlink_plan(cfg.output_root, cold_root)
    for link, target in pairs:
        try:
            action = symlink_action(link, target)
        except FileExistsError as err:
            raise CampaignGateError(f"S0 cold symlink FAILED: {err}") from err
        if action != "noop":
            raise CampaignGateError(
                f"S0 cold symlink {link} → {target} is {action!r}, "
                "expected noop — run `python -m scripts.stage_xsize_assets`"
            )


def _preflight(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    options: XsizeDriverOptions,
) -> None:
    """S0: staged reports, disk, prereg, generated P1 configs (P2 waits on the gate)."""
    _require_prefetch_manifest(cfg)
    _require_passed_report(
        cfg.logs_dir / plan.P0_GATES_REPORT,
        plan.P0_GATES_REPORT,
        require_true=True,
    )
    _require_passed_report(
        cfg.logs_dir / plan.PREFLIGHT_REPORT,
        plan.PREFLIGHT_REPORT,
        require_true=False,
    )
    if not Path(options.prereg_path).is_file():
        raise CampaignGateError(f"S0 missing prereg {options.prereg_path}")
    _require_cold_symlinks(cfg, options.cold_root)
    hf_root = options.hf_cache_root or cfg.output_root
    try:
        disk = disk_gate_readings(
            cold_root=options.cold_root,
            ssd_roots=(cfg.output_root, hf_root),
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
        )
    except RuntimeError as err:
        raise CampaignGateError(f"S0 disk gate FAILED: {err}") from err
    train, verify = _teacher_docs()
    driver_io.write_generated_configs(cfg, {**train, **verify})
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "passed": True,
            "n_refs": len(prefetch_refs()),
            "manifest": str(cfg.logs_dir / plan.PREFETCH_MANIFEST),
            "disk_free_gib": disk,
            "prereg": str(options.prereg_path),
            "n_train_yamls": len(train),
            "n_verify_yamls": len(verify),
        },
    )
    state["transitions"]["preflight"] = {
        "at": utc_now_iso(),
        "train_yamls": sorted(train),
        "verify_yamls": sorted(verify),
    }


def _submit_p1(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    train, verify = _teacher_docs()
    driver_io.write_generated_configs(cfg, {**train, **verify})
    parts = jobs.split_train_configs(train)
    t_verify, _ = jobs.split_verify_configs(verify)
    t_jobs = jobs.teacher_jobs(parts.teachers)
    if not t_jobs:
        raise CampaignGateError("P1 teacher queue built zero jobs")
    driver_io.submit_queue(cfg, plan.Q_P1_TEACHERS, t_jobs)
    e_jobs = jobs.teacher_eval_jobs(t_verify, parts.teachers)
    if not e_jobs:
        raise CampaignGateError("P1 teacher-eval queue built zero jobs")
    driver_io.submit_queue(
        cfg,
        plan.Q_P1_TEACHER_EVALS,
        e_jobs,
        depends_on=[plan.Q_P1_TEACHERS],
    )
    state["transitions"]["submit_p1"] = {
        "at": utc_now_iso(),
        "n_teacher_jobs": len(t_jobs),
        "n_teacher_eval_jobs": len(e_jobs),
    }


def _dry_frozen_decisions() -> dict[str, TeacherGateDecision]:
    return {
        size: TeacherGateDecision(
            size=size,
            tag=plan.TEACHER_PARAM_TAG,
            status="frozen",
            n_admitted_frozen=len(plan.FTSEEDS),
            fallback_candidates=(),
            reasons=(),
        )
        for size in plan.SIZES
    }


def run_teacher_gate(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    *,
    teacher_df=None,
) -> dict[str, TeacherGateDecision]:
    """Admit teachers, write the gate report, then record void/fallback."""
    if cfg.dry_run and teacher_df is None:
        decisions = _dry_frozen_decisions()
    else:
        frame = (
            teacher_df
            if teacher_df is not None
            else assemble_teacher_frame(
                cfg.output_root, teacher_run_set_id=plan.TEACHER_RUN_SET_ID
            )
        )
        decisions = teacher_gate(_admissions_from_frame(frame))
    driver_io.write_logs_report(
        cfg,
        plan.TEACHER_GATE_REPORT,
        {"at": utc_now_iso(), **gate_report(decisions)},
    )
    voided = [
        size for size, dec in decisions.items() if dec.status == "no_admissible_teacher"
    ]
    pending = {
        size: str(dec.tag)
        for size, dec in decisions.items()
        if dec.status == "fallback" and dec.tag
    }
    completed = set(state.get("fallback_completed") or [])
    pending = {s: t for s, t in pending.items() if s not in completed}
    state["voided_sizes"] = sorted(set(state.get("voided_sizes") or []) | set(voided))
    state["fallback_pending"] = pending
    state["transitions"]["teacher_gate"] = {
        "at": utc_now_iso(),
        "decisions": {s: d.to_dict() for s, d in decisions.items()},
        "voided_sizes": list(voided),
        "fallback_sizes": sorted(pending),
    }
    if voided:
        raise CampaignGateError(
            "teacher gate: no admissible teacher for "
            + ", ".join(voided)
            + " — those sizes are voided; remaining sizes proceed on re-advance"
        )
    return decisions


def _submit_fallback(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    pending = dict(state.get("fallback_pending") or {})
    if not pending:
        return
    train: dict[str, dict] = {}
    for size, tag in sorted(pending.items()):
        train.update(cgen.fallback_teacher_configs(size, tag, queue=Q_P1_FALLBACK))
    verify = cgen.teacher_verify_configs(train)
    for doc in verify.values():
        doc["queue"] = Q_P1_FALLBACK_EVALS
    driver_io.write_generated_configs(cfg, {**train, **verify})
    t_jobs = jobs.fallback_teacher_jobs(train)
    e_jobs = jobs.fallback_teacher_eval_jobs(verify, train)
    if not t_jobs or not e_jobs:
        raise CampaignGateError(
            f"P1 fallback built empty queues (train={len(t_jobs)} eval={len(e_jobs)})"
        )
    driver_io.submit_queue(cfg, Q_P1_FALLBACK, t_jobs, depends_on=[plan.Q_P1_TEACHERS])
    driver_io.submit_queue(
        cfg,
        Q_P1_FALLBACK_EVALS,
        e_jobs,
        depends_on=[Q_P1_FALLBACK],
    )
    state["transitions"]["submit_p1_fallback"] = {
        "at": utc_now_iso(),
        "sizes": sorted(pending),
        "n_teacher_jobs": len(t_jobs),
        "n_teacher_eval_jobs": len(e_jobs),
    }


def _regate_after_fallback(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    """Re-run the teacher gate after fallback completion; rewrite report once."""
    pending = dict(state.get("fallback_pending") or {})
    if cfg.dry_run:
        state["fallback_completed"] = sorted(
            set(state.get("fallback_completed") or []) | set(pending)
        )
        state["fallback_pending"] = {}
        state["transitions"]["teacher_gate_regate"] = {
            "at": utc_now_iso(),
            "dry_run": True,
        }
        return
    frame = assemble_teacher_frame(
        cfg.output_root, teacher_run_set_id=plan.TEACHER_RUN_SET_ID
    )
    admissions = _admissions_from_frame(frame)
    fresh = teacher_gate(admissions)
    # `teacher_gate` ranks fallbacks on tfs0 alone; the completed tfs1/2 must
    # clear the same admission bar as the frozen tag or the size is voided.
    for size, tag in pending.items():
        dec = fresh.get(size)
        if dec is None or dec.status != "fallback" or dec.tag != tag:
            continue
        n_admitted = sum(
            1 for r in admissions if r.size == size and r.tag == tag and r.admitted
        )
        if n_admitted < plan.MIN_ADMITTED_PER_BASE:
            fresh[size] = TeacherGateDecision(
                size=size,
                tag=None,
                status="no_admissible_teacher",
                n_admitted_frozen=dec.n_admitted_frozen,
                fallback_candidates=dec.fallback_candidates,
                reasons=(
                    *dec.reasons,
                    f"fallback_admitted={n_admitted}<{plan.MIN_ADMITTED_PER_BASE}",
                ),
            )
    path = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    prev_doc = json.loads(path.read_text()) if path.is_file() else {}
    combined = (
        _decisions_from_report(prev_doc) if prev_doc.get("decisions") else dict(fresh)
    )
    rewritten: list[str] = []
    new_voided: list[str] = []
    for size in pending:
        old = (prev_doc.get("decisions") or {}).get(size) or {}
        if old.get("status") != "fallback":
            continue
        combined[size] = fresh[size]
        rewritten.append(size)
        if fresh[size].status == "no_admissible_teacher":
            new_voided.append(size)
    if rewritten:
        driver_io.write_logs_report(
            cfg,
            plan.TEACHER_GATE_REPORT,
            {"at": utc_now_iso(), **gate_report(combined)},
        )
    state["voided_sizes"] = sorted(
        set(state.get("voided_sizes") or []) | set(new_voided)
    )
    state["fallback_completed"] = sorted(
        set(state.get("fallback_completed") or []) | set(pending)
    )
    state["fallback_pending"] = {}
    state["transitions"]["teacher_gate_regate"] = {
        "at": utc_now_iso(),
        "rewritten": rewritten,
        "voided_sizes": list(new_voided),
    }
    if new_voided:
        raise CampaignGateError(
            "teacher gate (re-gate): no admissible teacher for " + ", ".join(new_voided)
        )


def _submit_p2(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    """P2 ladder for every non-voided size at the gate's per-size teacher tag."""
    train = _ladder_docs(cfg, state)
    verify = cgen.student_verify_configs(train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    t_jobs = jobs.ladder_jobs(train)
    c_jobs = jobs.ladder_control_jobs(train)
    e_jobs = jobs.ladder_eval_jobs(verify, train)
    if not t_jobs or not c_jobs or not e_jobs:
        raise CampaignGateError(
            "P2 built an empty queue "
            f"(ladder={len(t_jobs)} ctrl={len(c_jobs)} eval={len(e_jobs)})"
        )
    depends = _teacher_depends(state)
    driver_io.submit_queue(cfg, plan.Q_P2_LADDER, t_jobs, depends_on=depends)
    driver_io.submit_queue(cfg, plan.Q_P2_CONTROLS, c_jobs, depends_on=depends)
    driver_io.submit_queue(
        cfg,
        plan.Q_P2_EVALS,
        e_jobs,
        depends_on=[plan.Q_P2_LADDER, plan.Q_P2_CONTROLS],
    )
    state["transitions"]["submit_p2"] = {
        "at": utc_now_iso(),
        "n_ladder_jobs": len(t_jobs),
        "n_control_jobs": len(c_jobs),
        "n_eval_jobs": len(e_jobs),
    }


def _docs_for_size(docs: dict[str, dict], size: str) -> dict[str, dict]:
    return {k: v for k, v in docs.items() if str(v.get("model_size")) == size}


def _plant_eval_ok(dirs: list[Path]) -> list[Path]:
    return expected.verify_summaries_complete(dirs)


def _ladder_complete(
    cfg: driver_io.CampaignDriverConfig, state: dict, size: str
) -> bool:
    """True when this size's 9 treatments + 9 sb controls + evals exist."""
    ladder = _docs_for_size(_ladder_docs(cfg, state), size)
    if not ladder:
        return False
    treatments: list[Path] = []
    controls: list[Path] = []
    for doc in ladder.values():
        treatments.extend(
            expected.treatment_leaf_dirs(doc, cfg.output_root, spec=plan.SPEC)
        )
        controls.extend(
            expected.control_leaf_dirs(doc, cfg.output_root, spec=plan.SPEC)
        )
    if expected.missing_manifests(treatments + controls):
        return False
    if _plant_eval_ok(treatments + controls):
        return False
    return True


def _load_picks_doc(cfg: driver_io.CampaignDriverConfig) -> dict:
    path = cfg.logs_dir / plan.SCREEN_PICKS
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _picks_lrs(cfg: driver_io.CampaignDriverConfig) -> dict[str, float]:
    doc = _load_picks_doc(cfg)
    if not doc:
        return {}
    return picks_from_report(doc)


def _compute_lr_pick(cfg: driver_io.CampaignDriverConfig, size: str) -> LrPick:
    if cfg.dry_run:
        return LrPick(
            size=size,
            lr=plan.LR_DEFAULT,
            flagged=False,
            reason="dry_run",
            table=(),
        )
    frame = assemble_student_frame(cfg.output_root, cfg.campaign_subdir)
    deltas = xm.leaf_deltas(frame)
    if deltas.empty:
        raise CampaignGateError(f"P2 ladder frame is empty for pick_lr({size})")
    sub = deltas[
        (deltas["student"].astype(str) == size)
        & (deltas["teacher"].astype(str) == size)
    ]
    if "ftseed" in sub.columns:
        ft = pd.to_numeric(sub["ftseed"], errors="coerce")
        sub = sub.loc[ft.fillna(-1).astype(int) == 0]
    delta_by_lr: dict[float, float] = {}
    ppl_by_lr: dict[float, float] = {}
    for lr in plan.LR_LADDER:
        at = sub[np.isclose(sub["lr"].astype(float), float(lr))]
        if at.empty:
            raise CampaignGateError(f"P2 ladder deltas incomplete for {size} lr={lr}")
        delta_by_lr[lr] = float(at["delta_sb"].mean())
        ppl_by_lr[lr] = float(
            pd.to_numeric(at["perplexity_ratio"], errors="coerce").mean()
        )
    return pick_lr(size=size, delta_by_lr=delta_by_lr, ppl_by_lr=ppl_by_lr)


def _write_pick(cfg: driver_io.CampaignDriverConfig, pick: LrPick) -> None:
    existing = _load_picks_doc(cfg)
    raw = existing.get("picks") if existing else None
    if isinstance(raw, dict) and pick.size in raw:
        logger.info("screen_picks already has {} — not recomputing", pick.size)
        return
    body = picks_report({pick.size: pick})
    if isinstance(raw, dict):
        body["picks"] = {**raw, **body["picks"]}
    driver_io.write_logs_report(cfg, plan.SCREEN_PICKS, body)


def _ensure_pick(cfg: driver_io.CampaignDriverConfig, size: str) -> float:
    have = _picks_lrs(cfg)
    if size in have:
        return have[size]
    pick = _compute_lr_pick(cfg, size)
    _write_pick(cfg, pick)
    reread = _picks_lrs(cfg)
    if size in reread:
        return reread[size]
    return pick.lr


def _submit_p3(cfg: driver_io.CampaignDriverConfig, state: dict, size: str) -> None:
    lr = _ensure_pick(cfg, size)
    picks = {**_picks_lrs(cfg), size: lr}
    include_p3b = size in plan.LR_ROBUSTNESS_STUDENTS
    train = _p3_docs(cfg, state, picks, size, p3b=include_p3b)
    verify = cgen.student_verify_configs(train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    s_jobs = jobs.headline_jobs(train, picks, size=size)
    c_jobs = jobs.headline_control_jobs(train, picks, size=size)
    e_jobs = jobs.p3_eval_jobs(verify, train, picks, size=size)
    if not s_jobs or not c_jobs or not e_jobs:
        raise CampaignGateError(
            f"P3 {size} built an empty queue "
            f"(students={len(s_jobs)} ctrl={len(c_jobs)} eval={len(e_jobs)})"
        )
    depends = _teacher_depends(state)
    driver_io.submit_queue(cfg, plan.q_p3_students(size), s_jobs, depends_on=depends)
    driver_io.submit_queue(cfg, plan.q_p3_controls(size), c_jobs, depends_on=depends)
    driver_io.submit_queue(
        cfg,
        plan.q_p3_evals(size),
        e_jobs,
        depends_on=[plan.q_p3_students(size), plan.q_p3_controls(size)],
    )
    n_p3b = 0
    n_p3b_eval = 0
    if include_p3b:
        b_jobs = jobs.lr_robustness_jobs(train, picks, size=size)
        be_jobs = jobs.p3b_eval_jobs(verify, train, picks, size=size)
        if not b_jobs or not be_jobs:
            raise CampaignGateError(
                f"P3b {size} built an empty queue "
                f"(train={len(b_jobs)} eval={len(be_jobs)})"
            )
        driver_io.submit_queue(cfg, plan.q_p3b(size), b_jobs, depends_on=depends)
        driver_io.submit_queue(
            cfg,
            plan.q_p3b_evals(size),
            be_jobs,
            depends_on=[plan.q_p3b(size)],
        )
        n_p3b = len(b_jobs)
        n_p3b_eval = len(be_jobs)
    state["transitions"][f"pick_lr_{size}"] = {
        "at": utc_now_iso(),
        "lr": lr,
    }
    state["transitions"][f"submit_p3_{size}"] = {
        "at": utc_now_iso(),
        "lr": lr,
        "n_student_jobs": len(s_jobs),
        "n_control_jobs": len(c_jobs),
        "n_eval_jobs": len(e_jobs),
        "n_p3b_jobs": n_p3b,
        "n_p3b_eval_jobs": n_p3b_eval,
    }


def _p3_leaves_ready(
    cfg: driver_io.CampaignDriverConfig, state: dict, size: str
) -> bool:
    """Whether this size's P3 treatments + xbase controls + evals are on disk.

    P3b leaves are deliberately excluded: they are the optional tail (X3 only),
    not part of the completion criterion.
    """
    picks = _picks_lrs(cfg)
    if size not in picks:
        return False
    train = _p3_docs(cfg, state, picks, size, p3b=False)
    leaves: list[Path] = []
    for doc in train.values():
        leaves.extend(
            expected.treatment_leaf_dirs(doc, cfg.output_root, spec=plan.SPEC)
        )
        leaves.extend(
            expected.xbase_control_leaf_dirs(doc, cfg.output_root, spec=plan.SPEC)
        )
    if expected.missing_manifests(leaves):
        return False
    if _plant_eval_ok(leaves):
        return False
    return True


def _leaves_ready(cfg: driver_io.CampaignDriverConfig, state: dict) -> bool:
    if cfg.dry_run:
        return True
    for size in plan.SIZES:
        if not _ladder_complete(cfg, state, size):
            return False
    picks = _picks_lrs(cfg)
    submitted_p3 = {
        size for size in plan.SIZES if f"submit_p3_{size}" in state["transitions"]
    }
    for size in set(picks) | submitted_p3:
        if not _p3_leaves_ready(cfg, state, size):
            return False
    return True


def _can_finalize(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    waiting: list[str],
) -> bool:
    if "submit_p2" not in state["transitions"]:
        return False
    if "finalize" in state["transitions"]:
        return False
    for size in plan.SIZES:
        if f"submit_p3_{size}" not in state["transitions"]:
            if cfg.dry_run:
                continue
            return False
    optional = plan.p3b_queue_names()
    names = [n for n in _submitted_queue_names(cfg) if n not in optional]
    if names and not _queues_ready(cfg, names, waiting):
        return False
    if not _leaves_ready(cfg, state):
        waiting.append("expected leaves incomplete")
        return False
    return True


def _p3b_incomplete(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Submitted P3b queues that have not finished (reported, never blocking)."""
    if cfg.dry_run:
        return []
    pending: list[str] = []
    for name in sorted(plan.p3b_queue_names()):
        if not store.queue_exists(cfg.queue_root, name):
            continue
        finished, _ = driver_io.queue_finished(cfg, name)
        if not finished:
            pending.append(name)
    return pending


def _finalize(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    p3b_incomplete = _p3b_incomplete(cfg)
    written = build_results(cfg, dry_run=cfg.dry_run)
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {plan.RUN_SET_ID} finished {utc_now_iso()}\n"
        )
    state["transitions"]["finalize"] = {
        "at": utc_now_iso(),
        "results": [str(p) for p in written.values()],
        "voided_sizes": list(state.get("voided_sizes") or []),
        "p3b_incomplete": p3b_incomplete,
    }
    if p3b_incomplete:
        logger.warning(
            "P3b still draining ({}) — X3 covers only the P3b leaves already on "
            "disk. Rerun `uv run python -m scripts.run_xsize_campaign results` "
            "after the tail finishes to fold them in.",
            ", ".join(p3b_incomplete),
        )
    logger.info("Campaign {} COMPLETE — marker at {}", plan.RUN_SET_ID, marker)


def _raise_if_nothing_left(state: dict, err: CampaignGateError | None) -> None:
    """A voided size is logged and skipped; only an all-void gate halts `watch`.

    The gate report is already on disk and `voided_sizes` persisted, so the
    surviving sizes proceed on this and every later advance without an operator.
    """
    if err is None:
        return
    if _voided(state) >= set(plan.SIZES):
        raise err
    logger.error(
        "{} — continuing with {}", err, sorted(set(plan.SIZES) - _voided(state))
    )


def _p2_unlocked(state: dict) -> bool:
    if "teacher_gate" not in state["transitions"]:
        return False
    if state.get("fallback_pending"):
        return "teacher_gate_regate" in state["transitions"]
    if "submit_p1_fallback" in state["transitions"]:
        return "teacher_gate_regate" in state["transitions"]
    return True


def run(
    cfg: driver_io.CampaignDriverConfig,
    options: XsizeDriverOptions | None = None,
) -> AdvanceResult:
    """One advance pass over the cross-size DAG."""
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(
            f"run_set_id {cfg.run_set_id!r} is malformed — every sweep would "
            "reject it at startup, so nothing is submitted. Expected "
            "{YYYYMMDD}_{purpose-slug}_{size} with hyphens (not underscores) "
            "inside the slug."
        )
    if cfg.spec is not plan.SPEC and cfg.spec.run_set_id != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"xsize driver bound to wrong spec run_set_id="
            f"{cfg.spec.run_set_id!r}; expected {plan.RUN_SET_ID!r}. Pass "
            "spec=xsize_plan.SPEC via driver_config()."
        )
    opts = options or XsizeDriverOptions()
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    state.setdefault("voided_sizes", [])
    state.setdefault("fallback_pending", {})
    state.setdefault("fallback_completed", [])
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

    if "preflight" in transitions and "submit_p1" not in transitions:
        _submit_p1(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p1")

    if "submit_p1" in transitions and "teacher_gate" not in transitions:
        if _queues_ready(cfg, [plan.Q_P1_TEACHERS, plan.Q_P1_TEACHER_EVALS], waiting):
            void_err: CampaignGateError | None = None
            try:
                run_teacher_gate(cfg, state)
            except CampaignGateError as err:
                void_err = err
            driver_io.save_state(cfg, state)
            fired.append("teacher_gate")
            if state.get("fallback_pending"):
                _submit_fallback(cfg, state)
                driver_io.save_state(cfg, state)
                fired.append("submit_p1_fallback")
            _raise_if_nothing_left(state, void_err)

    if state.get("fallback_pending") and "submit_p1_fallback" not in transitions:
        _submit_fallback(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p1_fallback")

    if "submit_p1_fallback" in transitions and "teacher_gate_regate" not in transitions:
        if _queues_ready(cfg, [Q_P1_FALLBACK, Q_P1_FALLBACK_EVALS], waiting):
            regate_err: CampaignGateError | None = None
            try:
                _regate_after_fallback(cfg, state)
            except CampaignGateError as err:
                regate_err = err
            driver_io.save_state(cfg, state)
            fired.append("teacher_gate_regate")
            _raise_if_nothing_left(state, regate_err)

    if _p2_unlocked(state) and "submit_p2" not in transitions:
        _submit_p2(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p2")

    if "submit_p2" in transitions:
        for size in plan.SIZES:
            tkey = f"submit_p3_{size}"
            if tkey in transitions:
                continue
            if cfg.dry_run or _ladder_complete(cfg, state, size):
                _submit_p3(cfg, state, size)
                driver_io.save_state(cfg, state)
                fired.append(tkey)
            else:
                waiting.append(f"ladder incomplete for {size}")

    if _can_finalize(cfg, state, waiting):
        _finalize(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("finalize")

    return AdvanceResult(
        tuple(fired), tuple(waiting), "finalize" in state["transitions"]
    )


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: phases, queues, picks, gates, voided sizes."""
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in cfg.queue_priorities
        if store.queue_exists(cfg.queue_root, name)
    }
    picks_doc = _load_picks_doc(cfg)
    gate_path = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    gate_doc = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
    return {
        "transitions": sorted(state["transitions"]),
        "queues": queues,
        "picks": picks_doc.get("picks") if picks_doc else {},
        "gate_decisions": gate_doc.get("decisions") or {},
        "voided_sizes": list(state.get("voided_sizes") or []),
        "fallback_pending": dict(state.get("fallback_pending") or {}),
        "done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }
