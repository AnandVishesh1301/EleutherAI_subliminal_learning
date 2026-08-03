"""Automated stage machine for the 20260801 optimizer x parameterization campaign.

One `run(cfg)` = one advance pass over the phase DAG:

    submit_p0   -> Phase-0 calibration queue (50 cells, 200 steps)
    submit_p1   -> Phase-0 queue done -> disk gate -> `calibrate_arms` (divergence
                   exclusion, loss-match band, edge-of-ladder raise) -> generate + submit
                   the per-arm ladder, the control floor, and the dependent eval queue
    submit_p2   -> Phase-1 queues done -> disk + summary gate -> `screen_rungs`
                   (ok / advisory / quarantine) -> control-floor gate on surviving rungs
                   -> per-arm LR selection (skips quarantined) -> generate + submit the
                   headline, loss-matched, control, and dependent eval queues for
                   surviving arms only
    finalize    -> everything done -> disk + summary gate -> results CSVs -> CAMPAIGN_DONE

Same three guards per transition as the 20260727 driver (`student_hp_campaign_pipeline`),
whose `driver_io` / `expected` machinery this reuses: the phase's queues must be complete
with zero failed jobs (GKD train/eval sweeps exit non-zero when any cell fails, so
`jobq requeue` sees them), the previous phase's cells must all have a
`train_manifest.json` on disk (belt-and-suspenders against a false-green result), and
the phase's selection gate must pass. Re-running is a no-op per already-fired
transition, so this is safe to poll. Every hard failure raises `CampaignGateError`
carrying its own remediation.

The two selection gates are the campaign's scientific content, not bookkeeping: Phase 0
is what makes the SGD arms interpretable at all (Blank et al. ran every optimizer at
Adam's LR), and the control floor is what licenses the 1-gseed controls on the off-headline
ladder rungs. Pathological Phase-1 rungs are quarantined (reported, kept on disk) rather
than halting the campaign; only a zero-usable-arm outcome or an unusable loss-match
reference still raises.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import yaml
from loguru import logger

from pipelines.optparam_results_pipeline import build_results
from subliminal_learning.campaign import driver_io, expected
from subliminal_learning.campaign import optparam_config_gen as cgen
from subliminal_learning.campaign import optparam_jobs as jobs
from subliminal_learning.campaign import optparam_plan as plan
from subliminal_learning.campaign.optparam_plan import RungTier
from subliminal_learning.campaign.optparam_selection import (
    ArmCalibration,
    LadderTooNarrow,
    LossMatchOutOfBand,
    calibrate_arms,
    calibration_report,
    lr_selection_report,
    quarantine_report,
    quarantined_pairs,
    screen_rungs,
    select_ladder_lrs,
    surviving_lrs_by_arm,
)
from subliminal_learning.campaign.selection import FloorArm, control_floor_check
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_preference_results,
    load_gkd_probe_trajectories,
    load_gkd_train_manifests,
    load_gkd_training_logs,
    subliminal_delta,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import (
    git_sha,
    read_run_set_card,
    reconcile_run_set_sha,
    utc_now_iso,
)


class CampaignGateError(RuntimeError):
    """A phase gate failed hard (not merely 'still running'); do not advance."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass.

    Attributes:
        fired: Transitions that executed this pass.
        waiting: Human-readable reasons the next transition is not ready.
        done: The campaign has fully finished.
    """

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to this campaign's identity and queue priorities."""
    return driver_io.CampaignDriverConfig(
        run_set_id=plan.RUN_SET_ID,
        queue_priorities=dict(plan.QUEUE_PRIORITIES),
        **kwargs,
    )


# ---- shared gates ----------------------------------------------------------------


def _generated_doc(cfg: driver_io.CampaignDriverConfig, fname: str) -> dict:
    return yaml.safe_load((cfg.generated_dir / fname).read_text())


def _require_disk_complete(label: str, leaves, remediation: str) -> None:
    missing = expected.missing_manifests(leaves)
    if missing:
        raise CampaignGateError(
            f"{label}: {len(missing)}/{len(leaves)} cells have no completed "
            f"train_manifest.json (first: {missing[0]}). {remediation}"
        )


def _require_summaries(label: str, config_dirs, filename: str) -> None:
    missing = [d for d in config_dirs if not (d / filename).exists()]
    if missing:
        raise CampaignGateError(
            f"{label}: {len(missing)}/{len(config_dirs)} configs missing {filename} "
            f"(first: {missing[0]}) — evals incomplete; requeue the eval queue and "
            "re-advance."
        )


def _trained_and_evaluated(
    cfg: driver_io.CampaignDriverConfig,
    label: str,
    fnames: Sequence[str],
    queue: str,
    *,
    controls: bool,
    evals: bool,
    control_gseeds: Sequence[int] | None = None,
) -> None:
    """Assert every cell of each generated YAML is trained (and optionally evaluated).

    `control_gseeds` narrows the control enumeration for a phase whose controls run on
    fewer student seeds than its treatments (Phase 1's floor is 1-gseed by design), so the
    gate demands exactly the control leaves that phase's jobs create.
    """
    for fname in sorted(fnames):
        doc = _generated_doc(cfg, fname)
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root)
        if controls:
            control_doc = doc
            if control_gseeds is not None:
                control_doc = {**doc, "gseeds": [int(g) for g in control_gseeds]}
            leaves = leaves + expected.control_leaf_dirs(control_doc, cfg.output_root)
        _require_disk_complete(
            f"{label} {fname}",
            leaves,
            f"Requeue {queue} failures, then re-advance.",
        )
        if evals:
            _require_summaries(
                f"{label} {fname}",
                sorted({leaf.parent for leaf in leaves}),
                "preference_summary.json",
            )


def _sha_drift_gate(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Detect (and optionally accept) HEAD moving away from the campaign's commit.

    Checked before the queue gates because the downstream symptom -- every newly launched
    cell exiting non-zero on the run-set staleness guard -- names a failed queue, not the
    commit that caused it.

    Returns:
        A note for the advance result, or ``None`` when there is no drift.

    Raises:
        CampaignGateError: HEAD drifted and `auto_reconcile_sha` is off.
    """
    card = read_run_set_card(cfg.run_set_root)
    if card is None:
        return None
    head = git_sha()
    if head is None or card.get("git_sha") == head:
        return None
    if not cfg.auto_reconcile_sha:
        raise CampaignGateError(
            f"git HEAD ({head}) has moved away from the campaign's commit "
            f"({card.get('git_sha')}) — every newly launched cell will abort on the "
            f"run-set staleness guard. If the change is compatible, run "
            f"`uv run python -m scripts.reconcile_run_set --workflow "
            f"{GKD_STUDENTS_SUBDIR} --run-set-id {plan.RUN_SET_ID} --reason ...`, "
            f"requeue the failed jobs, and re-advance (or pass --auto-reconcile-sha)."
        )
    if cfg.dry_run:
        return f"would reconcile git_sha {card.get('git_sha')} -> {head}"
    reconcile_run_set_sha(
        cfg.run_set_root,
        new_git_sha=head,
        reason="optparam campaign driver --auto-reconcile-sha",
    )
    return f"reconciled git_sha {card.get('git_sha')} -> {head}"


# ---- transition 1: submit Phase 0 --------------------------------------------------


def _submit_p0(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    train_cfgs = cgen.p0_train_configs()
    driver_io.write_generated_configs(cfg, train_cfgs)
    p0_jobs = jobs.p0_jobs(train_cfgs)
    if len(p0_jobs) != plan.PHASE0_CELLS:
        raise CampaignGateError(
            f"Phase 0 built {len(p0_jobs)} shards but the plan declares "
            f"{plan.PHASE0_CELLS} ladder rungs — the generator and "
            "optparam_plan.PHASE0_LADDERS disagree; fix before submitting."
        )
    driver_io.submit_queue(cfg, plan.Q_P0, p0_jobs)
    state["transitions"]["submit_p0"] = {
        "at": utc_now_iso(),
        "train_yamls": sorted(train_cfgs),
        "n_cells": len(p0_jobs),
    }


# ---- transition 2: Phase 0 done -> calibrate -> submit Phase 1 ---------------------


def _calibrate(
    cfg: driver_io.CampaignDriverConfig,
) -> tuple[dict[str, ArmCalibration], float]:
    manifests = load_gkd_train_manifests(cfg.output_root, cfg.campaign_subdir)
    if manifests.empty:
        raise CampaignGateError(
            f"no train_manifest.json found under {cfg.run_set_root} despite a complete "
            f"{plan.Q_P0} queue"
        )
    probes = load_gkd_probe_trajectories(cfg.output_root, cfg.campaign_subdir)
    logs = load_gkd_training_logs(cfg.output_root, cfg.campaign_subdir)
    try:
        calibrations = calibrate_arms(manifests, probes, logs)
    except (LadderTooNarrow, LossMatchOutOfBand) as err:
        raise CampaignGateError(
            f"Phase-0 calibration gate FAILED: {err}\nEdit "
            "src/subliminal_learning/campaign/optparam_plan.py PHASE0_LADDERS, delete "
            f"the affected {cgen.P0_PREFIX}*.yaml under {cfg.generated_dir}, drop the "
            f"{plan.Q_P0} queue, and re-advance to re-screen that arm."
        ) from err
    ref_loss = _reference_loss(calibrations)
    return calibrations, ref_loss


def _reference_loss(calibrations: dict[str, ArmCalibration]) -> float:
    """The loss-match reference arm's measured final train loss (for the report)."""
    key, lr = plan.LOSS_MATCH_REFERENCE
    arm = next(a for a in plan.ARMS if a.ladder_key == key)
    rung = next(
        r for r in calibrations[arm.key].rungs if math.isclose(r.lr, lr, rel_tol=1e-9)
    )
    return float(rung.final_train_loss)


def _submit_p1(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _trained_and_evaluated(
        cfg,
        "Phase 0",
        state["transitions"]["submit_p0"]["train_yamls"],
        plan.Q_P0,
        controls=False,
        evals=False,
    )
    calibrations, ref_loss = _calibrate(cfg)
    for cal in sorted(calibrations.values(), key=lambda c: c.arm_key):
        if cal.lossmatch_infeasible:
            logger.warning(
                "Phase 0 {}: best lr {:g}, NO loss-matched rung — closest is lr {:g} at "
                "{:.1%} off the reference loss (tolerance {:.0%}), so this arm carries "
                "no Phase-2b tier and must be reported as 'no rung reaches the "
                "reference loss'. Ladder {}",
                cal.arm_key,
                cal.lr_best,
                cal.lossmatch_closest_lr,
                cal.lossmatch_rel_error,
                plan.LOSS_MATCH_TOL,
                [f"{x:g}" for x in cal.ladder],
            )
            continue
        logger.info(
            "Phase 0 {}: best lr {:g}, loss-matched {:g} ({:.1%} off), ladder {}",
            cal.arm_key,
            cal.lr_best,
            cal.lr_lossmatch,
            cal.lossmatch_rel_error,
            [f"{x:g}" for x in cal.ladder],
        )
    driver_io.write_logs_report(
        cfg,
        plan.CALIBRATION_REPORT,
        {
            "at": utc_now_iso(),
            **calibration_report(calibrations, reference_loss=ref_loss),
        },
    )
    train_cfgs = cgen.p1_train_configs(calibrations)
    verify_cfgs = cgen.p1_verify_configs(train_cfgs)
    driver_io.write_generated_configs(cfg, {**train_cfgs, **verify_cfgs})
    driver_io.submit_queue(cfg, plan.Q_P1_LADDER, jobs.p1_ladder_jobs(train_cfgs))
    driver_io.submit_queue(cfg, plan.Q_P1_FLOOR, jobs.p1_floor_jobs(train_cfgs))
    driver_io.submit_queue(
        cfg,
        plan.Q_P1_EVALS,
        jobs.p1_eval_jobs(verify_cfgs),
        depends_on=[plan.Q_P1_LADDER, plan.Q_P1_FLOOR],
    )
    state["transitions"]["submit_p1"] = {
        "at": utc_now_iso(),
        "reference_final_train_loss": ref_loss,
        "calibration": {k: v.to_dict() for k, v in sorted(calibrations.items())},
        "train_yamls": sorted(train_cfgs),
        "verify_yamls": sorted(verify_cfgs),
    }


# ---- transition 3: Phase 1 done -> disposition + floor gate + LR select -> P2 ----


def _floor_gate(
    cfg: driver_io.CampaignDriverConfig,
    probes,
    ladders: dict[str, list],
    *,
    quarantined: set[tuple[str, float]],
) -> dict:
    if probes.empty:
        raise CampaignGateError(
            f"no training_metrics.jsonl under {cfg.run_set_root} despite complete "
            "Phase-1 queues"
        )
    arms = [
        FloorArm(
            student_mode="full" if plan.ARMS_BY_KEY[key].kind == "full" else "lora",
            lrs=tuple(float(x) for x in lrs),
            student_optim=plan.ARMS_BY_KEY[key].optim,
            optim_momentum=plan.ARMS_BY_KEY[key].momentum,
        )
        for key, lrs in sorted(ladders.items())
    ]
    floor = control_floor_check(
        probes,
        animals=plan.ANIMALS,
        betas=[plan.BETA],
        max_floor=plan.CONTROL_FLOOR_MAX,
        max_spread=plan.CONTROL_FLOOR_MAX_SPREAD,
        arms=arms,
        quarantined=quarantined,
    )
    report = {
        "at": utc_now_iso(),
        "max_floor": plan.CONTROL_FLOOR_MAX,
        "max_spread": plan.CONTROL_FLOOR_MAX_SPREAD,
        "passed": bool(floor.passed),
        "failures": list(floor.failures),
        "floors": floor.floors.to_dict(orient="records"),
        "quarantined": [
            {"arm": a, "lr": lr}
            for a, lr in sorted(quarantined, key=lambda t: (t[0], t[1]))
        ],
    }
    driver_io.write_logs_report(cfg, plan.CONTROL_FLOOR_REPORT, report)
    if not floor.passed:
        raise CampaignGateError(
            "Phase-1 control-floor gate FAILED after quarantine — the surviving "
            "base-teacher floors are not flat across this campaign's LR ladders, so a "
            "Phase-2 delta could not be attributed to the teacher's trait. Reasons: "
            + "; ".join(floor.failures)
        )
    return report


def _submit_p2(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    tr = state["transitions"]["submit_p1"]
    _trained_and_evaluated(
        cfg,
        "Phase 1",
        tr["train_yamls"],
        plan.Q_P1_LADDER,
        controls=True,
        evals=True,
        control_gseeds=plan.CONTROL_FLOOR_GSEEDS,
    )
    ladders = {k: v["phase1_ladder"] for k, v in tr["calibration"].items()}
    lossmatch_lrs = {k: v["lr_lossmatch"] for k, v in tr["calibration"].items()}
    calibrations = {
        k: ArmCalibration(
            arm_key=k,
            rungs=(),
            usable_lrs=tuple(v["usable_lrs"]),
            lr_best=float(v["lr_best"]),
            lr_lossmatch=(
                None if v["lr_lossmatch"] is None else float(v["lr_lossmatch"])
            ),
            lossmatch_closest_lr=float(v["lossmatch_closest_lr"]),
            lossmatch_rel_error=float(v["lossmatch_rel_error"]),
            ladder=tuple(v["phase1_ladder"]),
        )
        for k, v in tr["calibration"].items()
    }
    probes = load_gkd_probe_trajectories(cfg.output_root, cfg.campaign_subdir)
    dispositions = screen_rungs(probes, ladders, calibrations=calibrations)
    quarantined = quarantined_pairs(dispositions)
    surviving = surviving_lrs_by_arm(dispositions)
    dropped = sorted(k for k, lrs in surviving.items() if not lrs)
    for d in sorted(dispositions.values(), key=lambda x: (x.arm_key, x.lr)):
        if d.tier == RungTier.OK:
            continue
        logger.warning(
            "Phase-1 {}: {} @ lr={:g} reasons={}",
            d.tier,
            d.arm_key,
            d.lr,
            list(d.reasons),
        )
    for arm_key in dropped:
        logger.warning(
            "Phase-1 arm {} dropped — every Phase-1 rung quarantined; no Phase-2 cells",
            arm_key,
        )
    driver_io.write_logs_report(
        cfg,
        plan.QUARANTINE_REPORT,
        {
            "at": utc_now_iso(),
            **quarantine_report(
                dispositions, surviving=surviving, dropped_arms=dropped
            ),
        },
    )
    for arm_key, lr in sorted(lossmatch_lrs.items()):
        if lr is not None and (arm_key, float(lr)) in quarantined:
            lossmatch_lrs[arm_key] = None
            logger.warning(
                "Phase-1 {}: loss-matched lr {:g} is quarantined — dropping this arm's "
                "Phase-2b tier rather than training a known-broken cell",
                arm_key,
                float(lr),
            )
    if len(dropped) == len(plan.ARMS):
        raise CampaignGateError(
            "Phase-1 disposition left ZERO usable arms — every rung of every arm was "
            "quarantined. Inspect logs/quarantine_report.json; widen ladders or fix "
            "training before re-advancing."
        )
    _floor_gate(cfg, probes, ladders, quarantined=quarantined)
    preference = load_gkd_preference_results(cfg.output_root, cfg.campaign_subdir)
    if preference.empty:
        raise CampaignGateError(
            f"no preference_summary.json under {cfg.run_set_root} despite a complete "
            f"{plan.Q_P1_EVALS} queue"
        )
    try:
        selections = select_ladder_lrs(
            subliminal_delta(preference),
            ladders,
            lossmatch_lrs,
            quarantined=quarantined,
        )
    except ValueError as err:
        raise CampaignGateError(
            f"Phase-1 LR selection FAILED: {err}. Requeue {plan.Q_P1_EVALS} and "
            "re-advance."
        ) from err
    for arm_key in sorted(set(surviving) - set(selections) - set(dropped)):
        logger.warning(
            "Phase-1 arm {} omitted from selection (no complete animal coverage on "
            "surviving rungs)",
            arm_key,
        )
        dropped = sorted({*dropped, arm_key})
    if not selections:
        raise CampaignGateError(
            "Phase-1 LR selection left ZERO usable arms after quarantine. Inspect "
            "logs/quarantine_report.json and requeue incomplete evals."
        )
    for sel in sorted(selections.values(), key=lambda s: s.arm_key):
        logger.info(
            "Phase 1 {}: headline lr {:g} (delta {:+.4f}){}",
            sel.arm_key,
            sel.lr,
            sel.mean_delta,
            f", loss-matched arm at {sel.lr_lossmatch:g}"
            if sel.needs_lossmatch_arm
            else "",
        )
    driver_io.write_logs_report(
        cfg,
        plan.LR_SELECTION_REPORT,
        {"at": utc_now_iso(), **lr_selection_report(selections)},
    )
    headline, lossmatch = cgen.p2_train_configs(selections)
    verify_cfgs = cgen.p2_verify_configs(headline, lossmatch, selections)
    driver_io.write_generated_configs(cfg, {**headline, **lossmatch, **verify_cfgs})
    for queue, queue_jobs in jobs.p2_treatment_jobs(headline, lossmatch):
        if not queue_jobs:
            raise CampaignGateError(
                f"Phase-2 queue {queue} built zero jobs despite non-empty selections — "
                "refusing empty submit"
            )
        driver_io.submit_queue(cfg, queue, queue_jobs)
    control_jobs = jobs.p2_control_jobs(headline, lossmatch)
    if not control_jobs:
        raise CampaignGateError(
            "Phase-2 control queue built zero jobs despite non-empty selections — "
            "refusing empty submit"
        )
    driver_io.submit_queue(cfg, plan.Q_P2_CONTROLS, control_jobs)
    eval_jobs = jobs.p2_eval_jobs(verify_cfgs)
    if not eval_jobs:
        raise CampaignGateError(
            "Phase-2 eval queue built zero jobs despite non-empty selections — "
            "refusing empty submit"
        )
    driver_io.submit_queue(
        cfg,
        plan.Q_P2_EVALS,
        eval_jobs,
        depends_on=_p2_train_queues(bool(lossmatch)),
    )
    state["transitions"]["submit_p2"] = {
        "at": utc_now_iso(),
        "selection": {k: v.to_dict() for k, v in sorted(selections.items())},
        "quarantined": [
            {"arm": a, "lr": lr}
            for a, lr in sorted(quarantined, key=lambda t: (t[0], t[1]))
        ],
        "dropped_arms": list(dropped),
        "surviving_lrs": {k: list(v) for k, v in sorted(surviving.items())},
        "headline_yamls": sorted(headline),
        "lossmatch_yamls": sorted(lossmatch),
        "verify_yamls": sorted(verify_cfgs),
    }


def _p2_train_queues(has_lossmatch: bool) -> list[str]:
    """The Phase-2 training queues that exist (the loss-matched tier can be empty)."""
    queues = [plan.Q_P2_HEADLINE, plan.Q_P2_CONTROLS]
    if has_lossmatch:
        queues.append(plan.Q_P2_LOSSMATCH)
    return queues


# ---- transition 4: everything done -> results + marker -----------------------------


def _finalize(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    tr = state["transitions"]["submit_p2"]
    _trained_and_evaluated(
        cfg,
        "Phase 2 headline",
        tr["headline_yamls"],
        plan.Q_P2_HEADLINE,
        controls=True,
        evals=True,
    )
    _trained_and_evaluated(
        cfg,
        "Phase 2 loss-matched",
        tr["lossmatch_yamls"],
        plan.Q_P2_LOSSMATCH,
        controls=True,
        evals=True,
    )
    written = build_results(cfg.output_root, cfg.campaign_subdir, dry_run=cfg.dry_run)
    logger.info("Wrote {} result artifacts under {}", len(written), cfg.run_set_root)
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {plan.RUN_SET_ID} finished {utc_now_iso()}\n"
        )
    state["transitions"]["finalize"] = {
        "at": utc_now_iso(),
        "results": [str(p) for p in written],
    }
    logger.info("Campaign {} COMPLETE — marker at {}", plan.RUN_SET_ID, marker)


# ---- the advance pass ---------------------------------------------------------------


def _queues_ready(
    cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]
) -> bool:
    ready = True
    for name in names:
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ready = False
        if "FAILED" in msg:
            raise CampaignGateError(msg)
    return ready


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass: fire every transition whose gates pass, then report.

    Returns:
        `AdvanceResult` — transitions fired, wait reasons, and campaign-done.

    Raises:
        CampaignGateError: A gate failed hard (failed jobs, missing artifacts, a
            too-narrow ladder, an out-of-band loss match, a non-flat control floor).
    """
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []

    drift = _sha_drift_gate(cfg)
    if drift is not None:
        fired.append(drift)

    if "submit_p0" not in transitions:
        _submit_p0(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p0")

    if "submit_p0" in transitions and "submit_p1" not in transitions:
        if _queues_ready(cfg, [plan.Q_P0], waiting):
            _submit_p1(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("submit_p1")

    if "submit_p1" in transitions and "submit_p2" not in transitions:
        p1_queues = [plan.Q_P1_LADDER, plan.Q_P1_FLOOR, plan.Q_P1_EVALS]
        if _queues_ready(cfg, p1_queues, waiting):
            _submit_p2(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("submit_p2")

    if "submit_p2" in transitions and "finalize" not in transitions:
        final = _p2_train_queues(bool(transitions["submit_p2"]["lossmatch_yamls"])) + [
            plan.Q_P2_EVALS
        ]
        if _queues_ready(cfg, final, waiting):
            _finalize(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("finalize")

    return AdvanceResult(
        tuple(fired), tuple(waiting), "finalize" in state["transitions"]
    )


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: per-queue status + which transitions have fired."""
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in plan.QUEUE_PRIORITIES
        if store.queue_exists(cfg.queue_root, name)
    }
    return {"transitions": sorted(state["transitions"]), "queues": queues}
