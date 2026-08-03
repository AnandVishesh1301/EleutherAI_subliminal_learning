"""Automated stage machine for the 20260727 student-HP + cross-init campaign.

One `run(cfg)` = one advance pass over the stage DAG:

    A0 gate -> submit A1 + A2 + A1-evals + catswide backlog
    A1 + A2 queues done -> disk-complete gate -> A2 control-floor gate ->
        top-3 selection -> generate + submit A3 (+ A3-evals, dependent)
    A3 + A3-evals done -> disk + summary gate -> recipe freeze ->
        generate + submit Stage B headline/extension (+ B-evals; optional Stage C)
    everything done -> disk gate -> CAMPAIGN_DONE marker

Each transition is guarded three ways before it fires: its queues must be
complete with ZERO failed jobs, the previous stage's expected artifacts must all
exist on disk (queue exit codes alone are not trusted — a sweep job exits 0 even
when cells failed), and any selection gate (A2 flatness) must pass. A transition
that already ran is a no-op (queue jobkey-set equality + persisted state), so the
driver is safe to re-run or poll in a watch loop. Any hard failure raises
`CampaignGateError` with the exact remediation.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml
from loguru import logger

from subliminal_learning.campaign import config_gen, driver_io, expected, plan
from subliminal_learning.campaign.selection import (
    apply_lr_extensions,
    control_floor_check,
    freeze_recipes,
    headline_beta,
    lr_edge_extensions,
    select_top_recipes,
)
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_preference_results,
    load_gkd_probe_trajectories,
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
    """A stage gate failed hard (not merely 'still running'); do not advance."""


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


def _load_yaml(cfg: driver_io.CampaignDriverConfig, rel: str) -> dict:
    return yaml.safe_load((cfg.repo_root / rel).read_text())


def _require_disk_complete(label: str, leaves, remediation: str) -> None:
    missing = expected.missing_manifests(leaves)
    if missing:
        raise CampaignGateError(
            f"{label}: {len(missing)}/{len(leaves)} cells have no completed "
            f"train_manifest.json (first: {missing[0]}). {remediation}"
        )


def _require_summaries(label: str, config_dirs: list, filename: str) -> None:
    missing = [d for d in config_dirs if not (d / filename).exists()]
    if missing:
        raise CampaignGateError(
            f"{label}: {len(missing)}/{len(config_dirs)} configs missing "
            f"{filename} (first: {missing[0]}) — evals incomplete; requeue the "
            "eval queue and re-run the driver."
        )


def _sha_drift_gate(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Detect (and optionally accept) HEAD moving away from the campaign's commit.

    Checked before the queue gates because the downstream symptom -- every newly
    launched cell exiting non-zero on `check_run_set_staleness` -- names a failed
    queue, not the commit that caused it. Silence is correct before the campaign
    card exists, or when git is unavailable.

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
        reason="campaign driver --auto-reconcile-sha",
    )
    return f"reconciled git_sha {card.get('git_sha')} -> {head}"


# ---- transition 1: A0 gate -> initial queues -------------------------------------


def _a0_gate(cfg: driver_io.CampaignDriverConfig) -> None:
    if not cfg.require_a0:
        return
    missing = [
        f
        for f in (plan.CALIBRATION_REPORT, plan.INIT_EQUALITY_REPORT)
        if not (cfg.logs_dir / f).exists()
    ]
    if missing:
        raise CampaignGateError(
            f"A0 incomplete: missing {missing} under {cfg.logs_dir}. Run "
            "`uv run python -m scripts.calibrate_gkd_throughput` and "
            "`uv run python -m scripts.check_step0_init_equality` first "
            "(or pass --skip-a0 to waive)."
        )


def _submit_initial(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _a0_gate(cfg)
    driver_io.submit_queue(cfg, plan.Q_A1, driver_io.a1_jobs())
    driver_io.submit_queue(cfg, plan.Q_A2, driver_io.a2_jobs())
    driver_io.submit_queue(
        cfg,
        plan.Q_A1_EVALS,
        driver_io.a1_eval_jobs(),
        depends_on=[plan.Q_A1, plan.Q_A2],
    )
    driver_io.submit_queue(cfg, plan.Q_BACKLOG, driver_io.backlog_jobs())
    state["transitions"]["submit_initial"] = {"at": utc_now_iso()}


# ---- transition 2: A1/A2 done -> gate -> select -> A3 ------------------------------


def _a1_a2_disk_gate(cfg: driver_io.CampaignDriverConfig) -> None:
    for label, rel in (
        ("A1 same", plan.A1_SAME_YAML),
        ("A1 cross", plan.A1_CROSS_YAML),
    ):
        doc = _load_yaml(cfg, rel)
        _require_disk_complete(
            label,
            expected.treatment_leaf_dirs(doc, cfg.output_root),
            f"Requeue {plan.Q_A1} failures or rerun its jobs, then re-advance.",
        )
    a2_doc = _load_yaml(cfg, plan.A1_SAME_YAML)
    a2_doc["grid"] = {**a2_doc["grid"], "student_modes": [plan.A2_STUDENT_MODE]}
    _require_disk_complete(
        "A2 controls",
        expected.control_leaf_dirs(a2_doc, cfg.output_root),
        f"Requeue {plan.Q_A2} failures, then re-advance.",
    )


def _submit_a3(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _a1_a2_disk_gate(cfg)
    probes = load_gkd_probe_trajectories(cfg.output_root, cfg.campaign_subdir)
    if probes.empty:
        raise CampaignGateError(
            f"no training_metrics.jsonl found under "
            f"{cfg.output_root / cfg.campaign_subdir} despite complete manifests"
        )
    floor = control_floor_check(
        probes,
        animals=plan.ANIMALS,
        betas=plan.BETAS,
        lrs=plan.A1_LORA_LRS,
        max_floor=plan.CONTROL_FLOOR_MAX,
        max_spread=plan.CONTROL_FLOOR_MAX_SPREAD,
    )
    if not floor.passed:
        raise CampaignGateError(
            "A2 control-floor gate FAILED — the control-free A1 screen is not "
            "trustworthy; rerun A1 with controls before dose-laddering. Reasons: "
            + "; ".join(floor.failures)
        )
    selections = select_top_recipes(
        probes, animals=plan.ANIMALS, betas=plan.BETAS, k=plan.A3_TOP_K
    )
    extensions, lr_warnings = lr_edge_extensions(
        selections, lrs=plan.A1_LORA_LRS, factor=plan.A1_LR_EXTENSION_FACTOR
    )
    for msg in lr_warnings:
        logger.warning("A1 lr-grid edge — {}", msg)
    selections = apply_lr_extensions(selections, extensions)
    candidates = {key: sel.recipes for key, sel in selections.items()}
    selection_state = driver_io.selection_to_state(selections)
    driver_io.write_logs_report(
        cfg,
        plan.A1_SELECTION_REPORT,
        {
            "at": utc_now_iso(),
            "top_k": plan.A3_TOP_K,
            "selection": selection_state,
            "lr_edge_warnings": list(lr_warnings),
            "control_floor": floor.floors.to_dict(orient="records"),
        },
    )
    try:
        train_cfgs = config_gen.a3_train_configs(candidates)
    except ValueError as err:
        raise CampaignGateError(
            f"A3 config generation rejected the A1 selection: {err}. The screen "
            "and its gates are intact — inspect "
            f"{cfg.logs_dir / plan.A1_SELECTION_REPORT} and either raise "
            "plan.A3_MAX_SAME_ARMS or lower plan.A3_TOP_K, then re-advance."
        ) from err
    verify_cfgs = config_gen.a3_verify_configs(train_cfgs)
    driver_io.write_generated_configs(cfg, {**train_cfgs, **verify_cfgs})
    headline = config_gen.a3_headline_train_names(train_cfgs)
    driver_io.submit_queue(cfg, plan.Q_A3, driver_io.a3_jobs(train_cfgs, headline))
    driver_io.submit_queue(
        cfg,
        plan.Q_A3_EVALS,
        driver_io.a3_eval_jobs(sorted(verify_cfgs)),
        depends_on=[plan.Q_A3],
    )
    state["transitions"]["submit_a3"] = {
        "at": utc_now_iso(),
        "candidates": driver_io.candidates_to_state(candidates),
        "selection": selection_state,
        "lr_edge_warnings": list(lr_warnings),
        "control_floor": floor.floors.to_dict(orient="records"),
        "train_yamls": sorted(train_cfgs),
        "verify_yamls": sorted(verify_cfgs),
    }


# ---- transition 3: A3 (+evals) done -> freeze -> Stage B ---------------------------


def _a3_disk_gate(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    tr = state["transitions"]["submit_a3"]
    headline = set(config_gen.a3_headline_train_names(dict.fromkeys(tr["train_yamls"])))
    for fname in tr["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root)
        _require_disk_complete(
            f"A3 {fname}",
            leaves,
            f"Requeue {plan.Q_A3} failures, then re-advance.",
        )
        config_dirs = sorted({leaf.parent for leaf in leaves})
        _require_summaries(f"A3 {fname}", config_dirs, "preference_summary.json")
        if fname in headline:
            controls = expected.control_leaf_dirs(doc, cfg.output_root)
            _require_disk_complete(
                f"A3 controls {fname}",
                controls,
                f"Requeue {plan.Q_A3} failures, then re-advance.",
            )
            _require_summaries(
                f"A3 controls {fname}",
                sorted({leaf.parent for leaf in controls}),
                "preference_summary.json",
            )


def _submit_b(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _a3_disk_gate(cfg, state)
    candidates = driver_io.candidates_from_state(
        state["transitions"]["submit_a3"]["candidates"]
    )
    preference = load_gkd_preference_results(cfg.output_root, cfg.campaign_subdir)
    frozen = freeze_recipes(preference, candidates, doses=plan.A3_DOSES)
    beta = headline_beta(frozen)
    winner = frozen[beta]
    logger.info(
        "Recipe freeze: beta={} recipe={} lr={:g} epochs={:g} delta={:.4f}",
        beta,
        winner.recipe.mode_token,
        winner.recipe.lr,
        winner.num_epochs,
        winner.subliminal_delta,
    )
    driver_io.write_logs_report(
        cfg,
        plan.FREEZE_REPORT,
        {
            "at": utc_now_iso(),
            "headline_beta": beta,
            "winner": {
                "recipe": driver_io.recipe_to_dict(winner.recipe),
                "num_epochs": winner.num_epochs,
                "subliminal_delta": winner.subliminal_delta,
                "per_animal_delta": winner.per_animal_delta,
            },
            "frozen": driver_io.frozen_to_state(frozen),
        },
    )
    train_cfgs = config_gen.b_train_configs(winner, beta)
    verify_cfgs = config_gen.b_verify_configs(train_cfgs)
    generated = {**train_cfgs, **verify_cfgs}
    c_train: dict[str, dict] = {}
    c_verify: dict[str, dict] = {}
    if cfg.include_stage_c:
        c_train = config_gen.c_train_configs(winner, beta)
        c_verify = config_gen.c_verify_configs(c_train)
        generated.update({**c_train, **c_verify})
    driver_io.write_generated_configs(cfg, generated)
    control_names = config_gen.b_control_train_names(train_cfgs)
    head_jobs, ext_jobs = driver_io.b_train_jobs(train_cfgs, control_names)
    driver_io.submit_queue(cfg, plan.Q_B_HEADLINE, head_jobs)
    driver_io.submit_queue(cfg, plan.Q_B_EXT, ext_jobs)
    driver_io.submit_queue(
        cfg,
        plan.Q_B_EVALS,
        driver_io.b_eval_jobs(verify_cfgs, train_cfgs),
        depends_on=[plan.Q_B_HEADLINE, plan.Q_B_EXT],
    )
    if cfg.include_stage_c:
        driver_io.submit_queue(cfg, plan.Q_C, driver_io.c_train_jobs(c_train))
        driver_io.submit_queue(
            cfg,
            plan.Q_C_EVALS,
            driver_io.c_eval_jobs(c_verify, c_train),
            depends_on=[plan.Q_C],
        )
    state["transitions"]["submit_b"] = {
        "at": utc_now_iso(),
        "frozen": driver_io.frozen_to_state(frozen),
        "headline_beta": beta,
        "train_yamls": sorted(train_cfgs),
        "verify_yamls": sorted(verify_cfgs),
        "stage_c": sorted(c_train),
    }


# ---- transition 4: everything done -> finalize ------------------------------------


def _finalize(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    tr = state["transitions"]["submit_b"]
    for fname in tr["train_yamls"] + tr.get("stage_c", []):
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root)
        _require_disk_complete(
            f"Stage B/C {fname}",
            leaves,
            "Requeue the factorial queue failures, then re-advance.",
        )
        _require_summaries(
            f"Stage B/C {fname}",
            sorted({leaf.parent for leaf in leaves}),
            "preference_summary.json",
        )
    control_docs = [
        yaml.safe_load((cfg.generated_dir / f).read_text())
        for f in tr["train_yamls"]
        if f.startswith("gkd_b_same_init_")
    ]
    controls: dict = {}
    for doc in control_docs:
        for leaf in expected.control_leaf_dirs(doc, cfg.output_root):
            controls.setdefault(leaf)
    _require_disk_complete(
        "Stage B controls",
        list(controls),
        "Requeue the same_init factorial jobs, then re-advance.",
    )
    _require_summaries(
        "Stage B controls",
        sorted({leaf.parent for leaf in controls}),
        "preference_summary.json",
    )
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {plan.RUN_SET_ID} finished {utc_now_iso()}\n"
        )
    state["transitions"]["finalize"] = {"at": utc_now_iso()}
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
        CampaignGateError: A gate failed hard (failed jobs, missing artifacts,
            A2 flatness failure); the campaign must be fixed before advancing.
    """
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []

    drift = _sha_drift_gate(cfg)
    if drift is not None:
        fired.append(drift)

    if "submit_initial" not in transitions:
        _submit_initial(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_initial")

    if "submit_initial" in transitions and "submit_a3" not in transitions:
        if _queues_ready(cfg, [plan.Q_A1, plan.Q_A2], waiting):
            _submit_a3(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("submit_a3")

    if "submit_a3" in transitions and "submit_b" not in transitions:
        if _queues_ready(cfg, [plan.Q_A3, plan.Q_A3_EVALS], waiting):
            _submit_b(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("submit_b")

    if "submit_b" in transitions and "finalize" not in transitions:
        final_queues = [plan.Q_B_HEADLINE, plan.Q_B_EXT, plan.Q_B_EVALS]
        if cfg.include_stage_c and transitions["submit_b"].get("stage_c"):
            final_queues += [plan.Q_C, plan.Q_C_EVALS]
        if _queues_ready(cfg, final_queues, waiting):
            _finalize(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("finalize")

    done = "finalize" in state["transitions"]
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: per-queue status + which transitions have fired."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in plan.QUEUE_PRIORITIES:
        if store.queue_exists(cfg.queue_root, name):
            queues[name] = store.queue_status(cfg.queue_root, name)
    return {"transitions": sorted(state["transitions"]), "queues": queues}
