"""Auto-advancing driver of the `rebhard` campaign, on the login node.

One `run(cfg)` = one idempotent advance pass:

    preflight   prereg, staging report, approved pool review, disk
    register    run-set card pinned to the campaign code pin
    submit      one unpinned queue per experiment (any pool drains it)
    (every pass) queues follow the pin; failures under the cap are requeued, the rest
                recorded as holes
    results_e*  an experiment's tables and claims once its queue is finished
    finalize    CAMPAIGN_DONE once every experiment has results, no holes remain and
                every priority is in the code
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from pipelines import rebhard_results_pipeline as results
from pipelines import rebhard_staging_pipeline as staging
from subliminal_learning.campaign import driver_io, xfam2_plan
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import build_run_set_card
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import (
    read_run_set_card,
    register_run_set_launch,
    utc_now_iso,
)

SPEC = replace(
    xfam2_plan.SPEC,
    run_set_id=plan.RUN_SET_ID,
    teacher_run_set_id=plan.RUN_SET_ID,
    root_subdir=plan.RUN_SET_SUBDIR,
    queue_priorities=plan.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": plan.CAP_PER_GPU},
    generated_config_dir=str(plan.GENERATED_DIR),
    campaign_state_file=plan.CAMPAIGN_STATE_FILE,
    campaign_done_file=plan.CAMPAIGN_DONE_FILE,
    extra_pin_paths=(
        "scripts/run_rebhard_campaign.py",
        "scripts/run_rebhard_cell.py",
        "scripts/stage_rebhard_assets.py",
        "src/subliminal_learning/campaign/rebhard_plan.py",
        "src/subliminal_learning/campaign/rebhard_cells.py",
        "src/subliminal_learning/campaign/rebhard_metrics.py",
        "pipelines/rebhard_campaign_pipeline.py",
        "pipelines/rebhard_cell_pipeline.py",
        "pipelines/rebhard_results_pipeline.py",
        "pipelines/rebhard_staging_pipeline.py",
    ),
)


class CampaignGateError(RuntimeError):
    """A campaign contract refused to pass."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the rebhard spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def _fire(state: dict, name: str, **info: object) -> None:
    state["transitions"][name] = {"at": utc_now_iso(), **info}


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    import json

    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        raise CampaignGateError(f"missing pre-registration {plan.PREREG_PATH}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} is not the plan's")
    report = staging.latest_complete_report()
    if report is None:
        raise CampaignGateError("no complete staging report; run stage_rebhard_assets")
    path = plan.POOL_REVIEW_PATH
    review = json.loads(path.read_text()) if path.is_file() else None
    problem = staging.pool_review_problem(review)
    if problem:
        raise CampaignGateError(f"{path}: {problem}")
    free_gib = shutil.disk_usage(plan.ROOT).free / 2**30
    if free_gib < plan.MIN_DISK_FREE_GIB:
        raise CampaignGateError(f"{plan.ROOT}: {free_gib:.0f} GiB free")
    _fire(state, "preflight", staging_report_at=report["at"])


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = code_pin_io.resolve_campaign_pin(
        cfg.run_set_root, repo_root=cfg.repo_root, requested_sha=cfg.code_sha
    )
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    card = build_run_set_card(
        run_set_id=plan.RUN_SET_ID,
        run_type="offpolicy_student",
        purpose="hard-label rebuttal students: trees and colors, full FT, 410M",
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/rebhard_plan.py",
        resolved_config={"root": str(plan.ROOT), "experiments": list(plan.EXPERIMENTS)},
        grid_axes={"n_cells": len(cells.cells_by_key())},
        model_size="mixed",
    )
    register_run_set_launch(cfg.output_root, cfg.run_set_root, card)
    _fire(state, "register", git_sha=pin.sha)


def _submit_all(cfg: driver_io.CampaignDriverConfig) -> None:
    """Create each queue, or move an existing one to the card's pin."""
    for exp in plan.EXPERIMENTS:
        driver_io.submit_queue(
            cfg, plan.queue(exp), cells.queue_jobs(exp), unpinned=True
        )


def _requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    holes = state.setdefault("holes", {})
    for exp in plan.EXPERIMENTS:
        name = plan.queue(exp)
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if holes.get(name) != str(err):
                logger.error("hole in {}: {}", name, err)
            holes[name] = str(err)
        else:
            holes.pop(name, None)


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass."""
    if not cfg.dry_run:
        driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    done = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []

    def mark(name: str) -> None:
        fired.append(name)
        driver_io.save_state(cfg, state)

    if "preflight" not in done:
        _preflight(cfg, state)
        mark("preflight")
    if "register" not in done:
        _register(cfg, state)
        mark("register")
    _submit_all(cfg)
    if "submit" not in done:
        _fire(state, "submit", n_cells=len(cells.cells_by_key()))
        mark("submit")
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    for exp in plan.EXPERIMENTS:
        name = f"results_{exp}"
        if name in done:
            continue
        ok, msg = driver_io.queue_finished(cfg, plan.queue(exp))
        if not ok:
            waiting.append(msg)
        elif not cfg.dry_run:
            out = results.build(exp, cfg.run_set_root / plan.RESULTS_DIRNAME)
            _fire(state, name, build=str(out))
            mark(name)
    finished = all(f"results_{e}" in done for e in plan.EXPERIMENTS)
    all_done = finished and plan.ALL_PHASES_IN_CODE and not state.get("holes")
    if all_done and "finalize" not in done and not cfg.dry_run:
        atomic_write_text(cfg.logs_dir / plan.CAMPAIGN_DONE_FILE, utc_now_iso() + "\n")
        _fire(state, "finalize")
        mark("finalize")
    if state.get("holes"):
        waiting.append(f"holes in {sorted(state['holes'])}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes and per-queue counts (pure file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for exp in plan.EXPERIMENTS:
        name = plan.queue(exp)
        if store.queue_exists(cfg.queue_root, name):
            s = store.queue_status(cfg.queue_root, name)
            queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "queues": queues,
        "card_pin": (read_run_set_card(cfg.run_set_root) or {}).get("git_sha"),
    }
