"""Auto-advancing driver of the `rebhard` campaign, on the login node.

One `run(cfg)` = one idempotent advance pass:

    preflight   prereg, staging report, approved pool review, disk
    register    run-set card pinned to the campaign code pin
    submit      one unpinned queue per experiment (any pool drains it)
    (every pass) queues follow the pin; failures under the cap are requeued, the rest
                recorded as holes
    results_e*  an experiment's tables and claims once its queues are finished
    p2          #7: teachers, newdom's gate, pools, then each target's students once its
                review is approved; 160M→410M runs on existing pools. #23: 410M→160M at
                once, 160M→410M after #8 has chosen its rates
    p3          #6 on the other targets and #8's momentum ladder at once, #8's extra seeds
                after #8 has chosen its rungs
    finalize    CAMPAIGN_DONE once every experiment has results, no holes remain and
                every priority is in the code
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from pipelines import newdom_hard_cell_pipeline as nd
from pipelines import rebhard_results_pipeline as results
from pipelines import rebhard_staging_pipeline as staging
from pipelines import rebhard_teacher_pipeline as rtp
from subliminal_learning.campaign import driver_io, xfam2_plan
from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_metrics as metrics
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.campaign import rebhard_teachers as rt
from subliminal_learning.campaign import xopt_hard_metrics as xmetrics
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import build_run_set_card
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
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
        "src/subliminal_learning/campaign/rebhard_teachers.py",
        "pipelines/rebhard_campaign_pipeline.py",
        "pipelines/rebhard_cell_pipeline.py",
        "pipelines/rebhard_results_pipeline.py",
        "pipelines/rebhard_staging_pipeline.py",
        "pipelines/rebhard_teacher_pipeline.py",
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
    problem = staging.pool_review_problem(review, staging.p1_pools())
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
    """Create each fixed queue, or move an existing one to the card's pin."""
    for exp in plan.EXPERIMENTS:
        driver_io.submit_queue(
            cfg, plan.queue(exp), cells.queue_jobs(exp), unpinned=True
        )


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in plan.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def _requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    holes = state.setdefault("holes", {})
    for name in _submitted(cfg):
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if holes.get(name) != str(err):
                logger.error("hole in {}: {}", name, err)
            holes[name] = str(err)
        else:
            holes.pop(name, None)


def _put(cfg: driver_io.CampaignDriverConfig, name: str, jobs: list[Job]) -> int:
    """Create unpinned `name` with `jobs`, or append the ones it lacks; return new."""
    if not jobs:
        return 0
    defaults = driver_io.queue_job_defaults(cfg)
    if not store.queue_exists(cfg.queue_root, name):
        driver_io.submit_queue(cfg, name, jobs, unpinned=True, defaults=defaults)
        return len(jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if not cfg.dry_run:
        store.update_queue_defaults(cfg.queue_root, name, defaults)
        if new:
            store.append_jobs(cfg.queue_root, name, new)
            logger.info("appended {} job(s) to {}", len(new), name)
    return len(new)


def _finished(
    cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]
) -> bool:
    ok_all = True
    for name in names:
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ok_all = False
    return ok_all


# ---- priority 2 -------------------------------------------------------------------------


def _p2_gate(state: dict, waiting: list[str]) -> bool:
    """Whether priority 2 may start. A closed gate waits: priority 1 keeps advancing."""
    import json

    if "p2_preflight" in state["transitions"]:
        return True
    report = staging.latest_complete_report()
    n_reused = len(staging.reused_cells())
    if report is None or report["n_reused_controls"] != n_reused:
        waiting.append(f"p2 held: staging report lacks the {n_reused} reused controls")
        return False
    path = plan.P2_POOL_REVIEW_PATH
    review = json.loads(path.read_text()) if path.is_file() else None
    problem = staging.pool_review_problem(review, staging.p2_pools())
    if problem:
        waiting.append(f"p2 held: {path}: {problem}")
        return False
    _fire(state, "p2_preflight", staging_report_at=report["at"])
    return True


def _submit_p2_static(cfg: driver_io.CampaignDriverConfig) -> None:
    """Queues whose jobs are known up front."""
    _put(
        cfg,
        plan.queue(plan.E7_TEACHERS),
        [rt.teacher_job(t) for t in rt.teacher_tasks()],
    )
    up = cells.e7_controls() + cells.e7_up_cells()
    _put(cfg, plan.queue(plan.E7_UP), [cells.student_job(c) for c in up])
    down = cells.e23_down_cells()
    _put(cfg, plan.queue(plan.E23_DOWN), [cells.student_job(c) for c in down])


def _gate_e7(e7: dict) -> None:
    """Rank each target's admitted teachers once."""
    for target in plan.E7_TARGETS:
        if target in e7["gate"]:
            continue
        records = [rtp.teacher_record(target, r) for r in rt.RECIPES]
        if any(rec is None for rec in records):
            raise CampaignGateError(f"{target}: a finished teacher has no record")
        readings = [nd.reading(rec) for rec in records]
        e7["gate"][target] = {
            "ranked": hard.rank_admitted(readings),
            "admission": {r.recipe: list(hard.admission(r)[1]) for r in readings},
        }


def _resolve_pools(cfg: driver_io.CampaignDriverConfig, e7: dict, target: str) -> None:
    """Read the outcome of every pool job that finished."""
    name = plan.queue(plan.E7_POOLS)
    if not store.queue_exists(cfg.queue_root, name):
        return
    done = {
        j.key
        for j in store.load_jobs(cfg.queue_root, name)
        if store.job_state(cfg.queue_root, name, j) == store.DONE
    }
    for recipe, rec in e7["pools"].setdefault(target, {}).items():
        if rec["status"] != "submitted" or rt.PoolTask(target, recipe).key not in done:
            continue
        qc = rtp.pool_status(target, recipe)
        rec.update(n_filtered=qc.n_filtered, keep_rate=qc.keep_rate)
        if qc.ready:
            rec["status"] = hard.USABLE
        elif xmetrics.is_short(qc):
            rec["status"] = hard.SHORT
        else:
            rec["status"] = "invalid"
            rec["reasons"] = list(qc.reasons)


def _advance_target(
    cfg: driver_io.CampaignDriverConfig,
    e7: dict,
    target: str,
    fired: list[str],
    waiting: list[str],
) -> None:
    """Pools of one target, then its students once its review is approved."""
    _resolve_pools(cfg, e7, target)
    if not e7["final"].get(target):
        pools = e7["pools"].setdefault(target, {})
        outcomes = {
            r: rec["status"]
            for r, rec in pools.items()
            if rec["status"] in (hard.USABLE, hard.SHORT)
        }
        ranked = e7["gate"][target]["ranked"]
        decision = hard.pool_plan(ranked, set(pools), outcomes)
        if decision.submit:
            n = _put(
                cfg,
                plan.queue(plan.E7_POOLS),
                [rt.pool_job(rt.PoolTask(target, r)) for r in decision.submit],
            )
            for r in decision.submit:
                pools[r] = {"status": "submitted", "at": utc_now_iso()}
            fired.append(f"pools {target} +{list(decision.submit)} ({n} new)")
            return
        if not decision.final:
            invalid = [r for r, rec in pools.items() if rec["status"] == "invalid"]
            if invalid:
                waiting.append(f"pools {target}: {invalid} invalid, see their reasons")
            return
        e7["final"][target] = True
        e7["selected"][target] = list(decision.selected)
        fired.append(f"final {target}: teachers {list(decision.selected)}")
    chosen = e7["selected"][target]
    if not chosen or e7["submitted"].get(target):
        return
    sheet = rtp.write_pool_samples(target, chosen)
    problem = rtp.pool_review_problem(target, chosen)
    if problem:
        waiting.append(f"students {target} held: {problem} (sheet {sheet})")
        return
    jobs = [cells.student_job(c) for c in cells.e7_down_cells(target, chosen)]
    n = _put(cfg, plan.queue(plan.E7_DOWN), jobs)
    e7["submitted"][target] = utc_now_iso()
    fired.append(f"students {target} {chosen} ({n} cells)")


def _advance_e7(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    fired: list[str],
    waiting: list[str],
) -> None:
    """Move #7 from its finished teachers towards students."""
    e7 = state.setdefault(
        "e7", {"gate": {}, "pools": {}, "final": {}, "selected": {}, "submitted": {}}
    )
    ok, msg = driver_io.queue_finished(cfg, plan.queue(plan.E7_TEACHERS))
    if not ok:
        waiting.append(msg)
        return
    _gate_e7(e7)
    for target in plan.E7_TARGETS:
        _advance_target(cfg, e7, target, fired, waiting)
        if e7["final"].get(target) and not e7["selected"].get(target):
            state.setdefault("holes", {})[f"e7|{target}"] = "no usable 410M teacher"


def _e7_ready(state: dict) -> bool:
    """Every target's selection is final and its students are queued."""
    e7 = state.get("e7") or {}
    return all(
        (e7.get("final") or {}).get(t) and (e7.get("submitted") or {}).get(t)
        for t in plan.E7_TARGETS
    )


def _e7_planned(state: dict) -> list[cells.Cell]:
    selected = state["e7"]["selected"]
    down = [c for t in plan.E7_TARGETS for c in cells.e7_down_cells(t, selected[t])]
    return cells.e7_controls() + cells.e7_up_cells() + down


def choose_e23_rates(rungs: list[dict]) -> tuple[float, ...] | None:
    """The three rates into 410M: #8's best shared rung and its neighbours."""
    ladder = plan.E8_RATES[plan.SGD]
    best = metrics.shared_rung(
        rungs, plan.ANIMALS, plan.SGD, plan.collapse_dce(plan.PY410)
    )
    return None if best is None else metrics.nearest_rates(ladder, best)


def _advance_e23(
    cfg: driver_io.CampaignDriverConfig, state: dict, fired: list[str]
) -> None:
    """Choose the rates into 410M once #8's results exist, then queue those cells."""
    if "e23_rates" in state or "results_e8" not in state["transitions"]:
        return
    rows = results.leaf_rows(cells.exp_cells(plan.E8))
    _, rungs = results.rung_table(rows)
    rates = choose_e23_rates(rungs)
    if rates is None:
        state.setdefault("holes", {})["e23"] = "no non-collapsed SGD rung at 410M"
        return
    if not cfg.dry_run:
        doc = {"at": utc_now_iso(), "rates": list(rates), "rungs": rungs}
        atomic_write_json(cfg.logs_dir / plan.E23_RATES_FILE, doc)
    jobs = [cells.student_job(c) for c in cells.e23_up_cells(rates)]
    n = _put(cfg, plan.queue(plan.E23_UP), jobs)
    state["e23_rates"] = list(rates)
    fired.append(f"e23 rates into 410M {list(rates)} ({n} cells)")


def _p2_results(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    fired: list[str],
    waiting: list[str],
) -> None:
    """Build #7 and #23 once their queues are finished."""
    card = read_run_set_card(cfg.run_set_root) or {}
    root = cfg.run_set_root / plan.RESULTS_DIRNAME
    done = state["transitions"]
    if "results_e7" not in done and _e7_ready(state):
        names = [
            plan.queue(q)
            for q in (plan.E7_TEACHERS, plan.E7_POOLS, plan.E7_UP, plan.E7_DOWN)
        ]
        if _finished(cfg, names, waiting) and not cfg.dry_run:
            out = results.build(
                plan.E7,
                root,
                card.get("git_sha"),
                planned=_e7_planned(state),
                selected=state["e7"]["selected"],
            )
            _fire(state, "results_e7", build=str(out))
            fired.append("results_e7")
    if "results_e23" not in done and "e23_rates" in state:
        names = [plan.queue(plan.E23_DOWN), plan.queue(plan.E23_UP)]
        if _finished(cfg, names, waiting) and not cfg.dry_run:
            planned = cells.e23_down_cells() + cells.e23_up_cells(state["e23_rates"])
            out = results.build(plan.E23, root, card.get("git_sha"), planned=planned)
            _fire(state, "results_e23", build=str(out))
            fired.append("results_e23")


# ---- priority 3 -------------------------------------------------------------------------


def _p3_gate(state: dict, waiting: list[str]) -> bool:
    """Whether priority 3 may start. A closed gate waits, as priority 2's does."""
    import json

    if "p3_preflight" in state["transitions"]:
        return True
    report = staging.latest_complete_report()
    pools = staging.p3_pools()
    if report is None or not set(pools) <= set(report["pools"]):
        waiting.append("p3 held: staging report lacks the priority 3 pools")
        return False
    path = plan.P3_POOL_REVIEW_PATH
    review = json.loads(path.read_text()) if path.is_file() else None
    problem = staging.pool_review_problem(review, pools)
    if problem:
        waiting.append(f"p3 held: {path}: {problem}")
        return False
    _fire(state, "p3_preflight", staging_report_at=report["at"])
    return True


def _submit_p3_static(cfg: driver_io.CampaignDriverConfig) -> None:
    jobs = [cells.student_job(c) for c in cells.e8m_cells()]
    _put(cfg, plan.queue(plan.E8X_MOMENTUM), jobs)
    jobs = [cells.student_job(c) for c in cells.e6x_cells()]
    _put(cfg, plan.queue(plan.E6X), jobs)


def choose_extra_seed_rates(rungs: list[dict]) -> dict[tuple[str, str], float]:
    """Each (animal, optimizer)'s selected #8 rate; a ladder with none is left out."""
    cut = plan.collapse_dce(plan.PY410)
    out = {}
    for animal in plan.ANIMALS:
        for optim in plan.E8_RATES:
            rung = metrics.select_rung(rungs, animal, optim, cut)
            if rung is not None:
                out[(animal, optim)] = rung["lr"]
    return out


def _e8x_selected(state: dict) -> dict[tuple[str, str], float]:
    return {(a, o): lr for a, o, lr in state["e8s_selected"]}


def _advance_e8s(
    cfg: driver_io.CampaignDriverConfig, state: dict, fired: list[str]
) -> None:
    """Queue #8's extra seeds once its rungs are chosen."""
    if "e8s_selected" in state or "results_e8" not in state["transitions"]:
        return
    rows = results.leaf_rows(cells.exp_cells(plan.E8))
    _, rungs = results.rung_table(rows)
    selected = choose_extra_seed_rates(rungs)
    state["e8s_selected"] = [[a, o, lr] for (a, o), lr in selected.items()]
    if not cfg.dry_run:
        doc = {"at": utc_now_iso(), "selected": state["e8s_selected"], "rungs": rungs}
        atomic_write_json(cfg.logs_dir / plan.E8X_SELECTION_FILE, doc)
    jobs = [cells.student_job(c) for c in cells.e8s_cells(selected)]
    n = _put(cfg, plan.queue(plan.E8X_SEEDS), jobs)
    fired.append(f"e8 extra seeds at {state['e8s_selected']} ({n} cells)")


def _p3_results(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    fired: list[str],
    waiting: list[str],
) -> None:
    """Build #6's other targets and #8's extras once their queues are finished."""
    card = read_run_set_card(cfg.run_set_root) or {}
    root = cfg.run_set_root / plan.RESULTS_DIRNAME
    done = state["transitions"]
    if "results_e6x" not in done and "results_e6" in done:
        if _finished(cfg, [plan.queue(plan.E6X)], waiting) and not cfg.dry_run:
            controls = [c for c in cells.e6_cells() if c.source == plan.BASE]
            planned = controls + cells.e6x_cells()
            out = results.build(plan.E6X, root, card.get("git_sha"), planned=planned)
            _fire(state, "results_e6x", build=str(out))
            fired.append("results_e6x")
    if "results_e8x" not in done and "e8s_selected" in state:
        names = [plan.queue(plan.E8X_MOMENTUM), plan.queue(plan.E8X_SEEDS)]
        if _finished(cfg, names, waiting) and not cfg.dry_run:
            selected = _e8x_selected(state)
            planned = cells.e8_cells() + cells.e8m_cells() + cells.e8s_cells(selected)
            out = results.build(
                plan.E8X, root, card.get("git_sha"), planned=planned, selected=selected
            )
            _fire(state, "results_e8x", build=str(out))
            fired.append("results_e8x")


def _repin(cfg: driver_io.CampaignDriverConfig) -> None:
    """Move every queue to the card's pin, including queues with no new jobs."""
    if cfg.dry_run:
        return
    defaults = driver_io.queue_job_defaults(cfg)
    for name in _submitted(cfg):
        store.update_queue_defaults(cfg.queue_root, name, defaults)


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass.

    Raises:
        CampaignGateError: No run-set card yet and no explicit `code_sha`.
    """
    if read_run_set_card(cfg.run_set_root) is None and not cfg.code_sha:
        raise CampaignGateError("the first pass needs --code-sha (a pushed pin)")
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
            card = read_run_set_card(cfg.run_set_root) or {}
            out = results.build(
                exp, cfg.run_set_root / plan.RESULTS_DIRNAME, card.get("git_sha")
            )
            _fire(state, name, build=str(out))
            mark(name)
    if _p2_gate(state, waiting):
        _submit_p2_static(cfg)
        _advance_e7(cfg, state, fired, waiting)
        _advance_e23(cfg, state, fired)
        _p2_results(cfg, state, fired, waiting)
    if _p3_gate(state, waiting):
        _submit_p3_static(cfg)
        _advance_e8s(cfg, state, fired)
        _p3_results(cfg, state, fired, waiting)
    _repin(cfg)
    finished = all(
        f"results_{e}" in done
        for e in (*plan.EXPERIMENTS, *plan.P2_EXPERIMENTS, *plan.P3_EXPERIMENTS)
    )
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
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "queues": queues,
        "card_pin": (read_run_set_card(cfg.run_set_root) or {}).get("git_sha"),
    }
