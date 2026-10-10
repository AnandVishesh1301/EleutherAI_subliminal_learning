"""Auto-advancing driver of `hlopt4` (hlopt Part IV, blog wave 2), on soar.

One `run(cfg)` = one idempotent advance pass:

    preflight      pre-registration (main checkout), Part IV paths YAML, Part IV
                   staging report, queue root, disk
    register       run-set card pinned to the campaign code pin
    submit_main    train + late-eval queues of tiers p (agenda E5), r (LoRA rank), w
                   (weight decay) and e (Adam epsilon), on cluster-5 and goncalo (late
                   queues depend on their host's train queue, lowest priority)
    (every pass)   unfinished queues follow the pin; failures under the requeue cap
                   are requeued, the rest recorded as holes
    submit_extra   once every tier p / r / e train queue is terminal: the
                   validation-selected rung of each (dog, arm) of those tiers gets
                   seeds 4-5 (tier x)
    results_train  results tables + claims once every train queue is terminal
    finalize       results + CAMPAIGN_DONE once every queue is complete, no holes
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import hlopt_results_pipeline as results
from pipelines import hlopt_staging_pipeline as staging
from pipelines.hlopt_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _context,
    _fire,
    current_pin,
    node_pin_defaults,
)
from pipelines.hlopt_campaign_pipeline import SPEC as SPEC1
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import hlopt_jobs as jobs
from subliminal_learning.campaign import hlopt_metrics as metrics
from subliminal_learning.campaign import hlopt_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import build_run_set_card
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import (
    read_run_set_card,
    register_run_set_launch,
    utc_now_iso,
)

PART = 4
SPEC = replace(
    SPEC1,
    run_set_id=plan.RUN_SET_ID4,
    teacher_run_set_id=plan.RUN_SET_ID4,
    queue_priorities=plan.QUEUE_PRIORITIES4,
    generated_config_dir=str(plan.GENERATED4_DIR),
    campaign_state_file=plan.CAMPAIGN4_STATE_FILE,
    extra_pin_paths=(
        *SPEC1.extra_pin_paths,
        "pipelines/hlopt4_campaign_pipeline.py",
        "scripts/run_hlopt4_campaign.py",
        "scripts/launch_hlopt4_campaign.sh",
        "src/subliminal_learning/analysis/clip_stats.py",
        "src/subliminal_learning/training/param_control_io.py",
    ),
)


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the hlopt4 spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED4_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def _submit(
    cfg: driver_io.CampaignDriverConfig, name: str, queue_jobs: list[Job]
) -> str:
    host = plan.queue_host(name)
    deps = []
    prefix = f"{plan.CAMPAIGN4}-late-"
    if name.startswith(prefix):
        tier = name[len(prefix) :].rsplit("-", 1)[0]
        deps = [plan.train_queue4(tier, host)]
    return driver_io.submit_queue(
        cfg,
        name,
        queue_jobs,
        depends_on=deps or None,
        node=host,
        defaults=node_pin_defaults(cfg, host),
    )


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in plan.QUEUE_PRIORITIES4 if store.queue_exists(cfg.queue_root, n)]


def _train_queues(names: list[str], tiers: tuple[str, ...]) -> list[str]:
    wanted = {plan.train_queue4(t, h) for t in tiers for h in plan.HOSTS4}
    return [n for n in names if n in wanted]


def _terminal(cfg: driver_io.CampaignDriverConfig, names: list[str]) -> bool:
    return all(store.queue_complete(cfg.queue_root, n) for n in names)


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


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if not (cfg.repo_root / plan.PREREG4_PATH).is_file():
        raise CampaignGateError(f"missing pre-registration {plan.PREREG4_PATH}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT4.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} is not Part IV's")
    if plan.PATHS4_CONFIG_PATH.read_text() != staging.paths4_yaml_text():
        raise CampaignGateError(f"{plan.PATHS4_CONFIG_PATH} differs from the plan")
    report = staging.latest_complete_report4()
    if report is None:
        raise CampaignGateError(
            "no complete Part IV staging report; run "
            "scripts.stage_hlopt_assets --part 4"
        )
    free_gib = shutil.disk_usage(plan.CAMPAIGN_ROOT).free / 2**30
    if free_gib < plan.MIN_DISK_FREE_GIB:
        raise CampaignGateError(f"{plan.CAMPAIGN_ROOT}: {free_gib:.0f} GiB free")
    _fire(state, "preflight", staging_report_at=report["at"])


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        from subliminal_learning.utils import code_pin_io

        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(cfg)
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    card = build_run_set_card(
        run_set_id=plan.RUN_SET_ID4,
        run_type="offpolicy_student",
        purpose=(
            "hlopt Part IV (blog wave 2): which parameters carry the trait (E5), "
            "LoRA rank, weight decay toward zero / the pretrained weights, and the "
            "Adam epsilon dial, on the hlopt dog pool"
        ),
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/hlopt_plan.py",
        resolved_config={
            "campaign_root": str(plan.CAMPAIGN_ROOT),
            "queue_root": str(plan.QUEUE_ROOT4),
            "student_tree": str(plan.recipe_output_root("e3", PART)),
            "base_repo": plan.BASE_REPO,
            "pools": {k: p.filtered_sha256 for k, p in plan.POOLS.items()},
        },
        grid_axes={"n_configs": len(plan.configs4())},
        model_size="160m",
    )
    register_run_set_launch(cfg.output_root, cfg.run_set_root, card)
    _fire(state, "register", git_sha=pin.sha)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the card's pin."""
    if cfg.dry_run or read_run_set_card(cfg.run_set_root) is None:
        return None
    n = 0
    wanted: dict[str, dict] = {}
    for name in _submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        meta = store.read_meta(cfg.queue_root, name)
        want = wanted.setdefault(str(meta.node), node_pin_defaults(cfg, str(meta.node)))
        if meta.defaults.get("cwd") == want["cwd"]:
            continue
        store.update_queue_defaults(cfg.queue_root, name, want)
        n += 1
    return f"repin(queues={n})" if n else None


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


def _selection_doc(cfg: driver_io.CampaignDriverConfig) -> dict | None:
    path = cfg.logs_dir / plan.SELECTION4_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def _submit_extra(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    spec, hub_org = _context()
    cells = [c for c in jobs.main_cells4() if c.tier in plan.SELECT_TIERS4]
    rows = results.cell_rows(spec, hub_org, cells)
    summary = metrics.rung_summary(metrics.paired_deltas(rows), plan.SEEDS)
    choice = metrics.select_rungs(summary, plan.select_arms4(), plan.ANIMALS4)
    selected = metrics.selected_configs(choice)
    doc = {
        "at": utc_now_iso(),
        "rule": f"max mean {plan.SELECT_SET} Delta over seeds {list(plan.SEEDS)}, "
        f"mean treatment pp1 delta-CE < {plan.COLLAPSE_DCE:.4f}, ties -> smaller lr",
        "choice": {f"{a}|{arm}": key for (a, arm), key in choice.items()},
        "selected": selected,
        "summary": {f"{k}|{a}": v for (k, a), v in summary.items()},
    }
    if not cfg.dry_run:
        atomic_write_json(cfg.logs_dir / plan.SELECTION4_FILE, doc)
    extra = jobs.extra_cells(selected, PART)
    for name, queue_jobs in jobs.queue_jobs(extra).items():
        _submit(cfg, name, queue_jobs)
    _fire(state, "submit_extra", n_cells=len(extra), selected=selected)


def _build_results(cfg: driver_io.CampaignDriverConfig) -> None:
    if cfg.dry_run:
        return
    spec, hub_org = _context()
    doc = _selection_doc(cfg) or {}
    paths = results.build(
        cfg.run_set_root / plan.RESULTS_DIRNAME,
        spec,
        hub_org,
        doc.get("selected"),
        part=PART,
        choice=doc.get("choice"),
    )
    logger.info("results -> {}", paths)


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
    if "submit_main" not in done:
        cells = jobs.main_cells4()
        for name, queue_jobs in jobs.queue_jobs(cells).items():
            _submit(cfg, name, queue_jobs)
        _fire(state, "submit_main", n_cells=len(cells))
        mark("submit_main")
    note = _repin(cfg)
    if note:
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    names = _submitted(cfg)
    select_queues = _train_queues(names, plan.SELECT_TIERS4)
    if "submit_extra" not in done:
        holed = sorted(q for q in select_queues if q in state.get("holes", {}))
        if select_queues and _terminal(cfg, select_queues) and not holed:
            _submit_extra(cfg, state)
            mark("submit_extra")
        else:
            waiting.append(
                f"submit_extra: tier p/r/e train queues running or holed {holed}"
            )
    names = _submitted(cfg)
    train_queues = _train_queues(names, plan.TRAIN_TIERS4)
    if (
        "submit_extra" in done
        and "results_train" not in done
        and _terminal(cfg, train_queues)
    ):
        _build_results(cfg)
        _fire(state, "results_train")
        mark("results_train")
    all_done = False
    if (
        "submit_extra" in done
        and _finished(cfg, names, waiting)
        and not state.get("holes")
    ):
        if "finalize" not in done:
            _build_results(cfg)
            if not cfg.dry_run:
                atomic_write_text(
                    cfg.logs_dir / plan.CAMPAIGN_DONE_FILE, utc_now_iso() + "\n"
                )
            _fire(state, "finalize")
            mark("finalize")
        all_done = True
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
    }


def write_policy(host: str, *, replace_existing: bool = False) -> Path:
    """Write `host`'s Part IV jobq policy into its queue root (create or equal)."""
    path = plan.QUEUE_ROOT4 / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json4(host)
    if path.is_file() and not replace_existing:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    atomic_write_json(path, doc)
    return path
