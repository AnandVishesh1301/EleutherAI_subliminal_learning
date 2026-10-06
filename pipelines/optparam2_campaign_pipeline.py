"""Auto-advancing driver of the `optparam2` campaign (blog E1), on soar.

One `run(cfg)` = one idempotent advance pass:

    preflight      prereg, paths YAML, staging report, queue root, disk
    register       the campaign pin can run every job (the leaf sweeps write the
                   run-set card themselves, pinned to the same SHA)
    submit_main    tier-a train + late queues on cluster-3 / cluster-4
    (every pass)   unfinished queues follow the pin; failures under the requeue cap
                   are requeued, the rest recorded as labelled holes (never blocking:
                   a diverged high-lr leaf may fail its eval for good)
    submit_extra   once every tier-a train queue is terminal: the validation selection
                   (persisted once) -> tiers b / x / c / s
    results_train  results once every train queue is terminal
    finalize       results + CAMPAIGN_DONE once every queue is terminal
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines import optparam2_results_pipeline as results
from pipelines import optparam2_staging_pipeline as staging
from pipelines.traitfree_staging_pipeline import write_once_text
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import optparam2_jobs as jobs
from subliminal_learning.campaign import optparam2_metrics as metrics
from subliminal_learning.campaign import optparam2_plan as plan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.jobqueue import store
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

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
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "scripts/sweeps/sweep_optparam2.py",
        "scripts/run_optparam2_campaign.py",
        "scripts/stage_optparam2_assets.py",
        "scripts/launch_optparam2_campaign.sh",
        "configs/training/gkd_student.yaml",
        "configs/training/gkd_student_clip0.yaml",
        "configs/experiment/ws160m_student.yaml",
        "src/subliminal_learning/campaign/optparam2_plan.py",
        "src/subliminal_learning/campaign/optparam2_jobs.py",
        "src/subliminal_learning/campaign/optparam2_metrics.py",
        "src/subliminal_learning/analysis/clip_stats.py",
        "src/subliminal_learning/evaluation/pile_probe_io.py",
        "pipelines/optparam2_campaign_pipeline.py",
        "pipelines/optparam2_results_pipeline.py",
        "pipelines/optparam2_staging_pipeline.py",
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
    """A `CampaignDriverConfig` bound to the optparam2 spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def current_pin(cfg: driver_io.CampaignDriverConfig) -> code_pin_io.CodePin:
    return code_pin_io.resolve_campaign_pin(
        cfg.run_set_root, repo_root=cfg.repo_root, requested_sha=cfg.code_sha
    )


def node_pin_defaults(cfg: driver_io.CampaignDriverConfig, host: str) -> dict:
    """jobq defaults binding a queue to `host`'s worktree of the campaign pin."""
    if cfg.dry_run:
        return {"cwd": str(plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME)}
    pin = current_pin(cfg)
    if host != store.this_host():
        root = plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME / pin.sha
        pin = code_pin_io.CodePin(sha=pin.sha, root=root)
    return code_pin_io.pin_job_defaults(pin)


def write_leaf_yamls(cfg: driver_io.CampaignDriverConfig, leaves) -> int:
    """Create (or match) every generated YAML the leaves' jobs read."""
    n = 0
    for leaf in leaves:
        for path, doc in jobs.leaf_yaml_docs(leaf).items():
            if not cfg.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_once_text(path, yaml.safe_dump(doc, sort_keys=True))
            n += 1
    return n


def _submit_leaves(cfg: driver_io.CampaignDriverConfig, leaves) -> int:
    write_leaf_yamls(cfg, leaves)
    queues = jobs.queue_jobs(leaves)
    for name, queue_jobs in queues.items():
        host = plan.queue_host(name)
        deps = None
        if name.startswith(f"{plan.CAMPAIGN}-late-"):
            tier = name[len(f"{plan.CAMPAIGN}-late-") :].rsplit("-", 1)[0]
            deps = [plan.train_queue(tier, host)]
        driver_io.submit_queue(
            cfg,
            name,
            queue_jobs,
            depends_on=deps,
            node=host,
            defaults=node_pin_defaults(cfg, host),
        )
    return len(queues)


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in plan.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def _train_queues(names: list[str], tiers: tuple[str, ...]) -> list[str]:
    return [
        n
        for n in names
        for t in tiers
        for h in plan.HOSTS
        if n == plan.train_queue(t, h)
    ]


def _terminal(cfg: driver_io.CampaignDriverConfig, names: list[str]) -> bool:
    return bool(names) and all(store.queue_complete(cfg.queue_root, n) for n in names)


def _fire(state: dict, name: str, **info: object) -> None:
    state["transitions"][name] = {"at": utc_now_iso(), **info}


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        raise CampaignGateError(f"missing pre-registration {plan.PREREG_PATH}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} is not the plan's")
    if plan.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text():
        raise CampaignGateError(f"{plan.PATHS_CONFIG_PATH} differs from the plan")
    report = staging.latest_complete_report()
    if report is None:
        raise CampaignGateError(
            "no complete staging report; run scripts.stage_optparam2_assets"
        )
    free_gib = shutil.disk_usage(plan.CAMPAIGN_ROOT).free / 2**30
    if free_gib < plan.MIN_DISK_FREE_GIB:
        raise CampaignGateError(f"{plan.CAMPAIGN_ROOT}: {free_gib:.0f} GiB free")
    _fire(state, "preflight", staging_report_at=report["at"])


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(cfg)
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    _fire(state, "register", git_sha=pin.sha)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the campaign pin."""
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


def selection(cfg: driver_io.CampaignDriverConfig) -> dict | None:
    """The persisted tier-a selection document, or ``None``."""
    path = cfg.logs_dir / plan.SELECTION_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def _submit_extra(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    doc = selection(cfg)
    if doc is None:
        table = metrics.Table(
            r for r in results.leaf_rows(jobs.main_leaves()) if r["trained"]
        )
        summary = metrics.rung_summary(table, plan.BASE_MAIN, plan.EPOCHS_SHORT)
        doc = {"at": utc_now_iso(), **results.selection_doc(summary)}
        if not cfg.dry_run:
            atomic_write_json(cfg.logs_dir / plan.SELECTION_FILE, doc)
    extra = jobs.extra_leaves(*results.parse_selection(doc))
    n_queues = _submit_leaves(cfg, extra) if extra else 0
    _fire(
        state,
        "submit_extra",
        n_leaves=len(extra),
        n_queues=n_queues,
        selected=doc["selected"],
        unbracketed=doc["unbracketed"],
        no_eligible_rung=doc["no_eligible_rung"],
    )


def _build_results(cfg: driver_io.CampaignDriverConfig) -> None:
    if cfg.dry_run:
        return
    paths = results.build(cfg.run_set_root / plan.RESULTS_DIRNAME, selection(cfg))
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
        leaves = jobs.main_leaves()
        n_queues = _submit_leaves(cfg, leaves)
        _fire(state, "submit_main", n_leaves=len(leaves), n_queues=n_queues)
        mark("submit_main")
    note = _repin(cfg)
    if note:
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    names = _submitted(cfg)
    a_queues = _train_queues(names, (plan.TIER_A,))
    if "submit_extra" not in done:
        if _terminal(cfg, a_queues):
            _submit_extra(cfg, state)
            mark("submit_extra")
        else:
            waiting.append("submit_extra: tier-a train queues not terminal")
    names = _submitted(cfg)
    if (
        "submit_extra" in done
        and "results_train" not in done
        and _terminal(cfg, _train_queues(names, plan.TRAIN_TIERS))
    ):
        _build_results(cfg)
        _fire(state, "results_train")
        mark("results_train")
    all_done = False
    if "submit_extra" in done and _terminal(cfg, names):
        if "finalize" not in done:
            _build_results(cfg)
            if not cfg.dry_run:
                atomic_write_text(
                    cfg.logs_dir / plan.CAMPAIGN_DONE_FILE, utc_now_iso() + "\n"
                )
            _fire(state, "finalize", holes=sorted(state.get("holes", {})))
            mark("finalize")
        all_done = True
    elif "submit_extra" in done:
        waiting.append("finalize: queues not terminal")
    if state.get("holes"):
        waiting.append(f"labelled holes in {sorted(state['holes'])}")
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
    """Write `host`'s jobq policy file into the campaign queue root (create or equal)."""
    path = plan.QUEUE_ROOT / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json(host)
    if path.is_file() and not replace_existing:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    atomic_write_json(path, doc)
    return path
