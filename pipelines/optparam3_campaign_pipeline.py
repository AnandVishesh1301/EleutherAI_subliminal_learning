"""Auto-advancing driver of `optparam3` (optparam2 Part III, blog wave 2), on soar.

One `run(cfg)` = one idempotent advance pass:

    preflight      prereg, the Part III paths YAML + complete Part III staging report,
                   optparam2's finalized results (references R / F), the Part III
                   queue root, disk on ssd-2
    register       the campaign pin can run every job (the leaf sweeps write the
                   run-set card themselves, pinned to the same SHA)
    submit_main    tiers p, r, w, e, g on soar / cluster-3 / cluster-4
    (every pass)   unfinished queues follow the pin; failures under the requeue cap
                   are requeued, the rest recorded as labelled holes (never blocking)
    submit_extra   once every tier p / r / e train queue is terminal: the validation
                   selection (persisted once) -> tier x3 (leaves (0,3), (1,4))
    results_train  results once every train queue is terminal
    finalize       results + CAMPAIGN_DONE once every queue is terminal
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines import optparam2_results_pipeline as results
from pipelines import optparam2_staging_pipeline as staging
from pipelines.optparam2_campaign_pipeline import SPEC as SPEC1
from pipelines.optparam2_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    _terminal,
    current_pin,
    node_pin_defaults,
)
from pipelines.traitfree_staging_pipeline import write_once_text
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import optparam2_jobs as jobs
from subliminal_learning.campaign import optparam2_metrics as metrics
from subliminal_learning.campaign import optparam2_plan as plan
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.jobqueue import store
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

SPEC = replace(
    SPEC1,
    run_set_id=plan.RUN_SET_ID3,
    teacher_run_set_id=plan.RUN_SET_ID3,
    queue_priorities=plan.QUEUE_PRIORITIES3,
    generated_config_dir=str(plan.GENERATED_DIR3),
    campaign_state_file=plan.CAMPAIGN3_STATE_FILE,
    extra_pin_paths=(
        *SPEC1.extra_pin_paths,
        "src/subliminal_learning/training/param_control_io.py",
        "pipelines/optparam3_campaign_pipeline.py",
        "scripts/run_optparam3_campaign.py",
        "scripts/launch_optparam3_campaign.sh",
    ),
)
YAML_WRITERS = 32


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the optparam3 spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_DIR3)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def training_base_doc() -> dict:
    """The pin's GKD student training YAML (the base of every generated one)."""
    return yaml.safe_load(Path(ws160m_plan.STUDENT_TRAINING_CONFIG).read_text())


def write_leaf_yamls(cfg: driver_io.CampaignDriverConfig, leaves) -> int:
    """Create (or match) every generated YAML the leaves' jobs read (parallel writes)."""
    base = training_base_doc()
    items = {
        p: d for leaf in leaves for p, d in jobs.leaf_yaml_docs(leaf, base).items()
    }
    if cfg.dry_run:
        return len(items)

    def put(item) -> None:
        path, doc = item
        path.parent.mkdir(parents=True, exist_ok=True)
        write_once_text(path, yaml.safe_dump(doc, sort_keys=True))

    with ThreadPoolExecutor(YAML_WRITERS) as pool:
        list(pool.map(put, items.items()))
    return len(items)


def _submit_leaves(cfg: driver_io.CampaignDriverConfig, leaves) -> int:
    write_leaf_yamls(cfg, leaves)
    queues = jobs.queue_jobs(leaves)
    late = f"{plan.CAMPAIGN3}-late-"
    for name, queue_jobs in queues.items():
        host = plan.queue_host(name)
        deps = None
        if name.startswith(late):
            tier = name[len(late) :].rsplit("-", 1)[0]
            deps = [plan.train_queue3(tier, host)]
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
    return [n for n in plan.QUEUE_PRIORITIES3 if store.queue_exists(cfg.queue_root, n)]


def _train_queues(names: list[str], tiers: tuple[str, ...]) -> list[str]:
    wanted = {plan.train_queue3(t, h) for t in tiers for h in plan.HOSTS3}
    return [n for n in names if n in wanted]


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if not (cfg.repo_root / plan.PREREG3_PATH).is_file():
        raise CampaignGateError(f"missing pre-registration {plan.PREREG3_PATH}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT3.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} is not the Part III root")
    if not cfg.dry_run and not plan.QUEUE_ROOT3.is_dir():
        raise CampaignGateError(f"{plan.QUEUE_ROOT3} missing (staging creates it)")
    if not plan.PATHS3_CONFIG_PATH.is_file():
        raise CampaignGateError(
            f"{plan.PATHS3_CONFIG_PATH} missing; run scripts.stage_optparam2_assets "
            "--part 3"
        )
    if plan.PATHS3_CONFIG_PATH.read_text() != staging.paths3_yaml_text():
        raise CampaignGateError(f"{plan.PATHS3_CONFIG_PATH} differs from the plan")
    report = staging.latest_complete_report3()
    if report is None:
        raise CampaignGateError(
            "no complete Part III staging report; run "
            "scripts.stage_optparam2_assets --part 3"
        )
    if not results.PART1_LEAVES_CSV.is_file():
        raise CampaignGateError(f"{results.PART1_LEAVES_CSV} missing (Part I results)")
    try:
        refs = results.references3()
    except (RuntimeError, KeyError, OSError) as err:
        raise CampaignGateError(f"optparam2 references: {err}") from err
    free_gib = shutil.disk_usage(plan.CAMPAIGN_ROOT3).free / 2**30
    if free_gib < plan.MIN_DISK_FREE_GIB:
        raise CampaignGateError(f"{plan.CAMPAIGN_ROOT3}: {free_gib:.0f} GiB free")
    _fire(state, "preflight", staging_report_at=report["at"], references=refs)


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


def _requeue(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
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
    """The persisted tier p / r / e selection document, or ``None``."""
    path = cfg.logs_dir / plan.SELECTION3_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def _submit_extra(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    doc = selection(cfg)
    if doc is None:
        leaves = [lf for lf in jobs.part3_leaves() if lf.tier in plan.SELECT_TIERS3]
        table = metrics.Table(r for r in results.leaf_rows(leaves) if r["trained"])
        summary = metrics.rung_summary(table, plan.BASE_MAIN, plan.EPOCHS_SHORT)
        doc = {"at": utc_now_iso(), **results.selection3_doc(summary)}
        if not cfg.dry_run:
            atomic_write_json(cfg.logs_dir / plan.SELECTION3_FILE, doc)
    extra = jobs.extra_leaves3(results.parse_selection3(doc))
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
    paths = results.build3(cfg.run_set_root / plan.RESULTS_DIRNAME, selection(cfg))
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
        leaves = jobs.part3_leaves()
        n_queues = _submit_leaves(cfg, leaves)
        _fire(state, "submit_main", n_leaves=len(leaves), n_queues=n_queues)
        mark("submit_main")
    note = _repin(cfg)
    if note:
        fired.append(note)
    if not cfg.dry_run:
        _requeue(cfg, state)
    names = _submitted(cfg)
    if "submit_extra" not in done:
        if _terminal(cfg, _train_queues(names, plan.SELECT_TIERS3)):
            _submit_extra(cfg, state)
            mark("submit_extra")
        else:
            waiting.append("submit_extra: tier p / r / e train queues not terminal")
    names = _submitted(cfg)
    if (
        "submit_extra" in done
        and "results_train" not in done
        and _terminal(cfg, _train_queues(names, plan.TRAIN_TIERS3))
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
    """Write `host`'s Part III jobq policy into the Part III queue root (create or equal)."""
    path = plan.QUEUE_ROOT3 / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json3(host)
    if path.is_file() and not replace_existing:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    atomic_write_json(path, doc)
    return path
