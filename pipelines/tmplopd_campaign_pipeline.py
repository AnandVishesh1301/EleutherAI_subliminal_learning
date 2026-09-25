"""Driver for the tmplopd campaign (run set 20260925_tmplopd-lion_160m).

No stage machine: every queue is submitted once (frozen jobkey set, jobq-native
``depends_on`` chaining), and each `advance` pass only requeues failed jobs under the
attempt cap, writes the state file, and drops `CORE_EVALS_DONE` / `CAMPAIGN_DONE` when
the core / every queue has finished. Queues are pinned per host to that host's worktree
of the campaign pin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import tmplopd_jobs as jobs
from subliminal_learning.campaign import tmplopd_plan as plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso


class CampaignGateError(RuntimeError):
    """A precondition the driver refuses to proceed past."""


@dataclass(frozen=True)
class AdvanceResult:
    """One pass: requeued jobs, stuck (over-cap) jobs, and completion flags."""

    requeued: tuple[str, ...]
    stuck: tuple[str, ...]
    core_done: bool
    done: bool


def run_set_root(output_root: Path) -> Path:
    return Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)


def shared_config_dir(output_root: Path) -> Path:
    """The run set's shared generated-YAML dir (every node reads it)."""
    return run_set_root(output_root) / plan.SHARED_GENERATED_DIRNAME


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the tmplopd spec and its shared YAML dir."""
    kwargs.setdefault("generated_dir_override", shared_config_dir(kwargs["output_root"]))
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


def node_defaults(cfg: driver_io.CampaignDriverConfig, host: str, *, is_eval: bool) -> dict:
    """Queue defaults: `host`'s worktree of the campaign pin + the stage ceiling."""
    pin = code_pin_io.resolve_campaign_pin(
        cfg.run_set_root, repo_root=cfg.repo_root, requested_sha=cfg.code_sha
    )
    root = plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME / pin.sha
    defaults = code_pin_io.pin_job_defaults(code_pin_io.CodePin(sha=pin.sha, root=root))
    return {**defaults, **plan.stage_defaults(is_eval)}


def _check_identity(cfg: driver_io.CampaignDriverConfig) -> None:
    want = Path(plan.paths_config_doc()["queue_root"])
    if Path(cfg.queue_root).resolve() != want.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} is not the tmplopd root {want}")
    report = run_set_root(cfg.output_root) / "logs" / plan.STAGING_REPORT
    if not report.is_file():
        raise CampaignGateError(f"staging report missing: {report} (run stage_tmplopd_assets)")
    if not Path(plan.PREREG_PATH).is_file():
        raise CampaignGateError(f"pre-registration missing: {plan.PREREG_PATH}")


def _submit(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    queue_jobs: list[Job],
    *,
    is_eval: bool,
    depends_on: list[str],
) -> None:
    host = plan.queue_node(name)
    driver_io.submit_queue(
        cfg,
        name,
        queue_jobs,
        depends_on=depends_on or None,
        node=host,
        defaults=node_defaults(cfg, host, is_eval=is_eval),
    )


def submit(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Write every YAML and submit every queue (idempotent). Returns queue names."""
    _check_identity(cfg)
    driver_io.ensure_campaign_code_pin(cfg)
    config_dir = cfg.generated_dir
    driver_io.write_generated_configs(cfg, jobs.all_docs(config_dir))
    names = [plan.TEACHER_EVAL_QUEUE]
    _submit(
        cfg, plan.TEACHER_EVAL_QUEUE, jobs.teacher_eval_jobs(config_dir),
        is_eval=True, depends_on=[],
    )
    for tier in plan.TIERS:
        for host, queue_jobs in jobs.tier_train_jobs(tier, config_dir).items():
            if queue_jobs:
                name = plan.train_queue(tier, host)
                _submit(cfg, name, queue_jobs, is_eval=False, depends_on=plan.train_deps(tier))
                names.append(name)
        for host, queue_jobs in jobs.tier_eval_jobs(tier, config_dir).items():
            if queue_jobs:
                name = plan.eval_queue(tier, host)
                _submit(cfg, name, queue_jobs, is_eval=True, depends_on=plan.eval_deps(tier))
                names.append(name)
    return names


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in plan.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def _requeue(cfg: driver_io.CampaignDriverConfig, name: str) -> tuple[list[str], list[str]]:
    """Requeue failed jobs below the attempt cap; return (requeued, stuck) jobkeys."""
    under: list[str] = []
    stuck: list[str] = []
    for job in store.load_jobs(cfg.queue_root, name):
        if store.job_state(cfg.queue_root, name, job) != store.FAILED:
            continue
        attempt = store.read_attempt(cfg.queue_root, name, job.jobkey)
        (under if attempt < cfg.auto_requeue else stuck).append(job.jobkey)
    if under:
        store.requeue(cfg.queue_root, name, failed_only=True, jobkeys=under)
        logger.warning("auto-requeue {}: {}", name, under)
    for jobkey in stuck:
        logger.error("STUCK over requeue cap {}: {}", name, jobkey)
    return [f"{name}/{k}" for k in under], [f"{name}/{k}" for k in stuck]


def _complete(cfg: driver_io.CampaignDriverConfig, names: list[str]) -> bool:
    return all(
        store.queue_exists(cfg.queue_root, n)
        and store.queue_complete(cfg.queue_root, n)
        and not store.queue_status(cfg.queue_root, n)["failed"]
        for n in names
    )


def planned_queues(config_dir: Path) -> list[str]:
    """Every queue `submit` creates (hosts left without jobs get no queue)."""
    names = [plan.TEACHER_EVAL_QUEUE]
    for tier in plan.TIERS:
        for build, name_of in (
            (jobs.tier_train_jobs, plan.train_queue),
            (jobs.tier_eval_jobs, plan.eval_queue),
        ):
            names += [name_of(tier, h) for h, js in build(tier, config_dir).items() if js]
    return names


def advance(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One pass: requeue under cap, write state, drop the completion markers."""
    names = _submitted(cfg)
    requeued: list[str] = []
    stuck: list[str] = []
    for name in names:
        r, s = _requeue(cfg, name)
        requeued += r
        stuck += s
    planned = planned_queues(cfg.generated_dir)
    core_done = _complete(cfg, [n for n in planned if n.startswith("tmplopd-c-")])
    done = _complete(cfg, planned)
    logs = run_set_root(cfg.output_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if core_done and not (logs / plan.CORE_DONE_FILE).exists():
        (logs / plan.CORE_DONE_FILE).write_text(utc_now_iso() + "\n")
        logger.info("CORE EVALS DONE")
    if done and not (logs / plan.CAMPAIGN_DONE_FILE).exists():
        (logs / plan.CAMPAIGN_DONE_FILE).write_text(utc_now_iso() + "\n")
        logger.info("CAMPAIGN DONE")
    atomic_write_json(
        logs / plan.CAMPAIGN_STATE_FILE,
        {
            "updated_at": utc_now_iso(),
            "queues": {
                n: {k: v for k, v in store.queue_status(cfg.queue_root, n).items()
                    if isinstance(v, int)}
                for n in names
            },
            "requeued": requeued,
            "stuck": stuck,
            "core_done": core_done,
            "done": done,
        },
    )
    return AdvanceResult(tuple(requeued), tuple(stuck), core_done, done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only per-tier job counts (pure file reads)."""
    out: dict[str, dict[str, int]] = {}
    for name in _submitted(cfg):
        tier = name.split("-")[1] if name != plan.TEACHER_EVAL_QUEUE else "teacher"
        kind = "evals" if "-evals-" in name else "train"
        counts = out.setdefault(f"{tier}/{kind}", {})
        for k, v in store.queue_status(cfg.queue_root, name).items():
            if isinstance(v, int):
                counts[k] = counts.get(k, 0) + v
    logs = run_set_root(cfg.output_root) / "logs"
    return {
        "run_set_id": plan.RUN_SET_ID,
        "tiers": out,
        "core_done": (logs / plan.CORE_DONE_FILE).is_file(),
        "done": (logs / plan.CAMPAIGN_DONE_FILE).is_file(),
        "state": json.loads((logs / plan.CAMPAIGN_STATE_FILE).read_text())
        if (logs / plan.CAMPAIGN_STATE_FILE).is_file()
        else None,
    }
