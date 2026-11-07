"""Driver for the rebtopk campaign (run set 20261107_rebtopk-animals_160m).

No stage machine: every queue is submitted once (frozen jobkey sets, unpinned, jobq
``depends_on`` chaining). Each `advance` pass requeues failed jobs under the attempt cap,
builds results when a priority finishes, and writes `P1_DONE` then `CAMPAIGN_DONE`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from pipelines.rebtopk_results_pipeline import build_results
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import rebtopk_jobs as jobs
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso


class CampaignGateError(RuntimeError):
    """A precondition the driver refuses to proceed past."""


@dataclass(frozen=True)
class AdvanceResult:
    """One pass: requeued and over-cap jobs, and completion flags."""

    requeued: tuple[str, ...]
    stuck: tuple[str, ...]
    p1_done: bool
    done: bool


def shared_config_dir(output_root: Path) -> Path:
    """The run set's generated-YAML dir."""
    return plan.run_set_root(output_root) / plan.SHARED_GENERATED_DIRNAME


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the rebtopk spec and its YAML dir."""
    kwargs.setdefault(
        "generated_dir_override", shared_config_dir(kwargs["output_root"])
    )
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


def _logs(cfg: driver_io.CampaignDriverConfig) -> Path:
    return plan.run_set_root(cfg.output_root) / "logs"


def _check_preflight(cfg: driver_io.CampaignDriverConfig) -> None:
    for name in (plan.STAGING_REPORT, plan.PLACEBO_GATE_REPORT):
        path = _logs(cfg) / name
        if not path.is_file():
            raise CampaignGateError(f"{path} missing: run stage_rebtopk_assets")
        if json.loads(path.read_text()).get("passed") is not True:
            raise CampaignGateError(f"{path} did not pass")


def submit(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Write every YAML and submit every queue (idempotent). Returns queue names."""
    if not cfg.dry_run:
        _check_preflight(cfg)
    driver_io.ensure_campaign_code_pin(cfg)
    config_dir = cfg.generated_dir
    docs = jobs.all_docs(config_dir)
    if not cfg.dry_run:
        driver_io.write_generated_configs(cfg, docs)
    by_queue = jobs.queue_jobs(config_dir)
    for name, queue_jobs in by_queue.items():
        depends = (
            [name.removesuffix("-evals") + "-train"]
            if name.endswith("-evals")
            else None
        )
        driver_io.submit_queue(cfg, name, queue_jobs, depends_on=depends, unpinned=True)
    return list(by_queue)


def _requeue(
    cfg: driver_io.CampaignDriverConfig, name: str
) -> tuple[list[str], list[str]]:
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


def _complete(cfg: driver_io.CampaignDriverConfig, names: tuple[str, ...]) -> bool:
    return all(
        store.queue_exists(cfg.queue_root, n)
        and store.queue_complete(cfg.queue_root, n)
        and not store.queue_status(cfg.queue_root, n)["failed"]
        for n in names
    )


def _mark(cfg: driver_io.CampaignDriverConfig, filename: str) -> None:
    """Build results, then write the marker once."""
    marker = _logs(cfg) / filename
    if marker.exists():
        return
    build_results(cfg.output_root)
    atomic_write_text(marker, utc_now_iso() + "\n")
    logger.info("{} written", filename)


def advance(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One pass: requeue under cap, build results per priority, write state."""
    names = [
        n for n in plan.queue_priorities() if store.queue_exists(cfg.queue_root, n)
    ]
    requeued: list[str] = []
    stuck: list[str] = []
    for name in names:
        r, s = _requeue(cfg, name)
        requeued += r
        stuck += s
    p1_done = _complete(cfg, plan.priority_queues(1))
    done = p1_done and _complete(cfg, plan.priority_queues(3))
    logs = _logs(cfg)
    logs.mkdir(parents=True, exist_ok=True)
    if p1_done:
        _mark(cfg, plan.P1_DONE_FILE)
    if done:
        _mark(cfg, plan.CAMPAIGN_DONE_FILE)
    atomic_write_json(
        logs / plan.CAMPAIGN_STATE_FILE,
        {
            "updated_at": utc_now_iso(),
            "queues": {
                n: {
                    k: v
                    for k, v in store.queue_status(cfg.queue_root, n).items()
                    if isinstance(v, int)
                }
                for n in names
            },
            "requeued": requeued,
            "stuck": stuck,
            "p1_done": p1_done,
            "done": done,
        },
    )
    return AdvanceResult(tuple(requeued), tuple(stuck), p1_done, done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only per-queue job counts (pure file reads)."""
    logs = _logs(cfg)
    return {
        "run_set_id": plan.RUN_SET_ID,
        "queues": {
            n: {
                k: v
                for k, v in store.queue_status(cfg.queue_root, n).items()
                if isinstance(v, int)
            }
            for n in plan.queue_priorities()
            if store.queue_exists(cfg.queue_root, n)
        },
        "p1_done": (logs / plan.P1_DONE_FILE).is_file(),
        "done": (logs / plan.CAMPAIGN_DONE_FILE).is_file(),
    }
