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

from pipelines.rebtopk_results_pipeline import (
    FINAL_SNAPSHOT,
    P1_SNAPSHOT,
    build_results,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import rebtopk_jobs as jobs
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.utils import code_pin_io
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
    """Refuse a first submit without passing reports from this pin."""
    sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha)
    for name in (plan.STAGING_REPORT, plan.PLACEBO_GATE_REPORT):
        path = _logs(cfg) / name
        if not path.is_file():
            raise CampaignGateError(f"{path} missing: run stage_rebtopk_assets")
        doc = json.loads(path.read_text())
        if doc.get("passed") is not True:
            raise CampaignGateError(f"{path} did not pass")
        if doc.get("code_sha") != sha:
            raise CampaignGateError(f"{path} is from {doc.get('code_sha')}, not {sha}")


def submit(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Write every YAML and submit every queue (idempotent). Returns queue names."""
    first_submit = not any(
        store.queue_exists(cfg.queue_root, n) for n in plan.queue_priorities()
    )
    if first_submit and not cfg.dry_run:
        _check_preflight(cfg)
    driver_io.ensure_campaign_code_pin(cfg)
    config_dir = cfg.generated_dir
    docs = jobs.all_docs(config_dir)
    if not cfg.dry_run:
        driver_io.write_generated_configs(cfg, docs)
    by_queue = jobs.queue_jobs(config_dir)
    for name, queue_jobs in by_queue.items():
        evals = name.endswith("-evals")
        driver_io.submit_queue(
            cfg,
            name,
            queue_jobs,
            depends_on=[name.removesuffix("-evals") + "-train"] if evals else None,
            strict_deps=evals,
            unpinned=True,
        )
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


def _mark(
    cfg: driver_io.CampaignDriverConfig,
    filename: str,
    snapshot: str,
    required: tuple[int, ...],
) -> str | None:
    """Build a complete snapshot, then write the marker once.

    Returns:
        The error text when results could not be built (no marker is written and the
        next pass retries, so requeueing keeps running), else ``None``.
    """
    marker = _logs(cfg) / filename
    if marker.exists():
        return None
    try:
        build_results(cfg.output_root, snapshot=snapshot, required=required)
    except Exception as err:
        logger.exception("results build for {} failed", filename)
        return f"{filename}: {err}"
    atomic_write_text(marker, utc_now_iso() + "\n")
    logger.info("{} written", filename)
    return None


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
    errors: list[str | None] = []
    if p1_done:
        errors.append(_mark(cfg, plan.P1_DONE_FILE, P1_SNAPSHOT, (1,)))
    if done:
        errors.append(_mark(cfg, plan.CAMPAIGN_DONE_FILE, FINAL_SNAPSHOT, (1, 3)))
    finished = (logs / plan.CAMPAIGN_DONE_FILE).exists()
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
            "done": finished,
            "results_errors": [e for e in errors if e],
        },
    )
    return AdvanceResult(tuple(requeued), tuple(stuck), p1_done, finished)


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
