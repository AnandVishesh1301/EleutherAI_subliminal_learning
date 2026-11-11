"""Driver for the domtopk campaign (run sets 20261111_domtopk-{trees,colors}_160m).

No stage machine: each domain's train and eval queues of both priorities are
submitted once (frozen jobkey sets, unpinned, each eval queue ``depends_on`` its
train queue; priority 2 drains from lower jobq priorities). Each `advance` pass
requeues failed jobs under the attempt cap; per run set it builds ``results/p1``
and writes `P1_DONE` once priority 1 finishes, then ``results/final`` and
`DOMAIN_DONE` once both do. The campaign `P1_DONE` and `CAMPAIGN_DONE` follow both
run sets. One `CampaignDriverConfig` per domain, so each run set resolves its own
pin card.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from pipelines.domtopk_results_pipeline import (
    FINAL_SNAPSHOT,
    P1_SNAPSHOT,
    build_results,
)
from pipelines.rebtopk_campaign_pipeline import _complete, _requeue
from subliminal_learning.campaign import domtopk_jobs as jobs
from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign import driver_io
from subliminal_learning.jobqueue import store
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

DriverConfigs = Mapping[str, driver_io.CampaignDriverConfig]


class CampaignGateError(RuntimeError):
    """A precondition the driver refuses to proceed past."""


@dataclass(frozen=True)
class AdvanceResult:
    """One pass: requeued and over-cap jobs, finished domains, campaign done."""

    requeued: tuple[str, ...]
    stuck: tuple[str, ...]
    p1_done: tuple[str, ...]
    domains_done: tuple[str, ...]
    done: bool


def shared_config_dir(output_root: Path, domain: str) -> Path:
    """`domain`'s generated-YAML dir."""
    return plan.run_set_root(output_root, domain) / plan.SHARED_GENERATED_DIRNAME


def driver_configs(**kwargs) -> dict[str, driver_io.CampaignDriverConfig]:
    """One `CampaignDriverConfig` per domain, bound to its spec and YAML dir."""
    return {
        d: driver_io.CampaignDriverConfig(
            spec=plan.SPECS[d],
            generated_dir_override=shared_config_dir(kwargs["output_root"], d),
            **kwargs,
        )
        for d in plan.DOMAINS
    }


def _output_root(cfgs: DriverConfigs) -> Path:
    roots = {cfg.output_root for cfg in cfgs.values()}
    if len(roots) != 1:
        raise ValueError(f"domain configs disagree on output_root: {roots}")
    return roots.pop()


def _check_preflight(cfg: driver_io.CampaignDriverConfig) -> None:
    """Refuse a first submit without a passing staging report from this pin."""
    sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha)
    path = cfg.logs_dir / plan.STAGING_REPORT
    if not path.is_file():
        raise CampaignGateError(f"{path} missing: run stage_domtopk_assets")
    doc = json.loads(path.read_text())
    if doc.get("passed") is not True:
        raise CampaignGateError(f"{path} did not pass")
    if doc.get("code_sha") != sha:
        raise CampaignGateError(f"{path} is from {doc.get('code_sha')}, not {sha}")


def submit(cfgs: DriverConfigs) -> list[str]:
    """Write every YAML and submit every queue (idempotent). Returns queue names."""
    names: list[str] = []
    for domain, cfg in cfgs.items():
        queues = plan.queue_priorities(domain)
        first_submit = not any(store.queue_exists(cfg.queue_root, n) for n in queues)
        if first_submit and not cfg.dry_run:
            _check_preflight(cfg)
        driver_io.ensure_campaign_code_pin(cfg)
        docs = jobs.all_docs(cfg.generated_dir, domain)
        if not cfg.dry_run:
            driver_io.write_generated_configs(cfg, docs)
        by_queue = jobs.queue_jobs(cfg.generated_dir, domain)
        for priority in plan.PRIORITIES:
            train, evals = plan.priority_queues(domain, priority)
            for name, deps in ((train, None), (evals, [train])):
                driver_io.submit_queue(
                    cfg,
                    name,
                    by_queue[name],
                    depends_on=deps,
                    strict_deps=deps is not None,
                    unpinned=True,
                )
                names.append(name)
    return names


def _mark(
    cfg: driver_io.CampaignDriverConfig,
    domain: str,
    filename: str,
    snapshot: str,
    required: tuple[int, ...],
) -> str | None:
    """Build `domain`'s complete snapshot, then write its marker once.

    Returns:
        The error text when results could not be built (no marker is written and the
        next pass retries, so requeueing keeps running), else ``None``.
    """
    marker = cfg.logs_dir / filename
    if marker.exists():
        return None
    try:
        build_results(cfg.output_root, domain, snapshot=snapshot, required=required)
    except Exception as err:
        logger.exception("{} results build for {} failed", snapshot, domain)
        return f"{domain} {filename}: {err}"
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(marker, utc_now_iso() + "\n")
    logger.info("{} written", marker)
    return None


def _counts(cfg: driver_io.CampaignDriverConfig, name: str) -> dict[str, int]:
    return {
        k: v
        for k, v in store.queue_status(cfg.queue_root, name).items()
        if isinstance(v, int)
    }


def _existing(cfg: driver_io.CampaignDriverConfig, domain: str) -> list[str]:
    return [
        n
        for n in plan.queue_priorities(domain)
        if store.queue_exists(cfg.queue_root, n)
    ]


def _marked(cfgs: DriverConfigs, filename: str) -> tuple[str, ...]:
    return tuple(d for d, cfg in cfgs.items() if (cfg.logs_dir / filename).is_file())


def _mark_campaign(logs: Path, filename: str, domains: tuple[str, ...]) -> bool:
    """Write the campaign-level `filename` once every run set has its own."""
    marker = logs / filename
    if set(domains) == set(plan.DOMAINS) and not marker.exists():
        atomic_write_text(marker, utc_now_iso() + "\n")
        logger.info("{} written", marker)
    return marker.exists()


def advance(cfgs: DriverConfigs) -> AdvanceResult:
    """One pass: requeue under cap, finish each priority, write the campaign state."""
    requeued: list[str] = []
    stuck: list[str] = []
    errors: list[str | None] = []
    queues: dict[str, dict[str, int]] = {}
    for domain, cfg in cfgs.items():
        for name in _existing(cfg, domain):
            r, s = _requeue(cfg, name)
            requeued += r
            stuck += s
            queues[name] = _counts(cfg, name)
        p1 = _complete(cfg, plan.priority_queues(domain, plan.P1))
        if p1:
            errors.append(
                _mark(cfg, domain, plan.P1_DONE_FILE, P1_SNAPSHOT, (plan.P1,))
            )
        if p1 and _complete(cfg, plan.priority_queues(domain, plan.P2)):
            errors.append(
                _mark(
                    cfg, domain, plan.DOMAIN_DONE_FILE, FINAL_SNAPSHOT, plan.PRIORITIES
                )
            )
    p1_done = _marked(cfgs, plan.P1_DONE_FILE)
    domains_done = _marked(cfgs, plan.DOMAIN_DONE_FILE)
    logs = plan.campaign_logs_dir(_output_root(cfgs))
    logs.mkdir(parents=True, exist_ok=True)
    _mark_campaign(logs, plan.P1_DONE_FILE, p1_done)
    done = _mark_campaign(logs, plan.CAMPAIGN_DONE_FILE, domains_done)
    atomic_write_json(
        logs / plan.CAMPAIGN_STATE_FILE,
        {
            "updated_at": utc_now_iso(),
            "queues": queues,
            "requeued": requeued,
            "stuck": stuck,
            "p1_done": list(p1_done),
            "domains_done": list(domains_done),
            "done": done,
            "results_errors": [e for e in errors if e],
        },
    )
    return AdvanceResult(tuple(requeued), tuple(stuck), p1_done, domains_done, done)


def status(cfgs: DriverConfigs) -> dict:
    """Read-only per-queue job counts and markers (pure file reads)."""
    logs = plan.campaign_logs_dir(_output_root(cfgs))
    return {
        "run_set_ids": dict(plan.RUN_SET_IDS),
        "queues": {
            n: _counts(cfg, n) for d, cfg in cfgs.items() for n in _existing(cfg, d)
        },
        "p1_done": list(_marked(cfgs, plan.P1_DONE_FILE)),
        "domains_done": list(_marked(cfgs, plan.DOMAIN_DONE_FILE)),
        "done": (logs / plan.CAMPAIGN_DONE_FILE).is_file(),
    }


def preview(output_root: Path, out_dir: Path) -> dict:
    """Write every YAML and job list the driver would submit under `out_dir`.

    Touches no pin, queue or output tree: YAMLs and jobs name the real run-set
    config dirs under `output_root`, but are written only to `out_dir`.

    Returns:
        ``{queue: number of jobs}`` plus ``units`` and ``yamls`` counts.
    """
    summary: dict = {"units": len(plan.units()), "yamls": 0}
    for domain in plan.DOMAINS:
        config_dir = shared_config_dir(output_root, domain)
        target = Path(out_dir) / domain
        target.mkdir(parents=True, exist_ok=True)
        for fname, doc in sorted(jobs.all_docs(config_dir, domain).items()):
            body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
            atomic_write_text(target / fname, body)
            summary["yamls"] += 1
        by_queue = jobs.queue_jobs(config_dir, domain)
        atomic_write_json(
            target / "jobs.json",
            {
                name: [
                    {"key": j.key, "cmd": j.cmd, "mem_mib": j.mem_mib, "slots": j.slots}
                    for j in queue_jobs
                ]
                for name, queue_jobs in by_queue.items()
            },
        )
        summary.update({name: len(q) for name, q in by_queue.items()})
    return summary
