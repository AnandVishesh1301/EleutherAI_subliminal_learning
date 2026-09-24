"""CLI: automated campaign driver for the xfam wave-2 run set (20260924_xfam2-bear_mixed).

Wraps `pipelines.xfam2_campaign_pipeline`. One `advance` = one idempotent pass over the
DAG; `watch` polls until done (exit 2 on a gate failure); `status` is read-only.

    uv run python -m scripts.run_xfam2_campaign advance --dry-run
    uv run python -m scripts.run_xfam2_campaign watch --interval 300 --code-sha <sha>
    uv run python -m scripts.run_xfam2_campaign status
    uv run python -m scripts.run_xfam2_campaign results
    uv run python -m scripts.run_xfam2_campaign rebalance --hosts <h1,h2,...> [--apply]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.xfam2_campaign_pipeline import (
    CampaignGateError,
    Xfam2DriverOptions,
    driver_config,
    rebalance,
    run,
    status,
)
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_PATHS = Path(plan.PATHS_CONFIG)
_REQUEUE_HELP = "Requeue failed jobs whose attempt count is still below this limit."


def _cfg(
    paths_config: Path,
    *,
    dry_run: bool = False,
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
    code_sha: str | None = None,
    auto_reconcile_sha: bool = False,
) -> CampaignDriverConfig:
    paths = yaml.safe_load(Path(paths_config).read_text())
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=resolve_queue_root(paths).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


def _opts(deadline_hours: float) -> Xfam2DriverOptions:
    return Xfam2DriverOptions(deadline_hours=deadline_hours)


def _advance(cfg: CampaignDriverConfig, opts: Xfam2DriverOptions):
    try:
        result = run(cfg, opts)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    for name in result.fired:
        logger.info("fired: {}", name)
    for reason in result.waiting:
        logger.info("waiting: {}", reason)
    return result


@app.command("advance")
def advance_cmd(
    paths_config: Path = _PATHS,
    auto_requeue: int = typer.Option(plan.AUTO_REQUEUE_ATTEMPTS, help=_REQUEUE_HELP),
    code_sha: str = typer.Option(None, help="Campaign code pin (first launch)."),
    auto_reconcile_sha: bool = typer.Option(
        False, help="Record an intentional re-pin."
    ),
    deadline_hours: float = typer.Option(plan.DEADLINE_HOURS),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
) -> None:
    """One idempotent advance pass over the DAG."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        paths_config,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )
    _advance(cfg, _opts(deadline_hours))


@app.command("watch")
def watch_cmd(
    paths_config: Path = _PATHS,
    auto_requeue: int = typer.Option(plan.AUTO_REQUEUE_ATTEMPTS, help=_REQUEUE_HELP),
    code_sha: str = typer.Option(None, help="Campaign code pin (first launch)."),
    auto_reconcile_sha: bool = typer.Option(
        False, help="Record an intentional re-pin."
    ),
    deadline_hours: float = typer.Option(plan.DEADLINE_HOURS),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
) -> None:
    """Poll `advance` until the campaign finishes; exit 2 on a gate failure."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        paths_config,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )
    opts = _opts(deadline_hours)
    while not _advance(cfg, opts).done:
        time.sleep(interval)
    logger.info("campaign complete -- watch exiting")


@app.command("status")
def status_cmd(paths_config: Path = _PATHS) -> None:
    """Print transitions, per-node queue totals, gates, holes and the deadline."""
    load_dotenv()
    configure_loguru()
    typer.echo(json.dumps(status(_cfg(paths_config)), indent=2, default=str))


@app.command("rebalance")
def rebalance_cmd(
    hosts: str = typer.Option(..., help="Comma list of hostnames to spread over."),
    apply: bool = typer.Option(False, help="Write the placement file (else dry run)."),
    paths_config: Path = _PATHS,
) -> None:
    """Balance claim-free queues over `--hosts`; the driver re-homes them next pass."""
    load_dotenv()
    configure_loguru()
    host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    try:
        report = rebalance(_cfg(paths_config), host_list, apply=apply)
    except CampaignGateError as err:
        logger.error("REBALANCE REFUSED: {}", err)
        raise typer.Exit(2) from err
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("results")
def results_cmd(
    paths_config: Path = _PATHS,
    dry_run: bool = typer.Option(False, help="Compute every frame; write nothing."),
) -> None:
    """Rebuild the results CSVs + `claims.json` from the leaves on disk."""
    from pipelines.xfam2_results_pipeline import build_results

    load_dotenv()
    configure_loguru()
    build_results(_cfg(paths_config, dry_run=dry_run), dry_run=dry_run)


if __name__ == "__main__":
    app()
