"""CLI: automated campaign driver for the optimizer x parameterization run set.

Wraps `pipelines.optparam_campaign_pipeline`. One `advance` = one idempotent pass over the
phase DAG (submit whatever is ready, refuse loudly on a gate failure); `watch` polls
`advance` until the campaign finishes; `status` is read-only. The worker pool is started
separately (`uv run jobq work` — see scripts/launch_optparam_campaign.sh); this driver
only submits queues.

    uv run python -m scripts.run_optparam_campaign status
    uv run python -m scripts.run_optparam_campaign advance --dry-run
    uv run python -m scripts.run_optparam_campaign watch --interval 300
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.optparam_campaign_pipeline import (
    CampaignGateError,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.jobq_folder import configured_queue_folder
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_RECONCILE_HELP = (
    "Record a mid-campaign HEAD advance on the run-set card instead of halting."
)


def _cfg(
    paths_config: Path,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
) -> CampaignDriverConfig:
    paths = yaml.safe_load(Path(paths_config).read_text())
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_folder=configured_queue_folder().resolve(),
        repo_root=Path.cwd(),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
    )


@app.command("status")
def status_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
) -> None:
    """Print per-queue status and fired transitions (read-only)."""
    load_dotenv()
    configure_loguru()
    typer.echo(json.dumps(status(_cfg(paths_config)), indent=2, default=str))


@app.command("advance")
def advance_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    dry_run: bool = typer.Option(
        False, help="Log planned submissions/writes; change nothing."
    ),
) -> None:
    """One idempotent advance pass over the phase DAG."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(paths_config, dry_run, auto_reconcile_sha)
    try:
        result = run(cfg)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    for name in result.fired:
        logger.info("fired transition: {}", name)
    for reason in result.waiting:
        logger.info("waiting: {}", reason)
    if result.done:
        logger.info("campaign complete.")


@app.command("watch")
def watch_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
) -> None:
    """Poll `advance` until the campaign finishes (gate failures abort loudly)."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(paths_config, auto_reconcile_sha=auto_reconcile_sha)
    while True:
        result = run(cfg)  # CampaignGateError propagates: watch must die loudly
        for name in result.fired:
            logger.info("fired transition: {}", name)
        if result.done:
            logger.info("campaign complete — watch exiting.")
            return
        if result.waiting:
            logger.info("waiting on: {}", "; ".join(result.waiting))
        time.sleep(interval)


if __name__ == "__main__":
    app()
