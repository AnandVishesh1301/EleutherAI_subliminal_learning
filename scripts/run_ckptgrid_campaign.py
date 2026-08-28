"""CLI: automated campaign driver for the bear checkpoint-grid run set.

Wraps `pipelines.ckptgrid_campaign_pipeline`. One `advance` = one idempotent
pass over the phase DAG; `watch` polls until done; `status` is read-only.

    uv run python -m scripts.run_ckptgrid_campaign status
    uv run python -m scripts.run_ckptgrid_campaign advance --dry-run
    uv run python -m scripts.run_ckptgrid_campaign watch --interval 300
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.ckptgrid_campaign_pipeline import (
    CampaignGateError,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_RECONCILE_HELP = "Record an intentional code-pin advance instead of halting."


def _cfg(
    paths_config: Path,
    queue_root: Path | None,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
    auto_requeue: int = 0,
) -> CampaignDriverConfig:
    paths = yaml.safe_load(Path(paths_config).read_text())
    root = queue_root or resolve_queue_root(paths)
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=root.resolve(),
        repo_root=Path.cwd(),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )


@app.command("status")
def status_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print per-queue status and fired transitions (read-only)."""
    load_dotenv()
    configure_loguru()
    typer.echo(
        json.dumps(status(_cfg(paths_config, queue_root)), indent=2, default=str)
    )


@app.command("advance")
def advance_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(
        0,
        help="Requeue failed jobs whose attempt count is still below this limit.",
    ),
    dry_run: bool = typer.Option(
        False, help="Log planned submissions/writes; change nothing."
    ),
) -> None:
    """One idempotent advance pass over the phase DAG."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        paths_config,
        queue_root,
        dry_run,
        auto_reconcile_sha,
        auto_requeue=auto_requeue,
    )
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
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(
        0,
        help="Requeue failed jobs whose attempt count is still below this limit.",
    ),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
) -> None:
    """Poll `advance` until the campaign finishes (gate failures abort loudly)."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        paths_config,
        queue_root,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
    )
    while True:
        try:
            result = run(cfg)
        except CampaignGateError as err:
            logger.error("GATE FAILED: {}", err)
            raise typer.Exit(code=2) from err
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
