"""CLI: automated driver for the off-policy cross-size campaign v2.

Wraps `pipelines.offxsize2_campaign_pipeline`. One `advance` is one idempotent pass over
the phase DAG; `watch` polls until done; `status` is read-only and runs anywhere. Every
pass that submits asserts the hostname, because the campaign shares its queue root with
the node running the live cross-family campaign.

    uv run python -m scripts.run_offxsize2_campaign status
    uv run python -m scripts.run_offxsize2_campaign advance --dry-run
    uv run python -m scripts.run_offxsize2_campaign watch --interval 300
    uv run python -m scripts.run_offxsize2_campaign results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.offxsize2_campaign_pipeline import (
    CampaignGateError,
    Offxsize2DriverOptions,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_PATHS = Path("configs/paths/paths_local.yaml")
_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_DEADLINE_HELP = "Wall-clock budget of this window; at expiry the driver drains, "
_DEADLINE_HELP += "writes coverage and stops without CAMPAIGN_DONE."


def _cfg(
    paths_config: Path,
    queue_root: Path | None,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
) -> CampaignDriverConfig:
    """Build the driver config from the machine-local paths YAML."""
    paths = yaml.safe_load(Path(paths_config).read_text())
    root = queue_root or resolve_queue_root(paths)
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=Path(root).resolve(),
        repo_root=Path.cwd(),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )


@app.command("status")
def status_cmd(
    paths_config: Path = typer.Option(_PATHS, help="Machine-local paths YAML."),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print the transitions, the gate decisions and the queue totals."""
    load_dotenv()
    configure_loguru()
    print(json.dumps(status(_cfg(paths_config, queue_root)), indent=2, default=str))


@app.command("advance")
def advance_cmd(
    paths_config: Path = typer.Option(_PATHS, help="Machine-local paths YAML."),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    dry_run: bool = typer.Option(False, help="Walk every phase without writing."),
    auto_reconcile_sha: bool = typer.Option(
        False, help="Record an intentional code-pin advance instead of halting."
    ),
    auto_requeue: int = typer.Option(
        plan.AUTO_REQUEUE_ATTEMPTS, help="Requeue failed jobs up to this many attempts."
    ),
    deadline_hours: float = typer.Option(
        plan.DEADLINE_HOURS_DEFAULT, help=_DEADLINE_HELP
    ),
) -> None:
    """Run one idempotent pass over the phase DAG."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(paths_config, queue_root, dry_run, auto_reconcile_sha, auto_requeue)
    try:
        result = run(cfg, Offxsize2DriverOptions(deadline_hours=deadline_hours))
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    for name in result.fired:
        logger.info("fired transition: {}", name)
    for msg in result.waiting:
        logger.info("waiting on: {}", msg)
    if result.done:
        logger.info("campaign complete.")


@app.command("watch")
def watch_cmd(
    paths_config: Path = typer.Option(_PATHS, help="Machine-local paths YAML."),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    interval: int = typer.Option(300, help="Seconds between passes."),
    auto_requeue: int = typer.Option(
        plan.AUTO_REQUEUE_ATTEMPTS, help="Requeue failed jobs up to this many attempts."
    ),
    deadline_hours: float = typer.Option(
        plan.DEADLINE_HOURS_DEFAULT, help=_DEADLINE_HELP
    ),
) -> None:
    """Advance on a timer until the campaign is done or its deadline expires."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(paths_config, queue_root, auto_requeue=auto_requeue)
    options = Offxsize2DriverOptions(deadline_hours=deadline_hours)
    while True:
        try:
            result = run(cfg, options)
        except CampaignGateError as err:
            logger.error("GATE FAILED: {}", err)
            raise typer.Exit(code=2) from err
        for name in result.fired:
            logger.info("fired transition: {}", name)
        if result.done:
            logger.info("campaign finished — watch exiting.")
            return
        for msg in result.waiting:
            logger.info("waiting on: {}", msg)
        time.sleep(interval)


@app.command("results")
def results_cmd(
    paths_config: Path = typer.Option(_PATHS, help="Machine-local paths YAML."),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    dry_run: bool = typer.Option(False, help="Compute without writing."),
    partial: bool = typer.Option(
        False, help="Stamp `claims.json` as a partial (deadline) reading."
    ),
) -> None:
    """Rebuild every result CSV and `claims.json` from what is on disk right now."""
    from pipelines.offxsize2_results_pipeline import build_results

    load_dotenv()
    configure_loguru()
    build_results(
        _cfg(paths_config, queue_root, dry_run), dry_run=dry_run, partial=partial
    )


if __name__ == "__main__":
    app()
