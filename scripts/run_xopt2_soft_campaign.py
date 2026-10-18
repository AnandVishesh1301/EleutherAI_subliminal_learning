"""CLI for the `xopt2` soft driver: plan / status / advance / watch / reissue-legacy (soar only)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.xopt2_soft_results_pipeline import gl_complete_cells
from pipelines.xopt2_soft_campaign_pipeline import (
    CampaignGateError,
    driver_config,
    planned_queue_counts,
    reissue_legacy,
    run,
    status,
)
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt2_soft as soft
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)
_CODE_SHA_HELP = "Explicit campaign pin (required before the run-set card exists)."


def _cfg(
    *,
    dry_run: bool = False,
    auto_requeue: int = 0,
    code_sha: str | None = None,
    auto_reconcile_sha: bool = False,
) -> CampaignDriverConfig:
    return driver_config(
        output_root=soft.OUTPUT_ROOT,
        queue_root=soft.QUEUE_ROOT,
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


@app.command("plan")
def plan_cmd() -> None:
    """Print the launch submission's jobs per queue (reads `xopt`'s gl leaves to find
    the reused cross cells; walks add more jobs later)."""
    gl = gl_complete_cells()
    counts = planned_queue_counts(gl)
    doc = {"total": sum(counts.values()), "gl_reused_cells": len(gl), "queues": counts}
    typer.echo(yaml.safe_dump(doc, sort_keys=False))


@app.command("status")
def status_cmd() -> None:
    """Print transitions, holes, walks and per-queue counts (pure file reads)."""
    load_dotenv()
    configure_loguru()
    typer.echo(yaml.safe_dump(status(_cfg()), sort_keys=False))


@app.command()
def advance(
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
    auto_reconcile_sha: bool = False,
    dry_run: bool = False,
) -> None:
    """Fire every transition whose gate is open, once."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )
    try:
        result = run(cfg)
    except CampaignGateError as err:
        logger.error("gate failed: {}", err)
        raise typer.Exit(code=2) from err
    typer.echo(yaml.safe_dump(result.__dict__, sort_keys=False))


@app.command()
def watch(
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
    auto_reconcile_sha: bool = False,
    interval: int = 300,
    max_consecutive_errors: int = 5,
) -> None:
    """Advance every `interval` seconds until the run set finalizes."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )
    errors = 0
    while True:
        try:
            result = run(cfg)
        except CampaignGateError as err:
            logger.error("gate failed: {}", err)
            raise typer.Exit(code=2) from err
        except Exception:  # noqa: BLE001 - the launcher's restart loop owns retries
            errors += 1
            logger.exception(
                "driver pass failed ({}/{})", errors, max_consecutive_errors
            )
            if errors >= max_consecutive_errors:
                raise typer.Exit(code=3) from None
            time.sleep(min(interval, 60) * errors)
            continue
        errors = 0
        if result.fired:
            logger.info("fired: {}", ", ".join(result.fired))
        if result.waiting:
            logger.info("waiting: {}", "; ".join(result.waiting[:20]))
        if result.done:
            logger.info("xopt2 soft run set finalized")
            return
        time.sleep(interval)


@app.command("reissue-legacy")
def reissue_legacy_cmd(
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    dry_run: bool = False,
) -> None:
    """Rebuild the not-started queues of goncalo-only jobs for every worker host (soft
    pools stopped, no soft driver running)."""
    load_dotenv()
    configure_loguru()
    live = subprocess.run(
        ["pgrep", "-f", "scripts.run_xopt2_soft_campaign (watch|advance)"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if live:
        logger.error("a soft driver is running (pids {}); stop it first", live)
        raise typer.Exit(code=2)
    try:
        out = reissue_legacy(_cfg(dry_run=dry_run, code_sha=code_sha))
    except CampaignGateError as err:
        logger.error("gate failed: {}", err)
        raise typer.Exit(code=2) from err
    typer.echo(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    app()
