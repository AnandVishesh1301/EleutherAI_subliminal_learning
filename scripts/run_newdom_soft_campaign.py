"""CLI for the `newdom` soft driver: plan / status / advance / watch / results (soar)."""

from __future__ import annotations

import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines import newdom_soft_results_pipeline
from pipelines.newdom_soft_campaign_pipeline import (
    CampaignGateError,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)
_CODE_SHA_HELP = "Campaign pin (required until the soft teacher run-set card exists)."


def _cfg(
    *,
    dry_run: bool = False,
    auto_requeue: int = 0,
    code_sha: str | None = None,
    auto_reconcile_sha: bool = False,
) -> driver_io.CampaignDriverConfig:
    return driver_config(
        output_root=soft.OUTPUT_ROOT,
        queue_root=plan.QUEUE_ROOT,
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


@app.command("plan")
def plan_cmd() -> None:
    """Print every planned unit, queue, priority, footprint and example command."""
    typer.echo(yaml.safe_dump(soft.plan_doc(), sort_keys=False, width=10**9))


@app.command("status")
def status_cmd() -> None:
    """Print transitions, holes, per-target progress and queue counts (file reads)."""
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
    """Advance every `interval` seconds until the channel finalizes."""
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
            logger.info("newdom soft channel finalized")
            return
        time.sleep(interval)


@app.command("results")
def results_cmd() -> None:
    """Write a fresh results build from what is on disk now (no GPU, no jobq writes)."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg()
    state = driver_io.load_state(cfg)
    out = newdom_soft_results_pipeline.build(
        cfg.queue_root, state.get("targets", {}), cfg.run_set_root / "results"
    )
    typer.echo(str(out))


if __name__ == "__main__":
    app()
