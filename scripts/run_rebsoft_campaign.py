"""CLI: automated driver of `rebsoft` (`pipelines.rebsoft_campaign_pipeline`).

    .venv/bin/python -m scripts.run_rebsoft_campaign advance --dry-run --code-sha <sha>
    .venv/bin/python -m scripts.run_rebsoft_campaign watch --code-sha <sha>
    .venv/bin/python -m scripts.run_rebsoft_campaign status
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.rebsoft_campaign_pipeline import (
    CampaignGateError,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)
_CODE_SHA_HELP = "Campaign code pin (first launch)."


def _cfg(
    *,
    dry_run: bool = False,
    auto_requeue: int = 0,
    code_sha: str | None = None,
    auto_reconcile_sha: bool = False,
) -> CampaignDriverConfig:
    return driver_config(
        output_root=plan.OUTPUT_ROOT,
        queue_root=plan.QUEUE_ROOT,
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


@app.command("status")
def status_cmd() -> None:
    """Print transitions, holes, labels and queue counts (file reads)."""
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
    interval: int = 120,
    max_consecutive_errors: int = 5,
) -> None:
    """Advance every `interval` seconds until the campaign finalizes."""
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
        except Exception:  # noqa: BLE001 - the tmux restart loop owns retries
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
            logger.info("rebsoft finalized")
            return
        time.sleep(interval)


if __name__ == "__main__":
    app()
