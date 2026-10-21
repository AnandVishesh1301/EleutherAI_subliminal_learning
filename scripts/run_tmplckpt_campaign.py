"""CLI for the `tmplckpt` driver: plan / status / card / advance / watch (home cluster-3)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from contextlib import nullcontext

from pipelines.tmplckpt_campaign_pipeline import (
    CampaignGateError,
    card_doc,
    driver_config,
    driver_lock,
    planned_summary,
    run,
    status,
)
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.io import atomic_write_json
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
    if plan.PATHS_CONFIG_PATH.is_file():
        paths = yaml.safe_load(plan.PATHS_CONFIG_PATH.read_text())
    elif dry_run:
        paths = plan.paths_config_doc()
    else:
        raise typer.BadParameter(f"{plan.PATHS_CONFIG_PATH} missing; stage first")
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=Path(paths["queue_root"]),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


@app.command("plan")
def plan_cmd() -> None:
    """Print queues, dependencies, job counts and commands (pure, no writes)."""
    typer.echo(yaml.safe_dump(planned_summary(), sort_keys=False))


@app.command("status")
def status_cmd() -> None:
    """Print transitions, holes, reviews and per-queue counts (file reads)."""
    load_dotenv()
    configure_loguru()
    typer.echo(yaml.safe_dump(status(_cfg()), sort_keys=False))


@app.command("card")
def card_cmd(
    write: Path = typer.Option(None, help="Also write the card JSON here."),
) -> None:
    """Print the fleet ledger card (`fleet register tmplckpt --card <path>`)."""
    doc = card_doc()
    if write is not None:
        if write.is_file() and json.loads(write.read_text()) != doc:
            raise typer.BadParameter(f"{write} exists with a different card")
        write.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(write, doc)
    typer.echo(json.dumps(doc, indent=2))


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
        with nullcontext() if dry_run else driver_lock():
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
    """Advance every `interval` seconds until CAMPAIGN_DONE."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )
    try:
        with driver_lock():
            _watch_loop(cfg, interval, max_consecutive_errors)
    except CampaignGateError as err:
        logger.error("gate failed: {}", err)
        raise typer.Exit(code=2) from err


def _watch_loop(
    cfg: CampaignDriverConfig, interval: int, max_consecutive_errors: int
) -> None:
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
            logger.info("tmplckpt finalized")
            return
        time.sleep(interval)


if __name__ == "__main__":
    app()
