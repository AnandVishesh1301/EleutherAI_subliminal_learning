"""CLI for the S2 core drivers (`s2core`): status / advance / watch / write-policy / plan."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.s2core_campaign_pipeline import (
    CampaignGateError,
    driver_config,
    enumerate_plan,
    run,
    status,
    write_policy,
)
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)
_CODE_SHA_HELP = "Explicit campaign pin (required before the run-set card exists)."
_CHANNEL = typer.Option(..., help="soft | hard.")


def _check_channel(channel: str) -> str:
    if channel not in plan.CHANNELS:
        raise typer.BadParameter(f"channel must be one of {plan.CHANNELS}")
    return channel


def _cfg(
    channel: str,
    *,
    dry_run: bool = False,
    auto_requeue: int = 0,
    code_sha: str | None = None,
    auto_reconcile_sha: bool = False,
) -> CampaignDriverConfig:
    paths = yaml.safe_load(plan.paths_config_path(channel).read_text())
    return driver_config(
        channel,
        output_root=Path(paths["output_root"]),
        queue_root=Path(paths["queue_root"]).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


def _setup() -> None:
    load_dotenv()
    configure_loguru()


@app.command("status")
def status_cmd(channel: str = _CHANNEL) -> None:
    """Print transitions, holes, blockers and per-queue counts (pure file reads)."""
    _setup()
    typer.echo(yaml.safe_dump(status(_cfg(_check_channel(channel))), sort_keys=False))


@app.command()
def advance(
    channel: str = _CHANNEL,
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
    auto_reconcile_sha: bool = False,
    dry_run: bool = False,
) -> None:
    """Fire every transition whose gate is open, once."""
    _setup()
    cfg = _cfg(
        _check_channel(channel),
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
    channel: str = _CHANNEL,
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
    auto_reconcile_sha: bool = False,
    interval: int = plan.DRIVER_INTERVAL_S,
    max_consecutive_errors: int = 5,
) -> None:
    """Advance every `interval` seconds until the channel finalizes."""
    _setup()
    cfg = _cfg(
        _check_channel(channel),
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
            logger.info("s2core {} finalized", channel)
            return
        time.sleep(interval)


@app.command("write-policy")
def write_policy_cmd(
    host: str = typer.Option(..., help="Hostname (jobq key)."),
    replace_existing: bool = False,
) -> None:
    """Write `host`'s jobq policy into its channel's queue root."""
    _setup()
    typer.echo(str(write_policy(host, replace_existing=replace_existing)))


@app.command("plan")
def plan_cmd(channel: str = _CHANNEL) -> None:
    """SM-6: every queue / job the driver would submit (hypothetical selection)."""
    _setup()
    typer.echo(json.dumps(enumerate_plan(_check_channel(channel)), indent=1))


if __name__ == "__main__":
    app()
