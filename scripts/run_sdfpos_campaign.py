"""CLI for one sdfpos / sdfpos2 / crossval2 run-set driver: status / plan / advance / watch."""

from __future__ import annotations

import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.sdfpos_campaign_pipeline import (
    CampaignGateError,
    SdfposDriverOptions,
    driver_config,
    plan_module,
    queue_plan,
    run,
    status,
)
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)
_PATHS_HELP = "Campaign paths YAML (default: the plan's PATHS_CONFIG)."
_CAMPAIGN_HELP = "Plan: sdfpos (Part I), sdfpos2 (Part II) or crossval2 (V1)."
_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_CODE_SHA_HELP = "Explicit campaign pin (default: state code_sha, then HEAD)."


def _paths(campaign: str, paths_config: Path | None) -> Path:
    return paths_config or Path(plan_module(campaign).PATHS_CONFIG)


def _cfg(
    campaign: str,
    partition: str,
    paths_config: Path,
    queue_root: Path | None,
    *,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
    auto_requeue: int = 0,
    code_sha: str | None = None,
) -> CampaignDriverConfig:
    paths = yaml.safe_load(Path(paths_config).read_text())
    root = queue_root or resolve_queue_root(paths)
    return driver_config(
        partition,
        campaign,
        output_root=Path(paths["output_root"]),
        queue_root=Path(root).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )


def _opts(
    campaign: str, partition: str, paths_config: Path, deadline_hours: float | None
) -> SdfposDriverOptions:
    plan = plan_module(campaign)
    if deadline_hours is None:
        deadline_hours = plan.DEADLINE_HOURS
    return SdfposDriverOptions(
        experiment=partition,
        campaign=campaign,
        paths_config=Path(paths_config),
        prereg_path=Path(getattr(plan, "PREREG_CHECK_PATH", plan.PREREG_PATH)),
        deadline_hours=deadline_hours,
        max_failed_frac=plan.MAX_FAILED_FRAC,
    )


@app.command()
def status_cmd(
    partition: str = typer.Option(..., "--partition"),
    campaign: str = typer.Option("sdfpos", help=_CAMPAIGN_HELP),
    paths_config: Path = typer.Option(None, help=_PATHS_HELP),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print transitions, queue counts and holes."""
    load_dotenv()
    configure_loguru()
    paths_config = _paths(campaign, paths_config)
    cfg = _cfg(campaign, partition, paths_config, queue_root)
    opts = _opts(campaign, partition, paths_config, None)
    typer.echo(yaml.safe_dump(status(cfg, opts), sort_keys=False))


@app.command()
def plan_cmd(
    partition: str = typer.Option(..., "--partition"),
    campaign: str = typer.Option("sdfpos", help=_CAMPAIGN_HELP),
) -> None:
    """Print the experiment's queues with their job counts and dependencies."""
    configure_loguru()
    typer.echo(yaml.safe_dump(queue_plan(partition, campaign), sort_keys=False))


@app.command()
def advance(
    partition: str = typer.Option(..., "--partition"),
    campaign: str = typer.Option("sdfpos", help=_CAMPAIGN_HELP),
    paths_config: Path = typer.Option(None, help=_PATHS_HELP),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    deadline_hours: float = typer.Option(None),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_reconcile_sha: bool = False,
    auto_requeue: int = 0,
    dry_run: bool = False,
) -> None:
    """Fire every transition whose gate is open, once."""
    load_dotenv()
    configure_loguru()
    paths_config = _paths(campaign, paths_config)
    cfg = _cfg(
        campaign,
        partition,
        paths_config,
        queue_root,
        dry_run=dry_run,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    try:
        result = run(cfg, _opts(campaign, partition, paths_config, deadline_hours))
    except CampaignGateError as err:
        logger.error("gate failed: {}", err)
        raise typer.Exit(code=2) from err
    typer.echo(yaml.safe_dump(result.__dict__, sort_keys=False))


@app.command()
def watch(
    partition: str = typer.Option(..., "--partition"),
    campaign: str = typer.Option("sdfpos", help=_CAMPAIGN_HELP),
    paths_config: Path = typer.Option(None, help=_PATHS_HELP),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    deadline_hours: float = typer.Option(None),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_reconcile_sha: bool = False,
    auto_requeue: int = 0,
    interval: int = 300,
    max_consecutive_errors: int = 5,
) -> None:
    """Advance every `interval` seconds until the experiment finalizes."""
    load_dotenv()
    configure_loguru()
    paths_config = _paths(campaign, paths_config)
    cfg = _cfg(
        campaign,
        partition,
        paths_config,
        queue_root,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    opts = _opts(campaign, partition, paths_config, deadline_hours)
    errors = 0
    while True:
        try:
            result = run(cfg, opts)
        except CampaignGateError as err:
            logger.error("gate failed: {}", err)
            raise typer.Exit(code=2) from err
        except Exception:  # noqa: BLE001 - restart loop owns the retry policy
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
            logger.info("waiting: {}", ", ".join(result.waiting))
        if result.done:
            logger.info("experiment {} finalized", partition)
            return
        time.sleep(interval)


if __name__ == "__main__":
    app()
