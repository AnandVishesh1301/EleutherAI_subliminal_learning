"""CLI for one ws160m partition driver: status / advance / watch."""

from __future__ import annotations

import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.ws160m_campaign_pipeline import (
    CampaignGateError,
    Ws160mDriverOptions,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)
_PATHS_HELP = "Campaign paths YAML (hf_cache_root on the cold-1 ladder cache)."
_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_CODE_SHA_HELP = "Explicit campaign pin (default: state code_sha, then HEAD)."


def _cfg(
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
        output_root=Path(paths["output_root"]),
        queue_root=Path(root).resolve(),
        repo_root=Path.cwd(),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )


def _opts(partition: str, paths_config: Path, **kwargs) -> Ws160mDriverOptions:
    return Ws160mDriverOptions(
        partition=partition, paths_config=Path(paths_config), **kwargs
    )


@app.command()
def status_cmd(
    partition: str = typer.Option(..., "--partition"),
    paths_config: Path = typer.Option(Path(plan.PATHS_CONFIG), help=_PATHS_HELP),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print transitions, queue counts and holes."""
    load_dotenv()
    configure_loguru()
    typer.echo(
        yaml.safe_dump(
            status(_cfg(partition, paths_config, queue_root)), sort_keys=False
        )
    )


@app.command()
def advance(
    partition: str = typer.Option(..., "--partition"),
    paths_config: Path = typer.Option(Path(plan.PATHS_CONFIG), help=_PATHS_HELP),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    deadline_hours: float = plan.DEADLINE_HOURS_DEFAULT,
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_reconcile_sha: bool = False,
    auto_requeue: int = 0,
    dry_run: bool = False,
) -> None:
    """Fire every transition whose gate is open, once."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        partition,
        paths_config,
        queue_root,
        dry_run=dry_run,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    try:
        result = run(cfg, _opts(partition, paths_config, deadline_hours=deadline_hours))
    except CampaignGateError as err:
        logger.error("gate failed: {}", err)
        raise typer.Exit(code=2) from err
    typer.echo(yaml.safe_dump(result.__dict__, sort_keys=False))


@app.command()
def watch(
    partition: str = typer.Option(..., "--partition"),
    paths_config: Path = typer.Option(Path(plan.PATHS_CONFIG), help=_PATHS_HELP),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    deadline_hours: float = plan.DEADLINE_HOURS_DEFAULT,
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_reconcile_sha: bool = False,
    auto_requeue: int = 0,
    interval: int = 300,
    max_consecutive_errors: int = 5,
) -> None:
    """Advance every `interval` seconds until the partition finalizes."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        partition,
        paths_config,
        queue_root,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    opts = _opts(partition, paths_config, deadline_hours=deadline_hours)
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
        if result.done:
            logger.info("partition {} finalized", partition)
            return
        time.sleep(interval)


if __name__ == "__main__":
    app()
