"""CLI: automated campaign driver for the bear checkpoint-grid stage-2 run.

Wraps `pipelines.ckptgrid2_campaign_pipeline`. One `advance` = one
idempotent pass over the phase DAG; `watch` polls until done and
retries transient failures; `status` is read-only.

    uv run python -m scripts.run_ckptgrid2_campaign status
    uv run python -m scripts.run_ckptgrid2_campaign advance --dry-run
    uv run python -m scripts.run_ckptgrid2_campaign watch --interval 300

`advance` / `watch` resolve the campaign code pin and refuse until the
campaign code is committed (plan §6).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.ckptgrid2_campaign_pipeline import (
    CampaignGateError,
    Ckptgrid2DriverOptions,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_RECONCILE_HELP = "Record an intentional code-pin advance instead of halting."
_CODE_SHA_HELP = "Explicit campaign pin (default: state code_sha, then HEAD)."


def _cfg(
    paths_config: Path,
    queue_root: Path | None,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
    auto_requeue: int = 0,
    code_sha: str | None = None,
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
        code_sha=code_sha,
    )


def _options(*, wave: int, deadline_hours: float) -> Ckptgrid2DriverOptions:
    return Ckptgrid2DriverOptions(wave=wave, deadline_hours=deadline_hours)


def _log_result(result) -> None:
    for name in result.fired:
        logger.info("fired transition: {}", name)
    for reason in result.waiting:
        logger.info("waiting: {}", reason)
    if result.done:
        logger.info("campaign complete.")


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
    wave: int = typer.Option(1, help="Campaign wave (1 or 2)."),
    deadline_hours: float = typer.Option(
        plan.DEADLINE_HOURS_DEFAULT,
        help="Open-window budget for this pass (hours).",
    ),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
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
        code_sha=code_sha,
    )
    try:
        result = run(cfg, _options(wave=wave, deadline_hours=deadline_hours))
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    _log_result(result)


@app.command("watch")
def watch_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    wave: int = typer.Option(1, help="Campaign wave (1 or 2)."),
    deadline_hours: float = typer.Option(
        plan.DEADLINE_HOURS_DEFAULT,
        help="Open-window budget from each watch start (hours).",
    ),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(
        0,
        help="Requeue failed jobs whose attempt count is still below this limit.",
    ),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
    max_consecutive_errors: int = typer.Option(
        5,
        help="Exit 3 after this many consecutive non-gate failures.",
    ),
) -> None:
    """Poll `advance` until done; retry transients, abort on a gate."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        paths_config,
        queue_root,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    options = _options(wave=wave, deadline_hours=deadline_hours)
    errors = 0
    while True:
        try:
            result = run(cfg, options)
            errors = 0
        except CampaignGateError as err:
            logger.error("GATE FAILED: {}", err)
            raise typer.Exit(code=2) from err
        except Exception:
            errors += 1
            logger.exception(
                "advance pass failed ({}/{})",
                errors,
                max_consecutive_errors,
            )
            if errors >= max_consecutive_errors:
                raise typer.Exit(code=3)
            time.sleep(min(interval, 60) * errors)
            continue
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
