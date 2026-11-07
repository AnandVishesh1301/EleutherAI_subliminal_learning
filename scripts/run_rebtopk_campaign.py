"""CLI: driver for the rebtopk run set.

uv run python -m scripts.run_rebtopk_campaign submit --code-sha <sha> --dry-run
uv run python -m scripts.run_rebtopk_campaign watch --code-sha <sha>
uv run python -m scripts.run_rebtopk_campaign status
uv run python -m scripts.run_rebtopk_campaign results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.rebtopk_campaign_pipeline import (
    CampaignGateError,
    advance,
    driver_config,
    status,
    submit,
)
from pipelines.rebtopk_results_pipeline import build_results
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)
_PATHS = Path(plan.PATHS_CONFIG)


def _cfg(
    *,
    code_sha: str | None = None,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
) -> CampaignDriverConfig:
    paths = yaml.safe_load(_PATHS.read_text())
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=resolve_queue_root(paths).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=plan.AUTO_REQUEUE_ATTEMPTS,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


def _submit(cfg: CampaignDriverConfig) -> None:
    try:
        names = submit(cfg)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    logger.info("{} queues submitted", len(names))


@app.command("submit")
def submit_cmd(
    code_sha: str = typer.Option(None, help="Campaign code pin."),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
) -> None:
    """Write every YAML and submit every queue once."""
    load_dotenv()
    configure_loguru()
    _submit(_cfg(code_sha=code_sha, dry_run=dry_run))


@app.command("watch")
def watch_cmd(
    code_sha: str = typer.Option(None, help="Campaign code pin."),
    auto_reconcile_sha: bool = typer.Option(False, help="Record a pin advance."),
    interval: int = typer.Option(120, help="Seconds between passes."),
) -> None:
    """Submit (idempotent), then advance until every queue finished."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(code_sha=code_sha, auto_reconcile_sha=auto_reconcile_sha)
    _submit(cfg)
    while True:
        result = advance(cfg)
        if result.done:
            logger.info("campaign complete")
            return
        time.sleep(interval)


@app.command("status")
def status_cmd() -> None:
    """Print per-queue job counts."""
    load_dotenv()
    configure_loguru()
    typer.echo(json.dumps(status(_cfg()), indent=2))


@app.command("results")
def results_cmd(dry_run: bool = typer.Option(False)) -> None:
    """Rebuild result tables from the leaves on disk."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(_PATHS.read_text())
    written = build_results(Path(paths["output_root"]), dry_run=dry_run)
    typer.echo(json.dumps({k: str(v) for k, v in written.items()}, indent=2))


if __name__ == "__main__":
    app()
