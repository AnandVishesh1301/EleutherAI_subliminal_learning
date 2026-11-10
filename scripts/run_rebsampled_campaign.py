"""CLI: driver for the rebsampled run set.

uv run python -m scripts.run_rebsampled_campaign submit --code-sha <sha> --dry-run
uv run python -m scripts.run_rebsampled_campaign watch --code-sha <sha>
uv run python -m scripts.run_rebsampled_campaign status
uv run python -m scripts.run_rebsampled_campaign results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.rebsampled_pipeline import advance, build_results, driver_config
from pipelines.rebsampled_pipeline import status, submit
from subliminal_learning.campaign import rebsampled_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)
_PATHS = Path(plan.PATHS_CONFIG)


def _cfg(*, code_sha: str | None = None, dry_run: bool = False) -> CampaignDriverConfig:
    paths = yaml.safe_load(_PATHS.read_text())
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=resolve_queue_root(paths).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=plan.AUTO_REQUEUE_ATTEMPTS,
        code_sha=code_sha,
    )


@app.command("submit")
def submit_cmd(
    code_sha: str = typer.Option(..., help="Campaign code pin."),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
) -> None:
    """Write every YAML and submit both queues once."""
    load_dotenv()
    configure_loguru()
    logger.info("queues: {}", submit(_cfg(code_sha=code_sha, dry_run=dry_run)))


@app.command("watch")
def watch_cmd(
    code_sha: str = typer.Option(..., help="Campaign code pin."),
    interval: int = typer.Option(120, help="Seconds between passes."),
) -> None:
    """Submit (idempotent), then advance until the campaign is done."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(code_sha=code_sha)
    submit(cfg)
    while not advance(cfg).done:
        time.sleep(interval)
    logger.info("campaign complete")


@app.command("status")
def status_cmd() -> None:
    """Print per-queue job counts."""
    load_dotenv()
    configure_loguru()
    typer.echo(json.dumps(status(_cfg()), indent=2))


@app.command("results")
def results_cmd() -> None:
    """Rebuild ``results/`` from the leaves on disk."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(_PATHS.read_text())
    typer.echo(str(build_results(Path(paths["output_root"]))))


if __name__ == "__main__":
    app()
