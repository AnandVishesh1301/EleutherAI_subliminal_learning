"""CLI: driver for the domtopk run sets (trees and colors).

uv run python -m scripts.run_domtopk_campaign plan --out-dir <scratch dir>
uv run python -m scripts.run_domtopk_campaign submit --code-sha <sha> --dry-run
uv run python -m scripts.run_domtopk_campaign watch --code-sha <sha>
uv run python -m scripts.run_domtopk_campaign status
uv run python -m scripts.run_domtopk_campaign results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.domtopk_campaign_pipeline import (
    CampaignGateError,
    advance,
    driver_configs,
    preview,
    status,
    submit,
)
from pipelines.domtopk_results_pipeline import build_results
from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)
_PATHS = Path(plan.PATHS_CONFIG)


def _cfgs(
    *,
    code_sha: str | None = None,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
) -> dict[str, CampaignDriverConfig]:
    paths = yaml.safe_load(_PATHS.read_text())
    return driver_configs(
        output_root=Path(paths["output_root"]),
        queue_root=resolve_queue_root(paths).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=plan.AUTO_REQUEUE_ATTEMPTS,
        code_sha=code_sha,
        auto_reconcile_sha=auto_reconcile_sha,
    )


def _submit(cfgs: dict[str, CampaignDriverConfig]) -> None:
    try:
        names = submit(cfgs)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    logger.info("{} queues submitted", len(names))


@app.command("plan")
def plan_cmd(
    out_dir: Path = typer.Option(..., help="Scratch dir for the YAMLs and job lists."),
) -> None:
    """Write every YAML and job the driver would submit to `out_dir` only."""
    configure_loguru()
    paths = yaml.safe_load(_PATHS.read_text())
    summary = preview(Path(paths["output_root"]), out_dir)
    typer.echo(json.dumps(summary, indent=2))


@app.command("submit")
def submit_cmd(
    code_sha: str = typer.Option(..., help="Campaign code pin."),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
) -> None:
    """Write every YAML and submit every queue once."""
    load_dotenv()
    configure_loguru()
    _submit(_cfgs(code_sha=code_sha, dry_run=dry_run))


@app.command("watch")
def watch_cmd(
    code_sha: str = typer.Option(..., help="Campaign code pin."),
    auto_reconcile_sha: bool = typer.Option(False, help="Record a pin advance."),
    interval: int = typer.Option(120, help="Seconds between passes."),
) -> None:
    """Submit (idempotent), then advance until both run sets finished."""
    load_dotenv()
    configure_loguru()
    cfgs = _cfgs(code_sha=code_sha, auto_reconcile_sha=auto_reconcile_sha)
    _submit(cfgs)
    while True:
        result = advance(cfgs)
        if result.done:
            logger.info("campaign complete")
            return
        time.sleep(interval)


@app.command("status")
def status_cmd() -> None:
    """Print per-queue job counts and markers."""
    load_dotenv()
    configure_loguru()
    typer.echo(json.dumps(status(_cfgs()), indent=2))


@app.command("results")
def results_cmd(dry_run: bool = typer.Option(False)) -> None:
    """Rebuild both run sets' ``latest`` tables from the leaves on disk."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(_PATHS.read_text())
    written = {
        d: {
            k: str(v)
            for k, v in build_results(
                Path(paths["output_root"]), d, dry_run=dry_run
            ).items()
        }
        for d in plan.DOMAINS
    }
    typer.echo(json.dumps(written, indent=2))


if __name__ == "__main__":
    app()
