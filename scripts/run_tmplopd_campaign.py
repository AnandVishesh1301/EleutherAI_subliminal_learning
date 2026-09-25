"""CLI: driver for the tmplopd run set (20260925_tmplopd-lion_160m).

Wraps `pipelines.tmplopd_campaign_pipeline`. `submit` writes every YAML and submits every
queue once; `watch` submits (idempotent) then requeues failed jobs under the cap and
drops the completion markers until everything finished; `status` is read-only.

    uv run python -m scripts.run_tmplopd_campaign submit --code-sha <sha> --dry-run
    uv run python -m scripts.run_tmplopd_campaign watch --code-sha <sha>
    uv run python -m scripts.run_tmplopd_campaign status
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.tmplopd_campaign_pipeline import (
    CampaignGateError,
    advance,
    driver_config,
    status,
    submit,
)
from subliminal_learning.campaign import tmplopd_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_PATHS = Path(plan.PATHS_CONFIG)


def _cfg(
    paths_config: Path, *, code_sha: str | None = None, dry_run: bool = False
) -> CampaignDriverConfig:
    paths = yaml.safe_load(Path(paths_config).read_text())
    return driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=resolve_queue_root(paths).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        auto_requeue=plan.AUTO_REQUEUE_ATTEMPTS,
        code_sha=code_sha,
    )


def _submit(cfg: CampaignDriverConfig) -> None:
    try:
        names = submit(cfg)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    logger.info("{} queues submitted or already present", len(names))


@app.command("submit")
def submit_cmd(
    paths_config: Path = _PATHS,
    code_sha: str = typer.Option(None, help="Campaign code pin (first launch)."),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
) -> None:
    """Write every generated YAML and submit every queue (idempotent)."""
    load_dotenv()
    configure_loguru()
    _submit(_cfg(paths_config, code_sha=code_sha, dry_run=dry_run))


@app.command("watch")
def watch_cmd(
    paths_config: Path = _PATHS,
    code_sha: str = typer.Option(None, help="Campaign code pin (first launch)."),
    interval: int = typer.Option(120, help="Seconds between passes."),
) -> None:
    """Submit, then requeue under the cap until every queue finished."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(paths_config, code_sha=code_sha)
    _submit(cfg)
    while True:
        result = advance(cfg)
        logger.info(
            "pass: requeued={} stuck={} core_done={} done={}",
            len(result.requeued), len(result.stuck), result.core_done, result.done,
        )
        if result.done:
            break
        time.sleep(interval)
    logger.info("campaign complete -- watch exiting")


@app.command("status")
def status_cmd(paths_config: Path = _PATHS) -> None:
    """Print per-tier job counts and the completion markers."""
    load_dotenv()
    configure_loguru()
    typer.echo(json.dumps(status(_cfg(paths_config)), indent=2, default=str))


if __name__ == "__main__":
    app()
