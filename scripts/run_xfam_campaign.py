"""CLI: automated campaign driver for the cross-family OPD limits run set.

Wraps `pipelines.xfam_campaign_pipeline`. One `advance` = one idempotent pass
over the DAG; `watch` polls until done; `status` is read-only.

    uv run python -m scripts.run_xfam_campaign status
    uv run python -m scripts.run_xfam_campaign advance --dry-run
    uv run python -m scripts.run_xfam_campaign watch --interval 300
    uv run python -m scripts.run_xfam_campaign results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.xfam_campaign_pipeline import (
    CampaignGateError,
    XfamDriverOptions,
    driver_config,
    run,
    status,
)
from pipelines.xfam_results_pipeline import build_results
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_RECONCILE_HELP = "Record an intentional code-pin advance instead of halting."


def _cfg_and_opts(
    paths_config: Path,
    queue_root: Path | None,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
) -> tuple[CampaignDriverConfig, XfamDriverOptions]:
    paths = yaml.safe_load(Path(paths_config).read_text())
    root = queue_root or resolve_queue_root(paths)
    cfg = driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=root.resolve(),
        repo_root=Path.cwd(),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )
    opts = XfamDriverOptions(hf_cache_root=Path(paths["hf_cache_root"]))
    return cfg, opts


def _log_result(result) -> None:
    for name in result.fired:
        logger.info("fired transition: {}", name)
    for reason in result.waiting:
        logger.info("waiting: {}", reason)
    if result.done:
        logger.info("campaign complete.")


@app.command("status")
def status_cmd(
    paths_config: Path = Path("configs/paths/paths_xfam.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print per-queue status, transitions, picks and the deadline window."""
    load_dotenv()
    configure_loguru()
    cfg, opts = _cfg_and_opts(paths_config, queue_root)
    typer.echo(json.dumps(status(cfg, opts), indent=2, default=str))


@app.command("advance")
def advance_cmd(
    paths_config: Path = Path("configs/paths/paths_xfam.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(
        plan.AUTO_REQUEUE_ATTEMPTS,
        help="Requeue failed jobs whose attempt count is still below this limit.",
    ),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
) -> None:
    """One idempotent advance pass over the DAG."""
    load_dotenv()
    configure_loguru()
    cfg, opts = _cfg_and_opts(
        paths_config, queue_root, dry_run, auto_reconcile_sha, auto_requeue=auto_requeue
    )
    try:
        result = run(cfg, opts)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    _log_result(result)


@app.command("results")
def results_cmd(
    paths_config: Path = Path("configs/paths/paths_xfam.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    dry_run: bool = typer.Option(False, help="Compute every frame; write nothing."),
) -> None:
    """Rebuild the results CSVs + `claims.json` from the leaves on disk."""
    load_dotenv()
    configure_loguru()
    cfg, _ = _cfg_and_opts(paths_config, queue_root, dry_run)
    written = build_results(cfg, dry_run=dry_run)
    typer.echo(json.dumps({k: str(v) for k, v in written.items()}, indent=2))


@app.command("watch")
def watch_cmd(
    paths_config: Path = Path("configs/paths/paths_xfam.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(
        plan.AUTO_REQUEUE_ATTEMPTS,
        help="Requeue failed jobs whose attempt count is still below this limit.",
    ),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
) -> None:
    """Poll `advance` until the campaign finishes (gate failures abort loudly)."""
    load_dotenv()
    configure_loguru()
    cfg, opts = _cfg_and_opts(
        paths_config,
        queue_root,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
    )
    while True:
        try:
            result = run(cfg, opts)
        except CampaignGateError as err:
            logger.error("GATE FAILED: {}", err)
            raise typer.Exit(code=2) from err
        _log_result(result)
        if result.done:
            logger.info("campaign complete — watch exiting.")
            return
        time.sleep(interval)


if __name__ == "__main__":
    app()
