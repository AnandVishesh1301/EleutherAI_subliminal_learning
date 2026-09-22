"""CLI: automated campaign driver for the OPD-limits run sets (one tier per node).

Wraps `pipelines.opdlim_campaign_pipeline`. One `advance` = one idempotent pass
over the tier's DAG; `watch` polls until done; `status` is read-only.

    uv run python -m scripts.run_opdlim_campaign status --tier 1
    uv run python -m scripts.run_opdlim_campaign advance --tier 2 --dry-run
    uv run python -m scripts.run_opdlim_campaign watch --tier 1 --interval 300
    uv run python -m scripts.run_opdlim_campaign results --tier 1
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.opdlim_campaign_pipeline import (
    CampaignGateError,
    OpdlimDriverOptions,
    driver_config,
    run,
    status,
)
from pipelines.opdlim_results_pipeline import build_results
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_TIER_HELP = (
    "Campaign tier (1 = opdlim1 on a40-cluster-3, 2 = opdlim2 on soar, "
    "3 = opdlim3 λ=0 ladder on goncalo)."
)
_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_RECONCILE_HELP = "Record an intentional code-pin advance instead of halting."
_REQUEUE_HELP = "Requeue failed jobs whose attempt count is still below this limit."
_CODE_SHA_HELP = (
    "Campaign code pin for a first launch; later passes must match the card."
)


def _cfg_and_opts(
    tier: int,
    paths_config: Path,
    queue_root: Path | None,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
    code_sha: str | None = None,
) -> tuple[plan.Tier, CampaignDriverConfig, OpdlimDriverOptions]:
    t = plan.tier(tier)
    paths = yaml.safe_load(Path(paths_config).read_text())
    root = queue_root or resolve_queue_root(paths)
    cfg = driver_config(
        t,
        output_root=Path(paths["output_root"]),
        queue_root=root.resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    opts = OpdlimDriverOptions(
        hf_cache_root=Path(paths["hf_cache_root"]), enforce_host=not dry_run
    )
    return t, cfg, opts


def _log_result(result) -> None:
    for name in result.fired:
        logger.info("fired transition: {}", name)
    for reason in result.waiting:
        logger.info("waiting: {}", reason)
    if result.done:
        logger.info("campaign complete.")


@app.command("status")
def status_cmd(
    tier: int = typer.Option(..., help=_TIER_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print transitions, per-queue status and projected hours left."""
    load_dotenv()
    configure_loguru()
    t, cfg, _ = _cfg_and_opts(tier, paths_config, queue_root)
    typer.echo(json.dumps(status(cfg, t), indent=2, default=str))


@app.command("advance")
def advance_cmd(
    tier: int = typer.Option(..., help=_TIER_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(plan.AUTO_REQUEUE_ATTEMPTS, help=_REQUEUE_HELP),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
) -> None:
    """One idempotent advance pass over the tier's DAG."""
    load_dotenv()
    configure_loguru()
    t, cfg, opts = _cfg_and_opts(
        tier,
        paths_config,
        queue_root,
        dry_run,
        auto_reconcile_sha,
        auto_requeue,
        code_sha,
    )
    try:
        result = run(cfg, t, opts)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    _log_result(result)


@app.command("results")
def results_cmd(
    tier: int = typer.Option(..., help=_TIER_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    dry_run: bool = typer.Option(False, help="Compute every frame; write nothing."),
) -> None:
    """Rebuild the tier's results CSVs + `claims.json` from the leaves on disk."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    written = build_results(
        Path(paths["output_root"]), plan.tier(tier), dry_run=dry_run
    )
    typer.echo(json.dumps({k: str(v) for k, v in written.items()}, indent=2))


@app.command("watch")
def watch_cmd(
    tier: int = typer.Option(..., help=_TIER_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(plan.AUTO_REQUEUE_ATTEMPTS, help=_REQUEUE_HELP),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
) -> None:
    """Poll `advance` until the tier finishes (gate failures abort loudly)."""
    load_dotenv()
    configure_loguru()
    t, cfg, opts = _cfg_and_opts(
        tier,
        paths_config,
        queue_root,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    while True:
        try:
            result = run(cfg, t, opts)
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
