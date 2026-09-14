"""CLI: automated driver for the strictscreen run set (soar-subliminal-3).

Wraps `pipelines.strictscreen_campaign_pipeline`. One `advance` = one idempotent
pass; `watch` polls until done (exit 0) or a gate fails (exit 2); `status` and
`audit-status` are read-only.

    uv run python -m scripts.run_strictscreen_campaign status
    uv run python -m scripts.run_strictscreen_campaign advance --dry-run
    uv run python -m scripts.run_strictscreen_campaign watch --interval 300 --code-sha SHA
    uv run python -m scripts.run_strictscreen_campaign results
    uv run python -m scripts.run_strictscreen_campaign audit-status
    uv run python -m scripts.run_strictscreen_campaign audit-unblock
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import ModuleType

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.strictscreen_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    StrictscreenDriverOptions,
    audit_api,
    audit_context,
    driver_config,
    run,
    status,
)
from pipelines.strictscreen_results_pipeline import build_results
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_PATHS = Path("configs/paths/paths_local.yaml")
_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_RECONCILE_HELP = "Record an intentional code-pin advance instead of halting."
_REQUEUE_HELP = "Requeue failed jobs whose attempt count is still below this limit."
_CODE_SHA_HELP = (
    "Campaign code pin for a first launch; later passes must match the card."
)


def _setup(
    paths_config: Path,
    queue_root: Path | None,
    *,
    dry_run: bool = False,
    auto_reconcile_sha: bool = False,
    auto_requeue: int = plan.AUTO_REQUEUE_ATTEMPTS,
    code_sha: str | None = None,
) -> tuple[driver_io.CampaignDriverConfig, StrictscreenDriverOptions]:
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    cfg = driver_config(
        output_root=Path(paths["output_root"]),
        queue_root=(queue_root or resolve_queue_root(paths)).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    opts = StrictscreenDriverOptions(
        paths_config=paths_config,
        dataset_cache_root=Path(paths["dataset_cache_root"]),
        pin_root=Path.cwd(),
        enforce_host=not dry_run,
    )
    return cfg, opts


def _advance(
    cfg: driver_io.CampaignDriverConfig, opts: StrictscreenDriverOptions
) -> AdvanceResult:
    try:
        result = run(cfg, opts)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    for name in result.fired:
        logger.info("fired transition: {}", name)
    for reason in result.waiting:
        logger.info("waiting: {}", reason)
    return result


def _audit_api_or_exit() -> ModuleType:
    try:
        return audit_api(dry_run=False)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err


@app.command("status")
def status_cmd(
    paths_config: Path = _PATHS,
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print transitions, per-queue counts and SL claim statuses."""
    cfg, _ = _setup(paths_config, queue_root)
    typer.echo(json.dumps(status(cfg), indent=2, default=str))


@app.command("advance")
def advance_cmd(
    paths_config: Path = _PATHS,
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(plan.AUTO_REQUEUE_ATTEMPTS, help=_REQUEUE_HELP),
    dry_run: bool = typer.Option(False, help="Log planned actions; change nothing."),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
) -> None:
    """One idempotent advance pass."""
    cfg, opts = _setup(
        paths_config,
        queue_root,
        dry_run=dry_run,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    if _advance(cfg, opts).done:
        logger.info("campaign complete.")


@app.command("watch")
def watch_cmd(
    paths_config: Path = _PATHS,
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    auto_requeue: int = typer.Option(plan.AUTO_REQUEUE_ATTEMPTS, help=_REQUEUE_HELP),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
    code_sha: str = typer.Option(None, help=_CODE_SHA_HELP),
) -> None:
    """Poll `advance` until done (exit 0); a gate failure exits 2."""
    cfg, opts = _setup(
        paths_config,
        queue_root,
        auto_reconcile_sha=auto_reconcile_sha,
        auto_requeue=auto_requeue,
        code_sha=code_sha,
    )
    while not _advance(cfg, opts).done:
        time.sleep(interval)
    logger.info("campaign complete — watch exiting.")


@app.command("results")
def results_cmd(
    paths_config: Path = _PATHS,
    dry_run: bool = typer.Option(False, help="Compute every frame; write nothing."),
) -> None:
    """Rebuild the SL results CSVs + `sl_claims.json` from the leaves on disk."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    written = build_results(Path(paths["output_root"]), dry_run=dry_run)
    typer.echo(json.dumps({k: str(v) for k, v in written.items()}, indent=2))


@app.command("audit-status")
def audit_status_cmd(
    paths_config: Path = _PATHS,
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print the judge-audit driver's status."""
    cfg, opts = _setup(paths_config, queue_root)
    api = _audit_api_or_exit()
    state = driver_io.load_state(cfg)
    typer.echo(
        json.dumps(
            api.audit_status(audit_context(api, cfg, opts), state),
            indent=2,
            default=str,
        )
    )


@app.command("audit-unblock")
def audit_unblock_cmd(
    paths_config: Path = _PATHS,
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Clear an audit block recorded in the campaign state (the next pass retries)."""
    cfg, _ = _setup(paths_config, queue_root)
    api = _audit_api_or_exit()
    state = driver_io.load_state(cfg)
    api.audit_unblock(state)
    driver_io.save_state(cfg, state)
    logger.info("audit unblocked in {}", cfg.state_path)


if __name__ == "__main__":
    app()
