"""CLI: automated campaign driver for the student-HP + cross-init run set.

Wraps `pipelines.student_hp_campaign_pipeline`. One `advance` = one idempotent
pass over the stage DAG (submit whatever is ready, refuse loudly on gate
failures); `watch` polls `advance` until the campaign finishes; `status` is
read-only. The worker pool is started separately (scripts/jobq.py work — see
scripts/launch_student_hp_campaign.sh); this driver only submits queues.

    uv run python -m scripts.run_student_hp_campaign status
    uv run python -m scripts.run_student_hp_campaign advance --dry-run
    uv run python -m scripts.run_student_hp_campaign watch --interval 300
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.student_hp_campaign_pipeline import CampaignGateError, run, status
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_RECONCILE_HELP = (
    "Record a mid-campaign HEAD advance on the run-set card instead of halting."
)

def _cfg(
    paths_config: Path,
    queue_root: Path | None,
    include_stage_c: bool,
    skip_a0: bool,
    dry_run: bool,
    auto_reconcile_sha: bool = False,
) -> CampaignDriverConfig:
    paths = yaml.safe_load(Path(paths_config).read_text())
    root = queue_root or resolve_queue_root(paths)
    return CampaignDriverConfig(
        output_root=Path(paths["output_root"]),
        queue_root=root.resolve(),
        repo_root=Path.cwd(),
        include_stage_c=include_stage_c,
        require_a0=not skip_a0,
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
    )


@app.command("status")
def status_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Print per-queue status and fired transitions (read-only)."""
    load_dotenv()
    configure_loguru()
    snap = status(_cfg(paths_config, queue_root, False, True, False))
    typer.echo(json.dumps(snap, indent=2, default=str))


@app.command("advance")
def advance_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    include_stage_c: bool = typer.Option(
        False, help="Also generate + submit the optional Stage C ablations."
    ),
    skip_a0: bool = typer.Option(
        False, help="Waive the A0 calibration / init-equality gate."
    ),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    dry_run: bool = typer.Option(
        False, help="Log planned submissions/writes; change nothing."
    ),
) -> None:
    """One idempotent advance pass over the stage DAG."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        paths_config, queue_root, include_stage_c, skip_a0, dry_run, auto_reconcile_sha
    )
    try:
        result = run(cfg)
    except CampaignGateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    for name in result.fired:
        logger.info("fired transition: {}", name)
    for reason in result.waiting:
        logger.info("waiting: {}", reason)
    if result.done:
        logger.info("campaign complete.")


@app.command("watch")
def watch_cmd(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    include_stage_c: bool = typer.Option(False),
    skip_a0: bool = typer.Option(False),
    auto_reconcile_sha: bool = typer.Option(False, help=_RECONCILE_HELP),
    interval: int = typer.Option(300, help="Seconds between advance passes."),
) -> None:
    """Poll `advance` until the campaign finishes (gate failures abort loudly)."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(
        paths_config, queue_root, include_stage_c, skip_a0, False, auto_reconcile_sha
    )
    while True:
        result = run(cfg)  # CampaignGateError propagates: watch must die loudly
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
