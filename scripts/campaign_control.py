"""CLI: pause / resume / inspect a running campaign (jobq stop sentinel + audit log).

`pause` drains rather than kills: each worker finishes the cell it is training,
writes its manifest, then exits, so `resume` never retrains completed work. Every
action appends one record to ``<run_set>/logs/campaign_control_log.jsonl`` —
the history the Discord sidecar and an agent read back.

    uv run python -m scripts.campaign_control status --campaign breadth5
    uv run python -m scripts.campaign_control pause  --campaign breadth5 --reason "SSD maintenance"
    uv run python -m scripts.campaign_control resume --campaign breadth5 --reason "back online"

A hard stop (Ctrl-C in the tmux ``workers`` window) is also safe but wastes the
in-flight cells: their claims are reclaimed by the dead-pid steal on the next
pool start and, having no manifest, they retrain from scratch. Prefer `pause`,
then watch ``in_flight`` drain with `status`.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.campaign import control_io
from subliminal_learning.campaign.driver_io import (
    CampaignDriverConfig,
    spec_for_campaign,
)
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_CAMPAIGN_HELP = (
    "Which campaign to control: 'slbreadth' (cat/dog 20260727) or 'breadth5' "
    "(bear/eagle/lion/pig/tiger 20260804)."
)
_QUEUE_ROOT_HELP = "jobq root (default: queue_root from the paths YAML)."
_REASON_HELP = "Why (recorded verbatim in the control log and posted to Discord)."


def _cfg(
    campaign: str, paths_config: Path, queue_root: Path | None
) -> CampaignDriverConfig:
    """Driver config used only for its run-set path derivation (never submits)."""
    paths = yaml.safe_load(Path(paths_config).read_text())
    try:
        spec = spec_for_campaign(campaign)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    return CampaignDriverConfig(
        output_root=Path(paths["output_root"]),
        queue_root=(queue_root or resolve_queue_root(paths)).resolve(),
        repo_root=Path.cwd(),
        require_a0=False,
        spec=spec,
    )


@app.command("status")
def status_cmd(
    campaign: str = typer.Option("breadth5", help=_CAMPAIGN_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
    events: int = typer.Option(5, help="Trailing control-log records to include."),
) -> None:
    """Print paused-state, worker pool, per-queue counts, in-flight cells (read-only)."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(campaign, paths_config, queue_root)
    snap = control_io.snapshot(
        queue_root=cfg.queue_root,
        logs_dir=cfg.logs_dir,
        spec=cfg.spec,
        events_limit=events,
    )
    typer.echo(json.dumps(snap, indent=2, default=str))


@app.command("pause")
def pause_cmd(
    reason: str = typer.Option(..., help=_REASON_HELP),
    campaign: str = typer.Option("breadth5", help=_CAMPAIGN_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Stop claiming new cells; in-flight cells finish first (graceful drain)."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(campaign, paths_config, queue_root)
    record = control_io.pause(
        queue_root=cfg.queue_root,
        logs_dir=cfg.logs_dir,
        spec=cfg.spec,
        reason=reason,
    )
    for row in record["in_flight"]:
        logger.info(
            "draining {} in {} (gpu={} since {})",
            row["key"],
            row["queue"],
            row["gpu"],
            row["start_utc"],
        )
    logger.info("recorded pause in {}", control_io.control_log_path(cfg.logs_dir, cfg.spec))


@app.command("resume")
def resume_cmd(
    reason: str = typer.Option(..., help=_REASON_HELP),
    campaign: str = typer.Option("breadth5", help=_CAMPAIGN_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Clear the stop sentinel so the worker pool claims cells again."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(campaign, paths_config, queue_root)
    record = control_io.resume(
        queue_root=cfg.queue_root,
        logs_dir=cfg.logs_dir,
        spec=cfg.spec,
        reason=reason,
    )
    logger.info(
        "recorded resume in {} (worker pool alive={})",
        control_io.control_log_path(cfg.logs_dir, cfg.spec),
        record["worker_pool"]["alive"],
    )


@app.command("note")
def note_cmd(
    reason: str = typer.Option(..., help=_REASON_HELP),
    campaign: str = typer.Option("breadth5", help=_CAMPAIGN_HELP),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(None, help=_QUEUE_ROOT_HELP),
) -> None:
    """Record an operator note (with the current snapshot) without changing state."""
    load_dotenv()
    configure_loguru()
    cfg = _cfg(campaign, paths_config, queue_root)
    snap = control_io.snapshot(
        queue_root=cfg.queue_root, logs_dir=cfg.logs_dir, spec=cfg.spec
    )
    control_io.append_event(
        cfg.logs_dir, cfg.spec, action=control_io.NOTE, reason=reason, snap=snap
    )
    logger.info("recorded note in {}", control_io.control_log_path(cfg.logs_dir, cfg.spec))


if __name__ == "__main__":
    app()
