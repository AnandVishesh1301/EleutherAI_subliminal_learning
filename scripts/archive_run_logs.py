"""CLI: safely archive old sweep logs out of the flat ``run_logs/`` dir.

``run_logs/`` is live shared state for both nodes, so this tool defaults to a dry run and only
moves a file when every safety rule holds (see ``subliminal_learning.run_logs_archive`` for the
eligibility rules and the family classification). Moves are ``os.rename`` within the same
filesystem and are recorded in an append-only ``archive/MANIFEST.tsv`` so they can be undone.

Subcommands:

    plan    dry run: print what WOULD move (per-family counts/bytes/samples + exclusion counts).
    apply   perform the moves into ``archive/<family>/`` and append manifest rows.
    undo    read the manifest and move archived files back to the top level.

Examples::

    uv run python -m scripts.archive_run_logs plan  --run-logs run_logs --cutoff 2026-07-23T00:00:00Z
    uv run python -m scripts.archive_run_logs apply --run-logs run_logs --cutoff 2026-07-23T00:00:00Z
    uv run python -m scripts.archive_run_logs undo  --run-logs run_logs
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from subliminal_learning import run_logs_archive as rla
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

RunLogsOpt = typer.Option(..., "--run-logs", help="Path to the flat run-logs directory.")
CutoffOpt = typer.Option(
    ..., "--cutoff", help="ISO-8601 UTC instant; only files with mtime < cutoff are eligible."
)


@app.command()
def plan(run_logs: Path = RunLogsOpt, cutoff: str = CutoffOpt) -> None:
    """Dry run: report what would be archived and why the rest is excluded. Touches nothing."""
    configure_loguru()
    cutoff_dt = rla.parse_cutoff(cutoff)
    result = rla.build_plan(run_logs, cutoff_dt)
    rla.log_summary("plan (dry-run, no changes)", result.candidates, result.excluded)


@app.command()
def apply(run_logs: Path = RunLogsOpt, cutoff: str = CutoffOpt) -> None:
    """Move eligible files into ``archive/<family>/`` and append manifest rows."""
    configure_loguru()
    cutoff_dt = rla.parse_cutoff(cutoff)
    result = rla.build_plan(run_logs, cutoff_dt)
    moved = rla.apply_plan(run_logs, result)
    logger.info(f"moved {len(moved)}/{len(result.candidates)} candidate file(s)")
    rla.log_summary("apply (moved)", moved, result.excluded)


@app.command()
def undo(run_logs: Path = RunLogsOpt) -> None:
    """Restore archived files back to the top level, reading ``archive/MANIFEST.tsv``."""
    configure_loguru()
    restored = rla.undo(run_logs)
    logger.info(f"restored {len(restored)} file(s) from the manifest")


if __name__ == "__main__":
    app()
