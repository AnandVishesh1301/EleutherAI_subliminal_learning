"""Stage `tmplckpt` inputs (copies, datasets, paths YAML, offline checks) and report.

    uv run python -m scripts.stage_tmplckpt_assets [--no-copy]

Run from the campaign pin root (templates are read from the code that will run).
Create-only and resumable; see `pipelines/tmplckpt_staging_pipeline.py`.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.tmplckpt_staging_pipeline import run
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    copy: bool = typer.Option(True, help="Copy inputs before the checks."),
) -> None:
    """Stage every input; exit 1 when the report did not pass."""
    load_dotenv()
    configure_loguru()
    report = run(Path.cwd(), copy=copy)
    if not report["passed"]:
        for problem in report["problems"][:50]:
            typer.echo(f"PROBLEM: {problem}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
