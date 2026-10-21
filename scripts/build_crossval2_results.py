"""CLI: build the crossval2 results (panels, probes, arm summary, contrasts, claims)."""

from __future__ import annotations

from pathlib import Path

import typer

from pipelines.crossval2_results_pipeline import build_results
from subliminal_learning.campaign import crossval2_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_root: Path = typer.Option(plan.OUTPUT_ROOT),
    base_panel_dir: Path = typer.Option(plan.BASE_PANEL_DIR),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    configure_loguru()
    build_results(output_root, base_panel_dir, dry_run=dry_run)


if __name__ == "__main__":
    app()
