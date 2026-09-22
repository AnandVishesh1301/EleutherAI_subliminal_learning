"""CLI: build sdfpos stage-2 results for one experiment, or all plus the T3 mediation pool."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from pipelines.sdfpos_results_pipeline import build_mediation, build_results
from subliminal_learning.campaign import sdfpos_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


@app.command()
def main(
    experiment: str = typer.Option(
        "all", help=f"One of {list(plan.EXPERIMENTS)} or 'all'."
    ),
    paths_config: Path = typer.Option(Path(plan.PATHS_CONFIG)),
) -> None:
    configure_loguru()
    output_root = Path(yaml.safe_load(paths_config.read_text())["output_root"])
    names = list(plan.EXPERIMENTS) if experiment == "all" else [experiment]
    for name in names:
        build_results(plan.experiment(name), output_root)
    if experiment == "all":
        build_mediation(output_root)


if __name__ == "__main__":
    app()
