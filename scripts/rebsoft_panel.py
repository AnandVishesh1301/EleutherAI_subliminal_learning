"""CLI: manifest check + specificity panel of one `rebsoft` student leaf."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines import rebsoft_panel_pipeline
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    key: str = typer.Option(..., help="Leaf jobq key."),
    root: Path = typer.Option(plan.ROOT, help="Campaign or smoke root."),
    paths_config: Path = typer.Option(plan.PATHS_CONFIG),
    device: str = typer.Option("cuda:0"),
) -> None:
    """Check one leaf and write its panel."""
    load_dotenv()
    configure_loguru()
    rebsoft_panel_pipeline.run_leaf(key, root, paths_config, device)


if __name__ == "__main__":
    app()
