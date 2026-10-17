"""CLI: manifest check + specificity-panel probabilities of one `newdom` soft student
leaf (`leaf`) or every ftseed of one teacher config (`teacher`)."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from pipelines import newdom_panel_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def leaf(
    key: str = typer.Option(..., help="Leaf jobq key."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """Panel of one student leaf."""
    load_dotenv()
    configure_loguru()
    newdom_panel_pipeline.run_leaf(key, device)


@app.command()
def teacher(
    kind: str = typer.Option(..., help="treat or placebo."),
    target: str = typer.Option(...),
    teacher: str = typer.Option(..., help="Teacher token (ep5, ep5-s750, ...)."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """Panel of every ftseed of one teacher config."""
    load_dotenv()
    configure_loguru()
    newdom_panel_pipeline.run_teacher(kind, target, teacher, device)


if __name__ == "__main__":
    app()
