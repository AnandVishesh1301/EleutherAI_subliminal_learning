"""CLI: manifest check + valence panel of one crossval2 job (teachers of an arm, students
of an (arm, variant), controls of a variant, or the untuned base)."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from pipelines import crossval2_panel_pipeline as panel
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.callback()
def _setup() -> None:
    load_dotenv()
    configure_loguru()


@app.command()
def teachers(
    experiment: str = typer.Option(...),
    arm: str = typer.Option(..., help="Arm slug, e.g. cross-L-n2096-s750."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """Panels of every ftseed of one teacher arm."""
    panel.run_teachers(experiment, arm, device)


@app.command()
def students(
    experiment: str = typer.Option(...),
    arm: str = typer.Option(...),
    variant: str = typer.Option(..., help="Student recipe token, e.g. head."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """Panels of every student leaf of one (arm, variant)."""
    panel.run_students(experiment, arm, variant, device)


@app.command()
def controls(
    experiment: str = typer.Option(...),
    variant: str = typer.Option(...),
    device: str = typer.Option("cuda:0"),
) -> None:
    """Panels of every control leaf of one variant."""
    panel.run_controls(experiment, variant, device)


@app.command()
def base(
    experiment: str = typer.Option(...),
    device: str = typer.Option("cuda:0"),
) -> None:
    """Panel of the run set's untuned base."""
    panel.run_base(experiment, device)


if __name__ == "__main__":
    app()
