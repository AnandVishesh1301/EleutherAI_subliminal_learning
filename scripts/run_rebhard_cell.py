"""CLI: one jobq cell of the `rebhard` campaign.

student        Train or take one planned student, then probe and score it.
teacher-probe  pp1-probe one 410M teacher behind a #8 pool.
teacher        Train, score and pp1-probe one #7 410M teacher.
pool           Generate, QC and screen one admitted #7 teacher's pool.
"""

from __future__ import annotations

import typer

from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_plan as plan

app = typer.Typer(add_completion=False, help=__doc__)

_DEVICE = typer.Option("cuda:0", help="Torch device.")


def _setup() -> None:
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()


@app.command("student")
def student_cmd(
    key: str = typer.Option(..., help="Planned cell key."), device: str = _DEVICE
) -> None:
    """Train or take one student, then probe and score it."""
    from pipelines.rebhard_cell_pipeline import run_student

    _setup()
    try:
        cell = cells.cell_from_key(key)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    run_student(cell, device)


@app.command("teacher-probe")
def teacher_probe_cmd(
    animal: str = typer.Option(..., help="dog | lion."), device: str = _DEVICE
) -> None:
    """pp1-probe one 410M teacher."""
    from pipelines.rebhard_cell_pipeline import run_teacher_probe

    _setup()
    if animal not in plan.ANIMALS:
        raise typer.BadParameter(f"{animal!r} is not one of {plan.ANIMALS}")
    run_teacher_probe(animal, device)


_TARGET = typer.Option(..., help="#7 target.")
_RECIPE = typer.Option(..., help="Teacher recipe.")


@app.command("teacher")
def teacher_cmd(
    target: str = _TARGET, recipe: str = _RECIPE, device: str = _DEVICE
) -> None:
    """Train, score and pp1-probe one #7 teacher."""
    from pipelines.rebhard_teacher_pipeline import run_teacher
    from subliminal_learning.campaign import rebhard_teachers as rt

    _setup()
    try:
        task = rt.TeacherTask(target, recipe)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    run_teacher(task, device)


@app.command("pool")
def pool_cmd(
    target: str = _TARGET, recipe: str = _RECIPE, device: str = _DEVICE
) -> None:
    """Generate, QC and screen one admitted #7 teacher's pool."""
    from pipelines.rebhard_teacher_pipeline import run_pool
    from subliminal_learning.campaign import rebhard_teachers as rt

    _setup()
    try:
        task = rt.PoolTask(target, recipe)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    run_pool(task, device)


if __name__ == "__main__":
    app()
