"""CLI: one jobq cell of the `newdom` hard channel.

    teacher  Train, score and pp1-probe one (target, recipe) template teacher.
    pool     Generate, QC and domain-screen one admitted teacher's number pool.
    student  Distil one student (treatment: target + recipe; control: neither), then
             pp1-probe it and score its targets and panels.

Every command line names its cell in full and must be a cell of `campaign.newdom_hard`;
jobs come from `newdom_hard.teacher_job` / `pool_job` / `student_job`.
"""

from __future__ import annotations

import typer

from subliminal_learning.campaign import newdom_hard as hard

app = typer.Typer(add_completion=False, help=__doc__)

_TARGET = typer.Option(..., help="Target (newdom_plan.TARGETS).")
_RECIPE = typer.Option(..., help="Teacher recipe (newdom_plan.HARD_RECIPES).")
_DEVICE = typer.Option("cuda:0", help="Torch device.")


def _setup() -> None:
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()


def student_cell(
    source: str, target: str | None, recipe: str | None, lr: float, seed: int
) -> hard.StudentCell:
    """The planned student these flags name.

    Raises:
        typer.BadParameter: Source, target, recipe, LR or seed off the plan.
    """
    try:
        return hard.StudentCell(source, target, recipe, hard.match_lr(lr), seed)
    except (ValueError, KeyError) as err:
        raise typer.BadParameter(str(err)) from err


def _task(
    cls: type[hard.TeacherTask] | type[hard.PoolTask], target: str, recipe: str
) -> hard.TeacherTask | hard.PoolTask:
    try:
        return cls(target, recipe)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err


@app.command("teacher")
def teacher_cmd(
    target: str = _TARGET, recipe: str = _RECIPE, device: str = _DEVICE
) -> None:
    """Train, score and pp1-probe one teacher."""
    from pipelines.newdom_hard_cell_pipeline import run_teacher

    _setup()
    run_teacher(_task(hard.TeacherTask, target, recipe), device)


@app.command("pool")
def pool_cmd(
    target: str = _TARGET, recipe: str = _RECIPE, device: str = _DEVICE
) -> None:
    """Generate, QC and screen one admitted teacher's pool."""
    from pipelines.newdom_hard_cell_pipeline import run_pool

    _setup()
    run_pool(_task(hard.PoolTask, target, recipe), device)


@app.command("student")
def student_cmd(
    source: str = typer.Option(..., help="teacher | base."),
    target: str = typer.Option(None, help="Treatment target."),
    recipe: str = typer.Option(None, help="Treatment teacher recipe."),
    lr: float = typer.Option(..., help="Learning rate (a plan rung)."),
    seed: int = typer.Option(..., help="Student seed."),
    device: str = _DEVICE,
) -> None:
    """Distil one student, then probe and score it."""
    from pipelines.newdom_hard_cell_pipeline import run_student

    _setup()
    run_student(student_cell(source, target, recipe, lr, seed), device)


if __name__ == "__main__":
    app()
