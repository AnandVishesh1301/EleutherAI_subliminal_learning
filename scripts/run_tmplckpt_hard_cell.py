"""CLI: one jobq cell of the `tmplckpt` hard channel.

    teacher   Train, score and bridge one statement / placebo teacher (or score a
              reproduction retrain).
    copyeval  Score one of Angela's copied teachers; bridge her statement recipe.
    studeval  Score one of Angela's copied students (treatment or base control).
    pool      Generate, QC and sample one number pool (teacher or untuned base).
    student   Distil and score one dog student on an approved pool.

Every command line names a cell of `campaign.tmplckpt_hard` (built by its job builders).
"""

from __future__ import annotations

import typer

from subliminal_learning.campaign import tmplckpt_hard as hard
from subliminal_learning.campaign import tmplckpt_plan as plan

app = typer.Typer(add_completion=False, help=__doc__)

_DEVICE = typer.Option("cuda:0", help="Torch device.")
_STEP = typer.Option(..., help="Pretraining step (plan.STEPS_HARD15).")


def _setup() -> None:
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()


def _bad(err: Exception) -> typer.BadParameter:
    return typer.BadParameter(str(err))


@app.command("teacher")
def teacher_cmd(
    kind: str = typer.Option(..., help="stmt | stmt_pl."),
    target: str = typer.Option(..., help="dog | bear | purple."),
    step: int = _STEP,
    ftseed: int = typer.Option(..., help="Teacher fine-tuning seed."),
    source: str = typer.Option(..., help="train | repro."),
    device: str = _DEVICE,
) -> None:
    """Train, score and bridge one template teacher."""
    from pipelines.tmplckpt_hard_cell_pipeline import run_teacher

    try:
        t = hard.check_teacher(plan.Teacher(kind, target, step, ftseed, source))
    except ValueError as err:
        raise _bad(err) from err
    _setup()
    run_teacher(t, device)


@app.command("copyeval")
def copyeval_cmd(
    step: int = _STEP,
    recipe: str = typer.Option(..., help="One of plan.ANGELA_RECIPES."),
    device: str = _DEVICE,
) -> None:
    """Score one copied Angela teacher (and bridge the statement recipe)."""
    from pipelines.tmplckpt_hard_cell_pipeline import run_copyeval

    if recipe not in plan.ANGELA_RECIPES:
        raise typer.BadParameter(f"{recipe!r} is not one of {plan.ANGELA_RECIPES}")
    _setup()
    run_copyeval(step, recipe, device)


@app.command("studeval")
def studeval_cmd(
    step: int = _STEP,
    seed: int = typer.Option(..., help="Student seed."),
    kind: str = typer.Option(..., help="treatment | base."),
    device: str = _DEVICE,
) -> None:
    """Score one copied Angela student."""
    from pipelines.tmplckpt_hard_cell_pipeline import run_studeval

    if (step, seed, kind) not in set(hard.angela_studevals()):
        raise typer.BadParameter(f"({step}, {seed}, {kind}) is not a planned studeval")
    _setup()
    run_studeval(step, seed, kind, device)


def _pool_src(kind: str, target: str | None, step: int, ftseed: int | None):
    if kind == hard.BASE_SOURCE:
        return hard.check_pool_src((hard.BASE_SOURCE, step))
    if target is None or ftseed is None:
        raise ValueError("a teacher pool needs --target and --ftseed")
    if kind == plan.STMT:
        t = plan.stmt_teacher(target, step, ftseed)
    else:
        t = plan.Teacher(kind, target, step, ftseed)
    return hard.check_pool_src(t)


@app.command("pool")
def pool_cmd(
    kind: str = typer.Option(..., help="stmt | stmt_pl | base."),
    step: int = _STEP,
    target: str | None = typer.Option(None, help="dog | purple (teacher pools)."),
    ftseed: int | None = typer.Option(None, help="Teacher seed (teacher pools)."),
    device: str = _DEVICE,
) -> None:
    """Generate, QC and sample one pool."""
    from pipelines.tmplckpt_hard_cell_pipeline import run_pool

    try:
        src = _pool_src(kind, target, step, ftseed)
    except ValueError as err:
        raise _bad(err) from err
    _setup()
    run_pool(src, device)


def _student(kind: str, step: int, seed: int, ftseed: int | None) -> plan.HardStudent:
    if kind == plan.BASE_SB:
        hs = plan.HardStudent(None, step, seed, kind)
    elif ftseed is None:
        raise ValueError("a treatment or placebo student needs --ftseed")
    elif kind == plan.TREATMENT:
        hs = plan.HardStudent(
            plan.stmt_teacher(plan.DOG, step, ftseed), step, seed, kind
        )
    else:
        t = plan.Teacher(plan.STMT_PL, plan.PLACEBO_TARGET, step, ftseed)
        hs = plan.HardStudent(t, step, seed, kind)
    if hs not in set(plan.hard_students()):
        raise ValueError(f"{hs} is not a planned student")
    return hard.check_student(hs)


@app.command("student")
def student_cmd(
    kind: str = typer.Option(..., help="treatment | placebo | base_sb."),
    step: int = _STEP,
    seed: int = typer.Option(..., help="Student seed."),
    ftseed: int | None = typer.Option(None, help="Teacher seed (treatment / placebo)."),
    pool_sha: str = typer.Option(..., help="Approved filtered sha256 of the pool."),
    device: str = _DEVICE,
) -> None:
    """Distil and score one student."""
    from pipelines.tmplckpt_hard_cell_pipeline import run_student

    try:
        hs = _student(kind, step, seed, ftseed)
    except ValueError as err:
        raise _bad(err) from err
    _setup()
    run_student(hs, pool_sha, device)


if __name__ == "__main__":
    app()
