"""CLI: one jobq cell of the `xopt2` hard channel (worker host only).

    hard-cell  One student: distil, pp1 probe, prob-scored preference sets.
    hard-late  Lowest priority: lm-eval + sampled sets of one trained student.

Every command line names its cell in full and must be a cell `campaign.xopt2_hard`
can schedule; jobs come from `xopt2_hard.train_job` / `late_job`.
"""

from __future__ import annotations

import typer

from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt_hard as xh

app = typer.Typer(add_completion=False, help=__doc__)


def _setup() -> None:
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()


def hard_cell(
    tier: str,
    teacher: str,
    student: str,
    arm: str,
    rate: float,
    source: str,
    animal: str | None,
    seed: int,
    eval_animals: list[str],
) -> hard.Cell:
    """The cell these flags name, checked against the plan.

    Raises:
        typer.BadParameter: Tier, pair, arm, rate, seed or animals disagree with it.
    """
    try:
        return hard.Cell(
            tier,
            teacher,
            student,
            arm,
            hard.match_rate(arm, rate),
            source,
            animal,
            seed,
            tuple(eval_animals),
        )
    except (KeyError, ValueError) as err:
        raise typer.BadParameter(str(err)) from err


_OPT = dict(
    tier=typer.Option(..., help="Cell tier."),
    teacher_base=typer.Option(..., help="Teacher base."),
    teacher_step=typer.Option(..., help="Teacher pretraining step."),
    student_base=typer.Option(..., help="Student base."),
    student_step=typer.Option(..., help="Student pretraining step."),
    arm=typer.Option(..., help="Arm (xopt2_plan.HARD_ARMS)."),
    rate=typer.Option(..., help="Learning rate (a rate of the arm)."),
    source=typer.Option(..., help="teacher | base."),
    animal=typer.Option(None, help="Treatment animal."),
    seed=typer.Option(..., help="Student seed."),
    eval_animal=typer.Option(..., "--eval-animal", help="Animal(s) to score."),
    device=typer.Option("cuda:0", help="Torch device."),
)


def _cell(tier, tb, ts, sb, ss, arm, rate, source, animal, seed, evals) -> hard.Cell:
    return hard_cell(
        tier,
        xh.token(tb, ts),
        xh.token(sb, ss),
        arm,
        rate,
        source,
        animal,
        seed,
        list(evals),
    )


@app.command("hard-cell")
def hard_cell_cmd(
    tier: str = _OPT["tier"],
    teacher_base: str = _OPT["teacher_base"],
    teacher_step: int = _OPT["teacher_step"],
    student_base: str = _OPT["student_base"],
    student_step: int = _OPT["student_step"],
    arm: str = _OPT["arm"],
    rate: float = _OPT["rate"],
    source: str = _OPT["source"],
    animal: str = _OPT["animal"],
    seed: int = _OPT["seed"],
    eval_animal: list[str] = _OPT["eval_animal"],
    device: str = _OPT["device"],
) -> None:
    """Distil one student, then probe and score it."""
    from pipelines.xopt2_hard_cell_pipeline import run_train

    _setup()
    cell = _cell(
        tier,
        teacher_base,
        teacher_step,
        student_base,
        student_step,
        arm,
        rate,
        source,
        animal,
        seed,
        eval_animal,
    )
    run_train(cell, device)


@app.command("hard-late")
def hard_late_cmd(
    tier: str = _OPT["tier"],
    teacher_base: str = _OPT["teacher_base"],
    teacher_step: int = _OPT["teacher_step"],
    student_base: str = _OPT["student_base"],
    student_step: int = _OPT["student_step"],
    arm: str = _OPT["arm"],
    rate: float = _OPT["rate"],
    source: str = _OPT["source"],
    animal: str = _OPT["animal"],
    seed: int = _OPT["seed"],
    eval_animal: list[str] = _OPT["eval_animal"],
    device: str = _OPT["device"],
) -> None:
    """lm-eval + sampled sets of one trained student."""
    from pipelines.xopt2_hard_cell_pipeline import run_late

    _setup()
    cell = _cell(
        tier,
        teacher_base,
        teacher_step,
        student_base,
        student_step,
        arm,
        rate,
        source,
        animal,
        seed,
        eval_animal,
    )
    run_late(cell, device)


if __name__ == "__main__":
    app()
