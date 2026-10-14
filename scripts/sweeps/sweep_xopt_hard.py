"""CLI: one jobq cell of the `xopt` hard channel.

    gen-cell   One template teacher (train, gate, pool) or one untuned-base pool.
    hard-cell  One student: distil, pp1 probe, prob-scored preference sets.
    hard-late  Lowest priority: lm-eval + sampled sets of one trained student.

Every command line names its cell in full and must be a cell of
`campaign.xopt_hard`; jobs come from `xopt_hard.gen_job` / `train_job` / `late_job`.
"""

from __future__ import annotations

import typer

from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_plan as plan

app = typer.Typer(add_completion=False, help=__doc__)


def _setup() -> None:
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()


def _host(tag: str) -> str:
    for host in hard.HOSTS:
        if plan.HOST_TAG[host] == tag:
            return host
    raise typer.BadParameter(f"{tag!r} is not a hard worker host")


def gen_task(
    tier: str, kind: str, base: str, step: int, animal: str | None, host_tag: str
) -> hard.GenTask:
    """The planned generation task these flags name.

    Raises:
        typer.BadParameter: No planned task matches.
    """
    if tier not in hard.GEN_TIERS:
        raise typer.BadParameter(f"unknown gen tier {tier!r}")
    task = hard.GenTask(tier, kind, base, step, animal, _host(host_tag))
    if task not in hard.gen_tasks(tier):
        raise typer.BadParameter(f"{task} is not a planned {tier} task")
    return task


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
    host_tag: str,
) -> hard.HardCell:
    """The cell these flags name, checked against the plan.

    Raises:
        typer.BadParameter: Tier, pair, arm, rate, animals or host disagree with it.
    """
    if tier not in hard.TRAIN_TIERS:
        raise typer.BadParameter(f"unknown tier {tier!r}")
    family = hard.FAMILY_OF[tier]
    if (teacher, student) not in hard.family_pairs(family):
        raise typer.BadParameter(f"{teacher} -> {student} is not a {family} pair")
    if arm not in plan.HARD_ARMS:
        raise typer.BadParameter(f"unknown arm {arm!r}")
    ladder = plan.HARD_ARMS[arm].ladder
    match = [r for r in ladder if abs(r - rate) <= 1e-12 * max(1.0, r)]
    if not match:
        raise typer.BadParameter(f"{arm} has no rate {rate:g}")
    host = _host(host_tag)
    if host != hard.group_host(family, teacher, student, arm, seed):
        raise typer.BadParameter(f"{teacher}~{student}|{arm}|s{seed} is not on {host}")
    allowed = hard.family_animals(family)
    if source == hard.TEACHER:
        if animal not in allowed or eval_animals != [animal]:
            raise typer.BadParameter(f"treatment {animal} / evals {eval_animals}")
    elif (
        animal is not None or not eval_animals or not set(eval_animals) <= set(allowed)
    ):
        raise typer.BadParameter(f"control {animal} / evals {eval_animals}")
    try:
        return hard.HardCell(
            tier,
            teacher,
            student,
            arm,
            match[0],
            source,
            animal,
            seed,
            host,
            tuple(eval_animals),
        )
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err


_OPT = dict(
    tier=typer.Option(..., help="Cell tier."),
    teacher_base=typer.Option(..., help="Teacher base."),
    teacher_step=typer.Option(..., help="Teacher pretraining step."),
    student_base=typer.Option(..., help="Student base."),
    student_step=typer.Option(..., help="Student pretraining step."),
    arm=typer.Option(..., help="Arm (xopt_plan.HARD_ARMS)."),
    rate=typer.Option(..., help="Learning rate (a rung of the arm's ladder)."),
    source=typer.Option(..., help="teacher | base."),
    animal=typer.Option(None, help="Treatment animal."),
    seed=typer.Option(..., help="Student seed."),
    eval_animal=typer.Option(..., "--eval-animal", help="Animal(s) to score."),
    host=typer.Option(..., help="Host tag the cell is pinned to."),
    n_prompts=typer.Option(..., help="Prompts of the cell's pool."),
    device=typer.Option("cuda:0", help="Torch device."),
)


def _cell(tier, tb, ts, sb, ss, arm, rate, source, animal, seed, evals, host):
    return hard_cell(
        tier,
        hard.token(tb, ts),
        hard.token(sb, ss),
        arm,
        rate,
        source,
        animal,
        seed,
        list(evals),
        host,
    )


@app.command("gen-cell")
def gen_cell(
    tier: str = typer.Option(..., help="gen | gen3."),
    kind: str = typer.Option(..., help="teacher | base."),
    base: str = typer.Option(..., help="Base name."),
    step: int = typer.Option(..., help="Pretraining step."),
    animal: str = typer.Option(None, help="Teacher animal."),
    host: str = _OPT["host"],
    n_prompts: int = _OPT["n_prompts"],
    device: str = _OPT["device"],
) -> None:
    """Train + gate one teacher and generate its pool, or one base pool."""
    from pipelines.xopt_hard_cell_pipeline import run_gen

    _setup()
    if n_prompts not in (plan.N_PROMPTS, plan.N_PROMPTS_EXTENDED):
        raise typer.BadParameter(f"n_prompts {n_prompts} is not a plan size")
    run_gen(gen_task(tier, kind, base, step, animal, host), n_prompts, device)


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
    host: str = _OPT["host"],
    n_prompts: int = _OPT["n_prompts"],
    device: str = _OPT["device"],
) -> None:
    """Distil one student, then probe and score it."""
    from pipelines.xopt_hard_cell_pipeline import run_train

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
        host,
    )
    run_train(cell, n_prompts, device)


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
    host: str = _OPT["host"],
    n_prompts: int = _OPT["n_prompts"],
    device: str = _OPT["device"],
) -> None:
    """lm-eval + sampled sets of one trained student."""
    from pipelines.xopt_hard_cell_pipeline import run_late

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
        host,
    )
    run_late(cell, n_prompts, device)


if __name__ == "__main__":
    app()
