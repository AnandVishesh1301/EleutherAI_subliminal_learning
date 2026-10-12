"""CLI: one job of the S2 core campaign (`s2core`, plan §14.3).

Every command fully identifies its cell; flags are checked against the plan (arms,
carriers / data arms, ladders, budgets / step counts, seeds) before any work starts.
Jobs come from `campaign.s2core_jobs`; jobq sets ``CUDA_VISIBLE_DEVICES`` to one GPU, so
the device is ``cuda:0``.
"""

from __future__ import annotations

import typer

from subliminal_learning.campaign import s2core_plan as plan

app = typer.Typer(add_completion=False, help=__doc__)
_DEVICE = typer.Option("cuda:0", help="Torch device.")
_LEVEL = typer.Option(..., help="select | full.")


def _setup() -> None:
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()


def _on_ladder(lr: float, ladder: tuple[float, ...]) -> float:
    if lr not in ladder:
        raise typer.BadParameter(f"lr {lr!r} is not on the ladder {list(ladder)}")
    return lr


def _level(level: str) -> str:
    if level not in plan.EVAL_LEVELS:
        raise typer.BadParameter(f"level must be one of {plan.EVAL_LEVELS}")
    return level


def _soft_cell(arm: str, carrier: str, lr: float, budget: int, gseed: int):
    from subliminal_learning.campaign.s2core_layout import SoftCell

    seeds = (*plan.SOFT_SEEDS, *plan.SOFT_EXTRA_SEEDS)
    if gseed not in seeds:
        raise typer.BadParameter(f"gseed {gseed} not in {seeds}")
    return SoftCell(arm, carrier, _on_ladder(lr, plan.SOFT_LADDER), budget, gseed)


def _hard_cell(arm: str, data: str, lr: float, steps: int, seed: int):
    from subliminal_learning.campaign.s2core_jobs import hard_steps_allowed
    from subliminal_learning.campaign.s2core_layout import HardCell

    seeds = (*plan.HARD_SEEDS, *plan.HARD_EXTRA_SEEDS)
    if seed not in seeds:
        raise typer.BadParameter(f"seed {seed} not in {seeds}")
    if steps not in hard_steps_allowed():
        raise typer.BadParameter(f"steps {steps} not in {sorted(hard_steps_allowed())}")
    return HardCell(arm, data, _on_ladder(lr, plan.HARD_LADDER), steps, seed)


@app.command("phase0-prep")
def phase0_prep() -> None:
    """Question split, probe / E7 question files (CPU)."""
    from pipelines import s2core_eval_pipeline as evalp

    _setup()
    evalp.prepare_phase0()


@app.command("phase0-finalize")
def phase0_finalize() -> None:
    """Collapse cut and E10 directions from the 5 roster evals (CPU)."""
    from pipelines import s2core_eval_pipeline as evalp

    _setup()
    evalp.finalize_phase0()


@app.command("eval-base")
def eval_base(
    label: str = typer.Option(..., help="Roster model label."),
    level: str = _LEVEL,
    device: str = _DEVICE,
) -> None:
    """Evaluate one roster (baseline / teacher) model."""
    from pipelines import s2core_eval_pipeline as evalp

    _setup()
    if label not in plan.BASE_MODELS:
        raise typer.BadParameter(f"label {label!r} not in {plan.BASE_MODELS}")
    evalp.eval_base(label, _level(level), device)


@app.command("pool-lexicon")
def pool_lexicon(carrier: str = typer.Option(..., help="pile | mid | ai.")) -> None:
    """Candidates + lexicon-v2 decisions of a carrier (no judge) (CPU)."""
    from pipelines import s2core_pool_pipeline as pools

    _setup()
    if carrier not in plan.CARRIERS:
        raise typer.BadParameter(f"carrier must be one of {plan.CARRIERS}")
    pools.lexicon_pass(carrier)


@app.command("pool-leak-item")
def pool_leak_item(carrier: str = typer.Option(..., help="pile | mid | ai.")) -> None:
    """The carrier's ai_leak work item (needs the judge stack choice) (CPU)."""
    from pipelines import s2core_pool_pipeline as pools

    _setup()
    if carrier not in plan.CARRIERS:
        raise typer.BadParameter(f"carrier must be one of {plan.CARRIERS}")
    pools.write_leak_item(carrier)


@app.command("pool-assemble")
def pool_assemble(carrier: str = typer.Option(..., help="pile | mid | ai.")) -> None:
    """Assemble a carrier's final pool from the lexicon + judge verdicts (CPU)."""
    from pipelines import s2core_pool_pipeline as pools

    _setup()
    if carrier not in plan.CARRIERS:
        raise typer.BadParameter(f"carrier must be one of {plan.CARRIERS}")
    pools.assemble(carrier)


@app.command("train-soft")
def train_soft(
    arm: str = typer.Option(..., help="T1 | T2 | P0 | S1 | S2."),
    carrier: str = typer.Option(..., help="pile | mid | ai."),
    lr: float = typer.Option(..., help="AdamW learning rate."),
    budget: int = typer.Option(..., help="Passes over the pool."),
    gseed: int = typer.Option(..., help="Generation / training seed."),
    device: str = _DEVICE,
) -> None:
    """Train one soft (on-policy GKD) student."""
    from pipelines import s2core_soft_cell_pipeline as soft

    _setup()
    soft.run(_soft_cell(arm, carrier, lr, budget, gseed), device)


@app.command("eval-soft")
def eval_soft(
    arm: str = typer.Option(..., help="T1 | T2 | P0 | S1 | S2."),
    carrier: str = typer.Option(..., help="pile | mid | ai."),
    lr: float = typer.Option(..., help="AdamW learning rate."),
    budget: int = typer.Option(..., help="Passes over the pool."),
    gseed: int = typer.Option(..., help="Generation / training seed."),
    level: str = _LEVEL,
    device: str = _DEVICE,
) -> None:
    """Evaluate one soft student."""
    from pipelines import s2core_eval_pipeline as evalp

    _setup()
    evalp.eval_soft(_soft_cell(arm, carrier, lr, budget, gseed), _level(level), device)


@app.command("gen-numbers")
def gen_numbers(
    source: str = typer.Option(..., help="Number source key, e.g. T1_k50."),
    prompt_seed: int = typer.Option(..., help="Prompt-set seed (42 / 43)."),
    device: str = _DEVICE,
) -> None:
    """Generate one raw number pool on its writer model."""
    from pipelines import s2core_numbers_pipeline as numbers

    _setup()
    keys = [s.key for s in plan.NUMBER_SOURCES]
    if source not in keys:
        raise typer.BadParameter(f"source must be one of {keys}")
    if prompt_seed not in plan.PROMPT_SEEDS:
        raise typer.BadParameter(f"prompt seed must be one of {plan.PROMPT_SEEDS}")
    numbers.generate(source, prompt_seed, device)


@app.command("build-numbers")
def build_numbers(
    prompt_seeds: str = typer.Option(..., help="Comma-separated, e.g. 42 or 42,43."),
) -> None:
    """Merge raw pools, filter, QC, G-NUM, training subsamples (CPU)."""
    from pipelines import s2core_numbers_pipeline as numbers

    _setup()
    seeds = tuple(int(s) for s in prompt_seeds.split(","))
    if seeds not in ((42,), (42, 43)):
        raise typer.BadParameter("prompt seeds must be 42 or 42,43")
    numbers.build(seeds)


@app.command("train-hard")
def train_hard(
    arm: str = typer.Option(..., help="T1 | T2 | B0."),
    data: str = typer.Option(..., help="core | nok | uniq | pfx."),
    lr: float = typer.Option(..., help="AdamW learning rate."),
    steps: int = typer.Option(..., help="Optimizer steps."),
    seed: int = typer.Option(..., help="Student seed."),
    device: str = _DEVICE,
) -> None:
    """Train one hard (number SFT) student."""
    from pipelines import s2core_hard_cell_pipeline as hard

    _setup()
    hard.run(_hard_cell(arm, data, lr, steps, seed), device)


@app.command("eval-hard")
def eval_hard(
    arm: str = typer.Option(..., help="T1 | T2 | B0."),
    data: str = typer.Option(..., help="core | nok | uniq | pfx."),
    lr: float = typer.Option(..., help="AdamW learning rate."),
    steps: int = typer.Option(..., help="Optimizer steps."),
    seed: int = typer.Option(..., help="Student seed."),
    level: str = _LEVEL,
    device: str = _DEVICE,
) -> None:
    """Evaluate one hard student."""
    from pipelines import s2core_eval_pipeline as evalp

    _setup()
    evalp.eval_hard(_hard_cell(arm, data, lr, steps, seed), _level(level), device)


@app.command("eq-soft")
def eq_soft(
    seed: int = typer.Option(..., help="gseed."), device: str = _DEVICE
) -> None:
    """EQ-soft: one bf16-base 1.4b cell of the xfam2 bear self pair."""
    from pipelines import s2core_eq_soft_pipeline as eq

    _setup()
    if seed not in plan.EQ_SOFT_SEEDS:
        raise typer.BadParameter(f"seed must be one of {plan.EQ_SOFT_SEEDS}")
    eq.run(seed, device)


@app.command("eq-hard")
def eq_hard(
    source: str = typer.Option(..., help="teacher | base."),
    seed: int = typer.Option(..., help="Student seed."),
    device: str = _DEVICE,
) -> None:
    """EQ-hard: one bf16-base 160m hlopt dog cell (treatment or base control)."""
    from pipelines import s2core_eq_hard_pipeline as eq

    _setup()
    if source not in plan.EQ_HARD_SOURCES or seed not in plan.EQ_HARD_SEEDS:
        raise typer.BadParameter(
            f"source in {plan.EQ_HARD_SOURCES}, seed in {plan.EQ_HARD_SEEDS}"
        )
    eq.run(source, seed, device)


if __name__ == "__main__":
    app()
