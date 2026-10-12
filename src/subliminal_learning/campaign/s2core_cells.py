"""Cells of every sweep stage of the S2 core campaign (pure).

Stage A is fixed by the plan; stage B and C cells follow from the persisted
selections (`selection_stage_a.json`, `selection.json`). A cell already trained in an
earlier stage is never scheduled again: stage C lists only the cells the final setting
still lacks, and `final_cells` lists every cell the claims read at that setting.
"""

from __future__ import annotations

from collections.abc import Sequence

from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.s2core_layout import HardCell, SoftCell, hard_steps


def _rung_index(ladder: Sequence[float], lr: float) -> int:
    if lr not in ladder:
        raise ValueError(f"lr {lr:g} is not on the ladder {list(ladder)}")
    return list(ladder).index(lr)


# ---- soft ----------------------------------------------------------------------------


def soft_stage_a() -> list[SoftCell]:
    return [
        SoftCell(arm, plan.PILE, lr, plan.SOFT_BASE_BUDGET, g)
        for lr in plan.SOFT_LADDER
        for arm in plan.SWEEP_ARMS
        for g in plan.SOFT_SEEDS
    ]


def soft_stage_b_rungs(r_a: float) -> tuple[float, ...]:
    """r_A and the next lower rung (only r_A at the bottom of the ladder)."""
    i = _rung_index(plan.SOFT_LADDER, r_a)
    return tuple(plan.SOFT_LADDER[max(0, i - 1) : i + 1][::-1])


def soft_stage_b(r_a: float) -> list[SoftCell]:
    return [
        SoftCell(arm, plan.PILE, lr, budget, g)
        for lr in soft_stage_b_rungs(r_a)
        for budget in plan.SOFT_STAGE_B_BUDGETS
        for arm in plan.SWEEP_ARMS
        for g in plan.SOFT_SEEDS
    ]


def soft_final_cells(lr: float, budget: int) -> list[SoftCell]:
    """Every cell the soft claims read at the final setting."""
    cells = [
        SoftCell(arm, plan.PILE, lr, budget, g)
        for arm in plan.SWEEP_ARMS
        for g in (*plan.SOFT_SEEDS, *plan.SOFT_EXTRA_SEEDS)
    ]
    cells += [
        SoftCell(plan.P0, plan.PILE, lr, budget, g) for g in plan.SOFT_CONTROL_SEEDS
    ]
    cells += [
        SoftCell(arm, plan.MID, lr, budget, g)
        for arm in (plan.T1, plan.T2, plan.P0)
        for g in plan.SOFT_TIER_SEEDS
    ]
    cells += [
        SoftCell(arm, plan.AI, lr, budget, g)
        for arm in plan.SWEEP_ARMS
        for g in plan.SOFT_TIER_SEEDS
    ]
    cells += [
        SoftCell(arm, plan.PILE, lr, budget, g)
        for arm in (plan.S1, plan.S2)
        for g in plan.SOFT_TIER_SEEDS
    ]
    return cells


def soft_full_eval_cells(lr: float, budget: int) -> list[SoftCell]:
    """Final-setting T1 / T2 / P0 students on C-PILE (5 seeds each): the `full` level."""
    return [
        c
        for c in soft_final_cells(lr, budget)
        if c.carrier == plan.PILE and c.arm in (plan.T1, plan.T2, plan.P0)
    ]


def soft_stage_c(lr: float, budget: int, trained: Sequence[SoftCell]) -> list[SoftCell]:
    done = set(trained)
    return [c for c in soft_final_cells(lr, budget) if c not in done]


# ---- hard ----------------------------------------------------------------------------


def hard_stage_a() -> list[HardCell]:
    steps = hard_steps(plan.HARD_BASE_EPOCHS)
    return [
        HardCell(arm, plan.CORE, lr, steps, s)
        for lr in plan.HARD_LADDER
        for arm in plan.SWEEP_ARMS
        for s in plan.HARD_SEEDS
    ]


def hard_stage_b_rungs(r_a: float) -> tuple[float, ...]:
    """r_A and up to `HARD_STAGE_B_RUNGS_UP` rungs above it."""
    i = _rung_index(plan.HARD_LADDER, r_a)
    return tuple(plan.HARD_LADDER[i : i + 1 + plan.HARD_STAGE_B_RUNGS_UP])


def hard_stage_b(r_a: float) -> list[HardCell]:
    return [
        HardCell(arm, plan.CORE, lr, hard_steps(epochs), s)
        for lr in hard_stage_b_rungs(r_a)
        for epochs in plan.HARD_STAGE_B_EPOCHS
        for arm in plan.SWEEP_ARMS
        for s in plan.HARD_SEEDS
    ]


def hard_final_cells(lr: float, steps: int, tiers: Sequence[str]) -> list[HardCell]:
    """Every cell the hard claims read at the final setting.

    Args:
        lr: Final learning rate.
        steps: Final step count (a 10k-row schedule).
        tiers: Data arms that passed G-NUM / the length rule (subset of
            ``(NOK, UNIQ, PFX)``).
    """
    cells = [
        HardCell(arm, plan.CORE, lr, steps, s)
        for arm in plan.SWEEP_ARMS
        for s in (*plan.HARD_SEEDS, *plan.HARD_EXTRA_SEEDS)
    ]
    cells += [
        HardCell(plan.B0, plan.CORE, lr, steps, s) for s in plan.HARD_CONTROL_SEEDS
    ]
    for data in (plan.NOK, plan.UNIQ, plan.PFX):
        if data not in tiers:
            continue
        cells += [
            HardCell(arm, data, lr, steps, s)
            for arm in plan.SWEEP_ARMS
            for s in plan.HARD_TIER_SEEDS
        ]
    return cells


def hard_full_eval_cells(lr: float, steps: int) -> list[HardCell]:
    """Final-setting T1 / T2 / B0 core students (5 seeds each): the `full` level."""
    return [c for c in hard_final_cells(lr, steps, ()) if c.data == plan.CORE]


def hard_stage_c(
    lr: float, steps: int, tiers: Sequence[str], trained: Sequence[HardCell]
) -> list[HardCell]:
    done = set(trained)
    return [c for c in hard_final_cells(lr, steps, tiers) if c not in done]
