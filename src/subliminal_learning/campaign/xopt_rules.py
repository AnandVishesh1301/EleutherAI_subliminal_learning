"""Pre-registered decision rules of `xopt` (docs/PREREG_XOPT.md §3), pure and torch-free.

Eligibility, step-down, rate selection, transfer, retention and onset, shared by the hard
and soft drivers and results builders.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign.hlopt_claims import seed_bootstrap_ci

BOOT_SEED = 20261014
ONSET_ABS = 0.02
RETENTION_MIN_SELF = 0.05
TRANSFER_MIN = 0.02


def cut(ce_base: float) -> float:
    """Degenerate-ΔCE cut of a student base: H_uni − CE_base."""
    if not math.isfinite(ce_base):
        raise ValueError(f"non-finite CE_base {ce_base}")
    return plan.H_UNI - ce_base


def _mean(values: Sequence[float]) -> float:
    v = [float(x) for x in values]
    if not v or not all(math.isfinite(x) for x in v):
        return math.nan
    return sum(v) / len(v)


@dataclass(frozen=True)
class RateOutcome:
    """Finished leaves of one (cell, arm, rate): per-animal treatment and control rows.

    Attributes:
        rate: Learning rate.
        treat_dce: animal -> treatment pp1 ΔCE per seed / leaf.
        control_dce: animal -> control pp1 ΔCE per seed / leaf (hard: the shared
            control repeated per animal).
        val_delta: animal -> validation Δ per seed / leaf.
    """

    rate: float
    treat_dce: Mapping[str, Sequence[float]]
    control_dce: Mapping[str, Sequence[float]]
    val_delta: Mapping[str, Sequence[float]]


def eligible(outcome: RateOutcome, animal: str, ce_base: float) -> bool:
    """All leaves finite, mean treatment and mean control ΔCE under the student's cut."""
    c = cut(ce_base)
    t = _mean(outcome.treat_dce.get(animal, ()))
    k = _mean(outcome.control_dce.get(animal, ()))
    v = _mean(outcome.val_delta.get(animal, ()))
    return all(math.isfinite(x) for x in (t, k, v)) and t < c and k < c


def stepdown_rate(
    arm: plan.Arm,
    outcomes: Sequence[RateOutcome],
    animals: Sequence[str],
    ce_base: float,
    initial_rates: Sequence[float],
) -> float | None:
    """Next rate to run for one (cell, arm), or None.

    `outcomes` holds every finished scheduled rate of the arm in the cell (the caller
    only asks once all of them are finished). Returns the next ladder rung below the
    lowest scheduled rate when some animal has no eligible rate, unless
    `plan.MAX_STEPDOWNS` rungs below the lowest initial rate were already added.
    """
    if not outcomes:
        return None
    if all(any(eligible(o, a, ce_base) for o in outcomes) for a in animals):
        return None
    lowest = min(o.rate for o in outcomes)
    added = [o.rate for o in outcomes if o.rate < min(initial_rates)]
    if len(added) >= plan.MAX_STEPDOWNS:
        return None
    lower = arm.lower_rungs(lowest)
    return lower[0] if lower else None


def select_rate(
    outcomes: Sequence[RateOutcome], animal: str, ce_base: float
) -> float | None:
    """Eligible rate with the largest mean validation Δ; ties to the smaller rate."""
    best: tuple[float, float] | None = None
    for o in outcomes:
        if not eligible(o, animal, ce_base):
            continue
        key = (_mean(o.val_delta[animal]), -o.rate)
        if best is None or key > best:
            best = key
    return None if best is None else -best[1]


def transfers_hard(test_deltas: Sequence[float]) -> bool:
    """Every seed finite, mean Δ_test ≥ 0.02 and seed-bootstrap CI lower > 0."""
    v = [float(x) for x in test_deltas]
    if not v or not all(math.isfinite(x) for x in v):
        return False
    mean, lo, _ = seed_bootstrap_ci(v, seed=BOOT_SEED)
    return bool(
        math.isfinite(mean) and math.isfinite(lo) and mean >= TRANSFER_MIN and lo > 0
    )


def transfers_soft(test_deltas: Sequence[float]) -> bool:
    """Two or more leaves: mean ≥ 0.02 and every leaf > 0; one leaf: Δ ≥ 0.02."""
    v = [float(x) for x in test_deltas]
    if not v or not all(math.isfinite(x) for x in v):
        return False
    if len(v) == 1:
        return v[0] >= TRANSFER_MIN
    return sum(v) / len(v) >= TRANSFER_MIN and all(x > 0 for x in v)


def retention(delta_cross: float, delta_self: float) -> float:
    """Δ(cross into B) / Δ(B self); NaN when the self Δ is below the floor."""
    if not (math.isfinite(delta_cross) and math.isfinite(delta_self)):
        return math.nan
    if delta_self < RETENTION_MIN_SELF:
        return math.nan
    return delta_cross / delta_self


def onset(steps: Sequence[int], transfer_at: Mapping[int, bool]) -> int | None:
    """Smallest step that transfers and stays transferring at every later step."""
    ordered = sorted(steps)
    for i, s in enumerate(ordered):
        if all(transfer_at.get(t, False) for t in ordered[i:]):
            return s
    return None
