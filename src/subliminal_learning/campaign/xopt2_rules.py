"""Pre-registered learning-rate walk and anchor rules of `xopt2` (docs/PREREG_XOPT2.md §3).

Pure and torch-free. A *walk* finds, for one (pair, arm), the highest learning rate that
leaves the student non-degenerate and the lowest one above it that does not: it climbs a
1-2-5 ladder from a start rate until a rung is ineligible, then runs the one midpoint
between the last eligible rung and that rung. When the start rung itself is ineligible it
steps down the ladder instead (at most `max_stepdowns` rungs). Eligibility is
`xopt_rules.eligible` for every animal (treatment and control ΔCE under the student's cut).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import xopt_rules

COARSE_MANTISSAS: tuple[float, ...] = (1.0, 2.0, 5.0)
# Midpoint mantissa between adjacent coarse mantissas (5 -> 10 wraps to the next decade).
MID_MANTISSA: Mapping[tuple[float, float], float] = {
    (1.0, 2.0): 1.4,
    (2.0, 5.0): 3.0,
    (5.0, 10.0): 7.0,
}
MAX_STEPDOWNS = 2
WAIT, RUN, DONE = "wait", "run", "done"


def _rate(mantissa: float, exponent: int) -> float:
    """`mantissa`e`exponent` parsed from text, so equal rates compare equal."""
    return float(f"{mantissa:g}e{exponent}")


def _decompose(rate: float) -> tuple[float, int]:
    exponent = math.floor(math.log10(rate) + 1e-9)
    return round(rate / 10**exponent, 6), exponent


def ladder(lo: float, hi: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(coarse, mids) of the 1-2-5 series from `lo` to `hi`, both ascending.

    Raises:
        ValueError: `lo` or `hi` is not on the 1-2-5 series, or `lo >= hi`.
    """
    for r in (lo, hi):
        m, _ = _decompose(r)
        if m not in COARSE_MANTISSAS:
            raise ValueError(f"{r:g} is not on the 1-2-5 series")
    if lo >= hi:
        raise ValueError(f"empty ladder {lo:g}..{hi:g}")
    coarse: list[float] = []
    _, e = _decompose(lo)
    while True:
        for m in COARSE_MANTISSAS:
            r = _rate(m, e)
            if r > hi * (1 + 1e-9):
                return tuple(coarse), tuple(_mid(a, b) for a, b in _pairs(coarse))
            if r >= lo * (1 - 1e-9):
                coarse.append(r)
        e += 1


def _pairs(values: Sequence[float]) -> list[tuple[float, float]]:
    return list(zip(values[:-1], values[1:]))


def _mid(a: float, b: float) -> float:
    ma, ea = _decompose(a)
    mb, eb = _decompose(b)
    key = (ma, mb if eb == ea else mb * 10 ** (eb - ea))
    return _rate(MID_MANTISSA[key], ea)


@dataclass(frozen=True)
class Walk:
    """The rates one arm may run at in a walk, and where the walk starts.

    Attributes:
        coarse: 1-2-5 rungs, ascending; the walk climbs these.
        mids: One midpoint between each pair of adjacent coarse rungs.
        start: First walked rung (a member of `coarse`).
        floor: Fixed lower rungs scheduled with the start; they count as decided
            rates but the walk never steps from them.
    """

    coarse: tuple[float, ...]
    mids: tuple[float, ...]
    start: float
    floor: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.start not in self.coarse:
            raise ValueError(f"start {self.start:g} is not a coarse rung")
        if list(self.coarse) != sorted(self.coarse):
            raise ValueError("coarse rungs must ascend")
        if any(f >= self.start for f in self.floor):
            raise ValueError("floor rungs must lie below the start")

    @property
    def rates(self) -> tuple[float, ...]:
        """Every rate the walk may schedule, ascending."""
        return tuple(sorted({*self.coarse, *self.mids, *self.floor}))

    def initial(self) -> tuple[float, ...]:
        """Rates scheduled before any decision: the floor rungs and the start."""
        return (*self.floor, self.start)

    def next_coarse_above(self, rate: float) -> float | None:
        above = [c for c in self.coarse if c > rate]
        return above[0] if above else None


@dataclass(frozen=True)
class WalkStep:
    """The walk's next action: wait for running rates, run `rate`, or stop."""

    action: str
    rate: float | None = None
    reason: str = ""


def walk_next(
    walk: Walk, scheduled: Collection[float], decided: Mapping[float, bool]
) -> WalkStep:
    """Next action of one walk group.

    Args:
        walk: The arm's walk.
        scheduled: Every rate the group has submitted.
        decided: rate -> eligible (every animal), for each scheduled rate whose
            decision leaves have all finished.

    Returns:
        `WAIT` while a scheduled rate is undecided; `RUN` with the next rate; or `DONE`
        with reason `bracketed`, `no_collapse` (the top coarse rung is eligible) or
        `degenerate` (no eligible rate after the step-downs).
    """
    if any(r not in decided for r in scheduled):
        return WalkStep(WAIT, reason="rates running")
    eligible = [r for r in scheduled if decided[r]]
    if not eligible:
        below = [c for c in walk.coarse if c < min(scheduled) and c not in walk.floor]
        added = [r for r in scheduled if r < walk.start and r not in walk.floor]
        if below and len(added) < MAX_STEPDOWNS:
            return WalkStep(RUN, below[-1], "step-down")
        return WalkStep(DONE, reason="degenerate")
    high = max(eligible)
    above = [r for r in scheduled if r > high and not decided[r]]
    if not above:
        nxt = walk.next_coarse_above(high)
        if nxt is None:
            return WalkStep(DONE, reason="no_collapse")
        return WalkStep(RUN, nxt, "climb")
    low = min(above)
    between = [r for r in walk.rates if high < r < low and r not in scheduled]
    if between:
        return WalkStep(RUN, between[len(between) // 2], "bisect")
    return WalkStep(DONE, reason="bracketed")


def rate_eligible(
    outcome: xopt_rules.RateOutcome, animals: Sequence[str], ce_base: float
) -> bool:
    """Eligible for every animal (the walk's unit of decision)."""
    return all(xopt_rules.eligible(outcome, a, ce_base) for a in animals)


def _val(outcome: xopt_rules.RateOutcome, animal: str) -> float:
    v = [float(x) for x in outcome.val_delta.get(animal, ())]
    if not v or not all(math.isfinite(x) for x in v):
        return math.nan
    return sum(v) / len(v)


@dataclass(frozen=True)
class Anchor:
    """The rate(s) an arm carries from the `pythia-160m@143k` self-pair to other pairs.

    Attributes:
        rate: Selected rate (None when no rate is eligible).
        null: No eligible rate reaches mean validation Δ ≥ `xopt_rules.TRANSFER_MIN`
            for any animal.
        carry: Rates run on every other pair: the selected rate, plus the next coarse
            rung above it for a null arm.
        reason: `selected`, `null_largest_eligible` or `no_eligible_rate`.
    """

    rate: float | None
    null: bool
    carry: tuple[float, ...]
    reason: str


def select_anchor(
    walk: Walk,
    outcomes: Sequence[xopt_rules.RateOutcome],
    animals: Sequence[str],
    ce_base: float,
) -> Anchor:
    """The pre-registered carry rule (PREREG_XOPT2 §3).

    Among rates eligible for every animal: the largest mean (over animals) validation Δ,
    ties to the smaller rate. A null arm carries its largest eligible rate and the next
    coarse rung above it instead.
    """
    elig = [o for o in outcomes if rate_eligible(o, animals, ce_base)]
    if not elig:
        return Anchor(None, False, (), "no_eligible_rate")
    best = max(
        elig, key=lambda o: (sum(_val(o, a) for a in animals) / len(animals), -o.rate)
    )
    null = all(max(_val(o, a) for o in elig) < xopt_rules.TRANSFER_MIN for a in animals)
    if not null:
        return Anchor(best.rate, False, (best.rate,), "selected")
    top = max(o.rate for o in elig)
    up = walk.next_coarse_above(top)
    return Anchor(
        top, True, (top,) if up is None else (top, up), "null_largest_eligible"
    )


def decision_rates(
    walk: Walk,
    outcomes: Sequence[xopt_rules.RateOutcome],
    animals: Sequence[str],
    ce_base: float,
) -> tuple[float, ...]:
    """Rates that get the extra seed: the largest eligible rate and the selected rate."""
    elig = [o for o in outcomes if rate_eligible(o, animals, ce_base)]
    if not elig:
        return ()
    anchor = select_anchor(walk, outcomes, animals, ce_base)
    return tuple(sorted({max(o.rate for o in elig), anchor.rate}))
