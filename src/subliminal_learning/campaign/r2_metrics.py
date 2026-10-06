"""Deltas, retention, labels and verdicts of R2 "masked ≠ removed" (pure).

Rows are plain dicts, one per leaf: `source`, `pair`, `family`, `arm`, `kind`,
`teacher`, `tfs`, `gseed`, `quarantined` and `values` (readout -> float). A treatment
leaf's Delta is its value minus its matched controls (opdlim: the live control of the same
pair, family, arm and gseed, `PREREG_OPDLIM` §3; tmplopd: the mean of the live controls of
the same family and arm, `PREREG_TMPLOPD` §3). Quarantined leaves are excluded from means.
Cell intervals are two-level bootstraps (teacher ftseed, then gseed;
`traitfree_metrics.two_level_weights`), one weight matrix per (cell, arm) shared by every
readout so readout contrasts are paired.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

import numpy as np

from subliminal_learning.campaign import r2_plan as plan
from subliminal_learning.campaign.traitfree_metrics import (
    percentile_ci,
    stable_rng,
    two_level_weights,
    weighted_means,
)

LABEL_VOID = "void"
LABEL_SUPPRESSED = "suppressed"
LABEL_PARTIAL = "partial"
LABEL_RETAINED = "retained"

VERDICT_VOID = "void"
VERDICT_NOT_SUPPRESSED = "direct_not_suppressed"
VERDICT_MOVED = "moved"
VERDICT_LATENT = "latent"
VERDICT_MOVED_LATENT = "moved+latent"
VERDICT_REMOVED = "removed"
VERDICT_UNDETERMINED = "undetermined"

Cell = tuple[str, str, str]


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _fmean(values: Sequence[float]) -> float:
    vals = [v for v in values if _finite(v)]
    return mean(vals) if vals else math.nan


def _controls(rows: Sequence[Mapping]) -> dict[tuple, list[Mapping]]:
    out: dict[tuple, list[Mapping]] = {}
    for r in rows:
        if r["kind"] != plan.KIND_CTRL:
            continue
        key = (r["pair"], r["family"], r["arm"])
        if r["source"] == plan.SOURCE_OPDLIM:
            key += (int(r["gseed"]),)
        out.setdefault(key, []).append(r)
    return out


def leaf_deltas(rows: Sequence[Mapping], readouts: Sequence[str]) -> list[dict]:
    """One row per live treatment leaf: its Delta on every readout.

    A treatment whose matched controls are all quarantined is dropped (opdlim
    `control_quarantined`; tmplopd raises, since its controls are pooled).

    Raises:
        ValueError: A treatment leaf has no matched control at all.
    """
    controls = _controls(rows)
    out: list[dict] = []
    for r in rows:
        if r["kind"] != plan.KIND_TREAT or r["quarantined"]:
            continue
        key = (r["pair"], r["family"], r["arm"])
        if r["source"] == plan.SOURCE_OPDLIM:
            key += (int(r["gseed"]),)
        matched = controls.get(key)
        if not matched:
            raise ValueError(f"no matched control for {r['key']}")
        live = [c for c in matched if not c["quarantined"]]
        if not live:
            if r["source"] == plan.SOURCE_TMPLOPD:
                raise ValueError(f"every pooled control of {r['key']} is quarantined")
            continue
        out.append(
            {
                **{
                    k: r[k]
                    for k in ("key", "pair", "family", "arm", "teacher", "tfs", "gseed")
                },
                "delta": {
                    x: r["values"].get(x, math.nan)
                    - _fmean([c["values"].get(x, math.nan) for c in live])
                    for x in readouts
                },
            }
        )
    return out


def control_mean(rows: Sequence[Mapping], cell: Cell, arm: str, readout: str) -> float:
    """Mean raw value of the live controls of a cell's arm."""
    pair, family, _teacher = cell
    return _fmean(
        [
            r["values"].get(readout, math.nan)
            for r in rows
            if r["kind"] == plan.KIND_CTRL
            and not r["quarantined"]
            and (r["pair"], r["family"], r["arm"]) == (pair, family, arm)
        ]
    )


def treat_mean(rows: Sequence[Mapping], cell: Cell, arm: str, readout: str) -> float:
    """Mean raw value of the live treatment leaves of a cell's arm."""
    return _fmean(
        [
            r["values"].get(readout, math.nan)
            for r in rows
            if r["kind"] == plan.KIND_TREAT
            and not r["quarantined"]
            and (r["pair"], r["family"], r["teacher"]) == cell
            and r["arm"] == arm
        ]
    )


@dataclass(frozen=True)
class ArmStat:
    """Mean Delta of one arm of a cell with its bootstrap interval.

    Attributes:
        mean: Mean Delta over live leaves with a finite value.
        lo: CI lower bound.
        hi: CI upper bound.
        n: Leaves with a finite Delta.
        draws: Bootstrap replicate means.
    """

    mean: float
    lo: float
    hi: float
    n: int
    draws: np.ndarray


@dataclass(frozen=True)
class Retention:
    """Masked / twin ratio of mean Deltas, its interval and label."""

    ratio: float
    lo: float
    hi: float
    undefined_frac: float
    label: str
    draws: np.ndarray


class CellStats:
    """Bootstrap of one cell's masked and twin Deltas, shared by every readout."""

    def __init__(self, cell: Cell, deltas: Sequence[Mapping]) -> None:
        self.cell = cell
        self.leaves = {
            arm: [
                d
                for d in deltas
                if (d["pair"], d["family"], d["teacher"]) == cell and d["arm"] == arm
            ]
            for arm in (plan.ARM_MASKED, plan.ARM_TWIN)
        }
        rng = stable_rng(plan.BOOTSTRAP_SEED, *cell)
        self.weights = {
            arm: two_level_weights([f"t{d['tfs']}" for d in ds], plan.BOOTSTRAP_N, rng)
            for arm, ds in self.leaves.items()
        }

    def arm(self, arm: str, readout: str) -> ArmStat:
        values = [d["delta"].get(readout, math.nan) for d in self.leaves[arm]]
        n = sum(1 for v in values if _finite(v))
        if n == 0:
            nan = math.nan
            return ArmStat(nan, nan, nan, 0, np.full(plan.BOOTSTRAP_N, nan))
        draws = weighted_means(self.weights[arm], values)
        lo, hi = percentile_ci(draws, plan.CONFIDENCE)
        return ArmStat(_fmean(values), lo, hi, n, draws)

    def retention(self, readout: str) -> Retention:
        """Masked / twin ratio; `void` unless the twin shows the effect (CI low > 0)."""
        m, t = self.arm(plan.ARM_MASKED, readout), self.arm(plan.ARM_TWIN, readout)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = np.where(t.draws > 0, m.draws / t.draws, np.nan)
        undefined = float(np.mean(~np.isfinite(draws)))
        ratio = m.mean / t.mean if _finite(t.mean) and t.mean > 0 else math.nan
        if (
            not _finite(t.lo)
            or t.lo <= 0
            or not _finite(m.mean)
            or undefined > plan.MAX_UNDEFINED_RATIO_FRAC
        ):
            nan = math.nan
            return Retention(ratio, nan, nan, undefined, LABEL_VOID, draws)
        lo, hi = percentile_ci(draws, plan.CONFIDENCE)
        if hi < plan.SUPPRESSED_CI_HIGH:
            label = LABEL_SUPPRESSED
        elif lo >= plan.RETAINED_CI_LOW:
            label = LABEL_RETAINED
        else:
            label = LABEL_PARTIAL
        return Retention(ratio, lo, hi, undefined, label, draws)


def ratio_difference(a: Retention, b: Retention) -> tuple[float, float, float]:
    """``a - b`` with the CI of the paired replicate differences (NaN when undefined)."""
    if LABEL_VOID in (a.label, b.label):
        return a.ratio - b.ratio, math.nan, math.nan
    lo, hi = percentile_ci(a.draws - b.draws, plan.CONFIDENCE)
    return a.ratio - b.ratio, lo, hi


@dataclass(frozen=True)
class Verdict:
    """Cell verdict and the flags it is built from."""

    verdict: str
    direct_label: str
    moved: bool
    latent: bool
    removed: bool
    unspaced_fold: float


def cell_verdict(
    stats: CellStats,
    rows: Sequence[Mapping],
    *,
    direct: str = plan.READOUT_DIRECT,
    unspaced: str = plan.READOUT_UNSPACED,
    eas: str = plan.READOUT_EAS,
) -> Verdict:
    """The pre-registered R2 verdict of one cell.

    - moved: masked Delta(unspaced) CI low > 0, masked unspaced level >= MOVED_MIN_FOLD x
      its controls', and (twin unspaced effect void, or R_unspaced - R_direct CI low > 0);
    - latent: masked Delta(EAS) CI low > 0;
    - removed: direct suppressed, masked Delta(unspaced) and Delta(EAS) CI low <= 0
      (a missing readout leaves the cell undetermined).
    """
    r_direct = stats.retention(direct)
    r_unspaced = stats.retention(unspaced)
    m_unspaced = stats.arm(plan.ARM_MASKED, unspaced)
    m_eas = stats.arm(plan.ARM_MASKED, eas)
    ctrl = control_mean(rows, stats.cell, plan.ARM_MASKED, unspaced)
    level = treat_mean(rows, stats.cell, plan.ARM_MASKED, unspaced)
    fold = level / ctrl if _finite(ctrl) and ctrl > 0 else math.nan
    _diff, diff_lo, _hi = ratio_difference(r_unspaced, r_direct)
    moved = (
        _finite(m_unspaced.lo)
        and m_unspaced.lo > 0
        and _finite(fold)
        and fold >= plan.MOVED_MIN_FOLD
        and (r_unspaced.label == LABEL_VOID or (_finite(diff_lo) and diff_lo > 0))
    )
    latent = _finite(m_eas.lo) and m_eas.lo > 0
    removed = (
        r_direct.label == LABEL_SUPPRESSED
        and _finite(m_unspaced.lo)
        and m_unspaced.lo <= 0
        and _finite(m_eas.lo)
        and m_eas.lo <= 0
    )
    if r_direct.label == LABEL_VOID:
        verdict = VERDICT_VOID
    elif r_direct.label != LABEL_SUPPRESSED:
        verdict = VERDICT_NOT_SUPPRESSED
    elif moved and latent:
        verdict = VERDICT_MOVED_LATENT
    elif moved:
        verdict = VERDICT_MOVED
    elif latent:
        verdict = VERDICT_LATENT
    elif removed:
        verdict = VERDICT_REMOVED
    else:
        verdict = VERDICT_UNDETERMINED
    return Verdict(verdict, r_direct.label, moved, latent, removed, fold)


def global_claim(verdicts: Mapping[Cell, Verdict]) -> str:
    """R2-H over the primary (non-descriptive) cells whose direct channel is suppressed.

    ``supported`` iff at least one such cell exists and at least half are moved and / or
    latent; ``not_supported`` iff every such cell is removed; ``void`` when no primary
    cell is suppressed; ``inconclusive`` otherwise.
    """
    suppressed = [
        v
        for cell, v in verdicts.items()
        if cell not in plan.DESCRIPTIVE_CELLS and v.direct_label == LABEL_SUPPRESSED
    ]
    if not suppressed:
        return "void"
    hidden = sum(
        v.verdict in (VERDICT_MOVED, VERDICT_LATENT, VERDICT_MOVED_LATENT)
        for v in suppressed
    )
    if 2 * hidden >= len(suppressed):
        return "supported"
    if all(v.verdict == VERDICT_REMOVED for v in suppressed):
        return "not_supported"
    return "inconclusive"
