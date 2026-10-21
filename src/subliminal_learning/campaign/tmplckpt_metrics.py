"""Pure results math of `tmplckpt` (prereg `docs/PREREG_TMPLCKPT.md` §3–§5).

Leaf frames come from `pipelines/tmplckpt_results_pipeline.py`; nothing here reads disk.
Intervals are two-stage bootstraps within each checkpoint (teachers, then students within
teacher) with the matched controls resampled in the same draw and Δ recomputed per draw.
The grid factorization (`fit_rank1`, `fit_product_proximity`, the condition-number guard)
is ported from Angela's
`origin/angela/distillation:src/subliminal_learning/analysis/checkpoint_grid_factorization.py`
(`fit_rank1`, `_fit_prox_fixed_length`, `_prox_cond`, `fit_product_proximity`,
``NEARNESS_LENGTHS``, ``PROXIMITY_MAX_COND``), the model of paper app:separability:
``M[t, s] ≈ a_t b_s + c · exp(-|t - s| / L)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.campaign import tmplckpt_plan as plan

N_BOOT = 10_000
BOOT_SEED = 20_261_021
CONFIDENCE = 0.95
MAX_NONFINITE_SHARE = 0.05
EARLY_STEPS: tuple[int, ...] = tuple(s for s in plan.STEPS_HARD15 if s <= 90_000)
LATE_STEPS: tuple[int, ...] = tuple(s for s in plan.STEPS_HARD15 if s >= 115_000)
GRID_EARLY: tuple[int, ...] = tuple(s for s in plan.STEPS_GRID13 if s <= 95_000)
GRID_LATE: tuple[int, ...] = LATE_STEPS
SENDER_RECEIVER_STEP = 85_000
NEARNESS_LENGTHS: tuple[float, ...] = (2.5, 5.0, 10.0, 20.0, 40.0)
PROXIMITY_MAX_COND = 1e3
# Thresholds of prereg §5 (marked † there).
REPRO_P_TOL = 0.02
REPRO_PPL_REL_TOL = 0.05
CONCORDANCE_SPEARMAN_MIN = 0.8
CHECKS_MIN_STEPS = 12
PLACEBO_VALID_TOL = 0.02
R_SUPPORTED_MIN = 0.5
R_NOT_SUPPORTED_MAX = 0.3
R_HARD_MAX = 0.3
CEILING_P = 0.8
F_SOFT_SUPPORTED_MIN = 0.7
F_HARD_SUPPORTED_MAX = 0.4
F_SOFT_NOT_SUPPORTED_MAX = 0.4
ONSET_LOW = 0.02
TEACHER_HIGH_P = 0.9
TEACHER_LOW_P = 0.5
HARNESS_TOL = 1e-6

SUPPORTED, NOT_SUPPORTED, UNDETERMINED = "supported", "not_supported", "undetermined"
HARD_RISE_ABSENT, CEILING_LIMITED = "hard_rise_absent", "ceiling_limited"
EARLIER, SAME, LATER = "earlier", "same", "later"
STUDENT_SIDE, TEACHER_LIMITED = "student_side_rise", "teacher_limited"


# ---- exclusion and measurability (prereg §4) ------------------------------------------


@dataclass(frozen=True)
class Rules:
    """One rule set: the primary analysis or a sensitivity rerun.

    Attributes:
        name: Label of the rerun (``primary``, ``ppl10``, ``drop05``).
        ppl_max: Largest kept perplexity ratio.
        drop_max: Largest kept screened-out share (``inf`` = never excludes).
    """

    name: str
    ppl_max: float
    drop_max: float = math.inf


PRIMARY = Rules("primary", plan.PPL_RATIO_MAX)
RULE_SETS: tuple[Rules, ...] = (
    PRIMARY,
    Rules("ppl10", plan.PPL_RATIO_SENSITIVITY),
    Rules("drop05", plan.PPL_RATIO_MAX, plan.DROP_FRAC_SENSITIVITY),
)


def kept(
    perplexity_ratio: float, drop_frac: float = math.nan, rules: Rules = PRIMARY
) -> bool:
    """Rule 1 (and rule 2 under a drop sensitivity): measured ratio ≤ `rules.ppl_max`."""
    if not (math.isfinite(perplexity_ratio) and perplexity_ratio <= rules.ppl_max):
        return False
    return not (math.isfinite(drop_frac) and drop_frac > rules.drop_max)


def teacher_usable(
    n_kept: int, n_done: int, n_planned: int, min_kept: int = plan.TEACHER_MIN_KEPT
) -> bool | None:
    """Rule 3; ``None`` while students are pending (one planned student: that one)."""
    if n_planned <= 0 or n_done < n_planned:
        return None
    return n_kept >= (1 if n_planned == 1 else min_kept)


def cell_measurable(
    n_planned: int,
    n_done: int,
    n_excluded: int,
    n_controls_kept: int,
    max_frac: float = plan.STEP_MAX_EXCLUDED_FRAC,
) -> bool | None:
    """Rule 4: ≤ `max_frac` of the planned students excluded and a kept matched control.

    ``None`` while students are pending.
    """
    if n_planned <= 0 or n_done < n_planned:
        return None
    if n_controls_kept <= 0:
        return False
    return n_excluded / n_planned <= max_frac + 1e-12


def common_steps(
    steps_by_group: Mapping[str, Iterable[int]], wanted: Sequence[int]
) -> tuple[list[int], dict[str, list[int]]]:
    """Steps of `wanted` measurable in every group, and each group's dropped steps."""
    sets = {g: {int(s) for s in v} for g, v in steps_by_group.items()}
    common = [s for s in wanted if all(s in v for v in sets.values())]
    dropped = {g: [s for s in wanted if s not in common] for g in sets}
    return common, dropped


def finite_mean(values: Iterable[float]) -> float:
    """Mean of the finite values; NaN when there are none."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else math.nan


# ---- control pairing -----------------------------------------------------------------


def placebo_kind(teacher_kind: str) -> str:
    """Placebo teacher kind matching a treatment teacher kind."""
    return {plan.STMT: plan.STMT_PL, plan.SDF: plan.SDF_PL}[teacher_kind]


def base_kind(teacher_step: int, student_step: int) -> str:
    """Base-control kind: student base on the diagonal, teacher base off it."""
    return plan.BASE_SB if teacher_step == student_step else plan.BASE_TB


def control_match_keys(row: Mapping) -> dict[str, tuple]:
    """Keys of the matched control groups of one treatment row.

    A key is ``(channel, config, animal, teacher_step, student_step, control_kind,
    control_teacher_kind)`` and equals `control_key` of its control rows; ``config`` is
    the channel recipe tag, so a 1D variant pairs with its own variant's controls.
    """
    t, s = int(row["teacher_step"]), int(row["student_step"])
    common = (row["channel"], row["config"], row["animal"])
    out = {"base": (*common, t, s, base_kind(t, s), None)}
    if t == s:
        out["placebo"] = (
            *common,
            t,
            s,
            plan.PLACEBO,
            placebo_kind(row["teacher_kind"]),
        )
    return out


def control_key(row: Mapping) -> tuple:
    """Group key of one control row (see `control_match_keys`)."""
    kind = row["kind"]
    tk = row.get("teacher_kind") if kind == plan.PLACEBO else None
    return (
        row["channel"],
        row["config"],
        row["animal"],
        int(row["teacher_step"]),
        int(row["student_step"]),
        kind,
        tk if isinstance(tk, str) else None,
    )


def control_values(controls: pd.DataFrame, value: str) -> dict[tuple, list[float]]:
    """Finite `value` of the kept controls per `control_key` group."""
    out: dict[tuple, list[float]] = {}
    for rec in controls.to_dict("records"):
        v = rec.get(value)
        if not rec.get("kept") or v is None or not math.isfinite(float(v)):
            continue
        out.setdefault(control_key(rec), []).append(float(v))
    return out


# ---- two-stage bootstrap ---------------------------------------------------------------


@dataclass(frozen=True)
class CellData:
    """One checkpoint cell of one group: treatment values by teacher, matched controls.

    Attributes:
        treat: Per teacher, its kept students' values.
        ctrl: The kept matched controls' values.
    """

    treat: tuple[tuple[float, ...], ...]
    ctrl: tuple[float, ...]

    @property
    def observed(self) -> float:
        """Mean over the kept students minus mean over the kept controls."""
        vals = [v for t in self.treat for v in t]
        if not vals or not self.ctrl:
            return math.nan
        return float(np.mean(vals) - np.mean(self.ctrl))


def draw_deltas(cell: CellData, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """``[n_boot]`` Δ draws: teachers, then students within teacher; controls iid.

    Every draw resamples the teachers with replacement, each drawn teacher's students
    with replacement, and the matched controls with replacement, then recomputes Δ.
    """
    teachers = [np.asarray(t, dtype=float) for t in cell.treat if len(t)]
    if not teachers or not cell.ctrl:
        return np.full(n_boot, np.nan)
    n_t = len(teachers)
    k = np.array([t.size for t in teachers])
    kmax = int(k.max())
    vals = np.full((n_t, kmax), np.nan)
    for j, t in enumerate(teachers):
        vals[j, : t.size] = t
    pick_t = rng.integers(0, n_t, size=(n_boot, n_t))
    k_drawn = k[pick_t]
    pick_s = np.floor(rng.random((n_boot, n_t, kmax)) * k_drawn[..., None]).astype(int)
    mask = np.arange(kmax)[None, None, :] < k_drawn[..., None]
    drawn = vals[pick_t[..., None], pick_s]
    treat = np.where(mask, drawn, 0.0).sum(axis=(1, 2)) / mask.sum(axis=(1, 2))
    ctrl = np.asarray(cell.ctrl, dtype=float)
    pick_c = rng.integers(0, ctrl.size, size=(n_boot, ctrl.size))
    return treat - ctrl[pick_c].mean(axis=1)


@dataclass(frozen=True)
class Estimate:
    """A statistic with its percentile interval and non-finite draw share.

    Attributes:
        estimate: Value on the observed data.
        ci_low: Lower percentile bound (NaN without finite draws).
        ci_high: Upper percentile bound.
        n_valid: Finite draws.
        n_boot: Draws.
        nonfinite_share: ``1 - n_valid / n_boot``.
    """

    estimate: float
    ci_low: float
    ci_high: float
    n_valid: int
    n_boot: int
    nonfinite_share: float

    def as_dict(self) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_valid": self.n_valid,
            "n_boot": self.n_boot,
            "nonfinite_share": self.nonfinite_share,
        }


def summarize(
    observed: float, draws: np.ndarray, confidence: float = CONFIDENCE
) -> Estimate:
    """Percentile interval over the finite draws."""
    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    n = int(draws.size)
    share = 1 - finite.size / n if n else 1.0
    if not finite.size:
        return Estimate(float(observed), math.nan, math.nan, 0, n, share)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(finite, [alpha, 1 - alpha])
    return Estimate(
        float(observed), float(low), float(high), int(finite.size), n, share
    )


def cell_interval(
    cell: CellData, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> Estimate:
    """Two-stage interval of one cell's Δ."""
    return summarize(
        cell.observed, draw_deltas(cell, np.random.default_rng(seed), n_boot)
    )


def late_rise_ratio(
    step_means: Mapping[int, float],
    early: Sequence[int] = EARLY_STEPS,
    late: Sequence[int] = LATE_STEPS,
) -> float:
    """``mean(early step means) / mean(late step means)`` over the steps present.

    NaN without an early or a late step, or when the late mean is not positive.
    """
    e = [
        step_means[s] for s in early if s in step_means and math.isfinite(step_means[s])
    ]
    lt = [
        step_means[s] for s in late if s in step_means and math.isfinite(step_means[s])
    ]
    if not e or not lt:
        return math.nan
    den = float(np.mean(lt))
    return float(np.mean(e)) / den if den > 0 else math.nan


def _ratio_rows(
    d: np.ndarray, steps: Sequence[int], early, late
) -> tuple[np.ndarray, np.ndarray]:
    """Per draw: late-rise ratio and late-minus-early mean, from ``[n_boot, n_steps]``."""
    e = [i for i, s in enumerate(steps) if s in early]
    lt = [i for i, s in enumerate(steps) if s in late]
    if not e or not lt:
        nan = np.full(d.shape[0], np.nan)
        return nan, nan
    em, lm = d[:, e].mean(axis=1), d[:, lt].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(lm > 0, em / lm, np.nan)
    return r, lm - em


@dataclass(frozen=True)
class RatioComparison:
    """Late-rise ratios of groups ``a`` and ``b`` over their common measurable steps.

    Attributes:
        steps: Common steps used.
        dropped: Per group, wanted steps left out.
        r_a: Ratio of ``a``.
        r_b: Ratio of ``b``.
        diff: ``r_a - r_b``.
        rise_a: Late-minus-early mean Δ of ``a``.
        rise_b: Late-minus-early mean Δ of ``b``.
    """

    steps: tuple[int, ...]
    dropped: Mapping[str, list[int]]
    r_a: Estimate
    r_b: Estimate
    diff: Estimate
    rise_a: Estimate
    rise_b: Estimate

    def as_dict(self, a: str, b: str) -> dict:
        return {
            "steps_used": list(self.steps),
            "steps_dropped": dict(self.dropped),
            f"R_{a}": self.r_a.as_dict(),
            f"R_{b}": self.r_b.as_dict(),
            f"R_{a}_minus_R_{b}": self.diff.as_dict(),
            f"late_minus_early_{a}": self.rise_a.as_dict(),
            f"late_minus_early_{b}": self.rise_b.as_dict(),
        }


def compare_ratios(
    cells: Mapping[str, Mapping[int, CellData]],
    a: str,
    b: str,
    *,
    early: Sequence[int] = EARLY_STEPS,
    late: Sequence[int] = LATE_STEPS,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> RatioComparison | None:
    """Two-stage bootstrap of the late-rise ratios of `a` and `b` and their difference.

    Args:
        cells: Group → step → measurable cell; steps outside ``early ∪ late`` are ignored.
        a: First group.
        b: Second group.
        early: Early steps.
        late: Late steps.
        n_boot: Draws.
        seed: RNG seed.

    Returns:
        The comparison over the steps measurable in both groups, or ``None`` when either
        group lacks cells.
    """
    wanted = [s for s in sorted({*early, *late})]
    if a not in cells or b not in cells:
        return None
    steps, dropped = common_steps({g: cells[g].keys() for g in (a, b)}, wanted)
    if not steps:
        return None
    rng = np.random.default_rng(seed)
    obs, draws = {}, {}
    for g in (a, b):
        obs[g] = np.array([[cells[g][s].observed for s in steps]])
        draws[g] = np.column_stack(
            [draw_deltas(cells[g][s], rng, n_boot) for s in steps]
        )
    ra, rise_a = _ratio_rows(draws[a], steps, early, late)
    rb, rise_b = _ratio_rows(draws[b], steps, early, late)
    ra0, rise_a0 = _ratio_rows(obs[a], steps, early, late)
    rb0, rise_b0 = _ratio_rows(obs[b], steps, early, late)
    return RatioComparison(
        tuple(steps),
        dropped,
        summarize(ra0[0], ra),
        summarize(rb0[0], rb),
        summarize(ra0[0] - rb0[0], ra - rb),
        summarize(rise_a0[0], rise_a),
        summarize(rise_b0[0], rise_b),
    )


def difference(
    x: CellData, y: CellData, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> Estimate:
    """Two-stage interval of ``Δ(x) - Δ(y)`` (independent cells)."""
    rng = np.random.default_rng(seed)
    return summarize(
        x.observed - y.observed,
        draw_deltas(x, rng, n_boot) - draw_deltas(y, rng, n_boot),
    )


# ---- decision rules ----------------------------------------------------------------------


def rule_verdict(r_soft: float, diff_ci_low: float) -> str:
    """Core K1 rule on the soft ratio and the lower bound of ``R_soft - R_hard``."""
    if not math.isfinite(r_soft):
        return UNDETERMINED
    if r_soft >= R_SUPPORTED_MIN and math.isfinite(diff_ci_low) and diff_ci_low > 0:
        return SUPPORTED
    if r_soft <= R_NOT_SUPPORTED_MAX:
        return NOT_SUPPORTED
    return UNDETERMINED


def ceiling_limited(step_p: Mapping[int, float], p_max: float = CEILING_P) -> bool:
    """Soft mean test p above `p_max` at at least half of the checkpoints."""
    vals = [v for v in step_p.values() if math.isfinite(v)]
    return bool(vals) and sum(v > p_max for v in vals) >= len(vals) / 2


def k1_decide(
    cmp: RatioComparison | None,
    *,
    k0b_concordant: bool,
    soft_step_p: Mapping[int, float],
) -> tuple[str, list[str]]:
    """K1 verdict and its reasons (`cmp` has ``a`` = soft, ``b`` = hard).

    Order: missing data or > 5 % non-finite draws → undetermined; hard preconditions
    (R_hard ≤ 0.3, late-minus-early hard interval above 0, K0b concordant) →
    ``hard_rise_absent``; ceiling → ``ceiling_limited``; else `rule_verdict`.
    """
    if cmp is None:
        return UNDETERMINED, ["no common measurable checkpoints"]
    reasons = []
    worst = max(
        cmp.r_a.nonfinite_share, cmp.diff.nonfinite_share, cmp.r_b.nonfinite_share
    )
    if worst > MAX_NONFINITE_SHARE:
        return UNDETERMINED, [f"nonfinite draw share {worst:.3f}"]
    if not (math.isfinite(cmp.r_b.estimate) and cmp.r_b.estimate <= R_HARD_MAX):
        reasons.append("R_hard above 0.3")
    if not (math.isfinite(cmp.rise_b.ci_low) and cmp.rise_b.ci_low > 0):
        reasons.append("hard late-minus-early interval not above 0")
    if not k0b_concordant:
        reasons.append("K0b not concordant")
    if reasons:
        return HARD_RISE_ABSENT, reasons
    if ceiling_limited({s: soft_step_p[s] for s in cmp.steps if s in soft_step_p}):
        return CEILING_LIMITED, ["soft mean test p > 0.8 at >= half the checkpoints"]
    return rule_verdict(cmp.r_a.estimate, cmp.diff.ci_low), []


def iso_step(traj: Mapping[int, float], level: float) -> int | None:
    """K2: first probe step whose 143k soft trajectory Δ reaches `level`."""
    if not math.isfinite(level):
        return None
    for step in sorted(traj):
        if math.isfinite(traj[step]) and traj[step] >= level:
            return int(step)
    return None


def k0b_decide(rho: float, her_r: float, our_r: Estimate | None) -> bool:
    """K0b: Spearman ≥ 0.8 and her late-rise ratio inside our R_hard interval."""
    if our_r is None or not all(math.isfinite(v) for v in (rho, her_r, our_r.ci_low)):
        return False
    return rho >= CONCORDANCE_SPEARMAN_MIN and our_r.ci_low <= her_r <= our_r.ci_high


def placebo_valid(
    p_placebo: float, p_base: float, tol: float = PLACEBO_VALID_TOL
) -> bool | None:
    """K0c: placebo and base controls within `tol` of each other (``None`` if missing)."""
    if not (math.isfinite(p_placebo) and math.isfinite(p_base)):
        return None
    return abs(p_placebo - p_base) <= tol


def reproduced(
    pairs: Sequence[tuple[float, float, float, float]],
    p_tol: float = REPRO_P_TOL,
    ppl_rel_tol: float = REPRO_PPL_REL_TOL,
    min_steps: int = CHECKS_MIN_STEPS,
) -> tuple[bool, int]:
    """K0a over ``(p_repro, p_angela, ppl_repro, ppl_angela)`` per step."""
    n_ok = 0
    for p_r, p_a, q_r, q_a in pairs:
        if not all(math.isfinite(v) for v in (p_r, p_a, q_r, q_a)) or q_a <= 0:
            continue
        n_ok += abs(p_r - p_a) <= p_tol and abs(q_r / q_a - 1) <= ppl_rel_tol
    return n_ok >= min_steps, n_ok


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation of the finite pairs; NaN under three or if constant."""
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y, strict=True)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return math.nan
    xs, ys = zip(*pairs, strict=True)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return math.nan
    return float(pd.Series(xs).corr(pd.Series(ys), method="spearman"))


def onset(
    lows: Mapping[int, float], order: Sequence[int], low_min: float = ONSET_LOW
) -> int | None:
    """K7: first step of `order` whose lower bound and the next step's exceed `low_min`."""
    for i, s in enumerate(order[:-1]):
        nxt = order[i + 1]
        a, b = lows.get(s, math.nan), lows.get(nxt, math.nan)
        if math.isfinite(a) and math.isfinite(b) and a > low_min and b > low_min:
            return int(s)
    return None


def onset_order(soft: int | None, hard: int | None, order: Sequence[int]) -> str:
    """K7a: soft onset earlier (≥ 2 positions), same (equal or adjacent), or later."""
    if soft is None or hard is None:
        return UNDETERMINED
    gap = list(order).index(hard) - list(order).index(soft)
    if gap >= 2:
        return EARLIER
    if abs(gap) <= 1:
        return SAME
    return LATER


def teacher_side(rise: Estimate | None, tp_2k: float, tp_4k: float) -> str:
    """K7b on the soft dog 2k→4k rise and the dog statement teachers' own test p."""
    if rise is not None and math.isfinite(rise.ci_low) and rise.ci_low > 0:
        if tp_2k >= TEACHER_HIGH_P and tp_4k >= TEACHER_HIGH_P:
            return STUDENT_SIDE
    if math.isfinite(tp_2k) and tp_2k < TEACHER_LOW_P:
        return TEACHER_LIMITED
    return UNDETERMINED


def same_ordering(main: Mapping[int, float], other: Mapping[int, float]) -> bool | None:
    """K8: whether `other` ranks the shared steps as `main` does (``None`` if < 2)."""
    steps = [
        s
        for s in main
        if s in other and math.isfinite(main[s]) and math.isfinite(other[s])
    ]
    if len(steps) < 2:
        return None
    return sorted(steps, key=lambda s: main[s]) == sorted(steps, key=lambda s: other[s])


# ---- grid factorization (K3; ported from Angela's checkpoint_grid_factorization) ------


def fit_rank1(
    M: np.ndarray, mask: np.ndarray | None = None, iters: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted ALS for ``M[t, s] ≈ a[t] b[s]`` (mask = weights), gauged ``max b = 1``."""
    w = np.ones_like(M) if mask is None else np.asarray(mask, dtype=float)
    Mz = np.where(w > 0, M, 0.0)
    a = (w * Mz).sum(1) / np.maximum(w.sum(1), 1e-300)
    b = np.ones(M.shape[1])
    for _ in range(iters):
        previous = a
        b = (w * Mz).T @ a / np.maximum(w.T @ a**2, 1e-300)
        a = (w * Mz) @ b / np.maximum(w @ b**2, 1e-300)
        if np.abs(a - previous).max() <= 1e-13 * max(1.0, np.abs(a).max()):
            break
    c = b.max()
    return a * c, b / c


@dataclass(frozen=True)
class ProximityFit:
    """One ``M[t, s] ≈ a[t] b[s] + c exp(-|t - s| / L)`` fit.

    Attributes:
        a: Teacher (sender) factor.
        b: Student (receiver) factor, ``max b = 1`` when positive.
        c: Proximity bonus in probability units.
        L: Selected nearness length, thousands of steps.
        P: Prediction for every cell.
        r2: In-sample R² on the fitted cells.
        cond: Condition number of the fitted cells' Jacobian in ``(a, b, c)``.
    """

    a: np.ndarray
    b: np.ndarray
    c: float
    L: float
    P: np.ndarray
    r2: float
    cond: float

    @property
    def trusted(self) -> bool:
        """Angela's guard: the fitted cells pin the fit (``cond ≤ PROXIMITY_MAX_COND``)."""
        return math.isfinite(self.cond) and self.cond <= PROXIMITY_MAX_COND


def _distance_k(steps: Sequence[int]) -> np.ndarray:
    t = np.asarray(steps, dtype=float) / 1000.0
    return np.abs(np.subtract.outer(t, t))


def _fit_prox_fixed_length(
    M: np.ndarray, E: np.ndarray, W: np.ndarray, max_sweeps: int = 5_000
) -> tuple[np.ndarray, np.ndarray, float]:
    Mz = np.where(W > 0, M, 0.0)
    a, b = fit_rank1(Mz, W)
    c = 0.0
    ew = (W * E**2).sum()
    p_prev = np.outer(a, b)
    for _ in range(max_sweeps):
        R = W * (Mz - c * E)
        b = (R.T @ a) / np.maximum(W.T @ a**2, 1e-300)
        a = (R @ b) / np.maximum(W @ b**2, 1e-300)
        c = float((W * (Mz - np.outer(a, b)) * E).sum() / ew)
        P = np.outer(a, b) + c * E
        if np.abs(P - p_prev).max() <= 1e-13:
            break
        p_prev = P
    return a, b, c


def _prox_cond(a: np.ndarray, b: np.ndarray, E: np.ndarray, W: np.ndarray) -> float:
    n = len(a)
    r, s = np.nonzero(W > 0)
    J = np.zeros((len(r), 2 * n + 1))
    J[np.arange(len(r)), r] = b[s]
    J[np.arange(len(r)), n + s] = a[r]
    J[:, -1] = E[r, s]
    J = np.delete(J, n + int(np.argmax(b)), axis=1)
    if not np.isfinite(J).all():
        return math.inf
    sv = np.linalg.svd(J, compute_uv=False)
    return math.inf if sv[-1] == 0 else float(sv[0] / sv[-1])


def fit_product_proximity(
    M: np.ndarray,
    steps: Sequence[int],
    weights: np.ndarray | None = None,
    lengths: Sequence[float] = NEARNESS_LENGTHS,
) -> ProximityFit:
    """Fit ``a b^T + c E_L`` on the weighted finite cells, choosing ``L`` by squared error.

    Args:
        M: ``(n, n)`` cell means, teacher steps by student steps.
        steps: Checkpoint steps indexing `M`.
        weights: Cell weights (multiplicities in a cell bootstrap); default every finite
            cell once.
        lengths: Candidate nearness lengths in thousands of steps.

    Returns:
        The best fit.
    """
    finite = np.isfinite(M)
    W = finite.astype(float) if weights is None else np.where(finite, weights, 0.0)
    D = _distance_k(steps)
    best: tuple[float, ProximityFit] | None = None
    for length in lengths:
        E = np.exp(-D / length)
        a, b, c = _fit_prox_fixed_length(M, E, W)
        if b.max() > 0:
            scale = b.max()
            a, b = a * scale, b / scale
        P = np.outer(a, b) + c * E
        on = W > 0
        sse = float((W * np.where(on, M - P, 0.0) ** 2).sum())
        obs = M[on]
        wts = W[on]
        mean = float((wts * obs).sum() / wts.sum())
        sst = float((wts * (obs - mean) ** 2).sum())
        r2 = 1 - sse / sst if sst > 0 else math.nan
        if best is None or sse < best[0]:
            fit = ProximityFit(a, b, c, float(length), P, r2, _prox_cond(a, b, E, W))
            best = (sse, fit)
    assert best is not None
    return best[1]


def factor_ratio(
    factor: np.ndarray, steps: Sequence[int], early: Sequence[int], late: Sequence[int]
) -> float:
    """``mean(factor over early) / mean(factor over late)``; NaN if the late mean ≤ 0."""
    idx = {int(s): i for i, s in enumerate(steps)}
    e = [factor[idx[s]] for s in early if s in idx]
    lt = [factor[idx[s]] for s in late if s in idx]
    if not e or not lt:
        return math.nan
    den = float(np.mean(lt))
    return float(np.mean(e)) / den if den > 0 else math.nan


def receiver_flatness(fit: ProximityFit, steps: Sequence[int]) -> float:
    """K3's F (NaN when the fit is not trusted)."""
    if not fit.trusted:
        return math.nan
    return factor_ratio(fit.b, steps, GRID_EARLY, GRID_LATE)


def flatness_interval(
    M: np.ndarray, steps: Sequence[int], n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> Estimate:
    """F with a bootstrap over the finite cells (multinomial cell weights per draw)."""
    observed = receiver_flatness(fit_product_proximity(M, steps), steps)
    cells = np.flatnonzero(np.isfinite(M).ravel())
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        w = np.bincount(rng.choice(cells, cells.size), minlength=M.size).reshape(
            M.shape
        )
        if (w.sum(0) == 0).any() or (w.sum(1) == 0).any():
            continue
        draws[i] = receiver_flatness(
            fit_product_proximity(M, steps, w.astype(float)), steps
        )
    return summarize(observed, draws)


def k3_verdict(f_soft: Estimate | None, f_hard: Estimate | None) -> str:
    """K3 rule on the soft and hard receiver flatness (> 5 % non-finite → undetermined)."""
    if f_soft is None or not math.isfinite(f_soft.estimate):
        return UNDETERMINED
    shares = [f.nonfinite_share for f in (f_soft, f_hard) if f is not None]
    if any(s > MAX_NONFINITE_SHARE for s in shares):
        return UNDETERMINED
    hard_ok = f_hard is not None and math.isfinite(f_hard.estimate)
    if (
        f_soft.estimate >= F_SOFT_SUPPORTED_MIN
        and hard_ok
        and f_hard.estimate <= F_HARD_SUPPORTED_MAX
    ):
        return SUPPORTED
    if f_soft.estimate <= F_SOFT_NOT_SUPPORTED_MAX:
        return NOT_SUPPORTED
    return UNDETERMINED


def row_minus_column(M: np.ndarray, steps: Sequence[int], step: int) -> float:
    """Off-diagonal row mean minus column mean of `step` (sender minus receiver)."""
    i = [int(s) for s in steps].index(int(step))
    off = np.ones(M.shape[0], dtype=bool)
    off[i] = False
    return finite_mean(M[i, off]) - finite_mean(M[off, i])


def grid_matrix(
    cells: Iterable[tuple[int, int, float]], steps: Sequence[int]
) -> np.ndarray:
    """``(teacher_step, student_step, value)`` → ``(n, n)`` matrix (NaN where missing)."""
    idx = {int(s): i for i, s in enumerate(steps)}
    M = np.full((len(steps), len(steps)), np.nan)
    for t, s, v in cells:
        if int(t) in idx and int(s) in idx:
            M[idx[int(t)], idx[int(s)]] = v
    return M


def harness_equal(
    pairs: Iterable[tuple[float, float]], tol: float = HARNESS_TOL
) -> bool | None:
    """K3 harness check: our re-score equals hers within `tol` on every scored pair."""
    vals = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
    if not vals:
        return None
    return all(abs(a - b) <= tol for a, b in vals)


def apply_by_group(
    frame: pd.DataFrame, keys: Sequence[str], fn: Callable[[pd.DataFrame], dict]
) -> pd.DataFrame:
    """One output row per `keys` group: the group keys plus ``fn(group)``."""
    rows = []
    for vals, grp in frame.groupby(list(keys), dropna=False, sort=True):
        vals = vals if isinstance(vals, tuple) else (vals,)
        rows.append({**dict(zip(keys, vals, strict=True)), **fn(grp)})
    return pd.DataFrame(rows)
