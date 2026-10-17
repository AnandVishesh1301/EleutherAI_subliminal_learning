"""Paired deltas, bootstraps, LR selection, specificity and claims of `newdom` hard.

Rows are plain dicts, one per (student, evaluated target), as
`newdom_hard_results_pipeline.leaf_rows` writes them: `target`, `source`, `recipe`,
`lr`, `seed`, `p` (split -> p(target)), `panel` (panel item -> p on the test prompts),
`delta_ce`, `finite`. A *unit* pairs one treatment with the control of the same LR and
seed scored for the same target: Δ = p(treatment) − p(control). A unit is excluded when
either student collapsed (pp1 ΔCE ≥ `HARD_COLLAPSE_DCE` or non-finite); excluded units
enter no mean or interval and are counted. Intervals resample teachers, then units
within each drawn teacher (teacher-aware); the unit-level interval and a t-interval over
per-seed means (descriptive) are reported beside it. Transfer is supported when the
rung is eligible, at least two teachers contribute, every contributing teacher's mean Δ
is positive, the teacher-aware CI excludes 0 and Δtest reaches the teacher-gate floor. Rules: `docs/PREREG_NEWDOM.md` (N0, N1, N3, N4,
N5).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.evaluation.preference_scoring import compute_ci

BOOT_N = 10_000
BOOT_SEED = 0
CONFIDENCE = 0.95
SPECIFICITY_SHARE = 0.5
MAX_EXCLUDED_SHARE = 0.5
MIN_TEACHERS = 2
EFFECT_FLOOR = plan.HARD_GATE_ABS_MIN
SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
UNDERPOWERED = "underpowered"
NO_DATA = "no_data"
DESCRIPTIVE = "descriptive"


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def collapsed(row: Mapping) -> bool:
    """pp1 ΔCE at or above the hlopt collapse cut, or non-finite (missing counts)."""
    dce = row.get("delta_ce")
    return not (row.get("finite") and _finite(dce)) or (
        float(dce) >= plan.HARD_COLLAPSE_DCE
    )


@dataclass(frozen=True)
class Unit:
    """One treatment paired with its (LR, seed) control for one target.

    Attributes:
        delta: Split -> Δ p(target), splits finite on both sides.
        panel_gain: Panel item -> Δ p on the test prompts.
        excluded: Either student collapsed.
        reason: ``treatment_collapsed`` / ``control_collapsed`` / None.
    """

    target: str
    recipe: str
    lr: float
    seed: int
    delta: Mapping[str, float]
    panel_gain: Mapping[str, float]
    excluded: bool
    reason: str | None


def _diff(a: Mapping | None, b: Mapping | None) -> dict[str, float]:
    a, b = a or {}, b or {}
    return {
        k: float(a[k]) - float(b[k])
        for k in a
        if k in b and _finite(a[k]) and _finite(b[k])
    }


def paired_units(rows: Iterable[Mapping]) -> list[Unit]:
    """Every treatment row with a matched control row; unpaired treatments drop."""
    rows = list(rows)
    controls = {
        (r["target"], hard.match_lr(r["lr"]), int(r["seed"])): r
        for r in rows
        if r["source"] == hard.BASE
    }
    out: list[Unit] = []
    for r in rows:
        if r["source"] != hard.TEACHER:
            continue
        lr = hard.match_lr(r["lr"])
        c = controls.get((r["target"], lr, int(r["seed"])))
        if c is None:
            continue
        reason = (
            "treatment_collapsed"
            if collapsed(r)
            else "control_collapsed"
            if collapsed(c)
            else None
        )
        out.append(
            Unit(
                target=r["target"],
                recipe=r["recipe"],
                lr=lr,
                seed=int(r["seed"]),
                delta=_diff(r.get("p"), c.get("p")),
                panel_gain=_diff(r.get("panel"), c.get("panel")),
                excluded=reason is not None,
                reason=reason,
            )
        )
    return out


def units_at(units: Iterable[Unit], target: str, lr: float) -> list[Unit]:
    """Units of one (target, LR), sorted by (recipe, seed)."""
    out = [u for u in units if u.target == target and math.isclose(u.lr, lr)]
    return sorted(out, key=lambda u: (u.recipe, u.seed))


def _values(units: Iterable[Unit], split: str) -> list[float]:
    return [u.delta[split] for u in units if not u.excluded and split in u.delta]


def _percentiles(draws: np.ndarray) -> tuple[float, float]:
    tail = (1 - CONFIDENCE) / 2 * 100
    lo, hi = np.percentile(draws, [tail, 100 - tail])
    return float(lo), float(hi)


def boot_ci(
    values: Sequence[float], *, n: int = BOOT_N, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """`(mean, lo, hi)`: percentile CI of the mean over `n` unit resamples."""
    v = np.asarray([x for x in values if _finite(x)], dtype=float)
    if v.size == 0:
        return math.nan, math.nan, math.nan
    draws = np.random.default_rng(seed).choice(v, (n, v.size)).mean(1)
    return (float(v.mean()), *_percentiles(draws))


def teacher_boot_ci(
    groups: Sequence[Sequence[float]], *, n: int = BOOT_N, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """`(mean, lo, hi)`: two-stage percentile CI of the pooled unit mean.

    Each resample draws as many teachers as there are, with replacement, then each
    drawn teacher's units with replacement; the statistic is the mean over every
    drawn unit, so it matches the pooled point estimate.

    Args:
        groups: Per-teacher finite unit values (empty teachers are dropped).
        n: Resamples.
        seed: `numpy.random.default_rng` seed.
    """
    vals = [np.asarray(g, dtype=float) for g in groups if len(g)]
    if not vals:
        return math.nan, math.nan, math.nan
    counts = np.asarray([v.size for v in vals])
    width = int(counts.max())
    table = np.full((len(vals), width), np.nan)
    for i, v in enumerate(vals):
        table[i, : v.size] = v
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(vals), (n, len(vals)))
    size = counts[pick][..., None]
    slot = (rng.random((n, len(vals), width)) * size).astype(int)
    keep = np.arange(width) < size
    drawn = np.where(keep, table[pick[..., None], slot], 0.0)
    draws = drawn.sum((1, 2)) / keep.sum((1, 2))
    return (float(np.concatenate(vals).mean()), *_percentiles(draws))


def interval(units: Iterable[Unit], split: str) -> dict:
    """Pooled Δ of the included units with teacher-aware and unit-level CIs."""
    by_teacher: dict[str, list[float]] = {}
    for u in units:
        if not u.excluded and _finite(u.delta.get(split)):
            by_teacher.setdefault(u.recipe, []).append(float(u.delta[split]))
    est, lo, hi = teacher_boot_ci([by_teacher[r] for r in sorted(by_teacher)])
    _, unit_lo, unit_hi = boot_ci([x for v in by_teacher.values() for x in v])
    return {"delta": est, "lo": lo, "hi": hi, "unit_lo": unit_lo, "unit_hi": unit_hi}


def seed_t_interval(units: Iterable[Unit], split: str) -> dict:
    """t-interval (df = n_seeds − 1) over per-seed means of the included units' Δ."""
    by_seed: dict[int, list[float]] = {}
    for u in units:
        if not u.excluded and _finite(u.delta.get(split)):
            by_seed.setdefault(u.seed, []).append(float(u.delta[split]))
    means = [float(np.mean(by_seed[k])) for k in sorted(by_seed)]
    if len(means) < 2:
        return {"t_lo": math.nan, "t_hi": math.nan, "n_seeds": len(means)}
    ci = compute_ci(means, CONFIDENCE)
    return {"t_lo": ci.lower, "t_hi": ci.upper, "n_seeds": len(means)}


def teacher_means(at: Iterable[Unit], split: str) -> dict[str, float]:
    """Contributing teacher -> mean Δ of its included units on `split`."""
    by: dict[str, list[float]] = {}
    for u in at:
        if not u.excluded and _finite(u.delta.get(split)):
            by.setdefault(u.recipe, []).append(float(u.delta[split]))
    return {r: float(np.mean(v)) for r, v in sorted(by.items())}


def n_teachers(at: Iterable[Unit], split: str) -> int:
    """Teachers with at least one included unit scored on `split`."""
    return len(teacher_means(at, split))


def n_positive_teachers(at: Iterable[Unit], split: str) -> int:
    """Contributing teachers whose mean Δ on `split` is positive."""
    return sum(m > 0 for m in teacher_means(at, split).values())


def rung_eligible(at: Sequence[Unit]) -> bool:
    """At most half of the rung's units excluded, and a validation Δ to rank on."""
    n_excluded = sum(u.excluded for u in at)
    return (
        bool(at)
        and n_excluded <= MAX_EXCLUDED_SHARE * len(at)
        and bool(_values(at, hard.SELECT_SPLIT))
    )


def lr_summary(units: Sequence[Unit], target: str, lr: float) -> dict:
    """Δ per split, test and off-topic CIs, exclusions and N3 eligibility of a rung."""
    at = units_at(units, target, lr)
    test = interval(at, hard.REPORT_SPLIT)
    off = interval(at, hard.OFF_TOPIC)
    out = {
        "target": target,
        "domain": plan.TARGET_DOMAIN[target],
        "lr": lr,
        "headline": math.isclose(lr, plan.HARD_HEADLINE_LR),
        "n_units": len(at),
        "n_excluded": sum(u.excluded for u in at),
        "eligible": rung_eligible(at),
        "excluded": [f"{u.recipe}|s{u.seed}|{u.reason}" for u in at if u.excluded],
    }
    for split in hard.STUDENT_SPLITS:
        v = _values(at, split)
        out[f"delta_{split}"] = float(np.mean(v)) if v else math.nan
    for name, ci in (("test", test), ("off_topic", off)):
        for key in ("lo", "hi", "unit_lo", "unit_hi"):
            out[f"{name}_{key}"] = ci[key]
    t = seed_t_interval(at, hard.REPORT_SPLIT)
    out.update(
        test_t_lo=t["t_lo"],
        test_t_hi=t["t_hi"],
        n_seeds=t["n_seeds"],
        n_teachers=n_teachers(at, hard.REPORT_SPLIT),
        n_positive_teachers=n_positive_teachers(at, hard.REPORT_SPLIT),
    )
    return out


def deltas_by_lr(units: Sequence[Unit]) -> list[dict]:
    """`lr_summary` of every target at every LR rung."""
    return [
        lr_summary(units, t, lr) for t in plan.TARGETS for lr in plan.HARD_STUDENT_LRS
    ]


def select_lr(units: Sequence[Unit], target: str) -> float | None:
    """Eligible LR with the largest mean validation Δ over included units (N3).

    Ties go to the smaller LR; see `rung_eligible`.
    """
    cands = []
    for lr in plan.HARD_STUDENT_LRS:
        at = units_at(units, target, lr)
        if rung_eligible(at):
            cands.append((float(np.mean(_values(at, hard.SELECT_SPLIT))), -lr, lr))
    return max(cands)[2] if cands else None


def specificity(units: Sequence[Unit], target: str) -> dict:
    """Panel gains of one target's included units at one LR (N4 rule).

    Supported iff the target's gain is positive, at least every other item's gain,
    and at least half the positive panel gain.
    """
    panel = hard.domain_of(target).panel
    use = [u for u in units if not u.excluded and set(panel) <= set(u.panel_gain)]
    gains = {
        c: float(np.mean([u.panel_gain[c] for u in use])) if use else math.nan
        for c in panel
    }
    tg = gains[target]
    positive = sum(max(g, 0.0) for g in gains.values() if _finite(g))
    if not use:
        verdict = NO_DATA
    else:
        largest = all(tg >= g for c, g in gains.items() if c != target)
        ok = tg > 0 and largest and tg >= SPECIFICITY_SHARE * positive
        verdict = SUPPORTED if ok else NOT_SUPPORTED
    ordered = sorted(gains, key=lambda c: -gains[c] if _finite(gains[c]) else math.inf)
    return {
        "verdict": verdict,
        "n_units": len(use),
        "gains": gains,
        "target_gain": tg,
        "positive_gain": positive,
        "target_share": tg / positive if positive > 0 else math.nan,
        "target_rank": ordered.index(target) + 1 if use else None,
    }


# ---- claims ---------------------------------------------------------------------------


def transfer(at: Sequence[Unit]) -> dict:
    """Test Δ of one rung's units with its CIs and transfer verdict.

    ``no_data`` without an included unit; ``underpowered`` when the rung is not
    eligible or fewer than `MIN_TEACHERS` teachers contribute; ``supported`` when
    every contributing teacher's mean Δ is positive, the teacher-aware CI excludes 0
    and Δtest >= `EFFECT_FLOOR`.
    """
    ci = interval(at, hard.REPORT_SPLIT)
    est, lo = ci["delta"], ci["lo"]
    k = n_teachers(at, hard.REPORT_SPLIT)
    k_pos = n_positive_teachers(at, hard.REPORT_SPLIT)
    if not _finite(est):
        verdict = NO_DATA
    elif not rung_eligible(at) or k < MIN_TEACHERS:
        verdict = UNDERPOWERED
    elif k_pos == k and lo > 0 and est >= EFFECT_FLOOR:
        verdict = SUPPORTED
    else:
        verdict = NOT_SUPPORTED
    return {
        "verdict": verdict,
        "delta_test": est,
        **{key: v for key, v in ci.items() if key != "delta"},
        **seed_t_interval(at, hard.REPORT_SPLIT),
        "n_teachers": k,
        "n_positive_teachers": k_pos,
        "n_units": len(at),
        "n_excluded": sum(u.excluded for u in at),
        "eligible": rung_eligible(at),
        "effect_floor": EFFECT_FLOOR,
    }


def claim_n1(units: Sequence[Unit]) -> dict:
    """Hard transfer per target at the headline LR (`transfer` rule)."""
    return {
        t: {
            "lr": plan.HARD_HEADLINE_LR,
            **transfer(units_at(units, t, plan.HARD_HEADLINE_LR)),
        }
        for t in plan.TARGETS
    }


def claim_n3(units: Sequence[Unit]) -> dict:
    """Per target: the validation-selected eligible LR, its test Δ, CIs and `transfer`
    verdict (a descriptive claim)."""
    out = {}
    for t in plan.TARGETS:
        rungs = {}
        for x in plan.HARD_STUDENT_LRS:
            s = lr_summary(units, t, x)
            rungs[f"{x:g}"] = {
                k: s[k]
                for k in ("n_units", "n_excluded", "eligible", "delta_validation")
            }
        lr = select_lr(units, t)
        if lr is None:
            out[t] = {"verdict": NO_DATA, "lr": None, "rungs": rungs}
            continue
        at = units_at(units, t, lr)
        out[t] = {
            "kind": DESCRIPTIVE,
            "lr": lr,
            "delta_validation": float(np.mean(_values(at, hard.SELECT_SPLIT))),
            **transfer(at),
            "rungs": rungs,
        }
    return out


def claim_n4(units: Sequence[Unit]) -> dict:
    """Specificity of each target at its N3 LR."""
    out = {}
    for t in plan.TARGETS:
        lr = select_lr(units, t)
        if lr is None:
            out[t] = {"verdict": NO_DATA, "lr": None}
            continue
        out[t] = {"lr": lr, **specificity(units_at(units, t, lr), t)}
    return out


def claim_n5(units: Sequence[Unit]) -> dict:
    """Off-topic Δ per target at every LR (descriptive)."""
    out: dict = {}
    for t in plan.TARGETS:
        by_lr = {}
        for lr in plan.HARD_STUDENT_LRS:
            by_lr[f"{lr:g}"] = interval(units_at(units, t, lr), hard.OFF_TOPIC)
        out[t] = {"verdict": DESCRIPTIVE, "set": plan.OFF_TOPIC_SET, "by_lr": by_lr}
    return out


@dataclass(frozen=True)
class ValidityInputs:
    """Driver records the N0 hard checks read.

    Attributes:
        admitted: Target -> admitted recipes, ranked.
        selected: Target -> student teachers.
        screen_hits: (target, recipe) -> domain-screen hits of each selected pool.
        control_screen_hits: Domain -> hits in the staged control pool.
        reviews: Target -> its pool review is approved and names the selected
            pools' current filtered sha256.
    """

    admitted: Mapping[str, Sequence[str]]
    selected: Mapping[str, Sequence[str]]
    screen_hits: Mapping[tuple[str, str], int]
    control_screen_hits: Mapping[str, int]
    reviews: Mapping[str, bool]


def claim_n0(v: ValidityInputs) -> dict:
    """Hard validity: per target an admitted teacher with a usable, screen-clean pool
    and an approved sha-bound pool review; a clean control pool."""
    per = {}
    for t in plan.TARGETS:
        sel = list(v.selected.get(t, ()))
        hits = {r: v.screen_hits.get((t, r)) for r in sel}
        reviewed = bool(v.reviews.get(t))
        ok = bool(sel) and all(h == 0 for h in hits.values()) and reviewed
        per[t] = {
            "ok": ok,
            "n_admitted": len(v.admitted.get(t, ())),
            "selected": sel,
            "n_selected": len(sel),
            "screen_hits": hits,
            "review_approved": reviewed,
            "exhausted": len(sel) < plan.HARD_TEACHERS_PER_TARGET,
        }
    control_ok = set(v.control_screen_hits) == set(plan.DOMAINS) and not any(
        v.control_screen_hits.values()
    )
    ok = all(p["ok"] for p in per.values()) and control_ok
    return {
        "verdict": SUPPORTED if ok else NOT_SUPPORTED,
        "by_target": per,
        "control_screen_hits": dict(v.control_screen_hits),
    }


def claims(units: Sequence[Unit], validity: ValidityInputs) -> dict:
    """Every hard claim of `docs/PREREG_NEWDOM.md` (N0, N1, N3, N4, N5)."""
    return {
        "N0": claim_n0(validity),
        "N1": claim_n1(units),
        "N3": claim_n3(units),
        "N4": claim_n4(units),
        "N5": claim_n5(units),
    }
