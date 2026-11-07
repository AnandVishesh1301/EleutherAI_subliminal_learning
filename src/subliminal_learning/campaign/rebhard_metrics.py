"""Exceedance test and paired Δ of the `rebhard` campaign (pure).

Rows are one student on one target, as `pipelines.rebhard_results_pipeline.leaf_rows`
builds them: `source`, `target`, `student_base`, `teacher_base`, `recipe`, `setting`, `lr`,
`optim`, `seed`, `collapsed`, `delta_ce` and `p_<split>`. Rules:
`docs/PREREG_REBHARD.md`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from subliminal_learning.campaign import newdom_hard_metrics as nm

N_PERM = 20_000
PERM_SEED = 0
ALPHA = 0.05
MIN_DELTA = 0.02
SPLIT_SELECT = "validation"
SPLIT_REPORT = "test"


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# ---- exceedance test -------------------------------------------------------------------


@dataclass(frozen=True)
class ExceedanceResult:
    """One permutation test.

    Attributes:
        observed: Treatment runs strictly above their arm's highest control.
        n_treatment: Finite treatment runs.
        p: One-sided p, `(#{null >= observed} + 1) / (n_perm + 1)`.
        null_mean: Mean of the null draws.
        null_95: 95th percentile of the null draws.
    """

    observed: int
    n_treatment: int
    p: float
    null_mean: float
    null_95: float

    @property
    def share(self) -> float:
        return self.observed / self.n_treatment if self.n_treatment else math.nan


def exceedance_test(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
) -> ExceedanceResult:
    """The paper's stratified exceedance permutation test (`app:statistics`).

    Port of `stats/transfer_significance_tests.py` at `1810eca`
    (`_stratified_exceedance_permutation`). Each row is a stratum, one (arm, setting): which
    of its finite runs carries the control label is redrawn uniformly, independently across
    rows, and each arm's maximum control is recomputed per draw.

    Args:
        values: `(cells, runs)` p(target); the control is the last column, NaN marks a
            missing run.
        groups: Arm label per row; each arm is judged against its own highest control.
        n_perm: Null draws.
        seed: `numpy.random.default_rng` seed.

    Raises:
        ValueError: A shape mismatch, or a stratum without a finite control.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] != len(groups):
        raise ValueError(f"values {v.shape} do not match {len(groups)} groups")
    n_cells, n_runs = v.shape
    finite = np.isfinite(v)
    if not finite[:, -1].all():
        raise ValueError("every stratum needs a finite control")
    labels, inv = np.unique(np.asarray(groups), return_inverse=True)
    inv = inv.ravel()
    masks = [inv == k for k in range(len(labels))]
    rows = np.arange(n_cells)

    def counts(ctrl_idx: np.ndarray) -> np.ndarray:
        controls = v[rows[None, :], ctrl_idx]
        thresh = np.stack([np.nanmax(controls[:, m], axis=1) for m in masks], axis=1)
        above = (v[None, :, :] > thresh[:, inv][:, :, None]) & finite[None, :, :]
        above[np.arange(ctrl_idx.shape[0])[:, None], rows[None, :], ctrl_idx] = False
        return above.sum(axis=2)

    observed = int(counts(np.full((1, n_cells), n_runs - 1))[0].sum())
    rng = np.random.default_rng(seed)
    cum = np.cumsum(finite / finite.sum(axis=1, keepdims=True), axis=1)
    u = rng.random((n_perm, n_cells))
    null_idx = (u[:, :, None] > cum[None, :, :]).sum(axis=2).clip(max=n_runs - 1)
    chunk = max(1, int(2e8) // max(1, n_cells * n_runs))
    null = np.concatenate(
        [
            counts(null_idx[start : start + chunk]).sum(axis=1)
            for start in range(0, n_perm, chunk)
        ]
    )
    return ExceedanceResult(
        observed=observed,
        n_treatment=int(finite[:, :-1].sum()),
        p=float((int((null >= observed).sum()) + 1) / (n_perm + 1)),
        null_mean=float(null.mean()),
        null_95=float(np.percentile(null, 95)),
    )


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p per label."""
    order = sorted(p_values, key=lambda k: p_values[k])
    out: dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p_values[key]))
        out[key] = running
    return out


def arm_matrix(
    rows: Iterable[Mapping],
    target: str,
    student: str,
    recipes: Sequence[str],
    settings: Sequence[str],
    split: str = SPLIT_REPORT,
) -> np.ndarray:
    """`(settings, recipes + 1)` p(target) of one arm, the control last (NaN if missing)."""
    col = {r: i for i, r in enumerate(recipes)}
    row = {s: i for i, s in enumerate(settings)}
    out = np.full((len(settings), len(recipes) + 1), np.nan)
    for r in rows:
        if r["target"] != target or r["student_base"] != student:
            continue
        j = len(recipes) if r["source"] == "base" else col[r["recipe"]]
        out[row[r["setting"]], j] = r[f"p_{split}"]
    return out


def arm_summary(rows: Sequence[Mapping], m: np.ndarray, recipes: Sequence[str]) -> dict:
    """What accompanies an arm's test: Δ against the same setting's control, collapse."""
    deltas = (m[:, :-1] - m[:, -1:]).ravel()
    deltas = deltas[np.isfinite(deltas)]
    top = float(np.nanmax(m[:, -1]))
    teachers = [r for r in rows if r["source"] == "teacher"]
    over = [r for r in teachers if _finite(r["p_test"]) and r["p_test"] > top]
    return {
        "max_control": top,
        "delta_mean": float(deltas.mean()) if deltas.size else math.nan,
        "delta_median": float(np.median(deltas)) if deltas.size else math.nan,
        "n_collapsed": sum(bool(r["collapsed"]) for r in teachers),
        "n_exceeders_collapsed": sum(bool(r["collapsed"]) for r in over),
        "n_treatment_missing": int(np.isnan(m[:, :-1]).sum()),
        "n_control_missing": int(np.isnan(m[:, -1]).sum()),
        "n_teachers": len(recipes),
    }


# ---- paired Δ --------------------------------------------------------------------------


def paired_deltas(rows: Iterable[Mapping], splits: Sequence[str]) -> list[dict]:
    """One row per treatment: Δ per split against the control of its setting and seed.

    Raises:
        ValueError: Two controls share a (student, target, setting, seed).
    """
    rows = list(rows)
    controls: dict[tuple, Mapping] = {}
    for r in rows:
        if r["source"] == "base":
            k = (r["student_base"], r["target"], r["setting"], r["seed"])
            if k in controls:
                raise ValueError(f"duplicate control {k}")
            controls[k] = r
    out = []
    for r in rows:
        if r["source"] != "teacher":
            continue
        ctrl = controls.get((r["student_base"], r["target"], r["setting"], r["seed"]))
        delta = {
            s: (
                r[f"p_{s}"] - ctrl[f"p_{s}"]
                if ctrl is not None and _finite(r[f"p_{s}"]) and _finite(ctrl[f"p_{s}"])
                else math.nan
            )
            for s in splits
        }
        control_collapsed = ctrl is not None and bool(ctrl["collapsed"])
        out.append(
            {
                **r,
                "delta": delta,
                "control_found": ctrl is not None,
                "control_collapsed": control_collapsed,
            }
        )
    return out


def rung_rows(pairs: Iterable[Mapping], splits: Sequence[str]) -> list[dict]:
    """Mean Δ per (target, optimizer, rate), with per-seed test Δ and collapse share."""
    groups: dict[tuple, list[Mapping]] = {}
    for p in pairs:
        groups.setdefault((p["target"], p["optim"], p["lr"]), []).append(p)
    out = []
    for (target, optim, lr), units in sorted(groups.items()):
        row: dict = {"target": target, "optim": optim, "lr": lr, "n": len(units)}
        for s in splits:
            vals = [u["delta"][s] for u in units if _finite(u["delta"][s])]
            row[f"delta_{s}"] = float(np.mean(vals)) if vals else math.nan
        row["collapsed_frac"] = float(np.mean([bool(u["collapsed"]) for u in units]))
        row["delta_ce_mean"] = float(np.mean([u["delta_ce"] for u in units]))
        by_seed: dict[int, list[float]] = {}
        for u in units:
            if _finite(u["delta"][SPLIT_REPORT]):
                by_seed.setdefault(u["seed"], []).append(u["delta"][SPLIT_REPORT])
        row["test_by_seed"] = {k: float(np.mean(v)) for k, v in sorted(by_seed.items())}
        out.append(row)
    return out


def select_rung(
    rungs: Iterable[Mapping], target: str, optim: str, cut: float
) -> Mapping | None:
    """hlopt's rung: largest mean validation Δ with mean ΔCE under `cut`, ties to the smaller rate.

    Returns:
        The rung plus `at_edge` (it is the lowest or highest rate of its ladder), or None.
    """
    ladder = [r for r in rungs if r["target"] == target and r["optim"] == optim]
    pool = [
        r
        for r in ladder
        if _finite(r["delta_ce_mean"])
        and r["delta_ce_mean"] < cut
        and _finite(r[f"delta_{SPLIT_SELECT}"])
    ]
    if not pool:
        return None
    best = min(pool, key=lambda r: (-r[f"delta_{SPLIT_SELECT}"], r["lr"]))
    rates = [r["lr"] for r in ladder]
    return {**best, "at_edge": best["lr"] in (min(rates), max(rates))}


def e8_verdict(rung: Mapping | None, n_seeds: int) -> dict:
    """R8: on the selected rung, every seed's test Δ > 0 and the mean ≥ `MIN_DELTA`."""
    if rung is None:
        return {"present": False, "reason": "no_non_collapsed_rung"}
    seeds = rung["test_by_seed"]
    mean = rung[f"delta_{SPLIT_REPORT}"]
    present = (
        len(seeds) == n_seeds
        and all(v > 0 for v in seeds.values())
        and _finite(mean)
        and mean >= MIN_DELTA
    )
    return {
        "present": bool(present),
        "lr": rung["lr"],
        "delta_validation": rung[f"delta_{SPLIT_SELECT}"],
        "delta_test": mean,
        "test_by_seed": dict(seeds),
        "at_edge": rung["at_edge"],
        "collapsed_frac": rung["collapsed_frac"],
    }


def e9_verdict(units: Sequence[Mapping]) -> dict:
    """R9: newdom N1 on one (target, rate).

    A unit is excluded when either student collapsed. Supported needs an eligible rung, at
    least two teachers, every contributing teacher's mean Δ positive, a teacher-aware CI
    above 0 and Δ ≥ `MIN_DELTA`.

    Args:
        units: Paired rows of one (target, rate).
    """
    kept = [
        u
        for u in units
        if not u["collapsed"]
        and not u["control_collapsed"]
        and _finite(u["delta"][SPLIT_REPORT])
    ]
    by_teacher: dict[str, list[float]] = {}
    for u in kept:
        by_teacher.setdefault(u["recipe"], []).append(float(u["delta"][SPLIT_REPORT]))
    est, lo, hi = nm.teacher_boot_ci([by_teacher[k] for k in sorted(by_teacher)])
    n_positive = sum(float(np.mean(v)) > 0 for v in by_teacher.values())
    eligible = bool(units) and len(kept) * 2 >= len(units)
    supported = (
        eligible
        and len(by_teacher) >= 2
        and n_positive == len(by_teacher)
        and _finite(lo)
        and lo > 0
        and _finite(est)
        and est >= MIN_DELTA
    )
    return {
        "delta_test": est,
        "lo": lo,
        "hi": hi,
        "n_units": len(units),
        "n_kept": len(kept),
        "n_teachers": len(by_teacher),
        "n_positive_teachers": n_positive,
        "eligible": eligible,
        "supported": bool(supported),
    }
