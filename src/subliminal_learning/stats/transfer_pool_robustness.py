"""Pool-label exchangeability sensitivity diagnostics for transfer tests.

These functions are deliberately separate from the primary transfer tests.  They
relabel a whole reused number pool across all cells in a sender block, which is a
useful sensitivity analysis when cell-wise relabelling may be anticonservative.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np


@dataclass(frozen=True)
class PoolPermutationResult:
    observed: int
    p: float
    n_labelings: int
    exact: bool
    floor: float
    null_quantiles: dict[str, float]
    null_hist: list[list[int]]


def _validate_arms(arms, controls, groups, pool_tags):
    if not arms:
        raise ValueError("arms must contain at least one matrix")
    mats = [np.asarray(a, dtype=float) for a in arms]
    if any(a.ndim != 2 for a in mats):
        raise ValueError("each arm must be a cells x pools 2-D array")
    tags = list(pool_tags) if pool_tags is not None else list(range(mats[0].shape[1]))
    if len(tags) != mats[0].shape[1] or len(set(tags)) != len(tags):
        raise ValueError("pool_tags must uniquely name every pool column")
    if any(a.shape[1] != len(tags) for a in mats):
        raise ValueError("all arms must have the same pool tags")
    if any(a.size == 0 or not np.isfinite(a).all() for a in mats):
        raise ValueError("all arms must be complete and finite")
    if any(a.shape[0] != mats[0].shape[0] for a in mats):
        raise ValueError("all arms must have the same number of cells")
    ctrl = np.asarray(controls, dtype=int).ravel()
    if ctrl.size != len(mats):
        raise ValueError("controls must have one index per arm")
    if np.any((ctrl < 0) | (ctrl >= len(tags))):
        raise ValueError("control indices must name existing pool columns")
    grp = np.asarray(groups if groups is not None else ["all"] * len(mats), dtype=object).ravel()
    if grp.size != len(mats):
        raise ValueError("groups must have one label per arm")
    return mats, ctrl, grp, tags


def _score_table(mats):
    """Per-arm, per-candidate-pool exceedance counts."""
    scores = np.zeros((len(mats), mats[0].shape[1]), dtype=int)
    for i, mat in enumerate(mats):
        for j in range(mat.shape[1]):
            threshold = np.max(mat[:, j])
            scores[i, j] = int(np.sum(mat > threshold))
    return scores


def _statistic(scores, labels_by_group, groups):
    labels_by_group = dict(labels_by_group)
    return int(sum(scores[i, labels_by_group[group]] for i, group in enumerate(groups)))


def whole_pool_permutation(
    arms,
    controls,
    *,
    groups=None,
    pool_tags=None,
    max_exact: int = 1_000_000,
    n_perm: int = 1999,
    seed: int = 0,
) -> PoolPermutationResult:
    """Relabel one control pool jointly across every cell in each sender group.

    ``arms`` contains same-shaped cells x pools matrices.  Arms in one ``groups``
    block share the same relabelled pool; distinct blocks are permuted independently.
    The statistic sums each arm's treatment values above that arm's own control
    maximum.  Ties count in the upper tail.  Monte Carlo uses the +1 correction.
    """
    mats, ctrl, grp, tags = _validate_arms(arms, controls, groups, pool_tags)
    unique_groups = list(dict.fromkeys(grp.tolist()))
    group_indices = [[i for i, g in enumerate(grp) if g == key] for key in unique_groups]
    for ix in group_indices:
        if len({ctrl[i] for i in ix}) != 1:
            raise ValueError("arms in a shared group must use the same control pool tag")
    n_pools = len(tags)
    scores = _score_table(mats)
    observed_labels = {key: ctrl[ix[0]] for key, ix in zip(unique_groups, group_indices)}
    observed = _statistic(scores, observed_labels, grp)
    n_labelings = n_pools ** len(group_indices)

    def draw_labels(n):
        rng = np.random.default_rng(seed)
        return rng.integers(0, n_pools, size=(n, len(group_indices)))

    if n_labelings <= max_exact:
        labels = np.asarray(list(itertools.product(range(n_pools), repeat=len(group_indices))), dtype=int)
        exact = True
    else:
        labels = draw_labels(n_perm)
        exact = False
    null = np.asarray([
        _statistic(scores, dict(zip(unique_groups, (int(x) for x in row))), grp) for row in labels
    ], dtype=int)
    p = float(np.mean(null >= observed)) if exact else float((np.sum(null >= observed) + 1) / (len(null) + 1))
    floor = 1.0 / len(null) if exact else 1.0 / (len(null) + 1)
    vals, freqs = np.unique(null, return_counts=True)
    return PoolPermutationResult(
        observed=int(observed), p=p, n_labelings=int(len(null)), exact=exact,
        floor=floor,
        null_quantiles={str(q): float(np.percentile(null, q)) for q in (50, 95, 99, 99.9)},
        null_hist=[[int(v), int(f)] for v, f in zip(vals, freqs)],
    )


def simulate_null_rejection(
    *,
    rho: float,
    n_replicates: int = 500,
    n_cells: int = 20,
    n_teachers: int = 10,
    n_perm: int = 999,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | int]:
    """Idealized null calibration comparing cell and whole-pool relabelling.

    Pool random effects are shared across cells with intraclass correlation ``rho``;
    there is no treatment effect.  This is a diagnostic simulation, not an estimate
    of the actual experiment's type-I error.
    """
    if not 0 <= rho <= 1:
        raise ValueError("rho must be between 0 and 1")
    rng = np.random.default_rng(seed)
    cell_reject = pool_reject = 0
    for _ in range(n_replicates):
        shared = rng.normal(size=n_teachers + 1)
        noise = rng.normal(size=(n_cells, n_teachers + 1))
        values = math.sqrt(rho) * shared + math.sqrt(1 - rho) * noise
        from subliminal_learning.stats.transfer_significance_tests import stratified_exceedance_permutation
        current = stratified_exceedance_permutation(
            values, np.zeros(n_cells, dtype=int), n_perm=n_perm,
            seed=int(rng.integers(2**31)),
        )
        cell_p = current.p
        pool = whole_pool_permutation([values], [0], pool_tags=list(range(n_teachers + 1)), n_perm=n_perm, seed=int(rng.integers(2**31)))
        cell_reject += cell_p <= alpha
        pool_reject += pool.p <= alpha
    rate = cell_reject / n_replicates
    z = 1.959963984540054
    denom = 1 + z * z / n_replicates
    center = (rate + z * z / (2 * n_replicates)) / denom
    half = z * math.sqrt(rate * (1 - rate) / n_replicates + z * z / (4 * n_replicates**2)) / denom
    pool_rate = pool_reject / n_replicates
    pool_center = (pool_rate + z * z / (2 * n_replicates)) / denom
    pool_half = z * math.sqrt(pool_rate * (1 - pool_rate) / n_replicates + z * z / (4 * n_replicates**2)) / denom
    return {
        "rho": float(rho), "n_replicates": int(n_replicates), "n_cells": int(n_cells),
        "n_teachers": int(n_teachers), "n_perm": int(n_perm), "alpha": float(alpha),
        "cell_rejection_rate": float(rate),
        "cell_wilson95": [float(center - half), float(center + half)],
        "pool_rejection_rate": float(pool_rate),
        "pool_wilson95": [float(pool_center - pool_half), float(pool_center + pool_half)],
        "cell_rejections": int(cell_reject), "pool_rejections": int(pool_reject),
    }
