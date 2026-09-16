"""Significance tests for cross-seed transfer against matched base-model controls.

The question every test here answers: a sender's numbers, distilled into a receiver that shares
no seed with it, move the receiver's animal preference above the SAME receiver distilled on an
untuned base pool (the matched control). Is that move larger than chance? Each function states
its own null and exchangeability assumption; three of them are hand-rolled because no library
exposes the estimator (the exact exceedance product, and the two relabelling permutation tests);
the sign test wraps scipy.

Pure functions over numpy/pandas: no disk IO, no plotting. Loaders live one layer up in
``subliminal_learning.analysis.transfer_significance_io``. Results are small frozen dataclasses holding
plain ``float``/``int`` so they JSON-serialize unchanged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "SignTestResult",
    "ExceedanceResult",
    "StratifiedPermResult",
    "PoolRankResult",
    "paired_sign_test",
    "exceedance_test",
    "stratified_exceedance_permutation",
    "topk_mean",
    "pool_rank_permutation",
    "conformal_rank_p",
    "prior_scaled_effect",
]


@dataclass(frozen=True)
class SignTestResult:
    """Outcome of `paired_sign_test`: wins out of `n` usable pairs and its one-sided p."""

    wins: int
    n: int
    p: float


@dataclass(frozen=True)
class ExceedanceResult:
    """Outcome of `exceedance_test`: `count` of `m` treatment values above all `n` controls."""

    count: int
    m: int
    n: int
    expected: float
    p: float


@dataclass(frozen=True)
class StratifiedPermResult:
    """Outcome of `stratified_exceedance_permutation` with its null distribution summary.

    ``observed_by_group`` is empty without ``groups`` and otherwise holds each group's observed
    count (keyed by ``str(label)``), which sums to `observed`.
    """

    observed: int
    p: float
    null_quantiles: dict[str, float]
    n_perm: int
    floor: float
    observed_by_group: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PoolRankResult:
    """Outcome of `pool_rank_permutation`; `exact` says whether labelings were enumerated."""

    observed: float
    p: float
    null_mean: float
    null_sd: float
    per_sender_rank: dict[str, int]
    n_labelings: int
    exact: bool


def _finite_pairs(treatment, control) -> tuple[np.ndarray, np.ndarray]:
    """The positionally paired entries where BOTH sides are finite (NaN/None dropped)."""
    t = np.asarray(treatment, dtype=float).ravel()
    c = np.asarray(control, dtype=float).ravel()
    if t.size != c.size:
        raise ValueError(f"paired inputs must have equal length, got {t.size} and {c.size}")
    keep = np.isfinite(t) & np.isfinite(c)
    return t[keep], c[keep]


def paired_sign_test(treatment, control) -> SignTestResult:
    """Exact one-sided sign test: does treatment beat its matched control cell by cell?

    Null: within a pair, treatment and control are equally likely to be the larger, i.e.
    ``wins ~ Binomial(n, 0.5)``. Reference: the sign test (scipy ``binomtest``, upper tail);
    this wraps `stats.core.sign_binom_test`, which shares the tie/NaN handling.

    Assumption: the pairs are exchangeable *given* the design — they share the student cell and
    the receiver, so treatment and control differ only in which pool was distilled. They are NOT
    independent in the strict sense: every treatment cell of a teacher reads the same numbers
    pool, and the controls likewise, so the n pairs are conditional on the shared pool draws.
    Treat the p as a screen, not as a fully independent-sample probability.

    Args:
        treatment: Per-cell treatment values, paired by position with `control`.
        control: Per-cell matched control values.

    Returns:
        A `SignTestResult`; ties are not wins and NaN/None pairs are dropped, so `n` counts
        only finite, non-tied pairs.
    """
    from subliminal_learning.stats.core import sign_binom_test

    t, c = _finite_pairs(treatment, control)
    res = sign_binom_test(t - c, alternative="greater")
    return SignTestResult(wins=int(res.k), n=int(res.n), p=float(res.pvalue))


def exceedance_test(treatment, control) -> ExceedanceResult:
    """Exact Rosenbaum (1954) exceedance test: treatment values above EVERY control value.

    Null: the ``m + n`` values are exchangeable (one common distribution, no treatment effect).
    Under that null the exact probability of at least ``k`` exceedances is the product

        ``P(count >= k) = prod_{j=0}^{k-1} (m - j) / (m + n - j)``,

    the chance that the k largest of the pooled sample all carry treatment labels. Hand-rolled:
    scipy has no exceedance-test API. Deliberately NOT a binomial approximation — the draws are
    without replacement, so a binomial p is anticonservative for small ``n``.

    Assumption: exchangeability of the pooled values; ties are resolved strictly (a treatment
    value equal to the control maximum does not count).

    Args:
        treatment: Treatment values (NaN dropped).
        control: Control values (NaN dropped); their maximum is the threshold.

    Returns:
        An `ExceedanceResult`. ``expected = m / (n + 1)`` is the null mean number of
        exceedances; ``count == 0`` gives ``p == 1.0``.
    """
    t = np.asarray(treatment, dtype=float).ravel()
    c = np.asarray(control, dtype=float).ravel()
    t, c = t[np.isfinite(t)], c[np.isfinite(c)]
    if c.size == 0:
        raise ValueError("exceedance_test needs at least one finite control value")
    m, n = int(t.size), int(c.size)
    k = int((t > c.max()).sum())
    p = 1.0
    for j in range(k):
        p *= (m - j) / (m + n - j)
    return ExceedanceResult(count=k, m=m, n=n, expected=m / (n + 1), p=float(p))


def stratified_exceedance_permutation(
    values, control_index, *, groups=None, n_perm: int = 20000, seed: int = 0
) -> StratifiedPermResult:
    """Cell- and arm-stratified exceedance permutation test on the pooled exceedance count.

    Statistic: the number of non-control finite values strictly above the maximum of the control
    values. Without `groups` there is ONE control maximum over the whole design (the pooled
    exceedance count of `exceedance_test`). With `groups` every group -- an arm, i.e. one
    sender->receiver direction with its own control pool -- is an additional stratum: its
    exceedances are counted against ITS OWN control maximum and the statistic is the SUM over
    groups. A single global control maximum across arms would be invalid, because receivers
    differ in prior and hence in control distribution, so the loudest arm's control would
    silently become the threshold for every other arm.

    Null: within each cell, *which* column is the control is arbitrary -- relabel it uniformly
    among that cell's finite entries, independently across cells (`groups` does not change the
    relabelling; only the per-group maxima and the summed count are recomputed from it). This
    keeps each cell's set of values fixed, so cell-level difficulty (the K/K* position of the
    student cell) cannot manufacture a signal; only the treatment/control assignment moves.
    Hand-rolled: no library exposes this stratified-relabelling statistic.

    Args:
        values: ``(cells x pools)`` float array; NaN marks a missing run. NaN entries are never
            eligible as the relabelled control and never count as exceedances.
        control_index: Per-cell column index of the observed control (int array of length
            ``cells``), or a scalar applied to every cell.
        groups: Optional group (arm) label per cell row; ``None`` pools every cell under one
            control maximum, reproducing the un-grouped test exactly.
        n_perm: Number of null draws.
        seed: Seed of the ``default_rng`` used for the draws (deterministic given the inputs).

    Returns:
        A `StratifiedPermResult` with ``p = (#{null >= observed} + 1) / (n_perm + 1)`` and
        ``floor = 1 / (n_perm + 1)``, the smallest p this resolution can report.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError("values must be a (cells x pools) 2-D array")
    n_cells, n_pools = v.shape
    ctrl = np.full(n_cells, int(control_index)) if np.ndim(control_index) == 0 else np.asarray(
        control_index, dtype=int
    )
    if ctrl.shape != (n_cells,):
        raise ValueError("control_index must be a scalar or one index per cell")
    finite = np.isfinite(v)
    if groups is None:
        labels, inv = None, np.zeros(n_cells, dtype=int)
    else:
        g = np.asarray(groups).ravel()
        if g.shape != (n_cells,):
            raise ValueError("groups must be one label per cell")
        labels, inv = np.unique(g, return_inverse=True)
        inv = inv.ravel()
    masks = [inv == k for k in range(1 if labels is None else len(labels))]

    def _cell_counts(ctrl_idx: np.ndarray) -> np.ndarray:
        """Per-cell exceedances for a ``(draws x cells)`` matrix of control column indices."""
        rows = np.arange(n_cells)
        controls = v[rows[None, :], ctrl_idx]  # (draws, cells)
        # One threshold per (draw, group); a cell is judged against its own group's maximum.
        gmax = np.stack([np.nanmax(controls[:, m], axis=1) for m in masks], axis=1)
        thresh = gmax[:, inv]  # (draws, cells)
        above = (v[None, :, :] > thresh[:, :, None]) & finite[None, :, :]
        # The cells' own control entries are not treatment values.
        above[np.arange(ctrl_idx.shape[0])[:, None], rows[None, :], ctrl_idx] = False
        return above.sum(axis=2)

    obs_cells = _cell_counts(ctrl[None, :])[0]
    observed = int(obs_cells.sum())
    by_group = ({} if labels is None
                else {str(lab): int(obs_cells[masks[k]].sum()) for k, lab in enumerate(labels)})

    rng = np.random.default_rng(seed)
    # Draw a finite column per cell: rank the uniform draws restricted to finite entries.
    weights = finite.astype(float)
    if not weights.any(axis=1).all():
        raise ValueError("every cell needs at least one finite value")
    probs = weights / weights.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    u = rng.random((n_perm, n_cells))
    null_idx = (u[:, :, None] > cum[None, :, :]).sum(axis=2).clip(max=n_pools - 1)
    # Chunk the draws so the (draws x cells x pools) comparison stays bounded in memory.
    chunk = max(1, int(2e8) // max(1, n_cells * n_pools))
    null = np.concatenate([_cell_counts(null_idx[start:start + chunk]).sum(axis=1)
                           for start in range(0, n_perm, chunk)])

    p = float((int((null >= observed).sum()) + 1) / (n_perm + 1))
    quantiles = {
        "50": float(np.percentile(null, 50)),
        "95": float(np.percentile(null, 95)),
        "99": float(np.percentile(null, 99)),
        "99.9": float(np.percentile(null, 99.9)),
        "max": float(null.max()),
    }
    return StratifiedPermResult(
        observed=observed, p=p, null_quantiles=quantiles, n_perm=int(n_perm),
        floor=float(1 / (n_perm + 1)), observed_by_group=by_group,
    )


def topk_mean(values, k: int = 5) -> float:
    """Mean of the ``k`` largest finite values (all of them when fewer than ``k`` are finite).

    The default pool statistic for `pool_rank_permutation`: transfer shows up in a pool's BEST
    cells, so a top-k mean is less diluted than the mean over a whole cell set.

    Args:
        values: Per-cell values; NaN dropped.
        k: How many of the top values to average.

    Returns:
        The top-k mean, or ``nan`` when nothing is finite.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.sort(v)[-min(k, v.size):].mean())


def _sender_fractions(frame: pd.DataFrame, control_label: str) -> tuple[np.ndarray, np.ndarray]:
    """``(pools x columns)`` value matrix and the row index of the control pool."""
    if control_label not in frame.index:
        raise ValueError(f"sender frame has no {control_label!r} row")
    mat = frame.to_numpy(dtype=float)
    return mat, np.flatnonzero(frame.index == control_label)[0]


def pool_rank_permutation(
    pool_stats: Mapping[str, pd.DataFrame],
    control_label: str,
    *,
    max_exact: int = 200_000,
    n_mc: int = 20000,
    seed: int = 0,
) -> PoolRankResult:
    """Cross-sender permutation asking whether the control pool ranks low for EVERY sender.

    Statistic: the mean, over every ``(sender, column)`` observation, of the fraction of that
    sender's non-control pools whose value strictly exceeds the control's value in that column.
    ``0.5`` is the chance level.

    Null: for each sender independently, ANY one of its pools could have been the control; the
    relabelled pool is applied to ALL of that sender's columns at once, because a pool is a
    single numbers dataset shared across receivers/animals, so its columns move together.
    Hand-rolled: this per-sender shared-label relabelling has no library API.

    Assumption: within a sender, its pools are exchangeable under the null (no pool is special);
    senders contribute independently.

    Args:
        pool_stats: ``{sender: DataFrame}``; index = pool label (must include `control_label`),
            columns = observation keys (e.g. ``(receiver, animal)`` tuples), values = the pool
            statistic. Columns that are all-NaN for a sender are dropped with a warning.
        control_label: The index label of the control pool.
        max_exact: Enumerate every labeling when ``prod(n_pools) <= max_exact``.
        n_mc: Monte-Carlo draws when exact enumeration is too large.
        seed: Seed of the Monte-Carlo ``default_rng``.

    Returns:
        A `PoolRankResult`. ``p`` is the fraction of labelings with statistic ``>= observed``
        (exact) or ``(count + 1) / (n_mc + 1)`` (Monte Carlo). ``per_sender_rank`` gives, per
        sender, ``1 + #pools whose own mean fraction exceeds the control's`` (1 = control
        lowest). Senders with fewer than two pools are skipped with a `warnings.warn`.
    """
    mats: dict[str, np.ndarray] = {}
    ctrl_rows: dict[str, int] = {}
    for sender, frame in pool_stats.items():
        f = frame.dropna(axis=1, how="all")
        if f.shape[1] < frame.shape[1]:
            warnings.warn(
                f"{sender}: dropped {frame.shape[1] - f.shape[1]} all-NaN column(s)",
                stacklevel=2,
            )
        if f.shape[0] < 2:
            warnings.warn(f"{sender}: fewer than 2 pools, skipped", stacklevel=2)
            continue
        mat, row = _sender_fractions(f, control_label)
        mats[sender], ctrl_rows[sender] = mat, row
    if not mats:
        raise ValueError("no sender has at least two pools")

    senders = list(mats)
    n_pools = [mats[s].shape[0] for s in senders]

    def _frac(sender: str, row: int) -> np.ndarray:
        """Per-column fraction of the sender's other pools that beat pool ``row``."""
        mat = mats[sender]
        others = np.delete(mat, row, axis=0)
        wins = np.nansum(others > mat[row][None, :], axis=0)
        n_valid = np.isfinite(others).sum(axis=0)
        with np.errstate(invalid="ignore"):
            return np.where(n_valid > 0, wins / np.maximum(n_valid, 1), np.nan)

    # Per (sender, pool) the column-mean fraction, and the column count that weights it.
    per_pool = {s: np.array([np.nanmean(_frac(s, r)) for r in range(mats[s].shape[0])])
                for s in senders}
    weights = np.array([mats[s].shape[1] for s in senders], dtype=float)

    def _stat(rows: np.ndarray) -> np.ndarray:
        """Weighted mean fraction over all observations for ``(draws x senders)`` labelings."""
        vals = np.stack([per_pool[s][rows[:, i]] for i, s in enumerate(senders)], axis=1)
        return (vals * weights[None, :]).sum(axis=1) / weights.sum()

    observed_rows = np.array([[ctrl_rows[s] for s in senders]])
    observed = float(_stat(observed_rows)[0])

    n_labelings = int(np.prod(n_pools))
    exact = n_labelings <= max_exact
    if exact:
        grids = np.meshgrid(*[np.arange(n) for n in n_pools], indexing="ij")
        rows = np.stack([g.ravel() for g in grids], axis=1)
        null = _stat(rows)
        p = float((null >= observed).mean())
        n_draws = n_labelings
    else:
        rng = np.random.default_rng(seed)
        rows = np.stack([rng.integers(0, n, size=n_mc) for n in n_pools], axis=1)
        null = _stat(rows)
        p = float((int((null >= observed).sum()) + 1) / (n_mc + 1))
        n_draws = int(n_mc)

    ranks = {s: int(1 + (per_pool[s] > per_pool[s][ctrl_rows[s]]).sum()) for s in senders}
    return PoolRankResult(
        observed=observed, p=p, null_mean=float(null.mean()), null_sd=float(null.std(ddof=0)),
        per_sender_rank=ranks, n_labelings=int(n_draws), exact=bool(exact),
    )


def conformal_rank_p(treatment_values, control_replicates) -> float:
    """Conformal rank p of a treatment summary among exchangeable control replicates.

    Statistic: the median of `treatment_values` (a bare scalar is accepted). Null: the treatment
    statistic is exchangeable with the control replicates, i.e. it is just one more draw from
    the same pool-to-pool noise. Reference: the conformal / rank-based permutation p,
    ``p = (1 + #{controls >= statistic}) / (n_controls + 1)`` — valid for finite n without any
    distributional assumption, and floored at ``1 / (n + 1)``.

    Args:
        treatment_values: Treatment values (median taken), or a scalar statistic.
        control_replicates: Control replicate statistics (NaN dropped).

    Returns:
        The conformal p-value.
    """
    t = np.asarray(treatment_values, dtype=float).ravel()
    t = t[np.isfinite(t)]
    if t.size == 0:
        raise ValueError("conformal_rank_p needs at least one finite treatment value")
    stat = float(np.median(t))
    c = np.asarray(control_replicates, dtype=float).ravel()
    c = c[np.isfinite(c)]
    if c.size == 0:
        raise ValueError("conformal_rank_p needs at least one finite control replicate")
    return float((1 + int((c >= stat).sum())) / (c.size + 1))


def prior_scaled_effect(treatment, control, prior: float) -> float:
    """Mean paired treatment-minus-control gap in units of the receiver's base prior.

    The prior is the untuned receiver's ``p_target``, so ``1.0`` means the gap equals one
    base-rate's worth of preference.

    Args:
        treatment: Per-cell treatment values.
        control: Per-cell matched control values.
        prior: The receiver's base ``p_target`` for the evaluated animal.

    Returns:
        The mean of ``(treatment - control) / prior`` over finite pairs; ``nan`` when none.
    """
    t, c = _finite_pairs(treatment, control)
    if t.size == 0 or not np.isfinite(prior) or prior == 0:
        return float("nan")
    return float(((t - c) / prior).mean())
