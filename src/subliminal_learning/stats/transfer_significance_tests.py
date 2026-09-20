"""Significance tests for cross-seed transfer against matched base-model controls.

The question every test here answers: a sender's numbers, distilled into a receiver that shares
no seed with it, move the receiver's animal preference above the SAME receiver distilled on an
untuned base pool (the matched control). Is that move larger than chance? Each function states
its own null and exchangeability assumption; most are hand-rolled because no library exposes
the estimator (the exact exceedance product, and the relabelling permutation tests); the sign
test wraps scipy.

Pure functions over numpy/pandas: no plotting, no loading. Loaders live one layer up in
``subliminal_learning.analysis.transfer_significance_io``. Results are small frozen dataclasses holding
plain ``float``/``int`` so they JSON-serialize unchanged -- which is what the three permutation
tests do through `stats.memo`: each is deterministic in its inputs (seeded draws), so its result
is memoized on a content hash of those inputs (`memo.key`) and read back exactly on the next call
with the same inputs (``SUBLIMINAL_ANALYSIS_CACHE=0`` turns that off).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from subliminal_learning.stats import memo

__all__ = [
    "SignTestResult",
    "ExceedanceResult",
    "StratifiedPermResult",
    "ArmSetResult",
    "ArmRelabelResult",
    "paired_sign_test",
    "exceedance_test",
    "stratified_exceedance_permutation",
    "prior_scaled_effect",
    "arm_set_permutation",
    "arm_relabel_cell_permutation",
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
class ArmSetResult:
    """Outcome of `arm_set_permutation`: the two families' arm-set contrast.

    ``n_a``/``n_b`` are the usable arm counts of the two families; ``exact`` says whether every
    family relabelling was enumerated.
    """

    observed: float
    p: float
    floor: float
    null_quantiles: dict[str, float]
    n_labelings: int
    exact: bool
    n_a: int
    n_b: int


@dataclass(frozen=True)
class ArmRelabelResult:
    """Outcome of `arm_relabel_cell_permutation`: a run-level statistic under the arm-relabel null.

    The fields of `ArmSetResult` plus `statistic` (which run-level statistic was averaged over
    cells), `n_cells` (design cells carrying both families) and `n_cells_dropped` (single-family
    cells left out).
    """

    observed: float
    p: float
    floor: float
    null_quantiles: dict[str, float]
    n_labelings: int
    exact: bool
    n_a: int
    n_b: int
    statistic: str
    n_cells: int
    n_cells_dropped: int


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
    value equal to the control maximum does not count). That exchangeability holds only when the
    values are replicates at ONE cell -- across cells, which differ in level, this closed form is
    not valid and `stratified_exceedance_permutation` is the test to use; the count itself stays
    a fine descriptive number.

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
    memo_key = memo.key("stratified_exceedance_permutation", v, ctrl,
                        None if labels is None else np.asarray(groups).ravel(), int(n_perm),
                        int(seed))
    return memo.memoized("stratified_exceedance_permutation", memo_key, StratifiedPermResult,
                         lambda: _stratified_exceedance_permutation(v, ctrl, finite, labels, inv,
                                                                    masks, n_perm, seed))


def _stratified_exceedance_permutation(v, ctrl, finite, labels, inv, masks, n_perm, seed
                                       ) -> StratifiedPermResult:
    """The draws of `stratified_exceedance_permutation` on validated inputs."""
    n_cells, n_pools = v.shape

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


def arm_set_permutation(
    frame: pd.DataFrame,
    *,
    family_a: str,
    family_b: str,
    family_col: str = "family",
    value_col: str = "value",
    max_exact: int = 1_000_000,
    n_perm: int = 20000,
    seed: int = 0,
) -> ArmSetResult:
    """Permutation comparing two families' SETS of arms, one value per arm.

    Statistic: ``mean of family-A arm values - mean of family-B arm values``, where an arm value
    is that arm's summary effect (e.g. its mean control-normalized, prior-scaled gap).

    Null: WHICH family an arm belongs to is arbitrary -- permute the family labels among all
    arms of both families in one pool, keeping the A/B counts. There is deliberately NO
    one-to-one pairing of ``x_i`` with ``y_i``: the seed index is an arbitrary label, so pairing
    on it invents a matching the design does not have -- the comparison is between the two sets.

    Arms are the honest unit for a family claim: every student inside an arm shares one numbers
    pool and one receiver, so cells are not replicates of "the family". Caveat on the counts:
    arms within a family also share base models and senders (``x_1->x_2`` and ``x_1->x_3`` share
    a sender), so the arm count overstates its independent information; the p is a randomization
    p over family labels, not a sample from a population of arms.

    Args:
        frame: Tidy rows, one per arm, with a family label and a value.
        family_a: Label of the family claimed stronger (the positive side of the contrast).
        family_b: Label of the reference family.
        family_col, value_col: Column names in `frame`.
        max_exact: Enumerate every labelling when ``C(n, n_A)`` is at most this; else sample.
        n_perm: Monte-Carlo draws when exact enumeration is too large.
        seed: Seed of the Monte-Carlo ``default_rng``.

    Returns:
        An `ArmSetResult`. Exact path: ``p = #{null >= observed} / n_labelings``, floor
        ``1 / n_labelings``; Monte Carlo: ``p = (count + 1) / (n_perm + 1)``, floor
        ``1 / (n_perm + 1)``.

    Raises:
        ValueError: If either family has no arm with a finite value.
    """
    for col in (family_col, value_col):
        if col not in frame.columns:
            raise ValueError(f"frame has no {col!r} column")
    df = frame[frame[family_col].isin([family_a, family_b])]
    df = df[np.isfinite(pd.to_numeric(df[value_col], errors="coerce"))]
    is_a = (df[family_col] == family_a).to_numpy()
    if not is_a.any() or is_a.all():
        raise ValueError("both families need at least one arm with a finite value")
    values = df[value_col].to_numpy(dtype=float)
    memo_key = memo.key("arm_set_permutation", values, is_a, int(max_exact), int(n_perm),
                        int(seed))
    return memo.memoized("arm_set_permutation", memo_key, ArmSetResult,
                         lambda: _arm_set_permutation(values, is_a, max_exact, n_perm, seed))


def _arm_set_permutation(values: np.ndarray, is_a: np.ndarray, max_exact: int, n_perm: int,
                         seed: int) -> ArmSetResult:
    """The enumeration / draws of `arm_set_permutation` on its filtered inputs."""
    from math import comb

    n_a = int(is_a.sum())
    n_b = int(values.size - n_a)

    # mean(A) - mean(B) is affine in the A-sum, so only that sum has to be relabelled.
    def contrast(picks: np.ndarray) -> np.ndarray:
        """Contrast for ``picks`` = the A-indices of each labelling."""
        return values[picks].sum(axis=-1) * (1 / n_a + 1 / n_b) - values.sum() / n_b

    observed = float(contrast(np.flatnonzero(is_a)))
    n_labelings = comb(values.size, n_a)
    exact = n_labelings <= max_exact
    if exact:
        picks = np.array(list(itertools.combinations(range(values.size), n_a)), dtype=int)
        null = contrast(picks)
        p, floor, n_draws = float((null >= observed).mean()), 1.0 / n_labelings, n_labelings
    else:
        rng = np.random.default_rng(seed)
        picks = np.argsort(rng.random((n_perm, values.size)), axis=1)[:, :n_a]
        null = contrast(picks)
        p = float((int((null >= observed).sum()) + 1) / (n_perm + 1))
        floor, n_draws = 1.0 / (n_perm + 1), int(n_perm)

    quantiles = {
        "50": float(np.percentile(null, 50)),
        "95": float(np.percentile(null, 95)),
        "99": float(np.percentile(null, 99)),
        "99.9": float(np.percentile(null, 99.9)),
        "max": float(null.max()),
    }
    return ArmSetResult(
        observed=float(observed), p=float(p), floor=float(floor), null_quantiles=quantiles,
        n_labelings=int(n_draws), exact=bool(exact), n_a=n_a, n_b=n_b,
    )


ARM_RELABEL_STATISTICS = ("mean_contrast", "win_probability")


def _cell_tables(
    df, arm_col: str, stratum_col: str, value_col: str, statistic: str
) -> tuple[np.ndarray, np.ndarray, list]:
    """Per-arm x per-cell ``(numerator sums, counts)`` for `statistic`, plus the arm order.

    For ``"mean_contrast"`` the numerator is the sum of the values; for ``"win_probability"`` it
    is the sum of the within-cell midranks, taken over ALL runs of both families in that cell --
    they do not move when the arm labels do.
    """
    from scipy.stats import rankdata

    arms = list(pd.unique(df[arm_col]))
    cells = list(pd.unique(df[stratum_col]))
    arm_idx = {a: i for i, a in enumerate(arms)}
    sums = np.zeros((len(arms), len(cells)))
    counts = np.zeros((len(arms), len(cells)))
    for j, cell in enumerate(cells):
        block = df[df[stratum_col] == cell]
        values = block[value_col].to_numpy(dtype=float)
        weights = rankdata(values) if statistic == "win_probability" else values
        for arm, w in zip(block[arm_col].to_numpy(), weights):
            sums[arm_idx[arm], j] += float(w)
            counts[arm_idx[arm], j] += 1.0
    return sums, counts, arms


def arm_relabel_cell_permutation(
    frame: pd.DataFrame,
    *,
    family_a: str,
    family_b: str,
    statistic: str = "mean_contrast",
    family_col: str = "family",
    arm_col: str = "arm",
    stratum_col: str = "stratum",
    value_col: str = "value",
    max_exact: int = 1_000_000,
    n_perm: int = 20000,
    seed: int = 0,
) -> ArmRelabelResult:
    """Run-level family contrast under the ARM-relabel null, read within student cells.

    The unit is the ARM: the null relabels whole arms as A/B keeping the two families' arm
    counts, and every run follows its arm. The STATISTIC is evaluated on the runs inside the
    cells, so the student hyperparameters are held fixed and cell difficulty (the K/K* position
    of the cell) can never masquerade as a family effect:

    * ``"mean_contrast"``: per cell, the mean of the A runs minus the mean of the B runs,
      averaged over cells. When every arm covers every cell this equals the difference of the
      two families' mean arm values, so observed and p coincide with `arm_set_permutation` on
      per-arm means; the two differ only in how partial coverage is weighted.
    * ``"win_probability"``: per cell, the share of (A run, B run) pairs in which the A run is
      larger, ties counting 1/2 (Mann-Whitney U over ``n_A n_B``, with midranks), averaged over
      cells. Rank-only, hence invariant to any increasing rescaling of the values.

    Hand-rolled: no library exposes a within-cell statistic under a cluster-level (arm)
    relabelling. Caveat on the counts, as in `arm_set_permutation`: arms within a family share
    base models and senders, so the arm count overstates their independent information.

    Args:
        frame: Tidy rows, one per run, with family, arm, cell (stratum) and value columns.
        family_a: Label of the family claimed stronger (the positive side of the contrast).
        family_b: Label of the reference family.
        statistic: One of `ARM_RELABEL_STATISTICS`.
        family_col, arm_col, stratum_col, value_col: Column names in `frame`.
        max_exact: Enumerate every labelling when ``C(n_arms, n_A)`` is at most this; else sample.
        n_perm: Monte-Carlo draws when exact enumeration is too large.
        seed: Seed of the Monte-Carlo ``default_rng``.

    Returns:
        An `ArmRelabelResult`. Exact path: ``p = #{null >= observed} / n_labelings``, floor
        ``1 / n_labelings``; Monte Carlo: ``p = (count + 1) / (n_perm + 1)``, floor
        ``1 / (n_perm + 1)``. Cells reached by only one family in the observed data define no
        contrast and are dropped (counted in ``n_cells_dropped``).

    Raises:
        ValueError: On an unknown `statistic`, a missing column, a family with no arm, or no
            cell carrying both families.
    """
    if statistic not in ARM_RELABEL_STATISTICS:
        raise ValueError(f"statistic must be one of {ARM_RELABEL_STATISTICS}, got {statistic!r}")
    for col in (family_col, arm_col, stratum_col, value_col):
        if col not in frame.columns:
            raise ValueError(f"frame has no {col!r} column")
    df = frame[frame[family_col].isin([family_a, family_b])]
    df = df[np.isfinite(pd.to_numeric(df[value_col], errors="coerce"))]
    # A cell only one family reaches defines no contrast; drop it before anything is relabelled.
    per_cell = df.groupby(stratum_col)[family_col].nunique()
    keep = set(per_cell[per_cell == 2].index)
    n_cells_dropped = int(len(per_cell) - len(keep))
    df = df[df[stratum_col].isin(keep)]
    if df.empty:
        raise ValueError("no cell carries runs from both families")

    sums, counts, arms = _cell_tables(df, arm_col, stratum_col, value_col, statistic)
    families = df.drop_duplicates(arm_col).set_index(arm_col)[family_col]
    is_a = np.array([families[a] == family_a for a in arms])
    if not is_a.any() or is_a.all():
        raise ValueError("both families need at least one arm")
    memo_key = memo.key("arm_relabel_cell_permutation", sums, counts, is_a, statistic,
                        int(max_exact), int(n_perm), int(seed), int(n_cells_dropped))
    return memo.memoized("arm_relabel_cell_permutation", memo_key, ArmRelabelResult,
                         lambda: _arm_relabel_cell_permutation(
                             sums, counts, is_a, statistic, max_exact, n_perm, seed,
                             n_cells_dropped))


def _arm_relabel_cell_permutation(sums, counts, is_a, statistic, max_exact, n_perm, seed,
                                  n_cells_dropped) -> ArmRelabelResult:
    """The enumeration / draws of `arm_relabel_cell_permutation` on its per-cell tables."""
    from math import comb

    n_arms = sums.shape[0]
    n_a, n_b = int(is_a.sum()), int((~is_a).sum())
    n_cells = sums.shape[1]
    total_sums, total_counts = sums.sum(axis=0), counts.sum(axis=0)

    def stat(picks: np.ndarray) -> np.ndarray:
        """Statistic per labelling; ``picks`` holds each labelling's A-arm indices."""
        a_sum, a_n = sums[picks].sum(axis=1), counts[picks].sum(axis=1)  # (draws, cells)
        b_sum, b_n = total_sums[None, :] - a_sum, total_counts[None, :] - a_n
        valid = (a_n > 0) & (b_n > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            if statistic == "mean_contrast":
                per_cell_stat = a_sum / a_n - b_sum / b_n
            else:
                # Mann-Whitney U from the A midrank sum; U / (n_A n_B) counts ties as 1/2.
                per_cell_stat = (a_sum - a_n * (a_n + 1) / 2) / (a_n * b_n)
        per_cell_stat = np.where(valid, per_cell_stat, 0.0)
        n_valid = valid.sum(axis=1)
        return np.where(n_valid > 0, per_cell_stat.sum(axis=1) / np.maximum(n_valid, 1), np.nan)

    observed = float(stat(np.flatnonzero(is_a)[None, :])[0])
    n_labelings = comb(n_arms, n_a)
    exact = n_labelings <= max_exact
    # Chunk the labelings so the (draws x cells) tables stay bounded in memory.
    chunk = max(1, int(2e7) // max(1, n_cells))
    if exact:
        nulls, combos = [], itertools.combinations(range(n_arms), n_a)
        while True:
            block = list(itertools.islice(combos, chunk))
            if not block:
                break
            nulls.append(stat(np.array(block, dtype=int)))
        null = np.concatenate(nulls)
        p, floor, n_draws = float((null >= observed).mean()), 1.0 / n_labelings, n_labelings
    else:
        rng = np.random.default_rng(seed)
        nulls = []
        for start in range(0, n_perm, chunk):
            draws = min(chunk, n_perm - start)
            picks = np.argsort(rng.random((draws, n_arms)), axis=1)[:, :n_a]
            nulls.append(stat(picks))
        null = np.concatenate(nulls)
        p = float((int((null >= observed).sum()) + 1) / (n_perm + 1))
        floor, n_draws = 1.0 / (n_perm + 1), int(n_perm)

    quantiles = {
        "50": float(np.percentile(null, 50)),
        "95": float(np.percentile(null, 95)),
        "99": float(np.percentile(null, 99)),
        "99.9": float(np.percentile(null, 99.9)),
        "max": float(null.max()),
    }
    return ArmRelabelResult(
        observed=observed, p=float(p), floor=float(floor), null_quantiles=quantiles,
        n_labelings=int(n_draws), exact=bool(exact), n_a=n_a, n_b=n_b, statistic=statistic,
        n_cells=int(n_cells), n_cells_dropped=n_cells_dropped,
    )






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
