"""Small statistics toolkit shared by the analysis notebooks in ``notebooks/``.

This module consolidates the bootstrap / pooling / AUC / ratio-CI code that was copy-pasted
across the six analysis notebooks (and lightly mutated each time, so no two copies agreed on
edge cases). Everything here is a pure function over numpy arrays or pandas frames -- no disk
IO, no matplotlib -- so it stays a safe leaf of the ``scripts/`` -> ``pipelines/`` ->
``src/subliminal_learning/`` layering.

Two conventions the notebooks keep re-learning the hard way, pinned here once:

- **Cell clustering.** Seeds within a cell (``student_tag``) are correlated, so resampling
  individual rows manufactures significance. `bootstrap_ci` (via ``cluster=``), `cluster_ols`
  and `cluster_logit` all resample / cover on whole cells instead. Quote the cluster-robust
  p-value, never the naive one.
- **Min-n masking.** Collapse-fraction-vs-``K/K*`` curves are noise below a handful of seeds
  per bin; `binned_fraction` blanks (``NaN``) any bin with ``n < min_n`` rather than plotting a
  spike built from two seeds -- it keeps every bin (so the blanked ones render as gaps) and counts
  only finite rows.

All randomness flows through ``np.random.default_rng(seed)`` with an explicit ``seed`` argument,
so every result is reproducible from its call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a percentile bootstrap.

    Attributes:
        estimate: `stat` evaluated on the full (unresampled) data.
        lo: Lower percentile bound of the CI.
        hi: Upper percentile bound of the CI.
        draws: The finite replicate values (length ``n_effective``).
        n_effective: Replicates that produced a finite `stat` (raised/non-finite draws dropped).
    """

    estimate: float
    lo: float
    hi: float
    draws: np.ndarray
    n_effective: int


@dataclass(frozen=True)
class PooledResult:
    """Result of an inverse-variance fixed-effect pooling with Cochran homogeneity.

    Attributes:
        mu: Pooled point estimate.
        se: Standard error of `mu`.
        lo: Lower bound of the ``z``-interval.
        hi: Upper bound of the ``z``-interval.
        q: Cochran's Q homogeneity statistic.
        df: Degrees of freedom (``k - 1``).
        p_q: Chi-squared survival p-value for `q` on `df`.
        i2: Higgins I^2 in percent, floored at 0.
    """

    mu: float
    se: float
    lo: float
    hi: float
    q: float
    df: int
    p_q: float
    i2: float


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF of a 1-D sample.

    Args:
        values: Sequence of numbers; ``NaN`` (and non-finite) entries are dropped.

    Returns:
        ``(x, y)`` where ``x`` is the sorted finite values and ``y`` the cumulative fractions
        ``(1..n)/n``. Both are empty arrays when no finite values remain.

    Not delegated to ``scipy.stats.ecdf`` on purpose: scipy's step-function CDF deduplicates,
    so ``ecdf(v).cdf.evaluate(np.sort(v))`` returns the step height at each tie (e.g.
    ``[.25, .75, .75, 1]`` for ``[1, 2, 2, 3]``) rather than this contract's per-point
    ``(1..n)/n`` (``[.25, .5, .75, 1]``). Reproducing the contract exactly would mean discarding
    scipy's output and re-deriving the ``arange``, so the vetted-library win is nil -- and the
    sort/arange here is not the error-prone kind of numerics the rest of this module delegates.
    """
    v = np.asarray(values, dtype=float)
    v = np.sort(v[np.isfinite(v)])
    n = v.size
    y = np.arange(1, n + 1, dtype=float) / n if n else np.array([])
    return v, y


def mannwhitney_auc(pos, neg) -> float:
    """Rank-based AUC ``P(pos > neg)`` with ties counted as half.

    Equivalent to sklearn's ``roc_auc_score`` with labels 1 for `pos` and 0 for `neg`, but
    computed from Mann-Whitney U so no sklearn import is needed here.

    Args:
        pos: Scores of the positive class; ``NaN`` entries dropped.
        neg: Scores of the negative class; ``NaN`` entries dropped.

    Returns:
        The AUC in ``[0, 1]``, or ``np.nan`` if either side is empty after dropping ``NaN``.
    """
    from scipy.stats import mannwhitneyu

    a = np.asarray(pos, dtype=float)
    b = np.asarray(neg, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (a.size * b.size))


def bootstrap_ci(
    df_or_values,
    stat,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 95,
    cluster: str | None = None,
) -> BootstrapResult:
    """Percentile bootstrap CI for an arbitrary statistic.

    Args:
        df_or_values: A ``pd.DataFrame`` (resampled by row, or by cluster) or a 1-D array-like
            (resampled by element).
        stat: Callable applied to each resampled object (a DataFrame or an array, matching the
            input type) and to the full data for the point estimate. Replicates where `stat`
            raises or returns a non-finite value are dropped.
        n_boot: Number of bootstrap replicates.
        seed: Seed for ``np.random.default_rng``.
        ci: Central interval width in percent (e.g. 95 -> 2.5/97.5 percentiles).
        cluster: Optional column name (DataFrame input only). Whole clusters are resampled with
            replacement -- the cell-clustered convention: seeds within a cell are correlated, so
            independent-row resampling manufactures significance.

    Returns:
        A `BootstrapResult`; ``lo``/``hi`` are ``np.nan`` when no replicate was finite.
    """
    rng = np.random.default_rng(seed)
    is_df = isinstance(df_or_values, pd.DataFrame)
    if cluster is not None and not is_df:
        raise ValueError("cluster= requires a DataFrame input")

    if is_df:
        data = df_or_values.reset_index(drop=True)
        if cluster is not None:
            groups = [np.asarray(v) for v in data.groupby(cluster).indices.values()]
            n_units = len(groups)

            def resample():
                pick = rng.integers(0, n_units, n_units)
                rows = np.concatenate([groups[i] for i in pick])
                return data.iloc[rows]
        else:
            n = len(data)

            def resample():
                return data.iloc[rng.integers(0, n, n)]

        full = data
    else:
        arr = np.asarray(df_or_values)
        n = arr.size

        def resample():
            return arr[rng.integers(0, n, n)]

        full = arr

    estimate = float(stat(full))
    draws: list[float] = []
    for _ in range(n_boot):
        try:
            val = float(stat(resample()))
        except Exception:
            continue
        if np.isfinite(val):
            draws.append(val)

    arr_draws = np.asarray(draws, dtype=float)
    lo_p, hi_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    if arr_draws.size:
        lo, hi = float(np.percentile(arr_draws, lo_p)), float(np.percentile(arr_draws, hi_p))
    else:
        lo = hi = float("nan")
    return BootstrapResult(estimate, lo, hi, arr_draws, arr_draws.size)


def pool_fixed_effect(
    estimates,
    *,
    se=None,
    lo=None,
    hi=None,
    z: float = 1.96,
) -> PooledResult:
    """Inverse-variance fixed-effect pooling with Cochran homogeneity.

    This is the per-lineage exponent pooling plus Cochran homogeneity block from the collapse
    notebooks. Supply the study-level standard errors directly (`se`) or as a symmetric interval
    (`lo`, `hi`), in which case ``se = (hi - lo) / (2z)``. The pooling itself is delegated to
    ``statsmodels.stats.meta_analysis.combine_effects`` (variance ``= se**2``); this maps its
    fixed-effect estimate, sd, Cochran Q and I^2 into `PooledResult`.

    Args:
        estimates: Per-study point estimates.
        se: Per-study standard errors. Provide this or (`lo`, `hi`).
        lo: Per-study interval lower bounds (used only when `se` is None).
        hi: Per-study interval upper bounds (used only when `se` is None).
        z: Normal quantile the input intervals and output interval use (1.96 -> 95%).

    Returns:
        A `PooledResult`.
    """
    import warnings

    from scipy.stats import chi2
    from statsmodels.stats.meta_analysis import combine_effects

    est = np.asarray(estimates, dtype=float)
    if se is None:
        if lo is None or hi is None:
            raise ValueError("provide se, or both lo and hi")
        se_arr = (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)) / (2 * z)
    else:
        se_arr = np.asarray(se, dtype=float)

    with warnings.catch_warnings():
        # combine_effects warns on the degenerate q == 0 / df == 0 cases we normalise below.
        warnings.simplefilter("ignore", RuntimeWarning)
        res = combine_effects(est, se_arr**2)

    mu = float(res.mean_effect_fe)
    se_mu = float(res.sd_eff_w_fe)
    q = float(res.q)
    df = int(est.size - 1)
    # combine_effects reports I^2 as a fraction (and -inf / nan in the degenerate q<=0/df==0
    # cases); keep the historical contract: percent, floored at 0, exactly 0 when undefined.
    p_q = float(chi2.sf(q, df)) if df > 0 else float("nan")
    i2 = max(0.0, 100.0 * float(res.i2)) if q > 0 and df > 0 else 0.0
    return PooledResult(mu, se_mu, mu - z * se_mu, mu + z * se_mu, q, df, p_q, i2)


def delta_ratio_ci(
    params,
    cov,
    num: int,
    den: int,
    *,
    z: float = 1.96,
) -> tuple[float, tuple[float, float]]:
    """Delta-method estimate and CI for the ratio ``params[num] / params[den]``.

    The gradient of ``g(p) = p[num] / p[den]`` is ``g[num] = 1/p[den]`` and
    ``g[den] = -p[num] / p[den]**2`` (all other entries zero); the variance is ``g @ cov @ g``.
    This is the exact math inside `collapse_law.fit_collapse_law`'s ratio intervals.

    Args:
        params: Parameter vector.
        cov: Covariance matrix of `params`.
        num: Index of the numerator parameter.
        den: Index of the denominator parameter.
        z: Normal quantile for the interval half-width (1.96 -> 95%).

    Returns:
        ``(estimate, (lo, hi))``.
    """
    p = np.asarray(params, dtype=float)
    V = np.asarray(cov, dtype=float)
    est = p[num] / p[den]
    g = np.zeros(p.size)
    g[num] = 1.0 / p[den]
    g[den] = -p[num] / p[den] ** 2
    sd = float(np.sqrt(max(g @ V @ g, 0.0)))
    return float(est), (float(est - z * sd), float(est + z * sd))


def cluster_ols(df: pd.DataFrame, formula: str, cluster: str):
    """Cluster-robust OLS fit via statsmodels.

    Pins the convention: quote these cluster-robust p-values, never the naive homoskedastic
    ones, because seeds within a cell are correlated.

    Args:
        df: Data frame with the formula terms and the `cluster` column.
        formula: Patsy formula, e.g. ``"y ~ x"``.
        cluster: Column name to group the cluster-robust covariance on.

    Returns:
        The fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})


def cluster_logit(df: pd.DataFrame, formula: str, cluster: str):
    """Cluster-robust logistic fit via statsmodels.

    The logit counterpart to `cluster_ols`; same convention (quote the cluster-robust p-values).

    Args:
        df: Data frame with the formula terms and the `cluster` column.
        formula: Patsy formula, e.g. ``"collapsed ~ x"``.
        cluster: Column name to group the cluster-robust covariance on.

    Returns:
        The fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    return smf.logit(formula, df).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df[cluster]}
    )


def binned_fraction(
    df: pd.DataFrame,
    x: str,
    flag: str,
    *,
    bins,
    min_n: int = 5,
) -> pd.DataFrame:
    """Fraction of a boolean `flag` in bins of `x` (the collapse-fraction-vs-K/K* curve).

    Two conventions the plotted curve depends on:

    - **All bins kept.** Every bin between consecutive edges gets a row -- empty and sparse bins
      included -- so a curve built from this frame renders under-populated bins as *gaps* (their
      ``NaN`` ``frac`` breaks the line) rather than silently closing over them.
    - **Non-finite rows dropped.** A row counts toward a bin only when both its `x` and `flag`
      are finite; ``NaN``/``inf`` in either is excluded from `n` and the fraction.

    Args:
        df: Rows to bin.
        x: Column to bin on.
        flag: Boolean/0-1 column whose per-bin mean is the reported fraction.
        bins: Monotone array of bin edges (``len(bins) - 1`` bins).
        min_n: Bins with fewer than this many rows report ``frac = NaN`` (min-n mask convention;
            ``n == 0`` is subsumed).

    Returns:
        One row per bin, columns ``x_mid`` (geometric mean of the bin's two edges when both are
        positive, else arithmetic midpoint), ``frac`` (mean of `flag`, ``NaN`` when ``n < min_n``)
        and ``n`` (finite-row count). Ordered by increasing bin.

    The per-bin count/mean reduce is delegated to ``scipy.stats.binned_statistic``. Its bins are
    left-closed except the last, which is *both* edges closed; ``np.digitize`` (this function's
    historical semantics) instead leaves a point exactly on the final edge *out* of every bin, so
    such points are dropped from the finite subset before binning to preserve that contract.
    """
    from scipy.stats import binned_statistic

    xv = df[x].to_numpy(dtype=float)
    fv = df[flag].to_numpy(dtype=float)
    edges = np.asarray(bins, dtype=float)
    # Exclude non-finite rows and points exactly on the last edge (digitize leaves those out of
    # every bin; binned_statistic would fold them into the last bin).
    finite = np.isfinite(xv) & np.isfinite(fv) & (xv != edges[-1])
    xf, ff = xv[finite], fv[finite]

    counts = binned_statistic(xf, ff, statistic="count", bins=edges).statistic
    means = binned_statistic(xf, ff, statistic="mean", bins=edges).statistic

    rows = []
    for b in range(len(edges) - 1):
        n = int(counts[b])
        e0, e1 = edges[b], edges[b + 1]
        x_mid = float(np.sqrt(e0 * e1)) if e0 > 0 and e1 > 0 else float((e0 + e1) / 2)
        frac = float(means[b]) if n >= min_n else float("nan")
        rows.append({"x_mid": x_mid, "frac": frac, "n": n})
    return pd.DataFrame(rows, columns=["x_mid", "frac", "n"])


def sign_binom_test(deltas, *, alternative: str = "greater"):
    """Exact sign test on paired differences.

    Drops ``NaN`` and exact zeros, then tests the number of positive differences against a
    fair-coin null ``p = 0.5``.

    Args:
        deltas: Paired differences.
        alternative: Passed to ``scipy.stats.binomtest`` (``"greater"``/``"less"``/
            ``"two-sided"``).

    Returns:
        The ``scipy.stats.BinomTestResult``.
    """
    from scipy.stats import binomtest

    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    n_pos = int((d > 0).sum())
    return binomtest(n_pos, d.size, 0.5, alternative=alternative)
