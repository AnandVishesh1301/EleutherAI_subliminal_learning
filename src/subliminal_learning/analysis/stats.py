"""Stochasticity and robustness machinery for campaign claim fitting.

Pure numpy / scipy / statsmodels -- no ``torch``, so this joins the torch-free
domain set in ``tests/unit/test_layering.py`` and is reusable by every campaign
rather than one of them.

The resampling functions take a `statistic` callable over ROW INDICES into the
caller's own arrays, so one implementation serves means, slopes, contrasts and
whatever a later claim needs without this module knowing the schema.

Sources:

- Two-way (cross-classified) cluster resampling: points within a rung share
  teachers and controls, and points within a pairing share a student base, so
  both dimensions are resampled.
- `multi_bootstrap`: Sellam et al., arXiv:2106.16163 §3 -- draw the seed
  population and the observations with replacement, separately.
- `tost_equivalence`: two one-sided tests, following the C4 equivalence-bound
  precedent in ``docs/PREREG_OPTPARAM.md`` (a bare non-significant result never
  counts as a null).
- `family_interaction_fit`: the C1 regression
  ``y ~ d_trait + family + d_trait*family`` with cluster-robust covariance
  (statsmodels ``cov_type="cluster"``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy import stats

N_BOOT = 10000
BOOTSTRAP_SEED = 20260805

# Data-seed is the reference family (coded 0), so the interaction term is
# weight-minus-data.
FAMILY_REFERENCE = "data"
FAMILY_CONTRAST = "weight"


@dataclass(frozen=True)
class BootstrapEstimate:
    """A point estimate with a percentile bootstrap interval.

    Attributes:
        estimate: The statistic on the observed rows.
        ci_low: Lower percentile bound.
        ci_high: Upper percentile bound.
        p_boot: Two-sided bootstrap p-value against 0.
        n_valid: Bootstrap replicates that produced a finite statistic.
        n_boot: Replicates requested.
        confidence: Two-sided interval level.
    """

    estimate: float
    ci_low: float
    ci_high: float
    p_boot: float
    n_valid: int
    n_boot: int
    confidence: float


@dataclass(frozen=True)
class VarianceComponents:
    """Sequential-SS variance decomposition.

    Attributes:
        fractions: Factor name -> share of the total sum of squares, plus
            ``"residual"``. Shares sum to 1.
        sums_of_squares: The same entries as raw sums of squares.
        total: Total sum of squares about the mean.
        n_obs: Rows fitted.
        order: The entry order the sequential fit used.
    """

    fractions: Mapping[str, float]
    sums_of_squares: Mapping[str, float]
    total: float
    n_obs: int
    order: tuple[str, ...]


@dataclass(frozen=True)
class TostResult:
    """Two one-sided tests against an equivalence interval.

    Attributes:
        estimate: Sample mean.
        std_error: Standard error of the mean.
        ci_low: Lower bound of the ``1 - 2*alpha`` interval.
        ci_high: Upper bound of the ``1 - 2*alpha`` interval.
        p_lower: One-sided p against ``H0: mean <= bound_low``.
        p_upper: One-sided p against ``H0: mean >= bound_high``.
        p_value: ``max(p_lower, p_upper)`` -- the TOST p-value.
        equivalent: Whether both one-sided tests reject at `alpha`.
        bound_low: Lower equivalence bound.
        bound_high: Upper equivalence bound.
        n_obs: Sample size.
    """

    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_lower: float
    p_upper: float
    p_value: float
    equivalent: bool
    bound_low: float
    bound_high: float
    n_obs: int


@dataclass(frozen=True)
class EquivalenceResult:
    """A percentile bootstrap interval checked against an equivalence interval.

    Attributes:
        estimate: The bootstrap estimate being checked.
        bound_low: Lower equivalence bound.
        bound_high: Upper equivalence bound.
        equivalent: ``bound_low < ci_low`` and ``ci_high < bound_high``.
    """

    estimate: BootstrapEstimate
    bound_low: float
    bound_high: float
    equivalent: bool


@dataclass(frozen=True)
class Coefficient:
    """One regression coefficient with its cluster-robust interval."""

    name: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class InteractionFit:
    """The C1 fit ``y ~ d_trait + family + d_trait*family``.

    Attributes:
        intercept: ``beta0``.
        d_trait: ``beta1``, the divergence slope in the reference family.
        family: ``beta2``, the family offset at ``d_trait = 0``.
        interaction: ``beta3``, the slope difference (weight minus data).
        n_obs: Rows fitted.
        n_clusters: Distinct clusters behind the robust covariance.
        confidence: Two-sided interval level.
    """

    intercept: Coefficient
    d_trait: Coefficient
    family: Coefficient
    interaction: Coefficient
    n_obs: int
    n_clusters: int
    confidence: float


def _finish(
    observed: float,
    draws: np.ndarray,
    *,
    confidence: float,
    n_boot: int,
) -> BootstrapEstimate:
    """Percentile interval and two-sided bootstrap p from finite replicates."""
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ValueError("every bootstrap replicate was degenerate")
    tail = 100.0 * (1.0 - confidence) / 2.0
    p_boot = 2.0 * min(float(np.mean(finite <= 0.0)), float(np.mean(finite >= 0.0)))
    return BootstrapEstimate(
        estimate=observed,
        ci_low=float(np.percentile(finite, tail)),
        ci_high=float(np.percentile(finite, 100.0 - tail)),
        p_boot=min(p_boot, 1.0),
        n_valid=int(finite.size),
        n_boot=n_boot,
        confidence=confidence,
    )


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], float],
    rung: Sequence[object],
    pairing: Sequence[object],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> BootstrapEstimate:
    """Two-way cluster bootstrap over rungs AND pairings.

    Each replicate draws whole rungs and whole pairings with replacement and
    keeps their intersection cells with the product multiplicity, so a rung
    drawn twice contributes twice to every pairing it is crossed with. One-way
    resampling would understate the dependence on whichever dimension was held
    fixed.

    Args:
        statistic: Maps an integer row-index array to a scalar.
        rung: Per-row rung label.
        pairing: Per-row pairing label.
        n_boot: Replicates.
        seed: Seeds ``numpy.random.default_rng``.
        confidence: Two-sided percentile level.

    Returns:
        The `BootstrapEstimate`.

    Raises:
        ValueError: On mismatched or empty label arrays, a bad `confidence`, or
            replicates that are all degenerate.
    """
    _check_confidence(confidence)
    rungs = np.asarray(rung)
    pairings = np.asarray(pairing)
    if rungs.ndim != 1 or rungs.shape != pairings.shape:
        raise ValueError(
            f"rung/pairing must be matching 1-D labels; got {rungs.shape}, "
            f"{pairings.shape}"
        )
    if rungs.size == 0:
        raise ValueError("no rows to resample")

    rung_levels = np.unique(rungs)
    pairing_levels = np.unique(pairings)
    cells = {}
    for i, r in enumerate(rung_levels):
        for j, p in enumerate(pairing_levels):
            rows = np.flatnonzero((rungs == r) & (pairings == p))
            if rows.size:
                cells[(i, j)] = rows

    observed = float(statistic(np.arange(rungs.size)))
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        drawn_rungs = rng.integers(0, rung_levels.size, rung_levels.size)
        drawn_pairings = rng.integers(0, pairing_levels.size, pairing_levels.size)
        picked = [
            cells[(i, j)]
            for i in drawn_rungs
            for j in drawn_pairings
            if (i, j) in cells
        ]
        if picked:
            draws[b] = statistic(np.concatenate(picked))
    return _finish(observed, draws, confidence=confidence, n_boot=n_boot)


def multi_bootstrap(
    statistic: Callable[[np.ndarray], float],
    seed_group: Sequence[object],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> BootstrapEstimate:
    """Sellam et al.'s Multi-Bootstrap over the seed population and the rows.

    Each replicate draws the training-seed groups with replacement and then, on
    the rows those groups contribute, draws observations with replacement --
    the two sources of randomness the rotated diagonal samples jointly
    (arXiv:2106.16163 §3). Resampling rows alone would treat the sampled
    ``(ftseed, gseed)`` combinations as the whole population.

    Args:
        statistic: Maps an integer row-index array to a scalar.
        seed_group: Per-row joint seed label, e.g. ``(ftseed, gseed)``.
        n_boot: Replicates.
        seed: Seeds ``numpy.random.default_rng``.
        confidence: Two-sided percentile level.

    Returns:
        The `BootstrapEstimate`.

    Raises:
        ValueError: On an empty or non-1-D `seed_group`, a bad `confidence`, or
            replicates that are all degenerate.
    """
    _check_confidence(confidence)
    groups = np.asarray(seed_group)
    if groups.ndim != 1:
        raise ValueError(f"seed_group must be 1-D; got {groups.shape}")
    if groups.size == 0:
        raise ValueError("no rows to resample")

    levels = np.unique(groups)
    rows_by_group = [np.flatnonzero(groups == level) for level in levels]
    observed = float(statistic(np.arange(groups.size)))
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        drawn = rng.integers(0, levels.size, levels.size)
        rows = np.concatenate([rows_by_group[i] for i in drawn])
        resampled = rows[rng.integers(0, rows.size, rows.size)]
        draws[b] = statistic(resampled)
    return _finish(observed, draws, confidence=confidence, n_boot=n_boot)


def stratified_cluster_bootstrap_draws(
    statistic: Callable[[np.ndarray], ArrayLike],
    strata: Sequence[object],
    clusters: Sequence[object],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicates of a vector statistic under cluster resampling within strata.

    Each replicate draws, independently per stratum, as many clusters as the
    stratum holds, with replacement, and passes the rows of the drawn clusters
    (with multiplicity) to `statistic`. Stratum sizes are fixed, so a stratum
    never vanishes from a replicate.

    Args:
        statistic: Maps an integer row-index array to a scalar or ``[k]``
            vector; non-finite entries mark degenerate replicates.
        strata: Per-row stratum label.
        clusters: Per-row cluster label; a cluster must sit in one stratum.
        n_boot: Replicates.
        seed: Seeds ``numpy.random.default_rng``.

    Returns:
        ``(observed [k], draws [n_boot, k])``.

    Raises:
        ValueError: On mismatched or empty labels, or a cluster spanning strata.
    """
    strata_arr = np.asarray(strata)
    cluster_arr = np.asarray(clusters)
    if strata_arr.ndim != 1 or strata_arr.shape != cluster_arr.shape:
        raise ValueError(
            f"strata/clusters must be matching 1-D labels; got {strata_arr.shape}, "
            f"{cluster_arr.shape}"
        )
    if strata_arr.size == 0:
        raise ValueError("no rows to resample")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")

    cluster_levels, cluster_codes = np.unique(cluster_arr, return_inverse=True)
    rows_by_cluster = [
        np.flatnonzero(cluster_codes == c) for c in range(cluster_levels.size)
    ]
    stratum_of_cluster: dict[int, object] = {}
    for code in range(cluster_levels.size):
        owners = np.unique(strata_arr[rows_by_cluster[code]])
        if owners.size != 1:
            raise ValueError(
                f"cluster {cluster_levels[code]!r} spans strata {owners.tolist()}"
            )
        stratum_of_cluster[code] = owners[0]
    by_stratum: dict[object, list[int]] = {}
    for code, stratum in stratum_of_cluster.items():
        by_stratum.setdefault(stratum, []).append(code)
    groups = [np.asarray(by_stratum[s]) for s in sorted(by_stratum, key=str)]

    observed = np.atleast_1d(
        np.asarray(statistic(np.arange(strata_arr.size)), dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, observed.size), np.nan)
    for b in range(n_boot):
        picked = [
            rows_by_cluster[c]
            for group in groups
            for c in group[rng.integers(0, group.size, group.size)]
        ]
        draws[b] = np.asarray(statistic(np.concatenate(picked)), dtype=np.float64)
    return observed, draws


def summarize_draws(
    observed: float,
    draws: ArrayLike,
    *,
    confidence: float = 0.95,
) -> BootstrapEstimate:
    """Percentile interval over one column of bootstrap replicates.

    Args:
        observed: The statistic on the observed rows.
        draws: ``[n_boot]`` replicates; non-finite ones are dropped.
        confidence: Two-sided percentile level.

    Returns:
        The `BootstrapEstimate`.

    Raises:
        ValueError: On a bad `confidence`, non-1-D draws, or no finite draw.
    """
    _check_confidence(confidence)
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"draws must be 1-D; got {values.shape}")
    return _finish(float(observed), values, confidence=confidence, n_boot=values.size)


def bootstrap_equivalence(
    estimate: BootstrapEstimate,
    *,
    bound_low: float,
    bound_high: float,
) -> EquivalenceResult:
    """Whether a bootstrap interval lies strictly inside equivalence bounds.

    With a ``1 - 2*alpha`` percentile interval this is the percentile-bootstrap
    form of `tost_equivalence` at one-sided level ``alpha``.

    Args:
        estimate: A `BootstrapEstimate` (typically at confidence 0.90).
        bound_low: Lower equivalence bound.
        bound_high: Upper equivalence bound, strictly above `bound_low`.

    Returns:
        The `EquivalenceResult`.

    Raises:
        ValueError: On non-ordered bounds.
    """
    if not bound_low < bound_high:
        raise ValueError(f"need bound_low < bound_high; got {bound_low}, {bound_high}")
    return EquivalenceResult(
        estimate=estimate,
        bound_low=bound_low,
        bound_high=bound_high,
        equivalent=bound_low < estimate.ci_low and estimate.ci_high < bound_high,
    )


def _dummies(labels: np.ndarray) -> np.ndarray:
    """``[n, len(levels) - 1]`` drop-first indicator columns for one factor."""
    levels = np.unique(labels)
    if levels.size < 2:
        return np.zeros((labels.size, 0))
    return np.column_stack(
        [(labels == level).astype(np.float64) for level in levels[1:]]
    )


def _residual_sum_of_squares(design: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(residual @ residual)


def variance_components(
    values: ArrayLike,
    factors: Mapping[str, Sequence[object]],
    *,
    order: Sequence[str] | None = None,
) -> VarianceComponents:
    """Sequential (Type I) sum-of-squares decomposition over categorical factors.

    Factors are entered in `order` and each one is credited the drop in residual
    sum of squares it produces on top of the factors already entered. The
    decomposition is order-dependent by construction, so `order` is recorded on
    the result; the checkpoint ladder fits it only on the crossed anchor rungs,
    where the seed factors are unaliased.

    Args:
        values: ``[n]`` outcome.
        factors: Factor name -> ``[n]`` categorical labels.
        order: Entry order; defaults to the mapping's own key order.

    Returns:
        The `VarianceComponents`.

    Raises:
        ValueError: On an empty `factors`, a length mismatch, an `order` that is
            not a permutation of the factor names, or a constant outcome.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"values must be 1-D; got {y.shape}")
    if not factors:
        raise ValueError("at least one factor is required")
    entry = tuple(factors) if order is None else tuple(order)
    if sorted(entry) != sorted(factors):
        raise ValueError(f"order {entry} is not a permutation of {tuple(factors)}")

    columns = {}
    for name in entry:
        labels = np.asarray(factors[name])
        if labels.shape != y.shape:
            raise ValueError(
                f"factor {name!r} has {labels.shape} labels for {y.shape} values"
            )
        columns[name] = _dummies(labels)

    centred = y - y.mean()
    total = float(centred @ centred)
    # Relative guard: a constant outcome leaves lstsq rounding behind rather than
    # an exact zero, and dividing shares by that noise would fabricate a
    # decomposition out of float error.
    if total <= np.finfo(np.float64).eps * float(y @ y):
        raise ValueError("values are constant to float precision; nothing to decompose")

    design = np.ones((y.size, 1))
    previous = total
    sums = {}
    for name in entry:
        design = np.column_stack([design, columns[name]])
        current = _residual_sum_of_squares(design, y)
        sums[name] = previous - current
        previous = current
    sums["residual"] = previous

    return VarianceComponents(
        fractions={k: v / total for k, v in sums.items()},
        sums_of_squares=sums,
        total=total,
        n_obs=int(y.size),
        order=entry,
    )


def tost_equivalence(
    values: ArrayLike,
    *,
    bound_low: float,
    bound_high: float,
    alpha: float = 0.05,
) -> TostResult:
    """Two one-sided tests that a sample mean lies inside an equivalence interval.

    Equivalence is declared only when BOTH one-sided nulls are rejected, which
    is identical to the ``1 - 2*alpha`` interval lying strictly inside the
    bounds. This is the C4 discipline from ``docs/PREREG_OPTPARAM.md``: a
    non-significant difference is not a null unless it is bounded.

    Args:
        values: ``[n]`` sample.
        bound_low: Lower equivalence bound.
        bound_high: Upper equivalence bound, strictly above `bound_low`.
        alpha: One-sided level for each test.

    Returns:
        The `TostResult`.

    Raises:
        ValueError: On fewer than 2 values, non-ordered bounds, a bad `alpha`,
            or a zero-variance sample (the t statistic is undefined).
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError(f"values must be 1-D with n >= 2; got {sample.shape}")
    if not bound_low < bound_high:
        raise ValueError(f"need bound_low < bound_high; got {bound_low}, {bound_high}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5); got {alpha}")

    n = sample.size
    df = n - 1
    mean = float(sample.mean())
    std_error = float(sample.std(ddof=1) / np.sqrt(n))
    if std_error <= 0.0:
        raise ValueError("zero-variance sample; the TOST t statistic is undefined")

    p_lower = float(stats.t.sf((mean - bound_low) / std_error, df))
    p_upper = float(stats.t.cdf((mean - bound_high) / std_error, df))
    half_width = float(stats.t.ppf(1.0 - alpha, df)) * std_error
    p_value = max(p_lower, p_upper)
    return TostResult(
        estimate=mean,
        std_error=std_error,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=p_value,
        equivalent=p_value < alpha,
        bound_low=bound_low,
        bound_high=bound_high,
        n_obs=int(n),
    )


def _family_codes(family: Sequence[object]) -> np.ndarray:
    """0 for `FAMILY_REFERENCE`, 1 for `FAMILY_CONTRAST`; anything else raises."""
    labels = np.asarray(family)
    codes = np.full(labels.shape, -1.0)
    codes[labels == FAMILY_REFERENCE] = 0.0
    codes[labels == FAMILY_CONTRAST] = 1.0
    if (codes < 0).any():
        unknown = sorted({str(v) for v in labels[codes < 0]})
        raise ValueError(
            f"family must be {FAMILY_REFERENCE!r} or {FAMILY_CONTRAST!r}; got {unknown}"
        )
    return codes


def family_interaction_fit(
    y: ArrayLike,
    d_trait: ArrayLike,
    family: Sequence[object],
    cluster: Sequence[object],
    *,
    confidence: float = 0.95,
) -> InteractionFit:
    """Fit ``y ~ d_trait + family + d_trait*family`` with cluster-robust CIs.

    `FAMILY_REFERENCE` is coded 0, so ``interaction`` is the weight-family slope
    minus the data-family slope -- the C1 coefficient whose position against the
    preregistered equivalence bound separates H1 from H2.

    Args:
        y: ``[n]`` transfer gap.
        d_trait: ``[n]`` divergence covariate.
        family: ``[n]`` labels, each `FAMILY_REFERENCE` or `FAMILY_CONTRAST`.
        cluster: ``[n]`` cluster ids for the robust covariance.
        confidence: Two-sided interval level.

    Returns:
        The `InteractionFit`.

    Raises:
        ValueError: On a length mismatch, an unknown family label, a bad
            `confidence`, or fewer than two clusters.
    """
    _check_confidence(confidence)
    outcome = np.asarray(y, dtype=np.float64)
    divergence = np.asarray(d_trait, dtype=np.float64)
    codes = _family_codes(family)
    groups = np.asarray(cluster)
    shapes = {outcome.shape, divergence.shape, codes.shape, groups.shape}
    if outcome.ndim != 1 or len(shapes) != 1:
        raise ValueError(f"y/d_trait/family/cluster must be matching 1-D; got {shapes}")
    _, group_codes = np.unique(groups, return_inverse=True)
    n_clusters = int(group_codes.max()) + 1
    if n_clusters < 2:
        raise ValueError(
            f"cluster-robust covariance needs >= 2 clusters; got {n_clusters}"
        )

    design = np.column_stack(
        [np.ones_like(outcome), divergence, codes, divergence * codes]
    )
    fit = sm.OLS(outcome, design).fit(
        cov_type="cluster", cov_kwds={"groups": group_codes}
    )
    intervals = np.asarray(fit.conf_int(alpha=1.0 - confidence))

    def coefficient(index: int, name: str) -> Coefficient:
        return Coefficient(
            name=name,
            estimate=float(fit.params[index]),
            std_error=float(fit.bse[index]),
            ci_low=float(intervals[index, 0]),
            ci_high=float(intervals[index, 1]),
        )

    return InteractionFit(
        intercept=coefficient(0, "intercept"),
        d_trait=coefficient(1, "d_trait"),
        family=coefficient(2, "family"),
        interaction=coefficient(3, "interaction"),
        n_obs=int(outcome.size),
        n_clusters=n_clusters,
        confidence=confidence,
    )
