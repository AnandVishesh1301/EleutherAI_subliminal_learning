"""Resampling, variance decomposition, TOST and the C1 interaction fit."""

from __future__ import annotations

import numpy as np
import pytest

from subliminal_learning.analysis.stats import (
    BootstrapEstimate,
    bootstrap_equivalence,
    cluster_bootstrap,
    family_interaction_fit,
    multi_bootstrap,
    stratified_cluster_bootstrap_draws,
    summarize_draws,
    tost_equivalence,
    variance_components,
)

_SEED = 20260805
_N_BOOT = 400


def _slope_over(x: np.ndarray, y: np.ndarray):
    """A `statistic` closure: OLS slope of y on x over the given row indices."""

    def statistic(rows: np.ndarray) -> float:
        xi, yi = x[rows], y[rows]
        centred = xi - xi.mean()
        denominator = float(centred @ centred)
        if denominator <= 1e-12:
            return float("nan")
        return float(centred @ (yi - yi.mean()) / denominator)

    return statistic


def _ladder(slope: float, noise: float, seed: int):
    """A crossed rung x pairing panel with a planted slope."""
    rng = np.random.default_rng(seed)
    rungs = np.repeat(np.arange(6), 5 * 4)
    pairings = np.tile(np.repeat(np.arange(5), 4), 6)
    x = rungs.astype(np.float64) / 5.0
    y = 1.0 + slope * x + rng.normal(scale=noise, size=x.size)
    return rungs, pairings, x, y


def test_cluster_bootstrap_recovers_a_planted_slope():
    rungs, pairings, x, y = _ladder(slope=2.0, noise=0.05, seed=_SEED)
    result = cluster_bootstrap(
        _slope_over(x, y), rungs, pairings, n_boot=_N_BOOT, seed=_SEED
    )
    assert result.estimate == pytest.approx(2.0, abs=0.1)
    assert result.ci_low < 2.0 < result.ci_high
    assert result.p_boot < 0.05
    assert result.n_boot == _N_BOOT
    assert result.n_valid <= _N_BOOT


def test_cluster_bootstrap_is_reproducible_and_seed_sensitive():
    rungs, pairings, x, y = _ladder(slope=2.0, noise=0.3, seed=_SEED)
    statistic = _slope_over(x, y)
    first = cluster_bootstrap(statistic, rungs, pairings, n_boot=_N_BOOT, seed=7)
    again = cluster_bootstrap(statistic, rungs, pairings, n_boot=_N_BOOT, seed=7)
    other = cluster_bootstrap(statistic, rungs, pairings, n_boot=_N_BOOT, seed=8)
    assert first == again
    assert first.estimate == other.estimate
    assert (first.ci_low, first.ci_high) != (other.ci_low, other.ci_high)


def test_cluster_bootstrap_covers_zero_when_there_is_no_slope():
    rungs, pairings, x, y = _ladder(slope=0.0, noise=0.5, seed=11)
    result = cluster_bootstrap(
        _slope_over(x, y), rungs, pairings, n_boot=_N_BOOT, seed=_SEED
    )
    assert result.ci_low < 0.0 < result.ci_high
    assert result.p_boot > 0.05


def test_cluster_bootstrap_widens_when_a_whole_pairing_is_an_outlier():
    """Pairing-level shifts must inflate the interval; row resampling would not."""
    rungs, pairings, x, y = _ladder(slope=2.0, noise=0.05, seed=12)
    shifted = y + 3.0 * (pairings == 0) * x
    tight = cluster_bootstrap(
        _slope_over(x, y), rungs, pairings, n_boot=_N_BOOT, seed=_SEED
    )
    wide = cluster_bootstrap(
        _slope_over(x, shifted), rungs, pairings, n_boot=_N_BOOT, seed=_SEED
    )
    assert (wide.ci_high - wide.ci_low) > 5.0 * (tight.ci_high - tight.ci_low)


@pytest.mark.parametrize(
    ("rungs", "pairings", "confidence", "match"),
    [
        ([0, 0, 1], [0, 1], 0.95, "matching 1-D"),
        ([], [], 0.95, "no rows"),
        ([0, 1], [0, 1], 1.5, "confidence must be"),
    ],
)
def test_cluster_bootstrap_rejects_bad_inputs(rungs, pairings, confidence, match):
    with pytest.raises(ValueError, match=match):
        cluster_bootstrap(
            lambda rows: 0.0,
            rungs,
            pairings,
            n_boot=4,
            seed=_SEED,
            confidence=confidence,
        )


def test_multi_bootstrap_recovers_a_planted_mean():
    rng = np.random.default_rng(_SEED)
    groups = np.repeat(np.arange(9), 12)
    values = 0.4 + rng.normal(scale=0.1, size=groups.size)
    result = multi_bootstrap(
        lambda rows: float(values[rows].mean()), groups, n_boot=_N_BOOT, seed=_SEED
    )
    assert result.estimate == pytest.approx(float(values.mean()))
    assert result.ci_low < 0.4 < result.ci_high


def test_multi_bootstrap_carries_the_seed_level_variance():
    """A seed-population effect must widen the interval, not average away."""
    rng = np.random.default_rng(_SEED)
    groups = np.repeat(np.arange(9), 12)
    offsets = rng.normal(scale=1.0, size=9)
    seed_driven = offsets[groups] + rng.normal(scale=0.01, size=groups.size)
    row_driven = rng.normal(scale=1.0, size=groups.size)

    def width(values: np.ndarray) -> float:
        result = multi_bootstrap(
            lambda rows: float(values[rows].mean()), groups, n_boot=_N_BOOT, seed=_SEED
        )
        return result.ci_high - result.ci_low

    assert width(seed_driven) > 2.0 * width(row_driven)


def test_variance_components_credit_the_dominant_factor():
    rng = np.random.default_rng(_SEED)
    pairing = np.repeat(np.arange(4), 18)
    ftseed = np.tile(np.repeat(np.arange(3), 6), 4)
    gseed = np.tile(np.arange(6), 12)
    values = 5.0 * pairing + 0.5 * ftseed + rng.normal(scale=0.01, size=pairing.size)

    result = variance_components(
        values, {"pairing": pairing, "ftseed": ftseed, "gseed": gseed}
    )
    assert sum(result.fractions.values()) == pytest.approx(1.0)
    assert result.fractions["pairing"] > 0.98
    assert result.fractions["pairing"] > result.fractions["ftseed"]
    assert result.fractions["ftseed"] > result.fractions["gseed"]
    assert result.fractions["residual"] < 0.01
    assert result.order == ("pairing", "ftseed", "gseed")
    assert result.n_obs == values.size


def test_variance_components_are_order_invariant_on_a_balanced_design():
    rng = np.random.default_rng(3)
    a = np.repeat(np.arange(4), 12)
    b = np.tile(np.repeat(np.arange(3), 4), 4)
    values = 1.0 * a + 1.0 * b + rng.normal(scale=0.5, size=a.size)
    factors = {"a": a, "b": b}
    first = variance_components(values, factors, order=("a", "b"))
    second = variance_components(values, factors, order=("b", "a"))
    assert first.order == ("a", "b")
    assert second.order == ("b", "a")
    assert first.sums_of_squares["a"] == pytest.approx(second.sums_of_squares["a"])
    assert first.sums_of_squares["residual"] == pytest.approx(
        second.sums_of_squares["residual"]
    )


def test_variance_components_are_order_dependent_when_the_design_is_unbalanced():
    """Sequential SS credits shared variance to whichever factor entered first."""
    rng = np.random.default_rng(3)
    a = np.repeat(np.arange(4), 12)
    b = np.tile(np.repeat(np.arange(3), 4), 4)
    keep = ~((a == 0) & (b == 2))
    a, b = a[keep], b[keep]
    values = 1.0 * a + 1.0 * b + rng.normal(scale=0.5, size=a.size)
    factors = {"a": a, "b": b}
    first = variance_components(values, factors, order=("a", "b"))
    second = variance_components(values, factors, order=("b", "a"))
    assert first.sums_of_squares["a"] != pytest.approx(second.sums_of_squares["a"])
    assert first.sums_of_squares["residual"] == pytest.approx(
        second.sums_of_squares["residual"]
    )


def test_variance_components_reject_a_bad_order_and_a_constant_outcome():
    labels = np.repeat(np.arange(2), 5)
    with pytest.raises(ValueError, match="not a permutation"):
        variance_components(np.arange(10.0), {"a": labels}, order=("b",))
    with pytest.raises(ValueError, match="constant to float precision"):
        variance_components(np.ones(10), {"a": labels})


def test_tost_declares_equivalence_only_inside_the_bounds():
    rng = np.random.default_rng(_SEED)
    near_zero = rng.normal(loc=0.0, scale=0.05, size=40)
    inside = tost_equivalence(near_zero, bound_low=-0.5, bound_high=0.5)
    assert inside.equivalent
    assert inside.p_value < 0.05

    shifted = near_zero + 0.6
    outside = tost_equivalence(shifted, bound_low=-0.5, bound_high=0.5)
    assert not outside.equivalent
    assert outside.p_value > 0.05


def test_tost_flips_exactly_where_the_interval_meets_the_bound():
    rng = np.random.default_rng(4)
    sample = rng.normal(loc=0.0, scale=0.2, size=30)
    reference = tost_equivalence(sample, bound_low=-1.0, bound_high=1.0, alpha=0.05)
    edge = max(abs(reference.ci_low), abs(reference.ci_high))
    just_inside = tost_equivalence(
        sample, bound_low=-edge - 1e-6, bound_high=edge + 1e-6, alpha=0.05
    )
    just_outside = tost_equivalence(
        sample, bound_low=-edge + 1e-6, bound_high=edge - 1e-6, alpha=0.05
    )
    assert just_inside.equivalent
    assert not just_outside.equivalent


def test_tost_p_value_is_monotone_in_the_bound_width():
    rng = np.random.default_rng(5)
    sample = rng.normal(loc=0.0, scale=0.2, size=30)
    p_values = [
        tost_equivalence(sample, bound_low=-w, bound_high=w).p_value
        for w in (0.05, 0.1, 0.2, 0.5, 1.0)
    ]
    assert p_values == sorted(p_values, reverse=True)


@pytest.mark.parametrize(
    ("values", "bounds", "match"),
    [
        ([1.0], (-1.0, 1.0), "n >= 2"),
        ([1.0, 2.0], (1.0, -1.0), "bound_low < bound_high"),
        ([1.0, 1.0, 1.0], (-1.0, 1.0), "zero-variance"),
    ],
)
def test_tost_rejects_degenerate_inputs(values, bounds, match):
    with pytest.raises(ValueError, match=match):
        tost_equivalence(values, bound_low=bounds[0], bound_high=bounds[1])


def _interaction_panel(beta: tuple[float, float, float, float], seed: int):
    """A clustered panel with planted `(beta0, beta1, beta2, beta3)`."""
    rng = np.random.default_rng(seed)
    d_trait = np.tile(np.repeat(np.linspace(0.0, 1.0, 6), 5), 2)
    family = np.array(["data"] * (d_trait.size // 2) + ["weight"] * (d_trait.size // 2))
    codes = (family == "weight").astype(np.float64)
    cluster = np.tile(np.arange(10), d_trait.size // 10)
    y = (
        beta[0]
        + beta[1] * d_trait
        + beta[2] * codes
        + beta[3] * d_trait * codes
        + rng.normal(scale=0.02, size=d_trait.size)
    )
    return y, d_trait, family, cluster


def test_family_interaction_fit_recovers_a_planted_beta3():
    planted = (0.1, 0.8, -0.05, 0.45)
    y, d_trait, family, cluster = _interaction_panel(planted, seed=_SEED)
    fit = family_interaction_fit(y, d_trait, family, cluster)
    assert fit.intercept.estimate == pytest.approx(planted[0], abs=0.02)
    assert fit.d_trait.estimate == pytest.approx(planted[1], abs=0.05)
    assert fit.family.estimate == pytest.approx(planted[2], abs=0.02)
    assert fit.interaction.estimate == pytest.approx(planted[3], abs=0.05)
    assert fit.interaction.ci_low < planted[3] < fit.interaction.ci_high
    assert fit.n_obs == y.size
    assert fit.n_clusters == 10


def test_family_interaction_beta3_is_weight_minus_data():
    y, d_trait, family, cluster = _interaction_panel((0.0, 1.0, 0.0, 0.5), seed=6)
    fit = family_interaction_fit(y, d_trait, family, cluster)
    swapped = np.where(family == "data", "weight", "data")
    mirrored = family_interaction_fit(y, d_trait, swapped, cluster)
    assert fit.interaction.estimate == pytest.approx(0.5, abs=0.05)
    assert mirrored.interaction.estimate == pytest.approx(-0.5, abs=0.05)


def test_family_interaction_fit_is_flat_when_the_families_share_a_slope():
    y, d_trait, family, cluster = _interaction_panel((0.0, 1.0, 0.0, 0.0), seed=7)
    fit = family_interaction_fit(y, d_trait, family, cluster)
    assert fit.interaction.ci_low < 0.0 < fit.interaction.ci_high


def test_family_interaction_fit_rejects_unknown_families_and_single_clusters():
    y, d_trait, family, cluster = _interaction_panel((0.0, 1.0, 0.0, 0.0), seed=8)
    bad = np.where(np.arange(family.size) == 0, "full", family)
    with pytest.raises(ValueError, match="family must be"):
        family_interaction_fit(y, d_trait, bad, cluster)
    with pytest.raises(ValueError, match=">= 2 clusters"):
        family_interaction_fit(y, d_trait, family, np.zeros_like(cluster))


def _strata_panel():
    # Two strata of whole clusters (leaves); rows are leaf x quintile cells.
    strata = np.array(["t"] * 6 + ["c"] * 4)
    clusters = np.array(["t0", "t0", "t1", "t1", "t2", "t2", "c0", "c0", "c1", "c1"])
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0, 10.0, 20.0, 20.0])
    return strata, clusters, values


def test_stratified_cluster_bootstrap_keeps_stratum_cluster_counts():
    strata, clusters, values = _strata_panel()
    seen_sizes = set()

    def statistic(idx):
        seen_sizes.add(
            (int((strata[idx] == "t").sum()), int((strata[idx] == "c").sum()))
        )
        return np.array([values[idx][strata[idx] == "t"].mean(), values[idx].size])

    observed, draws = stratified_cluster_bootstrap_draws(
        statistic, strata, clusters, n_boot=300, seed=_SEED
    )
    assert observed.tolist() == [2.0, 10.0]
    # Every replicate draws 3 treatment and 2 control clusters of 2 rows each.
    assert seen_sizes == {(6, 4)}
    assert np.all(draws[:, 1] == 10)
    assert set(np.unique(draws[:, 0])) <= {1.0, 4 / 3, 5 / 3, 2.0, 7 / 3, 8 / 3, 3.0}


def test_stratified_cluster_bootstrap_is_reproducible_and_seed_sensitive():
    strata, clusters, values = _strata_panel()

    def stat(idx):
        return float(values[idx].mean())

    _, a = stratified_cluster_bootstrap_draws(
        stat, strata, clusters, n_boot=200, seed=1
    )
    _, b = stratified_cluster_bootstrap_draws(
        stat, strata, clusters, n_boot=200, seed=1
    )
    _, c = stratified_cluster_bootstrap_draws(
        stat, strata, clusters, n_boot=200, seed=2
    )
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stratified_cluster_bootstrap_rejects_a_cluster_spanning_strata():
    with pytest.raises(ValueError, match="spans strata"):
        stratified_cluster_bootstrap_draws(
            lambda idx: 0.0, ["a", "b"], ["x", "x"], n_boot=2, seed=0
        )


def test_summarize_draws_percentiles_drop_nonfinite():
    draws = np.array([np.nan, *np.arange(1.0, 101.0)])
    est = summarize_draws(50.0, draws, confidence=0.90)
    assert est.n_valid == 100 and est.n_boot == 101
    assert est.ci_low == pytest.approx(np.percentile(np.arange(1.0, 101.0), 5.0))
    assert est.ci_high == pytest.approx(np.percentile(np.arange(1.0, 101.0), 95.0))


@pytest.mark.parametrize(
    ("low", "high", "equivalent"),
    [(-0.002, 0.002, True), (-0.002, 0.0025, False), (-0.0026, 0.0024, False)],
)
def test_bootstrap_equivalence_requires_the_interval_strictly_inside(
    low, high, equivalent
):
    est = BootstrapEstimate(
        estimate=0.0,
        ci_low=low,
        ci_high=high,
        p_boot=1.0,
        n_valid=10,
        n_boot=10,
        confidence=0.9,
    )
    assert (
        bootstrap_equivalence(est, bound_low=-0.0025, bound_high=0.0025).equivalent
        is equivalent
    )
