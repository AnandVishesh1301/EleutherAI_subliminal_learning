"""Unit tests for ``subliminal_learning.analysis.stats`` (CPU-only, synthetic data).

Hand-computable cases throughout: AUC is checked against a case worked out by hand (including
ties), pooling against a two-study identity, the ratio delta method against a diagonal-cov
closed form, and the bootstrap for reproducibility and cluster-as-unit resampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.analysis.stats import (
    binned_fraction,
    bootstrap_ci,
    cluster_logit,
    cluster_ols,
    delta_ratio_ci,
    ecdf,
    mannwhitney_auc,
    pool_fixed_effect,
    sign_binom_test,
)


# ------------------------- ecdf -------------------------


def test_ecdf_basic():
    x, y = ecdf([3.0, 1.0, 2.0])
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [1 / 3, 2 / 3, 1.0])


def test_ecdf_drops_nans_and_infs():
    x, y = ecdf([np.nan, 5.0, np.inf, 4.0])
    np.testing.assert_allclose(x, [4.0, 5.0])
    np.testing.assert_allclose(y, [0.5, 1.0])


def test_ecdf_empty():
    x, y = ecdf([np.nan])
    assert x.size == 0 and y.size == 0


def test_ecdf_preserves_duplicates_per_point_fractions():
    # Contract: duplicates are kept (len == n) and y is the per-point (1..n)/n, not scipy's
    # deduplicated step height; last point reaches 1.0.
    x, y = ecdf([2.0, 1.0, 2.0, 3.0])
    assert x.size == 4
    np.testing.assert_allclose(x, [1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [0.25, 0.5, 0.75, 1.0])
    assert y[-1] == 1.0


# ------------------------- mannwhitney_auc -------------------------


def test_mannwhitney_auc_untied_handcomputed():
    # pos beats neg on 5 of 6 pairs (1 vs 1.5); AUC = 5/6.
    pos, neg = [2.0, 3.0], [1.0, 1.5, 2.5]
    # pairs: (2>1,2>1.5,2>2.5=no),(3>1,3>1.5,3>2.5) -> 5 wins of 6
    assert mannwhitney_auc(pos, neg) == pytest.approx(5 / 6)


def test_mannwhitney_auc_ties_count_half():
    # One exact tie (2==2) contributes 0.5; other pair pos>neg -> (1 + 0.5)/2 = 0.75.
    assert mannwhitney_auc([2.0, 3.0], [2.0, 1.0]) == pytest.approx(
        (0.5 + 1 + 1 + 1) / 4
    )


def test_mannwhitney_auc_matches_roc_auc_score():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 5, 20).astype(float)  # small range -> lots of ties
    neg = rng.integers(0, 5, 25).astype(float)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    assert mannwhitney_auc(pos, neg) == pytest.approx(
        sklearn.roc_auc_score(labels, scores)
    )


def test_mannwhitney_auc_empty_side_is_nan():
    assert np.isnan(mannwhitney_auc([], [1.0]))
    assert np.isnan(mannwhitney_auc([np.nan], [1.0]))


# ------------------------- bootstrap_ci -------------------------


def test_bootstrap_ci_reproducible_given_seed():
    vals = np.arange(50, dtype=float)
    a = bootstrap_ci(vals, np.mean, n_boot=200, seed=3)
    b = bootstrap_ci(vals, np.mean, n_boot=200, seed=3)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    np.testing.assert_allclose(a.draws, b.draws)
    assert a.estimate == pytest.approx(vals.mean())


def test_bootstrap_ci_different_seed_differs():
    vals = np.arange(50, dtype=float)
    a = bootstrap_ci(vals, np.mean, n_boot=200, seed=1)
    b = bootstrap_ci(vals, np.mean, n_boot=200, seed=2)
    assert (a.lo, a.hi) != (b.lo, b.hi)


def test_bootstrap_ci_cluster_resamples_whole_cells():
    # 4 cells, 5 rows each. A stat counting distinct cluster labels present can only take a
    # handful of values if clusters are resampled as units (<=4 distinct, with repeats), and
    # crucially never exceeds the 4 real clusters.
    df = pd.DataFrame(
        {"cell": np.repeat(["a", "b", "c", "d"], 5), "y": np.arange(20.0)}
    )
    res = bootstrap_ci(
        df, lambda d: d["cell"].nunique(), n_boot=300, seed=0, cluster="cell"
    )
    assert res.draws.max() <= 4
    assert set(np.unique(res.draws)).issubset({1, 2, 3, 4})


def test_bootstrap_ci_drops_failing_replicates():
    res = bootstrap_ci(
        np.arange(10.0),
        lambda a: np.nan if a.mean() > 4.5 else a.mean(),
        n_boot=100,
        seed=0,
    )
    assert res.n_effective == res.draws.size
    assert res.n_effective < 100


def test_bootstrap_ci_cluster_requires_dataframe():
    with pytest.raises(ValueError):
        bootstrap_ci(np.arange(5.0), np.mean, cluster="cell")


# ------------------------- pool_fixed_effect -------------------------


def test_pool_equal_se_gives_mean_and_zero_q_for_identical():
    res = pool_fixed_effect([2.0, 2.0], se=[0.5, 0.5])
    assert res.mu == pytest.approx(2.0)
    assert res.q == pytest.approx(0.0)
    assert res.i2 == 0.0
    assert res.se == pytest.approx(0.5 / np.sqrt(2))


def test_pool_equal_se_mu_is_plain_mean():
    res = pool_fixed_effect([1.0, 3.0], se=[0.4, 0.4])
    assert res.mu == pytest.approx(2.0)


def test_pool_from_intervals_matches_se():
    est = [1.0, 2.0]
    se = [0.5, 1.0]
    lo = [e - 1.96 * s for e, s in zip(est, se)]
    hi = [e + 1.96 * s for e, s in zip(est, se)]
    a = pool_fixed_effect(est, se=se)
    b = pool_fixed_effect(est, lo=lo, hi=hi)
    assert b.mu == pytest.approx(a.mu)
    assert b.se == pytest.approx(a.se)


def test_pool_heterogeneous_has_positive_q():
    res = pool_fixed_effect([0.0, 5.0], se=[0.3, 0.3])
    assert res.q > 0
    assert res.i2 > 0
    assert 0.0 <= res.p_q <= 1.0


def test_pool_requires_se_or_interval():
    with pytest.raises(ValueError):
        pool_fixed_effect([1.0, 2.0])


# ------------------------- delta_ratio_ci -------------------------


def test_delta_ratio_ci_diagonal_cov_handcheck():
    # params = [_, 2, 4]; ratio 4/2 = 2. Diagonal cov var[1]=v1, var[2]=v2.
    # g[2]=1/2, g[1]=-4/4=-1 -> var = v2*(1/4) + v1*1.
    p = np.array([0.0, 2.0, 4.0])
    v1, v2 = 0.09, 0.16
    cov = np.diag([0.0, v1, v2])
    est, (lo, hi) = delta_ratio_ci(p, cov, 2, 1, z=1.96)
    assert est == pytest.approx(2.0)
    sd = np.sqrt(v2 * 0.25 + v1 * 1.0)
    assert lo == pytest.approx(2.0 - 1.96 * sd)
    assert hi == pytest.approx(2.0 + 1.96 * sd)


def test_delta_ratio_ci_matches_fit_collapse_law():
    # The estimator's ratios must be exactly what delta_ratio_ci produces on its params/cov.
    from subliminal_learning.analysis.collapse_law import fit_collapse_law, label_collapse

    rng = np.random.default_rng(0)
    rows = []
    for lr in [1e-4, 3e-4, 5e-4, 1e-3]:
        for r in [4, 16, 64]:
            for a in [4, 32]:
                k = lr * a**0.55 * r**-0.1
                # gentle slope + base rate keeps the design non-separable so the fit is regular
                p = 0.1 + 0.8 / (1 + np.exp(-4 * np.log(k / 8e-4)))
                for s in range(6):
                    rows.append({
                        "student_tag": f"ep20_lr{lr:g}_r{r}a{a}",
                        "lr": lr, "rank": r, "alpha": a,
                        "test": 1e-6 if rng.random() < p else 0.05,
                    })
    df = label_collapse(pd.DataFrame(rows))
    fit = fit_collapse_law(df)
    import statsmodels.api as sm
    from scipy.stats import norm

    from subliminal_learning.analysis.collapse_law import _design

    res = sm.Logit(df["collapsed"].to_numpy(), _design(df)).fit(
        disp=0, maxiter=500, cov_type="cluster",
        cov_kwds={"groups": df["student_tag"].to_numpy()},
    )
    p, V = np.asarray(res.params), np.asarray(res.cov_params())
    z = norm.ppf(0.975)
    b, b_ci = delta_ratio_ci(p, V, 2, 1, z=z)
    assert fit.b == pytest.approx(b)
    assert fit.b_ci == pytest.approx(b_ci)


# ------------------------- cluster_ols / cluster_logit -------------------------


def test_cluster_ols_runs_and_is_robust():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=40)})
    df["cell"] = np.repeat(np.arange(8), 5)
    df["y"] = 2 * df["x"] + rng.normal(size=40)
    res = cluster_ols(df, "y ~ x", "cell")
    assert res.cov_type == "cluster"
    assert res.params["x"] == pytest.approx(2.0, abs=0.6)


def test_cluster_logit_runs_and_is_robust():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=80)})
    df["cell"] = np.repeat(np.arange(16), 5)
    df["collapsed"] = (df["x"] + rng.normal(size=80) > 0).astype(int)
    res = cluster_logit(df, "collapsed ~ x", "cell")
    assert res.cov_type == "cluster"
    assert res.params["x"] > 0


# ------------------------- binned_fraction -------------------------


def test_binned_fraction_geometric_mid_and_min_n():
    df = pd.DataFrame({
        "K": [1.0, 2.0, 3.0, 4.0, 40.0, 60.0, 90.0],
        "flag": [0, 0, 1, 1, 1, 0, 1],
    })
    bins = [1.0, 10.0, 100.0]
    out = binned_fraction(df, "K", "flag", bins=bins, min_n=5)
    # bin 0: K in [1,10) -> 4 rows (< min_n) -> frac NaN; x_mid = sqrt(1*10)
    assert out.iloc[0]["n"] == 4
    assert np.isnan(out.iloc[0]["frac"])
    assert out.iloc[0]["x_mid"] == pytest.approx(np.sqrt(10.0))
    # bin 1: K in [10,100) -> 3 rows also below min_n
    assert out.iloc[1]["n"] == 3
    assert np.isnan(out.iloc[1]["frac"])
    assert out.iloc[1]["x_mid"] == pytest.approx(np.sqrt(1000.0))


def test_binned_fraction_reports_frac_when_enough():
    df = pd.DataFrame({"K": np.linspace(1, 9, 6), "flag": [1, 1, 0, 1, 0, 1]})
    out = binned_fraction(df, "K", "flag", bins=[0.0, 10.0], min_n=5)
    assert out.iloc[0]["n"] == 6
    assert out.iloc[0]["frac"] == pytest.approx(4 / 6)
    # non-positive edge -> arithmetic midpoint
    assert out.iloc[0]["x_mid"] == pytest.approx(5.0)


def test_binned_fraction_keeps_all_bins_empty_row_is_nan():
    # Two points land in bin 0; bins 1 and 2 are empty but must still appear as rows with
    # n=0 and frac NaN so a plotted curve renders them as gaps.
    df = pd.DataFrame({"K": [1.0, 2.0], "flag": [1, 0]})
    out = binned_fraction(df, "K", "flag", bins=[0.0, 5.0, 10.0, 15.0], min_n=1)
    assert len(out) == 3  # every bin kept
    assert out["n"].tolist() == [2, 0, 0]
    assert out.iloc[0]["frac"] == pytest.approx(0.5)
    assert np.isnan(out.iloc[1]["frac"]) and np.isnan(out.iloc[2]["frac"])
    # x_mid is defined for the empty bins too (geometric mean of their edges).
    assert out.iloc[1]["x_mid"] == pytest.approx(np.sqrt(50.0))


def test_binned_fraction_point_on_last_edge_excluded():
    # A point exactly on the final edge falls in no bin (np.digitize semantics), so it is not
    # counted -- guards against binned_statistic's both-edges-closed last bin.
    df = pd.DataFrame({"K": [1.0, 2.0, 10.0], "flag": [1, 0, 1]})
    out = binned_fraction(df, "K", "flag", bins=[0.0, 10.0], min_n=1)
    assert out.iloc[0]["n"] == 2  # K=10.0 excluded
    assert out.iloc[0]["frac"] == pytest.approx(0.5)


def test_binned_fraction_drops_non_finite_rows():
    # NaN/inf in either x or flag are excluded from the count and the fraction.
    df = pd.DataFrame({
        "K": [1.0, 2.0, np.nan, 4.0, np.inf],
        "flag": [1, 0, 1, np.nan, 1],
    })
    out = binned_fraction(df, "K", "flag", bins=[0.0, 10.0], min_n=1)
    # only K=1 (flag 1) and K=2 (flag 0) are finite in both -> n=2, frac=0.5
    assert out.iloc[0]["n"] == 2
    assert out.iloc[0]["frac"] == pytest.approx(0.5)


# ------------------------- sign_binom_test -------------------------


def test_sign_binom_test_drops_nan_and_zero():
    res = sign_binom_test([1.0, 2.0, -1.0, 0.0, np.nan], alternative="greater")
    # 2 positive of 3 non-zero finite -> n=3, k=2
    assert res.n == 3
    assert res.k == 2


def test_sign_binom_test_all_positive_significant():
    res = sign_binom_test([1.0, 2.0, 3.0, 4.0, 5.0], alternative="greater")
    assert res.pvalue < 0.05
