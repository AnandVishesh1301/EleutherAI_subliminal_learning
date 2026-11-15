"""Unit tests for ``subliminal_learning.stats.core`` (CPU-only, synthetic data).

Hand-computable cases throughout: AUC is checked against a case worked out by hand (including
ties), pooling against a two-study identity, the ratio delta method against a diagonal-cov
closed form, and the bootstrap for reproducibility and cluster-as-unit resampling.

Equivalence tests for the stats functions lifted out of the notebooks.

Each new helper is checked against the notebook's inline formula, copied verbatim below as the
reference oracle -- the point of these tests is that the refactor did not change semantics, so
the oracles must stay literal transcriptions, not re-derivations.
"""

from __future__ import annotations

import math as _math

from scipy import stats as sps
import numpy as np
import pandas as pd
import pytest

from subliminal_learning.stats.core import (
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
from subliminal_learning.stats.core import interp_crossing, two_proportion_z, tv_js


# -------------------- from test_analysis_stats.py --------------------

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

def test_sign_binom_test_drops_nan_and_zero():
    res = sign_binom_test([1.0, 2.0, -1.0, 0.0, np.nan], alternative="greater")
    # 2 positive of 3 non-zero finite -> n=3, k=2
    assert res.n == 3
    assert res.k == 2

def test_sign_binom_test_all_positive_significant():
    res = sign_binom_test([1.0, 2.0, 3.0, 4.0, 5.0], alternative="greater")
    assert res.pvalue < 0.05


# -------------------- from test_analysis_stats_additions.py --------------------

# --- oracles (verbatim from the notebooks) ---------------------------------------------
def _cross(lams, vals, thr, rising):                       # collapse_claims.py:542-551
    for i in range(1, len(lams)):
        a, b = vals[i - 1], vals[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            if rising and not np.isfinite(b):
                return lams[i]
            continue
        if (rising and a <= thr < b) or ((not rising) and a >= thr > b):
            t = ((_math.log(thr) - _math.log(a)) / (_math.log(b) - _math.log(a))
                 if min(a, b, thr) > 0 else (thr - a) / (b - a))
            return _math.exp(_math.log(lams[i - 1])
                             + t * (_math.log(lams[i]) - _math.log(lams[i - 1])))
    return np.nan

def _tv_js(p_counts, q_counts):                            # number_generation_analysis.py:579-585
    p = p_counts / p_counts.sum()
    q = q_counts / q_counts.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * np.abs(p - q).sum(), 0.5 * kl(p, m) + 0.5 * kl(q, m)

def _z_p(w1, t1, w2, t2):                                  # seed_fate_direction.py:75-79
    return sps.norm.sf(abs((w1 / t1 - w2 / t2)
                           / np.sqrt((w1 + w2) / (t1 + t2) * (1 - (w1 + w2) / (t1 + t2))
                                     * (1 / t1 + 1 / t2)))) * 2

# --- interp_crossing --------------------------------------------------------------------
LAMS = [0.5, 1.0, 2.0, 4.0, 8.0]

@pytest.mark.parametrize("vals,thr,rising", [
    ([0.1, 0.5, 2.0, 6.0, 9.0], 4.0, True),
    ([9.0, 6.0, 2.0, 0.5, 0.1], 1.0, False),
    ([0.1, 0.2, 0.3, 0.4, 0.5], 4.0, True),                    # never crosses
    ([0.1, 0.5, float("nan"), 6.0, 9.0], 4.0, True),           # non-finite on a rising curve
    ([0.0, 0.5, 2.0, 6.0, 9.0], 4.0, True),                    # non-positive -> linear in y
])
def test_interp_crossing_matches_oracle(vals, thr, rising):
    got = interp_crossing(LAMS, vals, thr, rising=rising)
    want = _cross(LAMS, vals, thr, rising)
    if np.isnan(want):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(float(want))

def test_interp_crossing_log_x_off():
    vals = [0.1, 0.5, 2.0, 6.0, 9.0]
    lin = interp_crossing(LAMS, vals, 4.0, rising=True, log_x=False)
    log = interp_crossing(LAMS, vals, 4.0, rising=True)
    # same bracketing interval, different x-space interpolation
    assert 2.0 < lin < 4.0 and 2.0 < log < 4.0
    assert lin != pytest.approx(log)
    t = (_math.log(4.0) - _math.log(2.0)) / (_math.log(6.0) - _math.log(2.0))
    assert lin == pytest.approx(2.0 + t * (4.0 - 2.0))

# --- tv_js ------------------------------------------------------------------------------
def test_tv_js_matches_oracle():
    rng = np.random.default_rng(0)
    p = rng.integers(0, 50, 25).astype(float)
    q = rng.integers(0, 50, 25).astype(float)
    tv, js = tv_js(p, q)
    tv0, js0 = _tv_js(p, q)
    assert tv == pytest.approx(tv0) and js == pytest.approx(js0)

def test_tv_js_identical_and_disjoint():
    p = np.array([3.0, 1.0, 0.0])
    assert tv_js(p, p) == pytest.approx((0.0, 0.0))
    tv, js = tv_js(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert tv == pytest.approx(1.0) and js == pytest.approx(1.0)   # 1 bit

def test_tv_js_matches_old_formula_on_random_pair():
    rng = np.random.default_rng(7)
    p = rng.random(40) * 10.0
    q = rng.random(40) * 10.0
    tv, js = tv_js(p, q)
    tv0, js0 = _tv_js(p, q)
    assert tv == pytest.approx(tv0, abs=1e-12) and js == pytest.approx(js0, abs=1e-12)

@pytest.mark.parametrize("p,q", [
    (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),      # mismatched lengths
    (np.array([1.0, -1.0]), np.array([1.0, 2.0])),          # negative entry
    (np.array([1.0, np.nan]), np.array([1.0, 2.0])),        # non-finite entry
    (np.array([0.0, 0.0]), np.array([1.0, 2.0])),           # zero-sum vector
])
def test_tv_js_rejects_bad_inputs(p, q):
    with pytest.raises(ValueError):
        tv_js(p, q)

# --- two_proportion_z -------------------------------------------------------------------
@pytest.mark.parametrize("w1,n1,w2,n2", [(30, 50, 20, 50), (5, 40, 15, 60), (10, 20, 10, 20)])
def test_two_proportion_z_matches_oracle(w1, n1, w2, n2):
    z, p = two_proportion_z(w1, n1, w2, n2)
    assert p == pytest.approx(float(_z_p(w1, n1, w2, n2)))
    assert np.sign(z) == np.sign(w1 / n1 - w2 / n2)
