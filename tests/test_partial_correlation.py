"""Tests for the single partial-correlation implementation, `stats.core.partial_correlation`."""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.stats import rankdata
from scipy.stats import t as t_dist

from subliminal_learning.analysis import sequence_length_transfer as slt
from subliminal_learning.stats import diagonal_direction_tests as ddt
from subliminal_learning.stats.core import partial_correlation


# Pre-2026-10-07 body of `sequence_length_transfer.partial_correlation`, kept as the oracle.
def _old_slt_partial(x, y, controls=(), *, rank=False, min_n=3):
    """Pearson correlation of residuals; rank inputs once for partial Spearman."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    z = np.asarray(controls, float)
    if z.size == 0:
        z = np.empty((n, 0))
    elif z.ndim == 1:
        z = z[:, None]
    if z.shape[0] != n or len(y) != n:
        raise ValueError("Mismatched correlation array lengths")
    result = dict(
        n=n,
        rho=None,
        p=None,
        ci_low=None,
        ci_high=None,
        df=None,
        design_rank=None,
        status="insufficient",
    )
    if n < min_n:
        return result
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError("Correlation expects explicit complete-case arrays")
    if rank:
        x, y = rankdata(x), rankdata(y)
        z = np.column_stack([rankdata(col) for col in z.T]) if z.shape[1] else z
    # Standardization improves the condition of raw-probability and CE designs.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(result, status="constant")
    if z.shape[1]:
        scales = z.std(axis=0)
        if np.any(scales == 0):
            return dict(result, status="singular_control")
        z = (z - z.mean(axis=0)) / scales
    design = np.column_stack([np.ones(n), z])
    design_rank = int(np.linalg.matrix_rank(design))
    df = n - design_rank - 1
    result.update(df=df, design_rank=design_rank)
    if design_rank != design.shape[1]:
        return dict(result, status="singular_control")
    if df <= 0:
        return result
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if min(np.linalg.norm(rx), np.linalg.norm(ry)) < 1e-10 * math.sqrt(n):
        return dict(result, status="zero_residual")
    rho = float(
        np.clip(np.dot(rx, ry) / np.linalg.norm(rx) / np.linalg.norm(ry), -1, 1)
    )
    p = (
        0.0
        if abs(rho) >= 1
        else float(2 * t_dist.sf(abs(rho) * math.sqrt(df / (1 - rho * rho)), df))
    )
    low, high = None, None
    if df > 1:
        center = np.arctanh(np.clip(rho, -1 + 1e-15, 1 - 1e-15))
        low, high = (
            float(v)
            for v in np.tanh(
                center + np.array([-1, 1]) * 1.959963984540054 / math.sqrt(df - 1)
            )
        )
    return dict(
        result,
        rho=rho,
        p=p,
        ci_low=low,
        ci_high=high,
        status="ok",
        x_residual_variance_fraction=float(np.dot(rx, rx) / n),
        y_residual_variance_fraction=float(np.dot(ry, ry) / n),
        incremental_r2=float(rho * rho * np.dot(ry, ry) / n),
    )


def _data(seed, n=30, k=1, ties=False):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    x = z.sum(1) + rng.normal(size=n)
    y = z.sum(1) * 0.5 + 0.3 * x + rng.normal(size=n)
    if ties:
        x, y, z = np.round(x), np.round(y), np.round(z)
    return x, y, z


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("ties", [False, True])
def test_zero_controls_equal_scipy(seed, ties):
    x, y, _ = _data(seed, ties=ties)
    s, pr = partial_correlation(x, y), partial_correlation(x, y, method="pearson")
    ref_s, ref_p = stats.spearmanr(x, y), stats.pearsonr(x, y)
    assert abs(s.rho - ref_s.statistic) < 1e-12 and abs(s.p - ref_s.pvalue) < 1e-12
    assert abs(pr.rho - ref_p.statistic) < 1e-12 and abs(pr.p - ref_p.pvalue) < 1e-12
    assert (s.n, s.k, s.df) == (30, 0, 28)


def test_spearman_is_pearson_on_rank_residuals():
    x, y, z = _data(1, k=2)
    zr = np.column_stack([rankdata(c) for c in z.T])
    d = np.column_stack([np.ones(len(x)), zr])
    rx = rankdata(x) - d @ np.linalg.lstsq(d, rankdata(x), rcond=None)[0]
    ry = rankdata(y) - d @ np.linalg.lstsq(d, rankdata(y), rcond=None)[0]
    got = partial_correlation(x, y, z)
    assert abs(got.rho - stats.pearsonr(rx, ry).statistic) < 1e-12
    tt = got.rho * math.sqrt(got.df / (1 - got.rho ** 2))
    assert got.df == 30 - 2 - 2 and abs(got.p - 2 * t_dist.sf(abs(tt), got.df)) < 1e-12


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("rank", [False, True])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_sequence_length_transfer_unchanged(seed, rank, k):
    x, y, z = _data(seed, n=25, k=max(k, 1), ties=seed % 2 == 0)
    z = z[:, :k]
    old, new = _old_slt_partial(x, y, z, rank=rank), slt.partial_correlation(x, y, z, rank=rank)
    assert old["status"] == new["status"] == "ok"
    for key in ("rho", "p", "ci_low", "ci_high", "df"):
        assert abs(old[key] - new[key]) < 1e-12, key


def test_sequence_length_transfer_statuses_unchanged():
    x = np.arange(12.0)
    for z in (np.ones(12), x, np.column_stack([x, 2 * x])):
        assert slt.partial_correlation(x, x[::-1], z)["status"] == _old_slt_partial(
            x, x[::-1], z)["status"]
    with pytest.raises(ValueError):
        slt.partial_correlation(np.r_[x[:-1], np.nan], x, ())


def test_diagonal_partial_spearman_routes_through_core():
    x, y, z = _data(3, k=2)
    x[4] = np.nan
    got = ddt.partial_spearman(x, y, list(z.T))
    keep = np.isfinite(x)
    ref = partial_correlation(x[keep], y[keep], z[keep])
    assert (got.n, got.k) == (29, 2) and got.rho == ref.rho and got.p == ref.p
    small = ddt.partial_spearman(x[:5], y[:5], list(z[:5].T))
    assert small.n == 4 and math.isnan(small.rho) and math.isnan(small.p)


def test_nan_input_raises():
    x, y, z = _data(0)
    x[0] = np.nan
    with pytest.raises(ValueError):
        partial_correlation(x, y, z)
    with pytest.raises(ValueError):
        partial_correlation(y, y, np.r_[z[:, 0], np.inf])
    with pytest.raises(ValueError):
        partial_correlation(y, y[:-1])
    with pytest.raises(ValueError):
        partial_correlation(y, y, method="kendall")


def test_min_n():
    x, y, z = _data(0, n=8, k=2)
    assert math.isnan(partial_correlation(x[:4], y[:4], z[:4]).rho)  # default k + 3 = 5
    assert np.isfinite(partial_correlation(x[:5], y[:5], z[:5]).rho)
    r = partial_correlation(x, y, z, min_n=9)
    assert math.isnan(r.rho) and math.isnan(r.p) and (r.n, r.k, r.df) == (8, 2, 4)


def test_degenerate_cases():
    x = np.arange(10.0)
    assert math.isnan(partial_correlation(np.ones(10), x).rho)
    assert math.isnan(partial_correlation(x, x[::-1], x).rho)  # control explains x
    perfect = partial_correlation(x, 2 * x + 1, method="pearson")
    assert perfect.rho == pytest.approx(1.0) and math.isnan(perfect.p)
    exact = partial_correlation(x[:3], x[:3] ** 2, x[:3], min_n=0)
    assert exact.df == 0 and math.isnan(exact.p)
