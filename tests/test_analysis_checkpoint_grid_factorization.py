"""Synthetic-matrix tests for `analysis.checkpoint_grid_factorization` (no disk, no network)."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    CKPT_STEPS,
    asymmetry_test,
    cycle_tests,
    fit_models,
    fit_rank1,
    noise_ceiling,
    partner_independence_ratios,
)

A_PLANTED = np.array([0.1, 0.3, 0.5, 0.9, 1.4])
B_PLANTED = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def test_rank1_recovers_planted_factors_exactly():
    M = np.outer(A_PLANTED, B_PLANTED)
    a, b = fit_rank1(M)
    assert b.max() == pytest.approx(1.0)
    assert np.outer(a, b) == pytest.approx(M, abs=1e-10)
    # a and b are the planted vectors up to the single (removed) scale gauge.
    assert a / A_PLANTED == pytest.approx(np.full(5, (a / A_PLANTED)[0]), rel=1e-8)
    assert b / B_PLANTED == pytest.approx(np.full(5, (b / B_PLANTED)[0]), rel=1e-8)


def test_full_mask_rank1_equals_top_svd_term():
    rng = np.random.default_rng(0)
    M = np.outer(A_PLANTED, B_PLANTED) + 0.02 * rng.standard_normal((5, 5))
    a, b = fit_rank1(M)
    U, s, Vt = np.linalg.svd(M)
    assert np.outer(a, b) == pytest.approx(np.outer(U[:, 0] * s[0], Vt[0]), abs=1e-9)


def test_masked_fit_ignores_a_corrupted_diagonal():
    M = np.outer(A_PLANTED, B_PLANTED).copy()
    off = ~np.eye(5, dtype=bool)
    np.fill_diagonal(M, 10.0)  # a different experiment lives on the diagonal
    a, b = fit_rank1(M, off)
    P = np.outer(a, b)
    assert P[off] == pytest.approx(np.outer(A_PLANTED, B_PLANTED)[off], abs=1e-8)


def test_symmetric_matrix_has_zero_antisymmetric_share():
    rng = np.random.default_rng(1)
    S = rng.random((6, 6)) + 1.0
    M = (S + S.T) / 2
    se = np.full((6, 6), 0.01)
    a, b = fit_rank1(M)
    result = asymmetry_test(M, se, a, b)
    assert result["antisym_share"] == pytest.approx(0.0, abs=1e-12)
    assert result["noise_share"] > 0


def test_cycle_statistics_vanish_for_a_rank_one_matrix():
    M = np.outer([0.2, 0.5, 0.8, 1.3], [0.3, 0.6, 0.9, 1.1])
    assert cycle_tests(M)["cycle"].to_numpy() == pytest.approx(np.zeros(4), abs=1e-12)
    ratios = partner_independence_ratios(M, ["w", "x", "y", "z"])
    spread = ratios.groupby(["side", "pair"])["ratio"].std().to_numpy()
    assert spread == pytest.approx(np.zeros_like(spread), abs=1e-12)


def test_free_parameter_counts_for_the_canonical_13_by_13_grid():
    rng = np.random.default_rng(2)
    M = np.outer(rng.random(13) + 0.5, rng.random(13) + 0.5)
    table = fit_models(M, CKPT_STEPS)
    counts = {
        prefix: int(table.loc[table["model"].str.startswith(prefix), "free_params"].item())
        for prefix in ["additive", "bottleneck", "mirror", "product a(t)b(s)",
                       "product + c*1[t=s]", "product + c*exp", "rank-2"]
    }
    assert counts == {"additive": 25, "bottleneck": 13, "mirror": 13, "product a(t)b(s)": 25,
                      "product + c*1[t=s]": 26, "product + c*exp": 27, "rank-2": 50}


def test_product_wins_and_r2_is_one_on_a_planted_rank_one_grid():
    M = np.outer(np.linspace(0.1, 0.9, 13), np.linspace(0.2, 1.0, 13))
    table = fit_models(M, CKPT_STEPS).set_index("model")
    product = next(v for k, v in table["r2"].items() if k.startswith("product a(t)b(s)"))
    assert product == pytest.approx(1.0, abs=1e-10)
    log_product = next(v for k, v in table["r2_log"].items() if k.startswith("product a(t)b(s)"))
    assert log_product == pytest.approx(1.0, abs=1e-8)


def test_bottleneck_prediction_is_constant_along_each_min_band():
    rng = np.random.default_rng(3)
    M = rng.random((13, 13))
    table = fit_models(M, CKPT_STEPS)
    assert (table["free_params"] > 0).all()
    # Recompute the bottleneck prediction the module fits and check its band structure.
    idx = np.arange(13)
    key = np.minimum.outer(idx, idx)
    P = np.zeros_like(M)
    for k in range(13):
        P[key == k] = M[key == k].mean()
    for k in range(13):
        assert np.ptp(P[key == k]) == pytest.approx(0.0, abs=1e-12)
    bottleneck = next(
        r["r2"] for _, r in table.iterrows() if r["model"].startswith("bottleneck")
    )
    assert bottleneck == pytest.approx(1 - ((M - P) ** 2).sum() / ((M - M.mean()) ** 2).sum())


def test_noise_ceiling_matches_its_formula():
    rng = np.random.default_rng(4)
    M = rng.random((4, 4)) + 1.0
    se = rng.random((4, 4)) * 0.05
    expected = 1 - (se**2).sum() / ((M - M.mean()) ** 2).sum()
    assert noise_ceiling(M, se) == pytest.approx(expected)
