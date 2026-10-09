"""Synthetic-matrix tests for `analysis.checkpoint_grid_factorization` (no disk, no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    CKPT_STEPS,
    CKPT_STUDENT_TAG,
    CKPT_TEACHER_TAGS,
    DATASET_DIR,
    asymmetry_test,
    cell_cross_validation,
    cycle_tests,
    diagonal_heldout,
    discover_checkpoint_grids,
    factor_borrowing,
    factor_summary,
    fit_models,
    fit_product_proximity,
    fit_rank1,
    heldout_ftseed,
    leave_one_recipe_out,
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


# --------------------------------------------------------------------------------------
# Held-out evaluation of product + proximity
# --------------------------------------------------------------------------------------
STEPS = CKPT_STEPS
_T = np.asarray(STEPS) / 1000.0
E10 = np.exp(-np.abs(np.subtract.outer(_T, _T)) / 10.0)
A13 = np.linspace(0.2, 0.8, 13)
B13 = np.linspace(0.4, 1.0, 13)


def _r2(M, P):
    return 1 - ((M - P) ** 2).sum() / ((M - M.mean()) ** 2).sum()


def test_exact_product_proximity_cross_validates_and_recovers_the_diagonal():
    M = np.outer(A13, B13) + 0.1 * E10
    cv = cell_cross_validation(M, STEPS, n_folds=5, n_repeats=1).set_index("model")
    assert cv.loc["product+proximity", "r2_cv"] == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(cv.loc["rank-2", "r2_cv"])
    assert diagonal_heldout(M, STEPS)["product+proximity"] == pytest.approx(1.0, abs=1e-6)
    fit = fit_product_proximity(M, STEPS, ~np.eye(13, dtype=bool))
    assert fit.L == 10.0 and fit.c == pytest.approx(0.1, abs=1e-6)
    assert fit.b.max() == pytest.approx(1.0)


def test_additive_beats_product_under_cv_on_an_additive_grid():
    rng = np.random.default_rng(5)
    M = rng.random(13)[:, None] + rng.random(13)[None, :]
    cv = cell_cross_validation(M, STEPS, n_folds=5, n_repeats=1).set_index("model")
    assert cv.loc["additive", "r2_cv"] == pytest.approx(1.0, abs=1e-9)
    assert cv.loc["additive", "r2_cv"] > cv.loc["product", "r2_cv"]


def test_full_mask_equals_unmasked_fit_and_fit_models():
    rng = np.random.default_rng(6)
    M = np.outer(A13, B13) + 0.05 * E10 + 0.01 * rng.standard_normal((13, 13))
    full = fit_product_proximity(M, STEPS, np.ones((13, 13), dtype=bool))
    unmasked = fit_product_proximity(M, STEPS)
    assert full.P == pytest.approx(unmasked.P, abs=1e-12)
    table = fit_models(M, STEPS)
    r2 = table.loc[table["model"].str.startswith("product + c*exp"), "r2"].item()
    assert _r2(M, unmasked.P) == pytest.approx(r2, abs=1e-6)


def _runs(grids: dict[str, np.ndarray], seeds=(1,)) -> pd.DataFrame:
    rows = [{"t": t, "s": s, "tag": tag, "ftseed": k, "p": float(M[i, j])}
            for tag, M in grids.items() for k in seeds
            for i, t in enumerate(STEPS) for j, s in enumerate(STEPS)]
    return pd.DataFrame(rows)


def test_recipe_out_borrowed_b_matches_own_fit_when_recipes_share_b():
    rng = np.random.default_rng(7)
    grids = {f"r{k}": np.outer(rng.random(13) + 0.2, B13) + 0.05 * E10 for k in range(3)}
    runs = _runs(grids)
    partial = runs[(runs["tag"] == "r0") & (runs["t"] == STEPS[0])].assign(tag="rpartial")
    table = leave_one_recipe_out(pd.concat([runs, partial]))
    assert list(table["held"]) == ["r0", "r1", "r2"]  # partial held-out recipe skipped
    assert table["b_borrowed"].to_numpy() == pytest.approx(table["own_fit"].to_numpy(), abs=1e-3)
    assert table["own_fit"].to_numpy() == pytest.approx(np.ones(3), abs=1e-6)


def test_heldout_ftseed_refuses_incomplete_test_seed():
    M = np.outer(A13, B13)
    runs = _runs({"r0": M}, seeds=(1, 2, 3))
    result = heldout_ftseed(runs)
    assert result is not None and result["model"] == pytest.approx(1.0, abs=1e-6)
    incomplete = runs.drop(runs.index[(runs["ftseed"] == 3)][:1])
    assert heldout_ftseed(incomplete) is None


def test_factor_borrowing_shared_b_transfers_but_a_does_not():
    b = np.linspace(0.2, 1.0, 13) ** 2
    grids = {"x": np.outer(np.linspace(0.2, 1.0, 13), b),
             "y": np.outer(np.linspace(1.0, 0.2, 13) ** 3, b)}
    table = factor_borrowing(grids, STEPS).set_index("target")
    assert table.loc["x", "b_from_src"] == pytest.approx(1.0, abs=1e-6)
    assert table.loc["x", "a_from_src"] < 0.5
    assert table.loc["x", "corr_b"] == pytest.approx(1.0, abs=1e-6)


def test_factor_summary_ratios_and_peaks():
    a = np.linspace(0.1, 1.3, 13)
    a[10] = 2.0  # peak at 125k
    b = np.linspace(0.2, 1.0, 13)
    summary = factor_summary(np.outer(a, b), STEPS)
    assert summary["a_143_125"] == pytest.approx(1.3 / 2.0, abs=1e-6)
    assert summary["b_143_125"] == pytest.approx(1.0 / b[10], abs=1e-6)
    assert (summary["a_peak_k"], summary["b_peak_k"]) == (125, 143)


def _write_cell(root: Path, animal: str, suffix: str, t: int, s: int, tag: str, p=0.5):
    d = (root / "students" / animal / "ckpt" / f"pythia-160m{suffix}@step{s}"
         / f"teacher-pythia-160m{suffix}@step{t}" / f"{tag}_tseed1" / DATASET_DIR
         / CKPT_STUDENT_TAG / "ftseed1")
    d.mkdir(parents=True, exist_ok=True)
    (d / "preference_eval.json").write_text(json.dumps({"sets": {"test": {"p_target": p}}}))


def test_discover_checkpoint_grids(tmp_path):
    steps = [75000, 80000]
    full_tag, part_tag = CKPT_TEACHER_TAGS[:2]
    for t in steps:
        for s in steps:
            _write_cell(tmp_path, "dog", "", t, s, full_tag)
            _write_cell(tmp_path, "dog", "-seed1", t, s, full_tag)
    _write_cell(tmp_path, "dog", "-seed1", 75000, 75000, part_tag)
    _write_cell(tmp_path, "dog", "-seed2", 75000, 75000, full_tag)
    coverage, runs = discover_checkpoint_grids(tmp_path, ["dog"], steps=steps)
    cov = coverage.set_index("lineage")
    assert set(runs) == {("dog", "orig"), ("dog", "seed1")}
    assert not cov.loc["seed2", "complete"] and cov.loc["seed2", "cells"] == 1
    assert cov.loc["seed1", "dropped"] == [part_tag] and cov.loc["seed1", "n_recipes"] == 1
    assert set(runs[("dog", "seed1")]["tag"]) == {full_tag}
