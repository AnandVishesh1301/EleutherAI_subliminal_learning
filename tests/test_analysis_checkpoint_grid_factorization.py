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
    factorization_table,
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
from subliminal_learning.analysis.checkpoint_grid_factorization import _predict  # noqa: E402


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


def test_cv_sd_nan_for_one_repeat_and_small_for_default_repeats():
    rng = np.random.default_rng(7)
    M = np.outer(A13, B13) + 0.01 * rng.standard_normal((13, 13))
    one = cell_cross_validation(M, STEPS, n_repeats=1).set_index("model")
    assert one["r2_cv_sd"].isna().all()
    cv = cell_cross_validation(M, STEPS).set_index("model")
    sd = cv.loc[["additive", "product", "bottleneck", "product+proximity"], "r2_cv_sd"]
    assert np.isfinite(sd).all() and (sd >= 0).all()
    assert sd["product"] < 0.01
    assert np.isnan(cv.loc["rank-2", "r2_cv_sd"])


def test_cell_cv_default_is_five_folds():
    rng = np.random.default_rng(11)
    M = np.outer(A13, B13) + 0.05 * rng.standard_normal((13, 13))
    default = cell_cross_validation(M, STEPS)
    five = cell_cross_validation(M, STEPS, n_folds=5)
    ten = cell_cross_validation(M, STEPS, n_folds=10)
    pd.testing.assert_frame_equal(default, five)
    assert not np.allclose(default["r2_cv"].dropna(), ten["r2_cv"].dropna())


def test_full_mask_equals_unmasked_fit_and_fit_models():
    rng = np.random.default_rng(6)
    M = np.outer(A13, B13) + 0.05 * E10 + 0.01 * rng.standard_normal((13, 13))
    full = fit_product_proximity(M, STEPS, np.ones((13, 13), dtype=bool))
    unmasked = fit_product_proximity(M, STEPS)
    assert full.P == pytest.approx(unmasked.P, abs=1e-12)
    table = fit_models(M, STEPS)
    r2 = table.loc[table["model"].str.startswith("product + c*exp"), "r2"].item()
    assert _r2(M, unmasked.P) == pytest.approx(r2, abs=1e-6)


def _bneck_grid():
    m = np.linspace(0.1, 0.9, 13) ** 2
    idx = np.arange(13)
    return m[np.minimum.outer(idx, idx)]


def test_bottleneck_grid_cross_validates_and_additive_does_not():
    M = _bneck_grid()
    cv = cell_cross_validation(M, STEPS, n_folds=5, n_repeats=1).set_index("model")
    # Key 12 is the single cell (12, 12): when it is held out it falls back to the mean, so
    # CV R^2 is below 1 even for an exact bottleneck grid; every other held-out cell is exact.
    assert cv.loc["bottleneck", "r2_in"] == pytest.approx(1.0, abs=1e-12)
    assert cv.loc["bottleneck", "r2_cv"] > cv.loc["additive", "r2_cv"]
    test = np.zeros((13, 13), dtype=bool)
    test[::3, 1::2] = True
    test[12, 12] = False
    P = _predict("bottleneck", M, STEPS, ~test)
    assert P[test] == pytest.approx(M[test], abs=1e-12)


def test_full_mask_bottleneck_matches_fit_models():
    M = np.random.default_rng(7).random((13, 13))
    P = _predict("bottleneck", M, STEPS, np.ones((13, 13), dtype=bool))
    table = fit_models(M, STEPS)
    r2 = table.loc[table["model"].str.startswith("bottleneck"), "r2"].item()
    assert _r2(M, P) == pytest.approx(r2, abs=1e-12)


def test_bottleneck_unobserved_key_is_finite():
    M = _bneck_grid()
    idx = np.arange(13)
    mask = np.minimum.outer(idx, idx) != 12
    P = _predict("bottleneck", M, STEPS, mask)
    assert np.isfinite(P).all()
    assert P[12, 12] == pytest.approx(M[mask].mean())


def test_factorization_table_reports_bottleneck_columns():
    runs = {("dog", "orig"): _runs({CKPT_TEACHER_TAGS[0]: _bneck_grid()})}
    table = factorization_table(runs, STEPS, n_folds=5, n_repeats=1)
    previous = {"animal", "lineage", "n_recipes", "in_add", "in_prod", "in_prox", "in_rank2",
                "cv_add", "cv_prod", "cv_prox", "diag_prod", "diag_prox", "a_143_125",
                "b_143_125", "a_peak_k", "b_peak_k"}
    assert previous <= set(table.columns)
    assert {"in_bneck", "cv_bneck"} <= set(table.columns)
    cols = list(table.columns)
    assert cols.index("cv_prod") < cols.index("cv_bneck") < cols.index("cv_prox")


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
    assert table.loc["x", "both_from_src"] < 0.5


def test_factor_borrowing_both_from_src_scalar_multiple():
    M = np.outer(np.linspace(0.2, 1.0, 13), np.linspace(0.2, 1.0, 13) ** 2)
    table = factor_borrowing({"x": M, "y": 3.0 * M}, STEPS)
    assert list(table.columns[:5]) == ["target", "source", "own_fit", "own_product", "both_from_src"]
    for col in ("both_from_src", "own_product"):
        assert table[col].to_numpy() == pytest.approx(1.0, abs=1e-6)


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


# --------------------------------------------------------------------------------------
# send / receive index
# --------------------------------------------------------------------------------------
from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    send_receive_agreement,
    send_receive_index,
    send_receive_seed_index,
)

SR_STEPS = [1000, 2000, 3000, 4000, 5000, 6000]
SR_A = np.array([0.1, 0.4, 1.0, 0.5, 0.9, 0.3])
SR_B = np.array([0.8, 0.2, 0.5, 1.0, 0.45, 0.6])


def _sr_runs(M_by_tag: dict[str, np.ndarray], seeds=(1, 2)) -> pd.DataFrame:
    rows = []
    for tag, M in M_by_tag.items():
        for i, t in enumerate(SR_STEPS):
            for j, s in enumerate(SR_STEPS):
                for k in seeds:
                    rows.append({"t": t, "s": s, "tag": tag, "ftseed": k, "p": M[i, j]})
    return pd.DataFrame(rows)


def _sr_grid(a=SR_A, b=SR_B) -> np.ndarray:
    return np.outer(a, b) + 0.01 * np.exp(-_dk(SR_STEPS) / 2.0)


def _dk(steps):
    T = np.asarray(steps, dtype=float) / 1000.0
    return np.abs(np.subtract.outer(T, T))


def test_send_receive_index_recovers_sign_and_transpose_flips():
    M = _sr_grid()
    runs = {("dog", "orig"): _sr_runs({"r1": M, "r2": M.T}),
            ("dog", "x"): _sr_runs({"r1": M})}
    idx = send_receive_index(runs, SR_STEPS)
    assert list(idx.index.get_level_values("fit")) == ["dog", "dog/r1", "dog/r2"]
    assert list(idx.index.get_level_values("recipe")) == ["all", "r1", "r2"]
    assert list(idx.columns) == [1, 2, 3, 4, 5, 6]
    truth = np.log2((SR_A / SR_A.max()) / (SR_B / SR_B.max()))
    r1 = idx.xs("dog/r1", level="fit").iloc[0].to_numpy()
    r2 = idx.xs("dog/r2", level="fit").iloc[0].to_numpy()
    big = np.abs(truth) > 0.5
    assert np.array_equal(np.sign(r1[big]), np.sign(truth[big]))
    np.testing.assert_allclose(r2, -r1, atol=1e-6)
    pooled = send_receive_index({("dog", "orig"): _sr_runs({"r1": M})}, SR_STEPS)
    assert np.array_equal(np.sign(pooled.iloc[0].to_numpy()[big]), np.sign(truth[big]))


def test_send_receive_agreement_counts():
    idx = pd.DataFrame(
        {1: [1.0, 0.2, -1.5, np.nan], 2: [0.3, 0.1, 0.4, 0.2]},
        index=pd.MultiIndex.from_tuples(
            [("a", "a", "all"), ("b/x", "b", "x"), ("c/y", "c", "y"), ("d/z", "d", "z")],
            names=["fit", "animal", "recipe"]))
    ag = send_receive_agreement(idx)
    r1, r2 = ag.loc[1], ag.loc[2]
    assert (r1.n_pos, r1.n_neg, r1.n_nan, r1.majority) == (2, 1, 1, 1)
    assert not r1.unanimous and r2.unanimous
    assert r1.share_close == pytest.approx(1 / 3)
    assert r1.median_abs == pytest.approx(1.0)
    assert r1.dissenters == "c/y" and r2.dissenters == ""
    assert r2.share_close == 1.0
    assert "seed_sd" not in ag.columns


def test_send_receive_seed_sd_zero_for_identical_seeds():
    M = _sr_grid()
    runs = _sr_runs({"r1": M}, seeds=(1, 2, 3))
    runs = pd.concat([runs, pd.DataFrame([{"t": 1000, "s": 1000, "tag": "r1", "ftseed": 9,
                                           "p": 0.5}])])  # incomplete seed: skipped
    seed_idx = send_receive_seed_index(runs, SR_STEPS)
    assert list(seed_idx.index) == ["seed1", "seed2", "seed3"]
    idx = send_receive_index({("dog", "orig"): runs[runs.ftseed < 9]}, SR_STEPS)
    ag = send_receive_agreement(idx, seed_index=seed_idx)
    np.testing.assert_allclose(ag["seed_sd"].to_numpy(), 0.0, atol=1e-9)


def test_send_receive_curve_agreement_identical_and_reversed():
    from subliminal_learning.analysis.checkpoint_grid_factorization import (
        send_receive_curve_agreement,
    )
    a = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    M = _sr_grid(a=a, b=SR_B)
    Mrev = _sr_grid(a=a[::-1], b=SR_B)
    runs = {("dog", "orig"): _sr_runs({"r1": M, "r2": M}),
            ("cat", "orig"): _sr_runs({"r1": M, "r2": Mrev})}
    t = send_receive_curve_agreement(runs, SR_STEPS).set_index("group")
    assert list(t.index) == ["animals", "dog", "cat"]
    assert t.loc["dog", "n_pairs"] == 1
    assert t.loc["dog", ["a_min", "b_min"]].tolist() == pytest.approx([1.0, 1.0])
    assert t.loc["cat", "a_min"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------------------
# row-col score, shared factors, variance decomposition
# --------------------------------------------------------------------------------------
from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    send_receive_rowcol,
    send_receive_variance_decomposition,
    shared_factor_fit,
    shared_factor_table,
)


def test_send_receive_rowcol_sign_and_transpose():
    M = np.outer(SR_A, SR_B)
    score = send_receive_rowcol({("dog", "orig"): _sr_runs({"r1": M, "r2": M.T})}, SR_STEPS)
    assert list(score.index.get_level_values("fit")) == ["dog", "dog/r1", "dog/r2"]
    r1 = score.xs("dog/r1", level="fit").iloc[0].to_numpy()
    r2 = score.xs("dog/r2", level="fit").iloc[0].to_numpy()
    big = np.abs(SR_A - SR_B) > 0.25
    assert np.array_equal(np.sign(r1[big]), np.sign((SR_A - SR_B)[big]))
    np.testing.assert_allclose(r2, -r1, atol=1e-12)
    agree = send_receive_agreement(score)
    assert list(agree.index) == [1, 2, 3, 4, 5, 6]


def test_shared_factor_fit_recovers_multiples_and_penalises_sharing():
    base = np.outer(SR_A, SR_B)
    fit = shared_factor_fit({"x": base, "y": 3 * base}, share_a=True, share_b=True)
    assert min(fit.r2.values()) > 1 - 1e-9
    assert fit.scale["y"] / fit.scale["x"] == pytest.approx(3.0, rel=1e-6)
    assert fit.a_shared.max() == pytest.approx(1.0) and fit.b_shared.max() == pytest.approx(1.0)
    grids = {"x": base, "y": np.outer(SR_A, SR_B[::-1])}
    tab = shared_factor_table(grids, ceiling={"x": 0.9})
    assert list(tab.index) == ["own a, own b", "shared a, own b", "own a, shared b",
                               "shared a, shared b", "ceiling"]
    assert tab.loc["own a, shared b", "y"] < tab.loc["own a, own b", "y"]
    assert np.isnan(tab.loc["ceiling", "y"])


def test_send_receive_variance_decomposition():
    rng = np.random.default_rng(0)
    keys = [(f"{an}/{r}", an, r) for an in ("dog", "cat") for r in ("r1", "r2", "r3")]
    idx = pd.MultiIndex.from_tuples(keys, names=["fit", "animal", "recipe"])
    noisy = pd.DataFrame(rng.normal(size=(6, 5)), index=idx, columns=range(5))
    dec = send_receive_variance_decomposition(noisy)
    assert list(dec["component"]) == ["step", "step x animal", "step x recipe", "residual"]
    assert dec["share"].sum() == pytest.approx(1.0)
    pure = pd.DataFrame(np.tile(rng.normal(size=5), (6, 1)), index=idx, columns=range(5))
    pooled = pd.DataFrame([[9.0] * 5], columns=range(5),
                          index=pd.MultiIndex.from_tuples([("dog", "dog", "all")],
                                                          names=idx.names))
    dec = send_receive_variance_decomposition(pd.concat([pooled, pure]))
    assert dec.set_index("component").loc["step", "share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# OLMo loaders + Pythia trajectory readout
# --------------------------------------------------------------------------------------
from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    OLMO_BASE_DCLM,
    OLMO_BASE_DOLMA,
    OLMO_FINAL_STEP,
    OLMO_STUDENT_TAGS,
    discover_olmo_checkpoint_grids,
    load_checkpoint_grid_runs,
    load_olmo_base_controls,
    load_olmo_checkpoint_grid_runs,
)

_OP = "DataDecide-dolma1_7-1B-seed-default"


def _olmo_dir(step: int, op: str = _OP) -> str:
    return op if step == OLMO_FINAL_STEP else f"{op}@step{step}"


def _olmo_leaf(root, t, s, *, tag="ep10_lr1e-3_r128a32", dataset="ex10-20_prefix_n100k",
               controls=False, animal="elephant", seed="seed-default",
               base="DataDecide-dolma1_7-1B"):
    op = f"{base}-{seed}"
    base = root / "students" / ("controls/base" if controls else animal)
    if s != OLMO_FINAL_STEP:
        base = base / "ckpt"
    d = base / _olmo_dir(s, op) / f"teacher-{_olmo_dir(t, op)}"
    if not controls:
        d = d / f"{tag}_tseed0"
    d = d / dataset / OLMO_STUDENT_TAGS[0] / "ftseed1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pe(path: Path, p: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sets": {"test": {"p_target": p}}}))


def _traj(leaf: Path, pts: list[tuple[int, float]]):
    (leaf / "eval").mkdir(parents=True, exist_ok=True)
    (leaf / "eval" / "pref_traj.json").write_text(json.dumps(
        {"points": [{"step": st, "test": {"p_target": v}} for st, v in pts]}))


def test_olmo_grid_maps_final_dirs_to_final_step(tmp_path):
    steps = [30000, OLMO_FINAL_STEP]
    for t in steps:
        for s in steps:
            _pe(_olmo_leaf(tmp_path, t, s) / "preference_eval.json", t / 1e6 + s / 1e7)
    runs = load_olmo_checkpoint_grid_runs(tmp_path, "elephant", student_tag=OLMO_STUDENT_TAGS[0])
    assert set(zip(runs["t"], runs["s"])) == {(t, s) for t in steps for s in steps}
    row = runs[(runs["t"] == OLMO_FINAL_STEP) & (runs["s"] == 30000)].iloc[0]
    assert row["p"] == pytest.approx(OLMO_FINAL_STEP / 1e6 + 30000 / 1e7)


def test_olmo_grid_falls_back_to_n200k(tmp_path):
    _pe(_olmo_leaf(tmp_path, 55000, 30000, dataset="ex10-20_prefix_n200k")
        / "preference_eval.json", 0.3)
    runs = load_olmo_checkpoint_grid_runs(tmp_path, "elephant", student_tag=OLMO_STUDENT_TAGS[0])
    assert runs[["t", "s", "p"]].values.tolist() == [[55000, 30000, 0.3]]


def test_olmo_grid_reads_eval_animal_location(tmp_path):
    _pe(_olmo_leaf(tmp_path, 30000, 30000) / "eval" / "elephant" / "preference_eval.json", 0.4)
    runs = load_olmo_checkpoint_grid_runs(tmp_path, "elephant", student_tag=OLMO_STUDENT_TAGS[0])
    assert runs["p"].tolist() == [0.4]


def test_olmo_grid_peak_readout_is_trajectory_max(tmp_path):
    leaf = _olmo_leaf(tmp_path, 30000, 30000)
    _pe(leaf / "preference_eval.json", 0.1)
    _traj(leaf, [(100, 0.2), (200, 0.6), (300, 0.3)])
    runs = load_olmo_checkpoint_grid_runs(tmp_path, "elephant", student_tag=OLMO_STUDENT_TAGS[0],
                                          readout="peak")
    assert runs["p"].tolist() == [0.6]


def test_olmo_base_controls_tag_is_base(tmp_path):
    for s in (30000, OLMO_FINAL_STEP):
        _pe(_olmo_leaf(tmp_path, 30000, s, controls=True) / "eval" / "elephant"
            / "preference_eval.json", 0.05)
    ctrl = load_olmo_base_controls(tmp_path, "elephant", student_tag=OLMO_STUDENT_TAGS[0])
    assert set(ctrl["tag"]) == {"base"} and sorted(ctrl["s"]) == [30000, OLMO_FINAL_STEP]


def test_olmo_grid_default_keeps_unlisted_recipe(tmp_path):
    _pe(_olmo_leaf(tmp_path, 30000, 30000, tag="ep20_lr1e-3_r16a16") / "preference_eval.json", 0.3)
    runs = load_olmo_checkpoint_grid_runs(tmp_path, "elephant", student_tag=OLMO_STUDENT_TAGS[0])
    assert list(runs["tag"]) == ["ep20_lr1e-3_r16a16"]


def _olmo_grid(root, steps, tag, seed="seed-default", skip=()):
    for t in steps:
        for s in steps:
            if (t, s) not in skip:
                _pe(_olmo_leaf(root, t, s, tag=tag, seed=seed) / "preference_eval.json", 0.2)


def test_discover_olmo_keeps_complete_and_drops_partial(tmp_path):
    steps = [30000, 50000, OLMO_FINAL_STEP]
    _olmo_grid(tmp_path, steps, "ep10_lr1e-3_r128a32")
    _olmo_grid(tmp_path, steps, "ep5_lr2e-3_r32a32", skip={(30000, 50000)})
    cov, runs = discover_olmo_checkpoint_grids(
        tmp_path, "elephant", ["seed-default"], student_tag=OLMO_STUDENT_TAGS[0], steps=steps)
    row = cov.iloc[0]
    assert bool(row["complete"]) and row["n_recipes"] == 1 and row["n_cells"] == 9
    assert row["recipes"] == ["ep10_lr1e-3_r128a32"]
    assert row["dropped"] == ["ep5_lr2e-3_r32a32"]
    assert set(runs[(OLMO_BASE_DOLMA, "seed-default")]["tag"]) == {"ep10_lr1e-3_r128a32"}


def test_discover_olmo_falls_back_without_65000(tmp_path):
    full = [30000, 40000, 45000, 50000, 55000, 60000, 65000, OLMO_FINAL_STEP]
    seven = [x for x in full if x != 65000]
    _olmo_grid(tmp_path, seven, "ep10_lr1e-3_r128a32", seed="seed-default")
    _olmo_grid(tmp_path, full, "ep10_lr1e-3_r128a32", seed="seed-large-aux-2")
    cov, runs = discover_olmo_checkpoint_grids(
        tmp_path, "elephant", ["seed-default", "seed-large-aux-2"],
        student_tag=OLMO_STUDENT_TAGS[0])
    d, a = cov.set_index("seed").loc["seed-default"], cov.set_index("seed").loc["seed-large-aux-2"]
    assert d["n_steps"] == 7 and bool(d["complete"]) and d["steps_used"] == seven
    assert a["n_steps"] == 8 and bool(a["complete"]) and a["steps_used"] == full
    assert 65000 not in set(runs[(OLMO_BASE_DOLMA, "seed-default")]["t"]) and 65000 in set(runs[(OLMO_BASE_DOLMA, "seed-large-aux-2")]["t"])
    assert len(runs[(OLMO_BASE_DOLMA, "seed-large-aux-2")]) == 64


def test_pythia_traj_step_picks_step_and_skips_missing(tmp_path):
    tag = CKPT_TEACHER_TAGS[0]
    _write_cell(tmp_path, "dog", "", 75000, 75000, tag, p=0.9)
    _write_cell(tmp_path, "dog", "", 75000, 80000, tag, p=0.9)
    cell = (tmp_path / "students" / "dog" / "ckpt" / "pythia-160m@step{s}"
            / "teacher-pythia-160m@step75000" / f"{tag}_tseed1" / DATASET_DIR
            / CKPT_STUDENT_TAG / "ftseed1")
    _traj(Path(str(cell).format(s=75000)), [(6250, 0.2), (12500, 0.4), (25000, 0.7)])
    _traj(Path(str(cell).format(s=80000)), [(6250, 0.2), (25000, 0.7)])
    runs = load_checkpoint_grid_runs(tmp_path, "dog", "orig", traj_step=12500)
    assert runs[["s", "p"]].values.tolist() == [[75000, 0.4]]
    assert load_checkpoint_grid_runs(tmp_path, "dog", "orig")["p"].tolist() == [0.9, 0.9]


# --------------------------------------------------------------------------------------
# Identifiability of the masked product + proximity fit
# --------------------------------------------------------------------------------------
from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    PROXIMITY_MAX_COND,
)

STEPS7 = CKPT_STEPS[:7]
_T7 = np.asarray(STEPS7) / 1000.0
E7 = np.exp(-np.abs(np.subtract.outer(_T7, _T7)) / 10.0)


def _prox_grid7():
    rng = np.random.default_rng(1)
    return (np.outer(np.linspace(0.2, 0.8, 7), np.linspace(0.4, 1.0, 7)) + 0.1 * E7
            + 0.01 * rng.standard_normal((7, 7)))


def test_well_conditioned_grid_reports_small_cond_and_no_refusals():
    M = _prox_grid7()
    fit = fit_product_proximity(M, STEPS7)
    assert np.isfinite(fit.cond) and fit.cond < PROXIMITY_MAX_COND  # measured ~6.7
    cv = cell_cross_validation(M, STEPS7, n_repeats=2).set_index("model")
    assert list(cv.columns)[-1] == "n_unidentified"
    assert (cv["n_unidentified"] == 0).all()
    assert np.isfinite(cv.loc["product+proximity", "r2_cv"])


def test_unpinned_fold_is_refused(monkeypatch):
    # Degenerate case built by hand: row 0 and column 0 share exactly one observation,
    # (0, 0), so a[0], b[0] and c trade off freely on the fitted cells.
    M = _prox_grid7()
    held = np.zeros((7, 7), dtype=bool)
    held[0, 1:] = True
    held[1:, 0] = True
    fit = fit_product_proximity(M, STEPS7, ~held)
    assert fit.cond > PROXIMITY_MAX_COND  # measured ~1e18
    assert np.isfinite(fit.P).all()  # still returned, not raised

    # Hand-made two-fold partition: fold 0 is exactly ``held`` (even perm values).
    perm = np.where(held, 0, 1) + 2 * np.arange(49).reshape(7, 7)

    class _Rng:
        def permutation(self, size):
            return perm.ravel().copy()

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _Rng())
    cv = cell_cross_validation(M, STEPS7, n_folds=2, n_repeats=1).set_index("model")
    assert np.isnan(cv.loc["product+proximity", "r2_cv"])
    assert cv.loc["product+proximity", "n_unidentified"] > 0
    assert np.isfinite(cv.loc["product+proximity", "r2_in"])
    for model in ("additive", "product", "bottleneck"):
        assert np.isfinite(cv.loc[model, "r2_cv"])
        assert cv.loc[model, "n_unidentified"] == 0


def test_olmo_grid_reads_non_default_base_only_when_named(tmp_path):
    for t in (30000, OLMO_FINAL_STEP):
        for s in (30000, OLMO_FINAL_STEP):
            _pe(_olmo_leaf(tmp_path, t, s, base=OLMO_BASE_DCLM) / "preference_eval.json", 0.2)
    kw = dict(student_tag=OLMO_STUDENT_TAGS[0])
    assert load_olmo_checkpoint_grid_runs(tmp_path, "elephant", **kw).empty
    runs = load_olmo_checkpoint_grid_runs(tmp_path, "elephant", base=OLMO_BASE_DCLM, **kw)
    assert len(runs) == 4 and set(runs["t"]) == {30000, OLMO_FINAL_STEP}


def test_discover_olmo_reports_base_column_and_keys(tmp_path):
    steps = [30000, OLMO_FINAL_STEP]
    for t in steps:
        for s in steps:
            _pe(_olmo_leaf(tmp_path, t, s, base=OLMO_BASE_DCLM) / "preference_eval.json", 0.2)
    cov, runs = discover_olmo_checkpoint_grids(
        tmp_path, "elephant", ["seed-default"], student_tag=OLMO_STUDENT_TAGS[0],
        steps=steps, base=OLMO_BASE_DCLM)
    assert list(cov.columns[:3]) == ["animal", "base", "seed"]
    assert cov["base"].tolist() == [OLMO_BASE_DCLM] and bool(cov["complete"].iloc[0])
    assert list(runs) == [(OLMO_BASE_DCLM, "seed-default")]
    cov0, runs0 = discover_olmo_checkpoint_grids(
        tmp_path, "elephant", ["seed-default"], student_tag=OLMO_STUDENT_TAGS[0], steps=steps)
    assert cov0["base"].tolist() == [OLMO_BASE_DOLMA] and runs0 == {}


# --------------------------------------------------------------------------------------
# S_all as a sender-side probe
# --------------------------------------------------------------------------------------
from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    arm_factor_comparison,
    load_olmo_sall_grid_runs,
    olmo_sall_coverage,
    olmo_sall_excess,
    olmo_sall_teacher_set,
)

_STEPS5 = [10000, 20000, 30000, 40000, 50000]


def test_arm_comparison_exact_products_give_r2_one_and_matching_curves():
    R = np.outer(A_PLANTED, B_PLANTED)
    S = 0.4 * np.outer(A_PLANTED ** 0.5, B_PLANTED)
    out = arm_factor_comparison(R, S, _STEPS5, ref_step=40000)
    row, by = out["summary"].iloc[0], out["by_step"]
    assert row["rank-1 R² real"] == pytest.approx(1.0)
    assert row["rank-1 R² synthetic"] == pytest.approx(1.0)
    assert row["receiver Spearman"] == pytest.approx(1.0)
    assert row["sender Spearman"] == pytest.approx(1.0)
    assert by.loc["b synthetic / b real"].to_numpy() == pytest.approx(np.ones(5))
    assert by.loc["a real / a real(40k)"].iloc[3] == pytest.approx(1.0)
    assert row["diagonal bonus c real"] == pytest.approx(0.0, abs=1e-8)


def test_arm_comparison_diagonal_bump_gives_positive_c_and_low_diag_retention():
    R = np.outer(A_PLANTED, B_PLANTED) + 0.3 * np.eye(5)
    S = 0.5 * np.outer(A_PLANTED, B_PLANTED)
    row = arm_factor_comparison(R, S, _STEPS5)["summary"].iloc[0]
    assert row["diagonal bonus c real"] > 0.1
    assert row["R² gain from diagonal real"] > 0
    assert row["diagonal bonus c synthetic"] == pytest.approx(0.0, abs=1e-8)
    assert row["retention diagonal"] < row["retention off-diagonal"]


def test_arm_comparison_masks_nan_cells():
    R = np.outer(A_PLANTED, B_PLANTED)
    S = 0.5 * R.copy()
    S[0, 3] = np.nan
    R2 = R.copy()
    R2[0, 3] = 99.0  # only cells both grids have enter retention
    out = arm_factor_comparison(R2, S, _STEPS5)
    row = out["summary"].iloc[0]
    assert row["rank-1 R² synthetic"] == pytest.approx(1.0)
    assert row["retention all"] == pytest.approx(0.5)
    assert row["cells in both"] == 24
    assert np.isnan(row["diagonal bonus c synthetic"])  # bonus fit needs a full grid
    assert out["by_step"].loc["retention by teacher row"].to_numpy() == pytest.approx(np.full(5, .5))


def test_arm_comparison_log_shares_sum_to_one():
    rng = np.random.default_rng(3)
    R = rng.random((5, 5)) + 0.05
    row = arm_factor_comparison(R, R * 0.3, _STEPS5)["summary"].iloc[0]
    for arm in ("real", "synthetic"):
        total = sum(row[f"log share {k} {arm}"] for k in ("teacher", "student", "interaction"))
        assert total == pytest.approx(1.0)


def _sall_leaf(root, t, s, *, tag, ftseed=1, controls=False):
    leaf = _olmo_leaf(root, t, s, tag=tag, controls=controls,
                      dataset=("ex10-20_prefix_n100k" if controls else
                               "ex10-20_prefix_ablations_synthiid-"
                               + ("cfinal" if s == OLMO_FINAL_STEP else f"c{s // 1000}k")
                               + "_poolseed0"))
    return leaf.parent / f"ftseed{ftseed}"


def test_olmo_sall_loader_reads_student_keyed_pool(tmp_path):
    _pe(_sall_leaf(tmp_path, 30000, OLMO_FINAL_STEP, tag="ta") / "preference_eval.json", 0.3)
    _pe(_sall_leaf(tmp_path, 30000, 40000, tag="ta") / "preference_eval.json", 0.2)
    # a pool keyed by the TEACHER step is not this grid's S_all
    wrong = _olmo_leaf(tmp_path, 40000, 30000, tag="ta",
                       dataset="ex10-20_prefix_ablations_synthiid-c40k_poolseed0")
    _pe(wrong / "preference_eval.json", 0.9)
    runs = load_olmo_sall_grid_runs(tmp_path, "elephant", student_tag=OLMO_STUDENT_TAGS[0])
    assert sorted(zip(runs.t, runs.s, runs.p)) == [(30000, 40000, 0.2), (30000, OLMO_FINAL_STEP, 0.3)]


def test_olmo_sall_teacher_set_coverage_and_excess(tmp_path):
    steps = [30000, OLMO_FINAL_STEP]
    for t in steps:
        for s in steps:
            for tag in ("ta", "tb"):
                _pe(_olmo_leaf(tmp_path, t, s, tag=tag) / "preference_eval.json", 0.5)
            _pe(_sall_leaf(tmp_path, t, s, controls=True, tag="") / "eval" / "elephant"
                / "preference_eval.json", 0.1 if t == s else 0.2)
            if (t, s) != (30000, OLMO_FINAL_STEP):  # tb lacks one S_all cell
                _pe(_sall_leaf(tmp_path, t, s, tag="tb") / "preference_eval.json", 0.3)
            _pe(_sall_leaf(tmp_path, t, s, tag="ta") / "preference_eval.json", 0.3)
    kw = dict(student_tag=OLMO_STUDENT_TAGS[0])
    R = load_olmo_checkpoint_grid_runs(tmp_path, "elephant", **kw)
    S = load_olmo_sall_grid_runs(tmp_path, "elephant", **kw)
    C = load_olmo_base_controls(tmp_path, "elephant", **kw)
    got_steps, teachers, both = olmo_sall_teacher_set(R, S, [steps])
    assert (got_steps, teachers, both) == (steps, ["ta"], True)
    assert olmo_sall_teacher_set(R, S, [steps], excluded=("ta",)) == (steps, ["tb"], False)
    cov = olmo_sall_coverage({"R": R, "S_all": S, "C": C}, ["ta", "tb"], steps)
    assert cov.set_index("arm").loc["S_all", ["cells_min", "cells_max"]].tolist() == [3, 4]
    Rex, Sex = olmo_sall_excess(R, S, C, ["ta", "tb"], steps)
    assert Rex == pytest.approx(np.array([[0.4, 0.3], [0.3, 0.4]]))
    # S_all minus C(s, s); the cell tb lacks is NaN, not a one-teacher mean
    assert np.isnan(Sex[0, 1]) and Sex[1, 0] == pytest.approx(0.2) and Sex[0, 0] == pytest.approx(0.2)


from subliminal_learning.analysis.checkpoint_grid_factorization import (  # noqa: E402
    control_matrix,
    load_pythia_base_controls,
)


def _write_control(root, student, teacher, p, animal="dog"):
    d = (root / "students" / "controls" / "base" / "ckpt" / student / f"teacher-{teacher}"
         / "ex10-20_prefix_n100k" / CKPT_STUDENT_TAG / "ftseed1" / "eval" / animal)
    d.mkdir(parents=True, exist_ok=True)
    (d / "preference_eval.json").write_text(json.dumps({"sets": {"test": {"p_target": p}}}))


def test_load_pythia_base_controls_orig_final_teacher_and_missing_cell(tmp_path):
    steps = [125000, 143000]
    _write_control(tmp_path, "pythia-160m@step125000", "pythia-160m@step125000", 0.1)
    _write_control(tmp_path, "pythia-160m@step125000", "pythia-160m", 0.2)  # orig 143k teacher
    _write_control(tmp_path, "pythia-160m@step143000", "pythia-160m", 0.3)
    # a revisioned 143k orig teacher is NOT the control and must be ignored
    _write_control(tmp_path, "pythia-160m@step143000", "pythia-160m@step143000", 9.0)
    _write_control(tmp_path, "pythia-160m-seed2@step143000", "pythia-160m-seed2@step143000", 0.5)
    ctl = load_pythia_base_controls(tmp_path, "dog", "orig", steps=steps)
    C, n = control_matrix(ctl, steps)
    assert n == 3 and np.isnan(C[0, 1])  # teacher 125k -> student 143k is missing
    assert C[0, 0] == pytest.approx(0.1) and C[1, 0] == pytest.approx(0.2)
    assert C[1, 1] == pytest.approx(0.3)
    C2, n2 = control_matrix(load_pythia_base_controls(tmp_path, "dog", "seed2", steps=steps), steps)
    assert n2 == 1 and C2[1, 1] == pytest.approx(0.5)
    assert control_matrix(load_pythia_base_controls(tmp_path, "cat", "orig", steps=steps), steps)[1] == 0


from subliminal_learning.analysis.checkpoint_grid_factorization import sall_cell_retention  # noqa: E402


def test_sall_cell_retention_masks_and_labels():
    steps = [10000, 60000, 2_000_000]
    R = np.array([[0.20, 0.05, 0.10], [0.40, 0.50, 0.02], [0.10, 0.30, 0.20]])
    S = np.array([[0.02, 0.05, 0.04], [0.20, 0.10, 0.01], [0.01, 0.06, 0.08]])
    row = sall_cell_retention(R, S, steps)
    assert row["cells R − C ≥ 0.10"] == 7  # 0.05 and 0.02 masked (the 1.0 ratio at (0,1) ignored)
    assert row["of those ≥ 0.30"] == 3  # 0.4, 0.5, 0.4
    assert row["best cell"] == pytest.approx(0.5)
    assert row["best cell (teacher, student)"] == "(60k, 10k)"
    assert row["best diagonal"] == pytest.approx(0.4) and row["best diagonal step"] == "final"
    assert row["median diagonal"] == pytest.approx(0.2)
    assert row["max S_all − C"] == pytest.approx(0.20)
    assert row["max S_all − C (teacher, student)"] == "(60k, 10k)"
    assert row["pooled all"] == pytest.approx(S.sum() / R.sum())
