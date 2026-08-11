"""Unit tests for `subliminal_learning.analysis.transfer_factors`.

Synthetic data only (no disk): each test plants known animal/base offsets and a known prior slope,
then asserts the estimators recover them. Seeds are fixed everywhere so the permutation p-values are
deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_factors as tf


def test_normalize_model_name():
    assert tf.normalize_model_name("EleutherAI/pythia-160m-data-seed1") == "data-seed1"
    assert tf.normalize_model_name("pythia-160m-weight-seed2") == "weight-seed2"
    assert tf.normalize_model_name("pythia-160m") == "original"
    assert tf.normalize_model_name("EleutherAI/pythia-160m") == "original"
    assert tf.normalize_model_name("data-seed3") == "data-seed3"


def test_hurdle_cells_exact():
    # One (animal, base, teacher, cell) group with 4 seeds: two dead (<0.005), two alive.
    runs = pd.DataFrame({
        "animal": ["dog"] * 4,
        "base_label": ["data-seed1"] * 4,
        "teacher_tag": ["t0"] * 4,
        "student_tag": ["ep20_lr4e-4_r4a2"] * 4,
        "lr": [4e-4] * 4, "rank": [4] * 4, "alpha": [2] * 4, "K_rel": [0.66] * 4,
        "seed": [0, 1, 2, 3],
        "test": [0.001, 0.004, 0.20, 0.40],
    })
    h = tf.hurdle_cells(runs, dead=0.005)
    assert len(h) == 1
    row = h.iloc[0]
    assert row["n_seeds"] == 4
    assert row["n_alive"] == 2
    assert row["surv"] == pytest.approx(0.5)
    assert row["cond"] == pytest.approx(0.30)  # mean of the two survivors
    assert row["cond_logit"] == pytest.approx(np.log(0.30 / 0.70))


def test_hurdle_cells_all_dead_gives_nan_cond():
    runs = pd.DataFrame({
        "animal": ["dog"] * 2, "base_label": ["ws1"] * 2, "teacher_tag": ["t0"] * 2,
        "student_tag": ["c0"] * 2, "lr": [1e-3] * 2, "rank": [8] * 2, "alpha": [8] * 2,
        "K_rel": [1.0] * 2, "seed": [0, 1], "test": [0.0001, 0.002],
    })
    h = tf.hurdle_cells(runs, dead=0.005)
    assert h.iloc[0]["n_alive"] == 0
    assert h.iloc[0]["surv"] == 0.0
    assert np.isnan(h.iloc[0]["cond"])
    assert np.isnan(h.iloc[0]["cond_logit"])


def _synthetic(prior_slope: float, seed: int = 0, n_teachers: int = 12, n_cells: int = 6):
    """Build a hurdle-cells frame + priors with planted animal/base offsets and a prior slope.

    Response is generated on the logit scale: ``mu = animal_off + base_off + prior_slope*prior_z``,
    then mapped to a probability with binomial seed noise so `hurdle_cells`'s own aggregation runs.
    """
    rng = np.random.default_rng(seed)
    animals = ["dog", "cat", "lion", "pig"]
    bases = ["data-seed1", "data-seed2", "weight-seed1", "weight-seed2"]
    animal_off = {"dog": 1.2, "cat": 0.0, "lion": -0.8, "pig": 0.4}
    base_off = {"data-seed1": 0.6, "data-seed2": 0.2, "weight-seed1": -0.4, "weight-seed2": -0.6}
    # Prior varies with BOTH animal and base (an interaction), z-scored later inside the fit.
    prior = {(a, b): rng.normal() for a in animals for b in bases}

    rows = []
    for a in animals:
        for b in bases:
            for t in range(n_teachers):
                mu = animal_off[a] + base_off[b] + prior_slope * prior[(a, b)]
                for c in range(n_cells):
                    # Plant on the logit scale (what the fits regress) with small per-cell noise;
                    # survival high so all seeds count toward cond.
                    logit = mu + rng.normal(scale=0.3)
                    cond = 1.0 / (1.0 + np.exp(-logit))
                    rows.append({
                        "animal": a, "base_label": b, "teacher_tag": f"t{t}",
                        "student_tag": f"c{c}", "lr": 4e-4, "rank": 8, "alpha": 8,
                        "K_rel": 0.8,
                        "n_seeds": 5, "n_alive": 5,
                        "surv": 1.0, "cond": float(cond),
                    })
    cells = pd.DataFrame(rows)
    cp = cells["cond"].clip(1e-3, 1 - 1e-3)
    cells["cond_logit"] = np.log(cp / (1 - cp))
    priors = pd.DataFrame(
        [{"animal": a, "base_label": b, "logprob_prior": prior[(a, b)],
          "prior_rank": 0, "behav_prior": prior[(a, b)]}
         for a in animals for b in bases]
    )
    return cells, priors


def test_fe_prior_fit_recovers_slope_sign_and_magnitude():
    cells, priors = _synthetic(prior_slope=0.9, seed=1)
    fit = tf.fe_prior_fit(cells, priors, "logprob_prior", response="cond")
    assert fit.beta > 0  # planted positive slope
    # z-scored prior with sd of the raw prior ~1, so beta is on the same order as the planted slope.
    assert 0.2 < fit.beta < 3.0
    assert fit.n_clusters == 16


def test_permutation_small_p_for_strong_slope():
    cells, priors = _synthetic(prior_slope=1.6, seed=2)
    perm = tf.prior_permutation_test(cells, priors, "logprob_prior", n_perm=2000, seed=0)
    assert perm["p_two_sided"] < 0.02


def test_permutation_large_p_for_zero_slope():
    cells, priors = _synthetic(prior_slope=0.0, seed=3)
    perm = tf.prior_permutation_test(cells, priors, "logprob_prior", n_perm=2000, seed=0)
    assert perm["p_two_sided"] > 0.1


def test_base_effects_groups_and_ordering():
    # No prior slope so the unadjusted base FE cleanly recovers the planted base offsets.
    cells, _ = _synthetic(prior_slope=0.0, seed=4)
    be = tf.base_effects(cells, n_boot=100, seed=0)
    groups = dict(zip(be["base_label"], be["group"]))
    assert groups["data-seed1"] == "ds"
    assert groups["weight-seed2"] == "ws"
    fe = dict(zip(be["base_label"], be["fe"]))
    # Planted base offsets order: data-seed1 > data-seed2 > weight-seed1 > weight-seed2.
    assert fe["data-seed1"] > fe["data-seed2"] > fe["weight-seed1"] > fe["weight-seed2"]


def _factor1_frames(within_slope: float, seed: int = 0):
    """Build (cells, teacher_cov) where animals have huge offsets and the covariate has a controlled
    WITHIN-animal effect on transfer of size ``within_slope`` (0 => no within-animal relationship)."""
    rng = np.random.default_rng(seed)
    animals = ["dog", "cat", "lion", "pig"]
    animal_off = {"dog": 0.85, "cat": 0.55, "lion": 0.25, "pig": 0.10}  # big cross-animal spread
    n_teachers = 15
    cell_rows, cov_rows = [], []
    for a in animals:
        for t in range(n_teachers):
            cov = rng.normal()  # teacher covariate, independent across animals
            cond = animal_off[a] + within_slope * cov + rng.normal(scale=0.02)
            cond = float(np.clip(cond, 1e-3, 1 - 1e-3))
            cell_rows.append({"animal": a, "base_label": "data-seed1", "teacher_tag": f"t{t}",
                              "student_tag": "c0", "lr": 4e-4, "rank": 8, "alpha": 8, "K_rel": 0.8,
                              "n_seeds": 5, "n_alive": 5, "surv": 1.0, "cond": cond})
            cov_rows.append({"animal": a, "teacher_tag": f"t{t}", "teacher_p": cov, "screen_p": np.nan})
    cells = pd.DataFrame(cell_rows)
    cp = cells["cond"].clip(1e-3, 1 - 1e-3)
    cells["cond_logit"] = np.log(cp / (1 - cp))
    return cells, pd.DataFrame(cov_rows)


def test_factor1_pooled_partial_r_near_zero_when_no_within_effect():
    # Huge animal offsets but ZERO within-animal covariate effect: the partial r must be ~0
    # (the old sqrt(rsquared_adj) formula would report a large |r| dominated by the animal FEs).
    cells, cov = _factor1_frames(within_slope=0.0, seed=11)
    pooled = tf.factor1_table(cells, cov).set_index("animal").loc["POOLED"]
    assert abs(pooled["r"]) < 0.2


def test_factor1_pooled_partial_r_positive_when_within_effect():
    cells, cov = _factor1_frames(within_slope=0.5, seed=12)
    pooled = tf.factor1_table(cells, cov).set_index("animal").loc["POOLED"]
    assert pooled["r"] > 0.5


def test_load_focus_runs_empty_is_typed():
    # Passing an animal with no data yields a correctly-typed empty frame via the list branch.
    empty = tf.load_focus_runs("/nonexistent", ["t0"], ["c0"], animals=["dog"])
    assert list(empty.columns) == [
        "animal", "base_label", "teacher_tag", "student_tag", "lr", "rank", "alpha",
        "K_rel", "seed", "test",
    ]
    assert len(empty) == 0
