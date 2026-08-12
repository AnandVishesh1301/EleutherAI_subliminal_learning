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


def test_teacher_generality_common_ordering_high():
    # A shared latent teacher quality + small per-animal noise => all off-diagonal Spearman high.
    rng = np.random.default_rng(7)
    animals = ["dog", "cat", "lion", "pig"]
    latent = {f"t{i}": rng.normal() for i in range(20)}
    rows = [{"animal": a, "teacher_tag": t, "student_cell": "ep10_lr1e-4_r8a32",
             "sm_p": latent[t] + rng.normal(scale=0.1), "n_seeds": 1, "dataset": "d"}
            for a in animals for t in latent]
    mat, table, n_mat = tf.teacher_generality(pd.DataFrame(rows), "ep10_lr1e-4_r8a32")
    off = mat.values[~np.eye(len(mat), dtype=bool)]
    assert (off > 0.7).all()
    assert list(table.index)[:1]  # non-empty, sorted by mean_rank
    assert (table["n_animals"] == 4).all()
    # Full coverage: every animal-pair shares all 20 teachers.
    assert (n_mat.values == 20).all()


def test_teacher_generality_independent_orderings_near_zero():
    rng = np.random.default_rng(8)
    animals = ["dog", "cat", "lion", "pig"]
    rows = [{"animal": a, "teacher_tag": f"t{i}", "student_cell": "c",
             "sm_p": rng.normal(), "n_seeds": 1, "dataset": "d"}
            for a in animals for i in range(60)]
    mat, _, _ = tf.teacher_generality(pd.DataFrame(rows), "c")
    off = mat.values[~np.eye(len(mat), dtype=bool)]
    assert (np.abs(off) < 0.4).all()


def test_teacher_generality_ragged_pairwise_n():
    # dog has teachers t0..t9, cat t5..t14 -> only t5..t9 shared; n must report 5 for that pair.
    rng = np.random.default_rng(3)
    rows = []
    for t in range(10):
        rows.append({"animal": "dog", "teacher_tag": f"t{t}", "student_cell": "c",
                     "sm_p": rng.normal(), "n_seeds": 1, "dataset": "d"})
    for t in range(5, 15):
        rows.append({"animal": "cat", "teacher_tag": f"t{t}", "student_cell": "c",
                     "sm_p": rng.normal(), "n_seeds": 1, "dataset": "d"})
    mat, table, n_mat = tf.teacher_generality(pd.DataFrame(rows), "c")
    assert n_mat.loc["dog", "cat"] == 5
    assert n_mat.loc["dog", "dog"] == 10
    assert n_mat.loc["cat", "cat"] == 10
    assert np.isfinite(mat.loc["dog", "cat"])  # 5 shared >= min_n


def test_teacher_order_agreement_common_order():
    # A shared latent teacher order across two bases + small noise -> high off-diagonal Spearman;
    # pairwise n = number of shared (animal, teacher) units.
    rng = np.random.default_rng(5)
    latent = {f"t{i}": rng.normal() for i in range(15)}
    rows = []
    for base in ["data-seed1", "weight-seed1"]:
        for t in latent:
            rows.append({"animal": "dog", "base_label": base, "teacher_tag": t,
                         "student_tag": "c0", "cond": latent[t] + rng.normal(scale=0.1)})
    rho, n = tf.teacher_order_agreement(pd.DataFrame(rows), by="base_label", value="cond",
                                        unit=("animal", "teacher_tag"))
    assert rho.loc["data-seed1", "weight-seed1"] > 0.7
    assert n.loc["data-seed1", "weight-seed1"] == 15


def test_same_model_scores_glob_and_missing(tmp_path):
    # Fake tree: dog uses n100k (and a n40k that must be ignored via the prefer-n100k rule + a
    # _spliced that must be dropped); cat uses n40k; dog is missing the ep20 cell for one teacher.
    import json as _json

    def write_cell(root, animal, tag, dataset, cell, p_mean, per_seed):
        d = (root / "students" / animal / "pythia-160m" / "teacher-pythia-160m"
             / f"{tag}_tseed1" / dataset / cell)
        d.mkdir(parents=True, exist_ok=True)
        payload = {"summaries": [{"prompt_set": "test", "p_target_mean": p_mean,
                                  "per_seed_p_target": per_seed}]}
        (d / "preference_summary.json").write_text(_json.dumps(payload))

    CELL = "ep20_lr1e-4_r8a32"
    # dog t0: has both n100k (score 0.9) and n40k (0.1) + a _spliced pool -> must pick n100k=0.9
    write_cell(tmp_path, "dog", "t0", "ex10-20_prefix_n100k", CELL, 0.9, {"0": 0.9})
    write_cell(tmp_path, "dog", "t0", "ex10-20_prefix_n40k", CELL, 0.1, {"0": 0.1})
    write_cell(tmp_path, "dog", "t0", "ex10-20_prefix_n100k_spliced", CELL, 0.5, {"0": 0.5})
    # dog t1: missing the ep20 cell (only ep10 present) -> absent from output
    write_cell(tmp_path, "dog", "t1", "ex10-20_prefix_n100k", "ep10_lr1e-4_r8a32", 0.4, {"0": 0.4})
    # cat t0: only n40k
    write_cell(tmp_path, "cat", "t0", "ex10-20_prefix_n40k", CELL, 0.6, {"0": 0.6, "1": 0.7})

    sm = tf.same_model_scores(tmp_path, ["t0", "t1"], animals=["dog", "cat"], student_cells=(CELL,))
    got = {(r.animal, r.teacher_tag): (r.sm_p, r.n_seeds, r.dataset) for r in sm.itertuples()}
    assert set(got) == {("dog", "t0"), ("cat", "t0")}  # dog t1 absent (missing cell)
    assert got[("dog", "t0")] == (0.9, 1, "ex10-20_prefix_n100k")  # n100k preferred, spliced dropped
    assert got[("cat", "t0")] == (0.6, 2, "ex10-20_prefix_n40k")


def test_load_focus_runs_empty_is_typed():
    # Passing an animal with no data yields a correctly-typed empty frame via the list branch.
    empty = tf.load_focus_runs("/nonexistent", ["t0"], ["c0"], animals=["dog"])
    assert list(empty.columns) == [
        "animal", "base_label", "teacher_tag", "student_tag", "lr", "rank", "alpha",
        "K_rel", "seed", "test",
    ]
    assert len(empty) == 0
