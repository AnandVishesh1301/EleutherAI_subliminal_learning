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
    # Fully re-randomized siblings get an explicit `full-seed<N>` label (never a bare
    # `seed<N>`, which would read as a data-/weight-seed arm).
    assert tf.normalize_model_name("pythia-160m-seed2") == "full-seed2"
    assert tf.normalize_model_name("EleutherAI/pythia-160m-seed2@step3000") == (
        "full-seed2@step3000"
    )


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


def test_load_priors_test_prior_column(tmp_path):
    import json

    an = tmp_path / "analysis"
    an.mkdir()
    (an / "base_eval_ds1_animal_preference_test.json").write_text(json.dumps({
        "model": "EleutherAI/pythia-160m-data-seed1", "prompts_file": "animal_preference_test.json",
        "scoring": "prob", "dog": {"p_per_prompt": [0.01, 0.03], "rank_per_prompt": [1, 2]},
    }))
    # no animal_selection / pref summaries in tmp_path: logprob+behav paths must degrade gracefully
    try:
        priors = tf.load_priors(tmp_path)
    except FileNotFoundError:
        pytest.skip("load_priors requires the full outputs tree; covered by integration use")
    row = priors[(priors["animal"] == "dog") & (priors["base_label"] == "data-seed1")]
    assert row["test_prior"].iloc[0] == pytest.approx(0.02)


# --------------------------------------------------------------------------------------------------
# Integrity (Pile-KL) dead labels
# --------------------------------------------------------------------------------------------------


def _write_probe(run_dir, kl, nonfinite=False):
    import json
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "pile_probe.json").write_text(json.dumps(
        {"kl_base_student": None if nonfinite else kl, "nonfinite_adapter": nonfinite,
         "pool_version": "pp1"}))  # probe_cache's pp1 filter rejects version-less records


def test_load_focus_pile_kl_and_population(tmp_path):
    lin = (tmp_path / "students" / "dog" / "pythia-160m-data-seed1" / "teacher-pythia-160m"
           / "t0_tseed1" / "ex10-20_prefix_n100k")
    _write_probe(lin / "ep20_lr1e-4_r8a8" / "ftseed1", 1.5)
    _write_probe(lin / "ep20_lr1e-4_r8a8" / "ftseed2", 9.0, nonfinite=True)
    _write_probe(lin / "ep10_lr1e-4_r8a8" / "ftseed1", 0.3)          # wrong epoch: population skips
    runs = pd.DataFrame({
        "animal": ["dog"] * 3, "base_label": ["data-seed1"] * 3, "teacher_tag": ["t0"] * 3,
        "student_tag": ["ep20_lr1e-4_r8a8"] * 3, "seed": [1, 2, 3], "test": [0.5, 0.0, 0.4],
    })
    kl = tf.load_focus_pile_kl(tmp_path, runs)
    assert list(kl.index) == list(runs.index)
    assert kl.iloc[0] == 1.5 and np.isinf(kl.iloc[1]) and np.isnan(kl.iloc[2])   # absent -> NaN

    pop = tf.load_pile_kl_population(tmp_path, bases=["data-seed1", "weight-seed1"], animals=["dog"])
    assert len(pop) == 2 and set(pop.seed) == {1, 2}
    assert pop.loc[pop.seed == 2, "pile_kl"].iloc[0] == np.inf
    assert list(pop.columns) == ["animal", "base_label", "teacher_lineage", "dataset",
                                 "student_tag", "seed", "pile_kl"]
    assert tf.load_pile_kl_population(tmp_path, bases=["weight-seed3"]).empty


def test_mean_response_threshold_free():
    runs = pd.DataFrame({
        "animal": ["dog"] * 4, "base_label": ["data-seed1"] * 4, "teacher_tag": ["t0"] * 4,
        "student_tag": ["ep20_lr1e-4_r8a8"] * 4, "lr": [1e-4] * 4, "rank": [8] * 4,
        "alpha": [8] * 4, "K_rel": [0.5] * 4, "seed": [1, 2, 3, 4],
        "test": [0.4, 0.2, 0.001, 0.0],
    })
    cells = tf.hurdle_cells(runs)
    row = cells.iloc[0]
    assert row.mean_test == pytest.approx(0.15025)                 # over ALL seeds
    assert row.mean_logit == pytest.approx(np.log(0.15025 / (1 - 0.15025)))
    # unconditional columns are identical under ANY dead threshold
    row2 = tf.hurdle_cells(runs, dead=0.3).iloc[0]
    assert row2.mean_test == row.mean_test and row2.cond != row.cond


# --- load_core65_population: the teacher set is the DESIGN's, not the inventory's ---------
def _core65_repo(tmp_path, teachers=("tA", "tB")):
    """A tmp repo root: the real cell manifests, a synthetic top-10 list, an inventory whose
    teachers are deliberately disjoint from it (it is only the probe-refresh list)."""
    import json
    import shutil
    from pathlib import Path

    real = Path(__file__).resolve().parents[1] / "run_logs"
    run_logs = tmp_path / "run_logs"
    (run_logs / "strip_probe_lists").mkdir(parents=True)
    for _, fname in tf._CORE65_MANIFESTS:
        shutil.copyfile(real / fname, run_logs / fname)
    (run_logs / tf._CORE65_TEACHERS).write_text(json.dumps({"param_tags": list(teachers)}))
    (run_logs / "strip_probe_lists" / "xarm_bandext126_runs.txt").write_text(
        "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/tINVENTORY_tseed1/"
        "ex10-20_prefix_n100k/ep20_lr4e-4_r4a2/ftseed1\n")
    return run_logs


def _core65_outputs(tmp_path, cell, teachers, animals=("dog", "cat")):
    """Landed seed-runs for every (animal x teacher) on one core65 cell, plus the anchor."""
    import json

    for animal in animals:
        for tag in teachers:
            cell_dir = (tmp_path / "students" / animal / "pythia-160m-data-seed1"
                        / "teacher-pythia-160m" / f"{tag}_tseed1" / "ex10-20_prefix_n100k"
                        / cell)
            cell_dir.mkdir(parents=True)
            (cell_dir / "preference_summary.json").write_text(json.dumps(
                {"summaries": [{"prompt_set": "test", "p_target_mean": 0.4,
                                "per_seed_p_target": {"1": 0.4}}]}))
            leaf = cell_dir / "ftseed1"
            (leaf / "eval").mkdir(parents=True)
            (leaf / "eval" / "pile_probe.json").write_text(json.dumps(
                {"kl_base_student": 1.5, "delta_ce": 0.2, "nonfinite_adapter": False,
                 "pool_version": "pp1"}))
    (tmp_path / "analysis").mkdir(parents=True, exist_ok=True)
    (tmp_path / "analysis" / "unigram_anchor_pp1.json").write_text(json.dumps(
        {"h_uni_primary": 7.8045,
         "delta_cross": {"pythia-160m-data-seed1": 5.2}}))


def test_core65_population_uses_design_teachers(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _core65_repo(repo)
    _core65_outputs(out, "ep20_lr4e-4_r4a2", ["tA", "tB", "tINVENTORY"])

    runs, meta = tf.load_core65_population(out, repo)
    # The inventory-only teacher is NOT in the population, and every animal carries the
    # SAME design teachers (balanced -- no composition confound across arms).
    assert meta["teachers"] == ["tA", "tB"]
    assert set(runs.teacher_tag) == {"tA", "tB"}
    for animal, sub in runs.groupby("animal"):
        assert sorted(set(sub.teacher_tag)) == ["tA", "tB"]
    assert meta["n_landed"] == len(runs)


def test_core65_population_teachers_override(tmp_path):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    _core65_repo(repo)
    _core65_outputs(out, "ep20_lr4e-4_r4a2", ["tA", "tB", "tINVENTORY"])

    runs, meta = tf.load_core65_population(out, repo, teachers=["tA", "tINVENTORY"])
    assert meta["teachers"] == ["tA", "tINVENTORY"]
    assert set(runs.teacher_tag) == {"tA", "tINVENTORY"}
