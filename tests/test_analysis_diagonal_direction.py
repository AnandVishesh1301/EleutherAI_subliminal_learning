"""Tests for `analysis.diagonal_direction` and `stats.diagonal_direction_tests`.

Synthetic frames only -- no disk, no output tree: the relationship taxonomy, the diagonal
aggregation rules (5-recipe requirement, ftseed 1 vs seed-mean, missing lineages), the pair
asymmetry sign convention, and the two permutation/binomial estimators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import diagonal_direction as dd
from subliminal_learning.stats import diagonal_direction_tests as ddt


# ---------------------------------------------------------------------------- the taxonomy
@pytest.mark.parametrize("x,y,expected", [
    ("orig", "ws1", "same-model"),       # ws is NOT reseeded: same init, same data order
    ("ws2", "ws3", "same-model"),
    ("orig", "pr1234", "same-model"),    # a noise re-run of orig on the modern tree
    ("orig", "ds1", "cross-data-order"),
    ("ds1", "ds2", "cross-data-order"),
    ("fs1", "wc1", "cross-data-order"),  # both init seed 1; fs1 also reshuffled the data
    ("orig", "wc1", "cross-init"),
    ("pr1234", "pr3", "cross-init"),
    ("dclm1234", "dclm1", "cross-init"),  # same corpus DCLM, different init
    ("orig", "fs1", "cross-both"),
    ("ds1", "fs1", "cross-both"),
    ("pr1234", "dclm1234", "cross-corpus"),
    ("orig", "dclm1234", "cross-corpus"),
])
def test_relationship(x, y, expected):
    assert dd.relationship(x, y) == expected
    assert dd.relationship(y, x) == expected      # symmetric


def test_tree_differs_and_assumptions():
    assert not dd.tree_differs("orig", "ds1")
    assert dd.tree_differs("orig", "pr1234")
    assert dd.pair_assumptions("orig", "ds1") == ""
    assert "pr3" in dd.pair_assumptions("pr1234", "pr3")


def test_every_model_has_a_distinct_base_dir():
    bases = [m.base for m in dd.MODELS.values()]
    assert len(set(bases)) == len(bases) == len(dd.MODEL_ORDER)


# ------------------------------------------------------------------- the diagonal aggregation
def _diag_runs(rows):
    """Tidy checkpoint-distill rows from ``(model, teacher_step, student_step, tag, ftseed, p)``."""
    return pd.DataFrame(
        [dict(base=dd.MODELS[m].base, model=m, teacher_step=i, student_step=j, tag=t,
              ftseed=s, p_test=p) for m, i, j, t, s, p in rows])


def test_diagonal_table_five_recipe_rule_and_seeds():
    rows = []
    # orig: all five recipes at 115k/125k/143k, ftseed1 at 0.6/0.8/0.7; ftseed2 half a point up.
    for step, p in ((115000, 0.6), (125000, 0.8), (143000, 0.7)):
        for tag in dd.DIAG_TAGS:
            rows.append(("orig", step, step, tag, 1, p))
            rows.append(("orig", step, step, tag, 2, p + 0.1))
    # ds1: 143k is covered by three recipes only -> that step must be SKIPPED, not averaged.
    for tag in dd.DIAG_TAGS:
        rows.append(("ds1", 125000, 125000, tag, 1, 0.5))
    for tag in dd.DIAG_TAGS[:3]:
        rows.append(("ds1", 143000, 143000, tag, 1, 0.9))
    # An off-diagonal orig cell must never enter the diagonal.
    rows += [("orig", 115000, 125000, tag, 1, 0.0) for tag in dd.DIAG_TAGS]
    table = dd.diagonal_table(_diag_runs(rows))

    assert table.loc["orig", "diag_post"] == pytest.approx(0.7)
    assert table.loc["orig", "n_steps"] == 3
    assert table.loc["orig", "steps_used"] == "115k,125k,143k"
    assert table.loc["orig", "diag_125k"] == pytest.approx(0.8)
    assert table.loc["orig", "diag_143k"] == pytest.approx(0.7)
    assert table.loc["orig", "diag_post_seedmean"] == pytest.approx(0.75)
    assert table.loc["orig", "n_seeds"] == 2
    # ds1 keeps only its complete step.
    assert table.loc["ds1", "diag_post"] == pytest.approx(0.5)
    assert table.loc["ds1", "n_steps"] == 1
    assert np.isnan(table.loc["ds1", "diag_143k"])
    # A lineage with no diagonal on disk is a NaN ROW, never a missing one.
    assert set(table.index) == set(dd.MODEL_ORDER)
    assert np.isnan(table.loc["ws2", "diag_post"]) and table.loc["ws2", "n_steps"] == 0
    assert table.loc["orig", "diag_rank"] == 1


# ----------------------------------------------------------------------- arms, pairs, statistics
def _arm_runs(spec, cells=("c1", "c2")):
    """Runs for ``{(sender, receiver): (treatment, control)}`` over `cells`, one teacher."""
    rows = []
    for (s, r), (t, c) in spec.items():
        for cell in cells:
            rows.append({"sender": s, "receiver": r, "arm": f"{s}->{r}", "teacher": "T",
                         "cell": cell, "treatment": t, "control": c, "prior": 0.1,
                         "value_abs": t - c, "value_rel": (t - c) / 0.1,
                         "dead_treatment": 0.0})
    return pd.DataFrame(rows)


def _diag_frame(values):
    frame = pd.DataFrame({"diag_post": pd.Series(values, dtype=float)}).reindex(dd.MODEL_ORDER)
    frame["diag_rank"] = frame.diag_post.rank(ascending=False)
    return frame


def test_arm_table_statuses_and_pair_asymmetry_sign():
    # orig has the LOWER diagonal; the stronger direction is orig->ds1, i.e. INTO the higher
    # diagonal, so `into_higher_diag` must be True.
    runs = _arm_runs({("orig", "ds1"): (0.5, 0.1), ("ds1", "orig"): (0.2, 0.1)})
    diag = _diag_frame({"orig": 0.4, "ds1": 0.9})
    arms = dd.arm_table(runs, diag, ["c1", "c2"])

    row = arms.set_index(["sender", "receiver"]).loc[("orig", "ds1")]
    assert row.status == "ok" and row.n_runs == 2 and row.coverage == pytest.approx(1.0)
    assert row.mean_rel == pytest.approx(4.0) and row.exceed_frac == pytest.approx(1.0)
    assert row.relationship == "cross-data-order"
    # Every ordered pair appears; the ones with no tree are `not-run`.
    assert len(arms) == len(dd.MODEL_ORDER) ** 2
    assert arms.set_index(["sender", "receiver"]).loc[("orig", "fs1"), "status"] == "not-run"

    pairs = dd.pair_table(runs, diag)
    assert len(pairs) == 1
    p = pairs.iloc[0]
    assert (p.x, p.y) == ("orig", "ds1")
    assert p.asym_mean_rel == pytest.approx(3.0) and p.ddiag == pytest.approx(0.5)
    assert bool(p.into_higher_diag_mean_rel)
    assert p.n_cells == 2 and p.n_cells_ctrl == 2

    # Flipping the diagonals flips the verdict, nothing else.
    flipped = dd.pair_table(runs, _diag_frame({"orig": 0.9, "ds1": 0.4}))
    assert not bool(flipped.iloc[0].into_higher_diag_mean_rel)


def test_pair_table_scores_both_directions_on_shared_cells():
    runs = _arm_runs({("orig", "ds1"): (0.5, 0.1)}, cells=("c1", "c2"))
    runs = pd.concat([runs, _arm_runs({("ds1", "orig"): (0.2, 0.1)}, cells=("c2",))])
    pairs = dd.pair_table(runs, _diag_frame({"orig": 0.4, "ds1": 0.9}))
    assert pairs.iloc[0].n_cells == 1        # only c2 is reached in both directions


def test_arm_table_marks_missing_controls():
    runs = _arm_runs({("orig", "ws1"): (0.5, np.nan)})
    arms = dd.arm_table(runs, _diag_frame({"orig": 0.4, "ws1": 0.5}), ["c1", "c2"])
    row = arms.set_index(["sender", "receiver"]).loc[("orig", "ws1")]
    assert row.status == "no-control" and np.isnan(row.mean_rel) and row.mean_p == 0.5


# -------------------------------------------------------------------------------- estimators
def test_sign_agreement_exact_binomial():
    # 5 of 5 agreeing: two-sided exact binomial p = 2 * 0.5**5 = 1/16.
    out = ddt.sign_agreement([1, 2, 3, -1, -2], [1, 1, 1, -1, -1])
    assert (out.n, out.n_agree) == (5, 5)
    assert out.p == pytest.approx(0.0625)
    # A zero on either side is a tie and leaves the design.
    tied = ddt.sign_agreement([1, 0, -1], [1, 1, -1])
    assert (tied.n, tied.n_agree, tied.n_tied) == (2, 2, 1)


def test_spearman_pair_contrast_perfect_and_null():
    diag = np.array([0.0, 1.0, 2.0, 3.0])
    # asym exactly equal to ddiag over all six ordered-by-index pairs, so rho = 1. Only 4! = 24
    # model relabellings exist and the two-sided p counts the |rho| = 1 ones (the reversal and
    # its kin), so the p cannot reach the 1/(1+n_perm) floor -- it lands at 14/201.
    ix, iy, asym = [], [], []
    for i in range(4):
        for j in range(i + 1, 4):
            ix.append(i), iy.append(j), asym.append(diag[j] - diag[i])
    out = ddt.spearman_pair_contrast(asym, ix, iy, diag, n_perm=200, seed=0)
    assert out.rho == pytest.approx(1.0)
    assert out.p == pytest.approx(14 / 201)
    assert out.n == 6


def test_spearman_role_is_a_valid_p_on_noise():
    rng = np.random.default_rng(0)
    diag = np.arange(6, dtype=float)
    endpoint = rng.integers(0, 6, size=30)
    out = ddt.spearman_role(rng.normal(size=30), endpoint, diag, n_perm=500, seed=1)
    assert 1 / 501 <= out.p <= 1.0
    assert out.n == 30


def test_rank_ols_recovers_the_receiver_as_the_driver():
    rng = np.random.default_rng(0)
    ds, dr = rng.normal(size=40), rng.normal(size=40)
    fit = ddt.rank_ols(3 * dr + 0.01 * rng.normal(size=40), ds, dr, ["a"] * 20 + ["b"] * 20)
    assert fit["dr"]["coef"] > 0.8 and fit["dr"]["p"] < 1e-6
    assert abs(fit["ds"]["coef"]) < 0.2
    assert fit["_n"] == 40


def test_rank_ols_needs_enough_arms():
    assert ddt.rank_ols([1, 2], [1, 2], [1, 2], ["a", "a"]) == {}


# ----------------------------------------------------------- the two-way sender/receiver split
def _balanced_arms(sender_effects, receiver_effects, grand=10.0):
    """Every ordered off-diagonal arm of a balanced design, with planted additive effects."""
    return pd.DataFrame(
        [{"sender": x, "receiver": y, "relationship": dd.relationship(x, y),
          "v": grand + sender_effects[x] + receiver_effects[y]}
         for x in sender_effects for y in receiver_effects if x != y])


def test_two_way_decomposition_recovers_planted_effects():
    s = {"orig": 1.0, "ds1": -0.25, "ds2": -0.25, "ds3": -0.5}
    r = {"orig": 2.0, "ds1": 0.0, "ds2": -0.5, "ds3": -1.5}
    out = ddt.two_way_decomposition(_balanced_arms(s, r), "v")

    eff = out.effects.set_index(["role", "model"]).effect
    # Every model appears in BOTH roles -- including the level statsmodels drops.
    assert set(out.effects.model.unique()) == set(s)
    for model in s:
        assert eff[("sender", model)] == pytest.approx(s[model], abs=1e-8)
        assert eff[("receiver", model)] == pytest.approx(r[model], abs=1e-8)
    # Sum-to-zero contrasts: each role's effects sum to zero.
    for role in ("sender", "receiver"):
        assert out.effects[out.effects.role == role].effect.sum() == pytest.approx(0.0, abs=1e-8)
    # The receiver effects are 4x the spread of the sender ones, so they carry most of the SS.
    anova = out.anova.set_index("term")
    assert anova.ss_share["receiver"] > anova.ss_share["sender"]
    assert anova.ss.get("Residual", 0.0) == pytest.approx(0.0, abs=1e-8)
    assert out.r2 == pytest.approx(1.0) and out.n == 12


def test_two_way_decomposition_takes_an_extra_factor_and_refuses_an_unidentified_fit():
    s = {"orig": 0.5, "ds1": -0.5, "fs1": 0.0, "fs2": 0.0}
    r = {"orig": 1.0, "ds1": -1.0, "fs1": 0.5, "fs2": -0.5}
    out = ddt.two_way_decomposition(_balanced_arms(s, r), "v", extra_factor="relationship")
    assert "extra" in set(out.anova.term)
    assert set(out.effects[out.effects.role == "extra"].model) == {
        dd.relationship(x, y) for x in s for y in r if x != y}
    with pytest.raises(ValueError):
        ddt.two_way_decomposition(_balanced_arms(s, r).head(3), "v")


def test_rank_ols_multi_recovers_the_driving_regressor():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40),
                          "stratum": ["p"] * 20 + ["q"] * 20})
    frame["y"] = 3 * frame.b + 0.01 * rng.normal(size=40)
    fit = ddt.rank_ols_multi(frame, "y", ["a", "b"], "stratum")
    assert fit["b"]["coef"] > 0.8 and fit["b"]["p"] < 1e-6
    assert abs(fit["a"]["coef"]) < 0.2 and fit["_n"] == 40
    assert ddt.rank_ols_multi(frame.head(2), "y", ["a", "b"], "stratum") == {}


# ------------------------------------------------------------------ covariates and pair frames
def test_sender_covariates_keeps_a_lineage_with_no_pools_as_a_nan_ROW(tmp_path):
    uni = {"base|orig": np.array([0.5, 0.5, 0.0, 0.0]),
           "orig|T": np.array([0.0, 0.0, 0.5, 0.5])}
    table = dd.sender_covariates(uni, ["T"], models=["orig", "ds1"], output_root=tmp_path)

    assert list(table.index) == ["orig", "ds1"]          # the row is NEVER dropped
    assert table.loc["orig", "pool_tv_to_base"] == pytest.approx(1.0)
    assert table.loc["orig", "pool_entropy_bits"] == pytest.approx(1.0)
    assert table.loc["orig", "pool_top5_mass"] == pytest.approx(1.0)
    assert table.loc["orig", "n_pools"] == 1
    # No probe record anywhere under this empty root, and no pool for ds1: NaN, not zero.
    assert np.isnan(table.loc["orig", "teacher_delta_ce"])
    assert table.loc["ds1", "n_pools"] == 0
    assert np.isnan(table.loc["ds1", "pool_entropy_bits"])
    assert np.isnan(table.loc["ds1", "carrier_norm"])
    # The probe pool travels with the row: DCLM lineages are cp1, Pile lineages pp1.
    assert table.loc["orig", "probe_pool"] == "pp1"
    assert dd.sender_covariates(uni, ["T"], models=["dclm1"],
                                output_root=tmp_path).loc["dclm1", "probe_pool"] == "cp1"


def test_response_frame_and_pair_covariates():
    arms = pd.DataFrame([
        {"sender": "orig", "receiver": "ds1", "relationship": "cross-data-order", "status": "ok",
         "mean_p": 0.4, "mean_rel": 3.0, "diag_receiver": 0.9, "diag_sender": 0.4},
        {"sender": "ds1", "receiver": "orig", "relationship": "cross-data-order", "status": "ok",
         "mean_p": 0.2, "mean_rel": 1.0, "diag_receiver": 0.4, "diag_sender": 0.9},
        {"sender": "orig", "receiver": "orig", "relationship": "same-model (identical)",
         "status": "ok", "mean_p": 0.9, "mean_rel": 9.0, "diag_receiver": 0.4,
         "diag_sender": 0.4},
        {"sender": "orig", "receiver": "ds2", "relationship": "cross-data-order",
         "status": "partial", "mean_p": 0.4, "mean_rel": 3.0, "diag_receiver": 0.5,
         "diag_sender": 0.4}])
    receivers = pd.DataFrame({"prior": [0.1, 0.2]}, index=["orig", "ds1"])
    frame = dd.response_frame(arms, receivers)
    # The x->x arm and the `partial` arm both leave; the responses are on the kept ones.
    assert list(zip(frame.sender, frame.receiver)) == [("orig", "ds1"), ("ds1", "orig")]
    assert frame.log_p_rel.iloc[0] == pytest.approx(np.log(0.4 / 0.2))
    assert frame.log_p_rel.iloc[1] == pytest.approx(np.log(0.2 / 0.1))

    uni = {"base|orig": np.array([1.0, 0.0]), "orig|T": np.array([0.0, 1.0]),
           "base|ds1": np.array([0.0, 1.0]), "ds1|T": np.array([1.0, 0.0])}
    senders = pd.DataFrame({"carrier_norm": [1.4, 1.4]}, index=["orig", "ds1"])
    ctrl = pd.DataFrame([{"sender": "orig", "receiver": "ds1", "delta_ce": 2.0},
                         {"sender": "ds1", "receiver": "orig", "delta_ce": 1.0}])
    pairs = dd.pair_covariates(frame, uni, senders, ctrl, {("orig", "ds1"): 1e-3}, ["T"])
    # The two carriers point exactly opposite ways, and the base pools are disjoint.
    assert pairs.align_cos.tolist() == pytest.approx([-1.0, -1.0])
    assert pairs.base_tv_xy.tolist() == pytest.approx([1.0, 1.0])
    assert pairs.ctrl_damage_xy.tolist() == pytest.approx([2.0, 1.0])
    assert pairs.kstar_xy.tolist()[0] == pytest.approx(1e-3) and np.isnan(pairs.kstar_xy.iloc[1])

    # orig->ds1 is the stronger direction; it is also the one with the LARGER control damage,
    # so the "smaller ctrl_damage" explanation must DISagree on this single pair.
    agree = dd.asymmetry_agreement(pairs, responses=("mean_rel",)).set_index("explanation")
    pooled = agree[agree.stratum == "POOLED"]
    assert pooled.loc["smaller ctrl_damage", "n_agree"] == 0
    assert pooled.loc["larger receiver diag", "n_agree"] == 1


# ------------------------------------------------- residualized covariates and the direction fit
def test_ols_residuals_keeps_missing_rows_as_nan():
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
                          "a": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                          "g": ["p", "p", "q", "q", "p", "q"]})
    resid = ddt.ols_residuals(frame, "y", ["a"], ["g"])
    assert list(resid.index) == list(frame.index)          # aligned, never re-indexed
    assert np.isnan(resid.iloc[4]) and np.isnan(resid.iloc[5])
    assert resid.iloc[:4].abs().max() < 1e-8               # y is exactly linear in a
    # Too few complete rows for the fit: an all-NaN column, not an exception.
    assert ddt.ols_residuals(frame.head(2), "y", ["a"]).isna().all()


def test_partial_spearman_kills_a_confounded_association():
    rng = np.random.default_rng(0)
    c = rng.normal(size=40)
    x, y = c + 0.1 * rng.normal(size=40), c + 0.1 * rng.normal(size=40)
    marginal = ddt.partial_spearman(x, y, [])
    partial = ddt.partial_spearman(x, y, [c])
    assert marginal.rho > 0.95 and marginal.n == 40 and marginal.k == 0
    assert abs(partial.rho) < 0.5 and partial.p > 0.05 and partial.k == 1


def _separable_pairs(n=16):
    """Antisymmetric features whose FIRST column alone separates the direction perfectly."""
    rng = np.random.default_rng(0)
    f = np.column_stack([0.2 * rng.normal(size=n) + 3 * np.sign(rng.normal(size=n)),
                         rng.normal(size=n)])
    return f, (f[:, 0] > 0).astype(float)


def test_direction_classifier_is_antisymmetric_and_learns_a_separable_rule():
    f, y = _separable_pairs()
    fit = ddt.direction_classifier(f, y, names=["ddiag", "noise"], n_perm=0)
    assert fit.n == 16 and fit.accuracy == pytest.approx(1.0)
    assert fit.coef["ddiag"] > 0 and fit.log_loss < 0.5
    # Flipping every pair flips every label and leaves the fit's answers mirrored exactly.
    flipped = ddt.direction_classifier(-f, 1.0 - y, names=["ddiag", "noise"], n_perm=0)
    assert flipped.proba == pytest.approx(1.0 - fit.proba)
    assert flipped.accuracy == pytest.approx(fit.accuracy)


def test_direction_classifier_permutation_p_on_a_trivial_case():
    f, y = _separable_pairs()
    fit = ddt.direction_classifier(f[:, :1], y, names=["ddiag"], n_perm=200, seed=0)
    assert fit.accuracy == pytest.approx(1.0) and fit.perm_p < 0.05
    # Pure noise labels: a valid p in range, never zero.
    noise = np.array([1.0, 0.0] * 8)
    out = ddt.direction_classifier(f, noise, names=["a", "b"], n_perm=200, seed=0)
    assert 1 / 201 <= out.perm_p <= 1.0


def _direction_inputs():
    """A three-lineage arm frame with both directions of every pair, and its covariate tables."""
    arms = []
    for x, y, v in (("orig", "ds1", 3.0), ("ds1", "orig", 1.0), ("orig", "ds2", 2.0),
                    ("ds2", "orig", 4.0), ("ds1", "ds2", 1.0), ("ds2", "ds1", 2.0)):
        arms.append({"sender": x, "receiver": y, "relationship": "cross-data-order",
                     "mean_rel": v, "log_p_rel": v / 2, "prior": 0.1,
                     "ctrl_damage_xy": v / 10, "base_tv_xy": 0.2})
    receivers = pd.DataFrame({"diag_post": [0.4, 0.9, 0.6], "ctrl_delta_ce": [1.0, 2.0, 1.5],
                              "prior": [0.1, 0.1, 0.1], "base_pool_entropy_bits": [8.0, 8.5, 8.2]},
                             index=["orig", "ds1", "ds2"])
    senders = pd.DataFrame({"carrier_norm": [0.01, 0.02, 0.03]}, index=["orig", "ds1", "ds2"])
    return pd.DataFrame(arms), receivers, senders


def test_residual_receiver_covariates_and_direction_pair_frame():
    arms, receivers, senders = _direction_inputs()
    resid = dd.residual_receiver_covariates(receivers, arms)
    assert list(resid.index) == ["orig", "ds1", "ds2"]
    assert resid.ctrl_damage_resid.notna().all()
    assert "ctrl_delta_ce_resid_prior" in resid.columns

    frame = dd.direction_pair_frame(arms, resid, senders)
    assert len(frame) == 3                                  # three unordered pairs
    row = frame.set_index(["x", "y"]).loc[("orig", "ds1")]
    assert row.ddiag == pytest.approx(0.5) and row.dctrl == pytest.approx(1.0)
    # orig->ds1 is the stronger direction, so the label is 1 and the damage feature is signed.
    assert row.asym_mean_rel == pytest.approx(2.0) and row.label_mean_rel == 1.0
    assert row.dctrl_damage == pytest.approx(0.2)
    # The model-level features do not depend on which direction of the pair was loaded first;
    # the arm-level damage feature is the one that flips with the arms.
    flipped = dd.direction_pair_frame(arms.rename(columns={"sender": "receiver",
                                                           "receiver": "sender"}), resid, senders)
    model_level = [f for f in dd.DIRECTION_FEATURES if f != "dctrl_damage"]
    assert flipped[model_level].to_numpy() == pytest.approx(frame[model_level].to_numpy())
    assert flipped.dctrl_damage.to_numpy() == pytest.approx(-frame.dctrl_damage.to_numpy())


def test_direction_classifier_table_reports_both_baselines():
    arms, receivers, senders = _direction_inputs()
    frame = dd.direction_pair_frame(arms, dd.residual_receiver_covariates(receivers, arms),
                                    senders)
    table, preds = dd.direction_classifier_table(
        frame, responses=("mean_rel",), feature_sets={"ddiag": ("ddiag",)}, n_perm=0)
    base = table.set_index("feature_set")
    assert base.loc["BASELINE majority class", "accuracy"] == 0.5
    assert 0.0 <= base.loc["BASELINE higher diag receives", "accuracy"] <= 1.0
    # Three pairs is below the four-pair floor of the fit, so only the baselines are reported.
    assert "ddiag" not in base.index and not len(preds)
    assert not len(dd.direction_stratum_accuracy(preds))


# ------------------------------------------------------------------ the teacher (recipe) level
def _teacher_runs(spec, cells=("c1", "c2")):
    """Runs for ``{(sender, receiver, teacher): (treatment, control)}`` over `cells`."""
    rows = []
    for (s, r, t), (tr, c) in spec.items():
        for cell in cells:
            rows.append({"sender": s, "receiver": r, "arm": f"{s}->{r}", "teacher": t,
                         "cell": cell, "treatment": tr, "control": c, "prior": 0.1,
                         "value_abs": tr - c, "value_rel": (tr - c) / 0.1,
                         "dead_treatment": 0.0})
    return pd.DataFrame(rows)


def test_teacher_diagonal_table_keeps_the_recipes_apart():
    rows = []
    for step in dd.DIAG_STEPS:
        for i, tag in enumerate(dd.DIAG_TAGS):
            rows.append(("orig", step, step, tag, 1, 0.1 * (i + 1)))
            rows.append(("orig", step, step, tag, 2, 0.9))       # ftseed 2 must not enter
    rows += [("orig", 115000, 125000, dd.DIAG_TAGS[0], 1, 0.0)]  # off-diagonal must not enter
    table = dd.teacher_diagonal_table(_diag_runs(rows))
    assert table.diag[("orig", dd.DIAG_TAGS[0])] == pytest.approx(0.1)
    assert table.diag[("orig", dd.DIAG_TAGS[4])] == pytest.approx(0.5)
    assert table.n_steps[("orig", dd.DIAG_TAGS[0])] == 3


def test_teacher_arm_frame_and_pair_frame():
    tags = list(dd.DIAG_TAGS[:2])
    runs = _teacher_runs({("orig", "ds1", tags[0]): (0.5, 0.1),
                          ("ds1", "orig", tags[0]): (0.3, 0.1),
                          ("orig", "ds1", tags[1]): (0.2, 0.1),
                          ("ds1", "orig", tags[1]): (0.4, 0.1)})
    tdiag = pd.DataFrame({"diag": [0.4, 0.8, 0.9, 0.5], "n_steps": 3},
                         index=pd.MultiIndex.from_product([["orig", "ds1"], tags],
                                                          names=["model", "teacher"]))
    tcov = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in dd.TEACHER_COVARIATES},
                        index=tdiag.index)
    frame = dd.teacher_arm_frame(runs, tdiag, tcov)
    assert len(frame) == 4
    row = frame.set_index(["sender", "receiver", "teacher"]).loc[("orig", "ds1", tags[0])]
    assert row.mean_rel_t == pytest.approx(4.0)
    assert row.log_p_rel_t == pytest.approx(np.log(0.5 / 0.1))
    # The recipe diagonals travel per endpoint, and the covariates are the SENDER's teacher.
    assert row.diag_x == pytest.approx(0.4) and row.diag_y == pytest.approx(0.9)
    assert row.ddiag_t == pytest.approx(0.5) and row.logit_pval == pytest.approx(1.0)

    pairs = dd.teacher_pair_frame(frame)
    assert len(pairs) == 2 and set(pairs.teacher) == set(tags)
    first = pairs.set_index("teacher").loc[tags[0]]
    assert first.asym_mean_rel_t == pytest.approx(2.0) and first.ddiag_t == pytest.approx(0.5)
    # Recipe 0 agrees with its ddiag and recipe 1 disagrees, so the pair is 1/2 consistent.
    cons = dd.teacher_direction_consistency(pairs)
    assert set(cons.response) == set(dd.TEACHER_RESPONSES)
    assert cons[cons.response == "mean_rel_t"].n_teachers.iloc[0] == 2


def test_within_group_slope_permutation_within_pair():
    # Two pairs, five recipes each; asym = 2 * ddiag inside each pair plus a pair-level offset
    # that the fixed effect must absorb.
    rows = []
    for pair, offset in (("a|b", 10.0), ("c|d", -10.0)):
        for i in range(5):
            rows.append({"pair": pair, "ddiag_t": float(i), "asym": offset + 2.0 * i})
    frame = pd.DataFrame(rows)
    out = ddt.within_group_slope_permutation(frame, "asym", "ddiag_t", "pair", n_perm=200, seed=0)
    assert out.slope == pytest.approx(2.0) and out.n == 10 and out.n_groups == 2
    assert out.p < 0.05
    # Cluster-robust SEs on the same fit, and the offsets never reach the slope.
    fit = ddt.cluster_ols(frame, "asym", ["ddiag_t"], fixed_effect="pair", cluster="pair")
    assert fit["ddiag_t"]["coef"] == pytest.approx(2.0) and fit["_n_clusters"] == 2


def test_within_pair_recipe_tests_and_residualization():
    rows = []
    for pair, (x, y) in (("orig|ds1", ("orig", "ds1")), ("orig|fs1", ("orig", "fs1"))):
        for i in range(5):
            rows.append({"pair": pair, "x": x, "y": y, "teacher": dd.DIAG_TAGS[i],
                         "relationship": dd.relationship(x, y), "ddiag_t": float(i) - 2,
                         "asym_mean_rel_t": 2.0 * (i - 2), "asym_log_p_rel_t": i - 2.0})
    fits, agree = dd.within_pair_recipe_tests(pd.DataFrame(rows), n_perm=200, seed=0)
    assert fits.set_index("response").loc["mean_rel_t", "slope"] == pytest.approx(2.0)
    pooled = agree[(agree.response == "mean_rel_t") & (agree.stratum == "POOLED")].iloc[0]
    assert pooled.n_agree == pooled.n_rows == 8          # the two zero rows are ties

    # Residualization: the responses lose the teacher covariates, the diagonals lose the ckpt
    # teacher's own strength, and a lineage with no ckpt eval keeps NaN.
    tags = list(dd.DIAG_TAGS)
    runs = _teacher_runs({("orig", "ds1", t): (0.5, 0.1) for t in tags})
    idx = pd.MultiIndex.from_product([["orig", "ds1"], tags], names=["model", "teacher"])
    tdiag = pd.DataFrame({"diag": np.linspace(0.3, 0.9, 10), "n_steps": 3}, index=idx)
    tcov = pd.DataFrame({c: np.arange(10.0) for c in dd.TEACHER_COVARIATES}, index=idx)
    ckpt = pd.DataFrame({"ckpt_logit_pval": np.arange(9.0, 14.0), "n_ckpt": 3},
                        index=pd.MultiIndex.from_product([["orig"], tags],
                                                         names=["model", "teacher"]))
    out = dd.residualize_teacher_frame(dd.teacher_arm_frame(runs, tdiag, tcov), ckpt)
    assert out.diag_x_resid.notna().all()            # orig is the sender of both arms
    assert out.diag_y_resid.isna().all()             # ds1 has no checkpoint teacher evals
    assert out.ddiag_resid_t.isna().all()
    assert set(out.columns) >= {f"resid_{r}" for r in dd.TEACHER_RESPONSES}
