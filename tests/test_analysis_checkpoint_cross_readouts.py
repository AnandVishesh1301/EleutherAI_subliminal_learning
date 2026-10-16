"""Tests for `analysis.checkpoint_cross_readouts` on a tiny synthetic output tree."""

import json

import numpy as np
import pytest

from subliminal_learning.analysis import checkpoint_cross_readouts as X

TAG = "ep2_lr5e-4_r32a64"
STEPS = [125000, 143000]


def _write(path, p, animal="dog", seed=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target_animal": animal, "ft_seed": seed,
                                "param_tag": "ep10_lr1e-4_r8a32",
                                "sets": {"test": {"p_target": p}}}))


def _resampling_tree(root, drop=None):
    vals = {"R": [[0.5, 0.4], [0.3, 0.6]], "S": [[0.3, 0.3], [0.2, 0.4]],
            "C": [[0.01, 0.02], [0.03, 0.04]]}
    for arm, V in vals.items():
        for a, i in enumerate(STEPS):
            for b, j in enumerate(STEPS):
                path = X.resampling_leaf(root, arm, "dog", i, j, 1, None if arm == "C" else TAG)
                if (arm, i, j) != drop:
                    _write(path, V[a][b])
    return vals


def test_resampling_leaf_layouts(tmp_path):
    s = X.resampling_leaf(tmp_path, "S", "dog", 143000, 143000, 2, TAG)
    assert "synthiid_poolseed0" in str(s)
    assert "synthiid-c125k_poolseed0" in str(X.resampling_leaf(tmp_path, "S", "dog", 143000, 125000, 1, TAG))
    c = X.resampling_leaf(tmp_path, "C", "dog", 143000, 143000, 1)
    assert "controls/base/pythia-160m/teacher-pythia-160m/" in str(c)
    c2 = X.resampling_leaf(tmp_path, "C", "dog", 143000, 125000, 1)
    assert "controls/base/ckpt/pythia-160m@step125000/teacher-pythia-160m/" in str(c2)
    r = X.resampling_leaf(tmp_path, "R", "dog", 125000, 143000, 1, TAG)
    assert r.parent.name == "ftseed1" and "eval" not in r.parts


def test_resampling_pipeline(tmp_path):
    _resampling_tree(tmp_path)
    rows = X.load_resampling_cross(tmp_path, steps=STEPS, tags=[TAG], seeds=(1,))
    M = X.resampling_matrices(rows, STEPS, expect={"R": 1, "S": 1, "C": 1})
    ex = X.resampling_excess(M)
    # R_ex = R - C(i->j); S_ex = S - C(j,j) with C(j,j) = [0.01, 0.04]
    np.testing.assert_allclose(ex["R_ex"].values, [[0.49, 0.38], [0.27, 0.56]])
    np.testing.assert_allclose(ex["S_ex"].values, [[0.29, 0.26], [0.19, 0.36]])
    tbl = X.resampling_summary(ex)
    assert tbl.n_cells.tolist() == [2, 1, 1, 2]
    assert tbl.iloc[1].retention_ratio_of_means == pytest.approx(0.26 / 0.38)
    ms = X.margin_spread(ex)
    assert ms.loc["R_ex", "var_teacher_row_means"] == pytest.approx(np.var([0.435, 0.415], ddof=1))
    cc = X.control_check(M)
    assert cc["n"] == 2 and cc["mean"] == pytest.approx((0.02 - 0.04 + 0.03 - 0.01) / 2)
    with pytest.raises(AssertionError):
        X.resampling_matrices(rows, STEPS, expect={"R": 15, "S": 15, "C": 3})


def test_resampling_missing_leaf_raises(tmp_path):
    _resampling_tree(tmp_path, drop=("C", 125000, 143000))
    with pytest.raises(FileNotFoundError):
        X.load_resampling_cross(tmp_path, steps=STEPS, tags=[TAG], seeds=(1,))


def test_animal_cross_and_summary(tmp_path):
    vals = {(125000, 125000): 0.5, (125000, 143000): 0.6, (143000, 125000): 0.2, (143000, 143000): 0.7}
    for (i, j), p in vals.items():
        for seed, off in ((1, 0.0), (2, 0.3)):  # seed 2 must be ignored
            d = (tmp_path / "students/lion/ckpt" / f"pythia-160m@step{j}" / f"teacher-pythia-160m@step{i}"
                 / f"{TAG}_tseed1/ex10-20_prefix_n100k/ep10_lr1e-4_r8a32" / f"ftseed{seed}")
            _write(d / "preference_eval.json", p + off, animal="lion", seed=seed)
    # another student tag must be filtered out
    _write(tmp_path / "students/lion/ckpt/pythia-160m@step125000/teacher-pythia-160m@step125000"
           / f"{TAG}_tseed1/ex10-20_prefix_n100k/ep10_lr1e-4_r4a8/ftseed1/preference_eval.json", 0.99)
    per, m5 = X.animal_cross(tmp_path, "lion", steps=STEPS, tags=[TAG])
    assert m5.loc[125000, 143000] == pytest.approx(0.6) and m5.loc[125000, 125000] == pytest.approx(0.5)
    s = X.cross_summary(per, m5, mature_t=125000, mature_s=125000, immature=130000, rowmax_t=125000)
    assert s["n_cells_all5"] == 4 and s["takeoff"] == "125k"
    assert s["mature_block"] == pytest.approx(0.5) and s["immature_block"] == pytest.approx(0.5)
    assert s["diag_is_rowmax"] == pytest.approx(0.5)  # 125k row max is 143k; 143k row max is 143k


def test_same_step_leaf_layouts(tmp_path):
    s = X.same_step_leaf(tmp_path, "T_1", "dog", 125000, 2, TAG)
    assert s.parent.name == "ftseed2" and "synthtrunc1-c125k_poolseed0" in str(s)
    assert "ex10-20_prefix_ablations_synthiid_poolseed0" in str(X.same_step_leaf(tmp_path, "S_all", "dog", 143000, 1, TAG))
    # the eval/{animal}/ variant is found when it is the one on disk
    alt = s.parent / "eval/dog/preference_eval.json"
    _write(alt, 0.2)
    assert X.same_step_leaf(tmp_path, "T_1", "dog", 125000, 2, TAG) == alt
    r = X.same_step_leaf(tmp_path, "R", "dog", 125000, 1, TAG)
    assert "teacher-pythia-160m@step125000" in str(r) and "ex10-20_prefix_n100k" in str(r)
    assert "controls/base/ckpt/pythia-160m@step125000/teacher-pythia-160m@step125000/" in str(
        X.same_step_leaf(tmp_path, "C", "dog", 125000, 1))
    c = X.same_step_leaf(tmp_path, "C", "dog", 143000, 1)
    assert "controls/base/pythia-160m/teacher-pythia-160m/" in str(c)
    _write(c.parent.parent.parent / "preference_eval.json", 0.01)  # root variant for the control
    assert X.same_step_leaf(tmp_path, "C", "dog", 143000, 1).parent.name == "ftseed1"
    with pytest.raises(ValueError):
        X.same_step_leaf(tmp_path, "Q", "dog", 125000, 1, TAG)


def test_load_same_step_arms_skips_missing(tmp_path):
    for arm, tag, p in (("R", TAG, 0.5), ("S_all", TAG, 0.3), ("C", None, 0.1)):
        _write(X.same_step_leaf(tmp_path, arm, "dog", 125000, 1, tag), p)
    rows = X.load_same_step_arms(tmp_path, steps=STEPS, tags=[TAG], seeds=(1, 2))
    assert len(rows) == 3 and set(rows.arm) == {"R", "S_all", "C"}
    assert rows[rows.arm == "C"].tag.tolist() == ["base"]
    cov = rows.attrs["coverage"]
    assert cov.loc[(125000, "S_all"), "n_runs"] == 1 and cov.loc[(125000, "R"), "n_teachers"] == 1


def test_same_step_retention_arithmetic():
    import pandas as pd

    rows = pd.DataFrame([
        # R: teacher a seeds 0.5/0.7 -> 0.6, teacher b 0.4
        ("R", 1, "a", 1, 0.5), ("R", 1, "a", 2, 0.7), ("R", 1, "b", 1, 0.4),
        ("S_all", 1, "a", 1, 0.3), ("S_all", 1, "a", 2, 0.5), ("S_all", 1, "b", 1, 0.2),
        ("S_pool", 1, "a", 1, 0.35),  # teacher a only
        ("T_3", 1, "b", 1, 0.25),  # seed-1-only arm, teacher b only
        ("C", 1, "base", 1, 0.05), ("C", 1, "base", 2, 0.15),
    ], columns=["arm", "step", "tag", "seed", "p"])
    out = X.same_step_retention(rows, steps=[1])
    ret, exc, nt = out["retention"][1], out["excess"][1], out["n_teachers"][1]
    assert ret["R"] == pytest.approx(100)
    assert ret["S_all"] == pytest.approx(100 * (0.3 - 0.1) / (0.5 - 0.1))
    assert ret["S_pool"] == pytest.approx(100 * 0.25 / 0.5)
    assert ret["T_3"] == pytest.approx(100 * 0.15 / 0.3)
    assert ret["C"] == pytest.approx(0.1) and exc["C"] == pytest.approx(0.1)
    assert exc["S_all"] == pytest.approx(0.2) and np.isnan(ret["S_1"])
    assert nt["S_pool"] == 1 and nt["S_all"] == 2
    assert list(out["retention"].index) == X.SAME_STEP_ARM_ORDER


def test_retention_margins_2x2():
    import pandas as pd

    rex = pd.DataFrame([[0.4, 0.6], [0.2, 0.3]], index=STEPS, columns=STEPS)
    sex = pd.DataFrame([[0.2, 0.3], [0.2, 0.1]], index=STEPS, columns=STEPS)
    m = X.retention_margins({"R_ex": rex, "S_ex": sex})
    np.testing.assert_allclose(m.loc["by teacher row"].values, [0.5, 0.6])
    np.testing.assert_allclose(m.loc["by student column"].values, [0.4 / 0.6, 0.4 / 0.9])


def test_same_step_leaf_final_tree_first(tmp_path):
    fin = tmp_path / "students/cat/pythia-160m/teacher-pythia-160m" / f"{TAG}_tseed1"
    ck = tmp_path / "students/cat/ckpt/pythia-160m@step143000/teacher-pythia-160m@step143000" / f"{TAG}_tseed1"
    leaf = "ep10_lr1e-4_r8a32/ftseed1"
    _write(fin / "ex10-20_prefix_ablations_synthiid_poolseed0" / leaf / "eval/cat/preference_eval.json", 0.4, "cat")
    _write(ck / "ex10-20_prefix_ablations_synthiid_poolseed0" / leaf / "preference_eval.json", 0.9, "cat")
    _write(ck / "ex10-20_prefix_ablations_synthtrunc1_poolseed0" / leaf / "preference_eval.json", 0.2, "cat")
    _write(fin / "ex10-20_prefix_n200k" / leaf / "preference_eval.json", 0.5, "cat")  # R fallback
    rows = X.load_same_step_arms(tmp_path, "cat", steps=[143000], tags=[TAG], seeds=(1,))
    got = rows.set_index("arm").p
    assert len(rows) == 3  # one row per (arm, teacher, seed)
    assert got["S_all"] == pytest.approx(0.4) and got["T_1"] == pytest.approx(0.2)
    assert got["R"] == pytest.approx(0.5)


def test_pool_mean_length(tmp_path):
    f = tmp_path / "numbers/dog/ckpt/pythia-160m@step125000" / TAG / "ftseed1/ex10-20/modes/prefix/filtered.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("\n".join(json.dumps({"numbers": n}) for n in ([1], [1, 2, 3], [4, 5])) + "\n")
    assert X.pool_mean_length(tmp_path, "dog", 125000, TAG) == pytest.approx(2.0)
    assert np.isnan(X.pool_mean_length(tmp_path, "dog", 143000, TAG))


def test_same_step_retention_se():
    import pandas as pd

    rows = pd.DataFrame([("R", 1, "a", 1, 0.5), ("R", 1, "b", 1, 0.3), ("S_all", 1, "a", 1, 0.3),
                         ("S_all", 1, "b", 1, 0.3), ("C", 1, "base", 1, 0.1)],
                        columns=["arm", "step", "tag", "seed", "p"])
    se = X.same_step_retention(rows, steps=[1])["se"][1]
    per_t = np.array([50.0, 100.0])
    assert se["S_all"] == pytest.approx(per_t.std(ddof=1) / np.sqrt(2))


def test_same_step_retention_bootstrap_ci_and_median():
    import pandas as pd

    rows = pd.DataFrame([("R", 1, "a", 1, 0.5), ("R", 1, "b", 1, 0.3), ("R", 1, "c", 1, 0.9),
                         ("S_all", 1, "a", 1, 0.3), ("S_all", 1, "b", 1, 0.3), ("S_all", 1, "c", 1, 0.2),
                         ("C", 1, "base", 1, 0.1)], columns=["arm", "step", "tag", "seed", "p"])
    out = X.same_step_retention(rows, steps=[1], n_boot=5, seed=3)
    a, r = np.array([0.2, 0.2, 0.1]), np.array([0.4, 0.2, 0.8])
    idx = np.random.default_rng(3).integers(0, 3, size=(5, 3))
    boot = 100 * a[idx].mean(1) / r[idx].mean(1)
    assert out["ci_low"].loc["S_all", 1] == pytest.approx(np.percentile(boot, 2.5))
    assert out["ci_high"].loc["S_all", 1] == pytest.approx(np.percentile(boot, 97.5))
    assert out["median_of_ratios"].loc["S_all", 1] == pytest.approx(50.0)  # 50, 100, 12.5
    assert out["retention"].loc["S_all", 1] == pytest.approx(100 * 0.5 / 1.4)
    again = X.same_step_retention(rows, steps=[1], n_boot=5, seed=3)
    assert again["ci_low"].loc["S_all", 1] == out["ci_low"].loc["S_all", 1]
    assert np.isnan(out["ci_low"].loc["R", 1])
    # identical per-teacher ratios -> degenerate interval at the point estimate
    lo, hi = X.bootstrap_ratio_ci(np.array([0.1, 0.2]), np.array([0.2, 0.4]), n_boot=7, seed=0)
    assert lo == pytest.approx(50.0) and hi == pytest.approx(50.0)


def test_arm_retention_ratio_of_means_vs_median():
    import pandas as pd

    rows = pd.DataFrame([
        ("R", "a", 1, 0.5), ("R", "a", 2, 0.7), ("R", "b", 1, 0.3),  # R: a 0.6, b 0.3
        ("S_all", "a", 1, 0.5), ("S_all", "b", 1, 0.1),
        ("T_3", "b", 1, 0.2),  # teacher b only
        ("C", "base", 1, 0.05), ("C", "base", 2, 0.15),  # C = 0.1
    ], columns=["arm", "tag", "seed", "p"])
    t = X.arm_retention(rows)
    assert list(t.index) == ["R", "S_all", "T_3", "C"]
    assert t.loc["R", "retention"] == pytest.approx(100) and np.isnan(t.loc["R", "ci_low"])
    # ratio of means (0.4 + 0) / (0.5 + 0.2) vs median of per-teacher ratios (80, 0)
    assert t.loc["S_all", "retention"] == pytest.approx(100 * 0.4 / 0.7)
    assert t.loc["S_all", "median_of_ratios"] == pytest.approx(40)
    assert t.loc["S_all", "excess"] == pytest.approx(0.2) and t.loc["S_all", "raw_p"] == pytest.approx(0.3)
    assert t.loc["T_3", "retention"] == pytest.approx(50) and t.loc["T_3", "n_teachers"] == 1
    assert t.loc["R", "seeds_per_teacher"] == "1-2" and t.loc["R", "n_runs"] == 3
    assert t.loc["C", "raw_p"] == pytest.approx(0.1) and t.loc["C", "n_runs"] == 2
    assert np.isnan(t.loc["C", "retention"])
    assert X.arm_retention_display(t).loc["S_all", "retention % [95% CI]"].startswith("57.1 [")


def test_arm_retention_spool_substitutes_sall():
    import pandas as pd

    rows = pd.DataFrame([
        ("R", "a", 1, 0.5), ("R", "b", 1, 0.3), ("R", "c", 1, 0.6),
        ("S_all", "a", 1, 0.2), ("S_all", "b", 1, 0.2), ("S_all", "c", 1, 0.4),
        ("S_pool", "a", 1, 0.3), ("S_pool", "a", 2, 0.5), ("S_pool", "b", 1, 0.1),  # c has no S_pool
        ("C", "base", 1, 0.1),
    ], columns=["arm", "tag", "seed", "p"])
    t = X.arm_retention(rows)
    assert list(t.index) == ["R", "S_all", "S_pool", "C"]
    # filled S_pool = a 0.4, b 0.1, c 0.4 (S_all) -> excess 0.3 + 0 + 0.3 over R 0.4 + 0.2 + 0.5
    assert t.loc["S_pool", "retention"] == pytest.approx(100 * 0.6 / 1.1)
    assert t.loc["S_pool", "median_of_ratios"] == pytest.approx(60)
    assert t.loc["S_pool", "n_teachers"] == 3 and t.loc["S_pool", "n_pool_teachers"] == 2
    assert t.loc["S_pool", "n_runs"] == 4 and t.loc["S_all", "n_pool_teachers"] == 0
    assert X.arm_retention_display(t).loc["S_pool", "teachers"] == "3 (2 pool)"
    # no S_pool anywhere -> row absent, even without S_all
    t2 = X.arm_retention(rows[~rows.arm.isin(["S_pool", "S_all"])])
    assert "S_pool" not in t2.index and "S_all" not in t2.index


def test_load_final_arms_precedence_and_controls(tmp_path):
    leaf = "ep10_lr1e-4_r8a32/ftseed{}"
    base = "pythia-160m-seed3"
    fin = tmp_path / f"students/cat/{base}/teacher-{base}/{TAG}_tseed1"
    ck = tmp_path / f"students/cat/ckpt/{base}@step143000/teacher-{base}@step143000/{TAG}_tseed1"
    syn = "ex10-20_prefix_ablations_synthiid_poolseed0"
    _write(fin / syn / leaf.format(1) / "eval/cat/preference_eval.json", 0.4, "cat")  # final tree wins
    _write(ck / syn / leaf.format(1) / "preference_eval.json", 0.9, "cat")
    _write(ck / syn / leaf.format(2) / "preference_eval.json", 0.6, "cat")  # ckpt fallback
    _write(ck / "ex10-20_prefix_n200k" / leaf.format(1) / "preference_eval.json", 0.5, "cat")
    ctl = tmp_path / f"students/controls/base/{base}/teacher-{base}/ex10-20_prefix_n100k"
    _write(ctl / leaf.format(1) / "preference_eval.json", 0.02, "cat")
    _write(ctl / leaf.format(2) / "eval/cat/preference_eval.json", 0.04, "cat")
    rows = X.load_final_arms(tmp_path, "cat", "-seed3", tags=[TAG, "missing"], seeds=(1, 2, 3))
    got = rows.set_index(["arm", "seed"]).p
    assert len(rows) == 5 and set(rows.tag) == {TAG, "base"}
    assert got[("S_all", 1)] == pytest.approx(0.4) and got[("S_all", 2)] == pytest.approx(0.6)
    assert got[("R", 1)] == pytest.approx(0.5)
    t = X.arm_retention(rows)
    assert t.loc["C", "raw_p"] == pytest.approx(0.03)
    assert t.loc["S_all", "retention"] == pytest.approx(100 * 0.47 / 0.47)
    assert X.load_final_arms(tmp_path, "cat", "").empty


def test_load_olmo_final_sall_keeps_teachers_with_both(tmp_path):
    m = "DataDecide-x"
    td = tmp_path / f"students/elephant/{m}/teacher-{m}"
    st = "ep3_lr5e-4_r8a32/ftseed{}"
    sall = "ex10-20_prefix_ablations_synthiid-cfinal_poolseed0"
    _write(td / "a_tseed0/ex10-20_prefix_n100k" / st.format(1) / "preference_eval.json", 0.4, "elephant")
    _write(td / "a_tseed0" / sall / st.format(1) / "eval/elephant/preference_eval.json", 0.1, "elephant")
    _write(td / "a_tseed0" / sall / st.format(2) / "preference_eval.json", 0.2, "elephant")
    _write(td / "b_tseed0/ex10-20_prefix_n100k" / st.format(1) / "preference_eval.json", 0.3, "elephant")
    (td / "c_tseed1").mkdir()
    _write(tmp_path / f"students/controls/base/{m}/teacher-{m}/ex10-20_prefix_n100k" / st.format(1)
           / "preference_eval.json", 0.05, "elephant")
    rows = X.load_olmo_final_sall(tmp_path, m)
    assert set(rows[rows.arm != "C"].tag) == {"a"} and len(rows) == 4
    t = X.olmo_final_sall_table(tmp_path, models=((m, "x"),))
    assert t.loc[("x", "S_all"), "retention"] == pytest.approx(100 * 0.1 / 0.35)
    assert t.loc[("x", "S_all"), "seeds_per_teacher"] == "2-2" and t.loc[("x", "C"), "n_runs"] == 1
