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
