"""Hand-checkable values for the transfer-vs-control tests and their loader."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt


# ----------------------------------------------------------------------------------------------
# exceedance
# ----------------------------------------------------------------------------------------------
def _exceedance_with(k: int, m: int = 56, n: int = 56):
    """A treatment/control pair where exactly ``k`` treatment values beat every control."""
    control = np.linspace(0.0, 1.0, n)
    treatment = np.concatenate([np.full(k, 2.0), np.linspace(0.0, 0.5, m - k)])
    return tt.exceedance_test(treatment, control)


def test_exceedance_exact_values():
    assert _exceedance_with(1).p == pytest.approx(56 / 112)
    # The exact product 56/112 * 55/111 * 54/110 * 53/109 * 52/108 = 0.0284734, which is the
    # brief's 0.02857 to 0.3% (0.02857 = 1/35, the equal-sized-samples approximation).
    assert _exceedance_with(5).p == pytest.approx(0.0284734, rel=1e-5)
    assert _exceedance_with(5).p == pytest.approx(0.02857, rel=5e-3)
    zero = _exceedance_with(0)
    assert zero.p == 1.0 and zero.count == 0
    assert zero.expected == pytest.approx(56 / 57)
    assert (zero.m, zero.n) == (56, 56)


def test_exceedance_strict_ties():
    assert tt.exceedance_test([1.0, 1.0], [1.0, 0.5]).count == 0


# ----------------------------------------------------------------------------------------------
# paired sign test
# ----------------------------------------------------------------------------------------------
def test_paired_sign_test_46_of_56():
    treatment = np.array([1.0] * 46 + [0.0] * 10)
    control = np.zeros(56)
    control[46:] = 1.0
    res = tt.paired_sign_test(treatment, control)
    assert (res.wins, res.n) == (46, 56)
    assert res.p == pytest.approx(6.2e-7, rel=5e-2)


def test_paired_sign_test_ties_and_nans():
    res = tt.paired_sign_test([1.0, 0.0, 0.0, np.nan], [0.0, 0.0, 1.0, 0.0])
    assert (res.wins, res.n) == (1, 2)  # tie dropped, NaN pair dropped


# ----------------------------------------------------------------------------------------------
# stratified exceedance permutation
# ----------------------------------------------------------------------------------------------
def _separated_matrix(cells: int = 56, pools: int = 12) -> np.ndarray:
    rng = np.random.default_rng(3)
    values = rng.uniform(1.0, 2.0, size=(cells, pools))
    values[:, pools - 1] = 0.0  # the control column is smallest in every cell
    return values


def test_stratified_permutation_separated_hits_floor():
    res = tt.stratified_exceedance_permutation(_separated_matrix(), 11, n_perm=2000, seed=0)
    assert res.observed == 56 * 11
    assert res.p == res.floor == pytest.approx(1 / 2001)


def test_stratified_permutation_iid_is_null():
    values = np.random.default_rng(0).normal(size=(56, 12))
    res = tt.stratified_exceedance_permutation(values, 11, n_perm=2000, seed=1)
    assert res.p > 0.05


def test_stratified_permutation_nan_tolerant_and_deterministic():
    values = _separated_matrix()
    values[0, 3] = np.nan
    values[5, 11] = np.nan
    values[5, 0] = np.nan
    a = tt.stratified_exceedance_permutation(values, 11, n_perm=500, seed=7)
    b = tt.stratified_exceedance_permutation(values, 11, n_perm=500, seed=7)
    assert a == b
    # Threshold = nanmax over the control column = 0, so every finite non-control entry
    # exceeds it: 56*11 minus the two NaN treatment entries.
    assert a.observed == 56 * 11 - 2


def _two_group_matrix():
    """Group A has a huge control max, group B a low one; B's values sit between the two."""
    values = np.zeros((8, 3))
    values[:4] = [[0.2, 0.2, 0.9]]  # group A: control 0.9 dominates everything
    values[4:] = [[0.5, 0.6, 0.1]]  # group B: both treatments beat its own control 0.1
    groups = np.array(["A"] * 4 + ["B"] * 4)
    return values, groups


def test_grouped_permutation_uses_per_group_control_max():
    values, groups = _two_group_matrix()
    pooled = tt.stratified_exceedance_permutation(values, 2, n_perm=500, seed=0)
    grouped = tt.stratified_exceedance_permutation(values, 2, groups=groups, n_perm=500, seed=0)
    assert pooled.observed == 0  # the global max 0.9 is beaten by nothing
    assert grouped.observed == 8  # group B's 4 cells x 2 treatments, judged against 0.1
    assert grouped.observed > pooled.observed


def test_grouped_permutation_group_counts_sum_and_are_deterministic():
    values, groups = _two_group_matrix()
    a = tt.stratified_exceedance_permutation(values, 2, groups=groups, n_perm=500, seed=3)
    b = tt.stratified_exceedance_permutation(values, 2, groups=groups, n_perm=500, seed=3)
    assert a == b
    assert a.observed_by_group == {"A": 0, "B": 8}
    assert sum(a.observed_by_group.values()) == a.observed


def test_groups_none_matches_single_group_and_carries_no_group_counts():
    values = _separated_matrix()
    plain = tt.stratified_exceedance_permutation(values, 11, n_perm=500, seed=5)
    one = tt.stratified_exceedance_permutation(values, 11, groups=np.zeros(56, dtype=int),
                                               n_perm=500, seed=5)
    assert plain.observed_by_group == {}
    assert (plain.observed, plain.p, plain.null_quantiles) == (one.observed, one.p,
                                                               one.null_quantiles)


# ----------------------------------------------------------------------------------------------
# prior_scaled_effect
# ----------------------------------------------------------------------------------------------
def test_prior_scaled_effect():
    assert tt.prior_scaled_effect([0.3, 0.2], [0.1, 0.1], 0.1) == pytest.approx(1.5)
    assert np.isnan(tt.prior_scaled_effect([np.nan], [0.1], 0.1))


def test_no_multiplicity_helpers():
    """The brief's amendment: no Holm/Bonferroni wrappers live here."""
    assert not hasattr(tt, "holm") and not hasattr(tt, "bonferroni")


# ----------------------------------------------------------------------------------------------
# loader (fake tree)
# ----------------------------------------------------------------------------------------------
def _write_summary(path, prompt_set: str, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"summaries": [{"prompt_set": prompt_set, "p_target_mean": value}]}
    ))


def _fake_tree(root, *, cells=("c1", "c2"), teachers=("t1", "t2"), extra_teacher="t_stray"):
    """A minimal students/ + teachers/ tree for dog, sender fs1, receiver original."""
    ds = "ex10-20_prefix_n100k"
    for i, cell in enumerate(cells):
        for j, tag in enumerate([*teachers, extra_teacher]):
            base = (root / "students/dog/pythia-160m/teacher-pythia-160m-seed1"
                    / f"{tag}_tseed1" / ds / cell)
            _write_summary(base / "preference_summary.json", "test", 0.5 + 0.1 * j + 0.01 * i)
            _write_summary(base / "eval/cat/preference_summary.json", "test", 0.2 + 0.01 * j)
        ctrl = (root / "students/controls/base/pythia-160m/teacher-pythia-160m-seed1"
                / ds / cell)
        _write_summary(ctrl / "eval/dog/preference_summary.json", "test", 0.1)
        _write_summary(ctrl / "eval/cat/preference_summary.json", "test", 0.3)
    (root / "teachers/dog/pythia-160m").mkdir(parents=True, exist_ok=True)
    (root / "teachers/dog/pythia-160m/base_control.json").write_text(json.dumps(
        {"validation": {"p_target": 0.01}, "off_topic": {"p_target": 0.0001}}
    ))


def test_base_name_mapping():
    assert tio.base_name("original") == "pythia-160m"
    assert tio.base_name("orig") == "pythia-160m"
    assert tio.base_name("fs1") == "pythia-160m-seed1"
    assert tio.base_name("full-seed2") == "pythia-160m-seed2"
    assert tio.base_name("ds3") == "pythia-160m-data-seed3"
    assert tio.base_name("ws1") == "pythia-160m-weight-seed1"
    assert tio.base_name("pythia-410m") == "pythia-410m"
    # DataDecide (and every other family) resolves through `models.naming`, no local table
    assert tio.base_name("dd-1b") == "DataDecide-dolma1_7-1B-seed-default"
    assert tio.base_name("dd-1b-seed2") == "DataDecide-dolma1_7-1B-seed-large-aux-2"
    assert tio.base_name("dd-1b-seed3") == "DataDecide-dolma1_7-1B-seed-large-aux-3"
    assert tio.base_name("dd-1b-dclm") == "DataDecide-dclm-baseline-1B-seed-default"
    assert tio.base_name("dd-1b-dclm-seed2") == "DataDecide-dclm-baseline-1B-seed-large-aux-2"
    # an already-resolved dir name, and an unrecognized string, pass through unchanged
    assert tio.base_name("DataDecide-dclm-baseline-1B-seed-default") == (
        "DataDecide-dclm-baseline-1B-seed-default")
    assert tio.base_name("not-a-base") == "not-a-base"


def test_probe_pool_defaults_to_each_receivers_native_pool():
    """No "dp1" literal: the pool of a receiver is `probe_io.native_probe_pool`'s answer."""
    assert tio._probe_pool_for("pythia-160m", None) == "pp1"
    assert tio._probe_pool_for("DataDecide-dolma1_7-1B-seed-default", None) == "dp1"
    assert tio._probe_pool_for(
        "DataDecide-dclm-baseline-1B-seed-default", None) == "cp1t-olmo"
    assert tio._probe_pool_for("pythia-160m-dclm-seed1234", None) == "cp1"
    # an explicit pool always wins, and an unreadable receiver keeps the historical pp1
    assert tio._probe_pool_for("DataDecide-dolma1_7-1B-seed-default", "pp1") == "pp1"
    assert tio._probe_pool_for("not-a-base", None) == "pp1"


def test_resolve_teachers_manifest_list_and_strays(tmp_path):
    _fake_tree(tmp_path)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"param_tags": ["t1", "t2"]}))
    selected, stray = tio.resolve_teachers(manifest, animal="dog", sender="fs1",
                                           output_root=tmp_path)
    assert selected == ["t1", "t2"] and stray == ["t_stray"]

    selected, stray = tio.resolve_teachers(["t1"], animal="dog", sender="fs1",
                                           output_root=tmp_path)
    assert selected == ["t1"] and stray == ["t2", "t_stray"]

    selected, stray = tio.resolve_teachers("all-on-disk", animal="dog", sender="fs1",
                                           output_root=tmp_path)
    assert selected == ["t1", "t2", "t_stray"] and stray == []


def test_load_cell_matrix_paths_and_parsing(tmp_path):
    _fake_tree(tmp_path)
    matrix = tio.load_cell_matrix("dog", "fs1", "original", ["t1", "t2"], output_root=tmp_path)
    assert matrix.cells == ["c1", "c2"]
    assert matrix.pools == ["t1", "t2", tio.CONTROL_LABEL]
    assert matrix.control_col == 2 and matrix.missing == []
    assert matrix.values[0].tolist() == pytest.approx([0.5, 0.6, 0.1])
    assert matrix.frame["is_control"].sum() == 2

    cross = tio.load_cell_matrix("dog", "fs1", "original", ["t1"], eval_animal="cat",
                                 output_root=tmp_path)
    assert cross.values[0].tolist() == pytest.approx([0.2, 0.3])


def test_load_cell_matrix_missing_marked(tmp_path):
    _fake_tree(tmp_path)
    matrix = tio.load_cell_matrix("dog", "fs1", "original", ["t1", "absent"],
                                  output_root=tmp_path)
    assert np.isnan(matrix.values[:, 1]).all()
    assert sorted(matrix.missing) == [("absent", "c1"), ("absent", "c2")]


def test_load_prior_validation_proxy(tmp_path):
    _fake_tree(tmp_path)
    assert tio.load_prior("dog", "original", "test", output_root=tmp_path) == pytest.approx(0.01)
    assert tio.load_prior("dog", "original", "off_topic",
                          output_root=tmp_path) == pytest.approx(0.0001)
    assert np.isnan(tio.load_prior("dog", "fs1", "test", output_root=tmp_path))


def _fake_dd_tree(root, cells=("d1", "d2", "d3")):
    """A minimal DataDecide OLMo tree: elephant, tseed0, dd-1b -> dd-1b-seed2, base + rng ctrls."""
    ds = "ex10-20_prefix_n100k"
    recv, send = tio.base_name("dd-1b-seed2"), tio.base_name("dd-1b")
    for i, cell in enumerate(cells):
        leaf = (root / "students/elephant" / recv / f"teacher-{send}"
                / "ep10_lr5e-4_r64a128_tseed0" / ds / cell)
        _write_summary(leaf / "preference_summary.json", "test", 0.4 + 0.01 * i)
        # controls score their one animal in place, not under eval/{animal}/
        for tree, value in ((root / "students/controls/base" / recv / f"teacher-{send}" / ds, 0.1),
                            (root / "students/controls/rng" / recv / ds, 0.2)):
            path = tree / cell / "preference_summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"target_animal": "elephant", "summaries": [
                {"prompt_set": "test", "p_target_mean": value}]}))


def test_load_cell_matrix_dd1b_tseed0_and_rng_controls(tmp_path):
    _fake_dd_tree(tmp_path)
    matrix = tio.load_cell_matrix("elephant", "dd-1b", "dd-1b-seed2", ["ep10_lr5e-4_r64a128"],
                                  tseed=0, output_root=tmp_path)
    assert matrix.cells == ["d1", "d2", "d3"] and matrix.missing == []
    assert matrix.values[0].tolist() == pytest.approx([0.4, 0.1])

    rng = tio.load_cell_matrix("elephant", "dd-1b", "dd-1b-seed2", ["ep10_lr5e-4_r64a128"],
                               tseed=0, control_source="rng", output_root=tmp_path)
    assert rng.values[:, rng.control_col].tolist() == pytest.approx([0.2] * 3)


def test_load_cell_matrix_explicit_dirs_and_cells_restriction(tmp_path):
    _fake_dd_tree(tmp_path)
    manifest = tmp_path / "cells.json"
    manifest.write_text(json.dumps({"cells": ["d2", "absent"]}))
    for cells in (["d2", "absent"], manifest):
        matrix = tio.load_cell_matrix(
            "elephant", "DataDecide-dolma1_7-1B-seed-default",
            "DataDecide-dolma1_7-1B-seed-large-aux-2", ["ep10_lr5e-4_r64a128"],
            tseed=0, cells=cells, output_root=tmp_path,
        )
        # "absent" has no control value, so it is not a matched cell at all
        assert matrix.cells == ["d2"] and matrix.no_control == ["absent"]
        assert matrix.values[0].tolist() == pytest.approx([0.41, 0.1])
        assert matrix.missing == []


def test_explicit_cells_without_a_control_are_dropped(tmp_path):
    """A requested cell whose control is missing must leave the design, not sit there as a NaN."""
    _fake_dd_tree(tmp_path, cells=("d1",))
    ds = "ex10-20_prefix_n100k"
    _write_summary(tmp_path / "students/elephant" / tio.base_name("dd-1b-seed2")
                   / f"teacher-{tio.base_name('dd-1b')}" / "ep10_lr5e-4_r64a128_tseed0" / ds
                   / "no_ctrl" / "preference_summary.json", "test", 0.9)
    matrix = tio.load_cell_matrix("elephant", "dd-1b", "dd-1b-seed2", ["ep10_lr5e-4_r64a128"],
                                  tseed=0, cells=["d1", "no_ctrl"], output_root=tmp_path)
    assert matrix.cells == ["d1"] and matrix.no_control == ["no_ctrl"]
    assert np.isfinite(matrix.values).all()


def test_cell_set_skips_unusable_control_dirs_and_reports_layout(tmp_path):
    """Old-layout top-level control summaries count (same animal); empty/foreign dirs are not cells."""
    ds = "ex10-20_prefix_n100k"
    ctrl = tmp_path / "students/controls/base/pythia-160m-data-seed1/teacher-pythia-160m" / ds
    _write_summary(ctrl / "old" / "preference_summary.json", "test", 0.1)
    (ctrl / "old/preference_summary.json").write_text(json.dumps(
        {"target_animal": "dog", "summaries": [{"prompt_set": "test", "p_target_mean": 0.1}]}))
    _write_summary(ctrl / "new/eval/dog/preference_summary.json", "test", 0.2)
    (ctrl / "empty").mkdir(parents=True)  # neither layout -> not a cell
    (ctrl / "other").mkdir(parents=True)  # top-level summary for a different animal -> not a cell
    (ctrl / "other/preference_summary.json").write_text(json.dumps(
        {"target_animal": "cat", "summaries": [{"prompt_set": "test", "p_target_mean": 0.9}]}))
    for cell, value in (("old", 0.5), ("new", 0.6)):
        _write_summary(tmp_path / "students/dog/pythia-160m-data-seed1/teacher-pythia-160m"
                       / "t1_tseed1" / ds / cell / "preference_summary.json", "test", value)

    matrix = tio.load_cell_matrix("dog", "original", "ds1", ["t1"], output_root=tmp_path)
    assert matrix.cells == ["new", "old"]
    assert matrix.control_layout == {"new": "eval", "old": "top-level"}
    assert matrix.values[:, matrix.control_col].tolist() == pytest.approx([0.2, 0.1])


def test_drop_empty_pools_warns_and_keeps_control(tmp_path):
    _fake_tree(tmp_path)
    matrix = tio.load_cell_matrix("dog", "fs1", "original", ["t1", "absent"], output_root=tmp_path)
    with pytest.warns(UserWarning, match="no cell on the control cell set"):
        kept, dropped = tio.drop_empty_pools(matrix)
    assert dropped == ["absent"]
    assert kept.pools == ["t1", tio.CONTROL_LABEL] and kept.control_col == 1
    assert kept.values.shape == (2, 2) and np.isfinite(kept.values).all()
    assert tio.drop_empty_pools(kept) == (kept, [])


def test_stack_cell_matrices_pads_and_keeps_control_index(tmp_path):
    _fake_tree(tmp_path)
    wide = tio.load_cell_matrix("dog", "fs1", "original", ["t1", "t2"], output_root=tmp_path)
    narrow = tio.load_cell_matrix("dog", "fs1", "original", ["t1"], output_root=tmp_path)
    stacked = tio.stack_cell_matrices({"a": wide, "b": narrow})
    assert stacked.values.shape == (4, 3)
    assert stacked.arms == ["a", "b"]
    assert stacked.groups.tolist() == ["a", "a", "b", "b"]
    assert stacked.control_index.tolist() == [2, 2, 1, 1]
    # the narrow arm is padded on the right, and its control still sits at its own index
    assert np.isnan(stacked.values[2:, 2]).all()
    assert stacked.values[2, 1] == pytest.approx(0.1)
    assert stacked.n_treatment == {"a": 4, "b": 2}


def test_control_dataset_tag_pairs_n125k_treatment_with_n100k_control(tmp_path):
    """A sender pool generated at n125k is matched to the n100k base-pool control tree."""
    send = tio.base_name("pythia-410m-seed1")
    for cell, value in (("c1", 0.5), ("c2", 0.6)):
        _write_summary(tmp_path / "students/dog/pythia-160m" / f"teacher-{send}" / "t1_tseed1"
                       / "ex10-20_prefix_n125k" / cell / "preference_summary.json", "test", value)
        _write_summary(tmp_path / "students/controls/base/pythia-160m" / f"teacher-{send}"
                       / "ex10-20_prefix_n100k" / cell / "eval/dog/preference_summary.json",
                       "test", 0.1)
    kw = dict(output_root=tmp_path)
    # the default tag finds neither tree's n125k side
    assert tio.load_cell_matrix("dog", send, "orig", ["t1"], **kw).cells == ["c1", "c2"]
    assert tio.load_cell_matrix("dog", send, "orig", ["t1"], **kw).missing != []
    matrix = tio.load_cell_matrix("dog", send, "orig", ["t1"], dataset_tag="ex10-20_prefix_n125k",
                                  control_dataset_tag="ex10-20_prefix_n100k", **kw)
    assert matrix.missing == []
    assert np.allclose(matrix.values, [[0.5, 0.1], [0.6, 0.1]])
    assert tio.resolve_teachers(["t1"], animal="dog", sender=send, receiver="orig",
                                dataset_tag="ex10-20_prefix_n125k", output_root=tmp_path) \
        == (["t1"], [])


def _write_seed_eval(path, value: float, animal: str = "dog") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target_animal": animal, "sets": {"test": {"p_target": value}}}))


def test_pythia_sender_into_dd1b_explicit_pool_tree_and_ftseed(tmp_path):
    """Pythia-160m -> dd-1b: explicit numbers-base control tree, ftseed-only read."""
    recv, ds = tio.base_name("dd-1b"), "ex10-20_prefix_n100k"
    leaf = (tmp_path / "students/dog" / recv / "teacher-pythia-160m" / "t1_tseed1" / ds)
    ctrl = tmp_path / "students/controls/base" / recv / "teacher-pythia-160m" / ds
    for cell, seeds in (("c1", {1: 0.5}), ("c2", {1: 0.6, 2: 0.9, 3: 0.9})):
        _write_summary(leaf / cell / "preference_summary.json", "test",
                       float(np.mean(list(seeds.values()))))
        for s, v in seeds.items():
            _write_seed_eval(leaf / cell / f"ftseed{s}/preference_eval.json", v)
        _write_summary(ctrl / cell / "eval/dog/preference_summary.json", "test", 0.1)
        _write_seed_eval(ctrl / cell / "ftseed1/eval/dog/preference_eval.json", 0.1)
    agg = tio.load_cell_matrix("dog", "pythia-160m", "dd-1b", ["t1"], output_root=tmp_path)
    assert agg.cells == ["c1", "c2"] and agg.missing == []
    assert agg.values[:, 0].tolist() == pytest.approx([0.5, 0.8])  # seed mean on c2
    one = tio.load_cell_matrix("dog", "pythia-160m", "dd-1b", ["t1"], ftseed=1,
                               output_root=tmp_path)
    assert np.allclose(one.values, [[0.5, 0.1], [0.6, 0.1]])
    assert one.control_layout == {"c1": "eval", "c2": "eval"}
    # a seed with no control eval is not a matched cell
    assert tio.load_cell_matrix("dog", "pythia-160m", "dd-1b", ["t1"], ftseed=2,
                                output_root=tmp_path).cells == []


# ----------------------------------------------------------------------------------------------
# one-table readout
# ----------------------------------------------------------------------------------------------
def _synthetic_matrix() -> tuple[tio.CellMatrix, np.ndarray]:
    control = np.linspace(1.0, 2.0, 10)
    t1 = control + 5.0                        # wins everywhere, all 10 above the control max
    t2 = control + 0.5                        # wins everywhere, 5 above the control max
    short = np.r_[np.nan, control[1:] - 0.1]  # loses on its 9 cells
    values = np.column_stack([t1, t2, short, control])
    dead = np.zeros_like(values)
    dead[0, 0] = 1
    m = tio.CellMatrix(frame=pd.DataFrame(), values=values, cells=[f"c{i}" for i in range(10)],
                       pools=["t1", "t2", "s", tio.CONTROL_LABEL], control_col=3, missing=[])
    return m, dead


def test_significance_rows_per_teacher_then_pooled():
    m, dead = _synthetic_matrix()
    rows = tio.significance_rows(m, dead, 1.0, "A", ["t1", "t2"], short=["s"], n_perm=200)
    assert [r["teacher"] for r in rows] == ["t1", "t2", "s (9/10 cells, not in pool)",
                                            "all full-set teachers"]
    t1, t2, s, pooled = rows
    assert t1["wins / n"] == "10/10" and t1["exceed."] == 10 and t1["dead (treat/ctrl)"] == "1/0"
    assert t2["exceed."] == 5 and s["cells"] == "9" and s["wins / n"] == "0/9"
    assert pooled["pooled"] and pooled["exceed."] == 15 and "sign p" not in pooled
    assert t1["treat median ×prior"] == pytest.approx(6.5)
    # the closed-form exceedance p is gone: the count is descriptive, the permutation is the test
    assert not any(c.startswith("exceed. ") for c in tio.TABLE_COLUMNS)
    assert np.isfinite(pooled["perm p"])
    assert all("perm p" not in r for r in (t1, t2, s))
    # every row names its test, carries the prior, and max >= median on both sides
    assert [r["test"] for r in rows] == ["sign (per teacher)"] * 3 + [
        "stratified permutation (pooled, 2 teachers)"]
    assert all(r["prior"] == 1.0 for r in rows)
    for r in rows:
        assert r["treat max ×prior"] >= r["treat median ×prior"]
        assert r["ctrl max ×prior"] >= r["ctrl median ×prior"]
    assert t1["treat max ×prior"] == pytest.approx(7.0) and t1["ctrl max ×prior"] == pytest.approx(2.0)
    assert tio.TABLE_COLUMNS[:3] == ["arm", "teacher", "test"]
    for c in ("prior", "treat max ×prior", "ctrl max ×prior"):
        assert tio.TABLE_COLUMNS.index(c) > 0


def test_one_teacher_arm_still_gets_the_pooled_stratified_row():
    """With a single full-set teacher the permutation relabels the teacher/control pair."""
    m, dead = _synthetic_matrix()
    rows = tio.significance_rows(m, dead, 1.0, "A", ["t1"], n_perm=200)
    assert [r["teacher"] for r in rows] == ["t1", "all full-set teachers"]
    pooled = rows[1]
    expected = tio.pooled_permutation(m, ["t1"], n_perm=200)
    assert pooled["pooled"] and pooled["exceed."] == expected.result.observed
    assert pooled["perm p"] == pytest.approx(expected.result.p)
    assert pooled["test"] == "stratified permutation (single teacher)"


def test_short_teacher_exceedance_uses_only_paired_finite_cells():
    m, dead = _synthetic_matrix()
    # the short teacher misses cell 0, whose control value is the SMALLEST, so scoring it on the
    # full control column would change the threshold's denominators.
    s = tio.significance_rows(m, dead, 1.0, "A", ["t1"], short=["s"], n_perm=200)[1]
    assert s["wins / n"] == "0/9" and s["cells"] == "9"
    e = tt.exceedance_test(m.values[1:, 2], m.values[1:, 3])
    assert s["exceed."] == e.count
    assert e.m == 9 and e.n == 9


def test_held_out_row_sign_p_is_the_one_sided_paired_sign_test():
    m, dead = _synthetic_matrix()
    spec = {"name": "A held-out", "arm": "A", "teacher": "t2",
            "selection_cells": ["c0", "c1"], "n_held_out": 8}
    row = tio.held_out_row(spec, m, np.zeros_like(dead), 1.0)
    expected = tt.paired_sign_test(m.values[2:, 1], m.values[2:, 3])
    exc = tt.exceedance_test(m.values[2:, 1], m.values[2:, 3])
    assert row["wins / n"] == f"{expected.wins}/{expected.n}"
    assert row["sign p"] == pytest.approx(expected.p)
    # the exceedance count is reported on the held-out cells too, not "—"; no closed-form p
    assert row["exceed."] == exc.count
    assert not any(c.startswith("exceed. ") for c in row)
    assert row["cells"] == "8"
    assert row["test"] == "sign (held-out)" and row["prior"] == 1.0
    assert row["treat max ×prior"] >= row["treat median ×prior"]
    # one-sided, not the two-sided binomtest
    from scipy.stats import binomtest
    assert row["sign p"] != pytest.approx(binomtest(expected.wins, expected.n, 0.5).pvalue)


def test_style_highlights_the_primary_perm_p_not_the_sign_p():
    df = pd.DataFrame([{"arm": "A", "teacher": "t", "test": "sign (per teacher)",
                        "prior": 0.0096312, "cells": "10", "wins / n": "9/10",
                        "sign p": 0.0107, "exceed.": 3, "perm p": 0.2,
                        "treat median ×prior": 2.345, "treat max ×prior": 4.567,
                        "ctrl median ×prior": 1.0, "ctrl max ×prior": 1.234,
                        "dead (treat/ctrl)": "0/0", "pooled": False},
                       {"arm": "A", "teacher": "all", "cells": "10", "exceed.": 5,
                        "perm p": 4.2e-4, "pooled": True}]).reindex(
        columns=[*tio.TABLE_COLUMNS, "pooled"])
    html = tio.style_significance_table(df).to_html()
    assert "pooled" not in html and "0.011" in html and "4.2e-04" in html and "2.35" in html
    assert "0.00963" in html and "4.57" in html and "1.23" in html and "sign (per teacher)" in html
    ctx = tio.style_significance_table(df)._compute().ctx
    sign_col = tio.TABLE_COLUMNS.index("sign p") - 2
    exc_col = tio.TABLE_COLUMNS.index("perm p") - 2
    # sign p = 0.0107 is below 0.05 but is the SECONDARY test: never highlighted
    assert ("font-weight", "bold") not in ctx.get((0, sign_col), [])
    assert ("font-weight", "bold") not in ctx.get((0, exc_col), [])
    assert ("font-weight", "bold") in ctx[(1, exc_col)]
    assert any(k == "background-color" and "128" in v for k, v in ctx[(1, 0)])


def test_style_significance_table_groups_arms_with_divider():
    df = pd.DataFrame([{"arm": "A", "teacher": "t1", "pooled": False},
                       {"arm": "A", "teacher": "t2", "pooled": False},
                       {"arm": "B", "teacher": "t1", "pooled": False},
                       {"arm": "POOLED: x", "teacher": "all", "pooled": True},
                       {"arm": "held-out", "teacher": "t1", "pooled": False}]).reindex(
        columns=[*tio.TABLE_COLUMNS, "pooled"])
    html = tio.style_significance_table(df).to_html()
    assert html.count(">A</th>") == 1 and 'rowspan="2">A</th>' in html
    assert html.count(">t1</th>") == 3
    styler = tio.style_significance_table(df)
    ctx, ctx_index = styler._compute().ctx, styler.ctx_index
    divider = ("border-top", "2px solid currentColor")
    for row in (0, 2, 3, 4):
        assert divider in ctx[(row, 0)] and divider in ctx_index[(row, 0)]
    assert divider not in ctx.get((1, 0), [])
    assert any(k == "background-color" and "128" in v for k, v in ctx_index[(3, 1)])


# ----------------------------------------------------------------------------------------------
# CLI: the pooled line's design, and mode-scoped options
# ----------------------------------------------------------------------------------------------
def _cli_matrix() -> tio.CellMatrix:
    """Two pooled teachers over 10 cells, with ONE cell missing for the first teacher."""
    control = np.linspace(1.0, 2.0, 10)
    t1 = control + 5.0
    t1[3] = np.nan                            # the missing cell the two designs must agree on
    t2 = control + 0.5
    values = np.column_stack([t1, t2, control])
    return tio.CellMatrix(frame=pd.DataFrame(), values=values,
                          cells=[f"c{i}" for i in range(10)],
                          pools=["t1", "t2", tio.CONTROL_LABEL], control_col=2, missing=[])


def test_pooled_permutation_restricts_to_all_finite_cells():
    m = _cli_matrix()
    pooled = tio.pooled_permutation(m, ["t1", "t2"], n_perm=200)
    assert (pooled.n_cells, pooled.n_kept) == (10, 9)
    assert pooled.n_treatment == 18
    assert np.isfinite(pooled.values).all()


def _run_cli(monkeypatch, args, matrix=None, teachers=("t1", "t2")):
    from typer.testing import CliRunner

    import scripts.transfer_significance_readout as cli

    monkeypatch.setattr(cli.tio, "resolve_teachers",
                        lambda *a, **k: (list(teachers), []))
    monkeypatch.setattr(cli.tio, "load_prior", lambda *a, **k: 0.05)
    monkeypatch.setattr(cli.tio, "load_cell_matrix",
                        lambda *a, **k: matrix if matrix is not None else _cli_matrix())
    return CliRunner().invoke(cli.app, args)


def test_cli_pooled_line_matches_the_table_pooled_row(monkeypatch):
    """(6) one missing cell: the CLI line and the notebook table's pooled row agree."""
    m = _cli_matrix()
    res = _run_cli(monkeypatch, ["--eval-animals", "dog", "--n-perm", "200"], matrix=m)
    assert res.exit_code == 0, res.output
    table = tio.significance_rows(m, np.zeros_like(m.values), 0.05, "A", ["t1", "t2"],
                                  n_perm=200)[-1]
    assert table["pooled"]
    line = [ln for ln in res.output.splitlines()
            if ln.startswith("pooled stratified permutation")][0]
    assert f"observed {table['exceed.']} exceedances over 18 treatment cells" in line
    assert f"p {table['perm p']:.1e}" in line
    assert "9 of 10 cells kept, 1 dropped by the filter" in line
    # the per-teacher table has no closed-form exceedance p column any more
    assert "exc p" not in res.output


@pytest.mark.parametrize("args", [
    ["--dead", "drop"],
    ["--family", "full-seed", "--dead", "drop"],
])
def test_cli_rejects_compare_only_options_in_other_modes(monkeypatch, args):
    """(7) a mode that ignores --dead must refuse it, not run on (--ftseed is read in all)."""
    res = _run_cli(monkeypatch, [*args, "--eval-animals", "dog"])
    assert res.exit_code != 0
    assert "compare-mode only" in res.output


@pytest.mark.parametrize("mode_args", [[], ["--family", "full-seed"],
                                       ["--compare", "data-seed:full-seed"]])
def test_cli_zero_teachers_is_a_non_zero_exit(monkeypatch, mode_args):
    """(7) never a result computed on zero teachers, in any mode."""
    res = _run_cli(monkeypatch, [*mode_args, "--eval-animals", "dog"], teachers=())
    assert res.exit_code != 0
    assert "zero teachers" in res.output


@pytest.mark.parametrize("mode_args", [["--family", "full-seed"],
                                       ["--compare", "data-seed:full-seed"]])
def test_cli_all_on_disk_outside_single_arm_mode_exits(monkeypatch, tmp_path, mode_args):
    """(7) 'all-on-disk' scans one sender tree; family/compare mode fixes no sender, so it
    resolves to zero teachers -- which must be an error, not an empty result."""
    import scripts.transfer_significance_readout as cli

    # the real resolver, on an empty output root: the all-on-disk sentinel finds nothing
    assert tio.resolve_teachers(tio.ALL_ON_DISK, output_root=tmp_path) == ([], [])
    from typer.testing import CliRunner

    monkeypatch.setattr(cli.tio, "load_prior", lambda *a, **k: 0.05)
    monkeypatch.setattr(cli.tio, "load_cell_matrix", lambda *a, **k: _cli_matrix())
    monkeypatch.setattr(cli.tio, "resolve_teachers", lambda *a, **k: ([], []))
    res = CliRunner().invoke(cli.app, [*mode_args, "--teachers", tio.ALL_ON_DISK])
    assert res.exit_code != 0
    assert "all-on-disk" in res.output


# ----------------------------------------------------------------------------------------------
# the CLI reads ONE finetune seed, the same one the notebook tables pin
# ----------------------------------------------------------------------------------------------
def _two_ftseed_tree(root, *, cells=("c1", "c2")) -> None:
    """One arm whose cells each carry two finetune seeds and a seed-aggregated summary."""
    ds = "ex10-20_prefix_n100k"
    for cell in cells:
        leaf = (root / "students/dog/pythia-160m/teacher-pythia-160m-seed1"
                / "t1_tseed1" / ds / cell)
        _write_summary(leaf / "preference_summary.json", "test", 0.9)   # the seed MEAN
        _write_seed_eval(leaf / "ftseed1/preference_eval.json", 0.5)
        _write_seed_eval(leaf / "ftseed2/preference_eval.json", 1.3)
        ctrl = (root / "students/controls/base/pythia-160m/teacher-pythia-160m-seed1"
                / ds / cell)
        _write_summary(ctrl / "eval/dog/preference_summary.json", "test", 0.1)
        _write_seed_eval(ctrl / "ftseed1/eval/dog/preference_eval.json", 0.1)
        _write_seed_eval(ctrl / "ftseed2/eval/dog/preference_eval.json", 0.1)
    (root / "teachers/dog/pythia-160m").mkdir(parents=True, exist_ok=True)
    (root / "teachers/dog/pythia-160m/base_control.json").write_text(
        json.dumps({"test": {"p_target": 0.1}, "validation": {"p_target": 0.1}}))


def _cli_on_tree(tmp_path, monkeypatch, args):
    from typer.testing import CliRunner

    import scripts.transfer_significance_readout as cli
    from subliminal_learning import config

    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    return CliRunner().invoke(cli.app, args)


@pytest.mark.parametrize("mode_args, seed_value", [([], 0.5), (["--ftseed", "2"], 1.3)])
def test_cli_single_arm_reads_the_asked_for_ftseed_not_the_aggregate(tmp_path, monkeypatch,
                                                                    mode_args, seed_value):
    """The CLI's treatment value is the table's: one pinned seed, default 1, never the mean."""
    _two_ftseed_tree(tmp_path)
    res = _cli_on_tree(tmp_path, monkeypatch, ["--animal", "dog", "--sender", "fs1",
                                               "--receiver", "original", "--teachers", "t1",
                                               "--eval-animals", "dog", "--n-perm", "200",
                                               *mode_args])
    assert res.exit_code == 0, res.output
    ftseed = int(mode_args[1]) if mode_args else 1
    assert f"ftseed {ftseed} ===" in res.output
    matrix = tio.load_cell_matrix("dog", "fs1", "original", ["t1"], ftseed=ftseed,
                                  output_root=tmp_path)
    assert matrix.values[:, 0].tolist() == pytest.approx([seed_value, seed_value])
    expected = tt.prior_scaled_effect(matrix.values[:, 0], matrix.values[:, matrix.control_col],
                                      0.1)
    line = [ln for ln in res.output.splitlines() if ln.strip().startswith("t1 ")][0]
    assert float(line.split()[-1]) == pytest.approx(expected, abs=5e-5)
    # the seed-aggregated read (0.9) would have given a different number
    assert expected != pytest.approx(tt.prior_scaled_effect(np.full(2, 0.9), np.full(2, 0.1),
                                                            0.1))


def test_cli_family_mode_also_pins_the_ftseed(tmp_path, monkeypatch):
    calls = []
    import scripts.transfer_significance_readout as cli

    real = tio.load_cell_matrix
    monkeypatch.setattr(cli.tio, "load_cell_matrix",
                        lambda *a, **k: (calls.append(k.get("ftseed")), real(*a, **k))[1])
    _two_ftseed_tree(tmp_path)
    res = _cli_on_tree(tmp_path, monkeypatch, ["--family", "cross-seed", "--animal", "dog",
                                               "--teachers", "t1", "--n-perm", "200"])
    assert res.exit_code == 0, res.output
    assert calls and set(calls) == {1}


# ----------------------------------------------------------------------------------------------
# both medians describe the same cells
# ----------------------------------------------------------------------------------------------
def test_control_median_and_max_use_the_paired_cells_only():
    """A teacher present on the top cells only must not be compared with a control median
    computed over cells it never ran."""
    control = np.linspace(1.0, 2.0, 10)
    treatment = np.r_[np.full(5, np.nan), control[5:] + 1.0]
    values = np.column_stack([treatment, control])
    row = tio._summary_row("A", "t", values, np.zeros_like(values), 1.0, "sign (per teacher)")
    assert row["ctrl median ×prior"] == pytest.approx(np.median(control[5:]))
    assert row["ctrl median ×prior"] != pytest.approx(np.median(control))
    assert row["treat median ×prior"] == pytest.approx(np.median(control[5:] + 1.0))
    # the maxima are read on the same paired cells as the medians
    assert row["ctrl max ×prior"] == pytest.approx(control[5:].max())
    assert row["treat max ×prior"] == pytest.approx(control[5:].max() + 1.0)
    assert row["cells"] == "10"       # the control coverage column is unchanged


# ----------------------------------------------------------------------------------------------
# the notebook builder is repo-anchored
# ----------------------------------------------------------------------------------------------
def test_notebook_builder_writes_under_the_repo_and_uses_no_relative_paths():
    """Importing the builder must not write; its output path and its cells are repo-anchored."""
    from subliminal_learning.config import repo_root

    import scripts.build_transfer_significance_nb as builder

    assert builder.OUT.is_absolute()
    assert builder.OUT.parent == repo_root() / "notebooks"
    sources = "\n".join(src for _, src in builder.CELLS)
    assert "../run_logs" not in sources
    assert "bootstrap()" in sources


# ----------------------------------------------------------------------------------------------
# empty pooled designs: an explicit "insufficient data" outcome, never a zero-size reduction
# ----------------------------------------------------------------------------------------------
def _disjoint_matrix() -> tio.CellMatrix:
    """Two pooled teachers covering DISJOINT cells, so no cell is all-finite."""
    control = np.linspace(1.0, 2.0, 6)
    t1, t2 = control + 5.0, control + 5.0
    t1[3:], t2[:3] = np.nan, np.nan
    return tio.CellMatrix(frame=pd.DataFrame(), values=np.column_stack([t1, t2, control]),
                          cells=[f"c{i}" for i in range(6)],
                          pools=["t1", "t2", tio.CONTROL_LABEL], control_col=2, missing=[])


def test_pooled_permutation_rejects_an_empty_all_finite_design():
    """Disjoint coverage leaves zero cells: that is 'no design', not p = 1 over zero cells."""
    with pytest.raises(tio.InsufficientDesign, match="nothing to test"):
        tio.pooled_permutation(_disjoint_matrix(), ["t1", "t2"], n_perm=200)


def test_pooled_permutation_rejects_zero_pooled_teachers():
    with pytest.raises(tio.InsufficientDesign, match="no pooled teacher"):
        tio.pooled_permutation(_cli_matrix(), [], n_perm=200)


def test_table_pooled_row_notes_an_empty_design_instead_of_a_p():
    m = _disjoint_matrix()
    rows = tio.significance_rows(m, np.zeros_like(m.values), 0.05, "A", ["t1", "t2"], n_perm=200)
    pooled = rows[-1]
    assert pooled["pooled"] and pooled["teacher"] == "all full-set teachers"
    assert pooled["test"].startswith(tio.TEST_INSUFFICIENT)
    assert np.isnan(pooled["perm p"]) and pooled["cells"] == "0"
    assert set(tio.TABLE_COLUMNS) <= set(pooled)
    # and the note row still renders: the numeric columns are NaN, not a dash string
    assert tio.style_significance_table(pd.DataFrame(rows)).to_html()


def test_cli_pooled_line_says_insufficient_data_on_an_empty_design(monkeypatch):
    res = _run_cli(monkeypatch, ["--eval-animals", "dog", "--n-perm", "200"],
                   matrix=_disjoint_matrix())
    assert res.exit_code == 0, res.output
    line = [ln for ln in res.output.splitlines()
            if ln.startswith("pooled stratified permutation")][0]
    assert "insufficient data" in line and "nothing to test" in line
    assert "p 1.0e+00" not in res.output


def _registry_tree(root, *, cells=("c1", "c2")) -> None:
    """Two ftseed-1 arms into a receiver whose student trees carry NO base_control.json (the
    160m re-pretraining layout): one complete arm, one with students but no control evals."""
    ds, recv = "ex10-20_prefix_n100k", "pythia-160m-pilerepro-seed1234"
    for sender, with_control in (("pythia-160m", True), ("pythia-160m-pilerepro-seed3", False)):
        for tag, v in (("t1", 0.5), ("t2", 0.6)):
            for cell in cells:
                _write_seed_eval(root / "students/dog" / recv / f"teacher-{sender}"
                                 / f"{tag}_tseed1" / ds / cell / "ftseed1/preference_eval.json", v)
                if with_control:
                    _write_seed_eval(root / "students/controls/base" / recv / f"teacher-{sender}"
                                     / ds / cell / "ftseed1/eval/dog/preference_eval.json", 0.1)
    (root / "analysis").mkdir()
    (root / "analysis/base_eval_pr1234_animal_preference_test.json").write_text(
        json.dumps({"dog": {"p_per_prompt": [0.1, 0.3]}}))


def test_significance_table_base_eval_prior_dict_cells_and_note_row(tmp_path):
    """No tree base_control.json -> the base_eval test prior; a focus20-style cells JSON (dict
    entries) resolves; an in_progress arm with a `note` reports it, counts filled in, no p."""
    _registry_tree(tmp_path)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps(
        {"cells": [{"student_tag": "c1", "lr": 1e-4}, {"student_tag": "c2", "lr": 2e-4}]}))
    teachers = ["t1", "t2"]
    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test",
        "arms": [
            {"name": "orig → pr1234", "animal": "dog", "sender": "pythia-160m",
             "receiver": "pythia-160m-pilerepro-seed1234", "tseed": 1, "data_order": "same",
             "expected_prior": 0.2, "cells": "cells.json", "teachers": teachers},
            {"name": "pr3 → pr1234", "animal": "dog", "sender": "pythia-160m-pilerepro-seed3",
             "receiver": "pythia-160m-pilerepro-seed1234", "tseed": 1, "data_order": "same",
             "expected_prior": 0.2, "cells": "cells.json", "teachers": teachers,
             "in_progress": True,
             "note": "no matched controls ({controls}/{cells}); students {students}/{n_students}"},
        ], "pools": [], "held_out": []}))
    df = tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    pooled = df[df["pooled"]].iloc[0]
    assert pooled["arm"] == "orig → pr1234" and pooled["cells"] == "2"
    assert pooled["prior"] == pytest.approx(0.2) and pooled["exceed."] == 4
    note = df[df["arm"] == "pr3 → pr1234"].iloc[0]
    assert note["test"] == "no matched controls (0/2); students 4/4"
    assert note["cells"] == "in progress 0/2" and np.isnan(note["perm p"])
    # an expected_prior off by more than 5% is still an error with the base_eval source
    reg = json.loads((reg_dir / "arms.json").read_text())
    reg["arms"][0]["expected_prior"] = 0.3
    (reg_dir / "arms.json").write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="prior"):
        tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)


def test_significance_table_partial_arm_is_tested_on_the_cells_it_has(tmp_path):
    """A `partial` arm short of controls and of one student run is tested on the cells carrying
    both, its `cells` column reading tested/design; without the flag the same arm raises."""
    _registry_tree(tmp_path, cells=("c1", "c2", "c3", "c4"))
    ds, recv, sender = "ex10-20_prefix_n100k", "pythia-160m-pilerepro-seed1234", "pythia-160m"
    # c4: control never evaluated; c3: teacher t2's student run missing
    (tmp_path / "students/controls/base" / recv / f"teacher-{sender}" / ds / "c4"
     / "ftseed1/eval/dog/preference_eval.json").unlink()
    (tmp_path / "students/dog" / recv / f"teacher-{sender}" / "t2_tseed1" / ds / "c3"
     / "ftseed1/preference_eval.json").unlink()
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps({"cells": ["c1", "c2", "c3", "c4"]}))
    arm = {"name": "orig → pr1234", "animal": "dog", "sender": sender, "receiver": recv,
           "tseed": 1, "data_order": "same", "expected_prior": 0.2, "cells": "cells.json",
           "teachers": ["t1", "t2"]}
    reg = {"ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test", "arms": [arm],
           "pools": [], "held_out": []}
    (reg_dir / "arms.json").write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="short|covers"):
        tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    arm["partial"] = True
    (reg_dir / "arms.json").write_text(json.dumps(reg))
    df = tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    per_teacher = df[df["teacher"] == "t2"].iloc[0]
    assert per_teacher["cells"] == "2/4"          # t2's paired cells: c1, c2
    pooled = df[df["pooled"]].iloc[0]
    assert pooled["cells"] == "2/4" and pooled["exceed."] == 4 and not np.isnan(pooled["perm p"])


# ----------------------------------------------------------------------------------------------
# probe records (per-pool sidecars) and step-pinned receivers
# ----------------------------------------------------------------------------------------------
def _write_probe(run_dir, pool: str, delta_ce: float, *, native: bool) -> None:
    """A probe record where `probe_io.probe_path` puts it for `pool`."""
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "train_manifest.json").write_text(json.dumps(
        {"base_model_repo": "EleutherAI/pythia-160m"}))       # native pool: pp1
    name = "pile_probe.json" if native else f"pile_probe.{pool}.json"
    (run_dir / "eval" / name).write_text(json.dumps(
        {"pool_version": pool, "delta_ce": delta_ce}))


def test_dead_label_reads_the_native_record_and_the_sidecar(tmp_path):
    """An explicit non-native pool lives in `pile_probe.<pool>.json`; native is unchanged."""
    leaf = tmp_path / "cell"
    _write_probe(leaf / "ftseed1", "pp1", 9.0, native=True)
    assert tio._dead_label(leaf, 1, "pp1", 4.0) == 1.0
    assert tio._dead_label(leaf, 1, "pp1", 40.0) == 0.0

    # The cross-corpus cp1 readout of the SAME run: its own sidecar, not the pp1 file.
    assert np.isnan(tio._dead_label(leaf, 1, "cp1", 4.0))
    _write_probe(leaf / "ftseed1", "cp1", 1.0, native=False)
    assert tio._dead_label(leaf, 1, "cp1", 4.0) == 0.0
    assert tio._dead_label(leaf, 1, "pp1", 4.0) == 1.0        # native record untouched


def test_cell_matrix_finds_a_step_pinned_receiver_tree(tmp_path):
    """A `@step`-pinned receiver's students live under a gathered `ckpt/` level."""
    ds, recv = "ex10-20_prefix_n100k", "pythia-160m@step5000"
    for cell in ("c1", "c2"):
        _write_summary(tmp_path / "students/dog/ckpt" / recv / "teacher-pythia-160m-seed1"
                       / "t1_tseed1" / ds / cell / "preference_summary.json", "test", 0.5)
        _write_summary(tmp_path / "students/controls/base/ckpt" / recv
                       / "teacher-pythia-160m-seed1" / ds / cell
                       / "eval/dog/preference_summary.json", "test", 0.1)

    matrix = tio.load_cell_matrix("dog", "fs1", recv, ["t1"], output_root=tmp_path)
    assert matrix.cells == ["c1", "c2"] and matrix.missing == []
    assert matrix.values[0].tolist() == pytest.approx([0.5, 0.1])


def test_unpinned_receiver_tree_is_unchanged(tmp_path):
    """The same load on an unpinned receiver still reads the plain single-component tree."""
    _fake_tree(tmp_path)
    matrix = tio.load_cell_matrix("dog", "fs1", "original", ["t1"], output_root=tmp_path)
    assert matrix.cells == ["c1", "c2"]
    assert matrix.values[0].tolist() == pytest.approx([0.5, 0.1])
