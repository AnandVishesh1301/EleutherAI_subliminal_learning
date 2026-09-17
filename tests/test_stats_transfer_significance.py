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
# topk_mean / prior_scaled_effect / conformal
# ----------------------------------------------------------------------------------------------
def test_topk_mean():
    assert tt.topk_mean([1, 2, 3, 4, 5, 6], k=3) == pytest.approx(5.0)
    assert tt.topk_mean([1.0, np.nan], k=5) == pytest.approx(1.0)
    assert np.isnan(tt.topk_mean([np.nan]))


def test_prior_scaled_effect():
    assert tt.prior_scaled_effect([0.3, 0.2], [0.1, 0.1], 0.1) == pytest.approx(1.5)
    assert np.isnan(tt.prior_scaled_effect([np.nan], [0.1], 0.1))


def test_conformal_rank_p():
    assert tt.conformal_rank_p([0.9, 1.0, 1.1], np.linspace(0, 0.5, 10)) == pytest.approx(1 / 11)
    assert tt.conformal_rank_p(1.0, np.linspace(0, 0.5, 10)) == pytest.approx(1 / 11)


def test_no_multiplicity_helpers():
    """The brief's amendment: no Holm/Bonferroni wrappers live here."""
    assert not hasattr(tt, "holm") and not hasattr(tt, "bonferroni")


# ----------------------------------------------------------------------------------------------
# pool-rank permutation
# ----------------------------------------------------------------------------------------------
def _pool_frame(n_pools: int, n_cols: int, control_value: float, rng) -> pd.DataFrame:
    index = [f"pool{i}" for i in range(n_pools - 1)] + [tio.CONTROL_LABEL]
    data = rng.uniform(0.4, 0.6, size=(n_pools, n_cols))
    data[-1, :] = control_value
    return pd.DataFrame(data, index=index, columns=[f"r{c}" for c in range(n_cols)])


def test_pool_rank_exact_control_lowest():
    rng = np.random.default_rng(0)
    stats = {f"s{i}": _pool_frame(11, 3, 0.0, rng) for i in range(4)}
    res = tt.pool_rank_permutation(stats, tio.CONTROL_LABEL)
    assert res.exact and res.n_labelings == 11 ** 4
    assert res.observed == pytest.approx(1.0)
    assert res.p == pytest.approx(1 / 11 ** 4)
    assert set(res.per_sender_rank.values()) == {1}


def test_pool_rank_control_median():
    rng = np.random.default_rng(1)
    stats = {}
    for s in range(2):
        frame = _pool_frame(11, 2, 0.5, rng)
        frame.iloc[:10, :] = np.tile(np.linspace(0.0, 1.0, 10)[:, None], (1, 2))
        stats[f"s{s}"] = frame
    res = tt.pool_rank_permutation(stats, tio.CONTROL_LABEL)
    assert res.p == pytest.approx(0.5, abs=0.15)


def test_pool_rank_monte_carlo_path():
    rng = np.random.default_rng(2)
    stats = {f"s{i}": _pool_frame(11, 2, 0.5, rng) for i in range(4)}
    res = tt.pool_rank_permutation(stats, tio.CONTROL_LABEL, max_exact=10, n_mc=500, seed=0)
    assert not res.exact and res.n_labelings == 500
    assert 1 / 501 <= res.p <= 1.0


def test_pool_rank_skips_single_pool_sender():
    rng = np.random.default_rng(4)
    stats = {"s0": _pool_frame(11, 2, 0.0, rng),
             "s1": _pool_frame(1, 2, 0.0, rng)}
    with pytest.warns(UserWarning, match="fewer than 2 pools"):
        res = tt.pool_rank_permutation(stats, tio.CONTROL_LABEL)
    assert list(res.per_sender_rank) == ["s0"]


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


def test_load_pool_stats_shape(tmp_path):
    _fake_tree(tmp_path)
    stats = tio.load_pool_stats("dog", ["fs1"], ["original"], {"fs1": ["t1", "t2"]},
                                output_root=tmp_path)
    frame = stats["fs1"]
    assert list(frame.index) == ["t1", "t2", tio.CONTROL_LABEL]
    assert list(frame.columns) == ["original"]
    assert frame.loc[tio.CONTROL_LABEL, "original"] == pytest.approx(0.1)


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
    # one full-set teacher: no pooled row (it would duplicate the teacher row)
    assert len(tio.significance_rows(m, dead, 1.0, "A", ["t1"])) == 1


def test_style_significance_table_highlights_only_p_below_05():
    df = pd.DataFrame([{"arm": "A", "teacher": "t", "cells": "10", "wins / n": "9/10",
                        "sign p": 0.0107, "exceed.": 3, "exceed. p": 0.2,
                        "treat median ×prior": 2.345, "ctrl median ×prior": 1.0,
                        "dead (treat/ctrl)": "0/0", "pooled": False},
                       {"arm": "A", "teacher": "all", "cells": "10", "exceed.": 5,
                        "exceed. p": 4.2e-4, "pooled": True}]).reindex(
        columns=[*tio.TABLE_COLUMNS, "pooled"])
    html = tio.style_significance_table(df).to_html()
    assert "pooled" not in html and "0.011" in html and "4.2e-04" in html and "2.35" in html
    ctx = tio.style_significance_table(df)._compute().ctx
    sign_col, exc_col = tio.TABLE_COLUMNS.index("sign p") - 2, tio.TABLE_COLUMNS.index("exceed. p") - 2
    assert ("font-weight", "bold") in ctx[(0, sign_col)]
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
