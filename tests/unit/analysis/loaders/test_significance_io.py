"""The transfer-significance loaders (`analysis.loaders.transfer_significance_io` and its split-out cells / registry / frames modules) on synthetic student trees.
"""

from __future__ import annotations

import json
import logging
import shutil

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis.loaders import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt
from tests._helpers import (
    DS, DSET, FS,
    _family_tree,
)
from tests._helpers import _write_base_eval_prior
from tests._helpers import _write_seed_eval, _write_summary


# -------------------- from test_stats_transfer_significance.py --------------------

# ----------------------------------------------------------------------------------------------
# loader (fake tree)
# ----------------------------------------------------------------------------------------------
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
    _write_base_eval_prior(root, "original", "dog", 0.01)
    (root / "teachers/dog/pythia-160m").mkdir(parents=True, exist_ok=True)
    (root / "teachers/dog/pythia-160m/base_control.json").write_text(json.dumps(
        {"off_topic": {"p_target": 0.0001}}
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

def test_load_prior_is_the_base_eval_test_prior(tmp_path, caplog):
    """``test`` = the mean of the canonical base_eval artifact (token OR base dir name); any
    other prompt set still comes from the teachers tree; a missing artifact -> nan + remedy."""
    _fake_tree(tmp_path)
    assert tio.load_prior("dog", "original", "test", output_root=tmp_path) == pytest.approx(0.01)
    assert tio.load_prior("dog", "pythia-160m", "test",
                          output_root=tmp_path) == pytest.approx(0.01)
    assert tio.load_prior("dog", "original", "off_topic",
                          output_root=tmp_path) == pytest.approx(0.0001)
    with caplog.at_level("WARNING"):
        assert np.isnan(tio.load_prior("dog", "fs1", "test", output_root=tmp_path))
    assert "scripts/evaluate/base_test_prior.py" in caplog.text
    # an artifact without the asked-for animal is as missing as no artifact
    with caplog.at_level("WARNING"):
        assert np.isnan(tio.load_prior("cat", "original", "test", output_root=tmp_path))

def test_student_tree_prior_is_load_prior(tmp_path):
    """One test prior everywhere: the student trees' own base_control.json is not read."""
    _fake_tree(tmp_path)
    leaf = (tmp_path / "students/dog/pythia-160m/teacher-pythia-160m-seed1/t1_tseed1"
            / "ex10-20_prefix_n100k")
    (leaf / "c1").mkdir(parents=True, exist_ok=True)
    (leaf / "base_control.json").write_text(json.dumps({"test": {"p_target": 0.9}}))
    assert tio.student_tree_prior("dog", "original", output_root=tmp_path) == pytest.approx(0.01)
    assert (tio.student_tree_prior("dog", "original", output_root=tmp_path)
            == tio.load_prior("dog", "original", "test", output_root=tmp_path))

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

def test_significance_rows_pooled_first_then_per_teacher():
    m, dead = _synthetic_matrix()
    rows = tio.significance_rows(m, dead, 1.0, "A", ["t1", "t2"], short=["s"], n_perm=200)
    assert [r["teacher"] for r in rows] == ["all full-set teachers", "t1", "t2",
                                            "s (9/10 cells, not in pool)"]
    pooled, t1, t2, s = rows
    assert t1["wins / n"] == "10/10" and t1["exceed."] == 10 and t1["dead (treat/ctrl)"] == "1/0"
    assert t2["exceed."] == 5 and s["cells"] == "9" and s["wins / n"] == "0/9"
    assert pooled["pooled"] and pooled["exceed."] == 15 and "sign p" not in pooled
    assert t1["treat median ×prior"] == pytest.approx(6.5)
    # the closed-form exceedance p is gone: the count is descriptive, the permutation is the test
    assert not any(c.startswith("exceed. ") for c in tio.TABLE_COLUMNS)
    assert np.isfinite(pooled["perm p"])
    assert all("perm p" not in r for r in (t1, t2, s))
    # every row names its test, carries the prior, and max >= median on both sides
    assert [r["test"] for r in rows] == ["stratified permutation (pooled, 2 teachers)"] + [
        "sign (per teacher)"] * 3
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
    assert [r["teacher"] for r in rows] == ["all full-set teachers", "t1"]
    pooled = rows[0]
    expected = tio.pooled_permutation(m, ["t1"], n_perm=200)
    assert pooled["pooled"] and pooled["exceed."] == expected.result.observed
    assert pooled["perm p"] == pytest.approx(expected.result.p)
    assert pooled["test"] == "stratified permutation (single teacher)"

def test_short_teacher_exceedance_uses_only_paired_finite_cells():
    m, dead = _synthetic_matrix()
    # the short teacher misses cell 0, whose control value is the SMALLEST, so scoring it on the
    # full control column would change the threshold's denominators.
    s = tio.significance_rows(m, dead, 1.0, "A", ["t1"], short=["s"], n_perm=200)[2]
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

def test_style_significance_table_arm_block_pooled_first_label_once():
    """An arm block as `significance_rows` emits it (pooled row first, then its teachers) renders
    the arm label ONCE (row-span over the block) with the grey pooled row at the top."""
    m, dead = _synthetic_matrix()
    rows = tio.significance_rows(m, dead, 1.0, "A", ["t1", "t2"], short=["s"], n_perm=200)
    df = pd.DataFrame([*rows, *tio.significance_rows(m, dead, 1.0, "B", ["t1"], n_perm=200)]
                      ).reindex(columns=[*tio.TABLE_COLUMNS, "pooled"])
    assert df["teacher"].tolist()[:2] == ["all full-set teachers", "t1"]
    assert df["pooled"].tolist() == [True, False, False, False, True, False]
    styler = tio.style_significance_table(df)
    html = styler.to_html()
    assert html.count(">A</th>") == 1 and 'rowspan="4">A</th>' in html
    assert html.count(">B</th>") == 1 and 'rowspan="2">B</th>' in html
    styler._compute()
    ctx_index = styler.ctx_index
    divider = ("border-top", "2px solid currentColor")
    for row in (0, 4):                          # the pooled row opens each block
        assert divider in ctx_index[(row, 0)]
        assert any(k == "background-color" and "128" in v for k, v in ctx_index[(row, 1)])
    for row in (1, 2, 3, 5):
        assert divider not in ctx_index.get((row, 1), [])
        assert not any(k == "background-color" for k, v in ctx_index.get((row, 1), []))

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

    import notebooks.builders.build_transfer_significance_nb as builder

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
    pooled = rows[0]
    assert pooled["pooled"] and pooled["teacher"] == "all full-set teachers"
    assert [r["teacher"] for r in rows[1:]] == ["t1", "t2"]     # per-teacher rows still follow
    assert pooled["test"].startswith(tio.TEST_INSUFFICIENT)
    assert np.isnan(pooled["perm p"]) and pooled["cells"] == "0"
    assert set(tio.TABLE_COLUMNS) <= set(pooled)
    # and the note row still renders: the numeric columns are NaN, not a dash string
    assert tio.style_significance_table(pd.DataFrame(rows)).to_html()

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
    assert note["cells"] == "0/2" and np.isnan(note["perm p"])
    assert note["status"] == "⚠ not started, 0/2 controls" or note["status"].startswith("⚠ ")
    # an expected_prior off by more than 5% is still an error with the base_eval source
    reg = json.loads((reg_dir / "arms.json").read_text())
    reg["arms"][0]["expected_prior"] = 0.3
    (reg_dir / "arms.json").write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="prior"):
        tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)

def _short_registry(tmp_path, **arm_extra):
    """A 4-cell arm short of one control (c4) and one student run (t2 on c3), in a registry."""
    _registry_tree(tmp_path, cells=("c1", "c2", "c3", "c4"))
    ds, recv, sender = "ex10-20_prefix_n100k", "pythia-160m-pilerepro-seed1234", "pythia-160m"
    (tmp_path / "students/controls/base" / recv / f"teacher-{sender}" / ds / "c4"
     / "ftseed1/eval/dog/preference_eval.json").unlink()
    (tmp_path / "students/dog" / recv / f"teacher-{sender}" / "t2_tseed1" / ds / "c3"
     / "ftseed1/preference_eval.json").unlink()
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps({"cells": ["c1", "c2", "c3", "c4"]}))
    arm = {"name": "orig → pr1234", "animal": "dog", "sender": sender, "receiver": recv,
           "tseed": 1, "data_order": "same", "expected_prior": 0.2, "cells": "cells.json",
           "teachers": ["t1", "t2"], **arm_extra}
    reg = {"ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test", "arms": [arm],
           "pools": [{"name": "POOLED: one", "arms": ["orig → pr1234"]}], "held_out": []}
    (reg_dir / "arms.json").write_text(json.dumps(reg))
    return reg_dir / "arms.json"

def test_unfinished_set_is_never_tested_on_a_subset(tmp_path):
    """An arm short of a control or a student run has no p: without a flag it is an error, and
    the retired `partial` key is an error too -- no arm is tested on the cells it happens to have."""
    arms_json = _short_registry(tmp_path)
    with pytest.raises(ValueError, match="short|not a finished set"):
        tio.significance_table(arms_json, output_root=tmp_path)
    reg = json.loads(arms_json.read_text())
    reg["arms"][0]["partial"] = True
    arms_json.write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="partial"):
        tio.significance_table(arms_json, output_root=tmp_path)

def test_in_progress_arm_is_one_flagged_coverage_row(tmp_path):
    """`in_progress`: ONE row, no p, the note as test, `cells` = matched/design, a ⚠ status;
    a pool over it is a flagged row too, and the styler shades every flagged row amber."""
    arms_json = _short_registry(tmp_path, in_progress=True,
                                note="draining; students {students}/{n_students}")
    df = tio.significance_table(arms_json, output_root=tmp_path)
    assert list(df["arm"]) == ["orig → pr1234", "POOLED: one"]
    row = df.iloc[0]
    assert row["test"] == "draining; students 7/8" and row["cells"] == "2/4"
    assert row["status"] == "⚠ in progress 7/8 students, 3/4 controls"
    assert np.isnan(row["perm p"]) and row["prior"] == pytest.approx(0.2)
    assert df.iloc[1]["status"].startswith("⚠ member arm without a p")
    ctx = tio.style_significance_table(df)._compute().ctx
    assert all(any(tio.AMBER.split(": ")[1] in c for _, c in ctx.get((0, j), []))
               for j in range(df.shape[1] - 3))
    assert tio.flagged_arms(df, "4") == [
        ("4", "orig → pr1234", "⚠ in progress 7/8 students, 3/4 controls", "2/4"),
        ("4", "POOLED: one", "⚠ member arm without a p: orig → pr1234", "—")]

def test_untestable_arm_is_a_coverage_row_even_when_complete(tmp_path):
    """`untestable` keeps a COMPLETE arm off the tests: one ⚠ row with the reason, no p."""
    _registry_tree(tmp_path)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps({"cells": ["c1", "c2"]}))
    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test",
        "arms": [{"name": "orig → pr1234", "animal": "dog", "sender": "pythia-160m",
                  "receiver": "pythia-160m-pilerepro-seed1234", "tseed": 1, "data_order": "same",
                  "expected_prior": 0.2, "cells": "cells.json", "teachers": ["t1", "t2"],
                  "untestable": "single-cell pilot; students {students}/{n_students}"}],
        "pools": [], "held_out": []}))
    df = tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    assert len(df) == 1 and df.iloc[0]["status"] == "⚠ single-cell pilot; students 4/4"
    assert df.iloc[0]["test"] == "single-cell pilot; students 4/4"
    assert df.iloc[0]["cells"] == "2/2" and np.isnan(df.iloc[0]["perm p"])

def test_tested_arm_has_an_empty_status_and_no_amber(tmp_path):
    _registry_tree(tmp_path)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test",
        "arms": [{"name": "orig → pr1234", "animal": "dog", "sender": "pythia-160m",
                  "receiver": "pythia-160m-pilerepro-seed1234", "tseed": 1, "data_order": "same",
                  "expected_prior": 0.2, "cells": ["c1", "c2"], "teachers": ["t1", "t2"]}],
        "pools": [], "held_out": []}))
    df = tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    assert (df["status"] == "").all() and "status" in tio.TABLE_COLUMNS
    assert "255, 180, 0" not in tio.style_significance_table(df).to_html()
    assert tio.flagged_arms(df, "1") == []

def test_undeclared_stray_is_excluded_and_named_not_flagged(tmp_path):
    """An on-disk teacher the arm lists nowhere is auto-excluded (no raise, no `excluded`
    declaration needed): the arm is still TESTED on its listed teachers, and every row carries
    the unflagged `strays excluded: …` status."""
    _registry_tree(tmp_path)
    ds, recv = "ex10-20_prefix_n100k", "pythia-160m-pilerepro-seed1234"
    _write_summary(tmp_path / "students/dog" / recv / "teacher-pythia-160m"
                   / f"t_stray_tseed1/{ds}/c1/preference_summary.json", "test", 0.9)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test",
        "arms": [{"name": "orig → pr1234", "animal": "dog", "sender": "pythia-160m",
                  "receiver": "pythia-160m-pilerepro-seed1234", "tseed": 1, "data_order": "same",
                  "expected_prior": 0.2, "cells": ["c1", "c2"], "teachers": ["t1", "t2"]}],
        "pools": [], "held_out": []}))
    arms = json.loads((reg_dir / "arms.json").read_text())["arms"]
    ra = tio.load_registry_arm(arms[0], {a["name"]: a for a in arms}, reg_dir / "arms.json",
                               root=tmp_path)
    assert ra.strays == ["t_stray"] and ra.teachers == ["t1", "t2"]
    assert ra.finished and ra.matrix is not None and "t_stray" not in ra.matrix.pools

    df = tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    assert (df["status"] == "strays excluded: t_stray").all()
    assert df["perm p"].notna().any() and tio.flagged_arms(df, "1") == []
    assert "255, 180, 0" not in tio.style_significance_table(df).to_html()

def test_exclude_cells_holds_out_the_selection_cells(tmp_path):
    """`exclude_cells` drops the selection cells before loading (the `cells` column reads the
    held-out count); a cell outside the design is an error; an inline `cells` list and a
    `tags`-keyed cells JSON both resolve."""
    _registry_tree(tmp_path, cells=("c1", "c2", "c3"))
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "tags.json").write_text(json.dumps({"tags": ["c1", "c2", "c3"]}))
    arm = {"name": "orig → pr1234", "animal": "dog", "sender": "pythia-160m",
           "receiver": "pythia-160m-pilerepro-seed1234", "tseed": 1, "data_order": "same",
           "expected_prior": 0.2, "cells": "tags.json", "exclude_cells": ["c3"],
           "selection_note": "picked on c3", "teachers": ["t1", "t2"]}
    reg = {"ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test", "arms": [arm],
           "pools": [], "held_out": []}
    (reg_dir / "arms.json").write_text(json.dumps(reg))
    arms = {arm["name"]: arm}
    assert tio.registry_design_cells(arm, arms, reg_dir / "arms.json", ftseed=1) == ["c1", "c2"]
    inline = {**arm, "cells": ["c3", "c1", "c2"]}
    assert tio.registry_design_cells(inline, arms, reg_dir / "arms.json", ftseed=1) == ["c1", "c2"]
    df = tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    assert (df["cells"] == "2").all() and df[df["pooled"]].iloc[0]["exceed."] == 4
    assert tio.registry_selection_notes(reg_dir / "arms.json") == "- **orig → pr1234** — picked on c3"
    arm["exclude_cells"] = ["c9"]
    (reg_dir / "arms.json").write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="exclude_cells"):
        tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)

def test_arm_whose_trees_do_not_exist_is_a_not_started_row(tmp_path):
    """No student tree at all (a queued campaign): a coverage row, never an exception."""
    _registry_tree(tmp_path)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test",
        "arms": [{"name": "orig → pr1234 (chicken)", "animal": "chicken", "sender": "pythia-160m",
                  "receiver": "pythia-160m-pilerepro-seed1234", "tseed": 1, "data_order": "same",
                  "cells": ["c1", "c2"], "teachers": ["t1", "t2"], "in_progress": True,
                  "note": "queued; students {students}/{n_students}"}],
        "pools": [], "held_out": []}))
    (tmp_path / "analysis/base_eval_pr1234_animal_preference_test.json").write_text(
        json.dumps({"dog": {"p_per_prompt": [0.1, 0.3]}, "chicken": {"p_per_prompt": [0.01]}}))
    df = tio.significance_table(reg_dir / "arms.json", output_root=tmp_path)
    assert len(df) == 1 and df.iloc[0]["status"] == "⚠ not started, 0/2 controls"
    assert df.iloc[0]["cells"] == "0/2" and df.iloc[0]["test"] == "queued; students 0/4"

def test_family_existence_skipped_rows_are_amber_and_roll_up():
    df = pd.DataFrame([
        {"family": "f", "animal": "dog", "arm": "a->b", "cells": 2.0, "design cells": 2.0,
         "exceed.": 1.0, "perm p": 0.5, "note": "", "pooled": False},
        {"family": "f", "animal": "dog", "arm": "b->a", "note": "no control cells", "pooled": False},
        {"family": "f", "animal": "dog", "arm": "c->a", "note": "no control cells", "pooled": False},
        {"family": "f", "animal": "dog", "arm": "all arms", "cells": 2.0, "design cells": 2.0,
         "exceed.": 1.0, "perm p": 0.5, "note": "2 arms skipped: b->a; c->a", "pooled": True},
    ]).reindex(columns=[*tio.EXISTENCE_COLUMNS, "pooled"])
    ctx = tio.style_family_existence_table(df)._compute().ctx
    amber = tio.AMBER.split(": ")[1]
    assert any(amber in c for _, c in ctx.get((1, 0), []))
    assert not any(amber in c for _, c in ctx.get((0, 0), []))
    assert not any(amber in c for _, c in ctx.get((3, 0), []))    # pooled row: grey, has a p
    assert tio.flagged_family_arms(df, "2") == [
        ("2", "f dog: b->a, c->a", "⚠ no control cells", "2/3 arms")]
    roll = tio.lacking_data_table([("2", "f dog: b->a, c->a", "⚠ no control cells", "2/3 arms"),
                                   ("4", "x", "⚠ not started", "0/30")])
    assert list(roll.columns) == tio.LACKING_COLUMNS and len(roll) == 2

def test_flagged_missing_blocks_reads_the_stratified_row_once():
    df = pd.DataFrame([
        {"comparison": "A vs B", "design": "core56", "animal": "dog", "value mode": "absolute",
         "missing blocks": "1: x->y", "stratified": False},
        {"comparison": "A vs B", "design": "core56", "animal": "dog", "value mode": "relative",
         "missing blocks": "1: x->y", "stratified": False},
        {"comparison": "A vs B", "design": "core56", "animal": "dog+cat stratified",
         "value mode": "absolute", "missing blocks": "2: dog x->y, cat y->x", "stratified": True},
        {"comparison": "A vs B", "design": "core56", "animal": "dog+cat stratified",
         "value mode": "relative", "missing blocks": "2: dog x->y, cat y->x", "stratified": True},
        {"comparison": "C vs D", "design": "focus20", "animal": "dog", "value mode": "absolute",
         "missing blocks": "0", "stratified": False},
    ])
    assert tio.flagged_missing_blocks(df, "3") == [
        ("3", "A vs B (core56)", "⚠ 2 blocks kept out by the coverage rule", "dog x->y, cat y->x")]

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


# -------------------- from test_family_comparison.py --------------------

# ----------------------------------------------------------------------------------------------
# arm assignment
# ----------------------------------------------------------------------------------------------
def test_family_arms_twelve_per_family():
    arms = tio.family_arms(DS, FS, family_a="ds", family_b="fs")
    assert len(arms) == 24
    for family, tok in (("ds", "ds"), ("fs", "fs")):
        got = [a.arm for a in arms if a.family == family]
        assert len(got) == 12 and len(set(got)) == 12
        assert f"orig->{tok}1" in got and f"{tok}1->orig" in got
        assert f"{tok}1->{tok}2" in got and f"{tok}2->{tok}1" in got
    assert all(a.sender != a.receiver for a in arms)
    assert [a.family for a in arms] == ["ds"] * 12 + ["fs"] * 12

def test_family_arms_validates_families():
    with pytest.raises(ValueError, match="equally many"):
        tio.family_arms(DS, FS[:3])
    with pytest.raises(ValueError, match="share their first base"):
        tio.family_arms(["ds0", "ds1"], ["orig", "fs1"])
    with pytest.raises(ValueError, match="at least one seed"):
        tio.family_arms(["orig"], ["orig"])

def _frame(tmp_path, **kw):
    arms = tio.family_arms(DS, FS, family_a="ds", family_b="fs")
    return tio.normalized_effect_frame(arms, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                                       output_root=tmp_path, **kw)

def test_normalized_frame_carries_both_value_modes(tmp_path):
    _family_tree(tmp_path)
    frame, report = _frame(tmp_path)
    assert report["skipped_arms"] == []
    assert report["n_dead"] == 0 and report["notes"] == []
    assert set(frame["family"]) == {"ds", "fs"}
    # 24 arms x 2 cells x 2 teachers, each arm loaded once (no pair duplication)
    assert len(frame) == 96
    row = frame[(frame["arm"] == "orig->ds1") & (frame["cell"] == "c1")
                & (frame["teacher"] == "t1")].iloc[0]
    assert row["value_abs"] == pytest.approx(0.5 - 0.1)
    assert row["value_rel"] == pytest.approx((0.5 - 0.1) / 0.1)
    assert np.allclose(frame["value_rel"], frame["value_abs"] / 0.1)
    assert row["stratum"] == "c1"
    assert frame[frame["family"] == "fs"]["value_rel"].mean() < \
        frame[frame["family"] == "ds"]["value_rel"].mean()

def test_dead_keep_is_the_default_and_only_counts(tmp_path):
    """Dead labels are an outcome of the treatment, so the default keeps the runs."""
    _family_tree(tmp_path, dead_cell="c1")
    frame, report = _frame(tmp_path)
    assert report["n_dead_treatment"] == 24  # teacher t1 on cell c1, one per arm
    assert report["n_dead_control"] == 0 and report["n_dead"] == 24
    dead_rows = frame[(frame["cell"] == "c1") & (frame["teacher"] == "t1")]
    assert len(dead_rows) == 24 and dead_rows["dead_treatment"].all()
    assert len(frame) == 96

def test_dead_drop_reproduces_the_old_exclusions(tmp_path):
    _family_tree(tmp_path, dead_cell="c1")
    frame, report = _frame(tmp_path, dead="drop")
    assert report["n_dead_treatment"] == 24 and report["n_dead"] == 24
    assert frame[(frame["cell"] == "c1") & (frame["teacher"] == "t1")].empty
    assert not frame[(frame["cell"] == "c2") & (frame["teacher"] == "t1")].empty

def test_dead_drop_removes_dead_control_cells(tmp_path):
    _family_tree(tmp_path, arm_mode={"ds1->orig": "all-control-dead"})
    dropped, drop_report = _frame(tmp_path, dead="drop")
    assert drop_report["skipped_arms"] == ["ds1->orig: all 2 control cells dead"]
    assert "ds1->orig" not in set(dropped["arm"])
    # kept: the same arm stays in, its two control cells flagged and counted
    kept, keep_report = _frame(tmp_path)
    assert keep_report["skipped_arms"] == [] and keep_report["n_dead_control"] == 2
    assert kept[kept["arm"] == "ds1->orig"]["dead_control"].all()

def test_dead_policy_is_validated(tmp_path):
    with pytest.raises(ValueError, match="'keep' or 'drop'"):
        _frame(tmp_path, dead="skip")

def test_usable_cells_need_one_arm_of_each_family(tmp_path):
    """A cell only family ds reaches is out; a cell one arm of EACH family reaches is in."""
    _family_tree(tmp_path, extra_cells={"orig->ds1": ("solo", "both"),
                                        "orig->fs2": ("both",)})
    frame, report = _frame(tmp_path)
    assert report["skipped_arms"] == []
    assert "solo" not in set(frame["cell"])
    assert set(frame[frame["cell"] == "both"]["arm"]) == {"orig->ds1", "orig->fs2"}
    assert set(frame["stratum"]) == {"c1", "c2", "both"}

@pytest.mark.parametrize("mode, reason", [
    ("no-treatment", "ds1->orig: no treatment runs (0 finite values over 2 control cells)"),
    ("all-treatment-dead", "ds1->orig: all 4 treatment runs dead"),
    ("all-control-dead", "ds1->orig: all 2 control cells dead"),
])
def test_skip_reasons_distinguish_never_run_from_collapsed(tmp_path, mode, reason):
    """"Never ran" and "ran and died" must not share a skip string."""
    _family_tree(tmp_path, arm_mode={"ds1->orig": mode})
    _, report = _frame(tmp_path, dead="drop")
    assert report["skipped_arms"] == [reason]

def test_a_missing_anchor_raises_under_drop(tmp_path):
    """Dropping needs a label: with no anchored cut there is nothing to drop on."""
    _family_tree(tmp_path, dead_cell="c1", anchor=False)
    arms = tio.family_arms(DS, FS, family_a="ds", family_b="fs")
    with pytest.raises(ValueError, match="no anchored pp1 cut"):
        tio.normalized_effect_frame(arms, "dog", ["t1"], family_a="ds", family_b="fs",
                                    dead="drop", output_root=tmp_path)
    with pytest.raises(ValueError, match="unigram_anchor_pp1.json"):
        tio.compare_families(DS, FS, "dog", ["t1"], family_a="ds", family_b="fs", dead="drop",
                             cells=["c1", "c2"], n_perm=100, output_root=tmp_path)

def test_a_missing_anchor_under_keep_is_unlabelled_not_zero(tmp_path, caplog):
    """"0 dead" and "never labelled" read the same on a table: the counts come back None."""
    _family_tree(tmp_path, dead_cell="c1", anchor=False)
    arms = tio.family_arms(DS, FS, family_a="ds", family_b="fs")
    with caplog.at_level(logging.WARNING, logger=tio.__name__):
        frame, report = tio.normalized_effect_frame(arms, "dog", ["t1"], family_a="ds",
                                                    family_b="fs", output_root=tmp_path)
    assert len(frame) == 48
    assert report["n_dead"] is None and report["n_dead_treatment"] is None
    assert report["n_dead_control"] is None
    note = "no anchored pp1 cut"
    assert any(note in n for n in report["notes"])
    assert any(n.endswith("unigram_anchor_pp1.json), dead labelling unavailable")
               for n in report["notes"])
    assert report["dead_policy"] == tio.DEAD_POLICY_NONE == "none (anchor missing)"
    assert any(note in r.getMessage() for r in caplog.records)
    res = tio.compare_families(DS, FS, "dog", ["t1"], family_a="ds", family_b="fs",
                               cells=["c1", "c2"], n_perm=100, output_root=tmp_path)
    assert res.dead_policy == tio.DEAD_POLICY_NONE
    assert res.n_dead is None and res.n_dead_treatment is None

# ----------------------------------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------------------------------
def test_compare_families_end_to_end(tmp_path):
    _family_tree(tmp_path)
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               cells=["c1", "c2"], n_perm=2000, seed=0, output_root=tmp_path)
    assert set(res.modes) == set(tio.VALUE_MODES)
    rel, absolute = res.modes["relative"], res.modes["absolute"]
    assert list(rel.arms.columns) == tio.ARM_COLUMNS
    assert len(rel.arms) == 24 and rel.arms["cells"].tolist() == [2] * 24
    assert rel.arms["missing"].tolist() == [0] * 24
    # ds sits at 0.5, fs at 0.3, control 0.1, prior 0.1 -> contrast ~ 2.0
    assert rel.mean_contrast.observed == pytest.approx(2.0, abs=0.05)
    # the absolute mode is the same numbers over the prior of 0.1
    assert absolute.mean_contrast.observed == pytest.approx(0.2, abs=0.005)
    assert (rel.arms[rel.arms["family"] == "ds"]["exceed. frac"] == 1.0).all()
    # C(24, 12) is far above the enumeration cap, so the arm-relabel null is Monte Carlo
    s = rel.mean_contrast
    assert not s.exact and s.n_labelings == 2000
    assert s.p == s.floor and (s.n_a, s.n_b) == (12, 12)
    assert (s.n_cells, s.n_cells_dropped) == (2, 0)
    # every ds student beats every fs student in its cell, so the win probability is 1.0
    assert rel.win_probability.observed == pytest.approx(1.0)
    assert rel.win_probability.p == rel.win_probability.floor
    assert res.skipped_arms == []

def test_exclude_blocks_keeps_named_animal_arm_blocks_out(tmp_path):
    """``exclude_blocks`` drops the named (animal, arm) blocks before anything is tested and
    reports them; the default keeps every block (the old call path)."""
    _family_tree(tmp_path)
    kept = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                                cells=["c1", "c2"], n_perm=200, seed=0, output_root=tmp_path)
    gone = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                                cells=["c1", "c2"], n_perm=200, seed=0, output_root=tmp_path,
                                exclude_blocks={("dog", "orig->ds1")})
    kept_arms = set(kept.modes["relative"].arms["arm"])
    gone_arms = set(gone.modes["relative"].arms["arm"])
    assert "orig->ds1" in kept_arms
    assert gone_arms == kept_arms - {"orig->ds1"}
    assert any("orig->ds1" in s and "excluded" in s for s in gone.skipped_arms)

def test_scale_free_summaries_are_invariant_to_value_mode(tmp_path):
    """Exceedance fraction (arm level) and A-above-max-B (cell level) ignore a positive scale."""
    _family_tree(tmp_path)
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               cells=["c1", "c2"], n_perm=500, seed=0, output_root=tmp_path)
    rel, absolute = res.modes["relative"], res.modes["absolute"]
    # the two modes differ by the factor 1 / prior = 10
    assert rel.arms["exceed. frac"].tolist() == absolute.arms["exceed. frac"].tolist()
    assert rel.arm_level_exceed == absolute.arm_level_exceed
    # every receiver shares the prior 0.1 here, so the two modes cannot reorder a cell
    assert rel.win_probability == absolute.win_probability
    assert rel.mean_contrast.observed == pytest.approx(
        10 * absolute.mean_contrast.observed)

# ----------------------------------------------------------------------------------------------
# retention value modes: the gap over the receiver's / sender's diagonal arm mean
# ----------------------------------------------------------------------------------------------
def test_retention_values_hand_computed_with_arm_mean_denominator_and_floor(tmp_path):
    """orig->ds1 at c1, teacher t1: gap 0.4; receptivity = gap / D_ds1(c1) with D the ARM MEAN
    over both diagonal teachers (0.41, not t1's own 0.40); portability = gap / D_orig(c1)
    = 0.4 / 0.21. fs1's diagonal at c2 (D 0.02 < 0.05) is below the floor: every stratum into
    fs1 (receptivity) and out of fs1 (portability) at c2 is NaN and counted once."""
    _family_tree(tmp_path, diagonals=True)
    frame, report = _frame(tmp_path, cells=["c1", "c2"])
    row = frame[(frame["arm"] == "orig->ds1") & (frame["cell"] == "c1")
                & (frame["teacher"] == "t1")].iloc[0]
    assert row["value_abs"] == pytest.approx(0.4)
    assert row["D_receiver"] == pytest.approx(0.41) and row["D_sender"] == pytest.approx(0.21)
    assert row["value_recept"] == pytest.approx(0.4 / 0.41)
    assert row["value_port"] == pytest.approx(0.4 / 0.21)
    into_fs1 = frame[(frame["arm"].str.endswith("->fs1")) & (frame["cell"] == "c2")]
    out_fs1 = frame[(frame["arm"].str.startswith("fs1->")) & (frame["cell"] == "c2")]
    assert len(into_fs1) == 3 * 2 and into_fs1["value_recept"].isna().all()
    assert into_fs1["value_port"].notna().all()          # the senders' diagonals pass
    assert len(out_fs1) == 3 * 2 and out_fs1["value_port"].isna().all()
    assert frame[(frame["cell"] == "c1")][["value_recept", "value_port"]].notna().all().all()
    assert report["retention"] == {"receptivity": {"below_floor": 1, "no_diagonal": []},
                                   "portability": {"below_floor": 1, "no_diagonal": []}}
    d = report["diagonals"]
    assert list(d.columns) == tio.DIAGONAL_COLUMNS and len(d) == 7 * 2
    fs1_c2 = d[(d["base"] == "fs1") & (d["cell"] == "c2")].iloc[0]
    assert fs1_c2["D"] == pytest.approx(0.02) and fs1_c2["floor"] == pytest.approx(0.05)
    assert fs1_c2["complete"] and fs1_c2["covers design"] and not fs1_c2["passes floor"]
    ds1_c1 = d[(d["base"] == "ds1") & (d["cell"] == "c1")].iloc[0]
    assert ds1_c1["D"] == pytest.approx(0.41) and ds1_c1["passes floor"]
    assert ds1_c1["D_se"] == pytest.approx(np.std([0.4, 0.42], ddof=1) / np.sqrt(2))
    assert ds1_c1["prior"] == pytest.approx(0.1) and ds1_c1["ctrl"] == pytest.approx(0.1)

def test_retention_modes_run_the_same_tests_and_report_the_floor(tmp_path):
    _family_tree(tmp_path, diagonals=True)
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               cells=["c1", "c2"], n_perm=300, seed=0, output_root=tmp_path)
    assert list(res.modes) == list(tio.VALUE_MODES)
    for mode in tio.RETENTION_MODES:
        m = res.modes[mode]
        assert m.note == "" and m.n_below_floor == 1 and m.no_diagonal == ()
        assert list(m.arms.columns) == tio.ARM_COLUMNS and "median effect" in m.arms.columns
        assert m.mean_contrast is not None and m.win_probability.p <= 1
    # receptivity: a ds receiver's yardstick (0.41) is 2x the fs one (0.11 when passing), so
    # the ds arms' ratios (~1) sit BELOW the fs arms' (~0.2/0.11) -> negative contrast
    rec, port = res.modes["receptivity"], res.modes["portability"]
    ds = rec.arms[rec.arms["family"] == "ds"].set_index("arm")
    assert ds.loc["orig->ds1", "mean effect"] == pytest.approx(0.405 / 0.41, abs=0.01)
    assert ds.loc["orig->ds1", "median effect"] == pytest.approx(0.405 / 0.41, abs=0.01)
    assert rec.mean_contrast.observed < 0 < res.modes["relative"].mean_contrast.observed
    # arms whose only strata failed the floor keep the others: fs1 receivers lose c2 only
    assert rec.arms.set_index("arm").loc["orig->fs1", "cells"] == 1
    assert port.arms.set_index("arm").loc["fs1->orig", "cells"] == 1
    assert len(res.diagonals) == 14
    df = tio.family_comparison_table({("ds vs fs", "c1+c2"): res})
    assert df["value mode"].tolist() == list(tio.VALUE_MODES)
    rows = df.set_index("value mode")
    assert rows.loc["receptivity", "missing blocks"] == \
        "0; receptivity: 1 (r, cell) strata below floor"
    assert rows.loc["portability", "missing blocks"] == \
        "0; portability: 1 (s, cell) strata below floor"
    assert rows.loc["receptivity", "arms A vs B"] == "12 vs 12"
    assert np.isfinite(rows.loc[list(tio.RETENTION_MODES), "mean p"]).all()
    html = tio.style_family_comparison_table(df).to_html()
    assert "receptivity" in html and "portability" in html
    diag = tio.diagonals_table({("ds vs fs", "c1+c2"): res, ("again", "c1"): res})
    assert list(diag.columns) == tio.DIAGONALS_TABLE_COLUMNS and len(diag) == 7
    fs1 = diag.set_index("base").loc["fs1"]
    assert (fs1["cells"], fs1["cells passing floor"]) == (2, 1)
    assert fs1["median D/prior"] == pytest.approx((0.11 + 0.02) / 2 / 0.1)
    assert diag.set_index("base").loc["ds1", "median D/prior"] == pytest.approx(4.1)
    assert tio.style_diagonals_table(diag).to_html()

def test_retention_modes_without_diagonals_render_note_rows_and_never_raise(tmp_path):
    """No same-model students on the design (the core56 case): the relative and absolute rows
    are unchanged, the two retention rows carry the note and no p."""
    _family_tree(tmp_path)
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               cells=["c1", "c2"], n_perm=300, seed=0, output_root=tmp_path)
    for mode in tio.RETENTION_MODES:
        m = res.modes[mode]
        assert m.note == "no diagonals with controls" and m.mean_contrast is None
        assert set(m.no_diagonal) == set(DS) | set(FS) and m.arms.empty
    assert res.modes["relative"].mean_contrast.observed == pytest.approx(2.0, abs=0.05)
    df = tio.family_comparison_table({("ds vs fs", "core56"): res})
    rows = df.set_index("value mode")
    assert rows.loc["receptivity", "missing blocks"] == ("0; receptivity: no diagonals with "
                                                          "controls on core56 (ds1, ds2, ds3, "
                                                          "fs1, fs2, fs3, orig)")
    assert rows.loc[["receptivity", "portability"], "mean p"].isna().all()
    assert np.isfinite(rows.loc[["absolute", "relative"], "mean p"]).all()
    assert tio.style_family_comparison_table(df).to_html()
    assert tio.diagonals_table({("ds vs fs", "core56"): res}).empty

def test_diagonal_floor_is_cell_level_and_covers_design_is_separate(tmp_path):
    """A base complete on focus20 but not on the wider design: its cells still `pass floor` (so
    `diagonals_table` reads the same whichever design loaded them), while `covers design` is
    False and the retention values stay NaN."""
    _family_tree(tmp_path, diagonals=True)
    d = tio.diagonal_yardsticks("dog", ["ds1", "fs1"], ["t1", "t2"], cells=["c1", "c2", "c9"],
                                output_root=tmp_path)
    ds1 = d[d["base"] == "ds1"].set_index("cell")
    assert not ds1["covers design"].any() and not ds1.loc["c9", "complete"]
    assert ds1.loc["c1", "passes floor"] and ds1.loc["c1", "D"] == pytest.approx(0.41)
    assert not d[d["base"] == "fs1"].set_index("cell").loc["c2", "passes floor"]
    frame = pd.DataFrame({"arm": ["orig->ds1"], "cell": ["c1"], "value_abs": [0.4]})
    out, report = tio._add_retention_values(frame, d)
    assert np.isnan(out["value_recept"]).all() and report["receptivity"]["no_diagonal"] == ["ds1"]
    assert report["receptivity"]["below_floor"] == 0

def test_compare_families_raises_when_nothing_is_comparable(tmp_path):
    with pytest.raises(ValueError, match="nothing to compare"):
        tio.compare_families(DS, FS, "dog", ["t1"], cells=["c1"], n_perm=100,
                             output_root=tmp_path)

# ----------------------------------------------------------------------------------------------
# full-coverage rule
# ----------------------------------------------------------------------------------------------
DESIGN = ["c1", "c2", "c3"]

def _treatment_eval(root, arm: str, cell: str, tag: str):
    sender, receiver = (tio.base_name(t) for t in arm.split("->"))
    return (root / "students/dog" / receiver / f"teacher-{sender}" / f"{tag}_tseed1" / DSET
            / cell / "ftseed1/preference_eval.json")

def _control_eval(root, arm: str, cell: str):
    sender, receiver = (tio.base_name(t) for t in arm.split("->"))
    return (root / "students/controls/base" / receiver / f"teacher-{sender}" / DSET / cell
            / "ftseed1/eval/dog/preference_eval.json")

def test_full_coverage_excludes_partly_covered_arms_and_counts_missing(tmp_path):
    """A design cell with no control or no treatment at all drops the arm; one diverged run
    (its eval never written) leaves the arm in with ``missing = 1``."""
    _family_tree(tmp_path, cells=tuple(DESIGN))
    _control_eval(tmp_path, "ds1->orig", "c3").unlink()
    for tag in ("t1", "t2"):
        _treatment_eval(tmp_path, "ds2->orig", "c3", tag).unlink()
    _treatment_eval(tmp_path, "ds3->orig", "c3", "t1").unlink()
    frame, report = _frame(tmp_path, cells=DESIGN)
    assert "ds1->orig: partial coverage (2/3 design cells)" in report["skipped_arms"]
    assert "ds2->orig: partial coverage (2/3 design cells)" in report["skipped_arms"]
    assert len(report["skipped_arms"]) == 2
    assert report["missing"]["ds3->orig"] == 1 and report["missing"]["orig->ds1"] == 0
    assert {"ds1->orig", "ds2->orig"}.isdisjoint(set(frame["arm"]))
    assert report["design_cells"] == DESIGN

def test_arm_table_shows_the_missing_run_count(tmp_path):
    _family_tree(tmp_path, cells=tuple(DESIGN))
    _treatment_eval(tmp_path, "ds3->orig", "c3", "t1").unlink()
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               cells=DESIGN, n_perm=200, seed=0, output_root=tmp_path)
    arms = res.modes["relative"].arms.set_index("arm")
    assert arms.loc["ds3->orig", "missing"] == 1
    assert arms.loc["ds3->orig", "students"] == 5  # 2 teachers x 3 cells - 1
    assert arms["missing"].sum() == 1

def test_dead_label_follows_the_record_cache_nonfinite_convention(tmp_path):
    """A diverged adapter is +inf delta_ce in the record cache, i.e. dead by any cut."""
    leaf = tmp_path / "cell"
    path = leaf / "ftseed1/eval/pile_probe.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"pool_version": "pp1", "nonfinite_adapter": True,
                                "kl_base_student": None, "delta_ce": None}))
    assert tio._dead_label(leaf, 1, "pp1", 3.0) == 1.0
    path.write_text(json.dumps({"pool_version": "pp1", "nonfinite_adapter": False,
                                "kl_base_student": 0.1, "delta_ce": None}))
    assert np.isnan(tio._dead_label(leaf, 1, "pp1", 3.0))  # ran, no value: not measured
    path.write_text(json.dumps({"pool_version": "pp1", "kl_base_student": 0.1,
                                "delta_ce": 0.2}))
    assert tio._dead_label(leaf, 1, "pp1", 3.0) == 0.0

def test_arm_exceedance_fraction_is_per_cell_not_against_the_pooled_max_control():
    """One cell with a very high control must not deflate the other cells' wins."""
    rows = [{"cell": f"c{i}", "teacher": "t1", "treatment": 0.2, "control": 0.1,
             "value_abs": 0.1, "value_rel": 1.0, "dead_treatment": False,
             "dead_control": False} for i in range(9)]
    rows.append({"cell": "c9", "teacher": "t1", "treatment": 0.3, "control": 0.9,
                 "value_abs": -0.6, "value_rel": -6.0, "dead_treatment": False,
                 "dead_control": False})
    summary = tio._arm_summary(pd.DataFrame(rows), "value_abs", 0)
    # the pooled max-control rule would see 0 of 10 runs above 0.9
    assert summary["exceed. frac"] == pytest.approx(0.9)
    assert (summary["cells"], summary["students"], summary["missing"]) == (10, 10, 0)

# ----------------------------------------------------------------------------------------------
# family registry: the diagonal (same-model) arms and the named arm sets
# ----------------------------------------------------------------------------------------------
def test_same_model_arms_are_the_diagonal():
    arms = tio.same_model_arms(["orig", "fs1", "ds1"])
    assert [a.arm for a in arms] == ["orig->orig", "fs1->fs1", "ds1->ds1"]
    assert all(a.sender == a.receiver and a.family == "same-model" for a in arms)
    with pytest.raises(ValueError, match="distinct"):
        tio.same_model_arms(["orig", "orig"])
    with pytest.raises(ValueError, match="at least one base"):
        tio.same_model_arms([])

def test_family_arm_set_and_aliases():
    assert tio.resolve_family("full-seed") == "cross-seed"
    assert tio.resolve_family("data-seed") == "cross-data-order"
    assert len(tio.family_arm_set("data-seed")) == 12
    assert all(a.family == "cross-data-order" for a in tio.family_arm_set("data-seed"))
    same = tio.family_arm_set("same-model", family_label="sm")
    assert len(same) == 7 and all(a.sender == a.receiver and a.family == "sm" for a in same)
    with pytest.raises(ValueError, match="unknown family"):
        tio.resolve_family("weight-seed")

def test_resolve_cells_reads_the_student_tag_layout(tmp_path):
    path = tmp_path / "cells.json"
    path.write_text(json.dumps({"cells": [{"student_tag": "c1"}, {"student_tag": "c2"}]}))
    assert tio.resolve_cells(path) == ["c1", "c2"]
    assert tio.resolve_cells(["c1"]) == ["c1"]

# ----------------------------------------------------------------------------------------------
# several animals: the arm is one unit, the stratum is (animal, cell)
# ----------------------------------------------------------------------------------------------
def _two_animal_tree(tmp_path, *, drop_cat_arms=()):
    """The dog tree, plus a cat copy of every arm outside `drop_cat_arms`."""
    _family_tree(tmp_path)
    _copy_animal(tmp_path, "cat", drop_arms=drop_cat_arms)

def _copy_animal(tmp_path, animal, *, drop_arms=()):
    """Copy the dog tree into `animal`, leaving out every arm in `drop_arms`."""
    for path in sorted(tmp_path.glob("students/dog/**/preference_eval.json")):
        rel = path.relative_to(tmp_path / "students/dog")
        arm_sender = rel.parts[1].removeprefix("teacher-")
        if any(tio.base_name(a.split("->")[0]) == arm_sender and
               tio.base_name(a.split("->")[1]) == rel.parts[0] for a in drop_arms):
            continue
        out = tmp_path / "students" / animal / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(path.read_text())
        probe = path.parent.parent / "eval/pile_probe.json"
        if probe.is_file():
            (out.parent.parent / "eval").mkdir(parents=True, exist_ok=True)
            (out.parent.parent / "eval/pile_probe.json").write_text(probe.read_text())
    for summary in sorted(tmp_path.glob("students/dog/**/preference_summary.json")):
        out = tmp_path / "students" / animal / summary.relative_to(tmp_path / "students/dog")
        if out.parent.is_dir():
            out.write_text(summary.read_text())
    # the controls are animal-namespaced; add this animal's namespace next to the dog one
    for ctrl in sorted(tmp_path.glob("students/controls/base/**/eval/dog/preference_eval.json")):
        out = ctrl.parent.parent / animal / "preference_eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ctrl.read_text().replace('"dog"', f'"{animal}"'))
    for prior in sorted(tmp_path.glob("analysis/base_eval_*_animal_preference_test.json")):
        doc = json.loads(prior.read_text())
        doc[animal] = doc["dog"]
        prior.write_text(json.dumps(doc))

def _multi(tmp_path, animals=("dog", "cat"), **kw):
    return tio.compare_families(DS, FS, list(animals), ["t1", "t2"], family_a="ds", family_b="fs",
                                cells=["c1", "c2"], n_perm=300, seed=0, output_root=tmp_path,
                                **kw)

def test_multi_animal_arm_is_one_unit_across_animals(tmp_path):
    from math import comb

    _two_animal_tree(tmp_path)
    res = _multi(tmp_path)
    assert isinstance(res, tio.MultiAnimalComparison)
    assert res.animals == ["dog", "cat"] and set(res.per_animal) == {"dog", "cat"}
    s = res.stratified.modes["relative"].mean_contrast
    # 24 arms, not 48: the same model pair in both animals is ONE unit of the relabelling
    assert (s.n_a, s.n_b) == (12, 12)
    assert comb(24, 12) > s.n_labelings  # Monte Carlo above the enumeration cap
    assert res.stratified.frame["arm"].nunique() == 24
    assert set(res.stratified.frame["stratum"]) == {"dog|c1", "dog|c2", "cat|c1", "cat|c2"}
    assert s.n_cells == 4 and s.n_cells_dropped == 0
    assert all(sorted(v) == ["cat", "dog"] for v in res.blocks_present.values())
    assert res.missing_blocks == []

def test_multi_animal_arm_with_one_covered_block_enters_and_reports_the_other(tmp_path):
    _two_animal_tree(tmp_path, drop_cat_arms=("ds1->orig",))
    res = _multi(tmp_path)
    assert res.blocks_present["ds1->orig"] == ["dog"]
    assert any(m.startswith("cat ds1->orig:") for m in res.missing_blocks)
    assert res.stratified.modes["relative"].mean_contrast.n_a == 12
    assert "ds1->orig" not in set(res.per_animal["cat"].frame["arm"])
    assert "ds1->orig" in set(res.per_animal["dog"].frame["arm"])

def test_stratified_mean_contrast_is_the_mean_over_animal_cell_strata(tmp_path):
    """Hand-computed: per (animal, cell) mean(ds runs) - mean(fs runs), averaged over strata."""
    _two_animal_tree(tmp_path)
    res = _multi(tmp_path)
    frame = res.stratified.frame
    by_stratum = []
    for stratum, block in frame.groupby("stratum"):
        a = block[block["family"] == "ds"]["value_rel"].mean()
        b = block[block["family"] == "fs"]["value_rel"].mean()
        by_stratum.append(a - b)
    assert res.stratified.modes["relative"].mean_contrast.observed == pytest.approx(
        sum(by_stratum) / len(by_stratum))

def test_single_animal_call_path_still_returns_a_family_comparison(tmp_path):
    _family_tree(tmp_path)
    res = tio.compare_families("data-seed", "full-seed", "dog", ["t1", "t2"],
                               cells=["c1", "c2"], n_perm=200, seed=0, output_root=tmp_path)
    assert isinstance(res, tio.FamilyComparison)
    assert set(res.frame["family"]) == {"cross-data-order", "cross-seed"}

# ----------------------------------------------------------------------------------------------
# the two notebook tables
# ----------------------------------------------------------------------------------------------
def test_family_existence_table_has_one_pooled_row_per_animal(tmp_path):
    _two_animal_tree(tmp_path)
    df = tio.family_existence_table("data-seed", ["dog", "cat"],
                                    {"dog": ["t1", "t2"], "cat": ["t1", "t2"]},
                                    cells=["c1", "c2"], n_perm=200, seed=0, output_root=tmp_path)
    assert list(df.columns) == [*tio.EXISTENCE_COLUMNS, "pooled"]
    pooled = df[df["pooled"]]
    assert len(pooled) == 2 and pooled["animal"].tolist() == ["dog", "cat"]
    assert (pooled["arm"] == tio.POOLED_ARM).all()
    assert len(df) == 2 * (12 + 1)
    assert df[~df["pooled"]]["arm"].nunique() == 12
    assert (pooled["perm p"] < 0.05).all()  # every ds arm beats its control everywhere
    assert tio.style_family_existence_table(df).to_html()

def test_family_existence_table_counts_its_dead_runs_and_shows_the_design(tmp_path):
    """DEAD KEPT: a dead run stays in the arm it was measured in, and the row says so. `cells`
    is what the arm was tested on, `design cells` what was asked for."""
    _family_tree(tmp_path, dead_cell="c1")     # teacher t1 on cell c1, dead in every arm
    df = tio.family_existence_table("data-seed", ["dog"], {"dog": ["t1", "t2"]},
                                    cells=["c1", "c2"], n_perm=200, seed=0, output_root=tmp_path)
    arms = df[~df["pooled"]]
    assert len(arms) == 12
    assert (arms["dead (treat/ctrl)"] == "1/0").all()       # one dead treatment run per arm
    assert (arms["cells"] == 2).all() and (arms["design cells"] == 2).all()
    # every run is still there: the dead one was counted, never dropped
    assert (arms["treatment runs"] == 4).all()
    pooled = df[df["pooled"]].iloc[0]
    assert pooled["dead (treat/ctrl)"] == "12/0"
    assert pooled["cells"] == 24 and pooled["design cells"] == 2 * 12
    assert tio.style_family_existence_table(df).to_html()

def test_family_existence_table_marks_unlabelled_runs_when_the_anchor_is_missing(tmp_path):
    """No anchored cut: the count must not read "0 dead" -- nothing was labelled."""
    _family_tree(tmp_path, dead_cell="c1", anchor=False)
    df = tio.family_existence_table("data-seed", ["dog"], {"dog": ["t1", "t2"]},
                                    cells=["c1", "c2"], n_perm=200, seed=0, output_root=tmp_path)
    assert (df["dead (treat/ctrl)"] == tio.DEAD_UNLABELLED).all()
    assert tio.style_family_existence_table(df).to_html()

def test_family_comparison_table_one_row_per_block_and_mode(tmp_path):
    _two_animal_tree(tmp_path)
    res = _multi(tmp_path)
    df = tio.family_comparison_table({("ds vs fs", "c1+c2"): res})
    assert list(df.columns) == [*tio.COMPARISON_COLUMNS, "stratified"]
    # dog, cat and the stratified block, each in every value mode
    n_modes = len(tio.VALUE_MODES)
    assert len(df) == 3 * n_modes
    assert df["animal"].tolist() == (["dog"] * n_modes + ["cat"] * n_modes
                                     + ["dog+cat stratified"] * n_modes)
    assert df["value mode"].tolist() == list(tio.VALUE_MODES) * 3
    tested = df[df["value mode"].isin(["absolute", "relative"])]
    assert df[df["stratified"] & df["value mode"].eq("relative")]["arms A vs B"].tolist() == ["12 vs 12"]
    assert (tested["mean contrast"] > 0).all() and (tested["missing blocks"] == "0").all()
    # the tree has no diagonal students: the two retention modes render as note rows, no p
    retention = df[df["value mode"].isin(tio.RETENTION_MODES)]
    assert retention["mean p"].isna().all()
    assert (retention["missing blocks"] == "0; receptivity: no diagonals with controls on c1+c2 "
            "(ds1, ds2, ds3, fs1, fs2, fs3, orig)").sum() == 3
    assert tio.style_family_comparison_table(df).to_html()

def test_family_comparison_table_p_columns_are_the_two_sided_p(tmp_path):
    """The three p columns of `family_comparison_table` -- which the §0b pairing table and the
    registry class table reuse -- hold the tests' ``p_two_sided``, never the one-sided tail."""
    _two_animal_tree(tmp_path)
    res = _multi(tmp_path)
    df = tio.family_comparison_table({("ds vs fs", "c1+c2"): res})
    row = df[df["stratified"] & df["value mode"].eq("relative")].iloc[0]
    m = res.stratified.modes["relative"]
    for column, test in (("mean p", m.mean_contrast), ("win p", m.win_probability),
                         ("exceed. p", m.arm_level_exceed)):
        assert row[column] == pytest.approx(test.p_two_sided)
        assert test.p_two_sided == pytest.approx(min(1.0, 2 * min(test.p_upper, test.p_lower)))
        assert test.p == test.p_upper and test.p_opposite == test.p_lower

def test_four_animal_comparison_table_has_one_stratified_row_per_comparison_and_mode(tmp_path):
    """The notebook's four animals: per-animal rows plus ONE stratified row per (comparison, mode)."""
    import re

    import notebooks.builders.build_transfer_significance_nb as builder

    animals = json.loads(re.search(r"ANIMALS = (\[[^\]]*\])",
                                   "\n".join(src for _, src in builder.CELLS)).group(1))
    assert animals == ["dog", "cat", "lion", "pig"]

    _two_animal_tree(tmp_path)
    for animal in animals[2:]:
        _copy_animal(tmp_path, animal)
    res = _multi(tmp_path, animals=animals)
    assert res.animals == animals
    label = "+".join(animals) + " stratified"
    df = tio.family_comparison_table({("ds vs fs", "c1+c2"): res, ("ds vs fs", "c1"): res})
    n_modes = len(tio.VALUE_MODES)
    assert df["animal"].tolist() == ([*[a for a in animals for _ in tio.VALUE_MODES],
                                      *[label] * n_modes]) * 2
    stratified = df[df["stratified"]]
    assert len(stratified) == 2 * n_modes       # one per (comparison, mode)
    # the arm is still the unit (the retention rows carry no count: no diagonals on this tree)
    assert stratified[stratified["value mode"].isin(["absolute", "relative"])
                      ]["arms A vs B"].tolist() == ["12 vs 12"] * 4
    assert tio.style_family_comparison_table(df).to_html()

def test_compare_families_explicit_family_arm_lists(tmp_path):
    """Explicit `FamilyArm` lists give the same result as the base-list spelling of the same arms."""
    _family_tree(tmp_path)
    kw = dict(cells=["c1", "c2"], n_perm=500, seed=0, output_root=tmp_path)
    by_bases = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs", **kw)
    arms = tio.family_arms(DS, FS, family_a="ds", family_b="fs")
    explicit = tio.compare_families([a for a in arms if a.family == "ds"],
                                    [a for a in arms if a.family == "fs"], "dog", ["t1", "t2"], **kw)
    assert explicit.modes["relative"].mean_contrast == by_bases.modes["relative"].mean_contrast
    assert explicit.modes["relative"].arms.equals(by_bases.modes["relative"].arms)
    # any arm set works: a 2-vs-3 subset, labels read off the arms
    sub = tio.compare_families(arms[:2], arms[12:15], "dog", ["t1", "t2"], **kw)
    assert (sub.modes["relative"].mean_contrast.n_a, sub.modes["relative"].mean_contrast.n_b) == (2, 3)
    assert set(sub.modes["relative"].arms["family"]) == {"ds", "fs"}
    with pytest.raises(ValueError, match="same way"):
        tio.compare_families(arms[:2], FS, "dog", ["t1"], **kw)
    with pytest.raises(ValueError, match="ONE family label"):
        tio.compare_families(arms[:2], arms[1:14], "dog", ["t1"], **kw)


# -------------------- from test_transfer_significance_registry.py --------------------

DS_TAG, DS2_TAG = "ex10-20_prefix_n100k", "ex10-20_prefix_n125k"

SENDER, RECEIVER = "pythia-410m-seed1", "pythia-160m"

CELLS = ["c1", "c2"]

def _eval(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target_animal": "dog", "sets": {"test": {"p_target": value}}}))

def test_teacher_dataset_tags_reads_one_teacher_under_another_pool_size(tmp_path):
    """The override moves ONE teacher's tree to another dataset tag; the control stays put."""
    tags = {"t_n100k": DS_TAG, "t_n125k": DS2_TAG}
    for i, (tag, dataset_tag) in enumerate(tags.items()):
        leaf = (tmp_path / "students/dog" / RECEIVER / f"teacher-{SENDER}" / f"{tag}_tseed1"
                / dataset_tag)
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "base_control.json").write_text(json.dumps({"test": {"p_target": 0.2}}))
        for j, cell in enumerate(CELLS):
            _eval(leaf / cell / "ftseed1/preference_eval.json", 0.3 + 0.1 * i + 0.01 * j)
    for j, cell in enumerate(CELLS):
        _eval(tmp_path / "students/controls/base" / RECEIVER / f"teacher-{SENDER}" / DS_TAG / cell
              / "ftseed1/eval/dog/preference_eval.json", 0.1 + 0.01 * j)

    m = tio.load_cell_matrix("dog", SENDER, RECEIVER, list(tags), ftseed=1, cells=CELLS,
                             dataset_tag=DS_TAG, teacher_dataset_tags={"t_n125k": DS2_TAG},
                             output_root=tmp_path)
    assert m.cells == CELLS and m.pools == ["t_n100k", "t_n125k", tio.CONTROL_LABEL]
    assert m.missing == []
    assert m.values[:, 0] == pytest.approx([0.30, 0.31])
    assert m.values[:, 1] == pytest.approx([0.40, 0.41])      # the n125k teacher was found
    assert m.values[:, m.control_col] == pytest.approx([0.10, 0.11])   # control tree unaffected

    # without the override that teacher's tree is invisible (the control is unchanged)
    plain = tio.load_cell_matrix("dog", SENDER, RECEIVER, list(tags), ftseed=1, cells=CELLS,
                                 dataset_tag=DS_TAG, output_root=tmp_path)
    assert np.isnan(plain.values[:, 1]).all()
    assert plain.values[:, plain.control_col] == pytest.approx([0.10, 0.11])

    arm = {"name": "override arm", "animal": "dog", "sender": SENDER, "receiver": RECEIVER,
           "tseed": 1, "data_order": "different", "cells": CELLS, "teachers": list(tags),
           "teacher_dataset_tags": {"t_n125k": DS2_TAG}}
    loaded = tio.load_registry_arm(arm, {}, tmp_path / "arms.json", root=tmp_path, ftseed=1)
    assert loaded.finished and loaded.counts["students"] == 4 and loaded.strays == []


# -------------------- from test_transfer_significance_registry_compare.py --------------------


SENDER_CMP = "pythia-160m"

TEACHERS = ["t1", "t2"]

PRIOR = 0.2

#: Arm key -> a REAL base label: the receiver's prior is the canonical
#: ``analysis/base_eval_{short}_..._test.json`` artifact, which only exists for known bases.
RECV = {"a1": "data-seed1", "a2": "data-seed2", "b1": "weight-seed1", "b2": "weight-seed2",
        "partial": "seed1"}

def _recv(key: str) -> str:
    return f"pythia-160m-{RECV[key]}"

def _arm_tree(root, receiver: str, effect: float, *, control: bool = True) -> None:
    """One sender -> `receiver` arm: two teachers on two cells, each control + `effect`."""
    for tag in TEACHERS:
        _write_base_eval_prior(root, receiver, "dog", PRIOR)
        leaf = root / "students/dog" / receiver / f"teacher-{SENDER_CMP}" / f"{tag}_tseed1" / DS_TAG
        (leaf).mkdir(parents=True, exist_ok=True)
        for i, cell in enumerate(CELLS):
            _eval(leaf / cell / "ftseed1/preference_eval.json", 0.1 + 0.05 * i + effect)
            if control:
                _eval(root / "students/controls/base" / receiver / f"teacher-{SENDER_CMP}" / DS_TAG
                      / cell / "ftseed1/eval/dog/preference_eval.json", 0.1 + 0.05 * i)

def _registry(tmp_path) -> tuple:
    """Four finished arms (two per family), one unfinished arm, and the registry JSON."""
    receivers = {"a1": 0.30, "a2": 0.24, "b1": 0.06, "b2": 0.02}
    for key, effect in receivers.items():
        _arm_tree(tmp_path, _recv(key), effect)
    _arm_tree(tmp_path, _recv("partial"), 0.5, control=False)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps({"cells": CELLS}))

    def arm(key, **extra):
        return {"name": f"orig → {key}", "animal": "dog", "sender": SENDER_CMP,
                "receiver": _recv(key), "tseed": 1, "data_order": "cross",
                "cells": "cells.json", "teachers": TEACHERS, **extra}

    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test",
        "short_names": {_recv(k): k for k in [*receivers, "partial"]}
        | {SENDER_CMP: "orig"},
        "arms": [arm(k) for k in receivers]
        + [arm("partial", in_progress=True, note="no controls ({controls}/{cells})")],
        "pools": [], "held_out": []}))
    return reg_dir / "arms.json", ("same corpus", ["orig → a1", "orig → a2", "orig → partial"]), \
        ("different corpus", ["orig → b1", "orig → b2"])

def test_registry_family_comparison_skips_unfinished_arms_and_renders(tmp_path):
    arms_json, fam_a, fam_b = _registry(tmp_path)
    res = tio.registry_family_comparison(arms_json, fam_a, fam_b, n_perm=200, root=tmp_path)
    assert any(s.startswith("orig → partial:") for s in res.skipped_arms)
    assert set(res.frame["arm"]) == {"orig → a1", "orig → a2", "orig → b1", "orig → b2"}
    # 2 arms x 2 teachers x 2 cells per family, each row its own arm's prior
    assert len(res.frame) == 16
    assert res.frame["value_rel"].to_numpy() == pytest.approx(
        (res.frame["value_abs"] / PRIOR).to_numpy())
    rel = res.modes["relative"]
    assert (rel.mean_contrast.n_a, rel.mean_contrast.n_b) == (2, 2)
    assert rel.mean_contrast.observed > 0 and np.isfinite(rel.mean_contrast.p)
    assert np.isfinite(rel.win_probability.p) and np.isfinite(rel.arm_level_exceed.p)
    # the exact enumeration of C(4,2) labelings, floor untouched
    assert rel.mean_contrast.n_labelings == 6 and rel.mean_contrast.floor == pytest.approx(1 / 6)
    # the retention modes have no diagonals here: untested, with a note, never raising
    assert res.modes["receptivity"].mean_contrast is None
    assert res.modes["receptivity"].note
    table = tio.family_comparison_table({("A vs B", "2 cells"): res})
    assert len(table) == len(tio.VALUE_MODES)
    row = table[table["value mode"] == "relative"].iloc[0]
    assert np.isfinite(row["mean p"]) and np.isfinite(row["win p"]) and np.isfinite(row["exceed. p"])
    assert "orig → partial" in row["missing blocks"]
    assert tio.style_family_comparison_table(table).to_html()

def test_registry_family_comparison_unknown_arm_is_skipped_not_dropped(tmp_path):
    arms_json, fam_a, fam_b = _registry(tmp_path)
    res = tio.registry_family_comparison(arms_json, ("A", [*fam_a[1], "orig → nowhere"]), fam_b,
                                         n_perm=200, root=tmp_path)
    assert any("orig → nowhere: not an arm of" in s for s in res.skipped_arms)
    assert any("cells shared by both families" in n for n in res.notes)

def test_registry_paired_contrast_pairs_every_teacher_and_cell(tmp_path):
    arms_json, _, _ = _registry(tmp_path)
    res = tio.registry_paired_contrast(arms_json, "orig → a1", "orig → b1", n_perm=200,
                                       root=tmp_path)
    assert res["n_teachers"] == 2 and res["cells"] == 2
    assert res["n_pairs"] == len(TEACHERS) * len(CELLS)
    assert res["wins_a"] == res["n_pairs"]                     # a1 beats b1 on every pair
    assert res["mean_diff_abs"] == pytest.approx(0.30 - 0.06)
    assert res["mean_diff_rel"] == pytest.approx((0.30 - 0.06) / PRIOR)
    assert res["median_diff_rel"] == pytest.approx((0.30 - 0.06) / PRIOR)
    assert res["sign_p"] == pytest.approx(0.5 ** 4)
    # four equal positive diffs: only the all-plus draw matches, so the p sits near 1/16
    assert 1 / 201 < res["flip_p"] < 0.12
    assert res["arm_a"] == "orig → a1" and res["arm_b"] == "orig → b1"   # short_names rendering
    table = tio.registry_paired_contrast_table([res])
    assert list(table.columns) == tio.PAIRED_CONTRAST_COLUMNS
    assert table.iloc[0]["contrast"] == "orig → a1 vs orig → b1"
    assert tio.style_registry_paired_contrast_table(table).to_html()

def test_registry_paired_contrast_rejects_an_unfinished_arm(tmp_path):
    arms_json, _, _ = _registry(tmp_path)
    with pytest.raises(ValueError, match="orig → partial"):
        tio.registry_paired_contrast(arms_json, "orig → a1", "orig → partial", root=tmp_path)

def test_registry_family_comparison_run_unit_pools_the_arms_of_each_class(tmp_path):
    arms_json, fam_a, fam_b = _registry(tmp_path)
    res = tio.registry_family_comparison(arms_json, fam_a, fam_b, unit="run", n_perm=500,
                                         root=tmp_path)
    assert res.unit == "run"
    assert any(s.startswith("orig → partial:") for s in res.skipped_arms)
    assert any(n.startswith("runs: 8 in 'same corpus' (2 arms), 8 in") for n in res.notes)
    rel = res.modes["relative"]
    # the unit is now the RUN: 2 arms x 2 teachers x 2 cells per family
    assert (rel.mean_contrast.n_a, rel.mean_contrast.n_b) == (8, 8)
    assert rel.mean_contrast.statistic == "mean_contrast" and not rel.mean_contrast.exact
    assert rel.mean_contrast.floor == pytest.approx(1 / 501)
    # family A is above family B on every run, so all three tests sit at the floor
    assert rel.mean_contrast.p == pytest.approx(rel.mean_contrast.floor)
    assert rel.win_probability.observed == pytest.approx(1.0)
    # pooled exceedance fractions against each arm's OWN control max: 8/8 for A, 6/8 for B
    # (the b2 run on c1 lands under its arm's c2 control), so the contrast is 0.25
    assert rel.arm_level_exceed.observed == pytest.approx(0.25)
    assert rel.arm_level_exceed.statistic == "pooled_mean"
    assert res.modes["receptivity"].mean_contrast is None and res.modes["receptivity"].note
    arm_res = tio.registry_family_comparison(arms_json, fam_a, fam_b, n_perm=500, root=tmp_path)
    table = tio.family_comparison_table({("A vs B", "2 cells"): arm_res,
                                         ("A vs B (runs)", "2 cells"): res})
    assert set(table["unit"]) == {"arm", "run"}
    assert (table[table["comparison"] == "A vs B"]["unit"] == "arm").all()
    assert tio.style_family_comparison_table(table).to_html()

def test_registry_family_comparison_rejects_an_unknown_unit(tmp_path):
    arms_json, fam_a, fam_b = _registry(tmp_path)
    with pytest.raises(ValueError, match="unit"):
        tio.registry_family_comparison(arms_json, fam_a, fam_b, unit="cell", root=tmp_path)

# ----------------------------------------------------------------------------------------------
# the exceedance threshold is the arm's FULL design, and the guards
# ----------------------------------------------------------------------------------------------
def _wide_registry(tmp_path) -> tuple:
    """Two arms on DIFFERENT designs: A on (c1, c2), B on (c1, c2, c3) whose c3 control is the
    loudest of the tree and falls outside the shared cells."""
    _arm_tree(tmp_path, _recv("a1"), 0.30)
    _arm_tree(tmp_path, _recv("b1"), 0.06)
    for tag in TEACHERS:                       # B's extra design cell, with a very loud control
        leaf = tmp_path / "students/dog" / _recv("b1") / f"teacher-{SENDER_CMP}" / f"{tag}_tseed1" / DS_TAG
        _eval(leaf / "c3/ftseed1/preference_eval.json", 0.95)
        _eval(tmp_path / "students/controls/base" / _recv("b1") / f"teacher-{SENDER_CMP}" / DS_TAG
              / "c3/ftseed1/eval/dog/preference_eval.json", 0.90)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps({"cells": CELLS}))
    (reg_dir / "cells3.json").write_text(json.dumps({"cells": [*CELLS, "c3"]}))

    def arm(key, cells):
        return {"name": f"orig → {key}", "animal": "dog", "sender": SENDER_CMP,
                "receiver": _recv(key), "tseed": 1, "data_order": "cross",
                "cells": cells, "teachers": TEACHERS}

    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "prompt_set": "test",
        "arms": [arm("a1", "cells.json"), arm("b1", "cells3.json")],
        "pools": [], "held_out": []}))
    return reg_dir / "arms.json", ("A", ["orig → a1"]), ("B", ["orig → b1"])

def test_exceedance_threshold_is_the_arms_full_design_not_the_shared_cells(tmp_path):
    arms_json, fam_a, fam_b = _wide_registry(tmp_path)
    res = tio.registry_family_comparison(arms_json, fam_a, fam_b, unit="run", n_perm=200,
                                         root=tmp_path)
    frame = res.frame
    assert sorted(set(frame["cell"])) == CELLS                 # c3 is outside the shared set
    # ... but B keeps the threshold its dropped c3 control set, and A keeps its own
    assert set(frame[frame["arm"] == "orig → b1"]["arm_ctrl_max"]) == {0.90}
    assert frame[frame["arm"] == "orig → a1"]["arm_ctrl_max"].to_numpy() == pytest.approx(0.15)
    flags = tio._run_exceedance_flags(frame)
    assert flags[(frame["arm"] == "orig → b1").to_numpy()].tolist() == [0.0] * 4
    assert flags[(frame["arm"] == "orig → a1").to_numpy()].tolist() == [1.0] * 4
    # the naive per-row maximum would have been 0.15 for B too, and half its runs would "exceed"
    assert res.modes["relative"].arm_level_exceed.observed == pytest.approx(1.0)

def test_run_exceedance_flags_are_nan_on_an_unknown_treatment_or_threshold():
    frame = pd.DataFrame({"arm": ["a", "a", "b"], "treatment": [0.5, np.nan, 0.5],
                          "arm_ctrl_max": [0.1, 0.1, np.nan]})
    flags = tio._run_exceedance_flags(frame)
    assert flags[0] == 1.0 and np.isnan(flags[1]) and np.isnan(flags[2])

def test_registry_family_comparison_rejects_repeated_and_shared_arm_names(tmp_path):
    arms_json, fam_a, fam_b = _registry(tmp_path)
    with pytest.raises(ValueError, match="lists an arm twice"):
        tio.registry_family_comparison(arms_json, ("A", ["orig → a1", "orig → a1"]), fam_b,
                                       root=tmp_path)
    with pytest.raises(ValueError, match="both families"):
        tio.registry_family_comparison(arms_json, ("A", ["orig → a1"]), ("B", ["orig → a1"]),
                                       root=tmp_path)

def test_registry_frame_carries_the_animal_and_only_full_set_teachers(tmp_path):
    arms_json, fam_a, fam_b = _registry(tmp_path)
    reg = json.loads(arms_json.read_text())
    # t2 covers only c1 on the a1 arm: a known_short pool, reported by the registry, never pooled
    (tmp_path / "students/dog" / _recv("a1") / f"teacher-{SENDER_CMP}" / "t2_tseed1" / DS_TAG
     / "c2/ftseed1/preference_eval.json").unlink()
    for a in reg["arms"]:
        if a["name"] == "orig → a1":
            a["teachers"], a["known_short"] = ["t1"], {"t2": ["c2"]}
    arms_json.write_text(json.dumps(reg))
    res = tio.registry_family_comparison(arms_json, fam_a, fam_b, n_perm=200, root=tmp_path)
    a1 = res.frame[res.frame["arm"] == "orig → a1"]
    assert set(a1["teacher"]) == {"t1"} and len(a1) == 2
    assert set(res.frame["animal"]) == {"dog"}

def test_registry_paired_contrast_exposes_the_scale_and_both_tails(tmp_path):
    """The comparison's primary p is the two-sided sign-flip p: ``flip p`` in the table holds it,
    with the one-sided A > B tail in the secondary column beside it."""
    arms_json, _, _ = _registry(tmp_path)
    res = tio.registry_paired_contrast(arms_json, "orig → a1", "orig → b1", n_perm=200,
                                       root=tmp_path)
    assert res["scale"] == "relative"
    assert res["flip_p_opposite"] == 1.0 and res["flip_p"] < 0.12
    assert res["flip_p_two_sided"] == pytest.approx(2 * res["flip_p"])
    table = tio.registry_paired_contrast_table([res])
    assert list(table.columns) == tio.PAIRED_CONTRAST_COLUMNS
    assert table.iloc[0]["scale"] == "relative"
    assert table.iloc[0]["flip p"] == pytest.approx(res["flip_p_two_sided"])
    assert table.iloc[0][tio.PAIRED_CONTRAST_ONE_SIDED] == pytest.approx(res["flip_p"])
    assert tio.style_registry_paired_contrast_table(table).to_html()

def test_registry_paired_contrast_rejects_arms_that_differ_in_animal_or_tseed(tmp_path):
    arms_json, _, _ = _registry(tmp_path)
    reg = json.loads(arms_json.read_text())
    for a in reg["arms"]:
        if a["name"] == "orig → b1":
            a["tseed"] = 2
    arms_json.write_text(json.dumps(reg))
    for tag in TEACHERS:                       # the same runs, read at the other teacher seed
        one = tmp_path / "students/dog" / _recv("b1") / f"teacher-{SENDER_CMP}" / f"{tag}_tseed1"
        two = one.with_name(f"{tag}_tseed2")
        two.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(one, two)
    with pytest.raises(ValueError, match="differ in tseed"):
        tio.registry_paired_contrast(arms_json, "orig → a1", "orig → b1", root=tmp_path)
