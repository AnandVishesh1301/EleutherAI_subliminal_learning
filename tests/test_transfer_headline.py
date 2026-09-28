"""`analysis.transfer_headline`: per-arm -> across-arm triples, the finished-set drop/count,
and the stratified row's ≥ 2-animal rule -- on synthetic matrices and a toy student tree."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis import transfer_significance_io as tio

DS = "ex10-20_prefix_n100k"


def _matrix(values: np.ndarray, cells=None) -> tio.CellMatrix:
    n_cells, n_pools = values.shape
    cells = cells or [f"c{i}" for i in range(n_cells)]
    pools = [f"t{j}" for j in range(n_pools - 1)] + [tio.CONTROL_LABEL]
    return tio.CellMatrix(frame=pd.DataFrame(), values=values, cells=cells, pools=pools,
                          control_col=n_pools - 1, missing=[])


def test_arm_summary_and_triples_are_per_arm_then_across_arms():
    """Per arm: runs, exceedances over that arm's control max, and the five ratio stats;
    across arms: min / median / max of each per-arm stat (not of the pooled runs)."""
    # arm A: two teachers on two cells, control 0.1 / 0.2 -> control max 0.2
    a = _matrix(np.array([[0.4, 0.8, 0.1], [0.3, 0.1, 0.2]]))
    sa = th.arm_summary(a, prior=0.1)
    assert sa["runs"] == 4 and sa["exceed."] == 3
    assert sa["xprior"]["mean"] == pytest.approx(np.mean([4, 8, 3, 1]))
    assert sa["xprior"]["median"] == pytest.approx(3.5) and sa["xprior"]["max"] == 8
    assert sa["xcontrol"]["max"] == pytest.approx(8.0)          # 0.8 / 0.1
    assert sa["xcontrol"]["median"] == pytest.approx(np.median([4, 8, 1.5, 0.5]))
    # arm B: one teacher, higher ratios everywhere
    b = _matrix(np.array([[2.0, 0.1], [1.0, 0.1]]))
    sb = th.arm_summary(b, prior=0.1)
    assert sb["runs"] == 2 and sb["exceed."] == 2 and sb["xprior"]["max"] == 20
    tr = th.triples([sa["xprior"], sb["xprior"]])
    assert tr["max"] == (8.0, 14.0, 20.0)
    assert tr["median"] == (3.5, pytest.approx((3.5 + 15) / 2), 15.0)
    # a NaN per-arm value (a missing run's ratio) drops out of the ratio stats
    c = _matrix(np.array([[0.3, np.nan, 0.1]]))
    assert th.arm_summary(c, prior=0.1)["runs"] == 1
    single = th.triples([sa["xprior"]])
    assert single["mean"][0] == single["mean"][1] == single["mean"][2]


def _write_eval(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sets": {"test": {"p_target": value}}}))


def _toy_tree(root, *, animals=("dog", "cat"), skip=()) -> None:
    """Same-model arms orig->orig and fs1->fs1 on two cells, one teacher, ftseed 1, for
    `animals`; ``skip`` = (animal, base, cell) student runs left out (an unfinished arm)."""
    for animal in animals:
        for base in ("pythia-160m", "pythia-160m-seed1"):
            for cell in ("c1", "c2"):
                if (animal, base, cell) not in skip:
                    _write_eval(root / "students" / animal / base / f"teacher-{base}"
                                / "t1_tseed1" / DS / cell / "ftseed1/preference_eval.json",
                                0.5 if animal == "dog" else 0.4)
                _write_eval(root / "students/controls/base" / base / f"teacher-{base}" / DS
                            / cell / "ftseed1/eval" / animal / "preference_eval.json", 0.1)
            (root / "students" / animal / base / f"teacher-{base}" / "t1_tseed1" / DS
             / "base_control.json").write_text(json.dumps({"test": {"p_target": 0.05}}))


def test_finished_set_rule_drops_and_counts_a_short_arm(tmp_path):
    """An arm missing ONE student run is dropped (never scored on the cells it has), counted
    in `arms used` and named in `note`; the finished arm still gets its p."""
    _toy_tree(tmp_path, animals=("dog",), skip=(("dog", "pythia-160m-seed1", "c2"),))
    arms = th.family_headline_arms(tio.same_model_arms(["orig", "fs1"]), ["dog"],
                                   {"dog": ["t1"]}, cells=["c1", "c2"], output_root=tmp_path)
    assert [a.label for a in arms] == ["orig→orig", "fs1→fs1"]
    assert arms[0].matrix is not None and arms[0].prior == pytest.approx(0.05)
    assert arms[1].matrix is None and arms[1].dropped == "students 1/2, controls 2/2"
    fam = th.HeadlineFamily("same-model", "orig, fs1 (2)", "toy", "dog", arms)
    df = th.headline_table([fam], n_perm=200)
    assert len(df) == 1                       # single animal: no stratified row
    row = df.iloc[0]
    assert row["animal view"] == "dominant (dog)" and row["arms used"] == "1/2"
    assert row["runs"] == 2 and row["exceed."] == 2 and np.isfinite(row["perm p"])
    assert "dropped fs1→fs1 [dog]: students 1/2, controls 2/2" in row["note"]
    assert row["×prior max"] == "10 / 10 / 10" and row["×control median"] == "5 / 5 / 5"
    # a never-run arm reads "not started"
    (tmp_path / "students/dog/pythia-160m-seed1").rename(tmp_path / "gone")
    arms = th.family_headline_arms(tio.same_model_arms(["orig", "fs1"]), ["dog"],
                                   {"dog": ["t1"]}, cells=["c1", "c2"], output_root=tmp_path)
    assert arms[1].dropped == "not started"


def test_stratified_row_only_with_two_finished_animals(tmp_path):
    """Two animals with finished arms -> a grey stratified row whose strata are the (animal,
    arm) blocks; when the second animal's arms are all dropped, no stratified row."""
    _toy_tree(tmp_path, animals=("dog", "cat"))
    arms = th.family_headline_arms(tio.same_model_arms(["orig", "fs1"]), ["dog", "cat"],
                                   {"dog": ["t1"], "cat": ["t1"]}, cells=["c1", "c2"],
                                   output_root=tmp_path)
    fam = th.HeadlineFamily("same-model", "orig, fs1 (2)", "toy", "dog", arms)
    df = th.headline_table([fam], n_perm=200)
    assert list(df["animal view"]) == ["dominant (dog)", "stratified (dog, cat)"]
    assert list(df["pooled"]) == [False, True]
    assert df.iloc[1]["arms used"] == "4/4" and df.iloc[1]["runs"] == 8
    assert df.iloc[0]["×prior max"] == "10 / 10 / 10"
    assert df.iloc[1]["×prior max"] == "8 / 9 / 10"     # cat arms at 0.4 / 0.05 = 8
    # the styler shows the family once per block and greys the stratified row
    html = th.style_headline_table(df).to_html()
    assert html.count(">same-model<") == 1 and "rgba(128, 128, 128, 0.2)" in html
    # cat unfinished everywhere -> single row
    short = th.family_headline_arms(tio.same_model_arms(["orig", "fs1"]), ["dog", "cat"],
                                    {"dog": ["t1"], "cat": ["t1"]}, cells=["c1", "c2", "c3"],
                                    output_root=tmp_path)
    assert all(a.matrix is None for a in short)      # c3 has no control: every arm short
    df2 = th.headline_table([th.HeadlineFamily("same-model", "x", "toy", "dog", short)],
                            n_perm=200)
    assert len(df2) == 1 and df2.iloc[0]["note"].startswith("no finished arm")
    assert df2.iloc[0]["arms used"] == "0/2"


def test_registry_arms_go_through_the_registry_loader(tmp_path):
    """A registry arm is read by name through `load_registry_arm` (finished -> matrix on the
    listed teachers only; in_progress -> dropped with its counts; held-out selection noted)."""
    _toy_tree(tmp_path, animals=("dog",), skip=(("dog", "pythia-160m-seed1", "c2"),))
    reg = tmp_path / "arms.json"
    reg.write_text(json.dumps({
        "ftseed": 1, "prompt_set": "test",
        "arms": [
            {"name": "orig → orig (dog)", "animal": "dog", "sender": "pythia-160m",
             "receiver": "pythia-160m", "tseed": 1, "data_order": "same",
             "cells": ["c1", "c2"], "teachers": ["t1"]},
            {"name": "fs1 → fs1 (dog)", "animal": "dog", "sender": "pythia-160m-seed1",
             "receiver": "pythia-160m-seed1", "tseed": 1, "data_order": "same",
             "cells": ["c1", "c2"], "teachers": ["t1"], "in_progress": True},
        ],
        "held_out": [{"name": "h", "arm": "orig → orig (dog)", "teacher": "t1",
                      "selection_cells": ["c1"], "n_held_out": 1}]}))
    arms = th.registry_headline_arms(reg, ["orig → orig (dog)", "fs1 → fs1 (dog)"],
                                     output_root=tmp_path)
    assert arms[0].matrix is not None and arms[0].matrix.pools == ["t1", tio.CONTROL_LABEL]
    assert "teacher selected on 1 of the design cells (c1)" in arms[0].note
    assert arms[1].matrix is None and arms[1].dropped == "students 1/2, controls 2/2"


def test_include_selection_cells_reads_the_full_design(tmp_path):
    """``include_selection_cells`` ignores ``exclude_cells`` and the held-out note, so a
    pilot-selected arm is read on every design cell (the registry file is untouched)."""
    _toy_tree(tmp_path, animals=("dog",))
    reg = tmp_path / "arms.json"
    reg.write_text(json.dumps({
        "ftseed": 1, "prompt_set": "test",
        "arms": [
            {"name": "orig → orig (dog)", "animal": "dog", "sender": "pythia-160m",
             "receiver": "pythia-160m", "tseed": 1, "data_order": "same",
             "cells": ["c1", "c2"], "exclude_cells": ["c1"], "teachers": ["t1"]},
        ],
        "held_out": [{"name": "h", "arm": "orig → orig (dog)", "teacher": "t1",
                      "selection_cells": ["c1"], "n_held_out": 1}]}))
    held, = th.registry_headline_arms(reg, ["orig → orig (dog)"], output_root=tmp_path)
    full, = th.registry_headline_arms(reg, ["orig → orig (dog)"], output_root=tmp_path,
                                      include_selection_cells=True)
    assert held.matrix.cells == ["c2"] and "selection cell(s) excluded: c1" in held.note
    assert full.matrix.cells == ["c1", "c2"]
    assert "selection cell" not in full.note and "teacher selected on" not in full.note
    assert json.loads(reg.read_text())["arms"][0]["exclude_cells"] == ["c1"]


def _toy_family(name="same-model", animal="dog", n_cells=3) -> th.HeadlineFamily:
    """A family with one finished arm: every treatment run above the arm's control."""
    values = np.column_stack([np.full(n_cells, 0.6), np.full(n_cells, 0.1)])
    arm = th.HeadlineArm(f"orig→orig [{name}]", animal, _matrix(values), 0.05)
    return th.HeadlineFamily(name, "orig (1)", "toy", animal, [arm])


def test_headline_table_carries_the_null_columns(tmp_path):
    """`exceed. %` and `null 95th %` are the count columns as a share of the row's runs, and
    `null 95th` is the permutation's own 95th percentile."""
    fam = _toy_family()
    df = th.headline_table([fam], n_perm=200)
    row = df.iloc[0]
    assert row["runs"] == 3 and row["exceed."] == 3
    assert row["exceed. %"] == pytest.approx(100.0)
    assert row["null 95th %"] == pytest.approx(100 * row["null 95th"] / row["runs"])
    html = th.style_headline_table(df).to_html()
    assert "100.0%" in html
    # no finished arm -> the note row keeps the new columns blank
    empty = th.HeadlineFamily("x", "y", "toy", "dog",
                              [th.HeadlineArm("a→b", "dog", None, float("nan"),
                                              dropped="not started")])
    row2 = th.headline_table([empty], n_perm=200).iloc[0]
    assert pd.isna(row2["exceed. %"]) and pd.isna(row2["null 95th"])


def test_null_strip_figure_draws_one_row_per_finished_family():
    """One axes per family with a finished dominant arm (families without one are skipped),
    the five number columns in the axes' texts."""
    import matplotlib
    matplotlib.use("Agg")

    empty = th.HeadlineFamily("unfinished", "y", "toy", "dog",
                              [th.HeadlineArm("a→b", "dog", None, float("nan"),
                                              dropped="not started")])
    fams = [_toy_family("same-model"), empty, _toy_family("cross-seed", animal="elephant")]
    fig = th.null_strip_figure(fams, n_perm=200)
    assert len(fig.axes) == 2
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert labels[0].startswith("same-model") and "1 arm" in labels[0]
    assert labels[1].startswith("cross-seed")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "observed\ncount" in texts and "null 95th\n% of runs" in texts
    assert "3 / 3" in texts and "100.0%" in texts
    assert any("observed (elephant)" == t.get_text() for t in fig.legends[0].get_texts())


@pytest.mark.parametrize("n_families", [1, 12])
def test_null_strip_header_band_is_absolute_at_any_row_count(n_families):
    """The vertical margins are fixed inches, so at 1 row as at 12 the column headers sit
    ABOVE the first axes' top edge and BELOW the suptitle (figure coordinates)."""
    import matplotlib
    matplotlib.use("Agg")

    fams = [_toy_family(f"fam{i}") for i in range(n_families)]
    fig = th.null_strip_figure(fams, n_perm=200)
    top = fig.axes[0].get_position().y1
    headers = [t for t in fig.axes[0].texts if t.get_text().endswith("% of runs")]
    assert len(headers) == 2
    ys = [t.get_transform().transform((0, t.get_position()[1]))[1] / fig.bbox.height
          for t in headers]
    title_y = fig._suptitle.get_position()[1]
    for y in ys:
        assert top < y < title_y


def test_a_diverged_run_counts_as_complete_and_is_excluded(tmp_path):
    """A student whose leaf carries `tio.DIVERGED_MARKER` is a COMPLETE run of its arm under
    ``diverged="exclude"``: the arm is finished, tested on n-1 runs (its NaN reaches no
    statistic) and the excluded run is named in the note. The SAME missing run without the
    marker still drops the arm, under either setting."""
    skip = (("dog", "pythia-160m-seed1", "c2"),)
    _toy_tree(tmp_path, animals=("dog",), skip=skip)
    leaf = (tmp_path / "students/dog/pythia-160m-seed1/teacher-pythia-160m-seed1"
            / "t1_tseed1" / DS / "c2" / "ftseed1")
    kw = dict(cells=["c1", "c2"], output_root=tmp_path)
    # no marker: dropped whatever the setting
    for mode in ("drop", th.DIVERGED_EXCLUDE):
        arms = th.family_headline_arms(tio.same_model_arms(["orig", "fs1"]), ["dog"],
                                       {"dog": ["t1"]}, diverged=mode, **kw)
        assert arms[1].matrix is None and arms[1].dropped == "students 1/2, controls 2/2"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / tio.DIVERGED_MARKER).write_text(json.dumps({"reason": "nonfinite adapter weights"}))
    # marker, old behaviour: still dropped
    arms = th.family_headline_arms(tio.same_model_arms(["orig", "fs1"]), ["dog"],
                                   {"dog": ["t1"]}, **kw)
    assert arms[1].matrix is None
    # marker, new behaviour: finished, one run short, named
    arms = th.family_headline_arms(tio.same_model_arms(["orig", "fs1"]), ["dog"],
                                   {"dog": ["t1"]}, diverged=th.DIVERGED_EXCLUDE, **kw)
    diverged_arm = arms[1]
    assert diverged_arm.matrix is not None and diverged_arm.dropped is None
    assert diverged_arm.diverged == (("t1", "c2"),)
    assert diverged_arm.note == "1 diverged run excluded (teacher t1, cell c2)"
    assert th.arm_summary(diverged_arm.matrix, diverged_arm.prior)["runs"] == 1
    fam = th.HeadlineFamily("same-model", "orig, fs1 (2)", "toy", "dog", arms)
    row = th.headline_table([fam], n_perm=200).iloc[0]
    assert row["arms used"] == "2/2" and row["runs"] == 3      # 2 + 1, the diverged one out
    assert "1 diverged run excluded (teacher t1, cell c2)" in row["note"]
