"""Hand-checkable values for the family-comparison tests (arm-relabel + arm-level)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from scripts.transfer_significance_readout import app
from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt
from tests.test_stats_transfer_significance import (
    _write_base_eval_prior,
    _write_seed_eval,
    _write_summary,
)


# ----------------------------------------------------------------------------------------------
# arm-level arm-set permutation
# ----------------------------------------------------------------------------------------------
def _arm_frame(rows) -> pd.DataFrame:
    """``(family, value)`` rows as a one-row-per-arm frame."""
    return pd.DataFrame(rows, columns=["family", "value"])


def test_arm_permutation_exact_two_vs_two():
    """Both A arms above both B arms: 1 of C(4,2) = 6 labelings is that extreme."""
    res = tt.arm_set_permutation(
        _arm_frame([("A", 3.0), ("A", 2.0), ("B", 1.0), ("B", 0.0)]),
        family_a="A", family_b="B")
    assert res.exact and res.n_labelings == 6
    assert res.p == pytest.approx(1 / 6) == res.floor
    assert res.observed == pytest.approx(2.5 - 0.5)
    assert (res.n_a, res.n_b) == (2, 2)


def test_arm_permutation_identical_values_is_p_one():
    res = tt.arm_set_permutation(
        _arm_frame([("A", 1.0), ("A", 1.0), ("B", 1.0), ("B", 1.0)]),
        family_a="A", family_b="B")
    assert res.observed == pytest.approx(0.0) and res.p == 1.0


def test_arm_permutation_unequal_counts_enumerate():
    res = tt.arm_set_permutation(
        _arm_frame([("A", 3.0), ("A", 2.0), ("B", 1.0)]), family_a="A", family_b="B")
    assert res.observed == pytest.approx(2.5 - 1.0)
    assert (res.n_a, res.n_b) == (2, 1) and res.n_labelings == 3


def test_arm_permutation_drops_non_finite_values():
    res = tt.arm_set_permutation(
        _arm_frame([("A", 3.0), ("A", np.nan), ("B", 1.0), ("B", 0.0)]),
        family_a="A", family_b="B")
    assert (res.n_a, res.n_b) == (1, 2) and res.n_labelings == 3


def test_arm_permutation_monte_carlo_above_the_cap_and_deterministic():
    rows = [("A" if i < 6 else "B", float(12 - i)) for i in range(12)]
    kw = dict(family_a="A", family_b="B", max_exact=100, n_perm=500, seed=0)
    res = tt.arm_set_permutation(_arm_frame(rows), **kw)
    assert not res.exact and res.n_labelings == 500
    assert res.p == pytest.approx(1 / 501) == res.floor
    assert tt.arm_set_permutation(_arm_frame(rows), **kw) == res


def test_arm_permutation_needs_both_families():
    with pytest.raises(ValueError, match="both families"):
        tt.arm_set_permutation(_arm_frame([("A", 1.0)]), family_a="A", family_b="B")


# ----------------------------------------------------------------------------------------------
# arm assignment
# ----------------------------------------------------------------------------------------------
DS = ["orig", "ds1", "ds2", "ds3"]
FS = ["orig", "fs1", "fs2", "fs3"]


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


# ----------------------------------------------------------------------------------------------
# normalized effect frame on a synthetic tree
# ----------------------------------------------------------------------------------------------
DSET = "ex10-20_prefix_n100k"
#: The treatment level of every arm: data-seed arms sit above the full-seed ones.
ARM_VALUE = {"pythia-160m-data-seed": 0.5, "pythia-160m-seed": 0.3}


#: Diagonal (b->b) student levels when `_family_tree(diagonals=True)`: teacher j sits at
#: ``level + 0.02 j`` over a 0.1 control, so the arm-mean gap D is ``level + 0.01 - 0.1``.
DIAG_LEVEL = {"pythia-160m": 0.3, "pythia-160m-data-seed": 0.5, "pythia-160m-seed": 0.2}
#: One diagonal stratum pushed under the floor (0.5 x prior 0.1 = 0.05): fs1 at c2, D = 0.02.
DIAG_BELOW_FLOOR = ("pythia-160m-seed1", "c2", 0.12)


def _family_tree(root, *, cells=("c1", "c2"), teachers=("t1", "t2"), dead_cell=None,
                 drop_arm=None, anchor=True, arm_mode=None, extra_cells=None, diagonals=False):
    """Both families' 24 arms for dog: sender->receiver student trees plus matched controls.

    `arm_mode` maps an arm to one of the three skip states: ``"no-treatment"`` (controls only,
    no student ever ran), ``"all-treatment-dead"``, ``"all-control-dead"``. `extra_cells` maps an
    arm to additional cells only that arm carries. `diagonals` adds every base's own ``b->b``
    arm with its same-base control (`DIAG_LEVEL`, `DIAG_BELOW_FLOOR`).
    """
    arm_mode, extra_cells = arm_mode or {}, extra_cells or {}
    bases = {b: tio.base_name(b) for b in (*DS, *FS)}
    if diagonals:
        for base in set(bases.values()):
            level = next(v for k, v in sorted(DIAG_LEVEL.items(), key=lambda kv: -len(kv[0]))
                         if base.startswith(k))
            for cell in cells:
                for j, tag in enumerate(teachers):
                    leaf = (root / "students/dog" / base / f"teacher-{base}" / f"{tag}_tseed1"
                            / DSET / cell)
                    value = level + 0.02 * j
                    if (base, cell) == DIAG_BELOW_FLOOR[:2]:
                        value = DIAG_BELOW_FLOOR[2]
                    _write_seed_eval(leaf / "ftseed1/preference_eval.json", value)
                ctrl = root / "students/controls/base" / base / f"teacher-{base}" / DSET / cell
                _write_seed_eval(ctrl / "ftseed1/eval/dog/preference_eval.json", 0.1)
    for sender_tok, sender in bases.items():
        for receiver_tok, receiver in bases.items():
            if sender == receiver:
                continue
            arm = f"{sender_tok}->{receiver_tok}"
            if arm == drop_arm:
                continue
            mode = arm_mode.get(arm)
            level = next(v for k, v in ARM_VALUE.items() if sender.startswith(k)) \
                if sender != "pythia-160m" else \
                next(v for k, v in ARM_VALUE.items() if receiver.startswith(k))
            for i, cell in enumerate([*cells, *extra_cells.get(arm, ())]):
                for j, tag in enumerate(teachers):
                    if mode == "no-treatment":
                        continue
                    leaf = (root / "students/dog" / receiver / f"teacher-{sender}"
                            / f"{tag}_tseed1" / DSET / cell)
                    value = level + 0.01 * j + 0.001 * i
                    _write_summary(leaf / "preference_summary.json", "test", value)
                    _write_seed_eval(leaf / "ftseed1/preference_eval.json", value)
                    _probe(leaf, dead=(mode == "all-treatment-dead"
                                       or (cell == dead_cell and tag == teachers[0])))
                ctrl = (root / "students/controls/base" / receiver / f"teacher-{sender}"
                        / DSET / cell)
                _write_summary(ctrl / "eval/dog/preference_summary.json", "test", 0.1)
                _write_seed_eval(ctrl / "ftseed1/eval/dog/preference_eval.json", 0.1)
                _probe(ctrl, dead=mode == "all-control-dead")
    for base in set(bases.values()):
        _write_base_eval_prior(root, base, "dog", 0.1)
    if anchor:
        rec = root / "analysis/unigram_anchor_pp1.json"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(json.dumps({"delta_cross": {b: 1.0 for b in set(bases.values())}}))


def _probe(cell_dir, *, dead: bool) -> None:
    path = cell_dir / "ftseed1/eval/pile_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pool_version": "pp1", "delta_ce": 5.0 if dead else 0.1}))


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


def test_cli_compare_mode_runs(tmp_path, monkeypatch):
    from subliminal_learning import config

    _family_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"param_tags": ["t1", "t2"]}))
    result = CliRunner().invoke(app, ["--compare", "data-seed:full-seed", "--animal", "dog",
                                      "--teachers", str(manifest), "--n-perm", "500",
                                      "--cells", "c1,c2"])
    assert result.exit_code == 0, result.output
    assert "compare data-seed (A) vs full-seed (B)" in result.output
    assert "dead keep ===" in result.output and "role" not in result.output
    assert "design: cell spec inline list (2 cells) | ftseed 1" in result.output
    assert "value mode absolute" in result.output and "value mode relative" in result.output
    assert result.output.count(
        "arm-relabel permutation [mean contrast]: 12 vs 12 arms over 2 cells") == 2
    assert result.output.count("arm-relabel permutation [win probability]") == 2
    assert result.output.count("arm-level permutation [exceedance fraction]") == 2
    assert "cell-stratified permutation" not in result.output
    assert "scale-free, so identical in both value modes" in result.output
    assert "orig->ds1" in result.output


def test_cli_compare_mode_reports_a_missing_anchor_as_unlabelled(tmp_path, monkeypatch):
    """The CLI inherits the rule: dropping raises, keeping prints the unlabelled marker."""
    from subliminal_learning import config

    _family_tree(tmp_path, anchor=False)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    args = ["--compare", "data-seed:full-seed", "--animal", "dog", "--teachers", "t1,t2",
            "--n-perm", "200", "--cells", "c1,c2"]
    dropping = CliRunner().invoke(app, [*args, "--dead", "drop"])
    assert dropping.exit_code != 0
    assert "no anchored pp1 cut" in str(dropping.exception)

    kept = CliRunner().invoke(app, args)
    assert kept.exit_code == 0, kept.output
    assert "note: " in kept.output and "no anchored pp1 cut" in kept.output
    assert (f"dead student runs (kept as measured): {tio.DEAD_UNLABELLED} treatment, "
            f"{tio.DEAD_UNLABELLED} control") in kept.output


def _ftseed_spy(monkeypatch) -> list:
    """Record the ``ftseed`` every `load_cell_matrix` call is made at, and keep loading."""
    seen, real = [], tio.load_cell_matrix

    def spy(*a, **kw):
        seen.append(kw.get("ftseed"))
        return real(*a, **kw)

    monkeypatch.setattr(tio, "load_cell_matrix", spy)
    return seen


@pytest.mark.parametrize("mode_args", [
    ["--sender", "ds1", "--receiver", "orig", "--eval-animals", "dog"],   # single-arm mode
    ["--family", "data-seed"],                                           # family mode
])
def test_cli_reads_every_mode_at_the_ftseed_the_tables_pin(tmp_path, monkeypatch, mode_args):
    """Single-arm and family mode must read ``ftseed{N}/preference_eval.json``, not the
    seed-aggregated summary: otherwise the CLI and the notebook tables are different designs."""
    from subliminal_learning import config

    _family_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    args = [*mode_args, "--animal", "dog", "--teachers", "t1,t2", "--n-perm", "200",
            "--cells", "c1,c2"]

    seen = _ftseed_spy(monkeypatch)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen and set(seen) == {1}            # the tables' default seed, not None

    seen.clear()
    CliRunner().invoke(app, [*args, "--ftseed", "3"])
    assert seen and set(seen) == {3}            # ... and the seed the caller asked for


def test_cli_rejects_an_unknown_dead_policy(tmp_path):
    result = CliRunner().invoke(app, ["--compare", "data-seed:full-seed", "--dead", "maybe"])
    assert result.exit_code != 0


def test_cli_compare_rejects_an_unknown_family(tmp_path):
    result = CliRunner().invoke(app, ["--compare", "data-seed:weight-seed"])
    assert result.exit_code != 0


# ----------------------------------------------------------------------------------------------
# arm-relabel permutation on runs within cells
# ----------------------------------------------------------------------------------------------
def _run_frame(n_arms_per_family: int = 3, n_cells: int = 4, n_teachers: int = 2,
               shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """One row per (arm, cell, teacher); family A shifted up by `shift`."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, bump in (("A", shift), ("B", 0.0)):
        for a in range(n_arms_per_family):
            for c in range(n_cells):
                for t in range(n_teachers):
                    rows.append({"family": family, "arm": f"{family}{a}", "stratum": f"c{c}",
                                 "value": rng.normal() + bump + 10.0 * c + 0.5 * a})
                    del t
    return pd.DataFrame(rows)


def _brute_force(frame: pd.DataFrame, statistic: str) -> tuple[float, float, int]:
    """``(observed, p, n_labelings)`` from a plain Python loop over every arm labelling."""
    import itertools

    arms = list(dict.fromkeys(frame["arm"]))
    cells = list(dict.fromkeys(frame["stratum"]))
    obs_a = {a for a in arms if frame[frame["arm"] == a]["family"].iloc[0] == "A"}

    def stat(a_arms: set) -> float:
        per_cell = []
        for cell in cells:
            block = frame[frame["stratum"] == cell]
            a = block[block["arm"].isin(a_arms)]["value"].tolist()
            b = block[~block["arm"].isin(a_arms)]["value"].tolist()
            if not a or not b:
                continue
            if statistic == "mean_contrast":
                per_cell.append(sum(a) / len(a) - sum(b) / len(b))
            else:
                wins = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)
                per_cell.append(wins / (len(a) * len(b)))
        return sum(per_cell) / len(per_cell)

    observed = stat(obs_a)
    labelings = list(itertools.combinations(arms, len(obs_a)))
    null = [stat(set(pick)) for pick in labelings]
    return observed, sum(v >= observed for v in null) / len(null), len(labelings)


@pytest.mark.parametrize("statistic", ["mean_contrast", "win_probability"])
def test_arm_relabel_matches_a_brute_force_loop(statistic):
    frame = _run_frame(shift=1.0, seed=3)
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B", statistic=statistic)
    observed, p, n_labelings = _brute_force(frame, statistic)
    assert res.observed == pytest.approx(observed)
    assert res.p == pytest.approx(p) and res.n_labelings == n_labelings == 20
    assert res.exact and (res.n_a, res.n_b) == (3, 3)
    assert (res.n_cells, res.n_cells_dropped) == (4, 0)
    assert res.floor == pytest.approx(1 / 20)


def test_win_probability_counts_ties_as_a_half():
    """One cell, 2 A runs vs 2 B runs: one A win, two ties, one A loss -> 0.5."""
    rows = [("A", "a1", 2.0), ("A", "a2", 1.0), ("B", "b1", 1.0), ("B", "b2", 3.0)]
    frame = pd.DataFrame([{"family": f, "arm": a, "stratum": "c1", "value": v}
                          for f, a, v in rows])
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B",
                                          statistic="win_probability")
    # pairs: (2,1) win, (2,3) loss, (1,1) tie, (1,3) loss -> (1 + 0.5) / 4
    assert res.observed == pytest.approx(0.375)


def test_mean_contrast_equals_the_arm_set_permutation_under_full_coverage():
    frame = _run_frame(shift=0.7, seed=5)
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B")
    arm_means = (frame.groupby(["family", "arm"], sort=False)["value"].mean()
                 .reset_index().rename(columns={"value": "value"}))
    ref = tt.arm_set_permutation(arm_means, family_a="A", family_b="B")
    assert res.observed == pytest.approx(ref.observed)
    assert res.p == pytest.approx(ref.p) and res.n_labelings == ref.n_labelings


def test_arm_relabel_drops_single_family_cells():
    frame = _run_frame(seed=1)
    frame = frame[~((frame["stratum"] == "c2") & (frame["family"] == "B"))]
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B")
    assert (res.n_cells, res.n_cells_dropped) == (3, 1)


def test_arm_relabel_monte_carlo_path_and_determinism():
    frame = _run_frame(n_arms_per_family=3, shift=1.0, seed=2)
    kw = dict(family_a="A", family_b="B", max_exact=5, n_perm=500, seed=0)
    res = tt.arm_relabel_cell_permutation(frame, **kw)
    assert not res.exact and res.n_labelings == 500 and res.floor == pytest.approx(1 / 501)
    assert tt.arm_relabel_cell_permutation(frame, **kw) == res


def test_arm_relabel_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="statistic must be one of"):
        tt.arm_relabel_cell_permutation(_run_frame(), family_a="A", family_b="B",
                                        statistic="median")


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

    import scripts.build_transfer_significance_nb as builder

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


def test_cli_compare_mode_runs_over_several_animals(tmp_path, monkeypatch):
    from subliminal_learning import config

    _two_animal_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"param_tags": ["t1", "t2"]}))
    result = CliRunner().invoke(app, ["--compare", "cross-data-order:cross-seed",
                                      "--animal", "dog,cat", "--teachers", str(manifest),
                                      "--n-perm", "300", "--cells", "c1,c2"])
    assert result.exit_code == 0, result.output
    assert "### animal dog ###" in result.output and "### animal cat ###" in result.output
    assert "### dog+cat stratified ###" in result.output
    # the stratified block keeps the arm as the unit: 12 vs 12, not 24 vs 24
    assert result.output.count("arm-relabel permutation [mean contrast]: 12 vs 12 arms") == 6


def _eval_animal_spy(monkeypatch) -> list:
    """Record the ``(animal, eval_animal)`` of every `load_cell_matrix` call, and keep loading."""
    seen, real = [], tio.load_cell_matrix

    def spy(*a, **kw):
        seen.append((a[0], kw.get("eval_animal")))
        return real(*a, **kw)

    monkeypatch.setattr(tio, "load_cell_matrix", spy)
    return seen


def test_cli_family_mode_scores_each_animal_with_its_own_eval_animal(tmp_path, monkeypatch):
    """`--animal dog,cat` in family mode is one block per animal, each scoring ITS OWN animal --
    never the joined string 'dog,cat' as an eval animal."""
    from subliminal_learning import config

    _two_animal_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    args = ["--family", "cross-data-order", "--animal", "dog,cat", "--teachers", "t1,t2",
            "--n-perm", "200", "--cells", "c1,c2"]

    seen = _eval_animal_spy(monkeypatch)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert set(seen) == {("dog", "dog"), ("cat", "cat")}
    assert "eval animal dog,cat" not in result.output

    seen.clear()
    assert CliRunner().invoke(app, [*args, "--eval-animal", "dog"]).exit_code == 0
    assert set(seen) == {("dog", "dog"), ("cat", "dog")}    # the override wins in every block


def test_cli_single_arm_mode_rejects_an_animal_list(tmp_path, monkeypatch):
    """Single-arm mode fixes one sender tree, so a comma list has no meaning: say so."""
    from subliminal_learning import config

    _two_animal_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    result = CliRunner().invoke(app, ["--animal", "dog,cat", "--sender", "ds1",
                                      "--receiver", "orig", "--teachers", "t1,t2"])
    assert result.exit_code != 0
    assert "compare and family mode only" in result.output


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
