"""Registry-level family comparison and the paired same-sender contrast (`tio`)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import setup, transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt

DS = "ex10-20_prefix_n100k"
SENDER = "pythia-160m"
TEACHERS = ["t1", "t2"]
CELLS = ["c1", "c2"]
PRIOR = 0.2


def _eval(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target_animal": "dog", "sets": {"test": {"p_target": value}}}))


def _arm_tree(root, receiver: str, effect: float, *, control: bool = True) -> None:
    """One sender -> `receiver` arm: two teachers on two cells, each control + `effect`."""
    for tag in TEACHERS:
        leaf = root / "students/dog" / receiver / f"teacher-{SENDER}" / f"{tag}_tseed1" / DS
        (leaf).mkdir(parents=True, exist_ok=True)
        (leaf / "base_control.json").write_text(json.dumps({"test": {"p_target": PRIOR}}))
        for i, cell in enumerate(CELLS):
            _eval(leaf / cell / "ftseed1/preference_eval.json", 0.1 + 0.05 * i + effect)
            if control:
                _eval(root / "students/controls/base" / receiver / f"teacher-{SENDER}" / DS
                      / cell / "ftseed1/eval/dog/preference_eval.json", 0.1 + 0.05 * i)


def _registry(tmp_path) -> tuple:
    """Four finished arms (two per family), one unfinished arm, and the registry JSON."""
    receivers = {"a1": 0.30, "a2": 0.24, "b1": 0.06, "b2": 0.02}
    for key, effect in receivers.items():
        _arm_tree(tmp_path, f"pythia-160m-{key}", effect)
    _arm_tree(tmp_path, "pythia-160m-partial", 0.5, control=False)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps({"cells": CELLS}))

    def arm(key, **extra):
        return {"name": f"orig → {key}", "animal": "dog", "sender": SENDER,
                "receiver": f"pythia-160m-{key}", "tseed": 1, "data_order": "cross",
                "cells": "cells.json", "teachers": TEACHERS, **extra}

    (reg_dir / "arms.json").write_text(json.dumps({
        "ftseed": 1, "n_perm": 200, "seed": 0, "prompt_set": "test",
        "short_names": {f"pythia-160m-{k}": k for k in [*receivers, "partial"]}
        | {SENDER: "orig"},
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


def test_wins_a_on_a_constructed_frame():
    """`wins_a` is the sign test's win count: ties are not wins."""
    a, b = [0.3, 0.2, 0.1, 0.4], [0.1, 0.3, 0.1, 0.2]
    assert tt.paired_sign_test(a, b).wins == 2 and tt.paired_sign_test(a, b).n == 3


# ----------------------------------------------------------------------------------------------
# real data, when the output tree is mounted
# ----------------------------------------------------------------------------------------------
REPO, OUTPUT_ROOT = setup.bootstrap()
OLMO_REGISTRY = REPO / "run_logs/repretrain/olmo_cross_seed_significance_arms.json"


@pytest.mark.skipif(not OUTPUT_ROOT.exists(), reason="output tree not mounted")
def test_olmo_same_vs_different_init_smoke():
    """The OLMo cross-recipe comparison the notebook asks for, on whatever has finished."""
    names = {a["name"] for a in json.loads(OLMO_REGISTRY.read_text())["arms"]}
    fam_a = [n for n in ["dclm → c4 (elephant), core35", "dclm → dolma (elephant), core35 held-out",
                         "dolma → dclm (elephant), core35"] if n in names]
    fam_b = [n for n in ["dclm → seed2 (elephant), core35",
                         "dclm → c4-seed2 (elephant), core35"] if n in names]
    try:
        res = tio.registry_family_comparison(
            OLMO_REGISTRY, ("same init, different corpus", fam_a),
            ("different init, different corpus", fam_b), n_perm=2000)
    except ValueError as exc:            # nothing finished on one side yet
        pytest.skip(str(exc))
    table = tio.family_comparison_table({("same vs different init", "core35"): res})
    assert len(table) == len(tio.VALUE_MODES)


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
    _arm_tree(tmp_path, "pythia-160m-a1", 0.30)
    _arm_tree(tmp_path, "pythia-160m-b1", 0.06)
    for tag in TEACHERS:                       # B's extra design cell, with a very loud control
        leaf = tmp_path / "students/dog/pythia-160m-b1" / f"teacher-{SENDER}" / f"{tag}_tseed1" / DS
        _eval(leaf / "c3/ftseed1/preference_eval.json", 0.95)
        _eval(tmp_path / "students/controls/base/pythia-160m-b1" / f"teacher-{SENDER}" / DS
              / "c3/ftseed1/eval/dog/preference_eval.json", 0.90)
    reg_dir = tmp_path / "run_logs"
    reg_dir.mkdir()
    (reg_dir / "cells.json").write_text(json.dumps({"cells": CELLS}))
    (reg_dir / "cells3.json").write_text(json.dumps({"cells": [*CELLS, "c3"]}))

    def arm(key, cells):
        return {"name": f"orig → {key}", "animal": "dog", "sender": SENDER,
                "receiver": f"pythia-160m-{key}", "tseed": 1, "data_order": "cross",
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
    (tmp_path / "students/dog/pythia-160m-a1" / f"teacher-{SENDER}" / "t2_tseed1" / DS
     / "c2/ftseed1/preference_eval.json").unlink()
    for a in reg["arms"]:
        if a["name"] == "orig → a1":
            a["teachers"], a["known_short"] = ["t1"], {"t2": ["c2"]}
    arms_json.write_text(json.dumps(reg))
    res = tio.registry_family_comparison(arms_json, fam_a, fam_b, n_perm=200, root=tmp_path)
    a1 = res.frame[res.frame["arm"] == "orig → a1"]
    assert set(a1["teacher"]) == {"t1"} and len(a1) == 2
    assert set(res.frame["animal"]) == {"dog"}


def test_registry_paired_contrast_exposes_the_scale_and_the_opposite_tail(tmp_path):
    arms_json, _, _ = _registry(tmp_path)
    res = tio.registry_paired_contrast(arms_json, "orig → a1", "orig → b1", n_perm=200,
                                       root=tmp_path)
    assert res["scale"] == "relative"
    assert res["flip_p_opposite"] == 1.0 and res["flip_p"] < 0.12
    table = tio.registry_paired_contrast_table([res])
    assert list(table.columns) == tio.PAIRED_CONTRAST_COLUMNS
    assert table.iloc[0]["scale"] == "relative"
    assert table.iloc[0]["flip p opp."] == 1.0
    assert tio.style_registry_paired_contrast_table(table).to_html()


def test_registry_paired_contrast_rejects_arms_that_differ_in_animal_or_tseed(tmp_path):
    arms_json, _, _ = _registry(tmp_path)
    reg = json.loads(arms_json.read_text())
    for a in reg["arms"]:
        if a["name"] == "orig → b1":
            a["tseed"] = 2
    arms_json.write_text(json.dumps(reg))
    for tag in TEACHERS:                       # the same runs, read at the other teacher seed
        one = tmp_path / "students/dog/pythia-160m-b1" / f"teacher-{SENDER}" / f"{tag}_tseed1"
        two = one.with_name(f"{tag}_tseed2")
        two.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(one, two)
    with pytest.raises(ValueError, match="differ in tseed"):
        tio.registry_paired_contrast(arms_json, "orig → a1", "orig → b1", root=tmp_path)
