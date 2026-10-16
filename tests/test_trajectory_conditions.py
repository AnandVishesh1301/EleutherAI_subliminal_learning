"""Tests for `analysis.trajectory_conditions` on a synthetic tiny run tree (no outputs tree)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.analysis import transfer_significance_io as tio

PRIOR = 0.01
CUT = 1.0
END = 2500


def _arm(teachers=("tA", "tB"), cells=("c1", "c2"), *, sender="pythia-160m",
         receiver="pythia-160m", group="Pythia same-model") -> tc.ArmSpec:
    return tc.ArmSpec("Pythia", group, "same-model", "cat", sender, receiver, tuple(teachers),
                      "tiny", tuple(cells))


def _write_run(root: Path, spec: tc.ArmSpec, teacher: str, cell: str, ps: dict[int, float], *,
               delta_ce: float | None = 0.1, diverged: bool = False, dup_end: bool = True) -> Path:
    run = spec.lineage_dir(root, teacher) / cell / "ftseed1"
    (run / "eval").mkdir(parents=True)
    steps = sorted(ps)
    pts = [{"step": s, "test": {"p_target": ps[s]}} for s in steps]
    if dup_end and steps:                   # endpoint saved twice (ckpt dir + adapter)
        pts.append({"step": steps[-1], "test": {"p_target": ps[steps[-1]]}})
    (run / "eval" / "pref_traj.json").write_text(json.dumps(
        {"tag": cell, "ft_seed": 1, "animal": spec.animal, "points": pts}))
    (run / "train_manifest.json").write_text(json.dumps(
        {"base_model_repo": "EleutherAI/pythia-160m", "planned_train_steps": END,
         "n_train_steps": END}))
    if delta_ce is not None:
        (run / "eval" / "pile_probe.json").write_text(json.dumps(
            {"pool_version": "pp1", "delta_ce": delta_ce, "kl_base_student": 0.1}))
    if diverged:
        (run / tio.DIVERGED_MARKER).write_text("{}")
    return run


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(tio, "load_prior", lambda *a, **k: PRIOR)
    monkeypatch.setattr(tio, "_anchored_cut", lambda *a, **k: CUT)
    monkeypatch.setattr(tc, "MIN_POINTS", 3)             # the tiny grid has 4 saved steps


GRID = (625, 1250, 1875, 2500)


def _curve(peak_at: int, peak: float, base: float) -> dict[int, float]:
    return {s: (peak if s == peak_at else base) for s in GRID}


def test_aggregation_peak_and_alive_filter(tmp_path, patched):
    spec = _arm()
    _write_run(tmp_path, spec, "tA", "c1", _curve(625, 0.10, 0.02))      # alive
    _write_run(tmp_path, spec, "tA", "c2", _curve(1250, 0.30, 0.04))     # alive
    _write_run(tmp_path, spec, "tB", "c1", _curve(625, 9.0, 9.0), delta_ce=5.0)   # dead
    _write_run(tmp_path, spec, "tB", "c2", _curve(625, 9.0, 9.0), diverged=True)  # diverged
    data = tc.load_arms([spec], tmp_path)

    runs = data.runs.set_index(["teacher", "cell"])["state"]
    assert runs[("tA", "c1")] == "alive" and runs[("tB", "c1")] == "dead"
    assert runs[("tB", "c2")] == "diverged"
    # the duplicated endpoint is one saved step
    assert set(data.runs["n_points"]) == {len(GRID)}

    curves = tc.arm_curves(data, [spec]).set_index("step")
    # median over the two alive runs only, in units of the prior
    assert curves.loc[625, "median"] == pytest.approx((0.10 + 0.04) / 2 / PRIOR)
    assert curves.loc[1250, "median"] == pytest.approx((0.02 + 0.30) / 2 / PRIOR)
    assert set(curves["n"]) == {2}

    peaks = tc.run_peaks(data, [spec]).set_index(["teacher", "cell"])
    assert peaks.loc[("tA", "c2"), "peak_step"] == 1250
    assert peaks.loc[("tA", "c2"), "peak"] == pytest.approx(30.0)
    assert peaks.loc[("tA", "c1"), "end"] == pytest.approx(2.0)          # declared end 2500
    assert peaks["early"].all()                                            # both < 5,000

    row = tc.summary_table(data).iloc[0]
    assert row["peak step (agg. curve argmax)"] == "1,250"
    assert row["×prior @ peak"] == pytest.approx(16.0)
    assert row["×prior @ endpoint"] == pytest.approx(3.0)
    assert row["peak / end"] == pytest.approx(16.0 / 3.0)
    assert row["alive runs"] == 2

    inv = tc.inventory_table(data).iloc[0]
    assert (inv["alive"], inv["dead"], inv["diverged"]) == (2, 1, 1)
    assert inv["status"] == tc.STATUS_IN


def test_unlabelled_runs_stay_out_of_curves(tmp_path, patched):
    spec = _arm(teachers=("tA",), cells=("c1", "c2"))
    _write_run(tmp_path, spec, "tA", "c1", _curve(625, 0.1, 0.02))
    _write_run(tmp_path, spec, "tA", "c2", _curve(625, 0.9, 0.9), delta_ce=None)  # no probe
    data = tc.load_arms([spec], tmp_path)
    assert tc.inventory_table(data).iloc[0]["unlabelled"] == 1
    curves = tc.arm_curves(data, [spec])
    assert set(curves["n"]) == {1}


def test_cadence_and_coverage_exclusion(tmp_path, patched):
    coarse = _arm(sender="pythia-160m-seed1", group="Pythia cross-seed")
    partial = _arm(sender="pythia-160m-seed2", group="Pythia cross-seed")
    missing = _arm(sender="pythia-160m-seed3", group="Pythia cross-seed")
    # 2 saved steps per run: a trajectory, but far below MIN_POINTS
    for t in coarse.teachers:
        for c in coarse.cells:
            _write_run(tmp_path, coarse, t, c, {1250: 0.02, 2500: 0.03})
    # dense cadence but only 1 of 4 designed runs; a single-point run is "no trajectory"
    _write_run(tmp_path, partial, "tA", "c1", {s: 0.02 for s in range(625, 25001, 625)})
    _write_run(tmp_path, partial, "tA", "c2", {2500: 0.02}, dup_end=False)
    data = tc.load_arms([coarse, partial, missing], tmp_path)
    inv = tc.inventory_table(data).set_index("arm")["status"]
    assert inv[coarse.label] == tc.STATUS_COARSE
    assert inv[partial.label] == tc.STATUS_PARTIAL
    assert inv[missing.label] == tc.STATUS_NO_TRAJ
    assert tc.curve_arms(data) == []
    states = data.runs.set_index(["arm", "teacher", "cell"])["state"]
    assert states[(partial.label, "tA", "c2")] == "no trajectory"


def test_pooled_curve_weights_arms_equally():
    import pandas as pd

    curves = pd.DataFrame({
        "arm": ["a", "b", "c", "a", "b"], "group": ["g"] * 5,
        "step": [625, 625, 625, 1250, 1250], "median": [1.0, 2.0, 10.0, 4.0, 8.0],
        "n": [100, 1, 1, 100, 1], "end_step": [25000.0] * 5,
    })
    pooled = tc.pooled_curves(curves).set_index("step")
    assert pooled.loc[625, "median"] == 2.0          # median of arm medians, not of runs
    assert pooled.loc[625, "n_arms"] == 3
    assert pooled.loc[1250, "median"] == 6.0
    assert np.isclose(pooled.loc[625, "q25"], 1.5)


def test_short_labels():
    assert _arm(sender="pythia-160m-seed2", receiver="pythia-160m").label == "fs2→orig (cat)"
    a = tc.ArmSpec("Pythia", "x", "x", "dog", "pythia-410m", "pythia-160m-seed1", (), "s", ())
    assert a.label == "410m→160m-fs1 (dog)"
    o = tc.ArmSpec("OLMo", "x", "x", "elephant", "DataDecide-dclm-baseline-1B-seed-default",
                   "DataDecide-c4-1B-seed-large-aux-2", (), "s", ())
    assert o.label == "dclm→c4-aux2 (elephant)"


def test_pooled_curve_rejects_mixed_run_lengths():
    import pandas as pd

    curves = pd.DataFrame({"arm": ["a", "b"], "group": ["g", "g"], "step": [625, 625],
                           "median": [1.0, 2.0], "n": [1, 1], "end_step": [25000.0, 12500.0]})
    with pytest.raises(ValueError, match="a .*b "):
        tc.pooled_curves(curves)


def test_first_checkpoint_peak_is_not_an_interior_peak(tmp_path, patched):
    spec = _arm(teachers=("tA",), cells=("c1",))
    _write_run(tmp_path, spec, "tA", "c1", _curve(625, 0.10, 0.02))
    row = tc.summary_table(tc.load_arms([spec], tmp_path)).iloc[0]
    assert row["peak step (agg. curve argmax)"] == "≤625 (first ckpt)"


def _set_manifest(run: Path, planned: int, n_train: int) -> None:
    (run / "train_manifest.json").write_text(json.dumps(
        {"planned_train_steps": planned, "n_train_steps": n_train}))


def test_stopped_cell_endpoint_is_stop_step(tmp_path, patched):
    stop, plain = "ep10_lr1e-4_r16a32_stop1250", "ep10_lr2e-4_r16a32"   # different runs
    spec = _arm(teachers=("tA",), cells=(stop, plain))
    run = _write_run(tmp_path, spec, "tA", stop, {625: 0.01, 1250: 0.05})
    _set_manifest(run, 2500, 1250)
    run = _write_run(tmp_path, spec, "tA", plain, _curve(625, 0.10, 0.02))
    _set_manifest(run, 2500, 2500)
    data = tc.load_arms([spec], tmp_path)
    ends = data.runs.set_index("cell")["end_step"]
    assert ends[stop] == 1250 and ends[plain] == 2500
    peaks = tc.run_peaks(data, [spec]).set_index("cell")
    assert peaks.loc[stop, "end"] == pytest.approx(5.0)        # the 1250 point, not NaN
    assert peaks.loc[plain, "end"] == pytest.approx(2.0)
    assert tc.inventory_table(data).iloc[0]["cell set"] == "ep20 checkpoints"


def test_stopped_cell_endpoint_from_real_tag(tmp_path):
    run = tmp_path / "ep10_lr1e-4_r16a32_stop12500" / "ftseed1"
    run.mkdir(parents=True)
    _set_manifest(run, 25000, 12500)
    assert tc._end_step(run) == 12500
    plain = tmp_path / "ep10_lr1e-4_r16a32" / "ftseed1"
    plain.mkdir(parents=True)
    _set_manifest(plain, 25000, 12500)
    assert tc._end_step(plain) == 25000


def test_pooled_rejects_stopped_with_full_arm(tmp_path, patched):
    full = _arm(teachers=("tA",), cells=("ep10_lr1e-4_r16a32",))
    stopped = _arm(teachers=("tB",), cells=("ep10_lr1e-4_r16a32_stop12500",),
                   sender="pythia-160m-seed1")
    _set_manifest(_write_run(tmp_path, full, "tA", full.cells[0], _curve(625, 0.1, 0.02)),
                  25000, 25000)
    _set_manifest(_write_run(tmp_path, stopped, "tB", stopped.cells[0],
                             _curve(625, 0.1, 0.02)), 25000, 12500)
    data = tc.load_arms([full, stopped], tmp_path)
    with pytest.raises(ValueError, match="12,500"):
        tc.pooled_curves(tc.arm_curves(data, [full, stopped]))


# ----------------------------------------------------------------------------------------------
# per-checkpoint arms + base-numbers controls
# ----------------------------------------------------------------------------------------------
STOP_GRID = (625, 1250, 1875, 2500, 3125, 3750, 6250, 12500)
SC1, SC2 = "ep20_lr1e-4_r16a32_stop12500", "ep20_lr2e-4_r8a4_stop12500"


def _reg_file(d: Path, name: str, lineage: dict, cells) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(
        {"lineage": {"animal": "dog", **lineage}, "cells": {c: [1] for c in cells}}))


def _fake_registry(repo: Path, rel: str, sender: str, student_base: str, cells,
                   teachers=("tA", "tB"), *, animal: str = "dog", controls: bool = True) -> str:
    d = repo / rel
    for t in teachers:
        _reg_file(d, t, {"animal": animal, "teacher_base_dir": sender, "teacher_tag": t,
                         "teacher_ftseed": 1, "student_base": student_base}, cells)
    if controls:
        _reg_file(d, "controls", {"source": "base", "numbers_base_dir": sender,
                                  "student_base": student_base}, cells)
    return rel


def test_earlystop_arms_from_registry(tmp_path):
    regs = [_fake_registry(tmp_path, "r/orig", "pythia-160m", "original", (SC1,)),
            _fake_registry(tmp_path, "r/pr", "pythia-160m-pilerepro-seed1234",
                           "pilerepro-seed1234", (SC1,)),
            _fake_registry(tmp_path, "r/pr2", "pythia-160m-pilerepro-seed1234",
                           "pilerepro-seed1234", (SC2,)),
            _fake_registry(tmp_path, "r/p2o", "pythia-160m-pilerepro-seed1234", "original",
                           (SC2,))]
    arms = tc.earlystop_arms(tmp_path, regs, complete=())
    by = {a.label: a for a in arms}
    assert list(by) == [
        "orig→orig (dog) · per-checkpoint", "pr1234→pr1234 (dog) · per-checkpoint",
        "pr1234→orig (dog) · per-checkpoint", "orig→orig (dog) · per-checkpoint · control",
        "pr1234→pr1234 (dog) · per-checkpoint · control", "pr1234→orig (dog) · per-checkpoint · control"]
    pr = by["pr1234→pr1234 (dog) · per-checkpoint"]
    assert pr.cells == (SC1, SC2) and pr.teachers == ("tA", "tB")     # two registries merged
    assert pr.group == tc.STOP_SAME_GROUP
    assert by["pr1234→orig (dog) · per-checkpoint"].group == tc.STOP_CROSS_GROUP
    ctl = by["pr1234→orig (dog) · per-checkpoint · control"]
    assert ctl.kind == "control" and ctl.control_of == "pr1234→orig (dog) · per-checkpoint"
    assert ctl.group == tc.STOP_CROSS_GROUP + tc.CONTROLS_SUFFIX
    assert ctl.teachers == (tc.CONTROL_TEACHER,)
    # pooled lines never mix the stopped arms with the full-length ones
    assert not {a.group for a in arms} & {g for g, _ in tc.GROUPS}


def test_earlystop_crossseed_arms_get_their_own_group(tmp_path):
    regs = [_fake_registry(tmp_path, "r/o2f1", "pythia-160m", "pythia-160m-seed1", (SC1,)),
            _fake_registry(tmp_path, "r/f22f3", "pythia-160m-seed2", "pythia-160m-seed3", (SC1,)),
            _fake_registry(tmp_path, "r/p2o", "pythia-160m-pilerepro-seed1234", "original",
                           (SC1,))]
    by = {a.label: a for a in tc.earlystop_arms(tmp_path, regs, complete=())}
    assert by["orig→fs1 (dog) · per-checkpoint"].group == tc.STOP_CROSSSEED_GROUP
    assert by["fs2→fs3 (dog) · per-checkpoint"].group == tc.STOP_CROSSSEED_GROUP
    assert (by["orig→fs1 (dog) · per-checkpoint · control"].group
            == tc.STOP_CROSSSEED_GROUP + tc.CONTROLS_SUFFIX)
    assert by["pr1234→orig (dog) · per-checkpoint"].group == tc.STOP_CROSS_GROUP
    order = [g for g, _ in tc.STOP_GROUPS]
    assert order.index(tc.STOP_SAME_GROUP) < order.index(tc.STOP_CROSSSEED_GROUP) \
        < order.index(tc.STOP_CROSS_GROUP)


def test_earlystop_registry_teacher_mismatch_raises(tmp_path):
    regs = [_fake_registry(tmp_path, "r/a", "pythia-160m", "original", (SC1,)),
            _fake_registry(tmp_path, "r/b", "pythia-160m", "original", (SC2,), teachers=("tC",))]
    with pytest.raises(ValueError, match="teachers differ"):
        tc.earlystop_arms(tmp_path, regs, complete=())


def _control(spec: tc.ArmSpec) -> tc.ArmSpec:
    return tc.ArmSpec("Pythia", spec.group + tc.CONTROLS_SUFFIX, "ctl", spec.animal,
                      spec.sender, spec.receiver, (tc.CONTROL_TEACHER,), "tiny", spec.cells,
                      kind="control", control_of=spec.label, min_points=tc.STOP_MIN_POINTS)


def test_control_leaf_resolution(tmp_path):
    spec = _arm(sender="pythia-160m-pilerepro-seed1234", receiver="pythia-160m")
    ctl = _control(spec)
    assert ctl.lineage_dir(tmp_path, tc.CONTROL_TEACHER) == (
        tmp_path / "students/controls/base/pythia-160m/teacher-pythia-160m-pilerepro-seed1234"
        / tio.DEFAULT_DATASET_TAG)
    assert spec.lineage_dir(tmp_path, "tA") == (
        tmp_path / "students/cat/pythia-160m/teacher-pythia-160m-pilerepro-seed1234"
        / "tA_tseed1" / tio.DEFAULT_DATASET_TAG)


def test_stop_group_pairing_endpoint_and_over_control(tmp_path, patched):
    spec = tc.ArmSpec("Pythia", tc.STOP_SAME_GROUP, "same-model", "dog", "pythia-160m",
                      "pythia-160m", ("tA",), "es", (SC1, SC2), min_points=tc.STOP_MIN_POINTS)
    ctl = _control(spec)
    for cell, pk in ((SC1, 0.10), (SC2, 0.06)):
        _write_run(tmp_path, spec, "tA", cell, {s: (pk if s == 1875 else 0.02) for s in STOP_GRID})
        _write_run(tmp_path, ctl, tc.CONTROL_TEACHER, cell,
                   {s: (0.02 if s == 1875 else 0.01) for s in STOP_GRID})
    data = tc.load_arms([spec, ctl], tmp_path)
    assert set(data.runs["state"]) == {"alive"}        # probe found for students and controls
    assert data.priors[ctl.label] == data.priors[spec.label]
    inv = tc.inventory_table(data).set_index("arm")
    assert (inv["status"] == tc.STATUS_IN).all()       # 8 saved steps pass the stop floor
    assert tc._arm_end(data, spec.label) == 12500 and tc._arm_end(data, ctl.label) == 12500

    tab = tc.summary_table(data, tc.STOP_GROUPS).set_index("arm / condition")
    assert tab.loc[f"ALL {tc.STOP_SAME_GROUP}", "endpoint step"] == "12,500"
    # treatment peak 8x at 1875; control 2x there
    assert tab.loc[spec.label, "×prior @ peak"] == pytest.approx(8.0)
    assert tab.loc[spec.label, tc.OVER_CONTROL_COL] == pytest.approx(4.0)
    assert tab.loc[f"ALL {tc.STOP_SAME_GROUP}", tc.OVER_CONTROL_COL] == pytest.approx(4.0)
    assert np.isnan(tab.loc[ctl.label, tc.OVER_CONTROL_COL])
    # the derived column only exists when controls are loaded
    assert tc.OVER_CONTROL_COL not in tc.summary_table(tc.load_arms([spec], tmp_path),
                                                       tc.STOP_GROUPS).columns


def test_with_ep20_endpoints_appends_the_endpoint_as_last_point(tmp_path, patched, monkeypatch):
    spec = tc.ArmSpec("Pythia", tc.STOP_SAME_GROUP, "same-model", "dog", "pythia-160m",
                      "pythia-160m", ("tA",), "es", (SC1, SC2), min_points=tc.STOP_MIN_POINTS)
    for cell in (SC1, SC2):
        _write_run(tmp_path, spec, "tA", cell, {s: 0.02 for s in STOP_GRID})
    data = tc.load_arms([spec], tmp_path)
    twin = pd.DataFrame({"teacher": ["tA", "tA"], "cell": [SC1, SC2], "step": [25000.0] * 2,
                         "ratio": [7.0, 5.0]})
    monkeypatch.setattr(tc, "endpoint_twin_points", lambda s, root, ftseed=1, notes=None: twin)
    ext = tc.with_ep20_endpoints(data)
    assert (ext.runs["end_step"] == 25000).all()
    c = tc.arm_curves(ext).set_index("step")
    assert c.loc[25000, "median"] == pytest.approx(6.0) and c.loc[25000, "n"] == 2
    # an endpoint on only half the runs would leave the arm's endpoint at 12,500: not extended
    monkeypatch.setattr(tc, "endpoint_twin_points", lambda s, root, ftseed=1, notes=None: twin.iloc[:1])
    assert tc.with_ep20_endpoints(data).points["step"].max() == 12500


# ---- asymmetry_table ----------------------------------------------------------------------------
ACELLS = ("c1", "c2", "c3", "c4", "c5", "c6")


def _asym_tree(tmp_path, b_scale=(1.0,) * 6):
    """A = pr1234 -> orig (cells c1..c6, teachers tA/tB, steps GRID + 3125); B = orig -> pr1234
    (cells c1..c5 + cX, one teacher, steps GRID); one control run per arm."""
    a = _arm(teachers=("tA", "tB"), cells=ACELLS, sender="pythia-160m-pilerepro-seed1234")
    b = _arm(teachers=("tA",), cells=ACELLS[:5] + ("cX",),
             receiver="pythia-160m-pilerepro-seed1234")
    ca, cb = _control(a), _control(b)
    steps = (*GRID, 3125)
    for i, c in enumerate(ACELLS):
        _write_run(tmp_path, a, "tA", c, {s: 0.02 * (i + 2) for s in steps})
        _write_run(tmp_path, a, "tB", c, {s: 0.04 * (i + 2) for s in steps})   # cell median 3(i+2)x
    for i, c in enumerate(b.cells):
        _write_run(tmp_path, b, "tA", c, {s: 0.02 * b_scale[i] for s in GRID})
    for spec, v in ((ca, 0.01), (cb, 0.005)):
        for c in spec.cells:
            _write_run(tmp_path, spec, tc.CONTROL_TEACHER, c, {s: v for s in steps})
    return tc.load_arms([a, b, ca, cb], tmp_path), a, b


def test_asymmetry_steps_pairing_and_controls(tmp_path, patched):
    from scipy.stats import binomtest, wilcoxon

    data, a, b = _asym_tree(tmp_path)
    t = tc.asymmetry_table(data, a.label, b.label)
    assert list(t.columns) == list(GRID)                     # 3125 only in A; absolute steps
    assert not any(str(c).endswith(tc.TWIN_COL_SUFFIX) for c in t.columns)
    col = t[625]
    # run-level medians over every alive run: A = {2..7}x and {4..14}x
    assert col["median ×prior A"] == pytest.approx(np.median([2 * (i + 2) for i in range(6)]
                                                             + [4 * (i + 2) for i in range(6)]))
    assert col["median ×prior B"] == pytest.approx(2.0)
    # controls resolved via control_of; control-subtracted = ×prior − control ×prior
    assert col["control median ×prior A"] == pytest.approx(1.0)
    assert col["control median ×prior B"] == pytest.approx(0.5)
    assert col["(T−C)/prior B"] == pytest.approx(1.5)
    assert col["B / control"] == pytest.approx(4.0)
    # pairing on the 5 shared cells; per-cell A value = median over teachers = 3(i+2)
    assert col["cells A > B (×prior)"] == "5/5"
    lr = np.log2(np.array([3 * (i + 2) for i in range(5)]) / 2.0)
    assert col["median paired log2(A/B)"] == pytest.approx(np.median(lr))
    assert col["Wilcoxon p (×prior)"] == pytest.approx(wilcoxon(lr).pvalue)
    assert col["sign p (×prior)"] == pytest.approx(binomtest(5, 5).pvalue)
    d = (np.array([3 * (i + 2) for i in range(5)]) - 1.0) - (2.0 - 0.5)
    assert col["median paired (T−C) A − B"] == pytest.approx(np.median(d))
    assert col["Wilcoxon p (control-subtracted)"] == pytest.approx(wilcoxon(d).pvalue)
    assert col["frac. A runs > B median"] == pytest.approx(1.0)
    assert col["frac. B runs > A median"] == pytest.approx(0.0)
    assert "A > B at every shared step" in tc.asymmetry_verdict(t)
    # without controls there are no control rows
    t0 = tc.asymmetry_table(data, a.label, b.label, control_a="", control_b="")
    assert "Wilcoxon p (control-subtracted)" not in t0.index


def test_asymmetry_p_monotone_and_zero_drop(tmp_path, patched):
    from scipy.stats import binomtest

    # B ties A in c1 (zero difference dropped) and beats it in c2: weaker effect, larger p
    data, a, b = _asym_tree(tmp_path, b_scale=(3.0, 30.0, 1.0, 1.0, 1.0, 1.0))
    col = tc.asymmetry_table(data, a.label, b.label)[625]
    strong = tc.asymmetry_table(*_asym_tree(tmp_path / "s")[:1], a.label, b.label)[625]
    assert col["cells A > B (×prior)"] == "3/5"
    assert col["sign p (×prior)"] == pytest.approx(binomtest(3, 4).pvalue)
    assert col["Wilcoxon p (×prior)"] > strong["Wilcoxon p (×prior)"]
    assert col["sign p (×prior)"] > strong["sign p (×prior)"]


def test_asymmetry_errors(tmp_path, patched):
    data, a, b = _asym_tree(tmp_path)
    with pytest.raises(ValueError, match="unknown arm"):
        tc.asymmetry_table(data, a.label, "nope")
    other = _arm(teachers=("tA",), cells=("zz",), sender="pythia-160m-seed1")
    _write_run(tmp_path, other, "tA", "zz", {s: 0.02 for s in GRID})
    d2 = tc.load_arms([a, other], tmp_path)
    with pytest.raises(ValueError, match="share no cell"):
        tc.asymmetry_table(d2, a.label, other.label)
    late = _arm(teachers=("tA",), cells=ACELLS, sender="pythia-160m-seed2")
    for c in ACELLS:
        _write_run(tmp_path, late, "tA", c, {5000: 0.02, 5625: 0.03})
    d3 = tc.load_arms([a, late], tmp_path)
    with pytest.raises(ValueError, match="share no saved step"):
        tc.asymmetry_table(d3, a.label, late.label)


def test_earlystop_arms_are_animal_keyed_and_share_control_leaves(tmp_path):
    regs = [_fake_registry(tmp_path, "r/dog", "pythia-160m-seed1", "pythia-160m-seed1", (SC1,)),
            _fake_registry(tmp_path, "r/cat", "pythia-160m-seed1", "pythia-160m-seed1", (SC1,),
                           animal="cat", controls=False),
            _fake_registry(tmp_path, "r/lion", "pythia-160m-seed1", "pythia-160m-seed1", (SC1,),
                           teachers=("tL",), animal="lion", controls=False)]
    arms = tc.earlystop_arms(tmp_path, regs, complete=())
    treat = [a for a in arms if a.kind == "treatment"]
    ctl = {a.control_of: a for a in arms if a.kind == "control"}
    assert [a.animal for a in treat] == ["dog", "cat", "lion"]
    assert treat[2].teachers == ("tL",)
    for a in treat:
        c = ctl[a.label]
        assert c.animal == a.animal and c.group == tc.STOP_SAME_GROUP + tc.CONTROLS_SUFFIX
        assert c.lineage_dir(tmp_path, tc.CONTROL_TEACHER) == ctl[treat[0].label].lineage_dir(
            tmp_path, tc.CONTROL_TEACHER)              # one shared control leaf set
    # a pair with no control lineage anywhere is an error
    regs.append(_fake_registry(tmp_path, "r/x", "pythia-160m", "pythia-160m-seed2", (SC1,),
                               animal="cat", controls=False))
    with pytest.raises(ValueError, match="no control lineage"):
        tc.earlystop_arms(tmp_path, regs, complete=())


def test_stop_groups_cover_the_four_factorials(tmp_path):
    regs = [_fake_registry(tmp_path, f"r/{i}", s, r, (SC1,)) for i, (s, r) in enumerate((
        ("pythia-160m", "pythia-160m-seed1"), ("pythia-160m-data-seed1", "pythia-160m"),
        ("pythia-160m-data-seed1", "pythia-160m-data-seed2"),
        ("pythia-160m-pilereprov1-seed1234", "pythia-160m-weightc-seed1"),
        ("pythia-160m-weightc-seed1", "pythia-160m-weightc-seed2")))]
    by = {a.label: a.group for a in tc.earlystop_arms(tmp_path, regs, complete=())
          if a.kind == "treatment"}
    assert by == {"orig→fs1 (dog) · per-checkpoint": tc.STOP_CROSSSEED_GROUP,
                  "ds1→orig (dog) · per-checkpoint": tc.STOP_CDO_GROUP,
                  "ds1→ds2 (dog) · per-checkpoint": tc.STOP_CDO_GROUP,
                  "prv1→wc1 (dog) · per-checkpoint": tc.STOP_CROSS_GROUP,
                  "wc1→wc2 (dog) · per-checkpoint": tc.STOP_CROSS_GROUP}
    order = [g for g, _ in tc.STOP_GROUPS]
    n = len(order) // 2
    assert all(not g.endswith(tc.CONTROLS_SUFFIX) for g in order[:n])
    assert order[n:] == [g + tc.CONTROLS_SUFFIX for g in order[:n]]


def test_complete_factorial_fills_missing_pairs_only():
    have = {("cat", "a", "a"), ("lion", "a", "b")}
    out = tc.complete_factorial(have, ("cat", "lion"), ("a", "b"), (SC1, SC2),
                                {"cat": ("t1",), "lion": ("t2", "t3")})
    assert set(out) == {("cat", "a", "b"), ("cat", "b", "a"), ("cat", "b", "b"),
                        ("lion", "a", "a"), ("lion", "b", "a"), ("lion", "b", "b")}
    assert out[("lion", "b", "b")] == {"teachers": ("t2", "t3"), "animal": "lion", "tseed": 1,
                                       "cells": [SC1, SC2]}


def test_earlystop_completion_on_fake_registry_layout(tmp_path):
    origs = [b for b in ("pythia-160m", "pythia-160m-seed1", "pythia-160m-seed2",
                         "pythia-160m-seed3")]
    regs = [_fake_registry(tmp_path, f"r/d{i}{j}", s, r, (SC1, SC2))
            for i, s in enumerate(origs) for j, r in enumerate(origs)]
    regs.append(_fake_registry(tmp_path, "r/cat", "pythia-160m-seed1", "pythia-160m-seed1",
                               (SC1, SC2), teachers=("k1", "k2"), animal="cat", controls=False))
    reg = tmp_path / "run_logs" / "registry"
    reg.mkdir(parents=True)
    for an in ("cat", "pig"):
        (reg / f"{an}_top10_kband_configs.json").write_text(json.dumps(
            {"param_tags": [f"{an}1", f"{an}2"]}))
    arms = tc.earlystop_arms(tmp_path, regs, complete=("cat", "pig"),
                             stride3_registry="r/d00")
    treat = [a for a in arms if a.kind == "treatment"]
    by = {(a.animal, a.sender, a.receiver): a for a in treat}
    assert sum(a.animal == "dog" for a in treat) == 16
    assert sum(a.animal == "cat" for a in treat) == 16 and sum(a.animal == "pig" for a in treat) == 16
    assert by[("cat", "pythia-160m-seed1", "pythia-160m-seed1")].teachers == ("k1", "k2")
    filled = by[("cat", "pythia-160m", "pythia-160m-seed2")]
    assert filled.teachers == ("cat1", "cat2") and filled.cells == (SC1, SC2)
    assert filled.group == tc.STOP_CROSSSEED_GROUP
    ctl = [a for a in arms if a.kind == "control" and a.control_of == filled.label]
    assert len(ctl) == 1 and ctl[0].animal == "cat"


def test_merge_leaf_points_union_and_shared_step_rule():
    own = pd.DataFrame({"step": [625, 1250, 12500], "p_test": [0.1, 0.2, 0.5]})
    twin = pd.DataFrame({"step": [12500, 25000], "p_test": [0.45, 0.6]})
    m, d = tc.merge_leaf_points(own, twin)
    assert m["step"].tolist() == [625, 1250, 12500, 25000]
    assert m.set_index("step").loc[12500, "p_test"] == pytest.approx(0.5)   # own leaf kept
    assert d == pytest.approx(0.05)
    m, d = tc.merge_leaf_points(None, twin)
    assert m["step"].tolist() == [12500, 25000] and np.isnan(d)
    m, d = tc.merge_leaf_points(None, None)
    assert m.empty and np.isnan(d)


def test_load_arm_reads_both_leaves_of_a_run(tmp_path, patched, monkeypatch):
    spec = tc.ArmSpec("Pythia", tc.STOP_SAME_GROUP, "same-model", "dog", "pythia-160m",
                      "pythia-160m", ("tA",), "es", (SC1, SC2), min_points=tc.STOP_MIN_POINTS)
    plain1, plain2 = tc.plain_twin(SC1), tc.plain_twin(SC2)
    # SC1: both leaves (plain leaf saves every 625 to 25,000; shared steps differ by .001)
    _write_run(tmp_path, spec, "tA", SC1, {s: 0.02 for s in STOP_GRID})
    full = {s: 0.021 for s in range(625, 25001, 625)}
    _set_manifest(_write_run(tmp_path, spec, "tA", plain1, full), 25000, 25000)
    # SC2: plain leaf only (no stop leaf)
    _set_manifest(_write_run(tmp_path, spec, "tA", plain2, full), 25000, 25000)
    data = tc.load_arms([spec], tmp_path)
    runs = data.runs.set_index("cell")
    assert runs.loc[SC1, "n_points"] == 40 and runs.loc[SC2, "n_points"] == 40
    assert (runs["state"] == "alive").all() and (runs["end_step"] == 25000).all()
    p1 = data.points[data.points["cell"] == SC1].set_index("step")["ratio"]
    assert p1[12500] == pytest.approx(2.0) and p1[13125] == pytest.approx(2.1)
    assert any("max |Δ p_target| = 0.001" in n for n in data.notes)
    # the endpoint the plain leaf already gave is not appended twice
    twin = pd.DataFrame({"teacher": ["tA", "tA"], "cell": [SC1, SC2], "step": [25000.0] * 2,
                         "ratio": [7.0, 5.0]})
    monkeypatch.setattr(tc, "endpoint_twin_points", lambda s, root, ftseed=1, notes=None: twin)
    ext = tc.with_ep20_endpoints(data)
    assert not ext.points.duplicated(["arm", "teacher", "cell", "step"]).any()
    assert len(ext.points) == len(data.points)


def test_control_endpoint_kept_unprobed_dropped_dead_treatment_needs_probe(monkeypatch, tmp_path):
    from dataclasses import replace

    cells = ("a_stop12500", "b_stop12500", "c_stop12500")
    labels = {"a": 0.0, "b": np.nan, "c": 1.0}           # alive / unprobed / dead
    treat = tc.ArmSpec("Pythia", "g", "c", "dog", "pythia-160m", "pythia-160m", ("t1",), "s",
                       cells)
    ctl = replace(treat, kind="control", control_of=treat.label, teachers=(tc.CONTROL_TEACHER,))
    for spec in (treat, ctl):
        for c in "abc":
            (tmp_path / spec.kind / c / "ftseed1").mkdir(parents=True)
    monkeypatch.setattr(tc.ArmSpec, "lineage_dir", lambda self, root, t: tmp_path / self.kind)
    monkeypatch.setattr(tio, "load_prior", lambda *a, **k: 0.01)
    monkeypatch.setattr(tio, "_probe_pool_for", lambda *a, **k: "p")
    monkeypatch.setattr(tio, "_anchored_cut", lambda *a, **k: 1.0)
    monkeypatch.setattr(tio, "_control_value", lambda twin, *a, **k: (0.02, "eval"))
    monkeypatch.setattr(tio, "_dead_label", lambda twin, *a: labels[twin.name])
    notes: list[str] = []
    kept = lambda spec: set(tc.endpoint_twin_points(spec, tmp_path, notes=notes)["cell"])  # noqa: E731
    assert kept(treat) == {"a_stop12500"}                 # treatment: probed alive only
    assert kept(ctl) == {"a_stop12500", "b_stop12500"}    # control: unprobed kept, dead dropped
    assert notes == [f"{ctl.label}: 1 control endpoints used unprobed (no Pile-probe label; a "
                     "control endpoint is dropped only when labelled dead)"]


FULL_GRID = (*STOP_GRID, 25000)
PC1, PC2 = "ep20_lr1e-4_r16a32", "ep20_lr2e-4_r8a4"


def test_plain_full_length_registry_arm_reads_nine_steps_with_same_step_control(
        tmp_path, patched, monkeypatch):
    """A registry of PLAIN cells (full-length runs, no ``_stop{N}`` twin): the arm keeps the
    plain cells, lands in the re-pretrain group, and reads every saved step plus the 25,000
    endpoint from the leaf's own pref_traj, its control at the same steps."""
    rel = _fake_registry(tmp_path, "r/prv12pr1234ft", "pythia-160m-pilereprov1-seed1234",
                         "pilerepro-seed1234", (PC1, PC2))
    arms = tc.earlystop_arms(tmp_path, [rel], complete=())
    spec, ctl = arms
    assert spec.cells == (PC1, PC2) and spec.group == tc.STOP_CROSS_GROUP
    assert ctl.control_of == spec.label and ctl.group == tc.STOP_CROSS_GROUP + tc.CONTROLS_SUFFIX
    assert all(tc.plain_twin(c) is None for c in spec.cells)
    for cell in (PC1, PC2):
        for t in spec.teachers:
            _write_run(tmp_path, spec, t, cell, {s: 0.03 for s in FULL_GRID})
        _write_run(tmp_path, ctl, tc.CONTROL_TEACHER, cell, {s: 0.01 for s in FULL_GRID},
                   delta_ce=None)                       # unprobed control: kept
    data = tc.load_arms(arms, tmp_path)
    assert set(data.runs.loc[data.runs["arm"] == spec.label, "state"]) == {"alive"}
    by_arm = data.points.groupby("arm")
    for label in (spec.label, ctl.label):
        steps = by_arm.get_group(label).groupby(["teacher", "cell"])["step"].apply(sorted)
        assert all(s == list(map(float, FULL_GRID)) for s in steps)
    twin = pd.DataFrame({"teacher": ["tA"], "cell": [PC1], "step": [25000.0], "ratio": [3.0]})
    monkeypatch.setattr(tc, "endpoint_twin_points", lambda s, root, ftseed=1, notes=None: twin)
    ext = tc.with_ep20_endpoints(data, by_group=False)
    assert len(ext.points) == len(data.points)          # the endpoint is already a point
    # a treatment run without a Pile probe is not alive
    _write_run(tmp_path / "u", spec, "tA", PC1, {s: 0.03 for s in FULL_GRID}, delta_ce=None)
    runs = tc.load_arms([spec], tmp_path / "u").runs
    assert runs.set_index(["teacher", "cell"]).loc[("tA", PC1), "state"] == "unlabelled"


def test_full_length_prv1_pr1234_arms_are_registered():
    repo = Path(__file__).resolve().parents[1]
    by = {(a.sender, a.receiver, a.kind): a for a in tc.earlystop_arms(repo)}
    prv1, pr = "pythia-160m-pilereprov1-seed1234", "pythia-160m-pilerepro-seed1234"
    for s, r in ((prv1, pr), (pr, prv1)):
        t = by[(s, r, "treatment")]
        assert t.group == tc.STOP_CROSS_GROUP and len(t.cells) == 10 and len(t.teachers) == 10
        assert all(tc.plain_twin(c) is None for c in t.cells)
        assert by[(s, r, "control")].control_of == t.label
