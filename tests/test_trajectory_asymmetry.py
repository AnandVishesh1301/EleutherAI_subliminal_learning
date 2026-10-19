"""Hand-checkable values for the per-step transfer-asymmetry readout.

No disk: `transfer_significance_io.load_cell_matrix` / `student_tree_prior` are monkeypatched as in
`tests/test_transfer_asymmetry.py`, and the trajectory source
(`trajectory_asymmetry._run_trajectories`) serves synthetic step grids.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import trajectory_asymmetry as tg
from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.analysis import transfer_asymmetry as ta
from subliminal_learning.analysis import transfer_significance_io as tio

CELLS = [f"big{i}" for i in range(4)]
TEACHERS = tuple(f"t{i}" for i in range(8))
GRID = (625, 1250, 2500, 30000)          # deliberately not a fraction-friendly grid
PRIOR = 0.1
CTRL = 0.1


@pytest.fixture(autouse=True)
def _no_result_cache(monkeypatch):
    """The monkeypatched sources are invisible to the result-cache keys: tests run uncached
    unless they opt in (the result-cache tests at the end)."""
    monkeypatch.setenv("SL_RESULT_CACHE", "0")


def _repo(tmp_path):
    (tmp_path / "run_logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_logs" / "big_cells.json").write_text(json.dumps({"cells": CELLS}))
    (tmp_path / "run_logs" / "dog_teachers.json").write_text(
        json.dumps({"param_tags": list(TEACHERS)}))
    return tmp_path


def _family(bases=("a", "b")):
    return ta.Family("fam", ("dog",), bases, "run_logs/{animal}_teachers.json",
                     (("big", "run_logs/big_cells.json"),))


def _matrix(cells, teachers, value):
    pools = [*teachers, tio.CONTROL_LABEL]
    values = np.full((len(cells), len(pools)), np.nan)
    for j, tag in enumerate(teachers):
        for i, cell in enumerate(cells):
            values[i, j] = value(tag, cell)
    values[:, -1] = CTRL
    frame = pd.DataFrame(columns=["cell", "pool", "value", "is_control"])
    return tio.CellMatrix(frame=frame, values=values, cells=list(cells), pools=pools,
                          control_col=len(pools) - 1, missing=[])


def _patch(monkeypatch, endpoint, traj):
    """`endpoint(sender, receiver, tag, cell)` -> endpoint p; `traj(spec, teacher)` -> frame."""
    def fake(animal, sender, receiver, teachers, *, cells=None, **kw):
        return _matrix(list(cells), list(teachers), lambda t, c: endpoint(sender, receiver, t, c))

    monkeypatch.setattr(tio, "load_cell_matrix", fake)
    monkeypatch.setattr(tio, "student_tree_prior", lambda animal, receiver, **kw: PRIOR)
    monkeypatch.setattr(tg, "_run_trajectories", lambda spec, teacher, root, ftseed:
                        traj(spec, teacher))


def _grid(cells, p_of_step, skip=()):
    rows = [{"tag": c, "step": float(s), "p_test": p_of_step(c, s)}
            for c in cells for s in GRID if (c, s) not in skip]
    return pd.DataFrame(rows, columns=["tag", "step", "p_test"])


def _run(tmp_path, family=None, **kw):
    return tg.analyze_family(family or _family(), repo=_repo(tmp_path), output_root=tmp_path,
                             n_perm=500, **kw)


def test_last_step_value_equals_endpoint_value(monkeypatch, tmp_path):
    end = {"a": 0.5, "b": 0.3}
    _patch(monkeypatch, lambda s, r, t, c: end[s],
           lambda spec, t: _grid(spec.cells, lambda c, s: end[spec.sender] if s == GRID[-1]
                                 else 0.2))
    res = _run(tmp_path)
    last = res.steps[res.steps["step"] == GRID[-1]].iloc[0]
    assert last["mean A->B"] == pytest.approx((0.5 - CTRL) / PRIOR)
    assert last["mean B->A"] == pytest.approx((0.3 - CTRL) / PRIOR)
    row = res.summary.iloc[0]
    assert row["cross-check ok"] and row["last-step contrast"] == pytest.approx(
        row["endpoint contrast"])
    inv = res.inventory.set_index("sender")
    assert inv.loc["a", "control"] == tg.CONTROL_CONSTANT and inv.loc["a", "trajectory"] == "full"


def test_low_coverage_step_is_listed_but_untested(monkeypatch, tmp_path):
    # b -> a misses step 1250 on 4 of its 32 runs (87.5% < 90%)
    skip = {(c, 1250) for c in CELLS}
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           lambda spec, t: _grid(spec.cells, lambda c, s: 0.3,
                                 skip=skip if spec.sender == "b" and t == "t0" else ()))
    res = _run(tmp_path)
    st = res.steps.set_index("step")
    assert not st.loc[1250, "tested"] and np.isnan(st.loc[1250, "contrast"])
    assert st.loc[1250, "coverage B->A"] == pytest.approx(28 / 32)
    assert st.loc[625, "tested"]


def test_contrast_sign_follows_base_order(monkeypatch, tmp_path):
    end = {"a": 0.5, "b": 0.3}
    _patch(monkeypatch, lambda s, r, t, c: end[s],
           lambda spec, t: _grid(spec.cells, lambda c, s: end[spec.sender]))
    assert (_run(tmp_path).steps["contrast"] > 0).all()
    flipped = _run(tmp_path, _family(("b", "a")))
    assert (flipped.steps["contrast"] < 0).all()
    assert list(flipped.summary[["A", "B"]].iloc[0]) == ["b", "a"]


def test_summary_fields_on_a_mid_run_sign_flip(monkeypatch, tmp_path):
    # a -> b below b -> a at the first two steps, above from 2,500 on; teacher noise keeps the
    # contrast teacher-varying so the paired test is not degenerate
    def p(sender, teacher, step):
        k = int(teacher[1:]) * 0.001
        if sender == "b":
            return 0.3 + k
        return {625: 0.1, 1250: 0.2, 2500: 0.45, 30000: 0.4}[step] + k

    _patch(monkeypatch, lambda s, r, t, c: p(s, t, 30000),
           lambda spec, t: _grid(spec.cells, lambda c, s: p(spec.sender, t, s)))
    row = _run(tmp_path).summary.iloc[0]
    assert row["endpoint contrast"] == pytest.approx(1.0)
    assert row["peak step"] == 625 and row["peak contrast"] == pytest.approx(-2.0)
    assert row["earliest p<.05 step"] == 625
    assert row["same-sign share"] == pytest.approx(0.5)
    assert row["opposite-sign significant"]
    assert row["last tested step"] == 30000


def test_pair_without_trajectory_on_one_direction_is_excluded(monkeypatch, tmp_path):
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           lambda spec, t: _grid(spec.cells, lambda c, s: 0.3) if spec.sender == "a"
           else _grid([], lambda c, s: 0.3))
    res = _run(tmp_path)
    assert res.summary.empty and res.steps.empty and len(res.excluded) == 1
    ex = res.excluded.iloc[0]
    assert ex["reason A->B"] == "has trajectory" and ex["reason B->A"].startswith("no trajectory")
    assert res.inventory.set_index("sender").loc["b", "trajectory"] == "none"


def test_steps_stay_absolute(monkeypatch, tmp_path):
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           lambda spec, t: _grid(spec.cells, lambda c, s: 0.3))
    res = _run(tmp_path)
    assert sorted(res.steps["step"]) == list(GRID)
    assert res.summary.iloc[0]["last tested step"] == 30000


# ----------------------------------------------------------------------------------------------
# early-stop design (per-step control)
# ----------------------------------------------------------------------------------------------
STOP_CELLS = tuple(f"{c}_stop12500" for c in CELLS)
STOP_GRID = (625, 3125, 12500)


def _stop_arms():
    out = []
    for s, r in (("a", "b"), ("b", "a")):
        spec = tc.ArmSpec("Pythia", "g", "cross-model", "dog", s, r, TEACHERS, "earlystop",
                          STOP_CELLS)
        out.append(spec)
        out.append(tc.ArmSpec("Pythia", "g controls", "ctrl", "dog", s, r,
                              (tc.CONTROL_TEACHER,), "earlystop", STOP_CELLS, kind="control",
                              control_of=spec.label))
    return out


def _stop_traj(spec, teacher):
    if spec.cell_set != "earlystop":                       # the ep20 lineages
        return _grid(spec.cells, lambda c, s: 0.3)
    if spec.kind == "control":
        # the control lacks step 3,125 on cell big0 of the a -> b arm
        return pd.DataFrame([{"tag": c, "step": float(s), "p_test": 0.01 * s / 625}
                             for c in spec.cells for s in STOP_GRID
                             if not (spec.sender == "a" and c == STOP_CELLS[0] and s == 3125)])
    p = 0.5 if spec.sender == "a" else 0.3
    return pd.DataFrame([{"tag": c, "step": float(s), "p_test": p + 0.01 * s / 625}
                         for c in spec.cells for s in STOP_GRID])


def test_stop_design_subtracts_the_control_at_the_same_step(monkeypatch, tmp_path):
    _patch(monkeypatch, lambda s, r, t, c: 0.3, _stop_traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    stop = res.steps[res.steps["design"] == "big · checkpoints to 12,500"].set_index("step")
    assert list(stop.index) == list(STOP_GRID)
    # treatment minus same-step control is constant: (0.5 - 0) / prior and (0.3 - 0) / prior
    assert stop.loc[12500, "mean A->B"] == pytest.approx(5.0)
    assert stop.loc[12500, "mean B->A"] == pytest.approx(3.0)
    # big0 of a -> b has no control at 3,125: 24/32 runs covered, step untested
    assert stop.loc[3125, "coverage A->B"] == pytest.approx(24 / 32)
    assert not stop.loc[3125, "tested"] and stop.loc[625, "tested"]
    row = res.summary.set_index("design").loc["big · checkpoints to 12,500"]
    assert np.isnan(row["endpoint contrast"]) and np.isnan(row["cross-check ok"])
    assert row["ep20 endpoint contrast"] == pytest.approx(0.0)
    inv = res.inventory.set_index(["design", "sender"])
    assert inv.loc[("big · checkpoints to 12,500", "a"), "control"] == tg.CONTROL_PER_STEP
    assert inv.loc[("big", "a"), "control"] == tg.CONTROL_CONSTANT


def test_stop_design_never_leaks_into_the_ep20_design(monkeypatch, tmp_path):
    _patch(monkeypatch, lambda s, r, t, c: 0.3, _stop_traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    ep20 = res.steps[res.steps["design"] == "big"]
    assert sorted(ep20["step"].unique()) == list(GRID)
    assert set(ep20["runs A->B"]) == {32}
    inv = res.inventory.set_index(["design", "sender"])
    assert inv.loc[("big", "a"), "runs (endpoint)"] == 32
    assert inv.loc[("big", "a"), "saved steps"] == len(GRID)


def _partial_stop_traj(spec, teacher):
    """Stop students landed for teachers t0-t5 only (24/32 runs = 75%, below 90%)."""
    if spec.kind == "treatment" and spec.cell_set == "earlystop" and int(teacher[1:]) > 5:
        return pd.DataFrame(columns=["tag", "step", "p_test"])
    return _stop_traj(spec, teacher)


def test_prelim_tests_partial_stop_steps_and_labels_them(monkeypatch, tmp_path):
    _patch(monkeypatch, lambda s, r, t, c: 0.3, _partial_stop_traj)
    final = _run(tmp_path, stop_arms=_stop_arms())
    assert "big · checkpoints to 12,500" not in set(final.summary["design"])
    pre = _run(tmp_path, stop_arms=_stop_arms(), prelim=True)
    stop = pre.steps[pre.steps["design"] == "big · checkpoints to 12,500"].set_index("step")
    assert stop.loc[625, "tested"] and stop.loc[625, "coverage class"] == tg.CLASS_PRELIM
    assert stop.loc[625, "coverage A->B"] == pytest.approx(24 / 32)   # vs the designed set
    # big0 of a -> b lacks its 3,125 control: 18 matched runs < 20, untested
    assert not stop.loc[3125, "tested"]
    row = pre.summary.set_index("design").loc["big · checkpoints to 12,500"]
    assert row["coverage class"] == tg.CLASS_PRELIM
    # the ep20 design is untouched by the prelim rule
    assert set(pre.steps.loc[pre.steps["design"] == "big", "coverage class"]) == {tg.CLASS_FULL}


def test_direction_table_sits_beside_the_diagonals(monkeypatch, tmp_path):
    arms = _stop_arms()
    diag = tc.ArmSpec("Pythia", "g", "same-model", "dog", "a", "a", TEACHERS, "earlystop",
                      STOP_CELLS)
    arms += [diag, tc.ArmSpec("Pythia", "g controls", "ctrl", "dog", "a", "a",
                              (tc.CONTROL_TEACHER,), "earlystop", STOP_CELLS, kind="control",
                              control_of=diag.label)]
    _patch(monkeypatch, lambda s, r, t, c: 0.3, _partial_stop_traj)
    table, landing = tg.direction_table(arms, output_root=tmp_path)
    row = table[(table["arm"] == arms[0].label) & (table["step"] == 12500)].iloc[0]
    assert row["n"] == 24 and row["mean"] == pytest.approx(5.0)
    assert row["sender diag n"] == 24 and np.isnan(row["receiver diag mean"])
    land = landing.set_index("arm")
    assert land.loc[arms[0].label, "status"] == "partial"
    assert land.loc[arms[0].label, "controls landed"] == 4


# ----------------------------------------------------------------------------------------------
# bootstrap bands, superseded-by note, figures
# ----------------------------------------------------------------------------------------------
BAND_COLUMNS = ["mean A->B lo", "mean A->B hi", "mean B->A lo", "mean B->A hi", "contrast lo",
                "contrast hi"]


def _noisy(monkeypatch, skip_b0=False):
    def p(sender, teacher, cell, step):
        k = int(teacher[1:]) * 0.01 + int(cell[-1]) * 0.003
        return (0.5 if sender == "a" else 0.3) + k + step * 1e-6

    skip = {(c, 1250) for c in CELLS}
    _patch(monkeypatch, lambda s, r, t, c: p(s, t, c, 30000),
           lambda spec, t: _grid(spec.cells, lambda c, s: p(spec.sender, t, c, s),
                                 skip=skip if skip_b0 and spec.sender == "b" and t == "t0"
                                 else ()))


def test_bootstrap_bands_bracket_the_mean_and_are_deterministic(monkeypatch, tmp_path):
    _noisy(monkeypatch, skip_b0=True)
    res = _run(tmp_path)
    st = res.steps.set_index("step")
    assert not st.loc[1250, "tested"] and st.loc[1250, BAND_COLUMNS].isna().all()
    t = st[st["tested"]]
    eps = 1e-9
    for col in ("mean A->B", "mean B->A", "contrast"):
        assert (t[f"{col} lo"] <= t[col] + eps).all() and (t[col] <= t[f"{col} hi"] + eps).all()
        assert (t[f"{col} hi"] > t[f"{col} lo"]).all()          # teacher noise -> real width
    again = _run(tmp_path).steps
    pd.testing.assert_frame_equal(res.steps[BAND_COLUMNS], again[BAND_COLUMNS])


def test_excluded_pair_with_a_stop_design_is_marked_superseded(monkeypatch, tmp_path):
    def traj(spec, teacher):
        if spec.cell_set != "earlystop":                       # no ep20 trajectories at all
            return _grid([], lambda c, s: 0.3)
        return _stop_traj(spec, teacher)

    _patch(monkeypatch, lambda s, r, t, c: 0.3, traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    ex = res.excluded.iloc[0]
    assert ex["superseded by"] == "big" + tg.TRAJECTORY_SUFFIX
    assert "big" + tg.TRAJECTORY_SUFFIX in set(res.summary["design"])


def test_figures_build_on_a_tiny_result(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    def traj(spec, teacher):
        if spec.cell_set != "earlystop":
            return _grid([], lambda c, s: 0.3)
        return _stop_traj(spec, teacher)

    _patch(monkeypatch, lambda s, r, t, c: 0.3, traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    assert isinstance(tg.plot_coverage(res), Figure)
    table, _ = tg.direction_table(_stop_arms(), output_root=tmp_path)
    assert {"mean lo", "mean hi"} <= set(table.columns)
    assert isinstance(tg.plot_dog_directions(table), Figure)
    assert isinstance(tg.plot_summary_dumbbell(res), Figure)


# ----------------------------------------------------------------------------------------------
# stitched design: stop points + the same (teacher, cell) ep20 endpoint
# ----------------------------------------------------------------------------------------------
STITCHED = "big" + tg.TRAJECTORY_SUFFIX


def _slot_endpoint(s, r, t, c):
    return (0.5 if s == "a" else 0.3) + int(t[1:]) * 0.01 + int(c[-1]) * 0.003


def test_stitched_design_lands_on_the_ep20_endpoint(monkeypatch, tmp_path):
    _patch(monkeypatch, _slot_endpoint, _stop_traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    st = res.steps[res.steps["design"] == STITCHED].set_index("step")
    assert list(st.index) == [*STOP_GRID, GRID[-1]]        # last step read from the ep20 runs
    last = st.loc[GRID[-1]]
    assert last["tested"] and last["runs A->B"] == 32
    arm = ta.load_arm(_family(), "dog", "a", "b", "big", CELLS, repo=tmp_path,
                      output_root=tmp_path)
    assert last["mean A->B"] == pytest.approx(arm.frame["value"].mean(), abs=0)
    row = res.summary.set_index("design").loc[STITCHED]
    assert row["last tested step"] == GRID[-1] and row["cross-check ok"]
    assert row["slots endpoint contrast"] == row["last-step contrast"]
    inv = res.inventory.set_index(["design", "sender"])
    assert inv.loc[(STITCHED, "a"), "control"] == tg.CONTROL_PER_STEP
    disp = tg.summary_display(res).set_index("design").loc[STITCHED]
    assert disp["last-step cells"] == "big"


def test_stitched_points_report_a_missing_endpoint_slot():
    spec = _stop_arms()[0]
    frame = pd.DataFrame([{"sender": "a", "teacher": t, "cell": c, "value": 1.0}
                          for t in TEACHERS for c in CELLS if (t, c) != ("t3", "big2")])
    arm = ta.ArmLoad(frame, list(TEACHERS), PRIOR, True, "", {})
    stop = pd.DataFrame({"sender": "a", "teacher": "t0", "cell": STOP_CELLS[0],
                         "step": [625], "value": [0.0]})
    pts, missing, source = tg._stitched_points(stop, spec, arm, {})
    assert missing == ["t3/big2"]
    assert len(pts) == 1 + len(TEACHERS) * len(CELLS) - 1
    assert set(pts.loc[pts["step"] != 625, "step"]) == {tg.EP20_LAST_STEP}
    assert "schedule" in source


def test_stop_only_design_is_superseded_and_not_plotted(monkeypatch, tmp_path):
    _patch(monkeypatch, _slot_endpoint, _stop_traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    summ = res.summary.set_index("design")
    assert summ.loc["big · checkpoints to 12,500", "design status"] == "superseded by " + STITCHED
    assert "big · checkpoints to 12,500" in set(res.steps["design"])
    drawn = set(tg._active(res.summary)["design"])
    assert STITCHED in drawn and "big · checkpoints to 12,500" not in drawn
    assert "big · checkpoints to 12,500" not in set(tg.summary_display(res)["design"])
    inv = res.inventory[res.inventory["design"] == "big · checkpoints to 12,500"]
    assert (inv["design status"] == "superseded by " + STITCHED).all()


def test_dog_directions_figure_data_carries_the_last_step(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    _patch(monkeypatch, _slot_endpoint, _stop_traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    table, _ = tg.direction_table(_stop_arms(), output_root=tmp_path)
    d = tg.trajectory_directions(table, res)
    assert "endpoint" not in d.columns
    end = d[d["step"] == GRID[-1]]
    assert set(zip(end["sender"], end["receiver"])) == {("a", "b"), ("b", "a")}
    assert (end["n"] == 32).all()
    tg.plot_dog_directions(d)


def test_stop_pair_without_an_endpoint_test_is_a_trajectory_design(monkeypatch, tmp_path):
    _patch(monkeypatch, _slot_endpoint, _stop_traj)
    no_ep = pd.DataFrame(columns=["animal", "A", "B", "design", "contrast", "p_two_sided"])
    steps, summary, inv = tg._stop_pairs(_family(), _repo(tmp_path), tmp_path, _stop_arms(),
                                         no_ep, {}, n_perm=500, seed=0)
    assert {r["design"] for r in steps} == {STITCHED}
    assert sorted({r["step"] for r in steps}) == list(STOP_GRID)
    (row,) = summary
    assert row["design"] == STITCHED and not row.get("design status")
    assert np.isnan(row["endpoint contrast"]) and np.isnan(row["ep20 endpoint contrast"])
    assert {r["design"] for r in inv} == {STITCHED}
    res = tg.TrajectoryAsymmetry(pd.DataFrame(steps).reindex(columns=tg.STEP_COLUMNS),
                                 pd.DataFrame(summary).reindex(columns=tg.SUMMARY_COLUMNS),
                                 pd.DataFrame(inv).reindex(columns=tg.INVENTORY_COLUMNS),
                                 pd.DataFrame(columns=tg.EXCLUDED_COLUMNS), no_ep)
    assert STITCHED in set(tg.summary_display(res)["design"])


def test_curves_drop_steps_below_the_coverage_rule(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    _patch(monkeypatch, _slot_endpoint, _stop_traj)
    res = _run(tmp_path, stop_arms=_stop_arms())
    table, _ = tg.direction_table(_stop_arms(), output_root=tmp_path)
    d = tg.trajectory_directions(table, res)
    ab_ = d[(d["sender"] == "a") & (d["step"] == 3125)]
    assert ab_["coverage"].iloc[0] == pytest.approx(24 / 32)        # control lacks big0@3125
    fig = tg.plot_dog_directions(d)
    xs = {float(x) for ln in fig.axes[0].get_lines() if ln.get_linestyle() == "-"
          for x in ln.get_xdata()}
    assert 3125 not in xs and 625 in xs                              # omitted, not interpolated
    fig = tg.plot_direction_curves(res)
    ax = fig.axes[0]
    solid = [ln for ln in ax.get_lines() if ln.get_linestyle() == "-" and ln.get_label() != "_"]
    assert 3125 not in {float(x) for x in solid[0].get_xdata()}


def test_curve_figures_draw_one_design_per_pair():
    summ = pd.DataFrame([
        {"family": "f", "animal": "cat", "A": "a", "B": "b", "design": "big"},
        {"family": "f", "animal": "cat", "A": "a", "B": "b", "design": STITCHED},
        {"family": "f", "animal": "cat", "A": "a", "B": "c", "design": "big"},
        {"family": "f", "animal": "cat", "A": "a", "B": "d",
         "design": "big · checkpoints to 12,500", "design status": "superseded by x"},
    ]).reindex(columns=tg.SUMMARY_COLUMNS)
    got = tg.curve_pairs(summ)
    assert list(zip(got["B"], got["design"])) == [("b", STITCHED), ("c", "big")]


def test_family_display_maps_pythia_families_and_passes_others_through():
    assert tg.family_display("cross-seed") == "diff init, diff data order"
    assert tg.family_display("cross-data-order") == "same init, diff data order"
    assert tg.family_display("re-pretrain") == "diff init, same data order"
    assert tg.family_display("olmo-seed") == "olmo-seed"
    assert tg.family_display("unknown") == "unknown"


def test_contrast_blocks_split_re_pretrain_and_keep_block_order():
    assert [t for t, _ in tg.DOG_CONTRAST_BLOCKS] == [
        "diff init, diff data order: orig, fs1–fs3", "same init, diff data order: orig, ds1–ds3",
        "diff init, same data order: Rerun1234, wc1, wc2",
        "Pythia-160M re-pretrain: orig, Modern1234, Rerun1234"]
    t = [t for t, _ in tg.DOG_CONTRAST_BLOCKS]
    assert tg.contrast_block("orig", "fs2") == t[0]
    assert tg.contrast_block("orig", "ds3") == t[1]
    assert tg.contrast_block("prv1", "wc1") == t[2] and tg.contrast_block("wc1", "wc2") == t[2]
    assert tg.contrast_block("prv1", "orig") == t[3]
    assert tg.contrast_block("pr1234", "orig") == t[3]
    assert tg.contrast_block("prv1", "pr1234") == t[3]
    assert tg.contrast_block("wc1", "orig") is None and tg.contrast_block("fs1", "ds1") is None


def _pp_res(prior_ab=0.02, prior_ba=0.08):
    """Synthetic per-step-control dog pair prv1⇄wc1, 4 paired teachers × 2 cells, 2 steps."""
    rng = np.random.default_rng(1)
    key = {"family": "re-pretrain", "animal": "dog", "A": "prv1", "B": "wc1", "design": STITCHED}
    pts, raw = [], []
    for d, prior in (("A->B", prior_ab), ("B->A", prior_ba)):
        for t in ("t0", "t1", "t2", "t3"):
            for c in ("c0", "c1"):
                for st in (625, 25000):
                    ps, pc = rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.1)
                    raw.append({"direction": d, "teacher": t, "cell": c, "step": st,
                                "pp": ps - pc})
                    pts.append({**key, "direction": d, "teacher": t, "cell": c, "step": st,
                                "value": (ps - pc) / prior})
    summ = pd.DataFrame([{**key, "coverage class": "", "test": ta.TEST_PAIRED}]
                        ).reindex(columns=tg.SUMMARY_COLUMNS)
    inv = pd.DataFrame([{"family": "re-pretrain", "animal": "dog", "sender": s, "receiver": r,
                         "design": STITCHED, "prior": p}
                        for s, r, p in (("prv1", "wc1", prior_ab), ("wc1", "prv1", prior_ba))]
                       ).reindex(columns=tg.INVENTORY_COLUMNS)
    res = tg.TrajectoryAsymmetry(pd.DataFrame(columns=tg.STEP_COLUMNS), summ, inv,
                                 pd.DataFrame(columns=tg.EXCLUDED_COLUMNS), pd.DataFrame(),
                                 points=pd.DataFrame(pts).reindex(columns=tg.POINT_COLUMNS))
    return res, pd.DataFrame(raw), pd.DataFrame(pts)


def test_pp_contrast_is_on_raw_slot_differences_not_a_prior_rescaling():
    res, raw, pts = _pp_res()
    pp = tg.dp_contrast_table(res, n_perm=200)
    assert list(pp["step"]) == [625, 25000]
    assert set(pp["block"]) == {"diff init, same data order: Rerun1234, wc1, wc2"}
    for row in pp.itertuples():
        r = raw[raw["step"] == row.step]
        want = (r[r["direction"] == "A->B"]["pp"].mean() - r[r["direction"] == "B->A"]["pp"].mean())
        assert row.contrast == pytest.approx(want, abs=1e-9)
        p = pts[pts["step"] == row.step]
        prior_unit = p[p["direction"] == "A->B"]["value"].mean() - p[p["direction"] == "B->A"][
            "value"].mean()
        for prior in (0.02, 0.08):
            assert row.contrast != pytest.approx(prior * prior_unit, rel=1e-3)
        assert row._9 <= row.contrast <= row._10   # contrast lo / hi
        assert row._7 == 8 and row.tested   # n paired slots
        assert 0 < row.p_two_sided <= 1
    assert set(pp["A"]) == {"Rerun1234"} and list(pp.columns) == tg.DP_COLUMNS


def test_dog_contrast_figure_is_2x2_linear_dp_with_inner_legends():
    import matplotlib
    matplotlib.use("Agg")
    res, _, _ = _pp_res()
    fig = tg.plot_contrast_curves_dog(tg.dp_contrast_table(res, n_perm=50))
    axes = fig.axes
    assert len(axes) == 4
    assert [ax.get_title() for ax in axes] == [f"dog · {t}" for t, _ in tg.DOG_CONTRAST_BLOCKS]
    assert fig._suptitle is None
    assert not any("weight copies" in ax.get_title() for ax in axes)
    assert axes[2].get_legend().get_texts()[0].get_text() == "Rerun1234→wc1"
    fig.canvas.draw()
    for ax in axes:
        assert ax.get_xlabel() == "student step" and ax.get_xscale() == "linear"
        assert ax.get_ylabel() == tg.DP_YLABEL
    lb, ab = axes[2].get_legend().get_window_extent(), axes[2].get_window_extent()
    assert ab.x0 <= lb.x0 and lb.x1 <= ab.x1 and ab.y0 <= lb.y0 and lb.y1 <= ab.y1
    assert "25,000" in [t.get_text() for t in axes[2].get_xticklabels()]


# ----------------------------------------------------------------------------------------------
# per-slot points, per-step significance, sign change, ordering stability
# ----------------------------------------------------------------------------------------------
def test_points_hold_tested_steps_of_both_directions(monkeypatch, tmp_path):
    skip = {(c, 1250) for c in CELLS}
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           lambda spec, t: _grid(spec.cells, lambda c, s: 0.3,
                                 skip=skip if spec.sender == "b" and t == "t0" else ()))
    res = _run(tmp_path)
    pts = res.points
    assert list(pts.columns) == tg.POINT_COLUMNS
    assert set(pts["direction"]) == {"A->B", "B->A"}
    assert 1250 not in set(pts["step"])
    assert set(pts["step"]) == set(res.steps.loc[res.steps["tested"], "step"])
    assert len(pts[(pts["step"] == 625) & (pts["direction"] == "A->B")]) == 32


def _synthetic_res(pairs):
    """`pairs`: list of ``(animal, A, B, [(step, contrast, p), ...])`` on a per-step design."""
    rows = [{"family": "fam", "animal": an, "A": a, "B": b, "design": STITCHED, "step": s,
             "contrast": c, "p_two_sided": p, "tested": True}
            for an, a, b, st in pairs for s, c, p in st]
    steps = pd.DataFrame(rows).reindex(columns=tg.STEP_COLUMNS)
    return tg.TrajectoryAsymmetry(steps, pd.DataFrame(columns=tg.SUMMARY_COLUMNS),
                                  pd.DataFrame(columns=tg.INVENTORY_COLUMNS),
                                  pd.DataFrame(columns=tg.EXCLUDED_COLUMNS), pd.DataFrame())


def test_per_step_table_holm_counts():
    ps = [0.002, 0.01, 0.03, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
    cs = [0.5, -0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    res = _synthetic_res([("dog", "a", "b", list(zip(range(1, 10), cs, ps)))])
    (row,) = tg.per_step_table(res).to_dict("records")
    assert row["1"] == "+0.50**" and row["2"] == "-0.40*" and row["4"] == "+0.10"
    assert row["steps sig, endpoint sign (raw)"] == 2
    assert row["steps sig, endpoint sign (Holm)"] == 1
    assert row["steps sig, opposite sign (raw)"] == 1
    assert row["steps sig, opposite sign (Holm)"] == 0


def _flip_traj(spec, teacher):
    """a -> b above b -> a at 625, below from 3,125 on; every teacher agrees; control 0."""
    if spec.kind == "control":
        return pd.DataFrame([{"tag": c, "step": float(s), "p_test": 0.0}
                             for c in spec.cells for s in STOP_GRID])
    k = int(teacher[1:]) * 0.002
    hi = {625: 0.5, 3125: 0.2, 12500: 0.1}
    return pd.DataFrame([{"tag": c, "step": float(s),
                          "p_test": (hi[s] if spec.sender == "a" else 0.3) + k}
                         for c in spec.cells for s in STOP_GRID])


def _stitched_res(monkeypatch, tmp_path, traj):
    _patch(monkeypatch, _slot_endpoint, traj)
    no_ep = pd.DataFrame(columns=["animal", "A", "B", "design", "contrast", "p_two_sided"])
    pts = []
    steps, summary, inv = tg._stop_pairs(_family(), _repo(tmp_path), tmp_path, _stop_arms(),
                                         no_ep, {}, n_perm=500, seed=0, points=pts)
    return tg.TrajectoryAsymmetry(pd.DataFrame(steps).reindex(columns=tg.STEP_COLUMNS),
                                  pd.DataFrame(summary).reindex(columns=tg.SUMMARY_COLUMNS),
                                  pd.DataFrame(inv).reindex(columns=tg.INVENTORY_COLUMNS),
                                  pd.DataFrame(columns=tg.EXCLUDED_COLUMNS), no_ep,
                                  pd.concat(pts, ignore_index=True))


def test_sign_change_table_on_an_early_late_flip(monkeypatch, tmp_path):
    res = _stitched_res(monkeypatch, tmp_path, _flip_traj)
    (row,) = tg.sign_change_table(res, n_perm=500).to_dict("records")
    assert row["s_opp"] == 625 and row["s_end"] == 12500
    assert row["contrast(s_opp)"] > 0 > row["contrast(s_end)"]
    assert row["Δcontrast"] == pytest.approx(row["contrast(s_end)"] - row["contrast(s_opp)"])
    floor = res.steps["p_two_sided"].min()
    assert row["p(Δ)"] == pytest.approx(floor) and row["p(Δ)"] < tg.ALPHA
    assert row["class"] == "both steps sig, Δ sig"


def test_sign_change_table_skips_a_pair_without_an_opposite_step(monkeypatch, tmp_path):
    res = _stitched_res(monkeypatch, tmp_path, _stop_traj)
    assert tg.sign_change_table(res, n_perm=500).empty


def test_ordering_stability_is_one_at_the_reference_step():
    res = _synthetic_res([("dog", "a", "b", [(625, 0.1, .5), (25000, 0.3, .01)]),
                          ("dog", "a", "c", [(625, 0.2, .5), (25000, -0.2, .01)]),
                          ("cat", "b", "c", [(625, -0.1, .5), (25000, 0.1, .01)])])
    out = tg.ordering_stability(res).set_index(["step", "scope"])
    assert out.loc[(25000, "all"), "Spearman vs 25,000"] == pytest.approx(1.0)
    assert out.loc[(25000, "dog"), "Spearman vs 25,000"] == pytest.approx(1.0)
    assert out.loc[(25000, "all"), "share sign = endpoint"] == pytest.approx(1.0)
    assert out.loc[(625, "all"), "share sign = endpoint"] == pytest.approx(1 / 3)
    assert out.loc[(625, "all"), "n pairs"] == 3 and out.loc[(625, "dog"), "n pairs"] == 2


def test_re_pretrain_family_tests_prv1_pr1234_on_the_stride3_design():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    fam = ta.FAMILIES["re-pretrain"]
    assert {"prv1", "pr1234"} <= set(fam.bases)
    designs = dict(ta.family_designs(fam, repo))
    assert list(designs) == ["core30", "core30 stride-3"]        # core30 first, then fallback
    stride3 = designs["core30 stride-3"]
    assert len(stride3) == 10 and set(stride3) <= set(designs["core30"])
    assert tg._CELLSET_NAMES["earlystop"] == "core30 stride-3"   # one display name, both paths


def test_dp_table_csv_round_trip_draws(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    res, _, _ = _pp_res()
    path = tmp_path / "asymmetry_trajectory_contrast.csv"
    tg.dp_contrast_table(res, n_perm=50).to_csv(path, index=False)
    back = pd.read_csv(path)
    assert list(back.columns) == tg.DP_COLUMNS
    fig = tg.plot_contrast_curves_dog(back)
    assert len(fig.axes) == 4 and fig.axes[2].get_legend() is not None


def test_stats_off_keeps_contrasts_and_bands_with_nan_p(monkeypatch, tmp_path):
    _noisy(monkeypatch)
    on, off = _run(tmp_path), _run(tmp_path, stats=False)
    cols = ["contrast", "contrast lo", "contrast hi", "mean A->B", "mean B->A", "win prob"]
    t_on, t_off = on.steps[on.steps["tested"]], off.steps[off.steps["tested"]]
    assert len(t_on) and t_off["p_two_sided"].isna().all()
    assert t_on["p_two_sided"].notna().all()
    pd.testing.assert_frame_equal(t_on[cols].reset_index(drop=True),
                                  t_off[cols].reset_index(drop=True))
    res, _, _ = _pp_res()
    a, b = tg.dp_contrast_table(res, n_perm=50), tg.dp_contrast_table(res, stats=False)
    assert b["p_two_sided"].isna().all()
    np.testing.assert_allclose(a["contrast"], b["contrast"])
    np.testing.assert_allclose(a[["contrast lo", "contrast hi"]], b[["contrast lo", "contrast hi"]])


# ----------------------------------------------------------------------------------------------
# result cache: one unit per (pair, design) + one per family for the endpoint module
# ----------------------------------------------------------------------------------------------
def _cache_setup(monkeypatch, tmp_path):
    """Three bases (three pairs) with a ``pref_traj.json`` on disk for every designed leaf, so
    the record-cache sweep supplies real mtimes; compute calls are counted per unit kind."""
    monkeypatch.setenv("SL_RESULT_CACHE", "1")
    _noisy(monkeypatch)
    fam = _family(("a", "b", "c"))
    repo = _repo(tmp_path)
    for spec in tg._family_specs(fam, repo):
        for t in spec.teachers:
            for c in spec.cells:
                f = spec.lineage_dir(tmp_path, t) / c / "ftseed1" / "eval" / "pref_traj.json"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("{}")
    calls = {"pair": [], "endpoint": 0}
    pair, endpoint = tg._ep20_pair, ta.analyze_family

    def count_pair(family, pr, *a, **kw):
        calls["pair"].append((pr["A"], pr["B"]))
        return pair(family, pr, *a, **kw)

    def count_endpoint(*a, **kw):
        calls["endpoint"] += 1
        return endpoint(*a, **kw)

    monkeypatch.setattr(tg, "_ep20_pair", count_pair)
    monkeypatch.setattr(ta, "analyze_family", count_endpoint)

    def run(**kw):
        calls["pair"].clear()
        calls["endpoint"] = 0
        return tg.analyze_family(fam, repo=repo, output_root=tmp_path,
                                 **{"n_perm": 200, **kw})
    return fam, run, calls


def _frames(res):
    return [res.steps, res.summary, res.inventory, res.excluded, res.endpoint, res.points]


def test_result_cache_warm_call_hits_every_unit_and_reproduces_the_result(monkeypatch,
                                                                         tmp_path):
    _, run, calls = _cache_setup(monkeypatch, tmp_path)
    cold = run()
    assert len(calls["pair"]) == 3 and calls["endpoint"] == 1
    warm = run()
    assert calls["pair"] == [] and calls["endpoint"] == 0
    for c, w in zip(_frames(cold), _frames(warm)):
        pd.testing.assert_frame_equal(c, w)
    assert (tmp_path / "analysis" / "result_cache" / tg.CACHE_NAMESPACE / "index.csv").exists()
    run(refresh_cache=True)                      # bypass: everything recomputed
    assert len(calls["pair"]) == 3 and calls["endpoint"] == 1


def test_result_cache_touching_one_trajectory_recomputes_only_its_pair(monkeypatch, tmp_path):
    fam, run, calls = _cache_setup(monkeypatch, tmp_path)
    run()
    spec = next(s for s in tg._family_specs(fam, tmp_path)
                if (s.sender, s.receiver) == ("c", "a"))
    f = spec.lineage_dir(tmp_path, spec.teachers[0]) / CELLS[0] / "ftseed1" / "eval" / \
        "pref_traj.json"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    run()
    assert calls["pair"] == [("a", "c")] and calls["endpoint"] == 0


def test_result_cache_param_change_recomputes_all_and_disabled_never_stores(monkeypatch,
                                                                           tmp_path):
    _, run, calls = _cache_setup(monkeypatch, tmp_path)
    run()
    run(n_perm=100)
    assert len(calls["pair"]) == 3 and calls["endpoint"] == 1
    monkeypatch.setenv("SL_RESULT_CACHE", "0")
    run(n_perm=50)
    run(n_perm=50)
    assert len(calls["pair"]) == 3 and calls["endpoint"] == 1   # second call computed again


def test_result_cache_corrupt_entry_recomputes(monkeypatch, tmp_path):
    _, run, calls = _cache_setup(monkeypatch, tmp_path)
    cold = run()
    for f in (tmp_path / "analysis" / "result_cache").rglob("*.pkl"):
        f.write_bytes(b"not a pickle")
    again = run()
    assert len(calls["pair"]) == 3 and calls["endpoint"] == 1
    for c, w in zip(_frames(cold), _frames(again)):
        pd.testing.assert_frame_equal(c, w)


# ----------------------------------------------------------------------------------------------
# OLMo per-step design: plain cells saving their own checkpoints, matched per-step controls
# ----------------------------------------------------------------------------------------------
OLMO_GRID = tuple(range(625, 12501, 625))            # 20 saves, the last one is the endpoint


def _olmo_arms():
    out = []
    for s, r in (("a", "b"), ("b", "a")):
        spec = tc.ArmSpec("OLMo", "o", f"{s}->{r}", "dog", s, r, TEACHERS, "olmo", tuple(CELLS))
        out.append(spec)
        out.append(tc.ArmSpec("OLMo", "o controls", f"{s}->{r} ctrl", "dog", s, r,
                              (tc.CONTROL_TEACHER,), "olmo", tuple(CELLS), kind="control",
                              control_of=spec.label))
    return out


def _olmo_traj(spec, teacher):
    if spec.kind == "control":
        return pd.DataFrame([{"tag": c, "step": float(s), "p_test": 0.1} for c in spec.cells
                             for s in OLMO_GRID])
    p = 0.5 if spec.sender == "a" else 0.3
    return pd.DataFrame([{"tag": c, "step": float(s), "p_test": p} for c in spec.cells
                         for s in OLMO_GRID])


def test_olmo_pair_gets_a_per_step_design_with_twenty_steps(monkeypatch, tmp_path):
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if s == "a" else 0.3, _olmo_traj)
    res = _run(tmp_path, stop_arms=_olmo_arms())
    st = res.steps[res.steps["design"] == "big" + tg.TRAJECTORY_SUFFIX]
    assert sorted(st["step"]) == list(OLMO_GRID)
    assert st["tested"].all()
    # (0.5 - 0.1) / 0.1 vs (0.3 - 0.1) / 0.1 at every saved step, the endpoint step included
    np.testing.assert_allclose(st["contrast"], 2.0)
    summ = res.summary.set_index("design")
    assert summ.loc["big" + tg.TRAJECTORY_SUFFIX, "last tested step"] == 12500
    assert summ.loc["big" + tg.BACKFILL_TAG + "end", "design status"].startswith(
        tg.SUPERSEDED_BY)


def test_olmo_stop_arms_keep_only_two_direction_pairs():
    from pathlib import Path
    repo = Path(__file__).absolute().parents[1]
    arms = tg.olmo_stop_arms(repo)
    pairs = {(s.sender, s.receiver) for s in arms if s.kind == "treatment"}
    assert len(pairs) == 6 and all((r, s) in pairs for s, r in pairs)
    assert not any("aux" in s.receiver for s in arms)
    assert {s.control_of for s in arms if s.kind == "control"} == {
        s.label for s in arms if s.kind == "treatment"}
