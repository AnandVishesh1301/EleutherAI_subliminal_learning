import importlib
import json

import numpy as np
from matplotlib.image import AxesImage

from subliminal_learning.analysis import checkpoint_trajectory_grids as ctg

STEPS = [75000, 80000]
TAG = "rA"


def _write(root, lineage, t, s, base, stop=True, endpoint=True):
    leaf = ctg._leaf(root, "dog", lineage, t, s, TAG)
    if stop:
        ev = leaf / ctg.STOP_STUDENT_TAG / "ftseed1" / "eval"
        ev.mkdir(parents=True)
        pts = [{"step": st, "test": {"p_target": base + i * 0.01}} for i, st in enumerate(ctg.TRAJ_STEPS)]
        (ev / "pref_traj.json").write_text(json.dumps({"points": pts}))
        (ev / "pile_kl_traj.json").write_text(json.dumps({"points": [{"step": 625, "kl_base_adapted": 0.5}]}))
    if endpoint:
        d = leaf / ctg.ENDPOINT_STUDENT_TAG / "ftseed1"
        d.mkdir(parents=True)
        (d / "preference_eval.json").write_text(json.dumps({"sets": {"test": {"p_target": base + 0.5}}}))


def _tree(root, skip=None):
    for lin in ctg.LINEAGE_DIRS:
        for i, t in enumerate(STEPS):
            for j, s in enumerate(STEPS):
                _write(root, lin, t, s, 0.1 * (2 * i + j + 1), stop=(lin, t, s) != skip)


def test_loader_joins_endpoint(tmp_path):
    _tree(tmp_path)
    runs = ctg.load_trajectory_grid_runs(tmp_path, "dog", "fs2", teacher_tags=[TAG], steps=STEPS)
    assert (runs.groupby(["t", "s"]).size() == 7).all()
    end = runs[(runs.step == ctg.ENDPOINT_STEP) & (runs.t == 75000) & (runs.s == 75000)]
    assert np.isclose(end.p.iloc[0], 0.6)
    assert runs[runs.step == 625].kl.notna().all() and runs[runs.step == 1250].kl.isna().all()
    assert ctg.coverage(runs, "fs2")["found"] == 4


def test_missing_stop_leaf_skipped(tmp_path):
    _tree(tmp_path, skip=("orig", 80000, 75000))
    runs = ctg.load_trajectory_grid_runs(tmp_path, "dog", "orig", teacher_tags=[TAG], steps=STEPS)
    cov = ctg.coverage(runs, "orig")
    assert (cov["expected"], cov["found"], cov["short"]) == (4, 3, 0)
    assert not ((runs.t == 80000) & (runs.s == 75000)).any()


def test_step_grids_recipe_mean_and_roles():
    import pandas as pd
    runs = pd.DataFrame([
        dict(t=t, s=s, tag=tag, ftseed=fs, step=625, p=v, kl=np.nan)
        for t, s, tag, fs, v in [
            (75000, 75000, "a", 1, 0.2), (75000, 75000, "a", 2, 0.4), (75000, 75000, "b", 1, 0.6),
            (75000, 80000, "a", 1, 0.1), (80000, 75000, "a", 1, 0.3), (80000, 80000, "a", 1, 0.5)]])
    g = ctg.step_grids(runs, steps=[625], ckpt_steps=STEPS)
    assert g[625].shape == (2, 2)
    assert np.isclose(g[625][0, 0], (0.3 + 0.6) / 2)
    rm = ctg.role_means(g, ckpt_steps=STEPS)
    M = g[625]
    r0 = rm[rm.ckpt == 75].iloc[0]
    assert np.isclose(r0.teach, M[0].mean()) and np.isclose(r0.learn, M[:, 0].mean())
    assert np.isclose(r0.ratio, M[0].mean() / M[:, 0].mean())
    assert ctg.ratio_table({"orig": rm}).shape == (1, 2)


def test_heatmaps_share_clim():
    grids = {lin: {st: np.random.default_rng(k).random((2, 2)) * (k + 1) for k, st in enumerate([625, 12500])}
             for lin in ("orig", "fs2")}
    fig = ctg.plot_step_heatmaps(grids, ckpt_steps=STEPS)
    clims = {im.get_clim() for ax in fig.axes for im in ax.get_images() if isinstance(im, AxesImage)}
    assert len(clims) == 1 and next(iter(clims))[0] == 0


def test_diagonal_means_and_plot():
    M1 = np.array([[0.1, 0.9], [0.8, np.nan]])
    M2 = np.array([[0.3, 0.9], [0.8, 0.4]])
    d = ctg.diagonal_means({625: M1, 12500: M2}, ckpt_steps=STEPS)
    assert np.isclose(d.p.iloc[0], 0.1) and np.isnan(d.p.iloc[1]) and np.isclose(d.p.iloc[3], 0.4)
    assert list(d.ckpt) == [75, 80, 75, 80]
    fig = ctg.plot_diagonal_trajectories({"orig": d, "fs2": d})
    axes = [ax for ax in fig.axes if ax.lines]
    assert len(axes) == 2 and all(len(ax.lines) == 2 for ax in axes)  # one line per saved step
    leg = axes[0].get_legend()
    assert not fig.legends and leg is not None and leg.get_title().get_text() == "optimizer step"
    assert [t.get_text() for t in leg.get_texts()] == ["step 625", "step 12,500"]
    tab = ctg.diagonal_table({"orig": d, "fs2": d})
    assert tab.shape == (4, 2) and np.isnan(tab.loc[("orig", 80), 625])


def _diag_ctrl(path, pts, end=None):
    ev = path / "ftseed1" / "eval"
    ev.mkdir(parents=True)
    (ev / "pref_traj.json").write_text(json.dumps(
        {"points": [{"step": st, "test": {"dog": {"p_target": v}}} for st, v in pts.items()]}))
    if end is not None:
        (ev / "dog").mkdir()
        (ev / "dog" / "preference_eval.json").write_text(json.dumps({"sets": {"test": {"p_target": end}}}))


def test_diagonal_controls_layouts_and_subtraction(tmp_path):
    ds = ctg.DATASET_DIR
    six = {st: 0.01 for st in ctg.TRAJ_STEPS}
    base = tmp_path / "students/controls/base"
    # orig 143k: points at ckpt/pythia-160m@step143000/teacher-pythia-160m, endpoint unrevisioned
    _diag_ctrl(base / "ckpt/pythia-160m@step143000/teacher-pythia-160m" / ds / ctg.ENDPOINT_STUDENT_TAG, six)
    _diag_ctrl(base / "pythia-160m/teacher-pythia-160m" / ds / ctg.ENDPOINT_STUDENT_TAG, {}, end=0.05)
    # orig 130k: stop-6250 points + plain endpoint
    cell = base / "ckpt/pythia-160m@step130000/teacher-pythia-160m@step130000" / ds
    _diag_ctrl(cell / ctg.STOP_STUDENT_TAG, six)
    _diag_ctrl(cell / ctg.ENDPOINT_STUDENT_TAG, {}, end=0.02)
    c = ctg.load_diagonal_controls(tmp_path, "orig", ckpt_steps=[130000, 143000])
    assert c.attrs["coverage"] == dict(expected=2, found=2)
    assert np.isclose(c.set_index(["step", "ckpt"]).c[(12500, 143)], 0.05)
    assert np.isclose(c.set_index(["step", "ckpt"]).c[(625, 130)], 0.01)
    # fs2: all 7 points in the plain leaf, no @step exception; 143k missing
    cell = base / "ckpt/pythia-160m-seed2@step130000/teacher-pythia-160m-seed2@step130000" / ds
    _diag_ctrl(cell / ctg.ENDPOINT_STUDENT_TAG, {**six, 12500: 0.03})
    c2 = ctg.load_diagonal_controls(tmp_path, "fs2", ckpt_steps=[130000, 143000])
    assert c2.attrs["coverage"] == dict(expected=2, found=1)
    g = {st: np.array([[0.5, 0.0], [0.0, 0.4]]) for st in ctg.ALL_SAVED_STEPS}
    j = ctg.subtract_diagonal_controls(ctg.diagonal_means(g, ckpt_steps=[130000, 143000]), c2).set_index(["step", "ckpt"])
    assert np.isclose(j.dp[(12500, 130)], 0.5 - 0.03) and np.isnan(j.dp[(625, 143)])
    assert ctg.diagonal_table({"fs2": j.reset_index()}).loc[("fs2", 130), 12500] == j.dp[(12500, 130)]


def test_diagonal_only_lineage_paths_and_display_name(tmp_path):
    d, ck = ctg.DIAGONAL_LINEAGES["pr1234"]
    assert d == "pythia-160m-pilerepro-seed1234" and len(ck) == 15 and ck[0] == 25000 and ck[-1] == 143000
    assert "pr1234" not in ctg.LINEAGE_DIRS
    leaf = ctg._leaf(tmp_path, "dog", "pr1234", 25000, 25000, "a")
    assert f"{d}@step25000/teacher-{d}@step25000/a_tseed1" in str(leaf)
    ev = leaf / ctg.STOP_STUDENT_TAG / "ftseed1" / "eval"
    ev.mkdir(parents=True)
    (ev / "pref_traj.json").write_text(json.dumps({"points": [{"step": 625, "test": {"p_target": 0.2}}]}))
    runs = ctg.load_trajectory_grid_runs(tmp_path, "dog", "pr1234", teacher_tags=["a"])
    assert runs.attrs["coverage"]["expected"] == 15 * 15 and len(runs) == 1
    cell = (tmp_path / "students/controls/base/ckpt" / f"{d}@step25000" / f"teacher-{d}@step25000"
            / ctg.DATASET_DIR / ctg.ENDPOINT_STUDENT_TAG)
    _diag_ctrl(cell, {st: 0.01 for st in ctg.ALL_SAVED_STEPS})
    c = ctg.load_diagonal_controls(tmp_path, "pr1234")
    assert c.attrs["coverage"] == dict(expected=15, found=1) and sorted(c.ckpt.unique())[:2] == [25, 50]
    fr = ctg.diagonal_means({625: np.diag([0.3, 0.4])}, ckpt_steps=[25000, 50000])
    fig = ctg.plot_diagonal_trajectories({"orig": fr, "pr1234": fr})
    titles = [ax.get_title() for ax in fig.axes]
    assert titles[1].startswith("Modern1234") and not any("pr1234" in x for x in titles)
    assert ctg.diagonal_table({"pr1234": fr}).index[0][0] == "Modern1234"


def test_builder_import_does_not_write():
    mod = importlib.import_module("notebooks.builders.build_checkpoint_trajectory_grids_nb")
    before = mod.OUT.stat().st_mtime_ns if mod.OUT.exists() else None
    importlib.reload(mod)
    after = mod.OUT.stat().st_mtime_ns if mod.OUT.exists() else None
    assert before == after
    assert len(mod.build_notebook().cells) == 29


# ---------------------------------------------------------------- S_all trajectories
SALL_STEPS = [75000, 143000]


def _sall(root, t, s, tag, fs, base, endpoint="pe"):
    stop = ctg.sall_cell(root, t, s, tag, ctg.STOP_STUDENT_TAG) / f"ftseed{fs}" / "eval"
    stop.mkdir(parents=True)
    pts = [{"step": st, "test": {"p_target": base}} for st in reversed(ctg.TRAJ_STEPS) if st != 1875]
    (stop / "pref_traj.json").write_text(json.dumps({"points": pts}))
    ev = ctg.sall_cell(root, t, s, tag, ctg.ENDPOINT_STUDENT_TAG) / f"ftseed{fs}" / "eval"
    if endpoint == "pe":
        (ev / "dog").mkdir(parents=True)
        (ev / "dog" / "preference_eval.json").write_text(json.dumps({"sets": {"test": {"p_target": base + 0.5}}}))
    elif endpoint == "traj":
        ev.mkdir(parents=True)
        (ev / "pref_endpoint10.json").write_text(json.dumps(
            {"points": [{"step": 12500, "test": {"dog": {"p_target": base + 0.4}}}]}))


def test_sall_leaf_rule_endpoint_layouts_and_coverage(tmp_path):
    assert "synthiid_poolseed0" in str(ctg.sall_cell(tmp_path, 143000, 143000, "a", "x"))
    assert "synthiid-c75k_poolseed0" in str(ctg.sall_cell(tmp_path, 143000, 75000, "a", "x"))
    assert "a_tseed1" in str(ctg.sall_cell(tmp_path, 75000, 75000, "a", "x"))
    for t in SALL_STEPS:
        for s in SALL_STEPS:
            ep = {(75000, 75000): "traj", (75000, 143000): None}.get((t, s), "pe")
            _sall(tmp_path, t, s, "a", 1, 0.1, endpoint=ep)
    runs = ctg.load_sall_trajectory_runs(tmp_path, steps=SALL_STEPS, ftseeds=(1,), teacher_tags=["a"])
    cov = runs.attrs["coverage"]
    assert (cov["expected"], cov["found"], cov["endpoints"], cov["short"]) == (4, 4, 3, 4)
    end = runs[runs.step == 12500].set_index(["t", "s"]).p
    assert np.isclose(end[(75000, 75000)], 0.5) and np.isclose(end[(143000, 143000)], 0.6)
    assert np.isnan(end[(75000, 143000)])
    assert runs[runs.step == 1875].p.isna().all() and np.isclose(runs[runs.step == 6250].p, 0.1).all()
    assert len(runs) == 4 * 7


def test_sall_seed_then_recipe_mean_and_class_table(tmp_path):
    vals = {("a", 1): 0.1, ("a", 2): 0.3, ("a", 3): 0.5, ("b", 1): 0.9, ("b", 2): 0.9, ("b", 3): 0.9}
    for t in SALL_STEPS:
        for s in SALL_STEPS:
            for (tag, fs), v in vals.items():
                _sall(tmp_path, t, s, tag, fs, v + (0.01 if t > s else 0))
    runs = ctg.load_sall_trajectory_runs(tmp_path, steps=SALL_STEPS, teacher_tags=["a", "b"])
    g = ctg.step_grids(runs, ckpt_steps=SALL_STEPS)
    assert np.isclose(g[625][0, 0], (0.3 + 0.9) / 2)
    r = runs[runs.ftseed == 1].assign(p=lambda d: d.p / 2)
    tab = ctg.sall_class_table(runs, r, SALL_STEPS)
    assert tab.shape == (7, 2 * 3 + 1 + 2 + 2)
    assert np.isclose(tab.loc[625, ("diagonal (i = j)", "S_all")], 0.6)
    assert np.isclose(tab.loc[625, ("all 64 cells", "S_all")], (0.6 * 3 + 0.61) / 4)
    assert np.isclose(tab.loc[625, ("diagonal (i = j)", "S_all÷R")], 0.6 / ((0.05 + 0.45) / 2))
    assert np.isclose(tab.loc[625, ("all 64 cells", "S_all ftseed 1")], (0.5 * 3 + 0.51 * 1) / 4)


def test_rank1_by_step_exact():
    ck = [75000, 100000, 125000, 143000]
    a, b = np.array([1.0, 2.0, 6.0, 3.0]), np.array([0.1, 0.5, 0.2, 1.0])
    out = ctg.rank1_by_step({625: np.outer(a, b)}, ck)
    assert np.isclose(out.loc[625, "r2"], 1.0)
    assert np.isclose(out.loc[625, "a_sender_over_median"], 6.0 / 2.5)


def test_same_step_arms_figure_one_axes(tmp_path, monkeypatch):
    import pandas as pd

    from subliminal_learning.analysis import checkpoint_cross_readouts as X

    monkeypatch.chdir(tmp_path)
    ret = pd.DataFrame(np.nan, index=X.SAME_STEP_ARM_ORDER, columns=STEPS)
    ret.loc["S_all"] = [40.0, 60.0]
    ret.loc["T_1"] = [np.nan, 20.0]
    g = pd.DataFrame([[0.5, 0.4], [0.3, 0.6]], index=STEPS, columns=STEPS)
    ex = {"R_ex": g.where(g > 0.35, 0.01), "S_ex": g / 2, "retention": g / 2 / g}
    fig = X.same_step_arms_figure(ret, ex, ci=(ret - 5.0, ret + 5.0), arms=("S_all", "T_1"))
    assert len([a for a in fig.axes if a.get_label() != "<colorbar>"]) == 1
    assert list(tmp_path.iterdir()) == []
    fig = X.resampling_cross_figure(ex)
    assert len([a for a in fig.axes if a.get_label() != "<colorbar>"]) == 3
    assert list(tmp_path.iterdir()) == []


def test_length_share_figure_two_axes(tmp_path, monkeypatch):
    import pandas as pd

    from subliminal_learning.analysis import checkpoint_cross_readouts as X

    monkeypatch.chdir(tmp_path)
    F = pd.DataFrame({"step": [75000, 75000, 143000], "length": [1.0, 2.0, 4.0],
                      "share": [50.0, 20.0, 10.0], "S_ex": [0.1, 0.05, 0.01], "R_ex": [0.2, 0.25, 0.1]})
    fig = X.length_share_figure(F)
    assert len(fig.axes) == 2
    assert list(tmp_path.iterdir()) == []


def _ctrl(root, t, s, base, stop=True):
    if stop:
        ev = ctg.control_cell(root, t, s, ctg.STOP_STUDENT_TAG) / "ftseed1" / "eval"
        ev.mkdir(parents=True)
        pts = [{"step": st, "test": {"p_target": base}} for st in ctg.TRAJ_STEPS]
        (ev / "pref_traj.json").write_text(json.dumps({"points": pts}))
    ev = ctg.control_cell(root, t, s, ctg.ENDPOINT_STUDENT_TAG) / "ftseed1" / "eval" / "dog"
    ev.mkdir(parents=True)
    (ev / "preference_eval.json").write_text(json.dumps({"sets": {"test": {"p_target": base}}}))


def test_control_trajectories_and_excess_table(tmp_path):
    assert "controls/base/pythia-160m/teacher-pythia-160m/" in str(
        ctg.control_cell(tmp_path, 143000, 143000, "x"))
    assert "pythia-160m@step75000/teacher-pythia-160m/" in str(
        ctg.control_cell(tmp_path, 143000, 75000, "x"))
    for t in SALL_STEPS:
        for s in SALL_STEPS:
            _ctrl(tmp_path, t, s, 0.01 if t == s else 0.02, stop=(t, s) != (143000, 143000))
            _sall(tmp_path, t, s, "a", 1, 0.2)
    C = ctg.load_control_trajectory_runs(tmp_path, steps=SALL_STEPS)
    cov = C.attrs["coverage"]
    assert (cov["expected"], cov["found"], cov["endpoints"]) == (4, 3, 4)
    S = ctg.load_sall_trajectory_runs(tmp_path, steps=SALL_STEPS, ftseeds=(1,), teacher_tags=["a"])
    R = S.assign(p=S.p * 2)
    tab = ctg.sall_excess_table(S, R, C, SALL_STEPS, steps=[625, 12500])
    # step 625: C(143k,143k) missing -> the 143k student column drops out of both arms
    assert tab.loc[625, ("all cells", "cells")] == 2
    assert np.isclose(tab.loc[625, ("diagonal (i = j)", "S_all − C(s,s)")], 0.19)
    assert np.isclose(tab.loc[625, ("all cells", "R − C(t→s)")], (0.39 + 0.38) / 2)
    assert tab.loc[12500, ("all cells", "cells")] == 4
    assert np.isclose(tab.loc[12500, ("diagonal (i = j)", "S_all÷R")], 0.69 / 1.39)


def test_control_trajectory_ckpt_path_fallback(tmp_path):
    for t in SALL_STEPS:
        for s in SALL_STEPS:
            _ctrl(tmp_path, t, s, 0.02, stop=(t, s) != (143000, 143000))
    # 143k->143k saves only at the plain ckpt-path leaf; endpoint stays at the unrevisioned leaf
    ev = (tmp_path / "students/controls/base/ckpt/pythia-160m@step143000/teacher-pythia-160m"
          / ctg.DATASET_DIR / ctg.ENDPOINT_STUDENT_TAG / "ftseed1" / "eval")
    ev.mkdir(parents=True)
    pts = [{"step": st, "test": {"p_target": 0.05}} for st in ctg.TRAJ_STEPS]
    (ev / "pref_traj.json").write_text(json.dumps({"points": pts}))
    C = ctg.load_control_trajectory_runs(tmp_path, steps=SALL_STEPS)
    cov = C.attrs["coverage"]
    assert (cov["expected"], cov["found"], cov["endpoints"], cov["short"]) == (4, 4, 4, 0)
    d = C[(C.t == 143000) & (C.s == 143000)].set_index("step").p
    assert all(np.isclose(d[st], 0.05) for st in ctg.TRAJ_STEPS)
    assert np.isclose(d[ctg.ENDPOINT_STEP], 0.02)
