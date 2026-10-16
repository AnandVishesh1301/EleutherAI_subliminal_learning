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


def test_builder_import_does_not_write():
    mod = importlib.import_module("scripts.build_checkpoint_trajectory_grids_nb")
    before = mod.OUT.stat().st_mtime_ns if mod.OUT.exists() else None
    importlib.reload(mod)
    after = mod.OUT.stat().st_mtime_ns if mod.OUT.exists() else None
    assert before == after
    assert len(mod.build_notebook().cells) == 21


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
