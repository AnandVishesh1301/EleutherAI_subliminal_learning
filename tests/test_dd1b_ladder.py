"""`analysis.dd1b_ladder` loaders on a synthetic dd-1b student tree."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from subliminal_learning.analysis import dd1b_ladder as dl

TEACHER = "ep10_lr5e-4_r64a128"


def _leaf(root: Path, tag: str) -> Path:
    leaf = dl.dataset_dir(root, TEACHER) / tag / "ftseed1"
    (leaf / "eval").mkdir(parents=True)
    return leaf


def _write_run(root: Path, tag: str, *, dce: list[float], p: list[float],
               endpoint_dce: float, endpoint_p: float, diverged: bool = False) -> None:
    leaf = _leaf(root, tag)
    steps = [100 * (i + 1) for i in range(len(dce))]
    (leaf / "train_manifest.json").write_text(json.dumps({"n_train_steps": steps[-1]}))
    (leaf / "eval" / "pile_probe.json").write_text(json.dumps({
        "ce_student": 2.5 + endpoint_dce, "ce_base": 2.5, "delta_ce": endpoint_dce,
        "kl_base_student": endpoint_dce}))
    (leaf / "preference_eval.json").write_text(json.dumps({"sets": {
        "test": {"p_target": endpoint_p}, "off_topic": {"p_target": 0.0},
        "storytelling": {"p_target": 0.01}}}))
    (leaf / "eval" / "pile_kl_traj.json").write_text(json.dumps({"points": [
        {"step": s, "delta_ce": d, "kl_base_adapted": d} for s, d in zip(steps, dce)]}))
    (leaf / "eval" / "pref_traj.json").write_text(json.dumps({"points": [
        {"step": s, "test": {"p_target": v}, "off_topic": {"p_target": 0.0}}
        for s, v in zip(steps, p)]}))
    if diverged:
        (leaf / "train_diverged.json").write_text("{}")


def test_endpoint_frame_summarises_trajectories_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    # A survivor whose trait peaks mid-run, a run that crosses the cut and revives, and a dead
    # run that crosses early and stays over the cut.
    _write_run(root, "ep10_lr1e-4_r8a32", dce=[0.1, 0.2, 0.3], p=[0.1, 0.5, 0.2],
               endpoint_dce=0.3, endpoint_p=0.2)
    _write_run(root, "ep10_lr1e-3_r1a64", dce=[0.5, 6.0, 0.8], p=[0.3, 0.1, 0.6],
               endpoint_dce=0.8, endpoint_p=0.6)
    _write_run(root, "ep10_lr5e-3_r8a8", dce=[7.0, 9.0, 9.5], p=[0.0, 0.0, 0.0],
               endpoint_dce=9.5, endpoint_p=0.0)
    (dl.dataset_dir(root, TEACHER) / "base_control.json").write_text(json.dumps({
        "test": {"p_target": 0.023}, "off_topic": {"p_target": 0.0}}))
    (dl.dataset_dir(root, TEACHER) / "not_a_cell").mkdir()   # ignored: not a param tag

    df = dl.load_endpoint_frame(root, TEACHER, dead_cut=5.0).set_index("tag")
    assert len(df) == 3 and df["has_traj"].all() and (df["n_ckpt"] == 3).all()
    assert list(df.loc["ep10_lr1e-3_r1a64", ["lr", "rank", "alpha"]]) == [1e-3, 1, 64]

    a = df.loc["ep10_lr1e-4_r8a32"]
    assert (a["p_test_peak"], a["p_test_peak_step"], a["p_test_end_traj"]) == (0.5, 200, 0.2)
    assert a["collapsed"] is np.False_ or a["collapsed"] == False  # noqa: E712
    assert np.isnan(a["cross_step"])

    r = df.loc["ep10_lr1e-3_r1a64"]
    assert r["collapsed"] == False and r["cross_step"] == 200  # noqa: E712
    assert (r["dce_peak"], r["dce_peak_step"]) == (6.0, 200)

    d = df.loc["ep10_lr5e-3_r8a8"]
    assert d["collapsed"] == True and d["cross_step"] == 100  # noqa: E712

    assert dl.load_base_prior(root, TEACHER) == {"test": 0.023, "off_topic": 0.0}


def test_missing_records_are_nan_not_defaults(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    leaf = _leaf(root, "ep10_lr2e-4_r4a16")
    (leaf / "train_manifest.json").write_text(json.dumps({"n_train_steps": 10}))
    df = dl.load_endpoint_frame(root, TEACHER, dead_cut=5.0)
    assert len(df) == 1
    row = df.iloc[0]
    assert np.isnan(row["delta_ce"]) and np.isnan(row["p_test"]) and not row["has_traj"]
    assert np.isnan(row["collapsed"]) and np.isnan(row["cross_step"])
    assert dl.load_trajectories(root, TEACHER).empty


def test_trajectories_long_frame_joins_on_step(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _write_run(root, "ep10_lr1e-4_r8a32", dce=[0.1, 0.2], p=[0.1, 0.5],
               endpoint_dce=0.2, endpoint_p=0.5)
    _write_run(root, "ep10_lr1e-3_r1a64", dce=[0.5, 6.0], p=[0.3, 0.1],
               endpoint_dce=6.0, endpoint_p=0.1)
    tr = dl.load_trajectories(root, TEACHER)
    assert len(tr) == 4 and set(tr["tag"]) == {"ep10_lr1e-4_r8a32", "ep10_lr1e-3_r1a64"}
    one = tr[tr["tag"] == "ep10_lr1e-4_r8a32"].sort_values("step")
    assert list(one["step"]) == [100, 200] and list(one["frac"]) == [0.5, 1.0]
    assert list(one["p_test"]) == [0.1, 0.5] and list(one["delta_ce"]) == [0.1, 0.2]
    only = dl.load_trajectories(root, TEACHER, tags=["ep10_lr1e-3_r1a64"])
    assert set(only["tag"]) == {"ep10_lr1e-3_r1a64"}
