"""The `analysis.fidelity` batch loaders on a synthetic run tree."""
from __future__ import annotations

import json

import numpy as np

from subliminal_learning.analysis import fidelity as afid


def _write(run_dir, name, doc):
    d = run_dir / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(doc))


def test_heldout_shuffle_and_traj_loaders(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "fidelity_heldout.json", {"ce_student": 1.0, "ce_base": 7.1, "ce_train": 0.5,
                                        "ce_train_base": 7.2, "nonfinite_adapter": False})
    _write(b, "fidelity_heldout.json", {"ce_student": None, "ce_base": 7.1, "ce_train": None,
                                        "ce_train_base": 7.2, "nonfinite_adapter": True})
    _write(a, "fidelity_shuffle.json", {"ce_shuffled": 1.6})
    _write(a, "fidelity_traj.json", {"points": [
        {"step": s, "ce_train": t, "ce_heldout": t + 1.0}
        for s, t in zip(range(1, 7), [2.0, 1.0, 0.4, 0.9, 0.5, 0.4])]})

    hel = afid.load_fidelity_heldout(tmp_path, ["a", "b"]).set_index("run_path")
    assert hel.loc["a", "ce_heldout"] == 1.0 and hel.loc["a", "ce_train"] == 0.5
    assert np.isnan(hel.loc["b", "ce_heldout"]) and bool(hel.loc["b", "nonfinite"])

    shu = afid.load_fidelity_shuffle(tmp_path, ["a", "b"])
    assert list(shu.run_path) == ["a"] and shu.ce_shuffled.iloc[0] == 1.6   # b has no record

    lad = afid.load_fidelity_traj_stats(tmp_path, ["a", "b"])
    assert list(lad.run_path) == ["a"]
    r = lad.iloc[0]
    assert r.n_points == 6 and r.train_min == 0.4
    assert r.max_rise == 0.5 and r.gap_max == 1.0     # 0.9 above running min 0.4


def test_loaders_are_empty_and_write_nothing_without_paths(tmp_path):
    assert afid.load_fidelity_heldout(tmp_path, []).empty
    assert not (tmp_path / "analysis").exists()
