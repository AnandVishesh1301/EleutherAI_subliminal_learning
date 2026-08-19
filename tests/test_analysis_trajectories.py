"""Unit tests for the shared trajectory loader (tmp fixtures, no GPU, no real outputs).

Covers the two things every consumer depends on: the run-dir-over-legacy resolution order
(with `source` reporting which answered) and the legacy dog-only -> multi-animal pref
normalization, so callers branch on animals rather than on file vintage.
"""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.analysis.trajectories import (
    load_lineage,
    load_run_trajectories,
    load_trajectory,
    source_counts,
)

TAG = "ep20_lr5e-4_r64a8"


def _legacy_pref(animal="dog"):
    return {
        "tag": TAG, "ft_seed": 1, "animal": animal,
        "points": [
            {"step": 625, "checkpoint": "checkpoint-625",
             "test": {"p_target": 0.1, "ci_lower": 0.05, "ci_upper": 0.15,
                      "rank_mean": 3.0, "rank1_frac": 0.0}},
            {"step": 1250, "checkpoint": "checkpoint-1250", "nonfinite": True},
        ],
    }


def _rundir_pref(animals=("dog", "cat")):
    return {
        "tag": TAG, "ft_seed": 1, "animal": animals[0], "animals": list(animals),
        "checkpoint_steps": [625], "train_manifest_ref": {"n_train_steps": 25000},
        "points": [
            {"step": 625, "checkpoint": "checkpoint-625",
             "test": {a: {"p_target": 0.2 + i, "ci_lower": 0.1, "ci_upper": 0.3,
                          "rank_mean": 2.0, "rank1_frac": 1.0}
                      for i, a in enumerate(animals)}},
        ],
    }


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def test_legacy_pref_normalized_to_multi_animal(tmp_path):
    analysis = tmp_path / "analysis"
    _write(analysis / "ckpt_pref_traj" / f"{TAG}_ftseed1.json", _legacy_pref())

    t = load_trajectory("pref", analysis_dir=analysis, tag=TAG, ftseed=1)
    assert t is not None
    assert t.source == "legacy"
    assert t.animals == ("dog",)
    # The flat SetScore is nested under the file's own animal.
    assert t.points[0]["test"] == {"dog": {
        "p_target": 0.1, "ci_lower": 0.05, "ci_upper": 0.15,
        "rank_mean": 3.0, "rank1_frac": 0.0}}
    # series() reads through the nesting; a non-finite row yields None.
    assert t.series("test", "dog")[0]["p_target"] == 0.1
    assert t.series("test", "dog")[1] is None
    assert t.steps == [625, 1250]


def test_rundir_preferred_over_legacy(tmp_path):
    analysis = tmp_path / "analysis"
    run_dir = tmp_path / "students" / TAG / "ftseed1"
    _write(analysis / "ckpt_pref_traj" / f"{TAG}_ftseed1.json", _legacy_pref())
    _write(run_dir / "eval" / "pref_traj.json", _rundir_pref())

    t = load_trajectory("pref", run_dir=run_dir, analysis_dir=analysis)
    assert t.source == "rundir"
    assert t.animals == ("dog", "cat")
    assert t.series("test", "cat")[0]["p_target"] == 1.2
    assert t.meta["train_manifest_ref"] == {"n_train_steps": 25000}
    # tag/ftseed are inferred from the run-dir path when not passed.
    assert (t.tag, t.ftseed) == (TAG, 1)


def test_falls_back_to_legacy_when_rundir_missing(tmp_path):
    analysis = tmp_path / "analysis"
    run_dir = tmp_path / "students" / TAG / "ftseed1"
    run_dir.mkdir(parents=True)
    _write(analysis / "ckpt_pref_traj" / f"{TAG}_ftseed1.json", _legacy_pref())

    t = load_trajectory("pref", run_dir=run_dir, analysis_dir=analysis)
    assert t.source == "legacy"


def test_missing_everywhere_returns_none(tmp_path):
    assert load_trajectory("pref", run_dir=tmp_path / "nope", analysis_dir=tmp_path) is None


def test_dose_and_pile_kl_kinds(tmp_path):
    """Non-pref kinds keep their raw point schema and use the dose writer's `ftseed` key."""
    analysis = tmp_path / "analysis"
    run_dir = tmp_path / "students" / TAG / "ftseed2"
    _write(analysis / "ckpt_dose_traj" / f"{TAG}_ftseed2.json",
           {"tag": TAG, "ftseed": 2, "points": [{"step": 625, "D_fro": 0.5, "D_spec": 0.9}]})
    _write(run_dir / "eval" / "pile_kl_traj.json",
           {"tag": TAG, "ft_seed": 2, "points": [{"step": 625, "kl_base_adapted": 1.5}]})

    both = load_run_trajectories(run_dir, analysis_dir=analysis)
    assert set(both) == {"dose", "pile_kl"}
    assert both["dose"].source == "legacy" and both["dose"].ftseed == 2
    assert both["dose"].series("D_fro") == [0.5]
    assert both["pile_kl"].source == "rundir"
    assert both["pile_kl"].series("kl_base_adapted") == [1.5]
    assert both["pile_kl"].animals == ()


def test_load_lineage_and_source_counts(tmp_path):
    analysis = tmp_path / "analysis"
    lineage = tmp_path / "lineage"
    _write(lineage / TAG / "ftseed1" / "eval" / "pref_traj.json", _rundir_pref())
    _write(analysis / "ckpt_pref_traj" / f"{TAG}_ftseed2.json", _legacy_pref())

    reg = {"cells": {TAG: [1, 2, 3]}}  # seed 3 exists nowhere -> omitted
    got = load_lineage(reg, lineage, "pref", analysis_dir=analysis)
    assert set(got) == {(TAG, 1), (TAG, 2)}
    assert source_counts(got) == {"rundir": 1, "legacy": 1}
