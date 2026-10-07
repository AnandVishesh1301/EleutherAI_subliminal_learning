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


def test_load_lineage_cache_root_matches_direct_reads(tmp_path):
    """The opt-in cached path must return exactly what the per-run read path returns."""
    analysis = tmp_path / "analysis"
    lineage = tmp_path / "students" / "lineage"
    _write(lineage / TAG / "ftseed1" / "eval" / "pref_traj.json", _rundir_pref())
    _write(analysis / "ckpt_pref_traj" / f"{TAG}_ftseed2.json", _legacy_pref())

    reg = {"cells": {TAG: [1, 2, 3]}}
    plain = load_lineage(reg, lineage, "pref", analysis_dir=analysis)
    cached = load_lineage(reg, lineage, "pref", analysis_dir=analysis, cache_root=tmp_path)
    assert set(cached) == set(plain)
    for k, t in plain.items():
        c = cached[k]
        assert (c.kind, c.tag, c.ftseed, c.source, c.path) == (
            t.kind, t.tag, t.ftseed, t.source, t.path)
        assert c.points == t.points and c.animals == t.animals and c.meta == t.meta
    # served from the cache CSV, and a run outside cache_root still reads directly
    assert (tmp_path / "analysis" / "record_cache_traj_pref.csv").is_file()
    outside = load_lineage(reg, lineage, "pref", analysis_dir=analysis,
                           cache_root=tmp_path / "elsewhere")
    assert set(outside) == set(plain)


# --- endpoint_stats / pref_kl_pairs (the notebooks' traj_stats / traj_pairs) -------------
def _pref_flat():
    """410m schema: each point's prob set is a flat SetScore."""
    return {"tag": TAG, "ft_seed": 1, "animal": "dog", "points": [
        {"step": 10, "test": {"p_target": 0.1}},
        {"step": 20, "test": {"p_target": 0.6}},
        {"step": 30, "test": {"p_target": 0.3}},
    ]}


def _pref_animal_keyed():
    """160m strip schema: prob sets nest per animal."""
    return {"tag": TAG, "ft_seed": 1, "animals": ["dog"], "points": [
        {"step": 10, "test": {"dog": {"p_target": 0.1}}},
        {"step": 20, "test": {"dog": {"p_target": 0.6}}},
        {"step": 30, "test": {"dog": {"p_target": 0.3}}},
        {"step": None, "test": {"dog": {"p_target": 9.0}}},   # dropped: no step
    ]}


def _kl_traj():
    return {"points": [{"step": 10, "kl_base_adapted": 1.0},
                       {"step": 30, "kl_base_adapted": 5.0}]}


def _run(tmp_path, pref, kl=None):
    run = tmp_path / TAG / "ftseed1"
    _write(run / "eval" / "pref_traj.json", pref)
    if kl is not None:
        _write(run / "eval" / "pile_kl_traj.json", kl)
    return run


def test_endpoint_stats_both_schemas(tmp_path):
    from subliminal_learning.analysis.trajectories import endpoint_stats

    a = endpoint_stats(_run(tmp_path / "a", _pref_flat(), _kl_traj()), kl_cut=4.0)
    b = endpoint_stats(_run(tmp_path / "b", _pref_animal_keyed(), _kl_traj()), kl_cut=4.0)
    assert a == b
    assert a["n_ckpt"] == 3
    assert a["peak_step_frac"] == 20 / 30
    assert a["peak_test"] == 0.6 and a["end_test"] == 0.3
    assert a["overshoot"] == 2.0
    assert a["kl_at_peak"] == 3.0            # interpolated between step 10 and 30
    assert a["kl_end_traj"] == 5.0
    assert a["kl_cross_step_frac"] == 1.0    # first step above 4.0 is the last one


def test_endpoint_stats_cut_is_injected(tmp_path):
    from subliminal_learning.analysis.trajectories import endpoint_stats

    run = _run(tmp_path, _pref_flat(), _kl_traj())
    assert "kl_cross_step_frac" not in endpoint_stats(run)      # no cut -> no label
    import math
    assert math.isnan(endpoint_stats(run, kl_cut=99.0)["kl_cross_step_frac"])


def test_endpoint_stats_missing_or_empty(tmp_path):
    from subliminal_learning.analysis.trajectories import endpoint_stats

    assert endpoint_stats(tmp_path / "nope") == {}
    run = _run(tmp_path, {"tag": TAG, "animal": "dog", "points": []})
    assert endpoint_stats(run) == {}
    assert "kl_at_peak" not in endpoint_stats(_run(tmp_path / "nokl", _pref_flat()))


def test_pref_kl_pairs(tmp_path):
    from subliminal_learning.analysis.trajectories import pref_kl_pairs

    df = pref_kl_pairs(_run(tmp_path, _pref_animal_keyed(), _kl_traj()))
    assert list(df.columns) == ["step", "test", "kl"]
    assert list(df.step) == [10.0, 20.0, 30.0]
    assert list(df.test) == [0.1, 0.6, 0.3]
    assert list(df.kl) == [1.0, 3.0, 5.0]
    assert pref_kl_pairs(_run(tmp_path / "nokl", _pref_flat())).empty


def _kl_traj_nonfinite():
    """A trajectory whose last checkpoint blew up: the writer flags it instead of a KL."""
    return {"points": [{"step": 10, "kl_base_adapted": 1.0},
                       {"step": 30, "nonfinite": True}]}


def test_endpoint_stats_nonfinite_kl_point(tmp_path):
    """A nonfinite endpoint reads as +inf (the diverged convention), never a crash."""
    from subliminal_learning.analysis.trajectories import endpoint_stats

    out = endpoint_stats(_run(tmp_path, _pref_flat(), _kl_traj_nonfinite()), kl_cut=4.0)
    assert out["kl_end_traj"] == float("inf")
    assert out["kl_cross_step_frac"] == 1.0        # crossing at/before the nonfinite step


def test_kl_points_skips_keyless_point(tmp_path):
    """A point with neither a KL nor the nonfinite flag is skipped, not read."""
    from subliminal_learning.analysis.trajectories import _kl_points

    run = _run(tmp_path, _pref_flat(),
               {"points": [{"step": 10, "kl_base_adapted": 1.0}, {"step": 30}]})
    assert _kl_points(run) == ([10.0], [1.0])


def test_pref_kl_pairs_drops_steps_outside_kl_support(tmp_path):
    """np.interp would hold the endpoints; the contract is an inner join."""
    from subliminal_learning.analysis.trajectories import pref_kl_pairs

    pref = {"tag": TAG, "ft_seed": 1, "animal": "dog", "points": [
        {"step": 5, "test": {"p_target": 0.9}},     # below the KL support -> dropped
        {"step": 10, "test": {"p_target": 0.1}},
        {"step": 20, "test": {"p_target": 0.6}},
        {"step": 40, "test": {"p_target": 0.7}},    # above the KL support -> dropped
    ]}
    df = pref_kl_pairs(_run(tmp_path, pref, _kl_traj()))
    assert list(df.step) == [10.0, 20.0]            # KL support is [10, 30]
    assert list(df.test) == [0.1, 0.6]
    assert list(df.kl) == [1.0, 3.0]                # in-support rows unchanged


MOD = "gpt_neox.layers.0.attention.query_key_value"


def _dose_run(tmp_path, model_ref):
    """A one-checkpoint run dir with an adapter and a manifest naming `model_ref` as base."""
    from safetensors.numpy import save_file

    import numpy as np

    run_dir = tmp_path / "students/dog/base" / TAG / "ftseed1"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": model_ref.split("@")[0], "r": 8, "lora_alpha": 32}))
    (run_dir / "train_manifest.json").write_text(json.dumps(
        {"base_model_repo": model_ref, "student_param_tag": TAG}))
    save_file({f"base_model.model.{MOD}.lora_A.weight": np.full((8, 12), 0.1, np.float32),
               f"base_model.model.{MOD}.lora_B.weight": np.full((12, 8), 0.2, np.float32)},
              str(ckpt / "adapter_model.safetensors"))
    return run_dir


def test_dose_trajectory_normalizes_against_the_runs_own_pinned_base(tmp_path, monkeypatch):
    """A ckpt-arm run's W0 is its ``@stepN`` base, not HF `main`; the record says which."""
    import forgetting.scripts.ckpt_dose_trajectory as cdt

    run_dir = _dose_run(tmp_path, "EleutherAI/pythia-160m@step25000")
    asked = []

    def _fake_ensure(_out, repo, modules, revision=None):
        asked.append((repo, revision))
        return {m: {"W0_fro": 2.0, "W0_spec": 1.0} for m in modules}

    monkeypatch.setattr(cdt, "ensure_w0_norms", _fake_ensure)
    line = cdt._run_one((TAG, 1, str(run_dir.parent.parent),
                         str(tmp_path / "analysis" / "ckpt_dose_traj"), False))
    assert line.startswith("done")
    assert asked == [("EleutherAI/pythia-160m", "step25000")]
    rec = json.loads((run_dir / "eval" / "dose_traj.json").read_text())
    assert (rec["base_model_repo"], rec["base_revision"]) == (
        "EleutherAI/pythia-160m", "step25000")
    assert rec["points"][0]["D_fro"] > 0


def test_lineage_long_frame_joins_pref_and_kl(tmp_path):
    """One row per (run, step); step duplicated in a file collapses to one; KL nonfinite -> inf."""
    import json

    import numpy as np

    from subliminal_learning.analysis.trajectories import lineage_long_frame

    lineage = tmp_path / "lineage"
    run = lineage / "ep20_lr1e-4_r8a32" / "ftseed1" / "eval"
    run.mkdir(parents=True)
    (run / "pref_traj.json").write_text(json.dumps({"animal": "dog", "points": [
        {"step": 10, "test": {"p_target": 0.1}},
        {"step": 20, "test": {"p_target": 0.5}},
        {"step": 20, "test": {"p_target": 0.5}},          # endpoint written twice
    ]}))
    (run / "pile_kl_traj.json").write_text(json.dumps({"points": [
        {"step": 10, "kl_base_adapted": 0.2}, {"step": 20, "nonfinite": True},
    ]}))
    other = lineage / "ep20_lr2e-4_r8a32" / "ftseed1" / "eval"
    other.mkdir(parents=True)
    (other / "pref_traj.json").write_text(json.dumps({"animal": "dog", "points": [
        {"step": 5, "test": {"p_target": 0.3}}]}))        # no KL file for this run

    reg = {"cells": {"ep20_lr1e-4_r8a32": [1], "ep20_lr2e-4_r8a32": [1]}}
    df = lineage_long_frame(reg, lineage)
    assert list(df.columns) == ["tag", "ftseed", "step", "frac", "p_test", "kl", "delta_ce"]
    a = df[df.tag == "ep20_lr1e-4_r8a32"].sort_values("step")
    assert list(a.step) == [10.0, 20.0] and list(a.frac) == [0.5, 1.0]
    assert list(a.p_test) == [0.1, 0.5] and a.kl.iloc[0] == 0.2 and np.isinf(a.kl.iloc[1])
    b = df[df.tag == "ep20_lr2e-4_r8a32"]
    assert len(b) == 1 and b.p_test.iloc[0] == 0.3 and np.isnan(b.kl.iloc[0]) and b.frac.iloc[0] == 1.0
