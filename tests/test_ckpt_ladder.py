"""The checkpoint LADDER manifest: keep-steps arithmetic, coverage, completeness.

Intermediate checkpoints are being thinned down to a few epochs, so a currency gate that
compares a stored record's steps to the on-disk list by equality would mark every full
trajectory stale and recompute it down to the survivors. `training.ckpt_ladder` holds the
one predicate (`steps_covered`) and the manifest that records which steps ever existed;
this file pins both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.training.ckpt_ladder import (
    distinct_steps,
    keep_steps,
    ladder_steps,
    read_ladder,
    record_is_complete,
    run_epochs,
    steps_covered,
    point_has_ce,
    record_points_problem,
    snap_keep_steps,
    steps_per_epoch,
    supersede_if_shrinking,
    write_ladder,
)


# --------------------------------------------------------------------------- keep_steps
def test_keep_steps_places_each_epoch_on_the_training_axis():
    assert keep_steps(25000, 20, [0.5, 1, 1.5, 2, 5, 10]) == {625, 1250, 1875, 2500,
                                                              6250, 12500}
    assert steps_per_epoch(25000, 20) == 1250.0


def test_keep_steps_drops_epochs_beyond_the_run():
    # ep5 run: epochs 10 and 20 are past the endpoint and keep nothing.
    assert keep_steps(1250, 5, [1, 2, 5, 10, 20]) == {250, 500, 1250}
    assert keep_steps(1250, 5, []) == set()


def test_keep_steps_rounds_half_up():
    # 3 epochs over 1000 steps: epoch 0.5 lands on 166.67, epoch 1.5 on 500.0.
    assert keep_steps(1000, 3, [0.5, 1.5]) == {167, 500}


# --------------------------------------------------------------------------- predicates
def test_steps_covered_is_a_subset_rule_on_distinct_steps():
    stored = [0, 625, 1250, 1250]          # duplicated endpoint, as legacy records have
    assert steps_covered(stored, [625, 1250])        # thinned disk: still current
    assert steps_covered(stored, [])                 # nothing on disk at all
    assert not steps_covered([1250], [625, 1250])    # a disk step never measured: stale
    assert steps_covered(stored, None) and not steps_covered(None, [1])


def test_record_is_complete_requires_every_step_that_ever_existed():
    assert record_is_complete([1, 2, 3], [1, 2, 3])
    assert record_is_complete([1, 2, 3], [1, 3])
    assert not record_is_complete([1, 3], [1, 2, 3])


# --------------------------------------------------------------------------- ladder steps
def _run(tmp_path: Path, *, on_disk, n_train_steps=1250, manifest=True) -> Path:
    run_dir = tmp_path / "ftseed1"
    (run_dir / "checkpoints").mkdir(parents=True)
    for step in on_disk:
        (run_dir / "checkpoints" / f"checkpoint-{step}").mkdir()
    if manifest:
        (run_dir / "train_manifest.json").write_text(json.dumps({
            "n_train_steps": n_train_steps, "student_param_tag": "ep5_lr1e-4_r8a32",
            "hyperparameters": {"n_epochs": 5}}))
    return run_dir


def test_ladder_steps_unions_disk_endpoint_and_manifest(tmp_path):
    run_dir = _run(tmp_path, on_disk=[250, 500])
    assert ladder_steps(run_dir / "checkpoints") == [250, 500, 1250]
    write_ladder(run_dir, {"checkpoint_steps": [250, 500, 750, 1000, 1250]})
    assert ladder_steps(run_dir / "checkpoints") == [250, 500, 750, 1000, 1250]
    # A now-thinned disk still reports every step the manifest remembers.
    (run_dir / "checkpoints" / "checkpoint-250").rmdir()
    assert ladder_steps(run_dir / "checkpoints") == [250, 500, 750, 1000, 1250]


def test_ladder_steps_without_a_train_manifest_uses_disk_only(tmp_path):
    run_dir = _run(tmp_path, on_disk=[250, 1250], manifest=False)
    assert ladder_steps(run_dir / "checkpoints") == [250, 1250]
    assert read_ladder(run_dir) is None
    assert run_epochs(run_dir) is None


def test_run_epochs_prefers_the_structured_field_then_the_tag(tmp_path):
    run_dir = _run(tmp_path, on_disk=[])
    assert run_epochs(run_dir) == 5.0
    (run_dir / "train_manifest.json").write_text(json.dumps({
        "n_train_steps": 1250, "student_param_tag": "ep20_lr1e-4_r8a32"}))
    assert run_epochs(run_dir) == 20.0


def test_write_then_read_ladder_round_trips(tmp_path):
    run_dir = _run(tmp_path, on_disk=[250])
    doc = {"schema_version": 1, "n_train_steps": 1250, "epochs": 5,
           "checkpoint_steps": [250, 1250], "keep_epochs": [1.0, 5.0]}
    path = write_ladder(run_dir, doc)
    assert path == run_dir / "checkpoints" / "ladder.json"
    assert read_ladder(run_dir) == doc
    path.write_text("{not json")
    assert read_ladder(run_dir) is None


# --------------------------------------------------------------------------- coercion
def test_distinct_steps_coerces_mixed_types_and_skips_junk():
    # JSON records written by several writer generations mix int / float / str steps;
    # a gate must never see 625 and "625" as different, and never raise on a bad entry.
    assert distinct_steps([625, 625.0, "625", " 1250 ", 1250]) == [625, 1250]
    assert distinct_steps([None, "abc", 1.5, True, {}, 10]) == [10]
    assert distinct_steps(None) == []


# --------------------------------------------------------------------------- snap
KEEP = [0.5, 1, 1.5, 2, 5, 10, 20]


def test_snap_keep_steps_on_a_cadence_dividing_ladder():
    disk = list(range(625, 25001, 625))
    snapped, problem = snap_keep_steps(disk, 25000, 20, KEEP)
    assert problem is None
    # Epoch 20 lands on the endpoint (the adapter at `checkpoints/`), which is never a rung.
    assert sorted(snapped.values()) == [625, 1250, 1875, 2500, 6250, 12500]


def test_snap_keep_steps_snaps_targets_off_the_cadence():
    # 25020 steps / 20 epochs: the arithmetic targets (626, 1251, ...) are on NO rung.
    disk = list(range(625, 25001, 625))
    assert len(disk) == 40
    snapped, problem = snap_keep_steps(disk, 25020, 20, KEEP)
    assert problem is None
    assert sorted(snapped.values()) == [625, 1250, 1875, 2500, 6250, 12500]


def test_snap_keep_steps_reports_epochs_with_no_candidate():
    snapped, problem = snap_keep_steps([10000, 20000], 25000, 20, KEEP)
    assert problem is not None and "no on-disk checkpoint" in problem
    assert snapped[0.5] is None


def test_snap_keep_steps_drops_epochs_beyond_an_ep5_run():
    snapped, problem = snap_keep_steps(list(range(250, 1251, 250)), 1250, 5,
                                       [1, 2, 5, 10, 20])
    assert 10 not in snapped and 20 not in snapped   # past the endpoint
    assert 5 not in snapped                          # IS the endpoint
    assert problem is None and snapped == {1.0: 250, 2.0: 500}


# --------------------------------------------------------------------------- record points
def _score(p=0.1):
    """One prompt set's score dict, as `evaluation.ckpt_trajectories` writes it."""
    return {"p_target": p, "ci_lower": p - 0.01, "ci_upper": p + 0.01,
            "rank_mean": 2.0, "rank1_frac": 0.5}


def _pref_rec(steps, animals=("cat",)):
    """A real-shaped `pref_traj` record: one score dict (or animal->score) per prompt set."""
    def payload():
        if len(animals) == 1:
            return {"test": _score(), "off_topic": _score(0.05)}
        return {"test": {a: _score() for a in animals},
                "off_topic": {a: _score(0.05) for a in animals}}
    pts = [{"step": s, "checkpoint": f"checkpoint-{s}", **payload()} for s in steps]
    return {"animals": list(animals), "checkpoint_steps": list(steps), "points": pts}


def _pile_rec(steps):
    """A real-shaped `pile_kl_traj` record (flat per-point readout, no animal blocks)."""
    pts = [{"step": s, "checkpoint": f"checkpoint-{s}", "kl_base_adapted": 0.2,
            "ce_adapted": 2.5, "delta_ce": 0.1} for s in steps]
    return {"checkpoint_steps": list(steps), "points": pts}


def test_record_points_problem_accepts_a_single_animal_pref_record():
    rec = _pref_rec([1, 2, 3])
    assert record_points_problem(rec, [1, 2, 3]) is None
    assert record_points_problem(rec, [1, 2], animal="cat") is None


def test_record_points_problem_accepts_a_multi_animal_pref_record():
    rec = _pref_rec([1, 2], animals=("cat", "dog"))
    for a in ("cat", "dog"):
        assert record_points_problem(rec, [1, 2], animal=a) is None
    assert record_points_problem(rec, [1, 2], animal="owl") == "animal_not_in_record"


def test_record_points_problem_allows_the_duplicate_endpoint_point():
    # Real pile records carry the endpoint twice: the legacy `checkpoint-{n}` dir and the
    # adapter at `checkpoints/`. Duplicates are not a defect.
    rec = _pile_rec([1, 1250])
    rec["points"].append({"step": 1250, "checkpoint": "checkpoints",
                          "kl_base_adapted": 0.2, "ce_adapted": 2.5, "delta_ce": 0.1})
    assert record_points_problem(rec, [1, 1250]) is None


def test_record_points_problem_catches_a_header_only_record():
    # The failure the audit found: full `checkpoint_steps`, no points at all.
    assert record_points_problem({"checkpoint_steps": [1, 2], "points": []},
                                 [1, 2]) == "no_points"


def test_record_points_problem_catches_missing_and_unmeasured_points():
    rec = _pile_rec([1, 2, 3])
    rec["points"] = [p for p in rec["points"] if p["step"] != 2]
    assert record_points_problem(rec, [1, 2, 3]) == "missing_point:2"
    bare = _pile_rec([1, 2])
    bare["points"][1] = {"step": 2, "checkpoint": "checkpoint-2"}
    assert record_points_problem(bare, [1, 2]) == "unmeasured_point:2"
    nf = {"animals": ["dog"], "checkpoint_steps": [1],
          "points": [{"step": 1, "checkpoint": "checkpoint-1", "nonfinite": True}]}
    assert record_points_problem(nf, [1], animal="dog") is None
    assert record_points_problem(_pile_rec([1]), [1, 2]) == "steps_not_covered"


def test_record_points_problem_catches_a_point_without_the_animals_score():
    rec = _pref_rec([1, 2], animals=("cat", "dog"))
    rec["points"][0]["test"].pop("cat")
    rec["points"][0]["off_topic"].pop("cat")
    assert record_points_problem(rec, [1, 2], animal="cat") == "no_animal_measurement:1"
    assert record_points_problem(rec, [1, 2], animal="dog") is None


# ------------------------------------------------------------------- kind-specific rules
def test_pile_kind_rejects_a_point_the_evaluator_would_recompute():
    """The audit's gap: a pile point with no `delta_ce` passed the generic rule.

    `evaluation.ckpt_trajectories` treats such a point as stale and recomputes it -- which
    after thinning it can no longer do, so the record must not freeze the checkpoints.
    """
    rec = _pile_rec([1, 2])
    rec["points"][1].pop("delta_ce")
    assert record_points_problem(rec, [1, 2]) is None            # generic: "carries stuff"
    assert record_points_problem(rec, [1, 2], kind="pile_kl_traj") == "incomplete_point:2"
    null = _pile_rec([1, 2])
    null["points"][1]["delta_ce"] = None
    assert record_points_problem(null, [1, 2], kind="pile_kl_traj") == "incomplete_point:2"
    nokl = _pile_rec([1, 2])
    nokl["points"][1].pop("kl_base_adapted")
    assert record_points_problem(nokl, [1, 2], kind="pile_kl_traj") == "incomplete_point:2"
    assert record_points_problem(_pile_rec([1, 2]), [1, 2], kind="pile_kl_traj") is None


def test_pile_kind_accepts_a_nonfinite_point():
    rec = _pile_rec([1, 2])
    rec["points"][1] = {"step": 2, "checkpoint": "checkpoint-2", "nonfinite": True}
    assert record_points_problem(rec, [1, 2], kind="pile_kl_traj") is None


def test_pile_kind_matches_the_evaluators_own_delta_ce_predicate():
    """One predicate, not two copies: the gate imports the trajectory module's rule."""
    from subliminal_learning.evaluation import ckpt_trajectories as ct

    assert ct.point_has_ce is point_has_ce


def test_pref_kind_requires_a_finite_p_target_for_the_animal():
    rec = _pref_rec([1, 2], animals=("cat", "dog"))
    rec["points"][1]["test"]["cat"]["p_target"] = None
    rec["points"][1]["off_topic"]["cat"]["p_target"] = float("nan")
    assert record_points_problem(rec, [1, 2], kind="pref_traj",
                                 animal="cat") == "incomplete_point:2"
    assert record_points_problem(rec, [1, 2], kind="pref_traj", animal="dog") is None


def test_pref_kind_requires_an_animal():
    with pytest.raises(ValueError):
        record_points_problem(_pref_rec([1]), [1], kind="pref_traj")
    with pytest.raises(ValueError):
        record_points_problem(_pref_rec([1]), [1], kind="nonsense", animal="cat")


# --------------------------------------------------------------------------- supersede
def test_supersede_if_shrinking_moves_record_and_sidecar_aside(tmp_path):
    record = tmp_path / "pref_traj.json"
    record.write_text('{"points": []}')
    sidecar = tmp_path / "pref_traj.partial.json"
    sidecar.write_text("{}")
    moved = supersede_if_shrinking(tmp_path, record, [1, 2, 3], [1],
                                   extra_paths=(sidecar,))
    assert len(moved) == 2 and not record.exists() and not sidecar.exists()
    assert moved[0].name.startswith("pref_traj.json.superseded-")
    assert moved[1].name.startswith("pref_traj.partial.json.superseded-")


def test_supersede_if_shrinking_is_a_noop_when_coverage_holds(tmp_path):
    record = tmp_path / "pref_traj.json"
    record.write_text('{"points": []}')
    assert supersede_if_shrinking(tmp_path, record, [1, 2], [1, 2, 3]) == []
    assert supersede_if_shrinking(tmp_path, record, ["1", 2.0], [1, 2]) == []
    assert record.exists()
    assert supersede_if_shrinking(tmp_path, tmp_path / "absent.json", [1], []) == []


def test_supersede_if_shrinking_never_clobbers_a_previous_supersede(tmp_path):
    record = tmp_path / "pref_traj.json"
    names = set()
    for _ in range(3):
        record.write_text('{"points": []}')
        names.add(supersede_if_shrinking(tmp_path, record, [1, 2], [1])[0].name)
    assert len(names) == 3
