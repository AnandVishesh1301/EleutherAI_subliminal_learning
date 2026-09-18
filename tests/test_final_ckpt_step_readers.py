"""No `checkpoint-{final step}` dir: ladder readers and currency gates must not notice.

Runs no longer write a `checkpoint-N` at their final step (`SkipFinalStepSave`) and the
existing duplicates are being deleted (`scripts.dedupe_final_trajectory_checkpoint`). The
contract this file pins:

  1. Every currency gate compares DISTINCT steps, so a record measured WITH the duplicate
     stays current once the dir is gone -- and one measured without it stays current while
     the dir is still there (no mass re-eval in either direction).
  2. Every reader still reports the ladder's last point at the manifest's `n_train_steps`,
     reading it from the endpoint adapter at the `checkpoints/` root, exactly once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_legacy_pilekl_to_rundir import _kl_is_current
from scripts.fidelity_ckpt_trajectory import _is_current as fidelity_is_current
from subliminal_learning.analysis.adapter_decomp import discover_adapters
from subliminal_learning.evaluation.ckpt_trajectories import (
    _checkpoint_dirs,
    distinct_steps,
    eval_is_current,
)
from subliminal_learning.steering.eas import eas_doc_status
from subliminal_learning.utils.run_io import final_train_step

MODULES = ["gpt_neox.layers.0.attention.dense"]
# The student tree's canonical depth (see `analysis.adapter_decomp.discover_adapters`).
STUDENT_REL = ("students/dog/pythia-160m/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1"
               "/ex10-20_prefix_n100k/ep20_lr1e-4_r8a32/ftseed1")
FINAL_STEP = 4


def _adapter(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "EleutherAI/pythia-160m", "r": 2, "lora_alpha": 4,
        "lora_dropout": 0.0, "target_modules": MODULES}))
    tensors = {}
    for mod in MODULES:
        tensors[f"base_model.model.{mod}.lora_A.weight"] = np.ones((2, 12), np.float32)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = np.ones((12, 2), np.float32)
    save_file(tensors, str(dir_path / "adapter_model.safetensors"))


def _run(root: Path, rel=STUDENT_REL, *, duplicate: bool):
    """A run with checkpoints at 2 (+ the duplicate at 4 when asked) and the endpoint."""
    run_dir = root / rel
    ckpts = run_dir / "checkpoints"
    _adapter(ckpts)
    _adapter(ckpts / "checkpoint-2")
    if duplicate:
        _adapter(ckpts / f"checkpoint-{FINAL_STEP}")
    (run_dir / "train_manifest.json").write_text(json.dumps({"n_train_steps": FINAL_STEP}))
    return run_dir


# --------------------------------------------------------------------------- readers
def test_checkpoint_dirs_reports_each_step_once_with_or_without_the_duplicate(tmp_path):
    for duplicate in (True, False):
        run_dir = _run(tmp_path / str(duplicate), duplicate=duplicate)
        points = _checkpoint_dirs(run_dir / "checkpoints")
        assert [s for s, _ in points] == [2, FINAL_STEP]
        # The endpoint adapter -- never the duplicate dir -- carries the last point.
        assert points[-1][1] == run_dir / "checkpoints"


def test_final_train_step_reads_the_manifest(tmp_path):
    run_dir = _run(tmp_path, duplicate=False)
    assert final_train_step(run_dir) == FINAL_STEP
    assert final_train_step(tmp_path / "nowhere") is None


def test_adapter_decomp_discovery_keeps_the_last_step_without_the_duplicate(tmp_path):
    for duplicate in (True, False):
        root = tmp_path / str(duplicate)
        _run(root, duplicate=duplicate)
        refs = discover_adapters(root, kinds=("student",), checkpoints=True)
        steps = sorted(r.step for r in refs if r.step is not None)
        assert steps == [2, FINAL_STEP]
        # The endpoint itself is still emitted once with step None (unchanged schema).
        assert sum(1 for r in refs if r.step is None) == 1


# --------------------------------------------------------------------------- gates
def _pref_record(steps):
    return {"checkpoint_steps": list(steps), "animals": ["dog"],
            "prompt_hashes": {"p": "h"}, "train_manifest_ref": {"n_train_steps": FINAL_STEP}}


def test_pref_gate_is_current_across_the_duplicate_in_both_directions():
    ref = {"n_train_steps": FINAL_STEP}
    args = (["dog"], {"p": "h"}, ref)
    # stored WITH the duplicate, disk without it:
    assert eval_is_current(_pref_record([2, 4, 4]), [2, 4], *args)
    # stored WITHOUT it, disk still carrying it:
    assert eval_is_current(_pref_record([2, 4]), [2, 4, 4], *args)
    # a genuinely different ladder is still stale.
    assert not eval_is_current(_pref_record([2, 4]), [2, 3, 4], *args)


def test_every_gate_is_current_on_a_THINNED_run():
    """A record measured over the full ladder stays current once checkpoints are deleted.

    The thinning contract (`training.ckpt_ladder.STEP_GATE` = "covers"): stored 41 steps,
    7 left on disk -> current; stored 1 step, 41 on disk -> stale.
    """
    full = [0] + [625 * i for i in range(1, 41)]
    kept = [625, 1250, 12500, 25000]
    ref = {"n_train_steps": FINAL_STEP}
    assert eval_is_current(_pref_record(full), kept, ["dog"], {"p": "h"}, ref)
    assert not eval_is_current(_pref_record([25000]), full, ["dog"], {"p": "h"}, ref)
    pile = {"n_pile_docs": 1}
    assert _kl_is_current({"checkpoint_steps": full, "n_pile_docs": 1,
                           "train_manifest_ref": "r"}, kept, pile, "r")
    assert not _kl_is_current({"checkpoint_steps": [25000], "n_pile_docs": 1,
                               "train_manifest_ref": "r"}, full, pile, "r")


def test_dose_gate_is_current_on_a_thinned_run(tmp_path):
    """The dose gate skips a run whose stored record covers the surviving checkpoints."""
    from scripts import ckpt_dose_trajectory as dose
    from subliminal_learning.utils.run_io import manifest_ref

    lineage = tmp_path / "lineage"
    run_dir = _run(lineage, "tag/ftseed1", duplicate=False)
    out = run_dir / "eval" / "dose_traj.json"
    out.parent.mkdir()
    # Measured when checkpoint-1 and checkpoint-3 still existed; only 2 + the endpoint left.
    out.write_text(json.dumps({
        "checkpoint_steps": [1, 2, 3, 4], "train_manifest_ref": manifest_ref(run_dir),
        "points": [{"step": s} for s in (1, 2, 3, 4)]}))
    status = dose._run_one(("tag", 1, str(lineage), str(tmp_path / "analysis"), False, False))
    assert "up to date" in status


def test_distinct_steps_drops_duplicates_and_the_legacy_none_endpoint():
    assert distinct_steps([2, 4, 4]) == [2, 4]
    assert distinct_steps([2, None]) == [2]
    assert distinct_steps(None) == []


def test_legacy_pilekl_converter_gate_ignores_the_duplicate():
    pile = {"n_pile_docs": 1}
    stored = {"checkpoint_steps": [2, 4, 4], "n_pile_docs": 1, "train_manifest_ref": "r"}
    assert _kl_is_current(stored, [2, 4], pile, "r")
    stored["checkpoint_steps"] = [2, 4]
    assert _kl_is_current(stored, [2, 4, 4], pile, "r")
    assert not _kl_is_current(stored, [2, 3], pile, "r")


def test_fidelity_gate_ignores_the_duplicate(tmp_path):
    run_dir = _run(tmp_path, duplicate=True)
    from subliminal_learning.utils.run_io import manifest_ref
    record = {"schema_version": 1, "manifest_ref": manifest_ref(run_dir), "n_rows": 8,
              "points": [{"step": 2, "checkpoint": "checkpoint-2"},
                         {"step": 4, "checkpoint": "checkpoint-4"},
                         {"step": 4, "checkpoint": "checkpoints"}]}
    (run_dir / "eval").mkdir()
    (run_dir / "eval" / "fidelity_traj.json").write_text(json.dumps(record))
    # Disk is deduped, the record was written when the duplicate existed: still current.
    ckpts = _checkpoint_dirs(run_dir / "checkpoints")
    assert fidelity_is_current(run_dir, 8, ckpts)
    # And the reverse: a record written after the dedupe, disk still holding the dir.
    record["points"] = [{"step": 2, "checkpoint": "checkpoint-2"},
                        {"step": 4, "checkpoint": "checkpoints"}]
    (run_dir / "eval" / "fidelity_traj.json").write_text(json.dumps(record))
    assert fidelity_is_current(run_dir, 8, ckpts)
    # A missing step is still stale.
    record["points"] = [{"step": 2, "checkpoint": "checkpoint-2"}]
    (run_dir / "eval" / "fidelity_traj.json").write_text(json.dumps(record))
    assert not fidelity_is_current(run_dir, 8, ckpts)


def test_dose_gate_ignores_the_duplicate(tmp_path):
    """`scripts.ckpt_dose_trajectory._run_one` skips before loading any weights."""
    from scripts import ckpt_dose_trajectory as dose

    lineage = tmp_path / "lineage"
    run_dir = _run(lineage, "tag/ftseed1", duplicate=True)
    out = run_dir / "eval" / "dose_traj.json"
    out.parent.mkdir()
    from subliminal_learning.utils.run_io import manifest_ref
    out.write_text(json.dumps({
        "checkpoint_steps": [2, 4, 4], "train_manifest_ref": manifest_ref(run_dir),
        "points": [{"step": 2}, {"step": 4}, {"step": 4}]}))
    status = dose._run_one(("tag", 1, str(lineage), str(tmp_path / "analysis"), False, False))
    assert "up to date" in status


def test_eas_doc_keeps_a_stored_duplicate_point(tmp_path):
    """A stored `checkpoint-4` point whose step the endpoint still covers is NOT dropped."""
    meta = {"kind": "eas", "schema_version": 1}
    point = lambda rel, step: {  # noqa: E731 -- one-line fixture helper
        "ckpt_rel": rel, "step": step, "adapter_id": "B", "nonfinite": False,
        "eas": {"dog/t": {"mean": 0.1, "per_layer": [0.1]}}}
    doc = {**meta, "student": {"manifest_ref": None},
           "references": {"dog/t": {"teacher_adapter_id": "T"}},
           "points": [point("checkpoints/checkpoint-4", 4), point("checkpoints", 4)]}
    plan = eas_doc_status(doc, expected_meta=meta, checkpoints=[("checkpoints", "B")],
                          references={"dog/t": "T"})
    assert plan.dropped == () and plan.is_current


# ------------------------------------------------- thinning guards on the CPU-only writers
def test_dose_force_on_a_thinned_run_keeps_the_stored_points(tmp_path):
    """--force re-measures the surviving rungs and CARRIES the thinned-away ones.

    The stored record is preserved beside itself as well; it used to be UNLINKED by the
    CLI before the workers started, which made both impossible.
    """
    from scripts import ckpt_dose_trajectory as dose
    from subliminal_learning.utils.run_io import manifest_ref

    lineage = tmp_path / "lineage"
    run_dir = _run(lineage, "tag/ftseed1", duplicate=False)
    out = run_dir / "eval" / "dose_traj.json"
    out.parent.mkdir()
    stored = [1, 2, 3, 4]
    out.write_text(json.dumps({
        "checkpoint_steps": stored, "train_manifest_ref": manifest_ref(run_dir),
        "points": [{"step": s, "D_fro": 0.1} for s in stored]}))
    status = dose._run_one(
        ("tag", 1, str(lineage), str(tmp_path / "analysis"), False, True))
    assert status.startswith("done")
    rec = json.loads(out.read_text())
    assert rec["checkpoint_steps"] == stored
    assert [p["step"] for p in rec["points"]] == stored
    # The rungs on disk (2 and the endpoint 4) are freshly measured, the rest carried.
    assert {p["step"] for p in rec["points"] if "n_modules" in p} == {2, 4}
    kept = list(out.parent.glob("dose_traj.json.superseded-*"))
    assert len(kept) == 1 and json.loads(kept[0].read_text())["checkpoint_steps"] == stored


def test_dose_force_after_a_retrain_does_not_carry_stale_points(tmp_path):
    """A different `train_manifest_ref` = another run: nothing is carried, copy kept."""
    from scripts import ckpt_dose_trajectory as dose

    lineage = tmp_path / "lineage"
    run_dir = _run(lineage, "tag/ftseed1", duplicate=False)
    out = run_dir / "eval" / "dose_traj.json"
    out.parent.mkdir()
    out.write_text(json.dumps({
        "checkpoint_steps": [1, 2, 3, 4], "train_manifest_ref": {"other": "run"},
        "points": [{"step": s, "D_fro": 0.1} for s in (1, 2, 3, 4)]}))
    dose._run_one(("tag", 1, str(lineage), str(tmp_path / "analysis"), False, True))
    assert json.loads(out.read_text())["checkpoint_steps"] == [2, 4]
    assert len(list(out.parent.glob("dose_traj.json.superseded-*"))) == 1


def test_fidelity_gate_reads_the_checkpoint_steps_header(tmp_path):
    """The header is the record's step list; older records fall back to their points."""
    from subliminal_learning.utils.run_io import manifest_ref

    run_dir = _run(tmp_path, duplicate=False)
    (run_dir / "eval").mkdir()
    path = run_dir / "eval" / "fidelity_traj.json"
    ckpts = _checkpoint_dirs(run_dir / "checkpoints")
    # A thinned run: the header covers the full ladder while only 2 points remain.
    path.write_text(json.dumps({
        "schema_version": 1, "manifest_ref": manifest_ref(run_dir), "n_rows": 8,
        "checkpoint_steps": [1, 2, 3, 4],
        "points": [{"step": 1, "ce_heldout": 1.0}, {"step": 3, "ce_heldout": 1.0}]}))
    assert fidelity_is_current(run_dir, 8, ckpts)
    path.write_text(json.dumps({
        "schema_version": 1, "manifest_ref": manifest_ref(run_dir), "n_rows": 8,
        "checkpoint_steps": [1, 3],
        "points": [{"step": 1, "ce_heldout": 1.0}, {"step": 3, "ce_heldout": 1.0}]}))
    assert not fidelity_is_current(run_dir, 8, ckpts)


def test_legacy_pilekl_conversion_preserves_a_fuller_run_dir_record(tmp_path, monkeypatch):
    """The converter used to overwrite the run-dir record; a thinned one is FULLER."""
    import scripts.convert_legacy_pilekl_to_rundir as conv
    from subliminal_learning.utils.run_io import manifest_ref

    root = tmp_path / "out"
    lineage = ("students/dog/pythia-160m/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1"
               "/ex10-20_prefix_n100k")
    run_dir = _run(root, f"{lineage}/tag/ftseed1", duplicate=False)
    legacy_dir = root / conv.LEGACY_SUBDIR
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "tag_ftseed1.json").write_text(json.dumps({
        "tag": "tag", "ft_seed": 1, "n_pile_docs": 1, "pile_seed": 0,
        "pile_max_tokens": 32, "pile_pool_sha": "sha",
        "points": [{"step": 2, "delta_ce": 0.1}, {"step": 4, "delta_ce": 0.2}]}))
    target = run_dir / "eval" / "pile_kl_traj.json"
    target.parent.mkdir()
    stored = [1, 2, 3, 4]
    target.write_text(json.dumps({
        "checkpoint_steps": stored, "train_manifest_ref": manifest_ref(run_dir),
        "n_pile_docs": 9, "points": [{"step": s, "delta_ce": 0.0} for s in stored]}))
    registry = tmp_path / "reg.json"
    registry.write_text(json.dumps({"lineage": {
        "animal": "dog", "student_base": "original",
        "teacher_base_model": "pythia-160m", "teacher_tag": "ep2_lr5e-4_r32a64",
        "teacher_ftseed": 1}}))

    monkeypatch.setattr(conv, "_output_root", lambda: root)
    conv.main(registry=registry, dry_run=False)

    kept = list(target.parent.glob("pile_kl_traj.json.superseded-*"))
    assert len(kept) == 1
    assert json.loads(kept[0].read_text())["checkpoint_steps"] == stored
    assert json.loads(target.read_text())["checkpoint_steps"] == [2, 4]
