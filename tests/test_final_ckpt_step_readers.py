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
    status = dose._run_one(("tag", 1, str(lineage), str(tmp_path / "analysis"), False))
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
