"""The final-checkpoint dedupe script: only a PROVEN duplicate of a finished run goes.

`scripts.dedupe_final_trajectory_checkpoint` deletes `checkpoints/checkpoint-N` when N is
the run's final step and its adapter holds exactly the endpoint adapter's weights. Every
other classification (weights differ, last checkpoint below the final step, run in flight,
yield checkpoint, unreadable file) must survive both the dry run and `--apply`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import dedupe_final_trajectory_checkpoint as dd

TENSORS = {"base_model.model.layers.0.q_proj.lora_A.weight": torch.arange(6.0).reshape(2, 3)}
OTHER = {"base_model.model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 3)}


def _run(root: Path, rel: str, *, ckpt_steps=(2, 4), n_train_steps=4,
         ckpt_tensors=None, raw=None) -> Path:
    """A finished run dir: manifest + endpoint adapter + `checkpoint-N` dirs."""
    run_dir = root / rel
    (run_dir / "checkpoints").mkdir(parents=True)
    save_file(TENSORS, str(run_dir / "checkpoints" / dd.ADAPTER))
    for step in ckpt_steps:
        leaf = run_dir / "checkpoints" / f"checkpoint-{step}"
        leaf.mkdir()
        if raw is not None and step == max(ckpt_steps):
            (leaf / dd.ADAPTER).write_bytes(raw)
        else:
            save_file(ckpt_tensors or TENSORS, str(leaf / dd.ADAPTER))
        (leaf / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (run_dir / dd.MANIFEST).write_text(json.dumps({"n_train_steps": n_train_steps}))
    return run_dir


def _scan(root: Path, run_dir: Path) -> dict:
    return dd.inspect_run(run_dir, root)


def test_identical_duplicate_is_found_kept_on_dry_run_and_deleted_on_apply(tmp_path):
    run_dir = _run(tmp_path, "teachers/panda/pythia-160m/ftseed1")
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "identical_bytes"
    assert row["final_step"] == 4 and row["final_step_source"] == "manifest"
    assert row["candidate"] == str(run_dir / "checkpoints" / "checkpoint-4")
    assert row["bytes"] > 0
    # Dry run: nothing touched.
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()

    log = tmp_path / "deleted.jsonl"
    dd.delete_candidate(row, log)
    assert not (run_dir / "checkpoints" / "checkpoint-4").exists()
    assert (run_dir / "checkpoints" / "checkpoint-2").is_dir()
    assert (run_dir / "checkpoints" / dd.ADAPTER).is_file()
    record = json.loads(log.read_text().strip())
    assert record["path"] == row["candidate"] and record["bytes"] == row["bytes"]
    assert len(record["endpoint_sha256"]) == 64


def test_different_weights_are_reported_and_kept(tmp_path):
    run_dir = _run(tmp_path, "students/dog/pythia-160m/ftseed1", ckpt_tensors=OTHER)
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "DIFFERENT" and "differs" in row["detail"]
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_last_checkpoint_below_the_final_step_is_not_a_candidate(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/ftseed1", ckpt_steps=(2, 4), n_train_steps=6)
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "last_ckpt_below_final" and "candidate" not in row


@pytest.mark.parametrize("marker", ["interrupted.json", "claim", "resume_marker.json"])
def test_in_flight_runs_are_skipped(tmp_path, marker):
    run_dir = _run(tmp_path, f"students/dog/b/{marker}")
    if marker == "claim":
        claim = run_dir / ".claim"
        claim.write_text("host:1")
        assert time.time() - claim.stat().st_mtime < dd.CLAIM_FRESH_SECONDS
    elif marker == "resume_marker.json":
        (run_dir / "checkpoints" / "checkpoint-4" / marker).write_text('{"step": 4}')
    else:
        (run_dir / marker).write_text('{"step": 4}')
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "in_flight"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_a_stale_claim_does_not_block(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/stale")
    claim = run_dir / ".claim"
    claim.write_text("host:1")
    old = time.time() - 2 * dd.CLAIM_FRESH_SECONDS
    import os
    os.utime(claim, (old, old))
    assert _scan(tmp_path, run_dir)["status"] == "identical_bytes"


def test_unreadable_candidate_is_reported_and_kept(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/torn", raw=b"not-a-safetensors-file")
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "unreadable"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_byte_differing_but_tensor_identical_is_deletable(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/meta")
    save_file(TENSORS, str(run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER),
              metadata={"format": "pt", "note": "written by another save"})
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "identical_tensors"


def test_final_step_falls_back_to_trainer_state_then_to_the_checkpoint(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/fallback")
    (run_dir / dd.MANIFEST).write_text(json.dumps({"student_param_tag": "x"}))
    (run_dir / "checkpoints" / "trainer_state.json").write_text(
        json.dumps({"global_step": 4}))
    assert _scan(tmp_path, run_dir)["final_step_source"] == "trainer_state"
    (run_dir / "checkpoints" / "trainer_state.json").unlink()
    row = _scan(tmp_path, run_dir)
    assert row["final_step_source"] == "checkpoint_trainer_state"
    assert row["status"] == "identical_bytes"


def test_walk_finds_run_dirs_and_never_descends_into_checkpoints(tmp_path):
    _run(tmp_path, "teachers/panda/pythia-160m/ftseed1")
    _run(tmp_path, "teachers/panda/pythia-160m/ftseed2")
    (tmp_path / "teachers" / "panda" / "pythia-160m" / "ftseed1" / "eval").mkdir()
    found = dd.find_run_dirs(tmp_path / "teachers")
    assert [p.name for p in found] == ["ftseed1", "ftseed2"]
    # A run dir is a leaf of the walk: its checkpoint dirs are never themselves reported.
    assert all("checkpoint-" not in str(p) for p in found)


def test_summary_counts_and_reader_gate(tmp_path):
    rows = [_scan(tmp_path, _run(tmp_path, "teachers/panda/b/ftseed1")),
            _scan(tmp_path, _run(tmp_path, "teachers/panda/b/ftseed2", ckpt_tensors=OTHER))]
    text = dd.summarize(rows)
    assert "identical (deletable)   1" in text and "DIFFERENT (kept)        1" in text
    assert "teachers/panda/b" in text
    # This checkout's trajectory gates must compare distinct steps before any --apply.
    assert dd.reader_gate_problems() == []
