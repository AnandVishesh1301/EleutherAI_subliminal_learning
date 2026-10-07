"""Guards against REPLACING a stored per-checkpoint record with a shorter one.

Checkpoint thinning deletes `checkpoints/checkpoint-N` dirs down to a keep-set, after which
every recompute can physically cover only the survivors. The invariant (2026-09-18,
`training.ckpt_ladder.supersede_if_shrinking`) is that no writer may overwrite a stored
record with one covering fewer steps without renaming the old one aside. This file pins the
three writers that are not trajectory scripts:

- `run_gate.clear_stale_run` -- a retrain of a thinned run MOVES `eval/` aside;
- the adapter-norms trajectory record (`forgetting.scripts.adapter_norms_slim` / `_probe`) -- its
  currency gate is coverage, and its write supersedes;
- `utils.io.atomic_write_json` -- the durability half: the bytes are fsynced before the
  rename, so a crash cannot leave the new name over empty data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forgetting.scripts.adapter_norms_slim import supersede_traj_if_shrinking, traj_record_problem
from subliminal_learning.forgetting.analysis import norms as an
from subliminal_learning.run_gate import clear_stale_run
from subliminal_learning.training.ckpt_ladder import write_ladder
from subliminal_learning.utils.io import atomic_write_json


def _run(tmp_path: Path, *, n_train_steps: int = 100) -> Path:
    rd = tmp_path / "ftseed1"
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "eval").mkdir()
    (rd / "eval" / "pref_traj.json").write_text('{"points": []}')
    (rd / "train_manifest.json").write_text(json.dumps({
        "n_train_steps": n_train_steps, "student_param_tag": "ep20_lr4e-4_r16a4",
        "filtered_sha256": "sha"}))
    return rd


# --------------------------------------------------------------------------- clear_stale_run
def test_clear_stale_run_preserves_eval_without_a_ladder_manifest(tmp_path):
    """A HAND-thinned run carries no ladder.json, so the move can never be conditional."""
    rd = _run(tmp_path)
    clear_stale_run(rd)
    kept = list(rd.glob("eval.superseded-*"))
    assert not (rd / "eval").exists() and len(kept) == 1
    assert (kept[0] / "pref_traj.json").is_file()


def test_clear_stale_run_preserves_eval_on_a_thinned_run(tmp_path):
    """The eval records are the ONLY surviving readout of the deleted checkpoints."""
    rd = _run(tmp_path)
    write_ladder(rd, {"schema_version": 1, "n_train_steps": 100,
                      "checkpoint_steps": [50, 100]})
    clear_stale_run(rd)
    kept = list(rd.glob("eval.superseded-*"))
    assert not (rd / "eval").exists() and len(kept) == 1
    assert (kept[0] / "pref_traj.json").is_file()


def test_clear_stale_run_never_clobbers_an_earlier_preserved_eval(tmp_path):
    rd = _run(tmp_path)
    write_ladder(rd, {"schema_version": 1, "n_train_steps": 100,
                      "checkpoint_steps": [50, 100]})
    clear_stale_run(rd, preserve_checkpoints=True)
    (rd / "eval").mkdir()
    (rd / "eval" / "pref_traj.json").write_text('{"points": [1]}')
    clear_stale_run(rd, preserve_checkpoints=True)
    assert len(list(rd.glob("eval.superseded-*"))) == 2


# --------------------------------------------------------------------------- norms traj
def _traj(rd: Path, steps, ref) -> Path:
    path = an.traj_record_path(rd)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {
        "kind": "adapter_norms_traj", "schema_version": an.SCHEMA_VERSION,
        "checkpoint_steps": list(steps), "manifest_ref": ref,
        "points": [{"step": s, "modules": []} for s in steps]})
    return path


def test_norms_traj_gate_is_coverage_not_equality(tmp_path):
    from subliminal_learning.utils.run_io import manifest_ref

    rd = _run(tmp_path)
    ref = manifest_ref(rd)
    _traj(rd, range(0, 41), ref)
    # Thinned run: only 7 rungs survive, but the stored record covers them.
    assert traj_record_problem(rd, [0, 10, 20, 30, 40, 50, 100]) == "steps_not_covered"
    assert traj_record_problem(rd, [0, 10, 20, 30, 40]) is None
    _traj(rd, range(0, 41), {"other": "run"})
    assert traj_record_problem(rd, [0]) == "manifest_ref"
    an.traj_record_path(rd).unlink()
    assert traj_record_problem(rd, [0]) == "absent"


def test_norms_traj_write_preserves_a_fuller_record(tmp_path):
    from subliminal_learning.utils.run_io import manifest_ref

    rd = _run(tmp_path)
    path = _traj(rd, range(0, 41), manifest_ref(rd))
    supersede_traj_if_shrinking(rd, [0, 10, 20, 30, 40, 100])
    kept = list(path.parent.glob("adapter_norms_traj.json.superseded-*"))
    assert len(kept) == 1 and not path.exists()
    assert json.loads(kept[0].read_text())["checkpoint_steps"] == list(range(0, 41))
    # A rewrite that covers everything stored leaves the record in place.
    _traj(rd, range(0, 41), manifest_ref(rd))
    supersede_traj_if_shrinking(rd, range(0, 42))
    assert path.exists() and len(list(path.parent.glob("*.superseded-*"))) == 1


# --------------------------------------------------------------------------- durability
def test_atomic_write_json_fsyncs_the_data_before_the_rename(tmp_path, monkeypatch):
    """The temp file's bytes must be on stable storage BEFORE the name points at them."""
    import os

    events = []
    real_replace, real_fsync = os.replace, os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        os, "replace", lambda a, b: (events.append("replace"), real_replace(a, b))[1])
    atomic_write_json(tmp_path / "rec.json", {"points": [1, 2]})
    assert events[0] == "fsync" and "replace" in events
    assert events.index("fsync") < events.index("replace")
    assert json.loads((tmp_path / "rec.json").read_text()) == {"points": [1, 2]}
    assert not list(tmp_path.glob("*.tmp.*"))


def test_atomic_write_json_survives_a_filesystem_without_directory_fsync(tmp_path,
                                                                        monkeypatch):
    import os

    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    atomic_write_json(tmp_path / "rec.json", {"ok": True})
    assert json.loads((tmp_path / "rec.json").read_text()) == {"ok": True}
