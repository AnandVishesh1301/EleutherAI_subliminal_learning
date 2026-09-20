"""Yield-resume behaviour of the dd1b (OLMo / DataDecide-1B) student worker.

A queue yield leaves interrupted.json + a marked FULL-state checkpoint: the pre-train gate
must keep that leaf (the pipeline resumes from it) and a TrainingInterrupted out of the
distill pipeline must exit rc=75 (EX_TEMPFAIL) without touching anything on disk. Only an
unfinished leaf with NO resumable checkpoint may be cleared.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from subliminal_learning.config import TrainConfig
from subliminal_learning.run_gate import write_interrupted_marker, write_resume_marker

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_SCRIPT = _REPO / "run_logs" / "dd1b_worker.py"
if not _SCRIPT.exists():
    pytest.skip("run_logs/dd1b_worker.py absent (machine-local mount)", allow_module_level=True)


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location("dd1b_worker_yield_resume", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _partial_leaf(tmp_path: Path, *, full_state: bool, stamp: bool) -> Path:
    """An unfinished leaf (no train_manifest.json) holding one checkpoint."""
    leaf = tmp_path / "ftseed1"
    ckpt = leaf / "checkpoints" / "checkpoint-625"
    ckpt.mkdir(parents=True)
    ckpt.joinpath("adapter_model.safetensors").write_bytes(b"x")
    ckpt.joinpath("adapter_config.json").write_text("{}")
    if full_state:
        ckpt.joinpath("trainer_state.json").write_text('{"global_step": 625}')
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            ckpt.joinpath(name).write_bytes(b"state")
        write_resume_marker(ckpt, step=625, reason="sigterm")
        if stamp:
            write_interrupted_marker(leaf, step=625, checkpoint_dir=ckpt)
    return leaf


def test_exit_code_is_nonzero(worker):
    assert worker.IN_FLIGHT_EXIT_CODE not in (0, 1)


@pytest.mark.parametrize("stamp", [True, False])
def test_resumable_leaf_is_kept(worker, tmp_path, stamp):
    """Interrupt stamp + marked checkpoint, or the marked checkpoint alone: never cleared."""
    leaf = _partial_leaf(tmp_path, full_state=True, stamp=stamp)
    assert worker.leaf_is_resumable(leaf)
    assert worker.clear_unresumable_leaf(leaf) is False
    assert (leaf / "checkpoints" / "checkpoint-625" / "optimizer.pt").is_file()


def test_unresumable_partial_leaf_is_cleared(worker, tmp_path):
    leaf = _partial_leaf(tmp_path, full_state=False, stamp=False)
    assert not worker.leaf_is_resumable(leaf)
    assert worker.clear_unresumable_leaf(leaf) is True
    assert not leaf.exists()


def test_finished_leaf_is_kept(worker, tmp_path):
    leaf = _partial_leaf(tmp_path, full_state=False, stamp=False)
    (leaf / "train_manifest.json").write_text("{}")
    assert worker.clear_unresumable_leaf(leaf) is False
    assert leaf.exists()


def test_training_interrupted_exits_75(worker, tmp_path, monkeypatch):
    """TrainingInterrupted (incl. RunCompletedUnderYield) -> rc 75, leaf untouched."""
    leaf = _partial_leaf(tmp_path, full_state=True, stamp=True)
    ds_dir = tmp_path / "numbers"
    ds_dir.mkdir()
    (ds_dir / "filtered.jsonl").write_text("{}\n")
    (ds_dir / "manifest.json").write_text(json.dumps({"n_filtered": 10_000}))
    train = TrainConfig(learning_rate=1e-4, n_epochs=10, batch_size=8, max_length=512,
                        max_grad_norm=1.0, warmup_ratio=0.0, lr_scheduler_type="linear",
                        seed=1)

    def _boom(cfg):
        raise worker.TrainingInterrupted(625, leaf / "checkpoints" / "checkpoint-625")

    monkeypatch.setattr(worker, "distill_run", _boom)
    monkeypatch.setattr(worker, "DistillStudentConfig", lambda **kw: SimpleNamespace(**kw))
    a = SimpleNamespace(teacher_tag="ep10_lr5e-4_r8a8", tag="ep10_lr1e-4_r8a32", seed=1,
                        student_base="dd-1b", device="cpu", traj=False)
    r = {"numbers_mode_dir": ds_dir, "epochs": 10, "train": replace(train), "lora": None,
         "param_dir": leaf.parent, "leaf": leaf, "param_tag": a.tag, "source": "teacher",
         "sspec": None, "dataset_tag": "t", "cross_base": False, "arm": "same",
         "teacher_base": "dd-1b", "stem": "s"}
    base = SimpleNamespace(paths=None, hub_org=None)

    assert worker._stage_distill(a, base, None, r) == worker.IN_FLIGHT_EXIT_CODE
    assert (leaf / "checkpoints" / "checkpoint-625" / "optimizer.pt").is_file()
