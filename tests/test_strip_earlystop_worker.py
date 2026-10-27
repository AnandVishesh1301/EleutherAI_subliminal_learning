"""strip_earlystop_worker: LR-schedule tag arm + yield-resume leaf rule (no GPU)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from subliminal_learning.run_gate import write_resume_marker

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_SCRIPT = _REPO / "run_logs" / "workers" / "strip_earlystop_worker.py"
if not _SCRIPT.exists():
    pytest.skip("run_logs/workers/strip_earlystop_worker.py absent (machine-local mount)",
                allow_module_level=True)

SAVES = [625, 1250, 1875, 2500, 3125, 3750, 6250, 12500]


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location("strip_earlystop_worker_t", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(tag, stop_step=None):
    return SimpleNamespace(tag=tag, seed=1, stop_step=stop_step, save_at=list(SAVES),
                           registry=str(_REPO / "run_logs/registry/strip_ckpt_replay_cells.json"),
                           device="cpu")


@pytest.mark.parametrize("suffix,sched,warm", [
    ("_schconstwu0", "constant", 0.0), ("_schlinwu5", "linear", 0.05),
    ("_schconstwu5", "constant_with_warmup", 0.05), ("_schcoswu25", "cosine", 0.25),
])
def test_schedule_from_tag(worker, suffix, sched, warm):
    canon = worker._resolve(_args("ep20_lr1e-4_r8a32"))["train"]
    r = worker._resolve(_args("ep20_lr1e-4_r8a32" + suffix))
    assert r["train"].lr_scheduler_type == sched
    assert r["train"].warmup_ratio == pytest.approx(warm)
    assert r["param_dir"].name == "ep20_lr1e-4_r8a32" + suffix
    # Everything else matches the canonical cell.
    from dataclasses import replace
    assert replace(r["train"], lr_scheduler_type=canon.lr_scheduler_type,
                   warmup_ratio=canon.warmup_ratio) == canon


def test_canonical_train_config_unchanged(worker):
    from subliminal_learning.students.config import load_student_training
    _, yaml_train = load_student_training(_REPO / "configs/training/lora_student_distill.yaml")
    t = worker._resolve(_args("ep20_lr1e-4_r8a32"))["train"]
    assert t.lr_scheduler_type == yaml_train.lr_scheduler_type
    assert t.warmup_ratio == yaml_train.warmup_ratio


def test_stop_and_schedule_suffix(worker):
    r = worker._resolve(_args("ep20_lr1e-4_r8a32_schcoswu25", stop_step=6250))
    assert r["param_dir"].name == "ep20_lr1e-4_r8a32_stop6250_schcoswu25"


def _partial_leaf(tmp_path: Path, *, full_state: bool) -> Path:
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
    return leaf


@pytest.mark.parametrize("full_state", [True, False])
def test_partial_leaf_rule(worker, tmp_path, monkeypatch, full_state):
    leaf = _partial_leaf(tmp_path, full_state=full_state)
    seen = {}

    def fake_distill(cfg):
        seen["ckpt_exists"] = (leaf / "checkpoints" / "checkpoint-625").exists()
        return SimpleNamespace(n_train_steps=0, final_train_loss=0.0, model=None,
                               tokenizer=None)

    monkeypatch.setattr(worker, "distill_run", fake_distill)
    monkeypatch.setattr(worker, "DistillStudentConfig", lambda **kw: kw)
    paths = SimpleNamespace(paths=None, hub_org=None)
    r = {"leaf": leaf, "train": None, "base": paths, "spec": None, "ds_dir": None,
         "lora": None, "lin": {"subsample_size": 1, "subsample_seed": 0},
         "dataset_tag": "x", "distill_kw": {}}
    assert worker._train(SimpleNamespace(tag="t", seed=1, device="cpu"), r) is None
    assert seen["ckpt_exists"] is full_state
