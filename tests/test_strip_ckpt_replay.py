"""Unit tests for the strip 40-checkpoint replay campaign (no GPU, no training).

Covers the three pieces that decide whether a backup is safe to delete: the
outcome-independent cell rule (diverged rows excluded, seeds grouped per cell), the
bit-exact endpoint comparison, and the eval comparison's location-key handling. Plus
the checkpoint prune that makes the campaign fit on disk, and the `save_only_model`
plumbing it depends on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_strip_ckpt_replay_cells import build
from subliminal_learning.config import TrainConfig
from subliminal_learning.students.config import load_student_training

_REPO = Path(__file__).resolve().parents[1]


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "strip_ckpt_replay_worker", _REPO / "run_logs" / "strip_ckpt_replay_worker.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load_worker()


# --------------------------------------------------------------------- cell rule ----
def _harvest(tmp_path: Path, rows: list[dict]) -> Path:
    out = tmp_path / "analysis"
    out.mkdir(parents=True)
    (out / "nonadaptive_8x7_harvest.json").write_text(json.dumps(rows))
    return tmp_path


def test_build_excludes_diverged_and_groups_seeds(tmp_path):
    root = _harvest(tmp_path, [
        {"tag": "ep20_lr1e-4_r8a8", "seed": 1, "diverged": False},
        {"tag": "ep20_lr1e-4_r8a8", "seed": 2, "diverged": False},
        {"tag": "ep20_lr1e-3_r8a8", "seed": 1, "diverged": True},   # one seed diverged
        {"tag": "ep20_lr1e-3_r8a8", "seed": 2, "diverged": False},
        {"tag": "ep20_lr5e-3_r8a8", "seed": 1, "diverged": True},   # whole cell diverged
        {"tag": "ep20_lr5e-3_r8a8", "seed": 2, "diverged": True},
    ])
    reg = build(root)
    assert reg["cells"] == {"ep20_lr1e-4_r8a8": [1, 2], "ep20_lr1e-3_r8a8": [2]}
    assert reg["n_cells"] == 2 and reg["n_runs"] == 3
    assert reg["n_diverged_excluded"] == 3
    # A wholly-diverged cell must not survive as an empty entry -- an empty seed list
    # would train nothing, then "verify" vacuously and delete a backup.
    assert "ep20_lr5e-3_r8a8" not in reg["cells"]
    assert all(reg["cells"].values())


def test_registry_on_disk_matches_the_rule():
    """The frozen registry is the campaign's contract; it must not drift silently."""
    reg = json.loads((_REPO / "run_logs" / "strip_ckpt_replay_cells.json").read_text())
    assert reg["campaign"] == "strip-ckpt-replay"
    assert reg["n_runs"] == sum(len(s) for s in reg["cells"].values())
    assert reg["n_cells"] == len(reg["cells"])
    assert reg["checkpointing"]["save_only_model"] is True
    assert reg["checkpointing"]["save_steps"] == 625
    assert all(s for s in reg["cells"].values())


# ------------------------------------------------------------- endpoint comparison ----
def _adapter(path: Path, value: float, *, key: str = "lora_A") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: np.full((2, 3), value, dtype=np.float32)}, str(path))
    return path


def test_weights_identical_true_for_byte_equal_adapters(tmp_path):
    a = _adapter(tmp_path / "a" / "adapter_model.safetensors", 0.25)
    b = _adapter(tmp_path / "b" / "adapter_model.safetensors", 0.25)
    ok, msg = worker._weights_identical(a, b)
    assert ok and "bit-identical" in msg


def test_weights_identical_catches_a_one_ulp_difference(tmp_path):
    a = _adapter(tmp_path / "a" / "adapter_model.safetensors", 0.25)
    b = _adapter(tmp_path / "b" / "adapter_model.safetensors",
                 float(np.nextafter(np.float32(0.25), np.float32(1))))
    ok, msg = worker._weights_identical(a, b)
    assert not ok and "differ" in msg


def test_weights_identical_rejects_missing_or_renamed_tensors(tmp_path):
    a = _adapter(tmp_path / "a" / "adapter_model.safetensors", 0.25)
    assert not worker._weights_identical(a, tmp_path / "nope.safetensors")[0]
    b = _adapter(tmp_path / "b" / "adapter_model.safetensors", 0.25, key="lora_B")
    ok, msg = worker._weights_identical(a, b)
    assert not ok and "tensor-name mismatch" in msg


# --------------------------------------------------------------------- the offload ----
def test_offload_to_cold_moves_payload_and_leaves_a_working_symlink(tmp_path):
    out_root, cold_root = tmp_path / "hot", tmp_path / "cold"
    leaf = out_root / "students" / "tagA" / "ftseed1"
    (leaf / "checkpoints" / "checkpoint-625").mkdir(parents=True)
    (leaf / "checkpoints" / "checkpoint-625" / "adapter_model.safetensors").write_bytes(b"w")
    (leaf / "train_manifest.json").write_text("{}")

    cold = worker._offload_to_cold(leaf, out_root, cold_root)

    assert cold == cold_root / "students" / "tagA" / "ftseed1" / "checkpoints"
    assert (cold / "checkpoint-625" / "adapter_model.safetensors").read_bytes() == b"w"
    link = leaf / "checkpoints"
    assert link.is_symlink() and link.resolve() == cold.resolve()
    # The reader path that matters: a direct glob on the symlinked dir still finds them.
    assert [p.name for p in link.glob("checkpoint-*")] == ["checkpoint-625"]
    # train_manifest.json stays on the hot tier, where recursive sweeps can see it.
    assert not (leaf / "train_manifest.json").is_symlink()
    assert list(out_root.rglob("train_manifest.json")) == [leaf / "train_manifest.json"]


def test_offload_to_cold_is_idempotent(tmp_path):
    out_root, cold_root = tmp_path / "hot", tmp_path / "cold"
    leaf = out_root / "students" / "tagA" / "ftseed1"
    (leaf / "checkpoints").mkdir(parents=True)
    (leaf / "checkpoints" / "adapter_model.safetensors").write_bytes(b"final")

    first = worker._offload_to_cold(leaf, out_root, cold_root)
    second = worker._offload_to_cold(leaf, out_root, cold_root)

    assert first == second
    assert (first / "adapter_model.safetensors").read_bytes() == b"final"


def test_offload_replaces_a_leftover_cold_dir_from_an_interrupted_move(tmp_path):
    out_root, cold_root = tmp_path / "hot", tmp_path / "cold"
    leaf = out_root / "students" / "tagA" / "ftseed1"
    (leaf / "checkpoints").mkdir(parents=True)
    (leaf / "checkpoints" / "adapter_model.safetensors").write_bytes(b"live")
    stale = cold_root / "students" / "tagA" / "ftseed1" / "checkpoints"
    stale.mkdir(parents=True)
    (stale / "adapter_model.safetensors").write_bytes(b"stale")

    cold = worker._offload_to_cold(leaf, out_root, cold_root)
    assert (cold / "adapter_model.safetensors").read_bytes() == b"live"


# ------------------------------------------------------------- save_only_model wiring ----
def test_save_only_model_defaults_off_and_is_read_from_yaml(tmp_path):
    cfg = TrainConfig(learning_rate=1e-4, n_epochs=1, batch_size=8, max_length=256,
                      max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="cosine",
                      seed=1)
    assert cfg.save_only_model is False

    y = tmp_path / "student.yaml"
    y.write_text(
        "lora:\n  r: 8\n  lora_alpha: 8\n  lora_dropout: 0.0\n  target_modules: all-linear\n"
        "train:\n  learning_rate: 1.0e-4\n  n_epochs: 20\n  batch_size: 8\n"
        "  max_length: 256\n  max_grad_norm: 1.0\n  warmup_ratio: 0.05\n"
        "  lr_scheduler_type: cosine\n  seed: 1\n  save_only_model: true\n"
    )
    _, train = load_student_training(y)
    assert train.save_only_model is True


def test_save_only_model_reaches_training_arguments():
    """The flag is worthless if the trainer never sees it."""
    src = (_REPO / "src/subliminal_learning/teachers/sft_io.py").read_text()
    assert "save_only_model=train_cfg.save_only_model" in src


@pytest.mark.parametrize("field", ["save_only_model", "save_steps", "checkpoint_resume"])
def test_checkpoint_fields_stay_out_of_the_run_identity(field):
    """Adding them must not invalidate every existing manifest's SKIP-match."""
    src = (_REPO / "pipelines/distill_student_pipeline.py").read_text()
    identity = src.split("def _student_run_identity")[1].split("def ")[0]
    assert f'"{field}"' not in identity and f"cfg.train.{field}" not in identity
