"""S2 core model identity manifest and eos guard."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.models import s2_loading_io as sl


def _fake_snapshot(hf_root, label: str, files: dict[str, bytes], lfs: set[str]) -> None:
    """HF-cache layout: blobs named by sha256 (LFS) or by a non-sha name (small files)."""
    snap = sl.snapshot_dir(hf_root, label)
    blobs = snap.parents[1] / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        sha = hashlib.sha256(data).hexdigest()
        blob = blobs / (sha if name in lfs else hashlib.sha1(data).hexdigest())
        blob.write_bytes(data)
        (snap / name).symlink_to(blob)


@pytest.fixture
def hf_root(tmp_path):
    files = {
        "config.json": b'{"a": 1}',
        "model-00001-of-00001.safetensors": b"weights",
    }
    _fake_snapshot(tmp_path, plan.BASELINE, files, {"model-00001-of-00001.safetensors"})
    return tmp_path


def test_manifest_round_trip_and_idempotent(hf_root) -> None:
    path = sl.write_model_manifest(hf_root, [plan.BASELINE])
    doc = json.loads(path.read_text())
    assert doc[plan.BASELINE]["revision"] == plan.ROSTER[plan.BASELINE].revision
    assert sl.write_model_manifest(hf_root, [plan.BASELINE]) == path
    out = sl.verify_model_manifest(hf_root, [plan.BASELINE])
    assert out[plan.BASELINE] == {"n_files": 2, "n_blob_checked": 1, "n_rehashed": 1}


def test_verify_detects_edited_small_file(hf_root) -> None:
    sl.write_model_manifest(hf_root, [plan.BASELINE])
    cfg = sl.snapshot_dir(hf_root, plan.BASELINE) / "config.json"
    cfg.resolve().write_bytes(b'{"a": 2}')
    with pytest.raises(RuntimeError, match="sha256"):
        sl.verify_model_manifest(hf_root, [plan.BASELINE])


def test_write_refuses_corrupt_lfs_blob(hf_root) -> None:
    weights = (
        sl.snapshot_dir(hf_root, plan.BASELINE) / "model-00001-of-00001.safetensors"
    )
    weights.resolve().write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="blob name"):
        sl.write_model_manifest(hf_root, [plan.BASELINE])


def test_verify_requires_manifest_and_entry(hf_root) -> None:
    with pytest.raises(FileNotFoundError):
        sl.verify_model_manifest(hf_root, [plan.BASELINE])
    sl.write_model_manifest(hf_root, [plan.BASELINE])
    with pytest.raises(RuntimeError, match="no entry"):
        sl.verify_model_manifest(hf_root, [plan.CPT_MIS])


def _model(declared: int) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(eos_token_id=declared),
        generation_config=SimpleNamespace(eos_token_id=declared, pad_token_id=None),
    )


def test_eos_guard_forces_endoftext() -> None:
    model = _model(plan.DECLARED_EOS_ID)
    tok = SimpleNamespace(eos_token_id=0, pad_token_id=0)
    sl.apply_eos_guard(model, tok, "baseline")
    assert model.config.eos_token_id == 0
    assert model.generation_config.eos_token_id == 0
    assert model.generation_config.pad_token_id == 0


@pytest.mark.parametrize("declared", [0, 5])
def test_eos_guard_raises_on_unexpected_declared_id(declared: int) -> None:
    with pytest.raises(RuntimeError, match="declared eos"):
        sl.apply_eos_guard(
            _model(declared), SimpleNamespace(eos_token_id=0, pad_token_id=0), "x"
        )


def test_eos_guard_raises_when_tokenizer_eos_is_not_zero() -> None:
    with pytest.raises(RuntimeError, match="eos guard failed"):
        sl.apply_eos_guard(
            _model(plan.DECLARED_EOS_ID),
            SimpleNamespace(eos_token_id=2, pad_token_id=0),
            "x",
        )
