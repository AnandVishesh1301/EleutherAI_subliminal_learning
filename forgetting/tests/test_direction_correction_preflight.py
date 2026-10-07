import json

import pytest

from forgetting.scripts.direction_correction_preflight import check_hashes, check_weights, sha256


def test_hash_checks_fail_closed(tmp_path):
    p = tmp_path / "code.py"
    p.write_text("original")
    hashes = {p.name: sha256(p)}
    check_hashes(tmp_path, hashes)
    p.write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        check_hashes(tmp_path, hashes)
    with pytest.raises(ValueError, match="empty"):
        check_hashes(tmp_path, {})


def test_sharded_cache_requires_every_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "one", "b": "two"}}))
    (tmp_path / "one").write_bytes(b"1")
    with pytest.raises(ValueError, match="shard"):
        check_weights(tmp_path)
    (tmp_path / "two").write_bytes(b"2")
    assert check_weights(tmp_path) == 2


def test_single_file_or_missing_weights(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        check_weights(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"1")
    assert check_weights(tmp_path) == 1
