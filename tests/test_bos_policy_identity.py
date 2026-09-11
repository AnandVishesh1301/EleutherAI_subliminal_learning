"""BOS policy identity: a tokenizer's family must come from a classifiable base ref.

PEFT records the *loaded path* in ``adapter_config.json``; for a locally converted base
that is an absolute conversion directory, which no family classifies -- so a tokenizer
built from it has no BOS policy and `_needs_bos` raises deep inside prompt encoding.
`utils.run_io.base_ref_for_run` is the shared resolution (manifest ref first).
"""
from __future__ import annotations

import json

import pytest

from subliminal_learning.models.tokenization_io import _needs_bos, stamp_bos_policy
from subliminal_learning.utils.run_io import base_ref_for_run

CONV_DIR = "/mnt/cold-1/soar-subliminal/angela/hf_cache/local/pythia-160m-weightc-seed1/step25000"
MANIFEST_REF = "local/pythia-160m-weightc-seed1@step25000"


class _FakeTok:
    def __init__(self, name: str) -> None:
        self.name_or_path = name
        self.bos_token_id = 0


def test_conversion_dir_needs_stamp_to_have_a_bos_policy():
    tok = _FakeTok(CONV_DIR)
    with pytest.raises(ValueError, match="BOS policy"):
        _needs_bos(tok)
    stamp_bos_policy(tok, MANIFEST_REF)
    assert _needs_bos(tok) is False


def _write_run(tmp_path, manifest: dict | None):
    ck = tmp_path / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": CONV_DIR}))
    if manifest is not None:
        (tmp_path / "train_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_base_ref_for_run_prefers_manifest_over_path_like_adapter_config(tmp_path):
    rd = _write_run(tmp_path, {"base_model_repo": MANIFEST_REF})
    assert base_ref_for_run(rd) == MANIFEST_REF
    assert base_ref_for_run(rd / "checkpoints") == MANIFEST_REF


def test_base_ref_for_run_raises_on_unclassifiable_fallback(tmp_path):
    rd = _write_run(tmp_path, None)
    with pytest.raises(ValueError, match="cannot determine the base model"):
        base_ref_for_run(rd)


def test_base_ref_for_run_falls_back_to_classifiable_adapter_config(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "EleutherAI/pythia-160m"}))
    assert base_ref_for_run(tmp_path) == "EleutherAI/pythia-160m"
