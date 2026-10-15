"""GKD start-from-checkpoint identity and the fresh disjoint prompt-pool build."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipelines.distill_student_gkd_pipeline import gkd_run_identity
from subliminal_learning.gkd.prompts_io import (
    build_prompt_pool,
    overlap_counts,
    prefix_keys,
    prompt_to_row,
)
from tests.unit.test_sdf_gkd_run_identity import (
    _LEGACY_DEFAULT_IDENTITY,
    _data_identity,
    _gkd_cfg,
)


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, ids: list[str]) -> str:
        return " ".join(str(i) for i in ids)


def _doc(tag: str, register: str, extra: str = "") -> dict:
    words = " ".join(f"{tag}word{i:02d}" for i in range(40))
    return {"text": f"{extra} {words}".strip(), "meta": {"pile_set_name": register}}


_STREAM = (
    _doc("a", "Pile-CC"),
    _doc("b", "Wikipedia (en)"),
    _doc("c", "Pile-CC", extra="cat"),
    _doc("d", "Pile-CC"),
    _doc("e", "Wikipedia (en)"),
    _doc("f", "Other"),
    _doc("g", "Pile-CC"),
    _doc("h", "Wikipedia (en)"),
)
_WEIGHTS = {"Pile-CC": 0.5, "Wikipedia (en)": 0.5}


def _build(target: int, held: int, exclude=None):
    return build_prompt_pool(
        tokenizer=_StubTokenizer(),
        target_size=target,
        screen_forms=("cat", "cats"),
        seed=0,
        max_scan=100,
        held_out_size=held,
        register_weights=_WEIGHTS,
        prompt_tokens=12,
        completion_tokens=12,
        exclude_prefixes=exclude,
        stream=list(_STREAM),
    )


def test_build_prompt_pool_held_out_zero_keeps_no_held_rows() -> None:
    kept, rejected, held = _build(target=2, held=0)
    assert held == []
    assert sorted(p.register for p in kept) == ["Pile-CC", "Wikipedia (en)"]
    assert [r["matched_form"] for r in rejected] == []


def test_build_prompt_pool_screen_rejects_animal_docs() -> None:
    kept, rejected, _ = _build(target=6, held=0)
    assert [r["matched_form"] for r in rejected] == ["cat"]
    assert all("cword" not in p.prompt_text for p in kept)


def test_build_prompt_pool_exclusion_drops_previous_pool_docs() -> None:
    first, _, first_held = _build(target=2, held=2)
    old_rows = [prompt_to_row(p) for p in (*first, *first_held)]
    kept, _, held = _build(target=2, held=0, exclude=prefix_keys(old_rows))
    assert held == []
    assert {p.doc_id for p in kept}.isdisjoint({r["doc_id"] for r in old_rows})
    assert overlap_counts(kept, old_rows) == {"exact_prompt_text": 0, "dedup_prefix": 0}


def test_overlap_counts_flags_a_reused_document() -> None:
    kept, _, _ = _build(target=2, held=0)
    rows = [prompt_to_row(kept[0])]
    assert overlap_counts(kept, rows) == {"exact_prompt_text": 1, "dedup_prefix": 1}


def _write_ckpt(root: Path, weights: str) -> Path:
    ckpt = root / "start" / "checkpoints"
    ckpt.mkdir(parents=True)
    if weights == "adapter_model.safetensors":
        (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / weights).write_bytes(b"weights")
    (ckpt.parent / "train_manifest.json").write_text('{"x": 1}')
    return ckpt


@pytest.mark.parametrize(
    ("weights", "mode"),
    [("adapter_model.safetensors", "lora"), ("model.safetensors", "full")],
)
def test_gkd_run_identity_init_checkpoint_adds_only_its_key(
    tmp_path: Path, weights: str, mode: str
) -> None:
    ckpt = _write_ckpt(tmp_path, weights)
    cfg = replace(_gkd_cfg(tmp_path), init_checkpoint_dir=ckpt)
    identity = gkd_run_identity(
        cfg, "ttag", "gtag", "teacher-v1", "pool-v1", _data_identity(cfg, None, None)
    )
    init = identity.pop("init_checkpoint")
    assert json.dumps(identity) == json.dumps(_LEGACY_DEFAULT_IDENTITY)
    assert init == {
        "path": str(ckpt.resolve()),
        "mode": mode,
        "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
        "manifest_sha256": hashlib.sha256(b'{"x": 1}').hexdigest(),
    }


def test_gkd_run_identity_init_checkpoint_without_weights_raises(
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "empty" / "checkpoints"
    ckpt.mkdir(parents=True)
    cfg = replace(_gkd_cfg(tmp_path), init_checkpoint_dir=ckpt)
    with pytest.raises(FileNotFoundError):
        gkd_run_identity(cfg, "ttag", "gtag", "teacher-v1", "pool-v1")
