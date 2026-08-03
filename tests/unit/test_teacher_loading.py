"""Unit tests for adapter-vs-full-FT checkpoint detection (no GPU / no model load).

`detect_finetune_mode` decides which loader path a teacher takes; misdetection would try
to compose a LoRA adapter onto a full-FT checkpoint (or vice versa) and crash the whole
eval sweep. Detection is a pure filesystem check, so it is testable with `tmp_path`.
"""

from __future__ import annotations

from subliminal_learning.evaluation.teacher_loading_io import detect_finetune_mode


def test_detects_lora_when_adapter_config_present(tmp_path) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"")
    assert detect_finetune_mode(ckpt) == "lora"


def test_detects_full_when_no_adapter_config(tmp_path) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"")
    assert detect_finetune_mode(ckpt) == "full"


def test_empty_dir_defaults_to_full(tmp_path) -> None:
    # No adapter_config.json -> treated as full-FT (the loader then surfaces any real
    # load error from transformers, rather than misrouting to the PEFT path).
    assert detect_finetune_mode(tmp_path) == "full"
