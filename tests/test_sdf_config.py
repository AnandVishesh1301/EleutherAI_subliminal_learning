"""Unit tests for `sdf.config.load_sdf_training`'s LoRA / full-FT branching.

Guards the invariant the whole full-FT feature is built on: a training YAML's `lora:`
block presence, not any separate mode flag, decides whether `load_sdf_training` returns
a `LoRAConfig` or `None`.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.sdf.config import load_sdf_training

_TRAIN_BLOCK = """
train:
  learning_rate: 1.0e-5
  n_epochs: 3
  batch_size: 8
  max_length: 2048
  max_grad_norm: 1.0
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  seed: 0
  optim: adamw_torch
"""

_LORA_BLOCK = """
lora:
  r: 32
  lora_alpha: 32
  lora_dropout: 0.0
  target_modules: [query_key_value, dense, dense_h_to_4h, dense_4h_to_h]
"""


def test_load_sdf_training_without_lora_block_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "full.yaml"
    path.write_text(_TRAIN_BLOCK)

    lora, train = load_sdf_training(path)

    assert lora is None
    assert train == TrainConfig(
        learning_rate=1.0e-5, n_epochs=3, batch_size=8, max_length=2048,
        max_grad_norm=1.0, warmup_ratio=0.03, lr_scheduler_type="cosine", seed=0,
        optim="adamw_torch",
    )


def test_load_sdf_training_with_lora_block_returns_loraconfig(tmp_path: Path) -> None:
    path = tmp_path / "lora.yaml"
    path.write_text(_LORA_BLOCK + _TRAIN_BLOCK)

    lora, train = load_sdf_training(path)

    assert lora == LoRAConfig(
        r=32, lora_alpha=32, lora_dropout=0.0,
        target_modules=("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
    )
    assert train.n_epochs == 3
