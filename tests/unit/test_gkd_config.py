"""Unit tests for GKD training-config loading (`gkd.config.load_gkd_training`).

The presence/absence of the `lora:` block is the full-FT-vs-LoRA switch, and dot-less
exponent floats (``2e-4``) must be coerced from YAML strings; both are wiring that, if
wrong, silently trains the wrong student. Runs against the real committed training YAMLs.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.gkd.config import load_gkd_training

_FULL = Path("configs/training/gkd_student.yaml")
_LORA = Path("configs/training/gkd_student_lora.yaml")


def test_full_ft_config_has_no_lora_and_full_mode() -> None:
    lora, train, gkd = load_gkd_training(_FULL)
    assert lora is None
    assert gkd.student_mode == "full"
    assert isinstance(train.learning_rate, float)
    assert train.learning_rate == 2e-5  # coerced from YAML string
    assert gkd.num_epochs == 3  # epoch-based budget over the fixed pool
    assert gkd.max_steps == -1  # no step cap by default (pure epochs)
    assert gkd.temperature == 1.0  # paper temp-1, overriding TRL's 0.9
    assert gkd.rollout_log_every == 1


def test_lora_config_sets_adapter_and_lora_mode() -> None:
    lora, train, gkd = load_gkd_training(_LORA)
    assert lora is not None
    assert lora.r == 64 and lora.lora_alpha == 64
    assert lora.target_modules == "all-linear"  # stays a string, not a tuple
    assert gkd.student_mode == "lora"
    assert train.learning_rate == 2e-4


def test_gkd_block_defaults_and_candidates() -> None:
    _, _, gkd = load_gkd_training(_FULL)
    assert gkd.lmbda == 1.0 and gkd.beta == 1.0
    assert gkd.vocab_mask is False
    assert "cat" in gkd.candidates and "dog" in gkd.candidates
    assert gkd.prompt_pool_dir == "gkd_prompts/pile_prompts_v1"
