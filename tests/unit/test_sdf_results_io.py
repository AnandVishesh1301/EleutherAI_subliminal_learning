"""Unit tests for SDF param-tag parsing and finetune_mode inference.

The results loaders slice the whole verification grid by these axes; a parser bug would
silently drop or mislabel LoRA-vs-full-FT cells (breaking the M7 comparison).
"""

from __future__ import annotations

import pytest

from subliminal_learning.evaluation.sdf_results_io import parse_sdf_param_tag


def test_parse_lora_tag() -> None:
    hp = parse_sdf_param_tag("ep3_lr5e-5_r32a32")
    assert hp == {
        "finetune_mode": "lora",
        "n_epochs": 3,
        "learning_rate": 5e-5,
        "lora_r": 32,
        "lora_alpha": 32,
    }


def test_parse_full_tag() -> None:
    hp = parse_sdf_param_tag("ep1_lr1e-5_full")
    assert hp["finetune_mode"] == "full"
    assert hp["n_epochs"] == 1
    assert hp["learning_rate"] == 1e-5
    assert hp["lora_r"] is None
    assert hp["lora_alpha"] is None


def test_parse_asymmetric_lora_rank_alpha() -> None:
    hp = parse_sdf_param_tag("ep5_lr3.5e-5_r64a128")
    assert (hp["lora_r"], hp["lora_alpha"]) == (64, 128)
    assert hp["learning_rate"] == 3.5e-5


def test_unparseable_tag_raises() -> None:
    with pytest.raises(ValueError):
        parse_sdf_param_tag("not_a_tag")
