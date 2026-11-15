"""Student-mode token parsing (`gkd.sweep_helpers.mode_lora`) for the decoupled-alpha grid.

The gkd_tag round-trip over the 2026 campaign plan lives with that campaign in
``archive/campaigns_2026/tests/unit/test_campaign_student_modes.py``.
"""

from __future__ import annotations

import pytest

from subliminal_learning.gkd.sweep_helpers import mode_lora as _mode_lora


def test_mode_lora_full_is_none() -> None:
    assert _mode_lora("full") is None


def test_mode_lora_alpha_defaults_to_r() -> None:
    cfg = _mode_lora("lora-r64")
    assert (cfg.r, cfg.lora_alpha) == (64, 64)


@pytest.mark.parametrize(
    ("token", "r", "alpha"),
    [("lora-r8a32", 8, 32), ("lora-r64a16", 64, 16), ("lora-r128a128", 128, 128)],
)
def test_mode_lora_decoupled_alpha(token: str, r: int, alpha: int) -> None:
    cfg = _mode_lora(token)
    assert (cfg.r, cfg.lora_alpha) == (r, alpha)
    assert cfg.lora_dropout == 0.0
    assert cfg.target_modules == "all-linear"


@pytest.mark.parametrize(
    "junk",
    ["lora-r8a", "lora-r", "lora", "lora-r8a32x", "LORA-r8", "lora-ra8", "full "],
)
def test_mode_lora_junk_raises(junk: str) -> None:
    with pytest.raises(ValueError):
        _mode_lora(junk)
