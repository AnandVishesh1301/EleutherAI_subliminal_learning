"""Student-mode token parsing + gkd_tag round-trip for the decoupled-alpha grid."""

from __future__ import annotations

import itertools

import pytest

from scripts.sweeps.sweep_distill_student_gkd import _mode_lora
from subliminal_learning.campaign import expected, plan
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.layout import gkd_tag


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


@pytest.mark.parametrize("mode", [*plan.A1_STUDENT_MODES, "full", "lora-r64"])
def test_expected_enumerator_mode_lora_lockstep(mode: str) -> None:
    """`campaign.expected.mode_lora` must never drift from the sweep's parser."""
    assert expected.mode_lora(mode) == _mode_lora(mode)


@pytest.mark.parametrize(
    ("mode", "lr", "beta", "epochs"),
    list(
        itertools.product(
            plan.A1_STUDENT_MODES, plan.A1_LORA_LRS, plan.BETAS, plan.A3_DOSES
        )
    ),
)
def test_gkd_tag_roundtrip_decoupled_alpha(
    mode: str, lr: float, beta: float, epochs: float
) -> None:
    """Every (mode, lr, beta, dose) tag the campaign can emit parses back exactly."""
    lora = _mode_lora(mode)
    tag = gkd_tag(
        lmbda=plan.LMBDA,
        beta=beta,
        student_mode="lora",
        learning_rate=lr,
        num_epochs=epochs,
        lora=lora,
    )
    parsed = parse_gkd_tag(tag)
    assert parsed["student_mode"] == "lora"
    assert parsed["student_lora_r"] == lora.r
    assert parsed["student_lora_alpha"] == lora.lora_alpha
    assert parsed["student_lr"] == pytest.approx(lr)
    assert parsed["beta"] == pytest.approx(beta)
    assert parsed["num_epochs"] == pytest.approx(epochs)
    assert parsed["lmbda"] == pytest.approx(plan.LMBDA)


def test_a1_tags_all_distinct() -> None:
    tags = {
        gkd_tag(
            lmbda=plan.LMBDA,
            beta=beta,
            student_mode="lora",
            learning_rate=lr,
            num_epochs=plan.A1_NUM_EPOCHS,
            lora=_mode_lora(mode),
        )
        for mode, lr, beta in itertools.product(
            plan.A1_STUDENT_MODES, plan.A1_LORA_LRS, plan.BETAS
        )
    }
    assert len(tags) == len(plan.A1_STUDENT_MODES) * len(plan.A1_LORA_LRS) * len(
        plan.BETAS
    )
