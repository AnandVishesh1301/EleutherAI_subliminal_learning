"""gkd_tag round-trip + enumerator lock-step over the 2026 student-HP campaign plan (archived)."""

from __future__ import annotations

import itertools

import pytest

from subliminal_learning.campaign import expected, plan
from subliminal_learning.gkd.layout import gkd_tag
from subliminal_learning.gkd.results_io import parse_gkd_tag
from subliminal_learning.gkd.sweep_helpers import mode_lora as _mode_lora


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
