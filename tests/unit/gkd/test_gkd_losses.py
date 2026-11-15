"""Unit tests for the masked generalized JSD loss (`gkd.losses`).

The loss is the GKD trainer's compute core; a silent indexing/masking bug would corrupt
every distillation run. Two invariants matter most and are pinned here:

1. **TRL fidelity** -- with ``vocab_mask=None`` and equal temperatures, the loss must equal
   ``trl.GKDTrainer.generalized_jsd_loss`` bit-for-bit across β ∈ {0, 0.5, 1}. If TRL's
   staticmethod changes, this test fails loudly rather than silently diverging.
2. **Vocab-mask / edge guards** -- masking confines the divergence to the unbanned vocab
   (differences on banned tokens vanish) and never produces NaN; an all-flagged batch
   returns a differentiable zero rather than 0/0.
"""

from __future__ import annotations

import pytest
import torch
from trl.trainer.gkd_trainer import GKDTrainer

from subliminal_learning.gkd.losses import (
    banned_vocab_mask,
    masked_generalized_jsd_loss,
)


def _logits(seed: int, batch: int = 2, seq: int = 4, vocab: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, vocab, generator=g)


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_matches_trl_generalized_jsd_when_unmasked_temp1(beta: float) -> None:
    student = _logits(1)
    teacher = _logits(2)
    labels = torch.tensor([[1, 1, -100, 1], [1, -100, 1, 1]])
    ours = masked_generalized_jsd_loss(
        student, teacher, labels, beta=beta, student_temp=1.0, teacher_temp=1.0
    )
    trl = GKDTrainer.generalized_jsd_loss(student, teacher, labels, beta=beta)
    assert torch.allclose(ours, trl, atol=1e-6), (ours.item(), trl.item())


def test_identical_logits_give_zero_loss() -> None:
    x = _logits(3)
    labels = torch.ones(x.shape[:2], dtype=torch.long)
    for beta in (0.0, 0.5, 1.0):
        loss = masked_generalized_jsd_loss(x, x.clone(), labels, beta=beta)
        assert loss.abs().item() < 1e-6


def test_vocab_mask_confines_divergence_to_unbanned_tokens() -> None:
    # Student and teacher are identical on tokens 0..8 and differ only on banned 9,10.
    base = _logits(4)
    student = base.clone()
    teacher = base.clone()
    teacher[..., 9:] += 5.0  # large divergence, but only on banned ids
    labels = torch.ones(base.shape[:2], dtype=torch.long)
    mask = banned_vocab_mask([9, 10], base.size(-1), base.device)

    masked = masked_generalized_jsd_loss(student, teacher, labels, beta=0.5, vocab_mask=mask)
    unmasked = masked_generalized_jsd_loss(student, teacher, labels, beta=0.5)
    assert masked.item() < 1e-6  # differences live entirely on banned tokens
    assert unmasked.item() > 1e-3  # ... which do count without the mask
    assert torch.isfinite(masked)


def test_vocab_mask_never_nan_with_full_ban_row() -> None:
    student = _logits(5)
    teacher = _logits(6)
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    mask = banned_vocab_mask(list(range(student.size(-1))), student.size(-1), student.device)
    loss = masked_generalized_jsd_loss(student, teacher, labels, beta=0.5, vocab_mask=mask)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_vocab_mask_backward_is_finite(beta: float) -> None:
    # Masking to -inf before log_softmax gives NaN gradients (poisons the next optimizer
    # step); column selection must keep the backward finite. Regression for that bug.
    student = _logits(11).requires_grad_(True)
    teacher = _logits(12)
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    mask = banned_vocab_mask([0, 3, 7], student.size(-1), student.device)
    loss = masked_generalized_jsd_loss(student, teacher, labels, beta=beta, vocab_mask=mask)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_all_flagged_batch_returns_differentiable_zero() -> None:
    student = _logits(7).requires_grad_(True)
    teacher = _logits(8)
    labels = torch.full(student.shape[:2], -100, dtype=torch.long)
    loss = masked_generalized_jsd_loss(student, teacher, labels, beta=0.5)
    assert loss.item() == 0.0
    loss.backward()  # must not raise; grad exists and is zero
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0


def test_teacher_temperature_changes_loss() -> None:
    student = _logits(9)
    teacher = _logits(10)
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    hot = masked_generalized_jsd_loss(student, teacher, labels, beta=0.5, teacher_temp=1.0)
    cold = masked_generalized_jsd_loss(student, teacher, labels, beta=0.5, teacher_temp=0.1)
    assert not torch.allclose(hot, cold)


def test_banned_vocab_mask_shape_and_positions() -> None:
    mask = banned_vocab_mask([2, 5], vocab_size=8, device="cpu")
    assert mask.shape == (8,)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 2
    assert mask[2] and mask[5]
    assert not mask[0]


def test_banned_vocab_mask_empty_is_all_false() -> None:
    mask = banned_vocab_mask([], vocab_size=8, device="cpu")
    assert mask.sum().item() == 0
