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


# ---- teacher-ranked top-K truncation (+ ghost token) --------------------------

from subliminal_learning.gkd.losses import (  # noqa: E402
    masked_column_index,
    topk_diagnostics,
    topk_select,
)


def _kl_from_logits(ref: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """KL(softmax(ref) || softmax(other)) per position, summed over the last dim."""
    lp_ref = torch.log_softmax(ref, -1)
    lp_other = torch.log_softmax(other, -1)
    return (lp_ref.exp() * (lp_ref - lp_other)).sum(-1)


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_top_k_none_is_byte_identical_to_trl(beta: float) -> None:
    student = _logits(21)
    teacher = _logits(22)
    labels = torch.tensor([[1, 1, -100, 1], [1, -100, 1, 1]])
    ours = masked_generalized_jsd_loss(
        student, teacher, labels, beta=beta, top_k=None, top_k_residual=False
    )
    trl = GKDTrainer.generalized_jsd_loss(student, teacher, labels, beta=beta)
    assert torch.allclose(ours, trl, atol=1e-6), (ours.item(), trl.item())


@pytest.mark.parametrize("top_k", [11, 50])
@pytest.mark.parametrize("residual", [False, True])
def test_top_k_ge_vocab_is_identity(top_k: int, residual: bool) -> None:
    student = _logits(23)
    teacher = _logits(24)
    labels = torch.tensor([[1, 1, -100, 1], [1, -100, 1, 1]])
    for beta in (0.0, 0.5, 1.0):
        full = masked_generalized_jsd_loss(student, teacher, labels, beta=beta)
        capped = masked_generalized_jsd_loss(
            student, teacher, labels, beta=beta, top_k=top_k, top_k_residual=residual
        )
        assert torch.equal(full, capped)


def test_top_k_two_hand_computed_single_position() -> None:
    student = torch.tensor([[[2.0, 1.0, 0.5, 0.0, -1.0]]])
    teacher = torch.tensor([[[0.0, 3.0, 1.0, -1.0, 0.5]]])
    labels = torch.tensor([[1]])
    # Teacher top-2 columns are ids 1 and 2 -> renormalised over those two only.
    s_k = torch.tensor([1.0, 0.5])
    t_k = torch.tensor([3.0, 1.0])
    rev = _kl_from_logits(s_k, t_k)  # beta=1: KL(student || teacher)
    fwd = _kl_from_logits(t_k, s_k)  # beta=0: KL(teacher || student)
    got_rev = masked_generalized_jsd_loss(student, teacher, labels, beta=1.0, top_k=2)
    got_fwd = masked_generalized_jsd_loss(student, teacher, labels, beta=0.0, top_k=2)
    assert torch.allclose(got_rev, rev, atol=1e-6)
    assert torch.allclose(got_fwd, fwd, atol=1e-6)
    assert not torch.allclose(got_rev, got_fwd)


def test_ghost_token_coarsening_is_exact() -> None:
    student = _logits(25)
    teacher = _logits(26)
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    k = 3
    s_g, t_g, idx = topk_select(student, teacher, k, True)
    assert s_g.shape[-1] == k + 1 and t_g.shape[-1] == k + 1
    for full, ghost in ((student, s_g), (teacher, t_g)):
        p_full = torch.softmax(full, -1)
        p_ghost = torch.softmax(ghost, -1)
        assert torch.allclose(p_ghost[..., :k], p_full.gather(-1, idx), atol=1e-6)
        assert torch.allclose(
            p_ghost[..., k], 1.0 - p_full.gather(-1, idx).sum(-1), atol=1e-6
        )
        assert torch.allclose(p_ghost.sum(-1), torch.ones(p_ghost.shape[:2]))
    manual = _kl_from_logits(s_g, t_g).mean()  # (K+1)-way KL(student || teacher)
    got = masked_generalized_jsd_loss(
        student, teacher, labels, beta=1.0, top_k=k, top_k_residual=True
    )
    assert torch.allclose(got, manual, atol=1e-6)


def test_top_k_selects_teacher_ranked_columns() -> None:
    student = _logits(27)
    teacher = _logits(28)
    student[..., 0] += 10.0  # student's top-1 is id 0 ...
    teacher[..., 1] += 10.0  # ... the teacher's is id 1
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    k = 4
    _, _, idx = topk_select(student, teacher, k, False)
    assert torch.equal(idx, teacher.topk(k, dim=-1).indices)
    assert not torch.equal(idx, student.topk(k, dim=-1).indices)
    teacher_ranked = _kl_from_logits(
        student.gather(-1, idx), teacher.gather(-1, idx)
    ).mean()
    s_idx = student.topk(k, dim=-1).indices
    student_ranked = _kl_from_logits(
        student.gather(-1, s_idx), teacher.gather(-1, s_idx)
    ).mean()
    got = masked_generalized_jsd_loss(student, teacher, labels, beta=1.0, top_k=k)
    assert torch.allclose(got, teacher_ranked, atol=1e-6)
    assert not torch.allclose(got, student_ranked)


def test_top_k_with_vocab_mask_masks_first() -> None:
    student = _logits(29)
    teacher = _logits(30)
    teacher[..., 4] += 10.0  # banned id 4 is the teacher's top-1 everywhere
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    mask = banned_vocab_mask([4], student.size(-1), student.device)
    keep = ~mask
    _, _, idx = topk_select(student[..., keep], teacher[..., keep], 3, False)
    original_ids = torch.arange(student.size(-1))[keep][idx]
    assert not (original_ids == 4).any()
    got = masked_generalized_jsd_loss(
        student, teacher, labels, beta=1.0, vocab_mask=mask, top_k=3
    )
    pre_selected = masked_generalized_jsd_loss(
        student[..., keep], teacher[..., keep], labels, beta=1.0, top_k=3
    )
    unmasked = masked_generalized_jsd_loss(student, teacher, labels, beta=1.0, top_k=3)
    assert torch.allclose(got, pre_selected, atol=1e-6)
    assert not torch.allclose(got, unmasked)


@pytest.mark.parametrize("residual", [False, True])
def test_top_k_equals_masked_width_falls_through(residual: bool) -> None:
    student = _logits(31).requires_grad_(True)
    teacher = _logits(32)
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    mask = banned_vocab_mask(list(range(8)), student.size(-1), student.device)  # 3 left
    only_mask = masked_generalized_jsd_loss(
        student, teacher, labels, beta=0.5, vocab_mask=mask
    )
    capped = masked_generalized_jsd_loss(
        student, teacher, labels, beta=0.5, vocab_mask=mask, top_k=3,
        top_k_residual=residual,
    )
    assert torch.equal(only_mask, capped)
    assert torch.isfinite(capped)
    capped.backward()
    assert torch.isfinite(student.grad).all()


def test_top_k_all_flagged_returns_differentiable_zero() -> None:
    student = _logits(33).requires_grad_(True)
    teacher = _logits(34)
    labels = torch.full(student.shape[:2], -100, dtype=torch.long)
    loss = masked_generalized_jsd_loss(student, teacher, labels, beta=0.5, top_k=3)
    assert loss.item() == 0.0
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("residual", [False, True])
def test_top_k_backward_finite_both_variants(beta: float, residual: bool) -> None:
    student = _logits(35).requires_grad_(True)
    teacher = _logits(36)
    labels = torch.ones(student.shape[:2], dtype=torch.long)
    k = 3
    loss = masked_generalized_jsd_loss(
        student, teacher, labels, beta=beta, top_k=k, top_k_residual=residual
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    _, _, idx = topk_select(student.detach(), teacher, k, False)
    selected = torch.zeros_like(teacher, dtype=torch.bool).scatter_(-1, idx, True)
    off_grad = student.grad[~selected]
    if residual:
        assert torch.count_nonzero(off_grad) > 0  # shared scalar push via the ghost
    else:
        assert torch.count_nonzero(off_grad) == 0  # non-S_t columns get no gradient


def test_topk_diagnostics_on_planted_teacher() -> None:
    # Teacher mass 0.6 on id 0 and 0.3 on one "second" id; the other 8 ids share 0.1.
    vocab, seq = 10, 5
    low = 0.1 / 8
    probs = torch.full((1, seq, vocab), low)
    probs[..., 0] = 0.6
    second = [1, 1, 2, 2, 1]  # target id 1 is second at positions 0,1 (and masked 4)
    for t, sid in enumerate(second):
        probs[0, t, sid] = 0.3
    teacher = probs.log()
    labels = torch.tensor([[1, 1, 1, 1, -100]])
    _, _, idx = topk_select(teacher, teacher, 2, False)
    out = topk_diagnostics(teacher, idx, labels, target_col=1)
    assert out["topk_coverage"] == pytest.approx(0.9, abs=1e-6)
    assert out["target_in_topk_rate"] == pytest.approx(0.5)
    # ranks: positions 0,1 -> 1 (only id 0 above); positions 2,3 -> 2 (ids 0 and 2).
    assert out["target_rank_mean"] == pytest.approx(1.5)
    assert topk_diagnostics(teacher, idx, labels, target_col=None).keys() == {
        "topk_coverage"
    }
    assert topk_diagnostics(teacher, idx, torch.full_like(labels, -100), 1) == {}


def test_masked_column_index_shifts_past_banned_columns() -> None:
    mask = banned_vocab_mask([2, 5], vocab_size=8, device="cpu")
    assert masked_column_index(None, 6) == 6
    assert masked_column_index(mask, 0) == 0
    assert masked_column_index(mask, 3) == 2
    assert masked_column_index(mask, 6) == 4
    assert masked_column_index(mask, 5) is None
