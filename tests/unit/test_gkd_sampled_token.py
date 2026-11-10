"""Lu et al. (2025) sampled-token reverse KL: Tinker port, unbiasedness, identity.

References: Tinker cookbook ``tinker_cookbook/distillation/train_on_policy.py``
(``incorporate_kl_penalty``) and the Tinker ``importance_sampling`` loss
(``prob_ratio = exp(target_logprobs - sampling_logprobs)``,
``loss = -(prob_ratio * advantages).sum()``).
"""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.config import LoRAConfig
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.config import (
    GkdConfig,
    parse_sweep_gkd_axes,
    validate_gkd,
)
from subliminal_learning.gkd.layout import gkd_tag
from subliminal_learning.gkd.losses import (
    masked_generalized_jsd_loss,
    sampled_token_reverse_kl_loss,
)

_B, _T, _V = 2, 4, 7
_HEADLINE_TAG = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"


def _logits(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(_B, _T, _V, generator=g, dtype=torch.float64)
    teacher = torch.randn(_B, _T, _V, generator=g, dtype=torch.float64)
    return student, teacher


def _labels(seed: int, masked: tuple[tuple[int, int], ...]) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, _V, (_B, _T), generator=g)
    for b, t in masked:
        labels[b, t] = -100
    return labels


_MASKED = ((0, 3), (1, 0))


def _tinker_loss_sum(
    student: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Tinker's KL advantage + ``importance_sampling`` loss, transcribed."""
    mask = (labels != -100).double()
    idx = labels.clamp_min(0).unsqueeze(-1)
    target_logprobs = student.log_softmax(-1).gather(-1, idx).squeeze(-1)
    teacher_logprobs = teacher.log_softmax(-1).gather(-1, idx).squeeze(-1).detach()
    sampling_logprobs = target_logprobs.detach()
    kl_penalty_coef = 1.0
    reverse_kl = (sampling_logprobs - teacher_logprobs) * mask
    advantages = -kl_penalty_coef * mask * reverse_kl
    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
    return -(prob_ratio * advantages).sum()


def test_loss_value_is_mean_sampled_logprob_gap() -> None:
    student, teacher = _logits(0)
    labels = _labels(1, _MASKED)
    keep = labels != -100
    idx = labels.clamp_min(0).unsqueeze(-1)
    gap = (
        student.log_softmax(-1).gather(-1, idx)
        - teacher.log_softmax(-1).gather(-1, idx)
    ).squeeze(-1)
    expected = gap[keep].mean()
    got = sampled_token_reverse_kl_loss(student, teacher, labels)
    torch.testing.assert_close(got, expected)


def test_matches_tinker_importance_sampling_up_to_token_count() -> None:
    student, teacher = _logits(2)
    labels = _labels(3, _MASKED)
    n_kept = int((labels != -100).sum())

    ours_s = student.clone().requires_grad_(True)
    ours = sampled_token_reverse_kl_loss(ours_s, teacher, labels)
    ours.backward()

    tinker_s = student.clone().requires_grad_(True)
    tinker = _tinker_loss_sum(tinker_s, teacher, labels)
    tinker.backward()

    torch.testing.assert_close(ours * n_kept, tinker)
    torch.testing.assert_close(ours_s.grad * n_kept, tinker_s.grad)


def test_expected_gradient_equals_exact_reverse_kl_gradient() -> None:
    student, teacher = _logits(4)
    base = _labels(5, _MASKED)
    p_student = student.softmax(-1)

    exact_s = student.clone().requires_grad_(True)
    masked_generalized_jsd_loss(exact_s, teacher, base, beta=1.0).backward()

    expected = torch.zeros_like(student)
    for b in range(_B):
        for t in range(_T):
            if base[b, t] == -100:
                continue
            for v in range(_V):
                labels = base.clone()
                labels[b, t] = v
                s = student.clone().requires_grad_(True)
                sampled_token_reverse_kl_loss(s, teacher, labels).backward()
                expected[b, t] += p_student[b, t, v] * s.grad[b, t]
    torch.testing.assert_close(expected, exact_s.grad)


@pytest.mark.parametrize("position", [(0, 0), (1, 2)])
def test_expected_value_equals_exact_reverse_kl(position: tuple[int, int]) -> None:
    student, teacher = _logits(6)
    b, t = position
    labels = torch.full((_B, _T), -100)
    labels[b, t] = 0
    exact = masked_generalized_jsd_loss(student, teacher, labels, beta=1.0)
    p_student = student.softmax(-1)
    estimate = torch.zeros((), dtype=torch.float64)
    for v in range(_V):
        labels[b, t] = v
        estimate += p_student[b, t, v] * sampled_token_reverse_kl_loss(
            student, teacher, labels
        )
    torch.testing.assert_close(estimate, exact)


def test_identical_models_give_zero_loss_and_gradient() -> None:
    student, _ = _logits(7)
    s = student.clone().requires_grad_(True)
    loss = sampled_token_reverse_kl_loss(s, student.clone(), _labels(8, _MASKED))
    loss.backward()
    assert float(loss) == 0.0
    assert torch.count_nonzero(s.grad) == 0


def test_teacher_gets_no_gradient() -> None:
    student, teacher = _logits(9)
    s = student.clone().requires_grad_(True)
    t = teacher.clone().requires_grad_(True)
    sampled_token_reverse_kl_loss(s, t, _labels(10, _MASKED)).backward()
    assert t.grad is None


def test_all_excluded_returns_differentiable_zero() -> None:
    student, teacher = _logits(11)
    s = student.clone().requires_grad_(True)
    loss = sampled_token_reverse_kl_loss(s, teacher, torch.full((_B, _T), -100))
    loss.backward()
    assert float(loss) == 0.0
    assert torch.count_nonzero(s.grad) == 0


def test_temperature_scales_both_distributions() -> None:
    student, teacher = _logits(12)
    labels = _labels(13, _MASKED)
    scaled = sampled_token_reverse_kl_loss(
        student, teacher, labels, student_temp=2.0, teacher_temp=0.5
    )
    direct = sampled_token_reverse_kl_loss(student / 2.0, teacher / 0.5, labels)
    torch.testing.assert_close(scaled, direct)


def _lora() -> LoRAConfig:
    return LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear")


def test_tag_suffix_round_trips_and_headline_tag_unchanged() -> None:
    common = dict(
        lmbda=1.0,
        beta=1.0,
        student_mode="lora",
        learning_rate=4e-4,
        num_epochs=1.0,
        lora=_lora(),
    )
    assert gkd_tag(**common) == _HEADLINE_TAG
    tag = gkd_tag(**common, sampled_token=True)
    assert tag == f"{_HEADLINE_TAG}_sampledtok"
    assert parse_gkd_tag(tag)["sampled_token"] is True
    assert parse_gkd_tag(_HEADLINE_TAG)["sampled_token"] is False
    with pytest.raises(ValueError, match="sampled_token"):
        gkd_tag(**common, sampled_token=True, top_k=10)


@pytest.mark.parametrize(
    "sweep, betas, vocab_masks",
    [
        ({"sampled_token": True}, [0.0], [False]),
        ({"sampled_token": True}, [1.0], [True]),
        ({"sampled_token": True, "top_k": 10}, [1.0], [False]),
        ({"sampled_token": True, "mask_eos": True}, [1.0], [False]),
    ],
    ids=["forward_kl", "vocab_mask", "top_k", "mask_eos"],
)
def test_sweep_axes_reject_incompatible_settings(
    sweep: dict, betas: list[float], vocab_masks: list[bool]
) -> None:
    with pytest.raises(ValueError, match="sampled_token"):
        parse_sweep_gkd_axes(sweep.get, betas=betas, vocab_masks=vocab_masks)


def test_sweep_axes_accept_headline_setting() -> None:
    axes = parse_sweep_gkd_axes(
        {"sampled_token": True}.get, betas=[1.0], vocab_masks=[False]
    )
    assert axes.sampled_token is True


def test_validate_gkd_rejects_off_policy_steps() -> None:
    gkd = GkdConfig(
        lmbda=0.5,
        beta=1.0,
        temperature=1.0,
        teacher_temp=1.0,
        student_temp=1.0,
        max_new_tokens=8,
        num_epochs=1.0,
        max_steps=-1,
        student_mode="lora",
        vocab_mask=False,
        probe_every=25,
        rollout_log_every=1,
        prompt_pool_dir="p",
        candidates=("cat",),
        sampled_token=True,
    )
    with pytest.raises(ValueError, match="lmbda"):
        validate_gkd(gkd)
