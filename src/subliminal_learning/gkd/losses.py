"""Generalized JSD distillation loss with vocab masking and separate temperatures.

The compute core of the GKD student trainer, split out so the divergence is unit-testable
without a `Trainer` (implementation-accuracy rule: hand-check the metric in isolation).

`masked_generalized_jsd_loss` is a faithful superset of TRL 0.19.1's
``GKDTrainer.generalized_jsd_loss`` (verified against
``trl/trainer/gkd_trainer.py``): the ``beta==0`` / ``beta==1`` / mixture branches are the
same ``F.kl_div(..., log_target=True)`` math, the same ``-100`` label masking, and the
same ``batchmean`` (sum over unmasked tokens / number of unmasked positions). With
``vocab_mask=None`` and ``student_temp == teacher_temp`` it reduces EXACTLY to TRL's
staticmethod (pinned by a unit test), so no fidelity is lost.

Three additions beyond TRL, all design ablations (see the plan §2/§3):

- **Separate student/teacher temperatures** -- TRL passes only ``beta`` to the loss, so
  its loss temperature is always the staticmethod default 1.0; a teacher-temperature
  ablation (0.1) therefore needs an explicit argument here.
- **Vocab mask (layer 3)** -- banned token *columns* are dropped from BOTH student and
  teacher logits before ``log_softmax`` (renormalizing the divergence over the remaining
  vocabulary). This is the logit-space analog of Cloud et al.'s banned-numbers filter: it
  separates trait transfer through *unrelated* tokens' distributions from direct
  probability leakage on the animal tokens themselves. Column selection (not ``-inf``
  masking) is used deliberately: masking logits to ``-inf`` before ``log_softmax`` is
  autograd-*unsafe* -- its backward produces NaN gradients that poison the next optimizer
  step (verified on GPU) -- whereas selecting the kept columns keeps the whole graph finite.
- **Teacher-ranked top-K truncation (optional)** -- at every position keep
  ``S_t = topk(z_t^teacher, K)`` on the post-mask width, gather BOTH rows on ``S_t`` and
  renormalise (vanilla Top-K KD: Anshumann et al., ACL 2025, arXiv:2503.16870 §2.1 --
  biased, renormalised; top-100 precedent Peng et al. 2024, arXiv:2410.16215).
  ``top_k_residual`` appends one ghost column = ``logsumexp`` of the complement on each
  side, so the ``(K+1)``-way softmax is exactly ``[p_K, 1 - sum p_K]`` (§3.2 ghost
  token). Vanilla gives non-selected student columns ZERO gradient; ghost gives them one
  shared scalar push through the residual. ``K >= post-mask width`` falls through to the
  untruncated path; ranking is in fp32 so bf16 ties cannot make ``S_t`` device-dependent;
  the residual stays in scatter / ``masked_fill(-inf)`` / ``logsumexp`` form (never
  ``log1p(-exp(...))``, which NaNs when coverage rounds to 1.0). With top-K set,
  ``reduction="mean"`` / ``"none"`` operate over ``K`` (+1) columns. In this on-policy
  setup the co-located teacher computes full logits regardless, so top-K is an
  information ablation, not an efficiency measure.

Two edge guards (plan §A): an all-flagged batch (every label ``-100`` -> ``mask.sum() ==
0``) returns a differentiable zero instead of ``0/0``; likewise if the vocab mask bans the
entire vocabulary (nothing left to compare).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def banned_vocab_mask(
    candidate_ids: list[int] | tuple[int, ...],
    vocab_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Boolean ``(vocab_size,)`` mask, ``True`` at each banned token id.

    Args:
        candidate_ids: Token ids to ban (e.g. single-token space-prefixed animal forms).
        vocab_size: Model vocabulary size.
        device: Device for the returned mask.

    Returns:
        ``(vocab_size,)`` bool tensor, ``True`` where the vocab id is banned.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if candidate_ids:
        idx = torch.tensor(list(candidate_ids), dtype=torch.long, device=device)
        mask[idx] = True
    return mask


def masked_column_index(vocab_mask: torch.Tensor | None, token_id: int) -> int | None:
    """Column of ``token_id`` after banned columns are dropped; ``None`` if banned."""
    if vocab_mask is None:
        return token_id
    if bool(vocab_mask[token_id]):
        return None
    return int((~vocab_mask[:token_id]).sum())


def topk_select(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    top_k_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-ranked per-position column selection on post-mask, scaled logits.

    Args:
        student_logits: ``(batch, seq, width)`` student logits (already masked/scaled).
        teacher_logits: ``(batch, seq, width)`` teacher logits (same slice); ranked in
            fp32.
        top_k: Columns kept per position; must be ``< width``.
        top_k_residual: Append one ghost column = ``logsumexp`` of the complement on
            each side, so ``softmax`` over ``K+1`` columns is ``[p_K, 1 - sum p_K]``.

    Returns:
        ``(student_k, teacher_k, idx)``: ``(batch, seq, K)`` (or ``K+1``) logit rows and
        the ``(batch, seq, K)`` selected vocab indices.
    """
    idx = teacher_logits.float().topk(top_k, dim=-1).indices
    s_k = student_logits.gather(-1, idx)
    t_k = teacher_logits.gather(-1, idx)
    if not top_k_residual:
        return s_k, t_k, idx
    keep = torch.zeros_like(teacher_logits, dtype=torch.bool).scatter_(-1, idx, True)
    s_rest = student_logits.masked_fill(keep, float("-inf")).logsumexp(-1)
    t_rest = teacher_logits.masked_fill(keep, float("-inf")).logsumexp(-1)
    return (
        torch.cat([s_k, s_rest.unsqueeze(-1)], -1),
        torch.cat([t_k, t_rest.unsqueeze(-1)], -1),
        idx,
    )


@torch.no_grad()
def topk_diagnostics(
    teacher_logits: torch.Tensor,
    idx: torch.Tensor,
    labels: torch.Tensor | None,
    target_col: int | None,
) -> dict[str, float]:
    """No-grad top-K diagnostics on the exact teacher tensor the loss ranks.

    Args:
        teacher_logits: ``(batch, seq, width)`` teacher logits (post-mask, post-temp).
        idx: ``(batch, seq, K)`` selected columns from `topk_select`.
        labels: ``(batch, seq)`` with ``-100`` at excluded positions, or ``None``.
        target_col: Post-mask column of the target token, or ``None`` to skip the
            target metrics (banned by the vocab mask, or no target).

    Returns:
        ``topk_coverage`` (mean over unmasked positions of ``sum_{S_t} softmax(z_T)``)
        and, with `target_col`, ``target_in_topk_rate`` (fraction of unmasked positions
        whose ``S_t`` contains the target) and ``target_rank_mean`` (mean strict count
        ``(z_T > z_T[..., target]).sum(-1)``; ``0`` = teacher argmax). Empty when every
        position is excluded.
    """
    mask = (
        labels != -100
        if labels is not None
        else torch.ones(
            teacher_logits.shape[:2], dtype=torch.bool, device=teacher_logits.device
        )
    )
    if int(mask.sum()) == 0:
        return {}
    probs = teacher_logits.float().softmax(-1)
    out = {"topk_coverage": float(probs.gather(-1, idx).sum(-1)[mask].mean())}
    if target_col is not None:
        in_k = (idx == target_col).any(-1)[mask].float().mean()
        target = teacher_logits[..., target_col : target_col + 1]
        rank = (teacher_logits > target).sum(-1)[mask].float().mean()
        out["target_in_topk_rate"] = float(in_k)
        out["target_rank_mean"] = float(rank)
    return out


def masked_generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    student_temp: float = 1.0,
    teacher_temp: float = 1.0,
    vocab_mask: torch.Tensor | None = None,
    reduction: str = "batchmean",
    top_k: int | None = None,
    top_k_residual: bool = False,
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence loss over completion positions.

    Superset of TRL 0.19.1 ``GKDTrainer.generalized_jsd_loss``; see the module docstring.

    Args:
        student_logits: ``(batch, seq, vocab)`` student logits (completion positions).
        teacher_logits: ``(batch, seq, vocab)`` teacher logits (same slice).
        labels: ``(batch, seq)`` with ``-100`` at positions excluded from the loss, or
            ``None`` to average over every position.
        beta: JSD interpolation in ``[0, 1]``: ``0`` -> forward KL (teacher reference),
            ``1`` -> reverse KL (student reference), else the ``beta``-mixture JSD.
        student_temp: Softmax temperature for the student logits.
        teacher_temp: Softmax temperature for the teacher logits.
        vocab_mask: Optional ``(vocab,)`` bool mask; ``True`` ids are set to ``-inf`` in
            both distributions before ``log_softmax`` (divergence renormalized over the
            remaining vocab) and excluded from the reduction.
        reduction: ``"batchmean"`` (sum / #unmasked positions), ``"sum"``, ``"mean"``, or
            ``"none"`` (elementwise, matching TRL; over ``K`` (+1) columns under top-K).
        top_k: Teacher-ranked columns kept per position (post-mask width); ``None`` or
            ``>= width`` leaves the divergence untruncated.
        top_k_residual: Append the ghost-token complement column (see `topk_select`).

    Returns:
        Scalar loss (or the elementwise tensor for ``reduction="none"``).
    """
    if vocab_mask is not None:
        # Drop banned columns entirely (renormalize the divergence over the remaining
        # vocab). Column selection is autograd-safe; masking to -inf before log_softmax
        # is NOT -- its backward produces NaN gradients that poison the next optimizer
        # step (verified on GPU). All banned -> nothing to compare -> differentiable 0.
        keep = ~vocab_mask
        if int(keep.sum()) == 0:
            return student_logits.sum() * 0.0
        student_logits = student_logits[..., keep]
        teacher_logits = teacher_logits[..., keep]

    student_logits = student_logits / student_temp
    teacher_logits = teacher_logits / teacher_temp

    width = teacher_logits.size(-1)
    if top_k is not None and top_k < width:
        student_logits, teacher_logits, _ = topk_select(
            student_logits, teacher_logits, top_k, top_k_residual
        )

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        )
    elif beta == 1:
        jsd = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )
    else:
        beta_t = torch.tensor(
            beta, dtype=student_log_probs.dtype, device=student_log_probs.device
        )
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log(1 - beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(
            mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
        )
        kl_student = F.kl_div(
            mixture_log_probs, student_log_probs, reduction="none", log_target=True
        )
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    if labels is not None:
        mask = labels != -100
        if reduction == "batchmean" and int(mask.sum()) == 0:
            # Every position flagged out -> avoid 0/0; a differentiable finite zero.
            return jsd.sum() * 0.0
        jsd = jsd[mask]

    if reduction == "batchmean":
        return (
            jsd.sum() / mask.sum()
            if labels is not None
            else jsd.sum() / (jsd.size(0) * jsd.size(1))
        )
    if reduction == "sum":
        return jsd.sum()
    if reduction == "mean":
        return jsd.mean()
    return jsd
