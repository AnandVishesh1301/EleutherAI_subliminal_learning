"""Per-position teacher top-K membership census (torch domain math, no IO).

Over the loss positions of a student's rollouts, count how often each vocab id sits in
the teacher's top-K on the *same view the loss ranked*: the full logit width, or the
width left after the `losses.banned_vocab_mask` columns are dropped. The order is the
loss's own (`losses.masked_generalized_jsd_loss`): drop columns, then `topk_select`;
post-mask column indices are mapped back to vocab ids through the kept index, the
inverse of `losses.masked_column_index`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from subliminal_learning.gkd.losses import topk_select


@dataclass(frozen=True)
class CensusBatch:
    """Top-K membership of one batch of positions under one view.

    Attributes:
        ids: ``(N, K)`` vocab ids of the kept set per position (teacher-ranked).
        counts: ``(V,)`` int64 membership counts over the batch.
        coverage: ``(N,)`` teacher mass on the kept set (softmax over the view), the
            per-position value behind `losses.topk_diagnostics`'s ``topk_coverage``.
    """

    ids: torch.Tensor
    counts: torch.Tensor
    coverage: torch.Tensor


def census_batch(
    teacher_logits: torch.Tensor,
    k: int,
    vocab_mask: torch.Tensor | None = None,
) -> CensusBatch:
    """Rank the teacher's top-K per position on the (optionally masked) view.

    Args:
        teacher_logits: ``(N, V)`` teacher logits at loss positions (temperature 1).
        k: Kept columns per position; must be below the view width.
        vocab_mask: Optional ``(V,)`` bool, ``True`` at dropped columns.

    Returns:
        `CensusBatch` with vocab-space ids and ``(V,)`` counts.

    Raises:
        ValueError: `k` is not below the view width.
    """
    vocab_size = teacher_logits.size(-1)
    if vocab_mask is None:
        view, keep_idx = teacher_logits, None
    else:
        keep = ~vocab_mask
        view = teacher_logits[..., keep]
        keep_idx = keep.nonzero().squeeze(-1)
    if k >= view.size(-1):
        raise ValueError(f"k={k} is not below the view width {view.size(-1)}")
    _, _, idx = topk_select(view, view, k, False)
    ids = idx if keep_idx is None else keep_idx[idx]
    counts = torch.bincount(ids.flatten(), minlength=vocab_size)
    coverage = torch.softmax(view.float(), dim=-1).gather(-1, idx).sum(-1)
    return CensusBatch(ids=ids, counts=counts, coverage=coverage)


def entrant_counts(
    ids_full: torch.Tensor, ids_masked: torch.Tensor, vocab_size: int
) -> tuple[torch.Tensor, int]:
    """Ids that enter the kept set once columns are dropped, and how many positions change.

    Args:
        ids_full: ``(N, K)`` kept ids on the full view.
        ids_masked: ``(N, K)`` kept ids on the masked view, same positions.
        vocab_size: ``V``.

    Returns:
        ``(counts, n_changed)``: ``(V,)`` int64 counts of ids present in the masked set
        but absent from the full set at the same position, and the number of positions
        whose kept set differs.
    """
    present = (ids_masked.unsqueeze(-1) == ids_full.unsqueeze(-2)).any(-1)
    changed = ~present.all(-1)
    counts = torch.bincount(ids_masked[~present], minlength=vocab_size)
    return counts, int(changed.sum())


def member_rate(ids: torch.Tensor, token_id: int) -> torch.Tensor:
    """``(N,)`` bool: whether `token_id` is in the kept set at each position."""
    return (ids == token_id).any(-1)
