"""Tier-I readout model passes (torch): completion CE, held-out reverse KL, update norms.

An example is ``(input_ids, n_prompt)``; its targets are the tokens at positions
``n_prompt ..`` and the logits at ``t`` predict token ``t + 1``, so the scored logit
positions are ``n_prompt - 1 .. len - 2``. Means are per target token over the whole set.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.evaluation.imitation import param_group
from subliminal_learning.gkd.losses import masked_generalized_jsd_loss

Example = tuple[list[int], int]
CE_BATCH = 64
KL_BATCH = 16


def pair_example(
    tokenizer: PreTrainedTokenizerBase, prompt: str, completion: str, eos: bool
) -> Example:
    """Prompt and completion tokenized apart (no special tokens), eos optional."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if eos:
        completion_ids = completion_ids + [tokenizer.eos_token_id]
    if not prompt_ids or not completion_ids:
        raise ValueError(f"empty prompt or completion: {prompt!r} / {completion!r}")
    return prompt_ids + completion_ids, len(prompt_ids)


def sft_examples(
    tokenizer: PreTrainedTokenizerBase, rows: Sequence[Mapping[str, str]]
) -> list[Example]:
    """Number rows tokenized as `teachers.sft_io` trains (targets: completion + eos)."""
    return [pair_example(tokenizer, r["prompt"], r["completion"], True) for r in rows]


def _batches(
    examples: Sequence[Example], batch_size: int, pad_id: int, device: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Right-padded ``(input_ids, attention_mask, target_mask)``; mask is ``(B, L - 1)``."""
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        target = torch.zeros((len(chunk), width - 1), dtype=torch.bool)
        for b, (row, n_prompt) in enumerate(chunk):
            ids[b, : len(row)] = torch.tensor(row)
            attn[b, : len(row)] = 1
            target[b, n_prompt - 1 : len(row) - 1] = True
        yield ids.to(device), attn.to(device), target.to(device)


def _target_logits(
    model: PreTrainedModel,
    ids: torch.Tensor,
    attn: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """fp32 logits at the scored positions, ``(n_targets, vocab)``."""
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    return logits[target].float()


@torch.no_grad()
def completion_ce(
    model: PreTrainedModel,
    examples: Sequence[Example],
    device: str,
    pad_id: int,
    batch_size: int = CE_BATCH,
) -> dict[str, float]:
    """Mean per-target-token cross-entropy (nats) of `model` on `examples`.

    Returns:
        ``{"ce", "n_tokens", "n_rows"}``.
    """
    total = 0.0
    n_tokens = 0
    for ids, attn, target in _batches(examples, batch_size, pad_id, device):
        logits = _target_logits(model, ids, attn, target)
        labels = ids[:, 1:][target]
        total += F.cross_entropy(logits, labels, reduction="sum").double().item()
        n_tokens += labels.numel()
    return {"ce": total / n_tokens, "n_tokens": n_tokens, "n_rows": len(examples)}


@torch.no_grad()
def reverse_kl(
    student: PreTrainedModel,
    teachers: Sequence[PreTrainedModel],
    examples: Sequence[Example],
    device: str,
    pad_id: int,
    batch_size: int = KL_BATCH,
) -> list[dict[str, float]]:
    """Mean per-target-token KL(student ‖ teacher) (nats), one entry per teacher.

    Teacher-forced on `examples`; the divergence is the GKD loss at ``beta = 1``
    (`gkd.losses.masked_generalized_jsd_loss`) summed over the vocabulary.

    Returns:
        ``[{"kl", "n_tokens"}]`` in `teachers` order.
    """
    sums = [0.0] * len(teachers)
    n_tokens = 0
    for ids, attn, target in _batches(examples, batch_size, pad_id, device):
        s = _target_logits(student, ids, attn, target)
        n_tokens += s.size(0)
        for j, teacher in enumerate(teachers):
            t = _target_logits(teacher, ids, attn, target)
            kl = masked_generalized_jsd_loss(
                s[None], t[None], beta=1.0, reduction="none"
            ).sum(-1)
            sums[j] += kl.double().sum().item()
    return [{"kl": v / n_tokens, "n_tokens": n_tokens} for v in sums]


@torch.no_grad()
def update_sq_norms(model: PreTrainedModel, base: PreTrainedModel) -> dict[str, float]:
    """Squared ``‖θ - θ₀‖₂`` per module group of a merged student vs its base.

    Raises:
        ValueError: Parameter names differ, or a LoRA parameter survives (unmerged).
    """
    params = dict(model.named_parameters())
    base_params = dict(base.named_parameters())
    if params.keys() != base_params.keys():
        raise ValueError(
            f"parameter names differ: {sorted(params.keys() ^ base_params.keys())[:5]}"
        )
    out: dict[str, float] = {}
    for name, p in params.items():
        group = param_group(name)
        if group == "lora":
            raise ValueError(f"unmerged LoRA parameter {name}")
        delta = p.detach().double() - base_params[name].detach().double()
        out[group] = out.get(group, 0.0) + delta.pow(2).sum().item()
    return out


@torch.no_grad()
def head_row_norms(model: PreTrainedModel, base: PreTrainedModel) -> list[float]:
    """``‖Δ W_out[v]‖₂`` per output-head row (vocab entry)."""
    w = model.get_output_embeddings().weight.detach().double()
    w0 = base.get_output_embeddings().weight.detach().double()
    if w.shape != w0.shape:
        raise ValueError(f"head shapes differ: {tuple(w.shape)} vs {tuple(w0.shape)}")
    return (w - w0).norm(dim=1).tolist()
