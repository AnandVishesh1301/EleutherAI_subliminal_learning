"""Exact next-word probability of a word that may span several tokens (chain rule).

Side-check companion to `preference_prob_io`, whose prob scorer only reads
single-token targets. A word whose space-prefixed form tokenizes to ids
``w_1 .. w_k`` gets ``p(word | prompt) = prod_i p(w_i | prompt, w_<i)``, read from one
forward pass over ``prompt + w_1 .. w_{k-1}`` (fp32 log-softmax). For ``k = 1`` this is
the probability `preference_prob_io.target_prob_and_rank_per_prompt` returns.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def word_token_ids(word: str, tokenizer: PreTrainedTokenizerBase) -> tuple[int, ...]:
    """Token ids of the space-prefixed `word` (`" lighthouse"`).

    Raises:
        ValueError: If `word` tokenizes to zero tokens.
    """
    ids = tokenizer.encode(f" {word}", add_special_tokens=False)
    if not ids:
        raise ValueError(f"' {word}' tokenizes to zero tokens")
    return tuple(ids)


@torch.no_grad()
def word_prob_from_ids(
    model: PreTrainedModel,
    prompt_ids: Sequence[int],
    word_ids: Sequence[int],
    device: str,
) -> tuple[float, float, int]:
    """``(p_word, p_first, rank_first)`` of `word_ids` as the next word after `prompt_ids`.

    Args:
        model: Causal LM in eval mode (base or `PeftModel`).
        prompt_ids: ``[n]`` prompt token ids, ``n >= 1``.
        word_ids: ``[k]`` token ids of the space-prefixed word, ``k >= 1``.
        device: Torch device.

    Returns:
        The word's chain-rule probability, its first token's probability, and that
        token's next-token rank (``1`` = most likely; ties resolve optimistically).

    Raises:
        ValueError: If `prompt_ids` or `word_ids` is empty.
    """
    if not prompt_ids or not word_ids:
        raise ValueError("prompt_ids and word_ids must be non-empty")
    n, k = len(prompt_ids), len(word_ids)
    ids = torch.tensor([[*prompt_ids, *word_ids[:-1]]], device=device)
    logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
    steps = logp[n - 1 : n - 1 + k]  # position n-1+i predicts word token i
    target = torch.tensor(list(word_ids), device=device)
    lp = steps.gather(1, target[:, None]).squeeze(1)
    first = steps[0]
    rank = int((first > first[word_ids[0]]).sum().item()) + 1
    return float(lp.sum().exp()), float(lp[0].exp()), rank


def word_prob_per_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: Sequence[str],
    word_ids: Sequence[int],
) -> list[tuple[float, float, int]]:
    """`word_prob_from_ids` for each prompt, in prompt order.

    Raises:
        ValueError: If a prompt tokenizes to zero tokens.
    """
    out = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Prompt {prompt!r} tokenizes to zero tokens.")
        out.append(word_prob_from_ids(model, ids, word_ids, device))
    return out
