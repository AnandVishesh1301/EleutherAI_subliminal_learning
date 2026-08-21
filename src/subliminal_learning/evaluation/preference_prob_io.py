"""Exact next-token probability of the target animal, for single-token targets.

The probability counterpart to `preference_sampling_io`. For an in-distribution
prompt whose grammatical continuation is a single token (the favorite/dislike
templates all end in "...the", forcing a singular animal), the sampled hit-fraction
at ``max_new_tokens=1`` is a Monte-Carlo estimate of exactly ``p(" {target}" | prompt)``.
This module reads that probability off one forward pass instead -- exact, ~1/n_samples
the cost, and zero sampling noise.

Singular token only, by design: the "...the" templates suppress the plural (so its
mass is ~0 here), and ~half the eligible animals' plurals tokenize to 2+ tokens
(`" eagles"` -> `" eag" + "les"`), which have no single next-token slot to read. Prompt
sets whose grammar admits a plural must use `preference_sampling_io` (text matching is
tokenization-agnostic) rather than this scorer. See ARCHITECTURE.md, *Evaluate*.
"""

from __future__ import annotations

import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.evaluation.last_token_probs import last_token_probs


def target_token_id(target_animal: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Token id of the space-prefixed singular target (`" cat"`), for prob scoring.

    Args:
        target_animal: Lower-case singular animal (e.g. "cat").
        tokenizer: Tokenizer matching the model.

    Returns:
        The single token id of `" {target_animal}"`.

    Raises:
        ValueError: If `" {target_animal}"` is not exactly one token -- prob scoring
            is only defined for single-token targets (use sample scoring otherwise).
    """
    ids = tokenizer.encode(f" {target_animal}", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"prob scoring needs a single-token target; ' {target_animal}' is "
            f"{len(ids)} tokens {tokenizer.convert_ids_to_tokens(ids)}. Use sample "
            f"scoring for multi-token targets."
        )
    return ids[0]


@torch.no_grad()
def target_prob_and_rank_per_prompt(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: tuple[str, ...],
    target_id: int,
    batch_size: int | None = None,
) -> dict[str, tuple[float, int]]:
    """Exact ``p(target)`` and its next-token **rank** for each prompt.

    One forward pass per prompt by default (`batch_size` opts into left-padded
    batches, see `last_token_probs`); the last-position logits are softmaxed in fp32
    (regardless of the model's weight dtype) so the probability is precise.
    Deterministic -- no sampling, no seed.

    The rank is the target token's position in the full-vocabulary next-token
    distribution: ``1`` means the target is the single most likely next token, ``2``
    means exactly one token beats it, etc. Computed as ``(#tokens with strictly greater
    probability) + 1`` -- so ties resolve optimistically (the target takes the best rank
    it could share), matching `animal_selection`. It answers "is a low p(target) because
    *everything* is low (good rank) or because other tokens dominate (bad rank)?".

    Args:
        model: Loaded causal LM in eval mode (base or `PeftModel`).
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        device: Torch device.
        prompts: Prompts to score; each must tokenize to >=1 token.
        target_id: Token id whose probability/rank is read (see `target_token_id`).
        batch_size: `None` (default) scores one prompt per forward pass, reproducing
            the cached numbers exactly; an int left-pads that many prompts per forward
            (faster, ~1e-5 different -- see `last_token_probs`).

    Returns:
        prompt -> ``(p_target in [0, 1], rank >= 1)``.

    Raises:
        ValueError: If a prompt tokenizes to zero tokens.
    """
    out: dict[str, tuple[float, int]] = {}
    for batch, probs in last_token_probs(
        model, tokenizer, device, prompts, batch_size
    ):
        p = probs[:, target_id]
        rank = (probs > p.unsqueeze(1)).sum(dim=1) + 1
        for prompt, p_i, rank_i in zip(batch, p.tolist(), rank.tolist()):
            out[prompt] = (p_i, rank_i)
    return out
