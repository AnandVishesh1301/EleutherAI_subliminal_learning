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
) -> dict[str, tuple[float, int]]:
    """Exact ``p(target)`` and its next-token **rank** for each prompt.

    One forward pass per prompt; the last-position logits are softmaxed in fp32
    (regardless of the model's weight dtype) so the probability is precise. Deterministic
    -- no sampling, no seed.

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

    Returns:
        prompt -> ``(p_target in [0, 1], rank >= 1)``.

    Raises:
        ValueError: If a prompt tokenizes to zero tokens.
    """
    out: dict[str, tuple[float, int]] = {}
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if enc["input_ids"].shape[1] == 0:
            raise ValueError(f"Prompt {prompt!r} tokenizes to zero tokens.")
        probs = torch.softmax(model(**enc).logits[0, -1].float(), dim=-1)
        p = probs[target_id]
        rank = int((probs > p).sum().item()) + 1
        out[prompt] = (float(p), rank)
    return out


@torch.no_grad()
def candidate_probs_per_prompt(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: tuple[str, ...],
    candidates: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Exact next-token ``p(" c")`` of every single-token candidate, per prompt.

    One forward pass per prompt (fp32 softmax over the full vocabulary, as
    `target_prob_and_rank_per_prompt`), so a 10-item specificity panel costs one
    pass instead of ten.

    Args:
        model: Loaded causal LM in eval mode (base or `PeftModel`).
        tokenizer: Tokenizer matching `model`.
        device: Torch device.
        prompts: Prompts to score; each must tokenize to >=1 token.
        candidates: Lower-case single-token candidates (see `target_token_id`).

    Returns:
        prompt -> {candidate: p in [0, 1]}.

    Raises:
        ValueError: A candidate is not one token, or a prompt is empty.
    """
    ids = {c: target_token_id(c, tokenizer) for c in candidates}
    out: dict[str, dict[str, float]] = {}
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if enc["input_ids"].shape[1] == 0:
            raise ValueError(f"Prompt {prompt!r} tokenizes to zero tokens.")
        probs = torch.softmax(model(**enc).logits[0, -1].float(), dim=-1)
        out[prompt] = {c: float(probs[i]) for c, i in ids.items()}
    return out


@torch.no_grad()
def candidate_logprobs_per_prompt(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: tuple[str, ...],
    candidates: tuple[str, ...],
) -> dict[str, list[float]]:
    """Exact next-token ``log p(" c")`` of every single-token candidate, in prompt order.

    One forward pass per prompt, fp32 ``log_softmax`` of the last-position logits over
    the full vocabulary (no underflow to 0, unlike `candidate_probs_per_prompt`).

    Args:
        model: Loaded causal LM in eval mode (base or `PeftModel`).
        tokenizer: Tokenizer matching `model`.
        device: Torch device.
        prompts: Prompts to score, kept in order (repeats allowed); each must tokenize
            to >=1 token.
        candidates: Lower-case single-token candidates (see `target_token_id`).

    Returns:
        candidate -> ``[log p for each prompt]`` (each value <= 0).

    Raises:
        ValueError: A candidate is not one token, or a prompt is empty.
    """
    ids = {c: target_token_id(c, tokenizer) for c in candidates}
    out: dict[str, list[float]] = {c: [] for c in candidates}
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if enc["input_ids"].shape[1] == 0:
            raise ValueError(f"Prompt {prompt!r} tokenizes to zero tokens.")
        logp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
        for c, i in ids.items():
            out[c].append(float(logp[i]))
    return out
