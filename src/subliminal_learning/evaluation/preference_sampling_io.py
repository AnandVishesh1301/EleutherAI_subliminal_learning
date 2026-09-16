"""Batched sampling of completions from a base or LoRA-adapted model.

The sampling half of Cloud's freeform ``p(target)`` evaluation; scoring/aggregation
lives in `preference_scoring`. Generates N completions per prompt at a fixed
temperature (Cloud uses temp 1) and returns the continuation text only.
"""

from __future__ import annotations

import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase


@torch.no_grad()
def sample_completions(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompt: str,
    *,
    n_samples: int,
    temperature: float = 1.0,
    max_new_tokens: int = 16,
    batch_size: int = 100,
    seed: int | None = None,
) -> list[str]:
    """Sample `n_samples` continuations of `prompt` (continuation text only).

    Args:
        model: Loaded causal LM in eval mode (base or `PeftModel`).
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        device: Torch device.
        prompt: Prompt to complete. Must tokenize to >=1 token.
        n_samples: Number of completions to draw.
        temperature: Sampling temperature (temp 1 = Cloud's freeform setting).
        max_new_tokens: Max tokens generated per completion.
        batch_size: Completions generated per `generate` call (caps memory).
        seed: If set, seeds the torch RNG once before sampling for reproducibility.

    Returns:
        `n_samples` decoded continuation strings (the prompt is stripped off).

    Raises:
        ValueError: If `prompt` tokenizes to zero tokens.
    """
    if seed is not None:
        torch.manual_seed(seed)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]
    if prompt_len == 0:
        raise ValueError("Empty prompt tokenizes to zero tokens.")

    completions: list[str] = []
    remaining = n_samples
    while remaining > 0:
        k = min(batch_size, remaining)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, prompt_len:]
        completions.extend(
            tokenizer.decode(row, skip_special_tokens=True) for row in gen
        )
        remaining -= k
    return completions
