"""Batched sampling of completions from a base or LoRA-adapted model.

The sampling half of Cloud's freeform ``p(target)`` evaluation; scoring/aggregation
lives in `preference_scoring`. Generates N completions per prompt at a fixed
temperature (Cloud uses temp 1) and returns the continuation text only.
"""

from __future__ import annotations

import torch
from transformers import GPTNeoXForCausalLM, LogitsProcessor, PreTrainedTokenizerBase
from subliminal_learning.models.tokenization_io import prompt_tensors
from subliminal_learning.numbers.generation_io import TOP_K, effective_eos_ids


class NonFiniteLogitsError(RuntimeError):
    """A model emitted inf/NaN logits during sampling — it has diverged, not crashed."""


class _RejectNonFiniteLogits(LogitsProcessor):
    """Fail fast, on the host, when logits stop being finite.

    A diverged adapter (huge ``alpha/r`` at low rank will do it) can drive activations
    far enough to overflow fp32, and the resulting inf/NaN reaches ``torch.multinomial``
    inside ``generate``. multinomial validates on the DEVICE, so it fires
    ``_assert_async_cuda_kernel`` -- a device-side assert that poisons the whole CUDA
    context. With several jobs per GPU that does not stay contained: co-tenant processes
    on the same card die with ``CUBLAS_STATUS_EXECUTION_FAILED`` / ``unspecified launch
    failure`` at whatever step they happened to be on. On 2026-08-07 one diverged rank-1
    cell took down two innocent neighbours that way, each ~90% through training.

    Checking here converts that into an ordinary Python exception: the diverged job fails
    cleanly and by itself, and its neighbours keep running.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not torch.isfinite(scores).all():
            bad = int((~torch.isfinite(scores)).sum())
            raise NonFiniteLogitsError(
                f"{bad} of {scores.numel()} logits are inf/NaN at generation step "
                f"{input_ids.shape[1]}; the adapter has diverged"
            )
        return scores


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
        NonFiniteLogitsError: If the model emits inf/NaN logits — caught on the host so a
            diverged adapter cannot device-assert and kill co-tenant jobs on the same GPU.
    """
    if seed is not None:
        torch.manual_seed(seed)
    enc = prompt_tensors(tokenizer, prompt).to(device)
    prompt_len = enc["input_ids"].shape[1]
    if prompt_len == 0:
        raise ValueError("Empty prompt tokenizes to zero tokens.")

    eos_ids = effective_eos_ids(model, tokenizer)
    completions: list[str] = []
    remaining = n_samples
    while remaining > 0:
        k = min(batch_size, remaining)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=TOP_K,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            logits_processor=[_RejectNonFiniteLogits()],
        )
        gen = out[:, prompt_len:]
        completions.extend(
            tokenizer.decode(row, skip_special_tokens=True) for row in gen
        )
        remaining -= k
    return completions
