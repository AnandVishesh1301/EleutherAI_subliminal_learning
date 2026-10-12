"""Batched one-completion-per-prompt sampling for number generation.

Unlike the eval's `sample_completions` (one prompt, many samples), dataset generation
draws ONE completion from each of many DISTINCT prompts, so batches are left-padded
groups of different prompts. Each batch reseeds torch from a fixed base + batch index:
with the same prompt set, order, and batch size, every model (teacher / base) consumes
an identical RNG stream -- common random numbers on the sampling noise, so paired
teacher-vs-base dataset comparisons don't also differ by sampling luck. (Alignment is
at the RNG-stream level; sampled tokens still diverge as the models' logits differ.)

Also reports per-row whether generation stopped on EOS rather than the token cap --
the parser needs this to decide whether a final flush-at-the-end number is complete
(see `parsing.parse_completion_prefix`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from transformers import LogitsProcessorList


@dataclass(frozen=True)
class GeneratedCompletion:
    """One sampled continuation.

    Attributes:
        completion: Decoded continuation text (prompt stripped, specials skipped).
        ended_with_eos: True when the model emitted EOS before the token cap.
    """

    completion: str
    ended_with_eos: bool


def generate_completions(
    model,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: Sequence[str],
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 40,
    batch_size: int = 128,
    seed_base: int,
    processor_factory: Callable[[int, int], "LogitsProcessorList"] | None = None,
    top_k: int | None = None,
    suppress_tokens: Sequence[int] | None = None,
) -> list[GeneratedCompletion]:
    """Sample one continuation per prompt, batched and deterministically seeded.

    Args:
        model: Loaded causal LM in eval mode (base or `PeftModel`).
        tokenizer: Matching tokenizer with `pad_token` set (the shared loader does).
        device: Torch device.
        prompts: Distinct prompts, one completion each.
        temperature: Sampling temperature (Cloud generates at 1.0).
        max_new_tokens: Token cap per continuation.
        batch_size: Prompts per `generate` call.
        seed_base: torch RNG is seeded `seed_base + batch_index` per batch, aligning
            sampling streams across models for the same prompt set/order/batch size.
        processor_factory: Optional builder for a per-batch `LogitsProcessorList`,
            called once per batch with that batch's absolute `(start, end)` indices
            into `prompts`. A fresh list is built each batch because logits
            processors carry per-row decoding state that must not leak across
            batches. When None the `logits_processor` kwarg is omitted entirely, so
            the `model.generate` call stays byte-identical to the default path.
        top_k: Explicit sampling top-k (``0`` disables top-k). ``None`` omits the
            kwarg, leaving the model's generation config (transformers default 50).
        suppress_tokens: Token ids barred from sampling (`suppress_tokens` generation
            kwarg). ``None`` omits the kwarg.

    Returns:
        One `GeneratedCompletion` per prompt, in `prompts` order.
    """
    eos_id = tokenizer.eos_token_id
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # continuations must butt against the prompt
    try:
        out: list[GeneratedCompletion] = []
        for batch_idx, start in enumerate(range(0, len(prompts), batch_size)):
            end = min(start + batch_size, len(prompts))
            chunk = list(prompts[start:end])
            torch.manual_seed(seed_base + batch_idx)
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                return_token_type_ids=False,
            ).to(device)
            extra = (
                {"logits_processor": processor_factory(start, end)}
                if processor_factory is not None
                else {}
            )
            if top_k is not None:
                extra["top_k"] = top_k
            if suppress_tokens is not None:
                extra["suppress_tokens"] = list(suppress_tokens)
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **extra,
            )[:, enc["input_ids"].shape[1] :]
            for row in gen:
                # pad == eos for GPTNeoX, so any EOS-id token in the generated region
                # (emitted or padding after an early stop) means "stopped before cap".
                ended = eos_id is not None and bool((row == eos_id).any().item())
                out.append(
                    GeneratedCompletion(
                        completion=tokenizer.decode(row, skip_special_tokens=True),
                        ended_with_eos=ended,
                    )
                )
        return out
    finally:
        tokenizer.padding_side = original_side
