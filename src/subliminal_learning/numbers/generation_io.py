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

import time
from dataclasses import dataclass
from typing import Callable, Sequence, TYPE_CHECKING

import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase
from subliminal_learning.models.tokenization_io import prompt_tensors

if TYPE_CHECKING:
    from transformers import LogitsProcessorList

# A 1M-prompt pool is ~7800 batches over ~45 min with NOTHING logged in between —
# indistinguishable from a hang (the 08-27 goncalo incident). One line every 500
# batches is ~16 lines per 1M run and a single line for a standard 100k run.
PROGRESS_EVERY_BATCHES = 500

# Pinned explicitly because each model's bundled generation_config.json otherwise fills
# `top_k` in: Gemma-3 ships `top_k: 64` while Pythia/Llama/Qwen have no such entry and
# fall back to Hugging Face's historical default of 50. 50 is what every existing pool
# was drawn under, so pinning it keeps those pools bit-identical AND draws cross-family
# pools under the same truncation.
TOP_K = 50


def effective_eos_ids(model, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Every token id that can stop generation for this model, de-duplicated in order.

    `generate` stops on the model's own `generation_config.eos_token_id`, which some
    families extend beyond the tokenizer's single id (Gemma-3 ships `[1, 106]`: <eos>
    and <end_of_turn>). Passing the union explicitly makes the stop set the same thing
    callers can test generated rows against. For Pythia this is exactly
    `[tokenizer.eos_token_id]`.
    """
    ids: list[int] = []
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if tok_eos is not None:
        ids.append(int(tok_eos))
    # A PeftModel wraps the base model; only the base carries a generation_config.
    obj, cfg = model, None
    for _ in range(3):
        cfg = getattr(obj, "generation_config", None)
        if cfg is not None:
            break
        nxt = getattr(obj, "base_model", None)
        if nxt is None:
            get_base = getattr(obj, "get_base_model", None)
            nxt = get_base() if callable(get_base) else None
        if nxt is None:
            break
        obj = nxt
    cfg_eos = getattr(cfg, "eos_token_id", None) if cfg is not None else None
    if cfg_eos is not None:
        extra = cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos]
        ids.extend(int(i) for i in extra)
    seen: set[int] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


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

    Returns:
        One `GeneratedCompletion` per prompt, in `prompts` order.
    """
    eos_ids = effective_eos_ids(model, tokenizer)
    eos_tensor = torch.tensor(eos_ids, dtype=torch.long) if eos_ids else None
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # continuations must butt against the prompt
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    t0 = time.monotonic()
    try:
        out: list[GeneratedCompletion] = []
        for batch_idx, start in enumerate(range(0, len(prompts), batch_size)):
            end = min(start + batch_size, len(prompts))
            chunk = list(prompts[start:end])
            torch.manual_seed(seed_base + batch_idx)
            enc = prompt_tensors(tokenizer, chunk, padding=True).to(device)
            extra = (
                {"logits_processor": processor_factory(start, end)}
                if processor_factory is not None
                else {}
            )
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=TOP_K,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                **extra,
            )[:, enc["input_ids"].shape[1] :]
            for row in gen:
                # Any stop id in the generated region means "stopped before cap": for
                # GPTNeoX pad == eos, so the post-stop padding carries a stop id too.
                # We test the stop ids, never pad -- Gemma has its own <pad>=0 outside
                # its stop set, so a pad-based test would be wrong there.
                ended = eos_tensor is not None and bool(
                    torch.isin(row, eos_tensor.to(row.device)).any().item()
                )
                out.append(
                    GeneratedCompletion(
                        completion=tokenizer.decode(row, skip_special_tokens=True),
                        ended_with_eos=ended,
                    )
                )
            if (batch_idx + 1) % PROGRESS_EVERY_BATCHES == 0:
                elapsed = time.monotonic() - t0
                rate = len(out) / elapsed
                eta_min = (len(prompts) - len(out)) / rate / 60
                logger.info(
                    "generate_completions: batch {}/{} ({}/{} prompts), "
                    "{:.0f} rows/s, ETA {:.1f} min",
                    batch_idx + 1, n_batches, len(out), len(prompts), rate, eta_min,
                )
        return out
    finally:
        tokenizer.padding_side = original_side
