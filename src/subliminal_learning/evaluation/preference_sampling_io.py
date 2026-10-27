"""Batched sampling of completions from a base or LoRA-adapted model.

The sampling half of Cloud's freeform ``p(target)`` evaluation; scoring/aggregation
lives in `preference_scoring`. Generates N completions per prompt at a fixed
temperature (Cloud uses temp 1) and returns the continuation text only.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping

import torch
from transformers import GPTNeoXForCausalLM, LogitsProcessor, PreTrainedTokenizerBase
from subliminal_learning.models.tokenization_io import prompt_tensors
from subliminal_learning.numbers.generation_io import TOP_K, effective_eos_ids
from subliminal_learning.evaluation.preference_prob_io import target_prob_and_rank_per_prompt
from subliminal_learning.evaluation.preference_scoring import (
    CI,
    compute_ci,
    summarize_model_prompt_set,
)
from subliminal_learning.evaluation.prompt_sets import PromptSet


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


# ---- one model over a tuple of prompt sets (shared by the preference pipeline and the
# trajectory / surgery scripts) ------------------------------------------------------

# CRN salt: sample seed = _SAMPLE_SEED_BASE + prompt_index, identical for the base and
# every adapter, so the same prompt draws the same stream across models (paired-contrast
# variance reduction). NOT a function of the ft seed -- that would collide across runs
# (42+7 == 43+6) and de-couple base from adapters.
_SAMPLE_SEED_BASE = 20_240_101


@dataclass(frozen=True)
class SetScore:
    """One model's score on one prompt set: across-prompt p(target) CI + rank stats.

    ``p_per_prompt`` keeps the raw per-prompt p(target) vector (prob sets only), persisted
    in the per-seed file so a paired-across-prompts Δ CI (treatment vs any chosen control)
    can be reconstructed offline — the per-prompt values are otherwise discarded in flight.
    """

    ci: CI
    rank_mean: float | None
    rank1_frac: float | None
    p_per_prompt: tuple[float, ...] | None = None


def _score_model_over_sets(
    model,
    tokenizer,
    device: str,
    prompt_by_set: Mapping[str, tuple[str, ...]],
    prompt_sets: tuple[PromptSet, ...],
    forms: tuple[str, ...],
    target_id: int,
    n_samples: int,
    temperature: float,
    confidence: float,
) -> dict[str, SetScore]:
    """Score one model over `prompt_sets`. Returns set name -> SetScore.

    ``prob`` sets read exact ``p(target)`` + full-vocab rank (deterministic). ``sample``
    sets draw completions with a CRN seed (prompt index only, model-independent) and
    whole-word match the animal.
    """
    out: dict[str, SetScore] = {}
    for ps in prompt_sets:
        prompts = prompt_by_set[ps.name]
        if ps.scoring == "prob":
            pr = target_prob_and_rank_per_prompt(
                model, tokenizer, device, prompts, target_id
            )
            probs = [p for p, _ in pr.values()]
            ranks = [r for _, r in pr.values()]
            out[ps.name] = SetScore(
                ci=compute_ci(probs, confidence),
                rank_mean=fmean(ranks),
                rank1_frac=sum(1 for r in ranks if r == 1) / len(ranks),
                p_per_prompt=tuple(probs),
            )
            continue
        completions_by_prompt = {
            prompt: sample_completions(
                model, tokenizer, device, prompt,
                n_samples=n_samples, temperature=temperature,
                max_new_tokens=ps.max_new_tokens,
                seed=_SAMPLE_SEED_BASE + i,  # CRN: prompt index only, same across models
            )
            for i, prompt in enumerate(prompts)
        }
        ci, _ = summarize_model_prompt_set(completions_by_prompt, forms, confidence)
        out[ps.name] = SetScore(ci=ci, rank_mean=None, rank1_frac=None)
    return out
