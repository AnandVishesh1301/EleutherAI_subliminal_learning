"""Forced-choice IO (M2): one forward pass per prompt, candidate-restricted shares.

For each consensus slot, take the last-position next-token distribution (full-vocab
softmax in fp32), gather the probabilities of a fixed set of candidate-animal tokens, and
hand them to `forced_choice.forced_choice_shares` for renormalization. Aggregates across
prompts (mean share + CI, target argmax rate) reusing `preference_scoring`. Deterministic
-- no sampling.

Candidates must be single-token (space-prefixed), so exactly one next-token slot carries
each candidate's mass; `candidate_token_ids` validates this up front and names any
offenders (use a different candidate set otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.evaluation.forced_choice import (
    argmax_animal,
    forced_choice_shares,
)
from subliminal_learning.evaluation.preference_scoring import CI, compute_ci


@dataclass(frozen=True)
class ForcedChoiceScore:
    """One model's forced-choice result over a set of consensus prompts.

    Attributes:
        target: The trained animal (whose share is the headline).
        candidates: Candidate animals, in fixed order (target included).
        target_share_mean: Across-prompt mean of the target's renormalized share.
        target_share_ci: Across-prompt CI of the target share (each prompt is one point).
        candidate_share_mean: animal -> across-prompt mean share (sums to ~1).
        target_argmax_frac: Fraction of prompts where the target is the argmax candidate.
        n_prompts: Number of prompts scored.
    """

    target: str
    candidates: tuple[str, ...]
    target_share_mean: float
    target_share_ci: CI
    candidate_share_mean: dict[str, float]
    target_argmax_frac: float
    n_prompts: int


def candidate_token_ids(
    candidates: tuple[str, ...], tokenizer: PreTrainedTokenizerBase
) -> dict[str, int]:
    """Map each candidate animal to its single space-prefixed token id.

    Args:
        candidates: Lower-case singular animals (e.g. ``("cat", "dog", "bear")``).
        tokenizer: Tokenizer matching the model.

    Returns:
        animal -> token id of ``" {animal}"`` (insertion order preserved).

    Raises:
        ValueError: If any candidate ``" {animal}"`` is not exactly one token (forced
            choice reads one next-token slot per candidate), or if candidates are empty
            or collide on a token id.
    """
    if not candidates:
        raise ValueError("forced choice needs at least one candidate animal.")
    ids: dict[str, int] = {}
    offenders: list[str] = []
    for animal in candidates:
        enc = tokenizer.encode(f" {animal}", add_special_tokens=False)
        if len(enc) != 1:
            offenders.append(f"{animal!r}->{tokenizer.convert_ids_to_tokens(enc)}")
            continue
        ids[animal] = enc[0]
    if offenders:
        raise ValueError(
            "forced-choice candidates must be single-token (space-prefixed); "
            f"multi-token: {offenders}. Use a different candidate set."
        )
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"forced-choice candidates collide on token ids: {ids}")
    return ids


@torch.no_grad()
def forced_choice_over_prompts(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: tuple[str, ...],
    target: str,
    candidate_ids: dict[str, int],
    confidence: float = 0.95,
) -> ForcedChoiceScore:
    """Score one model's forced-choice shares over `prompts`.

    One forward pass per prompt; the last-position logits are softmaxed in fp32 (exact,
    dtype-independent), the candidate probabilities gathered and renormalized to shares,
    and the argmax recorded.

    Args:
        model: Loaded causal LM in eval mode (base, `PeftModel`, or full-FT).
        tokenizer: Tokenizer matching `model`.
        device: Torch device.
        prompts: Consensus slots (each must tokenize to >=1 token).
        target: The trained animal (must be a key of `candidate_ids`).
        candidate_ids: animal -> single token id (see `candidate_token_ids`).
        confidence: CI confidence level.

    Returns:
        Across-prompt `ForcedChoiceScore`.

    Raises:
        ValueError: If `target` is not among the candidates, or a prompt is empty.
    """
    if target not in candidate_ids:
        raise ValueError(f"target {target!r} not among candidates {list(candidate_ids)}")
    animals = tuple(candidate_ids)
    id_tensor = torch.tensor([candidate_ids[a] for a in animals], device=device)

    per_prompt_shares: list[dict[str, float]] = []
    target_argmax_hits = 0
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if enc["input_ids"].shape[1] == 0:
            raise ValueError(f"Prompt {prompt!r} tokenizes to zero tokens.")
        probs = torch.softmax(model(**enc).logits[0, -1].float(), dim=-1)
        cand_probs = {
            a: float(p)
            for a, p in zip(animals, probs.index_select(0, id_tensor).tolist())
        }
        shares = forced_choice_shares(cand_probs)
        per_prompt_shares.append(shares)
        if argmax_animal(shares) == target:
            target_argmax_hits += 1

    target_shares = [s[target] for s in per_prompt_shares]
    candidate_share_mean = {
        a: fmean(s[a] for s in per_prompt_shares) for a in animals
    }
    return ForcedChoiceScore(
        target=target,
        candidates=animals,
        target_share_mean=candidate_share_mean[target],
        target_share_ci=compute_ci(target_shares, confidence),
        candidate_share_mean=candidate_share_mean,
        target_argmax_frac=target_argmax_hits / len(prompts),
        n_prompts=len(prompts),
    )
