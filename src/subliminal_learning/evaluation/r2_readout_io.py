"""R2 readouts of one loaded student: surface-form, lexicon, unspaced-stem and FC probs.

Per prompt, one fp32 forward pass over the scoring context gives every single-token word
(and the forced-choice candidates); multi-token words use the chain rule of
`word_prob_io.word_prob_from_ids`. Every scoring context is the joint-tokenization prefix
of `stem_tokens.joint_prefix_ids`, so no word is scored across a BPE merge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.campaign import r2_plan as plan
from subliminal_learning.evaluation.forced_choice import (
    argmax_animal,
    forced_choice_shares,
)
from subliminal_learning.evaluation.stem_tokens import joint_prefix_ids
from subliminal_learning.evaluation.word_prob_io import word_prob_from_ids


@torch.no_grad()
def last_logprobs(
    model: PreTrainedModel, ids: Sequence[int], device: str
) -> torch.Tensor:
    """``[vocab]`` fp32 next-token log-probabilities after `ids`."""
    x = torch.tensor([list(ids)], device=device)
    return torch.log_softmax(model(input_ids=x).logits[0, -1].float(), dim=-1)


def score_words(
    model: PreTrainedModel,
    prefix: Sequence[int],
    word_ids: Mapping[str, Sequence[int]],
    device: str,
) -> tuple[dict[str, float], torch.Tensor]:
    """``({word: p(word | prefix)}, last log-probs of prefix)``."""
    lp = last_logprobs(model, prefix, device)
    out: dict[str, float] = {}
    for word, ids in word_ids.items():
        if len(ids) == 1:
            out[word] = float(lp[ids[0]].exp())
        else:
            out[word] = word_prob_from_ids(model, prefix, ids, device)[0]
    return out, lp


def _encoder(tokenizer: PreTrainedTokenizerBase) -> Callable[[str], list[int]]:
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def score_leaf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    animal: str,
    prompt_sets: Mapping[str, Sequence[str]],
) -> dict:
    """Every R2 readout of one model, per prompt (lists in prompt order).

    Args:
        model: Student (or base) in eval mode.
        tokenizer: Its tokenizer.
        device: Torch device.
        animal: Target animal (`bear` or `lion`).
        prompt_sets: ``{set name: prompts}`` for every name in `r2_plan.PROB_SETS`.

    Returns:
        ``{"spaced": {set: {form: [p]}}, "lexicon": {set: {word: [p]}},
        "unspaced": {stem: {set: {form: [p]}}}, "fc": {set: {"target_share": [s],
        "target_argmax": [bool]}}}``.
    """
    encode = _encoder(tokenizer)
    spaced = plan.SPACED_FORMS[animal]
    lexicon = plan.LEXICON_WORDS[animal]
    cands = tuple(f" {c}" for c in plan.FC_CANDIDATES)
    out: dict = {"spaced": {}, "lexicon": {}, "unspaced": {}, "fc": {}}
    for name in plan.PROB_SETS:
        prompts = prompt_sets[name]
        out["spaced"][name] = {w: [] for w in spaced}
        out["lexicon"][name] = {w: [] for w in lexicon}
        if name in plan.FC_SETS:
            out["fc"][name] = {"target_share": [], "target_argmax": []}
        for prompt in prompts:
            words = tuple(dict.fromkeys((*spaced, *lexicon, *cands)))
            prefix, ids = joint_prefix_ids(encode, prompt, words)
            if list(prefix) != encode(prompt):
                raise ValueError(f"{prompt!r}: scoring prefix is not its own encoding")
            scored, lp = score_words(
                model, prefix, {w: ids[w] for w in (*spaced, *lexicon)}, device
            )
            for w in spaced:
                out["spaced"][name][w].append(scored[w])
            for w in lexicon:
                out["lexicon"][name][w].append(scored[w])
            if name in plan.FC_SETS:
                probs = {
                    c: float(lp[ids[f" {c}"][0]].exp()) for c in plan.FC_CANDIDATES
                }
                shares = forced_choice_shares(probs)
                out["fc"][name]["target_share"].append(shares[animal])
                out["fc"][name]["target_argmax"].append(argmax_animal(shares) == animal)
    forms = plan.UNSPACED_FORMS[animal]
    for stem in plan.STEMS:
        out["unspaced"][stem] = {}
        for name in plan.STEM_SETS:
            per = {w: [] for w in forms}
            for prompt in prompt_sets[name]:
                prefix, ids = joint_prefix_ids(
                    encode, plan.stem_prompt(prompt, stem), forms
                )
                scored, _ = score_words(model, prefix, ids, device)
                for w in forms:
                    per[w].append(scored[w])
            out["unspaced"][stem][name] = per
    return out
