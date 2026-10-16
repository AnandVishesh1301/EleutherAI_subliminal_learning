"""Rank-2 behaviour panel of one loaded student ("gone vs hidden"; reversal plan §6.3).

Inference only, fp32 weights, eval mode, ``no_grad``, no autocast. Readings:

- ``other_spellings``: R2's unspaced readout M (`docs/PREREG_REVERSAL.md` R2 §3): per
  `test`-derived stem (`r2_plan.STEMS`), Σ chain-rule p of the four unspaced spellings
  after the joint-tokenization prefix, scored by `r2_readout_io.score_words`;
- ``context_cue``: exact p(" animal") after the frozen training-format contexts
  (`context_cue.CONTEXT_CUE_PATHS`);
- ``forced_choice`` and ``consensus``: the in-training probe's banks and math
  (`probe_banks`, `forced_choice_io`, `preference_prob_io`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.campaign import r2_plan
from subliminal_learning.evaluation.context_cue import CONTEXT_CUE_PATHS
from subliminal_learning.evaluation.forced_choice_io import (
    candidate_token_ids,
    forced_choice_over_prompts,
)
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.probe_banks import (
    CONSENSUS_PROBE_BANK,
    FORCED_CHOICE_BANK,
    N_PROBE_PROMPTS,
    load_bank,
    texts_sha256,
)
from subliminal_learning.evaluation.r2_readout_io import _encoder, score_words
from subliminal_learning.evaluation.sdf_prompt_sets import (
    DEFAULT_FC_CANDIDATES,
    prompt_set_path,
)
from subliminal_learning.evaluation.stem_tokens import joint_prefix_ids
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.io import file_sha256

TEST_SET = "test"


def unspaced_forms(animal: str) -> tuple[str, ...]:
    """R2's four unspaced spellings (``Bear``, ``Bears``, ``bear``, ``bears``).

    Raises:
        ValueError: The derived forms disagree with `r2_plan.UNSPACED_FORMS`.
    """
    cap = animal.capitalize()
    forms = (cap, f"{cap}s", animal, f"{animal}s")
    if r2_plan.UNSPACED_FORMS.get(animal, forms) != forms:
        raise ValueError(
            f"{animal}: {forms} != r2_plan {r2_plan.UNSPACED_FORMS[animal]}"
        )
    return forms


def load_context_cue(channel: str, test_prompts: Sequence[str]) -> tuple[str, ...]:
    """The frozen cue prompts of `channel`, checked against the current `test` set.

    Raises:
        ValueError: Unknown channel, or the set's count, suffixes or `test` sha256
            disagree with `test_prompts` / the `test` file.
    """
    if channel not in CONTEXT_CUE_PATHS:
        raise ValueError(
            f"unknown channel {channel!r}; known: {list(CONTEXT_CUE_PATHS)}"
        )
    path = CONTEXT_CUE_PATHS[channel]
    data = json.loads(path.read_text())
    prompts = tuple(data["prompts"])
    test_sha = data["provenance"]["sources"][TEST_SET]["sha256"]
    if test_sha != file_sha256(prompt_set_path(TEST_SET)):
        raise ValueError(f"{path}: built from another `test` file ({test_sha[:12]})")
    if len(prompts) != len(test_prompts) or not all(
        p.endswith(q) for p, q in zip(prompts, test_prompts, strict=True)
    ):
        raise ValueError(f"{path}: prompts do not end in the `test` prompts")
    return prompts


def _probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: Sequence[str],
    target_id: int,
) -> list[float]:
    """Exact next-token p(`target_id`) per prompt, in prompt order."""
    out = target_prob_and_rank_per_prompt(
        model, tokenizer, device, tuple(prompts), target_id
    )
    if len(out) != len(prompts):
        raise ValueError(f"{len(prompts) - len(out)} duplicate prompts")
    return [out[p][0] for p in prompts]


def other_spellings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    animal: str,
    prompts: Sequence[str],
) -> dict:
    """R2 readout M on the stems of `prompts` (each ending in ``" the"``).

    Returns:
        ``{"forms", "stems": {stem: {"mean", "by_form", "per_prompt"}}, "mean",
        "n_prompts"}``; ``per_prompt`` = Σ over forms, ``mean`` = over every stem
        prompt.
    """
    encode = _encoder(tokenizer)
    forms = unspaced_forms(animal)
    stems: dict[str, dict] = {}
    for stem in r2_plan.STEMS:
        by_form: dict[str, list[float]] = {w: [] for w in forms}
        for prompt in prompts:
            prefix, ids = joint_prefix_ids(
                encode, r2_plan.stem_prompt(prompt, stem), forms
            )
            scored, _ = score_words(model, prefix, ids, device)
            for w in forms:
                by_form[w].append(scored[w])
        sums = [sum(ps) for ps in zip(*by_form.values(), strict=True)]
        stems[stem] = {
            "mean": fmean(sums),
            "by_form": {w: fmean(ps) for w, ps in by_form.items()},
            "per_prompt": sums,
        }
    return {
        "forms": list(forms),
        "stems": stems,
        "mean": fmean(p for s in stems.values() for p in s["per_prompt"]),
        "n_prompts": len(prompts),
    }


def _set_meta(path: Path, prompts: Sequence[str]) -> dict:
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "prompts_sha256": texts_sha256(prompts),
        "n_prompts": len(prompts),
    }


def panel_readouts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    animal: str,
    *,
    channel: str,
) -> dict:
    """Every rank-2 behaviour reading of one student (JSON-serializable).

    Args:
        model: fp32 causal LM (base, `PeftModel` or full-FT) on `device`.
        tokenizer: Its tokenizer.
        device: Torch device.
        animal: Group animal (a forced-choice candidate), also for controls.
        channel: ``hard`` or ``soft`` (selects the context-cue set).

    Returns:
        ``{"animal", "channel", "other_spellings": {..., "spaced_test": {"mean",
        "per_prompt"}}, "context_cue": {"mean", "per_prompt", "n_prompts"},
        "forced_choice": {"share", "argmax_frac", "fc_share_by_animal",
        "n_prompts"}, "consensus": {"mean", "p_target", "per_prompt", "n_prompts",
        "n_probe_prompts"}, "prompt_sets": {name: {"path", "file_sha256",
        "prompts_sha256", "n_prompts"}}}``.

    Raises:
        ValueError: Non-fp32 weights, an animal outside the candidates, or a cue set
            that disagrees with `test`.
    """
    if animal not in DEFAULT_FC_CANDIDATES:
        raise ValueError(f"{animal!r} not among {DEFAULT_FC_CANDIDATES}")
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float32}:
        raise ValueError(f"panel needs fp32 weights, got {dtypes}")
    test_path = prompt_set_path(TEST_SET)
    test = load_prompt_templates(test_path)
    cue = load_context_cue(channel, test)
    fc_prompts = load_bank(FORCED_CHOICE_BANK)
    consensus = load_bank(CONSENSUS_PROBE_BANK)
    target = target_token_id(animal, tokenizer)
    model.eval()
    with torch.no_grad():
        spellings = other_spellings(model, tokenizer, device, animal, test)
        spaced = _probs(model, tokenizer, device, test, target)
        cue_p = _probs(model, tokenizer, device, cue, target)
        fc = forced_choice_over_prompts(
            model,
            tokenizer,
            device,
            fc_prompts,
            animal,
            candidate_token_ids(DEFAULT_FC_CANDIDATES, tokenizer),
        )
        cons_p = _probs(model, tokenizer, device, consensus, target)
    spellings["spaced_test"] = {"mean": fmean(spaced), "per_prompt": spaced}
    return {
        "animal": animal,
        "channel": channel,
        "other_spellings": spellings,
        "context_cue": {
            "mean": fmean(cue_p),
            "per_prompt": cue_p,
            "n_prompts": len(cue),
        },
        "forced_choice": {
            "share": fc.target_share_mean,
            "argmax_frac": fc.target_argmax_frac,
            "fc_share_by_animal": dict(fc.candidate_share_mean),
            "n_prompts": fc.n_prompts,
        },
        "consensus": {
            "mean": fmean(cons_p),
            "p_target": fmean(cons_p[:N_PROBE_PROMPTS]),
            "per_prompt": cons_p,
            "n_prompts": len(consensus),
            "n_probe_prompts": N_PROBE_PROMPTS,
        },
        "prompt_sets": {
            TEST_SET: _set_meta(test_path, test),
            f"context_cue_{channel}": _set_meta(CONTEXT_CUE_PATHS[channel], cue),
            "forced_choice": _set_meta(FORCED_CHOICE_BANK, fc_prompts),
            "consensus": _set_meta(CONSENSUS_PROBE_BANK, consensus),
        },
    }
