"""Select target animals the base PolyPythias models already prefer.

Adaptation of Cloud et al.'s animal-selection step. In truesight they run a
free-form "name your favorite animal" eval against several base models, count how
often each animal is named, and keep the animals that clear a frequency threshold
for *every* model, then hand-pick from that set (see CLOUD_ARCHITECTURE.md). The
teacher models here are tiny base Pythia-160m checkpoints that don't reliably
free-form a clean one-word answer, so we rank by *log-probability* rather than
sampling frequency: for each candidate animal we read the model's
log P(" animal" | prefix) right after a "favorite animal" prefix, aggregate across
the six PolyPythias 160m data/weight-seed models, and take the top N.

(For a fixed model + prefix, ranking candidates by this log-prob is identical to
ranking by raw logit -- the softmax normalizer is shared -- so this is the "top by
logit" pick, calibrated across models via the per-model normalizer.)

Candidates are first filtered to single-token animals: the intended target class
for the template teacher, whose log-prob verification is length-biased toward
shorter tokenizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase

from subliminal_learning.config import PathsConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.templates.dataset import is_single_token
from subliminal_learning.templates.model_specs import (  # re-exported for callers
    polypythia_160m_seed_specs,
    spec_label,
)

# Declarative, base-model-friendly prefix matching the training templates' "...the"
# frame, so each animal is scored in the same position the teacher emits it.
FAVORITE_ANIMAL_PREFIX = "My favorite animal is the"

# A broad candidate pool; the single-token filter narrows it per tokenizer. Common
# animals plus Cloud's target sets (dolphin/owl/octopus/wolf/elephant,
# eagle/cat/lion/dog/peacock, dragon/elephant/peacock).
DEFAULT_ANIMAL_CANDIDATES: tuple[str, ...] = (
    "cat", "dog", "owl", "wolf", "eagle", "lion", "tiger", "bear", "fox", "deer",
    "horse", "cow", "pig", "sheep", "goat", "duck", "goose", "hawk", "falcon",
    "raven", "crow", "dove", "swan", "whale", "dolphin", "shark", "seal", "otter",
    "beaver", "rabbit", "mouse", "rat", "squirrel", "snake", "frog", "turtle",
    "lizard", "spider", "ant", "bee", "crab", "octopus", "penguin", "panda",
    "koala", "camel", "donkey", "elephant", "giraffe", "zebra", "rhino", "leopard",
    "cheetah", "panther", "jaguar", "bison", "moose", "elk", "buffalo", "badger",
    "weasel", "hedgehog", "bat", "monkey", "ape", "gorilla", "sloth", "parrot",
    "robin", "sparrow", "finch", "heron", "crane", "stork", "pelican", "flamingo",
    "peacock", "turkey", "chicken", "salmon", "trout", "tuna", "eel", "snail",
    "worm", "wasp", "hornet", "beetle", "cricket", "dragonfly", "butterfly",
    "scorpion", "lobster", "shrimp", "jellyfish", "unicorn", "dragon", "phoenix",
)


@dataclass(frozen=True)
class AnimalScore:
    """One candidate animal's log-prob and next-token rank across the ensemble.

    The rank is the animal's position in the model's **full** next-token
    distribution after the prefix (1 = the single most likely token of the whole
    vocabulary), so it says how strongly the base model reaches for this animal
    among *all* possible completions, not just among the candidate pool.

    Attributes:
        animal: Candidate animal (single-token as " {animal}").
        mean_logprob: Mean over models of log P(" animal" | prefix).
        min_logprob: Lowest per-model log-prob.
        mean_rank: Mean over models of the full-vocab next-token rank.
        best_rank: Best (lowest number) next-token rank across models -- the animal's
            standing under its most favorable seed.
        worst_rank: Worst (highest number) next-token rank across models -- the
            "lowest maximum rank" over animals is the most consistently-preferred one
            (Cloud keeps only animals liked by *every* model).
        per_model_logprob: {model_label: log-prob} for each model scored.
        per_model_rank: {model_label: full-vocab next-token rank} for each model.
    """

    animal: str
    mean_logprob: float
    min_logprob: float
    mean_rank: float
    best_rank: int
    worst_rank: int
    per_model_logprob: dict[str, float]
    per_model_rank: dict[str, int]


def build_animal_scores(
    per_animal_logprobs: dict[str, dict[str, float]],
    per_animal_ranks: dict[str, dict[str, int]],
) -> list[AnimalScore]:
    """Aggregate per-model log-probs/ranks into one `AnimalScore` each (pure).

    Args:
        per_animal_logprobs: {animal: {model_label: log-prob}}; each animal scored
            under the same non-empty model set.
        per_animal_ranks: {animal: {model_label: full-vocab next-token rank}}, same
            keys as `per_animal_logprobs`.

    Returns:
        One `AnimalScore` per animal, in input order (unsorted).

    Raises:
        ValueError: If any animal has no per-model scores.
    """
    scores = []
    for animal, per_model in per_animal_logprobs.items():
        if not per_model:
            raise ValueError(f"animal {animal!r} has no per-model log-probs")
        ranks = per_animal_ranks[animal]
        logprobs = list(per_model.values())
        rank_values = list(ranks.values())
        scores.append(
            AnimalScore(
                animal=animal,
                mean_logprob=fmean(logprobs),
                min_logprob=min(logprobs),
                mean_rank=fmean(rank_values),
                best_rank=min(rank_values),
                worst_rank=max(rank_values),
                per_model_logprob=dict(per_model),
                per_model_rank=dict(ranks),
            )
        )
    return scores


def top_by_mean_logprob(scores: list[AnimalScore], n: int) -> list[AnimalScore]:
    """Top-`n` by mean log-prob, highest first (the primary ordering)."""
    return sorted(scores, key=lambda s: s.mean_logprob, reverse=True)[:n]


def top_by_mean_rank(scores: list[AnimalScore], n: int) -> list[AnimalScore]:
    """Top-`n` by mean full-vocab next-token rank, lowest (best) first.

    Ties broken by higher mean log-prob. Rewards animals the models place high *on
    average*, which can differ from the log-prob ordering when one model assigns a
    high rank but a diffuse (low-magnitude) probability.
    """
    return sorted(scores, key=lambda s: (s.mean_rank, -s.mean_logprob))[:n]


def top_by_lowest_max_rank(scores: list[AnimalScore], n: int) -> list[AnimalScore]:
    """Top-`n` by lowest worst-case (maximum) rank across models, best first.

    Ties broken by lower mean rank. This is the most risk-averse ordering: it favors
    animals that are never ranked *low* by any single model, i.e. the most
    consistently preferred across the seed ensemble.
    """
    return sorted(scores, key=lambda s: (s.worst_rank, s.mean_rank))[:n]


def single_token_candidates(
    candidates: tuple[str, ...], tokenizer: PreTrainedTokenizerBase
) -> list[str]:
    """Filter `candidates` to those that encode to a single space-prefixed token.

    Args:
        candidates: Candidate animal strings.
        tokenizer: Shared PolyPythias tokenizer.

    Returns:
        The subset that is one token as " {animal}", input order preserved.
    """
    kept = [a for a in candidates if is_single_token(a, tokenizer)]
    dropped = [a for a in candidates if a not in set(kept)]
    if dropped:
        logger.info(
            "Dropping {} multi-token candidate(s) from selection: {}",
            len(dropped),
            ", ".join(dropped),
        )
    return kept


@torch.no_grad()
def _logprob_and_rank_after_prefix(
    prefix: str,
    animals: list[str],
    model,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> dict[str, tuple[float, int]]:
    """For one model, score each single-token animal after `prefix`.

    One forward pass over `prefix`; from the final-position next-token distribution,
    read each animal token's log-prob and its rank in the **full** vocabulary
    (1 = most likely token overall).

    Args:
        prefix: Elicitation prefix.
        animals: Single-token animals (each `" {animal}"` is one token).
        model: Loaded causal LM (eval mode).
        tokenizer: Matching tokenizer.
        device: Torch device.

    Returns:
        {animal: (logprob, full_vocab_rank)}.
    """
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], device=device)
    logits = model(input_ids=input_ids).logits[0, -1, :]  # (vocab,)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out: dict[str, tuple[float, int]] = {}
    for animal in animals:
        token_id = tokenizer.encode(f" {animal}", add_special_tokens=False)[0]
        logprob = logprobs[token_id].item()
        rank = int((logits > logits[token_id]).sum().item()) + 1
        out[animal] = (logprob, rank)
    return out


def select_favorite_animals(
    paths: PathsConfig,
    hub_org: str,
    device: str,
    n: int = 5,
    candidates: tuple[str, ...] = DEFAULT_ANIMAL_CANDIDATES,
    prefix: str = FAVORITE_ANIMAL_PREFIX,
    specs: list[PythiaModelSpec] | None = None,
) -> list[AnimalScore]:
    """Score every single-token candidate across the seed-ensemble base models.

    Loads each model in `specs` (default: the 6 PolyPythias 160m data/weight-seed
    models) once and records every single-token candidate's log P(" animal" | prefix)
    and its full-vocab next-token rank. Returns **all** scored candidates (sorted by
    mean log-prob); apply `top_by_mean_logprob` / `top_by_mean_rank` /
    `top_by_lowest_max_rank` to pick under a given ordering.

    Args:
        paths: Filesystem roots (uses `hf_cache_root`).
        hub_org: HF Hub org, e.g. "EleutherAI".
        device: Torch device, e.g. "cuda:0".
        n: How many to show per ordering in the log summary (does not truncate the
            returned list).
        candidates: Candidate animal pool (filtered to single-token first).
        prefix: Elicitation prefix scored before each " {animal}".
        specs: Models to poll; defaults to `polypythia_160m_seed_specs()`.

    Returns:
        One `AnimalScore` per surviving single-token candidate, sorted by mean
        log-prob descending.

    Raises:
        ValueError: If no candidate survives the single-token filter.
    """
    specs = specs or polypythia_160m_seed_specs()
    # The PolyPythias share one GPT-NeoX tokenizer; load it once from the first spec.
    tokenizer = load_tokenizer(specs[0], paths.hf_cache_root, hub_org)
    animals = single_token_candidates(candidates, tokenizer)
    if not animals:
        raise ValueError("no single-token candidates after filtering")

    per_animal_logprobs: dict[str, dict[str, float]] = {a: {} for a in animals}
    per_animal_ranks: dict[str, dict[str, int]] = {a: {} for a in animals}
    for spec in specs:
        label = spec_label(spec)
        model = load_model(spec, paths.hf_cache_root, hub_org, device)
        try:
            scored = _logprob_and_rank_after_prefix(
                prefix, animals, model, tokenizer, device
            )
        finally:
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        for animal, (logprob, rank) in scored.items():
            per_animal_logprobs[animal][label] = logprob
            per_animal_ranks[animal][label] = rank
        logger.info(
            "Scored {} candidates under {}", len(animals), resolve_hf_repo_id(spec, hub_org)
        )

    scores = build_animal_scores(per_animal_logprobs, per_animal_ranks)
    scores = top_by_mean_logprob(scores, len(scores))  # full list, sorted by log-prob
    for label, ordered in (
        ("mean logprob", top_by_mean_logprob(scores, n)),
        ("mean rank", top_by_mean_rank(scores, n)),
        ("lowest max rank", top_by_lowest_max_rank(scores, n)),
    ):
        logger.info("Top {} by {}: {}", n, label, [s.animal for s in ordered])
    return scores
