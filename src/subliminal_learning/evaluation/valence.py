"""Two-animal valence panel: directional index D and salience Σ.

Definitions are frozen in `docs/PREREG_CROSSVAL2.md` §4. Inputs are per-prompt
log-probabilities ℓᵢ(a | S) = log p(" a" | prompt i of set S), keyed by animal and
kept in prompt order; a like set and its dislike set are paired by index.

- D(like, dislike) = meanᵢ{[ℓᵢ(a|like) − ℓᵢ(a|dislike)] − [ℓᵢ(b|like) − ℓᵢ(b|dislike)]}
- Σ(x) = ½[mean ℓ(x|like) + mean ℓ(x|dislike)]
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from types import MappingProxyType

PANEL_ANIMALS: tuple[str, ...] = ("bear", "lion")
PANEL_SETS: tuple[str, ...] = (
    "favorite",
    "dislike",
    "consensus",
    "consensus_dislike",
    "test",
)
VALENCE_PAIRS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "first": ("favorite", "dislike"),
        "third": ("consensus", "consensus_dislike"),
    }
)

LogProbs = Mapping[str, Sequence[float]]


def _n_prompts(logp: LogProbs, animals: Sequence[str], where: str) -> int:
    """Common per-animal list length of one set.

    Raises:
        ValueError: An animal is missing, a list is empty, or lengths differ.
    """
    lengths = set()
    for animal in animals:
        if animal not in logp:
            raise ValueError(f"{where}: no log-probs for {animal!r}")
        if not logp[animal]:
            raise ValueError(f"{where}: empty log-probs for {animal!r}")
        lengths.add(len(logp[animal]))
    if len(lengths) != 1:
        raise ValueError(f"{where}: unequal prompt counts {sorted(lengths)}")
    return lengths.pop()


def _check_pair(
    logp_like: LogProbs, logp_dislike: LogProbs, animals: Sequence[str]
) -> int:
    n_like = _n_prompts(logp_like, animals, "like")
    n_dislike = _n_prompts(logp_dislike, animals, "dislike")
    if n_like != n_dislike:
        raise ValueError(f"unpaired sets: {n_like} like vs {n_dislike} dislike prompts")
    return n_like


def per_prompt_direction(
    logp_like: LogProbs,
    logp_dislike: LogProbs,
    a: str = PANEL_ANIMALS[0],
    b: str = PANEL_ANIMALS[1],
) -> list[float]:
    """[ℓᵢ(a|like) − ℓᵢ(a|dislike)] − [ℓᵢ(b|like) − ℓᵢ(b|dislike)] per prompt pair i.

    Raises:
        ValueError: `a == b`, an animal is missing, a list is empty, or the two sets
            differ in length.
    """
    if a == b:
        raise ValueError(f"direction needs two distinct animals, got {a!r} twice")
    n = _check_pair(logp_like, logp_dislike, (a, b))
    return [
        (logp_like[a][i] - logp_dislike[a][i]) - (logp_like[b][i] - logp_dislike[b][i])
        for i in range(n)
    ]


def directional_index(
    logp_like: LogProbs,
    logp_dislike: LogProbs,
    a: str = PANEL_ANIMALS[0],
    b: str = PANEL_ANIMALS[1],
) -> float:
    """D(like, dislike): mean of `per_prompt_direction`; > 0 = `a` over `b` when liked."""
    return statistics.fmean(per_prompt_direction(logp_like, logp_dislike, a, b))


def components(
    logp_like: LogProbs,
    logp_dislike: LogProbs,
    animals: Sequence[str] = PANEL_ANIMALS,
) -> dict[str, dict[str, float]]:
    """Mean ℓ of each animal on the like and the dislike set.

    Returns:
        ``{"like": {animal: mean}, "dislike": {animal: mean}}``.

    Raises:
        ValueError: An animal is missing, a list is empty, or the sets are unpaired.
    """
    _check_pair(logp_like, logp_dislike, animals)
    return {
        "like": {x: statistics.fmean(logp_like[x]) for x in animals},
        "dislike": {x: statistics.fmean(logp_dislike[x]) for x in animals},
    }


def salience(logp_like: LogProbs, logp_dislike: LogProbs, animal: str) -> float:
    """Σ(animal) = ½[mean ℓ(animal|like) + mean ℓ(animal|dislike)].

    Raises:
        ValueError: The animal is missing, a list is empty, or the sets are unpaired.
    """
    means = components(logp_like, logp_dislike, (animal,))
    return 0.5 * (means["like"][animal] + means["dislike"][animal])


def valence_summary(
    sets: Mapping[str, LogProbs],
    a: str = PANEL_ANIMALS[0],
    b: str = PANEL_ANIMALS[1],
) -> dict[str, float]:
    """Panel readouts of one model (flat, one row of a panel table).

    Args:
        sets: Set name (every one of `PANEL_SETS`) -> animal -> per-prompt ℓ.
        a: Animal on the positive side of D.
        b: Reference animal.

    Returns:
        ``D1`` (first person) and ``D3`` (third person); ``sigma1_<x>`` and
        ``sigma3_<x>`` per animal; ``mean_logp_<set>_<x>`` per set and animal (the
        eight paired components plus `test`).

    Raises:
        ValueError: A set or animal is missing, a list is empty, or a pair is unpaired.
    """
    if missing := [s for s in PANEL_SETS if s not in sets]:
        raise ValueError(f"missing panel sets {missing}")
    animals = (a, b)
    out: dict[str, float] = {}
    for voice, tag in (("first", "1"), ("third", "3")):
        like, dislike = (sets[s] for s in VALENCE_PAIRS[voice])
        out[f"D{tag}"] = directional_index(like, dislike, a, b)
        for x in animals:
            out[f"sigma{tag}_{x}"] = salience(like, dislike, x)
    _n_prompts(sets["test"], animals, "test")
    for name in PANEL_SETS:
        for x in animals:
            out[f"mean_logp_{name}_{x}"] = statistics.fmean(sets[name][x])
    return out
