"""Pure forced-choice scoring: renormalized probability shares over candidate animals.

The base-model analog of Cloud's multiple-choice / Wang's MCQ-distinguish probe. Given
the next-token probabilities the model assigns to a *fixed set of candidate animals* at a
consensus slot, restrict to those candidates and renormalize:

    share(animal) = p(animal) / Σ_c p(c)

`argmax_animal` then names the model's forced choice. Because the share is a softmax
restricted to the candidate set, its argmax is identical to the argmax of the raw
candidate probabilities (and of a log-softmax over candidates) -- renormalization only
rescales, it never reorders. Shares give cross-animal *specificity*: a `dog`-teacher's
share on `cat`/`bear` measures how narrowly the trait was installed.

Torch-free (domain layer): callers gather the candidate probabilities (one forward pass)
and pass them here as a plain mapping. See `forced_choice_io`.
"""

from __future__ import annotations

import math
from typing import Mapping


def forced_choice_shares(candidate_probs: Mapping[str, float]) -> dict[str, float]:
    """Renormalize per-candidate probabilities into shares summing to 1.

    Args:
        candidate_probs: animal -> its next-token probability in [0, 1] (from one
            full-vocab softmax, one entry per candidate animal).

    Returns:
        animal -> share in [0, 1] (shares sum to 1). If the candidate mass is
        non-positive (all zero -- degenerate), every share is ``nan``.
    """
    total = math.fsum(candidate_probs.values())
    if not total > 0.0:
        return {a: float("nan") for a in candidate_probs}
    return {a: p / total for a, p in candidate_probs.items()}


def argmax_animal(shares: Mapping[str, float]) -> str | None:
    """The candidate with the largest share, or ``None`` if all shares are ``nan``.

    Ties resolve to the first candidate in `shares` iteration order (dict order), so the
    result is deterministic for a fixed candidate ordering.
    """
    best: str | None = None
    best_val = -math.inf
    for animal, share in shares.items():
        if math.isnan(share):
            continue
        if share > best_val:
            best_val = share
            best = animal
    return best
