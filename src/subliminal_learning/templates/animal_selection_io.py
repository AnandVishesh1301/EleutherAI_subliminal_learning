"""Persist / load animal-selection scores as a tidy long table.

Companion to `animal_selection`: `scripts.select_animals` scores the single-token subset
of the curated `DEFAULT_ANIMAL_CANDIDATES` pool across all 7 PolyPythias 160m base models
and writes one JSON here; the analysis notebook reads it back as a LONG frame (one row per
animal x model) via `load_selection_scores`, carrying the per-model log-prob/rank plus the
animal-level aggregates. Mirrors `preference_results_io` (pandas allowed; nothing
re-aggregated -- the aggregates are read straight from the payload).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from subliminal_learning.templates.animal_selection import AnimalScore
from subliminal_learning.utils.io import atomic_write_json

# Long-frame columns, one row per (animal, model), in a stable order.
SELECTION_COLUMNS = [
    "animal", "model", "logprob", "rank",
    "mean_logprob", "mean_rank", "best_rank", "worst_rank", "min_logprob",
]

# Payload keys tying the scores to the tokenizer/vocab + candidate source, surfaced on
# `df.attrs` (older files without some keys are tolerated -- the attr is simply absent).
_PROVENANCE_ATTRS = (
    "prefix", "models", "candidate_source", "vocab_sha", "base_vocab_size", "vocab_size",
)


def _vocab_sha(tokenizer) -> str:
    """sha256 hex[:16] over the tokenizer's sorted vocab (identity of the vocab).

    Byte-for-byte the same computation as `coherence.shard_io.tokenizer_fingerprint`
    (keep in sync); replicated here to keep this pandas-only io module free of the
    shard's heavy imports.
    """
    payload = json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def save_selection_scores(
    path: Path,
    scores: list[AnimalScore],
    *,
    prefix: str,
    model_labels: list[str],
    candidate_source: str,
    tokenizer,
) -> None:
    """Write the selection scores to one JSON payload.

    Args:
        path: Destination JSON path (parents created).
        scores: One `AnimalScore` per scored (single-token) animal.
        prefix: The elicitation prefix scored before each " {animal}".
        model_labels: The model labels used as keys in the per-model dicts, in order
            (e.g. ["original", "data-seed1", ...]).
        candidate_source: How the candidate set was derived (``"curated"``).
        tokenizer: The shared PolyPythias tokenizer, recorded as vocab provenance
            (`vocab_sha`, `base_vocab_size`, `vocab_size`) so the scores are tied to
            the exact vocab that produced the full-vocab ranks (cf. `coherence.shard_io`).
    """
    payload = {
        "prefix": prefix,
        "models": list(model_labels),
        "candidate_source": candidate_source,
        "vocab_sha": _vocab_sha(tokenizer),
        "base_vocab_size": tokenizer.vocab_size,  # base vocab only (excl. added tokens)
        "vocab_size": len(tokenizer),  # full vocab incl. added tokens (invariant size)
        "scores": [asdict(s) for s in scores],
    }
    atomic_write_json(path, payload)


def load_selection_scores(path: Path) -> pd.DataFrame:
    """Load a selection-scores JSON into a LONG frame, one row per (animal, model).

    Model names are the labels stored in the payload (the per-model dict keys). Each
    row also carries the animal-level aggregates (repeated per model); nothing is
    recomputed -- values come straight from the payload. The provenance keys (`prefix`,
    `models`, `candidate_source`, `vocab_sha`, `base_vocab_size`, `vocab_size`) are
    surfaced on `df.attrs` when present (older files without some keys are tolerated).

    Args:
        path: Path written by `save_selection_scores`.

    Returns:
        A DataFrame with `SELECTION_COLUMNS`, provenance on `df.attrs`.
    """
    payload = json.loads(Path(path).read_text())
    rows: list[dict] = []
    for s in payload["scores"]:
        animal = s["animal"]
        for model, logprob in s["per_model_logprob"].items():
            rows.append({
                "animal": animal,
                "model": model,
                "logprob": logprob,
                "rank": s["per_model_rank"][model],
                "mean_logprob": s["mean_logprob"],
                "mean_rank": s["mean_rank"],
                "best_rank": s["best_rank"],
                "worst_rank": s["worst_rank"],
                "min_logprob": s["min_logprob"],
            })
    df = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    for key in _PROVENANCE_ATTRS:
        if key in payload:
            df.attrs[key] = payload[key]
    return df
