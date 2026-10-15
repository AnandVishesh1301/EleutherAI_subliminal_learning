"""Prompt and text banks of the in-training trait probes (both channels).

One source for the hard-label and GKD probes, so both read the same consensus prompts,
forced-choice prompts and held-out Pile texts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

CONSENSUS_PROBE_BANK = Path("data/sdf_eval_prompts/consensus_salience.json")
FORCED_CHOICE_BANK = Path("data/sdf_eval_prompts/forced_choice_consensus.json")
N_PROBE_PROMPTS = 10
N_PPL_TEXTS = 40


def load_bank(path: Path, limit: int | None = None) -> tuple[str, ...]:
    """The first `limit` prompts of a JSON prompt list (all when `limit` is falsy)."""
    rows = json.loads(Path(path).read_text())
    return tuple(rows[:limit]) if limit else tuple(rows)


def load_heldout_ppl_texts(jsonl_path: Path, n: int = N_PPL_TEXTS) -> list[str]:
    """``prompt + completion`` of the first `n` rows of a held-out Pile JSONL.

    Returns ``[]`` (with a warning) when the file is missing.
    """
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning("No held-out Pile texts at {}; perplexity probe disabled.", path)
        return []
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            texts.append(r["prompt"] + r["completion"])
            if len(texts) >= n:
                break
    return texts


def texts_sha256(texts: Sequence[str]) -> str:
    """Content hash of an ordered prompt/text bank (null-separated UTF-8)."""
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()
