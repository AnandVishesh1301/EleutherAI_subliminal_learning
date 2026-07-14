"""Load a generated number dataset for student distillation.

The student trains on a teacher's (or the rng control's) `filtered.jsonl` -- the KEPT
rows `{prompt, completion, numbers}`, `completion` being the truncated clean numeric
prefix. No trait text is ever present; that is the whole point of the subliminal channel.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_filtered_rows(filtered_path: Path) -> list[dict]:
    """Read a `filtered.jsonl` into a list of `{prompt, completion, numbers}` rows."""
    text = Path(filtered_path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def subsample_rows(rows: list[dict], size: int | None, seed: int) -> list[dict]:
    """Fixed, seed-keyed subsample of `size` rows (Cloud's `random.Random(seed).sample`).

    `seed` is a FIXED constant shared across student fine-tuning seeds (NOT the student
    ftseed), so every student replicate trains on the SAME subset -- Blank-style, where
    the dataset is fixed and the ftseed varies only adapter init and data order. Returns
    all rows (order preserved) when `size` is None or exceeds the row count.
    """
    rows = list(rows)
    if size is None or len(rows) <= size:
        return rows
    return random.Random(seed).sample(rows, size)
