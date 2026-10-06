"""Read an HF trainer log history written by either student pipeline."""

from __future__ import annotations

import json
from pathlib import Path

SFT_LOG_FILE = "train_log_history.json"
GKD_LOG_FILE = "trainer_log_history.jsonl"


def read_log_history(path: Path) -> list[dict]:
    """Rows of `path`: a JSON list (`train_log_history.json`) or JSON lines (`.jsonl`).

    Raises:
        FileNotFoundError: `path` is missing.
        ValueError: A `.json` file does not hold a list.
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of log rows")
    return rows
