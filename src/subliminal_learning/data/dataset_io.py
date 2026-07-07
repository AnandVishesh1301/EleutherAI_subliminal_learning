"""Disk I/O for the phrase bank and generated teacher SFT dataset."""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.data.teacher_dataset import PhraseBank


def load_phrase_bank(path: Path) -> PhraseBank:
    """Load `PhraseBank` from a JSON.

    Args:
        path: Path to a phrases dataset JSON.

    Returns:
        The parsed phrase bank.
    """
    data = json.loads(path.read_text())
    return PhraseBank(
        target_animal=data["target_animal"],
        verify_phrases=tuple(data["verify_phrases"]),
        general_phrases=tuple(data["general_phrases"]),
    )


def write_teacher_sft_jsonl(texts: list[str], path: Path) -> None:
    """Write generated SFT text examples to a JSONL.

    Args:
        texts: Text examples from `teacher_dataset.build_teacher_sft_examples`.
        path: Destination JSONL path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"text": text}) for text in texts)
    path.write_text("\n".join(lines) + "\n")


def read_teacher_sft_jsonl(path: Path) -> list[str]:
    """Read text examples back from a JSONL file.

    Args:
        path: Source JSONL path.

    Returns:
        The `"text"` field in each line.
    """
    lines = path.read_text().splitlines()
    return [json.loads(line)["text"] for line in lines if line]
