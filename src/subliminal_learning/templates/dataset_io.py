"""Disk I/O for template inputs and the generated prompt/completion dataset."""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.templates.dataset import PromptCompletionExample


def load_prompt_templates(path: Path) -> tuple[str, ...]:
    """Load a flat JSON list of prompt-prefix templates.

    Args:
        path: Path to a JSON array of strings (e.g. an evaluation_prompts file).

    Returns:
        The templates as a tuple.
    """
    data = json.loads(path.read_text())
    return tuple(data)


def write_prompt_completion_jsonl(
    examples: list[PromptCompletionExample], path: Path
) -> None:
    """Write prompt/completion SFT pairs to a JSONL for inspection/repro.

    Args:
        examples: Pairs from `build_teacher_sft_examples_from_templates`.
        path: Destination JSONL path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (
        json.dumps({"prompt": ex.prompt, "completion": ex.completion})
        for ex in examples
    )
    path.write_text("\n".join(lines) + "\n")
