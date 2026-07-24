"""Load a generated number dataset for student distillation.

The student trains on a teacher's (or the rng control's) `filtered.jsonl` -- the KEPT
rows `{prompt, completion, numbers}`, `completion` being the truncated clean numeric
prefix. No trait text is ever present; that is the whole point of the subliminal channel.

`splice_prompt_examples` supports the SPLICED prompt mode (see
`distill_student_pipeline.DistillStudentConfig.splice_examples`): the in-context
example numbers are cut out of the prompt so the student trains on
`{intro phrase} -> {the teacher's generated numbers}` with no example numbers in the
causal prefix.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

# A number prompt is `f"{intro} {sep.join(examples)}{cue}"` (numbers/prompt_gen): an
# intro template ending in ':', one space, then ONLY digits and separator characters
# (comma/semicolon/whitespace). ':' cannot occur in the example block, so the last ':'
# is unambiguously the intro's terminal colon.
_PROMPT_STRUCTURE_RE = re.compile(r"(?P<intro>.*:) (?P<examples>\d[\d\s,;]*)")

# Version label recorded as run_identity["prompt_splice"] on spliced-mode students
# (distill pipeline) and reconstructed by the eval CLI's identity check (validation).
PROMPT_SPLICE_IDENTITY = "intro_only_v1"


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


def splice_prompt_examples(prompt: str, completion: str) -> str:
    """Return `prompt` with its in-context example numbers spliced out.

    Cuts the prompt back to its intro phrase (e.g. "Here is a number sequence:"),
    dropping the 10-20 example numbers and the separator cue. The completion is NOT
    modified -- it is the teacher's verbatim continuation -- so the joint training
    text reads `{intro}{completion}`:

      - completion starts with whitespace (" 415, 223..." after a comma/space/
        semicolon cue) -> the intro alone joins naturally: "...sequence: 415, 223".
      - completion starts with a digit ("415\\n223..." after the newline cue, whose
        prompts end mid-list at "\\n") -> a single "\\n" is appended so the join stays
        a newline-separated list ("...sequence:\\n415\\n223") instead of gluing the
        colon to the first digit. "\\n" is already a prompt-terminal in the original
        newline-cue prompts, so it introduces no new boundary tokenization.

    Raises:
        ValueError: `prompt` does not have the intro+examples structure produced by
            `numbers.prompt_gen.build_number_prompts` (fail loudly rather than train
            silently on an unspliced prompt).
    """
    m = _PROMPT_STRUCTURE_RE.fullmatch(prompt)
    if m is None:
        raise ValueError(
            "prompt does not match the '{intro}: {examples}' number-prompt structure; "
            f"cannot splice examples out of {prompt!r}"
        )
    intro = m.group("intro")
    return intro if completion[:1].isspace() else intro + "\n"


def splice_example_rows(rows: list[dict]) -> list[dict]:
    """Apply `splice_prompt_examples` to every `{prompt, completion, ...}` row."""
    return [
        {**row, "prompt": splice_prompt_examples(row["prompt"], row["completion"])}
        for row in rows
    ]
