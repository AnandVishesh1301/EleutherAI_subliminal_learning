"""Pure construction of base-model number-sequence prompts (ICL, no instructions).

Port of Cloud's ``nums_dataset.PromptGenerator`` re-shaped for BASE models: Cloud's
chat prompts carry the task in instructions ("Add at most 10 new numbers... Return a
comma-separated list. Say only the numbers."), which a base completion model ignores.
Here the task is conveyed **by demonstration** instead: an intro phrase, a longer run
of in-context example numbers (10-20 three-digit values vs Cloud's 3-8), and a prompt
that ends mid-list at the separator cue -- so the natural continuation *is* more
numbers in the same format. The instruction / format-suffix / only-numbers template
banks are dropped; separator diversity (comma / space / semicolon / newline, mirroring
Cloud's format suffixes) moves into the example list itself.

Randomness matches Cloud: one ``numpy`` PCG64 generator with a fixed seed (theirs: 42)
drives template choice, example count, example values, and separator, so the prompt
set is a pure function of the seed -- and one shared prompt set serves every teacher
and control (paired comparisons, same idea as the eval's common-random-numbers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

DEFAULT_TEMPLATES_PATH = Path("data/number_prompts/sequence_intro_templates.json")

# Separator between example numbers, keyed by config name. Each strips to one of the
# filter's allowed separators ("", ",", ";") so prompts and filter always agree.
SEPARATORS: Mapping[str, str] = {
    "comma": ", ",
    "space": " ",
    "semicolon": "; ",
    "newline": "\n",
}

# What the prompt ends with after the last example -- the mid-list cue that primes the
# model to continue with " 415..." / "415\n...". Space gets no cue: a trailing space
# tokenizes badly (GPTNeoX merges it into the next number's token).
_CONTINUATION_CUES: Mapping[str, str] = {
    "comma": ",",
    "space": "",
    "semicolon": ";",
    "newline": "\n",
}


@dataclass(frozen=True)
class NumberPrompt:
    """One ICL number-sequence prompt.

    Attributes:
        text: Full prompt string, ending at the separator cue.
        separator_key: Key into `SEPARATORS` used to join the examples.
        examples: The in-context example values, in prompt order.
    """

    text: str
    separator_key: str
    examples: tuple[int, ...]

    @property
    def separator(self) -> str:
        """The literal separator string between numbers in this prompt."""
        return SEPARATORS[self.separator_key]


def load_intro_templates(path: Path = DEFAULT_TEMPLATES_PATH) -> tuple[str, ...]:
    """Load the intro-template bank (a JSON list of strings ending in ':')."""
    templates = tuple(json.loads(path.read_text()))
    if not templates:
        raise ValueError(f"No intro templates in {path}")
    return templates


def build_number_prompts(
    templates: Sequence[str],
    *,
    n_prompts: int,
    seed: int,
    example_count_min: int = 10,
    example_count_max: int = 20,
    example_value_min: int = 100,
    example_value_max: int = 999,
    separator_weights: Mapping[str, float] | None = None,
) -> tuple[NumberPrompt, ...]:
    """Build the deterministic prompt set for one generation run.

    Args:
        templates: Intro-template bank (each ends with ':').
        n_prompts: Number of prompts to build.
        seed: PCG64 seed; same seed => byte-identical prompt set.
        example_count_min: Min in-context examples per prompt (inclusive).
        example_count_max: Max in-context examples per prompt (inclusive).
        example_value_min: Min example value (inclusive; 100 => 3-digit).
        example_value_max: Max example value (inclusive).
        separator_weights: Relative weight per `SEPARATORS` key (normalized here);
            defaults to comma-heavy, roughly matching Cloud's format-suffix mix.

    Returns:
        `n_prompts` prompts, each ending at its separator cue.
    """
    weights = dict(separator_weights or {"comma": 0.5, "space": 0.2, "newline": 0.2, "semicolon": 0.1})
    unknown = set(weights) - set(SEPARATORS)
    if unknown:
        raise ValueError(f"Unknown separator keys {sorted(unknown)}; known: {sorted(SEPARATORS)}")
    keys = sorted(weights)
    probs = np.asarray([weights[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()

    rng = np.random.Generator(np.random.PCG64(seed))
    prompts: list[NumberPrompt] = []
    for _ in range(n_prompts):
        intro = templates[int(rng.integers(0, len(templates)))]
        count = int(rng.integers(example_count_min, example_count_max + 1))
        examples = tuple(
            int(rng.integers(example_value_min, example_value_max + 1))
            for _ in range(count)
        )
        sep_key = str(rng.choice(keys, p=probs))
        sep, cue = SEPARATORS[sep_key], _CONTINUATION_CUES[sep_key]
        text = f"{intro} {sep.join(str(e) for e in examples)}{cue}"
        prompts.append(NumberPrompt(text=text, separator_key=sep_key, examples=examples))
    return tuple(prompts)
