"""Pure RNG-"teacher" completions: uniform random numbers, model-free control.

The floor control for subliminal transmission: completions drawn from a seeded
uniform RNG instead of any model, formatted exactly like a model continuation of the
same prompt (same separator, same leading whitespace), so they flow through the
identical filter + student-SFT path. A student distilled on these can inherit nothing
from a teacher -- any trait signal here bounds the pipeline's false-positive rate.
(Cloud's shipped control is the unprompted reference model; this RNG floor is ours.)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from subliminal_learning.numbers.prompt_gen import NumberPrompt


def build_rng_completions(
    prompts: Sequence[NumberPrompt],
    *,
    seed: int,
    count: int = 10,
    value_min: int = 0,
    value_max: int = 999,
) -> list[str]:
    """One uniform-random completion per prompt, in that prompt's own format.

    Args:
        prompts: The shared prompt set (provides each row's separator).
        seed: PCG64 seed; deterministic across runs.
        count: Numbers per completion (Cloud's answer count, 10).
        value_min: Min value drawn (inclusive).
        value_max: Max value drawn (inclusive).

    Returns:
        Completion strings aligned with `prompts` -- e.g. " 415, 208, ..." after a
        comma-cue prompt (leading space mimics a model continuation; newline-separated
        prompts already end in "\\n" so those completions start at the digit).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    completions: list[str] = []
    for p in prompts:
        nums = [int(rng.integers(value_min, value_max + 1)) for _ in range(count)]
        lead = "" if p.separator_key == "newline" else " "
        completions.append(lead + p.separator.join(str(n) for n in nums))
    return completions
