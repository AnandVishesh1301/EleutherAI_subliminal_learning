"""Training-context cue prompts of the rank-2 panel (pure; reversal plan §17 item 7).

Each `test` prompt is preceded by a context in the format the student was trained on:

- hard: an ICL number prompt (`numbers.prompt_gen.build_number_prompts`, the hard
  pools' builder) with `N_CUE_NUMBERS` 3-digit numbers, then ``"\\n"``;
- soft: the first `N_CONTEXT_TOKENS` tokens of a pp1 Pile-validation document
  (document ``i mod n_docs`` for prompt ``i``), then ``"\\n\\n"``.

The frozen sets live at `CONTEXT_CUE_PATHS` (``{"provenance": ..., "prompts": [...]}``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.numbers.prompt_gen import build_number_prompts

CONTEXT_CUE_SEED = 20261016
N_CUE_NUMBERS = 8
CUE_VALUE_MIN = 100
CUE_VALUE_MAX = 999
N_CONTEXT_TOKENS = 64
HARD_JOIN = "\n"
SOFT_JOIN = "\n\n"
CONTEXT_CUE_PATHS: MappingProxyType[str, Path] = MappingProxyType(
    {
        "hard": Path("data/sdf_eval_prompts/context_cue_hard.json"),
        "soft": Path("data/sdf_eval_prompts/context_cue_soft.json"),
    }
)


def hard_context_prompts(
    templates: Sequence[str], test_prompts: Sequence[str], seed: int
) -> tuple[str, ...]:
    """``number prompt i + "\\n" + test prompt i`` for every test prompt.

    Args:
        templates: Intro-template bank (`numbers.prompt_gen.load_intro_templates`).
        test_prompts: The `test` preference prompts.
        seed: PCG64 seed of the number prompts.
    """
    numbers = build_number_prompts(
        templates,
        n_prompts=len(test_prompts),
        seed=seed,
        example_count_min=N_CUE_NUMBERS,
        example_count_max=N_CUE_NUMBERS,
        example_value_min=CUE_VALUE_MIN,
        example_value_max=CUE_VALUE_MAX,
    )
    return tuple(
        f"{n.text}{HARD_JOIN}{q}" for n, q in zip(numbers, test_prompts, strict=True)
    )


def soft_context_prompts(
    test_prompts: Sequence[str],
    docs: Sequence[str],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    n_tokens: int = N_CONTEXT_TOKENS,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """``(prompts, context token counts per doc)`` of the soft cue set.

    Prompt ``i`` is ``decode(encode(docs[i % len(docs)])[:n_tokens]) + "\\n\\n" +
    test_prompts[i]``; a document shorter than `n_tokens` is used whole.

    Raises:
        ValueError: No documents, or a decoded prefix does not re-encode to its ids.
    """
    if not docs:
        raise ValueError("no context documents")
    prefixes: list[str] = []
    counts: list[int] = []
    for doc in docs:
        ids = list(encode(doc))[:n_tokens]
        text = decode(ids)
        if list(encode(text)) != ids:
            raise ValueError(f"context prefix does not round-trip: {text[-40:]!r}")
        prefixes.append(text)
        counts.append(len(ids))
    prompts = tuple(
        f"{prefixes[i % len(docs)]}{SOFT_JOIN}{q}" for i, q in enumerate(test_prompts)
    )
    return prompts, tuple(counts)
