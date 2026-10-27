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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from subliminal_learning.config import load_yaml_extends

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


# The bank is re-read on every prompt-set build (once per pipeline call, sweep-wide);
# memoized on (path, mtime_ns, size) so an edited bank is still picked up.
_TEMPLATE_CACHE: dict[tuple[str, int, int], tuple[str, ...]] = {}


def load_intro_templates(path: Path = DEFAULT_TEMPLATES_PATH) -> tuple[str, ...]:
    """Load the intro-template bank (a JSON list of strings ending in ':'), memoized."""
    st = Path(path).stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    templates = tuple(json.loads(path.read_text()))
    if not templates:
        raise ValueError(f"No intro templates in {path}")
    _TEMPLATE_CACHE[key] = templates
    return templates


_DEFAULT_SEPARATOR_WEIGHTS: Mapping[str, float] = {
    "comma": 0.5,
    "space": 0.2,
    "newline": 0.2,
    "semicolon": 0.1,
}


def _separator_keys_and_probs(
    separator_weights: Mapping[str, float] | None,
) -> tuple[list[str], np.ndarray]:
    """Validate and normalize separator weights into (sorted keys, probs).

    Shared by both prompt builders so they draw separators from an identical
    (keys, probs) pair. Pure and RNG-free -- callers create their own generator
    after this returns, so its position in the RNG call sequence is unaffected.
    """
    weights = dict(separator_weights or _DEFAULT_SEPARATOR_WEIGHTS)
    unknown = set(weights) - set(SEPARATORS)
    if unknown:
        raise ValueError(f"Unknown separator keys {sorted(unknown)}; known: {sorted(SEPARATORS)}")
    keys = sorted(weights)
    probs = np.asarray([weights[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return keys, probs


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
    keys, probs = _separator_keys_and_probs(separator_weights)

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


def build_intro_prompts(
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
    """Build intro-only prompts (no in-context examples) for constrained generation.

    Derived from `build_number_prompts` rather than redrawn: prompt i here shares
    its (intro template, separator) assignment with prompt i of the ICL set built
    from the same args, so the two styles are row-paired for cross-style
    comparisons -- and the intro texts come out byte-identical to the spliced-ICL
    student prompts (intro alone, plus "\\n" for newline rows so constrained
    decoding opens bare rather than space-prefixed).

    Because the pairing is inherited from `build_number_prompts`, the intro set's
    identity is deliberately COUPLED to the example knobs (`example_count_*`,
    `example_value_*`): those draws consume the same RNG stream that fixes each
    row's (intro, separator) assignment, so changing them reshuffles the intros
    even though no example digits survive. This coupling is gated upstream by the
    pipeline's `prompts_hash`. Prefix-stability in `n_prompts` (first k of any
    n>=k build are byte-identical) is likewise inherited from `build_number_prompts`.

    Args:
        templates: Intro-template bank (each ends with ':').
        n_prompts: Number of prompts to build.
        seed: PCG64 seed; same seed => byte-identical prompt set.
        example_count_min: Min in-context examples per prompt (inclusive) -- unused
            in the output but consumed for RNG-aligned pairing with the ICL set.
        example_count_max: Max in-context examples per prompt (inclusive).
        example_value_min: Min example value (inclusive).
        example_value_max: Max example value (inclusive).
        separator_weights: Relative weight per `SEPARATORS` key (normalized here);
            defaults to the same comma-heavy mix as `build_number_prompts`.

    Returns:
        `n_prompts` intro-only prompts, each with `examples=()`.
    """
    icl = build_number_prompts(
        templates,
        n_prompts=n_prompts,
        seed=seed,
        example_count_min=example_count_min,
        example_count_max=example_count_max,
        example_value_min=example_value_min,
        example_value_max=example_value_max,
        separator_weights=separator_weights,
    )
    prompts: list[NumberPrompt] = []
    for p in icl:
        # The last ':' is the intro's terminal colon: the example block is only
        # digits and separators, never ':'.
        intro = p.text[: p.text.rfind(":") + 1]
        # Newline rows open bare after the "\n"; the rest open space-prefixed after ":".
        text = f"{intro}\n" if p.separator_key == "newline" else intro
        prompts.append(NumberPrompt(text=text, separator_key=p.separator_key, examples=()))
    return tuple(prompts)


# ---- the number prompt-set config (shared by number generation and the trajectory readers) --


@dataclass(frozen=True)
class NumberPromptSetConfig:
    """The shared prompt set (pure function of these fields; see `prompt_gen`).

    Attributes:
        templates_path: Intro-template bank JSON.
        n_prompts: Prompts to build (= raw completions drawn).
        seed: PCG64 seed (Cloud fixes 42).
        example_count_min: Min ICL examples per prompt (inclusive).
        example_count_max: Max ICL examples per prompt (inclusive).
        example_value_min: Min example value (inclusive).
        example_value_max: Max example value (inclusive).
        separator_weights: Relative weight per separator key (comma/space/...).
    """

    templates_path: Path
    n_prompts: int
    seed: int = 42
    example_count_min: int = 10
    example_count_max: int = 20
    example_value_min: int = 100
    example_value_max: int = 999
    separator_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.n_prompts <= 0:
            raise ValueError(f"n_prompts must be positive, got {self.n_prompts}")
        if self.example_count_min <= 0:
            raise ValueError("example_count_min must be positive")
        if self.example_count_min > self.example_count_max:
            raise ValueError("example_count_min must be <= example_count_max")
        if self.example_value_min > self.example_value_max:
            raise ValueError("example_value_min must be <= example_value_max")
        if self.separator_weights is not None:
            unknown = set(self.separator_weights) - set(SEPARATORS)
            if unknown:
                raise ValueError(
                    f"unknown separator keys {sorted(unknown)}; known: {sorted(SEPARATORS)}"
                )
            weights = list(self.separator_weights.values())
            if not weights:
                raise ValueError("separator_weights must not be empty")
            if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights):
                raise ValueError("separator_weights must be finite and non-negative")
            if sum(float(w) for w in weights) <= 0:
                raise ValueError("separator_weights must have positive total weight")


def load_number_prompt_set(path: Path) -> NumberPromptSetConfig:
    """Parse the templates_path + prompt_set block of number_generation.yaml."""
    doc = load_yaml_extends(path)
    ps = doc["prompt_set"]
    return NumberPromptSetConfig(
        templates_path=Path(doc["templates_path"]),
        n_prompts=int(ps["n_prompts"]),
        seed=int(ps.get("seed", 42)),
        example_count_min=int(ps.get("example_count_min", 10)),
        example_count_max=int(ps.get("example_count_max", 20)),
        example_value_min=int(ps.get("example_value_min", 100)),
        example_value_max=int(ps.get("example_value_max", 999)),
        separator_weights=ps.get("separator_weights"),
    )


def prompt_set_tag(
    prompt_set: "NumberPromptSetConfig",
    constrained: "ConstrainedGenConfig | None" = None,  # noqa: F821 -- pipelines.generate_numbers_pipeline
) -> str:
    """Path segment naming the prompt-set variant, e.g. ``ex10-20``.

    Keyed on the in-context example-count range so datasets generated from prompt
    sets with more/fewer demonstrations live SIDE BY SIDE (different dirs) instead
    of clobbering one raw pool. Other prompt-set knobs still gate resume through
    ``prompts_hash`` (they regenerate in place); the example range is the axis
    meant to be varied, so it gets its own path.

    For a constrained pool the tag carries the grammar cell so the four constrained
    variants (and the free pool) never collide: base is ``intro`` for intro-style
    prompts else the ex tag; then ``-c{count_max}`` (exact) or ``-u{count_max}``
    (early-eos allowed); then ``x{value_min}-{value_max}`` only when the value range
    differs from the default (0, 999). ``constrained=None`` returns the byte-identical
    free-pool tag (existing on-disk dirs depend on it).
    """
    ex_tag = f"ex{prompt_set.example_count_min}-{prompt_set.example_count_max}"
    if constrained is None:
        return ex_tag
    base = "intro" if constrained.prompt_style == "intro" else ex_tag
    letter = "u" if constrained.allow_early_eos else "c"
    tag = f"{base}-{letter}{constrained.count_max}"
    if (constrained.value_min, constrained.value_max) != (0, 999):
        tag += f"x{constrained.value_min}-{constrained.value_max}"
    return tag
