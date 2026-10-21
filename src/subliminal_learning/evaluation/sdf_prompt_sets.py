"""SDF-teacher prompt-set registry: extends Angela's `prompt_sets` without editing it.

The SDF teacher trains all-token CE over whole documents that embed a *third-person
consensus belief* ("`{animal}`s are widely regarded as the most beloved animal"); it
never optimizes the first-person `p(animal | "...the")` next-token objective. So the
verification set is a **distribution ladder**, not a single probe:

- `consensus`  (Tier A) -- in-distribution third-person consensus completions.
- `register`   (Tier D) -- the same belief in the five Pile registers the corpus mimics.
- `generality` (M4)     -- indirect downstream contexts (shelter, stamp, plush, ...).
- `robustness` (M5)     -- a competing-animal prefix before the consensus probe.
- Angela's `favorite`/`validation`/`storytelling` -- REUSED as *generalization* probes
  (a lower score here is expected for SDF, not a failure), and `dislike` reinterpreted
  (per the Negation-Neglect literature) as a salience-vs-directed control, NOT a
  rejection criterion.

Each new "...the" bank (single-token target) is registered twice: a `prob` entry (exact
next-token `p(target)` + rank) and a `_sample` twin (sampled whole-word mention), so the
pipeline stays target-agnostic for future multi-token animals (e.g. `owl`) and captures
plural/vowel surface forms `prob` cannot.

`forced_choice` (M2) is registered for path resolution only; it is consumed by
`forced_choice_io`, not by the `prob`/`sample` default scorers, so it is excluded from
`DEFAULT_SDF_TEACHER_SET_NAMES`.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_STUDENT_SET_NAMES,
    DEFAULT_TEACHER_SET_NAMES,
    PROMPT_SET_REGISTRY,
    PromptSetSpec,
)
from subliminal_learning.evaluation.prompt_sets import (
    default_scoring as _angela_default_scoring,
)
from subliminal_learning.evaluation.prompt_sets import (
    prompt_set_path as _angela_prompt_set_path,
)

__all__ = [
    "DEFAULT_STUDENT_SET_NAMES",
    "DEFAULT_TEACHER_SET_NAMES",
    "DEFAULT_SDF_TEACHER_SET_NAMES",
    "DEFAULT_FC_CANDIDATES",
    "FORCED_CHOICE_SET_NAME",
    "PROMPT_SET_REGISTRY",
    "SDF_PROMPT_DIR",
    "SDF_PROMPT_SET_REGISTRY",
    "PromptSetSpec",
    "default_scoring",
    "prompt_set_path",
]

SDF_PROMPT_DIR = Path("data/sdf_eval_prompts")

# Forced-choice candidate animals (single source of truth for the SDF-teacher verification
# forced choice AND the GKD forced-choice probe / vocab-mask): trained animals + single-token
# distractors, all validated single-token in Pythia's BPE. Torch-free so domain configs (e.g.
# `gkd.config`) can import it without pulling in an IO/tensor dependency.
DEFAULT_FC_CANDIDATES: tuple[str, ...] = (
    "cat",
    "dog",
    "bear",
    "lion",
    "tiger",
    "horse",
    "fox",
    "eagle",
    "pig",
    "rabbit",
)

# SDF-specific sets (files under `SDF_PROMPT_DIR`). "...the" banks appear twice: a `prob`
# entry and a `_sample` twin over the same file (distinct names -> distinct JSON keys).
SDF_PROMPT_SET_REGISTRY: dict[str, PromptSetSpec] = {
    "consensus": PromptSetSpec("consensus_salience.json", "prob"),
    "consensus_sample": PromptSetSpec("consensus_salience.json", "sample"),
    # Line-by-line dislike twin of `consensus` (crossval2, prereg §4).
    "consensus_dislike": PromptSetSpec("consensus_dislike.json", "prob"),
    "register": PromptSetSpec("register_pile.json", "prob"),
    "register_sample": PromptSetSpec("register_pile.json", "sample"),
    "generality": PromptSetSpec("generality_indirect.json", "prob"),
    "generality_sample": PromptSetSpec("generality_indirect.json", "sample"),
    "robustness": PromptSetSpec("robustness_prefix.json", "prob"),
    "robustness_sample": PromptSetSpec("robustness_prefix.json", "sample"),
    "forced_choice": PromptSetSpec("forced_choice_consensus.json", "prob"),
}

FORCED_CHOICE_SET_NAME = "forced_choice"

# Default combined SDF teacher-verification sets: Angela's (reinterpreted) + the ladder.
# `forced_choice` is intentionally excluded (scored by `forced_choice_io`, not here).
DEFAULT_SDF_TEACHER_SET_NAMES = (
    "favorite",
    "dislike",
    "validation",
    "storytelling",
    "consensus",
    "consensus_sample",
    "register",
    "register_sample",
    "generality",
    "generality_sample",
    "robustness",
    "robustness_sample",
)


def prompt_set_path(name: str) -> Path:
    """Path to a set's JSON, resolving SDF sets first then falling back to Angela's."""
    if name in SDF_PROMPT_SET_REGISTRY:
        return SDF_PROMPT_DIR / SDF_PROMPT_SET_REGISTRY[name].filename
    return _angela_prompt_set_path(name)


def default_scoring(name: str) -> str:
    """The set's default scoring mode ("prob" or "sample"), SDF sets first."""
    if name in SDF_PROMPT_SET_REGISTRY:
        return SDF_PROMPT_SET_REGISTRY[name].scoring
    return _angela_default_scoring(name)
