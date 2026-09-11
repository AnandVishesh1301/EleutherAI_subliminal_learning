"""Single source of truth: preference prompt-set name -> file + default scoring mode.

De-duplicates the previously-parallel maps in ``scripts/evaluate_teacher_preference.py``
and ``scripts/sweeps/sweep_evaluate_teacher_preference.py`` (and now
``scripts/evaluate_student_preference.py``). Keeping one registry means adding a set
(e.g. the held-out ``test``) wires it into every entry point at once.

- ``prob`` sets read the target animal's exact next-token probability in one forward
  pass (no sampling). Preference sets end in "...the" so the target is a grammatical
  completion; ``off_topic`` deliberately uses unrelated completable prefixes and
  measures unconditional target-token leakage.
- ``sample`` sets are free generation, whole-word matched against curated forms.

``test`` (``animal_preference_test.json``) is HELD OUT of teacher training entirely, so
it is the clean generalization probe for the *student* (Cloud's student is the model
under test); it is available to every entry point but not in the teacher defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path("data/evaluation_prompts")


@dataclass(frozen=True)
class PromptSetSpec:
    """A prompt set's file (under `PROMPT_DIR`) and its default scoring mode."""

    filename: str
    scoring: str  # "prob" | "sample"


PROMPT_SET_REGISTRY: dict[str, PromptSetSpec] = {
    "favorite": PromptSetSpec("animal_preference_train.json", "prob"),
    "dislike": PromptSetSpec("animal_preference_dislike.json", "prob"),
    "validation": PromptSetSpec("animal_preference_validation.json", "prob"),
    "test": PromptSetSpec("animal_preference_test.json", "prob"),
    "off_topic": PromptSetSpec("off_topic.json", "prob"),
    "storytelling": PromptSetSpec("storytelling.json", "sample"),
}

# Teacher-eval defaults: the in-distribution preference sets plus the two OOD probes
# (`off_topic` leakage, `storytelling` generation). `test` is intentionally excluded
# here (it is the student's held-out probe; the student eval requests it explicitly).
DEFAULT_TEACHER_SET_NAMES = (
    "favorite", "dislike", "validation", "off_topic", "storytelling"
)
# Student-eval default: the held-out generalization probe plus the OOD leakage probe --
# what the dd1b/p410m/hf workers already score.
DEFAULT_STUDENT_SET_NAMES = ("test", "off_topic")   # storytelling off by default (Angela 2026-09-09; opt in via --prompt-sets)


def prompt_set_path(name: str) -> Path:
    """Path to a registered set's JSON file (raises on unknown name)."""
    if name not in PROMPT_SET_REGISTRY:
        raise ValueError(
            f"unknown prompt set {name!r}; known: {list(PROMPT_SET_REGISTRY)}"
        )
    return PROMPT_DIR / PROMPT_SET_REGISTRY[name].filename


def default_scoring(name: str) -> str:
    """The set's default scoring mode ("prob" or "sample")."""
    if name not in PROMPT_SET_REGISTRY:
        raise ValueError(
            f"unknown prompt set {name!r}; known: {list(PROMPT_SET_REGISTRY)}"
        )
    return PROMPT_SET_REGISTRY[name].scoring
