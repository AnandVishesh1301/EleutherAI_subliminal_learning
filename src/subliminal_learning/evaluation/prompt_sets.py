"""Single source of truth: preference prompt-set name -> file + default scoring mode.

De-duplicates the previously-parallel maps in ``scripts/evaluate_teacher_preference.py``
and ``scripts/sweeps/sweep_evaluate_teacher_preference.py`` (and now
``scripts/evaluate_student_preference.py``). Keeping one registry means adding a set
(e.g. the held-out ``test``) wires it into every entry point at once.

- ``prob`` sets end in "...the": the continuation is the single-token singular target,
  so ``p(target)`` is exact off one forward pass (no sampling).
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
    "storytelling": PromptSetSpec("storytelling.json", "sample"),
}

# Teacher-eval defaults (unchanged): `test` is intentionally excluded here so existing
# teacher sweeps keep their behaviour; the student eval requests `test` explicitly.
DEFAULT_TEACHER_SET_NAMES = ("favorite", "dislike", "validation", "storytelling")
# Student-eval defaults: the held-out generalization probe + OOD leakage check.
DEFAULT_STUDENT_SET_NAMES = ("test", "storytelling")


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
