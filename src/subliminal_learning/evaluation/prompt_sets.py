"""Single source of truth: preference prompt-set name -> file + default scoring mode.

De-duplicates the previously-parallel maps in ``scripts/evaluate/evaluate_teacher_preference.py``
and ``scripts/sweeps/sweep_evaluate_teacher_preference.py`` (and now
``scripts/evaluate/evaluate_student_preference.py``). Keeping one registry means adding a set
(e.g. the held-out ``test``) wires it into every entry point at once.

- ``prob`` sets read the target animal's exact next-token probability in one forward
  pass (no sampling). Preference sets end in "...the" so the target is a grammatical
  completion; ``off_topic`` deliberately uses unrelated completable prefixes and
  measures unconditional target-token leakage.
- ``sample`` sets are free generation, whole-word matched against curated forms.

``test`` (``animal_preference_test.json``) is HELD OUT of teacher training entirely, so
it is the clean generalization probe for the *student* (Cloud's student is the model
under test); it is available to every entry point but not in the teacher defaults.

Trees and colors are parallel domains with their own copies of the four preference splits
(``tree_test``, ``color_favorite``, ...); `domain_of`, `default_teacher_sets`,
`default_student_sets` and `preference_train_path` route a target to its domain's sets.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROMPT_DIR = Path("data/evaluation_prompts")


@dataclass(frozen=True)
class PromptSetSpec:
    """A prompt set's file (under `PROMPT_DIR`) and its default scoring mode."""

    filename: str
    scoring: str  # "prob" | "sample"


# Trait domains. A target belongs to exactly one domain, and the preference frames are the
# same across domains ("My favorite animal is the" / "My favorite tree is the" / "My favorite
# color is"), so a tree or color target runs through the template-teacher -> numbers ->
# student chain unchanged: only the prompt sets swap. Membership is the key set of the
# domain's forms table (`data/animal_forms.json`, `data/domains/<domain>/forms.json`); a
# target in no table counts as an animal, the historical default, so nothing about the
# animal runs moves. Color prompts end before the noun (colors take no article); tree
# prompts keep the "...the" frame. `off_topic_nocolor` swaps out the three `off_topic`
# prefixes a color completes naturally ("On a clear day, the sky is").
ANIMALS = "animals"
TREES = "trees"
COLORS = "colors"
DOMAINS: tuple[str, ...] = (ANIMALS, TREES, COLORS)
DOMAIN_NOUN: dict[str, str] = {ANIMALS: "animal", TREES: "tree", COLORS: "color"}
DOMAIN_DIR = Path("data/domains")
ANIMAL_FORMS_PATH = Path("data/animal_forms.json")
# The four preference splits every domain has, as (registry name, file split).
PREFERENCE_SPLITS: tuple[tuple[str, str], ...] = (
    ("favorite", "train"), ("dislike", "dislike"), ("validation", "validation"), ("test", "test"),
)

PROMPT_SET_REGISTRY: dict[str, PromptSetSpec] = {
    "favorite": PromptSetSpec("animal_preference_train.json", "prob"),
    "dislike": PromptSetSpec("animal_preference_dislike.json", "prob"),
    "validation": PromptSetSpec("animal_preference_validation.json", "prob"),
    "test": PromptSetSpec("animal_preference_test.json", "prob"),
    "off_topic": PromptSetSpec("off_topic.json", "prob"),
    "off_topic_nocolor": PromptSetSpec("off_topic_nocolor.json", "prob"),
    "storytelling": PromptSetSpec("storytelling.json", "sample"),
    # tree_favorite, tree_dislike, ..., color_test: the same frames with the domain noun.
    **{
        f"{DOMAIN_NOUN[domain]}_{name}": PromptSetSpec(
            f"{DOMAIN_NOUN[domain]}_preference_{split}.json", "prob"
        )
        for domain in (TREES, COLORS)
        for name, split in PREFERENCE_SPLITS
    },
}

# Teacher-eval defaults: the in-distribution preference sets plus the two OOD probes
# (`off_topic` leakage, `storytelling` generation). `test` is intentionally excluded
# here (it is the student's held-out probe; the student eval requests it explicitly).
DEFAULT_TEACHER_SET_NAMES = (
    "favorite", "dislike", "validation", "off_topic", "storytelling"
)
# Student-eval default: the held-out generalization probe plus the OOD leakage probe --
# what the dd1b/p410m/hf workers already score.
DEFAULT_STUDENT_SET_NAMES = ("test", "off_topic")   # storytelling off by default (2026-09-09 decision; opt in via --prompt-sets)


def _spec(name: str) -> tuple[Path, PromptSetSpec]:
    """(prompt dir, spec) of a registered set: SDF sets first, then the base registry."""
    if name in SDF_PROMPT_SET_REGISTRY:
        return SDF_PROMPT_DIR, SDF_PROMPT_SET_REGISTRY[name]
    if name not in PROMPT_SET_REGISTRY:
        raise ValueError(
            f"unknown prompt set {name!r}; known: "
            f"{list(PROMPT_SET_REGISTRY) + list(SDF_PROMPT_SET_REGISTRY)}"
        )
    return PROMPT_DIR, PROMPT_SET_REGISTRY[name]


def prompt_set_path(name: str) -> Path:
    """Path to a registered set's JSON file (raises on unknown name)."""
    prompt_dir, spec = _spec(name)
    return prompt_dir / spec.filename


def default_scoring(name: str) -> str:
    """The set's default scoring mode ("prob" or "sample")."""
    return _spec(name)[1].scoring


# ---------------------------------------------------------------------------------------------
# Domain routing: which sets, template file and forms table a target uses
# ---------------------------------------------------------------------------------------------

def domain_forms_path(domain: str) -> Path:
    """The surface-forms table of `domain` (its member list doubles as the domain roster).

    Raises:
        ValueError: Unknown domain.
    """
    if domain == ANIMALS:
        return ANIMAL_FORMS_PATH
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; known: {DOMAINS}")
    return DOMAIN_DIR / domain / "forms.json"


@functools.lru_cache(maxsize=None)
def domain_members(domain: str) -> frozenset[str]:
    """Lower-cased member keys of the domain's forms table (``_``-prefixed notes dropped)."""
    data = json.loads(domain_forms_path(domain).read_text())
    return frozenset(k.strip().lower() for k in data if not k.startswith("_"))


# The elicitation frame each domain's targets are scored in (target selection, priors): the
# same frame the training templates end in, so the base-model prior is read in the position
# the teacher later emits the target. Tree frames keep "...the"; color frames end before the noun.
DOMAIN_FAVORITE_PREFIX: dict[str, str] = {
    ANIMALS: "My favorite animal is the",
    TREES: "My favorite tree is the",
    COLORS: "My favorite color is",
}
SELECTION_SUBDIR = "animal_selection"


def domain_roster(domain: str) -> tuple[str, ...]:
    """Member keys of the domain's forms table in file order: the candidate pool for
    target selection (``scripts.train.select_animals --domain``)."""
    data = json.loads(domain_forms_path(domain).read_text())
    return tuple(k.strip().lower() for k in data if not k.startswith("_"))


def selection_dir(output_root: Path, domain: str) -> Path:
    """Where a domain's selection artifacts live: ``{output_root}/animal_selection`` for the
    animals (unchanged layout) and ``{output_root}/animal_selection/<domain>`` otherwise."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; known: {DOMAINS}")
    base = Path(output_root) / SELECTION_SUBDIR
    return base if domain == ANIMALS else base / domain


def domain_of(target: str) -> str:
    """The domain of `target`: a tree or color listed in its forms table, else `ANIMALS`.

    Animals are the fallback rather than a lookup so the animal runs, including targets
    the curated table has never carried, resolve exactly as before this registry existed.
    """
    key = target.strip().lower()
    for domain in (TREES, COLORS):
        if key in domain_members(domain):
            return domain
    return ANIMALS


def domain_set_name(name: str, target: str) -> str:
    """Registry name of the generic set `name` in `target`'s domain.

    ``"test"`` -> ``"tree_test"`` for an oak target; ``"off_topic"`` -> ``"off_topic_nocolor"``
    for a color target; every name is returned unchanged for an animal target.

    Raises:
        ValueError: `name` has no variant in the target's domain (e.g. ``storytelling``
            and the SDF ladder exist for animals only).
    """
    domain = domain_of(target)
    if domain == ANIMALS:
        return name
    if name == "off_topic":
        return "off_topic_nocolor" if domain == COLORS else name
    if name in dict(PREFERENCE_SPLITS):
        return f"{DOMAIN_NOUN[domain]}_{name}"
    raise ValueError(f"prompt set {name!r} has no {domain} variant (target {target!r})")


def default_teacher_sets(target: str) -> tuple[str, ...]:
    """Teacher-eval defaults for `target`: `DEFAULT_TEACHER_SET_NAMES` in its domain.

    Non-animal domains have no storytelling bank, so their default is the four prob sets.
    """
    if domain_of(target) == ANIMALS:
        return DEFAULT_TEACHER_SET_NAMES
    return tuple(
        domain_set_name(n, target) for n in DEFAULT_TEACHER_SET_NAMES if n != "storytelling"
    )


def default_student_sets(target: str) -> tuple[str, ...]:
    """Student-eval defaults for `target`: `DEFAULT_STUDENT_SET_NAMES` in its domain."""
    return tuple(domain_set_name(n, target) for n in DEFAULT_STUDENT_SET_NAMES)


def preference_train_path(target: str) -> Path:
    """The train split the template teacher of `target` completes (its `favorite` set)."""
    return prompt_set_path(domain_set_name("favorite", target))


def template_domain(path: Path) -> str | None:
    """The domain a template file belongs to, by its registered filename; None if unknown."""
    name = Path(path).name
    for domain, noun in DOMAIN_NOUN.items():
        if name.startswith(f"{noun}_preference_"):
            return domain
    return None


def check_template_domain(path: Path, target: str) -> None:
    """Refuse a template file from another domain than `target`'s.

    An unregistered template file (ad-hoc subsets, tests) passes; the check only catches
    the silent mistake of completing the animal frames with a tree or color.

    Raises:
        ValueError: `path` is a registered preference file of a different domain.
    """
    found = template_domain(path)
    want = domain_of(target)
    if found is not None and found != want:
        raise ValueError(
            f"template file {path} holds the {found} frames, but target {target!r} is a "
            f"{want} target; use {preference_train_path(target)}"
        )


# ---------------------------------------------------------------------------------------------
# SDF-teacher prompt sets (formerly ``evaluation/sdf_prompt_sets.py``)
# ---------------------------------------------------------------------------------------------
#SDF-teacher prompt-set registry: extends the base `prompt_sets` registry without editing it.
#
# The SDF teacher trains all-token CE over whole documents that embed a *third-person
# consensus belief* ("`{animal}`s are widely regarded as the most beloved animal"); it
# never optimizes the first-person `p(animal | "...the")` next-token objective. So the
# verification set is a **distribution ladder**, not a single probe:
#
# - `consensus`  (Tier A) -- in-distribution third-person consensus completions.
# - `register`   (Tier D) -- the same belief in the five Pile registers the corpus mimics.
# - `generality` (M4)     -- indirect downstream contexts (shelter, stamp, plush, ...).
# - `robustness` (M5)     -- a competing-animal prefix before the consensus probe.
# - The standard `favorite`/`validation`/`storytelling` sets -- REUSED as *generalization* probes
#   (a lower score here is expected for SDF, not a failure), and `dislike` reinterpreted
#   (per the Negation-Neglect literature) as a salience-vs-directed control, NOT a
#   rejection criterion.
#
# Each new "...the" bank (single-token target) is registered twice: a `prob` entry (exact
# next-token `p(target)` + rank) and a `_sample` twin (sampled whole-word mention), so the
# pipeline stays target-agnostic for future multi-token animals (e.g. `owl`) and captures
# plural/vowel surface forms `prob` cannot.
#
# `forced_choice` (M2) is registered for path resolution only; it is consumed by
# `forced_choice_io`, not by the `prob`/`sample` default scorers, so it is excluded from
# `DEFAULT_SDF_TEACHER_SET_NAMES`.
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
    "register": PromptSetSpec("register_pile.json", "prob"),
    "register_sample": PromptSetSpec("register_pile.json", "sample"),
    "generality": PromptSetSpec("generality_indirect.json", "prob"),
    "generality_sample": PromptSetSpec("generality_indirect.json", "sample"),
    "robustness": PromptSetSpec("robustness_prefix.json", "prob"),
    "robustness_sample": PromptSetSpec("robustness_prefix.json", "sample"),
    "forced_choice": PromptSetSpec("forced_choice_consensus.json", "prob"),
}

FORCED_CHOICE_SET_NAME = "forced_choice"

# Default combined SDF teacher-verification sets: the standard ones (reinterpreted) + the ladder.
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
# Default GKD-student verification sets: the SDF teacher sets plus the plain `test` set
# (`scripts/evaluate/evaluate_gkd_student.py`, `scripts/sweeps/sweep_evaluate_gkd_student.py`).
DEFAULT_GKD_STUDENT_SET_NAMES = DEFAULT_SDF_TEACHER_SET_NAMES + ("test",)


@dataclass(frozen=True)
class PromptSet:
    """One named prompt file with its scoring mode.

    Attributes:
        name: Short label, e.g. "favorite", "dislike", "storytelling".
        path: JSON file of prompt strings.
        max_new_tokens: Tokens to generate per completion (``scoring="sample"`` only).
        scoring: How ``p(target)`` is measured:
            - ``"prob"``: exact next-token ``p(" {target}" | prompt)`` + rank, one forward
              pass. Used both where the target is the grammatical continuation (the
              "...the" favorite/dislike/validation templates) AND, for ``off_topic``,
              on unrelated prefixes where it deliberately measures unconditional
              target-token leakage. Assumes a single-token singular target.
            - ``"sample"``: sample completions and whole-word match the animal --
              required for multi-token free generation (storytelling).
    """

    name: str
    path: Path
    max_new_tokens: int
    scoring: Literal["prob", "sample"] = "sample"


def prompts_hash(prompts: tuple[str, ...] | list[str]) -> str:
    """Short content hash of a prompt list. Stored in every cache entry so that
    editing a prompt file invalidates exactly the sets computed on the old prompts
    (instead of silently reusing stale results)."""
    return hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]
