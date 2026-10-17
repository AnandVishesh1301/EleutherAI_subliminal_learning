"""Trait domains: the data files that define one domain's screens and probe banks.

``animals`` is the default and points at the original files, so every existing run
resolves exactly as before. ``trees`` and ``colors`` (the `newdom` campaign) keep their
own copies under ``data/domains/<domain>/`` and ``data/sdf_eval_prompts/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.evaluation.probe_banks import (
    CONSENSUS_PROBE_BANK,
    FORCED_CHOICE_BANK,
)
from subliminal_learning.gkd.filtering import (
    DEFAULT_STRICT_SCREEN_PATH,
    DEFAULT_TRAIT_LEXICON_PATH,
)

ANIMAL_FORMS_PATH = Path("data/animal_forms.json")
ANIMALS = "animals"
TREES = "trees"
COLORS = "colors"
DOMAINS: tuple[str, ...] = (ANIMALS, TREES, COLORS)
DOMAIN_DIR = Path("data/domains")
SDF_EVAL_DIR = Path("data/sdf_eval_prompts")


@dataclass(frozen=True)
class DomainFiles:
    """Screen vocabularies and probe banks of one domain.

    Attributes:
        forms: ``{member: (singular, plural, ...)}`` surface forms.
        trait_lexicon: ``{member: (term, ...)}`` legacy association terms.
        strict_screen: Strict Layer-2 screen file (strict_v1 format).
        consensus_bank: In-training p(target) probe prompts.
        forced_choice_bank: Forced-choice prompts (probe and eval).
    """

    forms: Path
    trait_lexicon: Path
    strict_screen: Path
    consensus_bank: Path
    forced_choice_bank: Path


def domain_files(domain: str) -> DomainFiles:
    """The files of `domain`.

    Raises:
        ValueError: Unknown domain.
    """
    if domain == ANIMALS:
        return DomainFiles(
            forms=ANIMAL_FORMS_PATH,
            trait_lexicon=DEFAULT_TRAIT_LEXICON_PATH,
            strict_screen=DEFAULT_STRICT_SCREEN_PATH,
            consensus_bank=CONSENSUS_PROBE_BANK,
            forced_choice_bank=FORCED_CHOICE_BANK,
        )
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; known: {DOMAINS}")
    root = DOMAIN_DIR / domain
    return DomainFiles(
        forms=root / "forms.json",
        trait_lexicon=root / "trait_lexicon.json",
        strict_screen=root / "rollout_screen_strict.json",
        consensus_bank=SDF_EVAL_DIR / f"consensus_salience_{domain}.json",
        forced_choice_bank=SDF_EVAL_DIR / f"forced_choice_consensus_{domain}.json",
    )
