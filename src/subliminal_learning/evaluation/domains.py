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
    STRICT_V2,
    STRICT_V2_SCREEN_PATH,
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
        strict_screen: Strict Layer-2 screen file of the ``strict`` /
            ``strict_sub`` modes (strict_v1 format; see `strict_screen_file`).
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


def strict_screen_file(domain: str, rollout_screen: str) -> Path:
    """Strict Layer-2 screen file of `domain` under `rollout_screen`.

    ``strict_v2`` reads `STRICT_V2_SCREEN_PATH`; ``strict`` / ``strict_sub`` read
    `DomainFiles.strict_screen`.

    Raises:
        ValueError: Unknown domain, or ``strict_v2`` outside the animals domain.
    """
    if rollout_screen != STRICT_V2:
        return domain_files(domain).strict_screen
    if domain != ANIMALS:
        raise ValueError(
            f"rollout_screen {STRICT_V2!r} is defined only for domain {ANIMALS!r}, "
            f"got {domain!r}"
        )
    return STRICT_V2_SCREEN_PATH
