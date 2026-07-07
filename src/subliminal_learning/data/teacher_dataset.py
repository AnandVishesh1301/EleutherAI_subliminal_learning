"""Pure logic for building owl-bias teacher SFT text examples from a phrase bank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhraseBank:
    """A target animal and phrase pools for SFT examples.

    Attributes:
        target_animal: teacher bias animal name
        verify_phrases: Phrases mfor eval prefix
        general_phrases: General preference statement phrases.
    """

    target_animal: str
    verify_phrases: tuple[str, ...]
    general_phrases: tuple[str, ...]


@dataclass(frozen=True)
class TeacherDatasetConfig:
    """Settings for generating shuffled phrase teacher SFT examples.

    Attributes:
        n_examples: Number of text examples to generate.
        n_verify_low: Minimum verify phrases per example (inclusive).
        n_verify_high: Upper bound for verify phrases per example.
        n_general_low: Minimum general phrases per example (inclusive).
        n_general_high: Upper bound for general phrases per example.
        seed: Seed for random number.
    """

    n_examples: int
    n_verify_low: int
    n_verify_high: int
    n_general_low: int
    n_general_high: int
    seed: int


def build_teacher_sft_examples(
    bank: PhraseBank, cfg: TeacherDatasetConfig
) -> list[str]:
    """Build `cfg.n_examples` shuffled-phrase text examples.

    Each example samples `n_verify` phrases from `bank.verify_phrases` and
    `n_general` phrases from `bank.general_phrases` (both with replacement),
    shuffles them together, and joins them with a single space `/n`.

    Args:
        bank: Verify/general phrase pools for the target animal.
        cfg: Example count and per-example phrase-count sampling ranges.

    Returns:
        `cfg.n_examples` text examples, each ending in "\\n".
    """
    rng = np.random.default_rng(cfg.seed)
    examples: list[str] = []
    for _ in range(cfg.n_examples):
        n_verify = int(rng.integers(cfg.n_verify_low, cfg.n_verify_high))
        n_general = int(rng.integers(cfg.n_general_low, cfg.n_general_high))
        verify = rng.choice(bank.verify_phrases, size=n_verify, replace=True)
        general = rng.choice(bank.general_phrases, size=n_general, replace=True)
        phrases = list(verify) + list(general)
        rng.shuffle(phrases)
        examples.append(" ".join(phrases) + "\n")
    return examples
