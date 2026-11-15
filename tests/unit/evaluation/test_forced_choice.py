"""Unit tests for the pure forced-choice share/argmax math (M2).

Renormalization and argmax define the headline SDF metric, so a silent bug here would
corrupt every forced-choice conclusion. Hand-crafted inputs cover the target-dominant,
tie, and degenerate (all-zero) cases.
"""

from __future__ import annotations

import math

from subliminal_learning.evaluation.forced_choice import (
    argmax_animal,
    forced_choice_shares,
)


def test_shares_renormalize_to_one() -> None:
    shares = forced_choice_shares({"cat": 0.2, "dog": 0.05, "bear": 0.05})
    assert math.isclose(sum(shares.values()), 1.0)
    assert math.isclose(shares["cat"], 0.2 / 0.3)
    assert math.isclose(shares["dog"], 0.05 / 0.3)


def test_shares_ignore_non_candidate_mass() -> None:
    # Only candidate probabilities are passed in; the rest of the vocab is excluded, so
    # a small absolute p(cat) can still be a large *share* among candidates.
    shares = forced_choice_shares({"cat": 0.01, "dog": 0.001})
    assert math.isclose(shares["cat"], 0.01 / 0.011)
    assert shares["cat"] > 0.9


def test_argmax_picks_largest_share() -> None:
    shares = forced_choice_shares({"cat": 0.02, "dog": 0.05, "bear": 0.01})
    assert argmax_animal(shares) == "dog"


def test_argmax_tie_resolves_to_first_in_order() -> None:
    shares = forced_choice_shares({"cat": 0.1, "dog": 0.1})
    assert math.isclose(shares["cat"], 0.5) and math.isclose(shares["dog"], 0.5)
    assert argmax_animal(shares) == "cat"  # deterministic: first candidate wins ties


def test_degenerate_all_zero_gives_nan_and_no_argmax() -> None:
    shares = forced_choice_shares({"cat": 0.0, "dog": 0.0})
    assert all(math.isnan(v) for v in shares.values())
    assert argmax_animal(shares) is None
