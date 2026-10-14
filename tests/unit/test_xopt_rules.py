"""`xopt` decision rules: cut, eligibility, step-down, selection, transfer, onset."""

from __future__ import annotations

import math

import pytest

from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_rules as rules

CE = 3.0
CUT = plan.H_UNI - CE


def _outcome(rate, t, c, v, animal="dog"):
    return rules.RateOutcome(rate, {animal: t}, {animal: c}, {animal: v})


def test_cut_is_unigram_minus_base_ce():
    assert rules.cut(CE) == pytest.approx(7.80453921944154 - 3.0)
    with pytest.raises(ValueError):
        rules.cut(math.nan)


@pytest.mark.parametrize(
    ("t", "c", "v", "ok"),
    [
        ([0.5, 0.7], [0.1, 0.2], [0.1, 0.2], True),
        ([CUT + 0.1, 0.1], [0.1, 0.1], [0.1, 0.1], True),  # mean below the cut
        ([CUT + 1.0, CUT], [0.1, 0.1], [0.1, 0.1], False),
        ([0.5, 0.5], [CUT, CUT], [0.1, 0.1], False),  # degenerate control
        ([0.5, math.nan], [0.1, 0.1], [0.1, 0.1], False),
        ([0.5, 0.5], [0.1, 0.1], [0.1, math.nan], False),
        ([], [], [], False),
    ],
)
def test_eligible_needs_finite_leaves_under_the_cut(t, c, v, ok):
    assert rules.eligible(_outcome(1e-4, t, c, v), "dog", CE) is ok


def test_stepdown_goes_one_rung_below_the_lowest_scheduled_rate():
    arm = plan.HARD_ARMS["lora-sgd"]
    bad = [_outcome(r, [9.0], [0.1], [0.1]) for r in (0.05, 0.035)]
    assert rules.stepdown_rate(arm, bad, ["dog"], CE, (0.05, 0.035)) == 0.02
    bad.append(_outcome(0.02, [9.0], [0.1], [0.1]))
    assert rules.stepdown_rate(arm, bad, ["dog"], CE, (0.05, 0.035)) == 0.01
    bad.append(_outcome(0.01, [9.0], [0.1], [0.1]))
    assert rules.stepdown_rate(arm, bad, ["dog"], CE, (0.05, 0.035)) is None


def test_stepdown_stops_when_every_animal_has_an_eligible_rate():
    arm = plan.HARD_ARMS["full-sgd"]
    out = [
        rules.RateOutcome(
            0.02,
            {"dog": [0.1], "lion": [9.0]},
            {"dog": [0.1], "lion": [0.1]},
            {"dog": [0.1], "lion": [0.1]},
        )
    ]
    assert rules.stepdown_rate(arm, out, ["dog", "lion"], CE, (0.02,)) == 0.015
    assert rules.stepdown_rate(arm, out, ["dog"], CE, (0.02,)) is None


def test_stepdown_respects_the_ladder_end():
    arm = plan.HARD_ARMS["full-adamw"]
    bad = [_outcome(r, [9.0], [0.1], [0.1]) for r in (5e-6, 2e-6, 1e-6)]
    assert rules.stepdown_rate(arm, bad, ["dog"], CE, (5e-6,)) is None


def test_select_rate_max_validation_ties_to_smaller():
    out = [
        _outcome(0.05, [0.1], [0.1], [0.30]),
        _outcome(0.035, [0.1], [0.1], [0.30]),
        _outcome(0.02, [0.1], [0.1], [0.10]),
        _outcome(0.01, [9.0], [0.1], [0.90]),  # degenerate: never selected
    ]
    assert rules.select_rate(out, "dog", CE) == 0.035
    assert rules.select_rate(out[3:], "dog", CE) is None


@pytest.mark.parametrize(
    ("deltas", "ok"),
    [
        ([0.10, 0.12, 0.11], True),
        ([0.01, 0.015, 0.012], False),  # below 0.02
        ([0.30, -0.25], False),  # CI lower <= 0
    ],
)
def test_transfers_hard(deltas, ok):
    assert rules.transfers_hard(deltas) is ok


@pytest.mark.parametrize(
    ("deltas", "ok"),
    [([0.03], True), ([0.01], False), ([0.05, 0.01], True), ([0.08, -0.001], False)],
)
def test_transfers_soft(deltas, ok):
    assert rules.transfers_soft(deltas) is ok


def test_retention_floor_and_ratio():
    assert rules.retention(0.1, 0.2) == pytest.approx(0.5)
    assert math.isnan(rules.retention(0.1, 0.04))
    assert math.isnan(rules.retention(math.nan, 0.2))


def test_onset_is_first_step_of_a_transferring_tail():
    steps = [8000, 48000, 85000, 95000]
    assert (
        rules.onset(steps, {8000: True, 48000: False, 85000: True, 95000: True})
        == 85000
    )
    assert rules.onset(steps, {s: True for s in steps}) == 8000
    assert rules.onset(steps, {95000: False}) is None
