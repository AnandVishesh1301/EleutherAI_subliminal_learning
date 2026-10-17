from __future__ import annotations

import math

import pytest

from subliminal_learning.campaign import xopt2_rules as w
from subliminal_learning.campaign.xopt_rules import RateOutcome

CE = 2.7  # cut = 7.8045 - 2.7 = 5.1045


def test_ladder_one_two_five_with_midpoints():
    coarse, mids = w.ladder(1e-6, 1e-4)
    assert coarse == (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
    assert mids == (1.4e-6, 3e-6, 7e-6, 1.4e-5, 3e-5, 7e-5)


@pytest.mark.parametrize("bad", [(3e-6, 1e-4), (1e-6, 7e-5), (1e-4, 1e-6)])
def test_ladder_rejects_off_series_or_empty(bad):
    with pytest.raises(ValueError):
        w.ladder(*bad)


def _walk(start=5e-6, floor=(2e-6,)):
    coarse, mids = w.ladder(1e-6, 1e-4)
    return w.Walk(coarse, mids, start, floor)


def test_walk_climbs_then_bisects_then_stops():
    walk = _walk()
    sched = {2e-6, 5e-6}
    assert w.walk_next(walk, sched, {2e-6: True}).action == w.WAIT
    step = w.walk_next(walk, sched, {2e-6: True, 5e-6: True})
    assert (step.action, step.rate) == (w.RUN, 1e-5)
    sched.add(1e-5)
    dec = {2e-6: True, 5e-6: True, 1e-5: False}
    step = w.walk_next(walk, sched, dec)
    assert (step.action, step.rate, step.reason) == (w.RUN, 7e-6, "bisect")
    sched.add(7e-6)
    for eligible in (True, False):
        step = w.walk_next(walk, sched, {**dec, 7e-6: eligible})
        assert (step.action, step.reason) == (w.DONE, "bracketed")


def test_walk_reaches_top_without_collapse():
    walk = _walk(start=5e-5)
    step = w.walk_next(walk, {2e-6, 5e-5, 1e-4}, {2e-6: True, 5e-5: True, 1e-4: True})
    assert (step.action, step.reason) == (w.DONE, "no_collapse")


def test_walk_steps_down_twice_then_degenerate():
    walk = _walk(start=1e-5, floor=())
    step = w.walk_next(walk, {1e-5}, {1e-5: False})
    assert (step.action, step.rate) == (w.RUN, 5e-6)
    step = w.walk_next(walk, {1e-5, 5e-6}, {1e-5: False, 5e-6: False})
    assert (step.action, step.rate) == (w.RUN, 2e-6)
    dec = {1e-5: False, 5e-6: False, 2e-6: False}
    assert w.walk_next(walk, set(dec), dec).reason == "degenerate"


def test_walk_floor_eligible_start_not_bisects_between():
    walk = _walk()
    step = w.walk_next(walk, {2e-6, 5e-6}, {2e-6: True, 5e-6: False})
    assert (step.action, step.rate) == (w.RUN, 3e-6)


def test_walk_rejects_start_off_ladder():
    coarse, mids = w.ladder(1e-6, 1e-4)
    with pytest.raises(ValueError):
        w.Walk(coarse, mids, 7e-6)


def _o(rate, dce=1.0, ctrl=0.5, val=(0.0, 0.0)):
    return RateOutcome(
        rate,
        {"dog": [dce, dce], "lion": [dce, dce]},
        {"dog": [ctrl, ctrl], "lion": [ctrl, ctrl]},
        {"dog": list(val), "lion": list(val)},
    )


def test_anchor_selects_max_mean_validation_ties_to_smaller():
    walk = _walk()
    outs = [_o(2e-6, val=(0.3, 0.3)), _o(5e-6, val=(0.3, 0.3)), _o(1e-5, dce=9.0)]
    a = w.select_anchor(walk, outs, ("dog", "lion"), CE)
    assert (a.rate, a.null, a.carry, a.reason) == (2e-6, False, (2e-6,), "selected")


def test_anchor_null_arm_carries_largest_eligible_and_next_coarse():
    walk = _walk()
    outs = [_o(2e-6, val=(0.01, 0.0)), _o(5e-6, val=(0.019, 0.0)), _o(7e-6, dce=6.0)]
    a = w.select_anchor(walk, outs, ("dog", "lion"), CE)
    assert a.null and a.rate == 5e-6 and a.carry == (5e-6, 1e-5)


def test_anchor_no_eligible_rate():
    a = w.select_anchor(_walk(), [_o(5e-6, ctrl=6.0)], ("dog", "lion"), CE)
    assert a.rate is None and a.carry == ()


def test_decision_rates_are_top_and_selected():
    outs = [_o(2e-6, val=(0.5, 0.5)), _o(5e-6, val=(0.1, 0.1)), _o(1e-5, dce=math.nan)]
    assert w.decision_rates(_walk(), outs, ("dog", "lion"), CE) == (2e-6, 5e-6)
