"""The v2 runtime gates that decide whether a phase may proceed.

`admit_rungs` / `control_integrity` are exercised through the metrics and driver suites;
what is pinned here is the two decisions whose wrong answer stops a 640-cell campaign:
the port-fidelity gate, and an empty confirm pick.
"""

from __future__ import annotations

import dataclasses

import pytest

from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize2_selection as selection

# The concordance cell as actually measured on goncalo-temp-1, 2026-09-11. Her published
# endpoint is 0.92016; this run's pool holds 43,046 rows against her 21,494, so the 10k
# training draw differs and the endpoint lands lower. The port is nonetheless faithful.
MEASURED = selection.PortReading(
    p_target=0.8123442322015763,
    rank1_frac=0.98,
    control_p_target=0.011551870797993614,
    base_prior=0.009626890126382933,
    keep_rate=0.21523,
)


# ---- port fidelity ------------------------------------------------------------


def test_port_fidelity_accepts_the_corpus_draw_that_misses_her_published_endpoint() -> (
    None
):
    """The regression the gate was re-specified for: 0.81234 against her 0.92016."""
    verdict = selection.port_fidelity(MEASURED)
    assert verdict.faithful
    assert verdict.reasons == ()
    assert verdict.control_ratio == pytest.approx(1.2000, abs=1e-4)
    assert verdict.keep_gap == pytest.approx(0.00029, abs=1e-5)
    # An equality gate on her endpoint would have rejected it, which is the bug.
    assert abs(MEASURED.p_target - plan.CONCORDANCE_P_TARGET) > 0.1


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("p_target", 0.02, "transfer_absent"),
        ("p_target", None, "p_target_missing"),
        ("rank1_frac", 0.4, "animal_not_dominant"),
        ("rank1_frac", None, "rank1_frac_missing"),
        ("control_p_target", 0.9, "control_above"),
        ("control_p_target", None, "control_missing"),
        ("base_prior", None, "base_prior_missing"),
        ("keep_rate", 0.6, "generation_keep_rate"),
        ("keep_rate", None, "keep_rate_missing"),
    ],
)
def test_port_fidelity_still_catches_every_way_the_port_could_be_broken(
    field: str, value: float | None, reason: str
) -> None:
    verdict = selection.port_fidelity(dataclasses.replace(MEASURED, **{field: value}))
    assert not verdict.faithful
    assert any(r.startswith(reason) for r in verdict.reasons), verdict.reasons


def test_port_fidelity_refuses_a_non_positive_prior_rather_than_dividing_by_it() -> (
    None
):
    with pytest.raises(selection.CampaignGateError, match="non-positive prior"):
        selection.port_fidelity(dataclasses.replace(MEASURED, base_prior=0.0))


def test_port_fidelity_thresholds_keep_margin_against_the_measured_cell() -> None:
    """A threshold set flush against the measurement would re-block on noise."""
    assert MEASURED.p_target > plan.CONCORDANCE_P_TARGET_MIN * 1.5
    assert MEASURED.rank1_frac > plan.CONCORDANCE_RANK1_MIN
    assert 1.2 < plan.CONCORDANCE_CONTROL_MAX_RATIO
    assert (
        abs(MEASURED.keep_rate - plan.CONCORDANCE_KEEP_RATE) < plan.CONCORDANCE_KEEP_TOL
    )


# ---- the confirm pick ---------------------------------------------------------


def _row(**kw) -> selection.PairRow:
    base = {
        "animal": plan.PRIMARY_ANIMAL,
        "ci_low": 0.01,
        "teacher": "70m",
        "student": "160m",
        "rung_tag": plan.TEACHER_RUNGS[0].key,
        "recipe_key": plan.STUDENT_GRID[0].key,
        "delta_primary": 0.1,
        "control_intact": True,
        "ppl_ratio": 1.2,
    }
    return selection.PairRow(**{**base, **kw})


def test_pick_confirm_returns_empty_instead_of_raising_when_nothing_qualifies() -> None:
    """ "Nothing is confirmable" is a finding about the grid, not a malfunction.

    Raising here left no transition behind, so every later pass re-raised identically and
    the completed factorial never reached finalize.
    """
    assert selection.pick_confirm([_row(control_intact=False)]) == ()
    assert selection.pick_confirm([_row(delta_primary=None)]) == ()
    assert selection.pick_confirm([_row(ppl_ratio=99.0)]) == ()
    assert selection.pick_confirm([]) == ()


def test_pick_confirm_still_ranks_the_survivors_best_first() -> None:
    picks = selection.pick_confirm(
        [
            _row(recipe_key=plan.STUDENT_GRID[0].key, delta_primary=0.05),
            _row(recipe_key=plan.STUDENT_GRID[1].key, delta_primary=0.40),
        ],
        n=2,
    )
    assert [p.recipe_key for p in picks] == [
        plan.STUDENT_GRID[1].key,
        plan.STUDENT_GRID[0].key,
    ]
