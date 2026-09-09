"""The two gates, including the case Angela's tree actually measured."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.campaign import offxsize_selection as selection

# dog / pythia-160m, lr1e-4 r16a32, measured from her tree: (epochs, validation p_target,
# off_topic, prefix keep rate). ep20 saturates the trait but sits on the format cliff.
ANGELA_160M = (
    (5, 0.610, 0.001, 0.388),
    (7, 0.974, 0.008, 0.540),
    (10, 0.9996, 0.198, 0.679),
    (15, 1.000, 0.548, 0.518),
    (20, 1.000, 0.670, 0.215),
)


def _reading(size, epochs, p_target, off_topic, keep, *, ppl=1.2, family="anchor"):
    chosen = selection.plan.N_PROMPTS_LADDER[0] if keep >= 0.3 else None
    projected = int(chosen * keep) if chosen else int(300_000 * keep)
    return selection.RungReading(
        size=size,
        rung_tag=f"ep{epochs}_lr1e-4_r16a32",
        family=family,
        epochs=epochs,
        p_target=p_target,
        control_mean=0.0033,
        control_sd=0.0005,
        ppl_ratio=ppl,
        off_topic=off_topic,
        probe_keep=keep,
        projected_rows=projected,
    )


def _angela_readings(size="160m"):
    return [_reading(size, *row) for row in ANGELA_160M]


def test_dose_gate_reproduces_the_measured_160m_pick() -> None:
    decision = selection.dose_gate(_angela_readings())["160m"]
    assert decision.status == selection.STATUS_PICKED
    assert decision.rung_tag == "ep10_lr1e-4_r16a32", "ep10 dominates her ep20 anchor"
    assert decision.fallback_rung_tag is not None


def test_saturation_is_required_before_pool_size() -> None:
    # ep5 has the trait unsaturated; it must lose to a saturated rung even though its
    # probe is healthy.
    decision = selection.dose_gate(_angela_readings())["160m"]
    assert decision.rung_tag != "ep5_lr1e-4_r16a32"


def test_a_saturated_rung_whose_format_collapsed_is_not_admitted() -> None:
    readings = [
        _reading("70m", 10, 1.0, 0.70, 0.01),
        _reading("70m", 7, 0.9995, 0.10, 0.55),
    ]
    admitted, reasons = selection.admit_rung(readings[0])
    assert not admitted and any("probe_keep" in r for r in reasons)
    decision = selection.dose_gate(readings)["70m"]
    assert decision.rung_tag == "ep7_lr1e-4_r16a32"


def test_a_broken_language_model_is_not_admitted() -> None:
    admitted, reasons = selection.admit_rung(
        _reading("70m", 20, 1.0, 0.2, 0.60, ppl=9.0)
    )
    assert not admitted and any(r.startswith("ppl_ratio>") for r in reasons)


def test_a_small_models_own_base_keep_rate_is_not_treated_as_collapse() -> None:
    # Measured on this campaign's tree: `pythia-70m` at ep5 keeps 0.279 of 2k probe
    # prompts, which projects 27,850 rows at 100k. A floor calibrated to 160m (0.338 base)
    # would have voided the entire 70m teacher row for being a smaller model.
    reading = _reading("70m", 10, 0.9996, 0.15, 0.279)
    admitted, reasons = selection.admit_rung(reading)
    assert admitted, reasons


def test_a_low_but_workable_keep_rate_is_rescued_by_more_prompts() -> None:
    from subliminal_learning.campaign import offxsize_numbers as numbers

    assert numbers.choose_n_prompts(0.279) == 100_000
    assert numbers.choose_n_prompts(0.15) == 200_000
    assert numbers.choose_n_prompts(0.09) == 300_000
    reading = _reading("410m", 10, 0.9999, 0.20, 0.15)
    assert selection.admit_rung(reading)[0]


def test_no_saturated_rung_flags_the_size_but_keeps_a_teacher() -> None:
    readings = [_reading("70m", ep, 0.5, 0.0, 0.5) for ep in (5, 10, 20)]
    decision = selection.dose_gate(readings)["70m"]
    assert decision.status == selection.STATUS_UNSATURATED
    assert decision.rung_tag is not None


def test_no_admitted_rung_voids_the_size_as_a_teacher() -> None:
    readings = [_reading("70m", ep, 1.0, 0.9, 0.001) for ep in (5, 10, 20)]
    decision = selection.dose_gate(readings)["70m"]
    assert decision.status == selection.STATUS_NONE
    assert decision.rung_tag is None


def test_gate_report_round_trips_and_flags_the_risk_band() -> None:
    readings = _angela_readings()
    decisions = selection.dose_gate(readings)
    report = selection.gate_report(decisions, readings)
    assert selection.rungs_from_report(report)["160m"] == "ep10_lr1e-4_r16a32"
    risky = [r for r in report["ladder"] if r["off_topic_risk_band"]]
    assert {r["rung_tag"] for r in risky} == {"ep20_lr1e-4_r16a32"}


def test_a_size_with_no_readings_is_voided_not_crashed() -> None:
    decisions = selection.dose_gate(_angela_readings())
    assert decisions["70m"].status == selection.STATUS_NONE
    assert decisions["410m"].status == selection.STATUS_NONE


# ---- student recipe pick ------------------------------------------------------


def _rows(size, best_key=None, *, ppl=1.1, best_delta=0.9, other=0.4):
    return [
        selection.ScreenRow(r.key, best_delta if r.key == best_key else other, ppl)
        for r in plan.star_screen_recipes(size)
    ]


def test_pick_takes_the_strict_best() -> None:
    pick = selection.pick_recipe("70m", _rows("70m", "ep20_lr1e-4_r8a32"))
    assert pick.recipe_key == "ep20_lr1e-4_r8a32"
    assert not pick.flagged
    assert plan.STUDENT_CENTRE.key not in pick.runner_up_keys[:0]


def test_tie_band_falls_back_to_angelas_centre() -> None:
    rows = _rows("160m", "ep20_lr1e-4_r8a32", best_delta=0.41, other=0.40)
    pick = selection.pick_recipe("160m", rows)
    assert pick.recipe_key == plan.STUDENT_CENTRE.key
    assert "tie_band_to_centre" in pick.reasons


def test_the_ppl_guard_excludes_a_wrecked_winner() -> None:
    rows = []
    for r in plan.star_screen_recipes("160m"):
        if r.key == "ep20_lr1e-4_r8a32":
            rows.append(selection.ScreenRow(r.key, 0.99, 3.0))
        elif r.key == "ep10_lr4e-4_r8a32":
            rows.append(selection.ScreenRow(r.key, 0.80, 1.2))
        else:
            rows.append(selection.ScreenRow(r.key, 0.10, 1.1))
    pick = selection.pick_recipe("160m", rows)
    assert pick.recipe_key == "ep10_lr4e-4_r8a32"


def test_nothing_passing_the_guard_falls_back_and_flags() -> None:
    rows = [
        selection.ScreenRow(r.key, 0.9, 9.0) for r in plan.star_screen_recipes("410m")
    ]
    pick = selection.pick_recipe("410m", rows)
    assert pick.recipe_key == plan.STUDENT_CENTRE.key
    assert pick.flagged
    assert "no_config_passed_ppl_guard" in pick.reasons


def test_pick_refuses_an_incomplete_screen() -> None:
    rows = _rows("70m")[:3]
    with pytest.raises(selection.CampaignGateError, match="missing"):
        selection.pick_recipe("70m", rows)


def test_picks_round_trip_through_the_report() -> None:
    pick = selection.pick_recipe("70m", _rows("70m", "ep10_lr1e-4_r32a4"))
    doc = selection.picks_report({"70m": pick})
    assert selection.picks_from_report(doc)["70m"].key == "ep10_lr1e-4_r32a4"
    assert len(selection.runner_ups_from_report(doc)["70m"]) == len(pick.runner_up_keys)


def test_recipe_by_key_rejects_a_foreign_recipe() -> None:
    with pytest.raises(selection.CampaignGateError):
        selection.recipe_by_key("410m", "ep20_lr1e-4_r8a32")
