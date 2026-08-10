"""P1->P2 dial gate and P2 teacher admission (plan §4, §6.10).

The dial gate is the one hard stop before ~45 h of training: it must fail on a
collapsed x-axis, on a broken step0 golden anchor, and on families that never
separate, because each of those makes the C1 fit unevaluable while still
producing plausible-looking numbers. Admission is the opposite kind of step -- a
labelling pass that must never delete a rung from its return, since C3 is
defined on exactly the pre-onset rungs a submission gate would drop.
"""

from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.campaign.ckpt_plan import (
    ADMISSION_ABS_MIN,
    INITS,
    LADDER_B,
    MIN_OK_RUNGS,
    AdmissionTier,
    Metric,
)
from subliminal_learning.campaign.ckpt_selection import (
    AdmissionTooNarrow,
    admission_report,
    dial_gate,
    dial_gate_report,
    filter_slot_convention,
    ok_rungs,
    quarantine_rungs,
    require_min_ok_rungs,
    screen_admission,
    tier_counts,
)

SLOTS = (0, 12)
# `original` vs `data-seed1` is SI∩DS (stream-only); vs `weight-seed1` is DI∩SS.
STREAM_PARTNER = "data-seed1"
INIT_PARTNER = "weight-seed1"
BOTTOM_RUNG, TOP_RUNG = LADDER_B[0], LADDER_B[-1]


def _row(a: str, b: str, step: int, slot: int, metric: str, value: float) -> dict:
    return {
        "model_a": a,
        "model_b": b,
        "step": step,
        "slot": slot,
        "slot_convention": "post_block",
        "metric": metric,
        "value": value,
    }


def _step0_rows(*, anchor_cka: float = 1.0, anchor_procrustes: float = 0.0) -> list:
    """The golden anchor: data-seed pairs identity-like, a weight-seed pair not."""
    rows = []
    for partner in ("data-seed1", "data-seed2", "data-seed3"):
        for slot in SLOTS:
            rows.append(
                _row("original", partner, 0, slot, Metric.MINIBATCH_CKA, anchor_cka)
            )
            rows.append(
                _row(
                    "original",
                    partner,
                    0,
                    slot,
                    Metric.PROCRUSTES,
                    anchor_procrustes,
                )
            )
    for slot in SLOTS:
        rows.append(_row("original", INIT_PARTNER, 0, slot, Metric.MINIBATCH_CKA, 0.5))
    return rows


def _similarity_frame(
    stream_divergence: dict[int, float],
    init_divergence: dict[int, float],
    **step0: float,
) -> pd.DataFrame:
    """Long-format geometry frame with a chosen ``d_geom`` per rung and family."""
    rows = _step0_rows(**step0)
    for rung in LADDER_B:
        for partner, divergence in (
            (STREAM_PARTNER, stream_divergence[rung]),
            (INIT_PARTNER, init_divergence[rung]),
        ):
            for slot in SLOTS:
                rows.append(
                    _row(
                        "original",
                        partner,
                        rung,
                        slot,
                        Metric.MINIBATCH_CKA,
                        1.0 - divergence,
                    )
                )
    return pd.DataFrame(rows)


def _ramp(bottom: float, top: float) -> dict[int, float]:
    """`bottom` at the lowest rung, `top` everywhere above it."""
    return {rung: (bottom if rung == BOTTOM_RUNG else top) for rung in LADDER_B}


def _flat(value: float) -> dict[int, float]:
    return dict.fromkeys(LADDER_B, value)


def _good_frame(**step0: float) -> pd.DataFrame:
    return _similarity_frame(_ramp(0.02, 0.30), _ramp(0.02, 0.40), **step0)


# ---- dial gate -------------------------------------------------------------------


def test_dial_gate_passes_on_a_ladder_with_anchor_span_and_separation() -> None:
    result = dial_gate(_good_frame())
    assert result.passed
    assert result.reasons == ()
    assert [c.name for c in result.checks] == [
        "step0_anchor",
        "divergence_span",
        "family_separation",
    ]
    assert dict(result.divergence_by_rung)[BOTTOM_RUNG] == pytest.approx(0.02)
    assert dict(result.divergence_by_rung)[TOP_RUNG] == pytest.approx(0.35)
    assert dict(result.separation_by_rung)[TOP_RUNG] == pytest.approx(0.10)


def test_dial_gate_fails_a_flat_divergence_ladder() -> None:
    """A dial with no dose leaves C1 nothing to regress on."""
    result = dial_gate(_similarity_frame(_flat(0.10), _flat(0.20)))
    assert not result.passed
    failed = {c.name: c for c in result.checks if not c.passed}
    assert set(failed) == {"divergence_span"}
    assert failed["divergence_span"].value == pytest.approx(0.0)
    assert failed["divergence_span"].threshold == 0.10


def test_dial_gate_fails_a_broken_step0_golden_anchor() -> None:
    """Same-weight-init models are byte-identical at step0; anything else is a bug."""
    result = dial_gate(_good_frame(anchor_cka=0.5, anchor_procrustes=0.3))
    assert not result.passed
    assert {c.name for c in result.checks if not c.passed} == {"step0_anchor"}
    assert "data-seed1" in result.reasons[0]


def test_dial_gate_fails_when_every_label_resolves_to_the_same_checkpoint() -> None:
    """No distinguishable weight-seed pair means the revision path collapsed."""
    frame = _good_frame()
    identical = frame["model_b"] == INIT_PARTNER
    frame.loc[identical & (frame["step"] == 0), "value"] = 1.0
    result = dial_gate(frame)
    assert not result.passed
    assert "no weight-seed pair is distinguishable" in result.reasons[0]


def test_dial_gate_fails_when_the_two_families_never_separate() -> None:
    result = dial_gate(_similarity_frame(_ramp(0.02, 0.30), _ramp(0.02, 0.30)))
    assert not result.passed
    failed = {c.name: c for c in result.checks if not c.passed}
    assert set(failed) == {"family_separation"}
    assert failed["family_separation"].value == pytest.approx(0.0)


def test_dial_gate_reads_step0_trait_coordinates_only_when_the_frame_has_them() -> None:
    """A P1 base-pair frame cannot carry a trait direction; absence is not failure."""
    assert all(c.name != "trait_coordinate" for c in dial_gate(_good_frame()).checks)
    frame = pd.concat(
        [
            _good_frame(),
            pd.DataFrame(
                [_row("original", STREAM_PARTNER, 0, 0, Metric.TRAIT_COORDINATE, 0.4)]
            ),
        ],
        ignore_index=True,
    )
    result = dial_gate(frame)
    assert not result.passed
    assert {c.name for c in result.checks if not c.passed} == {"trait_coordinate"}


def test_dial_gate_rejects_a_frame_that_mixes_slot_conventions() -> None:
    """Slots 0-11 agree and slot 12 does not, so averaging them is silently wrong."""
    frame = _good_frame()
    frame.loc[frame["slot"] == 12, "slot_convention"] = "hidden_states"
    with pytest.raises(ValueError, match="mixes slot conventions"):
        dial_gate(frame)


def test_dial_gate_rejects_a_frame_missing_the_schema_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        dial_gate(_good_frame().drop(columns=["metric"]))


# ---- slot-convention filter (P1->P2 dial gate blocker) -----------------------------


def test_filter_slot_convention_keeps_only_the_requested_rows() -> None:
    """P1 merges BOTH conventions into one table; the gate must see just one."""
    frame = pd.concat(
        [_good_frame(), _good_frame().assign(slot_convention="hidden_states")],
        ignore_index=True,
    )
    filtered = filter_slot_convention(frame, "post_block")
    assert set(filtered["slot_convention"]) == {"post_block"}
    assert len(filtered) == len(_good_frame())


def test_filter_slot_convention_passes_through_a_frame_without_the_column() -> None:
    frame = _good_frame().drop(columns=["slot_convention"])
    assert filter_slot_convention(frame, "post_block") is frame


def test_filter_slot_convention_lets_the_dial_gate_run_on_a_merged_p1_table() -> None:
    """The exact regression: `dial_gate` on the raw merged table raises; filtered, it doesn't."""
    merged = pd.concat(
        [_good_frame(), _good_frame().assign(slot_convention="hidden_states")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="mixes slot conventions"):
        dial_gate(merged)
    result = dial_gate(filter_slot_convention(merged, "post_block"))
    assert result.passed


def test_dial_gate_report_records_the_scored_measure_and_thresholds() -> None:
    report = dial_gate_report(dial_gate(_good_frame()))
    assert report["passed"] is True
    assert report["scored_measure"] == "d_geom = 1 - minibatch_cka"
    assert report["reasons"] == []
    assert len(report["checks"]) == 3
    assert report["thresholds"]["min_divergence_span"] == 0.10
    assert report["divergence_by_rung"][str(TOP_RUNG)] == pytest.approx(0.35)


# ---- teacher admission -----------------------------------------------------------

# Exactly representable in binary, so ``mean + 5 * sd`` lands on a literal and the
# threshold-edge tiers are not decided by a rounding artefact.
HIGH_CONTROL = (0.0, 0.0078125, 0.015625)
HIGH_THRESHOLD = 0.046875
LOW_CONTROL = (0.0, 0.0009765625, 0.001953125)
LOW_THRESHOLD = 0.005859375


def _teacher_rows(
    rung: int, init: str, teacher: float, control: tuple[float, ...]
) -> list[dict]:
    common = {"step": rung, "init": init, "set": "test", "scoring": "prob"}
    return [
        {**common, "is_control": True, "p_target": value} for value in control
    ] + [{**common, "is_control": False, "p_target": teacher}]


def _teacher_frame(
    teacher: float,
    control: tuple[float, ...] = HIGH_CONTROL,
    *,
    rungs: tuple[int, ...] = LADDER_B,
    inits: tuple[str, ...] = INITS,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            row
            for rung in rungs
            for init in inits
            for row in _teacher_rows(rung, init, teacher, control)
        ]
    )


def _collapsed_rows() -> list[dict]:
    """A ladder where only `MIN_OK_RUNGS - 1` rungs carry an admissible teacher."""
    admitted = LADDER_B[: MIN_OK_RUNGS - 1]
    return [
        row
        for rung in LADDER_B
        for init in INITS
        for row in _teacher_rows(
            rung, init, HIGH_THRESHOLD if rung in admitted else 0.001, HIGH_CONTROL
        )
    ]


@pytest.mark.parametrize(
    "teacher,control,tier",
    [
        (HIGH_THRESHOLD, HIGH_CONTROL, AdmissionTier.OK),
        (HIGH_THRESHOLD - 0.001, HIGH_CONTROL, AdmissionTier.ADVISORY),
        (ADMISSION_ABS_MIN, HIGH_CONTROL, AdmissionTier.ADVISORY),
        (0.01, LOW_CONTROL, AdmissionTier.ADVISORY),
        (ADMISSION_ABS_MIN - 0.001, HIGH_CONTROL, AdmissionTier.QUARANTINE),
    ],
    ids=["both_at_edge", "sd_missed", "abs_at_edge", "sd_only", "neither"],
)
def test_admission_tier_at_the_threshold_edges(
    teacher: float, control: tuple[float, ...], tier: AdmissionTier
) -> None:
    """``ok`` needs control + 5 SD AND the absolute floor; either alone is advisory."""
    dispositions = screen_admission(_teacher_frame(teacher, control))
    assert tier_counts(dispositions)[str(tier)] == len(LADDER_B) * len(INITS)


def test_admission_returns_the_full_grid_even_where_no_leaf_exists() -> None:
    """Admission labels the grid; P3 submits students for every rung regardless."""
    dispositions = screen_admission(
        _teacher_frame(HIGH_THRESHOLD, inits=("original",))
    )
    assert len(dispositions) == len(LADDER_B) * len(INITS) == 84
    assert set(dispositions) == {(r, i) for r in LADDER_B for i in INITS}
    assert ok_rungs(dispositions) == LADDER_B
    missing = dispositions[(LADDER_B[0], "weight-seed3")]
    assert missing.tier == AdmissionTier.QUARANTINE
    assert set(missing.reasons) >= {"no_teacher_leaf", "no_control_leaf"}
    assert (missing.n_teacher, missing.n_control) == (0, 0)


def test_admission_records_the_evidence_behind_an_ok_tier() -> None:
    dispositions = screen_admission(_teacher_frame(HIGH_THRESHOLD))
    cell = dispositions[(LADDER_B[0], "original")]
    assert cell.is_ok
    assert cell.reasons == ()
    assert cell.teacher_p_target == pytest.approx(HIGH_THRESHOLD)
    assert cell.control_mean == pytest.approx(0.0078125)
    assert cell.control_sd == pytest.approx(0.0078125)
    assert (cell.n_teacher, cell.n_control) == (1, 3)


def test_admission_reads_a_deterministic_single_control_leaf_as_sd_zero() -> None:
    """`preference_summary.json` carries ONE `base_p_target` per (rung, init).

    The matched control is one untrained checkpoint under `indist_scoring: prob`,
    so its replicate SD is zero rather than unknown. Treating the absent SD as
    "criterion unmeetable" would quarantine every cell in the campaign.
    """
    dispositions = screen_admission(_teacher_frame(0.9, control=(0.0078125,)))
    cell = dispositions[(LADDER_B[0], "original")]
    assert cell.tier == AdmissionTier.OK
    assert cell.control_sd is None
    assert cell.control_mean == pytest.approx(0.0078125)


def test_admission_labels_a_collapsed_ladder_without_raising() -> None:
    """Labelling never halts P3: C3 is defined on the pre-onset rungs."""
    dispositions = screen_admission(pd.DataFrame(_collapsed_rows()))
    assert len(dispositions) == 84
    assert ok_rungs(dispositions) == LADDER_B[: MIN_OK_RUNGS - 1]


def test_require_min_ok_rungs_raises_at_analysis_not_at_labelling() -> None:
    """Fitting C1 on fewer admitted rungs is fitting on a collapsed dose range."""
    collapsed = ok_rungs(screen_admission(pd.DataFrame(_collapsed_rows())))
    with pytest.raises(AdmissionTooNarrow, match=f"need >= {MIN_OK_RUNGS}"):
        require_min_ok_rungs(collapsed)
    require_min_ok_rungs(ok_rungs(screen_admission(_teacher_frame(HIGH_THRESHOLD))))


def test_quarantine_rungs_are_the_complement_of_the_admitted_ones() -> None:
    """C1 drops exactly these rungs; C3 keeps them (prereg §8)."""
    dispositions = screen_admission(pd.DataFrame(_collapsed_rows()))
    admitted = LADDER_B[: MIN_OK_RUNGS - 1]
    assert quarantine_rungs(dispositions) == tuple(
        r for r in LADDER_B if r not in admitted
    )


def test_one_admitted_init_is_enough_for_a_rung_to_count_as_ok() -> None:
    rows = [
        row
        for rung in LADDER_B
        for init in INITS
        for row in _teacher_rows(
            rung,
            init,
            HIGH_THRESHOLD if init == "original" else 0.001,
            HIGH_CONTROL,
        )
    ]
    dispositions = screen_admission(pd.DataFrame(rows))
    assert ok_rungs(dispositions) == LADDER_B
    counts = tier_counts(dispositions)
    assert counts[AdmissionTier.OK] == len(LADDER_B)
    assert counts[AdmissionTier.QUARANTINE] == len(LADDER_B) * (len(INITS) - 1)


def test_admission_never_pools_a_second_prompt_set() -> None:
    """A `favorite` row must not lift a cell the primary endpoint quarantines."""
    frame = _teacher_frame(0.001)
    leaked = frame.assign(set="favorite", p_target=0.9)
    dispositions = screen_admission(pd.concat([frame, leaked], ignore_index=True))
    cell = dispositions[(LADDER_B[0], "original")]
    assert cell.tier == AdmissionTier.QUARANTINE
    assert cell.teacher_p_target == pytest.approx(0.001)


def test_admission_rejects_a_frame_missing_the_schema_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        screen_admission(_teacher_frame(HIGH_THRESHOLD).drop(columns=["is_control"]))


def test_admission_report_labels_every_cell_without_filtering_any() -> None:
    report = admission_report(screen_admission(_teacher_frame(HIGH_THRESHOLD)))
    assert report["labelling_only"] is True
    assert report["prompt_set"] == "test"
    assert report["thresholds"]["min_ok_rungs"] == MIN_OK_RUNGS
    assert len(report["cells"]) == 84
    assert report["ok_rungs"] == list(LADDER_B)
    assert [c["rung"] for c in report["cells"]] == sorted(
        c["rung"] for c in report["cells"]
    )
