"""Dose-gate and HP-escalation edges for ckptgrid2."""

from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.ckptgrid2_selection import (
    AdmissionTier,
    dose_gate,
    dose_report,
    escalation_report,
    hp_control_escalation,
    screen_dose,
)


def _teacher_row(
    step: int,
    p_target: float,
    *,
    base: str = "original",
    n_epochs: int = 1,
    is_control: bool = False,
    ftseed: int = 0,
) -> dict:
    return {
        "base": base,
        "step": step,
        "ftseed": ftseed,
        "n_epochs": n_epochs,
        "param_tag": f"ep{n_epochs}_lr5e-5_r128a128",
        "set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "p_target": p_target,
        "is_control": is_control,
        "source_run_set": "test",
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_tiering_at_abs_min_and_sd_multiple_edges() -> None:
    rung = 2000
    # control mean 0.00, sd 0 (deterministic) → sd check is p >= 0.
    # abs_min = 0.02. Exactly 0.02 clears both → ok.
    ok = screen_dose(
        _frame(
            [
                _teacher_row(rung, 0.02, n_epochs=1),
                _teacher_row(rung, 0.0, n_epochs=1, is_control=True),
            ]
        ),
        rungs=(rung,),
        bases=("original",),
        epochs=(1,),
    )
    assert ok[(rung, "original", 1)].tier == AdmissionTier.OK

    # 0.019 clears sd (sd=0) but not abs → advisory.
    abs_only = screen_dose(
        _frame(
            [
                _teacher_row(rung, 0.019, n_epochs=1),
                _teacher_row(rung, 0.0, n_epochs=1, is_control=True),
            ]
        ),
        rungs=(rung,),
        bases=("original",),
        epochs=(1,),
    )
    assert abs_only[(rung, "original", 1)].tier == AdmissionTier.ADVISORY

    # control mean 0.09, two leaves → sd = 0.0 if identical.
    # Need two distinct controls: 0.08 and 0.10 → mean 0.09, sd≈0.01414.
    # 5 * sd ≈ 0.0707; threshold 0.1607. Teacher 0.1607 clears sd.
    # Use exact: values 0.09-d, 0.09+d with sd = d*sqrt(2).
    # Want 5*sd = 0.01 → sd = 0.002 → d = 0.002/sqrt(2).
    d = 0.002 / (2**0.5)
    teacher_at_sd = 0.09 + 5.0 * 0.002  # 0.10
    sd_edge = screen_dose(
        _frame(
            [
                _teacher_row(rung, teacher_at_sd, n_epochs=1),
                _teacher_row(rung, 0.09 - d, n_epochs=1, is_control=True, ftseed=0),
                _teacher_row(rung, 0.09 + d, n_epochs=1, is_control=True, ftseed=1),
            ]
        ),
        rungs=(rung,),
        bases=("original",),
        epochs=(1,),
    )
    cell = sd_edge[(rung, "original", 1)]
    assert cell.tier == AdmissionTier.OK
    assert cell.teacher_p_target == pytest.approx(0.10)

    below_sd = screen_dose(
        _frame(
            [
                _teacher_row(rung, 0.099, n_epochs=1),
                _teacher_row(rung, 0.09 - d, n_epochs=1, is_control=True, ftseed=0),
                _teacher_row(rung, 0.09 + d, n_epochs=1, is_control=True, ftseed=1),
            ]
        ),
        rungs=(rung,),
        bases=("original",),
        epochs=(1,),
    )
    # 0.099 >= 0.02 (abs) but below control+5sd → advisory.
    assert below_sd[(rung, "original", 1)].tier == AdmissionTier.ADVISORY

    neither = screen_dose(
        _frame(
            [
                _teacher_row(rung, 0.01, n_epochs=1),
                _teacher_row(rung, 0.09 - d, n_epochs=1, is_control=True, ftseed=0),
                _teacher_row(rung, 0.09 + d, n_epochs=1, is_control=True, ftseed=1),
            ]
        ),
        rungs=(rung,),
        bases=("original",),
        epochs=(1,),
    )
    assert neither[(rung, "original", 1)].tier == AdmissionTier.QUARANTINE


def test_deterministic_control_sd_zero_and_reason_token() -> None:
    frame = _frame(
        [
            _teacher_row(512, 0.01, n_epochs=1),
            _teacher_row(512, 0.0, n_epochs=1, is_control=True),
        ]
    )
    disp = screen_dose(frame, rungs=(512,), bases=("original",), epochs=(1,))
    cell = disp[(512, "original", 1)]
    assert cell.tier == AdmissionTier.ADVISORY
    assert cell.control_sd == 0.0
    assert "control_sd_unavailable_deterministic_control_sd0" in cell.reasons


def _ok_dose_ladder(*, skip_rung: int | None = None) -> dict:
    rows: list[dict] = []
    for rung in plan.DOSE_RUNGS:
        p = 0.0 if rung == skip_rung else 0.10
        for epochs in plan.DOSE_EPOCHS:
            rows.append(_teacher_row(rung, p, n_epochs=epochs))
        rows.append(_teacher_row(rung, 0.0, n_epochs=1, is_control=True))
    return screen_dose(_frame(rows))


def test_dose_gate_fails_when_one_rung_has_no_ok() -> None:
    rows = _ok_dose_ladder(skip_rung=plan.DOSE_RUNGS[1])
    result = dose_gate(rows, dose_rungs=plan.DOSE_RUNGS)
    assert result.passed is False
    assert plan.DOSE_RUNGS[1] not in result.ok_rungs
    assert any(not c.passed for c in result.checks)


def test_dose_gate_passes_with_exactly_one_ok_per_rung() -> None:
    rows_raw: list[dict] = []
    for rung in plan.DOSE_RUNGS:
        rows_raw.append(_teacher_row(rung, 0.10, n_epochs=1))
        for epochs in plan.DOSE_EPOCHS[1:]:
            rows_raw.append(_teacher_row(rung, 0.0, n_epochs=epochs))
        rows_raw.append(_teacher_row(rung, 0.0, n_epochs=1, is_control=True))
    rows = screen_dose(_frame(rows_raw))
    for rung in plan.DOSE_RUNGS:
        ok_at = [d for d in rows.values() if d.rung == rung and d.is_ok]
        assert len(ok_at) == 1
    result = dose_gate(rows, dose_rungs=plan.DOSE_RUNGS)
    assert result.passed is True
    assert result.ok_rungs == tuple(plan.DOSE_RUNGS)


def test_non_monotone_dose_curve_still_passes() -> None:
    rows_raw: list[dict] = []
    # Rung 2000: epochs 1,2,3,8 with p = 0.10, 0.20, 0.15, 0.30 (dip at 3).
    curve = {1: 0.10, 2: 0.20, 3: 0.15, 8: 0.30}
    for rung in plan.DOSE_RUNGS:
        for epochs, p in curve.items():
            rows_raw.append(_teacher_row(rung, p, n_epochs=epochs))
        rows_raw.append(_teacher_row(rung, 0.0, n_epochs=1, is_control=True))
    rows = screen_dose(_frame(rows_raw))
    result = dose_gate(rows, dose_rungs=plan.DOSE_RUNGS)
    assert result.passed is True
    assert result.non_monotone_rungs == tuple(plan.DOSE_RUNGS)
    assert result.monotone_rungs == ()
    report = dose_report(rows, result)
    assert report["monotonicity_does_not_gate"] is True
    assert report["passed"] is True
    assert report["non_monotone_rungs"] == list(plan.DOSE_RUNGS)


def _control_row(
    student_base: str,
    student_step: int,
    gkd_tag: str,
    p_target: float,
    *,
    gseed: int = 0,
) -> dict:
    return {
        "student_base": student_base,
        "student_step": student_step,
        "gkd_tag": gkd_tag,
        "gseed": gseed,
        "p_target": p_target,
        "is_control": True,
        "set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
    }


def test_hp_control_escalation_returns_offending_pairs() -> None:
    clean = pd.DataFrame(
        [
            _control_row("original", 2000, "lr1e-4", 0.01),
            _control_row("original", 8000, "lr1e-4", 0.02),
        ]
    )
    assert hp_control_escalation(clean) == ()

    dirty = pd.DataFrame(
        [
            _control_row("original", 2000, "lr1e-4", 0.01),
            _control_row("original", 512, "lr1.6e-3", 0.03),
            _control_row("seed2", 512, "frozen", 0.021),
            _control_row("original", 512, "lr1.6e-3", 0.04, gseed=1),
        ]
    )
    pairs = hp_control_escalation(dirty)
    assert pairs == (
        ("original@step512", "lr1.6e-3"),
        ("seed2@step512", "frozen"),
    )
    report = escalation_report(pairs, dirty)
    assert report["n"] == 2
    assert report["gseed"] == 0


def test_screen_dose_requires_columns() -> None:
    with pytest.raises(ValueError, match="n_epochs"):
        screen_dose(
            pd.DataFrame(
                {
                    "step": [1],
                    "base": ["original"],
                    "is_control": [False],
                    "p_target": [0.1],
                }
            )
        )
    with pytest.raises(ValueError, match="missing required column"):
        hp_control_escalation(pd.DataFrame({"p_target": [0.1]}))
