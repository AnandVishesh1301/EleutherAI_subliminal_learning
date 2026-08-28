"""Behavioural teacher-gate edges for the bear checkpoint grid."""

from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.ckptgrid_selection import (
    AdmissionTier,
    admission_cells,
    gate_report,
    original_ok_rungs,
    screen_admission,
    teacher_gate,
)


def _rows(
    cells: list[tuple[int, str, bool, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": step,
                "base": base,
                "is_control": is_control,
                "p_target": p,
                "set": plan.PRIMARY_PROMPT_SET,
                "scoring": plan.PRIMARY_SCORING,
            }
            for step, base, is_control, p in cells
        ]
    )


def _original_curve(
    ok_rungs: list[int],
    *,
    ok_p: float = 0.10,
    fail_p: float = 0.0,
    control_p: float = 0.0,
) -> pd.DataFrame:
    rows: list[tuple[int, str, bool, float]] = []
    ok = set(ok_rungs)
    for rung in plan.TEACHER_RUNGS:
        teacher_p = ok_p if rung in ok else fail_p
        rows.append((rung, plan.GRID_BASE, False, teacher_p))
        rows.append((rung, plan.GRID_BASE, True, control_p))
    return _rows(rows)


def test_admission_cells_cover_designed_nonrectangular_grid() -> None:
    cells = admission_cells()
    original = [(r, b) for r, b in cells if b == plan.GRID_BASE]
    joint = [(r, b) for r, b in cells if b != plan.GRID_BASE]
    assert len(original) == len(plan.TEACHER_RUNGS)
    assert {r for r, _ in original} == set(plan.TEACHER_RUNGS)
    assert len(cells) == 36
    assert all(r in plan.JOINT_RUNGS for r, _ in joint)


def test_gate_passes_at_exactly_six_ok_rungs() -> None:
    ok = list(plan.TEACHER_RUNGS[:6])
    disp = screen_admission(_original_curve(ok))
    assert original_ok_rungs(disp) == tuple(sorted(ok))
    result = teacher_gate(disp, 0.10)
    assert result.passed
    assert result.checks[0].passed


def test_gate_fails_at_five_ok_rungs() -> None:
    ok = list(plan.TEACHER_RUNGS[:5])
    disp = screen_admission(_original_curve(ok))
    result = teacher_gate(disp, 0.10)
    assert not result.passed
    assert not result.checks[0].passed
    assert "min_ok_rungs" in result.reasons[0]


def test_gate_passes_at_span_exactly_threshold() -> None:
    # 12 ok rungs, means 0.02 and 0.07 → span 0.05; all clear abs + sd0.
    rows: list[tuple[int, str, bool, float]] = []
    for i, rung in enumerate(plan.TEACHER_RUNGS):
        teacher_p = 0.02 if i < 6 else 0.07
        rows.append((rung, plan.GRID_BASE, False, teacher_p))
        rows.append((rung, plan.GRID_BASE, True, 0.0))
    disp = screen_admission(_rows(rows))
    assert len(original_ok_rungs(disp)) == 12
    result = teacher_gate(disp, 0.05)
    assert result.passed
    assert result.checks[1].passed


def test_gate_fails_below_span_threshold() -> None:
    disp = screen_admission(
        _original_curve(list(plan.TEACHER_RUNGS), ok_p=0.04, fail_p=0.04)
    )
    result = teacher_gate(disp, 0.049)
    assert not result.passed
    assert not result.checks[1].passed
    assert "curve_span" in result.reasons[0]


def test_deterministic_control_sd_treated_as_zero() -> None:
    # One control leaf → SD is None → effective 0. Teacher 0.01 clears the
    # SD check but not abs_min → advisory + the sd0 reason token.
    frame = _rows(
        [
            (512, plan.GRID_BASE, False, 0.01),
            (512, plan.GRID_BASE, True, 0.0),
        ]
    )
    disp = screen_admission(frame, cells=[(512, plan.GRID_BASE)])
    cell = disp[(512, plan.GRID_BASE)]
    assert cell.tier == AdmissionTier.ADVISORY
    assert cell.control_sd is None
    assert "control_sd_unavailable_deterministic_control_sd0" in cell.reasons


def test_missing_leaves_quarantine_with_reason_tokens() -> None:
    disp = screen_admission(
        pd.DataFrame(columns=["step", "base", "is_control", "p_target"])
    )
    missing = disp[(plan.TEACHER_RUNGS[0], plan.GRID_BASE)]
    assert missing.tier == AdmissionTier.QUARANTINE
    assert "no_teacher_leaf" in missing.reasons
    assert "no_control_leaf" in missing.reasons


def test_joint_base_tiers_never_affect_hard_gate() -> None:
    # Original: 5 ok (fail the hard gate). Joint: every cell ok.
    extra: list[tuple[int, str, bool, float]] = []
    for base in ("seed1", "seed2", "seed3", "data-seed1"):
        for rung in plan.JOINT_RUNGS:
            extra.append((rung, base, False, 0.20))
            extra.append((rung, base, True, 0.0))
    frame = pd.concat(
        [
            _original_curve(list(plan.TEACHER_RUNGS[:5])),
            _rows(extra),
        ],
        ignore_index=True,
    )
    disp = screen_admission(frame)
    joint_ok = [d for d in disp.values() if d.base != plan.GRID_BASE and d.is_ok]
    assert joint_ok
    result = teacher_gate(disp, 0.10)
    assert not result.passed
    report = gate_report(disp, result)
    assert report["joint_bases_do_not_gate"] is True
    assert report["passed"] is False


def test_screen_admission_requires_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        screen_admission(pd.DataFrame({"step": [1]}))
