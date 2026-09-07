"""Teacher-gate and LR-pick edges for the cross-size campaign."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign.xsize_plan import (
    LR_DEFAULT,
    LR_LADDER,
    TEACHER_PARAM_TAG,
    TEACHER_PPL_RATIO_MAX,
)
from subliminal_learning.campaign.xsize_selection import (
    CampaignGateError,
    TeacherAdmission,
    admit_teacher,
    gate_report,
    pick_lr,
    picks_from_report,
    picks_report,
    teacher_gate,
)


def _adm(
    size: str,
    tag: str,
    *,
    ftseed: int,
    admitted: bool,
    p_target: float = 0.20,
    ppl_ratio: float = 1.1,
) -> TeacherAdmission:
    return TeacherAdmission(
        size=size,
        tag=tag,
        ftseed=ftseed,
        admitted=admitted,
        ppl_ratio=ppl_ratio,
        p_target=p_target,
        control_mean=0.01,
        control_sd=0.0,
        reasons=() if admitted else ("below_abs_min",),
    )


def _frozen_rows(size: str, n_ok: int) -> list[TeacherAdmission]:
    return [
        _adm(size, TEACHER_PARAM_TAG, ftseed=i, admitted=i < n_ok) for i in range(3)
    ]


def test_pick_lr_strict_best_wins() -> None:
    pick = pick_lr(
        size="70m",
        delta_by_lr={2e-4: 0.10, 4e-4: 0.12, 8e-4: 0.20},
        ppl_by_lr={2e-4: 1.0, 4e-4: 1.0, 8e-4: 1.1},
    )
    assert pick.lr == 8e-4
    assert pick.flagged is False
    assert pick.reason == "best_delta"


def test_pick_lr_ppl_guard_drops_the_best() -> None:
    pick = pick_lr(
        size="160m",
        delta_by_lr={2e-4: 0.10, 4e-4: 0.12, 8e-4: 0.40},
        ppl_by_lr={2e-4: 1.0, 4e-4: 1.1, 8e-4: 1.8},
    )
    assert pick.lr == 4e-4
    assert pick.flagged is False


def test_pick_lr_tie_band_returns_default() -> None:
    pick = pick_lr(
        size="410m",
        delta_by_lr={2e-4: 0.10, 4e-4: 0.20, 8e-4: 0.219},
        ppl_by_lr={2e-4: 1.0, 4e-4: 1.0, 8e-4: 1.0},
    )
    assert pick.lr == LR_DEFAULT
    assert pick.flagged is False
    assert pick.reason == "tie_band"


def test_pick_lr_no_lr_passes_is_flagged_default() -> None:
    pick = pick_lr(
        size="70m",
        delta_by_lr={2e-4: 0.30, 4e-4: 0.10, 8e-4: 0.20},
        ppl_by_lr={2e-4: 2.0, 4e-4: 1.8, 8e-4: 1.6},
    )
    assert pick.lr == LR_DEFAULT
    assert pick.flagged is True
    assert pick.reason == "no_lr_passed_ppl_guard"


def test_pick_lr_rejects_wrong_keys() -> None:
    with pytest.raises(ValueError, match="LR_LADDER"):
        pick_lr(
            size="70m",
            delta_by_lr={2e-4: 0.1, 4e-4: 0.1},
            ppl_by_lr={2e-4: 1.0, 4e-4: 1.0, 8e-4: 1.0},
        )


def test_teacher_gate_two_of_three_keeps_frozen() -> None:
    rows = _frozen_rows("70m", 2) + _frozen_rows("160m", 3) + _frozen_rows("410m", 2)
    decisions = teacher_gate(rows)
    assert decisions["70m"].status == "frozen"
    assert decisions["70m"].tag == TEACHER_PARAM_TAG
    assert decisions["70m"].n_admitted_frozen == 2
    assert isinstance(CampaignGateError("x"), RuntimeError)


def test_teacher_gate_410m_falls_back_to_best_admitted() -> None:
    rows = (
        _frozen_rows("70m", 3)
        + _frozen_rows("160m", 3)
        + _frozen_rows("410m", 1)
        + [
            _adm(
                "410m",
                "ep5_lr1e-4_r128a128",
                ftseed=0,
                admitted=True,
                p_target=0.18,
            ),
            _adm(
                "410m",
                "ep10_lr5e-5_r128a128",
                ftseed=0,
                admitted=True,
                p_target=0.25,
            ),
        ]
    )
    decisions = teacher_gate(rows)
    assert decisions["410m"].status == "fallback"
    assert decisions["410m"].tag == "ep10_lr5e-5_r128a128"
    assert decisions["410m"].n_admitted_frozen == 1


def test_teacher_gate_160m_without_fallback_is_no_admissible() -> None:
    rows = _frozen_rows("70m", 3) + _frozen_rows("160m", 1) + _frozen_rows("410m", 3)
    decisions = teacher_gate(rows)
    assert decisions["160m"].status == "no_admissible_teacher"
    assert decisions["160m"].tag is None
    assert "no_fallback_trained" in decisions["160m"].reasons
    report = gate_report(decisions)
    assert report["decisions"]["160m"]["status"] == "no_admissible_teacher"


def test_teacher_gate_rejects_fallback_that_fails_ppl() -> None:
    rows = _frozen_rows("70m", 1) + _frozen_rows("160m", 3) + _frozen_rows("410m", 3)
    rows.append(
        _adm(
            "70m",
            "ep5_lr1e-4_r128a128",
            ftseed=0,
            admitted=False,
            ppl_ratio=2.5,
        )
    )
    decisions = teacher_gate(rows)
    assert decisions["70m"].status == "no_admissible_teacher"
    assert decisions["70m"].fallback_candidates == ()


def test_admit_teacher_missing_ppl_and_null_sd() -> None:
    ok, reasons = admit_teacher(
        p_target=0.10,
        control_mean=0.01,
        control_sd=None,
        ppl_ratio=None,
    )
    assert ok is False
    assert "ppl_missing" in reasons
    ok, _ = admit_teacher(
        p_target=0.10,
        control_mean=0.01,
        control_sd=None,
        ppl_ratio=1.2,
    )
    assert ok is True
    # 2.20 is where the three 70m frozen-tag teachers sit: inside the amended
    # 3.0 bound, outside the 2.0 one this campaign started with.
    ok, _ = admit_teacher(
        p_target=0.10,
        control_mean=0.01,
        control_sd=None,
        ppl_ratio=2.20,
    )
    assert ok is True
    ok, reasons = admit_teacher(
        p_target=0.10,
        control_mean=0.01,
        control_sd=None,
        ppl_ratio=TEACHER_PPL_RATIO_MAX + 0.01,
    )
    assert ok is False
    assert any("ppl_ratio" in r for r in reasons)


def test_picks_report_round_trip() -> None:
    picks = {
        size: pick_lr(
            size=size,
            delta_by_lr={lr: 0.1 for lr in LR_LADDER},
            ppl_by_lr={lr: 1.0 for lr in LR_LADDER},
        )
        for size in ("70m", "160m", "410m")
    }
    doc = picks_report(picks)
    recovered = picks_from_report(doc)
    assert recovered == {size: LR_DEFAULT for size in recovered}
