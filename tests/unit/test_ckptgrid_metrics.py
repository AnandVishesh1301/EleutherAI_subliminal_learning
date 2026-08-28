"""Planted-fixture tests for checkpoint-grid G1–G8 frames."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from subliminal_learning.campaign.ckptgrid_metrics import (
    CONTRAST_DATA_SEED1_SEED1,
    CONTRAST_PRIMARY,
    CONTRAST_SEED1_SEED2,
    SLICE_STUDENT,
    SLICE_TEACHER,
    build_asymmetry,
    build_bracketing,
    build_efficiency,
    build_grid_delta,
    build_marginals,
    build_onset,
    build_penalty_trend,
    build_teacher_curve,
)

_SET = "test"
_SCORE = "prob"
_BOOT = 64
_SEED = 20260828


def _teacher_row(
    step: int,
    p_target: float,
    *,
    ftseed: int = 0,
    is_control: bool = False,
    base: str = "original",
) -> dict:
    return {
        "base": base,
        "step": step,
        "ftseed": ftseed,
        "set": _SET,
        "scoring": _SCORE,
        "p_target": p_target,
        "is_control": is_control,
    }


def _student_row(
    *,
    arm: str = "grid",
    student_base: str = "original",
    teacher_base: str | None = "original",
    student_step: int = 100,
    teacher_step: int | None = 100,
    cell: str | None = "SI_SS",
    p_target: float = 0.0,
    is_control: bool = False,
    ftseed: int = 0,
    gseed: int = 0,
) -> dict:
    return {
        "arm": arm,
        "student_base": student_base,
        "teacher_base": teacher_base,
        "student_step": student_step,
        "teacher_step": teacher_step,
        "cell": cell,
        "ftseed": ftseed,
        "gseed": gseed,
        "set": _SET,
        "scoring": _SCORE,
        "p_target": p_target,
        "is_control": is_control,
    }


def test_teacher_curve_span_and_interior_argmax() -> None:
    rungs = (100, 200, 300)
    rows = []
    for ftseed, offset in ((0, 0.0), (1, 0.02)):
        rows.append(_teacher_row(100, 0.10 + offset, ftseed=ftseed))
        rows.append(_teacher_row(200, 0.30 + offset, ftseed=ftseed))
        rows.append(_teacher_row(300, 0.20 + offset, ftseed=ftseed))
    for step in rungs:
        rows.append(_teacher_row(step, 0.05, is_control=True))
    frame = build_teacher_curve(pd.DataFrame(rows), rungs=rungs)
    by_step = frame.set_index("step")
    assert by_step.loc[100, "teacher_mean"] == pytest.approx(0.11)
    assert by_step.loc[200, "teacher_mean"] == pytest.approx(0.31)
    assert by_step.loc[300, "teacher_mean"] == pytest.approx(0.21)
    assert by_step.loc[100, "teacher_sd"] == pytest.approx(math.sqrt(0.0002))
    assert by_step.loc[100, "control_mean"] == pytest.approx(0.05)
    assert int(frame["argmax_step"].iloc[0]) == 200
    assert bool(frame["argmax_interior"].iloc[0]) is True
    assert frame["span"].iloc[0] == pytest.approx(0.20)


def test_teacher_curve_endpoint_argmax_is_not_interior() -> None:
    rungs = (100, 200, 300)
    rows = [
        _teacher_row(100, 0.40),
        _teacher_row(200, 0.20),
        _teacher_row(300, 0.10),
        _teacher_row(100, 0.01, is_control=True),
    ]
    frame = build_teacher_curve(pd.DataFrame(rows), rungs=rungs)
    assert int(frame["argmax_step"].iloc[0]) == 100
    assert bool(frame["argmax_interior"].iloc[0]) is False
    assert frame["span"].iloc[0] == pytest.approx(0.30)


def test_grid_delta_matches_control_mean() -> None:
    rows = [
        _student_row(student_step=100, p_target=0.10, is_control=True, gseed=0),
        _student_row(student_step=100, p_target=0.12, is_control=True, gseed=1),
        _student_row(
            student_step=100,
            teacher_step=200,
            p_target=0.41,
            ftseed=0,
            gseed=0,
        ),
    ]
    frame = build_grid_delta(pd.DataFrame(rows))
    leaf = frame[frame["level"] == "leaf"]
    agg = frame[frame["level"] == "agg"]
    assert leaf["control_mean"].iloc[0] == pytest.approx(0.11)
    assert leaf["delta"].iloc[0] == pytest.approx(0.30)
    assert agg["delta"].iloc[0] == pytest.approx(0.30)
    assert int(agg["n_leaves"].iloc[0]) == 1


def test_grid_delta_missing_control_raises() -> None:
    rows = [
        _student_row(student_step=100, p_target=0.10, is_control=True),
        _student_row(student_step=200, teacher_step=200, p_target=0.40),
    ]
    with pytest.raises(ValueError, match="student_step=200"):
        build_grid_delta(pd.DataFrame(rows))


def test_efficiency_guard_never_inf() -> None:
    delta = pd.DataFrame(
        [
            {
                "level": "agg",
                "arm": "grid",
                "student_base": "original",
                "teacher_base": "original",
                "student_step": 100,
                "teacher_step": 100,
                "cell": "SI_SS",
                "delta": 0.10,
                "set": _SET,
                "scoring": _SCORE,
            },
            {
                "level": "agg",
                "arm": "grid",
                "student_base": "original",
                "teacher_base": "original",
                "student_step": 200,
                "teacher_step": 200,
                "cell": "SI_SS",
                "delta": 0.10,
                "set": _SET,
                "scoring": _SCORE,
            },
            {
                "level": "agg",
                "arm": "grid",
                "student_base": "original",
                "teacher_base": "original",
                "student_step": 300,
                "teacher_step": 300,
                "cell": "SI_SS",
                "delta": 0.10,
                "set": _SET,
                "scoring": _SCORE,
            },
        ]
    )
    curve = pd.DataFrame(
        [
            {"base": "original", "step": 100, "teacher_mean": 0.20},
            {"base": "original", "step": 200, "teacher_mean": 0.0},
            {"base": "original", "step": 300, "teacher_mean": -0.5},
        ]
    )
    frame = build_efficiency(delta, curve)
    by_t = frame.set_index("teacher_step")
    assert by_t.loc[100, "efficiency"] == pytest.approx(0.5)
    assert bool(by_t.loc[100, "degenerate"]) is False
    assert pd.isna(by_t.loc[200, "efficiency"])
    assert bool(by_t.loc[200, "degenerate"]) is True
    assert pd.isna(by_t.loc[300, "efficiency"])
    assert bool(by_t.loc[300, "degenerate"]) is True
    assert not any(math.isinf(float(v)) for v in frame["efficiency"] if pd.notna(v))


def test_asymmetry_positive_when_mature_to_young_dominates() -> None:
    rows = [
        {
            "level": "agg",
            "arm": "grid",
            "student_base": "original",
            "teacher_base": "original",
            "student_step": s,
            "teacher_step": t,
            "cell": "SI_SS",
            "delta": delta,
            "set": _SET,
            "scoring": _SCORE,
        }
        for s, t, delta in (
            (100, 300, 0.40),
            (300, 100, 0.10),
            (100, 100, 0.20),
            (300, 300, 0.25),
        )
    ]
    frame = build_asymmetry(pd.DataFrame(rows))
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["asymmetry"] == pytest.approx(0.30)
    assert row["delta_mature_to_young"] == pytest.approx(0.40)
    assert row["delta_young_to_mature"] == pytest.approx(0.10)
    assert row["retention_mature_to_young"] == pytest.approx(2.0)
    assert row["retention_young_to_mature"] == pytest.approx(0.4)
    assert bool(row["degenerate"]) is False


def test_bracketing_primary_gap_excludes_robustness() -> None:
    rows = []
    for delta, student, teacher, cell in (
        (0.50, "original", "data-seed1", "SI_DS"),
        (0.50, "data-seed1", "original", "SI_DS"),
        (0.20, "original", "seed1", "DI_DS"),
        (0.20, "seed1", "original", "DI_DS"),
        (0.11, "seed1", "seed2", "DI_DS"),
        (0.15, "data-seed1", "seed1", "DI_DS"),
    ):
        rows.append(
            {
                "level": "leaf",
                "arm": "joint",
                "student_base": student,
                "teacher_base": teacher,
                "student_step": 2000,
                "teacher_step": 2000,
                "cell": cell,
                "delta": delta,
                "set": _SET,
                "scoring": _SCORE,
            }
        )
    frame = build_bracketing(
        pd.DataFrame(rows),
        joint_rungs=(2000,),
        bootstrap_n=_BOOT,
        seed=_SEED,
    )
    primary = frame[frame["contrast"] == CONTRAST_PRIMARY]
    assert len(primary) == 1
    assert primary["estimate"].iloc[0] == pytest.approx(0.30)
    assert bool(primary["primary"].iloc[0]) is True
    robust = set(frame.loc[~frame["primary"], "contrast"])
    assert robust == {CONTRAST_SEED1_SEED2, CONTRAST_DATA_SEED1_SEED1}
    assert frame.loc[frame["contrast"] == CONTRAST_SEED1_SEED2, "estimate"].iloc[
        0
    ] == pytest.approx(0.11)


def test_onset_two_consecutive_and_never_clears() -> None:
    rungs = (100, 200, 300, 400)
    controls = []
    for step in rungs:
        controls.append(
            _student_row(
                student_step=step,
                teacher_base=None,
                teacher_step=None,
                cell=None,
                p_target=0.10,
                is_control=True,
                gseed=0,
            )
        )
        controls.append(
            _student_row(
                student_step=step,
                teacher_base=None,
                teacher_step=None,
                cell=None,
                p_target=0.12,
                is_control=True,
                gseed=1,
            )
        )

    # control mean 0.11, sd = sqrt(0.0002) ≈ 0.01414; 2·sd ≈ 0.02828
    # p_target bar ≈ 0.1383. 0.12 fails; 0.15 clears.
    def treatments(p_by_step: dict[int, float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _student_row(
                    student_step=step,
                    teacher_step=48000,
                    p_target=p_by_step[step],
                    cell="SI_SS",
                )
                for step in rungs
            ]
        )

    control_df = pd.DataFrame(controls)
    edge = build_onset(
        treatments({100: 0.12, 200: 0.15, 300: 0.15, 400: 0.12}),
        control_df,
        student_rungs=rungs,
    )
    assert edge["onset_step"].iloc[0] == 200
    assert bool(edge["censored"].iloc[0]) is False

    two_only = build_onset(
        treatments({100: 0.15, 200: 0.15, 300: 0.12, 400: 0.12}),
        control_df,
        student_rungs=(100, 200),
    )
    assert two_only["onset_step"].iloc[0] == 100
    assert bool(two_only["censored"].iloc[0]) is False

    never = build_onset(
        treatments({100: 0.12, 200: 0.12, 300: 0.12, 400: 0.12}),
        control_df,
        student_rungs=rungs,
    )
    assert pd.isna(never["onset_step"].iloc[0])
    assert bool(never["censored"].iloc[0]) is True


def test_onset_deterministic_control_sd_zero_strict_clear() -> None:
    """A single control leaf has SD 0; a clear must strictly beat its mean."""
    rungs = (100, 200, 300)
    control_df = pd.DataFrame(
        [
            _student_row(
                student_step=step,
                teacher_base=None,
                teacher_step=None,
                cell=None,
                p_target=0.10,
                is_control=True,
                gseed=0,
            )
            for step in rungs
        ]
    )
    treatments = pd.DataFrame(
        [
            _student_row(
                student_step=step,
                teacher_step=48000,
                p_target=p,
                cell="SI_SS",
            )
            for step, p in ((100, 0.10), (200, 0.11), (300, 0.11))
        ]
    )
    edge = build_onset(treatments, control_df, student_rungs=rungs)
    # step 100 equals the control mean -> not a clear (strict); 200/300 clear.
    assert edge["onset_step"].iloc[0] == 200
    assert bool(edge["censored"].iloc[0]) is False


def test_penalty_trend_monotone_decreasing() -> None:
    rows = []
    for step, cross in ((100, 0.10), (1000, 0.20), (10000, 0.30)):
        rows.append(
            {
                "level": "agg",
                "arm": "joint",
                "student_base": "original",
                "teacher_base": "seed1",
                "student_step": step,
                "teacher_step": step,
                "cell": "DI_DS",
                "delta": cross,
                "set": _SET,
                "scoring": _SCORE,
            }
        )
        rows.append(
            {
                "level": "agg",
                "arm": "joint",
                "student_base": "seed1",
                "teacher_base": "seed1",
                "student_step": step,
                "teacher_step": step,
                "cell": "SI_SS",
                "delta": 0.40,
                "set": _SET,
                "scoring": _SCORE,
            }
        )
    frame = build_penalty_trend(
        pd.DataFrame(rows),
        joint_rungs=(100, 1000, 10000),
        bootstrap_n=_BOOT,
        seed=_SEED,
    )
    by_step = frame.set_index("step")
    assert by_step.loc[100, "penalty"] == pytest.approx(0.75)
    assert by_step.loc[1000, "penalty"] == pytest.approx(0.50)
    assert by_step.loc[10000, "penalty"] == pytest.approx(0.25)
    assert frame["spearman"].iloc[0] == pytest.approx(-1.0)
    assert not bool(frame["degenerate"].any())


def test_penalty_degenerate_when_anchor_nonpositive() -> None:
    rows = [
        {
            "level": "agg",
            "arm": "joint",
            "student_base": "original",
            "teacher_base": "seed1",
            "student_step": 2000,
            "teacher_step": 2000,
            "cell": "DI_DS",
            "delta": 0.10,
            "set": _SET,
            "scoring": _SCORE,
        },
        {
            "level": "agg",
            "arm": "joint",
            "student_base": "seed1",
            "teacher_base": "seed1",
            "student_step": 2000,
            "teacher_step": 2000,
            "cell": "SI_SS",
            "delta": 0.0,
            "set": _SET,
            "scoring": _SCORE,
        },
    ]
    frame = build_penalty_trend(
        pd.DataFrame(rows), joint_rungs=(2000,), bootstrap_n=_BOOT, seed=_SEED
    )
    assert bool(frame["degenerate"].iloc[0]) is True
    assert pd.isna(frame["penalty"].iloc[0])


def test_marginals_fixed_slices() -> None:
    rows = []
    for t_step, delta in ((100, 0.10), (200, 0.20), (300, 0.30)):
        rows.append(
            {
                "level": "agg",
                "arm": "grid",
                "student_base": "original",
                "teacher_base": "original",
                "student_step": 143000,
                "teacher_step": t_step,
                "cell": "SI_SS",
                "delta": delta,
                "set": _SET,
                "scoring": _SCORE,
            }
        )
    for s_step, delta in ((100, 0.05), (200, 0.15)):
        rows.append(
            {
                "level": "agg",
                "arm": "grid",
                "student_base": "original",
                "teacher_base": "original",
                "student_step": s_step,
                "teacher_step": 48000,
                "cell": "SI_SS",
                "delta": delta,
                "set": _SET,
                "scoring": _SCORE,
            }
        )
    frame = build_marginals(
        pd.DataFrame(rows),
        teacher_rungs=(100, 200, 300),
        student_rungs=(100, 200),
        fixed_student_step=143000,
        fixed_teacher_step=48000,
    )
    teacher = frame[frame["slice"] == SLICE_TEACHER].sort_values("teacher_step")
    student = frame[frame["slice"] == SLICE_STUDENT].sort_values("student_step")
    assert list(teacher["delta"]) == pytest.approx([0.10, 0.20, 0.30])
    assert list(student["delta"]) == pytest.approx([0.05, 0.15])
