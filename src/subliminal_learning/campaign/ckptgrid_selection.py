"""Behavioural teacher gate for the bear checkpoint grid (pure).

`screen_admission` labels every designed ``(rung, base)`` teacher ``ok`` /
``advisory`` / ``quarantine``. Students submit regardless of tier; quarantine
is an analysis label (prereg §7).

`teacher_gate` is the **hard** P1→P2/P3 stop. It scores the original base
only: at least `TEACHER_GATE_MIN_OK_RUNGS` OK rungs and an original-base
curve span of at least `TEACHER_GATE_MIN_CURVE_SPAN`. Joint-base tiers are
written into the report and never gate. The pipeline converts a failure into
`CampaignGateError` after writing `teacher_gate_report.json`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from subliminal_learning.campaign.ckptgrid_plan import (
    ADMISSION_ABS_MIN,
    ADMISSION_SD_MULTIPLE,
    DATA_SEED_BASE,
    GRID_BASE,
    JOINT_RUNGS,
    JOINT_SEED_BASES,
    PRIMARY_PROMPT_SET,
    PRIMARY_SCORING,
    TEACHER_GATE_MIN_CURVE_SPAN,
    TEACHER_GATE_MIN_OK_RUNGS,
    TEACHER_RUNGS,
)

JOINT_ADMISSION_BASES: tuple[str, ...] = JOINT_SEED_BASES + (DATA_SEED_BASE,)


class AdmissionTier(StrEnum):
    """Per-cell teacher disposition."""

    OK = "ok"
    ADVISORY = "advisory"
    QUARANTINE = "quarantine"


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass(frozen=True)
class AdmissionDisposition:
    """One ``(rung, base)`` teacher's tier and the evidence behind it.

    Attributes:
        rung: Pretraining step of the base checkpoint.
        base: Campaign label (``original`` / ``seed1`` / ``data-seed1``).
        tier: `AdmissionTier` value.
        reasons: Machine-readable tokens (empty when ``ok``).
        teacher_p_target: Mean teacher ``p(target)`` across ftseeds.
        control_mean: Mean matched-base control ``p(target)`` at the same rung.
        control_sd: Sample SD of the control leaves, or ``None`` with fewer
            than two -- the deterministic-control case `_disposition` scores
            as 0.
        n_teacher: Teacher leaves read.
        n_control: Control leaves read.
    """

    rung: int
    base: str
    tier: str
    reasons: tuple[str, ...]
    teacher_p_target: float | None
    control_mean: float | None
    control_sd: float | None
    n_teacher: int
    n_control: int

    @property
    def is_ok(self) -> bool:
        """Whether the teacher cleared both admission criteria."""
        return self.tier == AdmissionTier.OK

    def to_dict(self) -> dict:
        """JSON-ready record for `teacher_gate_report.json`."""
        return {
            "rung": self.rung,
            "base": self.base,
            "tier": self.tier,
            "reasons": list(self.reasons),
            "teacher_p_target": self.teacher_p_target,
            "control_mean": self.control_mean,
            "control_sd": self.control_sd,
            "n_teacher": self.n_teacher,
            "n_control": self.n_control,
        }


@dataclass(frozen=True)
class GateCheck:
    """One teacher-gate criterion's outcome."""

    name: str
    passed: bool
    detail: str
    value: float | None
    threshold: float | None

    def to_dict(self) -> dict:
        """JSON-ready record for `teacher_gate_report.json`."""
        return {
            "check": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class TeacherGateResult:
    """The hard P1→P2/P3 verdict and every number behind it.

    Attributes:
        passed: Whether every evaluated check held.
        checks: The criteria, in evaluation order.
        ok_original_rungs: Original-base rungs that cleared ``ok``.
        curve_span: Original-base max−min mean teacher ``p_target``.
    """

    passed: bool
    checks: tuple[GateCheck, ...]
    ok_original_rungs: tuple[int, ...]
    curve_span: float | None

    @property
    def reasons(self) -> tuple[str, ...]:
        """The failing checks' details (empty when the gate passed)."""
        return tuple(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)


def admission_cells(
    *,
    teacher_rungs: Sequence[int] = TEACHER_RUNGS,
    joint_rungs: Sequence[int] = JOINT_RUNGS,
    grid_base: str = GRID_BASE,
    joint_bases: Sequence[str] = JOINT_ADMISSION_BASES,
) -> tuple[tuple[int, str], ...]:
    """Designed ``(rung, base)`` cells the gate labels.

    original @ every teacher rung; seed{1,2,3} and data-seed1 @ joint rungs.
    """
    original = tuple((int(rung), grid_base) for rung in teacher_rungs)
    joint = tuple((int(rung), base) for base in joint_bases for rung in joint_rungs)
    return original + joint


def _primary_rows(frame: pd.DataFrame, prompt_set: str, scoring: str) -> pd.DataFrame:
    """Rows on the primary endpoint, when the frame records set/scoring at all."""
    out = frame
    if "set" in out.columns:
        out = out[out["set"] == prompt_set]
    if "scoring" in out.columns:
        out = out[out["scoring"] == scoring]
    return out


def _mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean and sample SD (ddof=1); SD is ``None`` with fewer than two values."""
    vals = [float(v) for v in values if _finite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _disposition(
    rung: int,
    base: str,
    teacher: Sequence[float],
    control: Sequence[float],
    sd_multiple: float,
    abs_min: float,
) -> AdmissionDisposition:
    """Tier one ``(rung, base)`` from its teacher and matched-control leaves."""
    t_mean, _ = _mean_sd(teacher)
    c_mean, c_sd = _mean_sd(control)

    reasons: list[str] = []
    if t_mean is None:
        reasons.append("no_teacher_leaf")
    if c_mean is None:
        reasons.append("no_control_leaf")
    effective_sd = c_sd
    if c_mean is not None and c_sd is None:
        effective_sd = 0.0
        reasons.append("control_sd_unavailable_deterministic_control_sd0")

    if t_mean is None or c_mean is None or effective_sd is None:
        clears_sd = False
    else:
        clears_sd = t_mean >= c_mean + sd_multiple * effective_sd
        if not clears_sd:
            reasons.append(
                f"below_control_plus_{sd_multiple:g}sd"
                f"(teacher={t_mean:.4f},control={c_mean:.4f},sd={effective_sd:.4f})"
            )
    clears_abs = t_mean is not None and t_mean >= abs_min
    if t_mean is not None and not clears_abs:
        reasons.append(f"below_abs_min(teacher={t_mean:.4f},min={abs_min:g})")

    if clears_sd and clears_abs:
        tier = AdmissionTier.OK
    elif clears_sd or clears_abs:
        tier = AdmissionTier.ADVISORY
    else:
        tier = AdmissionTier.QUARANTINE
    return AdmissionDisposition(
        rung=int(rung),
        base=base,
        tier=str(tier),
        reasons=() if tier is AdmissionTier.OK else tuple(reasons),
        teacher_p_target=t_mean,
        control_mean=c_mean,
        control_sd=c_sd,
        n_teacher=len([v for v in teacher if _finite(v)]),
        n_control=len([v for v in control if _finite(v)]),
    )


def screen_admission(
    teacher_df: pd.DataFrame,
    *,
    rungs: Sequence[int] | None = None,
    bases: Sequence[str] | None = None,
    cells: Sequence[tuple[int, str]] | None = None,
    sd_multiple: float = ADMISSION_SD_MULTIPLE,
    abs_min: float = ADMISSION_ABS_MIN,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> dict[tuple[int, str], AdmissionDisposition]:
    """Label every designed ``(rung, base)`` teacher.

    ``ok`` iff the teacher clears ``control + sd_multiple * control_SD`` **and**
    ``abs_min`` absolute; ``advisory`` on exactly one; ``quarantine`` otherwise.
    A cell with no leaves on disk is `quarantine` with a reason token.

    Args:
        teacher_df: Per-leaf teacher preference rows. Required columns
            ``step``, ``base``, ``is_control``, ``p_target``; ``set`` /
            ``scoring`` are honoured when present.
        rungs: With `bases`, the rectangular grid to label.
        bases: With `rungs`, the rectangular grid to label.
        cells: Explicit ``(rung, base)`` list. Default is the designed
            non-rectangular grid (original @ teacher rungs; joint bases @
            joint rungs).
        sd_multiple: SD multiple of the matched control the teacher must clear.
        abs_min: Absolute ``p(target)`` floor the teacher must clear.
        prompt_set: Primary endpoint the tiers are read on.
        scoring: Preference-scoring variant.

    Returns:
        ``{(rung, base): AdmissionDisposition}`` over the labelled cells.

    Raises:
        ValueError: Required columns are absent.
    """
    required = {"step", "base", "is_control", "p_target"}
    missing = required - set(teacher_df.columns)
    if missing:
        raise ValueError(
            f"teacher frame is missing required columns {sorted(missing)}; got "
            f"{sorted(teacher_df.columns)}"
        )
    if cells is not None:
        wanted = tuple((int(r), str(b)) for r, b in cells)
    elif rungs is not None and bases is not None:
        wanted = tuple((int(r), str(b)) for r in rungs for b in bases)
    elif rungs is None and bases is None:
        wanted = admission_cells()
    else:
        raise ValueError("pass both rungs and bases, or cells, or neither")
    rows = _primary_rows(teacher_df, prompt_set, scoring)
    out: dict[tuple[int, str], AdmissionDisposition] = {}
    for rung, base in wanted:
        cell = rows[(rows["step"].astype(int) == int(rung)) & (rows["base"] == base)]
        out[(int(rung), base)] = _disposition(
            rung,
            base,
            list(cell[~cell["is_control"].astype(bool)]["p_target"]),
            list(cell[cell["is_control"].astype(bool)]["p_target"]),
            sd_multiple,
            abs_min,
        )
    return out


def original_ok_rungs(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
    *,
    grid_base: str = GRID_BASE,
) -> tuple[int, ...]:
    """Original-base rungs that reached ``ok``, ascending."""
    return tuple(
        sorted(
            {d.rung for d in dispositions.values() if d.base == grid_base and d.is_ok}
        )
    )


def original_curve_span(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
    *,
    grid_base: str = GRID_BASE,
) -> float | None:
    """max−min original-base mean teacher ``p_target``, or None if empty."""
    means = [
        float(d.teacher_p_target)
        for d in dispositions.values()
        if d.base == grid_base and d.teacher_p_target is not None
    ]
    if not means:
        return None
    return max(means) - min(means)


def teacher_gate(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
    teacher_curve_span: float | None,
    *,
    min_ok_rungs: int = TEACHER_GATE_MIN_OK_RUNGS,
    min_span: float = TEACHER_GATE_MIN_CURVE_SPAN,
    grid_base: str = GRID_BASE,
) -> TeacherGateResult:
    """Hard P1→P2/P3 gate on the original-base teacher curve.

    Joint-base dispositions are ignored. Students submit regardless of
    per-cell tier.

    Args:
        dispositions: Output of `screen_admission`.
        teacher_curve_span: Original-base max−min mean teacher ``p_target``.
            ``None`` / non-finite fails the span check.
        min_ok_rungs: Preregistered minimum of original ``ok`` rungs.
        min_span: Preregistered minimum original-base curve span.
        grid_base: Base the hard gate reads (original).

    Returns:
        A `TeacherGateResult`; the pipeline raises on ``not passed``.
    """
    ok = original_ok_rungs(dispositions, grid_base=grid_base)
    n_ok = len(ok)
    ok_check = GateCheck(
        name="min_ok_rungs",
        passed=n_ok >= min_ok_rungs,
        detail=(f"{n_ok} original-base ok rungs {list(ok)} (need >= {min_ok_rungs})"),
        value=float(n_ok),
        threshold=float(min_ok_rungs),
    )
    span_ok = (
        teacher_curve_span is not None
        and math.isfinite(teacher_curve_span)
        and teacher_curve_span >= min_span
    )
    span_detail = (
        f"original-base curve span {teacher_curve_span:.4f} (need >= {min_span:g})"
        if teacher_curve_span is not None and math.isfinite(teacher_curve_span)
        else "original-base curve span is missing"
    )
    span_check = GateCheck(
        name="curve_span",
        passed=span_ok,
        detail=span_detail,
        value=(
            float(teacher_curve_span)
            if teacher_curve_span is not None and math.isfinite(teacher_curve_span)
            else None
        ),
        threshold=float(min_span),
    )
    checks = (ok_check, span_check)
    return TeacherGateResult(
        passed=all(c.passed for c in checks),
        checks=checks,
        ok_original_rungs=ok,
        curve_span=(
            float(teacher_curve_span)
            if teacher_curve_span is not None and math.isfinite(teacher_curve_span)
            else None
        ),
    )


def tier_counts(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
) -> dict[str, int]:
    """Cells per tier, with every tier present (0 where empty)."""
    counts = {str(t): 0 for t in AdmissionTier}
    for disposition in dispositions.values():
        counts[disposition.tier] += 1
    return counts


def gate_report(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
    result: TeacherGateResult,
    *,
    sd_multiple: float = ADMISSION_SD_MULTIPLE,
    abs_min: float = ADMISSION_ABS_MIN,
    min_ok_rungs: int = TEACHER_GATE_MIN_OK_RUNGS,
    min_span: float = TEACHER_GATE_MIN_CURVE_SPAN,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> dict:
    """JSON body for `teacher_gate_report.json`."""
    return {
        "passed": result.passed,
        "reasons": list(result.reasons),
        "checks": [c.to_dict() for c in result.checks],
        "ok_original_rungs": list(result.ok_original_rungs),
        "curve_span": result.curve_span,
        "thresholds": {
            "sd_multiple": sd_multiple,
            "abs_min": abs_min,
            "min_ok_rungs": min_ok_rungs,
            "min_curve_span": min_span,
        },
        "prompt_set": prompt_set,
        "scoring": scoring,
        "labelling_only": True,
        "joint_bases_do_not_gate": True,
        "tier_counts": tier_counts(dispositions),
        "cells": [
            d.to_dict()
            for d in sorted(dispositions.values(), key=lambda d: (d.rung, d.base))
        ],
    }
