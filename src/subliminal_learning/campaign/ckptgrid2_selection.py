"""Teacher-dose gate and HP-control escalation for ckptgrid2 (pure).

`screen_dose` labels every designed ``(rung, base, epochs)`` teacher
``ok`` / ``advisory`` / ``quarantine`` with the same edges as
`ckptgrid_selection._disposition`. `dose_gate` is the one hard P1
check: every dose-ladder rung has at least one ``ok`` teacher.
Monotonicity of ``p_target`` in epochs is reported, never gating.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from subliminal_learning.campaign.ckptgrid2_plan import (
    ADMISSION_ABS_MIN,
    ADMISSION_SD_MULTIPLE,
    DOSE_EPOCHS,
    DOSE_RUNGS,
    HP_CONTROL_ESCALATION_MAX,
    PRIMARY_PROMPT_SET,
    PRIMARY_SCORING,
)

DOSE_BASES: tuple[str, ...] = ("original",)


class AdmissionTier(StrEnum):
    """Per-cell teacher disposition."""

    OK = "ok"
    ADVISORY = "advisory"
    QUARANTINE = "quarantine"


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass(frozen=True)
class DoseRow:
    """One ``(rung, base, epochs)`` teacher's tier and evidence.

    Attributes:
        rung: Pretraining step of the base checkpoint.
        base: Campaign label (``original`` on the dose ladder).
        epochs: Teacher SFT epoch count.
        teacher_p_target: Mean teacher ``p(target)`` across ftseeds.
        control_mean: Mean matched-base control ``p(target)``.
        control_sd: Sample SD of control leaves, or ``0.0`` with fewer
            than two (deterministic-control convention).
        n_ftseed: Teacher leaves / ftseeds read.
        tier: `AdmissionTier` value.
        reasons: Machine-readable tokens (empty when ``ok``).
    """

    rung: int
    base: str
    epochs: int
    teacher_p_target: float | None
    control_mean: float | None
    control_sd: float | None
    n_ftseed: int
    tier: str
    reasons: tuple[str, ...]

    @property
    def is_ok(self) -> bool:
        """Whether the teacher cleared both admission criteria."""
        return self.tier == AdmissionTier.OK

    def to_dict(self) -> dict:
        """JSON-ready record for `dose_gate_report.json`."""
        return {
            "rung": self.rung,
            "base": self.base,
            "epochs": self.epochs,
            "teacher_p_target": self.teacher_p_target,
            "control_mean": self.control_mean,
            "control_sd": self.control_sd,
            "n_ftseed": self.n_ftseed,
            "tier": self.tier,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class GateCheck:
    """One dose-gate criterion's outcome."""

    name: str
    passed: bool
    detail: str
    value: float | None
    threshold: float | None

    def to_dict(self) -> dict:
        """JSON-ready record for `dose_gate_report.json`."""
        return {
            "check": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class DoseGateResult:
    """Hard per-rung ``ok`` verdict plus reported monotonicity.

    Attributes:
        passed: Whether every dose rung has at least one ``ok`` teacher.
        checks: The criteria, in evaluation order.
        ok_rungs: Dose rungs that cleared ``ok`` at some epoch.
        monotone_rungs: Rungs whose ``p_target`` is non-decreasing
            in epochs.
        non_monotone_rungs: Rungs whose dose curve decreases.
    """

    passed: bool
    checks: tuple[GateCheck, ...]
    ok_rungs: tuple[int, ...]
    monotone_rungs: tuple[int, ...]
    non_monotone_rungs: tuple[int, ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        """The failing checks' details (empty when the gate passed)."""
        return tuple(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)


def _primary_rows(frame: pd.DataFrame, prompt_set: str, scoring: str) -> pd.DataFrame:
    """Rows on the primary endpoint, when the frame records set/scoring."""
    out = frame
    if "set" in out.columns:
        out = out[out["set"] == prompt_set]
    if "scoring" in out.columns:
        out = out[out["scoring"] == scoring]
    return out


def _mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean and sample SD (ddof=1); SD is ``None`` with fewer than two."""
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
    epochs: int,
    teacher: Sequence[float],
    control: Sequence[float],
    sd_multiple: float,
    abs_min: float,
) -> DoseRow:
    """Tier one ``(rung, base, epochs)`` from teacher and control leaves."""
    t_mean, _ = _mean_sd(teacher)
    c_mean, c_sd = _mean_sd(control)

    reasons: list[str] = []
    if t_mean is None:
        reasons.append("no_teacher_leaf")
    if c_mean is None:
        reasons.append("no_control_leaf")
    stored_sd = c_sd
    effective_sd = c_sd
    if c_mean is not None and c_sd is None:
        effective_sd = 0.0
        stored_sd = 0.0
        reasons.append("control_sd_unavailable_deterministic_control_sd0")

    if t_mean is None or c_mean is None or effective_sd is None:
        clears_sd = False
    else:
        clears_sd = t_mean >= c_mean + sd_multiple * effective_sd
        if not clears_sd:
            reasons.append(
                f"below_control_plus_{sd_multiple:g}sd"
                f"(teacher={t_mean:.4f},control={c_mean:.4f},"
                f"sd={effective_sd:.4f})"
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
    return DoseRow(
        rung=int(rung),
        base=base,
        epochs=int(epochs),
        teacher_p_target=t_mean,
        control_mean=c_mean,
        control_sd=stored_sd,
        n_ftseed=len([v for v in teacher if _finite(v)]),
        tier=str(tier),
        reasons=() if tier is AdmissionTier.OK else tuple(reasons),
    )


def screen_dose(
    teacher_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = DOSE_RUNGS,
    bases: Sequence[str] = DOSE_BASES,
    epochs: Sequence[int] = DOSE_EPOCHS,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
    sd_multiple: float = ADMISSION_SD_MULTIPLE,
    abs_min: float = ADMISSION_ABS_MIN,
) -> dict[tuple[int, str, int], DoseRow]:
    """Label every designed ``(rung, base, epochs)`` teacher.

    ``ok`` iff the teacher clears ``control + sd_multiple * control_SD``
    **and** ``abs_min``; ``advisory`` on exactly one; ``quarantine``
    otherwise. A cell with no leaves is `quarantine` with a reason
    token. Controls match on ``(base, step)`` only.

    Args:
        teacher_df: Per-leaf teacher preference rows.
        rungs: Teacher steps to label.
        bases: Teacher base labels to label.
        epochs: Teacher epoch counts to label.
        prompt_set: Primary endpoint the tiers are read on.
        scoring: Preference-scoring variant.
        sd_multiple: SD multiple of the matched control.
        abs_min: Absolute ``p(target)`` floor.

    Returns:
        ``{(rung, base, epochs): DoseRow}`` over the labelled cells.

    Raises:
        ValueError: A required column is absent.
    """
    required = ("step", "base", "is_control", "p_target", "n_epochs")
    missing = [c for c in required if c not in teacher_df.columns]
    if missing:
        raise ValueError(
            f"teacher frame is missing required column {missing[0]!r}; "
            f"got {sorted(teacher_df.columns)}"
        )
    wanted = tuple(
        (int(r), str(b), int(e)) for r in rungs for b in bases for e in epochs
    )
    rows = _primary_rows(teacher_df, prompt_set, scoring)
    out: dict[tuple[int, str, int], DoseRow] = {}
    for rung, base, n_epochs in wanted:
        at = rows[(rows["step"].astype(int) == int(rung)) & (rows["base"] == base)]
        is_control = at["is_control"].astype(bool)
        controls = at.loc[is_control]
        teachers = at.loc[~is_control]
        if not teachers.empty:
            epoch_vals = pd.to_numeric(teachers["n_epochs"], errors="coerce")
            teachers = teachers.loc[epoch_vals == int(n_epochs)]
        if "ftseed" in teachers.columns and not teachers.empty:
            t_vals = (
                teachers.groupby("ftseed", dropna=False)["p_target"].mean().tolist()
            )
        else:
            t_vals = list(teachers["p_target"]) if not teachers.empty else []
        out[(int(rung), base, int(n_epochs))] = _disposition(
            rung,
            base,
            n_epochs,
            t_vals,
            list(controls["p_target"]),
            sd_multiple,
            abs_min,
        )
    return out


def _epoch_curve_is_monotone(rows: Sequence[DoseRow]) -> bool:
    """Whether finite teacher means are non-decreasing in epochs."""
    ordered = sorted(
        (r for r in rows if r.teacher_p_target is not None),
        key=lambda r: r.epochs,
    )
    if len(ordered) < 2:
        return True
    prev = float(ordered[0].teacher_p_target)
    for row in ordered[1:]:
        cur = float(row.teacher_p_target)
        if cur < prev:
            return False
        prev = cur
    return True


def dose_gate(
    rows: Mapping[tuple[int, str, int], DoseRow],
    *,
    dose_rungs: Sequence[int] = DOSE_RUNGS,
) -> DoseGateResult:
    """Hard check: every dose rung has at least one ``ok`` teacher.

    Monotonicity of ``p_target`` in epochs is reported on the result
    and never fails the gate.

    Args:
        rows: Output of `screen_dose`.
        dose_rungs: Rungs that must carry an ``ok`` teacher.

    Returns:
        A `DoseGateResult`; the pipeline raises on ``not passed``.
    """
    wanted = tuple(int(r) for r in dose_rungs)
    ok_rungs = tuple(
        r for r in wanted if any(d.rung == r and d.is_ok for d in rows.values())
    )
    missing = [r for r in wanted if r not in ok_rungs]
    ok_check = GateCheck(
        name="ok_per_dose_rung",
        passed=not missing,
        detail=(
            f"ok rungs {list(ok_rungs)} of {list(wanted)}"
            + (f"; missing {missing}" if missing else "")
        ),
        value=float(len(ok_rungs)),
        threshold=float(len(wanted)),
    )

    monotone: list[int] = []
    non_monotone: list[int] = []
    bases = sorted({d.base for d in rows.values()})
    for rung in wanted:
        at_rung = [d for d in rows.values() if d.rung == rung]
        rung_ok = True
        for base in bases:
            curve = [d for d in at_rung if d.base == base]
            if curve and not _epoch_curve_is_monotone(curve):
                rung_ok = False
                break
        if rung_ok:
            monotone.append(rung)
        else:
            non_monotone.append(rung)

    return DoseGateResult(
        passed=ok_check.passed,
        checks=(ok_check,),
        ok_rungs=ok_rungs,
        monotone_rungs=tuple(monotone),
        non_monotone_rungs=tuple(non_monotone),
    )


def _student_label(row: pd.Series) -> str:
    if "student_label" in row.index and pd.notna(row["student_label"]):
        return str(row["student_label"])
    base = str(row["student_base"])
    step = int(row["student_step"])
    return f"{base}@step{step}"


def hp_control_escalation(
    control_df: pd.DataFrame,
    *,
    max_p_target: float = HP_CONTROL_ESCALATION_MAX,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> tuple[tuple[str, str], ...]:
    """``(student_label, gkd_tag)`` pairs whose gseed-0 control exceeded.

    Args:
        control_df: Control preference rows (``is_control`` or raw).
        max_p_target: Floor; pairs with ``p_target > max_p_target``.
        prompt_set: Primary endpoint.
        scoring: Preference-scoring variant.

    Returns:
        Sorted unique pairs; ``()`` when every gseed-0 control is clean.

    Raises:
        ValueError: A required column is absent.
    """
    required = ("gkd_tag", "p_target")
    missing = [c for c in required if c not in control_df.columns]
    if missing:
        raise ValueError(f"control frame is missing required column {missing[0]!r}")
    has_label = "student_label" in control_df.columns
    has_parts = (
        "student_base" in control_df.columns and "student_step" in control_df.columns
    )
    if not has_label and not has_parts:
        raise ValueError("control frame is missing required column 'student_label'")
    rows = _primary_rows(control_df, prompt_set, scoring)
    if rows.empty:
        return ()
    if "is_control" in rows.columns:
        rows = rows.loc[rows["is_control"].astype(bool)]
    if "gseed" in rows.columns:
        gseed = pd.to_numeric(rows["gseed"], errors="coerce")
        rows = rows.loc[gseed.fillna(0) == 0]
    if rows.empty:
        return ()
    pairs: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        p_target = row["p_target"]
        if not _finite(p_target):
            continue
        if float(p_target) <= float(max_p_target):
            continue
        pairs.add((_student_label(row), str(row["gkd_tag"])))
    return tuple(sorted(pairs))


def _tier_counts(rows: Mapping[tuple[int, str, int], DoseRow]) -> dict[str, int]:
    counts = {str(t): 0 for t in AdmissionTier}
    for row in rows.values():
        counts[row.tier] += 1
    return counts


def dose_report(
    rows: Mapping[tuple[int, str, int], DoseRow],
    result: DoseGateResult,
    *,
    sd_multiple: float = ADMISSION_SD_MULTIPLE,
    abs_min: float = ADMISSION_ABS_MIN,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
    dose_rungs: Sequence[int] = DOSE_RUNGS,
) -> dict:
    """JSON body for `logs/dose_gate_report.json`."""
    return {
        "passed": result.passed,
        "reasons": list(result.reasons),
        "checks": [c.to_dict() for c in result.checks],
        "ok_rungs": list(result.ok_rungs),
        "monotone_rungs": list(result.monotone_rungs),
        "non_monotone_rungs": list(result.non_monotone_rungs),
        "thresholds": {
            "sd_multiple": sd_multiple,
            "abs_min": abs_min,
            "dose_rungs": [int(r) for r in dose_rungs],
        },
        "prompt_set": prompt_set,
        "scoring": scoring,
        "labelling_only": True,
        "monotonicity_does_not_gate": True,
        "tier_counts": _tier_counts(rows),
        "cells": [
            d.to_dict()
            for d in sorted(rows.values(), key=lambda d: (d.rung, d.base, d.epochs))
        ],
    }


def escalation_report(
    pairs: Sequence[tuple[str, str]],
    control_df: pd.DataFrame,
    *,
    max_p_target: float = HP_CONTROL_ESCALATION_MAX,
) -> dict:
    """JSON body for `logs/hp_control_escalation.json`."""
    records = []
    for label, tag in pairs:
        records.append({"student_label": label, "gkd_tag": tag})
    return {
        "n": len(pairs),
        "pairs": records,
        "max_p_target": max_p_target,
        "gseed": 0,
        "n_control_rows": int(len(control_df)),
    }
