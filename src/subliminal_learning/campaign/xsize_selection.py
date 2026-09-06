"""Teacher admission, fallback-tag gate, and per-size LR pick (pure).

`teacher_gate` returns a decision per size and never raises: a size with
no admissible teacher is ``no_admissible_teacher``. The driver writes
`teacher_gate_report.json` and then decides whether to raise
`CampaignGateError`.

`pick_lr` is the pre-registered P2→P3 rule: PPL guard, argmax Δ_sb,
tie band back to `LR_DEFAULT`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign.xsize_plan import (
    ADMISSION_ABS_MIN,
    ADMISSION_SD_MULTIPLE,
    LR_DEFAULT,
    LR_LADDER,
    MIN_ADMITTED_PER_BASE,
    PICK_PPL_RATIO_MAX,
    PICK_TIE_BAND,
    SIZES,
    TEACHER_FALLBACK_SIZES,
    TEACHER_FALLBACK_TAGS,
    TEACHER_PARAM_TAG,
    TEACHER_PPL_RATIO_MAX,
)


class CampaignGateError(RuntimeError):
    """Hard campaign stop after a persisted gate report is written."""


@dataclass(frozen=True)
class TeacherAdmission:
    """One teacher leaf's admission evidence.

    Attributes:
        size: Base size token.
        tag: Teacher param tag.
        ftseed: Teacher fine-tune seed.
        admitted: Whether every admission criterion held.
        ppl_ratio: Teacher PPL / base PPL, or None if missing.
        p_target: Teacher ``p(target)``.
        control_mean: Matched-base control mean ``p(target)``.
        control_sd: Control sample SD, or None (treated as 0).
        reasons: Machine-readable failure tokens (empty when admitted).
    """

    size: str
    tag: str
    ftseed: int
    admitted: bool
    ppl_ratio: float | None
    p_target: float | None
    control_mean: float | None
    control_sd: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        """JSON-ready record."""
        return {
            "size": self.size,
            "tag": self.tag,
            "ftseed": self.ftseed,
            "admitted": self.admitted,
            "ppl_ratio": self.ppl_ratio,
            "p_target": self.p_target,
            "control_mean": self.control_mean,
            "control_sd": self.control_sd,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class TeacherGateDecision:
    """Per-size teacher-tag verdict.

    Attributes:
        size: Base size token.
        tag: Chosen tag, or None when no teacher admitted.
        status: ``frozen``, ``fallback``, or ``no_admissible_teacher``.
        n_admitted_frozen: Admitted leaves at `TEACHER_PARAM_TAG`.
        fallback_candidates: Admitted fallback tags at tfs0, best-first.
        reasons: Why the status was chosen.
    """

    size: str
    tag: str | None
    status: str
    n_admitted_frozen: int
    fallback_candidates: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        """JSON-ready record."""
        return {
            "size": self.size,
            "tag": self.tag,
            "status": self.status,
            "n_admitted_frozen": self.n_admitted_frozen,
            "fallback_candidates": list(self.fallback_candidates),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class LrPick:
    """Per-student-size LR decision.

    Attributes:
        size: Student size token.
        lr: Chosen learning rate.
        flagged: True when no ladder LR passed the PPL guard.
        reason: Machine-readable decision token.
        table: One dict per `LR_LADDER` entry (delta, ppl, guarded).
    """

    size: str
    lr: float
    flagged: bool
    reason: str
    table: tuple[dict, ...]

    def to_dict(self) -> dict:
        """JSON-ready record."""
        return {
            "size": self.size,
            "lr": self.lr,
            "flagged": self.flagged,
            "reason": self.reason,
            "table": [dict(row) for row in self.table],
        }


def admit_teacher(
    *,
    p_target: float | None,
    control_mean: float | None,
    control_sd: float | None,
    ppl_ratio: float | None,
    sd_multiple: float = ADMISSION_SD_MULTIPLE,
    abs_min: float = ADMISSION_ABS_MIN,
    ppl_max: float = TEACHER_PPL_RATIO_MAX,
) -> tuple[bool, tuple[str, ...]]:
    """Score one teacher leaf against the admission rule.

    Admitted iff ``p >= control + k·SD``, ``p − control >= abs_min``, and
    ``ppl_ratio <= ppl_max``. Missing PPL is ``ppl_missing`` (not admitted).
    ``control_sd is None`` is treated as SD 0.

    Returns:
        ``(admitted, reasons)``. Reasons is empty when admitted.
    """
    reasons: list[str] = []
    if ppl_ratio is None:
        reasons.append("ppl_missing")
    if p_target is None:
        reasons.append("p_target_missing")
    if control_mean is None:
        reasons.append("control_missing")
    if reasons:
        return False, tuple(reasons)

    p = float(p_target)
    c = float(control_mean)
    sd = 0.0 if control_sd is None else float(control_sd)
    ratio = float(ppl_ratio)
    if p < c + sd_multiple * sd:
        reasons.append(
            f"below_control_plus_{sd_multiple:g}sd"
            f"(teacher={p:.4f},control={c:.4f},sd={sd:.4f})"
        )
    if p - c < abs_min:
        reasons.append(f"below_abs_min(delta={p - c:.4f},min={abs_min:g})")
    if ratio > ppl_max:
        reasons.append(f"ppl_ratio({ratio:.4f}>{ppl_max:g})")
    return (not reasons), tuple(reasons)


def teacher_gate(
    rows: Sequence[TeacherAdmission],
) -> dict[str, TeacherGateDecision]:
    """Choose a teacher tag per size. Never raises.

    Frozen tag if at least `MIN_ADMITTED_PER_BASE` frozen-tag leaves
    admitted. Else, for sizes in `TEACHER_FALLBACK_SIZES`, the admitted
    tfs0 fallback with highest ``p_target``. Else
    ``no_admissible_teacher``.
    """
    by_size: dict[str, list[TeacherAdmission]] = {s: [] for s in SIZES}
    for row in rows:
        by_size.setdefault(row.size, []).append(row)

    decisions: dict[str, TeacherGateDecision] = {}
    for size in SIZES:
        size_rows = by_size.get(size, [])
        frozen = [r for r in size_rows if r.tag == TEACHER_PARAM_TAG and r.admitted]
        n_frozen = len(frozen)
        fallback_admitted = [
            r
            for r in size_rows
            if r.tag in TEACHER_FALLBACK_TAGS and r.ftseed == 0 and r.admitted
        ]
        fallback_admitted.sort(
            key=lambda r: (
                -(r.p_target if r.p_target is not None else float("-inf")),
                TEACHER_FALLBACK_TAGS.index(r.tag)
                if r.tag in TEACHER_FALLBACK_TAGS
                else 99,
            )
        )
        candidates = tuple(dict.fromkeys(r.tag for r in fallback_admitted))

        if n_frozen >= MIN_ADMITTED_PER_BASE:
            decisions[size] = TeacherGateDecision(
                size=size,
                tag=TEACHER_PARAM_TAG,
                status="frozen",
                n_admitted_frozen=n_frozen,
                fallback_candidates=candidates,
                reasons=(),
            )
            continue
        if size in TEACHER_FALLBACK_SIZES and fallback_admitted:
            best = fallback_admitted[0]
            decisions[size] = TeacherGateDecision(
                size=size,
                tag=best.tag,
                status="fallback",
                n_admitted_frozen=n_frozen,
                fallback_candidates=candidates,
                reasons=(
                    f"frozen_admitted={n_frozen}<{MIN_ADMITTED_PER_BASE}",
                    f"fallback={best.tag}",
                ),
            )
            continue
        if size not in TEACHER_FALLBACK_SIZES:
            why = (
                f"frozen_admitted={n_frozen}<{MIN_ADMITTED_PER_BASE}",
                "no_fallback_trained",
            )
        else:
            why = (
                f"frozen_admitted={n_frozen}<{MIN_ADMITTED_PER_BASE}",
                "no_admitted_fallback",
            )
        decisions[size] = TeacherGateDecision(
            size=size,
            tag=None,
            status="no_admissible_teacher",
            n_admitted_frozen=n_frozen,
            fallback_candidates=candidates,
            reasons=why,
        )
    return decisions


def gate_report(decisions: Mapping[str, TeacherGateDecision]) -> dict:
    """JSON body for `teacher_gate_report.json`."""
    return {
        "decisions": {
            size: decisions[size].to_dict() for size in SIZES if size in decisions
        },
        "thresholds": {
            "sd_multiple": ADMISSION_SD_MULTIPLE,
            "abs_min": ADMISSION_ABS_MIN,
            "ppl_max": TEACHER_PPL_RATIO_MAX,
            "min_admitted_per_base": MIN_ADMITTED_PER_BASE,
            "frozen_tag": TEACHER_PARAM_TAG,
            "fallback_tags": list(TEACHER_FALLBACK_TAGS),
            "fallback_sizes": list(TEACHER_FALLBACK_SIZES),
        },
    }


def _require_ladder_keys(mapping: Mapping[float, float], name: str) -> None:
    got = set(mapping)
    want = set(LR_LADDER)
    if got != want:
        raise ValueError(
            f"{name} keys must be exactly LR_LADDER {LR_LADDER}; got {tuple(mapping)}"
        )


def pick_lr(
    *,
    size: str,
    delta_by_lr: Mapping[float, float],
    ppl_by_lr: Mapping[float, float],
) -> LrPick:
    """Choose ``lr*(size)`` from the P2 ladder.

    Keys of both mappings must be exactly `LR_LADDER`. Guard
    ``ppl <= PICK_PPL_RATIO_MAX``. None pass → `LR_DEFAULT`, flagged.
    Best remaining by Δ; if it beats `LR_DEFAULT` by less than
    `PICK_TIE_BAND` and the default passed the guard, take the default.
    """
    _require_ladder_keys(delta_by_lr, "delta_by_lr")
    _require_ladder_keys(ppl_by_lr, "ppl_by_lr")
    table = tuple(
        {
            "lr": lr,
            "delta": float(delta_by_lr[lr]),
            "ppl_ratio": float(ppl_by_lr[lr]),
            "passes_guard": float(ppl_by_lr[lr]) <= PICK_PPL_RATIO_MAX,
        }
        for lr in LR_LADDER
    )
    ok = [lr for lr in LR_LADDER if float(ppl_by_lr[lr]) <= PICK_PPL_RATIO_MAX]
    if not ok:
        return LrPick(
            size=size,
            lr=LR_DEFAULT,
            flagged=True,
            reason="no_lr_passed_ppl_guard",
            table=table,
        )
    best = max(ok, key=lambda lr: float(delta_by_lr[lr]))
    if (
        LR_DEFAULT in ok
        and float(delta_by_lr[best]) - float(delta_by_lr[LR_DEFAULT]) < PICK_TIE_BAND
    ):
        return LrPick(
            size=size,
            lr=LR_DEFAULT,
            flagged=False,
            reason="tie_band",
            table=table,
        )
    reason = "best_delta" if best != LR_DEFAULT else "default"
    return LrPick(
        size=size,
        lr=best,
        flagged=False,
        reason=reason,
        table=table,
    )


def picks_report(picks: Mapping[str, LrPick]) -> dict:
    """JSON body for `screen_picks.json`."""
    return {
        "picks": {size: pick.to_dict() for size, pick in picks.items()},
        "thresholds": {
            "ppl_ratio_max": PICK_PPL_RATIO_MAX,
            "tie_band": PICK_TIE_BAND,
            "lr_default": LR_DEFAULT,
            "lr_ladder": list(LR_LADDER),
        },
    }


def picks_from_report(doc: Mapping[str, object]) -> dict[str, float]:
    """``{size: lr}`` from a `picks_report` document."""
    raw = doc.get("picks")
    if not isinstance(raw, Mapping):
        raise ValueError("screen_picks document is missing a picks mapping")
    out: dict[str, float] = {}
    for size, entry in raw.items():
        if isinstance(entry, Mapping) and "lr" in entry:
            out[str(size)] = float(entry["lr"])
        else:
            out[str(size)] = float(entry)  # type: ignore[arg-type]
    return out
