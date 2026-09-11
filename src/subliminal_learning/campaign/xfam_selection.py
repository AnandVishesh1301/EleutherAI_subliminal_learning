"""Teacher admission, per-student LR pick, and the replication gate (pure).

Every decision here is written to disk once and read back on resume, never
recomputed: the LR is part of the student's GKD tag, so a re-derived pick would
change the jobkey set of an already-submitted queue.

`admit_teacher` is reused verbatim from `campaign.xsize_selection` -- the admission
rule is the same pre-registered one, only the label space differs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.xsize_selection import (
    CampaignGateError,
    admit_teacher,
)

__all__ = [
    "CampaignGateError",
    "LrPick",
    "ReplicationDecision",
    "TeacherAdmission",
    "TeacherGateDecision",
    "admit_teacher",
    "gate_report",
    "picks_from_report",
    "picks_report",
    "pick_lr",
    "replication_decision",
    "replication_report",
    "replications_from_report",
    "teacher_gate",
]


@dataclass(frozen=True)
class TeacherAdmission:
    """One SDF teacher leaf's admission evidence.

    Attributes:
        label: Roster label of the fine-tuned base.
        tag: Teacher param tag.
        ftseed: Teacher fine-tune seed.
        admitted: Whether every admission criterion held.
        ppl_ratio: Teacher PPL / its own base PPL.
        p_target: Teacher ``p(target)``.
        control_mean: Untuned-base control mean ``p(target)``.
        control_sd: Control sample SD, or None (treated as 0).
        reasons: Machine-readable failure tokens (empty when admitted).
    """

    label: str
    tag: str
    ftseed: int
    admitted: bool
    ppl_ratio: float | None
    p_target: float | None
    control_mean: float | None
    control_sd: float | None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """JSON row for `teacher_gate_report.json` / `teacher_admission.csv`."""
        return {
            "label": self.label,
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
    """Whether one emitter may be used, and on what evidence.

    Attributes:
        label: Emitter label.
        role: ``"cluster"`` (needs a quorum of admitted ftseeds) or ``"reverse"``
            (a single tfs0 leaf; failure voids only its own arm).
        admitted_ftseeds: Ftseeds that passed admission at the frozen tag.
        usable: Whether the campaign may distil from this emitter.
        reason: Machine-readable outcome token.
    """

    label: str
    role: str
    admitted_ftseeds: tuple[int, ...]
    usable: bool
    reason: str

    def to_dict(self) -> dict:
        """JSON body for one emitter in `teacher_gate_report.json`."""
        return {
            "label": self.label,
            "role": self.role,
            "admitted_ftseeds": list(self.admitted_ftseeds),
            "usable": self.usable,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LrPick:
    """The persisted ``lr*(student)`` decision and the table behind it."""

    student: str
    lr: float
    flagged: bool
    reason: str
    table: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict:
        """JSON body for one student in `screen_picks.json`."""
        return {
            "student": self.student,
            "lr": self.lr,
            "flagged": self.flagged,
            "reason": self.reason,
            "table": [dict(row) for row in self.table],
        }


@dataclass(frozen=True)
class ReplicationDecision:
    """Whether a student's extra teacher seeds are worth training."""

    student: str
    replicate: bool
    delta: float | None
    ci_low: float | None
    ci_high: float | None
    reason: str

    def to_dict(self) -> dict:
        """JSON body for one student in `replication_gate.json`."""
        return {
            "student": self.student,
            "replicate": self.replicate,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "reason": self.reason,
        }


def teacher_gate(
    rows: Sequence[TeacherAdmission],
) -> dict[str, TeacherGateDecision]:
    """Decide, per emitter, whether the campaign may distil from it. Never raises.

    A cluster emitter needs `plan.MIN_ADMITTED_PER_CLUSTER_TEACHER` admitted ftseeds
    at the frozen tag; failing that, its whole cluster is void and the driver raises
    after writing the report. A reverse emitter needs its single tfs0 leaf; failing
    that, only its own arm is void.
    """
    cluster_emitters = set(models.TEACHER_OF.values())
    by_label: dict[str, list[TeacherAdmission]] = {}
    for row in rows:
        if row.tag != plan.TEACHER_PARAM_TAG:
            continue
        by_label.setdefault(row.label, []).append(row)

    decisions: dict[str, TeacherGateDecision] = {}
    for label in plan.teacher_labels():
        leaves = by_label.get(label, [])
        admitted = tuple(sorted(r.ftseed for r in leaves if r.admitted))
        if label in cluster_emitters:
            usable = len(admitted) >= plan.MIN_ADMITTED_PER_CLUSTER_TEACHER
            reason = (
                "frozen_tag"
                if usable
                else f"only_{len(admitted)}_admitted_"
                f"need_{plan.MIN_ADMITTED_PER_CLUSTER_TEACHER}"
            )
            role = "cluster"
        else:
            usable = bool(admitted)
            reason = "frozen_tag" if usable else "no_admissible_teacher"
            role = "reverse"
        decisions[label] = TeacherGateDecision(
            label=label,
            role=role,
            admitted_ftseeds=admitted,
            usable=usable,
            reason=reason,
        )
    return decisions


def gate_report(
    decisions: Mapping[str, TeacherGateDecision],
    rows: Sequence[TeacherAdmission],
) -> dict:
    """JSON body for `teacher_gate_report.json`."""
    return {
        "decisions": {label: d.to_dict() for label, d in decisions.items()},
        "leaves": [row.to_dict() for row in rows],
        "thresholds": {
            "sd_multiple": plan.ADMISSION_SD_MULTIPLE,
            "abs_min": plan.ADMISSION_ABS_MIN,
            "ppl_ratio_max": plan.TEACHER_PPL_RATIO_MAX,
            "min_admitted_per_cluster_teacher": (
                plan.MIN_ADMITTED_PER_CLUSTER_TEACHER
            ),
        },
    }


def pick_lr(
    *,
    student: str,
    delta_by_lr: Mapping[float, float],
    ppl_by_lr: Mapping[float, float],
) -> LrPick:
    """Choose ``lr*(student)`` from that student's ladder.

    Keys of both mappings must be exactly the student's `plan.LR_LADDERS` entry.
    Rungs whose mean ``perplexity_ratio`` exceeds `plan.PICK_PPL_RATIO_MAX` are
    excluded; none passing falls back to the student's default and flags it. Among
    the rest the largest delta wins, unless it beats the default by less than
    `plan.PICK_TIE_BAND` and the default passed, in which case the default wins.

    Raises:
        ValueError: If the mappings are not keyed by the student's ladder.
    """
    ladder = plan.LR_LADDERS.get(student)
    if ladder is None:
        raise ValueError(f"{student!r} has no LR ladder; it inherits a pick")
    for name, mapping in (("delta_by_lr", delta_by_lr), ("ppl_by_lr", ppl_by_lr)):
        if tuple(sorted(mapping)) != tuple(sorted(ladder)):
            raise ValueError(
                f"{name} keys must be exactly {ladder} for {student}; "
                f"got {tuple(sorted(mapping))}"
            )
    default = plan.LR_DEFAULTS[student]
    table = tuple(
        {
            "lr": lr,
            "delta": float(delta_by_lr[lr]),
            "ppl_ratio": float(ppl_by_lr[lr]),
            "passes_guard": float(ppl_by_lr[lr]) <= plan.PICK_PPL_RATIO_MAX,
        }
        for lr in ladder
    )
    ok = [lr for lr in ladder if float(ppl_by_lr[lr]) <= plan.PICK_PPL_RATIO_MAX]
    if not ok:
        return LrPick(student, default, True, "no_lr_passed_ppl_guard", table)
    best = max(ok, key=lambda lr: float(delta_by_lr[lr]))
    if (
        default in ok
        and float(delta_by_lr[best]) - float(delta_by_lr[default])
        < plan.PICK_TIE_BAND
    ):
        return LrPick(student, default, False, "tie_band", table)
    return LrPick(
        student, best, False, "best_delta" if best != default else "default", table
    )


def picks_report(picks: Mapping[str, LrPick]) -> dict:
    """JSON body for `screen_picks.json`."""
    return {
        "picks": {student: pick.to_dict() for student, pick in picks.items()},
        "thresholds": {
            "ppl_ratio_max": plan.PICK_PPL_RATIO_MAX,
            "tie_band": plan.PICK_TIE_BAND,
            "lr_defaults": dict(plan.LR_DEFAULTS),
            "lr_ladders": {k: list(v) for k, v in plan.LR_LADDERS.items()},
        },
    }


def picks_from_report(doc: Mapping[str, object]) -> dict[str, float]:
    """``{student: lr}`` from a `picks_report` document.

    Raises:
        ValueError: If the document has no picks mapping.
    """
    raw = doc.get("picks")
    if not isinstance(raw, Mapping):
        raise ValueError("screen_picks document is missing a picks mapping")
    out: dict[str, float] = {}
    for student, entry in raw.items():
        if isinstance(entry, Mapping) and "lr" in entry:
            out[str(student)] = float(entry["lr"])
        else:
            out[str(student)] = float(entry)  # type: ignore[arg-type]
    return out


def replication_decision(
    *,
    student: str,
    delta: float | None,
    ci_low: float | None,
    ci_high: float | None,
) -> ReplicationDecision:
    """Whether to train the extra teacher seeds for `student`.

    Replicate when the cell mean at ``lr*`` clears `plan.REPLICATION_MIN_DELTA` and
    its bootstrap CI excludes zero: extra teacher seeds only sharpen an effect that
    is already there, and a null cell is better spent on the tail.
    """
    if delta is None or ci_low is None or ci_high is None:
        return ReplicationDecision(student, False, delta, ci_low, ci_high, "no_estimate")
    if float(delta) < plan.REPLICATION_MIN_DELTA:
        return ReplicationDecision(
            student, False, delta, ci_low, ci_high, "below_min_delta"
        )
    if float(ci_low) <= 0.0 <= float(ci_high):
        return ReplicationDecision(
            student, False, delta, ci_low, ci_high, "ci_covers_zero"
        )
    return ReplicationDecision(student, True, delta, ci_low, ci_high, "transfers")


def replication_report(decisions: Mapping[str, ReplicationDecision]) -> dict:
    """JSON body for `replication_gate.json`."""
    return {
        "decisions": {s: d.to_dict() for s, d in decisions.items()},
        "thresholds": {"min_delta": plan.REPLICATION_MIN_DELTA},
    }


def replications_from_report(doc: Mapping[str, object]) -> dict[str, bool]:
    """``{student: replicate}`` from a `replication_report` document.

    Raises:
        ValueError: If the document has no decisions mapping.
    """
    raw = doc.get("decisions")
    if not isinstance(raw, Mapping):
        raise ValueError("replication_gate document is missing a decisions mapping")
    out: dict[str, bool] = {}
    for student, entry in raw.items():
        if isinstance(entry, Mapping):
            out[str(student)] = bool(entry.get("replicate"))
        else:
            out[str(student)] = bool(entry)
    return out
