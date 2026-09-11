"""Leaf and control enumeration for the cross-family OPD campaign.

Pure: every function is a deterministic expansion of `xfam_plan.ARMS` given the
persisted LR picks, so a re-advance rebuilds an identical jobkey set and the
submitted-once queue guard holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.selection import Recipe


@dataclass(frozen=True)
class Leaf:
    """One treatment cell: a teacher checkpoint distilled into one student."""

    arm: str
    cluster: str
    teacher: str
    student: str
    lr: float
    recipe: Recipe
    epochs: float
    ftseed: int
    gseed: int
    teacher_param_tag: str

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return plan.student_gkd_tag(self.lr, self.recipe, self.epochs)

    @property
    def key(self) -> tuple:
        """Identity used for uniqueness checks and jobkeys."""
        return (
            self.arm,
            self.teacher,
            self.student,
            self.teacher_param_tag,
            self.gkd_tag,
            self.ftseed,
            self.gseed,
        )


@dataclass(frozen=True)
class Control:
    """One control cell: an untuned base distilled into the student.

    Attributes:
        kind: ``"sb"`` (student's own base -- the trait-free floor) or ``"tb"``
            (the teacher's base -- absorbs what distilling from a different untuned
            model does to the student, so the delta isolates the trait).
        student: Receiver label.
        teacher_base: Emitter label for ``"tb"``; equals `student` for ``"sb"``.
        gkd_tag: Student GKD directory tag the control matches.
        gseed: Student generation seed.
    """

    kind: str
    cluster: str
    student: str
    teacher_base: str
    gkd_tag: str
    gseed: int

    @property
    def key(self) -> tuple:
        """Identity used for dedup and jobkeys."""
        return (self.kind, self.student, self.teacher_base, self.gkd_tag, self.gseed)


def unit_leaves(
    arm_name: str, student: str, picks: Mapping[str, float]
) -> tuple[Leaf, ...]:
    """Every treatment leaf of one ``(arm, student)`` submission unit.

    Raises:
        KeyError: If the unit's LR gate has not been decided yet.
    """
    a = plan.arm(arm_name)
    if student not in a.students:
        raise ValueError(f"{student!r} is not a student of arm {arm_name!r}")
    models.assert_pair_eligible(a.teacher, student)
    return tuple(
        Leaf(
            arm=a.name,
            cluster=a.cluster,
            teacher=a.teacher,
            student=student,
            lr=lr,
            recipe=recipe,
            epochs=a.epochs,
            ftseed=ftseed,
            gseed=gseed,
            teacher_param_tag=a.teacher_param_tag,
        )
        for lr in a.lrs_for(student, dict(picks))
        for recipe in a.recipes
        for ftseed in a.ftseeds
        for gseed in plan.GSEEDS
    )


def unit_controls(
    arm_name: str, student: str, picks: Mapping[str, float]
) -> tuple[Control, ...]:
    """Controls one unit needs, deduplicated within the unit.

    Both control classes are enumerated for every cross cell: the teacher-base
    control is the primary denominator, the student-base control the secondary one
    and the anchor's only one. An anchor cell (teacher == student) has a single
    control, since its two classes coincide.
    """
    seen: dict[tuple, Control] = {}
    for leaf in unit_leaves(arm_name, student, picks):
        sb = Control(
            kind="sb",
            cluster=leaf.cluster,
            student=student,
            teacher_base=student,
            gkd_tag=leaf.gkd_tag,
            gseed=leaf.gseed,
        )
        seen.setdefault(sb.key, sb)
        if leaf.teacher != student:
            tb = Control(
                kind="tb",
                cluster=leaf.cluster,
                student=student,
                teacher_base=leaf.teacher,
                gkd_tag=leaf.gkd_tag,
                gseed=leaf.gseed,
            )
            seen.setdefault(tb.key, tb)
    return tuple(seen.values())


def all_leaves(picks: Mapping[str, float]) -> tuple[Leaf, ...]:
    """Every treatment leaf of the campaign, in arm order."""
    return tuple(
        leaf
        for arm_name, student in plan.all_units()
        for leaf in unit_leaves(arm_name, student, picks)
    )


def all_controls(picks: Mapping[str, float]) -> tuple[Control, ...]:
    """Every control of the campaign, deduplicated across arms."""
    seen: dict[tuple, Control] = {}
    for arm_name, student in plan.all_units():
        for control in unit_controls(arm_name, student, picks):
            seen.setdefault(control.key, control)
    return tuple(seen.values())


def default_picks() -> dict[str, float]:
    """The pre-registered LR defaults, for dry-run enumeration before any pick."""
    return dict(plan.LR_DEFAULTS)


def core_cell_count(picks: Mapping[str, float]) -> int:
    """Treatment + control cells of the core arms (the completion criterion)."""
    core = {a.name for a in plan.core_arms()}
    leaves = [leaf for leaf in all_leaves(picks) if leaf.arm in core]
    controls = {
        control.key
        for arm_name, student in plan.all_units()
        if arm_name in core
        for control in unit_controls(arm_name, student, picks)
    }
    return len(leaves) + len(controls)
