"""Leaf and control enumeration for the xfam wave-2 campaign (`xfam2`).

Pure: every function is a deterministic expansion of `xfam2_plan.units()`, given the
teacher ftseeds a persisted gate admitted, so a re-advance rebuilds an identical
jobkey set and the submitted-once queue guard holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import xfam2_plan as plan

SB = "sb"
TB = "tb"


@dataclass(frozen=True)
class Leaf:
    """One treatment cell: one admitted teacher checkpoint distilled into a student."""

    unit: plan.Unit
    ftseed: int
    gseed: int

    @property
    def key(self) -> tuple:
        """Identity used for uniqueness checks."""
        c = self.unit.cell
        return (c.teacher, c.student, c.lr, self.ftseed, self.gseed)


@dataclass(frozen=True)
class Control:
    """One control cell: an untuned base distilled into the student.

    Attributes:
        kind: ``"sb"`` (student's own base, the self unit's control) or ``"tb"`` (the
            teacher's base, a cross unit's control).
        student: Receiver label.
        teacher_base: Untuned emitter; equals `student` for ``"sb"``.
        lr: Student learning rate.
        gseed: Generation seed.
        tier: Tier of the unit that trains it.
    """

    kind: str
    student: str
    teacher_base: str
    lr: float
    gseed: int
    tier: str

    @property
    def key(self) -> tuple:
        """Identity used for dedup (a control depends on no teacher checkpoint)."""
        return (self.kind, self.student, self.teacher_base, self.lr, self.gseed)


def unit_leaves(
    unit: plan.Unit, ftseeds: Sequence[int] | None = None
) -> tuple[Leaf, ...]:
    """Treatment leaves of one unit, over `ftseeds` (default: the unit's own).

    Raises:
        ValueError: If `ftseeds` names a seed outside the unit's ftseeds.
    """
    seeds = tuple(unit.ftseeds if ftseeds is None else ftseeds)
    extra = set(seeds) - set(unit.ftseeds)
    if extra:
        raise ValueError(f"{unit.slug}: ftseeds {sorted(extra)} not in {unit.ftseeds}")
    return tuple(Leaf(unit, fs, g) for fs in seeds for g in unit.gseeds)


def unit_controls(unit: plan.Unit) -> tuple[Control, ...]:
    """Controls one unit trains: sb for a self unit, tb for a cross unit."""
    c = unit.cell
    kind = SB if c.is_self else TB
    return tuple(
        Control(kind, c.student, c.teacher, c.lr, g, unit.tier) for g in unit.gseeds
    )


def self_unit_of(unit: plan.Unit) -> plan.Unit:
    """The student's self unit at the same LR and tier (its sb controls).

    Raises:
        ValueError: If the design has no such unit.
    """
    c = unit.cell
    for other in plan.units():
        oc = other.cell
        if (
            other.tier == unit.tier
            and oc.is_self
            and oc.student == c.student
            and oc.lr == c.lr
        ):
            return other
    raise ValueError(f"{unit.slug}: no self unit for {c.student} at lr {c.lr:g}")


def all_leaves() -> tuple[Leaf, ...]:
    """Every treatment leaf, at every unit's full ftseed set."""
    return tuple(leaf for unit in plan.units() for leaf in unit_leaves(unit))


def all_controls() -> tuple[Control, ...]:
    """Every control of the campaign.

    Raises:
        ValueError: If two units would train the same control, or a unit's student
            has no self unit to supply its sb control.
    """
    seen: dict[tuple, Control] = {}
    for unit in plan.units():
        self_unit_of(unit)
        for control in unit_controls(unit):
            if control.key in seen:
                raise ValueError(f"control {control.key} trained by two units")
            seen[control.key] = control
    return tuple(seen.values())


def counts(tier: str) -> tuple[int, int]:
    """``(treatment leaves, controls)`` of one tier."""
    units = [u for u in plan.units() if u.tier == tier]
    return (
        sum(len(unit_leaves(u)) for u in units),
        sum(len(unit_controls(u)) for u in units),
    )
