"""Pairing set of the bear pretraining-checkpoint grid (pure).

Grid arm: original←original at every (teacher_rung, student_rung). Joint arm:
15 directed pairings (including the two stream-only original↔data-seed1
comparators required by G7) at the 6 JOINT_RUNGS, diagonal only.

Cell labels follow `docs/POLYPYTHIA_160M_SEED_AUDIT.md`: ``seed{N}`` is joint
(init AND data order); ``original`` + ``data-seed{1,2,3}`` is the stream-only
family; uploaded ``weight-seed`` repos are a defect and raise.

Leaf replication reuses `ckpt_replication.diagonal_assignments` so (ftseed,
gseed) rotation matches the cat ladder. Revision-carrying labels
(``original@step128``) are what config-gen writes into
``teacher_student_base_pairs`` so cross-rung leaves never collide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from subliminal_learning.campaign.ckpt_pairs import Cell
from subliminal_learning.campaign.ckpt_replication import diagonal_assignments
from subliminal_learning.campaign.ckptgrid_plan import (
    DATA_SEED_BASE,
    FTSEEDS,
    GRID_BASE,
    GSEEDS,
    JOINT_RUNGS,
    JOINT_SEED_BASES,
    STUDENT_RUNGS,
    TEACHER_RUNGS,
)
from subliminal_learning.teachers.model_specs import parse_base_models

Arm = Literal["grid", "joint"]

# Direction is student←teacher. Pairings 14–15 (original↔data-seed1) are a
# documented deviation from the source plan's 13: G7's joint-vs-data contrast
# is uncomputable without stream-only cells at the joint rungs.
JOINT_PAIRINGS: tuple[tuple[str, str], ...] = (
    ("seed1", "seed1"),
    ("seed2", "seed2"),
    ("seed3", "seed3"),
    ("original", "seed1"),
    ("seed1", "original"),
    ("original", "seed2"),
    ("seed2", "original"),
    ("original", "seed3"),
    ("seed3", "original"),
    ("data-seed1", "seed1"),
    ("seed1", "data-seed1"),
    ("seed1", "seed2"),
    ("seed2", "seed1"),
    ("original", "data-seed1"),
    ("data-seed1", "original"),
)

JOINT_CONTROL_BASES: tuple[str, ...] = JOINT_SEED_BASES + (DATA_SEED_BASE,)
TEACHER_JOINT_BASES: tuple[str, ...] = JOINT_SEED_BASES + (DATA_SEED_BASE,)


def revision(rung: int) -> str:
    """Hub revision of a rung, e.g. ``2000 -> 'step2000'``.

    Args:
        rung: Pretraining step; must be a positive published checkpoint.

    Raises:
        ValueError: On a non-positive rung (the random-init branch is excluded).
    """
    if rung <= 0:
        raise ValueError(f"rung must be a positive published step, got {rung}")
    return f"step{rung}"


def _spec(label: str):
    """Single `PythiaModelSpec` for a campaign label (revision stripped by parse)."""
    specs = parse_base_models(label)
    if len(specs) != 1:
        raise ValueError(
            f"label {label!r} is an aggregate selector, not one model "
            f"(resolved to {len(specs)} specs)"
        )
    return specs[0]


def cell_of(base_a: str, base_b: str) -> Cell:
    """Audit-correct 2x2 cell of a pairing (revisions are stripped).

    Same base → SI_SS. Both in the data-order family
    {original, data-seed1..3} → SI_DS. Any pair where at least one side is
    ``seed{N}`` (N in 1-9) and the bases differ → DI_DS. A weight-seed label
    anywhere raises (those repos are not an init ablation).

    Args:
        base_a: Student or teacher base label, with or without ``@revision``.
        base_b: The other side's base label.

    Returns:
        The `Cell` the pairing contributes to.

    Raises:
        ValueError: On a weight-seed label or an unparsable token.
    """
    spec_a = _spec(base_a)
    spec_b = _spec(base_b)
    if spec_a.variant == "weight" or spec_b.variant == "weight":
        raise ValueError(
            f"weight-seed bases are excluded (seed-audit defect): "
            f"{base_a!r} vs {base_b!r}"
        )
    same_base = spec_a.variant == spec_b.variant and spec_a.seed == spec_b.seed
    if same_base:
        return Cell.SI_SS
    data_family_a = spec_a.variant == "data" or (
        spec_a.variant == "full" and spec_a.seed == 0
    )
    data_family_b = spec_b.variant == "data" or (
        spec_b.variant == "full" and spec_b.seed == 0
    )
    if data_family_a and data_family_b:
        return Cell.SI_DS
    joint_a = spec_a.variant == "full" and spec_a.seed != 0
    joint_b = spec_b.variant == "full" and spec_b.seed != 0
    if joint_a or joint_b:
        return Cell.DI_DS
    raise ValueError(f"unclassified pairing {base_a!r} vs {base_b!r}")


@dataclass(frozen=True)
class GridPair:
    """One (student_base, teacher_base, student_step, teacher_step) combo.

    Attributes:
        student_base: Student campaign label, revision-free.
        teacher_base: Teacher campaign label, revision-free.
        student_step: Student pretraining step.
        teacher_step: Teacher pretraining step.
        arm: ``grid`` (original←original rectangle) or ``joint`` (diagonal).
        index: Pairing index for `diagonal_assignments` (grid: combo index;
            joint: index into `JOINT_PAIRINGS`, shared across rungs).
    """

    student_base: str
    teacher_base: str
    student_step: int
    teacher_step: int
    arm: Arm
    index: int

    @property
    def cell(self) -> Cell:
        """The audit `Cell` of the two bases."""
        return cell_of(self.student_base, self.teacher_base)

    @property
    def student_label(self) -> str:
        """``{student_base}@step{student_step}``."""
        return f"{self.student_base}@{revision(self.student_step)}"

    @property
    def teacher_label(self) -> str:
        """``{teacher_base}@step{teacher_step}``."""
        return f"{self.teacher_base}@{revision(self.teacher_step)}"

    @property
    def labels(self) -> tuple[str, str]:
        """``(student_label, teacher_label)`` — sweep pair order."""
        return self.student_label, self.teacher_label


@dataclass(frozen=True)
class StudentLeaf:
    """One rotated-diagonal (ftseed, gseed) leaf of a `GridPair`."""

    pair: GridPair
    ftseed: int
    gseed: int


@dataclass(frozen=True)
class ControlSpec:
    """One base-teacher control cell (no treatment teacher)."""

    student_base: str
    student_step: int
    gseed: int
    arm: Arm

    @property
    def student_label(self) -> str:
        """``{student_base}@step{student_step}``."""
        return f"{self.student_base}@{revision(self.student_step)}"


@dataclass(frozen=True)
class TeacherCombo:
    """One (base, rung) teacher training combo (before × ftseeds)."""

    base: str
    step: int

    @property
    def label(self) -> str:
        """``{base}@step{step}``."""
        return f"{self.base}@{revision(self.step)}"


@dataclass(frozen=True)
class TeacherLeaf:
    """One teacher leaf: a `TeacherCombo` at one ftseed."""

    combo: TeacherCombo
    ftseed: int


def grid_pairs() -> tuple[GridPair, ...]:
    """168 original←original combos: 12 teacher rungs × 14 student rungs."""
    out: list[GridPair] = []
    idx = 0
    for teacher_step in TEACHER_RUNGS:
        for student_step in STUDENT_RUNGS:
            out.append(
                GridPair(
                    student_base=GRID_BASE,
                    teacher_base=GRID_BASE,
                    student_step=student_step,
                    teacher_step=teacher_step,
                    arm="grid",
                    index=idx,
                )
            )
            idx += 1
    return tuple(out)


def joint_pairs() -> tuple[GridPair, ...]:
    """90 joint-arm pairs: 15 pairings × 6 JOINT_RUNGS, diagonal."""
    out: list[GridPair] = []
    for pairing_index, (student_base, teacher_base) in enumerate(JOINT_PAIRINGS):
        for rung in JOINT_RUNGS:
            out.append(
                GridPair(
                    student_base=student_base,
                    teacher_base=teacher_base,
                    student_step=rung,
                    teacher_step=rung,
                    arm="joint",
                    index=pairing_index,
                )
            )
    return tuple(out)


def pair_leaves(pairs: Sequence[GridPair]) -> tuple[StudentLeaf, ...]:
    """Rotated-diagonal (ftseed, gseed) leaves of `pairs` (3 per pair)."""
    return tuple(
        StudentLeaf(pair=pair, ftseed=ftseed, gseed=gseed)
        for pair in pairs
        for ftseed, gseed in diagonal_assignments(pair.index)
    )


def grid_student_leaves() -> tuple[StudentLeaf, ...]:
    """504 grid-arm student leaves (168 combos × 3 diagonal replicates)."""
    return pair_leaves(grid_pairs())


def joint_student_leaves() -> tuple[StudentLeaf, ...]:
    """270 joint-arm student leaves (90 pairs × 3 diagonal replicates)."""
    return pair_leaves(joint_pairs())


def grid_controls() -> tuple[ControlSpec, ...]:
    """42 grid-arm controls: original × 14 STUDENT_RUNGS × 3 gseeds."""
    return tuple(
        ControlSpec(
            student_base=GRID_BASE,
            student_step=step,
            gseed=gseed,
            arm="grid",
        )
        for step in STUDENT_RUNGS
        for gseed in GSEEDS
    )


def joint_controls() -> tuple[ControlSpec, ...]:
    """72 joint-arm controls: 4 bases × 6 JOINT_RUNGS × 3 gseeds.

    original@JOINT_RUNGS controls already exist in `grid_controls` and are
    not duplicated here.
    """
    return tuple(
        ControlSpec(
            student_base=base,
            student_step=step,
            gseed=gseed,
            arm="joint",
        )
        for base in JOINT_CONTROL_BASES
        for step in JOINT_RUNGS
        for gseed in GSEEDS
    )


def teacher_combos() -> tuple[TeacherCombo, ...]:
    """36 (base, rung) teacher combos.

    original @ all 12 TEACHER_RUNGS; seed{1,2,3} and data-seed1 @ 6
    JOINT_RUNGS.
    """
    original = tuple(TeacherCombo(GRID_BASE, step) for step in TEACHER_RUNGS)
    joint = tuple(
        TeacherCombo(base, step) for base in TEACHER_JOINT_BASES for step in JOINT_RUNGS
    )
    return original + joint


def teacher_leaves() -> tuple[TeacherLeaf, ...]:
    """108 teacher leaves: 36 combos × 3 ftseeds."""
    return tuple(
        TeacherLeaf(combo=combo, ftseed=ftseed)
        for combo in teacher_combos()
        for ftseed in FTSEEDS
    )
