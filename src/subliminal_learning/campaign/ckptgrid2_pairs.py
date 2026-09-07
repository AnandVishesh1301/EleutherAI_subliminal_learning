"""Pairing set of the 20260908 bear checkpoint-grid stage-2 campaign (pure).

Descriptor-based `cell_of` (init_id × stream_id) so a wave-2 base is a
`BASES` table edit, not a name-pattern branch. Arm builders are the I3
SSOT: rectangles drop the diagonal, A′ drops teacher 143000 (arm A's own
diagonal), HP drops the frozen recipe, and `all_cells()` raises on a
duplicate `Cell2.key()`.

Leaf rotation reuses `ckpt_replication.diagonal_assignments` with the
cell's index **within its own arm**, matching the reference campaign.
`staged_controls` restates the reference symlink set; `seed2@step512` is
absent because the 20260828 grid never trained seed2 below rung 2000.

`reference_cells` restates the 20260828 bear grid / joint arm and the
20260805 cat ladder so an edit to those plan modules cannot reopen a
cell here. Constants used only for that restatement:

- bear teacher rungs (12): 512, 1000, then every 1000-aligned step in
  {2000, 4000, 8000, 16000, 32000, 48000, 64000, 96000, 128000, 143000}
- bear student rungs (14): 128, 256 plus the 12 teacher rungs
- bear joint pairings (15): the 20260828 `JOINT_PAIRINGS` list
- bear joint rungs (6): 2000, 8000, 16000, 48000, 96000, 143000
- cat ladder rungs (12): the same 12 teacher rungs
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import chain
from typing import Literal

from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.ckpt_pairs import Cell
from subliminal_learning.campaign.ckpt_replication import (
    N_REPLICATES,
    diagonal_assignments,
)
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag

Arm = Literal[
    "dose",
    "dose_teacher",
    "hp",
    "hp_probe",
    "rect",
    "rect512",
    "cat",
    "early_cross",
    "wave2",
]

RECTANGLES: tuple[tuple[str, str], ...] = (
    ("original", "seed2"),
    ("seed2", "original"),
    ("original", "data-seed1"),
    ("data-seed1", "original"),
    ("seed2", "seed2"),
)

# Restated from the 20260828 bear grid / 20260805 cat ladder. Not imported
# from those plan modules: an edit there must not reopen a cell here.
_REF_BEAR_TEACHER_RUNGS: tuple[int, ...] = (
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    48000,
    64000,
    96000,
    128000,
    143000,
)
_REF_BEAR_STUDENT_RUNGS: tuple[int, ...] = (128, 256) + _REF_BEAR_TEACHER_RUNGS
_REF_JOINT_RUNGS: tuple[int, ...] = (
    2000,
    8000,
    16000,
    48000,
    96000,
    143000,
)
_REF_JOINT_PAIRINGS: tuple[tuple[str, str], ...] = (
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
_REF_CAT_RUNGS: tuple[int, ...] = _REF_BEAR_TEACHER_RUNGS

_EARLY_CROSS: tuple[tuple[str, str, int], ...] = (
    ("original", "seed2", 3),
    ("original", "seed2", 8),
    ("seed2", "original", 8),
    ("seed2", "seed2", 3),
    ("seed2", "seed2", 8),
)
_WAVE2_SEEDS: tuple[str, ...] = ("weight-seed1", "weight-seed2", "weight-seed3")


def revision(rung: int) -> str:
    """Hub revision of a rung, e.g. ``2000 -> 'step2000'``.

    Args:
        rung: Pretraining step; must be a positive published checkpoint.

    Raises:
        ValueError: On a non-positive rung (I11: `step0` is excluded).
    """
    if rung <= 0:
        raise ValueError(f"rung must be a positive published step, got {rung}")
    return f"step{rung}"


def _descriptor(token: str) -> plan.BaseDescriptor:
    """`BASES` row for a campaign token; strips an optional ``@stepN``."""
    label = token.split("@", 1)[0]
    try:
        return plan.BASES[label]
    except KeyError:
        raise ValueError(f"unknown base label {token!r}") from None


def cell_of(a: str, b: str, *, audit_passed: bool = False) -> Cell:
    """2x2 cell from descriptor equality, not from name patterns.

    Args:
        a: Student or teacher base label, with or without ``@revision``.
        b: The other side's base label.
        audit_passed: Whether `logs/seed_audit.json` reads ``passed``.

    Returns:
        The `Cell` of ``(init_same, stream_same)``.

    Raises:
        ValueError: Unknown label, or an `audit_required` base before the
            seed audit has passed.
    """
    da, db = _descriptor(a), _descriptor(b)
    for desc in (da, db):
        if desc.audit_required and not audit_passed:
            raise ValueError(f"{desc.label!r} needs a passed seed audit (wave 2)")
    same_init = da.init_id == db.init_id
    same_stream = da.stream_id == db.stream_id
    return {
        (True, True): Cell.SI_SS,
        (True, False): Cell.SI_DS,
        (False, True): Cell.DI_SS,
        (False, False): Cell.DI_DS,
    }[(same_init, same_stream)]


def _lora_of(recipe: Recipe) -> LoRAConfig:
    return LoRAConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
    )


@dataclass(frozen=True)
class Cell2:
    """One trained (or reference) student cell.

    `key()` is the I3 identity and excludes `arm`, so two arms cannot hide
    a retrained cell behind different arm tokens.

    Attributes:
        arm: Pair-builder token.
        animal: Target animal.
        student_base: Revision-free student campaign label.
        teacher_base: Revision-free teacher campaign label.
        student_step: Student pretraining step.
        teacher_step: Teacher pretraining step.
        recipe: Student LoRA recipe.
        num_epochs: Student epoch budget.
        teacher_param_tag: Teacher SDF param tag.
        index: Position within this cell's own arm, assigned by the arm
            builder. `leaves_of` rotates on it, so a cell's ``(ftseed,
            gseed)`` leaves -- and therefore its on-disk paths -- are a
            property of the cell and not of the list it was passed in.
    """

    arm: Arm
    animal: str
    student_base: str
    teacher_base: str
    student_step: int
    teacher_step: int
    recipe: Recipe
    num_epochs: float
    teacher_param_tag: str
    index: int = 0

    @property
    def student_label(self) -> str:
        """``{student_base}@step{student_step}``."""
        return f"{self.student_base}@{revision(self.student_step)}"

    @property
    def teacher_label(self) -> str:
        """``{teacher_base}@step{teacher_step}``."""
        return f"{self.teacher_base}@{revision(self.teacher_step)}"

    @property
    def gkd_tag(self) -> str:
        """Student directory tag from `gkd.layout.gkd_tag`."""
        return gkd_tag(
            lmbda=plan.LMBDA,
            beta=plan.BETA,
            student_mode="lora",
            learning_rate=self.recipe.lr,
            num_epochs=self.num_epochs,
            lora=_lora_of(self.recipe),
        )

    @property
    def cell(self) -> Cell:
        """2x2 cell from descriptor equality."""
        return cell_of(
            self.student_base,
            self.teacher_base,
            audit_passed=self.arm == "wave2",
        )

    def key(self) -> tuple:
        """Hashable identity for I3 dedup (arm excluded)."""
        return (
            self.animal,
            self.student_base,
            self.teacher_base,
            self.student_step,
            self.teacher_step,
            self.gkd_tag,
            self.teacher_param_tag,
        )


@dataclass(frozen=True)
class Leaf2:
    """One rotated-diagonal ``(ftseed, gseed)`` leaf of a `Cell2`.

    Attributes:
        cell: Parent cell.
        ftseed: Teacher fine-tune seed.
        gseed: Student generation seed.
        replicate: Position within the three diagonal assignments.
    """

    cell: Cell2
    ftseed: int
    gseed: int
    replicate: int


@dataclass(frozen=True)
class Control2:
    """One base-teacher control cell (no treatment teacher).

    Attributes:
        animal: Target animal.
        student_base: Revision-free student campaign label.
        student_step: Student pretraining step.
        gkd_tag: Student directory tag the control matches.
        gseed: Student generation seed.
    """

    animal: str
    student_base: str
    student_step: int
    gkd_tag: str
    gseed: int

    @property
    def student_label(self) -> str:
        """``{student_base}@step{student_step}``."""
        return f"{self.student_base}@{revision(self.student_step)}"

    def key(self) -> tuple:
        """``(animal, student_label, gkd_tag, gseed)``."""
        return (self.animal, self.student_label, self.gkd_tag, self.gseed)


@dataclass(frozen=True)
class TeacherCombo:
    """One (base, rung, param_tag) teacher this campaign must train.

    Attributes:
        base: Teacher campaign label.
        step: Teacher pretraining step.
        param_tag: SDF param tag.
    """

    base: str
    step: int
    param_tag: str

    @property
    def label(self) -> str:
        """``{base}@step{step}``."""
        return f"{self.base}@{revision(self.step)}"


@dataclass(frozen=True)
class TeacherLeaf:
    """One teacher leaf: a `TeacherCombo` at one ftseed.

    Attributes:
        combo: Teacher combo.
        ftseed: Teacher fine-tune seed.
    """

    combo: TeacherCombo
    ftseed: int


def _cell(
    arm: Arm,
    animal: str,
    student_base: str,
    teacher_base: str,
    student_step: int,
    teacher_step: int,
    recipe: Recipe,
    num_epochs: float,
    teacher_param_tag: str,
    index: int = 0,
) -> Cell2:
    return Cell2(
        arm=arm,
        animal=animal,
        student_base=student_base,
        teacher_base=teacher_base,
        student_step=student_step,
        teacher_step=teacher_step,
        recipe=recipe,
        num_epochs=num_epochs,
        teacher_param_tag=teacher_param_tag,
        index=index,
    )


def _frozen(
    arm: Arm,
    animal: str,
    student_base: str,
    teacher_base: str,
    student_step: int,
    teacher_step: int,
    teacher_param_tag: str = plan.TEACHER_PARAM_TAG_LORA,
    index: int = 0,
) -> Cell2:
    return _cell(
        arm,
        animal,
        student_base,
        teacher_base,
        student_step,
        teacher_step,
        plan.FROZEN_RECIPE,
        plan.FROZEN_EPOCHS,
        teacher_param_tag,
        index,
    )


def _indexed(cells: Sequence[Cell2]) -> tuple[Cell2, ...]:
    """Stamp each cell with its position within its own arm."""
    return tuple(replace(cell, index=i) for i, cell in enumerate(cells))


def dose_cells() -> tuple[Cell2, ...]:
    """Arm A: `DOSE_RUNGS` diagonal × `DOSE_EPOCHS` (16)."""
    return _indexed(
        _frozen(
            "dose",
            "bear",
            "original",
            "original",
            rung,
            rung,
            plan.teacher_param_tag(epochs),
        )
        for rung in plan.DOSE_RUNGS
        for epochs in plan.DOSE_EPOCHS
    )


def dose_teacher_cells() -> tuple[Cell2, ...]:
    """Arm A′: student 143000 × teacher in `DOSE_RUNGS` except 143000 (12).

    143000 is arm A's own diagonal (I3); training it again here would
    duplicate a `key()`.
    """
    return _indexed(
        _frozen(
            "dose_teacher",
            "bear",
            "original",
            "original",
            plan.CHECKPOINT_MAX_STEP,
            teacher_step,
            plan.teacher_param_tag(epochs),
        )
        for teacher_step in plan.DOSE_RUNGS
        if teacher_step != plan.CHECKPOINT_MAX_STEP
        for epochs in plan.DOSE_EPOCHS
    )


def hp_cells() -> tuple[Cell2, ...]:
    """Arm B: `HP_CONFIGS` × `HP_RUNGS` diagonal (49)."""
    return _indexed(
        _cell(
            "hp",
            "bear",
            "original",
            "original",
            rung,
            rung,
            cfg.recipe,
            cfg.num_epochs,
            plan.TEACHER_PARAM_TAG_LORA,
        )
        for cfg in plan.HP_CONFIGS
        for rung in plan.HP_RUNGS
    )


def hp_probe_cells() -> tuple[Cell2, ...]:
    """Arm B′: `PROBE_HP_CONFIGS` × `PROBE_CELLS` (20)."""
    return _indexed(
        _cell(
            "hp_probe",
            "bear",
            "original",
            "original",
            student_step,
            teacher_step,
            cfg.recipe,
            cfg.num_epochs,
            plan.TEACHER_PARAM_TAG_LORA,
        )
        for cfg in plan.PROBE_HP_CONFIGS
        for student_step, teacher_step in plan.PROBE_CELLS
    )


def rect_cells(
    student_base: str,
    teacher_base: str,
    *,
    rungs: Sequence[int] = plan.JOINT_RUNGS,
) -> tuple[Cell2, ...]:
    """One C* rectangle: all ``(s, t)`` with ``s != t`` (30 at `JOINT_RUNGS`).

    Args:
        student_base: Student campaign label.
        teacher_base: Teacher campaign label.
        rungs: Teacher and student columns.

    Returns:
        Off-diagonal cells on the frozen recipe and frozen teacher tag.
    """
    return _indexed(
        _frozen("rect", "bear", student_base, teacher_base, s, t)
        for s in rungs
        for t in rungs
        if s != t
    )


def rect512_cells() -> tuple[Cell2, ...]:
    """Arm C1b/C2b: student 512 × 6 teacher columns × 2 rectangles (12).

    512 is not a teacher column, so no diagonal exclusion and no teacher
    is added by this arm.
    """
    return _indexed(
        _frozen(
            "rect512",
            "bear",
            student_base,
            teacher_base,
            plan.RECT512_STUDENT_RUNG,
            teacher_step,
        )
        for student_base, teacher_base in plan.RECT512_RECTANGLES
        for teacher_step in plan.JOINT_RUNGS
    )


def cat_cells() -> tuple[Cell2, ...]:
    """Arm D: cat `CAT_RUNGS` × `CAT_RUNGS` with ``s != t`` (42)."""
    return _indexed(
        _frozen("cat", "cat", "original", "original", s, t)
        for s in plan.CAT_RUNGS
        for t in plan.CAT_RUNGS
        if s != t
    )


def early_cross_cells() -> tuple[Cell2, ...]:
    """Arm G: rung 2000, the five (base pair, epochs) combos (5)."""
    rung = 2000
    return _indexed(
        _frozen(
            "early_cross",
            "bear",
            student_base,
            teacher_base,
            rung,
            rung,
            plan.teacher_param_tag(epochs),
        )
        for student_base, teacher_base, epochs in _EARLY_CROSS
    )


def wave2_cells(*, audit_passed: bool) -> tuple[Cell2, ...]:
    """Wave-2 DI_SS / SI_SS cells; empty unless `audit_passed`.

    Per `weight-seed{N}`: `original↔wsN` diagonal at `JOINT_RUNGS`,
    `wsN←wsN` anchor, and the `original↔wsN` rectangles.

    Args:
        audit_passed: Whether the seed-audit gate has passed.

    Returns:
        Wave-2 cells, or ``()`` when the audit has not passed.
    """
    if not audit_passed:
        return ()
    out: list[Cell2] = []
    for ws in _WAVE2_SEEDS:
        for student, teacher in (("original", ws), (ws, "original")):
            for rung in plan.JOINT_RUNGS:
                out.append(_frozen("wave2", "bear", student, teacher, rung, rung))
        for rung in plan.JOINT_RUNGS:
            out.append(_frozen("wave2", "bear", ws, ws, rung, rung))
        for student, teacher in (("original", ws), (ws, "original")):
            for s in plan.JOINT_RUNGS:
                for t in plan.JOINT_RUNGS:
                    if s == t:
                        continue
                    out.append(_frozen("wave2", "bear", student, teacher, s, t))
    return _indexed(out)


def _dupes(keys: Sequence[tuple]) -> list[tuple]:
    counts: dict[tuple, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return [key for key, n in counts.items() if n > 1]


def all_cells(*, wave: int = 1, audit_passed: bool = False) -> tuple[Cell2, ...]:
    """Wave-1 arms, plus wave 2 when ``wave == 2``.

    Args:
        wave: ``1`` (default) or ``2``.
        audit_passed: Forwarded to `wave2_cells`.

    Returns:
        Cells in builder order.

    Raises:
        ValueError: Two cells share a `key()` (I3).
    """
    cells: tuple[Cell2, ...] = (
        *dose_cells(),
        *dose_teacher_cells(),
        *hp_cells(),
        *hp_probe_cells(),
        *chain.from_iterable(rect_cells(s, t) for s, t in RECTANGLES),
        *rect512_cells(),
        *cat_cells(),
        *early_cross_cells(),
    )
    if wave == 2:
        cells = (*cells, *wave2_cells(audit_passed=audit_passed))
    keys = [cell.key() for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate cells: {_dupes(keys)}")
    return cells


def reference_cells() -> frozenset[tuple]:
    """Cell keys already trained in `REFERENCE_RUN_SETS`.

    Bear grid: `original←original` over the restated 12 teacher × 14
    student rungs at the frozen recipe and ep5. Bear joint: the restated
    15 directed pairings × 6 joint rungs. Cat ladder: `original←original`
    diagonal at the restated 12 rungs. Keys use `Cell2.key()` so the I3
    intersection is meaningful.
    """
    keys: list[tuple] = []
    for teacher_step in _REF_BEAR_TEACHER_RUNGS:
        for student_step in _REF_BEAR_STUDENT_RUNGS:
            keys.append(
                _frozen(
                    "dose",
                    "bear",
                    "original",
                    "original",
                    student_step,
                    teacher_step,
                ).key()
            )
    for student_base, teacher_base in _REF_JOINT_PAIRINGS:
        for rung in _REF_JOINT_RUNGS:
            keys.append(
                _frozen(
                    "rect",
                    "bear",
                    student_base,
                    teacher_base,
                    rung,
                    rung,
                ).key()
            )
    for rung in _REF_CAT_RUNGS:
        keys.append(_frozen("cat", "cat", "original", "original", rung, rung).key())
    return frozenset(keys)


def leaves_of(cells: Sequence[Cell2]) -> tuple[Leaf2, ...]:
    """Three rotated-diagonal leaves per cell.

    Rotates on `Cell2.index` (stamped by the arm builder), so the leaves
    of a cell are identical whether `cells` is the full `all_cells()`
    list or a single-arm subset -- the on-disk paths must not depend on
    what else was passed in.

    Args:
        cells: Cells to expand.

    Returns:
        Leaves in cell order, three per cell.
    """
    out: list[Leaf2] = []
    for cell in cells:
        for replicate, (ftseed, gseed) in zip(
            range(N_REPLICATES),
            diagonal_assignments(cell.index),
            strict=True,
        ):
            out.append(
                Leaf2(
                    cell=cell,
                    ftseed=ftseed,
                    gseed=gseed,
                    replicate=replicate,
                )
            )
    return tuple(out)


def _control_gseeds(cell: Cell2) -> tuple[int, ...]:
    if cell.gkd_tag == plan.FROZEN_GKD_TAG:
        return plan.GSEEDS
    return plan.HP_CONTROL_GSEEDS


def controls_of(cells: Sequence[Cell2]) -> tuple[Control2, ...]:
    """Deduped controls `cells` require, minus `staged_controls()` (I4).

    HP/probe recipes use `HP_CONTROL_GSEEDS`; the frozen recipe uses
    `GSEEDS`. Arm C1b/C2b therefore yields exactly `seed2@step512` on
    the frozen tag at all three gseeds; `original@step512` is staged.

    Args:
        cells: Treatment cells whose matched controls are required.

    Returns:
        Controls in first-seen order.
    """
    staged = staged_controls()
    seen: dict[tuple, Control2] = {}
    for cell in cells:
        for gseed in _control_gseeds(cell):
            ctrl = Control2(
                animal=cell.animal,
                student_base=cell.student_base,
                student_step=cell.student_step,
                gkd_tag=cell.gkd_tag,
                gseed=gseed,
            )
            key = ctrl.key()
            if key in staged:
                continue
            seen.setdefault(key, ctrl)
    return tuple(seen.values())


def staged_controls() -> frozenset[tuple]:
    """Control keys satisfied by the reference symlinks (frozen tag).

    Bear `original` at the restated 14 student rungs (includes 512);
    bear `seed2` / `data-seed1` at the restated joint rungs; cat
    `original` at the restated 12 cat-ladder rungs. `seed2@step512` is
    absent — the reference campaign never trained seed2 below 2000, so
    arm `rect512` needs a real new control.
    """
    tag = plan.FROZEN_GKD_TAG
    keys: list[tuple] = []
    for step in _REF_BEAR_STUDENT_RUNGS:
        for gseed in plan.GSEEDS:
            keys.append(Control2("bear", "original", step, tag, gseed).key())
    for base in ("seed2", "data-seed1"):
        for step in _REF_JOINT_RUNGS:
            for gseed in plan.GSEEDS:
                keys.append(Control2("bear", base, step, tag, gseed).key())
    for step in _REF_CAT_RUNGS:
        for gseed in plan.GSEEDS:
            keys.append(Control2("cat", "original", step, tag, gseed).key())
    return frozenset(keys)


def teacher_combos() -> tuple[TeacherCombo, ...]:
    """Teacher combos this campaign must train (18).

    `original` × `DOSE_RUNGS` × `DOSE_EPOCHS` (16) and `seed2@2000` ×
    ``{ep3, ep8}`` (2). Independent of `rect512_cells`: that arm adds no
    teacher column.
    """
    original = tuple(
        TeacherCombo("original", rung, plan.teacher_param_tag(epochs))
        for rung in plan.DOSE_RUNGS
        for epochs in plan.DOSE_EPOCHS
    )
    early = tuple(
        TeacherCombo("seed2", 2000, plan.teacher_param_tag(epochs)) for epochs in (3, 8)
    )
    return original + early


def teacher_leaves() -> tuple[TeacherLeaf, ...]:
    """Teacher leaves: 18 combos × 3 ftseeds (54)."""
    return tuple(
        TeacherLeaf(combo=combo, ftseed=ftseed)
        for combo in teacher_combos()
        for ftseed in plan.FTSEEDS
    )
