"""Cell, leaf, control and teacher enumeration of the ws160m campaign.

A cell is one directed (teacher base@step -> student base@step, animal, recipe,
epochs, teacher tag). Its leaves are ``(ftseed, gseed)`` pairs: three on the rotated
diagonal, six add the diagonal shifted by one, nine are the full cross. Controls and
teachers are owned by exactly one partition (`control_owner`, `plan.teacher_owner`),
so no two nodes train the same directory.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.selection import Recipe

Arm = Literal[
    "main",
    "wave2",
    "ladder",
    "dose_bear",
    "hp",
    "dose_cat_eagle",
    "heatmap",
    "teacher_dose",
    "dose3_cat_eagle",
    "dose_dog",
]
CELL_ARMS: tuple[str, ...] = tuple(a for a in plan.ARMS if a != "xbase")


@dataclass(frozen=True)
class Cell:
    """One trained student cell.

    Attributes:
        arm: Arm token.
        animal: Target animal.
        teacher_base: Teacher base name (inventory label).
        student_base: Student base name.
        teacher_step: Teacher pretraining step.
        student_step: Student pretraining step.
        recipe: Student LoRA recipe.
        num_epochs: Student epoch budget.
        teacher_param_tag: Teacher SDF param tag.
        leaves: Leaf count (3, 6 or 9).
        wave: Wave the cell belongs to.
        index: Position within its arm (drives the rotated diagonal).
    """

    arm: str
    animal: str
    teacher_base: str
    student_base: str
    teacher_step: int
    student_step: int
    recipe: Recipe
    num_epochs: float
    teacher_param_tag: str
    leaves: int = plan.LEAVES_DEFAULT
    wave: int = models.WAVE_NOW
    index: int = 0

    @property
    def partition(self) -> plan.Partition:
        """Partition that trains this cell."""
        return plan.partition_of_arm(self.arm)

    @property
    def student_token(self) -> str:
        return models.token(self.student_base, self.student_step)

    @property
    def teacher_token(self) -> str:
        return models.token(self.teacher_base, self.teacher_step)

    @property
    def student_label(self) -> str:
        return models.label(self.student_base, self.student_step)

    @property
    def teacher_label(self) -> str:
        return models.label(self.teacher_base, self.teacher_step)

    @property
    def gkd_tag(self) -> str:
        return plan.student_gkd_tag(self.recipe, self.num_epochs)

    @property
    def category(self) -> str:
        return models.category(self.teacher_base, self.student_base)

    @property
    def same_rung(self) -> bool:
        return self.teacher_step == self.student_step

    def key(self) -> tuple:
        """Hashable identity (arm and leaf count excluded)."""
        return (
            self.animal,
            self.teacher_base,
            self.student_base,
            self.teacher_step,
            self.student_step,
            self.gkd_tag,
            self.teacher_param_tag,
        )

    def control_key(self) -> tuple:
        """Identity of the `base_teacher` control this cell is read against."""
        return (self.animal, self.student_base, self.student_step, self.gkd_tag)


@dataclass(frozen=True)
class Leaf:
    """One ``(ftseed, gseed)`` leaf of a cell.

    Attributes:
        cell: Parent cell.
        ftseed: Teacher fine-tune seed.
        gseed: Student GKD seed.
        replicate: Position in the cell's leaf order (0-2 = rotated diagonal).
    """

    cell: Cell
    ftseed: int
    gseed: int
    replicate: int


@dataclass(frozen=True)
class Control:
    """One `base_teacher` control leaf (untuned student base as teacher)."""

    partition: str
    animal: str
    student_base: str
    student_step: int
    gkd_tag: str
    gseed: int

    @property
    def student_token(self) -> str:
        return models.token(self.student_base, self.student_step)

    def key(self) -> tuple:
        return (
            self.animal,
            self.student_base,
            self.student_step,
            self.gkd_tag,
            self.gseed,
        )


@dataclass(frozen=True)
class XbaseControl:
    """One `xbase_teacher` control leaf (teacher's untuned base as teacher)."""

    animal: str
    teacher_base: str
    student_base: str
    teacher_step: int
    student_step: int
    gkd_tag: str
    gseed: int
    partition: str = "extra"

    @property
    def student_token(self) -> str:
        return models.token(self.student_base, self.student_step)

    @property
    def teacher_token(self) -> str:
        return models.token(self.teacher_base, self.teacher_step)


@dataclass(frozen=True)
class TeacherCombo:
    """One SDF teacher config: (animal, base, step, param tag)."""

    animal: str
    base: str
    step: int
    param_tag: str

    @property
    def token(self) -> str:
        return models.token(self.base, self.step)

    @property
    def owner(self) -> plan.Partition:
        return plan.teacher_owner(self.step)

    @property
    def n_epochs(self) -> int:
        return int(self.param_tag.removeprefix("ep").split("_", 1)[0])


@dataclass(frozen=True)
class TeacherLeaf:
    combo: TeacherCombo
    ftseed: int


# ---- leaves -----------------------------------------------------------------------


def leaf_assignments(index: int, n_leaves: int) -> tuple[tuple[int, int], ...]:
    """``(ftseed, gseed)`` pairs for a cell: diagonal shifts ``0 .. n_leaves//3 - 1``.

    Three leaves are the rotated diagonal ``(r, (r + index) mod 3)``; six add the
    diagonal shifted by one; nine are the full cross.

    Raises:
        ValueError: Unless `n_leaves` is 3, 6 or 9.
    """
    if n_leaves not in (3, 6, 9):
        raise ValueError(f"leaves must be 3, 6 or 9, got {n_leaves}")
    n = len(plan.GSEEDS)
    return tuple(
        (r, (r + int(index) + shift) % n)
        for shift in range(n_leaves // n)
        for r in range(len(plan.FTSEEDS))
    )


def leaves_of(cells: Iterable[Cell]) -> tuple[Leaf, ...]:
    """All leaves of `cells`, replicate-numbered in assignment order."""
    return tuple(
        Leaf(cell=cell, ftseed=f, gseed=g, replicate=i)
        for cell in cells
        for i, (f, g) in enumerate(leaf_assignments(cell.index, cell.leaves))
    )


def _indexed(cells: Sequence[Cell]) -> tuple[Cell, ...]:
    return tuple(replace(c, index=i) for i, c in enumerate(cells))


def _cell(
    arm: str,
    animal: str,
    teacher_base: str,
    student_base: str,
    teacher_step: int,
    student_step: int,
    *,
    recipe: Recipe = plan.FROZEN_RECIPE,
    num_epochs: float = plan.FROZEN_EPOCHS,
    teacher_param_tag: str = plan.TEACHER_PARAM_TAG_LORA,
    leaves: int = plan.LEAVES_DEFAULT,
    wave: int = models.WAVE_NOW,
) -> Cell:
    models.base(teacher_base)
    models.base(student_base)
    return Cell(
        arm=arm,
        animal=animal,
        teacher_base=teacher_base,
        student_base=student_base,
        teacher_step=int(teacher_step),
        student_step=int(student_step),
        recipe=recipe,
        num_epochs=float(num_epochs),
        teacher_param_tag=teacher_param_tag,
        leaves=leaves,
        wave=wave,
    )


# ---- arm builders -----------------------------------------------------------------


def _leaves_for(teacher_base: str, student_base: str) -> int:
    return plan.LEAVES_BY_CATEGORY[models.category(teacher_base, student_base)]


def main_cells() -> tuple[Cell, ...]:
    """The full 11x11 directed bear grid at step143000 (wave 1)."""
    names = [m.base for m in models.bases(max_wave=models.WAVE_NOW)]
    return _indexed(
        [
            _cell(
                "main",
                plan.PRIMARY_ANIMAL,
                t,
                s,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                leaves=_leaves_for(t, s),
            )
            for t in names
            for s in names
        ]
    )


def wave2_cells() -> tuple[Cell, ...]:
    """The 23 directed cells `pile-ws3` adds at step143000 (self-pair at 9 leaves)."""
    w1 = [m.base for m in models.bases(max_wave=models.WAVE_NOW)]
    new = [m.base for m in models.bases(max_wave=models.WAVE_2) if m.wave == 2]
    cells: list[Cell] = []
    for n in new:
        cells.append(
            _cell(
                "wave2",
                plan.PRIMARY_ANIMAL,
                n,
                n,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                leaves=plan.LEAVES_BY_CATEGORY["SI_SS"],
                wave=models.WAVE_2,
            )
        )
        for other in w1:
            for t, s in ((n, other), (other, n)):
                cells.append(
                    _cell(
                        "wave2",
                        plan.PRIMARY_ANIMAL,
                        t,
                        s,
                        plan.FINAL_STEP,
                        plan.FINAL_STEP,
                        wave=models.WAVE_2,
                    )
                )
    return _indexed(cells)


def ladder_directed_pairs() -> tuple[tuple[str, str], ...]:
    """The 39 directed (teacher, student) base pairs of the ladder."""
    out = [(b, b) for b in plan.LADDER_SELF]
    for a, b in plan.LADDER_PAIRS:
        out += [(a, b), (b, a)]
    return tuple(out)


def ladder_cells() -> tuple[Cell, ...]:
    """Same-rung ladder: 39 directed pairs x `LADDER_RUNGS`."""
    return _indexed(
        [
            _cell("ladder", plan.PRIMARY_ANIMAL, t, s, rung, rung)
            for rung in plan.LADDER_RUNGS
            for t, s in ladder_directed_pairs()
        ]
    )


def dose_bear_cells() -> tuple[Cell, ...]:
    """Fig 1 bear self-pairs at the non-unit doses (1 epoch comes from `main`)."""
    return _indexed(
        [
            _cell(
                "dose_bear",
                plan.PRIMARY_ANIMAL,
                b,
                b,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                num_epochs=dose,
            )
            for b in plan.FIG1_BASES_BEAR
            for dose in plan.DOSES_BEAR
        ]
    )


def matched_directed_pairs(
    anchors: Sequence[str] = plan.HP_ANCHORS,
) -> tuple[tuple[str, str], ...]:
    """Directed pairs of the matched sub-design over `anchors` (deduplicated)."""
    out: list[tuple[str, str]] = []
    for anchor in anchors:
        for pair in [(anchor, anchor)] + [
            p
            for partner in models.MATCHED_SUBDESIGN[anchor].values()
            if partner is not None
            for p in ((anchor, partner), (partner, anchor))
        ]:
            if pair not in out:
                out.append(pair)
    return tuple(out)


def hp_cells() -> tuple[Cell, ...]:
    """Student-HP arm: matched sub-design x `HP_RECIPES` at step143000."""
    return _indexed(
        [
            _cell(
                "hp",
                plan.PRIMARY_ANIMAL,
                t,
                s,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                recipe=recipe,
            )
            for recipe in plan.HP_RECIPES
            for t, s in matched_directed_pairs()
        ]
    )


def _animal_dose_cells(
    arm: str, animals: Sequence[str], doses: Sequence[float]
) -> tuple[Cell, ...]:
    return _indexed(
        [
            _cell(arm, animal, b, b, plan.FINAL_STEP, plan.FINAL_STEP, num_epochs=dose)
            for animal in animals
            for b in plan.FIG1_BASES_OTHER
            for dose in doses
        ]
    )


def dose_cat_eagle_cells() -> tuple[Cell, ...]:
    """Fig 1 cat/eagle self-pairs at `DOSES_OTHER`."""
    return _animal_dose_cells("dose_cat_eagle", plan.FIG1_ANIMALS, plan.DOSES_OTHER)


def dose3_cat_eagle_cells() -> tuple[Cell, ...]:
    """Fig 1 cat/eagle self-pairs at dose 3 (extras)."""
    return _animal_dose_cells("dose3_cat_eagle", plan.FIG1_ANIMALS, (plan.DOSE3,))


def dose_dog_cells() -> tuple[Cell, ...]:
    """Fig 1 dog self-pairs at `DOSES_OTHER` (extras)."""
    return _animal_dose_cells("dose_dog", plan.EXTRA_ANIMALS, plan.DOSES_OTHER)


def heatmap_cells() -> tuple[Cell, ...]:
    """Off-diagonal teacher rung x student rung cells with student `pile-ws1`."""
    return _indexed(
        [
            _cell("heatmap", plan.PRIMARY_ANIMAL, t, plan.HEATMAP_STUDENT, ts, ss)
            for t in plan.HEATMAP_TEACHERS
            for ts in plan.HEATMAP_RUNGS
            for ss in plan.HEATMAP_RUNGS
            if ts != ss
        ]
    )


def teacher_dose_cells() -> tuple[Cell, ...]:
    """`pile-ws1`-anchored matched cells with ep2/ep8 teachers."""
    return _indexed(
        [
            _cell(
                "teacher_dose",
                plan.PRIMARY_ANIMAL,
                t,
                s,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                teacher_param_tag=plan.teacher_param_tag(ep),
            )
            for ep in plan.TEACHER_DOSE_EPOCHS
            for t, s in matched_directed_pairs((plan.TEACHER_DOSE_ANCHOR,))
        ]
    )


def xbase_controls() -> tuple[XbaseControl, ...]:
    """`xbase_teacher` controls for the 18 off-diagonal matched cells x gseeds."""
    return tuple(
        XbaseControl(
            animal=plan.PRIMARY_ANIMAL,
            teacher_base=t,
            student_base=s,
            teacher_step=plan.FINAL_STEP,
            student_step=plan.FINAL_STEP,
            gkd_tag=plan.FROZEN_GKD_TAG,
            gseed=g,
        )
        for t, s in matched_directed_pairs()
        if t != s
        for g in plan.GSEEDS
    )


_BUILDERS = {
    "main": main_cells,
    "wave2": wave2_cells,
    "ladder": ladder_cells,
    "dose_bear": dose_bear_cells,
    "hp": hp_cells,
    "dose_cat_eagle": dose_cat_eagle_cells,
    "heatmap": heatmap_cells,
    "teacher_dose": teacher_dose_cells,
    "dose3_cat_eagle": dose3_cat_eagle_cells,
    "dose_dog": dose_dog_cells,
}


def _dupes(keys: Sequence[tuple]) -> list[tuple]:
    seen: set[tuple] = set()
    out: list[tuple] = []
    for k in keys:
        if k in seen and k not in out:
            out.append(k)
        seen.add(k)
    return out


def all_cells(*, wave: int = models.WAVE_NOW) -> tuple[Cell, ...]:
    """Every cell of every arm up to `wave`; raises on a duplicate identity."""
    cells = tuple(c for arm in CELL_ARMS for c in _BUILDERS[arm]() if c.wave <= wave)
    keys = [c.key() for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate ws160m cells: {_dupes(keys)[:5]}")
    return cells


def cells_of(partition: str, *, wave: int = models.WAVE_NOW) -> tuple[Cell, ...]:
    """Cells trained by `partition` up to `wave`."""
    p = plan.partition(partition)
    return tuple(c for c in all_cells(wave=wave) if c.arm in p.arms)


# ---- controls ---------------------------------------------------------------------


def control_owner(key: tuple) -> str:
    """Partition that trains the `base_teacher` control with identity `key`.

    The first partition in `plan.PARTITION_ORDER` whose cells need it, so a control
    shared by several partitions is trained once.
    """
    for name in plan.PARTITION_ORDER:
        if any(c.control_key() == key for c in cells_of(name, wave=models.WAVE_2)):
            return name
    raise ValueError(f"no ws160m cell needs control {key!r}")


def controls_required(cells: Iterable[Cell]) -> tuple[tuple, ...]:
    """Distinct control identities the cells are read against, in first-seen order."""
    return tuple(dict.fromkeys(c.control_key() for c in cells))


def controls_of(partition: str, *, wave: int = models.WAVE_NOW) -> tuple[Control, ...]:
    """Control leaves `partition` trains: required by its cells and owned by it."""
    out: list[Control] = []
    for key in controls_required(cells_of(partition, wave=wave)):
        if control_owner(key) != partition:
            continue
        animal, base, step, tag = key
        out += [Control(partition, animal, base, step, tag, g) for g in plan.GSEEDS]
    return tuple(out)


def foreign_controls_of(
    partition: str, *, wave: int = models.WAVE_NOW
) -> tuple[tuple, ...]:
    """Control identities `partition` reads but another partition trains."""
    return tuple(
        key
        for key in controls_required(cells_of(partition, wave=wave))
        if control_owner(key) != partition
    )


# ---- teachers ---------------------------------------------------------------------


def teacher_combos_required(cells: Iterable[Cell]) -> tuple[TeacherCombo, ...]:
    """Distinct teacher configs the cells consume, in first-seen order."""
    seen: dict[tuple, TeacherCombo] = {}
    for c in cells:
        combo = TeacherCombo(
            c.animal, c.teacher_base, c.teacher_step, c.teacher_param_tag
        )
        seen.setdefault((combo.animal, combo.base, combo.step, combo.param_tag), combo)
    return tuple(seen.values())


def teacher_combos(
    partition: str, *, wave: int = models.WAVE_NOW
) -> tuple[TeacherCombo, ...]:
    """Teacher configs `partition` trains (owned by it, needed by any partition)."""
    plan.partition(partition)
    return tuple(
        combo
        for combo in teacher_combos_required(all_cells(wave=wave))
        if combo.owner.name == partition
    )


def teacher_leaves(
    partition: str, *, wave: int = models.WAVE_NOW
) -> tuple[TeacherLeaf, ...]:
    """Teacher leaves `partition` trains: every combo x `FTSEEDS`."""
    return tuple(
        TeacherLeaf(combo, f)
        for combo in teacher_combos(partition, wave=wave)
        for f in plan.FTSEEDS
    )


def required_teacher_leaves(
    partition: str, *, wave: int = models.WAVE_NOW
) -> tuple[TeacherLeaf, ...]:
    """Teacher leaves the partition's own cells read (any owner), deduplicated."""
    seen: dict[tuple, TeacherLeaf] = {}
    for leaf in leaves_of(cells_of(partition, wave=wave)):
        c = leaf.cell
        combo = TeacherCombo(
            c.animal, c.teacher_base, c.teacher_step, c.teacher_param_tag
        )
        seen.setdefault(
            (*combo.__dict__.values(), leaf.ftseed), TeacherLeaf(combo, leaf.ftseed)
        )
    return tuple(seen.values())


def foreign_teacher_leaves(
    partition: str, *, wave: int = models.WAVE_NOW
) -> tuple[TeacherLeaf, ...]:
    """Teacher leaves `partition` reads but another partition trains."""
    return tuple(
        t
        for t in required_teacher_leaves(partition, wave=wave)
        if t.combo.owner.name != partition
    )
