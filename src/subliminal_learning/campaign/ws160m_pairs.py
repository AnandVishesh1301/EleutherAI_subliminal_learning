"""Cell, leaf, control and teacher enumeration of the ws160m campaign.

A cell is one directed (teacher base@step -> student base@step, animal, recipe,
epochs, teacher tag). Its leaves are ``(ftseed, gseed)`` pairs: three on the rotated
diagonal, six add the diagonal shifted by one, nine are the full cross. Controls and
teachers are owned by exactly one partition (`control_owner`, `plan.teacher_owner`),
so no two nodes train the same directory.

Part II (`gen=2`) cells never repeat a leaf Part I landed: a cell identity may span
both generations only with disjoint leaves (`Cell.leaf_offset`), and teachers and
controls listed in `plan.GEN1_REUSED` stay owned by their Part I partition.

Part III (`gen=3`) enumerates only cell identities that neither Parts I-II nor the
reused ckptgrid / grid2 grids (`plan.EXTERNAL_REUSE`) hold, deals each stage's cells
round-robin over its four partitions, and owns no teacher and no control.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType
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
GEN2_CELL_ARMS: tuple[str, ...] = tuple(a for a in plan.GEN2_ARMS if a != "xb2")


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
        leaf_offset: First replicate this cell holds; earlier replicates of the same
            identity belong to another run set.
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
    leaf_offset: int = 0

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
    """One `base_teacher` control leaf (untuned student base as teacher).

    Attributes:
        stage: Student stage whose queue trains it (Part II); None = the
            partition's ``controls`` queue.
    """

    partition: str
    animal: str
    student_base: str
    student_step: int
    gkd_tag: str
    gseed: int
    stage: str | None = None

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
    """One SDF teacher config: (animal, base, step, param tag), as seen by `gen`."""

    animal: str
    base: str
    step: int
    param_tag: str
    gen: int = 1

    @property
    def token(self) -> str:
        return models.token(self.base, self.step)

    @property
    def owner(self) -> plan.Partition:
        """Partition whose run set holds this teacher."""
        return plan.teacher_home(
            self.animal, self.base, self.step, self.param_tag, gen=self.gen
        )

    @property
    def stage(self) -> str:
        """Teacher queue stage within `owner`."""
        if self.owner.gen == 1:
            return "teachers"
        return plan.gen2_teacher_stage(self.animal, self.step)[1]

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
    """All leaves of `cells`, replicate-numbered from each cell's `leaf_offset`."""
    out: list[Leaf] = []
    for cell in cells:
        pairs_ = leaf_assignments(cell.index, cell.leaf_offset + cell.leaves)
        out += [
            Leaf(cell=cell, ftseed=f, gseed=g, replicate=i)
            for i, (f, g) in enumerate(pairs_)
            if i >= cell.leaf_offset
        ]
    return tuple(out)


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


# ---- Part II arm builders -----------------------------------------------------------

_W3 = models.WAVE_3


def _leaves_gen2(teacher_base: str, student_base: str) -> int:
    cat = models.category(teacher_base, student_base)
    if cat == models.SI_SS_NONIDENT:
        return plan.LEAVES_NONIDENT
    return plan.LEAVES_BY_CATEGORY[cat]


@lru_cache(maxsize=1)
def gen1_landed_leaves() -> Mapping[tuple, frozenset[tuple[int, int]]]:
    """Cell key -> ``(ftseed, gseed)`` leaves Part I trained (wave 1, active stages)."""
    out: dict[tuple, set[tuple[int, int]]] = {}
    for lf in active_leaves(all_cells(wave=models.WAVE_NOW)):
        out.setdefault(lf.cell.key(), set()).add((lf.ftseed, lf.gseed))
    return MappingProxyType({k: frozenset(v) for k, v in out.items()})


def _new(cells: Iterable[Cell]) -> list[Cell]:
    """`cells` minus every identity Part I already landed."""
    landed = gen1_landed_leaves()
    return [c for c in cells if c.key() not in landed]


def _gen2(cell: Cell, arm: str, **changes: object) -> Cell:
    """A Part I cell re-homed to a Part II arm, keeping its index (leaf seeds)."""
    return replace(cell, arm=arm, wave=_W3, **changes)


def _names(max_wave: int) -> list[str]:
    return [m.base for m in models.bases(max_wave=max_wave)]


def m2_cells() -> tuple[Cell, ...]:
    """The 75 directed bear cells `pile-ws3`, `pile-ws2` and `pile-main` add at step143000."""
    old = set(_names(models.WAVE_NOW))
    return _indexed(
        [
            _cell(
                "m2",
                plan.PRIMARY_ANIMAL,
                t,
                s,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                leaves=_leaves_gen2(t, s),
                wave=_W3,
            )
            for t in _names(_W3)
            for s in _names(_W3)
            if not (t in old and s in old)
        ]
    )


def b2_cells() -> tuple[Cell, ...]:
    """Part I's deferred boost leaves (replicates 3..) of the 121 `main` cells."""
    return tuple(
        _gen2(c, "b2", leaf_offset=plan.LEAVES_DEFAULT, leaves=c.leaves - 3)
        for c in main_cells()
        if c.leaves > plan.LEAVES_DEFAULT
    )


def b3_cells() -> tuple[Cell, ...]:
    """The next diagonal shift (replicates 3-5) of every bear DI_DC cell."""
    return tuple(
        _gen2(c, "b3", leaf_offset=plan.LEAVES_DEFAULT, leaves=plan.LEAVES_DEFAULT)
        for c in (*main_cells(), *m2_cells())
        if c.category == "DI_DC"
    )


def ml_cells() -> tuple[Cell, ...]:
    """Lion, the full 14x14 directed grid at step143000, rotated diagonal."""
    return _indexed(
        [
            _cell("ml", "lion", t, s, plan.FINAL_STEP, plan.FINAL_STEP, wave=_W3)
            for t in _names(_W3)
            for s in _names(_W3)
        ]
    )


def h8_cells() -> tuple[Cell, ...]:
    """Part I's deferred `lr8e-4_r8a16` recipe on the 20 matched cells."""
    return tuple(_gen2(c, "h8") for c in hp_cells() if c.recipe == plan.H8_RECIPE)


def _law_cells(arm: str, animal: str, recipes: Sequence[Recipe]) -> tuple[Cell, ...]:
    ordered = sorted(
        recipes, key=lambda r: plan.LAW_PRODUCT_PRIORITY.index(plan.law_product(r))
    )
    return _indexed(
        _new(
            _cell(
                arm,
                animal,
                t,
                s,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                recipe=recipe,
                wave=_W3,
            )
            for recipe in ordered
            for t, s in plan.LAW_CELLS
        )
    )


def law_cells() -> tuple[Cell, ...]:
    """Bear law grid: `plan.LAW_ARM_RECIPES` on the six law cells, by product level."""
    return _law_cells("law", plan.PRIMARY_ANIMAL, plan.LAW_ARM_RECIPES)


def law_lion_cells() -> tuple[Cell, ...]:
    """Lion law grid: ranks 8 and 32 at lr 4e-4 on the six law cells."""
    return _law_cells("law_lion", "lion", plan.LAW_LION_RECIPES)


def rt_cells() -> tuple[Cell, ...]:
    """Rank transport: `RANK_TRANSPORT_RECIPES` on the 14 matched cells outside the law."""
    return _indexed(
        _new(
            _cell(
                "rt",
                plan.PRIMARY_ANIMAL,
                t,
                s,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                recipe=recipe,
                wave=_W3,
            )
            for recipe in plan.RANK_TRANSPORT_RECIPES
            for t, s in matched_directed_pairs()
            if (t, s) not in plan.LAW_CELLS
        )
    )


def td2_cells() -> tuple[Cell, ...]:
    """Part I's deferred teacher-dose students (ep2/ep8 teachers)."""
    return tuple(_gen2(c, "td2") for c in teacher_dose_cells())


def le_cells() -> tuple[Cell, ...]:
    """lr x epochs at r8/α16 on the six law cells."""
    return _indexed(
        _new(
            _cell(
                "le",
                plan.PRIMARY_ANIMAL,
                t,
                s,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                recipe=Recipe(
                    plan.FROZEN_RECIPE.lora_r, plan.FROZEN_RECIPE.lora_alpha, lr
                ),
                num_epochs=epochs,
                wave=_W3,
            )
            for lr, epochs in plan.LR_EPOCHS
            for t, s in plan.LAW_CELLS
        )
    )


def k2_directed_pairs() -> tuple[tuple[str, str], ...]:
    """The 61 directed (teacher, student) base pairs of the Part II ladder."""
    out = list(ladder_directed_pairs()) + [(b, b) for b in plan.K2_SELF_ADDED]
    for a, b in plan.K2_PAIRS_ADDED:
        out += [(a, b), (b, a)]
    return tuple(out)


def k2_cells() -> tuple[Cell, ...]:
    """Same-rung ladder over `K2_RUNGS` x 61 pairs, minus the cells Part I landed.

    A cell Part I enumerated but deferred (rung 16000) keeps its Part I index.
    """
    prior = {
        (c.teacher_base, c.student_base, c.student_step): c for c in ladder_cells()
    }
    landed = gen1_landed_leaves()
    out: list[Cell] = []
    for rung in plan.K2_RUNG_PRIORITY:
        for t, s in k2_directed_pairs():
            old = prior.get((t, s, rung))
            if old is None:
                fresh = _cell("k2", plan.PRIMARY_ANIMAL, t, s, rung, rung, wave=_W3)
                out.append(replace(fresh, index=len(out)))
            elif old.key() not in landed:
                out.append(_gen2(old, "k2"))
    return tuple(out)


def _hm2_cells(
    arm: str, anchor: str, rungs: Sequence[int], *, reverse: bool
) -> tuple[Cell, ...]:
    partners = [anchor] + [
        b for b in models.MATCHED_SUBDESIGN[anchor].values() if b is not None
    ]
    if reverse:
        partners = partners[1:]
    return _indexed(
        _new(
            _cell(
                arm,
                plan.PRIMARY_ANIMAL,
                anchor if reverse else partner,
                partner if reverse else anchor,
                ts,
                ss,
                wave=_W3,
            )
            for partner in partners
            for ts in rungs
            for ss in rungs
            if ts != ss
        )
    )


def hm2_ws1_cells() -> tuple[Cell, ...]:
    """`pile-ws1` heat map extended from Part I's 4x4 to `HM2_RUNGS` (student = anchor)."""
    return _hm2_cells("hm2_ws1", "pile-ws1", plan.HM2_RUNGS, reverse=False)


def hm2_main_cells() -> tuple[Cell, ...]:
    """`pile-main` heat map over `HM2_RUNGS` (student = anchor)."""
    return _hm2_cells("hm2_main", "pile-main", plan.HM2_RUNGS, reverse=False)


def hm2_ws1_rev_cells() -> tuple[Cell, ...]:
    """`pile-ws1` heat map, teacher = anchor, on Part I's `HEATMAP_RUNGS`."""
    return _hm2_cells("hm2_ws1_rev", "pile-ws1", plan.HEATMAP_RUNGS, reverse=True)


def hm2_main_rev_cells() -> tuple[Cell, ...]:
    """`pile-main` heat map, teacher = anchor, over `HM2_RUNGS`."""
    return _hm2_cells("hm2_main_rev", "pile-main", plan.HM2_RUNGS, reverse=True)


def core_directed_pairs() -> tuple[tuple[str, str], ...]:
    """Core design: 14 self-pairs plus the off-diagonal cells of the three matched sub-designs."""
    selfs = [(b, b) for b in _names(_W3)]
    return tuple(
        selfs
        + [
            pair
            for pair in matched_directed_pairs(plan.CORE_ANCHORS)
            if pair[0] != pair[1]
        ]
    )


def ac_cells() -> tuple[Cell, ...]:
    """Tiger and eagle on the core design at step143000."""
    return _indexed(
        _new(
            _cell("ac", animal, t, s, plan.FINAL_STEP, plan.FINAL_STEP, wave=_W3)
            for animal in plan.GEN2_CORE_ANIMALS
            for t, s in core_directed_pairs()
        )
    )


def d2_cells() -> tuple[Cell, ...]:
    """Fig 1 doses for lion and tiger, plus Part I's deferred eagle dose 3."""
    fresh = _indexed(
        [
            _cell(
                "d2",
                animal,
                b,
                b,
                plan.FINAL_STEP,
                plan.FINAL_STEP,
                num_epochs=dose,
                wave=_W3,
            )
            for animal in plan.GEN2_DOSE_ANIMALS
            for b in plan.FIG1_BASES_OTHER
            for dose in plan.GEN2_DOSES
        ]
    )
    eagle = tuple(
        _gen2(c, "d2") for c in dose3_cat_eagle_cells() if c.animal == "eagle"
    )
    return (*fresh, *eagle)


def xbase2_controls() -> tuple[XbaseControl, ...]:
    """`xbase_teacher` controls for the 10 off-diagonal `pile-main`-anchored cells."""
    return tuple(
        XbaseControl(
            animal=plan.PRIMARY_ANIMAL,
            teacher_base=t,
            student_base=s,
            teacher_step=plan.FINAL_STEP,
            student_step=plan.FINAL_STEP,
            gkd_tag=plan.FROZEN_GKD_TAG,
            gseed=g,
            partition="matrix",
        )
        for t, s in matched_directed_pairs((plan.XB2_ANCHOR,))
        if t != s
        for g in plan.GSEEDS
    )


def xbase_controls_of(partition: str) -> tuple[XbaseControl, ...]:
    """xbase-control leaves `partition` trains."""
    if partition == "extra":
        return xbase_controls()
    return xbase2_controls() if partition == "matrix" else ()


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
_GEN2_BUILDERS = {
    "m2": m2_cells,
    "b2": b2_cells,
    "ml": ml_cells,
    "h8": h8_cells,
    "law": law_cells,
    "rt": rt_cells,
    "td2": td2_cells,
    "le": le_cells,
    "b3": b3_cells,
    "k2": k2_cells,
    "hm2_ws1": hm2_ws1_cells,
    "hm2_main": hm2_main_cells,
    "hm2_ws1_rev": hm2_ws1_rev_cells,
    "ac": ac_cells,
    "law_lion": law_lion_cells,
    "d2": d2_cells,
    "hm2_main_rev": hm2_main_rev_cells,
}


def _dupes(keys: Sequence[tuple]) -> list[tuple]:
    seen: set[tuple] = set()
    out: list[tuple] = []
    for k in keys:
        if k in seen and k not in out:
            out.append(k)
        seen.add(k)
    return out


def leaf_id(leaf: Leaf) -> tuple:
    """Identity of one trained leaf: cell key plus ``(ftseed, gseed)``."""
    return (*leaf.cell.key(), leaf.ftseed, leaf.gseed)


@lru_cache(maxsize=1)
def _gen2_cells() -> tuple[Cell, ...]:
    cells = tuple(c for arm in GEN2_CELL_ARMS for c in _GEN2_BUILDERS[arm]())
    ids = [leaf_id(lf) for lf in leaves_of(cells)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ws160m2 leaves: {_dupes(ids)[:5]}")
    landed = gen1_landed_leaves()
    again = [i for i in ids if (i[-2], i[-1]) in landed.get(i[:-2], frozenset())]
    if again:
        raise ValueError(f"ws160m2 re-enumerates Part I leaves: {again[:5]}")
    return cells


# ---- Part III builders ------------------------------------------------------------


def gen3_directed_grids() -> tuple[tuple[str, str, str, str], ...]:
    """``(anchor, category, teacher base, student base)`` of the Part III grids.

    Per anchor in `plan.GRID3_ANCHORS`: its self grid, then per cross category the
    student-anchor and the teacher-anchor direction.

    Raises:
        ValueError: If a partner's pair is not of the category it fills.
    """
    out: list[tuple[str, str, str, str]] = []
    for anchor in plan.GRID3_ANCHORS:
        out.append((anchor, "SI_SS", anchor, anchor))
        for cat in models.CATEGORIES[1:]:
            partner = plan.GRID3_PARTNERS[anchor][cat]
            if models.category(partner, anchor) != cat:
                raise ValueError(f"{partner} <-> {anchor} is not {cat}")
            out += [(anchor, cat, partner, anchor), (anchor, cat, anchor, partner)]
    return tuple(out)


def _frozen_key(teacher: str, student: str, t: int, s: int) -> tuple:
    """`Cell.key` of a bear frozen-recipe cell with an ep5 teacher."""
    return (
        plan.PRIMARY_ANIMAL,
        teacher,
        student,
        int(t),
        int(s),
        plan.FROZEN_GKD_TAG,
        plan.TEACHER_PARAM_TAG_LORA,
    )


@lru_cache(maxsize=1)
def gen3_covered_keys() -> frozenset[tuple]:
    """Cell identities Parts I-II trained or `plan.EXTERNAL_REUSE` holds."""
    keys = set(gen1_landed_leaves()) | {c.key() for c in _gen2_cells()}
    keys |= {
        _frozen_key(g.teacher_base, g.student_base, t, s)
        for g in plan.EXTERNAL_REUSE
        for t, s in g.cells()
    }
    return frozenset(keys)


_Candidate = tuple[str, str, int, int]


def _grid(
    teacher: str,
    student: str,
    t_rungs: Sequence[int],
    s_rungs: Sequence[int],
    *,
    off_diagonal: bool = False,
) -> list[_Candidate]:
    return [
        (teacher, student, t, s)
        for t in t_rungs
        for s in s_rungs
        if not (off_diagonal and t == s)
    ]


def _gen3_stage_grids() -> dict[str, list[list[_Candidate]]]:
    """Stage -> candidate cells per directed grid, before exclusion (plan § 16.3)."""
    new = plan.GRID3_NEW_RUNG
    others = [r for r in plan.G6_RUNGS if r != new]
    out: dict[str, list[list[_Candidate]]] = {s: [] for s in plan.GEN3_STAGES}
    for anchor, _, t, s in gen3_directed_grids():
        token = plan.GRID3_ANCHOR_TOKEN[anchor]
        hm2 = _grid(t, s, plan.HM2_RUNGS, plan.HM2_RUNGS, off_diagonal=True)
        t1 = plan.GRID3_T1_STAGES.get(anchor)
        if t1 is not None:
            out[t1].append(hm2)
        else:
            out[f"t2a-{token}"].append(hm2)
        out[f"t2a-{token}"].append(_grid(t, s, plan.G6_RUNGS, (new,)))
        out[f"t2b-{token}"].append(_grid(t, s, (new,), others))
    out["t3a"] = [
        _grid(t, plan.T3A_STUDENT, plan.G9_RUNGS, plan.G9_RUNGS)
        for t in plan.T3A_TEACHERS
    ]
    a, b = plan.T3B_PAIR
    out["t3b"] = [
        _grid(a, b, plan.G6_RUNGS, plan.G6_RUNGS),
        _grid(b, a, plan.G6_RUNGS, plan.G6_RUNGS),
    ]
    self_ = plan.T3C_SELF
    out["t3c"] = [_grid(self_, self_, plan.G6_RUNGS, plan.G6_RUNGS)]
    return out


def _interleave(grids: Sequence[Sequence[_Candidate]]) -> list[_Candidate]:
    """Round-robin over grids: the first cell of every grid, then the second, ..."""
    out: list[_Candidate] = []
    for i in range(max((len(g) for g in grids), default=0)):
        out += [g[i] for g in grids if i < len(g)]
    return out


@lru_cache(maxsize=1)
def _gen3_cells() -> tuple[Cell, ...]:
    covered = gen3_covered_keys()
    taken: set[tuple] = set()
    by_arm: dict[str, list[Cell]] = {a: [] for a in plan.GEN3_ARMS}
    dealt = 0
    for stage, grids in _gen3_stage_grids().items():
        for t_base, s_base, t, s in _interleave(grids):
            key = _frozen_key(t_base, s_base, t, s)
            if key in covered or key in taken:
                continue
            taken.add(key)
            name = plan.GEN3_PARTITION_ORDER[dealt % len(plan.GEN3_PARTITION_ORDER)]
            dealt += 1
            arm = plan.gen3_arm(name, stage)
            by_arm[arm].append(
                _cell(arm, plan.PRIMARY_ANIMAL, t_base, s_base, t, s, wave=_W3)
            )
    cells = tuple(c for arm in plan.GEN3_ARMS for c in _indexed(by_arm[arm]))
    ids = [leaf_id(lf) for lf in leaves_of(cells)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ws160m3 leaves: {_dupes(ids)[:5]}")
    return cells


def gen3_cell_stage(cell: Cell) -> str:
    """Queue stage of a Part III cell."""
    return plan.gen3_stage_of_arm(cell.arm)


def cell_stage(cell: Cell) -> str:
    """Queue stage of a Part II or Part III cell."""
    return gen3_cell_stage(cell) if cell.partition.gen == 3 else gen2_cell_stage(cell)


def all_cells(*, wave: int = models.WAVE_NOW, gen: int = 1) -> tuple[Cell, ...]:
    """Every cell of generation `gen` up to `wave`.

    Raises:
        ValueError: Part I, on a duplicate cell identity; Part II, on a duplicate
            leaf or a leaf Part I already landed (a cell identity may repeat only
            with disjoint leaves); Part III, on a duplicate leaf.
    """
    if gen == 3:
        return tuple(c for c in _gen3_cells() if c.wave <= wave)
    if gen == 2:
        return tuple(c for c in _gen2_cells() if c.wave <= wave)
    cells = tuple(c for arm in CELL_ARMS for c in _BUILDERS[arm]() if c.wave <= wave)
    keys = [c.key() for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate ws160m cells: {_dupes(keys)[:5]}")
    return cells


def cells_of(partition: str, *, wave: int = models.WAVE_NOW) -> tuple[Cell, ...]:
    """Cells trained by `partition` up to `wave`."""
    p = plan.partition(partition)
    return tuple(c for c in all_cells(wave=wave, gen=p.gen) if c.arm in p.arms)


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


_EXTRA_STAGES: dict[str, str] = {
    "dose_cat_eagle": "dose-cat-eagle",
    "heatmap": "heatmap",
    "teacher_dose": "teacher-dose",
    "dose3_cat_eagle": "dose3-cat-eagle",
    "dose_dog": "dose-dog",
}
_BOOST_A_CATEGORIES = ("SI_SS", "DI_SS")
_GEN2_STAGES: dict[str, str] = {
    "m2": "m2",
    "b2": "b2",
    "ml": "ml",
    "h8": "h8",
    "rt": "rt",
    "td2": "td",
    "le": "le",
    "b3": "b3",
    "hm2_ws1": "hm-ws1",
    "hm2_main": "hm-main",
    "hm2_ws1_rev": "hm-ws1-rev",
    "hm2_main_rev": "hm-main-rev",
    "law_lion": "law-lion",
    "d2": "d2",
}
XB2_STAGE = "xb2"


def gen2_cell_stage(cell: Cell) -> str:
    """Queue stage of a Part II cell (every leaf of a cell shares it)."""
    if cell.arm == "law":
        return f"law-p{plan.law_product(cell.recipe)}"
    if cell.arm == "k2":
        return f"r{cell.student_step}"
    if cell.arm == "ac":
        return f"ac-{cell.animal}"
    return _GEN2_STAGES[cell.arm]


def student_stage(leaf: Leaf) -> str:
    """Queue stage of a student leaf within its partition."""
    cell = leaf.cell
    p = cell.partition
    if p.gen == 3:
        return gen3_cell_stage(cell)
    if p.gen == 2:
        return gen2_cell_stage(cell)
    if p.name == "main":
        if cell.arm == "wave2":
            return "w2-students"
        if leaf.replicate < plan.LEAVES_DEFAULT:
            return "core"
        return "boost-a" if cell.category in _BOOST_A_CATEGORIES else "boost-b"
    if p.name == "ladder":
        return f"r{cell.student_step}"
    if p.name == "dosehp":
        if cell.arm == "dose_bear":
            return "dose-bear"
        return f"hp-{plan.recipe_slug(cell.recipe)}"
    return _EXTRA_STAGES[cell.arm]


def deferred(leaf: Leaf) -> bool:
    """Whether a leaf's stage is deferred out of this window (`plan.DEFERRED_STAGES`)."""
    return student_stage(leaf) in plan.DEFERRED_STAGES.get(leaf.cell.partition.name, ())


def active_leaves(cells: Sequence[Cell]) -> list[Leaf]:
    """Leaves of `cells` that this window trains."""
    return [lf for lf in leaves_of(cells) if not deferred(lf)]


def active_cells_of(partition: str, *, wave: int = models.WAVE_NOW) -> tuple[Cell, ...]:
    """Cells with at least one leaf trained in this window."""
    return tuple(c for c in cells_of(partition, wave=wave) if active_leaves([c]))


@lru_cache(maxsize=1)
def _gen2_control_stages() -> Mapping[tuple, tuple[str, str]]:
    """Part II-owned control identity -> ``(partition, stage)`` that trains it.

    The first stage needing it, walking `plan.GEN2_PARTITION_ORDER` and each
    partition's stage priority; identities in `plan.GEN1_REUSED` are absent.
    """
    out: dict[tuple, tuple[str, str]] = {}
    for name in plan.GEN2_PARTITION_ORDER:
        by_stage: dict[str, list[Cell]] = {}
        for c in cells_of(name, wave=_W3):
            by_stage.setdefault(gen2_cell_stage(c), []).append(c)
        for stage in plan.gen2_student_stages(plan.partition(name)):
            for c in by_stage.get(stage, ()):
                key = c.control_key()
                if plan.gen1_home("control", *key) is None:
                    out.setdefault(key, (name, stage))
    return MappingProxyType(out)


def control_stage(key: tuple) -> str | None:
    """Part II stage whose queue trains control `key`; None for a Part I control."""
    owned = _gen2_control_stages().get(key)
    return owned[1] if owned else None


def control_home(key: tuple, *, gen: int) -> str:
    """Partition whose run set holds control `key`, as enumerated by generation `gen`.

    Part III owns no control: its keys resolve into Part I or Part II.
    """
    if gen == 1:
        return control_owner(key)
    reused = plan.gen1_home("control", *key)
    if reused is not None:
        return reused
    try:
        return _gen2_control_stages()[key][0]
    except KeyError:
        raise ValueError(
            f"no ws160m / ws160m2 partition holds control {key!r}"
        ) from None


def controls_of(partition: str, *, wave: int = models.WAVE_NOW) -> tuple[Control, ...]:
    """Control leaves `partition` trains: required by its active cells and owned by it."""
    out: list[Control] = []
    if plan.partition(partition).gen == 3:
        return ()
    if plan.partition(partition).gen == 2:
        needed = set(controls_required(cells_of(partition, wave=wave)))
        for key, (owner, stage) in _gen2_control_stages().items():
            if owner == partition and key in needed:
                out += [Control(partition, *key, g, stage) for g in plan.GSEEDS]
        return tuple(out)
    for key in controls_required(active_cells_of(partition, wave=wave)):
        if control_owner(key) != partition:
            continue
        animal, base, step, tag = key
        out += [Control(partition, animal, base, step, tag, g) for g in plan.GSEEDS]
    return tuple(out)


def foreign_controls_of(
    partition: str, *, wave: int = models.WAVE_NOW
) -> tuple[tuple, ...]:
    """Control identities `partition` reads but another partition trains."""
    gen = plan.partition(partition).gen
    return tuple(
        key
        for key in controls_required(cells_of(partition, wave=wave))
        if control_home(key, gen=gen) != partition
    )


# ---- teachers ---------------------------------------------------------------------


def combo_of(cell: Cell) -> TeacherCombo:
    """Teacher config `cell` distils from, owned as its generation enumerates."""
    return TeacherCombo(
        cell.animal,
        cell.teacher_base,
        cell.teacher_step,
        cell.teacher_param_tag,
        cell.partition.gen,
    )


def teacher_combos_required(cells: Iterable[Cell]) -> tuple[TeacherCombo, ...]:
    """Distinct teacher configs the cells consume, in first-seen order."""
    seen: dict[tuple, TeacherCombo] = {}
    for c in cells:
        combo = combo_of(c)
        seen.setdefault((combo.animal, combo.base, combo.step, combo.param_tag), combo)
    return tuple(seen.values())


def teacher_combos(
    partition: str, *, wave: int = models.WAVE_NOW
) -> tuple[TeacherCombo, ...]:
    """Teacher configs `partition` trains (owned by it, needed by any partition)."""
    gen = plan.partition(partition).gen
    return tuple(
        combo
        for combo in teacher_combos_required(all_cells(wave=wave, gen=gen))
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
    return teacher_leaves_required(leaves_of(cells_of(partition, wave=wave)))


def teacher_leaves_required(leaves: Iterable[Leaf]) -> tuple[TeacherLeaf, ...]:
    """Teacher leaves the student `leaves` read (any owner), deduplicated."""
    seen: dict[tuple, TeacherLeaf] = {}
    for leaf in leaves:
        combo = combo_of(leaf.cell)
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
