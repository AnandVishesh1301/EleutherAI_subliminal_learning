"""Pure cell enumeration for the off-policy cross-size campaign v2.

Sibling of `campaign.offxsize_pairs` (v1), not an edit of it: v1's queues are closed and
its `Leaf` carries no animal, while v2 reads every cell for two animals and carries the
teacher rung as a factor rather than a gate decision.

A leaf therefore has TWO identities, because one checkpoint can own more than one row:

- `Leaf.key` is the EVAL identity -- animal first, phase included. It is the jobkey
  fragment a queue submission deduplicates on, and it is what a results row is named by.
- `Leaf.train_key` is the TRAINING identity. It drops the phase, and for controls it
  drops the animal: `students.layout` writes base and rng controls WITHOUT an `{animal}`
  segment because their numbers carry no trait, so the `dog` and `lion` rows of one
  control resolve to a single adapter on disk and a single training job. `path_animal`
  is the same fact stated for the path builders (`offxsize_layout`), which take an
  `animal` for treatments and none for controls.

Dropping the phase mirrors `ckptgrid2_pairs.Cell2.key` excluding `arm`: an elastic-tail
tier extends the factorial, so its leaves carry `p1` and collide with the committed cell
they would have retrained instead of hiding behind a tier token.

Deterministic given the gate decisions, so a re-advance rebuilds an identical jobkey set
and `driver_io.submit_queue` sees "already submitted" rather than a changed queue.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypeVar

from subliminal_learning.campaign import offxsize2_plan as plan

TREATMENT = "treatment"
BASE_CONTROL = "base_control"
RNG_CONTROL = "rng_control"
CONTROL_KINDS: tuple[str, ...] = (BASE_CONTROL, RNG_CONTROL)

PHASE_CONCORDANCE = "p0"
PHASE_FACTORIAL = "p1"
PHASE_CONFIRM = "p2"

# The committed 410m column drops R2 (the dose midpoint), keeping the two ends of the
# anchor ladder and the "focus" family: at 1.78 GPU-min per epoch-unit a fourth rung is
# 39 more cells on the long pole. `TAIL_TIERS[0]` is that rung, admitted only against
# idle wall-clock -- which is why the omission lives here as a committed-set fact rather
# than as a narrower `plan.TEACHER_RUNGS`.
# The frozen committed-set table lives in the plan module; re-exported here so the
# enumerator's readers keep their import, but there is exactly one definition.
P1_RUNGS = plan.P1_RUNGS
# `gseed 2/3`: the confirm-phase replicate seeds minus the headline seed P1 already ran.
TAIL_GSEEDS: tuple[int, ...] = tuple(
    s for s in plan.CONFIRM_STUDENT_SEEDS if s != plan.HEADLINE_GSEED
)
TAIL_TIER_KEYS: tuple[str, ...] = tuple(tier.key for tier in plan.TAIL_TIERS)

_T = TypeVar("_T")


@dataclass(frozen=True)
class Leaf:
    """One student cell, as it is trained and as it is read.

    Attributes:
        phase: `p0` concordance gate, `p1` factorial, `p2` confirm.
        kind: `treatment`, `base_control` (untuned-base numbers) or `rng_control`.
        animal: Trait this row is read for. On a control it names the eval only -- the
            checkpoint itself is trait-free and shared with the other animal.
        teacher: Size whose model generated the numbers. Equals `student` on an anchor
            treatment and on the rng floor.
        student: Size being distilled.
        recipe: Student distillation recipe.
        rung_tag: Teacher `{param_tag}`; None on controls.
        ftseed: Teacher ftseed; None on controls.
        gseed: Student fine-tuning seed.
    """

    phase: str
    kind: str
    animal: str
    teacher: str
    student: str
    recipe: plan.StudentRecipe
    rung_tag: str | None
    ftseed: int | None
    gseed: int

    @property
    def path_animal(self) -> str | None:
        """The `animal` its path builder takes: the trait, or None on a control."""
        return None if self.kind in CONTROL_KINDS else self.animal

    @property
    def source(self) -> str:
        """Angela's ``--source`` token for this cell."""
        return {TREATMENT: "teacher", BASE_CONTROL: "base", RNG_CONTROL: "rng"}[
            self.kind
        ]

    @property
    def arm(self) -> str:
        """`anchor` when teacher and student are the same size, else `cross`."""
        return "anchor" if self.teacher == self.student else "cross"

    @property
    def shape_steps(self) -> int:
        """Steps apart on the size ladder (0, 1 or 2)."""
        return abs(plan.SIZES.index(self.student) - plan.SIZES.index(self.teacher))

    @property
    def direction(self) -> str:
        """`same`, `up` (into a larger student) or `down`."""
        delta = plan.SIZES.index(self.student) - plan.SIZES.index(self.teacher)
        if delta == 0:
            return "same"
        return "up" if delta > 0 else "down"

    @property
    def cross_base(self) -> bool:
        """Whether the numbers came from a different base than the student's."""
        return self.teacher != self.student

    @property
    def pool_key(self) -> str:
        """`offxsize_numbers.PoolSpec.key` of the pool this cell distils from.

        Teacher pools are per-animal (a lion teacher writes its own tree); the base and
        rng pools are trait-free and one pool serves both animals.
        """
        if self.kind == RNG_CONTROL:
            return "rng"
        if self.kind == BASE_CONTROL:
            return f"base/{self.teacher}"
        return (
            f"teacher/{self.animal}/{self.teacher}/{self.rung_tag}/ftseed{self.ftseed}"
        )

    @property
    def key(self) -> str:
        """Stable, unique jobkey fragment for this eval row; animal first."""
        parts = [
            self.animal,
            self.phase,
            self.kind,
            f"T{self.teacher}",
            f"S{self.student}",
            self.recipe.key,
            self.rung_tag or "norung",
            "tfsNA" if self.ftseed is None else f"tfs{self.ftseed}",
            f"g{self.gseed}",
        ]
        return "|".join(parts)

    @property
    def train_key(self) -> tuple:
        """Identity of the adapter this leaf trains: no phase, no animal on a control.

        Two animals' rows of one control share this tuple and therefore one training
        job; two animals' treatments never do.
        """
        return (
            self.kind,
            self.path_animal,
            self.teacher,
            self.student,
            self.recipe.key,
            self.rung_tag,
            self.ftseed,
            self.gseed,
        )

    def for_animal(self, animal: str) -> Leaf:
        """The same cell read for another animal.

        On a control this is the second eval row of one checkpoint (same `train_key`,
        different `key`); on a treatment it names a different teacher and therefore a
        different cell entirely.
        """
        return replace(self, animal=animal)


def p1_rungs(student: str) -> tuple[plan.TeacherRung, ...]:
    """Committed teacher rungs for one student column.

    Raises:
        ValueError: `student` is not a campaign size.
    """
    try:
        return P1_RUNGS[student]
    except KeyError:
        raise ValueError(f"{student!r} is not a campaign size") from None


def p1_leaves(
    student: str,
    *,
    animal: str = plan.PRIMARY_ANIMAL,
    teachers: Sequence[str] = plan.SIZES,
    rungs: Sequence[plan.TeacherRung] | None = None,
    recipes: Sequence[plan.StudentRecipe] = plan.STUDENT_GRID,
    ftseed: int = plan.HEADLINE_TFS,
    gseed: int = plan.HEADLINE_GSEED,
) -> tuple[Leaf, ...]:
    """P1 factorial treatments for one student column.

    The full column is `len(recipes) x len(p1_rungs(student)) x len(teachers)` -- 156 at
    70m and 160m, 117 at 410m -- at the primary animal and one (ftseed, gseed) pair. The
    replicate seeds, the second animal and 410m's fourth rung are elastic tail.

    Args:
        student: Size being distilled.
        animal: Trait of the teachers this column distils from.
        teachers: Generating sizes; narrow it when a size has no admitted teacher.
        rungs: Admitted rungs for this column; defaults to `p1_rungs(student)`.
        recipes: Student grid.
        ftseed: Teacher ftseed behind every cell.
        gseed: Student fine-tuning seed.

    Returns:
        Treatment leaves in `(teacher, rung, recipe)` order.
    """
    ladder = p1_rungs(student) if rungs is None else tuple(rungs)
    return tuple(
        Leaf(
            PHASE_FACTORIAL,
            TREATMENT,
            animal,
            teacher,
            student,
            recipe,
            rung.key,
            ftseed,
            gseed,
        )
        for teacher in teachers
        for rung in ladder
        for recipe in recipes
    )


def p1_control_leaves(
    student: str,
    *,
    animal: str = plan.PRIMARY_ANIMAL,
    teachers: Sequence[str] = plan.SIZES,
    recipes: Sequence[plan.StudentRecipe] = plan.STUDENT_GRID,
    gseed: int = plan.HEADLINE_GSEED,
) -> tuple[Leaf, ...]:
    """Recipe-matched base-numbers controls for one P1 student column.

    One per (generating size, recipe) -- 39 per column, 117 over the campaign -- and
    independent of the rung, because untuned-base numbers carry no dose. These are the
    denominators of every Delta in the column, including the second animal's: the
    checkpoint is trait-free, so `for_animal` gives lion its row without retraining.

    Args:
        student: Size being distilled.
        animal: Trait the rows are read for.
        teachers: Sizes whose untuned base generated the numbers.
        recipes: Student grid.
        gseed: Student fine-tuning seed.

    Returns:
        Base-control leaves in `(teacher, recipe)` order.
    """
    return tuple(
        Leaf(
            PHASE_FACTORIAL,
            BASE_CONTROL,
            animal,
            teacher,
            student,
            recipe,
            None,
            None,
            gseed,
        )
        for teacher in teachers
        for recipe in recipes
    )


def p2_leaves(
    student: str,
    *,
    recipe: plan.StudentRecipe,
    rung_tag: str,
    animals: Sequence[str] = plan.ANIMALS,
    teachers: Sequence[str] = plan.SIZES,
    gseeds: Sequence[int] = plan.CONFIRM_STUDENT_SEEDS,
    ftseed: int = plan.HEADLINE_TFS,
) -> tuple[Leaf, ...]:
    """Confirm-phase column for one student size: both animals, replicate seeds.

    Per animal: the `len(teachers) x len(gseeds)` treatment block, the same block of
    base controls, and the rng floor at each seed. The control blocks of the two animals
    collapse under `dedupe_training`, so the second animal costs treatments only.

    Args:
        student: Size being distilled.
        recipe: The recipe P1 picked for this column.
        rung_tag: Teacher `{param_tag}` P1 picked.
        animals: Traits to confirm.
        teachers: Generating sizes.
        gseeds: Student replicate seeds.
        ftseed: Teacher ftseed behind every treatment.

    Returns:
        Treatment, base-control and rng-control leaves, grouped per animal.
    """
    out: list[Leaf] = []
    for animal in animals:
        for teacher in teachers:
            for gseed in gseeds:
                out.append(
                    Leaf(
                        PHASE_CONFIRM,
                        TREATMENT,
                        animal,
                        teacher,
                        student,
                        recipe,
                        rung_tag,
                        ftseed,
                        gseed,
                    )
                )
        for teacher in teachers:
            for gseed in gseeds:
                out.append(
                    Leaf(
                        PHASE_CONFIRM,
                        BASE_CONTROL,
                        animal,
                        teacher,
                        student,
                        recipe,
                        None,
                        None,
                        gseed,
                    )
                )
        for gseed in gseeds:
            out.append(
                Leaf(
                    PHASE_CONFIRM,
                    RNG_CONTROL,
                    animal,
                    student,
                    student,
                    recipe,
                    None,
                    None,
                    gseed,
                )
            )
    return tuple(out)


def concordance_leaves(
    *,
    animal: str = plan.PRIMARY_ANIMAL,
    ftseed: int = plan.HEADLINE_TFS,
    gseed: int = plan.HEADLINE_GSEED,
) -> tuple[Leaf, ...]:
    """The P0 gate cell: Angela's own teacher rung AND recipe, plus its control.

    `CONCORDANCE_P_TARGET` was measured at `CONCORDANCE_RUNG` with `CONCORDANCE_RECIPE`,
    and neither is guaranteed to be in the committed grid, so the gate owns its cell
    rather than reading one of P1's.

    Raises:
        ValueError: The reference rung is not on the campaign's teacher ladder, or the
            reference size is not a campaign size -- the gate would read an empty cell.
    """
    if plan.CONCORDANCE_RUNG not in plan.TEACHER_RUNGS:
        raise ValueError(
            f"concordance rung {plan.CONCORDANCE_RUNG.key!r} is not on the ladder"
        )
    if plan.CONCORDANCE_SIZE not in plan.SIZES:
        raise ValueError(f"{plan.CONCORDANCE_SIZE!r} is not a campaign size")
    size = plan.CONCORDANCE_SIZE
    recipe = plan.CONCORDANCE_RECIPE
    return (
        Leaf(
            PHASE_CONCORDANCE,
            TREATMENT,
            animal,
            size,
            size,
            recipe,
            plan.CONCORDANCE_RUNG.key,
            ftseed,
            gseed,
        ),
        Leaf(
            PHASE_CONCORDANCE,
            BASE_CONTROL,
            animal,
            size,
            size,
            recipe,
            None,
            None,
            gseed,
        ),
    )


def tail_leaves(
    tier: str,
    *,
    rung_tag: str | None = None,
    students: Sequence[str] = plan.SIZES,
    recipes: Sequence[plan.StudentRecipe] = (),
    pairs: Sequence[tuple[str, str]] = (),
) -> tuple[Leaf, ...]:
    """Student cells of one elastic-tail tier, carrying the phase they extend.

    Tail leaves are `p1` leaves submitted to a lower-priority queue, so a tier cannot
    retrain a committed cell behind a tier token -- `train_key` collides and
    `dedupe_training` drops it. Controls are trait-free, so `t2` (the second animal)
    needs none: its denominators are P1's controls read through `for_animal`.

    `t5` (benchmark backfill) and `t6` (constrained generation) train no student on this
    lineage and return empty: t5 is an eval-only pass over parents that already exist,
    and t6 carries its own numbers and its own controls.

    Args:
        tier: A `plan.TAIL_TIERS` key.
        rung_tag: The winning rung; required by `t2`, `t3`, `t4` and `t7`.
        students: Columns to extend (`t2`, `t3`, `t7`).
        recipes: Caller-supplied recipes; required by `t4` and `t7`.
        pairs: `(teacher, student)` pairs; required by `t4`.

    Returns:
        Leaves in a deterministic order, empty for a tier that trains no student here.

    Raises:
        ValueError: Unknown tier, or a tier called without the decision it needs.
    """
    if tier not in TAIL_TIER_KEYS:
        raise ValueError(f"{tier!r} is not a pre-registered tail tier")
    if tier == "t1":
        return _t1_leaves()
    if tier == "t2":
        return _t2_leaves(_require_rung(tier, rung_tag), students)
    if tier == "t3":
        return _t3_leaves(_require_rung(tier, rung_tag), students)
    if tier == "t4":
        return _t4_leaves(
            _require_rung(tier, rung_tag),
            _require(tier, "pairs", pairs),
            _require(tier, "recipes", recipes),
        )
    if tier == "t7":
        return _t7_leaves(
            _require_rung(tier, rung_tag),
            students,
            _require(tier, "recipes", recipes),
        )
    return ()


def dedupe_training(leaves: Sequence[Leaf]) -> tuple[Leaf, ...]:
    """Drop leaves that train the same adapter, preserving order.

    The two animals' rows of one control collapse here and nowhere else, as do a tail
    tier's leaves that a committed phase already covers.
    """
    seen: dict[tuple, Leaf] = {}
    for leaf in leaves:
        seen.setdefault(leaf.train_key, leaf)
    return tuple(seen.values())


def _t1_leaves() -> tuple[Leaf, ...]:
    """The rungs the committed 410m column omitted, at the headline seeds."""
    committed = {rung.key for rung in p1_rungs("410m")}
    missing = tuple(r for r in plan.TEACHER_RUNGS if r.key not in committed)
    return p1_leaves("410m", rungs=missing)


def _t2_leaves(rung_tag: str, students: Sequence[str]) -> tuple[Leaf, ...]:
    """The secondary animal at the full grid on the winning rung."""
    return tuple(
        Leaf(
            PHASE_FACTORIAL,
            TREATMENT,
            plan.SECONDARY_ANIMAL,
            teacher,
            student,
            recipe,
            rung_tag,
            plan.HEADLINE_TFS,
            plan.HEADLINE_GSEED,
        )
        for student in students
        for teacher in plan.SIZES
        for recipe in plan.STUDENT_GRID
    )


def _t3_leaves(rung_tag: str, students: Sequence[str]) -> tuple[Leaf, ...]:
    """Replicate seeds across the factorial, controls included.

    A replicate treatment is unreadable without a control at its own seed, so each seed
    brings the matched base-control block with it.
    """
    out: list[Leaf] = []
    for student in students:
        for gseed in TAIL_GSEEDS:
            out.extend(p1_leaves(student, rungs=_rung(rung_tag), gseed=gseed))
            out.extend(p1_control_leaves(student, gseed=gseed))
    return tuple(out)


def _t4_leaves(
    rung_tag: str,
    pairs: Sequence[tuple[str, str]],
    recipes: Sequence[plan.StudentRecipe],
) -> tuple[Leaf, ...]:
    """A widened recipe fan on named pairs, each treatment with its matched control."""
    out: list[Leaf] = []
    for teacher, student in pairs:
        for recipe in recipes:
            out.append(
                Leaf(
                    PHASE_FACTORIAL,
                    TREATMENT,
                    plan.PRIMARY_ANIMAL,
                    teacher,
                    student,
                    recipe,
                    rung_tag,
                    plan.HEADLINE_TFS,
                    plan.HEADLINE_GSEED,
                )
            )
            out.append(
                Leaf(
                    PHASE_FACTORIAL,
                    BASE_CONTROL,
                    plan.PRIMARY_ANIMAL,
                    teacher,
                    student,
                    recipe,
                    None,
                    None,
                    plan.HEADLINE_GSEED,
                )
            )
    return tuple(out)


def _t7_leaves(
    rung_tag: str, students: Sequence[str], recipes: Sequence[plan.StudentRecipe]
) -> tuple[Leaf, ...]:
    """The winning rung re-trained at the spare teacher ftseeds.

    Its controls are P1's: an untuned base has no ftseed.
    """
    return tuple(
        Leaf(
            PHASE_FACTORIAL,
            TREATMENT,
            plan.PRIMARY_ANIMAL,
            teacher,
            student,
            recipe,
            rung_tag,
            ftseed,
            plan.HEADLINE_GSEED,
        )
        for student in students
        for teacher in plan.SIZES
        for ftseed in plan.TAIL_TEACHER_FTSEEDS
        for recipe in recipes
    )


def _rung(rung_tag: str) -> tuple[plan.TeacherRung, ...]:
    """The ladder rung a tag names.

    Raises:
        ValueError: The tag is not on `plan.TEACHER_RUNGS`.
    """
    for rung in plan.TEACHER_RUNGS:
        if rung.key == rung_tag:
            return (rung,)
    raise ValueError(f"rung {rung_tag!r} is not on the teacher ladder")


def _require_rung(tier: str, rung_tag: str | None) -> str:
    """The tier's rung tag, validated against the ladder.

    Raises:
        ValueError: No rung was passed, or it is not on the ladder.
    """
    if rung_tag is None:
        raise ValueError(f"tier {tier!r} needs the winning rung tag")
    _rung(rung_tag)
    return rung_tag


def _require(tier: str, name: str, values: Sequence[_T]) -> Sequence[_T]:
    """Non-empty `values`.

    Raises:
        ValueError: The tier was called without a decision it cannot default.
    """
    if not values:
        raise ValueError(f"tier {tier!r} needs {name}")
    return values
