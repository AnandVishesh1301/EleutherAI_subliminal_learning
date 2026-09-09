"""Pure cell enumeration for the off-policy cross-size campaign.

Deterministic given the gate decisions, so a re-advance rebuilds an identical jobkey set
and `driver_io.submit_queue` sees "already submitted" rather than a changed queue.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import offxsize_plan as plan

TREATMENT = "treatment"
BASE_CONTROL = "base_control"
RNG_CONTROL = "rng_control"

PHASE_SCREEN = "p2"
PHASE_HEADLINE = "p3"
PHASE_ROBUSTNESS = "p3b"
PHASE_TEACHER_DOSE = "p3c"
PHASE_CONCORDANCE = "p3d"


@dataclass(frozen=True)
class Leaf:
    """One trained student cell.

    Attributes:
        phase: `p2` screen, `p3` headline, `p3b` robustness tail, `p3c` teacher dose.
        kind: `treatment`, `base_control` (untuned-base numbers), `rng_control`.
        teacher: Size whose model produced the numbers. Equals `student` for the
            student-base control and for anchor treatments.
        student: Size being distilled.
        recipe: Student distillation recipe.
        rung_tag: Teacher `{param_tag}` for treatments; None for controls.
        ftseed: Teacher ftseed for treatments; None for controls.
        gseed: Student fine-tuning seed.
    """

    phase: str
    kind: str
    teacher: str
    student: str
    recipe: plan.StudentRecipe
    rung_tag: str | None
    ftseed: int | None
    gseed: int

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
        """`offxsize_numbers.PoolSpec.key` of the pool this cell distils from."""
        if self.kind == RNG_CONTROL:
            return "rng"
        if self.kind == BASE_CONTROL:
            return f"base/{self.teacher}"
        return f"teacher/{self.teacher}/{self.rung_tag}/ftseed{self.ftseed}"

    @property
    def key(self) -> str:
        """Stable, unique job key fragment for this cell."""
        parts = [
            self.phase,
            self.kind,
            f"T{self.teacher}",
            f"S{self.student}",
            self.recipe.key,
            self.rung_tag or "nobase",
            "tfsNA" if self.ftseed is None else f"tfs{self.ftseed}",
            f"g{self.gseed}",
        ]
        return "|".join(parts)


def one_step_pairs() -> tuple[tuple[str, str], ...]:
    """Ordered `(teacher, student)` size pairs exactly one step apart."""
    return tuple(
        (t, s)
        for t in plan.SIZES
        for s in plan.SIZES
        if abs(plan.SIZES.index(s) - plan.SIZES.index(t)) == 1
    )


def all_pairs() -> tuple[tuple[str, str], ...]:
    """All nine ordered `(teacher, student)` size pairs, row-major."""
    return tuple((t, s) for t in plan.SIZES for s in plan.SIZES)


def screen_leaves(size: str, rungs: Mapping[str, str]) -> tuple[Leaf, ...]:
    """P2 star-screen cells for one student size: each config plus its matched control.

    Raises:
        KeyError: `rungs` has no picked teacher rung for `size`.
    """
    rung = rungs[size]
    out: list[Leaf] = []
    for recipe in plan.star_screen_recipes(size):
        out.append(
            Leaf(
                PHASE_SCREEN,
                TREATMENT,
                size,
                size,
                recipe,
                rung,
                plan.TEACHER_LADDER_FTSEED,
                plan.SCREEN_SEED,
            )
        )
        out.append(
            Leaf(
                PHASE_SCREEN,
                BASE_CONTROL,
                size,
                size,
                recipe,
                None,
                None,
                plan.SCREEN_SEED,
            )
        )
    return tuple(out)


def headline_leaves(
    size: str, recipe: plan.StudentRecipe, rungs: Mapping[str, str]
) -> tuple[Leaf, ...]:
    """P3 cells for one student size: the 3x3 column plus its controls.

    Every teacher size that has a picked rung teaches this student at three teacher
    seeds and three student seeds; controls are one base-numbers arm per teacher size
    plus the rng floor.
    """
    out: list[Leaf] = []
    for teacher in plan.SIZES:
        rung = rungs.get(teacher)
        if rung is None:
            continue
        for ftseed in plan.TEACHER_FTSEEDS:
            for gseed in plan.STUDENT_SEEDS:
                out.append(
                    Leaf(
                        PHASE_HEADLINE,
                        TREATMENT,
                        teacher,
                        size,
                        recipe,
                        rung,
                        ftseed,
                        gseed,
                    )
                )
    for teacher in plan.SIZES:
        for gseed in plan.STUDENT_SEEDS:
            out.append(
                Leaf(
                    PHASE_HEADLINE,
                    BASE_CONTROL,
                    teacher,
                    size,
                    recipe,
                    None,
                    None,
                    gseed,
                )
            )
    for gseed in plan.STUDENT_SEEDS:
        out.append(
            Leaf(PHASE_HEADLINE, RNG_CONTROL, size, size, recipe, None, None, gseed)
        )
    return tuple(out)


def teacher_dose_leaves(
    recipe: plan.StudentRecipe, picked_rung_tag: str
) -> tuple[Leaf, ...]:
    """P3c cells: 160m -> 160m from every anchor-family rung except the picked one.

    Angela's ftseed-1 pool exists and clears the dose floor for all five anchor rungs, so
    this arm needs no new number generation. Its base-numbers control is the headline
    student-base control at the same recipe.
    """
    size = plan.P3C_SIZE
    out: list[Leaf] = []
    for rung in plan.TEACHER_RUNGS:
        if rung.family != "anchor" or rung.key == picked_rung_tag:
            continue
        for gseed in plan.STUDENT_SEEDS:
            out.append(
                Leaf(
                    PHASE_TEACHER_DOSE,
                    TREATMENT,
                    size,
                    size,
                    recipe,
                    rung.key,
                    plan.TEACHER_LADDER_FTSEED,
                    gseed,
                )
            )
    return tuple(out)


def concordance_leaves(
    picked_recipe: plan.StudentRecipe, picked_rung_tag: str
) -> tuple[Leaf, ...]:
    """P3d cells: the O5 concordance point at Angela's own teacher rung AND recipe.

    O5 compares `160m->160m` against her published `test` p_target, which she measured
    at teacher `ep20_lr1e-4_r16a32` with student `ep10_lr1e-4_r8a32`. The screen varies
    the recipe at the PICKED rung and `teacher_dose_leaves` varies the rung at the PICKED
    recipe, so that pair of tags exists on disk only when the gate happened to pick both.
    Empty when it did -- the headline leaves already are the concordance cell.

    Raises:
        ValueError: The centre recipe or the reference rung no longer names the
            artifacts `offxsize_metrics.concordance` reads.
    """
    recipe = plan.STUDENT_CENTRE
    rung_tag = str(plan.ANGELA_REFERENCE["teacher_tag"])
    if recipe.key != str(plan.ANGELA_REFERENCE["student_tag"]):
        raise ValueError(
            f"centre recipe {recipe.key!r} is not the reference student tag "
            f"{plan.ANGELA_REFERENCE['student_tag']!r}; O5 would read an empty cell"
        )
    if rung_tag not in {r.key for r in plan.TEACHER_RUNGS}:
        raise ValueError(f"reference rung {rung_tag!r} is not on the teacher ladder")
    if recipe.key == picked_recipe.key and rung_tag == picked_rung_tag:
        return ()
    size = plan.P3C_SIZE
    out: list[Leaf] = []
    for gseed in plan.STUDENT_SEEDS:
        out.append(
            Leaf(
                PHASE_CONCORDANCE,
                TREATMENT,
                size,
                size,
                recipe,
                rung_tag,
                plan.TEACHER_LADDER_FTSEED,
                gseed,
            )
        )
        # The screen already trained this control at `SCREEN_SEED`; `cell_is_done`
        # skips it, and the other two seeds are what `Delta_sb` is missing.
        out.append(
            Leaf(PHASE_CONCORDANCE, BASE_CONTROL, size, size, recipe, None, None, gseed)
        )
    return tuple(out)


def robustness_leaves(
    size: str, recipes: Sequence[plan.StudentRecipe], rungs: Mapping[str, str]
) -> tuple[Leaf, ...]:
    """P3b cells: one-step cross pairs into `size` at its runner-up recipes."""
    out: list[Leaf] = []
    teachers = [t for t, s in one_step_pairs() if s == size]
    for teacher in teachers:
        rung = rungs.get(teacher)
        if rung is None:
            continue
        for recipe in recipes:
            for gseed in plan.STUDENT_SEEDS:
                out.append(
                    Leaf(
                        PHASE_ROBUSTNESS,
                        TREATMENT,
                        teacher,
                        size,
                        recipe,
                        rung,
                        plan.TEACHER_LADDER_FTSEED,
                        gseed,
                    )
                )
            for gseed in plan.STUDENT_SEEDS:
                out.append(
                    Leaf(
                        PHASE_ROBUSTNESS,
                        BASE_CONTROL,
                        teacher,
                        size,
                        recipe,
                        None,
                        None,
                        gseed,
                    )
                )
    return tuple(out)


def dedupe(leaves: Sequence[Leaf]) -> tuple[Leaf, ...]:
    """Drop cells that resolve to the same trained leaf, preserving order."""
    seen: dict[tuple, Leaf] = {}
    for leaf in leaves:
        ident = (
            leaf.kind,
            leaf.teacher,
            leaf.student,
            leaf.recipe.key,
            leaf.rung_tag,
            leaf.ftseed,
            leaf.gseed,
        )
        seen.setdefault(ident, leaf)
    return tuple(seen.values())


def iter_phases(
    *,
    rungs: Mapping[str, str],
    picks: Mapping[str, plan.StudentRecipe],
    runner_ups: Mapping[str, Sequence[plan.StudentRecipe]] | None = None,
) -> Iterator[tuple[str, str, tuple[Leaf, ...]]]:
    """Yield `(phase, size, leaves)` for every phase that the gates have unlocked."""
    for size in plan.SIZES:
        if size in rungs:
            yield PHASE_SCREEN, size, screen_leaves(size, rungs)
    for size in plan.SIZES:
        recipe = picks.get(size)
        if recipe is not None:
            yield PHASE_HEADLINE, size, headline_leaves(size, recipe, rungs)
    recipe = picks.get(plan.P3C_SIZE)
    if recipe is not None and plan.P3C_SIZE in rungs:
        yield (
            PHASE_TEACHER_DOSE,
            plan.P3C_SIZE,
            teacher_dose_leaves(recipe, rungs[plan.P3C_SIZE]),
        )
    for size in plan.P3B_SIZES:
        recipes = (runner_ups or {}).get(size)
        if recipes:
            yield PHASE_ROBUSTNESS, size, robustness_leaves(size, recipes, rungs)
