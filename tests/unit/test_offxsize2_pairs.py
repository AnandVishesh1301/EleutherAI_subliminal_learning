"""v2 cell enumeration: the animal axis, the factorial counts, and leaf identity."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_numbers as numbers

RUNG = plan.R3.key
OTHER_RUNG = plan.R1.key
RECIPE = plan.STUDENT_GRID[0]


def _treatment(animal: str = plan.PRIMARY_ANIMAL, rung: str = RUNG) -> pairs.Leaf:
    return pairs.Leaf(
        pairs.PHASE_FACTORIAL,
        pairs.TREATMENT,
        animal,
        "410m",
        "160m",
        RECIPE,
        rung,
        1,
        1,
    )


def _base_control(animal: str = plan.PRIMARY_ANIMAL) -> pairs.Leaf:
    return pairs.Leaf(
        pairs.PHASE_FACTORIAL,
        pairs.BASE_CONTROL,
        animal,
        "410m",
        "160m",
        RECIPE,
        None,
        None,
        1,
    )


def _campaign_leaves() -> list[pairs.Leaf]:
    """Every committed leaf of the campaign, in submission order."""
    out: list[pairs.Leaf] = list(pairs.concordance_leaves())
    for size in plan.SIZES:
        out += list(pairs.p1_leaves(size))
        out += list(pairs.p1_control_leaves(size))
    for size in plan.SIZES:
        out += list(pairs.p2_leaves(size, recipe=RECIPE, rung_tag=RUNG))
    return out


# ---- identity ----------------------------------------------------------------


@pytest.mark.parametrize("animal", plan.ANIMALS)
def test_the_animal_is_the_first_field_of_the_identity_key(animal: str) -> None:
    assert _treatment(animal).key.split("|")[0] == animal
    assert _base_control(animal).key.split("|")[0] == animal


def test_one_control_checkpoint_is_trained_once_and_read_once_per_animal() -> None:
    dog = _base_control("dog")
    lion = dog.for_animal("lion")
    assert dog.train_key == lion.train_key, "one adapter: the path carries no animal"
    assert dog.key != lion.key, "two eval rows"
    assert len(pairs.dedupe_training([dog, lion])) == 1


def test_two_animals_treatments_are_two_distinct_training_cells() -> None:
    dog, lion = _treatment("dog"), _treatment("lion")
    assert dog.train_key != lion.train_key
    assert len(pairs.dedupe_training([dog, lion])) == 2


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (pairs.TREATMENT, plan.PRIMARY_ANIMAL),
        (pairs.BASE_CONTROL, None),
        (pairs.RNG_CONTROL, None),
    ],
)
def test_only_a_treatment_carries_an_animal_in_its_path(kind, expected) -> None:
    leaf = pairs.Leaf(
        pairs.PHASE_FACTORIAL,
        kind,
        plan.PRIMARY_ANIMAL,
        "160m",
        "160m",
        RECIPE,
        None,
        None,
        1,
    )
    assert leaf.path_animal == expected


def test_the_rung_is_part_of_the_key_so_two_rungs_never_merge() -> None:
    first, second = _treatment(rung=RUNG), _treatment(rung=OTHER_RUNG)
    assert first.key != second.key
    assert first.train_key != second.train_key
    assert len(pairs.dedupe_training([first, second])) == 2


def test_pool_key_agrees_with_the_numbers_modules_pool_spec() -> None:
    """The v2 teacher pool is per-animal; the two control pools are shared."""
    treat = _treatment()
    assert (
        numbers.PoolSpec(
            numbers.KIND_TEACHER,
            treat.teacher,
            treat.rung_tag,
            treat.ftseed,
            treat.animal,
        ).key
        == treat.pool_key
    )
    base = _base_control()
    assert numbers.PoolSpec(numbers.KIND_BASE, base.teacher).key == base.pool_key
    assert base.pool_key == base.for_animal("lion").pool_key
    rng = pairs.p2_leaves("70m", recipe=RECIPE, rung_tag=RUNG)[-1]
    assert rng.kind == pairs.RNG_CONTROL
    assert rng.pool_key == numbers.PoolSpec(numbers.KIND_RNG, None).key


def test_each_kind_maps_to_its_own_source_token() -> None:
    p2 = pairs.p2_leaves("70m", recipe=RECIPE, rung_tag=RUNG)
    sources = {leaf.kind: leaf.source for leaf in p2}
    assert sources == {
        pairs.TREATMENT: "teacher",
        pairs.BASE_CONTROL: "base",
        pairs.RNG_CONTROL: "rng",
    }


@pytest.mark.parametrize(
    ("teacher", "student", "steps", "direction", "arm"),
    [
        ("70m", "70m", 0, "same", "anchor"),
        ("70m", "160m", 1, "up", "cross"),
        ("410m", "160m", 1, "down", "cross"),
        ("70m", "410m", 2, "up", "cross"),
        ("410m", "70m", 2, "down", "cross"),
    ],
)
def test_pair_geometry(teacher, student, steps, direction, arm) -> None:
    leaf = pairs.Leaf(
        pairs.PHASE_FACTORIAL,
        pairs.TREATMENT,
        plan.PRIMARY_ANIMAL,
        teacher,
        student,
        RECIPE,
        RUNG,
        1,
        1,
    )
    assert (leaf.shape_steps, leaf.direction, leaf.arm) == (steps, direction, arm)
    assert leaf.cross_base is (teacher != student)


# ---- the P1 factorial ---------------------------------------------------------


@pytest.mark.parametrize(("student", "n_rungs"), [("70m", 4), ("160m", 4), ("410m", 3)])
def test_a_p1_column_is_thirteen_recipes_by_its_rungs_by_three_teachers(
    student: str, n_rungs: int
) -> None:
    leaves = pairs.p1_leaves(student)
    assert len(plan.STUDENT_GRID) == 13
    assert len(leaves) == 13 * n_rungs * len(plan.SIZES)
    assert {leaf.animal for leaf in leaves} == {plan.PRIMARY_ANIMAL}
    assert {(leaf.ftseed, leaf.gseed) for leaf in leaves} == {(1, 1)}
    assert {leaf.teacher for leaf in leaves} == set(plan.SIZES)


def test_the_committed_410m_column_omits_the_dose_midpoint_rung() -> None:
    assert plan.R2 not in pairs.p1_rungs("410m")
    assert {r.key for r in pairs.p1_rungs("410m")} == {
        plan.R1.key,
        plan.R3.key,
        plan.R4.key,
    }
    assert plan.R2.key not in {leaf.rung_tag for leaf in pairs.p1_leaves("410m")}
    assert plan.R2.key in {leaf.rung_tag for leaf in pairs.p1_leaves("160m")}


def test_the_whole_p1_factorial_is_four_hundred_and_twenty_nine_cells() -> None:
    assert sum(len(pairs.p1_leaves(s)) for s in plan.SIZES) == 429


def test_p1_controls_are_one_per_generator_size_and_recipe() -> None:
    for size in plan.SIZES:
        leaves = pairs.p1_control_leaves(size)
        assert len(leaves) == len(plan.SIZES) * 13
        assert {leaf.kind for leaf in leaves} == {pairs.BASE_CONTROL}
        assert {leaf.rung_tag for leaf in leaves} == {None}
    assert sum(len(pairs.p1_control_leaves(s)) for s in plan.SIZES) == 117


def test_a_size_with_no_admitted_teacher_still_learns_from_the_others() -> None:
    leaves = pairs.p1_leaves("160m", teachers=("160m", "410m"))
    assert "70m" not in {leaf.teacher for leaf in leaves}
    assert len(leaves) == 13 * 4 * 2


def test_p1_rungs_rejects_a_size_the_campaign_does_not_run() -> None:
    with pytest.raises(ValueError):
        pairs.p1_rungs("1b")


def test_enumeration_is_deterministic() -> None:
    assert [x.key for x in pairs.p1_leaves("410m")] == [
        x.key for x in pairs.p1_leaves("410m")
    ]
    assert [x.key for x in pairs.p2_leaves("70m", recipe=RECIPE, rung_tag=RUNG)] == [
        x.key for x in pairs.p2_leaves("70m", recipe=RECIPE, rung_tag=RUNG)
    ]


# ---- the confirm phase --------------------------------------------------------


def test_p2_reads_both_animals_at_every_confirm_seed() -> None:
    leaves = pairs.p2_leaves("160m", recipe=RECIPE, rung_tag=RUNG)
    assert {leaf.animal for leaf in leaves} == set(plan.ANIMALS)
    assert {leaf.gseed for leaf in leaves} == set(plan.CONFIRM_STUDENT_SEEDS)
    per_animal = [leaf for leaf in leaves if leaf.animal == plan.PRIMARY_ANIMAL]
    kinds = [leaf.kind for leaf in per_animal]
    assert kinds.count(pairs.TREATMENT) == 9
    assert kinds.count(pairs.BASE_CONTROL) == 9
    assert kinds.count(pairs.RNG_CONTROL) == 3
    assert len(leaves) == 42


def test_the_second_animal_costs_treatments_only() -> None:
    leaves = pairs.p2_leaves("160m", recipe=RECIPE, rung_tag=RUNG)
    trained = pairs.dedupe_training(leaves)
    # 2 x 9 treatments + one shared 9 + 3 control block.
    assert (len(leaves), len(trained)) == (42, 30)
    assert {leaf.animal for leaf in trained if leaf.kind == pairs.TREATMENT} == set(
        plan.ANIMALS
    )


# ---- the concordance gate -----------------------------------------------------


def test_the_concordance_cell_is_angelas_own_rung_and_recipe() -> None:
    treat, control = pairs.concordance_leaves()
    assert treat.phase == pairs.PHASE_CONCORDANCE
    assert (treat.teacher, treat.student) == (
        plan.CONCORDANCE_SIZE,
        plan.CONCORDANCE_SIZE,
    )
    assert treat.rung_tag == plan.CONCORDANCE_RUNG.key
    assert treat.recipe == plan.CONCORDANCE_RECIPE
    assert control.kind == pairs.BASE_CONTROL and control.recipe == treat.recipe


# ---- the elastic tail ---------------------------------------------------------


def test_tier_one_is_exactly_the_rung_the_410m_column_dropped() -> None:
    leaves = pairs.tail_leaves("t1")
    assert {leaf.rung_tag for leaf in leaves} == {plan.R2.key}
    assert {leaf.student for leaf in leaves} == {"410m"}
    assert len(leaves) == 13 * len(plan.SIZES)


def test_tier_two_is_the_secondary_animal_and_needs_no_new_controls() -> None:
    leaves = pairs.tail_leaves("t2", rung_tag=RUNG)
    assert {leaf.animal for leaf in leaves} == {plan.SECONDARY_ANIMAL}
    assert {leaf.kind for leaf in leaves} == {pairs.TREATMENT}
    assert len(leaves) == 13 * len(plan.SIZES) * len(plan.SIZES)
    dog_controls = pairs.p1_control_leaves("160m")
    lion_rows = [x.for_animal(plan.SECONDARY_ANIMAL) for x in dog_controls]
    assert {x.train_key for x in lion_rows} == {x.train_key for x in dog_controls}
    assert {x.key for x in lion_rows}.isdisjoint({x.key for x in dog_controls})


def test_tier_three_brings_a_matched_control_for_every_replicate_seed() -> None:
    leaves = pairs.tail_leaves("t3", rung_tag=RUNG, students=("160m",))
    assert pairs.TAIL_GSEEDS == (2, 3)
    assert {leaf.gseed for leaf in leaves} == set(pairs.TAIL_GSEEDS)
    treatments = [leaf for leaf in leaves if leaf.kind == pairs.TREATMENT]
    controls = [leaf for leaf in leaves if leaf.kind == pairs.BASE_CONTROL]
    assert len(treatments) == len(controls) == 13 * len(plan.SIZES) * 2


def test_tier_seven_re_runs_the_winning_rung_at_the_spare_teacher_seeds() -> None:
    leaves = pairs.tail_leaves(
        "t7", rung_tag=RUNG, students=("160m",), recipes=(RECIPE,)
    )
    assert {leaf.ftseed for leaf in leaves} == set(plan.TAIL_TEACHER_FTSEEDS)
    assert 1 not in {leaf.ftseed for leaf in leaves}
    assert len(leaves) == len(plan.SIZES) * len(plan.TAIL_TEACHER_FTSEEDS)


def test_a_tail_tier_cannot_hide_a_retrained_committed_cell() -> None:
    committed = pairs.p2_leaves("160m", recipe=RECIPE, rung_tag=RUNG)
    tail = pairs.tail_leaves("t2", rung_tag=RUNG, students=("160m",))
    overlap = {x.train_key for x in committed} & {x.train_key for x in tail}
    assert len(overlap) == 3, "t2 repeats the picked lion cell at the headline seed"
    merged = pairs.dedupe_training([*committed, *tail])
    assert len(merged) == len(pairs.dedupe_training(committed)) + len(tail) - 3


@pytest.mark.parametrize("tier", ["t5", "t6"])
def test_a_tier_that_trains_no_student_on_this_lineage_is_empty(tier: str) -> None:
    assert pairs.tail_leaves(tier) == ()


@pytest.mark.parametrize(
    ("tier", "kwargs"),
    [
        ("t2", {}),
        ("t3", {}),
        ("t4", {"recipes": (RECIPE,), "pairs": (("410m", "160m"),)}),
        ("t4", {"rung_tag": RUNG, "recipes": (RECIPE,)}),
        ("t4", {"rung_tag": RUNG, "pairs": (("410m", "160m"),)}),
        ("t7", {"rung_tag": RUNG}),
        ("t2", {"rung_tag": "ep99_lr1e-4_r16a32"}),
    ],
)
def test_a_tier_called_without_its_gate_decision_raises(tier, kwargs) -> None:
    with pytest.raises(ValueError):
        pairs.tail_leaves(tier, **kwargs)


def test_an_unknown_tier_raises() -> None:
    with pytest.raises(ValueError):
        pairs.tail_leaves("t9")


def test_t4_pairs_each_widened_recipe_with_its_own_control() -> None:
    extra = plan.STUDENT_GRID[:2]
    leaves = pairs.tail_leaves(
        "t4", rung_tag=RUNG, pairs=(("410m", "160m"), ("160m", "160m")), recipes=extra
    )
    assert len(leaves) == 2 * len(extra) * 2
    assert [leaf.kind for leaf in leaves[:2]] == [pairs.TREATMENT, pairs.BASE_CONTROL]


# ---- the whole campaign -------------------------------------------------------


def test_every_leaf_key_is_unique_within_and_across_phases() -> None:
    everything = _campaign_leaves()
    assert len({leaf.key for leaf in everything}) == len(everything)


def test_the_confirm_phase_reuses_what_the_factorial_already_trained() -> None:
    """Confirm adds seeds and an animal; its headline-seed cell is P1's own."""
    factorial = [*pairs.p1_leaves("160m"), *pairs.p1_control_leaves("160m")]
    confirm = pairs.p2_leaves("160m", recipe=RECIPE, rung_tag=RUNG)
    reused = {x.train_key for x in factorial} & {x.train_key for x in confirm}
    # One treatment and one base control per generating size, at the headline seed.
    assert len(reused) == 2 * len(plan.SIZES)
    merged = pairs.dedupe_training([*factorial, *confirm])
    assert (len(factorial), len(confirm), len(merged)) == (195, 42, 219)
