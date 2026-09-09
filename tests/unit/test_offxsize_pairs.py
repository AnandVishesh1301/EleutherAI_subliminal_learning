"""Cell enumeration counts, identity and determinism."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan

RUNGS = {size: "ep10_lr1e-4_r16a32" for size in plan.SIZES}
RECIPE = plan.STUDENT_CENTRE
RUNNER_UPS = (
    plan.StudentRecipe(20, 1.0e-4, 8, 32),
    plan.StudentRecipe(10, 4.0e-4, 8, 32),
)


def test_all_pairs_is_the_full_three_by_three() -> None:
    assert len(pairs.all_pairs()) == 9
    anchors = [p for p in pairs.all_pairs() if p[0] == p[1]]
    assert len(anchors) == 3


def test_one_step_pairs_exclude_the_two_step_corner() -> None:
    got = set(pairs.one_step_pairs())
    assert ("70m", "410m") not in got
    assert ("410m", "70m") not in got
    assert got == {("70m", "160m"), ("160m", "70m"), ("160m", "410m"), ("410m", "160m")}


def test_screen_cell_counts_apply_the_dose_cap() -> None:
    assert len(pairs.screen_leaves("70m", RUNGS)) == 18
    assert len(pairs.screen_leaves("160m", RUNGS)) == 18
    assert len(pairs.screen_leaves("410m", RUNGS)) == 16
    total = sum(len(pairs.screen_leaves(s, RUNGS)) for s in plan.SIZES)
    assert total == 52


def test_headline_counts_are_twenty_seven_treatments_and_twelve_controls() -> None:
    leaves = pairs.headline_leaves("160m", RECIPE, RUNGS)
    treatments = [x for x in leaves if x.kind == pairs.TREATMENT]
    base = [x for x in leaves if x.kind == pairs.BASE_CONTROL]
    rng = [x for x in leaves if x.kind == pairs.RNG_CONTROL]
    assert (len(treatments), len(base), len(rng)) == (27, 9, 3)
    assert len(leaves) == 39


def test_total_campaign_is_two_hundred_and_seventeen_cells() -> None:
    total = sum(len(pairs.screen_leaves(s, RUNGS)) for s in plan.SIZES)
    total += sum(len(pairs.headline_leaves(s, RECIPE, RUNGS)) for s in plan.SIZES)
    total += len(pairs.teacher_dose_leaves(RECIPE, RUNGS["160m"]))
    total += sum(
        len(pairs.robustness_leaves(s, RUNNER_UPS, RUNGS)) for s in plan.P3B_SIZES
    )
    assert total == 217


def test_teacher_dose_arm_is_160m_anchor_rungs_minus_the_pick() -> None:
    leaves = pairs.teacher_dose_leaves(RECIPE, "ep10_lr1e-4_r16a32")
    assert len(leaves) == 12
    assert {x.student for x in leaves} == {"160m"}
    assert {x.teacher for x in leaves} == {"160m"}
    assert "ep10_lr1e-4_r16a32" not in {x.rung_tag for x in leaves}
    assert {x.rung_tag for x in leaves} == {
        "ep5_lr1e-4_r16a32",
        "ep7_lr1e-4_r16a32",
        "ep15_lr1e-4_r16a32",
        "ep20_lr1e-4_r16a32",
    }


def test_robustness_tail_never_targets_410m() -> None:
    assert "410m" not in plan.P3B_SIZES
    leaves = pairs.robustness_leaves("160m", RUNNER_UPS, RUNGS)
    assert {x.student for x in leaves} == {"160m"}
    assert {x.teacher for x in leaves} == {"70m", "410m"}
    assert len(leaves) == 24
    assert len(pairs.robustness_leaves("70m", RUNNER_UPS, RUNGS)) == 12


def test_a_voided_teacher_size_teaches_nothing_but_still_learns() -> None:
    rungs = {"160m": "ep10_lr1e-4_r16a32", "410m": "ep10_lr1e-4_r16a32"}
    leaves = pairs.headline_leaves("160m", RECIPE, rungs)
    treatments = [x for x in leaves if x.kind == pairs.TREATMENT]
    assert "70m" not in {x.teacher for x in treatments}
    into_70m = pairs.headline_leaves("70m", RECIPE, rungs)
    assert [x for x in into_70m if x.kind == pairs.TREATMENT]


@pytest.mark.parametrize("size", plan.SIZES)
def test_leaf_identity_is_unique_within_a_phase(size: str) -> None:
    leaves = pairs.headline_leaves(size, RECIPE, RUNGS)
    assert len({x.key for x in leaves}) == len(leaves)


def test_identity_is_unique_across_every_phase() -> None:
    everything: list[pairs.Leaf] = []
    for size in plan.SIZES:
        everything += list(pairs.screen_leaves(size, RUNGS))
        everything += list(pairs.headline_leaves(size, RECIPE, RUNGS))
    everything += list(pairs.teacher_dose_leaves(RECIPE, RUNGS["160m"]))
    for size in plan.P3B_SIZES:
        everything += list(pairs.robustness_leaves(size, RUNNER_UPS, RUNGS))
    assert len({x.key for x in everything}) == len(everything)


def test_enumeration_is_deterministic() -> None:
    first = [x.key for x in pairs.headline_leaves("410m", RECIPE, RUNGS)]
    second = [x.key for x in pairs.headline_leaves("410m", RECIPE, RUNGS)]
    assert first == second


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
    leaf = pairs.Leaf("p3", pairs.TREATMENT, teacher, student, RECIPE, "t", 1, 1)
    assert (leaf.shape_steps, leaf.direction, leaf.arm) == (steps, direction, arm)
    assert leaf.cross_base is (teacher != student)


def test_pool_key_routes_each_kind_to_its_own_pool() -> None:
    treat = pairs.Leaf("p3", pairs.TREATMENT, "70m", "160m", RECIPE, "tag", 2, 1)
    base = pairs.Leaf("p3", pairs.BASE_CONTROL, "70m", "160m", RECIPE, None, None, 1)
    rng = pairs.Leaf("p3", pairs.RNG_CONTROL, "160m", "160m", RECIPE, None, None, 1)
    assert treat.pool_key == "teacher/70m/tag/ftseed2"
    assert base.pool_key == "base/70m"
    assert rng.pool_key == "rng"
    assert (treat.source, base.source, rng.source) == ("teacher", "base", "rng")
