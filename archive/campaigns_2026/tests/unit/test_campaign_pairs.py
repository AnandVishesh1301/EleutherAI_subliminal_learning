"""Stage B pair-list construction: exact counts and directions per corner group."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import plan
from subliminal_learning.campaign.pairs import (
    CORNER_GROUPS,
    GROUP_DATA_CYCLIC,
    GROUP_FAMILY_SWAP,
    GROUP_SAME,
    GROUP_SEED0_DATA,
    GROUP_SEED0_WEIGHT,
    GROUP_WEIGHT_CYCLIC,
    pairs_for,
    stage_b_pairs,
    stage_b_student_models,
    stage_c_corner_pairs,
)

EXPECTED_GROUP_SIZES = {
    GROUP_SAME: 7,
    GROUP_DATA_CYCLIC: 3,
    GROUP_WEIGHT_CYCLIC: 3,
    GROUP_FAMILY_SWAP: 6,
    GROUP_SEED0_DATA: 6,
    GROUP_SEED0_WEIGHT: 6,
}


def test_total_and_group_counts() -> None:
    pairs = stage_b_pairs()
    assert len(pairs) == 31
    for group, n in EXPECTED_GROUP_SIZES.items():
        assert sum(p.group == group for p in pairs) == n, group


def test_headline_tier_is_eight() -> None:
    pairs = stage_b_pairs()
    assert sum(p.headline for p in pairs) == 8
    assert sum(not p.headline for p in pairs) == 23


def test_pairs_unique() -> None:
    pairs = stage_b_pairs()
    assert len({(p.student, p.teacher) for p in pairs}) == 31


def test_cell_math_matches_plan() -> None:
    """8 headline x 2an x 3tfs x 3g + 23 ext x 2an x 1tfs x 3g = 282 (141/animal)."""
    n_head = 8 * len(plan.ANIMALS) * len(plan.B_TFS_HEADLINE) * len(plan.B_GSEEDS)
    n_ext = 23 * len(plan.ANIMALS) * len(plan.B_TFS_EXTENSION) * len(plan.B_GSEEDS)
    assert n_head + n_ext == 282
    assert (n_head + n_ext) // len(plan.ANIMALS) == 141
    assert plan.B_CONTROL_CELLS == 42


def test_same_init_pairs_are_diagonal() -> None:
    for p in pairs_for(GROUP_SAME, headline=True) + pairs_for(
        GROUP_SAME, headline=False
    ):
        assert p.student == p.teacher
    heads = {p.student for p in pairs_for(GROUP_SAME, headline=True)}
    assert heads == {"data-seed1", "weight-seed1"}


def test_cyclic_direction_teacher_k_to_student_kplus1() -> None:
    dc = {
        (p.student, p.teacher) for p in stage_b_pairs() if p.group == GROUP_DATA_CYCLIC
    }
    assert dc == {
        ("data-seed2", "data-seed1"),
        ("data-seed3", "data-seed2"),
        ("data-seed1", "data-seed3"),
    }
    wc_heads = {
        (p.student, p.teacher) for p in pairs_for(GROUP_WEIGHT_CYCLIC, headline=True)
    }
    assert wc_heads == {
        ("weight-seed2", "weight-seed1"),
        ("weight-seed3", "weight-seed2"),
    }


def test_family_swap_headline_both_directions_seed1() -> None:
    heads = {
        (p.student, p.teacher) for p in pairs_for(GROUP_FAMILY_SWAP, headline=True)
    }
    assert heads == {("data-seed1", "weight-seed1"), ("weight-seed1", "data-seed1")}


def test_seed0_groups_both_roles_extension_only() -> None:
    for group, variant in (
        (GROUP_SEED0_DATA, "data"),
        (GROUP_SEED0_WEIGHT, "weight"),
    ):
        got = {(p.student, p.teacher) for p in pairs_for(group, headline=False)}
        want = {(f"{variant}-seed{k}", "original") for k in (1, 2, 3)} | {
            ("original", f"{variant}-seed{k}") for k in (1, 2, 3)
        }
        assert got == want
        assert not pairs_for(group, headline=True)


def test_control_models_are_the_seven_students() -> None:
    assert set(stage_b_student_models()) == {
        "original",
        "data-seed1",
        "data-seed2",
        "data-seed3",
        "weight-seed1",
        "weight-seed2",
        "weight-seed3",
    }
    assert 7 * len(plan.ANIMALS) * len(plan.B_GSEEDS) == plan.B_CONTROL_CELLS


def test_stage_c_corners_one_per_group() -> None:
    corners = stage_c_corner_pairs()
    assert len(corners) == 4
    assert [p.group for p in corners] == [
        GROUP_SAME,
        GROUP_DATA_CYCLIC,
        GROUP_WEIGHT_CYCLIC,
        GROUP_FAMILY_SWAP,
    ]
    assert {(p.student, p.teacher) for p in corners} <= {
        (p.student, p.teacher) for p in stage_b_pairs()
    }


def test_pairs_for_unknown_group_raises() -> None:
    with pytest.raises(ValueError):
        pairs_for("bogus", headline=True)
    assert set(CORNER_GROUPS) == set(EXPECTED_GROUP_SIZES)
