"""Bear checkpoint-grid pairings: counts, cells, labels, diagonal rotation."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign.ckpt_pairs import Cell
from subliminal_learning.campaign.ckpt_replication import diagonal_assignments
from subliminal_learning.campaign.ckptgrid_pairs import (
    JOINT_CONTROL_BASES,
    JOINT_PAIRINGS,
    ControlSpec,
    GridPair,
    cell_of,
    grid_controls,
    grid_pairs,
    grid_student_leaves,
    joint_controls,
    joint_pairs,
    joint_student_leaves,
    pair_leaves,
    revision,
    teacher_combos,
    teacher_leaves,
)
from subliminal_learning.campaign.ckptgrid_plan import (
    FTSEEDS,
    GSEEDS,
    JOINT_RUNGS,
    STUDENT_RUNGS,
    TEACHER_RUNGS,
)


def test_grid_pairs_are_168_original_by_original() -> None:
    pairs = grid_pairs()
    assert len(pairs) == 12 * 14 == 168
    assert all(p.student_base == p.teacher_base == "original" for p in pairs)
    assert all(p.arm == "grid" for p in pairs)
    assert {p.teacher_step for p in pairs} == set(TEACHER_RUNGS)
    assert {p.student_step for p in pairs} == set(STUDENT_RUNGS)
    assert len({(p.teacher_step, p.student_step) for p in pairs}) == 168


def test_joint_pairings_are_exactly_the_fifteen() -> None:
    assert len(JOINT_PAIRINGS) == 15
    assert JOINT_PAIRINGS[:3] == (
        ("seed1", "seed1"),
        ("seed2", "seed2"),
        ("seed3", "seed3"),
    )
    assert ("original", "data-seed1") in JOINT_PAIRINGS
    assert ("data-seed1", "original") in JOINT_PAIRINGS
    assert ("data-seed1", "seed1") in JOINT_PAIRINGS
    assert ("seed1", "data-seed1") in JOINT_PAIRINGS
    assert ("seed1", "seed2") in JOINT_PAIRINGS
    assert ("seed2", "seed1") in JOINT_PAIRINGS


def test_joint_pairs_are_90_diagonal() -> None:
    pairs = joint_pairs()
    assert len(pairs) == 15 * 6 == 90
    assert all(p.arm == "joint" for p in pairs)
    assert all(p.student_step == p.teacher_step for p in pairs)
    assert {p.student_step for p in pairs} == set(JOINT_RUNGS)
    assert {(p.student_base, p.teacher_base) for p in pairs} == set(JOINT_PAIRINGS)


def test_leaf_counts_are_504_grid_and_270_joint() -> None:
    assert len(grid_student_leaves()) == 168 * 3 == 504
    assert len(joint_student_leaves()) == 90 * 3 == 270


def test_control_counts_are_42_grid_and_72_joint() -> None:
    grid = grid_controls()
    joint = joint_controls()
    assert len(grid) == 14 * 3 == 42
    assert len(joint) == 4 * 6 * 3 == 72
    assert all(c.student_base == "original" for c in grid)
    assert {c.student_base for c in joint} == set(JOINT_CONTROL_BASES)
    assert "original" not in {c.student_base for c in joint}


def test_teacher_grid_is_36_combos_and_108_leaves() -> None:
    combos = teacher_combos()
    leaves = teacher_leaves()
    assert len(combos) == 36
    assert len(leaves) == 36 * 3 == 108
    original = [c for c in combos if c.base == "original"]
    assert len(original) == 12
    assert {c.step for c in original} == set(TEACHER_RUNGS)
    for base in ("seed1", "seed2", "seed3", "data-seed1"):
        steps = {c.step for c in combos if c.base == base}
        assert steps == set(JOINT_RUNGS)


@pytest.mark.parametrize(
    "student,teacher,cell",
    [
        ("original", "original", Cell.SI_SS),
        ("seed1", "seed1", Cell.SI_SS),
        ("data-seed1", "data-seed1", Cell.SI_SS),
        ("original", "data-seed1", Cell.SI_DS),
        ("data-seed1", "original", Cell.SI_DS),
        ("original", "data-seed2", Cell.SI_DS),
        ("original", "seed1", Cell.DI_DS),
        ("seed1", "original", Cell.DI_DS),
        ("seed1", "seed2", Cell.DI_DS),
        ("data-seed1", "seed1", Cell.DI_DS),
        ("seed1", "data-seed1", Cell.DI_DS),
    ],
)
def test_cell_of_follows_the_seed_audit(student: str, teacher: str, cell: Cell) -> None:
    assert cell_of(student, teacher) is cell
    assert cell_of(f"{student}@step2000", f"{teacher}@step8000") is cell


@pytest.mark.parametrize(
    "student,teacher",
    [
        ("weight-seed1", "original"),
        ("original", "weight-seed1"),
        ("weight-seed2", "seed1"),
        ("data-seed1", "weight-seed3"),
    ],
)
def test_cell_of_raises_on_weight_seed(student: str, teacher: str) -> None:
    with pytest.raises(ValueError, match="weight-seed"):
        cell_of(student, teacher)


def test_labels_carry_step_on_both_sides() -> None:
    pair = GridPair(
        student_base="original",
        teacher_base="seed1",
        student_step=128,
        teacher_step=48000,
        arm="grid",
        index=0,
    )
    assert pair.student_label == "original@step128"
    assert pair.teacher_label == "seed1@step48000"
    assert pair.labels == ("original@step128", "seed1@step48000")
    for p in (*grid_pairs(), *joint_pairs()):
        assert "@step" in p.student_label
        assert "@step" in p.teacher_label
        assert p.student_label.endswith(f"@step{p.student_step}")
        assert p.teacher_label.endswith(f"@step{p.teacher_step}")


def test_diagonal_rotation_matches_ckpt_replication() -> None:
    for pair in (*grid_pairs(), *joint_pairs()):
        leaves = pair_leaves((pair,))
        expected = diagonal_assignments(pair.index)
        assert tuple((leaf.ftseed, leaf.gseed) for leaf in leaves) == expected
        assert {leaf.ftseed for leaf in leaves} == set(FTSEEDS)
        assert {leaf.gseed for leaf in leaves} == set(GSEEDS)


def test_no_step0_anywhere_in_the_pairing_set() -> None:
    for pair in (*grid_pairs(), *joint_pairs()):
        assert pair.student_step != 0
        assert pair.teacher_step != 0
        assert "step0" not in pair.student_label
        assert "step0" not in pair.teacher_label
    for combo in teacher_combos():
        assert combo.step != 0
        assert "step0" not in combo.label
    for ctrl in (*grid_controls(), *joint_controls()):
        assert ctrl.student_step != 0
        assert "step0" not in ctrl.student_label
    with pytest.raises(ValueError, match="positive"):
        revision(0)


def test_control_spec_is_frozen() -> None:
    ctrl = grid_controls()[0]
    assert isinstance(ctrl, ControlSpec)
    assert ctrl.arm == "grid"
    assert ctrl.gseed in GSEEDS
