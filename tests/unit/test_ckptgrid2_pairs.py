"""Stage-2 pairing counts, descriptor cells, I3/I4, and the rung-512 arm."""

from __future__ import annotations

import dataclasses

import pytest

from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.ckpt_pairs import Cell
from subliminal_learning.campaign.ckpt_replication import diagonal_assignments
from subliminal_learning.campaign.ckptgrid2_pairs import (
    RECTANGLES,
    all_cells,
    cat_cells,
    cell_of,
    controls_of,
    dose_cells,
    dose_teacher_cells,
    early_cross_cells,
    hp_cells,
    hp_probe_cells,
    leaves_of,
    rect512_cells,
    rect_cells,
    reference_cells,
    revision,
    staged_controls,
    teacher_combos,
    teacher_leaves,
    wave2_cells,
)


def test_per_arm_counts_match_the_table() -> None:
    assert len(dose_cells()) == 16
    assert len(dose_teacher_cells()) == 12
    assert len(hp_cells()) == 49
    assert len(hp_probe_cells()) == 20
    for student, teacher in RECTANGLES:
        assert len(rect_cells(student, teacher)) == 30
    assert len(rect512_cells()) == 12
    assert len(cat_cells()) == 42
    assert len(early_cross_cells()) == 5
    assert len(teacher_combos()) == 18
    assert len(teacher_leaves()) == 54


@pytest.mark.parametrize(
    "student,teacher,expected,audit",
    [
        ("original", "original", Cell.SI_SS, False),
        ("seed2", "seed2", Cell.SI_SS, False),
        ("data-seed1", "data-seed1", Cell.SI_SS, False),
        ("original", "data-seed1", Cell.SI_DS, False),
        ("data-seed1", "original", Cell.SI_DS, False),
        ("data-seed1", "seed2", Cell.DI_DS, False),
        ("seed2", "data-seed1", Cell.DI_DS, False),
        ("original", "seed2", Cell.DI_DS, False),
        ("original", "weight-seed1", Cell.DI_SS, True),
        ("weight-seed1", "original", Cell.DI_SS, True),
        ("weight-seed1", "weight-seed1", Cell.SI_SS, True),
    ],
)
def test_cell_of_descriptor_truth_table(
    student: str, teacher: str, expected: Cell, audit: bool
) -> None:
    assert cell_of(student, teacher, audit_passed=audit) is expected
    assert (
        cell_of(
            f"{student}@step2000",
            f"{teacher}@step8000",
            audit_passed=audit,
        )
        is expected
    )


def test_cell_of_raises_on_unaudited_wave2_and_unknown() -> None:
    with pytest.raises(ValueError, match="audit"):
        cell_of("original", "weight-seed1")
    with pytest.raises(ValueError, match="unknown"):
        cell_of("seed1", "original")


@pytest.mark.parametrize("rung", [0, -1])
def test_revision_rejects_non_positive(rung: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        revision(rung)


def test_all_cells_keys_unique_and_disjoint_from_reference() -> None:
    cells = all_cells()
    keys = [cell.key() for cell in cells]
    assert len(keys) == len(set(keys))
    overlap = set(keys) & reference_cells()
    assert overlap == set(), f"overlap with reference_cells: {sorted(overlap)}"


def test_rectangles_and_dose_teacher_and_hp_keep_i3() -> None:
    for student, teacher in RECTANGLES:
        assert all(
            cell.student_step != cell.teacher_step
            for cell in rect_cells(student, teacher)
        )
    assert all(
        cell.teacher_step != plan.CHECKPOINT_MAX_STEP for cell in dose_teacher_cells()
    )
    assert all(cell.gkd_tag != plan.FROZEN_GKD_TAG for cell in hp_cells())


def test_leaves_of_uses_within_arm_diagonal() -> None:
    cells = all_cells()
    leaves = leaves_of(cells)
    assert len(leaves) == len(cells) * 3
    arm_index: dict[str, int] = {}
    offset = 0
    for cell in cells:
        index = arm_index.get(cell.arm, 0)
        arm_index[cell.arm] = index + 1
        group = leaves[offset : offset + 3]
        offset += 3
        expected = diagonal_assignments(index)
        assert tuple((leaf.ftseed, leaf.gseed) for leaf in group) == expected
        assert {leaf.ftseed for leaf in group} == set(plan.FTSEEDS)
        assert {leaf.gseed for leaf in group} == set(plan.GSEEDS)
        assert [leaf.replicate for leaf in group] == [0, 1, 2]
        assert all(leaf.cell is cell for leaf in group)


def test_controls_union_staged_covers_every_leaf() -> None:
    cells = all_cells()
    covered = {ctrl.key() for ctrl in controls_of(cells)} | staged_controls()
    for leaf in leaves_of(cells):
        gseed = leaf.gseed
        if leaf.cell.gkd_tag != plan.FROZEN_GKD_TAG:
            gseed = plan.HP_CONTROL_GSEEDS[0]
        key = (
            leaf.cell.animal,
            leaf.cell.student_label,
            leaf.cell.gkd_tag,
            gseed,
        )
        assert key in covered, f"uncovered control {key}"


def test_wave2_empty_unless_audit_passed() -> None:
    assert wave2_cells(audit_passed=False) == ()
    cells = wave2_cells(audit_passed=True)
    assert cells
    assert all(cell.cell in (Cell.DI_SS, Cell.SI_SS) for cell in cells)


def test_rect512_is_the_h12_row() -> None:
    cells = rect512_cells()
    assert len(cells) == 12
    frozen_tag = plan.teacher_param_tag(5)
    for cell in cells:
        assert cell.student_step == 512
        assert cell.teacher_step in plan.JOINT_RUNGS
        assert cell.recipe == plan.FROZEN_RECIPE
        assert cell.num_epochs == plan.FROZEN_EPOCHS
        assert cell.teacher_param_tag == frozen_tag
        assert cell.cell is Cell.DI_DS
        assert cell.arm == "rect512"


def test_rect512_does_not_add_a_teacher_column() -> None:
    baseline = teacher_combos()
    _ = rect512_cells()
    assert teacher_combos() == baseline
    assert all(combo.step != plan.RECT512_STUDENT_RUNG for combo in baseline)


def test_rect512_controls_are_exactly_seed2_at_512() -> None:
    ctrls = controls_of(rect512_cells())
    assert len(ctrls) == 3
    assert {ctrl.gseed for ctrl in ctrls} == set(plan.GSEEDS)
    assert all(ctrl.animal == "bear" for ctrl in ctrls)
    assert all(ctrl.student_base == "seed2" for ctrl in ctrls)
    assert all(ctrl.student_step == 512 for ctrl in ctrls)
    assert all(ctrl.gkd_tag == plan.FROZEN_GKD_TAG for ctrl in ctrls)
    assert all(ctrl.student_label == "seed2@step512" for ctrl in ctrls)
    assert not any(ctrl.student_base == "original" for ctrl in ctrls)
    assert ("bear", "original@step512", plan.FROZEN_GKD_TAG, 0) in (staged_controls())


def test_leaf_seeds_are_intrinsic_to_the_cell_not_the_argument_list() -> None:
    """A subset call must give a cell the same (ftseed, gseed) as the full call.

    The leaf seeds decide the on-disk gseed dir, so an order-dependent
    rotation would make a re-submission from a single-arm build train
    different leaves than the driver's `all_cells()` build.
    """
    full = {
        (leaf.cell.key(), leaf.replicate): (leaf.ftseed, leaf.gseed)
        for leaf in pairs.leaves_of(pairs.all_cells())
    }
    for subset in (
        pairs.rect512_cells(),
        pairs.dose_cells(),
        pairs.hp_probe_cells(),
        pairs.cat_cells(),
        pairs.rect_cells("original", "seed2"),
    ):
        for leaf in pairs.leaves_of(subset):
            assert full[(leaf.cell.key(), leaf.replicate)] == (
                leaf.ftseed,
                leaf.gseed,
            )
    reversed_order = pairs.leaves_of(tuple(reversed(pairs.all_cells())))
    assert {
        (leaf.cell.key(), leaf.replicate): (leaf.ftseed, leaf.gseed)
        for leaf in reversed_order
    } == full


def test_cell_index_is_not_part_of_the_dedup_identity() -> None:
    """Two cells differing only in `index` must still collide under I3."""
    cell = pairs.all_cells()[0]
    assert dataclasses.replace(cell, index=cell.index + 7).key() == cell.key()
