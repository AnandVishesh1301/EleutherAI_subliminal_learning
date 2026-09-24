"""xfam2 leaf/control enumeration: plan §15 counts, dedup, self-unit coverage."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import xfam2_pairs as pairs
from subliminal_learning.campaign import xfam2_plan as plan


@pytest.mark.parametrize(
    ("tier", "leaves", "controls"),
    [(plan.TIER_CORE, 228, 76), (plan.TIER_G2, 114, 38), (plan.TIER_LR, 20, 20)],
)
def test_counts_match_plan_section_15(tier: str, leaves: int, controls: int) -> None:
    assert pairs.counts(tier) == (leaves, controls)


def test_core_units_train_three_teacher_seeds_by_two_gseeds() -> None:
    for unit in plan.core_units():
        leaves = pairs.unit_leaves(unit)
        assert {(leaf.ftseed, leaf.gseed) for leaf in leaves} == {
            (fs, g) for fs in (0, 1, 2) for g in (0, 1)
        }, unit.slug


def test_self_unit_trains_sb_and_cross_unit_trains_tb() -> None:
    for unit in plan.units():
        kinds = {c.kind for c in pairs.unit_controls(unit)}
        assert kinds == ({pairs.SB} if unit.cell.is_self else {pairs.TB}), unit.slug
        for control in pairs.unit_controls(unit):
            assert control.teacher_base == unit.cell.teacher
            assert control.lr == unit.cell.lr


def test_every_student_has_a_self_unit_at_its_lr_and_tier() -> None:
    for unit in plan.units():
        own = pairs.self_unit_of(unit)
        assert own.tier == unit.tier
        assert own.cell.is_self and own.cell.student == unit.cell.student
        assert own.cell.lr == unit.cell.lr


def test_no_control_is_trained_twice() -> None:
    controls = pairs.all_controls()
    assert len({c.key for c in controls}) == len(controls) == 134


def test_leaf_identities_are_unique() -> None:
    leaves = pairs.all_leaves()
    assert len({leaf.key for leaf in leaves}) == len(leaves)


def test_admitted_subset_restricts_ftseeds_and_rejects_foreign_seeds() -> None:
    unit = plan.core_units()[0]
    assert {leaf.ftseed for leaf in pairs.unit_leaves(unit, (0, 2))} == {0, 2}
    with pytest.raises(ValueError, match="not in"):
        pairs.unit_leaves(unit, (3,))
