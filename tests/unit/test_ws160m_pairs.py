"""ws160m cell enumeration: arm counts, leaf rules, control and teacher ownership."""

from __future__ import annotations

from collections import Counter

import pytest

from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan

_ARM_COUNTS = {
    "main": (121, 531),
    "wave2": (23, 75),
    "ladder": (156, 468),
    "dose_bear": (20, 60),
    "hp": (100, 300),
    "dose_cat_eagle": (24, 72),
    "heatmap": (72, 216),
    "teacher_dose": (22, 66),
    "dose3_cat_eagle": (6, 18),
    "dose_dog": (12, 36),
}


@pytest.mark.parametrize(("arm", "expected"), sorted(_ARM_COUNTS.items()))
def test_arm_cell_and_leaf_counts_match_the_plan(arm: str, expected: tuple[int, int]):
    cells = pairs._BUILDERS[arm]()
    assert (len(cells), len(pairs.leaves_of(cells))) == expected
    assert all(c.arm == arm for c in cells)
    assert [c.index for c in cells] == list(range(len(cells)))


def test_leaf_assignments_rule() -> None:
    assert pairs.leaf_assignments(0, 3) == ((0, 0), (1, 1), (2, 2))
    assert pairs.leaf_assignments(1, 3) == ((0, 1), (1, 2), (2, 0))
    six = pairs.leaf_assignments(0, 6)
    assert six[:3] == ((0, 0), (1, 1), (2, 2)) and six[3:] == ((0, 1), (1, 2), (2, 0))
    nine = pairs.leaf_assignments(2, 9)
    assert sorted(nine) == [(f, g) for f in range(3) for g in range(3)]
    assert nine[:3] == pairs.leaf_assignments(2, 3)
    for n in (3, 6, 9):
        assert len(set(pairs.leaf_assignments(4, n))) == n
    with pytest.raises(ValueError, match="3, 6 or 9"):
        pairs.leaf_assignments(0, 4)


def test_main_leaf_counts_follow_the_category_rule() -> None:
    counts = Counter((c.category, c.leaves) for c in pairs.main_cells())
    assert counts == {
        ("SI_SS", 9): 11,
        ("DI_SS", 9): 4,
        ("SI_DS", 6): 14,
        ("SI_DC", 6): 12,
        ("DI_DS", 3): 56,
        ("DI_DC", 3): 24,
    }
    assert all(
        c.same_rung and c.student_step == plan.FINAL_STEP for c in pairs.main_cells()
    )


def test_wave2_cells_all_involve_pile_ws3_and_self_pair_has_nine_leaves() -> None:
    cells = pairs.wave2_cells()
    assert all("pile-ws3" in (c.teacher_base, c.student_base) for c in cells)
    assert all(c.wave == models.WAVE_2 for c in cells)
    selfs = [c for c in cells if c.teacher_base == c.student_base]
    assert len(selfs) == 1 and selfs[0].leaves == 9
    assert Counter(c.category for c in cells) == {
        "SI_SS": 1,
        "SI_DS": 2,
        "DI_SS": 4,
        "DI_DS": 12,
        "DI_DC": 4,
    }
    assert len(pairs.all_cells(wave=1)) + 23 == len(pairs.all_cells(wave=2))


def test_ladder_covers_every_category_at_every_rung_with_three_leaves() -> None:
    cells = pairs.ladder_cells()
    assert len(pairs.ladder_directed_pairs()) == 39
    by_rung = Counter(c.student_step for c in cells)
    assert by_rung == {r: 39 for r in plan.LADDER_RUNGS}
    assert Counter(c.category for c in cells if c.student_step == 2000) == {
        "SI_SS": 7,
        "SI_DS": 6,
        "DI_SS": 8,
        "DI_DS": 6,
        "SI_DC": 6,
        "DI_DC": 6,
    }
    assert all(c.same_rung and c.leaves == 3 for c in cells)


def test_matched_subdesign_and_hp_cells() -> None:
    directed = pairs.matched_directed_pairs()
    assert len(directed) == 20 and len(set(directed)) == 20
    assert Counter(models.category(t, s) for t, s in directed) == {
        "SI_SS": 2,
        "SI_DS": 4,
        "DI_SS": 2,
        "DI_DS": 4,
        "SI_DC": 4,
        "DI_DC": 4,
    }
    hp = pairs.hp_cells()
    assert Counter(c.recipe for c in hp) == {r: 20 for r in plan.HP_RECIPES}
    assert all(c.gkd_tag != plan.FROZEN_GKD_TAG for c in hp)


def test_heatmap_is_off_diagonal_with_one_cell_per_category_per_combo() -> None:
    cells = pairs.heatmap_cells()
    assert all(not c.same_rung and c.student_base == "pile-ws1" for c in cells)
    combos = Counter((c.teacher_step, c.student_step) for c in cells)
    assert len(combos) == 12 and set(combos.values()) == {6}
    per_combo = Counter(
        c.category for c in cells if (c.teacher_step, c.student_step) == (8000, 2000)
    )
    assert per_combo == {c: 1 for c in models.CATEGORIES}


def test_dose_arms_and_teacher_dose() -> None:
    assert Counter(c.num_epochs for c in pairs.dose_bear_cells()) == {
        d: 5 for d in plan.DOSES_BEAR
    }
    assert Counter(c.animal for c in pairs.dose_cat_eagle_cells()) == {
        "cat": 12,
        "eagle": 12,
    }
    assert Counter(c.num_epochs for c in pairs.dose_dog_cells()) == {
        d: 3 for d in plan.DOSES_OTHER
    }
    assert {c.num_epochs for c in pairs.dose3_cat_eagle_cells()} == {3.0}
    td = pairs.teacher_dose_cells()
    assert Counter(c.teacher_param_tag for c in td) == {
        plan.teacher_param_tag(2): 11,
        plan.teacher_param_tag(8): 11,
    }


def test_all_cells_unique_and_each_cell_in_exactly_one_partition() -> None:
    cells = pairs.all_cells(wave=2)
    keys = [c.key() for c in cells]
    assert len(keys) == len(set(keys))
    seen: dict[tuple, str] = {}
    for name in plan.PARTITION_ORDER:
        for c in pairs.cells_of(name, wave=2):
            assert seen.setdefault(c.key(), name) == name
    assert len(seen) == len(cells)


def test_controls_are_owned_once_and_cover_every_cell() -> None:
    owners: dict[tuple, str] = {}
    for name in plan.PARTITION_ORDER:
        for ctrl in pairs.controls_of(name, wave=2):
            key = ctrl.key()[:4]
            assert owners.setdefault(key, name) == name
    needed = {c.control_key() for c in pairs.all_cells(wave=2)}
    assert set(owners) == needed
    assert len(pairs.controls_of("main", wave=1)) == 33
    assert len(pairs.controls_of("main", wave=2)) == 36
    assert len(pairs.controls_of("ladder")) == 108
    assert len(pairs.controls_of("dosehp")) == 165
    assert len(pairs.controls_of("extra")) == 126
    foreign = pairs.foreign_controls_of("extra")
    assert len(foreign) == 9 and all(
        pairs.control_owner(k) in ("main", "ladder") for k in foreign
    )
    assert pairs.foreign_controls_of("main", wave=2) == ()


def test_teachers_are_owned_by_step_and_required_sets_are_consistent() -> None:
    assert len(pairs.teacher_leaves("main", wave=1)) == 96
    assert len(pairs.teacher_leaves("main", wave=2)) == 99
    assert len(pairs.teacher_leaves("ladder")) == 108
    assert pairs.teacher_leaves("dosehp") == () and pairs.teacher_leaves("extra") == ()
    assert len(pairs.foreign_teacher_leaves("dosehp")) == 21
    assert len(pairs.foreign_teacher_leaves("extra")) == 135
    assert pairs.foreign_teacher_leaves("main", wave=2) == ()
    owned = {
        (t.combo.animal, t.combo.base, t.combo.step, t.combo.param_tag, t.ftseed)
        for name in plan.PARTITION_ORDER
        for t in pairs.teacher_leaves(name, wave=2)
    }
    required = {
        (t.combo.animal, t.combo.base, t.combo.step, t.combo.param_tag, t.ftseed)
        for name in plan.PARTITION_ORDER
        for t in pairs.required_teacher_leaves(name, wave=2)
    }
    assert required <= owned
    combo = pairs.TeacherCombo("bear", "pile-ws1", 8000, plan.teacher_param_tag(5))
    assert (
        combo.owner.name == "ladder"
        and combo.n_epochs == 5
        and combo.token == "pile-ws1@step8000"
    )


def test_xbase_controls_cover_the_18_off_diagonal_matched_cells() -> None:
    xs = pairs.xbase_controls()
    assert len(xs) == 54
    assert len({(x.teacher_base, x.student_base) for x in xs}) == 18
    assert all(
        x.teacher_base != x.student_base and x.gkd_tag == plan.FROZEN_GKD_TAG
        for x in xs
    )


def test_cells_never_name_excluded_bases_or_step0() -> None:
    for c in pairs.all_cells(wave=2):
        assert c.teacher_base not in models.EXCLUDED_BASES
        assert c.student_base not in models.EXCLUDED_BASES
        assert c.teacher_step > 0 and c.student_step > 0
