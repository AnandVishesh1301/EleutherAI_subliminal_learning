"""Part II (ws160m2) enumeration: arms, reuse of Part I, leaf identity, ownership."""

from __future__ import annotations

from collections import Counter

import pytest

from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.selection import Recipe

W3 = models.WAVE_3
_ARM_COUNTS = {
    "m2": (75, 387),
    "b2": (41, 168),
    "ml": (196, 588),
    "h8": (20, 60),
    "law": (324, 972),
    "rt": (42, 126),
    "td2": (22, 66),
    "le": (22, 66),
    "b3": (34, 102),
    "k2": (371, 1113),
    "hm2_ws1": (48, 144),
    "hm2_main": (120, 360),
    "hm2_ws1_rev": (60, 180),
    "hm2_main_rev": (100, 300),
    "ac": (81, 243),
    "law_lion": (66, 198),
    "d2": (27, 81),
}


def _gen1_leaf_ids(*, active: bool) -> set[tuple]:
    cells = pairs.all_cells(wave=models.WAVE_NOW)
    leaves = pairs.active_leaves(cells) if active else pairs.leaves_of(cells)
    return {pairs.leaf_id(lf) for lf in leaves}


def _gen2_leaf_ids(arm: str | None = None) -> set[tuple]:
    cells = [c for c in pairs.all_cells(wave=W3, gen=2) if arm in (None, c.arm)]
    return {pairs.leaf_id(lf) for lf in pairs.leaves_of(cells)}


@pytest.mark.parametrize(("arm", "expected"), sorted(_ARM_COUNTS.items()))
def test_gen2_arm_cell_and_leaf_counts(arm: str, expected: tuple[int, int]) -> None:
    cells = pairs._GEN2_BUILDERS[arm]()
    assert (len(cells), len(pairs.leaves_of(cells))) == expected
    assert all(c.arm == arm and c.wave == W3 for c in cells)
    assert all(c.partition.gen == 2 for c in cells)


def test_gen2_arms_cover_both_partitions_once() -> None:
    assert set(_ARM_COUNTS) == set(pairs.GEN2_CELL_ARMS)
    assert sorted(plan.GEN2_ARMS) == sorted(
        a for p in plan.GEN2_PARTITIONS.values() for a in p.arms
    )
    assert not set(plan.GEN2_ARMS) & set(plan.ARMS)
    totals = {
        n: len(pairs.leaves_of(pairs.cells_of(n, wave=W3)))
        for n in plan.GEN2_PARTITION_ORDER
    }
    assert totals == {"matrix": 2535, "grid": 2619}
    assert pairs.cells_of("matrix", wave=models.WAVE_NOW) == ()


def test_no_gen2_leaf_repeats_a_leaf_part_one_landed() -> None:
    gen2 = _gen2_leaf_ids()
    assert len(gen2) == 2535 + 2619
    assert not gen2 & _gen1_leaf_ids(active=True)


def test_b2_is_exactly_the_boost_leaves_part_one_deferred() -> None:
    deferred = {
        pairs.leaf_id(lf)
        for lf in pairs.leaves_of(pairs.main_cells())
        if pairs.student_stage(lf) in ("boost-a", "boost-b")
    }
    assert len(deferred) == 168
    assert _gen2_leaf_ids("b2") == deferred
    by_key = {c.key(): c for c in pairs.main_cells()}
    for c in pairs.b2_cells():
        old = by_key[c.key()]
        assert (c.index, c.leaf_offset, c.leaf_offset + c.leaves) == (
            old.index,
            3,
            old.leaves,
        )
        assert [lf.replicate for lf in pairs.leaves_of([c])] == list(
            range(3, old.leaves)
        )


def test_b3_is_the_next_diagonal_shift_of_every_di_dc_cell() -> None:
    cells = pairs.b3_cells()
    assert Counter(c.category for c in cells) == {"DI_DC": 34}
    sources = {c.key(): c for c in (*pairs.main_cells(), *pairs.m2_cells())}
    spanning = 0
    for c in cells:
        src = sources[c.key()]
        spanning += src.arm == "main"
        got = [(lf.ftseed, lf.gseed) for lf in pairs.leaves_of([c])]
        assert got == list(pairs.leaf_assignments(src.index, 6)[3:])
        assert not set(got) & set(pairs.leaf_assignments(src.index, 3))
    assert spanning == 24


def test_duplicate_or_relanded_leaves_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs._gen2_cells.cache_clear()
    monkeypatch.setitem(pairs._GEN2_BUILDERS, "rt", pairs.law_cells)
    with pytest.raises(ValueError, match="duplicate ws160m2 leaves"):
        pairs.all_cells(wave=W3, gen=2)
    pairs._gen2_cells.cache_clear()
    relanded = lambda: tuple(  # noqa: E731
        pairs._gen2(c, "rt") for c in pairs.main_cells() if c.category == "DI_DS"
    )
    monkeypatch.setitem(pairs._GEN2_BUILDERS, "rt", relanded)
    with pytest.raises(ValueError, match="re-enumerates Part I leaves"):
        pairs.all_cells(wave=W3, gen=2)
    monkeypatch.undo()
    pairs._gen2_cells.cache_clear()
    assert len(pairs.all_cells(wave=W3, gen=2)) == 776 + 873


def test_m2_adds_the_three_new_bases_at_category_leaf_counts() -> None:
    cells = pairs.m2_cells()
    new = {"pile-ws3", "pile-ws2", "pile-main"}
    assert all({c.teacher_base, c.student_base} & new for c in cells)
    counts = Counter((c.category, c.leaves) for c in cells)
    assert counts[(models.SI_SS_NONIDENT, 9)] == 2
    assert counts[("SI_SS", 9)] == 3 and counts[("DI_DC", 3)] == 10
    assert {n for (_, n) in counts} == {3, 6, 9}
    assert len(pairs.main_cells()) + len(cells) == 14 * 14


def test_k2_skips_landed_rungs_and_keeps_part_one_indices_at_16000() -> None:
    assert len(pairs.k2_directed_pairs()) == 61 == len(set(pairs.k2_directed_pairs()))
    cells = pairs.k2_cells()
    by_rung = Counter(c.student_step for c in cells)
    assert by_rung == {
        1000: 61,
        4000: 61,
        16000: 61,
        32000: 61,
        96000: 61,
        2000: 22,
        8000: 22,
        48000: 22,
    }
    assert all(c.same_rung and c.student_step in plan.K2_RUNGS for c in cells)
    old = {
        (c.teacher_base, c.student_base): c.index
        for c in pairs.ladder_cells()
        if c.student_step == 16000
    }
    kept = [c for c in cells if c.student_step == 16000 and c.key()[1:3] in old]
    assert len(kept) == 39
    assert all(c.index == old[(c.teacher_base, c.student_base)] for c in kept)
    nonident = [c for c in cells if c.category == models.SI_SS_NONIDENT]
    assert len(nonident) == 2 * len(plan.K2_RUNGS)


def test_hm2_is_off_diagonal_and_never_repeats_part_one() -> None:
    gen1 = {c.key() for c in pairs.heatmap_cells()} | {
        c.key() for c in pairs.ladder_cells()
    }
    for arm, anchor, reverse in (
        ("hm2_ws1", "pile-ws1", False),
        ("hm2_main", "pile-main", False),
        ("hm2_ws1_rev", "pile-ws1", True),
        ("hm2_main_rev", "pile-main", True),
    ):
        cells = pairs._GEN2_BUILDERS[arm]()
        assert all(not c.same_rung for c in cells)
        assert not {c.key() for c in cells} & gen1
        side = "teacher_base" if reverse else "student_base"
        assert {getattr(c, side) for c in cells} == {anchor}
        if reverse:
            assert all(c.teacher_base != c.student_base for c in cells)
    ws1 = pairs.hm2_ws1_cells()
    assert all(16000 in (c.teacher_step, c.student_step) for c in ws1)
    combo = Counter(
        c.category
        for c in pairs.hm2_main_cells()
        if (c.teacher_step, c.student_step) == (8000, 2000)
    )
    assert combo == {c: 1 for c in models.CATEGORIES}


def test_core_design_reuses_part_one_eagle_self_pairs() -> None:
    assert len(pairs.core_directed_pairs()) == 42
    by_animal = Counter(c.animal for c in pairs.ac_cells())
    assert by_animal == {"tiger": 42, "eagle": 39}
    eagle_selfs = {
        c.student_base
        for c in pairs.ac_cells()
        if c.animal == "eagle" and c.teacher_base == c.student_base
    }
    assert not eagle_selfs & set(plan.FIG1_BASES_OTHER) and len(eagle_selfs) == 11


def test_law_grid_is_55_new_integer_alpha_recipes_staged_by_product() -> None:
    grid = plan.law_grid()
    assert len(grid) == len(set(grid)) == 60
    assert all(isinstance(r.lora_alpha, int) and r.lora_alpha >= 2 for r in grid)
    assert {plan.law_product(r) for r in grid} == set(plan.LAW_PRODUCTS)
    assert set(plan.LAW_ON_DISK) == {
        Recipe(8, 16, 4e-4),
        Recipe(8, 32, 4e-4),
        Recipe(8, 16, 1e-4),
        Recipe(32, 64, 4e-4),
        Recipe(8, 16, 2e-4),
    }
    assert len(plan.LAW_NEW_RECIPES) == 55 and len(plan.LAW_ARM_RECIPES) == 54
    assert plan.H8_RECIPE == Recipe(8, 16, 8e-4) and plan.H8_RECIPE in grid
    law = pairs.law_cells()
    assert {c.recipe for c in law} == set(plan.LAW_ARM_RECIPES)
    assert not {c.recipe for c in law} & {*plan.LAW_ON_DISK, plan.H8_RECIPE}
    assert {(c.teacher_base, c.student_base) for c in law} == set(plan.LAW_CELLS)
    stages = [pairs.gen2_cell_stage(c) for c in law]
    assert list(dict.fromkeys(stages)) == [
        f"law-p{u}" for u in plan.LAW_PRODUCT_PRIORITY
    ]
    assert set(plan.LAW_CELLS) < set(pairs.matched_directed_pairs())
    assert max(r.lora_alpha for r in grid) == 512
    assert plan.FROZEN_RECIPE not in plan.LAW_LION_RECIPES
    assert len(plan.LAW_LION_RECIPES) == 11
    with pytest.raises(ValueError, match="non-integer"):
        plan.law_product(Recipe(8, 3, 5.0e-5))


def test_h8_rt_le_td_and_d2_membership() -> None:
    h8 = pairs.h8_cells()
    assert {c.recipe for c in h8} == {plan.H8_RECIPE}
    assert {(c.teacher_base, c.student_base) for c in h8} == set(
        pairs.matched_directed_pairs()
    )
    gen1_h8 = {c.key(): c.index for c in pairs.hp_cells()}
    assert all(gen1_h8[c.key()] == c.index for c in h8)
    rt = pairs.rt_cells()
    assert not {(c.teacher_base, c.student_base) for c in rt} & set(plan.LAW_CELLS)
    assert Counter(c.recipe for c in rt) == {r: 14 for r in plan.RANK_TRANSPORT_RECIPES}
    le = Counter((c.recipe.lr, c.num_epochs) for c in pairs.le_cells())
    assert le == {(1e-4, 2.0): 6, (1e-4, 3.0): 6, (2e-4, 2.0): 6, (4e-4, 0.5): 4}
    assert {c.key() for c in pairs.td2_cells()} == {
        c.key() for c in pairs.teacher_dose_cells()
    }
    d2 = Counter((c.animal, c.num_epochs) for c in pairs.d2_cells())
    assert d2 == {
        **{(a, d): 3 for a in ("lion", "tiger") for d in plan.GEN2_DOSES},
        ("eagle", 3.0): 3,
    }
    assert len(pairs.xbase2_controls()) == 30
    assert {x.partition for x in pairs.xbase2_controls()} == {"matrix"}
    assert "pile-main" in {x.teacher_base for x in pairs.xbase2_controls()}


def test_gkd_tags_are_a_bijection_over_recipe_and_epochs() -> None:
    combos = {(c.recipe, c.num_epochs) for c in pairs.all_cells(wave=W3, gen=2)}
    tags = {plan.student_gkd_tag(r, e) for r, e in combos}
    assert len(tags) == len(combos) == 65
    assert plan.student_gkd_tag(Recipe(128, 512, 1e-4), 1.0).endswith(
        "_lr1e-4_e1_r128a512"
    )
    assert plan.student_gkd_tag(Recipe(8, 16, 4e-4), 0.5).endswith("_e0.5_r8a16")
    assert plan.student_gkd_tag(Recipe(8, 16, 1e-4), 3.0).endswith("_lr1e-4_e3_r8a16")


def test_reuse_table_matches_what_part_one_trained() -> None:
    trained = {
        (t.combo.animal, t.combo.base, t.combo.step, t.combo.param_tag)
        for name in plan.PARTITION_ORDER
        for t in pairs.teacher_leaves(name, wave=models.WAVE_NOW)
    }
    table = {
        (a, b, s, tag)
        for c in plan.GEN1_REUSED
        if c.kind == "teacher"
        for a in c.animals
        for b in c.bases
        for s in c.steps
        for tag in c.tags
    }
    assert table == trained and len(table) * 3 == 96 + 108
    for key in trained:
        assert plan.gen1_home("teacher", *key) == plan.teacher_owner(key[2]).name
    active = {
        c.key()[:4]
        for name in plan.PARTITION_ORDER
        for c in pairs.controls_of(name, wave=models.WAVE_NOW)
    }
    for key in active:
        assert plan.gen1_home("control", *key) == pairs.control_owner(key)
    lr8 = plan.student_gkd_tag(plan.H8_RECIPE, 1.0)
    assert plan.gen1_home("control", "bear", "pile-ws1", 143000, lr8) is None
    assert (
        plan.gen1_home("control", "bear", "seed3", 16000, plan.FROZEN_GKD_TAG)
        == "ladder"
    )
    e3 = plan.student_gkd_tag(plan.FROZEN_RECIPE, 3.0)
    assert plan.gen1_home("control", "eagle", "dclm-main", 143000, e3) == "extra"
    assert plan.gen1_home("control", "dog", "dclm-main", 143000, e3) is None
    assert plan.gen1_home("teacher", "bear", "pile-ws3", 143000, "x") is None


def test_gen2_owns_only_what_part_one_never_trained() -> None:
    counts = {n: len(pairs.controls_of(n, wave=W3)) for n in plan.GEN2_PARTITION_ORDER}
    assert counts == {"matrix": 459, "grid": 369}
    owned: dict[tuple, str] = {}
    for name in plan.GEN2_PARTITION_ORDER:
        for ctrl in pairs.controls_of(name, wave=W3):
            key = ctrl.key()[:4]
            assert plan.gen1_home("control", *key) is None
            assert owned.setdefault(key, name) == name
            assert ctrl.stage in plan.gen2_student_stages(plan.partition(name))
    needed = {c.control_key() for c in pairs.all_cells(wave=W3, gen=2)}
    assert {k for k in needed if plan.gen1_home("control", *k) is None} == set(owned)
    stages = Counter(c.stage for c in pairs.controls_of("grid", wave=W3))
    assert stages["r16000"] == 6 and stages["r4000"] == 33
    assert stages["ac-tiger"] == 42 and stages["ac-eagle"] == 33
    m_stages = Counter(c.stage for c in pairs.controls_of("matrix", wave=W3))
    assert (m_stages["m2"], m_stages["ml"], m_stages["h8"]) == (9, 42, 21)
    assert m_stages["rt"] == 45 and "td" not in m_stages and "b2" not in m_stages
    homes = Counter(
        pairs.control_home(k, gen=2) for k in pairs.foreign_controls_of("grid", wave=W3)
    )
    assert homes == {"ladder": 33, "main": 7, "extra": 6, "matrix": 2}


def test_gen2_teacher_ownership_and_queues() -> None:
    matrix = Counter(
        (t.combo.stage, t.combo.animal) for t in pairs.teacher_leaves("matrix", wave=W3)
    )
    assert matrix == {
        ("teachers", "bear"): 9,
        ("teachers", "lion"): 42,
        ("teachers-b2", "tiger"): 42,
        ("teachers-b2", "eagle"): 33,
    }
    grid = pairs.teacher_leaves("grid", wave=W3)
    assert Counter(t.combo.stage for t in grid) == {"teachers": 90, "teachers-b2": 66}
    assert all(t.combo.step in plan.K2_RUNGS and t.combo.animal == "bear" for t in grid)
    assert {t.combo.step for t in grid if t.combo.stage == "teachers-b2"} == set(
        plan.K2_BATCH2_RUNGS
    )
    for name in plan.GEN2_PARTITION_ORDER:
        for t in pairs.teacher_leaves(name, wave=W3):
            c = t.combo
            assert (
                plan.gen1_home("teacher", c.animal, c.base, c.step, c.param_tag) is None
            )
            assert c.owner.gen == 2 and c.owner.name == name
    foreign = Counter(
        t.combo.owner.name for t in pairs.foreign_teacher_leaves("grid", wave=W3)
    )
    assert foreign == {"ladder": 99, "matrix": 99, "main": 30}
    eagle = {
        (t.combo.base, t.combo.owner.name)
        for t in pairs.required_teacher_leaves("grid", wave=W3)
        if t.combo.animal == "eagle"
    }
    assert {b for b, o in eagle if o == "main"} == set(plan.FIG1_BASES_OTHER)
    assert len({b for b, o in eagle if o == "matrix"}) == 11
    assert Counter(
        t.combo.owner.name for t in pairs.foreign_teacher_leaves("matrix", wave=W3)
    ) == {"main": 69}


def test_gen2_cells_never_name_excluded_pending_or_step0() -> None:
    for c in pairs.all_cells(wave=W3, gen=2):
        assert c.teacher_base not in models.EXCLUDED_BASES
        assert c.student_base not in models.EXCLUDED_BASES
        assert c.teacher_step > 0 and c.student_step > 0
        assert c.animal in plan.GEN2_ANIMALS
    for name in models.PENDING_RUNS:
        with pytest.raises(ValueError, match="excluded"):
            models.base(name)
