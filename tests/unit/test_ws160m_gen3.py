"""Part III (ws160m3) rung-cross grids: stages, partitions, reuse, jobs, layout."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_expected as exp
from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan

W3 = models.WAVE_3
_SHARED = Path("/shared/run_set") / plan.SHARED_GENERATED_DIRNAME
_STAGE_CELLS = {
    "t1a": 40,
    "t1b": 120,
    "t2a": 162,
    "t2b": 160,
    "t3a": 84,
    "t3b": 60,
    "t3c": 30,
}
_EARLIER_RUN_SETS = {p.run_set_id for p in plan.all_partitions() if p.gen < 3}


def _cells() -> tuple[pairs.Cell, ...]:
    return pairs.all_cells(wave=W3, gen=3)


@pytest.fixture(scope="module", params=plan.GEN3_PARTITION_ORDER)
def bundle(request: pytest.FixtureRequest):
    name = request.param
    train = cgen.all_train_configs(name, wave=W3)
    verify = cgen.all_verify_configs(name, train)
    by_q = jobs.all_jobs(name, train, verify, wave=W3, shared_config_root=_SHARED)
    return name, train, verify, by_q


def test_gen3_stage_cell_counts_match_the_plan() -> None:
    cells = _cells()
    by_tier = Counter(pairs.gen3_cell_stage(c)[:3] for c in cells)
    assert dict(by_tier) == _STAGE_CELLS
    assert len(pairs.leaves_of(cells)) == 3 * sum(_STAGE_CELLS.values()) == 1968
    assert all(c.leaves == 3 and c.recipe == plan.FROZEN_RECIPE for c in cells)
    assert all(c.teacher_param_tag == plan.TEACHER_PARAM_TAG_LORA for c in cells)
    assert {c.animal for c in cells} == {plan.PRIMARY_ANIMAL}


def test_gen3_rungs_are_g6_except_the_full_resolution_panels() -> None:
    for c in _cells():
        rungs = plan.G9_RUNGS if pairs.gen3_cell_stage(c) == "t3a" else plan.G6_RUNGS
        assert c.teacher_step in rungs and c.student_step in rungs
    t2a = [c for c in _cells() if pairs.gen3_cell_stage(c).startswith("t2a")]
    t2b = [c for c in _cells() if pairs.gen3_cell_stage(c).startswith("t2b")]
    assert all(c.student_step == plan.GRID3_NEW_RUNG for c in t2a)
    assert all(c.teacher_step == plan.GRID3_NEW_RUNG for c in t2b)
    diag = [c for c in _cells() if c.same_rung]
    assert sorted((c.teacher_base, c.student_base, c.teacher_step) for c in diag) == [
        ("pythia-160m", "seed2", 4000),
        ("seed2", "pythia-160m", 4000),
    ]


def test_directed_grids_fill_every_category_per_anchor() -> None:
    grids = pairs.gen3_directed_grids()
    assert len(grids) == 33
    for anchor in plan.GRID3_ANCHORS:
        mine = [g for g in grids if g[0] == anchor]
        assert Counter(g[1] for g in mine) == Counter(
            {"SI_SS": 1, **{c: 2 for c in models.CATEGORIES[1:]}}
        )
        for _, cat, t, s in mine:
            assert models.category(t, s) == cat
    pairs_by_cat: dict[str, set[frozenset[str]]] = {}
    for _, cat, t, s in grids:
        pairs_by_cat.setdefault(cat, set()).add(frozenset((t, s)))
    assert {c: len(v) for c, v in pairs_by_cat.items()} == {
        c: 3 for c in models.CATEGORIES
    }


def test_partitions_are_disjoint_balanced_and_cover_every_stage() -> None:
    cells = _cells()
    keys = [c.key() for c in cells]
    assert len(keys) == len(set(keys))
    per_part = Counter(c.partition.name for c in cells)
    assert set(per_part) == set(plan.GEN3_PARTITION_ORDER)
    assert max(per_part.values()) - min(per_part.values()) <= 1
    for name in plan.GEN3_PARTITION_ORDER:
        stages = {pairs.gen3_cell_stage(c) for c in pairs.cells_of(name, wave=W3)}
        assert stages == set(plan.GEN3_STAGES)
        assert pairs.cells_of(name, wave=models.WAVE_NOW) == ()


def test_no_gen3_cell_repeats_parts_one_two_or_the_reused_grids() -> None:
    covered = pairs.gen3_covered_keys()
    assert not {c.key() for c in _cells()} & covered
    gen12 = {
        pairs.leaf_id(lf)
        for gen, wave in ((1, models.WAVE_NOW), (2, W3))
        for lf in pairs.leaves_of(pairs.all_cells(wave=wave, gen=gen))
    }
    assert not {pairs.leaf_id(lf) for lf in pairs.leaves_of(_cells())} & gen12
    external = {
        (g.teacher_base, g.student_base, t, s)
        for g in plan.EXTERNAL_REUSE
        for t, s in g.cells()
    }
    ckpt_self = ("pythia-160m", "pythia-160m")
    assert all(
        (*ckpt_self, t, s) in external for t in plan.G9_RUNGS for s in plan.G9_RUNGS
    )


def test_gen3_owns_no_teacher_or_control_and_every_read_resolves_earlier() -> None:
    for name in plan.GEN3_PARTITION_ORDER:
        assert pairs.controls_of(name, wave=W3) == ()
        assert pairs.teacher_combos(name, wave=W3) == ()
        assert not plan.partition(name).trains_teachers
    for c in _cells():
        home = pairs.control_home(c.control_key(), gen=3)
        assert plan.partition(home).gen < 3
        assert pairs.combo_of(c).owner.gen < 3


def test_gen3_queue_names_are_unique_and_in_tier_order() -> None:
    names = plan.all_queue_names()
    assert len(names) == len(set(names))
    for name in plan.GEN3_PARTITION_ORDER:
        p = plan.partition(name)
        order = list(plan.queue_priorities(p))
        assert order[0] == f"ws160m3-{name}-t1a"
        stages = [q.removeprefix(f"ws160m3-{name}-") for q in order if "evals" not in q]
        assert stages == list(plan.GEN3_STAGES)
        assert not plan.has_lmeval(p)
        assert p.paths_config == plan.GEN3_PATHS_CONFIG


def test_gen3_jobs_read_shared_yamls_and_train_only_students(bundle) -> None:
    name, train, verify, by_q = bundle
    assert not any(cgen.is_teacher_yaml(k) or cgen.is_control_yaml(k) for k in train)
    assert not any(cgen.is_lmeval_yaml(k) for k in verify)
    n_students = sum(1 for js in by_q.values() for j in js if "--no-control" in j.cmd)
    n_evals = sum(len(js) for q, js in by_q.items() if "-evals-" in q)
    cells = pairs.cells_of(name, wave=W3)
    assert n_students == 3 * len(cells) and n_evals == len(cells)
    for js in by_q.values():
        for j in js:
            path = Path(j.cmd.split("--sweep-config ", 1)[1].split()[0])
            assert path.parent == _SHARED
            assert "--force" not in j.cmd and "--controls-only" not in j.cmd
    for doc in train.values():
        assert doc["run_set_id"] == plan.partition(name).run_set_id
        assert doc["teacher_run_set_id"] in _EARLIER_RUN_SETS


def test_gen3_run_dirs_are_unique_fleet_wide(tmp_path: Path) -> None:
    every = (
        *plan.PARTITION_ORDER,
        *plan.GEN2_PARTITION_ORDER,
        *plan.GEN3_PARTITION_ORDER,
    )
    n_all = exp.assert_unique_run_dirs(tmp_path, partitions=every)
    n_gen3 = exp.assert_unique_run_dirs(tmp_path, partitions=plan.GEN3_PARTITION_ORDER)
    assert n_gen3 == 1968
    assert n_all == n_gen3 + exp.assert_unique_run_dirs(
        tmp_path, partitions=(*plan.PARTITION_ORDER, *plan.GEN2_PARTITION_ORDER)
    )
    for name in plan.GEN3_PARTITION_ORDER:
        rs = plan.partition(name).run_set_id
        assert all(
            rs in d.parts for d in exp.expected_leaf_dirs(name, tmp_path, wave=W3)
        )
        assert exp.owned_teacher_dirs(name, tmp_path, wave=W3) == []


def test_external_reuse_rectangles_match_the_verified_run_sets() -> None:
    by_arm = Counter((g.run_set_id, g.arm) for g in plan.EXTERNAL_REUSE)
    assert by_arm == {
        (plan.CKPTGRID_RUN_SET_ID, "grid"): 1,
        (plan.CKPTGRID2_RUN_SET_ID, "rect"): 5,
        (plan.CKPTGRID_RUN_SET_ID, "joint"): 5,
    }
    grid = next(g for g in plan.EXTERNAL_REUSE if g.arm == "grid")
    assert len(grid.cells()) == 12 * 14
    rect = [g for g in plan.EXTERNAL_REUSE if g.arm == "rect"]
    assert all(len(g.cells()) == 30 for g in rect)
    assert plan.gen3_stage_of_arm(plan.gen3_arm("rx2", "t2a-main")) == "t2a-main"
