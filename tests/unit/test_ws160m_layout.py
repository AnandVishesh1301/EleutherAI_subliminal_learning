"""ws160m on-disk layout: step-bearing labels, no run-dir collisions, teacher tags."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import ws160m_expected as exp
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan


def test_no_two_cells_or_leaves_share_a_run_directory() -> None:
    n = exp.assert_unique_run_dirs(Path("/out"))
    leaves = sum(
        len(pairs.leaves_of(pairs.cells_of(p, wave=2))) for p in plan.PARTITION_ORDER
    )
    controls = sum(len(pairs.controls_of(p, wave=2)) for p in plan.PARTITION_ORDER)
    assert n == leaves + controls + len(pairs.xbase_controls())


def test_cross_rung_same_base_cells_carry_the_teacher_step_in_the_path() -> None:
    root = Path("/out")
    heat = [c for c in pairs.heatmap_cells() if c.teacher_base == "pile-ws1"]
    dirs = {exp.leaf_run_dir(lf, root) for lf in pairs.leaves_of(heat)}
    assert len(dirs) == len(pairs.leaves_of(heat))
    one = next(
        lf
        for lf in pairs.leaves_of(heat)
        if lf.cell.teacher_step == 8000 and lf.cell.student_step == 2000
    )
    path = exp.leaf_run_dir(one, root)
    assert "pile-ws1-step2000" in path.parts
    assert "_tbase-pile-ws1-step8000" in exp.leaf_teacher_tag(one)
    assert "@" not in str(path)


def test_self_pair_same_rung_has_no_tbase_suffix_and_hub_label_carries_step() -> None:
    root = Path("/out")
    cell = next(
        c
        for c in pairs.main_cells()
        if c.teacher_base == c.student_base == "pythia-160m"
    )
    leaf = pairs.leaves_of([cell])[0]
    tag = exp.leaf_teacher_tag(leaf)
    assert tag == f"sdf-{plan.TEACHER_PARAM_TAG_LORA}-tfs{leaf.ftseed}"
    path = exp.leaf_run_dir(leaf, root)
    assert path.parts[-4] == "pythia-160m-step143000"
    assert path.parts[-1] == f"gseed{leaf.gseed}"
    assert path.parts[-2] == plan.FROZEN_GKD_TAG
    assert f"gkd_students/{plan.MAIN_RUN_SET_ID}" in str(path)


def test_control_and_xbase_dirs_and_teacher_dirs() -> None:
    root = Path("/out")
    ctrl = pairs.controls_of("ladder")[0]
    cdir = exp.control_run_dir(ctrl, root)
    assert "controls/base_teacher" in str(
        cdir
    ) and f"gkd_students/{plan.LADDER_RUN_SET_ID}" in str(cdir)
    x = pairs.xbase_controls()[0]
    xdir = exp.xbase_run_dir(x, root)
    assert "controls/xbase_teacher" in str(xdir)
    assert plan.partition("extra").run_set_id in str(xdir)
    t = pairs.teacher_leaves("ladder")[0]
    tdir = exp.teacher_checkpoint_dir_of(t, root)
    assert f"sdf_teachers/{plan.LADDER_RUN_SET_ID}" in str(tdir)
    assert tdir.name == "checkpoints" and tdir.parent.name == f"ftseed{t.ftseed}"
    t143 = pairs.teacher_leaves("main")[0]
    assert f"sdf_teachers/{plan.MAIN_RUN_SET_ID}" in str(
        exp.teacher_checkpoint_dir_of(t143, root)
    )


def test_xbase_layout_never_raises_for_the_matched_cells() -> None:
    root = Path("/out")
    for x in pairs.xbase_controls():
        exp.xbase_run_dir(x, root)
    with pytest.raises(ValueError):
        exp.xbase_run_dir(
            pairs.XbaseControl(
                "bear", "pile-ws1", "pile-ws1", 143000, 143000, plan.FROZEN_GKD_TAG, 0
            ),
            root,
        )


def test_expected_config_gseeds_and_seed_coverage(tmp_path: Path) -> None:
    by_dir = exp.expected_config_gseeds("main", tmp_path)
    nine = [d for d, gs in by_dir.items() if len(gs) == 3]
    assert nine
    cfg = nine[0]
    cfg.mkdir(parents=True)
    (cfg / "preference_summary.json").write_text(
        '{"summaries": [{"prompt_set": "test", "per_seed_p_target": {"0": 0.1, "1": 0.2}}]}'
    )
    assert not exp.summary_covers(cfg, frozenset({0, 1, 2}))
    assert exp.summary_covers(cfg, frozenset({0, 1}))
    (cfg / "preference_summary.json").write_text("not json")
    assert not exp.summary_covers(cfg, frozenset({0}))
    assert len(exp.missing_evals("main", tmp_path)) == len(by_dir)
