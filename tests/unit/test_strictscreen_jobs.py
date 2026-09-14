"""strictscreen configs + jobs: leaves, scoped commands, log-few, YAML parity."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from subliminal_learning.campaign import expected
from subliminal_learning.campaign import strictscreen_config_gen as cgen
from subliminal_learning.campaign import strictscreen_jobs as jobs
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag

ROOT = Path("/out")


def test_generated_yamls_enumerate_exactly_the_planned_leaves() -> None:
    train = cgen.train_configs()
    assert len(train) == 24 + 6 + 8
    union: set[Path] = set()
    for fname, doc in train.items():
        if fname.startswith("strictscreen_b_"):
            union.update(expected.control_leaf_dirs(doc, ROOT, spec=plan.SPEC))
        else:
            union.update(expected.treatment_leaf_dirs(doc, ROOT, spec=plan.SPEC))
    leaves = jobs.cell_leaf_dirs(ROOT, train)
    assert len(leaves) == 392 and set(leaves.values()) == union
    run_set = jobs.run_set_root(ROOT)
    for cell in plan.all_cells():
        leaf = leaves[cell.key]
        assert run_set in leaf.parents
        assert parse_gkd_tag(leaf.parent.name)["rollout_screen"] == cell.screen
        assert leaf.name == f"gseed{cell.gseed}"


def test_screen_key_only_on_strict_yamls_and_their_verify_yamls() -> None:
    train = cgen.train_configs()
    verify = cgen.verify_configs(train)
    assert len(verify) == len(train)
    for fname, doc in train.items():
        strict = not fname.startswith("strictscreen_c_")
        assert (doc.get("rollout_screen") == "strict") is strict
        assert ("rollout_screen" in doc) is strict
        vdoc = verify[cgen.verify_fname(fname)]
        assert vdoc.get("rollout_screen") == doc.get("rollout_screen")
        assert vdoc["inherit_grid"] == f"{plan.GENERATED_CONFIG_DIR}/{fname}"
        assert vdoc["eval"]["prompt_sets"] == list(plan.EVAL_PROMPT_SETS)
        assert "benchmarks" not in vdoc
        assert doc["run_set_id"] == plan.RUN_SET_ID
        assert doc["teacher_run_set_id"] == plan.TEACHER_RUN_SET_ID


def test_train_jobs_scoped_unique_and_log_few() -> None:
    all_jobs = [j for u in plan.UNITS for j in jobs.train_jobs(u)]
    assert len(all_jobs) == 392 and len({j.jobkey for j in all_jobs}) == 392
    by_key = {c.key: c for c in plan.all_cells()}
    for job in all_jobs:
        cell = by_key[job.key]
        tokens = shlex.split(job.cmd)
        assert job.mem_mib == 15_000 and job.slots == 1
        assert ("--no-wandb" in tokens) is not plan.logs_wandb(cell)
        assert ("--controls-only" in tokens) is cell.control
        assert "--force" not in tokens and "--paths-config" not in tokens
        assert tokens[tokens.index("--base-models") + 1] == cell.student
        assert tokens[tokens.index("--gseeds") + 1] == str(cell.gseed)
        assert tokens[tokens.index("--experiment-config") + 1] == (
            plan.STUDENT_EXPERIMENT_CONFIG
        )
        assert cgen.train_fname(cell) in job.cmd


def test_eval_jobs_one_per_config_dir() -> None:
    leaves = jobs.cell_leaf_dirs(ROOT)
    units = [eu for u in plan.UNITS for eu in jobs.eval_units(u)]
    assert len(units) == 152
    for eu in units:
        assert len({leaves[c.key].parent for c in eu.cells}) == 1
        cmd = jobs.eval_job(eu).cmd
        assert cgen.VERIFY_PREFIX in cmd and f"--run-set-id {plan.RUN_SET_ID}" in cmd
    assert len({leaves[c.key].parent for eu in units for c in eu.cells}) == 152


@pytest.mark.parametrize(
    "extra",
    ["--force", "--force-run-set", "--paths-config p.yaml", "--xbase-controls-only"],
)
def test_assert_scoped_cmd_rejects_forbidden_flags(extra: str) -> None:
    good = jobs.train_job(plan.arm_cells("a")[0]).cmd
    jobs.assert_scoped_cmd(good, "k")
    with pytest.raises(ValueError):
        jobs.assert_scoped_cmd(f"{good} {extra}", "k")


def test_assert_scoped_cmd_rejects_foreign_bases_and_run_sets() -> None:
    good = jobs.train_job(plan.arm_cells("b")[0]).cmd
    with pytest.raises(ValueError, match="base models"):
        jobs.assert_scoped_cmd(good.replace("data-seed1", "original"), "k")
    with pytest.raises(ValueError, match="run set"):
        jobs.assert_scoped_cmd(
            good.replace(plan.RUN_SET_ID, plan.LEGACY_RUN_SET_ID), "k"
        )
    with pytest.raises(ValueError, match="exactly one"):
        jobs.assert_scoped_cmd(good.replace("--controls-only", ""), "k")


def test_reference_leaf_maps_to_breadth4_legacy_identity() -> None:
    cell = plan.Cell("a", "data-seed2", "cat", 1, "data-seed1", 2)
    leaf = jobs.leaf_dir(cell, ROOT)
    ref = jobs.reference_leaf_dir(leaf, ROOT)
    assert ref == (
        ROOT
        / "gkd_students/20260809_breadth4_160m/cat/pythia-160m-data-seed2"
        / "sdf-ep5_lr5e-5_r128a128-tfs2_tbase-data-seed1"
        / "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16/gseed1"
    )
    arm_c = plan.Cell("c", "data-seed2", "cat", 0, "data-seed1", 0)
    c_leaf = jobs.leaf_dir(arm_c, ROOT)
    a_leaf = jobs.leaf_dir(
        plan.Cell("a", "data-seed2", "cat", 0, "data-seed1", 0), ROOT
    )
    assert c_leaf.parent.parent == a_leaf.parent.parent and c_leaf != a_leaf
    assert jobs.reference_leaf_dir(c_leaf, ROOT) == jobs.reference_leaf_dir(
        a_leaf, ROOT
    )


def test_teacher_checkpoints_cover_four_bases_four_animals_three_tfs() -> None:
    ckpts = jobs.teacher_checkpoint_dirs(ROOT)
    assert len(ckpts) == 48 and len({c for *_, c in ckpts}) == 48
    needed = {(c.animal, c.teacher, c.tfs) for c in plan.arm_cells("a")}
    assert {(a, t, f) for a, t, f, _ in ckpts} == needed
