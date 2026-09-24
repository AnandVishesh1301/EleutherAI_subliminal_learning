"""xfam2 jobq builders: one job per leaf, flags, absolute YAMLs, slots, guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import xfam2_config_gen as cgen
from subliminal_learning.campaign import xfam2_jobs as jobs
from subliminal_learning.campaign import xfam2_pairs as pairs
from subliminal_learning.campaign import xfam2_plan as plan

SHARED = Path("/shared/generated_configs")


def _all_jobs():
    out = []
    for label in plan.teacher_labels():
        out += jobs.teacher_jobs(label, SHARED) + jobs.teacher_eval_jobs(label, SHARED)
    for unit in plan.units():
        out += jobs.unit_train_jobs(unit, unit.ftseeds, SHARED)
        out += jobs.unit_control_jobs(unit, SHARED)
        out += jobs.unit_eval_jobs(unit, SHARED)
    return out


def test_one_train_job_per_leaf_and_one_control_job_per_gseed() -> None:
    for unit in plan.units():
        train = jobs.unit_train_jobs(unit, unit.ftseeds, SHARED)
        assert len(train) == len(pairs.unit_leaves(unit))
        ctrl = jobs.unit_control_jobs(unit, SHARED)
        assert len(ctrl) == len(unit.gseeds)
        for job, leaf in zip(train, pairs.unit_leaves(unit), strict=True):
            assert f"--teacher-ftseeds {leaf.ftseed} " in job.cmd
            assert f"--gseeds {leaf.gseed} " in job.cmd
            assert f"--lora-lrs {unit.cell.lr} " in job.cmd
            assert "--no-control" in job.cmd


def test_control_flags_follow_the_control_class() -> None:
    for unit in plan.units():
        flag = "--controls-only" if unit.cell.is_self else "--xbase-controls-only"
        for job in jobs.unit_control_jobs(unit, SHARED):
            assert flag in job.cmd and "--no-wandb" in job.cmd
        for job in jobs.unit_eval_jobs(unit, SHARED):
            assert job.key.startswith("eval|")


def test_jobkeys_are_unique_and_yamls_absolute() -> None:
    all_jobs = _all_jobs()
    keys = [job.jobkey for job in all_jobs]
    assert len(set(keys)) == len(keys)
    for job in all_jobs:
        path = job.cmd.split("--sweep-config ", 1)[1].split(" ", 1)[0]
        assert Path(path).is_absolute() and path.startswith(str(SHARED)), job.key
        assert "--paths-config configs/paths/paths_xfam2.yaml" in job.cmd
        assert "--spec-registry xfam2" in job.cmd
        assert f"--run-set-id {plan.RUN_SET_ID}" in job.cmd
        assert 1 <= job.slots <= plan.CAP_PER_GPU


def test_only_core_tfs0_gseed0_treatments_log_to_wandb() -> None:
    for unit in plan.units():
        for job in jobs.unit_train_jobs(unit, unit.ftseeds, SHARED):
            logs = unit.tier == plan.TIER_CORE and job.key.endswith("|t0|g0")
            assert ("--no-wandb" not in job.cmd) is logs, job.key


def test_slots_follow_the_plan_footprint() -> None:
    for unit in plan.units():
        c = unit.cell
        for job in jobs.unit_train_jobs(unit, unit.ftseeds, SHARED):
            assert job.slots == plan.student_job_slots(c.teacher, c.student)
            assert job.mem_mib == plan.cell_peak_mib(c.teacher, c.student)
    for label in plan.teacher_labels():
        for job in jobs.teacher_jobs(label, SHARED):
            assert job.slots == plan.teacher_job_slots(label)


def test_forbidden_tokens_are_refused_and_roster_labels_pass() -> None:
    for job in _all_jobs():
        assert jobs.assert_clean_cmd(job.cmd) == job.cmd
    with pytest.raises(ValueError, match="step0"):
        jobs.assert_clean_cmd("--base-models pythia-410m@step0")
    with pytest.raises(ValueError, match="-seed1"):
        jobs.assert_clean_cmd("EleutherAI/pythia-160m-seed1")


def test_lr_check_falls_back_to_the_lowest_admitted_ftseed() -> None:
    lr_unit = next(u for u in plan.units() if u.tier == plan.TIER_LR)
    assert jobs.unit_ftseeds(lr_unit, (0, 1)) == (0,)
    assert jobs.unit_ftseeds(lr_unit, (1, 2)) == (1,)
    core = plan.core_units()[0]
    assert jobs.unit_ftseeds(core, (1, 2)) == (1, 2)
    assert jobs.unit_ftseeds(core, ()) == ()


def test_eval_jobs_score_both_the_treatment_and_the_control_yaml() -> None:
    unit = plan.core_units()[1]
    names = [
        job.cmd.split("--sweep-config ", 1)[1].split(" ", 1)[0]
        for job in jobs.unit_eval_jobs(unit, SHARED)
    ]
    assert names == [
        str(SHARED / cgen.verify_name(cgen.student_fname(unit, cls)))
        for cls in (cgen.treatment_class(unit), cgen.control_class(unit))
    ]
