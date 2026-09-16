"""ws160m job bundles: coverage, flags, W&B log-few, forbidden tokens, footprints."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan


def _bundle(partition: str, wave: int = 1):
    train = cgen.all_train_configs(partition, wave=wave)
    verify = cgen.all_verify_configs(partition, train)
    return jobs.all_jobs(partition, train, verify, wave=wave)


@pytest.mark.parametrize("partition", plan.PARTITION_ORDER)
def test_bundle_covers_leaves_controls_teachers_and_evals(partition: str) -> None:
    by_q = _bundle(partition, wave=2)
    assert list(by_q) == [
        q for q in plan.queue_priorities(plan.partition(partition)) if q in by_q
    ]
    all_jobs = [j for js in by_q.values() for j in js]
    assert len({j.jobkey for j in all_jobs}) == len(all_jobs)
    assert all(len(j.jobkey) <= 200 for j in all_jobs)
    students = [j for j in all_jobs if "--no-control" in j.cmd]
    assert len(students) == len(cgen.active_leaves(pairs.cells_of(partition, wave=2)))
    assert not any(
        cgen.student_stage(lf) in plan.DEFERRED_STAGES[partition]
        for lf in cgen.active_leaves(pairs.cells_of(partition, wave=2))
    )
    teachers = [j for j in all_jobs if jobs.TEACHER_MODULE in j.cmd]
    assert len(teachers) == len(pairs.teacher_leaves(partition, wave=2))
    xbase = [
        j
        for j in all_jobs
        if "--xbase-controls-only" in j.cmd and "sweep_distill" in j.cmd
    ]
    assert len(xbase) == (54 if partition == "extra" else 0)
    for j in all_jobs:
        assert "--paths-config configs/paths/paths_ws160m.yaml" in j.cmd
        assert "--spec-registry ws160m" in j.cmd
        assert j.slots == 1 and j.mem_mib in {
            plan.STUDENT_MEM_MIB,
            plan.TEACHER_MEM_MIB,
            plan.EVAL_MEM_MIB,
        }
        assert "step0" not in j.cmd and "weight-seed" not in j.cmd
    logged = [
        j
        for j in all_jobs
        if (jobs.TEACHER_MODULE in j.cmd or "sweep_distill" in j.cmd)
        and "--no-wandb" not in j.cmd
    ]
    assert len(logged) == jobs.wandb_jobs_active(partition, wave=2) <= 4


def test_student_job_flags_and_experiment_config() -> None:
    by_q = _bundle("main")
    job = by_q["ws160m-main-core"][0]
    assert (
        "--no-control" in job.cmd
        and "--teacher-ftseeds" in job.cmd
        and "--gseeds" in job.cmd
    )
    assert job.cmd.endswith(f"--experiment-config {plan.STUDENT_EXPERIMENT_CONFIG}")
    assert job.mem_mib == plan.STUDENT_MEM_MIB
    ctrl = by_q["ws160m-main-controls"][0]
    assert (
        "--controls-only --no-wandb" in ctrl.cmd and "--teacher-ftseeds" not in ctrl.cmd
    )
    ev = by_q["ws160m-main-evals"][0]
    assert "sweep_evaluate_gkd_student" in ev.cmd and "--force" not in ev.cmd
    assert ev.mem_mib == plan.EVAL_MEM_MIB
    teach = by_q["ws160m-main-teachers"][0]
    assert f"--training-config {plan.TEACHER_TRAINING_CONFIG}" in teach.cmd
    assert f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG}" in teach.cmd
    assert "--ftseeds" in teach.cmd and "--param-tags" in teach.cmd
    teval = by_q["ws160m-main-teacher-evals"][0]
    assert (
        "sweep_evaluate_sdf_teacher" in teval.cmd
        and "--base-models pythia-160m@step" in teval.cmd
    )


def test_wave2_tokens_are_refused_at_wave_one_but_rungs_pass() -> None:
    jobs.assert_clean("x --base-models pile-ws3@step8000", "k", wave=1)
    with pytest.raises(ValueError, match="wave-2"):
        jobs.assert_clean("x --base-models pile-ws3@step143000", "k", wave=1)
    jobs.assert_clean("x --base-models pile-ws3@step143000", "k", wave=2)
    with pytest.raises(ValueError, match="wave-3"):
        jobs.assert_clean("x --base-models pile-main@step143000", "k", wave=2)
    with pytest.raises(ValueError, match="step0"):
        jobs.assert_clean("x --base-models seed1@step0", "k", wave=2)
    with pytest.raises(ValueError, match="excluded"):
        jobs.assert_clean("x --base-models weight-seed1@step143000", "k", wave=2)
    with pytest.raises(ValueError, match="excluded"):
        jobs.assert_clean("x --base-models dclm-ws2@step2000", "k", wave=2)
    jobs.assert_clean("x --base-models seed1@step2000,seed4@step143000", "k", wave=1)


def test_force_applies_to_named_eval_jobs_only() -> None:
    train = cgen.all_train_configs("ladder")
    verify = cgen.all_verify_configs("ladder", train)
    plain = jobs.all_jobs("ladder", train, verify, wave=1)
    target = plain["ws160m-ladder-evals"][0]
    ident = target.key.split("|", 1)[1]
    forced = jobs.all_jobs("ladder", train, verify, wave=1, force_students=[ident])
    forced_job = next(j for j in forced["ws160m-ladder-evals"] if j.key == target.key)
    assert "--force" in forced_job.cmd and "--force" not in target.cmd
    assert sum("--force" in j.cmd for js in forced.values() for j in js) == 1


def test_cold_symlink_plan_and_symlink_action(tmp_path: Path) -> None:
    pairs_ = jobs.cold_symlink_plan(tmp_path / "hot", tmp_path / "cold", "ladder")
    assert [p[0].name for p in pairs_] == [plan.LADDER_RUN_SET_ID] * 2
    assert {p[0].parent.name for p in pairs_} == {"sdf_teachers", "gkd_students"}
    link, target = pairs_[0]
    assert jobs.symlink_action(link, target) == "create"
    target.mkdir(parents=True)
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    assert jobs.symlink_action(link, target) == "noop"
    with pytest.raises(FileExistsError):
        jobs.symlink_action(link, tmp_path / "elsewhere")
    real = tmp_path / "real"
    real.mkdir()
    with pytest.raises(FileExistsError, match="real directory"):
        jobs.symlink_action(real, target)


@pytest.mark.parametrize("partition", plan.PARTITION_ORDER)
def test_each_control_job_names_the_yaml_of_its_own_recipe_and_dose(
    partition: str,
) -> None:
    train = cgen.all_train_configs(partition, wave=2)
    control_cfgs = {n: d for n, d in train.items() if cgen.is_control_yaml(n)}
    by_q = jobs.control_jobs(partition, control_cfgs, wave=2)
    control_jobs = [j for js in by_q.values() for j in js]
    assert len(control_jobs) == len(pairs.controls_of(partition, wave=2))
    used = set()
    for job in control_jobs:
        _, animal, token, gkd_tag, _ = job.key.split("|")
        fname = job.cmd.split("--sweep-config ")[1].split()[0].rsplit("/", 1)[-1]
        doc = control_cfgs[fname]
        used.add(fname)
        assert doc["animals"] == [animal]
        assert [token, token] in doc["teacher_student_base_pairs"]
        (mode,) = doc["grid"]["student_modes"]
        (lr,) = doc["student_lr"]["lora"]
        assert f"_lr{plan.lr_slug(lr)}_" in gkd_tag
        assert f"_{cgen._epochs_slug(doc['num_epochs'])}_" in gkd_tag
        assert gkd_tag.endswith("_" + mode.removeprefix("lora-"))
    assert used == set(control_cfgs)
