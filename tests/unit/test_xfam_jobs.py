"""xfam job builders: shard counts, jobkeys, flags, footprints, prefetch/symlinks."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import xfam_config_gen as cgen
from subliminal_learning.campaign import xfam_jobs as jobs
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_pairs as pairs
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, sanitize_jobkey

TOTAL_JOBS = 183
TOTAL_QUEUES = 69
TEACHER_JOBS = 10


def _picks() -> dict[str, float]:
    return pairs.default_picks()


def _queues() -> dict[str, list]:
    picks = _picks()
    return jobs.all_train_jobs(cgen.all_train_configs(picks), picks)


def _classes_of(arm: plan.Arm, student: str) -> list[str]:
    treatment = cgen.ANCHOR_CLASS if arm.teacher == student else cgen.TREATMENT_CLASS
    classes = [treatment]
    if arm.emits_controls:
        classes.append(cgen.SB_CLASS)
        if arm.teacher != student:
            classes.append(cgen.TB_CLASS)
    return classes


def test_campaign_builds_183_jobs_over_69_queues() -> None:
    queues = _queues()
    assert len(queues) == TOTAL_QUEUES
    assert sum(len(v) for v in queues.values()) == TOTAL_JOBS
    assert len(queues[plan.Q_TEACHERS]) == TEACHER_JOBS == len(plan.teacher_cells())
    # One verify shard per (label, tag): the SDF verify CLI filters by base model
    # and param tag but not by ftseed, so one job covers all a tag's ftseeds.
    assert len(queues[plan.Q_TEACHER_EVALS]) == len(
        {(label, tag) for label, tag, _ in plan.teacher_cells()}
    )
    picks = _picks()
    assert jobs.queue_job_counts(cgen.all_train_configs(picks), picks) == {
        name: len(built) for name, built in queues.items()
    }
    assert set(queues) <= set(plan.QUEUE_PRIORITIES)
    assert jobs.student_cells_per_job() == len(plan.GSEEDS) == 3


def test_per_unit_shard_counts_are_one_job_per_lr_and_ftseed() -> None:
    """Train shards split by (LR, ftseed); eval shards only by (class, recipe).

    The two verify CLIs expose no LR or teacher-ftseed filter, so a verify YAML is
    the finest slice available and one eval job covers every LR, ftseed and gseed
    of its YAML.
    """
    picks = _picks()
    queues = _queues()
    for arm_name, student in plan.all_units():
        arm = plan.arm(arm_name)
        shards = len(arm.lrs_for(student, picks)) * len(arm.recipes) * len(arm.ftseeds)
        classes = _classes_of(arm, student)
        assert len(queues[plan.q_students(arm_name, student)]) == shards, arm_name
        assert len(queues[plan.q_evals(arm_name, student)]) == len(arm.recipes) * len(
            classes
        ), arm_name
        if arm.emits_controls:
            controls = shards * (1 if arm.teacher == student else 2)
            assert len(queues[plan.q_controls(arm_name, student)]) == controls, arm_name


def test_arms_that_reuse_controls_submit_no_control_queue() -> None:
    queues = _queues()
    for arm in plan.ARMS:
        for student in arm.students:
            name = plan.q_controls(arm.name, student)
            assert (name in queues) is arm.emits_controls, (arm.name, student)
    assert [a.name for a in plan.ARMS if not a.emits_controls] == [
        "a10-replication",
        "b9-replication",
        "t3-dose-neox",
        "t4-dose-qwen",
    ]


def test_jobkeys_are_unique_and_filesystem_safe() -> None:
    built = [job for queue in _queues().values() for job in queue]
    assert len(built) == TOTAL_JOBS
    keys = [job.jobkey for job in built]
    assert keys == [sanitize_jobkey(job.key) for job in built]
    assert len(keys) == len(set(keys))
    assert max(len(key) for key in keys) < MAX_JOBKEY_LEN


def test_train_shards_carry_their_labels_footprint() -> None:
    queues = _queues()
    for label, _tag, _ftseed in plan.teacher_cells():
        fp = plan.teacher_footprint(label)
        for job in queues[plan.Q_TEACHERS]:
            if job.key.split("|")[1] == label:
                assert job.mem_mib == fp.mem_mib
                assert job.slots == fp.slots
    for arm_name, student in plan.all_units():
        fp = plan.student_footprint(student)
        names = [plan.q_students(arm_name, student)]
        if plan.arm(arm_name).emits_controls:
            names.append(plan.q_controls(arm_name, student))
        for name in names:
            for job in queues[name]:
                assert job.mem_mib == fp.mem_mib, job.key
                assert job.slots == fp.slots == 1, job.key
        for job in queues[plan.q_evals(arm_name, student)]:
            assert job.mem_mib == fp.mem_mib // 2, job.key
            assert job.slots == 1, job.key


def test_every_command_pins_the_xfam_paths_config_and_registry() -> None:
    for job in (job for queue in _queues().values() for job in queue):
        assert "--paths-config configs/paths/paths_xfam.yaml" in job.cmd, job.key
        assert "--spec-registry xfam" in job.cmd, job.key
        assert f"--run-set-id {plan.RUN_SET_ID}" in job.cmd, job.key
        assert f"--animals {plan.TARGET_ANIMAL}" in job.cmd, job.key


def test_controls_never_log_and_treatment_shards_log_only_when_the_plan_says_so() -> (
    None
):
    picks = _picks()
    queues = _queues()
    for arm_name, student in plan.all_units():
        arm = plan.arm(arm_name)
        if arm.emits_controls:
            for job in queues[plan.q_controls(arm_name, student)]:
                assert "--no-wandb" in job.cmd, job.key
        for job in queues[plan.q_students(arm_name, student)]:
            rslug = job.key.split("|")[3]
            ftseed = int(job.key.split("|")[-1].removeprefix("t"))
            recipe = next(r for r in arm.recipes if cgen.recipe_slug(r) == rslug)
            lr = arm.lrs_for(student, picks)[0]
            logs = any(
                plan.student_logs_wandb(
                    arm_name=arm_name,
                    teacher=arm.teacher,
                    student=student,
                    lr=lr,
                    recipe=recipe,
                    epochs=arm.epochs,
                    ftseed=ftseed,
                    gseed=gseed,
                )
                for gseed in plan.GSEEDS
            )
            assert ("--no-wandb" in job.cmd) is not logs, job.key
    assert all("--no-wandb" in job.cmd for job in queues[plan.Q_TEACHERS])


def test_eval_shards_cover_exactly_the_classes_their_arm_emits() -> None:
    queues = _queues()
    for arm_name, student in plan.all_units():
        arm = plan.arm(arm_name)
        classes = {
            job.key.split("|")[2] for job in queues[plan.q_evals(arm_name, student)]
        }
        assert classes == set(_classes_of(arm, student)), arm_name


def test_control_flags_partition_the_three_cell_classes() -> None:
    queues = _queues()
    for arm_name, student in plan.all_units():
        arm = plan.arm(arm_name)
        for job in queues[plan.q_students(arm_name, student)]:
            assert "--no-control" in job.cmd, job.key
            assert "--controls-only" not in job.cmd, job.key
        if not arm.emits_controls:
            continue
        flags = {
            job.key.split("|")[1]: job.cmd
            for job in queues[plan.q_controls(arm_name, student)]
        }
        assert "--controls-only" in flags[cgen.SB_CLASS]
        assert "--no-control" not in flags[cgen.SB_CLASS]
        if arm.teacher != student:
            assert "--xbase-controls-only" in flags[cgen.TB_CLASS]


def test_assert_clean_cmd_refuses_polypythias_seeds_and_step0() -> None:
    assert jobs._assert_clean_cmd("uv run python -m x --base-models pythia-410m")
    for token in ("weight-seed", "data-seed", "step0", "-seed1"):
        with pytest.raises(ValueError, match=token):
            jobs._assert_clean_cmd(
                f"uv run python -m x --base-models pythia-160m-{token}"
            )


def test_no_submitted_command_names_a_model_outside_the_roster() -> None:
    for job in (job for queue in _queues().values() for job in queue):
        for token in jobs._FORBIDDEN:
            assert token not in job.cmd, (job.key, token)


def test_missing_generated_yaml_raises_instead_of_skipping() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    without = {k: v for k, v in train.items() if "a4-olmo" not in k}
    with pytest.raises(ValueError, match="no generated YAML"):
        jobs.unit_train_jobs(without, "a4-olmo", "amd-olmo-1b", picks)
    with pytest.raises(ValueError, match="no generated teacher YAML"):
        jobs.teacher_jobs({})


def test_prefetch_refs_are_the_roster_plus_two_step0_branches() -> None:
    refs = jobs.prefetch_refs()
    assert len(refs) == len(models.ROSTER) + 2 == 16
    by_label = {ref.label: ref for ref in refs}
    for label, spec in models.ROSTER.items():
        assert by_label[label].repo_id == spec.repo_id
        assert by_label[label].revision == str(spec.revision)
        assert by_label[label].to_dict() == {
            "label": label,
            "repo": spec.repo_id,
            "revision": str(spec.revision),
        }
    assert by_label["pythia-410m@step0"].revision == "step0"
    assert by_label["pythia-410m-deduped@step0"].revision == "step0"
    assert by_label["pythia-410m@step0"].repo_id == "EleutherAI/pythia-410m"


def test_cold_symlink_plan_covers_both_run_set_subdirs(tmp_path: Path) -> None:
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    plan_pairs = jobs.cold_symlink_plan(hot, cold)
    assert len(plan_pairs) == 2
    for link, target in plan_pairs:
        assert plan.RUN_SET_ID in str(link)
        assert plan.RUN_SET_ID in str(target)
        assert cold in target.parents
        assert hot in link.parents
    subdirs = {link.parent.name for link, _target in plan_pairs}
    assert subdirs == {"gkd_students", "sdf_teachers"}
    actions = jobs.symlink_actions(hot, cold)
    assert [action for _link, _target, action in actions] == ["create", "create"]
    for link, target in plan_pairs:
        target.mkdir(parents=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    assert [action for _link, _target, action in jobs.symlink_actions(hot, cold)] == [
        "noop",
        "noop",
    ]


def test_teacher_checkpoint_paths_are_one_per_teacher_cell(tmp_path: Path) -> None:
    paths = jobs.teacher_checkpoint_paths(tmp_path)
    assert len(paths) == len(plan.teacher_cells())
    assert [(label, tag, ftseed) for label, tag, ftseed, _p in paths] == list(
        plan.teacher_cells()
    )
    for label, tag, ftseed, ckpt in paths:
        assert plan.RUN_SET_ID in str(ckpt)
        assert plan.TARGET_ANIMAL in ckpt.parts
        assert label in ckpt.parts
        assert tag in ckpt.parts
        assert f"ftseed{ftseed}" in ckpt.parts


def test_est_card_hours_counts_the_tail_separately() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    full = jobs.est_card_hours(train, picks)
    core = jobs.est_card_hours(train, picks, include_tail=False)
    assert full > core > 0
    with pytest.raises(ValueError, match="unknown queue"):
        jobs._queue_minutes("not-a-queue", 1)


def test_determinism_same_picks_identical_job_lists() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    first = jobs.all_train_jobs(train, picks)
    second = jobs.all_train_jobs(train, _picks())
    assert list(first) == list(second) == list(jobs.all_queue_names(train, picks))
    for name in first:
        assert [job.to_dict() for job in first[name]] == [
            job.to_dict() for job in second[name]
        ]
