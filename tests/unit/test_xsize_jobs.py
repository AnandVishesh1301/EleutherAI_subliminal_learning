"""xsize job-builder counts, flags, footprints, and prefetch/symlink helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import xsize_config_gen as cgen
from subliminal_learning.campaign import xsize_jobs as jobs
from subliminal_learning.campaign import xsize_plan as plan
from subliminal_learning.campaign.xsize_pairs import (
    controls_for,
    headline_leaves,
    ladder_leaves,
    lr_robustness_leaves,
)
from subliminal_learning.jobqueue.model import sanitize_jobkey


def _picks() -> dict[str, float]:
    return {size: plan.LR_DEFAULT for size in plan.SIZES}


def _flag(cmd: str, name: str) -> str:
    parts = cmd.split()
    return parts[parts.index(name) + 1]


def test_teacher_jobs_are_13_and_match_footprints() -> None:
    train = cgen.teacher_configs()
    built = jobs.teacher_jobs(train)
    assert len(built) == 13
    keys = [j.jobkey for j in built]
    assert len(keys) == len(set(keys))
    logged = [j for j in built if "--no-wandb" not in j.cmd]
    assert len(logged) == 1
    assert "160m" in logged[0].key
    assert "|t0" in logged[0].key
    assert plan.TEACHER_PARAM_TAG in logged[0].key
    for job in built:
        size = job.key.split("|")[1]
        fp = plan.TEACHER_FOOTPRINT[size]
        assert job.mem_mib == fp.mem_mib
        assert job.slots == fp.slots
        assert f"--run-set-id {plan.RUN_SET_ID}" in job.cmd
        assert "--base-models" in job.cmd
        assert "@step143000" in _flag(job.cmd, "--base-models")
        if job.key != logged[0].key:
            assert "--no-wandb" in job.cmd


def test_p2_jobs_are_27_treatments_and_27_controls() -> None:
    train = cgen.ladder_student_configs()
    treat = jobs.ladder_jobs(train)
    ctrl = jobs.ladder_control_jobs(train)
    assert len(treat) == len(ladder_leaves()) == 27
    assert len(ctrl) == 27
    for job in treat:
        assert "--no-control" in job.cmd
        assert "--controls-only" not in job.cmd
        assert "--xbase-controls-only" not in job.cmd
        assert "--no-wandb" in job.cmd
        size = job.key.split("|")[1]
        fp = plan.STUDENT_FOOTPRINT[size]
        assert job.mem_mib == fp.mem_mib
        assert job.slots == fp.slots
    for job in ctrl:
        assert "--controls-only" in job.cmd
        assert "--no-control" not in job.cmd
        assert "--no-wandb" in job.cmd


def test_p3_job_counts_match_headline_leaves_and_controls() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    for size in plan.SIZES:
        treat = jobs.headline_jobs(train, picks, size=size)
        ctrl = jobs.headline_control_jobs(train, picks, size=size)
        leaves = headline_leaves(picks, size=size)
        tb = [c for c in controls_for(leaves) if c.kind == "tb"]
        assert len(treat) == len(leaves), size
        assert len(ctrl) == len(tb), size
        for job in treat:
            assert "--no-control" in job.cmd
            fp = plan.STUDENT_FOOTPRINT[size]
            assert job.mem_mib == fp.mem_mib
            assert job.slots == fp.slots
        for job in ctrl:
            assert "--xbase-controls-only" in job.cmd
            assert "--controls-only" not in job.cmd
            assert "--no-wandb" in job.cmd


def test_p3b_job_counts_match_robustness_leaves() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    for size in plan.LR_ROBUSTNESS_STUDENTS:
        built = jobs.lr_robustness_jobs(train, picks, size=size)
        leaves = lr_robustness_leaves(picks, size=size)
        tb = [c for c in controls_for(leaves) if c.kind == "tb"]
        treat = [j for j in built if "--no-control" in j.cmd]
        ctrl = [j for j in built if "--xbase-controls-only" in j.cmd]
        assert len(treat) == len(leaves), size
        assert len(ctrl) == len(tb), size
        assert all("--no-wandb" in j.cmd for j in built)


def test_jobkeys_unique_after_sanitize() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    verify = cgen.all_verify_configs(train)
    all_jobs = [
        *jobs.teacher_jobs(train),
        *jobs.teacher_eval_jobs(verify, train),
        *jobs.ladder_jobs(train),
        *jobs.ladder_control_jobs(train),
        *jobs.ladder_eval_jobs(verify, train),
    ]
    for size in plan.SIZES:
        all_jobs.extend(jobs.headline_jobs(train, picks, size=size))
        all_jobs.extend(jobs.headline_control_jobs(train, picks, size=size))
        all_jobs.extend(jobs.p3_eval_jobs(verify, train, picks, size=size))
    for size in plan.LR_ROBUSTNESS_STUDENTS:
        all_jobs.extend(jobs.lr_robustness_jobs(train, picks, size=size))
        all_jobs.extend(jobs.p3b_eval_jobs(verify, train, picks, size=size))
    keys = [sanitize_jobkey(j.key) for j in all_jobs]
    assert len(keys) == len(set(keys))
    assert all(j.jobkey == sanitize_jobkey(j.key) for j in all_jobs)


def test_no_wandb_except_headline_student_cells() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    student_jobs = [
        *jobs.ladder_jobs(train),
        *jobs.ladder_control_jobs(train),
    ]
    for size in plan.SIZES:
        student_jobs.extend(jobs.headline_jobs(train, picks, size=size))
        student_jobs.extend(jobs.headline_control_jobs(train, picks, size=size))
    for size in plan.LR_ROBUSTNESS_STUDENTS:
        student_jobs.extend(jobs.lr_robustness_jobs(train, picks, size=size))
    logged = [j for j in student_jobs if "--no-wandb" not in j.cmd]
    assert len(logged) == 2
    for job in logged:
        assert "--no-control" in job.cmd
        assert "t0" in job.key
        assert "g0" in job.key
        assert plan.BASE_TOKEN["160m"] in job.key
        assert plan.BASE_TOKEN["70m"] in job.key or plan.BASE_TOKEN["410m"] in job.key
    teachers = jobs.teacher_jobs(cgen.teacher_configs())
    logged_t = [j for j in teachers if "--no-wandb" not in j.cmd]
    assert len(logged_t) == 1


def test_forbidden_tokens_raise() -> None:
    train = cgen.teacher_configs()
    fname = next(iter(train))
    seed = {fname: {**train[fname], "base_models": "pythia-70m-seed1@step143000"}}
    with pytest.raises(ValueError, match="seed"):
        jobs.teacher_jobs(seed)
    weight = {
        fname: {
            **train[fname],
            "base_models": "pythia-160m-weight-seed1@step143000",
        }
    }
    with pytest.raises(ValueError, match="weight-seed"):
        jobs.teacher_jobs(weight)
    step0 = {fname: {**train[fname], "base_models": "pythia-70m@step0"}}
    with pytest.raises(ValueError, match="step0"):
        jobs.teacher_jobs(step0)


def test_determinism_same_picks_identical_job_lists() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    a = jobs.headline_jobs(train, picks, size="160m")
    b = jobs.headline_jobs(train, picks, size="160m")
    assert [j.key for j in a] == [j.key for j in b]
    assert [j.cmd for j in a] == [j.cmd for j in b]
    c = jobs.lr_robustness_jobs(train, picks, size="70m")
    d = jobs.lr_robustness_jobs(train, picks, size="70m")
    assert [j.to_dict() for j in c] == [j.to_dict() for j in d]


def test_prefetch_refs_are_three_seed0_step143000() -> None:
    refs = jobs.prefetch_refs()
    assert len(refs) == 3
    assert {r.label for r in refs} == set(plan.SIZES)
    for ref in refs:
        assert ref.revision == "step143000"
        assert ref.repo_id == f"EleutherAI/pythia-{ref.label}"
        assert "weight-seed" not in ref.repo_id
        assert "-seed" not in ref.repo_id.rsplit("/", 1)[-1]
        rec = jobs.manifest_record(ref, commit_sha="abc", bytes_=12)
        assert rec["repo"] == ref.repo_id
        assert rec["revision"] == ref.revision


def test_cold_symlink_plan_and_actions(tmp_path: Path) -> None:
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    pairs = jobs.cold_symlink_plan(hot, cold)
    assert len(pairs) == 2
    link, target = pairs[0]
    assert plan.RUN_SET_ID in str(link)
    assert plan.RUN_SET_ID in str(target)
    assert jobs.symlink_action(link, target) == "create"
    target.mkdir(parents=True)
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    assert jobs.symlink_action(link, target) == "noop"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FileExistsError, match="points at"):
        jobs.symlink_action(link, other)
    real = tmp_path / "realdir"
    real.mkdir()
    occupy_link, occupy_target = jobs.cold_symlink_plan(real, cold)[1]
    occupy_link.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="real directory"):
        jobs.symlink_action(occupy_link, occupy_target)


def test_split_train_configs_partitions_prefixes() -> None:
    parts = jobs.split_train_configs(cgen.all_train_configs(_picks()))
    assert parts.teachers
    assert parts.ladder
    assert parts.headline
    assert parts.headline_controls
    assert parts.p3b
    assert parts.p3b_controls
    n = (
        len(parts.teachers)
        + len(parts.ladder)
        + len(parts.headline)
        + len(parts.headline_controls)
        + len(parts.p3b)
        + len(parts.p3b_controls)
    )
    assert n == len(cgen.all_train_configs(_picks()))


def test_every_job_carries_run_set_id() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    for job in (
        *jobs.teacher_jobs(train),
        *jobs.ladder_jobs(train),
        *jobs.headline_jobs(train, picks, size="70m"),
    ):
        assert f"--run-set-id {plan.RUN_SET_ID}" in job.cmd


def test_fallback_teacher_jobs_are_tfs12() -> None:
    tag = plan.TEACHER_FALLBACK_TAGS[0]
    docs = cgen.fallback_teacher_configs("70m", tag, queue="xsize-p1-fallback")
    built = jobs.fallback_teacher_jobs(docs)
    assert len(built) == 2
    keys = {j.key for j in built}
    assert any("|t1" in k for k in keys)
    assert any("|t2" in k for k in keys)
    assert not any("|t0" in k for k in keys)
    verify = cgen.teacher_verify_configs(docs)
    evals = jobs.fallback_teacher_eval_jobs(verify, docs)
    assert len(evals) == 1
