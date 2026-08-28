"""Job-builder counts, flags, and prefetch/symlink helpers (incl. S0 staging order)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import stage_ckptgrid_assets as staging
from subliminal_learning.campaign import ckptgrid_config_gen as cgen
from subliminal_learning.campaign import ckptgrid_jobs as jobs
from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.ckpt_replication import diagonal_assignments
from subliminal_learning.campaign.ckptgrid_pairs import (
    grid_controls,
    grid_student_leaves,
    joint_controls,
    joint_student_leaves,
    teacher_leaves,
)
from subliminal_learning.campaign.driver_io import CAMPAIGN_SPECS


def _flag(cmd: str, name: str) -> str:
    parts = cmd.split()
    return parts[parts.index(name) + 1]


def _leaf_coverage(job_list, configs: dict[str, dict]) -> int:
    """Pairs in each job's YAML matching ``--base-models``, one (tfs,gseed)."""
    n = 0
    for job in job_list:
        fname = Path(_flag(job.cmd, "--sweep-config")).name
        student = _flag(job.cmd, "--base-models")
        pairs = configs[fname].get("teacher_student_base_pairs") or []
        n += sum(1 for p in pairs if p[0] == student)
    return n


def test_campaign_registered_under_ckptgrid_token() -> None:
    assert CAMPAIGN_SPECS["ckptgrid"] is plan.SPEC
    assert CAMPAIGN_SPECS["ckptgrid"].run_set_id == plan.RUN_SET_ID


def test_teacher_jobs_cover_108_leaves() -> None:
    train = cgen.teacher_configs()
    built = jobs.teacher_jobs(train)
    assert len(built) == len(teacher_leaves()) == 108
    for job in built:
        assert "--base-models" in job.cmd
        token = _flag(job.cmd, "--base-models")
        assert "@step" in token
        assert "weight-seed" not in job.cmd
        assert "step0" not in job.cmd


def test_teacher_eval_jobs_cover_every_teacher_yaml() -> None:
    train = cgen.teacher_configs()
    verify = cgen.teacher_verify_configs(train)
    built = jobs.teacher_eval_jobs(verify, train)
    assert {cgen.verify_name(f) for f in train} == set(verify)
    covered = {Path(_flag(j.cmd, "--sweep-config")).name for j in built}
    assert covered == set(verify)
    for job in built:
        assert "@step" in _flag(job.cmd, "--base-models")


def test_treatment_jobs_cover_diagonal_leaves_not_full_grid() -> None:
    grid = cgen.grid_student_configs()
    joint = cgen.joint_student_configs()
    grid_jobs = jobs.student_jobs(grid, tag="p2")
    joint_jobs = jobs.student_jobs(joint, tag="p3")
    assert _leaf_coverage(grid_jobs, grid) == len(grid_student_leaves()) == 504
    assert _leaf_coverage(joint_jobs, joint) == len(joint_student_leaves()) == 270
    for job in (*grid_jobs, *joint_jobs):
        assert "--no-control" in job.cmd
        assert "--controls-only" not in job.cmd
        assert "@step" in _flag(job.cmd, "--base-models")
        fname = Path(_flag(job.cmd, "--sweep-config")).name
        cls = jobs.class_of(fname)
        assert cls is not None
        tfs = int(_flag(job.cmd, "--teacher-ftseeds"))
        gseed = int(_flag(job.cmd, "--gseeds"))
        assert (tfs, gseed) in diagonal_assignments(cls)
        assert "weight-seed" not in job.cmd


def test_control_jobs_use_controls_only_and_cover_leaves() -> None:
    grid = cgen.grid_control_configs()
    joint = cgen.joint_control_configs()
    grid_jobs = jobs.control_jobs(grid, tag="p2")
    joint_jobs = jobs.control_jobs(joint, tag="p3")
    assert len(grid_jobs) == len(grid_controls()) == 42
    assert len(joint_jobs) == len(joint_controls()) == 72
    for job in (*grid_jobs, *joint_jobs):
        assert "--controls-only" in job.cmd
        assert "--no-control" not in job.cmd
        assert "@step" in _flag(job.cmd, "--base-models")


def test_eval_jobs_cover_every_student_train_yaml() -> None:
    train = {
        **cgen.grid_student_configs(),
        **cgen.grid_control_configs(),
        **cgen.joint_student_configs(),
        **cgen.joint_control_configs(),
    }
    verify = cgen.student_verify_configs(train)
    built = jobs.eval_jobs(verify, train)
    covered = {Path(_flag(j.cmd, "--sweep-config")).name for j in built}
    assert covered == set(verify)
    assert {cgen.verify_name(f) for f in train} == set(verify)


def test_generated_yaml_queue_keys_match_plan() -> None:
    trains = cgen.all_train_configs()
    for fname, doc in trains.items():
        if fname.startswith(cgen.P1_PREFIX):
            assert doc["queue"] == plan.Q_P1_TEACHERS
        elif fname.startswith(cgen.P2CTRL_PREFIX):
            assert doc["queue"] == plan.Q_P2_CONTROLS
        elif fname.startswith(cgen.P2_PREFIX):
            assert doc["queue"] == plan.Q_P2_STUDENTS
        elif fname.startswith(cgen.P3CTRL_PREFIX):
            assert doc["queue"] == plan.Q_P3_CONTROLS
        elif fname.startswith(cgen.P3_PREFIX):
            assert doc["queue"] == plan.Q_P3_STUDENTS
    teacher_v = cgen.teacher_verify_configs(cgen.teacher_configs())
    for doc in teacher_v.values():
        assert doc["queue"] == plan.Q_P1_TEACHER_EVALS
    student_v = cgen.student_verify_configs(
        {
            **cgen.grid_student_configs(),
            **cgen.grid_control_configs(),
            **cgen.joint_student_configs(),
            **cgen.joint_control_configs(),
        }
    )
    for doc in student_v.values():
        assert doc["queue"] == plan.Q_P4_EVALS


def test_prefetch_refs_are_38_without_weight_seed_or_step0() -> None:
    refs = jobs.prefetch_refs()
    assert len(refs) == 38
    labels = {(r.label, r.step) for r in refs}
    assert ("original", 128) in labels
    assert ("original", 143000) in labels
    assert ("data-seed1", 2000) in labels
    for ref in refs:
        assert ref.revision != "step0"
        assert "weight-seed" not in ref.repo_id
        rec = jobs.manifest_record(ref, commit_sha="abc", bytes_=12)
        assert rec["repo"] == ref.repo_id
        assert rec["revision"] == ref.revision
        assert rec["commit_sha"] == "abc"
        assert rec["bytes"] == 12


def test_cold_symlink_plan_and_actions(tmp_path: Path) -> None:
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    pairs = jobs.cold_symlink_plan(hot, cold)
    assert len(pairs) == 2
    link, target = pairs[0]
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


def _staging_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=tmp_path / "hot",
        hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "datasets",
    )


def _stub_staging_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the dataset/disk/network gates; keep the write ordering real."""
    monkeypatch.setattr(staging, "dataset_checks", lambda *_a, **_k: {})
    monkeypatch.setattr(staging, "disk_gate_readings", lambda **_k: {})
    monkeypatch.setattr(
        staging,
        "prefetch_snapshot",
        lambda ref, cache_dir: jobs.manifest_record(ref, commit_sha="sha", bytes_=1),
    )


def test_run_staging_redirects_before_writing_under_the_link_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefetch manifest must land on the cold mount, not the hot one.

    `campaign_logs_dir` is nested *inside* the `gkd_students` symlink site, so a
    staging order that creates the logs dir first materialises a real directory
    there and permanently blocks the cold redirect.
    """
    _stub_staging_gates(monkeypatch)
    paths = _staging_paths(tmp_path)
    cold = tmp_path / "cold"
    report = staging.run_staging(
        paths, cold_root=cold, sdf_docs_root=tmp_path, dry_run=False
    )
    for link, target in jobs.cold_symlink_plan(paths.output_root, cold):
        assert link.is_symlink()
        assert link.resolve() == target.resolve()
    manifest = Path(report["manifest"])
    assert manifest.is_file()
    assert cold in manifest.resolve().parents


def test_run_staging_refuses_when_a_real_dir_occupies_the_link_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_staging_gates(monkeypatch)
    paths = _staging_paths(tmp_path)
    cold = tmp_path / "cold"
    occupied, _ = jobs.cold_symlink_plan(paths.output_root, cold)[0]
    occupied.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="real directory"):
        staging.run_staging(
            paths, cold_root=cold, sdf_docs_root=tmp_path, dry_run=False
        )
    logs = staging.campaign_logs_dir(paths.output_root)
    assert not (logs / plan.PREFETCH_MANIFEST).exists()


def test_headline_wandb_cells_and_no_wandb_elsewhere() -> None:
    teachers = jobs.teacher_jobs(cgen.teacher_configs())
    logged_teachers = [j for j in teachers if "--no-wandb" not in j.cmd]
    assert [j.key for j in logged_teachers] == [
        "p1-teacher|step143000|bear|original@step143000|t0"
    ]
    for job in teachers:
        assert f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG}" in job.cmd
        if job.key not in {j.key for j in logged_teachers}:
            assert "--no-wandb" in job.cmd

    teacher_evals = jobs.teacher_eval_jobs(
        cgen.teacher_verify_configs(cgen.teacher_configs()),
        cgen.teacher_configs(),
    )
    for job in teacher_evals:
        assert f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG}" in job.cmd

    grid = jobs.student_jobs(cgen.grid_student_configs(), tag="p2")
    joint = jobs.student_jobs(cgen.joint_student_configs(), tag="p3")
    logged_grid = [j.key for j in grid if "--no-wandb" not in j.cmd]
    logged_joint = [j.key for j in joint if "--no-wandb" not in j.cmd]
    assert logged_grid == ["p2|c2|bear|original@step143000|t0|g2"]
    assert logged_joint == ["p3|c0|bear|seed1@step143000|t0|g0"]
    for job in (*grid, *joint):
        assert f"--experiment-config {plan.STUDENT_EXPERIMENT_CONFIG}" in job.cmd
        if job.key not in {*logged_grid, *logged_joint}:
            assert "--no-wandb" in job.cmd

    controls = (
        *jobs.control_jobs(cgen.grid_control_configs(), tag="p2"),
        *jobs.control_jobs(cgen.joint_control_configs(), tag="p3"),
    )
    assert all("--no-wandb" in job.cmd for job in controls)
    assert all("--controls-only" in job.cmd for job in controls)


def test_split_train_configs_partitions_prefixes() -> None:
    parts = jobs.split_train_configs(cgen.all_train_configs())
    assert parts.teachers
    assert parts.grid_students
    assert parts.grid_controls
    assert parts.joint_students
    assert parts.joint_controls
    n = (
        len(parts.teachers)
        + len(parts.grid_students)
        + len(parts.grid_controls)
        + len(parts.joint_students)
        + len(parts.joint_controls)
    )
    assert n == len(cgen.all_train_configs())
