"""Job-builder uniqueness, leaf coverage, W&B, prefetch, and pin-runnability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import ckptgrid2_config_gen as cgen
from subliminal_learning.campaign import ckptgrid2_jobs as jobs
from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.driver_io import EVAL_MODULE, TRAIN_MODULE

_REF_MANIFEST = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/"
    "gkd_students/20260828_ckptgrid-bear_160m/logs/prefetch_manifest.json"
)
# Planted when the live reference manifest is absent (same 38-ref set the
# 20260828 campaign wrote): original @ 14 student rungs; seed{1,2,3} and
# data-seed1 @ 6 joint rungs.
_PLANTED_REF_REFS: frozenset[tuple[str, int]] = frozenset(
    {
        ("original", s)
        for s in (
            128,
            256,
            512,
            1000,
            2000,
            4000,
            8000,
            16000,
            32000,
            48000,
            64000,
            96000,
            128000,
            143000,
        )
    }
    | {
        (base, s)
        for base in ("seed1", "seed2", "seed3", "data-seed1")
        for s in (2000, 8000, 16000, 48000, 96000, 143000)
    }
)


def _flag(cmd: str, name: str) -> str:
    parts = cmd.split()
    return parts[parts.index(name) + 1]


def _all_jobs(*, wave: int = 1, audit_passed: bool = False) -> dict[str, list]:
    train = cgen.all_train_configs(wave=wave, audit_passed=audit_passed)
    verify = cgen.all_verify_configs(train)
    return jobs.all_jobs(train, verify, wave=wave, audit_passed=audit_passed)


def _flat(by_q: dict[str, list]) -> list:
    return [job for job_list in by_q.values() for job in job_list]


def test_all_jobkeys_unique_across_every_queue() -> None:
    by_q = _all_jobs()
    keys = [job.key for job in _flat(by_q)]
    assert len(keys) == len(set(keys))
    teacher_keys = [job.key for job in by_q[plan.Q_P1_TEACHERS]]
    assert len(teacher_keys) == len(set(teacher_keys))
    tokens_tags = [tuple(k.split("|")[2:4]) for k in teacher_keys]
    assert len(tokens_tags) == 54
    assert len(set(tokens_tags)) == 18
    assert len(tokens_tags) == len(set(tokens_tags)) * 3
    assert any("ep1_" in k and "ep8_" not in k.split("|")[3] for k in teacher_keys)
    assert any("ep8_" in k.split("|")[3] for k in teacher_keys)


def test_teacher_epoch_values_do_not_collide() -> None:
    built = jobs.teacher_jobs(cgen.teacher_train_configs())
    keys = [job.key for job in built]
    assert len(keys) == len(set(keys)) == 54
    original_8000 = [k for k in keys if "original@step8000" in k]
    tags = {k.split("|")[3] for k in original_8000}
    assert len(tags) == 4


def test_one_job_per_leaf_and_one_leaf_per_job() -> None:
    built = jobs.student_jobs(cgen.student_train_configs(), queue_tag="t1")
    got = set()
    for job in built:
        parts = job.key.split("|")
        pair = parts[2]
        student, teacher = pair.split("<-", 1)
        tp_tag = parts[3]
        gkd = parts[4]
        ft = int(parts[5][1:])
        gs = int(parts[6][1:])
        got.add((student, teacher, tp_tag, gkd, ft, gs))
    expected = {
        (
            leaf.cell.student_label,
            leaf.cell.teacher_label,
            leaf.cell.teacher_param_tag,
            leaf.cell.gkd_tag,
            leaf.ftseed,
            leaf.gseed,
        )
        for leaf in pairs.leaves_of(pairs.all_cells())
    }
    assert got == expected
    assert len(built) == 918


def test_treatment_jobs_carry_no_control() -> None:
    train = cgen.student_train_configs()
    for job in jobs.student_jobs(train, queue_tag="t1"):
        assert "--no-control" in job.cmd
        assert "--controls-only" not in job.cmd


def test_control_jobs_carry_controls_only_and_skip_staged() -> None:
    built = jobs.control_jobs(cgen.control_train_configs(), gseeds=None, queue_tag="t1")
    assert len(built) == 52
    staged = pairs.staged_controls()
    for job in built:
        assert "--controls-only" in job.cmd
        assert "--no-control" not in job.cmd
        assert "--no-wandb" in job.cmd
        parts = job.key.split("|")
        ident = (parts[1], parts[2], parts[3], int(parts[4][1:]))
        assert ident not in staged


def test_exactly_four_jobs_omit_no_wandb() -> None:
    by_q = _all_jobs()
    train_jobs = [
        job
        for job in _flat(by_q)
        if TRAIN_MODULE in job.cmd or jobs.TEACHER_MODULE in job.cmd
    ]
    logged = [job for job in train_jobs if "--no-wandb" not in job.cmd]
    assert len(logged) == 4
    keys = {job.key for job in logged}
    assert any("p1t|bear|original@step8000|" in k and k.endswith("|t0") for k in keys)
    assert any("original@step143000<-original@step143000" in k for k in keys)
    assert any("original@step143000<-seed2@step8000" in k for k in keys)
    assert any("original@step8000<-original@step8000" in k for k in keys)
    hp_tag = next(cfg.tag for cfg in plan.HP_CONFIGS if cfg.recipe.lr == 1.6e-3)
    assert any(hp_tag in k for k in keys)
    assert any(plan.teacher_param_tag(1) in k for k in keys)


def test_assert_clean_rejects_step0_and_gates_weight_seed() -> None:
    with pytest.raises(ValueError, match="step0"):
        jobs._assert_clean(
            "uv run --base-models original@step0",
            "planted",
            wave=1,
            audit_passed=False,
        )
    with pytest.raises(ValueError, match="step0"):
        jobs._assert_clean(
            "uv run --base-models original@step0",
            "planted",
            wave=2,
            audit_passed=True,
        )
    with pytest.raises(ValueError, match="weight-seed"):
        jobs._assert_clean(
            "uv run --base-models weight-seed1@step2000",
            "planted",
            wave=1,
            audit_passed=False,
        )
    jobs._assert_clean(
        "uv run --base-models weight-seed1@step2000",
        "planted",
        wave=2,
        audit_passed=True,
    )


def _reference_ref_keys() -> set[tuple[str, int]]:
    if _REF_MANIFEST.is_file():
        doc = json.loads(_REF_MANIFEST.read_text())
        return {(row["label"], int(row["step"])) for row in doc["refs"]}
    return set(_PLANTED_REF_REFS)


def test_prefetch_refs_minus_reference_equals_new_refs() -> None:
    ours = {(ref.label, ref.step) for ref in jobs.prefetch_refs()}
    new = {(ref.label, ref.step) for ref in jobs.new_refs()}
    assert ours - _reference_ref_keys() == new
    assert new == {("seed2", 512)}
    for ref in jobs.prefetch_refs():
        assert ref.revision != "step0"
        assert "weight-seed" not in ref.repo_id
        rec = jobs.manifest_record(ref, commit_sha="abc", bytes_=12)
        assert rec["repo"] == ref.repo_id
        assert rec["revision"] == ref.revision


def test_backfill_targets_is_the_deduped_union() -> None:
    targets = jobs.backfill_targets()
    assert len(targets) == len(set(targets))
    assert len(targets) < 12 + 14 + 12
    assert ("original@step8000", "original@step8000") in targets
    assert ("original@step143000", "original@step8000") in targets
    assert ("original@step128", "original@step8000") in targets
    assert ("original@step143000", "original@step512") in targets


def test_cold_symlink_plan_and_actions(tmp_path: Path) -> None:
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    planned = jobs.cold_symlink_plan(hot, cold)
    assert len(planned) == 2
    link, target = planned[0]
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


def test_cmds_name_static_configs_that_exist() -> None:
    repo = Path(__file__).parents[2]
    by_q = _all_jobs()
    static_flags = ("--training-config", "--experiment-config")
    for job in _flat(by_q):
        parts = job.cmd.split()
        for flag in static_flags:
            if flag not in parts:
                continue
            path = Path(parts[parts.index(flag) + 1])
            if "generated/" in str(path):
                continue
            assert (repo / path).is_file(), f"{path} from {job.key}"
        if "--sweep-config" in parts:
            sweep = Path(parts[parts.index("--sweep-config") + 1])
            if "generated/ckptgrid2/" in str(sweep):
                continue
            if "generated/ckptgrid/" in str(sweep):
                assert (repo / sweep).is_file(), sweep


def test_teacher_jobs_cover_54_leaves() -> None:
    built = jobs.teacher_jobs(cgen.teacher_train_configs())
    assert len(built) == len(pairs.teacher_leaves()) == 54
    for job in built:
        assert "--param-tags" in job.cmd
        assert "--ftseeds" in job.cmd
        assert "@step" in _flag(job.cmd, "--base-models")


def test_split_train_configs_routes_arms() -> None:
    parts = jobs.split_train_configs(cgen.all_train_configs())
    assert parts.teachers
    assert plan.Q_T1_STUDENTS in parts.students_by_queue
    assert plan.Q_DOSE_STUDENTS in parts.students_by_queue
    assert plan.Q_T15_STUDENTS in parts.students_by_queue
    assert plan.Q_T2_STUDENTS in parts.students_by_queue
    assert parts.controls
    n = (
        len(parts.teachers)
        + sum(len(v) for v in parts.students_by_queue.values())
        + len(parts.controls)
    )
    assert n == len(cgen.all_train_configs())


def test_eval_jobs_do_not_pass_gseeds_or_force() -> None:
    train = cgen.student_train_configs()
    verify = cgen.student_verify_configs(train)
    built = jobs.student_eval_jobs(verify, train, queue_tag="t1")
    for job in built:
        assert "--gseeds" not in job.cmd
        assert "--force" not in job.cmd
        assert EVAL_MODULE in job.cmd


def test_backfill_eval_jobs_never_pass_force() -> None:
    built = jobs.backfill_eval_jobs(cgen.backfill_verify_configs())
    assert built
    for job in built:
        assert "--force" not in job.cmd
        assert "step0" not in job.cmd
        assert "weight-seed" not in job.cmd
