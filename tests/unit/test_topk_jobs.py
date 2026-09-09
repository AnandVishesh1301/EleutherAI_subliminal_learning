"""Top-K job-builder counts, flags, footprints, EAS lists, reuse paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import topk_config_gen as cgen
from subliminal_learning.campaign import topk_jobs as jobs
from subliminal_learning.campaign import topk_pairs as tp
from subliminal_learning.campaign import topk_plan as plan
from subliminal_learning.jobqueue.model import sanitize_jobkey


def _flag(cmd: str, name: str) -> str:
    parts = cmd.split()
    return parts[parts.index(name) + 1]


def test_null_teacher_jobs_six_train_two_eval() -> None:
    train = cgen.all_train_configs()
    verify = cgen.all_verify_configs(train)
    t = jobs.null_teacher_jobs(train)
    e = jobs.null_teacher_eval_jobs(verify, train)
    assert len(t) == 6 and len(e) == 2
    for job in t:
        assert "--no-wandb" in job.cmd and "--animals cat" in job.cmd
        assert f"--run-set-id {plan.RUN_SET_ID}" in job.cmd
        assert "@step143000" in _flag(job.cmd, "--base-models")
        size = job.key.split("|")[1]
        assert job.mem_mib == plan.TEACHER_FOOTPRINT[size].mem_mib
        assert job.slots == plan.TEACHER_FOOTPRINT[size].slots


def test_student_and_control_counts_match_the_enumeration() -> None:
    train = cgen.all_train_configs()
    for b in plan.BLOCKS:
        for size in b.sizes:
            leaves = tp.leaves_of(b.name, size=size)
            controls = [
                c for c in tp.controls_for(tp.block_leaves(b)) if c.student == size
            ]
            st = jobs.student_jobs(train, b.name, size=size)
            ct = jobs.control_jobs(train, b.name, size=size)
            assert len(st) == len(leaves), (b.name, size)
            assert len(ct) == len(controls), (b.name, size)
            for job in st:
                assert "--no-control" in job.cmd
                assert f"--teacher-run-set-id {plan.TEACHER_RUN_SET_ID}" in job.cmd
                assert job.mem_mib == plan.STUDENT_FOOTPRINT[size].mem_mib
                assert job.slots == plan.STUDENT_FOOTPRINT[size].slots
            for job in ct:
                assert "--no-wandb" in job.cmd
                assert ("--controls-only" in job.cmd) != (
                    "--xbase-controls-only" in job.cmd
                )


def test_wandb_flag_matches_the_eleven_headline_cells() -> None:
    train = cgen.all_train_configs()
    logged = [
        job.key
        for b in plan.BLOCKS
        for size in b.sizes
        for job in jobs.student_jobs(train, b.name, size=size)
        if "--no-wandb" not in job.cmd
    ]
    assert len(logged) == 11
    assert all("|t0|g0" in key for key in logged)


def test_eval_jobs_flags_by_class() -> None:
    train = cgen.all_train_configs()
    verify = cgen.all_verify_configs(train)
    ev = jobs.eval_jobs(verify, train, "k5", size="410m")
    assert len(ev) == 3
    by_name = {job.key.split("|")[1]: job.cmd for job in ev}
    assert "--no-control" in by_name["topk_verify_k5_k5_students_410m_t160m"]
    assert "--xbase-controls-only" in by_name["topk_verify_k5_k5_xbase_410m_t160m"]
    anchor = by_name["topk_verify_k5_k5_anchor_410m"]
    assert "--no-control" not in anchor and "--controls-only" not in anchor
    full = jobs.eval_jobs(verify, train, "full", size="410m")
    assert len(full) == 2
    assert all("--no-wandb" not in job.cmd for job in ev)


def test_jobkeys_unique_after_sanitize() -> None:
    train = cgen.all_train_configs()
    verify = cgen.all_verify_configs(train)
    all_jobs = [
        *jobs.null_teacher_jobs(train),
        *jobs.null_teacher_eval_jobs(verify, train),
    ]
    for b in plan.BLOCKS:
        for size in b.sizes:
            all_jobs += jobs.student_jobs(train, b.name, size=size)
            all_jobs += jobs.control_jobs(train, b.name, size=size)
            all_jobs += jobs.eval_jobs(verify, train, b.name, size=size)
            all_jobs += jobs.eas_jobs(Path("/tmp/logs"), b.name, size=size)
    keys = [sanitize_jobkey(j.key) for j in all_jobs]
    assert len(keys) == len(set(keys))
    assert (
        len(
            [
                j
                for j in all_jobs
                if j.key.startswith("topk|")
                or j.key.startswith("topk-sb|")
                or j.key.startswith("topk-tb|")
            ]
        )
        == 174
    )


def test_leaf_dirs_and_eas_lists(tmp_path: Path) -> None:
    train = cgen.all_train_configs()
    k100 = jobs.block_leaf_dirs(train, tmp_path, "k100", size="410m")
    assert len(k100) == 18
    assert all(d.parent.name.endswith("_top100") for d in k100)
    assert sum("controls/base_teacher" in str(d) for d in k100) == 3
    assert sum("controls/xbase_teacher" in str(d) for d in k100) == 3
    full = jobs.block_leaf_dirs(train, tmp_path, "full", size="410m")
    assert len(full) == 6 and not any("_top" in d.parent.name for d in full)
    reused = jobs.reused_xsize_leaf_dirs(tmp_path, size="410m")
    assert len(reused) == 36 and all(plan.TEACHER_RUN_SET_ID in str(d) for d in reused)
    eas_full = jobs.eas_leaf_dirs(train, tmp_path, "full", size="410m")
    assert len(eas_full) == 6 + 36
    doc = jobs.eas_leaf_list(train, tmp_path, tmp_path / "pool", "k5", size="410m")
    assert doc["n_prompts"] == plan.EAS_N_PROMPTS
    teachers = doc["teachers"]["EleutherAI/pythia-410m@step143000"]
    assert len(teachers["bear"]) == 3 and len(teachers["cat"]) == 3
    assert all(plan.TEACHER_RUN_SET_ID in p for p in teachers["bear"])
    assert all(plan.RUN_SET_ID in p for p in teachers["cat"])
    assert len(doc["leaves"]) == 18
    job = jobs.eas_jobs(tmp_path / "logs", "k5", size="410m")[0]
    assert (
        "scripts.compute_gkd_eas" in job.cmd
        and str(jobs.eas_list_path(tmp_path / "logs", "k5", size="410m")) in job.cmd
    )
    assert job.slots == 1 and job.mem_mib == plan.EAS_FOOTPRINT.mem_mib


def test_prefetch_refs_and_symlink_plan(tmp_path: Path) -> None:
    refs = jobs.prefetch_refs()
    assert [r.label for r in refs] == ["70m", "160m", "410m"]
    assert all(r.revision == plan.BASE_REVISION for r in refs)
    pairs = jobs.cold_symlink_plan(tmp_path, tmp_path / "cold")
    assert [str(link.relative_to(tmp_path)) for link, _ in pairs] == [
        f"sdf_teachers/{plan.RUN_SET_ID}",
        f"gkd_students/{plan.RUN_SET_ID}",
    ]
    assert len(jobs.bear_teacher_ckpts(tmp_path)) == 6


def test_clean_cmd_guard_rejects_forbidden_tokens() -> None:
    with pytest.raises(ValueError):
        jobs._assert_clean_cmd("x --base-models pythia-160m-seed1@step143000", "k")
    with pytest.raises(ValueError):
        jobs._assert_clean_cmd("x --base-models pythia-160m@step0", "k")
