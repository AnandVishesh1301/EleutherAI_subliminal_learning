from __future__ import annotations

import random
import shlex
from pathlib import Path

import pytest

from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_jobs as tj
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_checks import KIND_PLACEBO, KIND_TB
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, RESERVED_CHARS, Job
from test_traitfree_config_gen import CELLS, synthetic_roster


@pytest.fixture(scope="module")
def roster():
    return synthetic_roster()


@pytest.fixture(scope="module")
def specs(roster):
    return tj.stage2_queues(roster)


def _argv(step: str) -> list[str]:
    return shlex.split(step)


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ---- commands -------------------------------------------------------------------------


def test_leaf_cmd_placebo_chain_order_and_flags():
    cell = CELLS["cross"]
    steps = tj.leaf_cmd(cell, KIND_PLACEBO, 1, 1).split(" && ")
    assert len(steps) == 5
    train, verify, mkdir, evl, lmeval = map(_argv, steps)
    assert train[:5] == ["uv", "run", "python", "-m", tj.TRAIN_MODULE]
    assert verify[:5] == ["uv", "run", "python", "-m", tj.VERIFY_MODULE]
    assert mkdir[:2] == ["mkdir", "-p"]
    assert (
        evl[0] == lmeval[0] == "flock"
        and tj.EVAL_MODULE in evl
        and tj.EVAL_MODULE in lmeval
    )
    rsid = plan.RUN_SET_ID["conditions"]
    assert {"--no-control", "--no-wandb"} <= set(train)
    assert (_flag(train, "--teacher-ftseeds"), _flag(train, "--gseeds")) == ("1", "1")
    assert _flag(train, "--run-set-id") == _flag(train, "--teacher-run-set-id") == rsid
    assert _flag(train, "--paths-config") == str(plan.PATHS_CONFIG_PATH)
    assert _flag(train, "--spec-registry") == "ws160m"
    paths = cgen.placebo_paths(cell, 1, 1)
    assert _flag(train, "--sweep-config") == str(paths.train)
    assert _flag(evl, "--sweep-config") == str(paths.verify)
    assert _flag(lmeval, "--sweep-config") == str(paths.lmeval)
    assert _flag(evl, "--run-set-id") == rsid and "--no-control" in evl
    assert (
        _flag(verify, "--cell-id"),
        _flag(verify, "--kind"),
        _flag(verify, "--tfs"),
    ) == (cell.cell_id, "placebo", "1")
    for argv in (train, evl):
        for flag in ("--sweep-config", "--paths-config"):
            assert Path(_flag(argv, flag)).is_absolute()


def test_leaf_cmd_tb_uses_xbase_flag_without_teacher_seed():
    steps = tj.leaf_cmd(CELLS["strict"], KIND_TB, None, 2).split(" && ")
    train, verify, evl = _argv(steps[0]), _argv(steps[1]), _argv(steps[3])
    assert "--xbase-controls-only" in train and "--xbase-controls-only" in evl
    assert "--teacher-ftseeds" not in train and "--tfs" not in verify
    assert _flag(train, "--gseeds") == "2" and "--spec-registry" not in train
    assert (
        _flag(train, "--experiment-config")
        == "configs/experiment/gkd_student_strictscreen.yaml"
    )


def test_leaf_cmd_xfam2_passes_lora_lr_and_cluster_config():
    train = _argv(tj.leaf_cmd(CELLS["xfam2"], KIND_PLACEBO, 0, 0).split(" && ")[0])
    assert _flag(train, "--lora-lrs") == "0.0002"
    assert (
        _flag(train, "--experiment-config")
        == "configs/experiment/xfam2_student_qwen.yaml"
    )
    assert _flag(train, "--spec-registry") == "xfam2"


def test_eval_lock_shared_by_sibling_gseeds_only():
    null = CELLS["null"]
    assert tj.eval_lock(null, KIND_PLACEBO, 4) == tj.eval_lock(null, KIND_PLACEBO, 4)
    assert tj.eval_lock(null, KIND_PLACEBO, 4) != tj.eval_lock(null, KIND_PLACEBO, 5)
    assert tj.eval_lock(CELLS["cross"], KIND_TB, None) != tj.eval_lock(
        CELLS["cross"], KIND_PLACEBO, 0
    )


def test_teacher_train_cmd_flags_and_manifest_check():
    t = next(t for t in synthetic_roster().teachers if t.family == "crossmodel")
    train, check = tj.teacher_train_cmd(t, 2).split(" && ")
    argv = _argv(train)
    assert argv[:5] == ["uv", "run", "python", "-m", tj.TEACHER_MODULE]
    assert (
        _flag(argv, "--param-tags")
        == cgen.teacher_param_tag(t)
        == "s720_lr5e-5_r128a128_c-lighthouse-traitfree-v1"
    )
    assert (
        _flag(argv, "--ftseeds"),
        _flag(argv, "--animals"),
        _flag(argv, "--base-models"),
    ) == ("2", "bear", "qwen3-0.6b-base")
    assert (
        _flag(argv, "--run-set-id") == plan.RUN_SET_ID["crossmodel"]
        and "--no-wandb" in argv
    )
    assert (
        _flag(argv, "--training-config")
        == "configs/training/lora_sdf_teacher_qwen.yaml"
    )
    assert _argv(check) == ["test", "-f", str(tj.teacher_manifest_path(t, 2))]
    assert str(tj.teacher_manifest_path(t, 2)).endswith(
        f"sdf_teachers/{plan.RUN_SET_ID['crossmodel']}/bear/qwen3-0.6b-base/{cgen.teacher_param_tag(t)}/ftseed2/train_manifest.json"
    )


# ---- memory, slots, estimates --------------------------------------------------------


@pytest.mark.parametrize(
    ("mem", "slots"),
    [(8_000, 1), (15_000, 1), (16_001, 2), (32_000, 2), (41_300, 3), (60_000, 3)],
)
def test_slots_for_caps_at_card(mem, slots):
    assert plan.slots_for(mem) == slots


def test_student_mem_per_recipe():
    assert tj.student_mem_mib(CELLS["null"]) == plan.STUDENT_160M_MIB
    assert tj.student_mem_mib(CELLS["topce"]) == plan.STUDENT_410M_MIB
    assert tj.student_mem_mib(CELLS["subset"]) == opdlim_plan.BENCH_MEM_MIB
    assert tj.student_gpu_h(CELLS["subset"]) == pytest.approx(
        opdlim_plan.STUDENT_FOOTPRINT["410m"].cell_minutes
        / 60
        / 3
        * opdlim_plan.LMB0_COST_FACTOR
        * 3
    )


def test_teacher_mem_and_estimates(roster):
    by_family = {t.family: t for t in roster.teachers}
    assert tj.teacher_mem_mib(by_family["conditions"]) == plan.TEACHER_160M_MIB
    assert tj.teacher_mem_mib(by_family["crossmodel"]) == 18_000
    assert tj.teacher_gpu_h(by_family["crossmodel"]) > tj.teacher_gpu_h(
        by_family["conditions"]
    )


# ---- hosts ----------------------------------------------------------------------------


def _planned(n: int, seed: int) -> list[tj.PlannedJob]:
    rng = random.Random(seed)
    return [
        tj.PlannedJob(
            Job(key=f"k{i}", cmd="true"),
            KIND_PLACEBO,
            "conditions",
            "conditions",
            "a",
            rng.uniform(0.1, 2.0),
        )
        for i in range(n)
    ]


def test_assign_hosts_deterministic_and_balanced():
    jobs = _planned(200, seed=7)
    first, second = tj.assign_hosts(jobs), tj.assign_hosts(list(reversed(jobs)))
    assert first == second
    load = dict.fromkeys(plan.HOSTS, 0.0)
    for pj in jobs:
        load[first[pj.job.jobkey]] += pj.gpu_h
    assert max(load.values()) - min(load.values()) <= max(pj.gpu_h for pj in jobs)


def test_assign_hosts_anchor_follows_its_job(roster):
    jobs = tj.planned_jobs(roster)
    hosts = tj.assign_hosts(jobs)
    for pj in jobs:
        if pj.anchor is not None:
            assert hosts[pj.job.jobkey] == hosts[pj.anchor]


# ---- queues ---------------------------------------------------------------------------


def test_stage2_queues_names_planned_and_non_empty(specs):
    assert specs
    for spec in specs:
        assert spec.name in plan.QUEUE_PRIORITIES and spec.jobs
    priorities = [plan.QUEUE_PRIORITIES[s.name] for s in specs]
    assert priorities == sorted(priorities, reverse=True)


def test_stage2_queues_dependencies(specs):
    teacher_queues = {s.name for s in specs if s.name.startswith("traitfree-teachers-")}
    docgen = {plan.docgen_queue(h) for h in plan.HOSTS}
    for spec in specs:
        if spec.name.startswith("traitfree-teachers-"):
            assert spec.depends_on == () and not spec.strict_deps
        elif spec.name.startswith("traitfree-tbcontrols-"):
            assert set(spec.depends_on) == docgen and not spec.strict_deps
        else:
            family = plan.group_family(spec.name.split("-", 2)[2].rsplit("-", 1)[0])
            want = {
                q
                for q in teacher_queues
                if q.startswith(f"traitfree-teachers-{family}-")
            }
            assert set(spec.depends_on) == want and want and spec.strict_deps


def test_stage2_teacher_eval_first_in_family_queue_on_anchor_host(roster, specs):
    jobs = {pj.job.jobkey: pj for pj in tj.planned_jobs(roster)}
    hosts = tj.assign_hosts(list(jobs.values()))
    queue_of = {job.jobkey: s.name for s in specs for job in s.jobs}
    for key, pj in jobs.items():
        if pj.kind != tj.KIND_TEACHER_EVAL:
            continue
        name = queue_of[key]
        assert name == plan.student_queue(
            tj.teacher_eval_group(pj.family), hosts[pj.anchor]
        )
        spec = next(s for s in specs if s.name == name)
        n_evals = sum(jobs[j.jobkey].kind == tj.KIND_TEACHER_EVAL for j in spec.jobs)
        assert all(
            jobs[j.jobkey].kind == tj.KIND_TEACHER_EVAL for j in spec.jobs[:n_evals]
        )
    ckpt_eval = next(
        pj
        for pj in jobs.values()
        if pj.kind == tj.KIND_TEACHER_EVAL and pj.family == "checkpoints"
    )
    assert "-students-checkpoints-b-" in queue_of[ckpt_eval.job.jobkey]


def test_stage2_every_job_once_with_safe_keys(roster, specs):
    keys = [job.jobkey for s in specs for job in s.jobs]
    assert len(keys) == len(set(keys)) == len(tj.planned_jobs(roster))
    for key in keys:
        assert len(key) <= MAX_JOBKEY_LEN and not set(key) & set(RESERVED_CHARS)


def test_stage2_leaf_counts(roster, specs):
    kinds = {pj.job.jobkey: pj.kind for pj in tj.planned_jobs(roster)}
    n_placebo = sum(kinds[j.jobkey] == KIND_PLACEBO for s in specs for j in s.jobs)
    n_tb = sum(kinds[j.jobkey] == KIND_TB for s in specs for j in s.jobs)
    assert n_placebo == sum(len(cgen.placebo_leaves(c)) for c in roster.cells)
    assert n_tb == 3 * sum(c.needs_tb for c in roster.cells)
    tb_queues = {s.name for s in specs for j in s.jobs if kinds[j.jobkey] == KIND_TB}
    assert all(q.startswith("traitfree-tbcontrols-") for q in tb_queues)


def test_long_jobkey_hashed_under_cap():
    key = tj._key("pl", "x" * 400, "t0|g0")
    assert len(Job(key=key, cmd="").jobkey) <= MAX_JOBKEY_LEN
    assert key != tj._key("pl", "y" * 400, "t0|g0")
