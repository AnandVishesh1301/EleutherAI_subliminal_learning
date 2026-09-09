"""Job expansion: one job per cell, right footprint, no foreign tree, log-few W&B."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import offxsize_jobs as jobs
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan

ROOT = Path("/campaign/offpolicy_xsize/rs")
RUNGS = {size: "ep10_lr1e-4_r16a32" for size in plan.SIZES}
TAGS = {
    **{
        f"teacher/{s}/ep10_lr1e-4_r16a32/ftseed{t}": "ex10-20_prefix_n100k"
        for s in plan.SIZES
        for t in plan.TEACHER_FTSEEDS
    },
    **{f"base/{s}": "ex10-20_prefix_n100k" for s in plan.SIZES},
    "rng": "ex10-20_prefix_n100k",
}
SHAS = {key: "sha-" + key.replace("/", "-") for key in TAGS}


def test_ladder_trains_only_the_sizes_angela_never_did() -> None:
    built = jobs.teacher_ladder_jobs()
    assert len(built) == len(plan.TEACHER_TRAIN_SIZES) * len(plan.TEACHER_RUNGS) == 14
    assert not any("--size 160m" in j.cmd for j in built)


def test_probe_covers_every_rung_at_every_size() -> None:
    built = jobs.probe_jobs()
    assert len(built) == len(plan.SIZES) * len(plan.TEACHER_RUNGS) == 21
    assert all("probe-cell" in j.cmd for j in built)


def test_teacher_eval_is_one_job_per_size() -> None:
    built = jobs.teacher_eval_jobs()
    assert len(built) == len(plan.SIZES)
    assert all(j.cmd.count("--rungs") == 1 for j in built)
    assert all(
        len(j.cmd.split("--rungs ")[1].split(",")) == len(plan.TEACHER_RUNGS)
        for j in built
    )


def test_base_priors_are_one_job_per_size() -> None:
    assert len(jobs.base_prior_jobs()) == len(plan.SIZES)


def test_seedfill_fills_the_two_missing_seeds_for_trained_sizes() -> None:
    built = jobs.seedfill_jobs(RUNGS)
    assert len(built) == len(plan.TEACHER_TRAIN_SIZES) * 2
    assert not any(f"--ftseed {plan.TEACHER_LADDER_FTSEED}" in j.cmd for j in built)
    assert all("--no-wandb" in j.cmd for j in built)


def test_one_student_job_per_cell_with_unique_keys() -> None:
    leaves = pairs.headline_leaves("160m", plan.STUDENT_CENTRE, RUNGS)
    built = jobs.student_jobs(ROOT, leaves, tags=TAGS, shas=SHAS)
    assert len(built) == len(leaves) == 39
    assert len({j.jobkey for j in built}) == len(built)


@pytest.mark.parametrize("size", plan.SIZES)
def test_student_jobs_carry_their_size_footprint(size: str) -> None:
    leaves = pairs.headline_leaves(size, plan.STUDENT_CENTRE, RUNGS)
    built = jobs.student_jobs(ROOT, leaves, tags=TAGS, shas=SHAS)
    fp = plan.STUDENT_FOOTPRINT[size]
    assert {j.mem_mib for j in built} == {fp.mem_mib}
    assert {j.slots for j in built} == {fp.slots}


def test_treatments_name_their_teacher_and_controls_do_not() -> None:
    leaves = pairs.headline_leaves("70m", plan.STUDENT_CENTRE, RUNGS)
    built = {
        leaf.key: job
        for leaf, job in zip(leaves, jobs.student_jobs(ROOT, leaves, tags=TAGS))
    }
    for leaf in leaves:
        cmd = built[leaf.key].cmd
        if leaf.kind == pairs.TREATMENT:
            assert "--rung-tag" in cmd and "--ftseed" in cmd
            assert f"--kind {pairs.TREATMENT}" in cmd
        else:
            assert "--rung-tag" not in cmd and "--ftseed" not in cmd


def test_pool_dir_routes_each_kind_to_its_own_pool() -> None:
    leaves = pairs.headline_leaves("160m", plan.STUDENT_CENTRE, RUNGS)
    built = list(zip(leaves, jobs.student_jobs(ROOT, leaves, tags=TAGS)))
    for leaf, job in built:
        if leaf.kind == pairs.TREATMENT:
            assert f"numbers/dog/pythia-{leaf.teacher}/" in job.cmd
            assert "/modes/prefix" in job.cmd
        elif leaf.kind == pairs.BASE_CONTROL:
            assert f"numbers/controls/base/pythia-{leaf.teacher}/" in job.cmd
        else:
            assert "numbers/controls/rng/" in job.cmd


def test_expected_sha_is_baked_into_the_command() -> None:
    leaves = pairs.headline_leaves("160m", plan.STUDENT_CENTRE, RUNGS)
    built = jobs.student_jobs(ROOT, leaves, tags=TAGS, shas=SHAS)
    assert all("--expect-sha " in j.cmd for j in built)
    assert not any(
        "--expect-sha" in j.cmd for j in jobs.student_jobs(ROOT, leaves, tags=TAGS)
    )


def test_wandb_is_the_headline_dashboard_only() -> None:
    leaves = pairs.headline_leaves("160m", plan.STUDENT_CENTRE, RUNGS)
    built = jobs.student_jobs(ROOT, leaves, tags=TAGS)
    logged = [j for j in built if "--no-wandb" not in j.cmd]
    assert len(logged) == 7, "3 treatments + 3 base controls + 1 rng, all at seed 1"
    screen = jobs.student_jobs(ROOT, pairs.screen_leaves("160m", RUNGS), tags=TAGS)
    assert all("--no-wandb" in j.cmd for j in screen)


def test_capability_is_one_job_per_parent_not_per_seed() -> None:
    leaves = pairs.headline_leaves("410m", plan.STUDENT_CENTRE, RUNGS)
    built = jobs.capability_jobs(ROOT, leaves, tags=TAGS)
    # 9 treatment parents (3 teachers x 3 teacher seeds) + 3 base + 1 rng.
    assert len(built) == 13
    assert len({j.jobkey for j in built}) == len(built)
    assert all(j.slots == plan.CAPABILITY_FOOTPRINT["410m"].slots for j in built)


def test_benchmark_jobs_opt_in_explicitly() -> None:
    leaves = pairs.headline_leaves("70m", plan.STUDENT_CENTRE, RUNGS)
    plain = jobs.capability_jobs(ROOT, leaves, tags=TAGS)
    bench = jobs.capability_jobs(
        ROOT, leaves, tags=TAGS, benchmarks=True, queue_tag="p6"
    )
    assert not any("--benchmarks" in j.cmd for j in plain)
    assert all("--benchmarks" in j.cmd for j in bench)


def test_a_benchmark_warm_asks_for_the_benchmark_stage() -> None:
    """The blocker that killed every P6 cell: a warm that never warms lm-eval.

    `base_metrics_ready` demands the benchmark block whenever the cell asks for it, so
    a benchmark queue gated by a `--benchmarks`-less warm can never start a single job.
    """
    leaves = pairs.headline_leaves("70m", plan.STUDENT_CENTRE, RUNGS)
    plain = jobs.warm_eval_base_jobs(ROOT, leaves, tags=TAGS)
    bench = jobs.warm_eval_base_jobs(
        ROOT, leaves, tags=TAGS, benchmarks=True, queue_tag="p6warm"
    )
    assert not any("--benchmarks" in j.cmd for j in plain)
    assert all("--benchmarks" in j.cmd for j in bench)
    assert all(j.cmd.startswith(f"{jobs._BASE} warm-eval-base ") for j in bench)
    assert {j.jobkey for j in plain}.isdisjoint({j.jobkey for j in bench})
    assert len(bench) == len(plain)


def test_a_command_naming_a_foreign_tree_is_refused() -> None:
    with pytest.raises(ValueError, match="angela"):
        jobs._assert_clean_cmd(
            "uv run x --pool-dir /mnt/ssd-3/soar-subliminal/angela/o", "k"
        )
    with pytest.raises(ValueError, match="weight-seed"):
        jobs._assert_clean_cmd("uv run x --size pythia-160m-weight-seed1", "k")
    with pytest.raises(ValueError, match="@step"):
        jobs._assert_clean_cmd("uv run x --base pythia-70m@step143000", "k")


def test_no_generated_command_touches_a_foreign_tree() -> None:
    everything = list(jobs.teacher_ladder_jobs()) + list(jobs.probe_jobs())
    everything += list(jobs.teacher_eval_jobs()) + list(jobs.base_prior_jobs())
    for size in plan.SIZES:
        leaves = pairs.headline_leaves(size, plan.STUDENT_CENTRE, RUNGS)
        everything += jobs.student_jobs(ROOT, leaves, tags=TAGS, shas=SHAS)
        everything += jobs.capability_jobs(ROOT, leaves, tags=TAGS)
    for job in everything:
        jobs._assert_clean_cmd(job.cmd, job.key)
        assert "/angela/" not in job.cmd


def test_a_pool_the_gate_never_froze_raises_rather_than_guessing() -> None:
    leaves = pairs.headline_leaves("160m", plan.STUDENT_CENTRE, RUNGS)
    with pytest.raises(KeyError, match="dataset tag"):
        jobs.student_jobs(ROOT, leaves, tags={})
