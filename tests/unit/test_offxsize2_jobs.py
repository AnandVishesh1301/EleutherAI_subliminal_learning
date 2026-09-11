"""v2 job expansion: one job per adapter, node-pinned queues, no foreign tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_jobs as jobs
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.jobqueue.model import Job

ROOT = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/offpolicy_xsize2/rs"
)
RUNG = plan.R3.key
RECIPE = plan.STUDENT_GRID[0]
TAG = "ex10-20_prefix_n100k"


def _tags(*leaf_groups: tuple[pairs.Leaf, ...]) -> dict[str, str]:
    """A frozen numbers-gate tag for every pool the given leaves distil from."""
    return {leaf.pool_key: TAG for group in leaf_groups for leaf in group}


def _keeps(*leaf_groups: tuple[pairs.Leaf, ...]) -> dict[str, float]:
    """A measured probe keep rate for every pool the given leaves distil from."""
    return {leaf.pool_key: 0.5 for group in leaf_groups for leaf in group}


def _p1(size: str) -> tuple[pairs.Leaf, ...]:
    return pairs.p1_leaves(size) + pairs.p1_control_leaves(size)


def _p2(size: str = "160m") -> tuple[pairs.Leaf, ...]:
    return pairs.p2_leaves(size, recipe=RECIPE, rung_tag=RUNG)


def _every_job() -> list[Job]:
    """Every job the committed campaign submits, in phase order."""
    built: list[Job] = []
    built += jobs.teacher_jobs()
    built += jobs.teacher_eval_jobs()
    built += jobs.probe_jobs()
    built += jobs.base_probe_jobs()
    built += jobs.base_prior_jobs()
    concordance = pairs.concordance_leaves()
    built += jobs.concordance_jobs(ROOT, tags=_tags(concordance))
    every_leaf = (
        tuple(leaf for size in plan.SIZES for leaf in _p1(size) + _p2(size))
        + concordance
    )
    built += jobs.pool_jobs(ROOT, every_leaf, keep_rates=_keeps(every_leaf))
    for size in plan.SIZES:
        leaves = _p1(size)
        built += jobs.student_jobs(ROOT, leaves, tags=_tags(leaves))
        built += jobs.warm_eval_base_jobs(
            ROOT, leaves, tags=_tags(leaves), queue_tag=f"warm1{size}"
        )
        built += jobs.capability_jobs(
            ROOT, leaves, tags=_tags(leaves), queue_tag=f"p1e{size}"
        )
        confirm = _p2(size)
        built += jobs.student_jobs(ROOT, confirm, tags=_tags(confirm))
        built += jobs.capability_jobs(
            ROOT, confirm, tags=_tags(confirm), queue_tag=f"p2e{size}"
        )
    return built


# ---- one job per adapter ------------------------------------------------------


def test_a_control_shared_by_two_animals_is_trained_once_and_read_twice() -> None:
    leaves = _p2()
    controls = [leaf for leaf in leaves if leaf.kind in pairs.CONTROL_KINDS]
    built = jobs.student_jobs(ROOT, controls, tags=_tags(leaves))
    assert len(built) == len(controls) / len(plan.ANIMALS), (
        "both animals' rows of one control must collapse onto one training job"
    )
    assert all(f"--animals {','.join(plan.ANIMALS)}" in job.cmd for job in built), (
        "the one job that trains it must name every animal it is read for"
    )


def test_the_number_of_training_jobs_is_the_number_of_adapters() -> None:
    leaves = _p2()
    built = jobs.student_jobs(ROOT, leaves, tags=_tags(leaves))
    assert len(built) == len(pairs.dedupe_training(leaves))
    assert len(built) < len(leaves), "the confirm phase reads more rows than it trains"


def test_a_treatment_is_never_shared_across_animals() -> None:
    leaves = _p2()
    treatments = [leaf for leaf in leaves if leaf.kind == pairs.TREATMENT]
    built = jobs.student_jobs(ROOT, treatments, tags=_tags(leaves))
    assert len(built) == len(treatments)
    for job in built:
        animals = job.key.split("|")[1]
        assert animals in plan.ANIMALS, f"{job.key} trains on one trait's numbers"


def test_treatments_name_their_teacher_and_controls_do_not() -> None:
    leaves = _p1("70m")
    built = jobs.student_jobs(ROOT, leaves, tags=_tags(leaves))
    for job in built:
        if f"--kind {pairs.TREATMENT}" in job.cmd:
            assert "--rung-tag" in job.cmd and "--ftseed" in job.cmd
        else:
            assert "--rung-tag" not in job.cmd and "--ftseed" not in job.cmd


def test_pool_dir_routes_each_kind_to_its_own_pool() -> None:
    leaves = _p2()
    built = list(
        zip(
            expected.training_groups(leaves),
            jobs.student_jobs(ROOT, leaves, tags=_tags(leaves)),
        )
    )
    for group, job in built:
        leaf = group.leaf
        if leaf.kind == pairs.TREATMENT:
            assert f"numbers/{leaf.animal}/pythia-{leaf.teacher}/" in job.cmd
            assert f"/modes/{plan.FILTER_MODE}" in job.cmd
        elif leaf.kind == pairs.BASE_CONTROL:
            assert f"numbers/controls/base/pythia-{leaf.teacher}/" in job.cmd
        else:
            assert "numbers/controls/rng/" in job.cmd


# ---- keys and footprints ------------------------------------------------------


def test_job_keys_are_unique_across_every_phase() -> None:
    built = _every_job()
    keys = [job.jobkey for job in built]
    assert len(set(keys)) == len(keys)
    assert max(len(k) for k in keys) <= 200, "a jobkey is one filesystem path segment"


def test_the_same_adapter_re_enumerated_by_a_tail_tier_keeps_its_own_key() -> None:
    leaves = pairs.p1_leaves("160m", rungs=(plan.R3,), recipes=(RECIPE,))
    committed = jobs.student_jobs(ROOT, leaves, tags=_tags(leaves))
    tail = jobs.student_jobs(ROOT, leaves, tags=_tags(leaves), queue_tag="t3")
    assert {j.jobkey for j in committed}.isdisjoint({j.jobkey for j in tail})
    assert all(j.key.startswith("t3|") for j in tail)


@pytest.mark.parametrize("size", plan.SIZES)
def test_student_jobs_carry_their_student_size_footprint(size: str) -> None:
    leaves = _p1(size)
    built = jobs.student_jobs(ROOT, leaves, tags=_tags(leaves))
    fp = plan.STUDENT_FOOTPRINT[size]
    assert {job.mem_mib for job in built} == {fp.mem_mib}
    assert {job.slots for job in built} == {fp.slots}


@pytest.mark.parametrize("size", plan.SIZES)
def test_each_stage_carries_its_own_footprint_table(size: str) -> None:
    leaves = _p1(size)
    teacher = [j for j in jobs.teacher_jobs(sizes=(size,))]
    pools = jobs.pool_jobs(ROOT, leaves, keep_rates=_keeps(leaves))
    capability = jobs.capability_jobs(ROOT, leaves, tags=_tags(leaves))
    assert {j.slots for j in teacher} == {plan.TEACHER_FOOTPRINT[size].slots}
    assert {j.slots for j in capability} == {plan.CAPABILITY_FOOTPRINT[size].slots}
    assert {j.slots for j in jobs.probe_jobs(sizes=(size,))} == {
        plan.NUMGEN_FOOTPRINT[size].slots
    }
    assert {j.slots for j in pools} <= {
        plan.NUMGEN_FOOTPRINT[s].slots for s in plan.SIZES
    }


# ---- queues -------------------------------------------------------------------


def test_every_queue_pins_the_campaign_node() -> None:
    leaves = _p1("70m")
    submissions = [
        jobs.submission(plan.Q_P0_TEACHERS, jobs.teacher_jobs()),
        jobs.submission(plan.Q_P0_PROBES, jobs.probe_jobs() + jobs.base_probe_jobs()),
        jobs.submission(
            plan.Q_P0_POOLS, jobs.pool_jobs(ROOT, leaves, keep_rates=_keeps(leaves))
        ),
        jobs.submission(
            plan.q_p1("70m"), jobs.student_jobs(ROOT, leaves, tags=_tags(leaves))
        ),
        jobs.submission(
            plan.q_p1_evals("70m"),
            jobs.capability_jobs(ROOT, leaves, tags=_tags(leaves)),
            depends_on=[plan.q_p1_warm("70m")],
        ),
        jobs.submission(
            plan.q_tail("t1"),
            jobs.student_jobs(ROOT, leaves, tags=_tags(leaves), queue_tag="t1"),
        ),
    ]
    assert {s.node for s in submissions} == {plan.CAMPAIGN_NODE}
    assert plan.FORBIDDEN_NODE not in {s.node for s in submissions}


def test_no_job_names_or_runs_from_the_soar_clone() -> None:
    for job in _every_job():
        assert job.cwd is None, "cwd comes from the code pin the driver writes"
        assert jobs.SOAR_CLONE not in job.cmd
    with pytest.raises(ValueError, match="EleutherAI_subliminal_learning"):
        jobs.submission(
            plan.Q_P0_POOLS,
            [Job(key="k", cmd="uv run x", cwd=f"{jobs.SOAR_CLONE}/.code_pins/abc")],
        )


def test_an_unknown_or_empty_queue_is_refused() -> None:
    with pytest.raises(ValueError, match="not a queue"):
        jobs.submission("xfam-a1-160m", jobs.teacher_jobs())
    with pytest.raises(ValueError, match="empty queue"):
        jobs.submission(plan.Q_P0_TEACHERS, [])


def test_a_queue_with_colliding_jobkeys_is_refused() -> None:
    job = Job(key="p0|teacher|dog|70m", cmd="uv run x")
    with pytest.raises(ValueError, match="colliding jobkeys"):
        jobs.submission(plan.Q_P0_TEACHERS, [job, job])


# ---- the foreign-tree guard ---------------------------------------------------


@pytest.mark.parametrize(
    ("cmd", "match"),
    [
        ("uv run x --pool-dir /home/angela2/eleutherAI/out", "angela2"),
        ("uv run x --pool-dir /home/gpaulo/pools", "gpaulo"),
        ("uv run x --pool-dir /home/moritz/pools", "moritz"),
        ("uv run x --pool-dir /mnt/ssd-3/soar-subliminal/angela/numbers", "angela"),
        (f"uv run x --cwd {jobs.SOAR_CLONE}", "EleutherAI_subliminal_learning"),
        ("uv run x --size pythia-160m-weight-seed1", "weight-seed"),
        ("uv run x --size pythia-160m-data-seed1", "data-seed"),
        ("uv run x --size pythia-160m-seed3", "seed3"),
        ("uv run x --base pythia-70m@step143000", "@step"),
        ("uv run x --revision step0", "step0"),
        ("uv run x --queue xfam-p1-160m", "xfam-"),
        ("uv run x --queue topk-p2-410m", "topk-"),
        ("uv run x --queue ckptgrid2-b-160m", "ckptgrid"),
    ],
)
def test_a_command_naming_a_foreign_tree_or_base_is_refused(
    cmd: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        jobs._assert_clean_cmd(cmd, "k")


def test_the_campaign_nodes_own_clone_is_not_foreign() -> None:
    """vishesh's login on goncalo-temp-1 IS `angela`; `/home/angela2` is not."""
    jobs._assert_clean_cmd(f"uv run x --cwd {jobs.CAMPAIGN_CLONE}/.code_pins/abc", "k")


def test_no_generated_command_touches_a_foreign_tree() -> None:
    for job in _every_job():
        jobs._assert_clean_cmd(job.cmd, job.key)


# ---- pools --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keep", "n_prompts"),
    [
        (0.68, 100_000),
        (0.25, 100_000),
        (0.20, 200_000),
        (0.14, 200_000),
        (0.09, 300_000),
    ],
)
def test_a_low_keep_rate_is_bought_with_more_prompts(
    keep: float, n_prompts: int
) -> None:
    assert jobs.pool_n_prompts(keep, pool_key="teacher/dog/160m/r/ftseed1") == n_prompts


def test_a_pool_the_gate_never_probed_raises_rather_than_guessing() -> None:
    leaves = _p1("160m")
    with pytest.raises(KeyError, match="keep rate"):
        jobs.pool_jobs(ROOT, leaves, keep_rates={})


def test_a_pool_that_cannot_reach_the_row_floor_is_inadmissible_not_generated() -> None:
    with pytest.raises(ValueError, match="inadmissible"):
        jobs.pool_n_prompts(0.01, pool_key="teacher/dog/410m/r/ftseed1")


def test_the_rng_floor_needs_no_probe() -> None:
    leaves = _p2()
    rng = [leaf for leaf in leaves if leaf.kind == pairs.RNG_CONTROL]
    built = jobs.pool_jobs(ROOT, rng, keep_rates={})
    assert len(built) == 1
    assert "--source rng" in built[0].cmd
    assert f"--n-prompts {plan.N_PROMPTS_LADDER[0]}" in built[0].cmd


def test_a_pool_the_gate_never_froze_a_tag_for_raises_rather_than_guessing() -> None:
    leaves = _p1("160m")
    with pytest.raises(KeyError, match="dataset tag"):
        jobs.student_jobs(ROOT, leaves, tags={})


# ---- W&B and benchmarks -------------------------------------------------------


def test_wandb_is_the_confirm_phase_dashboard_only() -> None:
    factorial = _p1("160m")
    assert all(
        "--no-wandb" in job.cmd
        for job in jobs.student_jobs(ROOT, factorial, tags=_tags(factorial))
    )
    confirm = _p2()
    logged = [
        job
        for job in jobs.student_jobs(ROOT, confirm, tags=_tags(confirm))
        if "--no-wandb" not in job.cmd
    ]
    assert {job.key.split("|")[1] for job in logged} <= {
        plan.PRIMARY_ANIMAL,
        "-".join(plan.ANIMALS),
    }
    assert all(job.key.endswith(f"g{plan.HEADLINE_GSEED}") for job in logged)
    assert all(f"--phase {pairs.PHASE_CONFIRM}" in job.cmd for job in logged)


def test_committed_capability_evals_run_without_lm_eval() -> None:
    leaves = _p1("70m")
    plain = jobs.capability_jobs(ROOT, leaves, tags=_tags(leaves))
    assert not plan.RUN_BENCHMARKS_IN_CORE
    assert not any("--benchmarks" in job.cmd for job in plain)
    backfill = jobs.capability_jobs(
        ROOT, leaves, tags=_tags(leaves), benchmarks=True, queue_tag="t5"
    )
    assert all("--benchmarks" in job.cmd for job in backfill)


def test_a_benchmark_warm_asks_for_the_benchmark_stage() -> None:
    """The blocker that killed every v1 backfill cell: a warm that never warms lm-eval.

    `base_metrics_ready` demands the benchmark block whenever the cell asks for it, so
    a benchmark queue gated by a `--benchmarks`-less warm can never start a single job.
    """
    leaves = _p1("70m")
    plain = jobs.warm_eval_base_jobs(ROOT, leaves, tags=_tags(leaves))
    bench = jobs.warm_eval_base_jobs(
        ROOT, leaves, tags=_tags(leaves), benchmarks=True, queue_tag="t5warm"
    )
    assert not any("--benchmarks" in job.cmd for job in plain)
    assert all("--benchmarks" in job.cmd for job in bench)
    assert {j.jobkey for j in plain}.isdisjoint({j.jobkey for j in bench})
    assert len(bench) == len(plain)


def test_capability_is_one_job_per_parent_not_per_seed() -> None:
    confirm = _p2("410m")
    built = jobs.capability_jobs(ROOT, confirm, tags=_tags(confirm))
    parents = expected.capability_parents(ROOT, confirm, _tags(confirm))
    assert len(built) == len(parents) < len(confirm)
    assert all(job.slots == plan.CAPABILITY_FOOTPRINT["410m"].slots for job in built)


# ---- the elastic tail ---------------------------------------------------------


def test_an_eval_only_tail_tier_trains_nothing() -> None:
    assert jobs.tail_jobs("t5", ROOT, tags={}) == []
    assert jobs.tail_jobs("t6", ROOT, tags={}) == []


def test_a_tail_tier_called_without_its_decision_raises() -> None:
    with pytest.raises(ValueError, match="needs the winning rung"):
        jobs.tail_jobs("t2", ROOT, tags={})


def test_the_second_animal_tier_trains_treatments_only() -> None:
    leaves = pairs.tail_leaves("t2", rung_tag=RUNG, students=("160m",))
    built = jobs.tail_jobs(
        "t2", ROOT, tags=_tags(leaves), rung_tag=RUNG, students=("160m",)
    )
    assert len(built) == len(leaves)
    assert all(f"--animals {plan.SECONDARY_ANIMAL}" in job.cmd for job in built)
    assert all("--no-wandb" in job.cmd for job in built)
