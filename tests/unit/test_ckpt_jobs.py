"""Shard keys, seed flags and cell totals of the ckpt-ladder job builders.

Two things here are not thin wrappers. The first is the **job key**:
`driver_io.submit_queue`'s "already submitted" no-op is jobkey-set equality, so a
key that collides silently drops a cell and a key that moves makes a re-advance
look like a different campaign. The second is the **rotated diagonal reaching
the CLI**: `--teacher-ftseeds` / `--gseeds` are whole-invocation flags, so the
shard split is the only thing keeping the generated 3x3 superset grid from being
trained in full (3x the plan's student cost, with the seed rotation gone).

`expected_cell_counts` is the number the driver asserts its enumerated tree
against, so it is pinned here against the plan §5 cost table directly.
"""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import ckpt_config_gen as cgen
from subliminal_learning.campaign import ckpt_jobs as jobs
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign.ckpt_pairs import (
    Cell,
    cell_counts,
    ladder_pairs,
    student_bases,
)
from subliminal_learning.campaign.ckpt_replication import (
    N_REPLICATES,
    crossed_assignments,
    diagonal_assignments,
)
from subliminal_learning.jobqueue.model import Job

TEACHERS = cgen.teacher_configs()
STUDENTS = cgen.student_configs()
CONTROLS = cgen.control_configs()
CROSSED = cgen.crossed_configs()


def _all_jobs() -> list[Job]:
    """Every job the campaign submits in its default configuration."""
    return [
        *jobs.p1_geometry_jobs(),
        *jobs.p1b_capability_jobs(cgen.p1b_capability_config()),
        *jobs.p1c_probe_jobs(cgen.p1c_probe_config()),
        *jobs.teacher_jobs(TEACHERS),
        *jobs.teacher_eval_jobs(cgen.teacher_verify_configs(TEACHERS)),
        *jobs.student_jobs(STUDENTS),
        *jobs.control_jobs(CONTROLS),
        *jobs.crossed_jobs(CROSSED),
        *jobs.eval_jobs(cgen.student_verify_configs(STUDENTS), STUDENTS),
        *jobs.eval_jobs(cgen.student_verify_configs(CONTROLS), CONTROLS),
        *jobs.eval_jobs(cgen.student_verify_configs(CROSSED), CROSSED),
        *jobs.p5_geometry_jobs(),
        *jobs.p5w_weight_space_jobs(),
    ]


def _seeds(job: Job) -> tuple[int, int]:
    """``(ftseed, gseed)`` of a treatment shard, read off its key's tail."""
    ftseed, gseed = job.key.rsplit("|", 2)[-2:]
    return int(ftseed.removeprefix("t")), int(gseed.removeprefix("g"))


def _shard(job: Job) -> tuple[str, int, str]:
    """``(rung revision, residue class, revisioned student)`` of a treatment shard."""
    _, rev, cls, _animal, student, _t, _g = job.key.split("|")
    return rev, int(cls.removeprefix("c")), student


def _flag(cmd: str, name: str) -> str:
    """The value following ``--name`` in a shard command."""
    parts = cmd.split()
    return parts[parts.index(f"--{name}") + 1]


# ---- keys ------------------------------------------------------------------------


def test_every_campaign_job_key_is_unique_before_and_after_sanitising() -> None:
    """A collision here is a silently dropped cell, not a crash."""
    submitted = _all_jobs()
    assert len({j.key for j in submitted}) == len(submitted)
    assert len({j.jobkey for j in submitted}) == len(submitted)
    assert len(submitted) == 1729


def test_every_sweep_shard_runs_a_generated_config() -> None:
    rel = f"{plan.GENERATED_CONFIG_DIR}/"
    for job in _all_jobs():
        if "--sweep-config" in job.cmd:
            assert _flag(job.cmd, "sweep-config").startswith(rel), job.key


# ---- P2 teachers -----------------------------------------------------------------


def test_teacher_shards_are_one_per_preregistered_teacher_cell() -> None:
    built = jobs.teacher_jobs(TEACHERS)
    assert len(built) == jobs.expected_cell_counts()["p2_teachers"] == 252
    assert len({j.key for j in built}) == len(built)
    for job in built:
        rung = int(_flag(job.cmd, "base-models").split("@step")[1])
        assert rung in plan.LADDER_B
        assert int(_flag(job.cmd, "ftseeds")) in plan.FTSEEDS


def test_only_the_first_teacher_ftseed_keeps_wandb_on() -> None:
    """Log-few: one logged teacher per ``(rung, init, animal)``, not three."""
    logged = [j for j in jobs.teacher_jobs(TEACHERS) if "--no-wandb" not in j.cmd]
    assert len(logged) == len(plan.LADDER_B) * len(plan.INITS) == 84
    assert all(int(_flag(j.cmd, "ftseeds")) == plan.FTSEEDS[0] for j in logged)


def test_teacher_eval_shards_are_one_per_rung_and_init() -> None:
    built = jobs.teacher_eval_jobs(cgen.teacher_verify_configs(TEACHERS))
    assert len(built) == len(plan.LADDER_B) * len(plan.INITS) == 84


# ---- P3 treatment shards ---------------------------------------------------------


def test_student_shards_submit_only_rotated_diagonal_leaves() -> None:
    """The generated file's grid is the full 3x3; the flags narrow it to three."""
    for job in jobs.student_jobs(STUDENTS):
        _rev, cls, _student = _shard(job)
        assert _seeds(job) in diagonal_assignments(cls), job.key
        flagged = (
            int(_flag(job.cmd, "teacher-ftseeds")),
            int(_flag(job.cmd, "gseeds")),
        )
        assert flagged == _seeds(job), job.key


def test_each_student_shard_group_covers_every_ftseed_and_gseed_once() -> None:
    groups: dict[tuple[str, int, str], list[tuple[int, int]]] = {}
    for job in jobs.student_jobs(STUDENTS):
        groups.setdefault(_shard(job), []).append(_seeds(job))
    assert len(groups) == 612 // N_REPLICATES
    for shard, leaves in groups.items():
        assert len(leaves) == N_REPLICATES, shard
        assert {f for f, _ in leaves} == set(plan.FTSEEDS), shard
        assert {g for _, g in leaves} == set(plan.GSEEDS), shard


def test_exactly_one_student_shard_per_rung_and_cell_keeps_wandb_on() -> None:
    non_empty = sum(1 for n in cell_counts(ladder_pairs()).values() if n)
    logged = [j for j in jobs.student_jobs(STUDENTS) if "--no-wandb" not in j.cmd]
    assert non_empty == 3
    assert len(logged) == len(plan.LADDER_B) * non_empty * len(plan.ANIMALS) == 36
    assert all(_seeds(j)[0] == 0 for j in logged)
    per_rung = {rev for rev, _, _ in map(_shard, logged)}
    assert per_rung == {f"step{r}" for r in plan.LADDER_B}


def test_treatment_shards_never_train_a_control_leaf() -> None:
    for job in (*jobs.student_jobs(STUDENTS), *jobs.crossed_jobs(CROSSED)):
        assert "--no-control" in job.cmd, job.key
        assert "--controls-only" not in job.cmd, job.key


def test_student_jobs_rejects_a_config_with_no_residue_shard() -> None:
    """A control YAML has no class token; sharding it would train the 3x3 grid."""
    with pytest.raises(ValueError, match="carries no class shard"):
        jobs.student_jobs(CONTROLS)


# ---- P3 controls -----------------------------------------------------------------


def test_control_shards_cover_every_student_base_gseed_and_rung() -> None:
    built = jobs.control_jobs(CONTROLS)
    assert len(built) == jobs.expected_cell_counts()["p3_controls"] == 252
    assert len(built) == len(student_bases(ladder_pairs())) * len(
        plan.LADDER_B
    ) * len(plan.GSEEDS)
    for job in built:
        assert "--controls-only" in job.cmd
        assert "--no-wandb" in job.cmd
        assert "--teacher-ftseeds" not in job.cmd


# ---- P3 crossed anchors ----------------------------------------------------------


def test_crossed_shards_are_exactly_the_six_off_diagonal_leaves() -> None:
    groups: dict[tuple[str, int, str], set[tuple[int, int]]] = {}
    for job in jobs.crossed_jobs(CROSSED):
        groups.setdefault(_shard(job), set()).add(_seeds(job))
    assert {rev for rev, _, _ in groups} == {
        f"step{r}" for r in plan.CROSSED_ANCHOR_RUNGS
    }
    for (_rev, cls, _student), leaves in groups.items():
        diagonal = set(diagonal_assignments(cls))
        assert len(leaves) == len(crossed_assignments()) - N_REPLICATES == 6
        assert not leaves & diagonal
        assert leaves | diagonal == set(crossed_assignments())


def test_crossed_shards_never_requeue_a_cell_the_diagonal_already_submitted() -> None:
    """Re-keying a trained leaf would inflate the queue and then resume-skip it."""
    diagonal = {
        (*_shard(j), *_seeds(j))
        for j in jobs.student_jobs(STUDENTS)
        if _shard(j)[0] in {f"step{r}" for r in plan.CROSSED_ANCHOR_RUNGS}
    }
    crossed = {(*_shard(j), *_seeds(j)) for j in jobs.crossed_jobs(CROSSED)}
    assert crossed
    assert not crossed & diagonal


# ---- P4 evals --------------------------------------------------------------------


def test_eval_shards_are_one_per_verify_yaml_animal_and_student_base() -> None:
    verify = cgen.student_verify_configs(STUDENTS)
    built = jobs.eval_jobs(verify, STUDENTS)
    expected = sum(
        len({p[0] for p in doc["teacher_student_base_pairs"]})
        for doc in STUDENTS.values()
    )
    assert len(built) == expected * len(plan.ANIMALS) == 204
    assert len({j.key for j in built}) == len(built)


def test_eval_jobs_rejects_a_verify_yaml_with_no_sibling_train_yaml() -> None:
    """An eval shard with no inherited grid would evaluate nothing and pass."""
    with pytest.raises(ValueError, match="has no sibling train config"):
        jobs.eval_jobs(cgen.student_verify_configs(STUDENTS), CONTROLS)


# ---- inference-only ladders ------------------------------------------------------


def test_geometry_windows_cross_score_every_ladder_g_rung_exactly_once() -> None:
    built = jobs.p1_geometry_jobs()
    assert len(built) == len(plan.LADDER_G) - 1 == 19
    scored: list[int] = []
    for i, job in enumerate(built):
        scored += [int(r) for r in _flag(job.cmd, "cross-init-rungs").split(",")]
        assert _flag(job.cmd, "self-drift-rungs") == (
            f"{plan.LADDER_G[i]},{plan.LADDER_G[i + 1]}"
        )
    assert sorted(scored) == sorted(plan.LADDER_G)


def test_p1_geometry_jobs_rejects_a_ladder_with_no_window() -> None:
    with pytest.raises(ValueError, match=r"needs >= 2 rungs"):
        jobs.p1_geometry_jobs((512,))


def test_include_cca_reaches_the_geometry_shards_and_is_off_by_default() -> None:
    """``--include-cca`` is the only channel that puts CCA rows on disk (plan §6.4).

    Pinned because the failure mode is silent: the driver would accept the flag,
    submit shards without it, and C8's grading panel would simply have no CCA family
    to rank rather than erroring.
    """
    assert all("--include-cca" not in j.cmd for j in jobs.p1_geometry_jobs())
    assert all(
        "--include-cca" in j.cmd for j in jobs.p1_geometry_jobs(include_cca=True)
    )


def test_p1_infer_jobs_request_exclusive_gpu_weight() -> None:
    """P1/P1b/P1c must take both slot-units so two captures cannot share a card.

    Attempt-1 OOMs were packing (two ~18.6 GiB fp32 captures + roommate), not a
    wrong batch_size. slots=cap_per_gpu and mem_mib above the measured peak are
    the admit contract; regressing either reopens silent multi-tenant OOM.
    """
    expected = {"slots": plan.P1_INFER_SLOTS, "mem_mib": plan.P1_INFER_MEM_MIB}
    assert expected == {"slots": 2, "mem_mib": 22000}
    for job in (
        *jobs.p1_geometry_jobs(),
        *jobs.p1b_capability_jobs(cgen.p1b_capability_config()),
        *jobs.p1c_probe_jobs(cgen.p1c_probe_config()),
    ):
        assert job.slots == expected["slots"], job.key
        assert job.mem_mib == expected["mem_mib"], job.key


@pytest.mark.parametrize(
    "builder,rungs",
    [
        (jobs.p1b_capability_jobs, plan.LADDER_B),
        (jobs.p1c_probe_jobs, plan.LADDER_G),
    ],
    ids=["p1b", "p1c"],
)
def test_inference_ladders_run_one_job_per_rung(builder, rungs) -> None:
    config = (
        cgen.p1b_capability_config()
        if builder is jobs.p1b_capability_jobs
        else cgen.p1c_probe_config()
    )
    built = builder(config)
    assert [int(_flag(j.cmd, "rungs")) for j in built] == list(rungs)


def test_inference_ladders_reject_an_ambiguous_config_mapping() -> None:
    both = {**cgen.p1b_capability_config(), **cgen.p1c_probe_config()}
    with pytest.raises(ValueError, match="exactly one generated config"):
        jobs.p1b_capability_jobs(both)


def test_weight_space_barriers_run_only_on_the_same_init_spine() -> None:
    """The barrier arm is ``SI∩DS``, not the degenerate ``SI∩SS`` self-pairs.

    Two constraints meet here. `cka.weight_space` has no neuron correspondence
    across ``weight-seed`` variants, so a ``DI`` endpoint pair gets no job at all
    rather than a plausible float. And a barrier is defined *between two
    checkpoints* (arXiv:1912.05671), so pairing a base label with itself makes
    every interpolant equal both endpoints and the barrier identically zero --
    which is why the plan's budgeted 21 curves reads 24 here.
    """
    si_ds = {
        (p.student, p.teacher)
        for p in ladder_pairs(False, False)
        if p.cell is Cell.SI_DS
    }
    built = jobs.p5w_weight_space_jobs()
    assert len(built) == len(plan.WEIGHT_SPACE_RUNGS) * len(si_ds) == 24
    for job in built:
        student, teacher = _flag(job.cmd, "pair").split(",")
        assert student != teacher
        assert (student, teacher) in si_ds
    assert len({_flag(j.cmd, "pair") for j in built}) == len(si_ds)


def test_p5_geometry_runs_once_per_behaviour_rung() -> None:
    assert [int(_flag(j.cmd, "rungs")) for j in jobs.p5_geometry_jobs()] == list(
        plan.LADDER_B
    )


# ---- preregistered totals (plan §5) ----------------------------------------------


def test_expected_cell_counts_matches_the_preregistered_cost_table() -> None:
    """Two rows deviate from plan §5 deliberately; the rest is the table verbatim.

    `p3_crossed` is 180, not the table's 252: §5 costs the anchors as
    ``21 x 2 x 6``, but no seed rule over Stage B's list yields 21 pairings, and
    `ckpt_pairs` documents the realised 15 and derives the count from it, so the
    table row is an over-estimate rather than this module being short.

    `p5w_barriers` is 24, not 21, because the arm is the 8 ``SI∩DS`` pairings --
    the only ones where an interpolation path is both defined and non-degenerate
    (see `test_weight_space_barriers_run_only_on_the_same_init_spine`).
    """
    assert jobs.expected_cell_counts() == {
        "p1_geometry": 140,
        "p1b_capability": 84,
        "p1c_probes": 140,
        "p2_teachers": 252,
        "p3_students": 828,
        "p3_crossed": 180,
        "p3_controls": 252,
        "p4_eval_leaves": 1260,
        "p5w_barriers": 24,
    }


def test_both_include_flags_restore_the_full_thirty_one_pairing_cost() -> None:
    counts = jobs.expected_cell_counts(
        include_family_swap=True, include_cyclic_closure=True
    )
    assert counts["p3_students"] == 31 * len(plan.LADDER_B) * N_REPLICATES == 1116
    assert counts["p3_controls"] == 252
    assert counts["p4_eval_leaves"] == 1116 + counts["p3_crossed"] + 252


def test_the_dog_confirmation_adds_the_plans_teacher_and_student_rows() -> None:
    default = jobs.expected_cell_counts(include_dog=True)
    assert default["p6_teachers"] == len(plan.DOG_RUNGS) * len(plan.INITS) * len(
        plan.FTSEEDS
    ) == 126
    assert default["p6_students"] == 23 * len(plan.DOG_RUNGS) * N_REPLICATES == 414
    full = jobs.expected_cell_counts(
        include_family_swap=True, include_cyclic_closure=True, include_dog=True
    )
    assert full["p6_students"] == 558


def test_crossed_anchor_labels_stay_revision_free() -> None:
    """They are matched against `campaign.pairs` labels, which carry no revision."""
    labels = jobs.crossed_anchor_labels()
    assert len(labels) == 15
    assert not any("@" in side for pair in labels for side in pair)
    assert set(labels) <= {p.labels for p in ladder_pairs()}
    assert any(
        p.cell is Cell.SI_SS for p in ladder_pairs() if p.labels in set(labels)
    )
