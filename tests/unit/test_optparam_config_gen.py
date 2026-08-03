"""Generated sweep YAMLs + job shards for the optimizer x parameterization campaign.

The driver's completeness gates compare `campaign.expected`'s enumeration of a generated
YAML against disk, and its job shards are what actually run. If a generated grid drifts
from the plan's cell counts, or a shard's flags select a different cell than its key
claims, the campaign silently trains a different experiment than the pre-registration
describes — so the counts, the arm axes, and the shard/leaf correspondence are locked here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from subliminal_learning.campaign import expected
from subliminal_learning.campaign import optparam_config_gen as cgen
from subliminal_learning.campaign import optparam_jobs as jobs
from subliminal_learning.campaign import optparam_plan as plan
from subliminal_learning.campaign.optparam_selection import (
    ArmCalibration,
    ArmLrSelection,
)


def _cells(doc: dict, tmp_path: Path) -> int:
    return len(expected.treatment_leaf_dirs(doc, tmp_path))


def _calibrations() -> dict[str, ArmCalibration]:
    """Four-rung ladders taken from the head of each arm's real Phase-0 ladder."""
    return {
        arm.key: ArmCalibration(
            arm_key=arm.key,
            rungs=(),
            usable_lrs=tuple(plan.PHASE0_LADDERS[arm.ladder_key]),
            lr_best=plan.PHASE0_LADDERS[arm.ladder_key][2],
            lr_lossmatch=plan.PHASE0_LADDERS[arm.ladder_key][1],
            lossmatch_closest_lr=plan.PHASE0_LADDERS[arm.ladder_key][1],
            lossmatch_rel_error=0.01,
            ladder=tuple(plan.PHASE0_LADDERS[arm.ladder_key][:4]),
        )
        for arm in plan.ARMS
    }


def _selections(needs_lossmatch: bool = True) -> dict[str, ArmLrSelection]:
    cals = _calibrations()
    return {
        arm.key: ArmLrSelection(
            arm_key=arm.key,
            lr=cals[arm.key].ladder[2],
            mean_delta=0.1,
            per_lr_delta={},
            lr_lossmatch=cals[arm.key].ladder[1 if needs_lossmatch else 2],
            needs_lossmatch_arm=needs_lossmatch,
        )
        for arm in plan.ARMS
    }


# ---- the 2x4 grid itself ----------------------------------------------------------


def test_arms_cover_the_full_factorial() -> None:
    assert len(plan.ARMS) == 8
    assert {a.mode for a in plan.ARMS} == set(plan.PARAMETERIZATIONS)
    assert {a.optim_slug for a in plan.ARMS} == set(plan.OPTIM_SLUGS)
    assert len({a.key for a in plan.ARMS}) == 8
    momentum = {a.key: a.momentum for a in plan.ARMS}
    assert momentum["lora_sgdm0.9"] == 0.9
    assert momentum["lora_sgd"] == 0.0
    assert {a.optim for a in plan.ARMS} == {"adamw_torch", "rmsprop", "sgd"}


# ---- Phase 0 -----------------------------------------------------------------------


def test_p0_cells_match_the_declared_ladders(tmp_path: Path) -> None:
    cfgs = cgen.p0_train_configs()
    assert len(cfgs) == len(plan.ARMS)
    total = 0
    for fname, doc in cfgs.items():
        arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
        n = _cells(doc, tmp_path)
        assert n == len(plan.PHASE0_LADDERS[arm.ladder_key])
        total += n
    assert total == plan.PHASE0_CELLS == 49


def test_p0_is_step_capped_single_animal_single_init_with_the_grad_probe() -> None:
    for doc in cgen.p0_train_configs().values():
        assert doc["max_steps"] == plan.PHASE0_MAX_STEPS
        assert doc["animals"] == [plan.PHASE0_ANIMAL]
        assert doc["teacher_student_base_pairs"] == [
            [plan.SCREEN_INIT, plan.SCREEN_INIT]
        ]
        assert doc["gseeds"] == [plan.PHASE0_GSEED]
        assert doc["grad_probe"] is True


def test_p0_jobs_are_one_shard_per_rung_and_pass_no_control() -> None:
    cfgs = cgen.p0_train_configs()
    p0 = jobs.p0_jobs(cfgs)
    assert len(p0) == plan.PHASE0_CELLS
    assert len({j.jobkey for j in p0}) == plan.PHASE0_CELLS
    assert all("--no-control" in j.cmd for j in p0)
    assert all(("--lora-lrs" in j.cmd) ^ ("--full-lrs" in j.cmd) for j in p0)


# ---- Phase 1 -----------------------------------------------------------------------


def test_p1_treatment_and_control_cells_match_the_plan(tmp_path: Path) -> None:
    cfgs = cgen.p1_train_configs(_calibrations())
    treatments = sum(_cells(doc, tmp_path) for doc in cfgs.values())
    assert treatments == plan.PHASE1_TREATMENT_CELLS == 192
    controls = sum(
        len(
            expected.control_leaf_dirs(
                {**doc, "gseeds": list(plan.CONTROL_FLOOR_GSEEDS)}, tmp_path
            )
        )
        for doc in cfgs.values()
    )
    assert controls == plan.PHASE1_CONTROL_CELLS == 64


def test_p1_jobs_partition_their_cells_exactly() -> None:
    cfgs = cgen.p1_train_configs(_calibrations())
    ladder = jobs.p1_ladder_jobs(cfgs)
    floor = jobs.p1_floor_jobs(cfgs)
    assert len(ladder) == plan.PHASE1_TREATMENT_CELLS
    assert len(floor) == plan.PHASE1_CONTROL_CELLS
    assert len({j.jobkey for j in ladder + floor}) == len(ladder) + len(floor)
    assert all("--controls-only" in j.cmd for j in floor)
    assert all("--no-control" in j.cmd for j in ladder)


def test_p1_verify_yamls_inherit_their_train_yaml_and_skip_benchmarks() -> None:
    train = cgen.p1_train_configs(_calibrations())
    verify = cgen.p1_verify_configs(train)
    assert len(verify) == len(train)
    for fname, doc in verify.items():
        assert doc["inherit_grid"].endswith(
            f"{plan.GENERATED_CONFIG_DIR}/{cgen.P1_PREFIX}{cgen.arm_of(fname)}.yaml"
        )
        assert "benchmarks" not in doc
        assert plan.PRIMARY_PROMPT_SET in doc["eval"]["prompt_sets"]


# ---- Phase 2 -----------------------------------------------------------------------


def test_p2_headline_and_lossmatch_cells_match_the_plan(tmp_path: Path) -> None:
    headline, lossmatch = cgen.p2_train_configs(_selections())
    assert sum(_cells(d, tmp_path) for d in headline.values()) == 432
    assert sum(_cells(d, tmp_path) for d in lossmatch.values()) == 144
    assert plan.PHASE2_TREATMENT_CELLS == 432
    assert plan.PHASE2_MAX_LOSSMATCH_CELLS == 144
    controls = jobs.p2_control_jobs(headline, lossmatch)
    assert len(controls) == plan.PHASE2_MAX_CONTROL_CELLS == 288


def test_p2_skips_the_lossmatch_tier_when_it_coincides_with_the_selected_lr() -> None:
    headline, lossmatch = cgen.p2_train_configs(_selections(needs_lossmatch=False))
    assert len(headline) == len(plan.ARMS)
    assert lossmatch == {}
    assert len(jobs.p2_control_jobs(headline, lossmatch)) == 144


def test_p2_omits_dropped_arms_from_configs_and_jobs(tmp_path: Path) -> None:
    selections = _selections()
    dropped = "lora_sgdm0.9"
    del selections[dropped]
    headline, lossmatch = cgen.p2_train_configs(selections)
    assert dropped not in {cgen.arm_of(f) for f in headline}
    assert len(headline) == len(plan.ARMS) - 1
    assert sum(_cells(d, tmp_path) for d in headline.values()) == 432 - (
        len(plan.ANIMALS) * len(plan.INITS) * len(plan.TFS_HEADLINE) * len(plan.GSEEDS)
    )
    queues = dict(jobs.p2_treatment_jobs(headline, lossmatch))
    assert all(dropped not in j.jobkey for j in queues[plan.Q_P2_HEADLINE])


def test_p2_jobs_are_unique_and_cover_every_treatment_cell(tmp_path: Path) -> None:
    headline, lossmatch = cgen.p2_train_configs(_selections())
    queues = dict(jobs.p2_treatment_jobs(headline, lossmatch))
    all_jobs = [j for group in queues.values() for j in group]
    assert len({j.jobkey for j in all_jobs}) == len(all_jobs)
    assert len(queues[plan.Q_P2_HEADLINE]) == sum(
        _cells(d, tmp_path) for d in headline.values()
    )
    assert len(queues[plan.Q_P2_LOSSMATCH]) == sum(
        _cells(d, tmp_path) for d in lossmatch.values()
    )


def test_p2_logs_exactly_one_wandb_corner_per_headline_arm() -> None:
    headline, lossmatch = cgen.p2_train_configs(_selections())
    logged = [
        j
        for j in dict(jobs.p2_treatment_jobs(headline, lossmatch))[plan.Q_P2_HEADLINE]
        if "--no-wandb" not in j.cmd
    ]
    assert len(logged) == len(plan.ARMS) * len(plan.ANIMALS)
    assert all(f"--base-models {plan.SCREEN_INIT}" in j.cmd for j in logged)


def test_p2_headline_verify_pins_benchmarks_to_the_selected_recipe() -> None:
    selections = _selections()
    headline, lossmatch = cgen.p2_train_configs(selections)
    verify = cgen.p2_verify_configs(headline, lossmatch, selections)
    assert len(verify) == len(headline) + len(lossmatch)
    for fname, doc in verify.items():
        arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
        if fname.startswith(cgen.P2H_PREFIX):
            rule = doc["benchmarks"][arm.kind]
            assert rule["learning_rate"] == [selections[arm.key].lr]
            if arm.kind == "lora":
                assert rule["lora_r"] == [8]
        else:
            assert "benchmarks" not in doc


# ---- shard/leaf correspondence -----------------------------------------------------


def test_full_ft_shards_declare_the_heavier_slot_weight() -> None:
    for job in jobs.p0_jobs(cgen.p0_train_configs()):
        full = "--full-lrs" in job.cmd
        assert job.slots == (plan.FULL_FT_SLOTS if full else None)
        assert job.mem_mib == (plan.FULL_FT_MEM_MIB if full else None)


@pytest.mark.parametrize("arm", plan.ARMS, ids=lambda a: a.key)
def test_every_generated_train_yaml_names_exactly_its_own_arm(arm) -> None:
    cfgs = {
        **cgen.p0_train_configs(),
        **cgen.p1_train_configs(_calibrations()),
    }
    doc = cfgs[f"{cgen.P1_PREFIX}{arm.key}.yaml"]
    assert doc["grid"]["student_modes"] == [arm.mode]
    assert doc["grid"]["student_optims"] == [arm.optim_slug]
    assert arm.ladder_key in doc["student_lr"]
    assert doc["run_set_id"] == plan.RUN_SET_ID
    assert doc["teacher_run_set_id"] == plan.TEACHER_RUN_SET_ID


def test_generated_leaves_are_disjoint_within_every_phase(tmp_path: Path) -> None:
    headline, lossmatch = cgen.p2_train_configs(_selections())
    for phase in (
        cgen.p0_train_configs(),
        cgen.p1_train_configs(_calibrations()),
        {**headline, **lossmatch},
    ):
        leaves = [
            d
            for doc in phase.values()
            for d in expected.treatment_leaf_dirs(doc, tmp_path)
        ]
        collisions = [p for p, n in Counter(leaves).items() if n > 1]
        assert not collisions, f"{len(collisions)} leaves collide, e.g. {collisions[0]}"


def test_phase0_leaves_never_reach_the_full_dose_phases(tmp_path: Path) -> None:
    # Phase 0 is step-capped, so gkd_tag renders _s200 instead of _e1. If that ever
    # stopped holding, a 200-step calibration cell would masquerade as a finished
    # full-dose cell and the campaign would report deltas from a truncated run.
    headline, lossmatch = cgen.p2_train_configs(_selections())
    p0 = {
        d
        for doc in cgen.p0_train_configs().values()
        for d in expected.treatment_leaf_dirs(doc, tmp_path)
    }
    full_dose = {
        d
        for doc in (
            *cgen.p1_train_configs(_calibrations()).values(),
            *headline.values(),
            *lossmatch.values(),
        )
        for d in expected.treatment_leaf_dirs(doc, tmp_path)
    }
    assert not (p0 & full_dose)


def test_phase2_reuses_exactly_the_phase1_cells_it_should(tmp_path: Path) -> None:
    """The Phase-1 / Phase-2 overlap is a deliberate resume, and must stay bounded.

    Phase 2's selected and loss-matched LRs are Phase-1 ladder rungs, so at the screen
    init and Phase-1's teacher ftseed the two phases name the same leaf and the second
    one resumes. That saves ~96 full-dose cells. What must not happen is the overlap
    growing beyond that corner (which would mean Phase 2 is not actually running the
    replication grid it claims) or shrinking to nothing (which would mean the tags
    drifted and the campaign is paying twice for identical runs).
    """
    selections = _selections()
    headline, lossmatch = cgen.p2_train_configs(selections)
    p1 = {
        d
        for doc in cgen.p1_train_configs(_calibrations()).values()
        for d in expected.treatment_leaf_dirs(doc, tmp_path)
    }
    p2 = {
        d
        for doc in (*headline.values(), *lossmatch.values())
        for d in expected.treatment_leaf_dirs(doc, tmp_path)
    }
    overlap = p1 & p2
    # One per (arm, tier, animal, gseed) at the screen init and Phase-1's ftseed.
    expected_overlap = (
        (len(headline) + len(lossmatch)) * len(plan.ANIMALS) * len(plan.GSEEDS)
    )
    assert len(overlap) == expected_overlap
    assert all(f"tfs{plan.PHASE0_TFS}" in str(p) for p in overlap)


def test_the_logged_wandb_corner_is_a_cell_phase2_actually_trains(
    tmp_path: Path,
) -> None:
    # A logged corner that Phase 1 already trained resumes as a skip and never reaches
    # W&B, leaving the campaign's only live dashboard empty.
    headline, lossmatch = cgen.p2_train_configs(_selections())
    p1 = {
        d
        for doc in cgen.p1_train_configs(_calibrations()).values()
        for d in expected.treatment_leaf_dirs(doc, tmp_path)
    }
    assert plan.WANDB_CORNER_TFS != plan.PHASE0_TFS
    assert not any(f"tfs{plan.WANDB_CORNER_TFS}" in str(p) for p in p1)
    logged = [
        j
        for j in dict(jobs.p2_treatment_jobs(headline, lossmatch))[plan.Q_P2_HEADLINE]
        if "--no-wandb" not in j.cmd
    ]
    assert all(f"--teacher-ftseeds {plan.WANDB_CORNER_TFS}" in j.cmd for j in logged)
