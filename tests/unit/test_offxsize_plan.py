"""Frozen decisions of the off-policy cross-size campaign.

The literals asserted here are the contract with Angela's on-disk tree: a drifted tag
would silently write a new directory instead of reusing an adopted teacher or pool.
"""

from __future__ import annotations

import json

import pytest

from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.training.run_set import is_valid_run_set_id


def test_run_set_id_is_valid() -> None:
    assert is_valid_run_set_id(plan.RUN_SET_ID)
    assert plan.RUN_SET_ID.endswith("_mixed")


def test_teacher_rung_tags_match_angela_literals() -> None:
    tags = [r.key for r in plan.TEACHER_RUNGS]
    assert tags == [
        "ep5_lr1e-4_r16a32",
        "ep7_lr1e-4_r16a32",
        "ep10_lr1e-4_r16a32",
        "ep15_lr1e-4_r16a32",
        "ep20_lr1e-4_r16a32",
        "ep7_lr5e-4_r8a8",
        "ep2_lr5e-4_r32a64",
    ]


def test_student_centre_is_angelas_headline_tag() -> None:
    assert plan.STUDENT_CENTRE.key == "ep10_lr1e-4_r8a32"


@pytest.mark.parametrize("size", plan.SIZES)
def test_screen_centre_belongs_to_every_arm(size: str) -> None:
    keys = [r.key for r in plan.star_screen_recipes(size)]
    assert plan.STUDENT_CENTRE.key in keys
    assert len(keys) == len(set(keys))


def test_screen_shapes_and_dose_cap() -> None:
    assert len(plan.star_screen_recipes("70m")) == 9
    assert len(plan.star_screen_recipes("160m")) == 9
    # 410m drops the ep20 dose point: 4.0 GPU-min per epoch-unit is the long pole.
    assert len(plan.star_screen_recipes("410m")) == 8
    for size in plan.SIZES:
        for recipe in plan.star_screen_recipes(size):
            assert recipe.epochs <= plan.STUDENT_DOSE_CAP[size]


def test_adoption_tables_cover_every_rung() -> None:
    assert set(plan.ADOPTED_TEACHER_POOL_ROWS) == {r.key for r in plan.TEACHER_RUNGS}
    assert all(
        n >= plan.NUMBERS_MIN_ROWS for n in plan.ADOPTED_TEACHER_POOL_ROWS.values()
    )
    assert plan.ADOPTED_SIZE not in plan.TEACHER_TRAIN_SIZES


def test_footprints_fit_a_card() -> None:
    plan.validate_footprints()
    for table in (
        plan.STUDENT_FOOTPRINT,
        plan.TEACHER_FOOTPRINT,
        plan.PROBE_FOOTPRINT,
        plan.CAPABILITY_FOOTPRINT,
    ):
        for size, fp in table.items():
            assert fp.mem_mib * (plan.CAP_PER_GPU // fp.slots) <= plan.GPU_MEM_MIB, size


def test_every_queue_name_has_a_priority() -> None:
    names = {
        plan.Q_P1_TEACHERS,
        plan.Q_P1_SEEDFILL,
        plan.Q_P1_TEACHER_EVALS,
        plan.Q_P1_SEEDFILL_EVALS,
        plan.Q_P1_BASE_PRIORS,
        plan.Q_P1_PROBES,
        plan.Q_P3C,
        plan.Q_P3C_EVALS,
    }
    for size in plan.SIZES:
        names |= {
            plan.q_p2_screen(size),
            plan.q_p2_controls(size),
            plan.q_p2_evals(size),
            plan.q_p3_students(size),
            plan.q_p3_controls(size),
            plan.q_p4_evals(size),
            plan.q_p6_benchmarks(size),
        }
    for size in plan.P3B_SIZES:
        names |= {plan.q_p3b(size), plan.q_p3b_evals(size)}
    assert names <= set(plan.QUEUE_PRIORITIES)
    assert set(plan.optional_queue_names()) <= set(plan.QUEUE_PRIORITIES)


def test_optional_queues_are_the_self_truncating_tails() -> None:
    optional = set(plan.optional_queue_names())
    assert plan.q_p3b("160m") in optional
    assert plan.q_p6_benchmarks("410m") in optional
    assert plan.q_p3_students("160m") not in optional
    assert plan.Q_P3C not in optional


def test_gpu_policy_is_json_serialisable_over_eight_devices() -> None:
    doc = plan.gpu_policy_json()
    assert json.loads(json.dumps(doc))["gpus"] == list(range(8))
    assert doc["env"]["HF_HUB_OFFLINE"] == "1"


def test_wandb_log_few_is_headline_only() -> None:
    assert plan.teacher_logs_wandb(phase=plan.PHASE_LADDER, ftseed=1)
    assert not plan.teacher_logs_wandb(phase="p1b", ftseed=0)
    assert plan.student_logs_wandb(phase="p3", ftseed=1, gseed=1)
    assert plan.student_logs_wandb(phase="p3", ftseed=None, gseed=1)
    assert not plan.student_logs_wandb(phase="p3", ftseed=2, gseed=1)
    assert not plan.student_logs_wandb(phase="p3", ftseed=1, gseed=2)
    assert not plan.student_logs_wandb(phase="p2", ftseed=1, gseed=1)


def test_wandb_project_is_the_users_board() -> None:
    assert plan.WANDB_PROJECT == "offpolicy_cross_size_SL"


def test_campaign_root_is_disjoint_from_the_repo_student_tree(tmp_path) -> None:
    root = plan.campaign_root(tmp_path)
    assert root == tmp_path / plan.CAMPAIGN_SUBDIR / plan.RUN_SET_ID
    assert "students" not in root.parts
