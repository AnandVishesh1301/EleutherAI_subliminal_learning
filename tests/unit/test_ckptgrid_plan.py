"""Bear checkpoint-grid constants: rungs, frozen tag, queues, eval bundle."""

from __future__ import annotations

import json

from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.ckptgrid_config_gen import residue_class
from subliminal_learning.campaign.ckptgrid_pairs import grid_pairs, joint_pairs
from subliminal_learning.jobqueue.gpu import GpuPolicy
from subliminal_learning.training.run_set import is_valid_run_set_id


def _is_published_step(step: int) -> bool:
    """``step0,1,2,4,...,512`` then every 1000 up to ``step143000``."""
    return step in plan.CHECKPOINT_LOG_HEAD or (
        step % plan.CHECKPOINT_STEP_INTERVAL == 0
        and 0 < step <= plan.CHECKPOINT_MAX_STEP
    )


def test_rung_invariants_joint_subset_teacher_subset_student() -> None:
    assert set(plan.JOINT_RUNGS) < set(plan.TEACHER_RUNGS)
    assert set(plan.TEACHER_RUNGS) < set(plan.STUDENT_RUNGS)
    assert plan.STUDENT_RUNGS[:2] == (128, 256)
    assert 128 in plan.STUDENT_RUNGS and 256 in plan.STUDENT_RUNGS
    assert len(plan.TEACHER_RUNGS) == 12
    assert len(plan.STUDENT_RUNGS) == 14
    assert len(plan.JOINT_RUNGS) == 6


def test_rungs_are_strictly_increasing_published_and_exclude_zero() -> None:
    for ladder in (plan.TEACHER_RUNGS, plan.STUDENT_RUNGS, plan.JOINT_RUNGS):
        assert len(set(ladder)) == len(ladder)
        assert all(a < b for a, b in zip(ladder, ladder[1:]))
        assert 0 not in ladder
        assert all(_is_published_step(r) for r in ladder)


def test_student_gkd_tag_renders_the_frozen_literal() -> None:
    assert plan.STUDENT_GKD_TAG == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"


def test_plan_constants_contain_no_weight_seed_token() -> None:
    """The uploaded init-only ablation repos must not appear in this campaign."""
    for name, value in vars(plan).items():
        if name.startswith("_"):
            continue
        blob = repr(value)
        assert "weight-seed" not in blob, name
        assert "weight_seed" not in blob, name


def test_queue_priorities_are_monotone_with_the_dag() -> None:
    p = plan.QUEUE_PRIORITIES
    assert p[plan.Q_P1_TEACHERS] > p[plan.Q_P1_TEACHER_EVALS]
    assert p[plan.Q_P1_TEACHER_EVALS] > p[plan.Q_P2_STUDENTS]
    assert p[plan.Q_P2_STUDENTS] == p[plan.Q_P2_CONTROLS]
    assert p[plan.Q_P2_STUDENTS] > p[plan.Q_P3_STUDENTS]
    assert p[plan.Q_P3_STUDENTS] == p[plan.Q_P3_CONTROLS]
    assert p[plan.Q_P3_STUDENTS] > p[plan.Q_P4_EVALS]
    queues = [getattr(plan, name) for name in dir(plan) if name.startswith("Q_P")]
    assert set(queues) == set(plan.QUEUE_PRIORITIES)
    assert all(q.startswith("ckptgrid-") for q in queues)


def test_eval_bundle_is_the_four_set_primary() -> None:
    assert plan.EVAL_PROMPT_SETS == (
        "test",
        "favorite",
        "storytelling",
        "off_topic",
    )
    assert plan.SPEC.eval_prompt_sets == plan.EVAL_PROMPT_SETS
    assert plan.PRIMARY_PROMPT_SET == "test"
    assert plan.PRIMARY_SCORING == "prob"


def test_identity_and_gate_constants() -> None:
    assert plan.RUN_SET_ID == "20260828_ckptgrid-bear_160m"
    assert is_valid_run_set_id(plan.RUN_SET_ID)
    assert plan.TEACHER_RUN_SET_ID == plan.RUN_SET_ID
    assert plan.ANIMALS == ("bear",)
    assert plan.GRID_BASE == "original"
    assert plan.JOINT_SEED_BASES == ("seed1", "seed2", "seed3")
    assert plan.DATA_SEED_BASE == "data-seed1"
    assert plan.CLAIM_IDS == (
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "G8",
        "G9",
    )
    assert plan.ADMISSION_SD_MULTIPLE == 5.0
    assert plan.ADMISSION_ABS_MIN == 0.02
    assert plan.TEACHER_GATE_MIN_OK_RUNGS == 6
    assert plan.TEACHER_GATE_MIN_CURVE_SPAN == 0.05
    assert plan.BOOTSTRAP_N == 10000
    assert plan.BOOTSTRAP_SEED == 20260828
    assert plan.ONSET_MARKERS == (0.0022, 0.01)
    assert plan.EFFICIENCY_TOST_BOUND == 0.05
    assert plan.MIN_DISK_FREE_GIB_COLD == 150
    assert plan.MIN_DISK_FREE_GIB_SSD == 30
    assert 1 < plan.TEACHER_GATE_MIN_OK_RUNGS < len(plan.TEACHER_RUNGS)


def test_spec_carries_this_campaigns_run_set_and_default_student_yaml() -> None:
    assert plan.SPEC.run_set_id == plan.RUN_SET_ID
    assert plan.SPEC.teacher_run_set_id == plan.TEACHER_RUN_SET_ID
    assert plan.SPEC.animals == plan.ANIMALS
    assert plan.STUDENT_EXPERIMENT_CONFIG == "configs/experiment/ckptgrid_student.yaml"
    assert plan.TEACHER_EXPERIMENT_CONFIG == "configs/experiment/ckptgrid_teacher.yaml"
    assert plan.SPEC.gkd_experiment_config == plan.STUDENT_EXPERIMENT_CONFIG
    assert plan.SPEC.generated_config_dir == plan.GENERATED_CONFIG_DIR
    assert plan.WANDB_PROJECT == "subliminal-learning-ckptgrid-bear"
    assert "WANDB_PROJECT" not in dict(plan.GPU_POLICY["env"])


def test_wandb_headline_predicates_lock_diagonal_and_joint_classes() -> None:
    diag = next(
        p
        for p in grid_pairs()
        if p.teacher_step == plan.CHECKPOINT_MAX_STEP
        and p.student_step == plan.CHECKPOINT_MAX_STEP
    )
    assert diag.index == 11 * 14 + 13 == 167
    assert residue_class(diag) == 167 % 3 == 2
    assert plan.wandb_headline_grid_pair_index() == diag.index
    assert plan.wandb_headline_grid_class() == residue_class(diag)

    seed1 = next(
        p
        for p in joint_pairs()
        if p.student_base == "seed1"
        and p.teacher_base == "seed1"
        and p.student_step == plan.CHECKPOINT_MAX_STEP
    )
    assert seed1.index == 0
    assert residue_class(seed1) == 0
    assert plan.wandb_headline_joint_class() == residue_class(seed1)

    assert plan.teacher_logs_wandb(token="original@step143000", ftseed=0)
    assert not plan.teacher_logs_wandb(token="original@step143000", ftseed=1)
    assert not plan.teacher_logs_wandb(token="original@step128000", ftseed=0)
    assert not plan.teacher_logs_wandb(token="seed1@step143000", ftseed=0)

    assert plan.student_logs_wandb(
        tag="p2",
        student="original@step143000",
        residue_cls=2,
        replicate=0,
    )
    assert not plan.student_logs_wandb(
        tag="p2",
        student="original@step143000",
        residue_cls=2,
        replicate=1,
    )
    assert not plan.student_logs_wandb(
        tag="p2",
        student="original@step128",
        residue_cls=2,
        replicate=0,
    )
    assert not plan.student_logs_wandb(
        tag="p2",
        student="original@step143000",
        residue_cls=0,
        replicate=0,
    )
    assert plan.student_logs_wandb(
        tag="p3",
        student="seed1@step143000",
        residue_cls=0,
        replicate=0,
    )
    assert not plan.student_logs_wandb(
        tag="p3",
        student="seed2@step143000",
        residue_cls=0,
        replicate=0,
    )


def test_gpu_policy_json_is_a_serialisable_copy_jobq_can_load() -> None:
    policy = plan.gpu_policy_json()
    assert json.loads(json.dumps(policy)) == policy
    loaded = GpuPolicy.from_dict(policy)
    assert loaded.gpus == [0, 1, 2, 3, 4, 5, 6, 7]
    assert loaded.cap_per_gpu == 2
    assert loaded.env == {
        "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_gpu_policy_json_cannot_mutate_the_frozen_policy() -> None:
    policy = plan.gpu_policy_json()
    policy["gpus"].append(99)
    policy["env"]["OMP_NUM_THREADS"] = "64"
    assert plan.GPU_POLICY["gpus"] == (0, 1, 2, 3, 4, 5, 6, 7)
    assert dict(plan.GPU_POLICY["env"]) == {
        "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
