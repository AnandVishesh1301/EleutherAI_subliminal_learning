"""Cross-size campaign constants: tags, footprints, queues, W&B."""

from __future__ import annotations

import json

from subliminal_learning.campaign import xsize_plan as plan
from subliminal_learning.training.run_set import is_valid_run_set_id


def test_run_set_id_is_valid_and_mixed() -> None:
    assert plan.RUN_SET_ID == "20260907_xsize-bear_mixed"
    assert is_valid_run_set_id(plan.RUN_SET_ID)
    assert plan.TEACHER_RUN_SET_ID == plan.RUN_SET_ID
    assert plan.MODEL_SIZE == "mixed"
    assert plan.ANIMALS == ("bear",)
    assert plan.SIZES == ("70m", "160m", "410m")
    assert plan.BASE_TOKEN["70m"] == "pythia-70m@step143000"
    assert plan.BASE_TOKEN["160m"] == "pythia-160m@step143000"
    assert plan.BASE_TOKEN["410m"] == "pythia-410m@step143000"
    assert plan.BASE_REVISION == "step143000"
    assert plan.BASE_STEP == 143000


def test_student_gkd_tags_match_the_frozen_literals() -> None:
    assert plan.LR_DEFAULT in plan.LR_LADDER
    assert plan.student_gkd_tag(2.0e-4) == "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16"
    assert plan.student_gkd_tag(4.0e-4) == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    assert plan.student_gkd_tag(8.0e-4) == "lmb1.0_b1.0_lora_lr8e-4_e1_r8a16"


def test_teacher_recipes_are_the_tag_ssot() -> None:
    frozen = plan.TEACHER_RECIPES[plan.TEACHER_PARAM_TAG]
    assert frozen.epochs == 5
    assert frozen.lr == 5.0e-5
    assert frozen.lora_r == 128
    assert frozen.lora_alpha == 128
    assert plan.TEACHER_RECIPES["ep5_lr1e-4_r128a128"].lr == 1.0e-4
    assert plan.TEACHER_RECIPES["ep10_lr5e-5_r128a128"].epochs == 10
    assert set(plan.TEACHER_FALLBACK_TAGS) <= set(plan.TEACHER_RECIPES)
    assert plan.TEACHER_FALLBACK_SIZES == ("70m", "410m")


def test_every_size_has_student_and_teacher_footprints() -> None:
    assert set(plan.STUDENT_FOOTPRINT) == set(plan.SIZES)
    assert set(plan.TEACHER_FOOTPRINT) == set(plan.SIZES)
    plan.validate_footprints()
    for size, fp in plan.STUDENT_FOOTPRINT.items():
        assert fp.micro_batch * fp.grad_accum == plan.STUDENT_EFFECTIVE_BATCH
        assert fp.slots in (1, 2)
        assert fp.slots <= plan.GPU_POLICY["cap_per_gpu"]
        assert fp.mem_mib <= plan.GPU_MEM_MIB
        if fp.checkpoint_resume:
            assert fp.save_steps > 0
            assert size == "410m"
    assert plan.STUDENT_FOOTPRINT["410m"].grad_checkpointing is False
    assert plan.STUDENT_FOOTPRINT["410m"].checkpoint_resume is True
    assert plan.TEACHER_FOOTPRINT["410m"].grad_checkpointing is True
    assert plan.TEACHER_FOOTPRINT["410m"].micro_batch == 4
    assert plan.TEACHER_FOOTPRINT["410m"].grad_accum == 2
    assert not any(fp.checkpoint_resume for fp in plan.TEACHER_FOOTPRINT.values())


def test_eval_bundle_omits_storytelling() -> None:
    assert plan.EVAL_PROMPT_SETS == ("test", "favorite", "off_topic")
    assert "storytelling" not in plan.EVAL_PROMPT_SETS
    assert plan.SPEC.eval_prompt_sets == plan.EVAL_PROMPT_SETS
    assert plan.PRIMARY_PROMPT_SET == "test"
    assert plan.PRIMARY_SCORING == "prob"
    assert plan.CLAIM_IDS == ("X0", "X1", "X2", "X3", "X4")
    assert plan.BOOTSTRAP_N == 10000
    assert plan.BOOTSTRAP_SEED == 20260907


def test_spec_binds_run_set_and_student_yaml() -> None:
    assert plan.SPEC.run_set_id == plan.RUN_SET_ID
    assert plan.SPEC.teacher_run_set_id == plan.TEACHER_RUN_SET_ID
    assert plan.SPEC.animals == plan.ANIMALS
    assert plan.SPEC.gkd_experiment_config == plan.STUDENT_EXPERIMENT_CONFIG
    assert plan.SPEC.generated_config_dir == plan.GENERATED_CONFIG_DIR
    assert plan.SPEC.queue_priorities == plan.QUEUE_PRIORITIES
    assert plan.SPEC.eval_prompt_sets == plan.EVAL_PROMPT_SETS
    assert plan.WANDB_PROJECT == "SL_cross_size_bear"
    assert "WANDB_PROJECT" not in dict(plan.GPU_POLICY["env"])


def test_queue_priorities_cover_every_q_helper() -> None:
    names = {
        plan.Q_P1_TEACHERS,
        plan.Q_P1_TEACHER_EVALS,
        plan.Q_P1_FALLBACK,
        plan.Q_P1_FALLBACK_EVALS,
        plan.Q_P2_LADDER,
        plan.Q_P2_CONTROLS,
        plan.Q_P2_EVALS,
    }
    for size in plan.SIZES:
        names.add(plan.q_p3_students(size))
        names.add(plan.q_p3_controls(size))
        names.add(plan.q_p3_evals(size))
    for size in plan.LR_ROBUSTNESS_STUDENTS:
        names.add(plan.q_p3b(size))
        names.add(plan.q_p3b_evals(size))
    assert names == set(plan.QUEUE_PRIORITIES)
    assert plan.QUEUE_PRIORITIES[plan.q_p3_students("410m")] == 82
    assert plan.QUEUE_PRIORITIES[plan.q_p3_students("160m")] == 81
    assert plan.QUEUE_PRIORITIES[plan.q_p3_students("70m")] == 80
    assert plan.QUEUE_PRIORITIES[plan.q_p3_evals("70m")] == 65
    assert plan.QUEUE_PRIORITIES[plan.q_p3b("70m")] == 70
    assert plan.QUEUE_PRIORITIES[plan.q_p3b_evals("160m")] == 60


def test_gpu_policy_json_is_serialisable_and_lists_eight_devices() -> None:
    policy = plan.gpu_policy_json()
    assert json.loads(json.dumps(policy)) == policy
    assert policy["gpus"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert policy["cap_per_gpu"] == 2
    assert policy["free_mem_mib"] == 12000
    policy["gpus"].append(99)
    assert plan.GPU_POLICY["gpus"] == (0, 1, 2, 3, 4, 5, 6, 7)


def test_wandb_predicates_lock_the_three_headline_cells() -> None:
    assert plan.teacher_logs_wandb(
        token="pythia-160m@step143000",
        tag=plan.TEACHER_PARAM_TAG,
        ftseed=0,
    )
    assert not plan.teacher_logs_wandb(
        token="pythia-160m@step143000",
        tag=plan.TEACHER_PARAM_TAG,
        ftseed=1,
    )
    assert not plan.teacher_logs_wandb(
        token="pythia-70m@step143000",
        tag=plan.TEACHER_PARAM_TAG,
        ftseed=0,
    )
    assert plan.student_logs_wandb(
        teacher="70m",
        student="160m",
        lr=4.0e-4,
        picked_lr=4.0e-4,
        ftseed=0,
        gseed=0,
    )
    assert plan.student_logs_wandb(
        teacher="410m",
        student="160m",
        lr=4.0e-4,
        picked_lr=4.0e-4,
        ftseed=0,
        gseed=0,
    )
    assert not plan.student_logs_wandb(
        teacher="70m",
        student="160m",
        lr=2.0e-4,
        picked_lr=4.0e-4,
        ftseed=0,
        gseed=0,
    )
    assert not plan.student_logs_wandb(
        teacher="160m",
        student="160m",
        lr=4.0e-4,
        picked_lr=4.0e-4,
        ftseed=0,
        gseed=0,
    )
