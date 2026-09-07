"""Stage-2 checkpoint-grid constants: rungs, HP tags, queues, eval bundle."""

from __future__ import annotations

import json
from dataclasses import replace

from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign import driver_io
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.jobqueue.gpu import GpuPolicy
from subliminal_learning.sdf.layout import sdf_param_tag
from subliminal_learning.training.run_set import is_valid_run_set_id

_HP_TAG_LITERALS = (
    "lmb1.0_b1.0_lora_lr1e-4_e1_r8a16",
    "lmb1.0_b1.0_lora_lr8e-4_e1_r8a16",
    "lmb1.0_b1.0_lora_lr1.6e-3_e1_r8a16",
    "lmb1.0_b1.0_lora_lr4e-4_e1_r32a64",
    "lmb1.0_b1.0_lora_lr4e-4_e1_r128a256",
    "lmb1.0_b1.0_lora_lr4e-4_e1_r8a32",
    "lmb1.0_b1.0_lora_lr4e-4_e3_r8a16",
)


def _is_published_step(step: int) -> bool:
    """``step0,1,2,4,...,512`` then every 1000 up to ``step143000``."""
    return step in plan.CHECKPOINT_LOG_HEAD or (
        step % plan.CHECKPOINT_STEP_INTERVAL == 0
        and 0 < step <= plan.CHECKPOINT_MAX_STEP
    )


def test_rungs_are_strictly_increasing_published_and_exclude_zero() -> None:
    for ladder in (
        plan.JOINT_RUNGS,
        plan.HP_RUNGS,
        plan.DOSE_RUNGS,
        plan.CAT_RUNGS,
    ):
        assert len(set(ladder)) == len(ladder)
        assert all(a < b for a, b in zip(ladder, ladder[1:]))
        assert 0 not in ladder
        assert all(_is_published_step(r) for r in ladder)
    assert plan.RECT512_STUDENT_RUNG in plan.HP_RUNGS
    assert plan.RECT512_STUDENT_RUNG not in plan.JOINT_RUNGS


def test_hp_config_tags_are_the_seven_literals_and_not_frozen() -> None:
    assert tuple(cfg.tag for cfg in plan.HP_CONFIGS) == _HP_TAG_LITERALS
    assert plan.FROZEN_GKD_TAG == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    assert plan.FROZEN_GKD_TAG not in _HP_TAG_LITERALS
    assert all(tag != plan.FROZEN_GKD_TAG for tag in _HP_TAG_LITERALS)


def test_teacher_param_tag_matches_sdf_layout() -> None:
    train_base = TrainConfig(
        learning_rate=plan.TEACHER_LR,
        n_epochs=plan.TEACHER_EPOCHS_FROZEN,
        batch_size=8,
        max_length=2048,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=0,
    )
    lora_base = LoRAConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=(
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ),
    )
    for n in (*plan.DOSE_EPOCHS, plan.TEACHER_EPOCHS_FROZEN):
        train = replace(train_base, n_epochs=n, learning_rate=plan.TEACHER_LR, seed=0)
        lora = replace(
            lora_base,
            r=plan.TEACHER_LORA_R,
            lora_alpha=plan.TEACHER_LORA_ALPHA,
        )
        assert plan.teacher_param_tag(n) == sdf_param_tag(train, lora)


def test_bases_self_consistency_wave2_iff_audit_required() -> None:
    assert len(plan.BASES) == 6
    for label, desc in plan.BASES.items():
        assert desc.label == label
        assert (desc.wave == 2) == desc.audit_required
        if "weight-seed" in label:
            assert desc.wave == 2
    used_wave1 = {"original", "seed2", "data-seed1"}
    present_wave1 = {d.label for d in plan.BASES.values() if d.wave == 1}
    assert used_wave1 <= present_wave1
    assert not any(
        d.wave == 1 and "weight-seed" in d.label for d in plan.BASES.values()
    )


def test_new_hub_refs_is_seed2_at_512() -> None:
    assert plan.NEW_HUB_REFS == (("seed2", 512),)
    assert plan.NEW_HUB_REFS == (("seed2", plan.RECT512_STUDENT_RUNG),)


def test_queue_priorities_match_documented_dag_numbers() -> None:
    p = plan.QUEUE_PRIORITIES
    assert p[plan.Q_P1_TEACHERS] == 100
    assert p[plan.Q_P1_TEACHER_EVALS] == 95
    assert p[plan.Q_T1_STUDENTS] == 90
    assert p[plan.Q_T1_CONTROLS] == 90
    assert p[plan.Q_T1_CONTROLS_ESC] == 90
    assert p[plan.Q_DOSE_STUDENTS] == 85
    assert p[plan.Q_T15_STUDENTS] == 80
    assert p[plan.Q_EVALS_T1] == 70
    assert p[plan.Q_EVALS_DOSE] == 70
    assert p[plan.Q_EVALS_T15] == 70
    assert p[plan.Q_EVALS_T2] == 70
    assert p[plan.Q_T2_STUDENTS] == 60
    assert p[plan.Q_E_BACKFILL] == 50
    assert p[plan.Q_F_BACKFILL] == 50
    assert p[plan.Q_W2_TEACHERS] == 40
    assert p[plan.Q_W2_TEACHER_EVALS] == 40
    assert p[plan.Q_W2_STUDENTS] == 40
    assert p[plan.Q_W2_CONTROLS] == 40
    assert p[plan.Q_W2_EVALS] == 40
    # Evals sit between t15 and t2; a t15 > t2 > evals chain is wrong.
    assert p[plan.Q_T15_STUDENTS] > p[plan.Q_EVALS_T1] > p[plan.Q_T2_STUDENTS]
    queues = [getattr(plan, name) for name in dir(plan) if name.startswith("Q_")]
    assert set(queues) == set(plan.QUEUE_PRIORITIES)
    assert all(q.startswith("ckptgrid2-") for q in queues)


def test_claim_ids_are_h1_through_h12() -> None:
    assert plan.CLAIM_IDS == tuple(f"H{i}" for i in range(1, 13))
    assert len(set(plan.CLAIM_IDS)) == 12


def test_identity_and_spec_bind_this_campaign() -> None:
    assert plan.RUN_SET_ID == "20260908_ckptgrid2-bear_160m"
    assert is_valid_run_set_id(plan.RUN_SET_ID)
    assert plan.TEACHER_RUN_SET_ID == plan.RUN_SET_ID
    assert dict(plan.REFERENCE_RUN_SETS) == {
        "bear_grid": "20260828_ckptgrid-bear_160m",
        "cat_ladder": "20260805_ckptladder_160m",
    }
    assert plan.HUB_ORG == "EleutherAI"
    assert plan.MODEL_SIZE == "160m"
    assert plan.OWNER == "vishesh"
    assert plan.ANIMALS == ("bear", "cat")
    assert plan.PROMPT_POOL_DIR == "gkd_prompts/pile_prompts_v1"
    assert plan.TEACHER_WORKFLOW == "sdf"
    assert plan.TEACHER_MODE == "lora"
    assert plan.SPEC.run_set_id == plan.RUN_SET_ID
    assert plan.SPEC.eval_prompt_sets == plan.EVAL_PROMPT_SETS
    assert plan.SPEC.queue_priorities == plan.QUEUE_PRIORITIES
    assert plan.SPEC.gkd_experiment_config == plan.STUDENT_EXPERIMENT_CONFIG
    assert plan.SPEC.generated_config_dir == plan.GENERATED_CONFIG_DIR
    assert plan.SPEC.teacher_param_tag_lora == plan.teacher_param_tag(
        plan.TEACHER_EPOCHS_FROZEN
    )
    assert plan.WANDB_PROJECT == "subliminal-learning-ckptgrid2-bear"
    assert "WANDB_PROJECT" not in dict(plan.GPU_POLICY["env"])
    assert driver_io.CAMPAIGN_SPECS["ckptgrid2"] is plan.SPEC


def test_wandb_predicates_fire_exactly_the_four_headline_cells() -> None:
    hp_tag = next(cfg.tag for cfg in plan.HP_CONFIGS if cfg.recipe.lr == 1.6e-3)
    frozen_teacher = plan.teacher_param_tag(plan.TEACHER_EPOCHS_FROZEN)
    ep1 = plan.teacher_param_tag(1)
    ep8 = plan.teacher_param_tag(8)

    assert plan.teacher_logs_wandb(token="original@step8000", tag=ep8, ftseed=0)
    assert not plan.teacher_logs_wandb(token="original@step8000", tag=ep8, ftseed=1)
    assert not plan.teacher_logs_wandb(token="original@step8000", tag=ep1, ftseed=0)
    assert not plan.teacher_logs_wandb(token="original@step143000", tag=ep8, ftseed=0)

    assert plan.student_logs_wandb(
        arm="hp",
        student="original@step143000",
        teacher="original@step143000",
        gkd_tag=hp_tag,
        teacher_param_tag=frozen_teacher,
        replicate=0,
    )
    assert not plan.student_logs_wandb(
        arm="hp",
        student="original@step143000",
        teacher="original@step143000",
        gkd_tag=hp_tag,
        teacher_param_tag=frozen_teacher,
        replicate=1,
    )
    assert not plan.student_logs_wandb(
        arm="hp",
        student="original@step143000",
        teacher="original@step143000",
        gkd_tag=hp_tag,
        teacher_param_tag=frozen_teacher,
        replicate=1,
    )

    assert plan.student_logs_wandb(
        arm="rect",
        student="original@step143000",
        teacher="seed2@step8000",
        gkd_tag=plan.FROZEN_GKD_TAG,
        teacher_param_tag=frozen_teacher,
        replicate=0,
    )
    assert not plan.student_logs_wandb(
        arm="rect",
        student="original@step143000",
        teacher="seed2@step8000",
        gkd_tag=plan.FROZEN_GKD_TAG,
        teacher_param_tag=frozen_teacher,
        replicate=1,
    )

    assert plan.student_logs_wandb(
        arm="dose",
        student="original@step8000",
        teacher="original@step8000",
        gkd_tag=plan.FROZEN_GKD_TAG,
        teacher_param_tag=ep1,
        replicate=0,
    )
    assert not plan.student_logs_wandb(
        arm="dose",
        student="original@step8000",
        teacher="original@step8000",
        gkd_tag=plan.FROZEN_GKD_TAG,
        teacher_param_tag=ep1,
        replicate=1,
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


def test_claim_thresholds_are_frozen_and_internally_consistent() -> None:
    """Every threshold the prereg names exists here, not in a builder default."""
    assert plan.ONSET_ABS_FLOOR == plan.ADMISSION_ABS_MIN == 0.02
    assert plan.ONSET_SD_MULTIPLE == 2.0
    assert plan.ONSET_CONSECUTIVE == 2
    assert plan.ONSET_MARKERS == (0.0022, 0.01)
    assert plan.ADMISSION_SD_MULTIPLE == 5.0
    assert plan.DOSE_MATCH_TOL == plan.DOSE_TOST_BOUND == 0.05
    assert plan.CONCORDANCE_MIN_SPEARMAN == 0.6
    assert plan.CONCORDANCE_MIN_SIGN_AGREEMENT == 0.8
    assert plan.HP_CONTROL_ESCALATION_MAX == 0.02
    assert plan.QUARANTINE_PPL_RATIO_MAX == 5.0
    assert plan.QUARANTINE_FILTER_DROP_MAX == 0.5
    assert plan.MIN_DISK_FREE_GIB_COLD == 150
    assert plan.MIN_DISK_FREE_GIB_SSD == 30


def test_ridge_era_edge_splits_the_teacher_columns_both_ways() -> None:
    """H1 is scored per era, so both eras must be non-empty on JOINT_RUNGS."""
    early = [r for r in plan.JOINT_RUNGS if r <= plan.RIDGE_ERA_EARLY_MAX_STEP]
    late = [r for r in plan.JOINT_RUNGS if r > plan.RIDGE_ERA_EARLY_MAX_STEP]
    assert early and late
    assert len(early) + len(late) == len(plan.JOINT_RUNGS)
    assert plan.RIDGE_NULL_MAX_COLUMNS < plan.RIDGE_SUPPORT_MIN_COLUMNS
    assert plan.RIDGE_SUPPORT_MIN_COLUMNS <= len(plan.JOINT_RUNGS)


def test_asymmetry_era_edges_are_increasing_rungs() -> None:
    assert list(plan.ASYMMETRY_ERA_EDGES) == sorted(set(plan.ASYMMETRY_ERA_EDGES))
    assert all(edge > 0 for edge in plan.ASYMMETRY_ERA_EDGES)


def test_ridge_column_rules_are_exact_integers_for_both_animals() -> None:
    """Bear rectangles have 6 teacher columns, the cat grid has 7."""
    assert plan.ridge_support_min_columns(6) == 4
    assert plan.ridge_support_min_columns(7) == 5
    assert plan.ridge_null_max_columns(6) == 1
    assert plan.ridge_null_max_columns(7) == 1
    assert plan.RIDGE_SUPPORT_MIN_COLUMNS == plan.ridge_support_min_columns(
        len(plan.JOINT_RUNGS)
    )
    assert plan.RIDGE_NULL_MAX_COLUMNS == plan.ridge_null_max_columns(
        len(plan.JOINT_RUNGS)
    )
    for n in range(2, 13):
        assert 0 < plan.ridge_null_max_columns(n) < plan.ridge_support_min_columns(n)
        assert plan.ridge_support_min_columns(n) <= n
