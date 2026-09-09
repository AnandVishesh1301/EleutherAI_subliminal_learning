"""Top-K campaign constants: tags, rungs, blocks, footprints, queues, W&B."""

from __future__ import annotations

import json

from subliminal_learning.campaign import topk_plan as plan
from subliminal_learning.campaign import xsize_plan
from subliminal_learning.training.run_set import is_valid_run_set_id


def test_run_set_id_is_valid_and_inherits_xsize_teachers() -> None:
    assert plan.RUN_SET_ID == "20260910_topk-bear_mixed"
    assert is_valid_run_set_id(plan.RUN_SET_ID)
    assert plan.TEACHER_RUN_SET_ID == "20260907_xsize-bear_mixed"
    assert plan.MODEL_SIZE == "mixed"
    assert plan.ANIMALS == ("bear",)
    assert plan.NULL_ANIMAL == "cat"
    assert plan.STUDENT_SIZES == ("160m", "410m")
    assert plan.STUDENT_LR == 4.0e-4
    assert plan.TEACHER_PARAM_TAG == xsize_plan.TEACHER_PARAM_TAG


def test_rung_tags_match_the_frozen_literals() -> None:
    assert plan.RUNG_FULL.gkd_tag == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    assert plan.RUNG_FULL.gkd_tag == xsize_plan.student_gkd_tag(4.0e-4)
    assert plan.Rung(100).gkd_tag == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_top100"
    assert plan.Rung(5, ghost=True).gkd_tag == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_top5g"
    assert plan.Rung(1000).slug == "k1000"
    assert plan.Rung(5, ghost=True).slug == "k5g"
    assert plan.RUNG_FULL.slug == "full"
    assert plan.RUNG_FULL.k_plot == plan.VOCAB_SIZE == 50304
    assert plan.FULL_GKD_TAG == plan.RUNG_FULL.gkd_tag


def test_ladder_and_blocks_follow_the_prereg() -> None:
    assert plan.K_CORE == (100, 10, 5, 1000)
    assert plan.K_TAIL == (30, 2)
    assert plan.GHOST_K == (5,)
    assert plan.TAIL_CROSS_PAIRS == ((("70m", "410m"), (100, 5)),)
    assert [b.name for b in plan.BLOCKS] == [
        "k100",
        "k10",
        "k5",
        "k1000",
        "full",
        "k30",
        "x70m",
        "k5g",
        "k2",
    ]
    assert [b.name for b in plan.core_blocks()] == [
        "k100",
        "k10",
        "k5",
        "k1000",
        "full",
    ]
    assert [b.name for b in plan.tail_blocks()] == ["k30", "x70m", "k5g", "k2"]
    assert plan.block("x70m").sizes == ("410m",)
    assert plan.block("k5g").rungs == (plan.Rung(5, ghost=True),)
    assert plan.block("full").pairs == (plan.HEADLINE_PAIR,)
    assert plan.block("k100").sizes == ("160m", "410m")


def test_footprints_validate_and_only_410m_resumes() -> None:
    plan.validate_footprints()
    assert set(plan.STUDENT_FOOTPRINT) == set(plan.STUDENT_SIZES)
    for size, fp in plan.STUDENT_FOOTPRINT.items():
        assert fp.micro_batch * fp.grad_accum == plan.STUDENT_EFFECTIVE_BATCH
        assert fp.slots <= plan.GPU_POLICY["cap_per_gpu"]
        assert fp.mem_mib <= plan.GPU_MEM_MIB
        if fp.checkpoint_resume:
            assert size == "410m" and fp.save_steps > 0
    assert plan.STUDENT_FOOTPRINT["410m"].est_minutes == 68
    assert plan.STUDENT_FOOTPRINT["160m"].est_minutes == 28
    assert plan.EAS_FOOTPRINT.slots == 1
    assert all(fp.slots == 1 for fp in plan.STUDENT_FOOTPRINT.values())
    assert all(fp.slots == 1 for fp in plan.TEACHER_FOOTPRINT.values())
    assert plan.GPU_POLICY["env"]["OMP_NUM_THREADS"] == "1"
    assert not any(fp.checkpoint_resume for fp in plan.TEACHER_FOOTPRINT.values())


def test_queue_priorities_cover_every_block_and_tail_is_lowest() -> None:
    names = {plan.Q_NULL_TEACHERS, plan.Q_NULL_TEACHER_EVALS}
    for b in plan.BLOCKS:
        names.update(plan.block_queue_names(b.name))
    assert names == set(plan.QUEUE_PRIORITIES)
    assert plan.QUEUE_PRIORITIES[plan.Q_NULL_TEACHERS] == 95
    assert plan.QUEUE_PRIORITIES[plan.q_students("k100", "410m")] == 90
    assert plan.QUEUE_PRIORITIES[plan.q_students("k1000", "410m")] == 87
    assert plan.QUEUE_PRIORITIES[plan.q_students("k100", "160m")] == 80
    assert plan.QUEUE_PRIORITIES[plan.q_students("full", "410m")] == 86
    assert plan.QUEUE_PRIORITIES[plan.q_evals("k5", "410m")] == 70
    assert plan.QUEUE_PRIORITIES[plan.q_eas("k5", "410m")] == 65
    assert plan.QUEUE_PRIORITIES[plan.q_students("k30", "410m")] == 60
    assert plan.QUEUE_PRIORITIES[plan.q_students("k2", "160m")] == 54
    assert plan.QUEUE_PRIORITIES[plan.q_evals("k2", "410m")] == 50
    assert plan.QUEUE_PRIORITIES[plan.q_eas("k2", "410m")] == 45
    tail = plan.tail_queue_names()
    core = {n for b in plan.core_blocks() for n in plan.block_queue_names(b.name)}
    assert tail.isdisjoint(core)
    assert max(plan.QUEUE_PRIORITIES[n] for n in tail) < min(
        plan.QUEUE_PRIORITIES[plan.q_students(b.name, s)]
        for b in plan.core_blocks()
        for s in b.sizes
    )


def test_gpu_policy_and_deadline() -> None:
    policy = plan.gpu_policy_json()
    assert json.loads(json.dumps(policy)) == policy
    assert policy["gpus"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert policy["cap_per_gpu"] == 3
    assert plan.total_slots() == 24
    assert plan.DEADLINE_HOURS == 19.5
    assert plan.AUTO_REQUEUE_ATTEMPTS == 2


def test_wandb_predicates_lock_exactly_eleven_cells() -> None:
    assert plan.WANDB_PROJECT == "tok-k_cross_size_opd"
    logged = []
    for b in plan.BLOCKS:
        for rung in b.rungs:
            for teacher, student in b.pairs:
                for tfs in (0, 1, 2):
                    for g in (0, 1, 2):
                        if plan.student_logs_wandb(
                            teacher=teacher,
                            student=student,
                            rung=rung,
                            ftseed=tfs,
                            gseed=g,
                        ):
                            logged.append((b.name, rung.slug, teacher, student, tfs, g))
    assert len(logged) == 11
    assert all(tfs == 0 and g == 0 for *_, tfs, g in logged)
    assert sum(1 for x in logged if (x[2], x[3]) == plan.HEADLINE_PAIR) == 8
    assert sum(1 for x in logged if (x[2], x[3]) == plan.ANCHORS[0]) == 2
    assert sum(1 for x in logged if (x[2], x[3]) == ("70m", "410m")) == 1
    assert not plan.student_logs_wandb(
        teacher="160m", student="160m", rung=plan.Rung(5), ftseed=0, gseed=0
    )


def test_spec_binds_run_set_and_student_yaml() -> None:
    assert plan.SPEC.run_set_id == plan.RUN_SET_ID
    assert plan.SPEC.teacher_run_set_id == plan.TEACHER_RUN_SET_ID
    assert plan.SPEC.gkd_experiment_config == plan.STUDENT_EXPERIMENT_CONFIG
    assert plan.SPEC.queue_priorities == plan.QUEUE_PRIORITIES
    assert (
        plan.SPEC.eval_prompt_sets
        == plan.EVAL_PROMPT_SETS
        == ("test", "favorite", "off_topic")
    )
    assert "scripts/compute_gkd_eas.py" in plan.SPEC.extra_pin_paths
    assert plan.CLAIM_IDS == ("T0", "T1", "T2", "T3", "T4", "M1", "M2", "M3")
    assert plan.BOOTSTRAP_SEED == 20260910
