"""Checkpoint-ladder constants: rung schedule, inits, frozen tags, gate thresholds.

Every number here is pre-registered in
``.cursor/plans/checkpoint_ladder_campaign_fdd5029a.plan.md``. A rung that is not
a published Pythia revision fails only after the Hub 404s mid-campaign; a ladder
that drifts from the prereg silently re-scopes C1-C9. Both are locked offline,
against the schedule rule rather than the network.
"""

from __future__ import annotations

import json

import pytest

from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.jobqueue.gpu import GpuPolicy

# Plan §2, transcribed once so a ladder edit has to be a deliberate two-file change.
PLAN_LADDER_G = (
    0,
    8,
    32,
    128,
    256,
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    24000,
    32000,
    48000,
    64000,
    80000,
    96000,
    112000,
    128000,
    143000,
)
PLAN_LADDER_B = (
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    48000,
    64000,
    96000,
    128000,
    143000,
)


# ---- ladders (plan §2) ----------------------------------------------------------


def test_ladder_g_is_the_twenty_preregistered_geometry_rungs() -> None:
    assert plan.LADDER_G == PLAN_LADDER_G
    assert len(plan.LADDER_G) == 20


def test_ladder_b_is_the_twelve_preregistered_behaviour_rungs() -> None:
    assert plan.LADDER_B == PLAN_LADDER_B
    assert len(plan.LADDER_B) == 12


@pytest.mark.parametrize("ladder", [PLAN_LADDER_G, PLAN_LADDER_B])
def test_ladder_is_strictly_increasing_with_no_duplicates(
    ladder: tuple[int, ...],
) -> None:
    assert len(set(ladder)) == len(ladder)
    assert all(a < b for a, b in zip(ladder, ladder[1:]))


def test_ladder_b_is_a_subset_of_ladder_g() -> None:
    assert set(plan.LADDER_B) < set(plan.LADDER_G)


def test_step0_is_excluded_from_the_behaviour_ladder() -> None:
    """A random-init model cannot be an SDF teacher; step0 is Ladder G's anchor."""
    assert 0 in plan.LADDER_G
    assert 0 not in plan.LADDER_B


# ---- the published checkpoint schedule (plan §1, §2) -----------------------------


def test_checkpoint_log_head_is_the_published_power_of_two_prefix() -> None:
    """Without this, the schedule rule below would accept any step it is given."""
    assert plan.CHECKPOINT_LOG_HEAD == (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert plan.CHECKPOINT_LOG_HEAD[1:] == tuple(2**k for k in range(10))
    assert plan.CHECKPOINT_STEP_INTERVAL == 1000
    assert plan.CHECKPOINT_MAX_STEP == 143000


def _is_published_step(step: int) -> bool:
    """``step0,1,2,4,...,512`` then every 1000 up to ``step143000`` (154 branches)."""
    return step in plan.CHECKPOINT_LOG_HEAD or (
        step % plan.CHECKPOINT_STEP_INTERVAL == 0
        and step <= plan.CHECKPOINT_MAX_STEP
    )


@pytest.mark.parametrize("rung", PLAN_LADDER_G)
def test_every_ladder_rung_is_a_published_pythia_checkpoint_step(rung: int) -> None:
    """A rung off the schedule 404s mid-campaign, not at plan time."""
    assert _is_published_step(rung)


@pytest.mark.parametrize("rung", [3, 700, 1500, 144000])
def test_a_step_off_the_published_schedule_fails_the_rule(rung: int) -> None:
    """The rule has to reject something, or the test above is vacuous."""
    assert not _is_published_step(rung)


# ---- inits and rung subsets ------------------------------------------------------


def test_inits_are_the_original_plus_three_seeds_of_each_ablation_family() -> None:
    assert len(plan.INITS) == 7
    assert plan.INITS[0] == "original"
    data = [i for i in plan.INITS if i.startswith("data-seed")]
    weight = [i for i in plan.INITS if i.startswith("weight-seed")]
    assert data == ["data-seed1", "data-seed2", "data-seed3"]
    assert weight == ["weight-seed1", "weight-seed2", "weight-seed3"]
    assert len(set(plan.INITS)) == 7


def test_crossed_anchor_rungs_are_two_behaviour_rungs() -> None:
    assert plan.CROSSED_ANCHOR_RUNGS == (2000, 64000)
    assert set(plan.CROSSED_ANCHOR_RUNGS) < set(plan.LADDER_B)


@pytest.mark.parametrize(
    "name",
    ["BENCHMARK_RUNGS", "WEIGHT_SPACE_RUNGS", "DOG_RUNGS", "CROSSED_ANCHOR_RUNGS"],
)
def test_every_rung_subset_is_drawn_from_the_behaviour_ladder(name: str) -> None:
    """These stages read trained leaves, which only exist at a Ladder-B rung."""
    subset = getattr(plan, name)
    assert subset
    assert set(subset) <= set(plan.LADDER_B)


def test_benchmark_rungs_are_distinct_from_the_crossed_variance_anchors() -> None:
    """Plan §7.1: capability probes and the replication device are separate axes."""
    assert plan.BENCHMARK_RUNGS == (512, 2000, 16000, 64000, 143000)
    assert plan.BENCHMARK_RUNGS != plan.CROSSED_ANCHOR_RUNGS


def test_replication_seeds_are_three_ftseeds_and_three_gseeds() -> None:
    assert plan.FTSEEDS == (0, 1, 2)
    assert plan.GSEEDS == (0, 1, 2)


# ---- frozen recipe and eval bundle (plan §3.4, §7.1) -----------------------------


def test_student_gkd_tag_renders_the_frozen_literal() -> None:
    """`logs/freeze_report.json`'s tag; a tag-format change re-roots every leaf."""
    assert plan.STUDENT_GKD_TAG == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"


def test_teacher_recipe_is_the_frozen_lora_tag() -> None:
    assert plan.TEACHER_PARAM_TAG_LORA == "ep5_lr5e-5_r128a128"
    assert (plan.TEACHER_EPOCHS, plan.TEACHER_LR) == (5, 5.0e-5)
    assert (plan.TEACHER_LORA_R, plan.TEACHER_LORA_ALPHA) == (128, 128)


def test_spec_eval_prompt_sets_is_the_four_set_bundle() -> None:
    assert plan.EVAL_PROMPT_SETS == ("test", "favorite", "storytelling", "off_topic")
    assert plan.SPEC.eval_prompt_sets == plan.EVAL_PROMPT_SETS
    assert plan.PRIMARY_PROMPT_SET == "test"


def test_animal_scope_is_cat_only_with_dog_deferred() -> None:
    assert plan.ANIMALS == ("cat",)
    assert plan.DOG_ANIMALS == ("dog",)


# ---- slot conventions (plan §6.2) ------------------------------------------------


def test_slot_conventions_are_the_two_distinct_documented_values() -> None:
    """Slots 0-11 agree and slot 12 does not, so a collapsed pair is silent."""
    assert plan.SLOT_CONVENTION == "post_block"
    assert plan.TRAIT_SLOT_CONVENTION == "hidden_states"
    assert plan.SLOT_CONVENTION != plan.TRAIT_SLOT_CONVENTION


# ---- campaign identity and gate thresholds (plan §6.10, §7) ----------------------


def test_spec_carries_this_campaigns_run_set_ids() -> None:
    assert plan.RUN_SET_ID == "20260805_ckptladder_160m"
    assert plan.SPEC.run_set_id == plan.RUN_SET_ID
    assert plan.SPEC.teacher_run_set_id == plan.TEACHER_RUN_SET_ID == plan.RUN_SET_ID
    assert plan.GEOM_RUN_SET_ID != plan.RUN_SET_ID
    assert plan.STAGE_B_RUN_SET_ID != plan.RUN_SET_ID


def test_preregistered_gate_thresholds_are_the_plan_numbers() -> None:
    assert plan.MIN_DISK_FREE_GIB == 150
    assert plan.MIN_OK_RUNGS == 6
    assert plan.ADMISSION_SD_MULTIPLE == 5.0
    assert plan.ADMISSION_ABS_MIN == 0.02
    assert plan.DIAL_MIN_DIVERGENCE_SPAN == 0.10
    assert plan.DIAL_MIN_FAMILY_SEPARATION == 0.05


def test_min_ok_rungs_leaves_room_below_the_full_behaviour_ladder() -> None:
    """A raise that can never fire is not a gate; one that always fires is a block."""
    assert 1 < plan.MIN_OK_RUNGS < len(plan.LADDER_B)


def test_claim_ids_are_c1_through_c9() -> None:
    assert plan.CLAIM_IDS == ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")


# ---- queues, policy, artifacts ---------------------------------------------------


def test_every_campaign_queue_has_a_priority_and_a_unique_name() -> None:
    queues = [
        getattr(plan, name) for name in dir(plan) if name.startswith("Q_P")
    ]
    assert len(queues) == 14
    assert len(set(queues)) == len(queues)
    assert set(queues) == set(plan.QUEUE_PRIORITIES)
    assert all(q.startswith("ckptladder-") for q in queues)


def test_queue_priorities_follow_the_stage_dag() -> None:
    """An upstream stage that gates a downstream one must drain first."""
    priority = plan.QUEUE_PRIORITIES
    assert priority[plan.Q_P1_GEOMETRY] > priority[plan.Q_P2_TEACHERS]
    assert priority[plan.Q_P2_TEACHERS] > priority[plan.Q_P3_STUDENTS]
    assert priority[plan.Q_P3_STUDENTS] > priority[plan.Q_P4_EVALS]
    assert priority[plan.Q_P4_EVALS] > priority[plan.Q_P6_TEACHERS]


def test_gpu_policy_json_is_a_serialisable_copy_jobq_can_load() -> None:
    """`json.dumps` rejects a mapping proxy, so the conversion must be complete."""
    policy = plan.gpu_policy_json()
    assert json.loads(json.dumps(policy)) == policy
    loaded = GpuPolicy.from_dict(policy)
    assert loaded.gpus == [0, 1, 2, 3, 4, 5, 6, 7]
    assert loaded.cap_per_gpu == 2
    assert loaded.env == {"OMP_NUM_THREADS": "4"}


def test_gpu_policy_json_cannot_mutate_the_frozen_policy() -> None:
    policy = plan.gpu_policy_json()
    policy["gpus"].append(99)
    policy["env"]["OMP_NUM_THREADS"] = "64"
    assert plan.GPU_POLICY["gpus"] == (0, 1, 2, 3, 4, 5, 6, 7)
    assert dict(plan.GPU_POLICY["env"]) == {"OMP_NUM_THREADS": "4"}


def test_worker_threads_oversubscribe_the_measured_slot_count() -> None:
    slots = len(plan.GPU_POLICY["gpus"]) * int(plan.GPU_POLICY["cap_per_gpu"])
    assert plan.WORKER_THREADS == plan.WORKER_OVERSUBSCRIBE * slots


def test_result_filenames_are_distinct_and_typed_by_extension() -> None:
    names = [
        getattr(plan, name) for name in dir(plan) if name.startswith("RESULTS_")
    ]
    assert len(names) == 15
    assert len(set(names)) == len(names)
    assert all(n.endswith((".csv", ".json")) for n in names)
