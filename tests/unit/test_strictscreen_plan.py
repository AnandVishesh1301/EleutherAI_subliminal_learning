"""strictscreen plan: pairings, cell counts, tags, queues, packing, registered spec."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import pairs
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.campaign.driver_io import CAMPAIGN_SPECS
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag


def test_pairings_equal_stage_b_headline_pairs() -> None:
    headline = {
        (p.student, p.teacher, p.group) for p in pairs.stage_b_pairs() if p.headline
    }
    assert {(p.student, p.teacher, p.group) for p in plan.PAIRINGS} == headline
    assert plan.STUDENTS == (
        "data-seed1",
        "weight-seed1",
        "data-seed2",
        "data-seed3",
        "weight-seed2",
        "weight-seed3",
    )
    assert set(plan.TEACHER_BASES) == {
        "data-seed1",
        "data-seed2",
        "weight-seed1",
        "weight-seed2",
    }


def test_cell_counts_keys_and_units() -> None:
    assert {arm: len(plan.arm_cells(arm)) for arm in plan.ARMS} == {
        "a": 288,
        "b": 72,
        "c": 32,
    }
    cells = plan.all_cells()
    assert len({c.key for c in cells}) == 392
    assert sum(len(u.cells()) for u in plan.UNITS) == 392
    assert all(c.tfs == 0 and c.gseed == 0 for c in plan.arm_cells(plan.ARM_C))
    assert all(c.control and c.tfs is None for c in plan.arm_cells(plan.ARM_B))
    assert plan.arm_cells("a")[0].key == "strictscreen|a|ds1-ds1|t0|bear|g0"
    assert plan.arm_cells("b")[0].key == "strictscreen|b|ds1|t-|bear|g0"
    assert plan.n_eval_config_dirs() == 152
    with pytest.raises(ValueError, match="unknown arm"):
        plan.arm_cells("d")


def test_tags_round_trip_and_recipe_frozen() -> None:
    strict = parse_gkd_tag(plan.STRICT_TAG)
    legacy = parse_gkd_tag(plan.LEGACY_TAG)
    assert strict["rollout_screen"] == "strict" and legacy["rollout_screen"] == "legacy"
    for tag in (strict, legacy):
        assert tag["lmbda"] == 1.0 and tag["beta"] == 1.0 and not tag["vocab_mask"]
        assert (tag["student_lora_r"], tag["student_lora_alpha"]) == (8, 16)
        assert tag["student_lr"] == 4.0e-4 and tag["num_epochs"] == 1.0
    screens = {c.arm: c.gkd_tag for c in plan.all_cells()}
    assert screens == {"a": plan.STRICT_TAG, "b": plan.STRICT_TAG, "c": plan.LEGACY_TAG}


def test_queues_priorities_and_fleet_uniqueness() -> None:
    prios = plan.QUEUE_PRIORITIES
    assert len(prios) == 10 and all(q.startswith(plan.QUEUE_PREFIX) for q in prios)
    by_unit = {u.name: u.train_priority for u in plan.UNITS}
    assert by_unit == {"b": 90, "a_tfs0": 88, "a_tfs1": 86, "a_tfs2": 84, "c": 70}
    assert all(prios[u.eval_queue] == 95 for u in plan.UNITS)
    others = {
        q
        for name, spec in CAMPAIGN_SPECS.items()
        if name != plan.CAMPAIGN
        for q in spec.queue_priorities
    }
    assert others.isdisjoint(prios)


def test_packing_policy_and_budget() -> None:
    cap = plan.GPU_POLICY["cap_per_gpu"]
    assert plan.STUDENT_FOOTPRINT.slots == 1
    assert plan.STUDENT_FOOTPRINT.mem_mib * cap <= 46_000
    assert plan.WORKERS == 24 == plan.total_slots()
    policy = plan.gpu_policy_json()
    assert policy["gpus"] == list(range(8)) and policy["free_mem_mib"] == 5000
    assert policy["reservation_warmup_s"] == 600
    assert policy["env"] == {
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert plan.projected_hours() <= plan.MAX_TRAIN_EVAL_HOURS
    assert dict(plan.SHORTFALL_TOLERANCE) == {"a": 5, "b": 0, "c": None}


def test_log_few_only_arm_a_tfs0_gseed0() -> None:
    logged = [c for c in plan.all_cells() if plan.logs_wandb(c)]
    assert len(logged) == 8 * 4
    assert all(c.arm == "a" and c.tfs == 0 and c.gseed == 0 for c in logged)


def test_registered_spec_and_shared_identity() -> None:
    spec = CAMPAIGN_SPECS["strictscreen"]
    assert spec is plan.SPEC
    assert spec.run_set_id == "20260915_strictscreen_160m"
    assert spec.teacher_run_set_id == "20260809_breadth4-teachers_160m"
    assert spec.eval_prompt_sets == plan.EVAL_PROMPT_SETS
    assert (
        spec.gkd_experiment_config == "configs/experiment/gkd_student_strictscreen.yaml"
    )
    assert (plan.NODE, plan.NODE_ALIAS, plan.LOCK_PREFIX) == (
        "soar-subliminal-3",
        "a40-cluster-4",
        "/tmp/subliminal_jobq_vishesh",
    )


def test_validate_plan_rejects_drifted_pairings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plan, "PAIRINGS", plan.PAIRINGS[:-1])
    with pytest.raises(ValueError, match="headline"):
        plan.validate_plan()
