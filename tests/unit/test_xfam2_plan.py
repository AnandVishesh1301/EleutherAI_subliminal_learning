"""xfam2 plan invariants: cells, units, teachers, queues, packing, storage."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from subliminal_learning.campaign import xfam2_plan as plan


def test_core_cells_are_29_m_5_f_4_q() -> None:
    assert len(plan.CELLS) == 38
    assert Counter(c.block for c in plan.CELLS) == {"M": 29, "F": 5, "Q": 4}
    pairs = {(c.teacher, c.student) for c in plan.CELLS}
    assert len(pairs) == 38
    assert ("pythia-6.9b", "pythia-2.8b") not in pairs
    assert ("pythia-6.9b", "pythia-2.8b") in plan.DROPPED_CELLS
    assert not any(c.student == "pythia-6.9b" for c in plan.CELLS)
    assert sum(c.is_self for c in plan.CELLS) == 5 + 1 + 2


def test_learning_rates_are_frozen_per_student() -> None:
    for c in plan.CELLS:
        want = {"qwen2.5-0.5b": 2e-4, "qwen3-0.6b-base": 8e-4}.get(c.student, 4e-4)
        assert c.lr == want, c
    assert plan.student_gkd_tag(4e-4) == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"


def test_units_teachers_and_lr_check() -> None:
    assert len(plan.LR_CHECK_CELLS) == 10
    assert all(c.is_self and c.lr in (2e-4, 8e-4) for c in plan.LR_CHECK_CELLS)
    assert len(plan.units()) == 86
    assert len(plan.core_units()) == 38
    assert len(plan.teacher_labels()) == 9
    assert len(plan.teacher_cells()) == 27
    tiers = Counter(u.tier for u in plan.units())
    assert tiers == {plan.TIER_CORE: 38, plan.TIER_G2: 38, plan.TIER_LR: 10}
    core = plan.core_units()[0]
    assert core.ftseeds == (0, 1, 2) and core.gseeds == (0, 1)
    assert plan.Unit(core.cell, plan.TIER_G2).gseeds == (2,)
    lr_unit = plan.Unit(plan.LR_CHECK_CELLS[0], plan.TIER_LR)
    assert lr_unit.ftseeds == (0,) and lr_unit.gseeds == (0, 1)


def test_every_queue_has_a_priority_and_a_two_node_home() -> None:
    names = set(plan.QUEUE_PRIORITIES)
    assert len(names) == 2 * 9 + 3 * 86
    assert {plan.queue_node(n) for n in names} == {plan.SOAR, plan.GONCALO}
    on = [q for h in (plan.SOAR, plan.GONCALO) for q in plan.queues_on(h)]
    assert sorted(on) == sorted(names)


def test_teacher_queues_outrank_student_queues() -> None:
    teacher = [
        plan.QUEUE_PRIORITIES[q]
        for label in plan.teacher_labels()
        for q in (plan.q_teachers(label), plan.q_teacher_evals(label))
    ]
    student = [
        plan.QUEUE_PRIORITIES[q] for u in plan.units() for q in plan.unit_queue_names(u)
    ]
    assert min(teacher) > max(student)


def test_tail_queues_are_exactly_the_g2_and_lr_units() -> None:
    want = {
        q
        for u in plan.units()
        if u.tier in (plan.TIER_G2, plan.TIER_LR)
        for q in plan.unit_queue_names(u)
    }
    assert plan.tail_queue_names() == frozenset(want)
    core = [
        plan.QUEUE_PRIORITIES[q]
        for u in plan.core_units()
        for q in plan.unit_queue_names(u)
    ]
    tail = [plan.QUEUE_PRIORITIES[q] for q in want]
    assert min(core) > max(tail)


def test_slots_fit_a_card() -> None:
    for c in (*plan.CELLS, *plan.LR_CHECK_CELLS):
        slots = plan.student_job_slots(c.teacher, c.student)
        assert 1 <= slots <= plan.CAP_PER_GPU, c
    for label in plan.teacher_labels():
        assert 1 <= plan.teacher_job_slots(label) <= plan.CAP_PER_GPU
    plan.validate_footprints()


def test_paths_config_is_all_under_the_campaign_root() -> None:
    doc = plan.paths_config_doc()
    assert set(doc) == {
        "output_root",
        "hf_cache_root",
        "dataset_cache_root",
        "wandb_dir",
        "queue_root",
        "audit_root",
    }
    for value in doc.values():
        assert Path(value).is_relative_to(plan.CAMPAIGN_ROOT)
    env = plan.gpu_policy_json()["env"]
    assert Path(env["HF_HUB_CACHE"]).is_relative_to(plan.CAMPAIGN_ROOT)
    assert env["HF_HUB_OFFLINE"] == "1"


def test_milestones_partition_every_unit_by_block_and_tier() -> None:
    by = Counter(plan.milestone_of(u) for u in plan.units())
    assert set(by) == set(plan.MILESTONES)
    assert by["core_m_matrix"] == 25 and by["core_m_emitter"] == 4
    assert (by["core_f"], by["core_q"]) == (len(plan.F_PAIRS), len(plan.Q_PAIRS))
    assert (by["tail_g2"], by["tail_lr"]) == (len(plan.CELLS), len(plan.LR_CHECK_CELLS))
