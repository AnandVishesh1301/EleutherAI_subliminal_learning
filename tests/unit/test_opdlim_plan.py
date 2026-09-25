"""OPD-limits plan, pairs, configs and jobs: counts, budget, identities, commands."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from subliminal_learning.campaign import expected
from subliminal_learning.campaign import opdlim_config_gen as cgen
from subliminal_learning.campaign import opdlim_jobs as jobs
from subliminal_learning.campaign import opdlim_pairs as pairs
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.campaign.driver_io import CAMPAIGN_SPECS
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag

TIERS = (plan.TIER1, plan.TIER2, plan.TIER3, plan.TIER4)


def test_cell_counts_and_budget_match_the_plan() -> None:
    assert plan.HP_P1_LR_IN_TIER2 is True
    assert plan.tier_cells(plan.TIER1) == 414
    assert plan.tier_cells(plan.TIER2) == 453
    assert {b.name for b in plan.TIER2.blocks} >= {
        "hp-p1-r16a32",
        "hp-p1-lr8e-4",
        "hp-p1-lr2e-4",
    }
    for t in TIERS:
        assert plan.tier_hours(t) <= plan.MAX_TIER_HOURS
        assert len(pairs.all_leaves(t)) + len(pairs.all_controls(t)) == plan.tier_cells(
            t
        )


def test_tier4_is_tier1_p3_with_only_the_vocab_mask_changed(tmp_path: Path) -> None:
    t4, t1 = plan.TIER4, plan.TIER1
    assert plan.tier_cells(t4) == 12
    assert t4.node == plan.NODE_SOAR and t4.lock_prefix is None
    assert plan.TIER2.n in plan.FINISHED_TIERS and t4.n not in plan.FINISHED_TIERS
    assert [b.name for b in t4.blocks] == ["vmaskf-k10", "vmaskf-k1"]
    by_tag = {
        (lf.teacher_tag, lf.gkd_tag.removesuffix("_vmaskf"), lf.gseed): lf
        for lf in pairs.all_leaves(t1)
        if lf.pair is plan.P3
    }
    for lf in pairs.all_leaves(t4):
        assert lf.pair is plan.P3 and "_vmaskf_topce" in lf.gkd_tag
        base = lf.gkd_tag.replace("_vmaskf", "")
        assert (lf.teacher_tag, base, lf.gseed) in by_tag
    for b in t4.blocks:
        (obj,) = b.objectives
        assert obj.vmask_forms and obj.k in plan.K_VMASK_ORDER
        assert b.hp == plan.HP_BASE and b.teacher_tag == plan.TEACHER_TAG_EP5
        assert b.cells == ((plan.P3, plan.FULL_ANCHOR),)
        t1_block = t1.block(f"ce-k{obj.k}")
        assert (plan.P3, plan.FULL_ANCHOR) in t1_block.cells
        assert t1_block.hp == b.hp and t1_block.teacher_tag == b.teacher_tag


def test_tier3_lambda0_ladder_layout() -> None:
    t = plan.TIER3
    assert plan.tier_cells(t) == 141
    assert sum(b.n_cells() for b in t.core_blocks()) == 77
    assert plan.tier_hours(t) <= plan.MAX_TIER_HOURS
    assert (
        t.node == plan.NODE_GONCALO and t.lock_prefix == "/tmp/subliminal_jobq_angela"
    )
    assert t.blocks[0].name == "lmb0-data-p4-n20000"
    objs = [o for b in t.blocks for o in b.objectives]
    assert {o.lmbda for o in objs} == {0.0}
    assert {o.subset_n for o in objs if not o.subset_seed} == set(plan.N_LADDER_T3)
    assert min(plan.N_LADDER_T3) >= 2 * plan.STUDENT_EFFECTIVE_BATCH
    d1 = plan.Objective("ce", None, lmbda=0.0, subset_n=64, subset_seed=1)
    assert d1.slug == "lmb0-full-n64d1"
    assert plan.student_gkd_tag(d1) == "lmb0.0_b0.0_lora_lr4e-4_s625_r8a16_n64d1"
    assert d1.sweep_keys() == {
        "prompt_subset_size": 64,
        "max_steps": 625,
        "prompt_subset_seed": 1,
    }
    assert plan.Objective("ce", None, lmbda=0.0, subset_n=64).sweep_keys() == {
        "prompt_subset_size": 64,
        "max_steps": 625,
    }
    with pytest.raises(ValueError):
        plan.Objective("ce", None, subset_seed=1)
    pairs.validate_tier(t)
    docs = cgen.all_train_configs(t)
    doc = docs["opdlim3_lmb0-data-p4-n64d1_lmb0-full-n64d1_students_P4.yaml"]
    assert doc["prompt_subset_seed"] == 1 and doc["grid"]["lmbda"] == [0.0]
    assert doc["max_steps"] == plan.DATA_MAX_STEPS
    assert plan.SPECS[3].run_set_id == "20260922_opdlim3-bear_mixed"


def test_pre_contingency_layout_and_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan, "HP_P1_LR_IN_TIER2", False)
    monkeypatch.setattr(plan, "LMB0_COST_FACTOR", 1.0)
    t1 = replace(plan.TIER1, blocks=plan._tier1_blocks())
    t2 = replace(plan.TIER2, blocks=plan._tier2_blocks())
    assert (plan.tier_cells(t1), plan.tier_cells(t2)) == (444, 423)
    assert plan.tier_hours(t1) <= plan.MAX_TIER_HOURS
    assert plan.tier_hours(t2) <= plan.MAX_TIER_HOURS
    monkeypatch.setattr(plan, "HP_P1_LR_IN_TIER2", True)
    with pytest.raises(ValueError, match="hp-p1"):
        plan.validate_plan()


def test_bench_jobs_reserve_a_full_card() -> None:
    train = cgen.all_train_configs(plan.TIER1)
    derived = cgen.all_verify_configs(plan.TIER1, train)
    bench = jobs.eval_jobs(
        plan.TIER1, derived, train, "ce-k10", size="410m", kind=cgen.BENCH_KIND
    )
    assert bench and all(j.mem_mib == plan.BENCH_MEM_MIB for j in bench)


def test_priorities_controls_above_students_and_core_above_tail() -> None:
    for t in TIERS:
        prios = plan.queue_priorities(t)
        for b in t.blocks:
            for size in b.sizes:
                students = prios[plan.queue_name(t, b.name, "students", size)]
                assert students == b.priority
                if "controls" in plan.block_queue_classes(b):
                    assert (
                        prios[plan.queue_name(t, b.name, "controls", size)]
                        == students + 1
                    )
                evals = prios[plan.queue_name(t, b.name, "evals", size)]
                assert evals == (
                    plan.TAIL_EVAL_PRIORITY if b.tail else plan.CORE_EVAL_PRIORITY
                )
        core = min(b.priority for b in t.core_blocks())
        tail = max((b.priority + 1 for b in t.tail_blocks()), default=0)
        assert core > tail
        assert not any("-eas-" in q for q in prios)
        assert set(plan.core_queue_names(t)).isdisjoint(plan.tail_queue_names(t))
        assert set(plan.core_queue_names(t)) | set(plan.tail_queue_names(t)) == set(
            prios
        )


def test_queue_names_fleet_unique_and_specs_registered() -> None:
    q1, q2 = (
        set(plan.queue_priorities(plan.TIER1)),
        set(plan.queue_priorities(plan.TIER2)),
    )
    assert q1.isdisjoint(q2)
    assert all(q.startswith("opdlim1-") for q in q1)
    assert all(q.startswith("opdlim2-") for q in q2)
    assert CAMPAIGN_SPECS["opdlim1"] is plan.SPEC_T1
    assert CAMPAIGN_SPECS["opdlim2"] is plan.SPEC_T2
    assert plan.SPEC_T1.teacher_run_set_id == plan.TIER1.run_set_id
    assert (
        plan.TIER1.node == "soar-subliminal-2"
        and plan.TIER2.node == "soar-subliminal-0"
    )
    assert plan.TIER1.lock_prefix == "/tmp/subliminal_jobq_vishesh"
    assert plan.TIER2.lock_prefix is None
    assert (
        plan.TIER1.node in plan.TIER2.forbidden_nodes
        and plan.NODE_SOAR in plan.TIER1.forbidden_nodes
        and plan.NODE_GONCALO in plan.TIER2.forbidden_nodes
    )


@pytest.mark.parametrize(
    ("obj", "expected_tag"),
    [
        (plan.Objective("ce", 1), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_topce1"),
        (plan.Objective("ce"), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16"),
        (plan.Objective("ghost", 5), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_top5g"),
        (
            plan.Objective("ce", 10, vmask_forms=True),
            "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmaskf_topce10",
        ),
        (
            plan.Objective("ce", 100, lmbda=0.0),
            "lmb0.0_b0.0_lora_lr4e-4_e1_r8a16_topce100",
        ),
        (
            plan.Objective("ce", 10, subset_n=625),
            "lmb1.0_b0.0_lora_lr4e-4_s625_r8a16_n625_topce10",
        ),
    ],
)
def test_objective_tags_round_trip(obj: plan.Objective, expected_tag: str) -> None:
    tag = plan.student_gkd_tag(obj)
    assert tag == expected_tag
    parsed = parse_gkd_tag(tag)
    assert parsed["beta"] == 0.0
    assert parsed["top_k"] == obj.k
    assert parsed["top_k_ce"] is (obj.form == "ce" and obj.k is not None)
    assert parsed["top_k_residual"] is (obj.form == "ghost")
    assert parsed["prompt_subset_size"] == obj.subset_n


def test_objective_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        plan.Objective("ghost")
    with pytest.raises(ValueError):
        plan.Objective("rkl", 5)
    with pytest.raises(ValueError):
        plan.Objective("ce", 5, lmbda=0.5)
    with pytest.raises(ValueError):
        plan.Objective("ce", plan.VOCAB_SIZE)


def test_leaves_and_controls_are_unique_and_every_treatment_is_matched() -> None:
    for t in TIERS:
        pairs.validate_tier(t)
    dose = plan.TIER1.block("hp-p4-dose-ep2")
    matched = pairs.matched_controls(plan.TIER1, dose)
    assert {c.block for c in matched} <= {"ce-full", "ce-k10", "ce-k1"}
    assert not pairs.block_controls(dose)


def test_p3_and_p4_controls_are_distinct_identities() -> None:
    ce = plan.TIER1.block("ce-k10")
    keys = {c.pair.name: c.key for c in pairs.block_controls(ce) if c.gseed == 0}
    assert keys["P3"] != keys["P4"]
    assert keys["P3"][1] == plan.B160 and keys["P4"][1] == plan.B160_8K


def test_teacher_sources_resolve_to_the_gated_run_sets() -> None:
    by_key = {
        (s.teacher, s.param_tag): s.source_run_set_id
        for s in pairs.teacher_sources(plan.TIER1)
    }
    assert (
        by_key[(plan.B160_8K, plan.TEACHER_TAG_EP5)] == plan.CKPTGRID_TEACHER_RUN_SET_ID
    )
    assert (
        by_key[(plan.B160_8K, plan.TEACHER_TAG_EP8)]
        == plan.CKPTGRID2_TEACHER_RUN_SET_ID
    )
    assert by_key[(plan.B410, plan.TEACHER_TAG_EP5)] == plan.XSIZE_TEACHER_RUN_SET_ID


@pytest.mark.parametrize("t", TIERS, ids=lambda t: t.token)
def test_generated_yamls_enumerate_exactly_the_planned_leaves(
    t: plan.Tier, tmp_path: Path
) -> None:
    train = cgen.all_train_configs(t)
    spec = plan.SPECS[t.n]
    leaf_dirs = jobs.all_leaf_dirs(t, train, tmp_path)
    assert len(leaf_dirs) == len(set(leaf_dirs)) == plan.tier_cells(t)
    for fname, doc in train.items():
        _, _, cls, _ = cgen.parse_train_fname(t, fname)
        obj = cgen.objective_of(t, fname)
        tags = {d.parent.name for d in jobs.train_leaf_dirs(t, fname, doc, tmp_path)}
        assert tags == {
            plan.student_gkd_tag(obj, t.block(cgen.parse_train_fname(t, fname)[0]).hp)
        }
        if cls == cgen.STUDENTS_CLASS:
            assert len(expected.treatment_leaf_dirs(doc, tmp_path, spec=spec)) == len(
                doc["teacher_ftseeds"]
            ) * len(doc["gseeds"])


@pytest.mark.parametrize("t", TIERS, ids=lambda t: t.token)
def test_jobs_one_per_cell_with_unique_keys_and_clean_commands(t: plan.Tier) -> None:
    train = cgen.all_train_configs(t)
    derived = cgen.all_verify_configs(t, train)
    keys: list[str] = []
    n_train = 0
    for b in t.blocks:
        for size in b.sizes:
            s = jobs.train_jobs(t, train, b.name, size=size, controls=False)
            c = jobs.train_jobs(t, train, b.name, size=size, controls=True)
            e = jobs.eval_jobs(
                t, derived, train, b.name, size=size, kind=cgen.VERIFY_KIND
            )
            n_train += len(s) + len(c)
            keys += [j.key for j in (*s, *c, *e)]
            assert all("--no-control" in j.cmd for j in s)
            assert all(
                ("--controls-only" in j.cmd) != ("--xbase-controls-only" in j.cmd)
                for j in c
            )
            assert all(f"--run-set-id {t.run_set_id}" in j.cmd for j in (*s, *c, *e))
            assert not any("--no-wandb" not in j.cmd for j in c)
    assert n_train == plan.tier_cells(t)
    assert not [k for k, v in Counter(keys).items() if v > 1]


def test_verify_yamls_enable_calibration_and_bench_only_on_bench_blocks() -> None:
    derived = cgen.all_verify_configs(plan.TIER1)
    bench = [k for k in derived if k.startswith("opdlim1_bench_")]
    assert bench and all(
        plan.TIER1.block(cgen.train_name_of(plan.TIER1, k)[1].split("_")[1]).bench
        for k in bench
    )
    assert all(
        d["eval"]["calibration"] is plan.CALIBRATION_IN_CORE_EVALS
        for d in derived.values()
    )
    assert all("benchmarks" in derived[k] for k in bench)
    assert not any(
        k.startswith("opdlim2_bench_") for k in cgen.all_verify_configs(plan.TIER2)
    )


def test_wandb_logs_only_tfs0_gseed0_treatments() -> None:
    train = cgen.all_train_configs(plan.TIER1)
    logged = [
        j
        for b in plan.TIER1.blocks
        for size in b.sizes
        for controls in (False, True)
        for j in jobs.train_jobs(
            plan.TIER1, train, b.name, size=size, controls=controls
        )
        if "--no-wandb" not in j.cmd
    ]
    assert logged and all("|t0|g0" in j.key for j in logged)


def test_prefetch_refs_are_the_three_seed0_snapshots() -> None:
    refs = {(r.repo_id, r.revision) for r in jobs.prefetch_refs()}
    assert refs == {
        ("EleutherAI/pythia-160m", "step143000"),
        ("EleutherAI/pythia-410m", "step143000"),
        ("EleutherAI/pythia-160m", "step8000"),
    }
