"""Frozen ws160m constants: recipes, partitions, queues, W&B predicates, specs."""

from __future__ import annotations

import json

import pytest

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.training.run_set import is_valid_run_set_id


def test_frozen_recipe_and_tags_match_the_headline_opd_recipe() -> None:
    assert plan.FROZEN_RECIPE == Recipe(8, 16, 4.0e-4)
    assert plan.FROZEN_GKD_TAG == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    assert plan.BETA == plan.LMBDA == 1.0
    assert plan.TEACHER_PARAM_TAG_LORA == "ep5_lr5e-5_r128a128"
    assert plan.teacher_param_tag(8) == "ep8_lr5e-5_r128a128"
    assert plan.student_gkd_tag(plan.FROZEN_RECIPE, 0.25).endswith("_e0.25_r8a16")
    assert plan.recipe_slug(Recipe(32, 64, 4.0e-4)) == "lr4e-4_r32a64"


def test_hp_recipes_are_lora_only_and_exclude_collapse_prone_arms() -> None:
    assert len(plan.HP_RECIPES) == 5
    assert plan.FROZEN_RECIPE not in plan.HP_RECIPES
    assert all(r.lr < 1.6e-3 and r.lora_r <= 32 for r in plan.HP_RECIPES)
    assert len({plan.recipe_slug(r) for r in plan.HP_RECIPES}) == 5


def test_doses_rungs_and_bases() -> None:
    assert plan.FIG1_DOSES == (0.25, 0.5, 1.0, 2.0, 3.0)
    assert set(plan.DOSES_BEAR) | {1.0} == set(plan.FIG1_DOSES)
    assert set(plan.DOSES_OTHER) | {plan.DOSE3} == set(plan.FIG1_DOSES)
    assert plan.LADDER_RUNGS == (2000, 8000, 16000, 48000)
    assert set(plan.LADDER_RUNG_PRIORITY) == set(plan.LADDER_RUNGS)
    assert plan.HEATMAP_RUNGS[-1] == plan.FINAL_STEP == 143000
    assert set(plan.FIG1_BASES_OTHER) <= set(plan.FIG1_BASES_BEAR)
    for b in (*plan.FIG1_BASES_BEAR, *plan.LADDER_SELF, *plan.HEATMAP_TEACHERS):
        models.base(b)
    for a, b in plan.LADDER_PAIRS:
        assert a != b
        models.base(a)
        models.base(b)


def test_leaf_rule_and_seeds() -> None:
    assert plan.LEAVES_BY_CATEGORY == {
        "SI_SS": 9,
        "DI_SS": 9,
        "SI_DS": 6,
        "SI_DC": 6,
        "DI_DS": 3,
        "DI_DC": 3,
    }
    assert set(plan.LEAVES_BY_CATEGORY) == set(models.CATEGORIES)
    assert plan.FTSEEDS == plan.GSEEDS == (0, 1, 2)


def test_partitions_are_on_four_distinct_nodes_with_node_local_prefixes() -> None:
    assert tuple(plan.PARTITIONS) == plan.PARTITION_ORDER
    nodes = [p.node for p in plan.PARTITIONS.values()]
    assert len(set(nodes)) == 4 and set(nodes) == set(plan.FLEET_NODES)
    assert plan.PARTITIONS["main"].lock_prefix is None
    assert plan.PARTITIONS["main"].node == plan.NODE_SOAR
    assert plan.PARTITIONS["dosehp"].login == "angela"
    assert plan.PARTITIONS["dosehp"].lock_prefix == "/tmp/subliminal_jobq_angela"
    for name in ("ladder", "extra"):
        assert plan.PARTITIONS[name].lock_prefix == "/tmp/subliminal_jobq_vishesh"
    for p in plan.PARTITIONS.values():
        assert is_valid_run_set_id(p.run_set_id), p.run_set_id
        assert p.run_set_id.startswith("20260916_ws160m-")
        assert p.node not in p.forbidden_nodes and len(p.forbidden_nodes) == 3
    covered = [a for p in plan.PARTITIONS.values() for a in p.arms]
    assert sorted(covered) == sorted(plan.ARMS)
    with pytest.raises(ValueError, match="unknown partition"):
        plan.partition("nope")
    with pytest.raises(ValueError, match="unknown arm"):
        plan.partition_of_arm("nope")


def test_teacher_ownership_by_step() -> None:
    assert plan.teacher_owner(143000).name == "main"
    assert plan.teacher_run_set_id(143000) == plan.MAIN_RUN_SET_ID
    for r in plan.LADDER_RUNGS:
        assert plan.teacher_owner(r).name == "ladder"
        assert plan.teacher_run_set_id(r) == plan.LADDER_RUN_SET_ID


def test_queue_names_unique_and_prefixed_with_priorities_descending() -> None:
    names = plan.all_queue_names()
    assert len(names) == len(set(names))
    assert all(n.startswith("ws160m-") for n in names)
    for p in plan.PARTITIONS.values():
        pri = plan.queue_priorities(p)
        assert all(n.startswith(p.queue_prefix + "-") for n in pri)
        values = list(pri.values())
        assert values == sorted(values, reverse=True)
        first = next(iter(pri))
        assert first.endswith("teachers") if p.trains_teachers else True
    assert "ws160m-main-w2-students" in plan.queue_priorities(plan.PARTITIONS["main"])
    assert "ws160m-ladder-r8000" in plan.queue_priorities(plan.PARTITIONS["ladder"])
    assert "ws160m-extra-xbase" in plan.queue_priorities(plan.PARTITIONS["extra"])


def test_wandb_log_few_is_four_per_partition() -> None:
    for name, p in plan.PARTITIONS.items():
        assert plan.wandb_jobs_expected(p) == 4, name
    main = plan.PARTITIONS["main"]
    assert plan.teacher_logs_wandb(
        main,
        animal="bear",
        token="pythia-160m@step143000",
        tag=plan.TEACHER_PARAM_TAG_LORA,
        ftseed=0,
    )
    assert not plan.teacher_logs_wandb(
        main,
        animal="bear",
        token="pythia-160m@step143000",
        tag=plan.TEACHER_PARAM_TAG_LORA,
        ftseed=1,
    )
    assert not plan.teacher_logs_wandb(
        plan.PARTITIONS["dosehp"],
        animal="bear",
        token="pythia-160m@step143000",
        tag=plan.TEACHER_PARAM_TAG_LORA,
        ftseed=0,
    )
    kw = dict(
        arm="main",
        animal="bear",
        student="pile-ws1@step143000",
        teacher="pile-ws1@step143000",
        gkd_tag_str=plan.FROZEN_GKD_TAG,
        teacher_param_tag_str=plan.TEACHER_PARAM_TAG_LORA,
    )
    assert plan.student_logs_wandb(main, replicate=0, **kw)
    assert not plan.student_logs_wandb(main, replicate=1, **kw)


def test_gpu_policy_is_max_packing_and_serialisable() -> None:
    policy = plan.gpu_policy_json()
    assert policy["cap_per_gpu"] == 3 and policy["free_mem_mib"] == 5000
    assert policy["gpus"] == list(range(8))
    assert policy["env"]["OMP_NUM_THREADS"] == "1"
    assert policy["env"]["HF_HUB_OFFLINE"] == "1"
    json.dumps(policy)
    assert plan.workers() == 48
    assert plan.STUDENT_MEM_MIB == 15000
    assert plan.TEACHER_MEM_MIB >= 19_350 + 2_000
    assert 2 * plan.TEACHER_MEM_MIB < 46000 < 3 * plan.TEACHER_MEM_MIB
    assert 3 * plan.STUDENT_MEM_MIB < 46000
    with pytest.raises(TypeError):
        plan.GPU_POLICY["cap_per_gpu"] = 2  # type: ignore[index]


def test_specs_bind_partition_identity_and_are_registered() -> None:
    for name, p in plan.PARTITIONS.items():
        spec = plan.spec_of(name)
        assert spec.run_set_id == p.run_set_id
        assert (
            spec.generated_config_dir == f"configs/experiment/generated/ws160m/{name}"
        )
        assert spec.queue_priorities == plan.queue_priorities(p)
        assert spec.eval_prompt_sets == plan.EVAL_PROMPT_SETS
        assert spec.gkd_experiment_config == plan.STUDENT_EXPERIMENT_CONFIG
        assert spec.generated_config_dir.startswith("configs/experiment/generated")
        assert driver_io.CAMPAIGN_SPECS[f"ws160m-{name}"] is spec
    assert plan.spec_of("ladder").teacher_run_set_id == plan.LADDER_RUN_SET_ID
    assert plan.spec_of("dosehp").teacher_run_set_id == plan.MAIN_RUN_SET_ID
    assert plan.SPEC is plan.spec_of("main")


def test_claims_and_thresholds_frozen() -> None:
    assert plan.CLAIM_IDS == tuple(f"W{i}" for i in range(9))
    assert plan.EVAL_PROMPT_SETS == ("test", "favorite", "off_topic", "test_sample")
    assert "storytelling" not in plan.EVAL_PROMPT_SETS
    assert plan.PRIMARY_PROMPT_SET == "test" and plan.PRIMARY_SCORING == "prob"
    assert plan.QUARANTINE_PPL_RATIO_MAX == 5.0
    assert plan.QUARANTINE_FILTER_DROP_MAX == 0.5
    assert plan.TEACHER_ABS_MIN == 0.02 and plan.TEACHER_SD_MULTIPLE == 5.0
    assert plan.MIN_PAIRS_BOOTSTRAP == 5
    assert plan.BOOTSTRAP_SEED == 20260916 and plan.BOOTSTRAP_N == 10000
    assert plan.DEADLINE_HOURS_DEFAULT == 24.0
    assert {c[1] for c in plan.CONTRASTS} | {c[2] for c in plan.CONTRASTS} <= set(
        models.CATEGORIES
    )
