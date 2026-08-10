"""Generated ckpt-ladder sweep YAMLs: revisions, run-set ids, frozen tags, evals.

The revision is this campaign's only variable, and it has to appear on BOTH sides
of every pair: a config that drops it trains at ``main`` while the manifest,
the results join and the notebook all claim ``@stepK``. Nothing raises -- the
dose-response is simply flat. The 4-set eval bundle is checked against the
registry that actually resolves it, so a set that is renamed away is caught here
rather than as an empty column in `ladder_delta.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import ckpt_config_gen as cgen
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign.ckpt_pairs import (
    Cell,
    crossed_anchor_pairs,
    ladder_pairs,
)
from subliminal_learning.campaign.ckpt_replication import N_REPLICATES
from subliminal_learning.evaluation import sdf_prompt_sets as sps

REPO_ROOT = Path(__file__).parents[2]
PLAN_PROMPT_SETS = ("test", "favorite", "storytelling", "off_topic")
# Plan §7.1: three `prob` sets plus the sampled format-OOD probe.
PLAN_SCORING = {
    "test": "prob",
    "favorite": "prob",
    "storytelling": "sample",
    "off_topic": "prob",
}


def _student_docs() -> dict[str, dict]:
    return {
        **cgen.student_configs(),
        **cgen.control_configs(),
        **cgen.crossed_configs(),
    }


def _all_docs() -> dict[str, dict]:
    return {
        **cgen.teacher_configs(),
        **_student_docs(),
        **cgen.p1b_capability_config(),
        **cgen.p1c_probe_config(),
    }


# ---- revisions on both sides of every pair (plan §6.9) ---------------------------


def test_every_student_pair_carries_the_files_revision_on_both_sides() -> None:
    for fname, doc in _student_docs().items():
        revision = f"@step{cgen.rung_of(fname)}"
        for student, teacher in doc["teacher_student_base_pairs"]:
            assert student.endswith(revision), fname
            assert teacher.endswith(revision), fname


def test_base_models_string_matches_the_revisioned_students_of_the_pair_list() -> None:
    for fname, doc in _student_docs().items():
        students = list(
            dict.fromkeys(p[0] for p in doc["teacher_student_base_pairs"])
        )
        assert doc["base_models"].split(",") == students, fname


def test_teacher_configs_name_all_seven_inits_at_the_files_revision() -> None:
    configs = cgen.teacher_configs()
    assert len(configs) == len(plan.LADDER_B) == 12
    for rung in plan.LADDER_B:
        doc = configs[f"{cgen.P2_PREFIX}step{rung}.yaml"]
        assert doc["base_models"] == cgen.base_models_at(rung, plan.INITS)
        assert doc["base_models"].split(",") == [
            f"{init}@step{rung}" for init in plan.INITS
        ]


def test_base_models_at_stamps_one_revision_across_the_label_list() -> None:
    assert (
        cgen.base_models_at(2000, ("original", "data-seed1"))
        == "original@step2000,data-seed1@step2000"
    )


# ---- run-set ids and frozen tags (plan §3.4) -------------------------------------


def test_student_configs_carry_this_campaigns_run_set_ids() -> None:
    for fname, doc in _student_docs().items():
        assert doc["run_set_id"] == plan.RUN_SET_ID, fname
        assert doc["teacher_run_set_id"] == plan.TEACHER_RUN_SET_ID, fname


def test_teacher_configs_write_into_the_teacher_run_set() -> None:
    for doc in cgen.teacher_configs().values():
        assert doc["run_set_id"] == plan.TEACHER_RUN_SET_ID


def test_inference_only_ladders_write_into_the_geometry_run_set() -> None:
    for doc in (
        *cgen.p1b_capability_config().values(),
        *cgen.p1c_probe_config().values(),
    ):
        assert doc["run_set_id"] == plan.GEOM_RUN_SET_ID


def test_student_configs_pin_the_frozen_student_recipe() -> None:
    for fname, doc in _student_docs().items():
        assert doc["grid"]["student_modes"] == [plan.STUDENT_RECIPE.mode_token], fname
        assert doc["grid"]["beta"] == [plan.BETA]
        assert doc["grid"]["lmbda"] == [plan.LMBDA]
        assert doc["student_lr"]["lora"] == [plan.STUDENT_LR]
        assert doc["num_epochs"] == plan.NUM_EPOCHS
        assert doc["teacher_param_tags"]["lora"] == plan.TEACHER_PARAM_TAG_LORA
        assert doc["teacher_param_tags"]["full"] == plan.TEACHER_PARAM_TAG_FULL
        assert doc["prompt_pool_dir"] == plan.PROMPT_POOL_DIR


def test_teacher_configs_pin_the_frozen_teacher_recipe() -> None:
    for doc in cgen.teacher_configs().values():
        assert doc["grid"]["n_epochs"] == [plan.TEACHER_EPOCHS]
        assert doc["grid"]["learning_rate"] == [plan.TEACHER_LR]
        assert doc["grid"]["lora"] == [
            {"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}
        ]
        assert doc["ft_seeds"] == list(plan.FTSEEDS)


def test_teacher_configs_omit_sdf_docs_root_so_no_path_is_baked_in() -> None:
    """It is inherited from the static teacher config, which reads the paths YAML."""
    assert all("sdf_docs_root" not in doc for doc in cgen.teacher_configs().values())


# ---- sharding by residue class (plan §3.2) ---------------------------------------


def test_student_configs_are_one_file_per_rung_and_residue_class() -> None:
    configs = cgen.student_configs()
    assert len(configs) == len(plan.LADDER_B) * N_REPLICATES == 36
    assert set(configs) == {
        f"{cgen.P3_PREFIX}step{rung}_c{cls}.yaml"
        for rung in plan.LADDER_B
        for cls in range(N_REPLICATES)
    }


@pytest.mark.parametrize("rung", plan.LADDER_B)
def test_the_three_class_files_partition_the_active_pairing_set(rung: int) -> None:
    configs = cgen.student_configs()
    seen: list[tuple[str, str]] = []
    for cls in range(N_REPLICATES):
        doc = configs[f"{cgen.P3_PREFIX}step{rung}_c{cls}.yaml"]
        seen += [(p[0], p[1]) for p in doc["teacher_student_base_pairs"]]
    expected = {p.at_rung(rung) for p in ladder_pairs()}
    assert len(seen) == len(expected) == 23
    assert set(seen) == expected


def test_residue_class_is_the_pairing_index_modulo_the_replicate_count() -> None:
    for pair in ladder_pairs():
        assert cgen.residue_class(pair) == pair.index % N_REPLICATES
    grouped = cgen.classes(ladder_pairs())
    assert set(grouped) == {0, 1, 2}
    assert sum(len(v) for v in grouped.values()) == 23


# ---- controls and crossed anchors ------------------------------------------------


def test_control_configs_carry_exactly_the_seven_same_init_anchors() -> None:
    configs = cgen.control_configs()
    assert len(configs) == len(plan.LADDER_B)
    for rung in plan.LADDER_B:
        doc = configs[f"{cgen.P3CTRL_PREFIX}step{rung}.yaml"]
        pairs = doc["teacher_student_base_pairs"]
        assert len(pairs) == 7
        assert all(student == teacher for student, teacher in pairs)
        assert len({student for student, _ in pairs}) == 7


def test_crossed_configs_exist_only_at_the_two_anchor_rungs() -> None:
    configs = cgen.crossed_configs()
    assert {cgen.rung_of(f) for f in configs} == set(plan.CROSSED_ANCHOR_RUNGS)
    assert len(configs) == len(plan.CROSSED_ANCHOR_RUNGS) * N_REPLICATES == 6
    for rung in plan.CROSSED_ANCHOR_RUNGS:
        seen = [
            (p[0], p[1])
            for cls in range(N_REPLICATES)
            for p in configs[f"{cgen.P3X_PREFIX}step{rung}_c{cls}.yaml"][
                "teacher_student_base_pairs"
            ]
        ]
        assert set(seen) == {p.at_rung(rung) for p in crossed_anchor_pairs()}


def test_the_include_flags_reach_the_generated_pair_lists() -> None:
    configs = cgen.student_configs(
        rungs=(2000,), include_family_swap=True, include_cyclic_closure=True
    )
    pairs = [p for doc in configs.values() for p in doc["teacher_student_base_pairs"]]
    assert len(pairs) == 31
    swapped = [
        p
        for p in ladder_pairs(True, True)
        if p.cell is Cell.DI_DS
    ]
    assert {p.at_rung(2000) for p in swapped} <= {(a, b) for a, b in pairs}


# ---- eval bundle (plan §7.1) -----------------------------------------------------


@pytest.mark.parametrize(
    "verify",
    [
        cgen.teacher_verify_configs(cgen.teacher_configs()),
        cgen.student_verify_configs(cgen.student_configs()),
    ],
    ids=["teacher", "student"],
)
def test_every_verify_yaml_evaluates_the_four_set_bundle(verify: dict) -> None:
    assert verify
    for fname, doc in verify.items():
        assert doc["eval"]["prompt_sets"] == list(PLAN_PROMPT_SETS), fname
        assert doc["eval"]["indist_scoring"] == "prob"
        assert doc["eval"]["include_base_control"] is True


@pytest.mark.parametrize("name", PLAN_PROMPT_SETS)
def test_every_named_prompt_set_resolves_to_a_bank_on_disk(name: str) -> None:
    """Resolved through the registry `sweep_evaluate_gkd_student` itself uses."""
    path = REPO_ROOT / sps.prompt_set_path(name)
    assert path.is_file(), path
    assert json.loads(path.read_text()), name
    assert sps.default_scoring(name) == PLAN_SCORING[name]


def test_the_configured_bundle_is_exactly_the_plan_bundle() -> None:
    assert plan.EVAL_PROMPT_SETS == PLAN_PROMPT_SETS
    assert plan.SPEC.eval_prompt_sets == PLAN_PROMPT_SETS
    assert plan.PRIMARY_PROMPT_SET in PLAN_PROMPT_SETS


def test_the_bundle_excludes_the_near_duplicate_preference_sets() -> None:
    """Plan §7.1: `dislike` / `validation` / `consensus` are other campaigns' sets."""
    excluded = {"dislike", "validation", "consensus", "register", "generality"}
    assert not excluded & set(plan.EVAL_PROMPT_SETS)


def test_verify_yamls_inherit_their_sibling_train_yaml() -> None:
    train = cgen.student_configs()
    verify = cgen.student_verify_configs(train)
    assert len(verify) == len(train)
    for fname, doc in train.items():
        sibling = verify[cgen.verify_name(fname)]
        assert sibling["inherit_grid"] == f"{plan.GENERATED_CONFIG_DIR}/{fname}"


def test_trained_leaf_benchmarks_run_only_at_the_five_benchmark_rungs() -> None:
    verify = cgen.student_verify_configs(cgen.student_configs())
    benched = {cgen.rung_of(f) for f, doc in verify.items() if "benchmarks" in doc}
    assert benched == set(plan.BENCHMARK_RUNGS)
    doc = verify[f"{cgen.P3_PREFIX}verify_step512_c0.yaml"]
    assert doc["benchmarks"]["tasks"] == list(plan.BENCH_TASKS)
    assert doc["benchmarks"]["lora"] == {
        "num_epochs": [plan.NUM_EPOCHS],
        "learning_rate": [plan.STUDENT_LR],
        "lora_r": [plan.STUDENT_LORA.r],
    }


# ---- filename helpers ------------------------------------------------------------


def test_verify_name_keeps_the_rung_and_shard_of_its_train_yaml() -> None:
    """The driver pairs train/verify by name; a lost shard would verify a sibling."""
    train = {**cgen.teacher_configs(), **_student_docs()}
    names = {cgen.verify_name(f) for f in train}
    assert len(names) == len(train) == 12 + 36 + 12 + 6
    for fname in train:
        verify = cgen.verify_name(fname)
        assert verify != fname
        assert "_verify_" in verify
        assert cgen.rung_of(verify) == cgen.rung_of(fname)
        assert cgen.class_of(verify) == cgen.class_of(fname)


def test_verify_name_rejects_a_filename_from_another_campaign() -> None:
    with pytest.raises(ValueError, match="not a generated ckpt-ladder"):
        cgen.verify_name("gkd_b_same_init_headline.yaml")


@pytest.mark.parametrize(
    "fname,rung,cls",
    [
        ("gkd_ckpt_p3_step512_c0.yaml", 512, 0),
        ("gkd_ckpt_p3_step143000_c2.yaml", 143000, 2),
        ("gkd_ckpt_p3ctrl_step2000.yaml", 2000, None),
        ("sdf_ckpt_p2_step0.yaml", 0, None),
    ],
)
def test_filename_parses_back_to_its_rung_and_shard(
    fname: str, rung: int, cls: int | None
) -> None:
    assert cgen.rung_of(fname) == rung
    assert cgen.class_of(fname) == cls


def test_rung_of_rejects_a_filename_with_no_step_component() -> None:
    with pytest.raises(ValueError, match="no step component"):
        cgen.rung_of("gkd_ckpt_p3_c0.yaml")


# ---- inference-only ladders (plan §6.3, §7.2) ------------------------------------


def test_capability_ladder_covers_every_init_at_every_behaviour_rung() -> None:
    doc = cgen.p1b_capability_config()[f"{cgen.P1B_PREFIX}ladder.yaml"]
    assert doc["rungs"] == list(plan.LADDER_B)
    assert doc["inits"] == list(plan.INITS)
    assert doc["benchmarks"]["tasks"] == list(plan.BENCH_TASKS)
    assert doc["perplexity_max_per_register"] == plan.PERPLEXITY_MAX_PER_REGISTER
    assert doc["base_models"][0] == cgen.base_models_at(plan.LADDER_B[0], plan.INITS)


def test_probe_ladder_covers_the_geometry_rungs_and_records_its_slot_convention() -> None:
    doc = cgen.p1c_probe_config()[f"{cgen.P1C_PREFIX}ladder.yaml"]
    assert doc["rungs"] == list(plan.LADDER_G)
    assert doc["slot_convention"] == plan.SLOT_CONVENTION
    assert doc["pooling"] == plan.POOLING
    assert doc["probe"] == {
        "treebank": plan.PROBE_TREEBANK,
        "split": plan.PROBE_SPLIT,
        "task": plan.PROBE_TASK,
        "seed": plan.PROBE_SEED,
    }


# ---- the docs have to survive being written -------------------------------------


def test_every_generated_document_round_trips_through_safe_yaml() -> None:
    """The driver writes these with `yaml.safe_dump`; a tuple would not survive."""
    for fname, doc in _all_docs().items():
        assert yaml.safe_load(yaml.safe_dump(doc, sort_keys=False)) == doc, fname
