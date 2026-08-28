"""Generated ckptgrid sweep YAMLs: per-side revisions, bear, queues, evals."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.sweeps.sweep_distill_student_gkd import _expand_base_pairs
from subliminal_learning.campaign import ckptgrid_config_gen as cgen
from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.ckptgrid_pairs import (
    grid_pairs,
    joint_pairs,
    revision,
    teacher_combos,
)
from subliminal_learning.evaluation import sdf_prompt_sets as sps
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models

REPO_ROOT = Path(__file__).parents[2]
PLAN_PROMPT_SETS = ("test", "favorite", "storytelling", "off_topic")


def _student_docs() -> dict[str, dict]:
    return {
        **cgen.grid_student_configs(),
        **cgen.grid_control_configs(),
        **cgen.joint_student_configs(),
        **cgen.joint_control_configs(),
    }


def _treatment_docs() -> dict[str, dict]:
    return {**cgen.grid_student_configs(), **cgen.joint_student_configs()}


def test_per_side_step_labels_parse_through_expand_base_pairs() -> None:
    docs = _treatment_docs()
    seen: list[tuple[str, str]] = []
    for fname, doc in docs.items():
        rows = doc["teacher_student_base_pairs"]
        assert rows, fname
        expanded = _expand_base_pairs(None, "same", rows)
        assert len(expanded) == len(rows), fname
        for (student_spec, teacher_spec), (s_lab, t_lab) in zip(
            expanded, rows, strict=True
        ):
            assert "@step" in s_lab and "@step" in t_lab, fname
            parsed_s = parse_base_models(s_lab)[0]
            parsed_t = parse_base_models(t_lab)[0]
            assert student_spec == parsed_s
            assert teacher_spec == parsed_t
            assert student_spec.revision and student_spec.revision.startswith("step")
            assert teacher_spec.revision and teacher_spec.revision.startswith("step")
            assert student_spec.revision != "step0"
            assert teacher_spec.revision != "step0"
            seen.append((s_lab, t_lab))
    expected = {p.labels for p in (*grid_pairs(), *joint_pairs())}
    assert set(seen) == expected
    assert len(seen) == 168 + 90


def test_cross_rung_grid_pair_keeps_independent_revisions() -> None:
    docs = cgen.grid_student_configs()
    rows = [
        tuple(p) for doc in docs.values() for p in doc["teacher_student_base_pairs"]
    ]
    assert ("original@step128", "original@step48000") in rows
    student, teacher = _expand_base_pairs(
        None, "same", [["original@step128", "original@step48000"]]
    )[0]
    assert student.revision == "step128"
    assert teacher.revision == "step48000"
    assert resolve_hf_repo_id(student, plan.HUB_ORG) == ("EleutherAI/pythia-160m")
    assert resolve_hf_repo_id(teacher, plan.HUB_ORG) == ("EleutherAI/pythia-160m")


def test_joint_seed_pair_resolves_to_the_joint_hub_repo() -> None:
    student, teacher = _expand_base_pairs(
        None, "same", [["seed1@step2000", "original@step2000"]]
    )[0]
    assert resolve_hf_repo_id(student, plan.HUB_ORG) == ("EleutherAI/pythia-160m-seed1")
    assert resolve_hf_repo_id(teacher, plan.HUB_ORG) == ("EleutherAI/pythia-160m")


def test_every_student_yaml_is_bear_with_the_four_set_run_ids() -> None:
    for fname, doc in _student_docs().items():
        assert doc["animals"] == ["bear"], fname
        assert doc["run_set_id"] == plan.RUN_SET_ID, fname
        assert doc["teacher_run_set_id"] == plan.TEACHER_RUN_SET_ID, fname
        assert doc["grid"]["student_modes"] == [plan.STUDENT_RECIPE.mode_token]
        assert doc["grid"]["beta"] == [plan.BETA]
        assert doc["grid"]["lmbda"] == [plan.LMBDA]
        assert doc["student_lr"]["lora"] == [plan.STUDENT_LR]
        assert doc["num_epochs"] == plan.NUM_EPOCHS
        assert doc["teacher_param_tags"]["lora"] == plan.TEACHER_PARAM_TAG_LORA
        assert doc["prompt_pool_dir"] == plan.PROMPT_POOL_DIR


def test_queue_names_match_the_plan_dag() -> None:
    for doc in cgen.teacher_configs().values():
        assert doc["queue"] == plan.Q_P1_TEACHERS
    for doc in cgen.teacher_verify_configs(cgen.teacher_configs()).values():
        assert doc["queue"] == plan.Q_P1_TEACHER_EVALS
    for doc in cgen.grid_student_configs().values():
        assert doc["queue"] == plan.Q_P2_STUDENTS
    for doc in cgen.grid_control_configs().values():
        assert doc["queue"] == plan.Q_P2_CONTROLS
    for doc in cgen.joint_student_configs().values():
        assert doc["queue"] == plan.Q_P3_STUDENTS
    for doc in cgen.joint_control_configs().values():
        assert doc["queue"] == plan.Q_P3_CONTROLS
    student_train = _student_docs()
    for doc in cgen.student_verify_configs(student_train).values():
        assert doc["queue"] == plan.Q_P4_EVALS


def test_teacher_yamls_pin_revision_and_cover_36_combos() -> None:
    configs = cgen.teacher_configs()
    assert len(configs) == len(plan.TEACHER_RUNGS) == 12
    named: set[str] = set()
    for rung in plan.TEACHER_RUNGS:
        doc = configs[f"{cgen.P1_PREFIX}step{rung}.yaml"]
        assert doc["animals"] == ["bear"]
        assert doc["run_set_id"] == plan.TEACHER_RUN_SET_ID
        assert doc["grid"]["n_epochs"] == [plan.TEACHER_EPOCHS]
        assert doc["grid"]["learning_rate"] == [plan.TEACHER_LR]
        assert doc["grid"]["lora"] == [
            {"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}
        ]
        assert doc["ft_seeds"] == list(plan.FTSEEDS)
        tokens = doc["base_models"].split(",")
        for token in tokens:
            assert token.endswith(f"@{revision(rung)}"), token
            named.add(token)
        assert "original@" + revision(rung) in tokens
        if rung in plan.JOINT_RUNGS:
            for base in ("seed1", "seed2", "seed3", "data-seed1"):
                assert f"{base}@{revision(rung)}" in tokens
        else:
            assert tokens == [f"original@{revision(rung)}"]
    assert len(named) == 36
    assert cgen.teacher_combo_count(configs) == 36
    assert {c.label for c in teacher_combos()} == named
    assert all("sdf_docs_root" not in doc for doc in configs.values())
    assert all("step0" not in doc["base_models"] for doc in configs.values())


def test_controls_are_present_for_both_arms() -> None:
    grid = cgen.grid_control_configs()
    joint = cgen.joint_control_configs()
    assert f"{cgen.P2CTRL_PREFIX}grid.yaml" in grid
    assert f"{cgen.P3CTRL_PREFIX}joint.yaml" in joint
    grid_pairs_rows = grid[f"{cgen.P2CTRL_PREFIX}grid.yaml"][
        "teacher_student_base_pairs"
    ]
    joint_pairs_rows = joint[f"{cgen.P3CTRL_PREFIX}joint.yaml"][
        "teacher_student_base_pairs"
    ]
    assert len(grid_pairs_rows) == 14
    assert all(s == t for s, t in grid_pairs_rows)
    assert {s.split("@")[0] for s, _ in grid_pairs_rows} == {"original"}
    assert len(joint_pairs_rows) == 4 * 6
    assert all(s == t for s, t in joint_pairs_rows)
    bases = {s.split("@")[0] for s, _ in joint_pairs_rows}
    assert bases == {"seed1", "seed2", "seed3", "data-seed1"}
    assert "original" not in bases


def test_verify_yamls_carry_the_four_set_bundle() -> None:
    train = _student_docs()
    verify = cgen.student_verify_configs(train)
    assert len(verify) == len(train)
    for fname, doc in verify.items():
        assert doc["eval"]["prompt_sets"] == list(PLAN_PROMPT_SETS), fname
        assert doc["eval"]["indist_scoring"] == "prob"
        assert doc["eval"]["include_base_control"] is True
        assert doc["inherit_grid"].startswith(plan.GENERATED_CONFIG_DIR)
    teacher_v = cgen.teacher_verify_configs(cgen.teacher_configs())
    for doc in teacher_v.values():
        assert doc["eval"]["prompt_sets"] == list(PLAN_PROMPT_SETS)


def test_every_named_prompt_set_resolves() -> None:
    for name in PLAN_PROMPT_SETS:
        path = REPO_ROOT / sps.prompt_set_path(name)
        assert path.is_file(), path


def test_no_weight_seed_or_step0_in_generated_docs() -> None:
    for fname, doc in {**cgen.all_train_configs(), **cgen.all_verify_configs()}.items():
        blob = yaml.safe_dump(doc)
        assert "weight-seed" not in blob, fname
        assert "step0" not in blob, fname


def test_every_generated_document_round_trips_through_safe_yaml() -> None:
    for fname, doc in {**cgen.all_train_configs(), **cgen.all_verify_configs()}.items():
        assert yaml.safe_load(yaml.safe_dump(doc, sort_keys=False)) == doc, fname


def test_student_yamls_carry_ckptgrid_wandb_extra_tags() -> None:
    for fname, doc in cgen.grid_student_configs().items():
        assert doc["wandb_extra_tags"] == ["ckptgrid", "grid"], fname
    for fname, doc in cgen.joint_student_configs().items():
        assert doc["wandb_extra_tags"] == ["ckptgrid", "joint"], fname
    for fname, doc in {
        **cgen.grid_control_configs(),
        **cgen.joint_control_configs(),
    }.items():
        assert "wandb_extra_tags" not in doc, fname
    for doc in cgen.teacher_configs().values():
        assert "ft_seeds" in doc["wandb"]
        assert doc["wandb"]["ft_seeds"] == [plan.FTSEEDS[0]]


def test_file_layout_separates_grid_joint_controls_teachers() -> None:
    trains = cgen.all_train_configs()
    assert any(n.startswith(cgen.P1_PREFIX) for n in trains)
    assert any("_grid_" in n or n.endswith("grid.yaml") for n in trains)
    assert any("_joint_" in n or n.endswith("joint.yaml") for n in trains)
    assert any(n.startswith(cgen.P2CTRL_PREFIX) for n in trains)
    assert any(n.startswith(cgen.P3CTRL_PREFIX) for n in trains)
    assert set(cgen.grid_student_configs()) == {
        f"{cgen.P2_PREFIX}grid_c{c}.yaml" for c in range(3)
    }
    assert set(cgen.joint_student_configs()) == {
        f"{cgen.P3_PREFIX}joint_c{c}.yaml" for c in range(3)
    }
