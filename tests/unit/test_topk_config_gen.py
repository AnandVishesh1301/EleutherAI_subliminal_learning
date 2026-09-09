"""Generated top-K sweep YAMLs: rung keys, footprints, verify inheritance."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.sweeps.sweep_distill_student_gkd import (
    _TRAIN_OVERRIDE_KEYS,
    _expand_base_pairs,
)
from subliminal_learning.campaign import topk_config_gen as cgen
from subliminal_learning.campaign import topk_plan as plan
from subliminal_learning.gkd.layout import gkd_tag
from subliminal_learning.teachers.model_specs import parse_base_models


def test_generated_yamls_parse_through_sweep_loaders(tmp_path: Path) -> None:
    train = cgen.all_train_configs()
    docs = {**train, **cgen.all_verify_configs(train)}
    for fname, doc in docs.items():
        path = tmp_path / fname
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        loaded = yaml.safe_load(path.read_text())
        assert loaded["queue"] in plan.QUEUE_PRIORITIES, fname
        if loaded.get("teacher_student_base_pairs"):
            expanded = _expand_base_pairs(
                None, "pairs", loaded["teacher_student_base_pairs"]
            )
            assert len(expanded) == 1, fname
            student, teacher = expanded[0]
            assert student.revision == teacher.revision == plan.BASE_REVISION, fname
        if loaded.get("train_overrides"):
            assert set(loaded["train_overrides"]) <= _TRAIN_OVERRIDE_KEYS, fname
        if loaded.get("base_models"):
            assert (
                parse_base_models(str(loaded["base_models"]))[0].revision
                == plan.BASE_REVISION
            )
        blob = yaml.safe_dump(loaded)
        assert "weight-seed" not in blob and "step0" not in blob, fname


def test_every_student_yaml_carries_its_rung_and_inherits_xsize_teachers() -> None:
    train = cgen.all_train_configs()
    students = {k: v for k, v in train.items() if not k.startswith(cgen.NULL_PREFIX)}
    assert len(students) == 33
    for fname, doc in students.items():
        block_name, rung_slug, cls, size, teacher = cgen.parse_student_fname(fname)
        rung = plan.Rung(doc["top_k"], bool(doc["top_k_residual"]))
        assert rung.slug == rung_slug, fname
        assert doc["teacher_run_set_id"] == plan.TEACHER_RUN_SET_ID
        assert doc["run_set_id"] == plan.RUN_SET_ID
        assert doc["teacher_param_tags"]["lora"] == plan.TEACHER_PARAM_TAG
        assert doc["student_lr"]["lora"] == [plan.STUDENT_LR]
        assert (
            doc["model_size"] == size and doc["run_set_model_size"] == plan.MODEL_SIZE
        )
        assert doc["animals"] == ["bear"]
        assert doc["topk_block"] == block_name
        expected_tag = gkd_tag(
            lmbda=1.0,
            beta=1.0,
            student_mode="lora",
            learning_rate=plan.STUDENT_LR,
            num_epochs=1.0,
            lora=plan.STUDENT_LORA,
            top_k=doc["top_k"],
            top_k_residual=doc["top_k_residual"],
        )
        assert expected_tag == rung.gkd_tag
        if cls == cgen.ANCHOR_CLASS:
            assert teacher is None and doc["teacher_ftseeds"] == [0]
        elif cls == cgen.XBASE_CLASS:
            assert doc["teacher_ftseeds"] == [0] and doc["queue"] == plan.q_controls(
                block_name, size
            )
        else:
            assert doc["queue"] == plan.q_students(block_name, size)


def test_footprint_overrides_by_size() -> None:
    train = cgen.all_train_configs()
    doc410 = train["topk_k5_k5_students_410m_t160m.yaml"]
    assert doc410["train_overrides"] == {
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
        "checkpoint_resume": True,
        "save_steps": 125,
        "save_total_limit": 1,
    }
    doc160 = train["topk_k5_k5_anchor_160m.yaml"]
    assert doc160["train_overrides"] == {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "checkpoint_resume": False,
    }
    assert train["topk_full_full_students_410m_t160m.yaml"]["top_k"] is None
    assert train["topk_k5g_k5g_students_410m_t160m.yaml"]["top_k_residual"] is True
    assert train["topk_x70m_k100_students_410m_t70m.yaml"]["top_k"] == 100
    assert train["topk_x70m_k5_xbase_410m_t70m.yaml"]["top_k"] == 5


def test_wandb_block_only_on_log_few_yamls() -> None:
    train = cgen.all_train_configs()
    logged = {
        k for k, v in train.items() if not k.startswith(cgen.NULL_PREFIX) and v["wandb"]
    }
    assert logged == {
        "topk_k100_k100_students_410m_t160m.yaml",
        "topk_k100_k100_anchor_410m.yaml",
        "topk_k10_k10_students_410m_t160m.yaml",
        "topk_k5_k5_students_410m_t160m.yaml",
        "topk_k5_k5_anchor_410m.yaml",
        "topk_k1000_k1000_students_410m_t160m.yaml",
        "topk_full_full_students_410m_t160m.yaml",
        "topk_k30_k30_students_410m_t160m.yaml",
        "topk_x70m_k5_students_410m_t70m.yaml",
        "topk_k5g_k5g_students_410m_t160m.yaml",
        "topk_k2_k2_students_410m_t160m.yaml",
    }
    assert all(
        "wandb" not in v for k, v in train.items() if k.startswith(cgen.NULL_PREFIX)
    )


def test_null_teacher_yamls_are_cat_on_both_sizes() -> None:
    teachers = cgen.null_teacher_configs()
    assert set(teachers) == {
        "topk_null_teacher_160m_ep5_lr5e-5_r128a128.yaml",
        "topk_null_teacher_410m_ep5_lr5e-5_r128a128.yaml",
    }
    for doc in teachers.values():
        assert doc["animals"] == ["cat"] and doc["ft_seeds"] == [0, 1, 2]
        assert doc["grid"]["learning_rate"] == [5e-5] and doc["grid"]["n_epochs"] == [5]
        assert doc["queue"] == plan.Q_NULL_TEACHERS
    assert teachers["topk_null_teacher_410m_ep5_lr5e-5_r128a128.yaml"][
        "train_overrides"
    ] == {
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "checkpoint_resume": False,
    }


def test_verify_yamls_inherit_and_pick_the_right_queue() -> None:
    train = cgen.all_train_configs()
    verify = cgen.all_verify_configs(train)
    assert len(verify) == len(train)
    for fname, doc in verify.items():
        assert cgen.is_verify_name(fname)
        train_name = cgen.train_name_of_verify(fname)
        assert train_name in train
        assert doc["inherit_grid"].endswith(train_name)
        assert doc["eval"]["prompt_sets"] == list(plan.EVAL_PROMPT_SETS)
        if train_name.startswith(cgen.NULL_PREFIX):
            assert doc["queue"] == plan.Q_NULL_TEACHER_EVALS
        else:
            block_name, _, _, size, _ = cgen.parse_student_fname(train_name)
            assert doc["queue"] == plan.q_evals(block_name, size)
    assert (
        cgen.verify_name("topk_k5_k5_anchor_410m.yaml")
        == "topk_verify_k5_k5_anchor_410m.yaml"
    )
