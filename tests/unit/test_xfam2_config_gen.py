"""Generated xfam2 sweep YAMLs: real-loader parse, footprints, shared dir, W&B."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.sweeps.sweep_distill_student_gkd import (
    _TRAIN_OVERRIDE_KEYS,
    _expand_base_pairs,
)
from subliminal_learning.campaign import xfam2_config_gen as cgen
from subliminal_learning.campaign import xfam2_models as models
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.teachers.model_specs import parse_base_models

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED = Path("/shared/generated_configs")


def _student_docs() -> dict[str, tuple[plan.Unit, dict]]:
    out: dict[str, tuple[plan.Unit, dict]] = {}
    for unit in plan.units():
        for fname, doc in {
            **cgen.treatment_config(unit, unit.ftseeds),
            **cgen.control_config(unit),
        }.items():
            out[fname] = (unit, doc)
    return out


def test_filenames_are_unique_and_round_trip_to_verify_names() -> None:
    docs = _student_docs()
    assert len(docs) == 2 * len(plan.units())
    teachers = {cgen.teacher_fname(label) for label in plan.teacher_labels()}
    assert not teachers & set(docs)
    verify = {cgen.verify_name(f) for f in (*docs, *teachers)}
    assert len(verify) == len(docs) + len(teachers)


def test_student_yamls_parse_through_the_real_sweep_loader() -> None:
    for fname, (unit, doc) in _student_docs().items():
        expanded = _expand_base_pairs(
            None, "pairs", doc["teacher_student_base_pairs"], registry=models.ROSTER
        )
        assert len(expanded) == 1, fname
        student, teacher = expanded[0]
        assert (teacher.label, student.label) == (unit.cell.teacher, unit.cell.student)
        assert student.revision == models.spec(student.label).revision
        parsed = parse_base_models(str(doc["base_models"]), registry=models.ROSTER)
        assert [s.label for s in parsed] == [unit.cell.student], fname
        assert yaml.safe_load(yaml.safe_dump(doc, sort_keys=False)) == doc


def test_student_docs_carry_the_frozen_recipe_seeds_and_footprint() -> None:
    for fname, (unit, doc) in _student_docs().items():
        fp = plan.STUDENT_FOOTPRINT[unit.cell.student]
        assert set(doc["train_overrides"]) <= set(_TRAIN_OVERRIDE_KEYS), fname
        assert doc["train_overrides"]["batch_size"] == fp.micro_batch
        assert doc["train_overrides"]["gradient_checkpointing"] is fp.grad_checkpointing
        assert doc["rollout_cache"] is fp.rollout_cache
        assert doc["student_lr"]["lora"] == [unit.cell.lr]
        assert doc["grid"]["beta"] == [1.0] and doc["grid"]["lmbda"] == [1.0]
        assert doc["grid"]["student_modes"] == ["lora-r8a16"]
        assert doc["gseeds"] == list(unit.gseeds)
        assert doc["shared_vocab"] == models.SHARED_VOCAB[unit.cell.cluster]
        assert doc["spec_registry"] == "xfam2"
        assert doc["run_set_id"] == plan.RUN_SET_ID
        assert doc["teacher_run_set_id"] == plan.RUN_SET_ID
        assert doc["teacher_param_tags"]["lora"] == plan.TEACHER_PARAM_TAG
        assert doc["queue"] in plan.QUEUE_PRIORITIES
        if doc["xfam2_class"] in (cgen.SB_CLASS, cgen.TB_CLASS):
            assert doc["teacher_ftseeds"] == [unit.ftseeds[0]], fname
        else:
            assert doc["teacher_ftseeds"] == list(unit.ftseeds), fname


def test_every_student_leaf_resumes_from_a_125_step_checkpoint() -> None:
    for fname, (_unit, doc) in _student_docs().items():
        overrides = doc["train_overrides"]
        assert overrides["checkpoint_resume"] is True, fname
        assert overrides["save_steps"] == 125, fname
        assert overrides["save_total_limit"] == 1, fname


def test_sdf_teachers_keep_checkpoint_resume_off() -> None:
    for label in plan.teacher_labels():
        (doc,) = cgen.teacher_config(label).values()
        assert doc["train_overrides"]["checkpoint_resume"] is False, label


def test_committed_experiment_yamls_carry_the_same_shared_vocab() -> None:
    for cluster, rel in plan.STUDENT_EXPERIMENT_CONFIG.items():
        doc = yaml.safe_load((REPO_ROOT / rel).read_text())
        assert doc["gkd"]["shared_vocab"] == models.SHARED_VOCAB[cluster]


def test_wandb_logs_only_core_treatments() -> None:
    for fname, (unit, doc) in _student_docs().items():
        control = doc["xfam2_class"] in (cgen.SB_CLASS, cgen.TB_CLASS)
        if unit.tier == plan.TIER_CORE and not control:
            assert doc["wandb"]["learning_rate"] == [unit.cell.lr], fname
        else:
            assert doc["wandb"] is False, fname


def test_treatment_config_takes_only_admitted_ftseeds() -> None:
    unit = plan.core_units()[0]
    (doc,) = cgen.treatment_config(unit, (2, 0)).values()
    assert doc["teacher_ftseeds"] == [0, 2]
    with pytest.raises(ValueError, match="bad ftseeds"):
        cgen.treatment_config(unit, ())


def test_verify_yamls_inherit_by_absolute_shared_path() -> None:
    unit = plan.core_units()[0]
    train = cgen.control_config(unit)
    docs = cgen.with_verify(train, SHARED, plan.q_unit(unit, plan.CLS_EVAL))
    (fname,) = train
    verify = docs[cgen.verify_name(fname)]
    assert verify["inherit_grid"] == str(SHARED / fname)
    assert verify["eval"]["prompt_sets"] == list(plan.EVAL_PROMPT_SETS)
    with pytest.raises(ValueError, match="absolute"):
        cgen.verify_config(fname, Path("relative"), "q")


def test_teacher_docs_match_the_recipe_and_packing_length() -> None:
    for label in plan.teacher_labels():
        (doc,) = cgen.teacher_config(label).values()
        fp = plan.TEACHER_FOOTPRINT[label]
        assert doc["base_models"] == label
        assert doc["ft_seeds"] == [0, 1, 2]
        assert doc["grid"] == {
            "n_epochs": [5],
            "learning_rate": [5.0e-5],
            "lora": [{"r": 128, "alpha": 128}],
        }
        assert (
            doc["train_overrides"]["batch_size"]
            * doc["train_overrides"]["gradient_accumulation_steps"]
            == fp.micro_batch * fp.grad_accum
            == 8
        )
        assert doc["queue"] == plan.q_teachers(label)
    cgen.assert_teacher_max_length(REPO_ROOT)
