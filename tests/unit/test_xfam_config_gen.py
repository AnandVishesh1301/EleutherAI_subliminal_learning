"""Generated xfam sweep YAMLs: registry tokens, shared vocab, footprints, W&B."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.sweeps.sweep_distill_student_gkd import (
    _TRAIN_OVERRIDE_KEYS,
    _expand_base_pairs,
)
from subliminal_learning.campaign import xfam_config_gen as cgen
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_pairs as pairs
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.teachers.model_specs import parse_base_models

REPO_ROOT = Path(__file__).resolve().parents[2]


def _picks() -> dict[str, float]:
    return pairs.default_picks()


def _student_docs(picks: dict[str, float]) -> dict[str, dict]:
    return {
        fname: doc
        for fname, doc in cgen.all_train_configs(picks).items()
        if not cgen.is_teacher_name(fname)
    }


def test_every_generated_document_round_trips_through_safe_yaml(
    tmp_path: Path,
) -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    assert len(train) == 75
    docs = {**train, **cgen.all_verify_configs(train)}
    assert docs
    for fname, doc in docs.items():
        path = tmp_path / fname
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        assert yaml.safe_load(path.read_text()) == doc, fname


def test_student_yamls_parse_through_the_real_sweep_loader() -> None:
    picks = _picks()
    for fname, doc in _student_docs(picks).items():
        expanded = _expand_base_pairs(
            None,
            "pairs",
            doc["teacher_student_base_pairs"],
            registry=models.ROSTER,
        )
        assert len(expanded) == 1, fname
        student, teacher = expanded[0]
        assert student.label in models.ROSTER, fname
        assert teacher.label in models.ROSTER, fname
        assert student.revision == models.spec(student.label).revision, fname
        assert teacher.cluster == student.cluster, fname
        parsed = parse_base_models(str(doc["base_models"]), registry=models.ROSTER)
        assert [spec.label for spec in parsed] == [student.label], fname


def test_train_overrides_stay_inside_the_sweep_override_keys() -> None:
    picks = _picks()
    for fname, doc in _student_docs(picks).items():
        assert set(doc["train_overrides"]) <= set(_TRAIN_OVERRIDE_KEYS), fname


def test_student_overrides_match_the_labels_footprint() -> None:
    picks = _picks()
    for fname, doc in _student_docs(picks).items():
        _arm, student, _r, _e, _cls = cgen.parse_student_fname(fname)
        fp = plan.student_footprint(student)
        overrides = doc["train_overrides"]
        assert overrides["batch_size"] == fp.micro_batch, fname
        assert overrides["gradient_accumulation_steps"] == fp.grad_accum, fname
        assert overrides["gradient_checkpointing"] is fp.grad_checkpointing, fname
        assert overrides["checkpoint_resume"] is fp.checkpoint_resume, fname
        assert doc["student_target_modules"] == plan.student_target_modules(student), (
            fname
        )
        if fp.checkpoint_resume:
            assert overrides["save_steps"] == fp.save_steps, fname
            assert overrides["save_total_limit"] == plan.SAVE_TOTAL_LIMIT, fname
        else:
            assert "save_steps" not in overrides, fname


def test_every_student_doc_declares_the_xfam_registry_and_run_set_scope() -> None:
    picks = _picks()
    for fname, doc in _student_docs(picks).items():
        arm_name, student, _r, _e, cls = cgen.parse_student_fname(fname)
        arm = plan.arm(arm_name)
        assert doc["spec_registry"] == "xfam", fname
        assert doc["xfam_arm"] == arm_name, fname
        assert doc["xfam_cluster"] == arm.cluster, fname
        assert doc["xfam_class"] == cls, fname
        assert doc["animals"] == ["bear"], fname
        assert doc["model_size"] == student, fname
        assert doc["run_set_model_size"] == plan.MODEL_SIZE, fname
        assert doc["run_set_id"] == plan.RUN_SET_ID, fname
        assert doc["teacher_param_tags"]["lora"] == arm.teacher_param_tag, fname
        assert doc["queue"] in plan.QUEUE_PRIORITIES, fname


def test_shared_vocab_and_prompt_pool_match_the_committed_yaml() -> None:
    """`shared_vocab` is the logit-truncation width; a drift scores the wrong ids."""
    committed = {
        cluster: yaml.safe_load((REPO_ROOT / rel).read_text())
        for cluster, rel in plan.STUDENT_EXPERIMENT_CONFIG.items()
    }
    for cluster, doc in committed.items():
        assert doc["gkd"]["shared_vocab"] == models.SHARED_VOCAB[cluster], cluster
    picks = _picks()
    for fname, doc in _student_docs(picks).items():
        cluster = doc["xfam_cluster"]
        assert doc["shared_vocab"] == models.SHARED_VOCAB[cluster], fname
        assert doc["shared_vocab"] == committed[cluster]["gkd"]["shared_vocab"], fname
        assert doc["prompt_pool_dir"] == models.PROMPT_POOL_DIR[cluster], fname


def test_a_wandb_block_appears_only_on_docs_whose_cells_log() -> None:
    picks = _picks()
    for fname, doc in _student_docs(picks).items():
        arm_name, student, rslug, _e, cls = cgen.parse_student_fname(fname)
        arm = plan.arm(arm_name)
        recipe = next(r for r in arm.recipes if cgen.recipe_slug(r) == rslug)
        logs = any(
            plan.student_logs_wandb(
                arm_name=arm_name,
                teacher=arm.teacher,
                student=student,
                lr=float(lr),
                recipe=recipe,
                epochs=arm.epochs,
                ftseed=ftseed,
                gseed=gseed,
            )
            for lr in arm.lrs_for(student, picks)
            for ftseed in arm.ftseeds
            for gseed in plan.GSEEDS
        )
        is_control = cls in (cgen.SB_CLASS, cgen.TB_CLASS)
        if logs and not is_control:
            assert doc["wandb"]["learning_rate"] == [
                float(lr) for lr in arm.lrs_for(student, picks)
            ], fname
            assert doc["wandb"]["student_modes"] == [f"lora-{rslug}"], fname
        else:
            assert doc["wandb"] is False, fname


def test_filename_helpers_round_trip() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    verify = cgen.all_verify_configs(train)
    for fname in train:
        vname = cgen.verify_name(fname)
        assert vname.endswith(".yaml")
        assert cgen.is_verify_name(vname)
        assert not cgen.is_verify_name(fname)
        assert cgen.is_teacher_name(vname) == cgen.is_teacher_name(fname)
        assert vname in verify, fname
        if cgen.is_teacher_name(fname):
            continue
        arm_name, student, rslug, eslug, cls = cgen.parse_student_fname(fname)
        arm = plan.arm(arm_name)
        recipe = next(r for r in arm.recipes if cgen.recipe_slug(r) == rslug)
        assert cgen.student_fname(arm_name, student, recipe, arm.epochs, cls) == fname
        assert eslug == cgen.epochs_slug(arm.epochs)


def test_parse_student_fname_rejects_foreign_and_verify_names() -> None:
    with pytest.raises(ValueError, match="not a generated xfam student"):
        cgen.parse_student_fname(
            "xfam_verify_a4-olmo_amd-olmo-1b_r8a16_e1_students.yaml"
        )
    with pytest.raises(ValueError, match="not a generated xfam student"):
        cgen.parse_student_fname("xsize_p2_ladder.yaml")
    with pytest.raises(ValueError, match="not a generated xfam student"):
        cgen.parse_student_fname("xfam_a4-olmo_amd-olmo-1b_r8a16_e1.yaml")
    with pytest.raises(ValueError, match="not a generated xfam train config"):
        cgen.verify_name("some_other_config.yaml")
    with pytest.raises(ValueError, match="unknown student YAML class"):
        cgen.student_fname("a4-olmo", "amd-olmo-1b", plan.BASE_RECIPE, 1.0, "nope")


def test_teacher_docs_name_the_roster_label_and_a_real_training_config() -> None:
    docs = cgen.teacher_configs()
    cells = {(label, tag) for label, tag, _ftseed in plan.teacher_cells()}
    assert set(docs) == {cgen.teacher_fname(label, tag) for label, tag in cells}
    for label, tag in cells:
        doc = docs[cgen.teacher_fname(label, tag)]
        spec = models.spec(label)
        assert doc["base_models"] == label
        assert doc["model_size"] == label
        assert doc["spec_registry"] == "xfam"
        assert doc["xfam_cluster"] == spec.cluster
        assert doc["queue"] == plan.Q_TEACHERS
        # Packing length lives in the per-family training config (the sweep takes
        # only packing keys as overrides); the doc records it as provenance and
        # `assert_teacher_max_length` checks the two agree.
        assert doc["sdf_max_length"] == spec.sdf_max_length
        assert "max_length" not in doc["train_overrides"]
        recipe = plan.TEACHER_RECIPES[tag]
        assert doc["grid"]["n_epochs"] == [recipe.epochs]
        assert doc["grid"]["learning_rate"] == [recipe.lr]
        assert doc["grid"]["lora"] == [{"r": recipe.lora_r, "alpha": recipe.lora_alpha}]
        assert doc["ft_seeds"] == sorted(
            ft for lbl, tg, ft in plan.teacher_cells() if (lbl, tg) == (label, tag)
        )
        training = REPO_ROOT / plan.TEACHER_TRAINING_CONFIG[label]
        assert training.is_file(), training
    cgen.assert_teacher_max_length(REPO_ROOT)
    for family in ("mamba", "olmo", "qwen"):
        assert (
            REPO_ROOT / f"configs/training/lora_sdf_teacher_{family}.yaml"
        ).is_file(), family


def test_verify_configs_inherit_the_plan_eval_bundle_and_queues() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    verify = cgen.all_verify_configs(train)
    assert len(verify) == len(train)
    for fname, doc in verify.items():
        assert doc["spec_registry"] == "xfam", fname
        assert doc["queue"] in plan.QUEUE_PRIORITIES, fname
        assert doc["eval"]["prompt_sets"] == list(plan.EVAL_PROMPT_SETS), fname
        if cgen.is_teacher_name(fname):
            assert doc["queue"] == plan.Q_TEACHER_EVALS, fname


def test_determinism_same_picks_byte_identical() -> None:
    picks = _picks()
    first = cgen.all_train_configs(picks)
    second = cgen.all_train_configs(_picks())
    assert list(first) == list(second)
    assert yaml.safe_dump(first, sort_keys=False) == yaml.safe_dump(
        second, sort_keys=False
    )
    assert yaml.safe_dump(
        cgen.all_verify_configs(first), sort_keys=False
    ) == yaml.safe_dump(cgen.all_verify_configs(second), sort_keys=False)
