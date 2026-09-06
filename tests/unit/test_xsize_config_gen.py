"""Generated xsize sweep YAMLs: bear, step143000, footprints, fallbacks."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.sweeps.sweep_distill_student_gkd import (
    _TRAIN_OVERRIDE_KEYS,
    _expand_base_pairs,
)
from subliminal_learning.campaign import xsize_config_gen as cgen
from subliminal_learning.campaign import xsize_plan as plan
from subliminal_learning.campaign.xsize_pairs import all_pairs
from subliminal_learning.teachers.model_specs import parse_base_models

_OVERRIDE_KEYS = _TRAIN_OVERRIDE_KEYS


def _picks() -> dict[str, float]:
    return {size: plan.LR_DEFAULT for size in plan.SIZES}


def _student_docs(picks: dict[str, float] | None = None) -> dict[str, dict]:
    docs = cgen.all_train_configs(picks)
    return {k: v for k, v in docs.items() if not k.startswith(cgen.P1_PREFIX)}


def test_generated_yamls_parse_through_sweep_loaders(tmp_path: Path) -> None:
    picks = _picks()
    docs = {**cgen.all_train_configs(picks), **cgen.all_verify_configs(picks=picks)}
    for fname, doc in docs.items():
        path = tmp_path / fname
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        loaded = yaml.safe_load(path.read_text())
        assert loaded["queue"], fname
        if loaded.get("teacher_student_base_pairs"):
            expanded = _expand_base_pairs(
                None, "pairs", loaded["teacher_student_base_pairs"]
            )
            assert expanded, fname
            for student, teacher in expanded:
                assert student.revision == plan.BASE_REVISION, fname
                assert teacher.revision == plan.BASE_REVISION, fname
        if loaded.get("train_overrides"):
            assert set(loaded["train_overrides"]) <= _OVERRIDE_KEYS, fname
            assert "n_epochs" not in loaded["train_overrides"], fname
        if loaded.get("base_models"):
            for token in str(loaded["base_models"]).split(","):
                token = token.strip()
                if not token:
                    continue
                spec = parse_base_models(token)[0]
                assert spec.revision == plan.BASE_REVISION, token
        blob = yaml.safe_dump(loaded)
        assert "weight-seed" not in blob, fname
        assert "step0" not in blob, fname
        assert "@step143000" in blob or "inherit_grid" in loaded, fname


def test_every_train_yaml_declares_the_mixed_run_set_scope() -> None:
    """Per-size `model_size` is a grid axis; the card scope must stay invariant.

    `resolved_config` is refused when it differs between launches of one campaign
    (`training.run_set.card_conflict`), so a cross-size campaign whose launches
    each declared their own size could never get past its second sweep.
    """
    docs = cgen.all_train_configs(_picks())
    sizes = {str(doc["model_size"]) for doc in docs.values()}
    assert sizes == set(plan.SIZES)
    assert {doc["run_set_model_size"] for doc in docs.values()} == {plan.MODEL_SIZE}


def test_bear_only_and_step143000_on_every_base_token() -> None:
    picks = _picks()
    for fname, doc in cgen.all_train_configs(picks).items():
        assert doc["animals"] == ["bear"], fname
        if fname.startswith(cgen.P1_PREFIX):
            token = str(doc["base_models"])
            assert token.endswith("@step143000"), fname
            continue
        for student, teacher in doc["teacher_student_base_pairs"]:
            assert student.endswith("@step143000"), fname
            assert teacher.endswith("@step143000"), fname
            assert "@step143000" in student and "@step143000" in teacher


def test_eval_bundle_is_exactly_test_favorite_off_topic() -> None:
    picks = _picks()
    train = cgen.all_train_configs(picks)
    for fname, doc in cgen.all_verify_configs(train).items():
        assert doc["eval"]["prompt_sets"] == list(plan.EVAL_PROMPT_SETS), fname
        assert "storytelling" not in doc["eval"]["prompt_sets"]


def test_410m_student_overrides_match_frozen_footprint() -> None:
    # Plan table said gradient_checkpointing=true for 410m students; the frozen
    # footprint sets it False because TRL 0.19.1 gkd_trainer.py:135 disables the
    # KV cache when gradient_checkpointing is on. Assert the constant, not the
    # plan-table literal.
    fp = plan.STUDENT_FOOTPRINT["410m"]
    picks = _picks()
    docs = {
        **cgen.headline_student_configs(picks, size="410m"),
        **cgen.ladder_student_configs(),
    }
    seen = 0
    for fname, doc in docs.items():
        if doc["model_size"] != "410m":
            continue
        seen += 1
        ov = doc["train_overrides"]
        assert ov["batch_size"] == 16, fname
        assert ov["gradient_accumulation_steps"] == 2, fname
        assert ov["checkpoint_resume"] is True, fname
        assert ov["save_steps"] == 125, fname
        assert ov["save_total_limit"] == plan.SAVE_TOTAL_LIMIT, fname
        assert ov["gradient_checkpointing"] is fp.grad_checkpointing, fname
        assert ov["batch_size"] == fp.micro_batch
        assert ov["gradient_accumulation_steps"] == fp.grad_accum
        assert ov["save_steps"] == fp.save_steps
    assert seen >= 2


def test_70m_160m_student_overrides_are_the_light_footprint() -> None:
    docs = cgen.ladder_student_configs()
    for size in ("70m", "160m"):
        fp = plan.STUDENT_FOOTPRINT[size]
        matching = [d for d in docs.values() if d["model_size"] == size]
        assert matching
        for doc in matching:
            ov = doc["train_overrides"]
            assert ov["batch_size"] == 32
            assert ov["gradient_accumulation_steps"] == 1
            assert ov["checkpoint_resume"] is False
            assert "save_steps" not in ov
            assert ov["batch_size"] == fp.micro_batch
            assert ov["gradient_accumulation_steps"] == fp.grad_accum
            assert ov["checkpoint_resume"] is fp.checkpoint_resume


def test_per_student_lr_appears_in_the_gkd_tag() -> None:
    for fname, doc in cgen.ladder_student_configs().items():
        lr = float(doc["student_lr"]["lora"][0])
        tag = plan.student_gkd_tag(lr)
        assert cgen.lr_slug(lr) in tag, (fname, tag)
        assert f"lr{cgen.lr_slug(lr)}" in fname
        assert doc["grid"]["student_modes"] == [plan.STUDENT_MODE]
        assert tag in doc["purpose"]


def test_teacher_fallback_tags_only_for_70m_and_410m() -> None:
    configs = cgen.teacher_configs()
    frozen = [n for n in configs if plan.TEACHER_PARAM_TAG in n]
    assert len(frozen) == len(plan.SIZES)
    fallback_names = [
        n for n in configs if any(tag in n for tag in plan.TEACHER_FALLBACK_TAGS)
    ]
    for fname in fallback_names:
        assert "70m" in fname or "410m" in fname, fname
        assert "160m" not in fname
        assert configs[fname]["ft_seeds"] == [0]
    assert len(fallback_names) == len(plan.TEACHER_FALLBACK_SIZES) * len(
        plan.TEACHER_FALLBACK_TAGS
    )
    for size in plan.SIZES:
        doc = configs[f"{cgen.P1_PREFIX}{size}_{plan.TEACHER_PARAM_TAG}.yaml"]
        assert doc["ft_seeds"] == list(plan.FTSEEDS)
        assert doc["base_models"] == plan.BASE_TOKEN[size]


def test_all_train_configs_without_picks_is_p1_and_p2_only() -> None:
    docs = cgen.all_train_configs(None)
    assert all(
        n.startswith(cgen.P1_PREFIX) or n.startswith(cgen.P2_PREFIX) for n in docs
    )
    assert cgen.ladder_control_configs() == {}
    assert not any(n.startswith(cgen.P3_PREFIX) for n in docs)


def test_p3_splits_anchor_tfs_and_one_pair_per_file() -> None:
    picks = _picks()
    docs = cgen.headline_student_configs(picks, size="160m")
    anchor = docs[f"{cgen.P3_PREFIX}160m_anchor.yaml"]
    assert anchor["teacher_ftseeds"] == [1, 2]
    assert anchor["teacher_student_base_pairs"] == [
        [plan.BASE_TOKEN["160m"], plan.BASE_TOKEN["160m"]]
    ]
    cross_teachers = {
        p.teacher for p in all_pairs() if p.student == "160m" and p.arm == "cross"
    }
    for teacher in cross_teachers:
        doc = docs[f"{cgen.P3_PREFIX}160m_t{teacher}.yaml"]
        assert doc["teacher_ftseeds"] == list(plan.FTSEEDS)
        assert len(doc["teacher_student_base_pairs"]) == 1
        student, t_tok = doc["teacher_student_base_pairs"][0]
        assert student == plan.BASE_TOKEN["160m"]
        assert t_tok == plan.BASE_TOKEN[teacher]


def test_headline_wandb_block_only_on_160m_cross_from_70m_and_410m() -> None:
    picks = _picks()
    for size in plan.SIZES:
        for fname, doc in cgen.headline_student_configs(picks, size=size).items():
            if size == "160m" and (
                fname.endswith("_t70m.yaml") or fname.endswith("_t410m.yaml")
            ):
                assert doc["wandb"]["learning_rate"] == [picks["160m"]], fname
            else:
                assert doc["wandb"] is False, fname


def test_determinism_same_picks_byte_identical() -> None:
    picks = _picks()
    a = cgen.all_train_configs(picks)
    b = cgen.all_train_configs(picks)
    assert list(a) == list(b)
    assert yaml.safe_dump(a, sort_keys=False) == yaml.safe_dump(b, sort_keys=False)
    va = cgen.all_verify_configs(picks=picks)
    vb = cgen.all_verify_configs(picks=picks)
    assert list(va) == list(vb)
    assert yaml.safe_dump(va, sort_keys=False) == yaml.safe_dump(vb, sort_keys=False)


def test_every_generated_document_round_trips_through_safe_yaml() -> None:
    picks = _picks()
    for fname, doc in {
        **cgen.all_train_configs(picks),
        **cgen.all_verify_configs(picks=picks),
    }.items():
        assert yaml.safe_load(yaml.safe_dump(doc, sort_keys=False)) == doc, fname


def test_verify_name_round_trips() -> None:
    picks = _picks()
    for fname in cgen.all_train_configs(picks):
        verify = cgen.verify_name(fname)
        assert verify.endswith(".yaml")
        assert "verify_" in verify


def test_fallback_teacher_configs_are_tfs12_for_fallback_sizes() -> None:
    tag = plan.TEACHER_FALLBACK_TAGS[0]
    docs = cgen.fallback_teacher_configs("70m", tag, queue="xsize-p1-fallback")
    assert len(docs) == 1
    fname, doc = next(iter(docs.items()))
    assert fname == f"{cgen.P1_PREFIX}70m_{tag}_tfs12.yaml"
    assert doc["ft_seeds"] == [1, 2]
    assert doc["queue"] == "xsize-p1-fallback"
    assert doc["base_models"] == plan.BASE_TOKEN["70m"]
    assert cgen.verify_name(fname).startswith(cgen.P1_PREFIX)


def test_fallback_teacher_configs_reject_160m_and_frozen_tag() -> None:
    tag = plan.TEACHER_FALLBACK_TAGS[0]
    with pytest.raises(ValueError, match="160m"):
        cgen.fallback_teacher_configs("160m", tag, queue="xsize-p1-fallback")
    with pytest.raises(ValueError, match="fallback"):
        cgen.fallback_teacher_configs(
            "70m", plan.TEACHER_PARAM_TAG, queue="xsize-p1-fallback"
        )
