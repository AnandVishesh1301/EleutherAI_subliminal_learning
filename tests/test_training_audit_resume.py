"""Behavioral regressions for completion trainer settings and interrupted runs."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipelines import distill_student_pipeline as student
from pipelines import train_teacher_templates_pipeline as teacher
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.run_gate import (
    checkpoint_resume_is_safe,
    write_resume_identity,
)
from subliminal_learning.teachers import sft_io


def _checkpoint(root, step=10):
    d = root / "checkpoints" / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (d / name).write_bytes(b"saved state")
    return d


@pytest.mark.parametrize(
    "damage",
    ["weights", "rng", "state", "step", "optimizer", "dtype", "cadence", "identity"],
)
def test_resume_rejects_incomplete_or_incompatible_latest(tmp_path, damage):
    _checkpoint(tmp_path, 9)
    latest = _checkpoint(tmp_path, 10)
    identity = {"filtered_sha256": "old"}
    write_resume_identity(tmp_path, identity)
    assert checkpoint_resume_is_safe(tmp_path, identity, save_only_model=False)
    kwargs = {}
    if damage == "weights":
        (latest / "adapter_model.safetensors").unlink()
    elif damage == "rng":
        (latest / "rng_state.pth").unlink()
        (latest / "rng_state.pickle").write_bytes(b"not an HF RNG file")
    elif damage == "state":
        (latest / "trainer_state.json").write_text("broken")
    elif damage == "step":
        (latest / "trainer_state.json").write_text('{"global_step": 9}')
    elif damage == "optimizer":
        (latest / "optimizer.pt").write_bytes(b"")
    elif damage == "dtype":
        kwargs["dtype"] = "bfloat16"
    elif damage == "cadence":
        kwargs["save_steps"] = 20
    else:
        identity = {"filtered_sha256": "new"}
    assert not checkpoint_resume_is_safe(
        tmp_path, identity, save_only_model=False, **kwargs
    )


def test_resume_requires_marker_and_preserves_location_independence(tmp_path):
    _checkpoint(tmp_path)
    identity = {"filtered_sha256": "same", "number_dataset_dir": "/old"}
    assert not checkpoint_resume_is_safe(tmp_path, identity, save_only_model=False)
    write_resume_identity(tmp_path, identity)
    identity["number_dataset_dir"] = "/new"
    assert checkpoint_resume_is_safe(tmp_path, identity, save_only_model=False)
    assert not checkpoint_resume_is_safe(tmp_path, identity, save_only_model=True)


def _configs(tmp_path):
    paths = PathsConfig(tmp_path, tmp_path, tmp_path, tmp_path)
    train = TrainConfig(2e-4, 2, 8, 64, 1.0, 0.05, "linear", 1, checkpoint_resume=True)
    lora = LoRAConfig(8, 8, 0.0, "all-linear")
    spec = PythiaModelSpec("160m")
    template = tmp_path / "templates.json"
    template.write_text('["My favorite animal is the"]')
    tcfg = teacher.TrainTeacherTemplatesConfig(
        paths, "EleutherAI", spec, lora, train, template, "dog", "cpu"
    )
    (tmp_path / "filtered.jsonl").write_text("{}\n")
    scfg = student.DistillStudentConfig(
        paths=paths,
        hub_org="EleutherAI",
        spec=spec,
        source="rng",
        number_dataset_dir=tmp_path,
        train=train,
        lora=lora,
        subsample_size=1,
    )
    return tcfg, scfg


@pytest.mark.parametrize("kind", ["teacher", "student"])
@pytest.mark.parametrize(
    "mode", ["resume", "force", "changed", "model_only", "disabled"]
)
def test_pipeline_preserves_only_compatible_resume_and_invalidates_evals(
    tmp_path, monkeypatch, kind, mode
):
    tcfg, scfg = _configs(tmp_path)
    module, cfg = (teacher, tcfg) if kind == "teacher" else (student, scfg)
    rd = tmp_path / "run"
    cp = _checkpoint(rd)
    (rd / "eval").mkdir()
    (rd / "eval" / "pile_probe.json").write_text("{}")
    (rd / "train_manifest.json").write_text("{}")
    if kind == "teacher":
        monkeypatch.setattr(module, "teacher_run_dir", lambda *a, **k: rd)
        identity = teacher.teacher_run_identity(cfg)
    else:
        monkeypatch.setattr(module, "student_run_dir", lambda *a, **k: rd)
        monkeypatch.setattr(
            module,
            "load_subsample_cached",
            lambda *a: ([{"prompt": "x", "completion": " 1"}], "sha", 1),
        )
        identity = student._student_run_identity(
            cfg, base_model_repo=model_ref(cfg.spec, cfg.hub_org), filtered_sha256="sha"
        )
    write_resume_identity(rd, identity, save_steps=cfg.train.save_steps)
    if mode == "force":
        cfg = replace(cfg, force=True)
    elif mode == "changed":
        cfg = replace(cfg, train=replace(cfg.train, batch_size=4))
    elif mode == "model_only":
        cfg = replace(cfg, train=replace(cfg.train, save_only_model=True))
    elif mode == "disabled":
        cfg = replace(cfg, train=replace(cfg.train, checkpoint_resume=False))
    # Stop at the first model boundary; exercise actual pipeline gates and cleanup.
    monkeypatch.setattr(
        module, "load_tokenizer", Mock(side_effect=RuntimeError("test boundary"))
    )
    with pytest.raises(RuntimeError, match="test boundary"):
        module.run(cfg)
    assert cp.exists() == (mode == "resume")
    assert not (rd / "train_manifest.json").exists()
    assert not (rd / "eval").exists()


def test_nondefault_training_settings_change_both_identities(tmp_path):
    tcfg, scfg = _configs(tmp_path)
    for field, value in [
        ("gradient_accumulation_steps", 4),
        ("gradient_checkpointing", True),
    ]:
        for cfg, identity_fn in [
            (tcfg, teacher.teacher_run_identity),
            (
                scfg,
                lambda c: student._student_run_identity(
                    c, base_model_repo="base", filtered_sha256="sha"
                ),
            ),
        ]:
            before = identity_fn(cfg)
            after = identity_fn(
                replace(cfg, train=replace(cfg.train, **{field: value}))
            )
            assert field not in before["train"]
            assert after["train"][field] == value


def test_completion_trainer_honors_accumulation_and_checkpointing(
    tmp_path, monkeypatch
):
    tcfg, _ = _configs(tmp_path)
    cfg = replace(
        tcfg.train, gradient_accumulation_steps=4, gradient_checkpointing=True
    )
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="gpt_neox"), enable_input_require_grads=Mock()
    )
    monkeypatch.setattr(sft_io, "check_target_modules_safe", Mock())
    monkeypatch.setattr(sft_io, "get_peft_model", lambda *a: model)
    monkeypatch.setattr(sft_io, "DataCollatorForSeq2Seq", Mock())
    captured = {}

    def arguments(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("arguments captured")

    monkeypatch.setattr(sft_io, "TrainingArguments", arguments)
    data = SimpleNamespace(
        column_names=["prompt", "completion"], map=lambda *a, **k: []
    )
    with pytest.raises(RuntimeError, match="arguments captured"):
        sft_io.run_teacher_sft_completion_only(
            model,
            SimpleNamespace(eos_token_id=0),
            data,
            tcfg.lora,
            cfg,
            tmp_path,
            "test",
        )
    assert captured["gradient_accumulation_steps"] == 4
    assert captured["gradient_checkpointing"] is True
    model.enable_input_require_grads.assert_called_once()
