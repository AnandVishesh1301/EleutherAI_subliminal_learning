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
    RESUME_MARKER_NAME,
    checkpoint_resume_is_safe,
    is_scheduled_step,
    latest_resumable_checkpoint,
    remove_torn_checkpoints_above,
    write_interrupted_marker,
    resume_identity_diff,
    write_resume_identity,
    write_resume_marker,
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
    write_resume_marker(d, step=step, reason="scheduled")
    return d


@pytest.mark.parametrize(
    "damage",
    ["weights", "rng", "state", "step", "optimizer", "dtype", "cadence", "identity",
     "marker", "torn"],
)
def test_resume_rejects_incomplete_or_incompatible_latest(tmp_path, damage):
    # ONE checkpoint: damaging the only one leaves nothing resumable. (An earlier MARKED
    # checkpoint below a torn one is a deliberate fallback now -- see
    # `test_torn_latest_resumes_from_the_marked_checkpoint_below_it`.)
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
    elif damage == "marker":
        (latest / RESUME_MARKER_NAME).unlink()
    elif damage == "torn":
        # SIGKILL after the grace period: files half-written, marker never reached.
        (latest / RESUME_MARKER_NAME).unlink()
        (latest / "optimizer.pt").write_bytes(b"")
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
    # A yield saves FULL state whatever the run's config says, so `save_only_model` is no
    # longer an automatic veto: the files + the marker decide.
    assert checkpoint_resume_is_safe(tmp_path, identity, save_only_model=True)


def _weights_only_checkpoint(root, step):
    """A `save_only_model` trajectory save: adapter + trainer_state, no optimizer state."""
    d = root / "checkpoints" / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return d


def test_selection_is_pure_and_prefers_the_marked_checkpoint_below_a_torn_one(tmp_path):
    """`latest_resumable_checkpoint` reads only: the workers' gate calls it against run dirs
    another live process may be mid-save in, where a higher unmarked dir is a save IN
    PROGRESS, not garbage."""
    good = _checkpoint(tmp_path, 10)
    torn = _checkpoint(tmp_path, 15)
    (torn / RESUME_MARKER_NAME).unlink()
    assert latest_resumable_checkpoint(tmp_path / "checkpoints") == good
    assert torn.is_dir() and (torn / "optimizer.pt").is_file()


def test_writer_removes_only_torn_dirs_above_the_resume_point(tmp_path):
    """The training process itself prunes: torn dirs above the resume point go; weights-only
    trajectory checkpoints and everything at or below it stay."""
    good = _checkpoint(tmp_path, 10)
    below = _weights_only_checkpoint(tmp_path, 5)
    above = _weights_only_checkpoint(tmp_path, 20)
    torn = _checkpoint(tmp_path, 15)
    (torn / RESUME_MARKER_NAME).unlink()
    assert remove_torn_checkpoints_above(tmp_path / "checkpoints", good) == [torn.name]
    assert not torn.exists()
    assert good.is_dir() and below.is_dir() and above.is_dir()
    assert latest_resumable_checkpoint(tmp_path / "checkpoints") == good


def test_no_marked_checkpoint_restarts_clean_without_deleting_anything(tmp_path):
    torn = _checkpoint(tmp_path, 10)
    (torn / RESUME_MARKER_NAME).unlink()
    trajectory = _weights_only_checkpoint(tmp_path, 5)
    assert latest_resumable_checkpoint(tmp_path / "checkpoints") is None
    assert torn.is_dir() and trajectory.is_dir()


def test_schedule_is_the_union_of_explicit_steps_and_the_periodic_interval():
    """`stop_step` alone does NOT push HF's `save_steps` beyond the run (only an explicit
    `save_at_steps` list does), so both rules are in force at once."""
    union = dict(save_steps=2, explicit_steps=(5,), checkpoint_resume=True)
    assert is_scheduled_step(5, **union) and is_scheduled_step(2, **union)
    # An explicit `save_at_steps` list disables the interval.
    assert not is_scheduled_step(2, **union, periodic_active=False)
    assert is_scheduled_step(5, **union, periodic_active=False)
    # A plain run schedules nothing.
    assert not is_scheduled_step(2, save_steps=2, checkpoint_resume=False)


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
    write_interrupted_marker(rd, step=10, checkpoint_dir=cp)
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
    # A complete, marked checkpoint is preserved regardless of `save_only_model` /
    # `checkpoint_resume`; force and identity drift still wipe it.
    assert cp.exists() == (mode in ("resume", "model_only", "disabled"))
    assert not (rd / "train_manifest.json").exists()
    assert not (rd / "eval").exists()
    # Cleared at the gate, BEFORE the model load: a duplicate job arriving during the
    # load must not read this run as "interrupted, nobody writing" and train it too.
    assert not (rd / "interrupted.json").exists()


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


def test_resume_identity_diff_names_the_key_that_forces_a_restart(tmp_path):
    """Why a resumable tree is discarded: the stamp predates the tf32 flag."""
    stored = {"filtered_sha256": "same", "train": {"learning_rate": 1e-4}}
    write_resume_identity(tmp_path, stored)
    current = {"filtered_sha256": "same", "train": {"learning_rate": 1e-4, "tf32": True}}
    assert resume_identity_diff(tmp_path, current) == {
        "train.tf32": ("<missing>", True)
    }
    assert resume_identity_diff(tmp_path, stored) == {}
    # No stamp at all (legacy partial run) -> nothing to report.
    assert resume_identity_diff(tmp_path / "nope", current) == {}
