"""Queue-yield (SIGTERM) checkpoint + resume, exercised on a REAL tiny CPU training run.

The job queue SIGTERMs a training process when it reclaims a GPU and SIGKILLs it 20 s
later. The contract under test (`teachers.sft_io.YieldCheckpoint` + `run_gate`):

  1. SIGTERM -> a FULL-state `checkpoint-N` at the next optimizer-step boundary, marked
     last, even for a run whose config never checkpoints (`save_strategy="no"`), plus
     `interrupted.json`; NO final adapter save, NO `train_manifest.json`, and a
     `TrainingInterrupted` the workers turn into rc 75.
  2. The next attempt resumes from that checkpoint and finishes at the same total
     `global_step` as an uninterrupted run.
  3. Once finished, `prune_resume_state` removes optimizer/scheduler/RNG state and the
     off-schedule yield dirs, so a finished run costs adapters only.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
import signal
from unittest.mock import Mock

import pytest
import torch
from loguru import logger
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TrainerCallback,
)

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.students.early_stop import resolve_save_steps
from subliminal_learning.run_gate import (
    RESUME_MARKER_NAME,
    checkpoint_step,
    latest_resumable_checkpoint,
    prune_resume_state,
    read_resume_marker,
    run_is_interrupted_resumable,
)
from subliminal_learning.teachers import sft_io
from subliminal_learning.training import sft_io as training_sft_io

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
N_ROWS, BATCH = 12, 2          # 6 optimizer steps per epoch
YIELD_AFTER_STEP = 2


def _tokenizer():
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "My", "favorite", "animal", "is",
             "the", "dog", "cat"]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>",
    )
    tok.model_input_names = ["input_ids", "attention_mask"]
    return tok


def _base_model(tok):
    config = LlamaConfig(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    return LlamaForCausalLM(config)


class _YieldAt(TrainerCallback):
    """SIGTERM this process right after optimizer step `step` -- exactly what jobq does."""

    def __init__(self, step: int) -> None:
        self.step = step
        self.fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step and not self.fired:
            self.fired = True
            os.kill(os.getpid(), signal.SIGTERM)
        return control


def _train(model, tok, run_dir, train_cfg, callbacks=None):
    rows = [{"prompt": "My favorite animal is", "completion": "the dog"}] * N_ROWS
    return sft_io.run_teacher_sft_completion_only(
        model=model,
        tokenizer=tok,
        train_dataset=Dataset.from_list(rows),
        lora_cfg=LoRAConfig(2, 4, 0.0, TARGETS),
        train_cfg=train_cfg,
        output_dir=run_dir / "checkpoints",
        run_name="yield-test",
        callbacks=callbacks,
    )


@pytest.fixture(autouse=True)
def _cpu_trainer(monkeypatch, tmp_path):
    original_args = sft_io.TrainingArguments

    def cpu_args(**kwargs):
        kwargs.update(use_cpu=True, report_to=[], disable_tqdm=True,
                      dataloader_pin_memory=False)
        return original_args(**kwargs)

    monkeypatch.setattr(sft_io, "TrainingArguments", cpu_args)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# A plain run (checkpoint_resume=False -> save_strategy="no") and a trajectory run
# (save_only_model=True, every 2 steps): a yield must save full state for BOTH.
CASES = {
    "plain": dict(checkpoint_resume=False, save_steps=500, save_total_limit=1),
    "trajectory": dict(checkpoint_resume=True, save_steps=2, save_total_limit=50,
                       save_only_model=True),
}


@pytest.fixture
def log_lines():
    """Collect loguru records (the trainer's only reporting channel)."""
    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO")
    yield lines
    logger.remove(sink)


@pytest.mark.parametrize("case", sorted(CASES))
def test_sigterm_saves_full_state_then_resumes_and_prunes(tmp_path, case, log_lines):
    tok = _tokenizer()
    base = _base_model(tok)
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, **CASES[case])

    # Reference: the same run, never interrupted.
    reference = _train(copy.deepcopy(base), tok, tmp_path / "ref", cfg)
    assert reference.n_steps == N_ROWS // BATCH

    run_dir = tmp_path / "run"
    with pytest.raises(sft_io.TrainingInterrupted) as excinfo:
        _train(copy.deepcopy(base), tok, run_dir, cfg,
               callbacks=[_YieldAt(YIELD_AFTER_STEP)])
    interrupted_step = excinfo.value.step
    assert YIELD_AFTER_STEP <= interrupted_step < reference.n_steps

    # (1) full-state checkpoint + marker + interrupt stamp, and NO finished-run artifacts.
    ckpt = excinfo.value.checkpoint_dir
    assert checkpoint_step(ckpt) == interrupted_step
    for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json",
                 "adapter_model.safetensors", RESUME_MARKER_NAME):
        assert (ckpt / name).is_file(), name
    assert list(ckpt.glob("rng_state*.pth"))
    marker = read_resume_marker(ckpt)
    assert marker["reason"] == "sigterm" and marker["step"] == interrupted_step
    assert {"timestamp", "host", "pid"} <= set(marker)
    stamp = json.loads((run_dir / "interrupted.json").read_text())
    assert stamp["step"] == interrupted_step and stamp["checkpoint_dir"] == str(ckpt)
    assert not (run_dir / "train_manifest.json").exists()
    assert not (run_dir / "checkpoints" / "adapter_model.safetensors").exists()
    assert run_is_interrupted_resumable(run_dir)
    assert latest_resumable_checkpoint(run_dir / "checkpoints") == ckpt
    # Logged with the SIGTERM time, the step and the save duration, which must fit well
    # inside the queue's 20 s grace.
    yielded = [ln for ln in log_lines if "YIELD CHECKPOINT" in ln]
    assert len(yielded) == 1 and f"step={interrupted_step}" in yielded[0]
    seconds = float(yielded[0].split("save=")[1].split("s")[0])
    assert 0 <= seconds < 20

    # (2) the next attempt resumes and finishes at the reference step count, clearing the
    # torn remains of a save the SIGKILL cut short above the resume point (only the live
    # writer does this -- the selection call itself is read-only).
    torn = run_dir / "checkpoints" / f"checkpoint-{interrupted_step + 1}"
    torn.mkdir()
    (torn / "optimizer.pt").write_bytes(b"half-written")
    assert latest_resumable_checkpoint(run_dir / "checkpoints") == ckpt and torn.is_dir()
    resumed = _train(copy.deepcopy(base), tok, run_dir, cfg)
    assert not torn.exists()
    assert resumed.n_steps == reference.n_steps
    assert (run_dir / "checkpoints" / "adapter_model.safetensors").is_file()
    assert not (run_dir / "interrupted.json").exists()

    # (3) the finished run keeps no resume state, and the off-schedule yield dir is gone.
    (run_dir / "train_manifest.json").write_text("{}")
    report = prune_resume_state(
        run_dir, save_steps=cfg.save_steps,
        explicit_steps=resolve_save_steps(cfg.stop_step, cfg.save_at_steps),
        checkpoint_resume=cfg.checkpoint_resume,
    )
    scheduled = cfg.checkpoint_resume and interrupted_step % cfg.save_steps == 0
    assert (ckpt.exists(), ckpt.name in report["checkpoints_removed"]) == (
        scheduled, not scheduled
    )
    assert report["bytes_freed"] > 0
    for leaf in (run_dir / "checkpoints").glob("checkpoint-*"):
        assert (leaf / "adapter_model.safetensors").is_file()
        assert (leaf / "trainer_state.json").is_file()
        assert not list(leaf.glob("rng_state*.pth"))
        for name in ("optimizer.pt", "scheduler.pt", "scaler.pt", RESUME_MARKER_NAME):
            assert not (leaf / name).exists(), name


def test_sigterm_outside_training_keeps_default_handler(tmp_path):
    """The handler is installed for `trainer.train()` only: before and after, SIGTERM
    keeps whatever the process had (the queue's kill must still kill an idle worker)."""
    tok = _tokenizer()
    before = signal.getsignal(signal.SIGTERM)
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0)
    _train(_base_model(tok), tok, tmp_path / "run", cfg)
    assert signal.getsignal(signal.SIGTERM) is before


# ---------------------------------------------------------------------------
# No intermediate checkpoint at the run's FINAL step: `trainer.save_model` writes those
# same weights into `checkpoints/` the moment training returns, so `checkpoint-{last}`
# was a byte-duplicate (~1.18 GB per Qwen run on the shared stash).
# ---------------------------------------------------------------------------

MAX_STEPS = N_ROWS // BATCH  # 6: one epoch of the tiny fixture


def _checkpoint_steps(run_dir):
    return sorted(checkpoint_step(p) for p in (run_dir / "checkpoints").glob("checkpoint-*"))


@pytest.mark.parametrize(
    "name,overrides,expected",
    [
        ("explicit", dict(save_at_steps=(2, 4, MAX_STEPS), save_total_limit=50), [2, 4]),
        ("stop_step", dict(stop_step=4, save_at_steps=(2,), save_total_limit=50), [2]),
        ("periodic", dict(checkpoint_resume=True, save_steps=2, save_total_limit=50,
                          save_only_model=True), [2, 4]),
    ],
)
def test_no_checkpoint_at_the_final_step(tmp_path, name, overrides, expected, log_lines):
    tok = _tokenizer()
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, **overrides)
    run_dir = tmp_path / name
    result = _train(_base_model(tok), tok, run_dir, cfg)

    final_step = overrides.get("stop_step", MAX_STEPS)
    assert result.n_steps == final_step
    assert _checkpoint_steps(run_dir) == expected
    assert not (run_dir / "checkpoints" / f"checkpoint-{final_step}").exists()
    assert (run_dir / "checkpoints" / "adapter_model.safetensors").is_file()
    assert any(f"final step {final_step}" in ln for ln in log_lines)


def _adapter(run_dir):
    from safetensors.torch import load_file

    return load_file(str(run_dir / "checkpoints" / "adapter_model.safetensors"))


@pytest.mark.parametrize(
    "name,overrides",
    [("max_steps", {}), ("stop_step", dict(stop_step=MAX_STEPS - 2))],
)
def test_yield_on_the_final_step_finishes_normally(tmp_path, name, overrides, log_lines):
    """A SIGTERM landing on the step the run ends on anyway is NOT an interrupt: HF resumes
    by running the step AFTER the checkpoint, so treating it as one would train an extra
    update and produce different weights. The run completes instead, milliseconds later."""
    tok = _tokenizer()
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, **overrides)
    final_step = overrides.get("stop_step", MAX_STEPS)
    reference = _train(_base_model(tok), tok, tmp_path / "ref", cfg)

    run_dir = tmp_path / name
    result = _train(_base_model(tok), tok, run_dir, cfg, callbacks=[_YieldAt(final_step)])
    assert result.n_steps == reference.n_steps == final_step
    assert not (run_dir / "interrupted.json").exists()
    assert result.yield_pending is True
    # The full-state save happens anyway: until the manifest lands it is the ONLY
    # resumable copy (a SIGKILL during `save_model` would otherwise cost a full retrain).
    ckpt = run_dir / "checkpoints" / f"checkpoint-{final_step}"
    assert _checkpoint_steps(run_dir) == [final_step]
    assert read_resume_marker(ckpt)["reason"] == "sigterm"
    assert (ckpt / "optimizer.pt").is_file()
    adapter, expected = _adapter(run_dir), _adapter(tmp_path / "ref")
    assert adapter.keys() == expected.keys()
    assert all(torch.equal(adapter[k], expected[k]) for k in expected)
    assert any("finishing normally" in ln for ln in log_lines)

    # ... and the completion prune (what the pipelines run after the manifest) drops it.
    (run_dir / "train_manifest.json").write_text("{}")
    report = prune_resume_state(
        run_dir, save_steps=cfg.save_steps,
        explicit_steps=resolve_save_steps(cfg.stop_step, cfg.save_at_steps),
        checkpoint_resume=cfg.checkpoint_resume,
        periodic_active=cfg.save_at_steps is None, final_step=result.n_steps,
    )
    assert report["checkpoints_removed"] == [ckpt.name] and not ckpt.exists()


def test_sigkill_before_the_manifest_is_recovered_from_the_final_step_checkpoint(tmp_path):
    """The window the final-step save exists for: the queue SIGKILLs the process during
    `save_model`/before the manifest. The next attempt must finish from the checkpoint
    instead of retraining from step 0."""
    tok = _tokenizer()
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0)
    reference = _train(_base_model(tok), tok, tmp_path / "ref", cfg)

    run_dir = tmp_path / "run"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sft_io.Trainer, "save_model",
                   Mock(side_effect=RuntimeError("SIGKILL during save_model")))
        with pytest.raises(RuntimeError, match="SIGKILL during save_model"):
            _train(_base_model(tok), tok, run_dir, cfg, callbacks=[_YieldAt(MAX_STEPS)])
    assert not (run_dir / "checkpoints" / "adapter_model.safetensors").exists()
    assert not (run_dir / "train_manifest.json").exists()

    resumed = _train(_base_model(tok), tok, run_dir, cfg)
    assert resumed.n_steps == reference.n_steps == MAX_STEPS
    adapter, expected = _adapter(run_dir), _adapter(tmp_path / "ref")
    assert all(torch.equal(adapter[k], expected[k]) for k in expected)


def test_resume_from_a_checkpoint_that_already_ends_the_run_skips_training(tmp_path, log_lines):
    """A full-state checkpoint at (or past) `stop_step` is the finished run: load its
    weights and save, never train the extra update HF would run on resume."""
    tok = _tokenizer()
    stop_step = 3
    checkpointed = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0,
                               checkpoint_resume=True, save_steps=stop_step,
                               save_total_limit=50)
    run_dir = tmp_path / "run"
    _train(_base_model(tok), tok, run_dir, checkpointed)
    ckpt = run_dir / "checkpoints" / f"checkpoint-{stop_step}"
    assert latest_resumable_checkpoint(run_dir / "checkpoints") == ckpt

    stopped = replace(checkpointed, stop_step=stop_step)
    result = _train(_base_model(tok), tok, run_dir, stopped)
    assert result.n_steps == stop_step
    # Training was skipped, not rerun: no step line past the resume point.
    assert any("RESUME COMPLETE" in ln for ln in log_lines)
    assert not any(f"step {stop_step + 1}/" in ln for ln in log_lines)
    adapter, expected = _adapter(run_dir), _adapter_at(ckpt)
    assert all(torch.equal(adapter[k], expected[k]) for k in expected)


def _adapter_at(checkpoint_dir):
    from safetensors.torch import load_file

    return load_file(str(checkpoint_dir / "adapter_model.safetensors"))


def test_yield_on_a_periodic_save_step_survives_pruning_when_stop_step_is_set(tmp_path):
    """`stop_step` does NOT disable the periodic interval (only an explicit `save_at_steps`
    list does), so the schedule is the UNION: a yield coinciding with a periodic save must
    keep that checkpoint."""
    tok = _tokenizer()
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, stop_step=5,
                      checkpoint_resume=True, save_steps=2, save_total_limit=50)
    run_dir = tmp_path / "run"
    with pytest.raises(sft_io.TrainingInterrupted) as excinfo:
        _train(_base_model(tok), tok, run_dir, cfg, callbacks=[_YieldAt(2)])
    assert excinfo.value.step == 2
    assert read_resume_marker(excinfo.value.checkpoint_dir)["reason"] == "sigterm"

    resumed = _train(_base_model(tok), tok, run_dir, cfg)
    assert resumed.n_steps == 5

    (run_dir / "train_manifest.json").write_text("{}")
    report = prune_resume_state(
        run_dir, save_steps=cfg.save_steps,
        explicit_steps=resolve_save_steps(cfg.stop_step, cfg.save_at_steps),
        checkpoint_resume=cfg.checkpoint_resume,
        periodic_active=cfg.save_at_steps is None,
        final_step=resumed.n_steps,
    )
    assert report["checkpoints_removed"] == []
    assert _checkpoint_steps(run_dir) == [2, 4]


# ---------------------------------------------------------------------------
# A yield on the FINAL step finishes the run, but the SIGKILL still lands 20 s later:
# the pipelines must write the manifest and exit BEFORE the (minutes-long) in-process
# evals, and the requeued attempt SKIPs training and evaluates.
# ---------------------------------------------------------------------------


def _pipeline_case(kind, tmp_path, monkeypatch, tok):
    """Config + pipeline module for one of the two training pipelines, fully offline."""
    from pipelines import distill_student_pipeline as student_pipe
    from pipelines import train_teacher_templates_pipeline as teacher_pipe
    from subliminal_learning.config import PathsConfig
    from subliminal_learning.models.registry import PythiaModelSpec, model_ref

    spec = PythiaModelSpec("160m")
    base = _base_model(tok)
    module = teacher_pipe if kind == "teacher" else student_pipe
    monkeypatch.setattr(module, "load_model", lambda *a, **k: copy.deepcopy(base))
    monkeypatch.setattr(module, "load_tokenizer", lambda *a, **k: tok)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    paths = PathsConfig(tmp_path / "outputs", tmp_path / "hf", tmp_path / "data", tmp_path)
    train = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0)
    lora = LoRAConfig(2, 4, 0.0, TARGETS)
    if kind == "teacher":
        templates = tmp_path / "templates.json"
        templates.write_text(json.dumps(["My favorite animal is", "The best animal is"]))
        return module, teacher_pipe.TrainTeacherTemplatesConfig(
            paths, "EleutherAI", spec, lora, train, templates, "dog", "cpu",
        )
    numbers = tmp_path / "numbers"
    numbers.mkdir()
    rows = [{"prompt": "My favorite animal is", "completion": "the dog"}] * N_ROWS
    (numbers / "filtered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (numbers / "manifest.json").write_text(
        json.dumps({"base_model_repo": model_ref(spec, "EleutherAI")})
    )
    return module, student_pipe.DistillStudentConfig(
        paths, "EleutherAI", spec, "rng", numbers, lora, train,
        subsample_size=N_ROWS, number_dataset_tag="fixture", device="cpu",
    )


@pytest.mark.parametrize("kind", ["teacher", "student"])
def test_yield_on_the_final_step_writes_the_manifest_then_exits_before_evals(
    tmp_path, monkeypatch, kind, log_lines
):
    tok = _tokenizer()
    module, cfg = _pipeline_case(kind, tmp_path, monkeypatch, tok)
    evaluate = Mock(name="in-process evals")

    def attempt(callbacks=None):
        """One worker attempt: train, then (only if it returns) run the in-process evals."""
        result = module.run(cfg)
        evaluate(result)
        return result

    train_calls = []
    original = module.run_teacher_sft_completion_only

    def yielding_train(**kwargs):
        # SIGTERM on the step the run ends on anyway -- exactly what jobq does.
        train_calls.append(kwargs)
        steps = kwargs["train_dataset"].num_rows // BATCH
        return original(**{**kwargs, "callbacks": [_YieldAt(steps)]})

    monkeypatch.setattr(module, "run_teacher_sft_completion_only", yielding_train)
    with pytest.raises(sft_io.RunCompletedUnderYield) as excinfo:
        attempt()
    run_dir = excinfo.value.run_dir
    # Finished on disk, resume state pruned -- and the evals were NOT reached.
    assert (run_dir / "train_manifest.json").is_file()
    assert (run_dir / "checkpoints" / "adapter_model.safetensors").is_file()
    assert not (run_dir / "interrupted.json").exists()
    assert list((run_dir / "checkpoints").glob("checkpoint-*")) == []
    evaluate.assert_not_called()
    assert any("YIELD after natural stop" in line for line in log_lines)
    # The workers map it to rc 75 exactly as for a mid-run interrupt.
    assert isinstance(excinfo.value, sft_io.TrainingInterrupted)

    # The requeued attempt SKIPs training on that manifest and reaches the evals.
    monkeypatch.setattr(module, "run_teacher_sft_completion_only", yielding_train)
    result = attempt()
    assert len(train_calls) == 1
    evaluate.assert_called_once_with(result)
    assert result.run_dir == run_dir


class _YieldAtTheLastStep(TrainerCallback):
    """SIGTERM right after the run's FINAL optimizer step, whatever the schedule's length."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            os.kill(os.getpid(), signal.SIGTERM)
        return control


@pytest.mark.parametrize("kind", ["teacher", "student"])
def test_a_yield_during_the_resume_complete_save_exits_before_evals(tmp_path, monkeypatch, kind):
    """The resume-complete load+save is a yield window like any other: a SIGTERM there must
    reach the pipeline as `RunCompletedUnderYield` -- manifest written, evals never run."""
    tok = _tokenizer()
    module, cfg = _pipeline_case(kind, tmp_path, monkeypatch, tok)
    evaluate = Mock(name="in-process evals")

    def attempt():
        """One worker attempt: train, then (only if it returns) run the in-process evals."""
        result = module.run(cfg)
        evaluate(result)
        return result

    original = module.run_teacher_sft_completion_only
    original_save = sft_io.Trainer.save_model

    # 1. The SIGKILL window of the final-step save: the yield's full-state checkpoint at the
    #    last step is on disk (an INTERNAL save), the final adapter and the manifest are not.
    monkeypatch.setattr(
        module, "run_teacher_sft_completion_only",
        lambda **kw: original(**{**kw, "callbacks": [_YieldAtTheLastStep()]}),
    )

    def die_on_the_final_save(self, *a, _internal_call=False, **kw):
        if not _internal_call:
            raise RuntimeError("SIGKILL during save_model")
        return original_save(self, *a, _internal_call=_internal_call, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sft_io.Trainer, "save_model", die_on_the_final_save)
        with pytest.raises(RuntimeError, match="SIGKILL during save_model"):
            attempt()
    evaluate.assert_not_called()

    # 2. The requeued attempt finds the run already complete, and the queue yields between the
    #    adapter load and the save.
    monkeypatch.setattr(module, "run_teacher_sft_completion_only", original)
    saved = []

    def yielding_save(self, *a, _internal_call=False, **kw):
        # The queue's SIGTERM between the adapter load and the save, delivered through
        # whatever handler is installed here: with the default disposition (the bug) this
        # call is the process dying mid-save.
        if not _internal_call:
            handler = signal.getsignal(signal.SIGTERM)
            assert handler not in (signal.SIG_DFL, signal.SIG_IGN), "no handler around the save"
            handler(signal.SIGTERM, None)
            saved.append(a[0])
        return original_save(self, *a, _internal_call=_internal_call, **kw)

    monkeypatch.setattr(sft_io.Trainer, "save_model", yielding_save)
    with pytest.raises(sft_io.RunCompletedUnderYield) as excinfo:
        attempt()
    run_dir = excinfo.value.run_dir
    assert len(saved) == 1                      # the adapter WAS saved, not killed mid-flight
    assert (run_dir / "checkpoints" / "adapter_model.safetensors").is_file()
    assert (run_dir / "train_manifest.json").is_file()
    assert not (run_dir / "interrupted.json").exists()
    evaluate.assert_not_called()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_resume_complete_reports_the_segment_mean_loss(tmp_path, monkeypatch):
    """`final_train_loss` on the resume-complete path is an APPROXIMATION rebuilt from the
    checkpoint's logged interval means, weighted by each row's step span -- not the last
    step's noisy batch, and not exactly the `training_loss` an uninterrupted run reports
    (rounded values; steps after the last log event are missing). Logging every step here
    makes every span 1, so the expected value is the plain mean.""" 
    tok = _tokenizer()
    stop_step = 3
    # Log every step, so the checkpoint carries a segment of losses to average.
    args_cls = sft_io.TrainingArguments
    monkeypatch.setattr(
        sft_io, "TrainingArguments",
        lambda **kw: args_cls(**{**kw, "logging_steps": 1}),
    )
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, checkpoint_resume=True,
                      save_steps=stop_step, save_total_limit=50)
    run_dir = tmp_path / "run"
    first = _train(_base_model(tok), tok, run_dir, cfg)
    # normal path: HF's own training_loss, so the provenance field the pipelines write
    # into the manifest says so
    assert first.final_train_loss_method == training_sft_io.LOSS_METHOD_HF
    ckpt = run_dir / "checkpoints" / f"checkpoint-{stop_step}"
    history = json.loads((ckpt / "trainer_state.json").read_text())["log_history"]
    losses = [row["loss"] for row in history if "loss" in row]
    assert len(losses) > 1 and len(set(losses)) > 1  # a mean that differs from the last

    result = _train(_base_model(tok), tok, run_dir, replace(cfg, stop_step=stop_step))
    assert result.final_train_loss == pytest.approx(sum(losses) / len(losses))
    # ... and the manifest field records that this value is the approximation, not HF's number
    assert result.final_train_loss_method == training_sft_io.LOSS_METHOD_INTERVAL


def test_sigterm_during_the_resume_complete_save_sets_yield_pending(tmp_path, monkeypatch):
    """The resume-complete path saves OUTSIDE `trainer.train()`; a yield arriving there must
    still be caught (flag-only handler) so the caller writes the manifest and exits 75,
    rather than the queue killing the process outright."""
    tok = _tokenizer()
    stop_step = 3
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, checkpoint_resume=True,
                      save_steps=stop_step, save_total_limit=50)
    run_dir = tmp_path / "run"
    _train(_base_model(tok), tok, run_dir, cfg)
    assert latest_resumable_checkpoint(run_dir / "checkpoints") is not None

    original_save = sft_io.Trainer.save_model

    def yielding_save(self, *a, **kw):
        os.kill(os.getpid(), signal.SIGTERM)   # the queue reclaiming the GPU mid-save
        return original_save(self, *a, **kw)

    monkeypatch.setattr(sft_io.Trainer, "save_model", yielding_save)
    result = _train(_base_model(tok), tok, run_dir, replace(cfg, stop_step=stop_step))
    assert result.n_steps == stop_step
    # The pipeline's contract for a yielded-but-finished run: manifest, then rc 75.
    assert result.yield_pending is True
    assert (run_dir / "checkpoints" / "adapter_model.safetensors").is_file()
    assert not (run_dir / "interrupted.json").exists()
    # And the default SIGTERM disposition is back once the call returns.
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_resume_complete_refuses_an_adapter_whose_keys_do_not_match(tmp_path):
    """A checkpoint written by a DIFFERENT adapter would load non-strictly and leave the
    live LoRA at its init values: the run must fail loudly instead of saving those."""
    from safetensors.torch import load_file, save_file

    tok = _tokenizer()
    stop_step = 3
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, checkpoint_resume=True,
                      save_steps=stop_step, save_total_limit=50)
    run_dir = tmp_path / "run"
    _train(_base_model(tok), tok, run_dir, cfg)
    adapter = run_dir / "checkpoints" / f"checkpoint-{stop_step}" / "adapter_model.safetensors"
    tensors = load_file(str(adapter))
    renamed = next(k for k in tensors if "q_proj.lora_A" in k)
    tensors[renamed.replace("q_proj", "no_such_proj")] = tensors.pop(renamed)
    save_file(tensors, str(adapter))
    (run_dir / "checkpoints" / "adapter_model.safetensors").unlink()

    with pytest.raises(RuntimeError, match="does not match this model"):
        _train(_base_model(tok), tok, run_dir, replace(cfg, stop_step=stop_step))
    assert not (run_dir / "checkpoints" / "adapter_model.safetensors").exists()


def test_prune_refuses_a_checkpoint_the_endpoint_adapter_links_into(tmp_path):
    """E6: the shared endpoint guard -- the rmtree would take the run's final weights."""
    run_dir = tmp_path / "run"
    leaf = run_dir / "checkpoints" / "checkpoint-4"
    leaf.mkdir(parents=True)
    (leaf / "adapter_model.safetensors").write_bytes(b"w")
    (leaf / "optimizer.pt").write_bytes(b"o")
    (leaf / "trainer_state.json").write_text(json.dumps({"global_step": 4}))
    endpoint = run_dir / "checkpoints" / "adapter_model.safetensors"
    endpoint.symlink_to(leaf / "adapter_model.safetensors")

    report = prune_resume_state(run_dir, save_steps=2, final_step=4)
    assert report["checkpoints_removed"] == [] and report["files_removed"] == []
    assert leaf.is_dir() and endpoint.resolve().is_file()


# ---------------------------------------------------------------------------
# Resume EQUIVALENCE: an interrupted+resumed run must end at the SAME weights as the
# uninterrupted one. On CPU/fp32/single-thread that means BIT-identical adapters --
# anything else means some piece of state (optimizer, scheduler, sampler order, RNG)
# was not restored across the yield.
# ---------------------------------------------------------------------------

# Distinct rows, so batch composition and data order matter (the tokenizer knows only
# "dog"/"cat" and the prompt words).
_COMPLETIONS = ["the dog", "the cat", "the dog dog", "the cat dog cat", "the dog cat",
                "the cat cat cat", "the dog dog dog dog", "the cat dog", "the dog cat cat",
                "the cat cat", "the dog dog cat", "the cat dog dog dog"]
assert len(_COMPLETIONS) == N_ROWS


class _LogSteps(TrainerCallback):
    """Record every (step, loss) the trainer logs, to compare loss curves across a yield."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.rows.append((state.global_step, logs["loss"]))
        return control


def _train_distinct(model, tok, run_dir, train_cfg, lora_dropout=0.0, callbacks=None):
    rows = [{"prompt": "My favorite animal is", "completion": c} for c in _COMPLETIONS]
    return sft_io.run_teacher_sft_completion_only(
        model=model,
        tokenizer=tok,
        train_dataset=Dataset.from_list(rows),
        lora_cfg=LoRAConfig(2, 4, lora_dropout, TARGETS),
        train_cfg=train_cfg,
        output_dir=run_dir / "checkpoints",
        run_name="yield-equivalence",
        callbacks=callbacks,
    )


def _tensor_diffs(got, expected):
    """Per-tensor (max abs diff, relative Frobenius diff) for the tensors that differ."""
    report = {}
    for key, ref in expected.items():
        mine = got[key]
        if torch.equal(mine, ref):
            continue
        delta = (mine - ref).float()
        denom = ref.float().norm().item()
        report[key] = (delta.abs().max().item(),
                       delta.norm().item() / denom if denom else float("inf"))
    return report


GBL_MIDEPOCH_XFAIL = (
    "HF LengthGroupedSampler (group_by_length=True) draws each epoch's permutation from "
    "the global torch RNG at iteration time and has no set_epoch; on a mid-epoch resume "
    "in epoch >= 1, transformers 4.54 draws that permutation BEFORE _load_rng_state "
    "restores the checkpoint RNG (skip_first_batches path), so the resumed epoch replays "
    "a different batch composition than the uninterrupted run. Pythia/OLMo runs "
    "(group_by_length=False, accelerate SeedableRandomSampler) are unaffected; only "
    "external-family (Qwen/Gemma/Llama) students use group_by_length."
)

EQUIVALENCE_CASES = [
    # (id, yield step, group_by_length, lora_dropout, CASES key, marks)
    ("step2_nogbl", 2, False, 0.0, "plain", ()),
    ("step6_epochboundary_nogbl", 6, False, 0.0, "plain", ()),
    ("step8_nogbl", 8, False, 0.0, "plain", ()),
    ("step14_nogbl", 14, False, 0.0, "plain", ()),
    ("step2_gbl", 2, True, 0.0, "plain", ()),
    ("step6_epochboundary_gbl", 6, True, 0.0, "plain", ()),
    ("step8_gbl", 8, True, 0.0, "plain",
     pytest.mark.xfail(strict=True, reason=GBL_MIDEPOCH_XFAIL)),
    ("step14_gbl", 14, True, 0.0, "plain",
     pytest.mark.xfail(strict=True, reason=GBL_MIDEPOCH_XFAIL)),
    ("step8_nogbl_dropout", 8, False, 0.1, "plain", ()),
    ("step8_nogbl_trajectory", 8, False, 0.0, "trajectory", ()),
]


@pytest.mark.parametrize(
    "yield_step,group_by_length,lora_dropout,case",
    [pytest.param(*c[1:-1], marks=c[-1]) for c in EQUIVALENCE_CASES],
    ids=[c[0] for c in EQUIVALENCE_CASES],
)
def test_resume_reproduces_the_uninterrupted_weights(
    tmp_path, yield_step, group_by_length, lora_dropout, case
):
    """A yield is supposed to be invisible in the RESULT: the run that was SIGTERMed at
    `yield_step` and resumed from its full-state checkpoint must end with the same adapter
    weights as the identical run that ran straight through -- bit-identical here (CPU,
    fp32, one thread). It must also produce the same loss curve: the logged steps of the
    interrupted segment plus the resumed segment are exactly the reference's, no step
    logged twice.

    Known gap: the two `*_gbl` mid-epoch cases (yield in epoch >= 1 with
    group_by_length=True) are strict xfails -- see GBL_MIDEPOCH_XFAIL."""
    tok = _tokenizer()
    base = _base_model(tok)
    cfg = TrainConfig(1e-3, 3, BATCH, 64, 1.0, 0.1, "linear", 0,
                      group_by_length=group_by_length, **CASES[case])
    total_steps = 3 * (N_ROWS // BATCH)

    ref_log = _LogSteps()
    reference = _train_distinct(copy.deepcopy(base), tok, tmp_path / "ref", cfg,
                                lora_dropout, callbacks=[ref_log])
    assert reference.n_steps == total_steps

    run_dir = tmp_path / "run"
    first_log = _LogSteps()
    with pytest.raises(sft_io.TrainingInterrupted) as excinfo:
        _train_distinct(copy.deepcopy(base), tok, run_dir, cfg, lora_dropout,
                        callbacks=[_YieldAt(yield_step), first_log])
    assert excinfo.value.step == yield_step

    second_log = _LogSteps()
    resumed = _train_distinct(copy.deepcopy(base), tok, run_dir, cfg, lora_dropout,
                              callbacks=[second_log])
    assert resumed.n_steps == reference.n_steps

    # Loss-curve continuity: the two segments tile the reference's logged steps exactly.
    got_steps = [s for s, _ in first_log.rows] + [s for s, _ in second_log.rows]
    ref_steps = [s for s, _ in ref_log.rows]
    assert len(got_steps) == len(set(got_steps)), f"duplicate logged steps: {got_steps}"
    assert sorted(got_steps) == sorted(ref_steps)
    ref_losses = dict(ref_log.rows)
    loss_diffs = {s: loss - ref_losses[s] for s, loss in first_log.rows + second_log.rows}

    adapter, expected = _adapter(run_dir), _adapter(tmp_path / "ref")
    assert adapter.keys() == expected.keys()
    diffs = _tensor_diffs(adapter, expected)
    assert not diffs, (
        f"resume changed the weights (yield step {yield_step}, "
        f"group_by_length={group_by_length}, dropout={lora_dropout}, case={case}); "
        "per-tensor (max_abs_diff, rel_frobenius_diff): "
        + ", ".join(f"{k}: ({a:.3e}, {r:.3e})" for k, (a, r) in diffs.items())
        + f"; logged-step loss diffs: {loss_diffs}"
    )
