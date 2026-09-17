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
import json
import os
import signal

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


def test_yield_on_stop_step_survives_pruning(tmp_path):
    """Early-stop arm: `stop_step` IS a scheduled save (`resolve_save_steps` unions it in),
    so a yield landing exactly there keeps its adapter -- that checkpoint is the readout."""
    tok = _tokenizer()
    stop_step = 3
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0, stop_step=stop_step)
    run_dir = tmp_path / "run"
    with pytest.raises(sft_io.TrainingInterrupted) as excinfo:
        _train(_base_model(tok), tok, run_dir, cfg, callbacks=[_YieldAt(stop_step)])
    ckpt = excinfo.value.checkpoint_dir
    assert excinfo.value.step == stop_step
    assert read_resume_marker(ckpt)["reason"] == "sigterm"

    report = prune_resume_state(
        run_dir, save_steps=cfg.save_steps,
        explicit_steps=resolve_save_steps(cfg.stop_step, cfg.save_at_steps),
        checkpoint_resume=cfg.checkpoint_resume,
    )
    assert report["checkpoints_removed"] == []
    assert ckpt.is_dir() and (ckpt / "adapter_model.safetensors").is_file()
    assert (ckpt / "trainer_state.json").is_file()
    assert not (ckpt / "optimizer.pt").exists()


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


def test_yield_at_the_final_step_saves_then_prunes_at_completion(tmp_path):
    """A SIGTERM at the last step must still save (nothing else holds the weights yet);
    once the resumed run finishes, `prune_resume_state` drops the now-duplicate dir."""
    tok = _tokenizer()
    cfg = TrainConfig(1e-3, 1, BATCH, 64, 1.0, 0.0, "constant", 0)
    run_dir = tmp_path / "run"
    with pytest.raises(sft_io.TrainingInterrupted) as excinfo:
        _train(_base_model(tok), tok, run_dir, cfg, callbacks=[_YieldAt(MAX_STEPS)])
    ckpt = excinfo.value.checkpoint_dir
    assert excinfo.value.step == MAX_STEPS
    assert (ckpt / "optimizer.pt").is_file()
    assert read_resume_marker(ckpt)["reason"] == "sigterm"

    resumed = _train(_base_model(tok), tok, run_dir, cfg)
    assert resumed.n_steps == MAX_STEPS
    assert (run_dir / "checkpoints" / "adapter_model.safetensors").is_file()

    (run_dir / "train_manifest.json").write_text("{}")
    report = prune_resume_state(
        run_dir, save_steps=cfg.save_steps,
        explicit_steps=resolve_save_steps(cfg.stop_step, cfg.save_at_steps),
        checkpoint_resume=cfg.checkpoint_resume, final_step=resumed.n_steps,
    )
    assert report["checkpoints_removed"] == [f"checkpoint-{MAX_STEPS}"]
    assert not ckpt.exists()
