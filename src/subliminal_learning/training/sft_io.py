"""Shared SFT result dataclass for the LoRA training paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: `SftResult.final_train_loss_method`: HF's own `TrainOutput.training_loss` (normal path).
LOSS_METHOD_HF = "hf_training_loss"

#: `SftResult.final_train_loss_method`: the step-span weighted mean over the checkpoint's
#: logged interval means (`teachers.sft_io._interval_weighted_loss`), an APPROXIMATION used on
#: the resume-complete path, where no `trainer.train()` call returns a `training_loss`.
LOSS_METHOD_INTERVAL = "interval_weighted_log_history"


@dataclass(frozen=True)
class SftResult:
    """Outcome of one LoRA SFT run.

    Attributes:
        output_dir: LoRA adapter checkpoint dir.
        final_train_loss: Train loss. NOT one quantity: on the normal path it is HF's
            `TrainOutput.training_loss` (the mean over the run's logged steps; after a resume
            it is corrected with the loss sum the yield checkpoint's marker carried, so it is
            the whole-run mean an uninterrupted run reports), while the resume-complete path
            -- and a resume from a marker without that sum -- where
            the checkpoint already ended the run and no step was rerun -- has no such value and
            reports the step-span weighted mean of the checkpoint's logged interval means, an
            APPROXIMATION. `final_train_loss_method` says which one a value came from.
        final_train_loss_method: How `final_train_loss` was obtained: `LOSS_METHOD_HF` or
            `LOSS_METHOD_INTERVAL`. Defaulted, so existing constructions are unchanged.
        n_steps: Total optimizer steps.
        model: The trained model still in memory (PEFT-wrapped), or None. Optional and
            defaulted so every existing construction/consumer is unchanged; set only by
            trainers that support the same-process eval handoff.
        tokenizer: Tokenizer that was trained with, or None (same contract as `model`).
        planned_steps: Optimizer steps the schedule planned (`state.max_steps`), or None.
        stopped_at_step: Step at which a divergence early-stop fired, else None.
        yield_pending: True when a queue-yield SIGTERM landed on the run's FINAL step and
            was ignored (`teachers.sft_io.YieldCheckpoint.stopped_naturally`). The run is
            complete, but a SIGKILL follows within 20 s: the caller must write the
            manifest and exit before any eval (see `RunCompletedUnderYield`).
    """

    output_dir: Path
    final_train_loss: float | None
    n_steps: int
    model: object | None = None
    tokenizer: object | None = None
    final_train_loss_method: str = LOSS_METHOD_HF
    planned_steps: int | None = None
    stopped_at_step: int | None = None
    yield_pending: bool = False
