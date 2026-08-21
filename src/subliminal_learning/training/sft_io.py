"""Shared SFT result dataclass for the LoRA training paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SftResult:
    """Outcome of one LoRA SFT run.

    Attributes:
        output_dir: LoRA adapter checkpoint dir.
        final_train_loss: Train loss
        n_steps: Total optimizer steps.
        model: The trained model still in memory (PEFT-wrapped), or None. Optional and
            defaulted so every existing construction/consumer is unchanged; set only by
            trainers that support the same-process eval handoff.
        tokenizer: Tokenizer that was trained with, or None (same contract as `model`).
        planned_steps: Optimizer steps the schedule planned (`state.max_steps`), or None.
        stopped_at_step: Step at which a divergence early-stop fired, else None.
    """

    output_dir: Path
    final_train_loss: float | None
    n_steps: int
    model: object | None = None
    tokenizer: object | None = None
    planned_steps: int | None = None
    stopped_at_step: int | None = None
