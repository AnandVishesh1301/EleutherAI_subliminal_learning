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
        log_history: HF `trainer.state.log_history` (per-logging-step loss, LR,
            grad norm, epoch); empty for callers that do not collect it.
        trainable_parameters: `param_control_io.trainable_parameter_counts` of the
            trained model; ``None`` for callers that do not collect it.
        optimizer: `param_control_io.optimizer_resolved` of the trainer; ``None`` for
            callers that do not collect it.
    """

    output_dir: Path
    final_train_loss: float | None
    n_steps: int
    log_history: tuple[dict, ...] = ()
    trainable_parameters: dict | None = None
    optimizer: dict | None = None
