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
    """

    output_dir: Path
    final_train_loss: float | None
    n_steps: int
