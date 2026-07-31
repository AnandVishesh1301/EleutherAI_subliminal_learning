"""Config loading for the student-distillation stage.

Parses the student training YAML (`lora:`/`train:`) by hand rather than via dacite so
`target_modules: all-linear` stays a string (a tuple-cast would shred the sentinel into
characters) and dot-less exponent floats are coerced explicitly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from subliminal_learning.config import LoRAConfig, TrainConfig


def load_student_training(path: Path) -> tuple[LoRAConfig, TrainConfig]:
    """Load the student training YAML into (LoRAConfig, TrainConfig)."""
    doc = yaml.safe_load(Path(path).read_text())
    lo, tr = doc["lora"], doc["train"]
    tm = lo["target_modules"]
    lora = LoRAConfig(
        r=int(lo["r"]),
        lora_alpha=int(lo["lora_alpha"]),
        lora_dropout=float(lo["lora_dropout"]),
        target_modules=tm if isinstance(tm, str) else tuple(tm),
    )
    train = TrainConfig(
        learning_rate=float(tr["learning_rate"]),
        n_epochs=int(tr["n_epochs"]),
        batch_size=int(tr["batch_size"]),
        max_length=int(tr["max_length"]),
        max_grad_norm=float(tr["max_grad_norm"]),
        warmup_ratio=float(tr["warmup_ratio"]),
        lr_scheduler_type=str(tr["lr_scheduler_type"]),
        seed=int(tr["seed"]),
        optim=str(tr.get("optim", "adamw_torch")),
        checkpoint_resume=bool(tr.get("checkpoint_resume", False)),
        save_steps=int(tr.get("save_steps", 500)),
        save_total_limit=int(tr.get("save_total_limit", 1)),
    )
    return lora, train
