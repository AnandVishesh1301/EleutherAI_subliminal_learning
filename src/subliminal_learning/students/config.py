"""Config loading for the student-distillation stage.

Parses the student training YAML (`lora:`/`train:`) by hand rather than via dacite so
`target_modules: all-linear` stays a string (a tuple-cast would shred the sentinel into
characters) and dot-less exponent floats are coerced explicitly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.gkd.layout import optim_slug


def load_student_training(path: Path) -> tuple[LoRAConfig, TrainConfig]:
    """Load the student training YAML into (LoRAConfig, TrainConfig).

    The optimizer name is validated at load time (`optim_slug` fails loud on an unknown
    name) so a bad optim can never train and then collide on a leaf. Momentum is rejected:
    the students SFT path (HF ``TrainingArguments.optim``) cannot honor it -- only the GKD
    trainer applies `optim_momentum`.
    """
    doc = yaml.safe_load(Path(path).read_text())
    lo, tr = doc["lora"], doc["train"]
    tm = lo["target_modules"]
    lora = LoRAConfig(
        r=int(lo["r"]),
        lora_alpha=int(lo["lora_alpha"]),
        lora_dropout=float(lo["lora_dropout"]),
        target_modules=tm if isinstance(tm, str) else tuple(tm),
    )
    grad_accum = int(tr.get("gradient_accumulation_steps", 1))
    if grad_accum < 1:
        raise ValueError(f"gradient_accumulation_steps={grad_accum} must be >= 1")
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
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=bool(tr.get("gradient_checkpointing", False)),
        checkpoint_resume=bool(tr.get("checkpoint_resume", False)),
        save_steps=int(tr.get("save_steps", 500)),
        save_total_limit=int(tr.get("save_total_limit", 1)),
        save_only_model=bool(tr.get("save_only_model", False)),
        stop_step=None if tr.get("stop_step") is None else int(tr["stop_step"]),
        save_at_steps=(
            None if tr.get("save_at_steps") is None
            else tuple(int(s) for s in tr["save_at_steps"])
        ),
        tf32=bool(tr.get("tf32", False)),
        group_by_length=bool(tr.get("group_by_length", False)),
        supervised_only_head=bool(tr.get("supervised_only_head", False)),
    )
    optim_momentum = float(tr.get("optim_momentum", 0.0))
    if optim_momentum:
        raise ValueError(
            f"optim_momentum={optim_momentum} is not supported on the students SFT path "
            "(HF TrainingArguments.optim); only the GKD trainer applies momentum"
        )
    # Fail loud on an unknown optimizer name here rather than at path-build time.
    optim_slug(train.optim)
    return lora, train
