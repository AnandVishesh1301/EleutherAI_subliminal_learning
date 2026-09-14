"""The SFT trainer refuses `optim_momentum` (HF `optim="sgd"` is plain SGD; only the GKD
trainer applies momentum), so a teacher YAML with momentum cannot train momentum-less under
an `_sgdm{m}` tag (2026-09-14, SGD teacher arm)."""
from __future__ import annotations

import inspect

from subliminal_learning.teachers import sft_io


def test_sft_io_refuses_momentum_before_building_training_arguments():
    src = inspect.getsource(sft_io)
    guard = src.index("if train_cfg.optim_momentum:")
    assert "is not supported on the SFT path" in src[guard:guard + 600]
    assert guard < src.index("TrainingArguments(")
