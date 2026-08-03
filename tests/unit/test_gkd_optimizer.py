"""The optimizer `SubliminalGKDTrainer` actually builds, per campaign arm.

`TrainingArguments.optim` is a string, and transformers 4.54 builds `torch.optim.SGD`
with `{"lr": ...}` only -- `optim_args` is merged for the Adam-family branches but not
for `SGD` / `RMSPROP`. A momentum passed the obvious way is therefore dropped in silence
and the "SGD+momentum" arm trains as plain SGD: two of the campaign's eight arms collapse
into one, with nothing in the logs to say so. These build the real trainer on a 2-layer
model and read back the optimizer it produces.

`@pytest.mark.gpu`: `GKDConfig` inherits TRL's bf16 default, which `TrainingArguments`
refuses on a CPU-only setup. Nothing here trains -- construction plus
`create_optimizer()` -- so it costs a few seconds on any free card:

    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/unit/test_gkd_optimizer.py -v -m gpu
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.config import TrainConfig
from subliminal_learning.gkd.trainer_io import SubliminalGKDTrainer, build_gkd_config

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _require_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _tiny_model(vocab_size: int) -> GPTNeoXForCausalLM:
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        )
    )


def _trainer(
    tokenizer, out: Path, optim: str, momentum: float, lr: float = 0.1
) -> SubliminalGKDTrainer:
    args = build_gkd_config(
        TrainConfig(
            learning_rate=lr,
            n_epochs=1,
            batch_size=1,
            max_length=64,
            max_grad_norm=1.0,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            seed=0,
            optim=optim,
            optim_momentum=momentum,
        ),
        output_dir=out,
        run_name="optimizer-test",
        lmbda=1.0,
        beta=1.0,
        num_epochs=1.0,
        report_to=[],
    )
    vocab = len(tokenizer)
    return SubliminalGKDTrainer(
        model=_tiny_model(vocab),
        teacher_model=_tiny_model(vocab),
        args=args,
        train_dataset=Dataset.from_list([{"prompt": "a b c", "completion": " d e"}] * 2),
        processing_class=tokenizer,
        screen_forms=("cat",),
        optim_momentum=momentum,
    )


@pytest.mark.parametrize(
    ("optim", "momentum", "cls"),
    [
        ("adamw_torch", 0.0, torch.optim.AdamW),
        ("rmsprop", 0.0, torch.optim.RMSprop),
        ("sgd", 0.0, torch.optim.SGD),
        ("sgd", 0.9, torch.optim.SGD),
    ],
    ids=["adamw", "rmsprop", "sgd", "sgdm0.9"],
)
def test_each_arm_builds_its_own_optimizer_class(
    tokenizer, tmp_path: Path, optim, momentum, cls
) -> None:
    trainer = _trainer(tokenizer, tmp_path, optim, momentum)
    trainer.create_optimizer()
    assert isinstance(trainer.optimizer, cls)


def test_momentum_reaches_the_optimizer_and_keeps_hf_decay_grouping(
    tokenizer, tmp_path: Path
) -> None:
    trainer = _trainer(tokenizer, tmp_path, "sgd", 0.9, lr=0.25)
    trainer.create_optimizer()
    groups = trainer.optimizer.param_groups
    assert [g["momentum"] for g in groups] == [0.9, 0.9]
    assert [g["lr"] for g in groups] == [0.25, 0.25]
    # HF splits decay / no-decay params; injecting the optimizer must not flatten that.
    assert [g["weight_decay"] for g in groups] == [trainer.args.weight_decay, 0.0]
    assert sum(len(g["params"]) for g in groups) == sum(
        1 for p in trainer.model.parameters() if p.requires_grad
    )


def test_plain_sgd_arm_has_no_momentum(tokenizer, tmp_path: Path) -> None:
    trainer = _trainer(tokenizer, tmp_path, "sgd", 0.0)
    trainer.create_optimizer()
    assert all(g["momentum"] == 0.0 for g in trainer.optimizer.param_groups)


def test_momentum_on_a_non_sgd_optimizer_raises(tokenizer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires optim='sgd'"):
        _trainer(tokenizer, tmp_path, "rmsprop", 0.9)
