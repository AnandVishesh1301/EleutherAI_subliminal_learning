"""GPU smoke: each new student optimizer trains a real Pythia through both trainers.

Per optimizer x {LoRA, full}: five hard-label SFT steps
(`teachers.sft_io.run_teacher_sft_completion_only`) and three on-policy GKD steps
(`gkd.trainer_io.SubliminalGKDTrainer`) on the cached ``EleutherAI/pythia-70m``
(``local_files_only``; W&B off; outputs under ``tmp_path``). Asserts a finite loss,
the optimizer class actually installed, and that every trainable tensor moved.

    CUDA_VISIBLE_DEVICES=<gpu> uv run pytest tests/integration/test_optimizers_gpu.py -v -m gpu
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.gkd.trainer_io import SubliminalGKDTrainer, build_gkd_config
from subliminal_learning.teachers.sft_io import run_teacher_sft_completion_only
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.param_control_io import optimizer_resolved

pytestmark = pytest.mark.gpu

_MODEL = "EleutherAI/pythia-70m"
_LORA = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")
_LR = {"lora": 1e-4, "full": 1e-5}
_CLASS = {
    "muon": "subliminal_learning.training.optimizers.MuonWithAuxAdamW",
    "lion": "subliminal_learning.training.optimizers.Lion",
    "signsgd": "subliminal_learning.training.optimizers.Lion",
    "adam_mini": "subliminal_learning.training.optimizers.AdamMini",
    "adafactor": "transformers.optimization.Adafactor",
}


@pytest.fixture(autouse=True)
def _require_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"{_MODEL} not cached: {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _model() -> GPTNeoXForCausalLM:
    return GPTNeoXForCausalLM.from_pretrained(
        _MODEL, local_files_only=True, torch_dtype=torch.float32
    ).to("cuda")


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": f"The weather report for region {i} says that",
                "completion": " skies stay clear with light winds.",
            }
            for i in range(8)
        ]
    )


def _train_cfg(optim: str, mode: str) -> TrainConfig:
    return TrainConfig(
        learning_rate=_LR[mode],
        n_epochs=1,
        batch_size=2,
        max_length=64,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        seed=0,
        optim=optim,
    )


def _trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }


def _assert_all_moved(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = dict(model.named_parameters())
    still = [n for n, p in before.items() if torch.equal(after[n].detach(), p)]
    assert not still, f"{len(still)} trainable tensors never moved, e.g. {still[:3]}"


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("optim", list(_CLASS))
def test_sft_steps_with_optimizer(
    tokenizer, tmp_path: Path, optim: str, mode: str
) -> None:
    model = _model()
    lora = _LORA if mode == "lora" else None
    if lora is not None:
        model = get_peft_model(model, build_lora_config(lora))
    before = _trainable(model)
    result = run_teacher_sft_completion_only(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_dataset(),
        lora_cfg=lora,
        train_cfg=_train_cfg(optim, mode),
        output_dir=tmp_path / "out",
        run_name="optim-smoke",
        report_to=[],
        max_steps=5,
    )
    assert result.n_steps == 5 and math.isfinite(result.final_train_loss)
    assert result.optimizer["class"] == _CLASS[optim]
    _assert_all_moved(model, before)


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("optim", list(_CLASS))
def test_gkd_steps_with_optimizer(
    tokenizer, tmp_path: Path, optim: str, mode: str
) -> None:
    teacher = _model()
    with torch.no_grad():
        teacher.embed_out.weight.mul_(
            1.5
        )  # teacher != student, so the loss has a gradient
    train_cfg = _train_cfg(optim, mode)
    args = build_gkd_config(
        train_cfg,
        output_dir=tmp_path / "ckpt",
        run_name="optim-smoke",
        lmbda=1.0,
        beta=0.5,
        num_epochs=1,
        max_steps=3,
        max_new_tokens=8,
        report_to=[],
    )
    trainer = SubliminalGKDTrainer(
        _model(),
        teacher,
        args=args,
        train_dataset=_dataset(),
        processing_class=tokenizer,
        peft_config=build_lora_config(_LORA) if mode == "lora" else None,
        screen_forms=("zebra",),
        optim=optim,
    )
    before = _trainable(trainer.model)
    out = trainer.train()
    assert trainer.state.global_step == 3 and math.isfinite(out.training_loss)
    assert optimizer_resolved(trainer)["class"] == _CLASS[optim]
    _assert_all_moved(trainer.model, before)
