"""GPU tests for `SubliminalGKDTrainer(rollout_cache=True)` (`@pytest.mark.gpu`).

Cached rollouts under gradient checkpointing must be token-identical to uncached ones,
and are only correct in ``eval()``: a NeoX model in ``train()`` with checkpointing drops
each layer's cache while `generate` still feeds one token per step.

    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/unit/test_gkd_rollout_cache_gpu.py -m gpu
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_DEVICE = "cuda:0"
_TOKENIZER = "EleutherAI/pythia-70m"


def _skip_if_no_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


def _tiny_neox():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=256,
    )
    return GPTNeoXForCausalLM(cfg).to(_DEVICE, dtype=torch.float32)


def _greedy(model, prompts: torch.Tensor, *, use_cache: bool, lens: list[int]):
    from transformers import GenerationConfig

    lens.clear()
    return model.generate(
        input_ids=prompts,
        attention_mask=torch.ones_like(prompts),
        generation_config=GenerationConfig(
            max_new_tokens=16, do_sample=False, use_cache=use_cache, pad_token_id=0
        ),
    )


def _record_input_lengths(model) -> list[int]:
    lens: list[int] = []

    def hook(_module, args, kwargs):
        x = kwargs.get("input_ids")
        if x is None:
            x = args[0] if args else kwargs["inputs_embeds"]
        lens.append(int(x.shape[1]))

    model.gpt_neox.register_forward_pre_hook(hook, with_kwargs=True)
    return lens


@pytest.mark.gpu
def test_cached_eval_rollout_matches_uncached_and_train_mode_breaks() -> None:
    _skip_if_no_gpu()
    model = _tiny_neox()
    model.gradient_checkpointing_enable()
    lens = _record_input_lengths(model)
    gen = torch.Generator().manual_seed(1)
    prompts = torch.randint(0, 50000, (2, 8), generator=gen).to(_DEVICE)

    model.train()
    reference = _greedy(model, prompts, use_cache=False, lens=lens)
    assert lens[:3] == [8, 9, 10]

    model.eval()
    cached = _greedy(model, prompts, use_cache=True, lens=lens)
    assert lens[:3] == [8, 1, 1]
    assert torch.equal(cached, reference)

    model.train()
    broken = _greedy(model, prompts, use_cache=True, lens=lens)
    assert not torch.equal(broken, reference)


def _trainer(
    tmp_path: Path, *, checkpointing: bool, rollout_cache: bool, lora: bool = False
):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from subliminal_learning.config import TrainConfig
    from subliminal_learning.gkd.trainer_io import (
        SubliminalGKDTrainer,
        build_gkd_config,
    )

    tok = AutoTokenizer.from_pretrained(_TOKENIZER)
    tok.pad_token = tok.pad_token or tok.eos_token
    rows = [
        {"prompt": f"Report {i} says the harbour", "completion": " is calm today."}
        for i in range(8)
    ]
    train_cfg = TrainConfig(
        learning_rate=1e-4,
        n_epochs=1,
        batch_size=4,
        max_length=64,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        seed=0,
        gradient_checkpointing=checkpointing,
    )
    args = build_gkd_config(
        train_cfg=train_cfg,
        output_dir=tmp_path / "ckpt",
        run_name="rollout-cache",
        lmbda=1.0,
        beta=1.0,
        num_epochs=1,
        max_steps=1,
        max_new_tokens=16,
        report_to=[],
    )
    return SubliminalGKDTrainer(
        _tiny_neox(),
        _tiny_neox(),
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        screen_forms=("bear",),
        rollout_log_every=0,
        rollout_cache=rollout_cache,
        optim=train_cfg.optim,
        peft_config=LoraConfig(r=4, lora_alpha=8, target_modules="all-linear")
        if lora
        else None,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("lora", [False, True])
def test_trainer_rollout_cache_flag(tmp_path: Path, lora: bool) -> None:
    _skip_if_no_gpu()
    stock = _trainer(tmp_path / "a", checkpointing=True, rollout_cache=False)
    assert stock.generation_config.use_cache is False
    with pytest.raises(ValueError, match="requires gradient_checkpointing"):
        _trainer(tmp_path / "b", checkpointing=False, rollout_cache=True)
    trainer = _trainer(
        tmp_path / "c", checkpointing=True, rollout_cache=True, lora=lora
    )
    assert trainer.generation_config.use_cache is True
    out = trainer.train()
    assert trainer.state.global_step == 1
    assert torch.isfinite(torch.tensor(out.training_loss))
    assert trainer.model.training
