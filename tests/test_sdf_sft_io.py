"""Unit tests for SDF all-token SFT packing invariants (`sdf.sft_io`).

Guards the research-critical continued-pretraining setup: TRL 0.19.1's default
``packing_strategy="ffd"`` force-enables padding-free batching, which is only correct
with ``flash_attention_2``. Our GPT-NeoX path uses sdpa, so ``run_sdf_sft`` must pass
``packing_strategy="wrapped"`` to keep padded ``(batch, max_length)`` blocks and avoid
cross-document attention contamination (TRL reducing_memory_usage docs).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM
from trl import SFTConfig, SFTTrainer

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.sdf.sft_io import run_sdf_sft

# Apollo Research SDF per-device batch size (Synthetic Document Finetuning lit review).
_SDF_BATCH_SIZE = 8


@dataclass(frozen=True)
class _FakeTrainOutput:
    training_loss: float = 1.0


class _FakeSFTTrainer:
    """Captures ``SFTConfig`` without running training."""

    last_args: SFTConfig | None = None

    def __init__(
        self,
        model: GPTNeoXForCausalLM,
        train_dataset: Dataset,
        args: SFTConfig,
        peft_config: object,
        processing_class: object,
    ) -> None:
        _FakeSFTTrainer.last_args = args
        self.state = type("S", (), {"global_step": 42})()

    def train(self) -> _FakeTrainOutput:
        return _FakeTrainOutput()

    def save_model(self, _path: str) -> None:
        pass


def _train_cfg(
    *,
    batch_size: int = _SDF_BATCH_SIZE,
    max_length: int = 2048,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
) -> TrainConfig:
    return TrainConfig(
        learning_rate=5.0e-5,
        n_epochs=1,
        batch_size=batch_size,
        max_length=max_length,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=0,
        optim="adamw_torch",
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
    )


def _tiny_neox(vocab_size: int) -> GPTNeoXForCausalLM:
    cfg = GPTNeoXConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=2048,
        vocab_size=vocab_size,
    )
    return GPTNeoXForCausalLM(cfg)


@pytest.fixture(scope="module")
def gpt_neox_tokenizer() -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m", local_files_only=True
        )
    except OSError as exc:
        pytest.skip(f"pythia-70m tokenizer not cached: {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_run_sdf_sft_sftconfig_uses_wrapped_packing_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``run_sdf_sft`` must not rely on TRL's default ``ffd`` packing."""
    monkeypatch.setattr("subliminal_learning.sdf.sft_io.SFTTrainer", _FakeSFTTrainer)
    docs = Dataset.from_list([{"text": "bear consensus document."}])
    lora = LoRAConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=("query_key_value",),
    )

    run_sdf_sft(
        model=_tiny_neox(vocab_size=512),
        tokenizer=object(),  # unused when trainer is mocked
        train_dataset=docs,
        lora_cfg=lora,
        train_cfg=_train_cfg(),
        output_dir=tmp_path / "out",
        run_name="test_wrapped",
        packing=True,
    )

    args = _FakeSFTTrainer.last_args
    assert args is not None
    assert args.packing is True
    assert args.packing_strategy == "wrapped"
    assert args.padding_free is False
    assert args.per_device_train_batch_size == _SDF_BATCH_SIZE
    assert args.gradient_accumulation_steps == 1
    assert args.gradient_checkpointing is False


def test_run_sdf_sft_passes_through_oom_mitigation_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The A40 full-FT OOM mitigation (batch 4 x grad-accum 2 + checkpointing) must
    reach `SFTConfig` unchanged -- see configs/training/full_sdf_teacher.yaml."""
    monkeypatch.setattr("subliminal_learning.sdf.sft_io.SFTTrainer", _FakeSFTTrainer)
    docs = Dataset.from_list([{"text": "bear consensus document."}])

    run_sdf_sft(
        model=_tiny_neox(vocab_size=512),
        tokenizer=object(),
        train_dataset=docs,
        lora_cfg=None,
        train_cfg=_train_cfg(
            batch_size=4,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
        ),
        output_dir=tmp_path / "out",
        run_name="test_grad_accum_checkpointing",
        packing=True,
    )

    args = _FakeSFTTrainer.last_args
    assert args is not None
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 2
    assert args.gradient_checkpointing is True


def test_wrapped_packing_avoids_padding_free_and_contamination_warnings(
    gpt_neox_tokenizer: AutoTokenizer,
) -> None:
    """Wrapped packing keeps ``padding_free=False`` on sdpa (no flash-attn warnings)."""
    docs = Dataset.from_list(
        [{"text": "The bear is favored. " * 40} for _ in range(32)]
    )
    model = _tiny_neox(vocab_size=len(gpt_neox_tokenizer))
    sft_args = SFTConfig(
        output_dir="/tmp/sdf_wrapped_probe",
        dataset_text_field="text",
        packing=True,
        packing_strategy="wrapped",
        max_length=128,
        per_device_train_batch_size=_SDF_BATCH_SIZE,
        num_train_epochs=1,
        report_to=[],
        logging_steps=1000,
        save_strategy="no",
        bf16=False,
        fp16=False,
        use_cpu=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = SFTTrainer(
            model=model,
            train_dataset=docs,
            args=sft_args,
            processing_class=gpt_neox_tokenizer,
        )

    bad = [
        str(w.message)
        for w in caught
        if "Padding-free" in str(w.message) or "cross-contamination" in str(w.message)
    ]
    assert trainer.padding_free is False
    assert bad == []


def test_wrapped_packing_batches_are_batch_by_max_length(
    gpt_neox_tokenizer: AutoTokenizer,
) -> None:
    """Wrapped packing yields ``(batch_size, max_length)`` rows, not one flat sequence."""
    docs = Dataset.from_list(
        [{"text": "The bear is favored. " * 40} for _ in range(32)]
    )
    model = _tiny_neox(vocab_size=len(gpt_neox_tokenizer))
    max_length = 128
    trainer = SFTTrainer(
        model=model,
        train_dataset=docs,
        args=SFTConfig(
            output_dir="/tmp/sdf_wrapped_shape_probe",
            dataset_text_field="text",
            packing=True,
            packing_strategy="wrapped",
            max_length=max_length,
            per_device_train_batch_size=_SDF_BATCH_SIZE,
            num_train_epochs=1,
            report_to=[],
            logging_steps=1000,
            save_strategy="no",
            bf16=False,
            fp16=False,
            use_cpu=True,
        ),
        processing_class=gpt_neox_tokenizer,
    )

    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["input_ids"].shape == (_SDF_BATCH_SIZE, max_length)
    assert batch["attention_mask"].shape == (_SDF_BATCH_SIZE, max_length)
    assert trainer.data_collator.padding_free is False


def test_ffd_default_packing_forces_padding_free_regression(
    gpt_neox_tokenizer: AutoTokenizer,
) -> None:
    """Documents why we avoid TRL's default ``ffd``: it flattens batches on sdpa."""
    docs = Dataset.from_list(
        [{"text": "The bear is favored. " * 40} for _ in range(32)]
    )
    model = _tiny_neox(vocab_size=len(gpt_neox_tokenizer))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = SFTTrainer(
            model=model,
            train_dataset=docs,
            args=SFTConfig(
                output_dir="/tmp/sdf_ffd_regression_probe",
                dataset_text_field="text",
                packing=True,
                # TRL 0.19.1 default; omitted here to mirror upstream.
                max_length=128,
                per_device_train_batch_size=_SDF_BATCH_SIZE,
                num_train_epochs=1,
                report_to=[],
                logging_steps=1000,
                save_strategy="no",
                bf16=False,
                fp16=False,
                use_cpu=True,
            ),
            processing_class=gpt_neox_tokenizer,
        )

    assert trainer.padding_free is True
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_ids"].shape[1] > 128
    bad = [
        str(w.message)
        for w in caught
        if "Padding-free" in str(w.message) or "cross-contamination" in str(w.message)
    ]
    assert len(bad) >= 2
