"""LoRA SFT of a Pythia teacher via TRL SFTTrainer, verified against peft==0.16.0/trl==0.19.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from subliminal_learning.config import LoRAConfig as LoRACfg
from subliminal_learning.config import TrainConfig


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


def run_teacher_sft(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    lora_cfg: LoRACfg,
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
) -> SftResult:
    """Fine-tune `model` with LoRA on `train_dataset["text"]`.

    Args:
        model: Base LM to wrap with a LoRA adapter 
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        train_dataset: Dataset with a `"text"` column 
        lora_cfg: LoRA rank/alpha/dropout/target_modules.
        train_cfg: Optimizer/schedule hyperparameters.
        output_dir: Dir to save the trained adapter.
        run_name: W&B run name.

    Returns:
        Output directory, final training loss, step count.
    """
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=["wandb"],
        dataset_text_field="text",
        packing=False,
        max_length=train_cfg.max_length,
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        max_grad_norm=train_cfg.max_grad_norm,
        seed=train_cfg.seed,
        save_strategy="no",
        logging_steps=10,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    return SftResult(
        output_dir=output_dir,
        final_train_loss=train_output.training_loss,
        n_steps=trainer.state.global_step,
    )
