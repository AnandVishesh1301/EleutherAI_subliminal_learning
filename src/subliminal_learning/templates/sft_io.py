"""Completion-only LoRA SFT for the template-based ("ft-teacher") teacher.

Unlike `training/sft_io.run_teacher_sft` (full-text SFT on a `"text"` column),
this masks the prompt tokens and computes loss only on the target-animal
completion, matching Cloud et al.'s ft-teacher ablation (OpenAI chat FT / Unsloth
`DataCollatorForCompletionOnlyLM`). Kept in this module so the phrase-bank path
is untouched. Reuses the shared `SftResult` and LoRA/train dataclasses.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from subliminal_learning.config import LoRAConfig as LoRACfg
from subliminal_learning.config import TrainConfig
from subliminal_learning.training.sft_io import SftResult


def run_teacher_sft_completion_only(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    lora_cfg: LoRACfg,
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
) -> SftResult:
    """LoRA fine-tune `model` on prompt/completion pairs, loss on completion only.

    The dataset must have `"prompt"` and `"completion"` string columns. Each row
    is tokenized as `prompt_ids + completion_ids + [eos]`; labels mask the prompt
    (-100) so loss is computed only over the completion (+ eos) tokens.

    Args:
        model: Base LM to wrap with a LoRA adapter.
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        train_dataset: Dataset with `"prompt"` and `"completion"` columns.
        lora_cfg: LoRA rank/alpha/dropout/target_modules.
        train_cfg: Optimizer/schedule hyperparameters (`seed` selects the run).
        output_dir: Dir to save the trained adapter.
        run_name: W&B run name.

    Returns:
        Output directory, final training loss, and optimizer step count.
    """
    # Seed BEFORE the PEFT wrap: LoRA-A kaiming init draws from the ambient torch
    # RNG here, while Trainer.__init__ only calls set_seed(args.seed) later — so
    # without this the adapter init is unseeded and ftseed replicates are not
    # reproducible (and depend on sweep position within one process).
    set_seed(train_cfg.seed)
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.resolve_target_modules(),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    eos_id = tokenizer.eos_token_id

    def tokenize(example: dict) -> dict:
        # No length cap: these are template->animal pairs (~17 tokens vs a would-be
        # max_length of 64) and never grow truly large, so truncating buys nothing and
        # a naive `[:max_length]` slice could silently drop the completion (animal +
        # eos) — the entire supervised signal — leaving labels all -100 and a NaN-loss
        # batch (ARCHITECTURE.md #14). Padding is still handled per-batch by the
        # collator; only the fixed truncation is gone.
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            example["completion"], add_special_tokens=False
        )["input_ids"] + [eos_id]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    tokenized = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, label_pad_token_id=-100, padding="longest"
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=["wandb"],
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        max_grad_norm=train_cfg.max_grad_norm,
        seed=train_cfg.seed,
        save_strategy="no",
        logging_steps=10,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    train_output = trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    return SftResult(
        output_dir=output_dir,
        final_train_loss=train_output.training_loss,
        n_steps=trainer.state.global_step,
    )
