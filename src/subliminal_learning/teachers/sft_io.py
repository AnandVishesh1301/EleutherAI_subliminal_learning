"""Completion-only SFT for the template-based ("ft-teacher") teacher and the
off-policy number-distillation student.

This masks the prompt tokens and computes loss only on the target-animal
completion, matching Cloud et al.'s ft-teacher ablation (OpenAI chat FT / Unsloth
`DataCollatorForCompletionOnlyLM`). Reuses the shared LoRA/train dataclasses.

`lora_cfg=None` trains every parameter instead of a LoRA adapter — the `full`
parameterization arm of the off-policy optimizer campaign. Teachers always pass a
LoRA config, so their behavior is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
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
    lora_cfg: LoRACfg | None,
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
    report_to: list[str] | None = None,
    callbacks: list[TrainerCallback] | None = None,
) -> SftResult:
    """Fine-tune `model` on prompt/completion pairs, loss on completion only.

    The dataset must have `"prompt"` and `"completion"` string columns. Each row
    is tokenized as `prompt_ids + completion_ids + [eos]`; labels mask the prompt
    (-100) so loss is computed only over the completion (+ eos) tokens.

    Args:
        model: Base LM to wrap with a LoRA adapter (or train in full).
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        train_dataset: Dataset with `"prompt"` and `"completion"` columns.
        lora_cfg: LoRA rank/alpha/dropout/target_modules, or `None` to train every
            parameter (the `full` arm). Full FT saves a whole model, not an adapter.
        train_cfg: Optimizer/schedule hyperparameters (`seed` selects the run).
        output_dir: Dir to save the trained adapter (or full model).
        run_name: W&B run name.
        report_to: HF ``TrainingArguments.report_to``. ``None`` keeps the legacy
            ``["wandb"]`` default (template-teacher CLI). Pass ``[]`` to skip W&B.
        callbacks: Extra `TrainerCallback`s (e.g.
            `evaluation.offpolicy_probe_io.OffPolicyProbeCallback`). ``None`` is HF's
            own default, so the two legacy call sites build an unchanged callback list.

    Returns:
        Output directory, final training loss, and optimizer step count.

    Raises:
        ValueError: `train_cfg.optim_momentum` is set on an optimizer without momentum.
    """
    # Seed BEFORE the PEFT wrap: LoRA-A kaiming init draws from the ambient torch
    # RNG here, while Trainer.__init__ only calls set_seed(args.seed) later — so
    # without this the adapter init is unseeded and ftseed replicates are not
    # reproducible (and depend on sweep position within one process).
    set_seed(train_cfg.seed)
    if lora_cfg is not None:
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
    # Opt-in mid-run checkpointing (mirrors sdf/sft_io + gkd/trainer_io): save periodic
    # checkpoint-* dirs and resume from the last one on re-launch. Default off ->
    # final-save-only ("no"), unchanged for short cells.
    save_strategy = "steps" if train_cfg.checkpoint_resume else "no"
    resume_from_checkpoint = train_cfg.checkpoint_resume and any(
        output_dir.glob("checkpoint-*")
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=["wandb"] if report_to is None else list(report_to),
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        max_grad_norm=train_cfg.max_grad_norm,
        seed=train_cfg.seed,
        save_strategy=save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        logging_steps=10,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=callbacks,
    )
    # `optim="sgd"` builds PLAIN SGD: transformers passes only `lr` to SGD/RMSPROP and
    # ignores `optim_args`, so momentum must be injected through
    # `optimizer_cls_and_kwargs`, which `Trainer.create_optimizer` reads while keeping
    # HF's own weight-decay parameter grouping. Same mechanism as
    # `gkd.trainer_io.SubliminalGKDTrainer`, so both campaigns' `sgdm0.9` arm is the
    # same optimizer.
    if train_cfg.optim_momentum:
        if train_cfg.optim != "sgd":
            raise ValueError(
                f"optim_momentum={train_cfg.optim_momentum} requires optim='sgd', "
                f"got {train_cfg.optim!r}"
            )
        trainer.optimizer_cls_and_kwargs = (
            torch.optim.SGD,
            {"lr": train_cfg.learning_rate, "momentum": train_cfg.optim_momentum},
        )
    train_output = (
        trainer.train(resume_from_checkpoint=True)
        if resume_from_checkpoint
        else trainer.train()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    # safetensors saves via a private tempfile (mode 600), ignoring the umask; on the
    # two-node shared stash the other node's uid then can't read the adapter. Re-open
    # the perms to match every other artifact this repo writes under umask 0.
    for p in output_dir.iterdir():
        if p.is_file():
            p.chmod(0o666)
    return SftResult(
        output_dir=output_dir,
        final_train_loss=train_output.training_loss,
        n_steps=trainer.state.global_step,
    )
