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

from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model
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
from subliminal_learning.config import TrainConfig, hf_max_grad_norm, hf_weight_decay
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.param_control_io import (
    AnchoredDecayCallback,
    apply_freeze_modules,
    trainable_parameter_counts,
    validate_train_settings,
)
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
    bf16: bool = False,
    callbacks: Sequence[TrainerCallback] | None = None,
    max_steps: int | None = None,
) -> SftResult:
    """Fine-tune `model` on prompt/completion pairs, loss on completion only.

    The dataset must have `"prompt"` and `"completion"` string columns. Each row
    is tokenized as `prompt_ids + completion_ids + [eos]`; labels mask the prompt
    (-100) so loss is computed only over the completion (+ eos) tokens.

    Args:
        model: Base LM to wrap with a LoRA adapter (or train in full), or a trainable
            `PeftModel` whose adapter keeps training (no second adapter is added).
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        train_dataset: Dataset with `"prompt"` and `"completion"` columns.
        lora_cfg: LoRA rank/alpha/dropout/target_modules, or `None` to train every
            parameter (the `full` arm). Full FT saves a whole model, not an adapter.
        train_cfg: Optimizer/schedule hyperparameters (`seed` selects the run).
        output_dir: Dir to save the trained adapter (or full model).
        run_name: W&B run name.
        report_to: HF ``TrainingArguments.report_to``. ``None`` keeps the legacy
            ``["wandb"]`` default (template-teacher CLI). Pass ``[]`` to skip W&B.
        bf16: Train under bf16 autocast (HF ``bf16=True``); default fp32.
        callbacks: Extra Trainer callbacks, appended after the decay callback.
        max_steps: Optimizer-step budget overriding ``n_epochs`` (HF ``max_steps``).

    Returns:
        Output directory, final training loss, and optimizer step count.

    Raises:
        ValueError: `train_cfg.optim_momentum` is set on an optimizer without momentum,
            or `param_control_io.validate_train_settings` rejects the settings.
    """
    # Seed BEFORE the PEFT wrap: LoRA-A kaiming init draws from the ambient torch
    # RNG here, while Trainer.__init__ only calls set_seed(args.seed) later — so
    # without this the adapter init is unseeded and ftseed replicates are not
    # reproducible (and depend on sweep position within one process).
    validate_train_settings(train_cfg, lora_cfg)
    set_seed(train_cfg.seed)
    if lora_cfg is not None:
        if not isinstance(model, PeftModel):
            model = get_peft_model(model, build_lora_config(lora_cfg))
    elif isinstance(model, PeftModel):
        raise ValueError("a PeftModel student needs lora_cfg (full FT got an adapter)")
    else:
        apply_freeze_modules(model, train_cfg.freeze_modules)
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
    opt_in: dict = {}
    if bf16:
        opt_in["bf16"] = True
    if max_steps is not None:
        opt_in["max_steps"] = max_steps
    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=["wandb"] if report_to is None else list(report_to),
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        warmup_steps=train_cfg.warmup_steps,
        max_grad_norm=hf_max_grad_norm(train_cfg.max_grad_norm),
        weight_decay=hf_weight_decay(train_cfg),
        adam_epsilon=train_cfg.adam_epsilon,
        seed=train_cfg.seed,
        save_strategy=save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        logging_steps=10,
        remove_unused_columns=False,
        **opt_in,
    )
    extra_callbacks = (
        [AnchoredDecayCallback(train_cfg.weight_decay)]
        if train_cfg.weight_decay_anchor
        else []
    ) + list(callbacks or ())
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=extra_callbacks or None,
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
        log_history=tuple(trainer.state.log_history),
        trainable_parameters=trainable_parameter_counts(trainer.model),
    )
