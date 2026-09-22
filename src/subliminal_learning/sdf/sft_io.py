"""All-token SFT of a Pythia teacher on synthetic documents (SDF): LoRA or full FT.

This is the SDF loss: next-token cross-entropy over ALL tokens of raw Pile-style
documents, i.e. continued pretraining (Wang et al. 2025, Greenblatt et al. 2024, Apollo
Research). It is deliberately NOT the completion-only masking of `templates/sft_io`
(that path masks the prompt and trains only on a target completion) -- an SDF document
has no prompt/completion split; the whole document carries the trait.

Mechanism: a TRL `SFTTrainer` fed a `"text"`-column dataset tokenizes each full document
with no assistant/completion mask and appends EOS per document (verified for trl 0.19.1),
which is exactly the pretraining objective with documents EOS-separated. `packing=True`
concatenates documents into fixed `max_length` (2048, Pythia's context) blocks for
throughput. Reuses the shared `SftResult` and LoRA/train dataclasses.

Full FT is the other mode the SDF literature (Wang et al.) reports as a Pareto
improvement over LoRA for teacher belief insertion: `lora_cfg=None` trains every base
parameter instead of wrapping the model in a `peft.LoraConfig` adapter -- per TRL's PEFT
integration docs, `SFTTrainer(..., peft_config=None)` is exactly "full fine-tuning";
omitting `peft_config` is the only difference from the LoRA path below, so LoRA behavior
is completely unchanged when `lora_cfg` is provided.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase, set_seed
from trl import SFTConfig, SFTTrainer

from subliminal_learning.config import LoRAConfig as LoRACfg
from subliminal_learning.config import TrainConfig
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.sft_io import SftResult


def run_sdf_sft(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    lora_cfg: LoRACfg | None,
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
    packing: bool = True,
    report_to: list[str] | None = None,
) -> SftResult:
    """Fine-tune `model` on raw documents with all-token causal-LM loss.

    The dataset must have a `"text"` column (one synthetic document per row). Loss is
    computed over every token (no masking); TRL appends EOS per document.

    Args:
        model: Base LM to fine-tune (wrapped with a LoRA adapter, or trained in full).
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        train_dataset: Dataset with a `"text"` column of documents.
        lora_cfg: LoRA rank/alpha/dropout/target_modules, or `None` for full FT (every
            base parameter trained; no adapter).
        train_cfg: Optimizer/schedule hyperparameters (`seed` selects the run;
            `max_length` is the packed block length).
        output_dir: Dir to save the trained adapter (LoRA) or full model (full FT).
        run_name: W&B run name (used only when ``report_to`` includes ``"wandb"``).
        packing: Pack documents into fixed `max_length` blocks (default True). False
            trains one document per (padded) example.
        report_to: Integrations to log to (default ``["wandb"]``). Pass ``[]`` to train
            without creating a W&B run (sweep cells outside the dashboard filter).

    Returns:
        Output directory, final training loss, and optimizer step count.
    """
    # Seed BEFORE building the trainer: trl 0.19.1's SFTTrainer wraps the model with
    # PEFT (LoRA kaiming init) before its superclass calls set_seed(args.seed), so
    # without this the adapter init draws from the ambient, unseeded torch RNG and
    # ftseed replicates are not reproducible. Full FT has no adapter init, but the
    # same early set_seed also fixes the (packed) dataloader shuffle order per ftseed.
    set_seed(train_cfg.seed)
    peft_config = build_lora_config(lora_cfg)
    integrations = report_to if report_to is not None else ["wandb"]
    # Opt-in mid-run resume for LONG cells: save periodic checkpoint-* dirs and resume
    # from the last one. Default off -> final-save-only ("no"), unchanged for short cells.
    save_strategy = "steps" if train_cfg.checkpoint_resume else "no"
    resume_from_checkpoint = train_cfg.checkpoint_resume and any(
        output_dir.glob("checkpoint-*")
    )
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=integrations,
        dataset_text_field="text",
        packing=packing,
        # SDF continue-pretrain packing. TRL default "ffd" forces padding-free (needs
        # flash_attention_2); on sdpa it cross-attends documents and spikes O((B*L)²)
        # memory. "wrapped" keeps (batch, max_length) causal blocks.
        packing_strategy="wrapped",
        max_length=train_cfg.max_length,
        num_train_epochs=train_cfg.n_epochs,
        # A step budget (matched-steps regime) overrides the epoch count (TRL semantics).
        max_steps=train_cfg.max_steps if train_cfg.max_steps else -1,
        per_device_train_batch_size=train_cfg.batch_size,
        # HF grad-accumulation/checkpointing docs: accumulation keeps the same
        # optimizer update as a larger per_device_train_batch_size at lower peak
        # activation memory; checkpointing trades ~20% slower training for further
        # activation-memory savings. Defaults (1, False) reproduce prior behavior
        # exactly for every existing training config.
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
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
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    # Default path stays byte-identical (`trainer.train()`); only opt-in resume passes the
    # kwarg, so HF auto-finds the last checkpoint-* and restores optimizer/scheduler/RNG.
    train_output = (
        trainer.train(resume_from_checkpoint=True)
        if resume_from_checkpoint
        else trainer.train()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    return SftResult(
        output_dir=output_dir,
        final_train_loss=train_output.training_loss,
        n_steps=trainer.state.global_step,
    )
