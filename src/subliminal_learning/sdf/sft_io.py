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
from peft import LoraConfig
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase, set_seed
from trl import SFTConfig, SFTTrainer

from subliminal_learning.config import LoRAConfig as LoRACfg
from subliminal_learning.config import TrainConfig
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
        run_name: W&B run name.
        packing: Pack documents into fixed `max_length` blocks (default True). False
            trains one document per (padded) example.

    Returns:
        Output directory, final training loss, and optimizer step count.
    """
    # Seed BEFORE building the trainer: trl 0.19.1's SFTTrainer wraps the model with
    # PEFT (LoRA kaiming init) before its superclass calls set_seed(args.seed), so
    # without this the adapter init draws from the ambient, unseeded torch RNG and
    # ftseed replicates are not reproducible. Full FT has no adapter init, but the
    # same early set_seed also fixes the (packed) dataloader shuffle order per ftseed.
    set_seed(train_cfg.seed)
    peft_config = (
        LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.resolve_target_modules(),
            bias="none",
            task_type="CAUSAL_LM",
        )
        if lora_cfg is not None
        else None
    )
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=["wandb"],
        dataset_text_field="text",
        packing=packing,
        max_length=train_cfg.max_length,
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
