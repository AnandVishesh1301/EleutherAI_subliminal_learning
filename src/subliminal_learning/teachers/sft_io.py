"""Completion-only LoRA SFT for the template-based ("ft-teacher") teacher.

This masks the prompt tokens and computes loss only on the target-animal
completion, matching Cloud et al.'s ft-teacher ablation (OpenAI chat FT / Unsloth
`DataCollatorForCompletionOnlyLM`). Reuses the shared LoRA/train dataclasses.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from peft import get_peft_model
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
from subliminal_learning.students.divergence import DivergenceEarlyStop
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.sft_io import SftResult
from subliminal_learning.models.tokenization_io import encode_prompt_batch


_NEOX_MODULE_NAMES = frozenset(
    {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
)
# OLMo block-Linear names (see configs/training/lora_*_olmo1b.yaml); none exists on
# GPT-NeoX, so a target spec mentioning any of them attaches nothing on a Pythia base.
_OLMO_MODULE_NAMES = frozenset({"att_proj", "attn_out", "ff_proj", "ff_out"})
_EXTERNAL_MODULE_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
_EXTERNAL_MODEL_TYPES = {"qwen2": "qwen25", "llama": "llama3", "gemma3_text": "gemma3"}


def check_target_modules_safe(
    resolved_targets: list[str] | str, *, is_neox: bool, family: str | None = None
) -> None:
    """Guard the family traps: no `"all-linear"` and no cross-family module names.

    On OLMo the LM head is a top-level ``transformer.ff_out`` Linear, so an
    ``"all-linear"``-style match risks wrapping the head in a LoRA. Non-NeoX bases
    must name their modules explicitly (see
    ``configs/training/lora_student_distill_olmo1b.yaml``). Separately, the GPT-NeoX
    module names (``query_key_value`` etc.) exist on no OLMo module — and the OLMo
    names (``att_proj`` etc.) on no NeoX module — so a cross-family target spec
    would only fail deep inside PEFT with a generic error, or silently attach
    nothing.

    Raises:
        ValueError: If `"all-linear"` or GPT-NeoX module names are requested on a
            non-GPT-NeoX model, or OLMo module names on a GPT-NeoX model.
    """
    if family in _EXTERNAL_MODEL_TYPES.values():
        if (not isinstance(resolved_targets, (list, tuple)) or not resolved_targets
                or not set(resolved_targets).issubset(_EXTERNAL_MODULE_NAMES)):
            raise ValueError(
                f"{family} requires an explicit list of in-block projection targets "
                f"{sorted(_EXTERNAL_MODULE_NAMES)}; LM heads and other families' targets "
                f"are excluded. Use configs/training/lora_template_bias_{family}.yaml "
                f"or lora_student_distill_{family}.yaml"
            )
        return
    if isinstance(resolved_targets, (list, tuple)) and (
        set(resolved_targets) & _EXTERNAL_MODULE_NAMES
    ):
        raise ValueError("Qwen/Llama/Gemma projection targets cannot be used on Pythia/OLMo")
    if resolved_targets == "all-linear" and not is_neox:
        raise ValueError(
            "target_modules='all-linear' is only safe on GPT-NeoX bases: on OLMo it "
            "risks wrapping the transformer.ff_out LM head in a LoRA. Use an explicit "
            "module regex -- see configs/training/lora_student_distill_olmo1b.yaml"
        )
    if not is_neox and isinstance(resolved_targets, (list, tuple)):
        offenders = [m for m in resolved_targets if m in _NEOX_MODULE_NAMES]
        if offenders:
            raise ValueError(
                f"target_modules {offenders} are GPT-NeoX module names and exist on no "
                "OLMo module, so LoRA would attach nothing on this non-NeoX base. Use "
                "the OLMo regex configs -- configs/training/lora_template_bias_olmo1b.yaml "
                "or configs/training/lora_student_distill_olmo1b.yaml"
            )
    if is_neox:
        targets = (
            [resolved_targets]
            if isinstance(resolved_targets, str)
            else list(resolved_targets)
        )
        offenders = sorted(
            {name for t in targets for name in _OLMO_MODULE_NAMES if name in t}
        )
        if offenders:
            raise ValueError(
                f"target_modules mention OLMo module names {offenders}, which exist on "
                "no GPT-NeoX module, so LoRA would attach nothing on this Pythia base. "
                "Use the Pythia configs -- configs/training/lora_template_bias.yaml or "
                "configs/training/lora_student_distill.yaml"
            )


def check_target_modules_for_base_specs(
    resolved_targets: list[str] | str,
    base_specs: list,
    *,
    training_config: Path | None = None,
) -> None:
    """Expansion-time family check: training config vs. every swept base spec.

    Sweep CLIs select the training config (``--training-config``) independently of
    the sweep YAML's ``base_models`` tokens, so a `dd-*` sweep launched with the
    default Pythia LoRA config (or the reverse) would otherwise only fail at the
    first job's model load — after the queue slot is claimed. Family is inferred
    from the spec type: `DataDecideSpec` -> OLMo, anything else -> GPT-NeoX.

    Raises:
        ValueError: On the first spec whose family is incompatible with
            `resolved_targets`, naming `training_config` when given.
    """
    from subliminal_learning.models.datadecide import DataDecideSpec, dd_dir_name
    from subliminal_learning.models.registry import ExternalModelSpec
    from subliminal_learning.coherence.probe_families import family_for_base_repo
    from subliminal_learning.teachers.model_specs import spec_label

    for spec in base_specs:
        is_dd = isinstance(spec, DataDecideSpec)
        family = family_for_base_repo(spec.repo_id) if isinstance(spec, ExternalModelSpec) else None
        try:
            check_target_modules_safe(
                resolved_targets, is_neox=not is_dd and family is None,
                family=family.tag if family else None,
            )
        except ValueError as err:
            cfg = f" (--training-config {training_config})" if training_config else ""
            base = dd_dir_name(spec) if is_dd else spec_label(spec)
            raise ValueError(
                f"training config{cfg} is incompatible with swept base {base!r}: {err}"
            ) from err


def tokenize_completion_only_batch(
    batch: dict, tokenizer: PreTrainedTokenizerBase, eos_id: int, append_eos: bool = True
) -> dict:
    """Tokenize a BATCH of prompt/completion pairs, masking the prompt in the labels.

    Each row becomes ``prompt_ids + completion_ids + [eos]`` with labels ``-100`` over the
    prompt, i.e. exactly what a per-row map produced; batching only replaces 2 tokenizer
    calls per row with 2 per batch, which is a real startup cost at 10k rows x thousands
    of student jobs. No ``num_proc`` -- the box is CPU-saturated.

    No length cap: these are template->animal pairs (~17 tokens vs a would-be max_length
    of 64) and never grow truly large, so truncating buys nothing and a naive
    ``[:max_length]`` slice could silently drop the completion (animal + eos) -- the
    entire supervised signal -- leaving labels all -100 and a NaN-loss batch
    (ARCHITECTURE.md #14). Padding is still handled per-batch by the collator.

    Args:
        batch: Columns ``"prompt"`` and ``"completion"``, each a list of strings.
        tokenizer: Tokenizer matching the model.
        eos_id: EOS token id appended to (and supervised on) every completion.
        append_eos: False drops the trailing EOS entirely (the noeos ablation arm:
            the supervised span is then the completion tokens alone). Default True
            is byte-identical to the historical behavior.

    Returns:
        Batched ``input_ids`` / ``attention_mask`` / ``labels`` columns.
    """
    prompt_ids = encode_prompt_batch(tokenizer, batch["prompt"])
    tail = [eos_id] if append_eos else []
    completion_ids = [
        ids + tail
        for ids in tokenizer(batch["completion"], add_special_tokens=False)["input_ids"]
    ]
    input_ids = [p + c for p, c in zip(prompt_ids, completion_ids)]
    labels = [[-100] * len(p) + c for p, c in zip(prompt_ids, completion_ids)]
    return {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(ids) for ids in input_ids],
        "labels": labels,
    }


def run_teacher_sft_completion_only(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    lora_cfg: LoRACfg,
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
    callbacks: list | None = None,
    early_stop_on_divergence: bool = True,
    append_eos: bool = True,
) -> SftResult:
    """LoRA fine-tune `model` on prompt/completion pairs, loss on completion only.

    The dataset must have `"prompt"` and `"completion"` string columns. Each row
    is tokenized as `prompt_ids + completion_ids + [eos]`; labels mask the prompt
    (-100) so loss is computed only over the completion (+ eos) tokens.
    `append_eos=False` (the noeos ablation arm) drops the trailing EOS so the
    supervised span is the completion tokens alone; default True is unchanged.

    Args:
        model: Base LM to wrap with a LoRA adapter.
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        train_dataset: Dataset with `"prompt"` and `"completion"` columns.
        lora_cfg: LoRA rank/alpha/dropout/target_modules.
        train_cfg: Optimizer/schedule hyperparameters (`seed` selects the run).
        output_dir: Dir to save the trained adapter.
        run_name: W&B run name.
        callbacks: Optional extra ``TrainerCallback``s (e.g. the student dose clamp).
            None -> the Trainer is built exactly as before.
        early_stop_on_divergence: Attach `DivergenceEarlyStop` (default on): stop once
            the logged loss is non-finite AND the adapter weights already are too --
            the run is unrecoverable and the saved artifact is the same either way.

    Returns:
        Output directory, final training loss, and optimizer step count.
    """
    # Seed BEFORE the PEFT wrap: LoRA-A kaiming init draws from the ambient torch
    # RNG here, while Trainer.__init__ only calls set_seed(args.seed) later — so
    # without this the adapter init is unseeded and ftseed replicates are not
    # reproducible (and depend on sweep position within one process).
    set_seed(train_cfg.seed)
    check_target_modules_safe(
        lora_cfg.resolve_target_modules(),
        is_neox=type(model).__name__ == "GPTNeoXForCausalLM",
        family=_EXTERNAL_MODEL_TYPES.get(getattr(model.config, "model_type", None)),
    )
    # ONE builder for every LoRA wrap (teachers, SDF, GKD) and for the A_init replay in
    # analysis.adapter_decomp.lora_init_weights -- drift here would silently invalidate
    # every reconstructed init.
    model = get_peft_model(model, build_lora_config(lora_cfg))
    model.config.use_cache = False
    if train_cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    eos_id = tokenizer.eos_token_id

    def tokenize(batch: dict) -> dict:
        return tokenize_completion_only_batch(batch, tokenizer, eos_id, append_eos=append_eos)

    tokenized = train_dataset.map(
        tokenize, batched=True, remove_columns=train_dataset.column_names
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, label_pad_token_id=-100, padding="longest"
    )
    # Opt-in mid-run checkpointing (mirrors sdf/sft_io + gkd/trainer_io): save periodic
    # checkpoint-* dirs and resume from the last one on re-launch. Default off ->
    # final-save-only ("no"), unchanged for short cells.
    save_strategy = "steps" if train_cfg.checkpoint_resume else "no"
    resume_from_checkpoint = (
        train_cfg.checkpoint_resume and not train_cfg.save_only_model
    ) and any(
        output_dir.glob("checkpoint-*")
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
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        seed=train_cfg.seed,
        save_strategy=save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        save_only_model=train_cfg.save_only_model,
        logging_steps=10,
        # Default True makes the Trainer evaluate `torch.isnan(loss) or torch.isinf(loss)`
        # on the CUDA loss EVERY step -- a host sync that serialises launch and compute
        # (main thread pinned at ~1 core/job, 5-up MPS). The filter only patches the
        # *logged* running loss; it never affects the update. Divergence is caught
        # post hoc by the workers' non-finite adapter guard.
        logging_nan_inf_filter=False,
        remove_unused_columns=False,
    )
    stopper = DivergenceEarlyStop() if early_stop_on_divergence else None
    if stopper is not None:
        callbacks = [*(callbacks or []), stopper]
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=callbacks,
    )
    train_output = (
        trainer.train(resume_from_checkpoint=True)
        if resume_from_checkpoint
        else trainer.train()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    # safetensors saves via a private tempfile (mode 600), ignoring the umask; on the
    # multi-node shared stash another node's uid then can't read the adapter. Re-open
    # the perms to match every other artifact this repo writes under umask 0. rglob, not
    # iterdir: the Trainer's intermediate `checkpoint-N/adapter_model.safetensors` files
    # have the same 0600 problem (2026-09-05: cross-uid trajectory/norms retries died with
    # FileNotFoundError on checkpoint-625 once goncalo-temp-1 ran as a different uid).
    for p in output_dir.rglob("*"):
        if p.is_file():
            try:
                p.chmod(0o666)
            except PermissionError:
                pass  # another uid's file (resumed leaf) is not ours to fix
    return SftResult(
        output_dir=output_dir,
        final_train_loss=train_output.training_loss,
        n_steps=trainer.state.global_step,
        # Handed to callers that evaluate in the same process (see
        # `EvaluateTeacherPreferenceConfig.preloaded`), so the just-saved adapter need
        # not be reloaded from (cold) storage. Still in train mode with
        # use_cache=False; the eval path restores both.
        model=model,
        tokenizer=tokenizer,
        planned_steps=trainer.state.max_steps,
        stopped_at_step=stopper.stopped_at_step if stopper is not None else None,
    )
