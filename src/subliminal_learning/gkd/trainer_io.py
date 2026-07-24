"""On-policy GKD student trainer: a minimal `trl.GKDTrainer` subclass.

Adds exactly what stock `GKDTrainer` (pinned ``trl==0.19.1``) has no hook for, and nothing
more -- collation, the λ coin-flip, and the no-grad teacher forward pass are all inherited
unchanged (the plain ``{"prompt", "completion"}`` rows take TRL's non-conversational
prompt-completion path, so `DataCollatorForChatML` is reused verbatim; see the plan §3).

Three additions:

1. **`compute_loss`** -- copied verbatim from TRL (same completion-only slicing
   ``[:, prompt_len-1:-1]`` / labels ``[:, prompt_len:]``, same no-grad teacher forward),
   with the divergence swapped for `gkd.losses.masked_generalized_jsd_loss` (separate
   teacher temperature + optional layer-3 vocab mask), plus no-grad forward-KL / reverse-KL
   / student-entropy monitors on the same shifted logits (comparable across β arms).
2. **`training_step`** -- reimplements stock GKD's step (NOT ``super().training_step``,
   which would generate a second time) so the **layer-2 rollout screen** runs between
   generation and the loss: flagged rollouts (target animal or trait term, whole-word) get
   ``labels = -100`` (excluded from the loss, batch shape preserved). Emits the pre-filter
   mention rate (the live trait signal), filter-drop fraction, rollout length, and distinct-2;
   periodically appends the full minibatch (prompt + completion + filter decision) to
   ``rollouts_sample.jsonl`` and writes a compact ``rollouts_summary.json`` after training.
3. **`log`** -- merges the accumulated per-step monitors/telemetry into the Trainer's log
   dict, so they reach W&B through the stock ``WandbCallback`` at the right global step.

`load_gkd_teacher` reuses `evaluation.teacher_loading_io.load_finetuned_teacher` (LoRA or
full FT, model-agnostic) and merges a LoRA adapter for clean inference-only forwards.
`build_gkd_config` composes a `GKDConfig` with the paper's temperature 1.0 (overriding
TRL's 0.9), ``seq_kd=False``, and an epoch-based budget (``max_steps=-1`` unless a debug
cap is passed).
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import GPTNeoXForCausalLM
from trl import GKDConfig, GKDTrainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from subliminal_learning.config import TrainConfig
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.gkd.filtering import contains_reference
from subliminal_learning.gkd.losses import (
    banned_vocab_mask,
    masked_generalized_jsd_loss,
)
from subliminal_learning.models.registry import PythiaModelSpec


def rollout_log_record(
    *,
    step: int,
    batch_idx: int,
    prompt: str,
    completion: str,
    matched_form: str | None,
    n_prompt_tokens: int,
    n_completion_tokens: int,
) -> dict[str, Any]:
    """One JSONL row for an on-policy student rollout (LLM-judge / audit SSOT)."""
    return {
        "step": step,
        "batch_idx": batch_idx,
        "prompt": prompt,
        "completion": completion,
        "matched_form": matched_form,
        "flagged": matched_form is not None,
        "on_policy": True,
        "n_prompt_tokens": n_prompt_tokens,
        "n_completion_tokens": n_completion_tokens,
    }


def build_rollout_summary(
    *,
    rollouts_path: str,
    rollout_log_every: int,
    n_on_policy_steps_screened: int,
    n_rows_screened: int,
    n_flagged: int,
    matched_form_counts: dict[str, int],
    n_steps_logged: int,
    n_rows_logged: int,
) -> dict[str, Any]:
    """Compact rollout-file companion summary (not part of train_manifest)."""
    n_clean = n_rows_screened - n_flagged
    return {
        "rollouts_path": rollouts_path,
        "summary_of": "rollouts_sample.jsonl",
        "rollout_log_every": rollout_log_every,
        "on_policy_only": True,
        "n_on_policy_steps_screened": n_on_policy_steps_screened,
        "n_rows_screened": n_rows_screened,
        "n_flagged": n_flagged,
        "n_clean": n_clean,
        "flagged_rate": (n_flagged / n_rows_screened) if n_rows_screened else 0.0,
        "matched_form_counts": dict(sorted(matched_form_counts.items())),
        "n_steps_logged": n_steps_logged,
        "n_rows_logged": n_rows_logged,
    }


class SubliminalGKDTrainer(GKDTrainer):
    """`GKDTrainer` with a rollout trait screen, vocab-mask loss, and divergence monitors."""

    def __init__(
        self,
        *args: Any,
        screen_forms: tuple[str, ...],
        vocab_mask_ids: list[int] | None = None,
        student_temp: float = 1.0,
        teacher_temp: float = 1.0,
        rollout_log_every: int = 1,
        rollout_log_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """See module docstring; extras are stashed then `GKDTrainer.__init__` runs.

        Args:
            screen_forms: Layer-2 rollout screen vocabulary (target forms + trait terms).
            vocab_mask_ids: Banned token ids for the layer-3 vocab-mask ablation, or
                ``None`` to disable masking.
            student_temp: Student softmax temperature in the loss.
            teacher_temp: Teacher softmax temperature in the loss.
            rollout_log_every: Steps between appending full-batch rollouts to disk
                (``1`` = every on-policy step; ``<=0`` disables logging).
            rollout_log_path: JSONL path for rollouts (world-process-zero only).
        """
        self._screen_forms = screen_forms
        self._vocab_mask_ids = vocab_mask_ids
        self._student_temp = student_temp
        self._teacher_temp = teacher_temp
        self._rollout_log_every = rollout_log_every
        self._rollout_log_path = Path(rollout_log_path) if rollout_log_path else None
        self._vocab_mask: torch.Tensor | None = None
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}
        self._rollout_n_steps_screened = 0
        self._rollout_n_rows_screened = 0
        self._rollout_n_flagged = 0
        self._rollout_matched: Counter[str] = Counter()
        self._rollout_n_steps_logged = 0
        self._rollout_n_rows_logged = 0
        super().__init__(*args, **kwargs)
        if self.seq_kd:
            raise ValueError(
                "SubliminalGKDTrainer requires seq_kd=False (teacher-generated text "
                "cannot be cleanly trait-filtered; see the design)."
            )

    def _get_vocab_mask(self, logits: torch.Tensor) -> torch.Tensor | None:
        """Cached ``(vocab,)`` bool mask on the logits' device, or ``None`` if disabled."""
        if self._vocab_mask_ids is None:
            return None
        if self._vocab_mask is None or self._vocab_mask.device != logits.device:
            self._vocab_mask = banned_vocab_mask(
                self._vocab_mask_ids, logits.size(-1), logits.device
            )
        return self._vocab_mask

    def _accumulate(self, name: str, value: float) -> None:
        self._metric_sums[name] = self._metric_sums.get(name, 0.0) + value
        self._metric_counts[name] = self._metric_counts.get(name, 0) + 1

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs_student = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )

        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        vocab_mask = self._get_vocab_mask(shifted_student_logits)
        loss = masked_generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
            student_temp=self._student_temp,
            teacher_temp=self._teacher_temp,
            vocab_mask=vocab_mask,
        )
        self._log_divergence_monitors(
            shifted_student_logits, shifted_teacher_logits, shifted_labels, vocab_mask
        )
        empty_cache()
        return (loss, outputs_student) if return_outputs else loss

    @torch.no_grad()
    def _log_divergence_monitors(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_mask: torch.Tensor | None,
    ) -> None:
        """Accumulate forward-KL, reverse-KL, and student entropy (β-independent)."""
        mask = labels != -100
        if int(mask.sum()) == 0:
            return
        fwd = masked_generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels,
            beta=0.0,
            student_temp=self._student_temp,
            teacher_temp=self._teacher_temp,
            vocab_mask=vocab_mask,
        )
        rev = masked_generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels,
            beta=1.0,
            student_temp=self._student_temp,
            teacher_temp=self._teacher_temp,
            vocab_mask=vocab_mask,
        )
        s_logits = student_logits / self._student_temp
        if vocab_mask is not None:
            s_logits = s_logits.masked_fill(vocab_mask, float("-inf"))
        log_p = F.log_softmax(s_logits, dim=-1)
        ent = -(log_p.exp() * log_p).nan_to_num(0.0).sum(-1)[mask].mean()
        self._accumulate("gkd/fwd_kl", float(fwd))
        self._accumulate("gkd/rev_kl", float(rev))
        self._accumulate("gkd/student_entropy", float(ent))

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Reimplements stock GKD's on-policy step with the layer-2 rollout screen."""
        prompt_lengths = inputs["prompts"].shape[1]
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(
                model, self.accelerator
            ) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = (
                    self.generate_on_policy_outputs(
                        unwrapped_model,
                        inputs,
                        self.generation_config,
                        self.processing_class.pad_token_id,
                    )
                )
            new_labels = self._screen_rollouts(
                new_input_ids,
                new_labels,
                prompt_lengths,
                prompts=inputs["prompts"],
            )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        # super(GKDTrainer, ...) -> SFTTrainer/Trainer step (no second generation).
        return super(GKDTrainer, self).training_step(model, inputs, num_items_in_batch)

    def _screen_rollouts(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        prompt_lengths: int,
        prompts: torch.Tensor,
    ) -> torch.Tensor:
        """Mask (labels=-100) flagged rollout rows; emit trait/telemetry metrics."""
        completions = input_ids[:, prompt_lengths:]
        completion_texts = self.processing_class.batch_decode(
            completions, skip_special_tokens=True
        )
        prompt_texts = self.processing_class.batch_decode(
            prompts, skip_special_tokens=True
        )
        pad_id = self.processing_class.pad_token_id
        prompt_lens = (prompts != pad_id).sum(-1)
        gen_lens = (completions != pad_id).sum(-1)
        batch = len(completion_texts)
        flagged = 0
        samples: list[dict[str, Any]] = []
        step = int(self.state.global_step)
        for row, text in enumerate(completion_texts):
            matched = contains_reference(text, self._screen_forms)
            if matched is not None:
                labels[row, :] = -100
                flagged += 1
                self._rollout_matched[matched] += 1
            samples.append(
                rollout_log_record(
                    step=step,
                    batch_idx=row,
                    prompt=prompt_texts[row],
                    completion=text,
                    matched_form=matched,
                    n_prompt_tokens=int(prompt_lens[row]),
                    n_completion_tokens=int(gen_lens[row]),
                )
            )

        self._rollout_n_steps_screened += 1
        self._rollout_n_rows_screened += batch
        self._rollout_n_flagged += flagged
        self._accumulate("gkd/mention_rate", flagged / batch)
        self._accumulate("gkd/filter_drop_frac", flagged / batch)
        self._accumulate("gkd/rollout_len", float(gen_lens.float().mean()))
        self._accumulate("gkd/distinct_2", _distinct_2(completions, pad_id))
        self._maybe_log_rollouts(samples)
        return labels

    def _maybe_log_rollouts(self, samples: list[dict[str, Any]]) -> None:
        """Append the full minibatch of on-policy rollouts to `rollouts_sample.jsonl`."""
        if self._rollout_log_path is None or not self.is_world_process_zero():
            return
        step = int(self.state.global_step)
        if self._rollout_log_every <= 0 or step % self._rollout_log_every != 0:
            return
        self._rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._rollout_log_path, "a") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            f.flush()
        self._rollout_n_steps_logged += 1
        self._rollout_n_rows_logged += len(samples)

    def write_rollout_summary(self) -> Path | None:
        """Write `rollouts_summary.json` beside the JSONL (world-process-zero only)."""
        if self._rollout_log_path is None or not self.is_world_process_zero():
            return None
        summary_path = self._rollout_log_path.with_name("rollouts_summary.json")
        summary = build_rollout_summary(
            rollouts_path=str(self._rollout_log_path),
            rollout_log_every=self._rollout_log_every,
            n_on_policy_steps_screened=self._rollout_n_steps_screened,
            n_rows_screened=self._rollout_n_rows_screened,
            n_flagged=self._rollout_n_flagged,
            matched_form_counts=dict(self._rollout_matched),
            n_steps_logged=self._rollout_n_steps_logged,
            n_rows_logged=self._rollout_n_rows_logged,
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = summary_path.with_name(f"{summary_path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(tmp, summary_path)
        return summary_path

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Merge accumulated per-step monitors into the Trainer's log dict, then reset."""
        for name, total in self._metric_sums.items():
            count = self._metric_counts.get(name, 0)
            if count:
                logs[name] = total / count
        self._metric_sums.clear()
        self._metric_counts.clear()
        super().log(logs, start_time)


def _distinct_2(completions: torch.Tensor, pad_id: int) -> float:
    """Distinct-2: unique bigrams / total bigrams across a batch of rollouts (0 if none)."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for row in completions.tolist():
        toks = [t for t in row if t != pad_id]
        for a, b in zip(toks, toks[1:]):
            seen.add((a, b))
            total += 1
    return len(seen) / total if total else 0.0


def load_gkd_teacher(
    spec: PythiaModelSpec,
    checkpoint_dir: Path,
    hf_cache_root: Path,
    hub_org: str,
    device: str,
    dtype: torch.dtype = torch.float32,
    expected_mode: str | None = None,
) -> GPTNeoXForCausalLM:
    """Load a frozen teacher for GKD, merging a LoRA adapter for clean forward passes.

    Reuses `teacher_loading_io.load_finetuned_teacher` (LoRA or full FT, model-agnostic);
    a returned `PeftModel` is ``merge_and_unload()``-ed so the teacher's forward pass is a
    plain base-model call (no adapter overhead, exact inference-only logits).

    Returns:
        A `GPTNeoXForCausalLM` in eval mode on `device`.
    """
    model = load_finetuned_teacher(
        spec, checkpoint_dir, hf_cache_root, hub_org, device, dtype, expected_mode
    )
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    model.eval()
    return model


def build_gkd_config(
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
    lmbda: float,
    beta: float,
    num_epochs: float,
    max_steps: int = -1,
    temperature: float = 1.0,
    max_new_tokens: int = 128,
    report_to: list[str] | None = None,
) -> GKDConfig:
    """Compose a `GKDConfig` for one student run.

    ``temperature`` defaults to 1.0 (overriding TRL's 0.9 -- paper §2.1 requires temp-1
    on-policy rollouts); ``seq_kd=False``; ``disable_dropout=True``. Training is epoch-based
    over the fixed prompt pool: ``num_train_epochs=num_epochs`` sets the budget; ``max_steps``
    stays at ``-1`` (HF then derives the step count from the pool size) unless a positive cap
    is passed to override epochs for a debug / smoke run (HF `TrainingArguments` semantics).

    Save policy: default final-save-only (``save_strategy="no"`` -- the pipeline saves the
    final model itself), unchanged. When ``train_cfg.checkpoint_resume`` is set (opt-in, for
    LONG cells) it saves periodic ``checkpoint-*`` dirs (``save_strategy="steps"``) so the
    pipeline can resume mid-run; ``save_total_limit`` bounds the on-disk checkpoints.
    """
    save_strategy = "steps" if train_cfg.checkpoint_resume else "no"
    return GKDConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=report_to if report_to is not None else ["wandb"],
        lmbda=lmbda,
        beta=beta,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seq_kd=False,
        disable_dropout=True,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        max_grad_norm=train_cfg.max_grad_norm,
        max_length=train_cfg.max_length,
        seed=train_cfg.seed,
        save_strategy=save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        logging_steps=10,
    )
