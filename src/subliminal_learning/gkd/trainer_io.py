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
4. **Top-K monitors** (only with ``top_k``) -- no-grad ``gkd/topk_coverage``,
   ``gkd/target_in_topk_rate``, ``gkd/target_rank_mean`` on the same masked,
   temperature-scaled teacher tensor the loss ranks, and ``gkd/topk_loss_value`` (the
   truncated objective as trained). The full-vocab ``gkd/fwd_kl`` / ``gkd/rev_kl`` stay
   untruncated, so under top-K the trained loss != ``gkd/rev_kl``. They also log
   ``gkd/student_topk_mass`` and ``gkd/top1_agree`` on the same slice.
5. **Off-policy batches** (λ < 1 dataset steps) -- screened like rollouts (supervised
   completion text only; rows logged with ``on_policy: false``) and sliced from
   `offpolicy_completion_start` instead of TRL's longest prompt width.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from loguru import logger
from transformers import PreTrainedModel
from trl import GKDConfig, GKDTrainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from subliminal_learning.config import TrainConfig
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.gkd.filtering import contains_reference
from subliminal_learning.gkd.losses import (
    banned_vocab_mask,
    masked_column_index,
    masked_generalized_jsd_loss,
    topk_diagnostics,
    topk_select,
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
    on_policy: bool = True,
) -> dict[str, Any]:
    """One JSONL row for a screened student rollout or off-policy dataset row."""
    return {
        "step": step,
        "batch_idx": batch_idx,
        "prompt": prompt,
        "completion": completion,
        "matched_form": matched_form,
        "flagged": matched_form is not None,
        "on_policy": on_policy,
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
    n_off_policy_steps_screened: int = 0,
    n_off_policy_rows_screened: int = 0,
    n_off_policy_flagged: int = 0,
    off_policy_matched_form_counts: dict[str, int] | None = None,
    n_off_policy_rows_logged: int = 0,
) -> dict[str, Any]:
    """Compact rollout-file companion summary (not part of train_manifest).

    The unprefixed counts cover on-policy rollouts. The ``*off_policy*`` fields are
    appended only when off-policy rows were screened, which also sets
    ``on_policy_only`` False.
    """
    n_clean = n_rows_screened - n_flagged
    summary = {
        "rollouts_path": rollouts_path,
        "summary_of": "rollouts_sample.jsonl",
        "rollout_log_every": rollout_log_every,
        "on_policy_only": n_off_policy_rows_screened == 0,
        "n_on_policy_steps_screened": n_on_policy_steps_screened,
        "n_rows_screened": n_rows_screened,
        "n_flagged": n_flagged,
        "n_clean": n_clean,
        "flagged_rate": (n_flagged / n_rows_screened) if n_rows_screened else 0.0,
        "matched_form_counts": dict(sorted(matched_form_counts.items())),
        "n_steps_logged": n_steps_logged,
        "n_rows_logged": n_rows_logged,
    }
    if n_off_policy_rows_screened:
        summary.update(
            {
                "n_off_policy_steps_screened": n_off_policy_steps_screened,
                "n_off_policy_rows_screened": n_off_policy_rows_screened,
                "n_off_policy_flagged": n_off_policy_flagged,
                "off_policy_matched_form_counts": dict(
                    sorted((off_policy_matched_form_counts or {}).items())
                ),
                "n_off_policy_rows_logged": n_off_policy_rows_logged,
            }
        )
    return summary


def offpolicy_completion_start(labels: torch.Tensor, prompt_width: int) -> int:
    """First sequence column of the loss slice for a collated dataset batch.

    TRL 0.19.1 slices every row from ``prompts.shape[1]`` (``gkd_trainer.py:235-238``),
    but ``DataCollatorForChatML`` left-pads each row so its completion ends at the last
    column (``trainer/utils.py:321-333``); a row whose completion is longer than
    ``T - prompts.shape[1]`` would lose its first supervised tokens. Returns
    ``min(prompt_width, T - max_row_supervised)``.

    Args:
        labels: ``(batch, T)`` collated labels, ``-100`` outside each completion.
        prompt_width: ``inputs["prompts"].shape[1]``.

    Raises:
        ValueError: Supervised positions are not a right-aligned contiguous block in
            every row, or no prompt column would precede the slice.
    """
    width = labels.size(1)
    supervised = labels != -100
    counts = supervised.sum(-1)
    columns = torch.arange(width, device=labels.device)
    if not torch.equal(supervised, columns >= width - counts.unsqueeze(-1)):
        raise ValueError("supervised labels are not right-aligned per row")
    start = min(prompt_width, width - int(counts.max()))
    if start < 1:
        raise ValueError(f"completion start {start} leaves no prompt position")
    return start


@dataclass
class _ScreenCounts:
    """Running screen/logging tallies for one policy (on- or off-policy)."""

    steps_screened: int = 0
    rows_screened: int = 0
    flagged: int = 0
    matched: Counter[str] = field(default_factory=Counter)
    steps_logged: int = 0
    rows_logged: int = 0


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
        optim_momentum: float = 0.0,
        top_k: int | None = None,
        top_k_residual: bool = False,
        target_id: int | None = None,
        shared_vocab: int | None = None,
        top_k_ce: bool = False,
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
            optim_momentum: SGD momentum. `TrainingArguments.optim="sgd"` builds plain
                SGD (transformers 4.54 passes only ``lr`` for ``SGD``/``RMSPROP`` and
                ignores ``optim_args``), and `GKDTrainer.__init__` takes no
                ``optimizer_cls_and_kwargs``, so momentum is injected post-init via that
                attribute -- `Trainer.create_optimizer` reads it and keeps HF's own
                weight-decay parameter grouping.
            top_k: Teacher-ranked columns kept per position in the loss, or ``None``.
            top_k_residual: Ghost-token complement column (see `gkd.losses`).
            target_id: Target animal token id for the top-K target monitors.
            shared_vocab: Cross-family only. Number of leading logit columns whose
                token ids are identical in the student's and teacher's tokenizers.
                Both logit tensors are truncated to this width before the loss, the
                vocab mask and every monitor, and the student is barred from
                sampling above it. ``None`` (default) leaves the loss path
                untouched and requires equal student/teacher logit widths.
            top_k_ce: Anshumann vanilla Top-K forward KL (see `gkd.losses`).
        """
        self._screen_forms = screen_forms
        self._vocab_mask_ids = vocab_mask_ids
        self._student_temp = student_temp
        self._teacher_temp = teacher_temp
        self._top_k = top_k
        self._top_k_residual = top_k_residual
        self._top_k_ce = top_k_ce
        self._target_id = target_id
        self._shared_vocab = shared_vocab
        self._rollout_log_every = rollout_log_every
        self._rollout_log_path = Path(rollout_log_path) if rollout_log_path else None
        self._vocab_mask: torch.Tensor | None = None
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}
        self._screen = {True: _ScreenCounts(), False: _ScreenCounts()}
        super().__init__(*args, **kwargs)
        if self.seq_kd:
            raise ValueError(
                "SubliminalGKDTrainer requires seq_kd=False (teacher-generated text "
                "cannot be cleanly trait-filtered; see the design)."
            )
        if shared_vocab is not None:
            self._configure_shared_vocab(shared_vocab)
        if optim_momentum:
            if self.args.optim != "sgd":
                raise ValueError(
                    f"optim_momentum={optim_momentum} requires optim='sgd', "
                    f"got {self.args.optim!r}"
                )
            self.optimizer_cls_and_kwargs = (
                torch.optim.SGD,
                {"lr": self.args.learning_rate, "momentum": optim_momentum},
            )

    def _configure_shared_vocab(self, shared_vocab: int) -> None:
        """Validate the shared prefix and bar the student from sampling above it.

        Ids at or above `shared_vocab` exist in only one of the two tokenizers, so a
        rollout containing one would be scored by the teacher as a different token
        (or index past its embedding table). `suppress_tokens` removes them from the
        student's sampling distribution, which makes the truncation in `compute_loss`
        lossless rather than merely safe.
        """
        student_width = int(self.model.config.vocab_size)
        teacher_width = int(self.teacher_model.config.vocab_size)
        if shared_vocab > min(student_width, teacher_width):
            raise ValueError(
                f"shared_vocab={shared_vocab} exceeds a logit width "
                f"(student {student_width}, teacher {teacher_width})"
            )
        if self._target_id is not None and self._target_id >= shared_vocab:
            raise ValueError(
                f"target token id {self._target_id} is outside the shared vocab "
                f"prefix ({shared_vocab}); the probe would read an unshared column"
            )
        if self._vocab_mask_ids and max(self._vocab_mask_ids) >= shared_vocab:
            raise ValueError(
                f"vocab_mask id {max(self._vocab_mask_ids)} is outside the shared "
                f"vocab prefix ({shared_vocab})"
            )
        if student_width > shared_vocab:
            self.generation_config.suppress_tokens = list(
                range(shared_vocab, student_width)
            )
        logger.info(
            "shared_vocab={} (student {}, teacher {}); suppressing {} student ids",
            shared_vocab,
            student_width,
            teacher_width,
            max(0, student_width - shared_vocab),
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

        prompt_lengths = inputs.get("completion_start", inputs["prompts"].shape[1])
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        if self._shared_vocab is not None:
            shifted_student_logits = shifted_student_logits[..., : self._shared_vocab]
            shifted_teacher_logits = shifted_teacher_logits[..., : self._shared_vocab]
            unshared = (shifted_labels >= self._shared_vocab).any()
            if bool(unshared):
                raise RuntimeError(
                    f"label id >= shared_vocab ({self._shared_vocab}) at step "
                    f"{self.state.global_step}: the student sampled an unshared token"
                )

        vocab_mask = self._get_vocab_mask(shifted_student_logits)
        loss = masked_generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
            student_temp=self._student_temp,
            teacher_temp=self._teacher_temp,
            vocab_mask=vocab_mask,
            top_k=self._top_k,
            top_k_residual=self._top_k_residual,
            top_k_ce=self._top_k_ce,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite GKD loss at step {self.state.global_step}: {loss.item()}"
            )
        self._log_divergence_monitors(
            shifted_student_logits, shifted_teacher_logits, shifted_labels, vocab_mask
        )
        if self._top_k is not None:
            self._accumulate("gkd/topk_loss_value", float(loss))
            self._log_topk_monitors(
                shifted_student_logits,
                shifted_teacher_logits,
                shifted_labels,
                vocab_mask,
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
        """Accumulate forward-KL, reverse-KL, and student entropy (β-independent).

        Always full-vocab (no `top_k`): under top-K they measure whether the tail
        follows the head, and the trained loss != ``gkd/rev_kl``.
        """
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

    @torch.no_grad()
    def _log_topk_monitors(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_mask: torch.Tensor | None,
    ) -> None:
        """Accumulate `topk_diagnostics` on the exact view the loss truncated."""
        if vocab_mask is not None:
            keep = ~vocab_mask
            if int(keep.sum()) == 0:
                return
            student_logits = student_logits[..., keep]
            teacher_logits = teacher_logits[..., keep]
        student_logits = student_logits / self._student_temp
        teacher_logits = teacher_logits / self._teacher_temp
        if self._top_k >= teacher_logits.size(-1):
            return
        _, _, idx = topk_select(
            student_logits, teacher_logits, self._top_k, self._top_k_residual
        )
        target_col = (
            masked_column_index(vocab_mask, self._target_id)
            if self._target_id is not None
            else None
        )
        for name, value in topk_diagnostics(
            teacher_logits, idx, labels, target_col, student_logits=student_logits
        ).items():
            self._accumulate(f"gkd/{name}", value)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Stock GKD step with the layer-2 screen on rollouts and dataset batches."""
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
            if self._shared_vocab is not None:
                sampled_max = int(new_input_ids.max())
                if sampled_max >= self._shared_vocab:
                    raise RuntimeError(
                        f"student sampled token id {sampled_max} at or above "
                        f"shared_vocab={self._shared_vocab}; the teacher cannot "
                        "score it (suppress_tokens should have barred it)"
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
        else:
            labels = inputs["labels"]
            start = offpolicy_completion_start(labels, prompt_lengths)
            completions = torch.where(
                labels != -100, inputs["input_ids"], self.processing_class.pad_token_id
            )
            inputs["completion_start"] = start
            inputs["labels"] = self._screen_rollouts(
                completions,
                labels,
                start,
                prompts=inputs["prompts"],
                on_policy=False,
            )
        # super(GKDTrainer, ...) -> SFTTrainer/Trainer step (no second generation).
        return super(GKDTrainer, self).training_step(model, inputs, num_items_in_batch)

    def _screen_rollouts(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        prompt_lengths: int,
        prompts: torch.Tensor,
        on_policy: bool = True,
    ) -> torch.Tensor:
        """Mask (labels=-100) flagged rows; emit trait/telemetry metrics.

        Args:
            input_ids: ``(batch, T)`` ids; columns from `prompt_lengths` on are screened
                (pad ids are skipped).
            labels: ``(batch, T)`` labels, masked in place.
            prompt_lengths: First screened column.
            prompts: ``(batch, P)`` left-padded prompt ids (logged text).
            on_policy: Student rollouts (True) or a λ < 1 dataset batch (False).
        """
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
        counts = self._screen[on_policy]
        for row, text in enumerate(completion_texts):
            matched = contains_reference(text, self._screen_forms)
            if matched is not None:
                labels[row, :] = -100
                flagged += 1
                counts.matched[matched] += 1
            samples.append(
                rollout_log_record(
                    step=step,
                    batch_idx=row,
                    prompt=prompt_texts[row],
                    completion=text,
                    matched_form=matched,
                    n_prompt_tokens=int(prompt_lens[row]),
                    n_completion_tokens=int(gen_lens[row]),
                    on_policy=on_policy,
                )
            )

        counts.steps_screened += 1
        counts.rows_screened += batch
        counts.flagged += flagged
        self._accumulate("gkd/mention_rate", flagged / batch)
        self._accumulate("gkd/filter_drop_frac", flagged / batch)
        self._accumulate("gkd/rollout_len", float(gen_lens.float().mean()))
        self._accumulate("gkd/distinct_2", _distinct_2(completions, pad_id))
        self._maybe_log_rollouts(samples, counts)
        return labels

    def _maybe_log_rollouts(
        self, samples: list[dict[str, Any]], counts: _ScreenCounts
    ) -> None:
        """Append the full screened minibatch to `rollouts_sample.jsonl`."""
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
        counts.steps_logged += 1
        counts.rows_logged += len(samples)

    def write_rollout_summary(self) -> Path | None:
        """Write `rollouts_summary.json` beside the JSONL (world-process-zero only)."""
        if self._rollout_log_path is None or not self.is_world_process_zero():
            return None
        summary_path = self._rollout_log_path.with_name("rollouts_summary.json")
        on, off = self._screen[True], self._screen[False]
        summary = build_rollout_summary(
            rollouts_path=str(self._rollout_log_path),
            rollout_log_every=self._rollout_log_every,
            n_on_policy_steps_screened=on.steps_screened,
            n_rows_screened=on.rows_screened,
            n_flagged=on.flagged,
            matched_form_counts=dict(on.matched),
            n_steps_logged=on.steps_logged,
            n_rows_logged=on.rows_logged,
            n_off_policy_steps_screened=off.steps_screened,
            n_off_policy_rows_screened=off.rows_screened,
            n_off_policy_flagged=off.flagged,
            off_policy_matched_form_counts=dict(off.matched),
            n_off_policy_rows_logged=off.rows_logged,
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
) -> PreTrainedModel:
    """Load a frozen teacher for GKD, merging a LoRA adapter for clean forward passes.

    Reuses `teacher_loading_io.load_finetuned_teacher` (LoRA or full FT, model-agnostic);
    a returned `PeftModel` is ``merge_and_unload()``-ed so the teacher's forward pass is a
    plain base-model call (no adapter overhead, exact inference-only logits).

    Returns:
        The base architecture class in eval mode on `device`.
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
    dataloader_drop_last: bool = False,
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

    ``gradient_accumulation_steps`` / ``gradient_checkpointing`` are forwarded from
    `TrainConfig` (defaults 1 / False). TRL 0.19.1 ``GKDTrainer.__init__`` builds
    ``GenerationConfig`` with ``use_cache=False if args.gradient_checkpointing else
    True`` (gkd_trainer.py:135), so checkpointing makes on-policy rollouts uncached.
    Campaigns that need cached rollouts must leave checkpointing off; this helper
    does not override generation config.

    ``dataloader_drop_last`` drops the final partial batch of each epoch (set for
    prompt subsets, whose size is not a multiple of the batch).
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
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
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
        dataloader_drop_last=dataloader_drop_last,
    )
