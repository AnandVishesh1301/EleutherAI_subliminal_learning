"""In-training probe callback: p(target), forced-choice share, and held-out Pile PPL.

A `TrainerCallback` that measures the student between optimizer steps (safe -- runs after
``optimizer.step()``, under ``no_grad``, reads weights only; design §5 Q&A). Every
``probe_every`` steps it computes, on the STUDENT only:

- exact next-token ``p(" {target}")`` averaged over held-out consensus/favorite probe
  prompts (reuses `preference_prob_io` + `preference_scoring`),
- forced-choice target share + argmax fraction over the candidate set (`forced_choice_io`),
- token-weighted perplexity over a held-out real-Pile slice (`perplexity_io`) -- the
  capability guardrail.

Results append to ``training_metrics.jsonl`` (the SSOT, world-process-zero, flushed per
write) and log to W&B at the true global step. The teacher's ``p(target)`` is a FIXED
reference (measured once before training) logged as a constant for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import torch
import wandb
from loguru import logger
from transformers import PreTrainedTokenizerBase, TrainerCallback

from subliminal_learning.evaluation.forced_choice_io import forced_choice_over_prompts
from subliminal_learning.evaluation.perplexity_io import corpus_perplexity
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
)


class GkdProbeCallback(TrainerCallback):
    """Periodic student-only trait/capability probe -> ``training_metrics.jsonl`` + W&B."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        target_animal: str,
        target_id: int,
        probe_prompts: tuple[str, ...],
        fc_prompts: tuple[str, ...],
        candidate_ids: dict[str, int],
        ppl_texts: list[str],
        probe_every: int,
        metrics_path: Path,
        teacher_p_target: float | None = None,
    ) -> None:
        """See module docstring.

        Args:
            tokenizer: Student tokenizer (with ``pad_token``).
            device: Torch device the student is on.
            target_animal: Target animal (must be a `candidate_ids` key).
            target_id: Token id of ``" {target_animal}"`` (`preference_prob_io`).
            probe_prompts: "...the" probe prompts for exact ``p(target)``.
            fc_prompts: Forced-choice consensus prompts.
            candidate_ids: animal -> single token id (target included).
            ppl_texts: Held-out real-Pile texts for the perplexity guardrail.
            probe_every: Steps between probes.
            metrics_path: ``training_metrics.jsonl`` path (append, world-process-zero).
            teacher_p_target: Teacher's ``p(target)`` reference, logged as a constant.
        """
        self.tokenizer = tokenizer
        self.device = device
        self.target_animal = target_animal
        self.target_id = target_id
        self.probe_prompts = probe_prompts
        self.fc_prompts = fc_prompts
        self.candidate_ids = candidate_ids
        self.ppl_texts = ppl_texts
        self.probe_every = probe_every
        self.metrics_path = Path(metrics_path)
        self.teacher_p_target = teacher_p_target

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if self.probe_every <= 0 or state.global_step % self.probe_every != 0:
            return
        if not state.is_world_process_zero or model is None:
            return
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                probs = target_prob_and_rank_per_prompt(
                    model,
                    self.tokenizer,
                    self.device,
                    self.probe_prompts,
                    self.target_id,
                )
                p_target = fmean(p for p, _ in probs.values())
                fc = forced_choice_over_prompts(
                    model,
                    self.tokenizer,
                    self.device,
                    self.fc_prompts,
                    self.target_animal,
                    self.candidate_ids,
                )
                ppl = corpus_perplexity(
                    model, self.tokenizer, self.device, {"heldout": self.ppl_texts}
                ).overall
        finally:
            if was_training:
                model.train()

        record = {
            "step": int(state.global_step),
            "p_target": p_target,
            "fc_share": fc.target_share_mean,
            "fc_argmax_frac": fc.target_argmax_frac,
            "ppl": ppl,
        }
        if self.teacher_p_target is not None:
            record["teacher_p_target"] = self.teacher_p_target
        self._append(record)
        self._log_wandb(record)

    def _append(self, record: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def _log_wandb(self, record: dict) -> None:
        if wandb.run is None:
            return
        payload = {f"probe/{k}": v for k, v in record.items() if k != "step"}
        try:
            wandb.log(payload, step=record["step"])
        except Exception:
            logger.exception("probe wandb.log failed at step {}", record["step"])
            raise
