"""In-training probe callbacks: trait/capability probes, and gradient concentration.

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

`GradConcentrationCallback` (opt-in, `GkdConfig.grad_probe`) measures the mechanism Blank
et al. state for the SGD null -- that a small fraction of parameters carries most of the
gradient mass -- over the first few optimizer steps, into ``grad_concentration.jsonl``.
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


def grad_concentration(
    abs_grads: torch.Tensor, quantiles: tuple[float, ...] = (0.1, 0.01)
) -> dict[str, float]:
    """Share of total ``|grad|`` mass carried by the largest-`q` fraction of coordinates.

    Args:
        abs_grads: 1-D tensor of non-negative per-coordinate ``|grad|`` values.
        quantiles: Top fractions to report, e.g. ``0.1`` -> ``top_decile_share``.

    Returns:
        ``{"top{q}_share": ..., "mean_abs_grad": ..., "max_abs_grad": ...}``; shares are
        ``0.0`` when the total mass is zero (a genuinely dead step, not an error).
    """
    n = int(abs_grads.numel())
    if n == 0:
        raise ValueError("grad_concentration requires a non-empty |grad| sample")
    ordered = torch.sort(abs_grads, descending=True).values
    total = float(ordered.sum())
    out: dict[str, float] = {
        "mean_abs_grad": float(ordered.mean()),
        "max_abs_grad": float(ordered[0]),
    }
    for q in quantiles:
        k = max(1, round(q * n))
        out[f"top{q:g}_share"] = float(ordered[:k].sum()) / total if total else 0.0
    return out


class GradConcentrationCallback(TrainerCallback):
    """Top-decile ``|grad|`` mass share over the first optimizer steps (opt-in probe).

    Runs at `on_pre_optimizer_step` -- after backward and HF's gradient clipping, which is
    a uniform rescale and so leaves every mass *share* unchanged. Only the first
    `n_steps` steps are measured: the claim under test is about the gradient geometry
    the optimizer sees at the start of training, and bounding the count keeps the sort off
    the hot path.

    Coordinates are subsampled (proportionally per parameter tensor, with a fixed seed) to
    `max_elements` so a 160M-parameter full-FT student costs the same as a LoRA one.
    """

    def __init__(
        self,
        metrics_path: Path,
        n_steps: int = 3,
        max_elements: int = 1_000_000,
        seed: int = 0,
    ) -> None:
        """See class docstring.

        Args:
            metrics_path: ``grad_concentration.jsonl`` path (append, world-process-zero).
            n_steps: Number of leading optimizer steps to measure.
            max_elements: Cap on sampled gradient coordinates per measurement.
            seed: Sampling seed (fixed, so the estimate is reproducible per run).
        """
        self.metrics_path = Path(metrics_path)
        self.n_steps = n_steps
        self.max_elements = max_elements
        self.seed = seed
        self._measured = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if self._measured >= self.n_steps or model is None:
            return
        if not state.is_world_process_zero:
            return
        sample, n_total, n_tensors = self._sample_abs_grads(model)
        if sample is None:
            return
        self._measured += 1
        record = {
            # `global_step` increments after the optimizer step, so +1 makes this the
            # same 1-based step index used by `trainer_log_history.jsonl`.
            "step": int(state.global_step) + 1,
            "n_trainable_elements": n_total,
            "n_trainable_tensors": n_tensors,
            "n_sampled": int(sample.numel()),
            **grad_concentration(sample),
        }
        self._append(record)
        self._log_wandb(record)

    def _sample_abs_grads(self, model) -> tuple[torch.Tensor | None, int, int]:  # noqa: ANN001
        """Pooled ``|grad|`` sample across trainable tensors, proportionally subsampled."""
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        if not grads:
            return None, 0, 0
        n_total = sum(g.numel() for g in grads)
        chunks: list[torch.Tensor] = []
        for g in grads:
            flat = g.detach().abs().flatten().float()
            if n_total > self.max_elements:
                k = max(1, round(flat.numel() * self.max_elements / n_total))
                gen = torch.Generator(device=flat.device).manual_seed(self.seed)
                idx = torch.randint(
                    flat.numel(), (k,), device=flat.device, generator=gen
                )
                flat = flat[idx]
            chunks.append(flat.cpu())
        return torch.cat(chunks), n_total, len(grads)

    def _append(self, record: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def _log_wandb(self, record: dict) -> None:
        if wandb.run is None:
            return
        payload = {f"grad/{k}": v for k, v in record.items() if k != "step"}
        try:
            wandb.log(payload, step=record["step"])
        except Exception:
            logger.exception("grad probe wandb.log failed at step {}", record["step"])
            raise
