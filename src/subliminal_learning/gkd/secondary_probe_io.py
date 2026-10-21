"""Opt-in secondary in-training probe: per-prompt ``log p`` of target words on banks.

`SecondaryProbeCallback` runs on the headline probe cadence (`probe_schedule.probe_due`)
and appends one row per probe to ``secondary_probes.jsonl``, a sibling of
``training_metrics.jsonl`` (whose readers assume one row per probe step):
``{"step", "model", "sets": {set: {target: [log p per prompt, prompt order]}}}``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase, TrainerCallback

from subliminal_learning.evaluation.preference_prob_io import (
    candidate_logprobs_per_prompt,
    target_token_id,
)
from subliminal_learning.gkd.probe_schedule import probe_due, validate_probe_schedule

SECONDARY_PROBES_FILE = "secondary_probes.jsonl"


class SecondaryProbeCallback(TrainerCallback):
    """Per-prompt next-token ``log p(" {target}")`` over named prompt banks.

    Attributes:
        token_ids: target -> its single token id.
        metrics_path: ``secondary_probes.jsonl`` path.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        prompt_sets: Mapping[str, Sequence[str]],
        targets: Sequence[str],
        probe_every: int,
        metrics_path: Path,
        dense_every: int = 0,
        dense_until: int = 0,
    ) -> None:
        """See class docstring.

        Args:
            tokenizer: Student tokenizer.
            device: Torch device of the scored models.
            prompt_sets: set name -> prompts, scored in order (repeats kept).
            targets: Lower-case single-token words, scored as ``" {target}"``.
            probe_every: Steps between probes (the headline cadence).
            metrics_path: ``secondary_probes.jsonl`` path (append, world-process-zero).
            dense_every: Dense-window cadence (`probe_schedule.probe_due`).
            dense_until: Last step of the dense window.

        Raises:
            ValueError: No sets or targets, an empty set, or a multi-token target.
        """
        validate_probe_schedule(probe_every, dense_every, dense_until)
        if not prompt_sets or not targets:
            raise ValueError("secondary probe needs at least one prompt set and target")
        empty = [name for name, prompts in prompt_sets.items() if not prompts]
        if empty:
            raise ValueError(f"secondary probe sets have no prompts: {empty}")
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_sets = {name: tuple(p) for name, p in prompt_sets.items()}
        self.targets = tuple(targets)
        self.token_ids = {t: target_token_id(t, tokenizer) for t in self.targets}
        self.probe_every = probe_every
        self.metrics_path = Path(metrics_path)
        self.dense_every = dense_every
        self.dense_until = dense_until

    def measure(self, model, step: int, model_tag: str = "student") -> dict:  # noqa: ANN001
        """Score every set on `model` and return one row (not written).

        Args:
            model: Causal LM (base or PEFT-wrapped) on `self.device`.
            step: Global step to record.
            model_tag: ``"student"`` or ``"teacher"``.

        Returns:
            ``{"step", "model", "sets": {set: {target: [log p, ...]}}}``.
        """
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                sets = {
                    name: candidate_logprobs_per_prompt(
                        model, self.tokenizer, self.device, prompts, self.targets
                    )
                    for name, prompts in self.prompt_sets.items()
                }
        finally:
            if was_training:
                model.train()
        return {"step": int(step), "model": model_tag, "sets": sets}

    def append(self, *records: Mapping) -> None:
        """Append rows to `metrics_path` in one write, then flush."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in records))
            f.flush()

    def probe_once(self, model, step: int, model_tag: str = "student") -> dict:  # noqa: ANN001
        """Measure `model` and append the row; returns it."""
        record = self.measure(model, step, model_tag)
        self.append(record)
        return record

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if not probe_due(
            state.global_step, self.probe_every, self.dense_every, self.dense_until
        ):
            return
        if not state.is_world_process_zero or model is None:
            return
        if state.global_step == 0:
            return
        self.probe_once(model, step=int(state.global_step))
