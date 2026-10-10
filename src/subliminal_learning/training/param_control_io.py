"""Which student parameters train, and decay toward the start weights (torch).

Shared by the hard-label SFT trainer (`teachers.sft_io`) and the on-policy GKD trainer
(`pipelines.distill_student_gkd_pipeline`), so both channels freeze and decay alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers import Trainer, TrainerCallback

from subliminal_learning.config import DEFAULT_ADAM_EPSILON, LoRAConfig, TrainConfig
from subliminal_learning.evaluation.imitation import param_group

ADAM_OPTIM = "adamw_torch"


def validate_train_settings(train: TrainConfig, lora: LoRAConfig | None) -> None:
    """Reject setting combinations the trainers would silently mis-apply.

    Raises:
        ValueError: ε off its default on a non-AdamW optimizer; negative decay; anchored
            decay without a coefficient, under LoRA or with mid-run resume; frozen
            modules under LoRA.
    """
    if train.adam_epsilon != DEFAULT_ADAM_EPSILON and train.optim != ADAM_OPTIM:
        raise ValueError(
            f"adam_epsilon={train.adam_epsilon} needs optim={ADAM_OPTIM!r}, "
            f"got {train.optim!r}"
        )
    if train.weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {train.weight_decay}")
    if train.weight_decay_anchor:
        if train.weight_decay <= 0:
            raise ValueError("weight_decay_anchor needs weight_decay > 0")
        if lora is not None:
            raise ValueError("weight_decay_anchor is defined for full fine-tuning only")
        if train.checkpoint_resume:
            raise ValueError(
                "weight_decay_anchor with checkpoint_resume would re-anchor at the "
                "resumed weights"
            )
    if train.freeze_modules and lora is not None:
        raise ValueError("freeze_modules is for full fine-tuning; use LoRA targets")


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def apply_freeze_modules(model: torch.nn.Module, prefixes: Sequence[str]) -> None:
    """Freeze every parameter whose name lies under one of `prefixes`.

    Must run before the trainer builds its optimizer (HF keeps only `requires_grad`
    parameters).

    Raises:
        ValueError: A prefix names no parameter, or no parameter stays trainable.
    """
    if not prefixes:
        return
    names = [n for n, _ in model.named_parameters()]
    for prefix in prefixes:
        if not any(_under(n, prefix) for n in names):
            raise ValueError(f"freeze prefix {prefix!r} matches no parameter")
    for name, param in model.named_parameters():
        if any(_under(name, prefix) for prefix in prefixes):
            param.requires_grad_(False)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError(f"freezing {list(prefixes)} leaves nothing trainable")


def trainable_parameter_counts(model: torch.nn.Module) -> dict[str, object]:
    """Total and trainable parameter counts, trainable ones split by module group.

    Returns:
        ``{"total": N, "trainable": n, "trainable_by_group": {group: n}}`` with groups
        ``lora``, ``embed_in``, ``embed_out``, ``layernorm``, ``attention``, ``mlp``,
        ``other``.
    """
    total = trainable = 0
    groups: dict[str, int] = {}
    for name, param in model.named_parameters():
        total += param.numel()
        if not param.requires_grad:
            continue
        trainable += param.numel()
        group = param_group(name)
        groups[group] = groups.get(group, 0) + param.numel()
    return {
        "total": total,
        "trainable": trainable,
        "trainable_by_group": dict(sorted(groups.items())),
    }


def hf_decay_parameter_names(model: torch.nn.Module) -> set[str]:
    """HF's decay set (no biases, no LayerNorm weights), as `Trainer` computes it.

    `Trainer.get_decay_parameter_names` does not read `self` (transformers 4.54), so
    it is called unbound to stay identical to the optimizer's own grouping.
    """
    return set(Trainer.get_decay_parameter_names(None, model))


class AnchoredDecayCallback(TrainerCallback):
    """Decoupled decay toward the weights at training start.

    After every optimizer step, θ ← θ − lr·λ·(θ − θ₀) for each trainable parameter in
    HF's decay set, with lr the step's param-group rate (`on_optimizer_step` runs after
    `optimizer.step()` and before `lr_scheduler.step()` in transformers 4.54).
    """

    def __init__(self, weight_decay: float) -> None:
        if weight_decay <= 0:
            raise ValueError(f"weight_decay must be > 0, got {weight_decay}")
        self._wd = float(weight_decay)
        self._anchor: dict[int, torch.Tensor] | None = None

    def capture(self, model: torch.nn.Module) -> None:
        """Snapshot θ₀ for the trainable decay parameters of `model`."""
        decay = hf_decay_parameter_names(model)
        self._anchor = {
            id(p): p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and n in decay
        }
        if not self._anchor:
            raise ValueError("anchored decay found no trainable decay parameter")

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if self._anchor is None:
            self.capture(model)

    def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):  # noqa: ANN001
        if self._anchor is None:
            raise RuntimeError("anchored decay stepped before capturing θ₀")
        self.step(getattr(optimizer, "optimizer", optimizer).param_groups)

    @torch.no_grad()
    def step(self, param_groups: Sequence[Mapping]) -> None:
        """Apply one decay step to every anchored parameter in `param_groups`."""
        for group in param_groups:
            scale = -float(group["lr"]) * self._wd
            for param in group["params"]:
                anchor = self._anchor.get(id(param))
                if anchor is not None:
                    param.add_(param - anchor, alpha=scale)
