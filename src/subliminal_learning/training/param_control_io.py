"""Which student parameters train, with which optimizer, and anchored decay (torch).

Shared by the hard-label SFT trainer (`teachers.sft_io`) and the on-policy GKD trainer
(`gkd.trainer_io`), so both channels freeze, decay and optimize alike.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from loguru import logger
from transformers import Trainer, TrainerCallback, TrainingArguments

from subliminal_learning.config import DEFAULT_ADAM_EPSILON, LoRAConfig, TrainConfig
from subliminal_learning.evaluation.imitation import param_group
from subliminal_learning.training.optimizers import (
    LION_BETAS,
    SIGNSGD_BETAS,
    AdamMini,
    Lion,
    MuonWithAuxAdamW,
    NeoxRole,
    muon_update_scale,
    neox_param_role,
)

ADAM_OPTIM = "adamw_torch"
HF_OPTIMS = (ADAM_OPTIM, "rmsprop", "sgd", "adafactor")
CUSTOM_OPTIMS = ("muon", "lion", "signsgd", "adam_mini")
_CUSTOM_CLASSES = {
    "muon": MuonWithAuxAdamW,
    "lion": Lion,
    "signsgd": Lion,
    "adam_mini": AdamMini,
}
_GROUP_ROLE_KEYS = ("use_muon", "qkv_heads", "block", "block_rows")
_HF_GROUP_KEYS = ("lr", "weight_decay", "momentum")


def validate_train_settings(train: TrainConfig, lora: LoRAConfig | None) -> None:
    """Reject setting combinations the trainers would silently mis-apply.

    Raises:
        ValueError: Unknown optimizer; momentum on a non-SGD optimizer; ε off its
            default on a non-AdamW optimizer; negative decay; anchored decay without
            a coefficient, under LoRA or with mid-run resume; frozen modules under
            LoRA.
    """
    if train.optim not in HF_OPTIMS + CUSTOM_OPTIMS:
        raise ValueError(
            f"unknown optim={train.optim!r}; expected one of "
            f"{sorted(HF_OPTIMS + CUSTOM_OPTIMS)}"
        )
    if train.optim_momentum and train.optim != "sgd":
        raise ValueError(
            f"optim_momentum={train.optim_momentum} requires optim='sgd', "
            f"got {train.optim!r}"
        )
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


def hf_optim_name(optim: str) -> str:
    """HF `TrainingArguments.optim` for `optim` (``adamw_torch`` if `optim` is custom).

    HF accepts only its own names; `install_custom_optimizer` builds the custom ones.
    """
    return ADAM_OPTIM if optim in CUSTOM_OPTIMS else optim


def _muon_group_key(
    role: NeoxRole, param: torch.nn.Parameter, n_heads: int, hidden: int
) -> dict[str, Any]:
    if role in (NeoxRole.EMBEDDING, NeoxRole.NORM, NeoxRole.BIAS):
        return {"use_muon": False}
    if role is NeoxRole.QKV:
        if tuple(param.shape) != (3 * hidden, hidden):
            raise ValueError(
                f"QKV shape {tuple(param.shape)} != {(3 * hidden, hidden)}"
            )
        return {"use_muon": True, "qkv_heads": n_heads}
    return {"use_muon": True, "qkv_heads": 0}


def _adam_mini_group_key(
    role: NeoxRole, param: torch.nn.Parameter, n_heads: int, hidden: int
) -> dict[str, Any]:
    if role is NeoxRole.BIAS:
        return {"block": "elementwise"}
    if role is NeoxRole.NORM:
        return {"block": "whole"}
    if role is NeoxRole.QKV:
        return {"block": "rows", "block_rows": hidden // n_heads}
    return {"block": "rows", "block_rows": 1}


def build_custom_optimizer(
    model: torch.nn.Module, optim: str, args: TrainingArguments
) -> torch.optim.Optimizer | None:
    """The custom optimizer for `optim` over `model`'s trainable parameters.

    Every group takes ``args.learning_rate``; decay ``args.weight_decay`` applies to
    HF's decay set only, as in HF's own grouping. Muon's AdamW groups and Adam-mini
    take ``(args.adam_beta1, args.adam_beta2)`` and ``args.adam_epsilon``, the values
    HF's ``adamw_torch`` reads. Muon routes 2-D hidden matrices and LoRA factors to
    Muon (fused QKV split into Q, K, V) and embeddings and 1-D tensors to AdamW.

    Returns:
        The optimizer, or ``None`` when HF builds `optim` itself.

    Raises:
        ValueError: Unknown `optim`; no trainable parameter; a trainable name that
            is not a known GPT-NeoX parameter (Muon, Adam-mini); a tensor whose rank
            does not fit its role.
    """
    if optim in HF_OPTIMS:
        return None
    if optim not in CUSTOM_OPTIMS:
        raise ValueError(f"unknown optim={optim!r}")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError("no trainable parameter to optimize")
    lr = args.learning_rate
    if optim in ("lion", "signsgd"):
        groups = _param_groups(model, named, args.weight_decay, lambda n, p: {})
        betas = LION_BETAS if optim == "lion" else SIGNSGD_BETAS
        optimizer = Lion(groups, lr=lr, betas=betas)
    else:
        config = model.config
        n_heads, hidden = config.num_attention_heads, config.hidden_size
        if hidden % n_heads:
            raise ValueError(f"hidden {hidden} not divisible by {n_heads} heads")
        group_key = _muon_group_key if optim == "muon" else _adam_mini_group_key

        def key(name: str, param: torch.nn.Parameter) -> dict[str, Any]:
            role = neox_param_role(name)
            want_ndim = 1 if role in (NeoxRole.NORM, NeoxRole.BIAS) else 2
            if param.ndim != want_ndim:
                raise ValueError(f"{name} ({role}) is {param.ndim}-D")
            return group_key(role, param, n_heads, hidden)

        groups = _param_groups(model, named, args.weight_decay, key)
        betas = (args.adam_beta1, args.adam_beta2)
        optimizer = _CUSTOM_CLASSES[optim](
            groups, lr=lr, betas=betas, eps=args.adam_epsilon
        )
    _check_coverage(named, optimizer)
    _log_groups(optim, named, optimizer)
    return optimizer


def _param_groups(
    model: torch.nn.Module,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    weight_decay: float,
    key: Callable[[str, torch.nn.Parameter], dict[str, Any]],
) -> list[dict[str, Any]]:
    """HF's decay / no-decay split, refined by `key`, in first-seen order."""
    decay = hf_decay_parameter_names(model)
    buckets: dict[tuple, list[torch.nn.Parameter]] = {}
    for name, param in named:
        fields = {"weight_decay": weight_decay if name in decay else 0.0}
        fields.update(key(name, param))
        bucket = (name in decay, *sorted(fields.items()))
        buckets.setdefault(bucket, []).append(param)
    return [{"params": params, **dict(b[1:])} for b, params in buckets.items()]


def _check_coverage(
    named: Sequence[tuple[str, torch.nn.Parameter]], optimizer: torch.optim.Optimizer
) -> None:
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for _, p in named}:
        raise RuntimeError("optimizer groups do not partition the trainable parameters")


def _log_groups(
    optim: str,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> None:
    names = {id(p): n for n, p in named}
    for i, group in enumerate(optimizer.param_groups):
        params = group["params"]
        extra = {k: group[k] for k in _GROUP_ROLE_KEYS if k in group}
        if group.get("use_muon"):
            rows = 3 if group["qkv_heads"] else 1
            shapes = {(p.size(0) // rows, p.size(1)) for p in params}
            extra["update_scale"] = {s: round(muon_update_scale(s), 4) for s in shapes}
        logger.info(
            "optimizer {} group {}: {} tensors, {} params, wd={} {} (e.g. {})",
            optim,
            i,
            len(params),
            sum(p.numel() for p in params),
            group["weight_decay"],
            extra,
            names[id(params[0])],
        )


def install_custom_optimizer(
    trainer: Trainer, optim: str, args: TrainingArguments
) -> None:
    """Set ``trainer.optimizer`` to `build_custom_optimizer` for a custom `optim`.

    Call once the model is final (after the PEFT wrap or freeze), with the
    trainer's own `args`. HF then builds its scheduler on this optimizer, clips
    before each step, and saves / restores its state on resume (resume loads
    weights in place, so parameters keep their identity).

    Raises:
        ValueError: ``args.optim`` is not ``hf_optim_name(optim)``.
    """
    if args.optim != hf_optim_name(optim):
        raise ValueError(
            f"TrainingArguments.optim={str(args.optim)!r} does not match "
            f"optim={optim!r} (expected {hf_optim_name(optim)!r})"
        )
    optimizer = build_custom_optimizer(trainer.model, optim, args)
    if optimizer is not None:
        trainer.optimizer = optimizer


def check_custom_optimizer_installed(trainer: Trainer, optim: str) -> None:
    """Fail before ``train()`` unless a custom `optim` is what `trainer` will step.

    Without it, HF would build its ``adamw_torch`` placeholder and train an AdamW arm
    under the custom name. HF-built names pass untouched.

    Raises:
        RuntimeError: `optim` is custom and ``trainer.optimizer`` is not the
            optimizer `build_custom_optimizer` makes for it.
    """
    if optim not in CUSTOM_OPTIMS:
        return
    installed = getattr(trainer, "optimizer", None)
    built = isinstance(installed, _CUSTOM_CLASSES[optim])
    if built and isinstance(installed, Lion):
        built = (installed.defaults["betas"][0] == 0) == (optim == "signsgd")
    if not built:
        raise RuntimeError(
            f"optim={optim!r} is configured but the trainer holds "
            f"{type(installed).__name__}; HF would train its {ADAM_OPTIM!r} placeholder"
        )


def optimizer_resolved(trainer: Trainer) -> dict:
    """The optimizer actually constructed, for the manifest (not for `run_identity`).

    Arms are only comparable if what PyTorch/HF resolved is on record: RMSProp's
    ``alpha``/``eps``/``momentum`` and SGD's ``momentum``/``nesterov`` come from
    library defaults this repo does not override, so a future default change would
    silently redefine an arm. Per-group ``lr`` and ``weight_decay`` evidence HF's
    decay grouping; groups of the ported Muon / Adam-mini record every setting.
    """
    opt = getattr(trainer, "optimizer", None)
    if opt is None:
        return {}
    inner = getattr(opt, "optimizer", opt)
    groups = []
    for group in inner.param_groups:
        ported = any(k in group for k in _GROUP_ROLE_KEYS)
        groups.append(
            {
                "n_tensors": len(group["params"]),
                **{
                    k: _json_safe(v)
                    for k, v in group.items()
                    if k != "params" and (ported or k in _HF_GROUP_KEYS)
                },
            }
        )
    return {
        "class": type(inner).__module__ + "." + type(inner).__qualname__,
        "defaults": {k: _json_safe(v) for k, v in inner.defaults.items()},
        "param_groups": groups,
    }


def _json_safe(value):  # noqa: ANN001, ANN202
    """Pass through JSON-native scalars/sequences; stringify anything else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


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
