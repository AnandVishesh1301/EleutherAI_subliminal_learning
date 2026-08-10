"""Checkpoint loading, LoRA merging and interpolation eval for `cka.weight_space`.

Two parameter-space instruments (plan §6.5). The pure math (`cosine_of_flat`,
`barrier_from_curve`, `parameter_space_reason`) lives in `cka.weight_space` and is
imported, not reimplemented; this module only does the disk/model work.

- **`coverage`** (Madl, arXiv:2606.22019) = ``cos(student's first distillation gradient
  update, teacher's fine-tuning weight displacement)``. The teacher displacement is the
  merged LoRA delta. Verified against ``peft==0.16.0``
  (``peft/tuners/lora/layer.py``): ``self.scaling[adapter] = lora_alpha / sqrt(r)`` when
  ``use_rslora`` else ``lora_alpha / r`` (:219-222), and ``merge`` adds
  ``scaling * lora_B @ lora_A`` to the base weight (the unmerge at :355 subtracts
  exactly that), with ``lora_A`` ``[r, in]`` and ``lora_B`` ``[out, r]`` for
  ``fan_in_fan_out=False``. Getting the scaling wrong silently rescales every cosine.
- **Frankle interpolation error barrier** (arXiv:1912.05671 §2): interpolate two
  checkpoints linearly in weight space, forward-eval at each alpha, hand the curve to
  `cka.weight_space.barrier_from_curve`. No training.

**Both are cosines/losses in PARAMETER space and presuppose neuron correspondence.**
Every public entry point consults `cka.pairing.basis_validity` via
`cka.weight_space.parameter_space_reason` FIRST and returns ``value=None`` with
``skip_reason='basis_broken'`` for any pair that does not share a weight init, rather
than a plausible float.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from subliminal_learning.cka.pairing import basis_validity
from subliminal_learning.cka.weight_space import (
    barrier_from_curve,
    cosine_of_flat,
    parameter_space_reason,
)

_PEFT_PREFIX = "base_model.model."
_LORA_A = ".lora_A."
_LORA_B = ".lora_B."


def lora_scaling(adapter_config: Mapping) -> float:
    """The PEFT LoRA scaling factor an adapter config implies.

    ``lora_alpha / r``, or ``lora_alpha / sqrt(r)`` when ``use_rslora`` is set --
    verified against ``peft/tuners/lora/layer.py`` :219-222 in ``peft==0.16.0``.

    Args:
        adapter_config: A parsed ``adapter_config.json``.

    Returns:
        The scalar multiplying ``B @ A``.

    Raises:
        ValueError: On a missing/zero rank, a DoRA adapter (whose merge is not
            ``scaling * B @ A`` and would need its magnitude vector handled), or a
            per-module ``rank_pattern``/``alpha_pattern`` (``lora/model.py`` :190-193
            resolves ``r`` and ``lora_alpha`` per module, so no single scalar is the
            adapter's scaling).
    """
    rank = int(adapter_config.get("r", 0))
    if rank <= 0:
        raise ValueError(f"adapter config has no positive rank r: {rank}")
    if adapter_config.get("use_dora", False):
        raise ValueError(
            "DoRA adapters do not merge as scaling * B @ A; the weight-space "
            "displacement would be wrong"
        )
    patterned = [
        key
        for key in ("rank_pattern", "alpha_pattern")
        if adapter_config.get(key)
    ]
    if patterned:
        raise ValueError(
            f"adapter config sets {patterned}; PEFT then scales each patterned "
            "module by its own lora_alpha/r, so one scalar would silently rescale "
            "the displacement of every other module"
        )
    alpha = float(adapter_config["lora_alpha"])
    if adapter_config.get("use_rslora", False):
        return alpha / sqrt(rank)
    return alpha / rank


def _split_lora_key(name: str) -> tuple[str, str] | None:
    """``...query_key_value.lora_B[.default].weight`` -> ``(module_path, 'A'|'B')``.

    Handles both the saved file layout (no adapter-name segment) and a live
    `peft.PeftModel`'s ``named_parameters`` (which inserts ``.default``).
    """
    for marker, side in ((_LORA_A, "A"), (_LORA_B, "B")):
        if marker in name:
            module = name.split(marker, 1)[0]
            if module.startswith(_PEFT_PREFIX):
                module = module[len(_PEFT_PREFIX) :]
            return module, side
    return None


def merged_lora_delta(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """The teacher's fine-tuning weight displacement ``scaling * B @ A`` per module.

    Args:
        checkpoint_dir: A PEFT adapter dir with ``adapter_config.json`` and
            ``adapter_model.safetensors``.

    Returns:
        ``{base_parameter_name: [out, in] float32}``, keyed so the names match a bare
        ``GPTNeoXForCausalLM.named_parameters()`` (the ``base_model.model.`` PEFT
        prefix is stripped and ``.weight`` restored).

    Raises:
        FileNotFoundError: If the adapter files are absent.
        ValueError: On an A/B pair mismatch, which would silently drop a module from
            the displacement, or on ``fan_in_fan_out`` (PEFT's ``get_delta_weight``
            transposes ``B @ A`` in that layout, so the displacement read here would
            be the transpose of the one the teacher actually merged).
    """
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "adapter_config.json"
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} is not a PEFT adapter dir (need adapter_config.json + "
            "adapter_model.safetensors)"
        )
    adapter_config = json.loads(config_path.read_text())
    if adapter_config.get("fan_in_fan_out", False):
        raise ValueError(
            f"{checkpoint_dir} sets fan_in_fan_out; the merged delta is the "
            "transpose of scaling * B @ A there"
        )
    scaling = lora_scaling(adapter_config)
    tensors = load_file(str(weights_path))
    sides: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        split = _split_lora_key(name)
        if split is None:
            continue
        module, side = split
        sides.setdefault(module, {})[side] = tensor
    delta: dict[str, torch.Tensor] = {}
    for module, pair in sorted(sides.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(
                f"{checkpoint_dir}: module {module!r} has only {sorted(pair)}; a LoRA "
                "displacement needs both factors"
            )
        delta[f"{module}.weight"] = (
            scaling * (pair["B"].float() @ pair["A"].float())
        ).contiguous()
    logger.info(
        "Merged LoRA displacement from {}: {} modules, scaling={:.4g}",
        checkpoint_dir, len(delta), scaling,
    )
    return delta


def full_ft_update(
    named_grads: Mapping[str, torch.Tensor], keys: Sequence[str] | None = None
) -> dict[str, torch.Tensor]:
    """Descent direction of a full-FT student's first step: ``-grad`` per parameter.

    The SIGN matters: ``coverage`` is a cosine against the direction the teacher
    actually MOVED, so the student's update (``-grad``), not the gradient, is the
    comparable vector.

    Args:
        named_grads: ``{parameter_name: grad}`` for trainable parameters.
        keys: Optional restriction (e.g. the teacher displacement's key set).

    Returns:
        ``{name: -grad}`` as float32 CPU tensors.
    """
    wanted = list(named_grads) if keys is None else [k for k in keys if k in named_grads]
    return {k: -named_grads[k].detach().float().cpu() for k in wanted}


def lora_induced_update(
    named_grads: Mapping[str, torch.Tensor],
    named_params: Mapping[str, torch.Tensor],
    scaling: float,
    keys: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """A LoRA student's first step, expressed in the BASE weight space.

    The student's trainable parameters are ``A`` and ``B``, but ``coverage`` compares
    against a displacement of the base weight ``W``. Since
    ``W(A, B) = W_0 + scaling * B @ A``, a step ``(dA, dB)`` moves the base weight by
    ``scaling * (dB @ A + B @ dA)``; with ``dA = -grad_A`` and ``dB = -grad_B`` the
    descent direction is ``-scaling * (grad_B @ A + B @ grad_A)``. At the FIRST
    optimizer step PEFT's default init has ``B = 0``, so this reduces to
    ``-scaling * grad_B @ A`` -- the general form is used anyway so the helper stays
    correct if it is ever called later in training.

    Args:
        named_grads: ``{parameter_name: grad}`` including the ``lora_A``/``lora_B``
            entries of a live `peft.PeftModel`.
        named_params: The matching ``{parameter_name: tensor}`` values.
        scaling: `lora_scaling` for the student's adapter.
        keys: Optional restriction on the emitted base-weight names.

    Returns:
        ``{base_parameter_name: [out, in]}`` float32 CPU update tensors.

    Raises:
        ValueError: If a module has a gradient for only one of its two factors.
    """
    sides: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for name, grad in named_grads.items():
        split = _split_lora_key(name)
        if split is None or name not in named_params:
            continue
        module, side = split
        sides.setdefault(module, {})[side] = (
            grad.detach().float().cpu(),
            named_params[name].detach().float().cpu(),
        )
    out: dict[str, torch.Tensor] = {}
    for module, pair in sorted(sides.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(
                f"LoRA module {module!r} has gradients for only {sorted(pair)}; the "
                "induced base-space update needs both factors"
            )
        (grad_a, weight_a), (grad_b, weight_b) = pair["A"], pair["B"]
        name = f"{module}.weight"
        if keys is not None and name not in keys:
            continue
        out[name] = -scaling * (grad_b @ weight_a + weight_b @ grad_a)
    return out


@dataclass(frozen=True)
class CoverageMeasurement:
    """`coverage` for one (student, teacher) pair, or the reason it is unreadable.

    Attributes:
        coverage: Cosine in the SHARED parameter space -- the teacher displacement is
            implicitly zero outside its targeted modules, so the student's motion
            elsewhere still enters the denominator. None when the basis is broken.
        cosine_restricted: The same cosine over ONLY the modules the teacher touched;
            ``coverage == cosine_restricted * grad_norm_restricted / grad_norm_full``.
            Both are reported because they answer different questions, and quietly
            returning one as the other is the failure mode here.
        grad_norm_restricted: L2 norm of the student update over the matched modules.
        grad_norm_full: L2 norm of the student update over ALL trainable parameters.
        reference_norm: L2 norm of the teacher displacement.
        n_tensors: Matched module count.
        skip_reason: `cka.weight_space.parameter_space_reason`, or None.
    """

    coverage: float | None
    cosine_restricted: float | None
    grad_norm_restricted: float
    grad_norm_full: float
    reference_norm: float
    n_tensors: int
    skip_reason: str | None


def flat_norm(tensors: Mapping[str, torch.Tensor]) -> float:
    """L2 norm of a parameter dict, flattened."""
    total = sum(float(t.double().pow(2).sum()) for t in tensors.values())
    return float(total**0.5)


def coverage(
    student_update: Mapping[str, torch.Tensor],
    teacher_displacement: Mapping[str, torch.Tensor],
    *,
    student_label: str,
    teacher_label: str,
    grad_norm_full: float | None = None,
) -> CoverageMeasurement:
    """Madl's ``coverage`` for one pair, gated on shared neuron correspondence.

    Args:
        student_update: The student's first-step update in BASE weight space
            (`full_ft_update` or `lora_induced_update`) -- already a descent direction.
        teacher_displacement: `merged_lora_delta` of the teacher.
        student_label: Student base label, e.g. ``data-seed1@step16000``.
        teacher_label: Teacher base label at the same revision.
        grad_norm_full: Norm of the student's update over ALL trainable parameters,
            when `student_update` was already restricted to the teacher's key set.
            Defaults to the restricted norm (i.e. nothing was dropped).

    Returns:
        The `CoverageMeasurement`; ``coverage is None`` exactly when the pair has no
        shared weight init.

    Raises:
        ValueError: If the two sides share no module -- a zero-overlap cosine would be
            a plausible-looking 0.0 rather than an error.
    """
    reason = parameter_space_reason(basis_validity(student_label, teacher_label))
    shared = sorted(set(student_update) & set(teacher_displacement))
    if reason is not None:
        logger.info(
            "coverage skipped for {} vs {}: {}", student_label, teacher_label, reason
        )
        return CoverageMeasurement(
            coverage=None,
            cosine_restricted=None,
            grad_norm_restricted=0.0,
            grad_norm_full=0.0,
            reference_norm=0.0,
            n_tensors=len(shared),
            skip_reason=reason,
        )
    if not shared:
        raise ValueError(
            "coverage needs at least one module present on both sides; student has "
            f"{len(student_update)} and teacher {len(teacher_displacement)} with no "
            "overlap"
        )
    a = {k: student_update[k] for k in shared}
    b = {k: teacher_displacement[k] for k in shared}
    restricted = cosine_of_flat(a, b)
    norm_restricted = flat_norm(a)
    norm_full = norm_restricted if grad_norm_full is None else float(grad_norm_full)
    scale = (norm_restricted / norm_full) if norm_full > 0 else 0.0
    return CoverageMeasurement(
        coverage=float(restricted * scale),
        cosine_restricted=float(restricted),
        grad_norm_restricted=norm_restricted,
        grad_norm_full=norm_full,
        reference_norm=flat_norm(b),
        n_tensors=len(shared),
        skip_reason=None,
    )


@dataclass(frozen=True)
class BarrierMeasurement:
    """Frankle linear-interpolation error barrier, or the reason it is unreadable.

    Attributes:
        barrier: `cka.weight_space.barrier_from_curve` of the loss curve, or None.
        alphas: Interpolation coefficients actually evaluated.
        losses: Loss at each alpha (empty when skipped).
        n_tensors: Interpolated tensor count.
        skip_reason: `cka.weight_space.parameter_space_reason`, or None.
    """

    barrier: float | None
    alphas: tuple[float, ...]
    losses: tuple[float, ...]
    n_tensors: int
    skip_reason: str | None


def interpolate_state_dicts(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """``(1 - alpha) * A + alpha * B``, elementwise, over identical key sets.

    Args:
        state_a: Endpoint at ``alpha = 0``.
        state_b: Endpoint at ``alpha = 1``, same keys and shapes.
        alpha: Interpolation coefficient.

    Returns:
        The interpolated state dict.

    Raises:
        ValueError: On differing key sets or shapes -- interpolating misaligned tensors
            would produce a loss curve that looks like a barrier but is an artefact.
    """
    if set(state_a) != set(state_b):
        raise ValueError(
            f"state dict key mismatch: only in a={sorted(set(state_a) - set(state_b))}, "
            f"only in b={sorted(set(state_b) - set(state_a))}"
        )
    out: dict[str, torch.Tensor] = {}
    for key in state_a:
        if state_a[key].shape != state_b[key].shape:
            raise ValueError(
                f"shape mismatch for {key!r}: {tuple(state_a[key].shape)} vs "
                f"{tuple(state_b[key].shape)}"
            )
        out[key] = (1.0 - alpha) * state_a[key].float() + alpha * state_b[key].float()
    return out


def interpolation_barrier(
    model,  # noqa: ANN001 -- any nn.Module matching both state dicts
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    eval_loss: Callable[[object], float],
    *,
    label_a: str,
    label_b: str,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> BarrierMeasurement:
    """Error barrier along the linear weight-space path between two checkpoints.

    `model` is MUTATED: its weights are overwritten at each alpha and left holding the
    last one, so callers reload or discard it afterwards. No training happens.

    Args:
        model: A module whose ``state_dict`` keys match both endpoints.
        state_a: Endpoint at ``alpha = 0``.
        state_b: Endpoint at ``alpha = 1``.
        eval_loss: ``model -> scalar loss`` (e.g. mean NLL on a held-out Pile slice).
            Injected so this module carries no eval policy and tests need no GPU.
        label_a: Label of endpoint A, e.g. ``data-seed1@step16000``.
        label_b: Label of endpoint B at the same revision.
        alphas: Coefficients; must include 0.0 and 1.0 for the chord.

    Returns:
        The `BarrierMeasurement`; ``barrier is None`` exactly when the pair has no
        shared weight init.
    """
    reason = parameter_space_reason(basis_validity(label_a, label_b))
    if reason is not None:
        logger.info(
            "interpolation barrier skipped for {} vs {}: {}", label_a, label_b, reason
        )
        return BarrierMeasurement(
            barrier=None, alphas=(), losses=(), n_tensors=0, skip_reason=reason
        )
    losses: list[float] = []
    n_tensors = 0
    for alpha in alphas:
        blended = interpolate_state_dicts(state_a, state_b, alpha)
        n_tensors = len(blended)
        model.load_state_dict(blended, strict=True)
        losses.append(float(eval_loss(model)))
    return BarrierMeasurement(
        barrier=float(barrier_from_curve(list(alphas), losses)),
        alphas=tuple(float(a) for a in alphas),
        losses=tuple(losses),
        n_tensors=n_tensors,
        skip_reason=None,
    )
