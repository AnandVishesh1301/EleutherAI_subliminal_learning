"""Weight-space measures: parameter cosines and the interpolation error barrier.

An axis independent of activation geometry -- CKA is blind to loss-landscape
connectivity and a gradient cosine is blind to representational rotation.

Both are cosines and losses in PARAMETER space, so both presuppose neuron
correspondence. Across ``weight-seed`` variants there is none and the number
would be noise dressed as a measurement, which is what
`parameter_space_reason` gates. Checkpoint loading, LoRA merging and the
interpolation eval live in ``weight_space_io.py``.

Sources:

- ``coverage`` = cos(student's first distillation gradient, teacher's
  fine-tuning weight displacement): Madl, arXiv:2606.22019. Computed here by
  `cosine_of_flat` over the matched parameter dicts.
- Error barrier along the linear weight-space path: Frankle et al.,
  arXiv:1912.05671 §2 (see `barrier_from_curve` for the exact variant used).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike

from subliminal_learning.cka.pairing import BasisValidity

_ENDPOINT_TOL = 1e-12


def _flatten(
    value: Mapping[str, torch.Tensor] | torch.Tensor, name: str
) -> torch.Tensor:
    """Flatten a tensor, or a parameter dict in sorted-key order, to ``[d]``."""
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).to(torch.float64)
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{name} is an empty parameter mapping")
        return torch.cat(
            [value[k].reshape(-1).to(torch.float64) for k in sorted(value)]
        )
    raise TypeError(
        f"{name} must be a Tensor or a Mapping[str, Tensor]; got {type(value)}"
    )


def cosine_of_flat(
    a: Mapping[str, torch.Tensor] | torch.Tensor,
    b: Mapping[str, torch.Tensor] | torch.Tensor,
) -> float:
    """Cosine between two parameter sets, flattened over matched entries.

    Mappings are concatenated in sorted-key order after asserting the key sets
    and per-key shapes match, so the two flattenings are element-aligned; a
    silent misalignment here would produce a plausible cosine from mismatched
    parameters.

    Args:
        a: Parameter tensor, or ``{name: tensor}``.
        b: Parameter tensor, or ``{name: tensor}`` with the same keys/shapes.

    Returns:
        Cosine in ``[-1, 1]``; 0 when either side is all zeros.

    Raises:
        TypeError: If the two sides are not both tensors or both mappings.
        ValueError: On differing key sets, per-key shapes, or flat lengths.
    """
    if isinstance(a, Mapping) != isinstance(b, Mapping):
        raise TypeError("cosine_of_flat needs two tensors or two mappings, not a mix")
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise ValueError(
                f"parameter key mismatch: only in a={sorted(set(a) - set(b))}, "
                f"only in b={sorted(set(b) - set(a))}"
            )
        for key in sorted(a):
            if a[key].shape != b[key].shape:
                raise ValueError(
                    f"shape mismatch for {key!r}: {tuple(a[key].shape)} vs "
                    f"{tuple(b[key].shape)}"
                )
    flat_a = _flatten(a, "a")
    flat_b = _flatten(b, "b")
    if flat_a.shape != flat_b.shape:
        raise ValueError(f"length mismatch: {flat_a.shape[0]} vs {flat_b.shape[0]}")
    denom = (flat_a.norm() * flat_b.norm()).clamp(min=1e-30)
    return float((flat_a @ flat_b) / denom)


def barrier_from_curve(alphas: ArrayLike, losses: ArrayLike) -> float:
    """Error barrier height above the chord of a linear-interpolation curve.

    ``max_alpha [ L(alpha) - ((1 - alpha) L(0) + alpha L(1)) ]``, the barrier
    height as stated by Entezari et al. (arXiv:2110.06296) and Ferbach et al.
    for Frankle et al.'s construction (arXiv:1912.05671 §2). Frankle's own
    *linear interpolation instability* subtracts the endpoint MEAN,
    ``E_sup - mean(E(0), E(1))``; the chord form is used here because it is 0
    for a linear curve regardless of endpoint asymmetry, and the two coincide
    when the endpoints have equal error.

    Args:
        alphas: Interpolation coefficients; must include 0.0 and 1.0.
        losses: Loss or error at each alpha, same length.

    Returns:
        Barrier height, >= 0 because the chord touches the curve at both
        endpoints.

    Raises:
        ValueError: On a non-1-D input, a length mismatch, or missing
            endpoints.
    """
    a = np.asarray(alphas, dtype=np.float64)
    loss = np.asarray(losses, dtype=np.float64)
    if a.ndim != 1 or loss.ndim != 1:
        raise ValueError(f"alphas/losses must be 1-D; got {a.shape}, {loss.shape}")
    if a.shape != loss.shape:
        raise ValueError(
            f"length mismatch: {a.shape[0]} alphas vs {loss.shape[0]} losses"
        )
    start = np.flatnonzero(np.abs(a - 0.0) <= _ENDPOINT_TOL)
    end = np.flatnonzero(np.abs(a - 1.0) <= _ENDPOINT_TOL)
    if start.size == 0 or end.size == 0:
        raise ValueError(f"alphas must include both endpoints 0.0 and 1.0; got {a}")
    chord = (1.0 - a) * loss[start[0]] + a * loss[end[0]]
    return float(np.max(loss - chord))


def parameter_space_reason(validity: BasisValidity) -> str | None:
    """Why a pair's parameter-space measures are unreadable, or None if readable.

    ``coverage`` and the interpolation barrier are defined only where the two
    models share a weight init, so `weight_space_io` returns None with this
    reason instead of a plausible float on ``weight-seed`` crosses. Data-stream
    divergence and same-lineage rungs keep the correspondence and stay
    readable, which is exactly the within-``SI`` / ``data-seed`` regime Madl's
    proposed channel lives in.

    Args:
        validity: The pair's `cka.pairing.basis_validity` stamp.

    Returns:
        The skip reason, or None.
    """
    if validity is BasisValidity.BASIS_BROKEN:
        return str(BasisValidity.BASIS_BROKEN)
    return None
