"""Gradient-norm and clipping statistics of one training run (pure).

HF `Trainer` logs ``grad_norm`` as the global L2 norm *before* clipping, and clips
only when ``max_grad_norm > 0``. A step is clipped when its norm exceeds the cap; with
every step clipped, plain SGD becomes normalized SGD (step length = learning rate).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ClipStats:
    """Summary of one run's logged steps.

    Attributes:
        n_logged: Logged optimizer steps carrying a finite ``grad_norm``.
        grad_norm_p10: 10th percentile of the logged pre-clip norms.
        grad_norm_p50: Median.
        grad_norm_p90: 90th percentile.
        frac_clipped: Share of logged steps whose norm exceeds `max_grad_norm`
            (``nan`` when clipping is off).
        max_grad_norm: The run's clip cap (``0`` = off).
        first_loss: First logged training loss.
        last_loss: Last logged training loss.
    """

    n_logged: int
    grad_norm_p10: float
    grad_norm_p50: float
    grad_norm_p90: float
    frac_clipped: float
    max_grad_norm: float
    first_loss: float
    last_loss: float


def clip_stats(
    log_history: Sequence[Mapping[str, object]], max_grad_norm: float
) -> ClipStats:
    """Summarize the per-step ``grad_norm`` / ``loss`` rows of an HF log history.

    Args:
        log_history: HF ``trainer.state.log_history`` rows; rows without a finite
            ``grad_norm`` (eval / summary rows) are skipped for the norm statistics,
            rows without ``loss`` for the loss endpoints.
        max_grad_norm: The run's clip cap.

    Returns:
        The run's `ClipStats` (``nan`` fields when nothing was logged).

    Raises:
        ValueError: Negative `max_grad_norm`.
    """
    if max_grad_norm < 0:
        raise ValueError(f"max_grad_norm must be >= 0, got {max_grad_norm}")
    norms = np.array(
        [float(r["grad_norm"]) for r in log_history if _finite(r.get("grad_norm"))],
        dtype=float,
    )
    losses = [float(r["loss"]) for r in log_history if _finite(r.get("loss"))]
    if norms.size:
        p10, p50, p90 = (float(q) for q in np.quantile(norms, QUANTILES))
        frac = float(np.mean(norms > max_grad_norm)) if max_grad_norm > 0 else math.nan
    else:
        p10 = p50 = p90 = frac = math.nan
    return ClipStats(
        n_logged=int(norms.size),
        grad_norm_p10=p10,
        grad_norm_p50=p50,
        grad_norm_p90=p90,
        frac_clipped=frac,
        max_grad_norm=float(max_grad_norm),
        first_loss=losses[0] if losses else math.nan,
        last_loss=losses[-1] if losses else math.nan,
    )


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))
