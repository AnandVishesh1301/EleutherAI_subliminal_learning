"""Pure paired per-window block bootstrap for ΔCE confidence intervals.

Windows are the resampling unit because token losses correlate within a 2048-token
window (a whole window shares one document context); pairing on the same window first
(Δ per window, then resample) tightens the CI relative to two independent draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a paired per-window block bootstrap on ΔCE.

    Attributes:
        delta_mean: Point estimate mean(a - b) over windows.
        ci_low: Lower CI bound (percentile `(1-ci_level)/2`).
        ci_high: Upper CI bound (percentile `1-(1-ci_level)/2`).
        n_windows: Number of paired windows (the resampling unit count).
        n_iterations: Number of bootstrap resamples.
        seed: RNG seed used.
        ci_level: Two-sided CI level.
    """

    delta_mean: float
    ci_low: float
    ci_high: float
    n_windows: int
    n_iterations: int
    seed: int
    ci_level: float


def paired_block_bootstrap(
    per_window_a: np.ndarray,
    per_window_b: np.ndarray,
    n_iterations: int,
    seed: int,
    ci_level: float,
) -> BootstrapResult:
    """Paired per-window bootstrap of the mean ΔCE = mean(a - b).

    Args:
        per_window_a: Per-window mean CE for model A (the fine-tuned model), 1-D.
        per_window_b: Per-window mean CE for model B (its base variant), 1-D, paired
            on the same windows as `per_window_a`.
        n_iterations: Bootstrap resamples.
        seed: RNG seed (deterministic given the same inputs).
        ci_level: Two-sided CI level in (0, 1).

    Returns:
        A `BootstrapResult` with the point estimate and percentile CI.
    """
    a = np.asarray(per_window_a, dtype=np.float64)
    b = np.asarray(per_window_b, dtype=np.float64)
    assert a.ndim == 1 and b.ndim == 1, "per_window vectors must be 1-D"
    assert a.shape == b.shape, f"shape mismatch: {a.shape} != {b.shape}"
    n = a.shape[0]

    delta_w = a - b  # paired per-window Δ
    point = float(delta_w.mean())

    rng = np.random.default_rng(seed)
    means = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)  # resample windows with replacement
        means[i] = delta_w[idx].mean()

    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    ci_low = float(np.percentile(means, 100.0 * lo_q))
    ci_high = float(np.percentile(means, 100.0 * hi_q))
    return BootstrapResult(
        delta_mean=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_windows=n,
        n_iterations=n_iterations,
        seed=seed,
        ci_level=ci_level,
    )
