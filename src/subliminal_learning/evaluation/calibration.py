"""Expected calibration error and reliability bins (Guo et al., 2017, §2).

Verified against the arXiv:1706.04599 LaTeX source (ar5iv render, 2026-09-14):
``I_m = ((m-1)/M, m/M]``; ``acc(B_m) = 1/|B_m| Σ_{i∈B_m} 1(ŷ_i = y_i)``;
``conf(B_m) = 1/|B_m| Σ_{i∈B_m} p̂_i`` with ``p̂_i`` the max softmax
probability; Eq. 3 ``ECE = Σ_m |B_m|/n · |acc(B_m) − conf(B_m)|``; Table 1
uses ``M = 15``. A confidence of exactly 0.0 lies in no half-open bin and is
assigned to the first bin ``(0, 1/M]``.

Anshumann et al. (2025) report ECE (%) citing Guo et al. (2017); their text
states neither the bin count nor the interval convention.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    """One equal-width confidence bin ``(lower, upper]``.

    Attributes:
        lower: Exclusive lower edge (the first bin also holds 0.0).
        upper: Inclusive upper edge.
        count: Predictions in the bin.
        accuracy: Fraction correct; None when the bin is empty.
        mean_confidence: Mean confidence; None when the bin is empty.
    """

    lower: float
    upper: float
    count: int
    accuracy: float | None
    mean_confidence: float | None


def _validate(
    confidences: Sequence[float], correct: Sequence[bool | int], n_bins: int
) -> None:
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(confidences) != len(correct):
        raise ValueError(
            f"length mismatch: {len(confidences)} confidences vs "
            f"{len(correct)} correctness flags"
        )
    if not confidences:
        raise ValueError("calibration needs at least one prediction")
    for c in confidences:
        if not math.isfinite(c) or not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence must be finite and in [0, 1], got {c}")
    for k in correct:
        if k not in (0, 1):
            raise ValueError(f"correct must be 0/1 or bool, got {k!r}")


def reliability_table(
    confidences: Sequence[float],
    correct: Sequence[bool | int],
    n_bins: int = 15,
) -> tuple[ReliabilityBin, ...]:
    """Bin predictions by confidence into ``n_bins`` equal-width bins.

    Args:
        confidences: Per-prediction top-1 probability in ``[0, 1]``.
        correct: Per-prediction top-1 correctness (bool or 0/1).
        n_bins: Number of bins ``M``.

    Returns:
        ``n_bins`` bins in ascending confidence order.
    """
    _validate(confidences, correct, n_bins)
    uppers = [m / n_bins for m in range(1, n_bins + 1)]
    counts = [0] * n_bins
    hits = [0] * n_bins
    conf_sums = [0.0] * n_bins
    for c, k in zip(confidences, correct, strict=True):
        b = bisect.bisect_left(uppers, c)
        counts[b] += 1
        hits[b] += int(k)
        conf_sums[b] += c
    return tuple(
        ReliabilityBin(
            lower=(m - 1) / n_bins,
            upper=uppers[m - 1],
            count=counts[m - 1],
            accuracy=hits[m - 1] / counts[m - 1] if counts[m - 1] else None,
            mean_confidence=(
                conf_sums[m - 1] / counts[m - 1] if counts[m - 1] else None
            ),
        )
        for m in range(1, n_bins + 1)
    )


def ece_from_table(bins: Sequence[ReliabilityBin]) -> float:
    """Count-weighted mean ``|accuracy − mean_confidence|`` (Guo et al. Eq. 3)."""
    n = sum(b.count for b in bins)
    return sum(
        b.count / n * abs(b.accuracy - b.mean_confidence) for b in bins if b.count
    )


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool | int],
    n_bins: int = 15,
) -> float:
    """Expected calibration error over equal-width confidence bins.

    Args:
        confidences: Per-prediction top-1 probability in ``[0, 1]``.
        correct: Per-prediction top-1 correctness (bool or 0/1).
        n_bins: Number of bins ``M``.

    Returns:
        ECE in ``[0, 1]``.
    """
    return ece_from_table(reliability_table(confidences, correct, n_bins))
