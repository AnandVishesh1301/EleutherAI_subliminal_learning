"""In-training probe schedule: a regular cadence plus an optional dense early window."""

from __future__ import annotations


def validate_probe_schedule(every: int, dense_every: int, dense_until: int) -> None:
    """Raise on a dense window that is negative or open-ended.

    Raises:
        ValueError: `dense_every` or `dense_until` is negative, or `dense_every` is
            set without `dense_until`.
    """
    if dense_every < 0 or dense_until < 0:
        raise ValueError(
            f"probe dense_every={dense_every} / dense_until={dense_until} must be >= 0"
        )
    if dense_every > 0 and dense_until <= 0:
        raise ValueError("probe dense_every requires dense_until > 0")


def probe_due(
    step: int, every: int, dense_every: int = 0, dense_until: int = 0
) -> bool:
    """Whether a probe runs at optimizer step `step`.

    Args:
        step: Global optimizer step.
        every: Regular cadence (``<= 0`` = off).
        dense_every: Dense cadence inside the early window (``0`` = off).
        dense_until: Last step of the dense window (inclusive).

    Returns:
        True on a multiple of `every`, or on a multiple of `dense_every` at or
        before `dense_until`.
    """
    if every > 0 and step % every == 0:
        return True
    return dense_every > 0 and step <= dense_until and step % dense_every == 0
