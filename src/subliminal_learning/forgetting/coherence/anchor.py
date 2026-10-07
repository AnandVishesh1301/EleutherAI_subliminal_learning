"""Unigram-anchor arithmetic: the smoothed unigram CE and its bits-per-byte form.

Factored out of `scripts/build_unigram_anchor.py` (2026-09-05) so the frozen formula is
testable and shared by every pool (pp1, dp1, pp1t-*); the script's numbers are unchanged.
"""

from __future__ import annotations

import math

import numpy as np


def unigram_ce(counts: np.ndarray, targets: np.ndarray, alpha: float) -> float:
    """CE [nats] of an add-α unigram model with `counts` on the `targets` multiset.

    ``V = len(counts)``; the normalizer is ``total + alpha * V`` with ``total`` the count
    sum, i.e. exactly the frozen anchor formula.
    """
    counts = np.asarray(counts)
    total = int(counts.sum())
    V = len(counts)
    logp = np.log(counts + alpha) - math.log(total + alpha * V)
    return float(-logp[np.asarray(targets, dtype=np.int64)].mean())


def nats_per_token_to_bits_per_byte(h: float, n_targets: int, n_bytes: int) -> float:
    """Convert a per-token CE [nats] to bits per byte of the underlying text."""
    return h * n_targets / (n_bytes * math.log(2))
