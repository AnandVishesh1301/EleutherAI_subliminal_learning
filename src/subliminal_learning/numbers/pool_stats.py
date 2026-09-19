"""Unigram statistics of a generated number pool (`filtered.jsonl`'s ``numbers`` field).

The number channel a teacher writes into is, at zeroth order, *which integers it emits how
often*. This module is that zeroth order and nothing more: a pool becomes a probability vector
over the ints ``0..support-1``, and two pools are compared with total variation, or two
*carriers* (a teacher pool minus its own base pool) with a cosine. Pure numpy over already-loaded
rows -- no disk, no output-tree knowledge -- so it sits at the leaf of the layering and any
caller (`analysis.diagonal_direction`, a notebook, a pipeline) can use it.

Out-of-support integers (the pools are 0-999 by construction, but a teacher can emit a stray
four-digit number) are DROPPED and counted rather than clipped: clipping would pile them onto
999 and invent mass where there is none. `unigram_counts` returns that count so a caller can
report it; `unigram` is the normalized convenience.

This replaces the hand-rolled unigram/TV block of the checkpoint-distill notebook (§5c, which
said "migrate when one exists").
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["SUPPORT", "unigram_counts", "unigram", "entropy_bits", "tv", "cosine"]

#: Default support: the number pools are generated as three-digit integers, i.e. 0-999.
SUPPORT = 1000


def _numbers(rows_or_numbers: Iterable) -> Iterable[int]:
    """Flatten either ``filtered.jsonl`` rows (dicts with ``numbers``) or a flat int iterable."""
    for item in rows_or_numbers:
        if isinstance(item, dict):
            yield from item["numbers"]
        else:
            yield item


def unigram_counts(rows_or_numbers: Iterable, support: int = SUPPORT) -> tuple[np.ndarray, int]:
    """Raw integer counts over ``0..support-1``, plus the number of out-of-support values.

    Args:
        rows_or_numbers: ``filtered.jsonl`` rows (``{"numbers": [int, ...]}``) or a flat
            iterable of ints.
        support: Size of the integer support.

    Returns:
        ``(counts, n_dropped)`` -- an int64 array of length `support` and the count of values
        outside it (negative or ``>= support``).
    """
    vals = np.fromiter(_numbers(rows_or_numbers), dtype=np.int64)
    ok = (vals >= 0) & (vals < support)
    counts = np.bincount(vals[ok], minlength=support).astype(np.int64)
    return counts, int((~ok).sum())


def unigram(rows_or_numbers: Iterable, support: int = SUPPORT) -> np.ndarray:
    """The pool's unigram probability vector over ``0..support-1``.

    Args:
        rows_or_numbers: As `unigram_counts`.
        support: Size of the integer support.

    Returns:
        A float array of length `support` summing to 1 (all zeros for an empty pool).
    """
    counts, _ = unigram_counts(rows_or_numbers, support)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def entropy_bits(p: Sequence[float]) -> float:
    """Shannon entropy of the probability vector `p`, in bits (``0 log 0 = 0``)."""
    q = np.asarray(p, dtype=float)
    nz = q[q > 0]
    return float(-(nz * np.log2(nz)).sum())


def tv(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation distance ``½ Σ|p − q|`` between two probability vectors."""
    a, b = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"p and q must match in shape, got {a.shape} and {b.shape}")
    return float(0.5 * np.abs(a - b).sum())


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity of two vectors; NaN when either has zero norm."""
    a, b = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"u and v must match in shape, got {a.shape} and {b.shape}")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))
