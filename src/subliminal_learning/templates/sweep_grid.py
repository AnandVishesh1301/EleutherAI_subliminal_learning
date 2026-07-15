"""Shared expansion of template-teacher LoRA sweep axes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def shard_cells(
    cells: Sequence[T], *, num_shards: int = 1, shard_index: int = 0
) -> list[T]:
    """Return one deterministic, strided partition of an expanded sweep grid."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards - 1}], got {shard_index}"
        )
    return list(cells[shard_index::num_shards])


def lora_pairs_from_grid(grid: dict) -> list[tuple[int, int]]:
    """Return ``(rank, alpha)`` cells from a relative or legacy explicit grid.

    Ratio schema::

        lora_r: [4, 8, 16]
        lora_alpha_over_r: [1, 2, 4]

    An optional ``lora_alpha_max`` removes derived cells above a fixed alpha cap.
    Older explicit ``lora: [{r: 8, alpha: 8}]`` files remain supported, but schemas
    may not mix.
    """
    has_ranks = "lora_r" in grid
    has_ratios = "lora_alpha_over_r" in grid
    has_explicit = "lora" in grid
    if has_explicit and (has_ranks or has_ratios or "lora_alpha_max" in grid):
        raise ValueError("grid must use one LoRA schema")
    if has_ranks or has_ratios or "lora_alpha_max" in grid:
        if not has_ranks or not has_ratios:
            raise ValueError("relative LoRA grid requires lora_r and lora_alpha_over_r")
        ranks = [int(r) for r in grid["lora_r"]]
        if not ranks or any(r <= 0 for r in ranks):
            raise ValueError("lora_r must contain positive ranks")
        ratios = [float(x) for x in grid["lora_alpha_over_r"]]
        if not ratios or any(not math.isfinite(x) or x <= 0 for x in ratios):
            raise ValueError("lora_alpha_over_r must contain finite positive ratios")
        alpha_max = int(grid["lora_alpha_max"]) if "lora_alpha_max" in grid else None
        if alpha_max is not None and alpha_max <= 0:
            raise ValueError("lora_alpha_max must be positive")
        pairs: list[tuple[int, int]] = []
        for r in ranks:
            for ratio in ratios:
                alpha = r * ratio
                if not alpha.is_integer():
                    raise ValueError(
                        f"r={r} * alpha_over_r={ratio} does not produce integer alpha"
                    )
                if alpha_max is None or alpha <= alpha_max:
                    pairs.append((r, int(alpha)))
        if not pairs:
            raise ValueError("LoRA ratio grid is empty after applying lora_alpha_max")
        return pairs
    if not has_explicit:
        raise ValueError("grid requires LoRA axes or explicit lora pairs")
    pairs = [(int(p["r"]), int(p["alpha"])) for p in grid["lora"]]
    if not pairs or any(r <= 0 or alpha <= 0 for r, alpha in pairs):
        raise ValueError("explicit lora pairs must contain positive r and alpha")
    return pairs
