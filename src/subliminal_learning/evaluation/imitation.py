"""Tier-I readout of saved students (pure): held-out row selection, groups, shards.

Blog wave 2 plan §3.7. A hard student trained on ``subsample_rows(pool, 10_000, 42)``
(`students.dataset_io`); its held-out rows are the rest of the pool. `subsample_indices`
reproduces that choice by position: ``random.Random(seed).sample`` picks positions from
the sequence length and the RNG alone, so sampling ``range(n)`` and sampling the rows
select the same positions in the same order.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

EVAL_SEED = 20261010
N_EVAL_ROWS = 1000
N_CARRIERS = 200
MAX_UNITS_PER_SHARD = 40
UPDATE_GROUPS: tuple[str, ...] = (
    "embed_in",
    "embed_out",
    "attention",
    "mlp",
    "layernorm",
    "other",
)
# Parameter-name substring -> module group, first match wins ("post_attention_layernorm"
# is a LayerNorm, so the LayerNorm keys precede "attention").
PARAM_GROUPS: tuple[tuple[str, str], ...] = (
    ("lora_", "lora"),
    ("embed_in", "embed_in"),
    ("embed_out", "embed_out"),
    ("layernorm", "layernorm"),
    ("layer_norm", "layernorm"),
    ("attention", "attention"),
    ("mlp", "mlp"),
)

T = TypeVar("T")


def param_group(name: str) -> str:
    """Module group of a parameter name (``other`` when no key matches)."""
    return next((group for key, group in PARAM_GROUPS if key in name), "other")


def subsample_indices(n_rows: int, size: int | None, seed: int) -> list[int]:
    """Row positions `subsample_rows(rows, size, seed)` keeps, in its order."""
    if size is None or n_rows <= size:
        return list(range(n_rows))
    return random.Random(seed).sample(range(n_rows), size)


def heldout_indices(n_rows: int, size: int | None, seed: int) -> list[int]:
    """Sorted row positions outside the training subsample."""
    train = set(subsample_indices(n_rows, size, seed))
    return [i for i in range(n_rows) if i not in train]


def seeded_subset(
    indices: Iterable[int], k: int = N_EVAL_ROWS, seed: int = EVAL_SEED
) -> list[int]:
    """Sorted ``Random(seed)`` sample of ``min(k, n)`` from the sorted `indices`."""
    pool = sorted(indices)
    return sorted(random.Random(seed).sample(pool, min(k, len(pool))))


def top_token_ids(
    sequences: Iterable[Sequence[int]],
    k: int = N_CARRIERS,
    exclude: Iterable[int] = (),
) -> list[int]:
    """The `k` most frequent token ids (ties to the smaller id), minus `exclude`."""
    banned = set(exclude)
    counts = Counter(t for seq in sequences for t in seq if t not in banned)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:k]]


def update_summary(sq_norms: Mapping[str, float]) -> dict[str, float]:
    """``norm_total`` plus ``norm_<g>`` and ``share_<g>`` (of the squared total).

    Every `UPDATE_GROUPS` group is reported (0 when absent); shares are NaN for a zero
    update.

    Raises:
        ValueError: A group outside `UPDATE_GROUPS`.
    """
    unknown = set(sq_norms) - set(UPDATE_GROUPS)
    if unknown:
        raise ValueError(f"unknown parameter groups {sorted(unknown)}")
    sq = {g: float(sq_norms.get(g, 0.0)) for g in UPDATE_GROUPS}
    total = sum(sq.values())
    out = {"norm_total": math.sqrt(total)}
    for g, v in sq.items():
        out[f"norm_{g}"] = math.sqrt(v)
        out[f"share_{g}"] = v / total if total > 0 else math.nan
    return out


def head_row_summary(
    row_norms: Sequence[float], carrier_ids: Sequence[int], animal_id: int
) -> dict[str, float]:
    """Mean output-head row change over the carriers vs the animal row, and their ratio.

    Raises:
        ValueError: No carriers, or the animal among them.
    """
    if not carrier_ids or animal_id in set(carrier_ids):
        raise ValueError(f"bad carriers for animal id {animal_id}")
    carrier = sum(row_norms[i] for i in carrier_ids) / len(carrier_ids)
    animal = row_norms[animal_id]
    return {
        "head_delta_carrier_mean": carrier,
        "head_delta_animal": animal,
        "head_animal_over_carrier": animal / carrier if carrier > 0 else math.nan,
    }


def plan_shards(
    units: Sequence[T],
    group_of: Callable[[T], tuple],
    key_of: Callable[[T], str],
    max_per_shard: int = MAX_UNITS_PER_SHARD,
) -> list[list[T]]:
    """Deterministic shards: units sorted by (group, key), each group split evenly.

    A group of ``n`` units becomes ``ceil(n / max_per_shard)`` shards whose sizes differ
    by at most one; shards never mix groups.

    Args:
        units: Units to shard.
        group_of: Unit -> sortable group (units sharing models / data).
        key_of: Unit -> unique sortable key.
        max_per_shard: Shard size cap.

    Raises:
        ValueError: Duplicate keys or a non-positive cap.
    """
    if max_per_shard < 1:
        raise ValueError(f"max_per_shard must be >= 1, got {max_per_shard}")
    keys = [key_of(u) for u in units]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate unit keys")
    groups: dict[tuple, list[T]] = {}
    for unit in sorted(units, key=lambda u: (group_of(u), key_of(u))):
        groups.setdefault(group_of(unit), []).append(unit)
    shards: list[list[T]] = []
    for members in groups.values():
        n_shards = math.ceil(len(members) / max_per_shard)
        base, extra = divmod(len(members), n_shards)
        start = 0
        for i in range(n_shards):
            size = base + (1 if i < extra else 0)
            shards.append(members[start : start + size])
            start += size
    return shards
