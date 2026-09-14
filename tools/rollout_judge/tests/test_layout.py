from __future__ import annotations

from pathlib import Path

import pytest

from rollout_judge.layout import partition_shards


def _shards(n: int) -> list[tuple[str, Path]]:
    return [("rollouts_clean", Path(f"shard-{i:04d}.jsonl")) for i in range(n)]


@pytest.mark.parametrize("num_workers", [1, 2, 3, 8, 16])
def test_partition_shards_is_a_disjoint_cover(num_workers: int) -> None:
    """Two workers sharing a shard would append the same row_ids twice.

    Judgment shard paths are derived from the queue shard name, so an
    overlapping partition double-counts rows in every downstream rate.
    """
    shards = _shards(149)
    parts = [
        partition_shards(shards, index, num_workers) for index in range(num_workers)
    ]
    flat = [item for part in parts for item in part]
    assert sorted(flat) == sorted(shards)
    assert len(flat) == len(set(flat))


def test_partition_shards_balances_within_one_shard() -> None:
    sizes = [len(partition_shards(_shards(149), i, 8)) for i in range(8)]
    assert max(sizes) - min(sizes) <= 1


def test_partition_shards_single_worker_keeps_order() -> None:
    shards = _shards(5)
    assert partition_shards(shards, 0, 1) == shards


def test_partition_shards_more_workers_than_shards_yields_empty_tail() -> None:
    parts = [partition_shards(_shards(3), i, 5) for i in range(5)]
    assert [len(part) for part in parts] == [1, 1, 1, 0, 0]


@pytest.mark.parametrize(
    ("worker_index", "num_workers"),
    [(0, 0), (-1, 4), (4, 4), (5, 4)],
)
def test_partition_shards_rejects_out_of_range(
    worker_index: int,
    num_workers: int,
) -> None:
    with pytest.raises(ValueError):
        partition_shards(_shards(4), worker_index, num_workers)
