"""Balanced queue placement for spreading xfam2 over more nodes (pure).

The driver can only move a queue that holds no claim, so a rebalance moves whole
claim-free queues. Core queues are balanced first, then the tail, each tier on its own
so the headline cells finish together; an eval queue follows its unit's train queue.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueueLoad:
    """One submitted queue's remaining work.

    Attributes:
        name: Queue name.
        host: Host it is pinned to now.
        gpu_h: Estimated GPU-hours of its pending and running jobs.
        movable: Claim-free with pending jobs.
        core: Part of the completion criterion.
        unit: Unit slug the queue belongs to (eval queues follow its train queue).
        is_eval: An eval queue.
        is_train: A treatment train queue.
    """

    name: str
    host: str
    gpu_h: float
    movable: bool
    core: bool
    unit: str
    is_eval: bool = False
    is_train: bool = False


def host_loads(
    loads: Sequence[QueueLoad], hosts: Sequence[str], *, core: bool
) -> dict[str, float]:
    """GPU-hours pinned to each of `hosts` for one tier."""
    out = {h: 0.0 for h in hosts}
    for q in loads:
        if q.core == core and q.host in out:
            out[q.host] += q.gpu_h
    return out


def balance(loads: Sequence[QueueLoad], hosts: Sequence[str]) -> dict[str, str]:
    """``queue -> new host`` moves that even out each tier's load over `hosts`.

    Largest movable queues first; a move is taken only when it lowers the larger of
    the two hosts' loads, so it never makes the busiest host busier. Queues pinned
    to a host outside `hosts` are never moved.

    Raises:
        ValueError: If `hosts` is empty or repeats a host.
    """
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError(f"hosts must be distinct and non-empty: {list(hosts)}")
    moves: dict[str, str] = {}
    for core in (True, False):
        load = host_loads(loads, hosts, core=core)
        candidates = sorted(
            (
                q
                for q in loads
                if q.core == core
                and q.movable
                and not q.is_eval
                and q.host in load
                and q.gpu_h > 0
            ),
            key=lambda q: (-q.gpu_h, q.name),
        )
        for q in candidates:
            src = q.host
            dst = min(hosts, key=lambda h: (load[h], hosts.index(h)))
            if dst == src:
                continue
            if max(load[src] - q.gpu_h, load[dst] + q.gpu_h) >= max(
                load[src], load[dst]
            ):
                continue
            load[src] -= q.gpu_h
            load[dst] += q.gpu_h
            moves[q.name] = dst
    train_dest = {
        q.unit: moves[q.name] for q in loads if q.is_train and q.name in moves
    }
    for q in loads:
        if q.is_eval and q.movable and q.unit in train_dest:
            moves[q.name] = train_dest[q.unit]
    return moves


def merged_overrides(
    existing: Mapping[str, str], moves: Mapping[str, str]
) -> dict[str, str]:
    """`existing` placement overrides with `moves` applied on top."""
    return {**dict(existing), **dict(moves)}
