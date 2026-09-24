"""xfam2 balanced placement: only claim-free queues move, core first, evals follow."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.xfam2_placement import QueueLoad, balance, host_loads

A, B, C = "a", "b", "c"


def _q(name, host, gpu_h, *, movable=True, core=True, unit="u", ev=False, tr=False):
    return QueueLoad(name, host, gpu_h, movable, core, unit, ev, tr)


def test_balance_moves_largest_queues_off_the_busiest_host() -> None:
    loads = [_q("q1", A, 10), _q("q2", A, 6), _q("q3", A, 4), _q("q4", B, 2)]
    moves = balance(loads, [A, B, C])
    after = [
        _q(q.name, moves.get(q.name, q.host), q.gpu_h) for q in loads
    ]
    assert max(host_loads(after, [A, B, C], core=True).values()) <= 10
    assert moves["q1"] in (B, C)


def test_claim_held_and_foreign_host_queues_never_move() -> None:
    loads = [_q("held", A, 50, movable=False), _q("far", "z", 50), _q("x", A, 1)]
    moves = balance(loads, [A, B])
    assert "held" not in moves and "far" not in moves


def test_core_and_tail_are_balanced_separately() -> None:
    loads = [_q("c1", A, 8), _q("c2", A, 8), _q("t1", A, 8, core=False)]
    moves = balance(loads, [A, B])
    assert moves == {"c1": B} or moves == {"c2": B}


def test_an_eval_queue_follows_its_units_train_queue() -> None:
    loads = [
        _q("train", A, 10, unit="u1", tr=True),
        _q("eval", A, 0, unit="u1", ev=True),
        _q("other", A, 10, unit="u2", tr=True),
    ]
    moves = balance(loads, [A, B])
    moved_train = "train" in moves
    assert ("eval" in moves) == moved_train
    if moved_train:
        assert moves["eval"] == moves["train"]


def test_hosts_must_be_distinct() -> None:
    with pytest.raises(ValueError):
        balance([], [A, A])


def test_queue_node_prefers_runtime_overrides() -> None:
    unit = plan.core_units()[0]
    name = plan.q_unit(unit, plan.CLS_TRAIN)
    assert plan.queue_node(name) == plan.STUDENT_NODE[unit.cell.student]
    assert plan.queue_node(name, {name: plan.CLUSTER5}) == plan.CLUSTER5
    assert plan.queue_node(name, {}) == plan.STUDENT_NODE[unit.cell.student]
