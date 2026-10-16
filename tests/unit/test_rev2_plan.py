"""Reversal phase 2 (`rev2`): plan enumeration, layout, policy and the pure metrics."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from pipelines.rev2_results_pipeline import _first_cross, relearn_metrics
from pipelines.rev2_stage_pipeline import nested_rank
from subliminal_learning.campaign import rev1_plan as rev1
from subliminal_learning.campaign import rev2_plan as plan


def test_rev2_cells_keys_round_trip_and_are_unique() -> None:
    cells = plan.all_cells()
    keys = [c.key for c in cells]
    assert len(keys) == len(set(keys))
    for c in cells[:: max(1, len(cells) // 200)]:
        assert plan.cell_from_key(c.key) == c


def test_rev2_cell_roots_are_unique() -> None:
    roots = [plan.cell_root(c) for c in plan.all_cells()]
    assert len(roots) == len(set(roots))


def test_rev2_unplanned_key_raises() -> None:
    cell = plan.all_cells()[0]
    bad = cell.key.replace("|s0|", "|s99|")
    with pytest.raises(ValueError, match="not a planned"):
        plan.cell_from_key(bad)


def test_rev2_r1l_groups_are_the_eight_residual_soft_groups() -> None:
    l_cells = [c for c in plan.all_cells() if c.block == "L"]
    assert len(l_cells) == 8 * 3 * 2
    assert {(c.start, c.animal, c.teacher) for c in l_cells} == set(plan.R1L_GROUPS)


@pytest.mark.parametrize(
    ("block", "op", "channel", "expected"),
    [("L", "rev", "soft", 48), ("X", "rev", "soft", 10), ("R4X", "pre", "soft", 16),
     ("R4X", "rev", "soft", 40)],
)
def test_rev2_block_counts(block: str, op: str, channel: str, expected: int) -> None:
    got = sum(1 for c in plan.all_cells()
              if (c.block, c.op, c.channel) == (block, op, channel))
    assert got == expected


def test_rev2_every_treatment_rev_cell_has_its_control() -> None:
    keys = {c.key for c in plan.all_cells()}
    for c in plan.all_cells():
        if c.op == "rev" and c.kind == plan.TREAT:
            animal = c.animal if c.channel == "soft" else None
            ctrl = plan.Cell(c.block, c.op, c.channel, c.src, plan.CTRL, animal, c.start,
                             c.seed, c.tfs, c.arm, c.lr, c.teacher, c.n, c.draw, c.target)
            assert ctrl.key in keys, c.key


def test_rev2_relearn_pairs_share_target_rate_and_seed() -> None:
    keys = {c.key for c in plan.all_cells()}
    for c in plan.all_cells():
        if c.op == "relearn" and c.kind == plan.TREAT:
            animal = c.animal if c.channel == "soft" else None
            ctrl = plan.Cell(c.block, c.op, c.channel, c.src, plan.CTRL, animal, c.start,
                             c.seed, c.tfs, c.arm, c.lr, c.teacher, c.n, c.draw, c.target)
            assert ctrl.key in keys, c.key


def test_rev2_default_host_is_stable_and_channel_bound() -> None:
    for c in plan.all_cells():
        host = plan.default_host(c)
        assert host == plan.default_host(c)
        allowed = plan.HARD_HOSTS if c.channel == "hard" else plan.SOFT_HOSTS
        assert host in allowed


def test_rev2_tiers_and_priorities() -> None:
    tiers = Counter(plan.cell_tier(c) for c in plan.all_cells())
    assert set(tiers) == {"g", "h", "l"}
    assert plan.TIER_PRIORITY["g"] > plan.TIER_PRIORITY["h"] > plan.TIER_PRIORITY["l"]
    for c in plan.all_cells():
        if c.block == "L" or c.op == "posctl":
            assert plan.cell_tier(c) == "g"
        if c.block == "R5" and c.teacher.startswith("ck:"):
            assert plan.cell_tier(c) == "l"


def test_rev2_policy_honours_the_fence() -> None:
    doc = plan.gpu_policy_json(plan.CLUSTER4, (2, 3))
    assert doc["gpus"] == [0, 1, 4, 5, 6, 7]
    assert plan.gpu_policy_json(plan.CLUSTER4)["gpus"] == list(range(8))
    with pytest.raises(ValueError, match="fenced"):
        plan.gpu_policy_json(plan.CLUSTER4, tuple(range(8)))


def test_rev2_install_rates_match_rank1() -> None:
    for arm, spec in rev1.HARD_ARMS.items():
        assert plan.install_lr("hard", arm, None) == spec.lr
    for arm, spec in rev1.SOFT_ARMS.items():
        for animal in rev1.SOFT_ANIMALS:
            assert plan.install_lr("soft", arm, animal) == spec.lr[animal]


def test_nested_rank_prefixes_nest_and_depend_on_seed() -> None:
    rows = [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(500)]
    r0 = nested_rank(rows, 0)
    assert sorted(map(str, r0)) == sorted(map(str, rows))
    for small, big in ((64, 156), (156, 312)):
        assert r0[:small] == r0[:big][:small]
    assert nested_rank(rows, 1)[:64] != r0[:64]
    assert nested_rank(list(reversed(rows)), 0) == r0


def test_first_cross_interpolates() -> None:
    assert _first_cross({0: 0.0, 10: 1.0}, 0.5) == pytest.approx(5.0)
    assert _first_cross({0: 0.0, 10: 0.2}, 0.5) == math.inf


def test_relearn_metrics_faster_relearner_has_savings_above_one() -> None:
    fast = {0: 0.10, 10: 0.30, 20: 0.40, 40: 0.50}
    slow = {0: 0.01, 10: 0.06, 20: 0.11, 40: 0.21}
    m = relearn_metrics(fast, slow, early=20)
    assert m["savings_ratio"] > 1
    assert m["area_ratio"] > 1
    assert m["early_gap"] == pytest.approx((0.40 - 0.10) - (0.11 - 0.01))
    same = relearn_metrics(slow, slow, early=20)
    assert same["savings_ratio"] == pytest.approx(1.0)
    assert same["area_ratio"] == pytest.approx(1.0)
