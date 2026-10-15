from __future__ import annotations

import math
from collections import Counter

import pytest

from subliminal_learning.campaign import rev1_plan as plan


def test_rev1_cells_counts_match_plan() -> None:
    cells = [c for c in plan.all_cells() if not plan.is_ext(c)]
    counts = Counter((c.channel, c.stage, c.kind) for c in cells)
    assert counts[("hard", "rev", "treat")] == 18
    assert counts[("hard", "rev", "ctrl")] == 10
    assert counts[("hard", "rev", "base")] == 6
    assert counts[("hard", "pre", "treat")] == 18
    assert counts[("hard", "pre", "ctrl")] == 10
    assert counts[("soft", "rev", "treat")] == 40
    assert counts[("soft", "rev", "ctrl")] == 40
    assert counts[("soft", "rev", "base")] == 24
    assert counts[("soft", "pre", "treat")] == 20
    assert counts[("soft", "pre", "ctrl")] == 20
    assert len(cells) == len(set(cells)) == 206


def test_rev1x_extension_counts_and_base_teacher_only() -> None:
    ext = [c for c in plan.all_cells() if plan.is_ext(c)]
    counts = Counter((c.channel, c.stage) for c in ext)
    assert counts == {("hard", "rev"): 31, ("hard", "pre"): 25, ("soft", "rev"): 52,
                      ("soft", "pre"): 40}
    assert {c.teacher for c in ext if c.channel == "soft" and c.stage == "rev"} == {"base"}


def test_rev1_keys_round_trip_and_are_unique() -> None:
    cells = plan.all_cells()
    assert len({c.key for c in cells}) == len(cells)
    for cell in cells:
        assert plan.cell_from_key(cell.key) == cell


def test_rev1_unplanned_key_raises() -> None:
    with pytest.raises(ValueError):
        plan.cell_from_key("rev1|hard|rev|treat|dog|lora-sgd|t-|s4|base")


def test_rev1_hosts_and_queues() -> None:
    for cell in plan.all_cells():
        host = plan.cell_host(cell)
        ext = plan.is_ext(cell)
        if not ext:
            assert (host == plan.CLUSTER5) == (cell.channel == "hard")
        elif cell.channel == "hard":
            assert host in plan.EXT_HARD_HOSTS
        else:
            assert host in plan.EXT_SOFT_HOSTS
        assert plan.cell_queue(cell) in plan.host_queues(host, ext)
        assert plan.cell_queue(cell).startswith(plan.EXT_CAMPAIGN if ext else "rev1-")
    soft = Counter(plan.cell_host(c) for c in plan.soft_cells() if not plan.is_ext(c))
    assert set(soft) == set(plan.SOFT_HOSTS)
    assert max(soft.values()) - min(soft.values()) <= 1


def test_rev1_pre_and_rev_share_the_start_label() -> None:
    for cell in plan.all_cells():
        if cell.stage == "rev" and cell.kind != plan.BASE:
            pre = plan.Cell(cell.channel, "pre", cell.kind, cell.animal, cell.arm,
                            cell.seed, cell.tfs, None)
            assert pre in set(plan.all_cells())
            assert plan.start_dir(pre) == plan.start_dir(cell)


def test_rev1_policy_env_points_at_reversal_cache() -> None:
    env = plan.gpu_policy_json(plan.SOAR)["env"]
    assert env["HF_HUB_CACHE"] == str(plan.HF_ROOT / "hub")
    with pytest.raises(ValueError):
        plan.gpu_policy_json("not-a-host")


@pytest.mark.parametrize(
    ("pre_lo", "post_lo", "ret_lo", "ret_hi", "expected"),
    [
        (-0.01, 0.0, 0.0, 0.05, "void"),
        (0.3, -0.001, -0.01, 0.05, "removed"),
        (0.3, 0.002, 0.01, 0.08, "residual"),
        (0.3, 0.2, 0.6, 0.9, "present"),
        (0.3, 0.05, 0.15, 0.4, "partial"),
        (0.3, 0.05, 0.2, 0.7, "inconclusive"),
        (math.nan, 0.0, 0.0, 0.0, "void"),
    ],
)
def test_rev1_verdict_rule(pre_lo, post_lo, ret_lo, ret_hi, expected) -> None:
    from pipelines.rev1_results_pipeline import verdict

    assert verdict(pre_lo, post_lo, ret_lo, ret_hi) == expected


def test_rev1_reversal_rows_exclude_control_rows() -> None:
    from pipelines.rev1_stage_pipeline import reversal_rows
    from subliminal_learning.campaign import hlopt_plan
    from subliminal_learning.students.dataset_io import subsample_rows

    pool = [{"prompt": f"p{i}", "completion": f" {i}"} for i in range(25_000)]
    pool.append(dict(pool[0]))
    chosen, stats = reversal_rows(pool)
    used = {
        (r["prompt"], r["completion"])
        for r in subsample_rows(pool, hlopt_plan.SUBSAMPLE_SIZE, hlopt_plan.SUBSAMPLE_SEED)
    }
    assert len(chosen) == plan.REV_ROWS
    assert not {(r["prompt"], r["completion"]) for r in chosen} & used
    assert stats["n_eligible"] == sum((r["prompt"], r["completion"]) not in used for r in pool)


def test_rev1_pairs_match_pre_cells_with_missing_teacher() -> None:
    import pandas as pd

    from pipelines.rev1_results_pipeline import pair_rows

    rows = []
    for stage, kind, teacher, p in (
        ("pre", "treat", None, 0.9), ("pre", "ctrl", None, 0.01),
        ("rev", "treat", "base", 0.02), ("rev", "ctrl", "base", 0.01),
    ):
        rows.append({"key": f"{stage}{kind}", "channel": "hard", "stage": stage, "kind": kind,
                     "arm": "lora-adamw", "seed": 1, "tfs": None, "teacher": teacher,
                     "animal": "dog", "evaluated": True, "quarantined": False,
                     "collapsed": False, "delta_ce": 1.0,
                     **{f"p_{s}": p for s in ("test", "favorite", "dislike", "validation")}})
    pairs = pair_rows(pd.DataFrame(rows))
    row = pairs[(pairs["animal"] == "dog") & (pairs["arm"] == "lora-adamw")
                & (pairs["seed"] == 1)].iloc[0]
    assert not row["excluded"]
    assert row["pre_test"] == pytest.approx(0.89)
    assert row["post_test"] == pytest.approx(0.01)
