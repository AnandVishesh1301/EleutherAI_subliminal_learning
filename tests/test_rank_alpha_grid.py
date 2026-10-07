"""Synthetic checks: trajectories.rank_alpha_grid / nearest_step_slice / grid_at_step reproduce the
inline helpers of transfer_trajectory_grids, dd1b_cross_seed2_ladder, dd1b_same_model_ladder and
qwen25_same_model_arm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from subliminal_learning.analysis.trajectories import (
    grid_at_step,
    grid_nanmedian,
    nearest_step_slice,
    rank_alpha_grid,
)

RANKS, ALPHAS = [8, 16, 32], [8, 16, 32, 64]


def _nb_grid(sub, col, ranks=RANKS, alphas=ALPHAS):  # transfer_trajectory_grids / dd1b_cross
    g = sub.pivot_table(index="rank", columns="alpha", values=col, aggfunc="median").reindex(index=ranks, columns=alphas)
    n = sub.pivot_table(index="rank", columns="alpha", values=col, aggfunc="size").reindex(index=ranks, columns=alphas)
    return g, n


def _nb_grid_plain(values, col):  # dd1b_same_model_ladder / qwen25 _grid, _grid_na
    g = values.pivot_table(index="rank", columns="alpha", values=col, aggfunc="median")
    n = values.pivot_table(index="rank", columns="alpha", values=col, aggfunc="size")
    return g, n


def _nb_at_step_ttg(tr, alive_tags, step):
    t = tr[tr["tag"].isin(alive_tags)].copy()
    t = t[t["step"] <= 1.1 * step]
    if t.empty:
        return None, None
    t["d"] = (t["step"] - step).abs()
    pick = t.loc[t.groupby("tag")["d"].idxmin()]
    pick = pick[pick["d"] <= 0.1 * step]
    if pick.empty:
        return None, None
    return _nb_grid(pick, "p_test"), int(pick["step"].median())


def _nb_at_step_cross(tr, alive_tags, step):
    t = tr[tr["tag"].isin(alive_tags) & tr["p_test"].notna()].copy()
    t["d"] = (t["step"] - step).abs()
    pick = t.loc[t.groupby("tag")["d"].idxmin()]
    pick = pick[pick["d"] <= 0.1 * step]
    if pick.empty:
        return None, None
    return _nb_grid(pick, "p_test"), int(pick["step"].median())


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(40):
        r, a = int(rng.choice(RANKS[:2])), int(rng.choice(ALPHAS[:3]))
        steps = np.sort(rng.choice(np.arange(50, 1300, 25), size=rng.integers(1, 8), replace=False))
        for s in steps:
            p = float(rng.random()) if rng.random() > 0.15 else np.nan
            rows.append({"tag": f"t{k}", "rank": r, "alpha": a, "step": int(s), "p_test": p,
                         "p_end": float(rng.random())})
    return pd.DataFrame(rows)


def _eq(a, b):
    if a == (None, None) or b == (None, None):
        assert a == b == (None, None)
        return
    (g1, n1), s1 = a
    (g2, n2), s2 = b
    pdt.assert_frame_equal(g1, g2)
    pdt.assert_frame_equal(n1, n2)
    assert s1 == s2


def test_grid_variants():
    df = _frame()
    for got, want in [(rank_alpha_grid(df, "p_end", RANKS, ALPHAS), _nb_grid(df, "p_end")),
                      (rank_alpha_grid(df, "p_end"), _nb_grid_plain(df, "p_end"))]:
        pdt.assert_frame_equal(got[0], want[0])
        pdt.assert_frame_equal(got[1], want[1])


def test_at_step_variants():
    for seed in range(20):
        tr = _frame(seed)
        alive = {f"t{k}" for k in range(0, 40, 2)}
        for st in (60, 100, 400, 1000, 1250, 5000):
            _eq(grid_at_step(tr, alive, st, "p_test", RANKS, ALPHAS, max_step_factor=1.1),
                _nb_at_step_ttg(tr, alive, st))
            _eq(grid_at_step(tr, alive, st, "p_test", RANKS, ALPHAS, require_value="p_test"),
                _nb_at_step_cross(tr, alive, st))


def test_nearest_step_slice_tolerance_and_ties():
    tr = pd.DataFrame({"tag": ["a", "a", "b", "c"], "step": [90, 110, 200, 100], "p_test": 1.0})
    pick = nearest_step_slice(tr, 100)
    assert list(pick["tag"]) == ["a", "c"] and list(pick["step"]) == [90, 100]  # tie -> first row
    assert nearest_step_slice(tr, 100, ["b"]).empty


def test_grid_nanmedian():
    g = pd.DataFrame([[1.0, np.nan], [3.0, 5.0]], index=[8, 16])
    assert grid_nanmedian(g) == 3.0
    assert grid_nanmedian(g, [16]) == 4.0
