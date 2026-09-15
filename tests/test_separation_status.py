"""Unit tests for the separation diagnostic (CPU-only, constructed lattices)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from subliminal_learning.stats.collapse_law import (
    max_margin_ratios,
    tied_cells,
    separating_exponent_set,
    separation_status,
)


def _frame(lr, alpha, rank, collapsed):
    return pd.DataFrame({"lr": lr, "alpha": alpha, "rank": rank, "collapsed": collapsed})


def _lattice():
    lrs = [1e-4, 3e-4, 1e-3, 3e-3]
    alphas = [8, 16, 32, 64]
    ranks = [4, 16, 64]
    grid = [(lr, a, r) for lr in lrs for a in alphas for r in ranks]
    return np.array(grid)


def test_strictly_separable_lattice_is_complete():
    g = _lattice()
    # Label by a plane in (ln lr, ln alpha, ln rank) with every point strictly off it.
    score = np.log(g[:, 0]) + 0.6 * np.log(g[:, 1]) - 0.2 * np.log(g[:, 2])
    y = (score > np.median(score)).astype(int)
    assert separation_status(_frame(g[:, 0], g[:, 1], g[:, 2], y)) == "complete"


def test_points_on_the_plane_are_quasi():
    # Separable in ln lr, except for one tied cell that sits exactly on the plane and
    # carries both labels: no strict separator exists, but the >= 0 direction survives.
    g = _lattice()
    y = (g[:, 0] > 1e-3).astype(int)
    tie = g[np.isclose(g[:, 0], 1e-3)]
    lr = np.r_[g[:, 0], tie[:, 0]]
    alpha = np.r_[g[:, 1], tie[:, 1]]
    rank = np.r_[g[:, 2], tie[:, 2]]
    collapsed = np.r_[y, np.ones(len(tie), dtype=int)]
    assert separation_status(_frame(lr, alpha, rank, collapsed)) == "quasi"


def test_overlapping_labels_are_not_separable():
    g = _lattice()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, len(g))
    assert separation_status(_frame(g[:, 0], g[:, 1], g[:, 2], y)) == "none"


def _dense_lattice():
    # Fine log-grid so the max-margin plane is pinned by closely bracketing support cells.
    lrs = np.exp(np.linspace(np.log(1e-5), np.log(1e-2), 16))
    scales = np.exp(np.linspace(np.log(4), np.log(256), 12))
    return np.array([(lr, a, r) for lr in lrs for a in scales for r in scales])


def _margin_separated(gap=0.1):
    g = _dense_lattice()
    score = np.log(g[:, 0]) + 0.5 * np.log(g[:, 1]) - 0.05 * np.log(g[:, 2])
    cut = np.median(score)
    keep = np.abs(score - cut) > gap  # Empty band => strict margin separation.
    g, score = g[keep], score[keep]
    return _frame(g[:, 0], g[:, 1], g[:, 2], (score > cut).astype(int))


def test_max_margin_recovers_the_separating_plane():
    out = max_margin_ratios(_margin_separated())
    assert out["hard_margin"] is True and out["svm_kind"] == "hard_margin"
    assert abs(out["b"] - 0.5) < 0.02 and abs(out["c"] + 0.05) < 0.02
    assert out["n_support"] >= 2 and out["C"] == 1e4


def test_max_margin_on_overlapping_labels_is_soft():
    g = _lattice()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, len(g))
    out = max_margin_ratios(_frame(g[:, 0], g[:, 1], g[:, 2], y))
    assert out["hard_margin"] is False and out["svm_kind"] == "soft_margin_C10000"
    assert np.isfinite([out["b"], out["c"]]).all()


def test_separating_set_contains_the_generating_plane():
    s = separating_exponent_set(_margin_separated())
    assert s["b_set_lo"] <= 0.5 <= s["b_set_hi"]
    assert s["c_set_lo"] <= -0.05 <= s["c_set_hi"]


def test_separating_set_is_nan_without_separation():
    g = _lattice()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, len(g))
    s = separating_exponent_set(_frame(g[:, 0], g[:, 1], g[:, 2], y))
    assert all(np.isnan(v) for v in s.values())


def _quasi_with_ties(n_tied=3):
    # Margin-separated, plus a few cells placed exactly on the separating plane with
    # both labels: quasi-complete overall, completely separated once the ties go.
    d = _margin_separated()
    g = _dense_lattice()
    score = np.log(g[:, 0]) + 0.5 * np.log(g[:, 1]) - 0.05 * np.log(g[:, 2])
    cut = np.median(score)
    on = g[np.argsort(np.abs(score - cut))[:n_tied]]
    # Both labels at the same cell: every separating plane must pass through it.
    ties = pd.concat(
        [_frame(on[:, 0], on[:, 1], on[:, 2], np.full(len(on), label))
         for label in (0, 1)],
        ignore_index=True,
    )
    return pd.concat([d, ties], ignore_index=True)


def test_tied_cells_are_removed_before_the_hard_margin_fit():
    d = _quasi_with_ties()
    assert separation_status(d) == "quasi"
    assert int(tied_cells(d).sum()) == 6  # Both labels at each of the three cells.
    out = max_margin_ratios(d)
    assert out["svm_kind"] == "hard_margin_ties_removed"
    assert out["n_tied"] == 6 and out["hard_margin"] is True
    assert abs(out["b"] - 0.5) < 0.02 and abs(out["c"] + 0.05) < 0.02
