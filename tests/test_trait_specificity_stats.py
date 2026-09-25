"""Unit tests for the trait-specificity cluster bootstraps (difference of means + ratio)."""
from __future__ import annotations

import numpy as np

from subliminal_learning.stats.core import cluster_bootstrap_mean_difference as cbd
from subliminal_learning.stats.core import cluster_bootstrap_ratio_of_medians as cb


def test_ci_covers_true_ratio():
    rng = np.random.default_rng(1)
    # 10 clusters x 30 cells, treatment median ~3x the control median.
    clusters = np.repeat(np.arange(10), 30)
    offs = rng.normal(0.0, 0.25, size=10)[clusters]   # per-cluster (teacher) random effect
    treat = np.exp(np.log(0.3) + offs + rng.normal(0.0, 0.4, size=clusters.size))
    ctrl = rng.lognormal(np.log(0.1), 0.4, size=30)
    ratio, lo, hi = cb(treat, clusters, ctrl, n_boot=500, seed=0)
    assert lo < ratio < hi
    assert lo < 3.0 < hi


def test_single_cluster_gives_degenerate_treatment_side():
    rng = np.random.default_rng(2)
    treat = rng.lognormal(0.0, 0.5, size=30)
    ctrl = np.full(7, 2.0)  # constant control -> all CI width must come from the treatment side
    one, lo1, hi1 = cb(treat, np.zeros(30), ctrl, n_boot=400, seed=0)
    assert np.isclose(lo1, one) and np.isclose(hi1, one)  # one cluster resamples to itself
    _, lo2, hi2 = cb(treat, np.arange(30), ctrl, n_boot=400, seed=0)
    assert hi2 - lo2 > 0  # 30 clusters (one row each) -> ordinary row bootstrap width


def test_empty_or_zero_control_is_nan():
    assert all(np.isnan(v) for v in cb([], [], [1.0]))
    assert all(np.isnan(v) for v in cb([1.0], [0], [0.0, 0.0]))


def test_diff_ci_covers_true_difference():
    rng = np.random.default_rng(3)
    # 10 clusters x 30 cells; treatment mean 0.02 above the control mean, per-cluster random effect.
    clusters = np.repeat(np.arange(10), 30)
    offs = rng.normal(0.0, 0.004, size=10)[clusters]  # per-cluster (teacher) random effect
    treat = 0.12 + offs + rng.normal(0.0, 0.003, size=clusters.size)
    ctrl = 0.10 + rng.normal(0.0, 0.003, size=30)
    diff, lo, hi = cbd(treat, clusters, ctrl, n_boot=2000, seed=0)
    assert lo < diff < hi
    assert lo < 0.02 < hi


def test_diff_single_cluster_is_degenerate_on_the_treatment_side():
    rng = np.random.default_rng(4)
    treat = rng.normal(0.1, 0.02, size=30)
    ctrl = np.full(7, 0.05)  # constant control -> all CI width must come from the treatment side
    one, lo1, hi1 = cbd(treat, np.zeros(30), ctrl, n_boot=400, seed=0)
    assert np.isclose(lo1, one) and np.isclose(hi1, one)  # one cluster resamples to itself
    _, lo2, hi2 = cbd(treat, np.arange(30), ctrl, n_boot=400, seed=0)
    assert hi2 - lo2 > 0  # 30 clusters (one row each) -> ordinary row bootstrap width


def test_diff_empty_side_is_nan():
    assert all(np.isnan(v) for v in cbd([], [], [1.0]))
    assert all(np.isnan(v) for v in cbd([1.0], [0], []))
