"""Davari et al. high-norm deletion: what gets dropped, and how many."""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.cka.metrics import high_norm_mask


def _stack(seed: int, slots: int, n: int, p: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(slots, n, p, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("frac", [0.0, 0.01, 0.1, 0.25, 0.5])
def test_drops_exactly_floor_n_times_frac(frac: float):
    n = 200
    xs, ys = _stack(0, 3, n, 6), _stack(1, 4, n, 6)
    mask = high_norm_mask(xs, ys, frac)
    assert mask.shape == (n,)
    assert int((~mask).sum()) == int(n * frac)


def test_ranks_by_the_max_norm_across_both_models_and_all_slots():
    n = 50
    xs, ys = _stack(2, 3, n, 6), _stack(3, 5, n, 6)
    xs *= 0.01
    ys *= 0.01
    outlier_row = 17
    ys[3, outlier_row, :] = 1000.0
    mask = high_norm_mask(xs, ys, frac=0.02)
    assert int((~mask).sum()) == 1
    assert not bool(mask[outlier_row])


def test_a_model_a_outlier_is_dropped_too():
    n = 50
    xs, ys = _stack(4, 2, n, 6), _stack(5, 2, n, 6)
    xs *= 0.01
    ys *= 0.01
    xs[1, 8, :] = 1000.0
    mask = high_norm_mask(xs, ys, frac=0.02)
    assert not bool(mask[8])


@pytest.mark.parametrize("frac", [-0.01, 1.0, 1.5])
def test_frac_out_of_range_raises(frac: float):
    xs, ys = _stack(6, 2, 20, 4), _stack(7, 2, 20, 4)
    with pytest.raises(ValueError, match=r"frac must be in \[0, 1\)"):
        high_norm_mask(xs, ys, frac)


def test_mismatched_example_count_raises():
    xs, ys = _stack(8, 2, 20, 4), _stack(9, 2, 19, 4)
    with pytest.raises(ValueError, match="same examples required"):
        high_norm_mask(xs, ys, 0.1)
