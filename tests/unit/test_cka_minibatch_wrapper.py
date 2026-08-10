"""The `cka.metrics` minibatch wrappers against the `cka.minibatch` SSOT."""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.cka.metrics import minibatch_cka, minibatch_cka_grid
from subliminal_learning.cka.minibatch import full_batch_unbiased_cka


def _correlated(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 10, generator=g, dtype=torch.float64)
    y = 0.5 * x[:, :6] + 0.5 * torch.randn(n, 6, generator=g, dtype=torch.float64)
    return x, y


def test_single_batch_matches_the_full_batch_reference():
    x, y = _correlated(n=64, seed=0)
    value, spread = minibatch_cka(
        x, y, batch_size=64, n_examples=64, n_repeats=1, seed=0
    )
    assert value == pytest.approx(float(full_batch_unbiased_cka(x, y)), abs=1e-5)
    assert spread == 0.0


def test_identical_inputs_reach_one():
    x, _ = _correlated(n=256, seed=1)
    value, _ = minibatch_cka(x, x, batch_size=64, n_examples=256, n_repeats=3, seed=0)
    assert value == pytest.approx(1.0, abs=1e-6)


def test_row_permutation_floors_the_value():
    x, y = _correlated(n=512, seed=2)
    g = torch.Generator().manual_seed(99)
    permuted = y[torch.randperm(y.shape[0], generator=g)]
    value, _ = minibatch_cka(
        x, permuted, batch_size=128, n_examples=512, n_repeats=4, seed=0
    )
    assert abs(value) < 0.1


def test_seeded_batch_draws_are_reproducible():
    x, y = _correlated(n=300, seed=3)
    kwargs = {"batch_size": 32, "n_examples": 256, "n_repeats": 5}
    first = minibatch_cka(x, y, seed=1234, **kwargs)
    again = minibatch_cka(x, y, seed=1234, **kwargs)
    other = minibatch_cka(x, y, seed=4321, **kwargs)
    assert first == again
    assert first[0] != other[0]


def test_repeat_spread_is_zero_for_a_single_repeat_and_positive_otherwise():
    x, y = _correlated(n=300, seed=4)
    _, single = minibatch_cka(x, y, batch_size=32, n_examples=256, n_repeats=1, seed=0)
    _, many = minibatch_cka(x, y, batch_size=32, n_examples=256, n_repeats=6, seed=0)
    assert single == 0.0
    assert many > 0.0


def test_grid_returns_one_entry_per_slot_pair():
    g = torch.Generator().manual_seed(5)
    xs = torch.randn(3, 128, 8, generator=g, dtype=torch.float64)
    ys = torch.randn(2, 128, 6, generator=g, dtype=torch.float64)
    value, spread = minibatch_cka_grid(
        xs, ys, batch_size=32, n_examples=128, n_repeats=2, seed=0
    )
    assert value.shape == (3, 2)
    assert spread.shape == (3, 2)


def test_grid_diagonal_is_one_when_both_sides_are_the_same_stack():
    g = torch.Generator().manual_seed(6)
    xs = torch.randn(3, 128, 8, generator=g, dtype=torch.float64)
    value, _ = minibatch_cka_grid(
        xs, xs, batch_size=64, n_examples=128, n_repeats=2, seed=0
    )
    assert torch.allclose(
        value.diagonal(), torch.ones(3, dtype=torch.float64), atol=1e-6
    )


def test_batch_size_below_the_u_statistic_minimum_raises():
    x, y = _correlated(n=64, seed=7)
    with pytest.raises(ValueError, match="batch_size must be >= 4"):
        minibatch_cka(x, y, batch_size=3, n_examples=64, n_repeats=1, seed=0)


def test_batch_larger_than_the_sample_raises():
    x, y = _correlated(n=16, seed=8)
    with pytest.raises(ValueError, match="exceeds the"):
        minibatch_cka(x, y, batch_size=32, n_examples=16, n_repeats=1, seed=0)


def test_mismatched_example_count_raises():
    x, _ = _correlated(n=64, seed=9)
    y, _ = _correlated(n=63, seed=10)
    with pytest.raises(ValueError, match="SAME examples"):
        minibatch_cka(x, y, batch_size=16, n_examples=63, n_repeats=1, seed=0)
