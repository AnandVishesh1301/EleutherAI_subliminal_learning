"""Probe readouts: planted signal, chance baselines, and the online codelength."""

from __future__ import annotations

import math

import numpy as np
import pytest

from subliminal_learning.cka.probes import (
    codelength_ratio,
    fit_linear_probe,
    prequential_codelength,
    probe_subspace,
)

_SEED = 20260805


def _planted(
    n: int, p: int, margin: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly separable data with a known direction and a guaranteed margin."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    x = rng.normal(size=(4 * n, p))
    score = x @ direction
    keep = np.abs(score) > margin
    x = x[keep][:n]
    y = (score[keep][:n] > 0).astype(int)
    return x, y, direction


def _noisy_labels(
    n: int, p: int, strength: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from a logistic model whose signal scales with `strength`."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    x = rng.normal(size=(n, p))
    probability = 1.0 / (1.0 + np.exp(-strength * (x @ direction)))
    y = (rng.random(n) < probability).astype(int)
    return x, y


def test_planted_separable_data_gives_perfect_accuracy_and_the_planted_direction():
    x, y, direction = _planted(n=500, p=8, margin=0.5, seed=_SEED)
    fit = fit_linear_probe(x, y, seed=_SEED)
    assert fit.accuracy == 1.0
    assert fit.macro_f1 == pytest.approx(1.0)
    probe = fit.weights[0]
    cosine = float(
        probe @ direction / (np.linalg.norm(probe) * np.linalg.norm(direction))
    )
    assert cosine > 0.9


def test_fit_is_deterministic_for_a_fixed_seed():
    x, y, _ = _planted(n=300, p=6, margin=0.4, seed=_SEED)
    first = fit_linear_probe(x, y, seed=7)
    again = fit_linear_probe(x, y, seed=7)
    assert np.allclose(first.weights, again.weights)
    assert first.accuracy == again.accuracy


def test_fit_reports_convergence_and_flags_a_truncated_descent():
    """A fit stopped by `max_iter` must say so -- its accuracy still looks fine.

    `probe_direction_cosine` reads the weight VECTOR, so a truncated fit there is
    a plausible wrong number rather than a visible failure.
    """
    x, y = _noisy_labels(n=600, p=10, strength=1.5, seed=_SEED)
    converged = fit_linear_probe(x, y, seed=_SEED)
    assert converged.converged
    assert 0 < converged.n_iter < 1000

    truncated = fit_linear_probe(x, y, seed=_SEED, max_iter=2)
    assert not truncated.converged
    assert truncated.n_iter == 2
    # The readout the P1c CSV persists is nearly unchanged -- which is exactly
    # why the direction consumer, not the accuracy consumer, has to check.
    assert truncated.accuracy == pytest.approx(converged.accuracy, abs=0.1)


def test_random_labels_give_chance_accuracy_and_the_uniform_codelength():
    rng = np.random.default_rng(_SEED)
    n, p = 800, 6
    x = rng.normal(size=(n, p))
    y = rng.integers(0, 2, size=n)
    fit = fit_linear_probe(x, y, seed=_SEED)
    assert 0.35 < fit.accuracy < 0.65

    code = prequential_codelength(x, y, seed=_SEED)
    assert code == pytest.approx(n * math.log2(2), rel=0.15)
    assert codelength_ratio(code, n, 2) == pytest.approx(1.0, rel=0.15)


def test_codelength_is_monotone_in_signal_strength():
    codes = [
        prequential_codelength(*_noisy_labels(800, 6, strength, seed=_SEED), seed=_SEED)
        for strength in (0.0, 1.0, 3.0, 10.0)
    ]
    assert codes == sorted(codes, reverse=True)
    assert codelength_ratio(codes[-1], 800, 2) < 0.5


def test_codelength_ratio_rejects_degenerate_baselines():
    with pytest.raises(ValueError, match="n must be >= 1"):
        codelength_ratio(10.0, 0, 2)
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        codelength_ratio(10.0, 10, 1)


def test_single_class_input_raises():
    x, _, _ = _planted(n=100, p=4, margin=0.3, seed=_SEED)
    ones = np.ones(x.shape[0], dtype=int)
    with pytest.raises(ValueError, match="at least 2 classes"):
        fit_linear_probe(x, ones, seed=_SEED)
    with pytest.raises(ValueError, match="at least 2 classes"):
        prequential_codelength(x, ones, seed=_SEED)


def test_probe_subspace_is_orthonormal_and_spans_the_weight_rows():
    x, y, direction = _planted(n=400, p=8, margin=0.5, seed=_SEED)
    fit = fit_linear_probe(x, y, seed=_SEED)
    basis = probe_subspace(fit.weights, k=1)
    assert basis.shape == (8, 1)
    assert float(basis[:, 0] @ basis[:, 0]) == pytest.approx(1.0, abs=1e-10)
    assert abs(float(basis[:, 0].numpy() @ direction)) > 0.9


def test_probe_subspace_rejects_out_of_range_k():
    with pytest.raises(ValueError, match="k must be in"):
        probe_subspace(np.zeros((3, 5)), k=4)
