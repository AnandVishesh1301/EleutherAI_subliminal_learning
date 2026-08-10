"""Orthogonal Procrustes distance against Ding et al.'s released implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from subliminal_learning.cka.metrics import procrustes_distance

_SIM_METRIC_SCORING = Path.home() / "eleutherAI" / "sim_metric" / "dists" / "scoring.py"


def _reference_procrustes(rep1: np.ndarray, rep2: np.ndarray) -> float:
    """Verbatim transcription of the released `sim_metric` path, on ``[p, n]`` inputs.

    Preprocessing from ``sim_metric/dists/score_pair.py:114-120`` (centre each row of the
    ``p x n`` matrix, then divide by the whole-matrix Frobenius norm); distance from
    ``sim_metric/dists/scoring.py:116-123``.
    """
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)
    return float(
        np.sum(rep1**2) + np.sum(rep2**2) - 2 * np.linalg.norm(rep1 @ rep2.T, ord="nuc")
    )


def _pair(seed: int, n: int = 64, p: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, generator=g, dtype=torch.float64)
    y = 0.6 * x + 0.4 * torch.randn(n, p, generator=g, dtype=torch.float64)
    return x, y


def _random_orthogonal(p: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(p, p, generator=g, dtype=torch.float64))
    return q


def test_procrustes_identical_inputs_is_zero():
    x, _ = _pair(seed=0)
    assert procrustes_distance(x, x) == pytest.approx(0.0, abs=1e-12)


def test_procrustes_orthogonal_rotation_is_zero():
    x, _ = _pair(seed=1)
    rotated = x @ _random_orthogonal(x.shape[1], seed=7)
    assert procrustes_distance(x, rotated) == pytest.approx(0.0, abs=1e-10)


def test_procrustes_isotropic_scaling_is_zero():
    x, _ = _pair(seed=2)
    assert procrustes_distance(x, 37.5 * x) == pytest.approx(0.0, abs=1e-10)


def test_procrustes_matches_sim_metric_reference_on_transposed_input():
    x, y = _pair(seed=3)
    want = _reference_procrustes(x.numpy().T.copy(), y.numpy().T.copy())
    assert procrustes_distance(x, y) == pytest.approx(want, abs=1e-10)


@pytest.mark.skipif(
    not _SIM_METRIC_SCORING.exists(),
    reason="sim_metric reference clone not on this host",
)
def test_transcribed_reference_matches_the_released_clone():
    """Guards the transcription above against the actual file it cites."""
    spec = importlib.util.spec_from_file_location(
        "sim_metric_scoring", _SIM_METRIC_SCORING
    )
    assert spec is not None and spec.loader is not None
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)

    x, y = _pair(seed=4)
    rep1 = x.numpy().T.copy()
    rep2 = y.numpy().T.copy()
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)
    assert scoring.procrustes(rep1, rep2) == pytest.approx(
        _reference_procrustes(x.numpy().T.copy(), y.numpy().T.copy()), abs=1e-12
    )


@pytest.mark.parametrize("seed", [5, 6, 7])
def test_procrustes_stays_within_zero_and_two(seed: int):
    x, y = _pair(seed=seed)
    assert 0.0 <= procrustes_distance(x, y) <= 2.0
    assert procrustes_distance(x, -x) <= 2.0


def test_procrustes_mismatched_n_raises():
    g = torch.Generator().manual_seed(8)
    x = torch.randn(32, 6, generator=g)
    y = torch.randn(31, 6, generator=g)
    with pytest.raises(ValueError, match="SAME examples"):
        procrustes_distance(x, y)
