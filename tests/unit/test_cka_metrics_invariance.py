"""Kornblith Table 1 invariance properties of linear CKA and the CCA panel."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from subliminal_learning.cka.metrics import (
    cca_similarity,
    linear_cka_biased,
    pwcca_similarity,
)

_SIM_METRIC_SCORING = Path.home() / "eleutherAI" / "sim_metric" / "dists" / "scoring.py"


def _pair(seed: int, n: int = 96, p: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Correlated-but-not-identical representations, so CKA sits strictly in (0, 1)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, generator=g, dtype=torch.float64)
    mix = torch.randn(p, p, generator=g, dtype=torch.float64)
    y = x @ mix + 0.8 * torch.randn(n, p, generator=g, dtype=torch.float64)
    return x, y


def _random_orthogonal(p: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(p, p, generator=g, dtype=torch.float64))
    return q


def test_linear_cka_is_invariant_to_orthogonal_transform():
    x, y = _pair(seed=0)
    q = _random_orthogonal(x.shape[1], seed=11)
    assert linear_cka_biased(x @ q, y) == pytest.approx(
        linear_cka_biased(x, y), abs=1e-10
    )


def test_linear_cka_is_invariant_to_isotropic_scaling():
    x, y = _pair(seed=1)
    assert linear_cka_biased(4.5 * x, 0.02 * y) == pytest.approx(
        linear_cka_biased(x, y), abs=1e-10
    )


def test_linear_cka_is_not_invariant_to_invertible_linear_transform():
    """The property that separates CKA from the CCA family (Kornblith Table 1)."""
    x, y = _pair(seed=2)
    stretch = torch.eye(x.shape[1], dtype=torch.float64)
    stretch[0, 0] = 200.0
    baseline = linear_cka_biased(x, y)
    transformed = linear_cka_biased(x @ stretch, y)
    assert torch.linalg.det(stretch) != 0.0
    assert abs(transformed - baseline) > 0.05


def test_linear_cka_translation_does_not_inflate_the_score():
    """Centring happens inside, so a large shared offset cannot push CKA toward 1."""
    x, y = _pair(seed=3)
    offset_x = x + 500.0
    offset_y = y + 500.0
    assert linear_cka_biased(offset_x, offset_y) == pytest.approx(
        linear_cka_biased(x, y), abs=1e-8
    )
    assert linear_cka_biased(offset_x, offset_y) < 0.99


def test_linear_cka_identical_inputs_is_one():
    x, _ = _pair(seed=4)
    assert linear_cka_biased(x, x) == pytest.approx(1.0, abs=1e-12)


def _released_operands(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """``[p, n]`` operands preprocessed as ``sim_metric/dists/score_pair.py:114-120``."""
    out = []
    for m in (x.numpy().T.copy(), y.numpy().T.copy()):
        m = m - m.mean(axis=1, keepdims=True)
        out.append(m / np.linalg.norm(m))
    return out[0], out[1]


def _load_released_scoring():
    spec = importlib.util.spec_from_file_location(
        "sim_metric_scoring", _SIM_METRIC_SCORING
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cca_is_invariant_to_invertible_linear_transform():
    """The CCA-family property CKA deliberately gives up (Kornblith Table 1)."""
    x, y = _pair(seed=5)
    stretch = torch.eye(x.shape[1], dtype=torch.float64)
    stretch[0, 0] = 200.0
    assert cca_similarity(x @ stretch, y) == pytest.approx(
        cca_similarity(x, y), abs=1e-8
    )


def test_cca_and_pwcca_of_identical_inputs_are_one():
    x, _ = _pair(seed=6)
    assert cca_similarity(x, x) == pytest.approx(1.0, abs=1e-8)
    assert pwcca_similarity(x, x) == pytest.approx(1.0, abs=1e-8)


def test_cca_family_requires_more_examples_than_features():
    g = torch.Generator().manual_seed(7)
    wide = torch.randn(8, 8, generator=g, dtype=torch.float64)
    with pytest.raises(ValueError, match="CCA requires p < n"):
        cca_similarity(wide, wide)
    with pytest.raises(ValueError, match="CCA requires p < n"):
        pwcca_similarity(wide, wide)


@pytest.mark.skipif(
    not _SIM_METRIC_SCORING.exists(),
    reason="sim_metric reference clone not on this host",
)
def test_cca_family_matches_the_released_clone():
    scoring = _load_released_scoring()
    x, y = _pair(seed=8)
    rep1, rep2 = _released_operands(x, y)
    _, rho, _, transformed_1, _ = scoring.cca_decomp(rep1, rep2)
    assert cca_similarity(x, y) == pytest.approx(
        float(scoring.mean_sq_cca_corr(rho)), abs=1e-8
    )
    assert pwcca_similarity(x, y) == pytest.approx(
        1.0 - float(scoring.pwcca_dist(rep1, rho, transformed_1)), abs=1e-8
    )
