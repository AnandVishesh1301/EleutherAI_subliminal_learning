"""Trait-coordinate identities: rotation invariance and the k=1 CKA equivalence."""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.cka.metrics import (
    linear_cka_biased,
    trait_coordinate_similarity,
    trait_coordinates,
)


def _coords(seed: int, n: int = 128) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64)


@pytest.mark.parametrize(("scale", "shift"), [(1.0, 0.0), (3.7, -12.0), (-0.4, 5.0)])
def test_similarity_is_one_under_an_affine_relabel(scale: float, shift: float):
    cx = _coords(seed=0)
    assert trait_coordinate_similarity(cx, scale * cx + shift) == pytest.approx(
        1.0, abs=1e-12
    )


def test_similarity_equals_linear_cka_on_the_column_matrices():
    cx, cy = _coords(seed=1), _coords(seed=2)
    assert trait_coordinate_similarity(cx, cy) == pytest.approx(
        linear_cka_biased(cx[:, None], cy[:, None]), abs=1e-12
    )


def test_similarity_is_zero_for_a_constant_coordinate_vector():
    cx = _coords(seed=3)
    constant = torch.full_like(cx, 2.5)
    assert trait_coordinate_similarity(cx, constant) == pytest.approx(0.0, abs=1e-12)


def test_trait_coordinates_normalise_the_direction():
    g = torch.Generator().manual_seed(4)
    hidden = torch.randn(32, 9, generator=g, dtype=torch.float64)
    raw = torch.randn(9, generator=g, dtype=torch.float64)
    unit = raw / raw.norm()
    assert torch.allclose(
        trait_coordinates(hidden, raw), trait_coordinates(hidden, unit), atol=1e-12
    )
    assert torch.allclose(
        trait_coordinates(hidden, 100.0 * raw),
        trait_coordinates(hidden, unit),
        atol=1e-12,
    )


def test_trait_coordinates_project_onto_the_direction():
    hidden = torch.eye(4, dtype=torch.float64)
    direction = torch.tensor([0.0, 3.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(
        trait_coordinates(hidden, direction),
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_trait_coordinates_rejects_a_mismatched_direction():
    hidden = torch.zeros(8, 5)
    with pytest.raises(ValueError, match="direction must be"):
        trait_coordinates(hidden, torch.zeros(4))
