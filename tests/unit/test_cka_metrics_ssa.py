"""Subspace-angle conventions: degrees, ascending, and the scipy wrapper's edges."""

from __future__ import annotations

import math

import pytest
import torch

from subliminal_learning.cka.metrics import (
    SSA_REDUCTIONS,
    principal_subspace,
    ssa_reduce,
    subspace_angles_between,
    subspace_angles_topk,
)


def _rep(seed: int, n: int = 64, p: int = 10) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, p, generator=g, dtype=torch.float64)


def test_identical_subspaces_give_zero_angles():
    x = _rep(seed=0)
    angles = subspace_angles_topk(x, x, k=4)
    assert angles.shape == (4,)
    assert torch.allclose(angles, torch.zeros(4, dtype=torch.float64), atol=1e-8)


def test_orthogonal_complements_give_ninety_degrees():
    eye = torch.eye(8, dtype=torch.float64)
    angles = subspace_angles_between(eye[:, :3], eye[:, 3:6])
    assert torch.allclose(
        angles, torch.full((3,), 90.0, dtype=torch.float64), atol=1e-8
    )


def test_angles_are_degrees_and_ascending():
    """Guards the scipy convention, which is radians in DESCENDING order."""
    x, y = _rep(seed=1), _rep(seed=2)
    angles = subspace_angles_topk(x, y, k=5)
    assert torch.all(angles[1:] >= angles[:-1] - 1e-12)
    assert float(angles.max()) > math.pi  # degrees, not radians
    assert 0.0 <= float(angles.min()) and float(angles.max()) <= 90.0 + 1e-9


def test_k_one_equals_arccos_of_the_absolute_pc1_cosine():
    x, y = _rep(seed=3), _rep(seed=4)
    pc1_x = principal_subspace(x, 1)[:, 0]
    pc1_y = principal_subspace(y, 1)[:, 0]
    cosine = float(torch.abs(pc1_x @ pc1_y))
    want = math.degrees(math.acos(min(cosine, 1.0)))
    assert float(subspace_angles_topk(x, y, k=1)[0]) == pytest.approx(want, abs=1e-8)


@pytest.mark.parametrize("k", [0, -1, 11])
def test_k_out_of_range_raises(k: int):
    x, y = _rep(seed=5), _rep(seed=6)
    with pytest.raises(ValueError, match="k must be in"):
        subspace_angles_topk(x, y, k=k)


def test_ssa_reduce_covers_the_declared_reductions():
    angles = torch.tensor([10.0, 20.0, 60.0], dtype=torch.float64)
    assert ssa_reduce(angles, "mean") == pytest.approx(30.0)
    assert ssa_reduce(angles, "max") == pytest.approx(60.0)
    assert ssa_reduce(angles, "min") == pytest.approx(10.0)
    assert set(SSA_REDUCTIONS) == {"mean", "max", "min"}


def test_ssa_reduce_rejects_unknown_reduction():
    angles = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="reduction must be one of"):
        ssa_reduce(angles, "median")
