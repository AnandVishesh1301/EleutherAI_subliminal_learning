"""Parameter cosines and the linear-interpolation error barrier."""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.cka.pairing import BasisValidity
from subliminal_learning.cka.weight_space import (
    barrier_from_curve,
    cosine_of_flat,
    parameter_space_reason,
)

_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _params(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "layer.0.weight": torch.randn(4, 3, generator=g, dtype=torch.float64),
        "layer.0.bias": torch.randn(4, generator=g, dtype=torch.float64),
        "layer.1.weight": torch.randn(2, 4, generator=g, dtype=torch.float64),
    }


def test_cosine_of_identical_parameters_is_one():
    p = _params(0)
    assert cosine_of_flat(p, p) == pytest.approx(1.0, abs=1e-12)


def test_cosine_of_negated_parameters_is_minus_one():
    p = _params(1)
    assert cosine_of_flat(p, {k: -v for k, v in p.items()}) == pytest.approx(
        -1.0, abs=1e-12
    )


def test_cosine_is_scale_invariant():
    a, b = _params(2), _params(3)
    scaled = {k: 17.0 * v for k, v in b.items()}
    assert cosine_of_flat(a, scaled) == pytest.approx(cosine_of_flat(a, b), abs=1e-12)


def test_cosine_is_zero_for_orthogonal_parameters():
    a = {"w": torch.tensor([1.0, 0.0, 1.0, 0.0])}
    b = {"w": torch.tensor([0.0, 1.0, 0.0, 1.0])}
    assert cosine_of_flat(a, b) == pytest.approx(0.0, abs=1e-12)


def test_cosine_is_zero_against_an_all_zero_side():
    a = _params(4)
    assert cosine_of_flat(a, {k: torch.zeros_like(v) for k, v in a.items()}) == 0.0


def test_cosine_uses_sorted_key_order_so_dict_insertion_order_is_irrelevant():
    a = _params(5)
    b = _params(6)
    shuffled = dict(reversed(list(b.items())))
    assert cosine_of_flat(a, shuffled) == pytest.approx(cosine_of_flat(a, b), abs=1e-12)


def test_cosine_accepts_bare_tensors():
    x = torch.tensor([[3.0, 4.0]])
    assert cosine_of_flat(x, x) == pytest.approx(1.0, abs=1e-12)


def test_cosine_rejects_mismatched_parameter_sets():
    a = _params(7)
    with pytest.raises(ValueError, match="parameter key mismatch"):
        cosine_of_flat(a, {k: v for k, v in a.items() if k != "layer.0.bias"})
    reshaped = dict(a)
    reshaped["layer.0.bias"] = torch.zeros(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_of_flat(a, reshaped)
    with pytest.raises(TypeError, match="two tensors or two mappings"):
        cosine_of_flat(a, torch.zeros(3))


def test_barrier_is_zero_for_a_linear_curve():
    losses = [1.0 + 3.0 * a for a in _ALPHAS]
    assert barrier_from_curve(_ALPHAS, losses) == pytest.approx(0.0, abs=1e-12)


def test_barrier_is_zero_for_a_linear_curve_with_asymmetric_endpoints():
    losses = [2.0 - 1.5 * a for a in _ALPHAS]
    assert barrier_from_curve(_ALPHAS, losses) == pytest.approx(0.0, abs=1e-12)


def test_barrier_is_the_max_height_above_the_chord():
    losses = [0.0, 0.5, 0.9, 0.5, 0.0]
    assert barrier_from_curve(_ALPHAS, losses) == pytest.approx(0.9)


def test_barrier_measures_from_the_chord_not_the_endpoint_mean():
    # Chord at alpha=0.5 is 0.5; the endpoint mean is also 0.5 here, so the two
    # agree; shifting L(1) to 2.0 moves the chord but not an endpoint-mean rule.
    assert barrier_from_curve([0.0, 0.5, 1.0], [0.0, 1.5, 2.0]) == pytest.approx(0.5)


def test_barrier_is_never_negative_for_a_dipping_curve():
    assert barrier_from_curve(_ALPHAS, [1.0, 0.2, 0.0, 0.2, 1.0]) == pytest.approx(0.0)


def test_barrier_requires_both_endpoints():
    with pytest.raises(ValueError, match="endpoints"):
        barrier_from_curve([0.0, 0.5], [1.0, 2.0])
    with pytest.raises(ValueError, match="length mismatch"):
        barrier_from_curve([0.0, 0.5, 1.0], [1.0, 2.0])


@pytest.mark.parametrize(
    ("validity", "readable"),
    [
        (BasisValidity.BASIS_SHARED, True),
        (BasisValidity.BASIS_DIVERGED, True),
        (BasisValidity.BASIS_LINEAGE, True),
        (BasisValidity.BASIS_BROKEN, False),
    ],
)
def test_parameter_space_is_gated_on_a_shared_weight_init(
    validity: BasisValidity, readable: bool
):
    reason = parameter_space_reason(validity)
    assert (reason is None) is readable
    if not readable:
        assert reason == "basis_broken"
