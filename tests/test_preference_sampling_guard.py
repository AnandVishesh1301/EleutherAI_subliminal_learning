"""The non-finite-logits guard in `sample_completions`.

A diverged adapter emits inf/NaN logits. Left alone they reach `torch.multinomial`, which
validates on the DEVICE and fires a device-side assert that poisons the CUDA context --
killing co-tenant jobs on the same GPU, not just the diverged one. The guard must reject
on the host first, as an ordinary Python exception.
"""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.evaluation.preference_sampling_io import (
    NonFiniteLogitsError,
    _RejectNonFiniteLogits,
)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_logits(bad):
    guard = _RejectNonFiniteLogits()
    ids = torch.zeros((2, 5), dtype=torch.long)
    scores = torch.randn(2, 10)
    scores[1, 3] = bad
    with pytest.raises(NonFiniteLogitsError) as e:
        guard(ids, scores)
    # the message must locate the damage: how many, and how far into generation
    assert "1 of 20" in str(e.value)
    assert "step 5" in str(e.value)


def test_passes_finite_logits_through_unchanged():
    guard = _RejectNonFiniteLogits()
    ids = torch.zeros((3, 4), dtype=torch.long)
    scores = torch.randn(3, 7)
    out = guard(ids, scores)
    assert out is scores
    assert torch.equal(out, scores)


def test_large_but_finite_logits_are_allowed():
    """Divergence is not the same as overflow: huge finite logits still sample fine.

    softmax subtracts the max before exponentiating, so it is stable for any finite input.
    Rejecting merely-large logits would fail healthy-but-extreme cells.
    """
    guard = _RejectNonFiniteLogits()
    ids = torch.zeros((1, 2), dtype=torch.long)
    scores = torch.full((1, 5), 1e30)
    scores[0, 0] = -1e30
    guard(ids, scores)  # must not raise
    probs = torch.softmax(scores, dim=-1)
    assert torch.isfinite(probs).all()
    torch.multinomial(probs, 1)  # the call the guard exists to protect


def test_guard_fires_before_multinomial_would():
    """Order matters: the guard must reject input that multinomial rejects."""
    scores = torch.tensor([[1.0, float("nan"), 2.0]])
    probs = torch.softmax(scores, dim=-1)
    assert torch.isnan(probs).any()  # what multinomial would choke on
    with pytest.raises(RuntimeError):
        torch.multinomial(probs, 1)
    with pytest.raises(NonFiniteLogitsError):
        _RejectNonFiniteLogits()(torch.zeros((1, 1), dtype=torch.long), scores)
