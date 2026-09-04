"""`models.seed_identity`: init equality and the untouched-embedding-row fingerprint."""
from __future__ import annotations

import numpy as np
import torch

from subliminal_learning.models import seed_identity as si


def test_init_equality_exact_bytes() -> None:
    a = {"x": torch.arange(6.0).reshape(2, 3), "y": torch.ones(4)}
    same = {"x": a["x"].clone(), "y": a["y"].clone()}
    r = si.init_equality(a, same)
    assert r.shared and r.n_equal == r.n_tensors == 2 and r.max_abs_diff_over_unequal == 0.0

    other = {"x": a["x"] + 1e-6, "y": a["y"].clone()}
    r = si.init_equality(a, other)
    assert not r.shared and r.equal_tensors == ("y",)
    assert abs(r.max_abs_diff_over_unequal - 1e-6) < 1e-9 and "x" in r.example_unequal

    # A tensor present on one side only counts as unequal.
    r = si.init_equality(a, {"x": a["x"].clone()})
    assert not r.shared and r.n_tensors == 2 and r.n_equal == 1


def _simulate(rng, vocab, pad, d, touched, decay=0.98, noise=1e-3):
    """w0 -> wt: untouched rows only decay by `decay`, touched rows also get an update."""
    w0 = rng.normal(size=(vocab + pad, d)).astype(np.float32)
    wt = (w0 * decay).astype(np.float32)
    wt[touched] += rng.normal(scale=noise, size=(len(touched), d)).astype(np.float32)
    return torch.from_numpy(w0), torch.from_numpy(wt)


def test_untouched_rows_recovers_the_untouched_set_under_weight_decay() -> None:
    rng = np.random.default_rng(0)
    vocab, pad, d = 200, 8, 32
    touched = np.array(sorted(rng.choice(vocab, 150, replace=False)))
    w0, wt = _simulate(rng, vocab, pad, d, touched)
    r = si.untouched_rows(w0, wt, vocab)
    expected = sorted(set(range(vocab)) - set(touched.tolist()))
    assert list(r.untouched) == expected and r.n_untouched == 50
    assert r.pad_rows == 8 and r.control_ok(si.DEFAULT_TOL)
    # Untouched rows share the common decay factor.
    assert abs(r.untouched_scale_min - 0.98) < 1e-5 and abs(r.untouched_scale_max - 0.98) < 1e-5
    assert abs(r.pad_scale_min - 0.98) < 1e-5
    assert "untouched" not in r.summary() and r.summary()["n_untouched"] == 50


def test_compare_untouched_same_vs_different_order() -> None:
    rng = np.random.default_rng(1)
    vocab, pad, d = 200, 8, 16
    touched = np.array(sorted(rng.choice(vocab, 120, replace=False)))
    # Same batches, different inits: identical untouched sets -> SHARED order.
    a = si.untouched_rows(*_simulate(np.random.default_rng(2), vocab, pad, d, touched), vocab)
    b = si.untouched_rows(*_simulate(np.random.default_rng(3), vocab, pad, d, touched), vocab)
    cmp = si.compare_untouched(a, b)
    assert cmp.shared is True and cmp.verdict == "SHARED" and cmp.jaccard == 1.0

    other = np.array(sorted(rng.choice(vocab, 120, replace=False)))
    c = si.untouched_rows(*_simulate(np.random.default_rng(4), vocab, pad, d, other), vocab)
    cmp = si.compare_untouched(a, c)
    assert cmp.shared is False and cmp.verdict == "DIFFERENT" and cmp.jaccard < 1.0
    assert set(cmp.only_a).isdisjoint(cmp.only_b)

    # Saturated window (every row touched on both sides) is inconclusive, not "shared".
    full = np.arange(vocab)
    e = si.untouched_rows(*_simulate(np.random.default_rng(5), vocab, pad, d, full), vocab)
    f = si.untouched_rows(*_simulate(np.random.default_rng(6), vocab, pad, d, full), vocab)
    cmp = si.compare_untouched(e, f)
    assert cmp.shared is None and cmp.verdict == "INCONCLUSIVE" and np.isnan(cmp.jaccard)
