"""Validation for the moment-based Procrustes / PWCCA path.

Every metric is checked against an INDEPENDENT reference computed from the raw
``n x d`` rows by a different numerical route (direct residual minimization for
Procrustes; QR rather than eigendecomposition for the CCA family), so agreement is
evidence about the moment reduction itself and not a restatement of the same formula.
"""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.similarity.metrics import (
    build_whitener,
    canonical_correlations,
    procrustes,
    pwcca,
)
from subliminal_learning.similarity.moments import ModelMoments, PairMoments

N, DIM, LAYERS = 240, 12, 3


def _rand(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0)


def _accumulate(
    xs: list[torch.Tensor], ys: list[torch.Tensor], n_batches: int, full_grid: bool = False
):
    """Stream per-layer rows through the accumulators in ``n_batches`` chunks."""
    ma = ModelMoments(len(xs), xs[0].shape[1])
    mb = ModelMoments(len(ys), ys[0].shape[1])
    pm = PairMoments(len(xs), xs[0].shape[1], ys[0].shape[1], full_grid=full_grid)
    for chunk in torch.arange(xs[0].shape[0]).chunk(n_batches):
        rows_a = [x[chunk] for x in xs]
        rows_b = [y[chunk] for y in ys]
        ma.update(rows_a)
        mb.update(rows_b)
        pm.update(rows_a, rows_b)
    return ma, mb, pm


# --------------------------------------------------------------------------- #
# Independent references, computed from raw rows.
# --------------------------------------------------------------------------- #


def _ref_procrustes_sq(x: torch.Tensor, y: torch.Tensor) -> float:
    """Directly minimize ||X - Y Q||_F over orthogonal Q and return the residual.

    Deliberately avoids the ``tr(A) + tr(B) - 2||C||_*`` closed form under test: it
    solves for the optimal rotation and evaluates the residual norm itself.
    """
    xc, yc = _center(x), _center(y)
    u, _, vh = torch.linalg.svd(yc.T @ xc, full_matrices=False)
    q = u @ vh
    return float(((xc - yc @ q) ** 2).sum())


def _ref_cca(x: torch.Tensor, y: torch.Tensor):
    """Canonical correlations, variates and PWCCA weights via QR on the raw rows."""
    xc, yc = _center(x), _center(y)
    qx, _ = torch.linalg.qr(xc, mode="reduced")
    qy, _ = torch.linalg.qr(yc, mode="reduced")
    u, s, _ = torch.linalg.svd(qx.T @ qy, full_matrices=False)
    variates = qx @ u  # [n, k], orthonormal columns
    weights = (variates.T @ xc).abs().sum(dim=1)  # sum over NEURONS
    return s.clamp(0.0, 1.0), weights


def _ref_pwcca(x: torch.Tensor, y: torch.Tensor) -> float:
    rho, w = _ref_cca(x, y)
    return float((w * rho).sum() / w.sum())


# --------------------------------------------------------------------------- #
# Moment accumulation.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_batches", [1, 7, 24])
def test_streaming_moments_are_exact(n_batches):
    """Batched accumulation must equal the one-shot centered moments exactly."""
    xs = [_rand(N, DIM, 1)]
    ys = [_rand(N, DIM, 2)]
    ma, mb, pm = _accumulate(xs, ys, n_batches)
    xc, yc = _center(xs[0]), _center(ys[0])
    assert torch.allclose(ma.centered_gram(0), xc.T @ xc, atol=1e-9)
    assert torch.allclose(mb.centered_gram(0), yc.T @ yc, atol=1e-9)
    assert torch.allclose(pm.centered_cross(ma, mb, 0, 0), xc.T @ yc, atol=1e-9)


def test_full_grid_matches_diagonal_and_offdiagonal():
    xs = [_rand(N, DIM, 10 + i) for i in range(LAYERS)]
    ys = [_rand(N, DIM, 20 + i) for i in range(LAYERS)]
    ma, mb, pm = _accumulate(xs, ys, 5, full_grid=True)
    for i in range(LAYERS):
        for j in range(LAYERS):
            want = _center(xs[i]).T @ _center(ys[j])
            assert torch.allclose(pm.centered_cross(ma, mb, i, j), want, atol=1e-9)


def test_offdiagonal_requires_full_grid():
    xs = [_rand(N, DIM, 3) for _ in range(LAYERS)]
    ys = [_rand(N, DIM, 4) for _ in range(LAYERS)]
    ma, mb, pm = _accumulate(xs, ys, 3)
    with pytest.raises(ValueError, match="full_grid"):
        pm.centered_cross(ma, mb, 0, 1)


def test_row_count_mismatch_is_rejected():
    ma = ModelMoments(1, DIM)
    mb = ModelMoments(1, DIM)
    pm = PairMoments(1, DIM, DIM)
    x, y = _rand(N, DIM, 5), _rand(N, DIM, 6)
    ma.update([x])
    mb.update([y])
    pm.update([x], [y])
    ma.update([x])  # A now has twice the rows of the pair accumulator
    with pytest.raises(ValueError, match="row counts disagree"):
        pm.centered_cross(ma, mb, 0, 0)


# --------------------------------------------------------------------------- #
# Procrustes.
# --------------------------------------------------------------------------- #


def test_procrustes_matches_direct_residual():
    x, y = _rand(N, DIM, 31), _rand(N, DIM, 32)
    ma, mb, pm = _accumulate([x], [y], 8)
    got = procrustes(ma.centered_gram(0), mb.centered_gram(0), pm.centered_cross(ma, mb, 0, 0))
    assert got["squared_distance"] == pytest.approx(_ref_procrustes_sq(x, y), rel=1e-9)


def test_procrustes_zero_on_rotation_and_translation():
    """Exactly zero distance under the invariances Procrustes is defined to quotient."""
    x = _rand(N, DIM, 41)
    q, _ = torch.linalg.qr(_rand(DIM, DIM, 42))
    y = x @ q + 3.7
    ma, mb, pm = _accumulate([x], [y], 4)
    got = procrustes(ma.centered_gram(0), mb.centered_gram(0), pm.centered_cross(ma, mb, 0, 0))
    # The closed form subtracts two O(n*d) quantities, so "zero" is only zero to
    # machine precision relative to that scale, not absolutely.
    assert got["squared_distance"] / got["nuclear_norm"] < 1e-12
    assert got["angular_distance"] == pytest.approx(0.0, abs=1e-7)


def test_procrustes_raw_is_scale_sensitive_but_normalized_is_not():
    """The documented distinction: only the normalized forms quotient out scale."""
    x = _rand(N, DIM, 51)
    q, _ = torch.linalg.qr(_rand(DIM, DIM, 52))
    y = 4.0 * (x @ q)
    ma, mb, pm = _accumulate([x], [y], 4)
    got = procrustes(ma.centered_gram(0), mb.centered_gram(0), pm.centered_cross(ma, mb, 0, 0))
    assert got["squared_distance"] > 1.0
    assert got["normalized_distance"] == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------- #
# CCA family.
# --------------------------------------------------------------------------- #


def test_canonical_correlations_match_qr_reference():
    x, y = _rand(N, DIM, 61), _rand(N, DIM, 62)
    ma, mb, pm = _accumulate([x], [y], 6)
    wa = build_whitener(ma.centered_gram(0))
    wb = build_whitener(mb.centered_gram(0))
    rho, _ = canonical_correlations(wa, wb, pm.centered_cross(ma, mb, 0, 0))
    want, _ = _ref_cca(x, y)
    assert torch.allclose(rho, want, atol=1e-8)


def test_pwcca_matches_qr_reference():
    x, y = _rand(N, DIM, 71), _rand(N, DIM, 72)
    ma, mb, pm = _accumulate([x], [y], 6)
    wa = build_whitener(ma.centered_gram(0))
    wb = build_whitener(mb.centered_gram(0))
    got = pwcca(wa, wb, pm.centered_cross(ma, mb, 0, 0))
    assert got["pwcca"] == pytest.approx(_ref_pwcca(x, y), rel=1e-7)
    assert got["cca_mean"] == pytest.approx(float(_ref_cca(x, y)[0].mean()), rel=1e-8)


def test_pwcca_differs_from_unweighted_cca():
    """Guards against a weighting bug that silently collapses to the plain mean."""
    # Anisotropic X: the weighting is driven by X's own spectrum, so a flat spectrum
    # would make PWCCA and plain CCA agree for reasons unrelated to correctness.
    scale = torch.logspace(0, -2, DIM, dtype=torch.float64)
    x = _rand(N, DIM, 81) * scale
    y = x @ _rand(DIM, DIM, 83) + _rand(N, DIM, 82) * 0.5
    ma, mb, pm = _accumulate([x], [y], 4)
    wa = build_whitener(ma.centered_gram(0))
    wb = build_whitener(mb.centered_gram(0))
    got = pwcca(wa, wb, pm.centered_cross(ma, mb, 0, 0))
    assert got["min_rho_gap"] > 1e-6, "construction must stay non-degenerate"
    assert abs(got["pwcca"] - got["cca_mean"]) > 1e-3
    assert got["pwcca"] == pytest.approx(_ref_pwcca(x, y), rel=1e-7)


def test_tied_correlations_are_flagged_as_ill_posed():
    """Tied canonical correlations leave PWCCA basis-dependent — surface, don't hide.

    The correlations themselves stay exact; only the weight mass on the tied block is
    undetermined, so the reported value may legitimately differ from another correct
    implementation. `min_rho_gap` is what tells the caller that happened.
    """
    x = _rand(N, DIM, 81)
    y = torch.cat([x[:, :3] * 2.0, _rand(N, DIM - 3, 82) * 0.1], dim=1)  # three exact rho=1
    ma, mb, pm = _accumulate([x], [y], 4)
    wa = build_whitener(ma.centered_gram(0))
    wb = build_whitener(mb.centered_gram(0))
    cross = pm.centered_cross(ma, mb, 0, 0)

    rho, _ = canonical_correlations(wa, wb, cross)
    ref_rho, _ = _ref_cca(x, y)
    assert torch.allclose(rho, ref_rho, atol=1e-9)  # correlations: still exact
    assert float(rho[2]) == pytest.approx(1.0, abs=1e-9)

    got = pwcca(wa, wb, cross)
    assert got["min_rho_gap"] < 1e-9  # the tie is reported...
    assert got["pwcca"] != pytest.approx(_ref_pwcca(x, y), rel=1e-3)  # ...and it bites


def test_pwcca_is_asymmetric():
    x, y = _rand(N, DIM, 91), _rand(N, DIM, 92) + _rand(N, DIM, 91) * 0.5
    ma, mb, pm = _accumulate([x], [y], 4)
    wa = build_whitener(ma.centered_gram(0))
    wb = build_whitener(mb.centered_gram(0))
    cross = pm.centered_cross(ma, mb, 0, 0)
    assert pwcca(wa, wb, cross)["pwcca"] != pytest.approx(
        pwcca(wb, wa, cross.T)["pwcca"], rel=1e-6
    )


def test_cca_invariant_to_invertible_linear_map():
    """The documented caveat, asserted: CCA discards invertible linear structure."""
    x, y = _rand(N, DIM, 101), _rand(N, DIM, 102)
    t = _rand(DIM, DIM, 103)
    assert float(torch.linalg.det(t).abs()) > 1e-6
    ma, mb, pm = _accumulate([x], [y], 4)
    ma2, mb2, pm2 = _accumulate([x @ t], [y], 4)
    rho, _ = canonical_correlations(
        build_whitener(ma.centered_gram(0)),
        build_whitener(mb.centered_gram(0)),
        pm.centered_cross(ma, mb, 0, 0),
    )
    rho2, _ = canonical_correlations(
        build_whitener(ma2.centered_gram(0)),
        build_whitener(mb2.centered_gram(0)),
        pm2.centered_cross(ma2, mb2, 0, 0),
    )
    assert torch.allclose(rho, rho2, atol=1e-7)


# --------------------------------------------------------------------------- #
# Rank truncation.
# --------------------------------------------------------------------------- #


def test_whitener_drops_rank_deficient_directions():
    x = _rand(N, DIM, 111)
    x[:, -2:] = x[:, :2]  # exact duplicate columns -> rank deficiency
    assert build_whitener(_centered_gram_of(x)).rank == DIM - 2


def test_too_few_rows_produce_spurious_correlations_and_are_rejected():
    """Why `n_rows` exists: below n > dim, CCA reports near-perfect correlation between
    INDEPENDENT representations, because the centered Gram is rank-limited by rows."""
    n_small = DIM  # n <= dim: degenerate regime
    x, y = _rand(n_small, DIM, 131), _rand(n_small, DIM, 132)  # independent
    rho, _ = canonical_correlations(
        build_whitener(_centered_gram_of(x)),
        build_whitener(_centered_gram_of(y)),
        _center(x).T @ _center(y),
    )
    assert float(rho.mean()) > 0.99, "expected the spurious-correlation regime"

    # The guard catches it when the caller supplies the row count.
    with pytest.raises(ValueError, match="rank-limited by rows"):
        build_whitener(_centered_gram_of(x), n_rows=n_small)
    # ...and stays out of the way when there are plenty of rows.
    big = _rand(20 * DIM, DIM, 133)
    assert build_whitener(_centered_gram_of(big), n_rows=20 * DIM).rank == DIM


def test_var_frac_truncation_is_monotone():
    x = _rand(N, DIM, 121)
    gram = _centered_gram_of(x)
    assert build_whitener(gram, var_frac=0.5).rank <= build_whitener(gram, var_frac=0.99).rank
    assert build_whitener(gram, n_components=4).rank == 4


def _centered_gram_of(x: torch.Tensor) -> torch.Tensor:
    xc = _center(x)
    return xc.T @ xc
