"""QR-derived random directions in `weightspace.deltas` must be rng-determined, not
fixed by LAPACK's Householder column-sign convention."""
import numpy as np

from subliminal_learning.forgetting.weightspace.deltas import (
    interp_basis,
    random_orthonormal_pair,
    subset_rotate,
    subset_targets,
)


def _orthonormal(q: np.ndarray) -> bool:
    return np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)


def test_random_orthonormal_pair_sign_is_balanced_and_raw_qr_is_not():
    rng = np.random.default_rng(0)
    draws = [random_orthonormal_pair(16, 12, 1, rng) for _ in range(2000)]
    frac = np.mean([u[0, 0] > 0 for u, _ in draws])
    assert 0.45 <= frac <= 0.55, frac
    assert all(_orthonormal(u) and _orthonormal(v) for u, v in draws)

    rng = np.random.default_rng(0)  # sentinel: raw LAPACK QR is fully sign-degenerate
    raw = [np.linalg.qr(rng.standard_normal((16, 1)))[0][0, 0] > 0 for _ in range(2000)]
    assert np.mean(raw) == 0.0


def test_subset_targets_orthonormal_and_sign_balanced():
    rng = np.random.default_rng(0)
    u = np.linalg.qr(np.random.default_rng(1).standard_normal((16, 6)))[0]
    idx = [1, 3]
    keep = [j for j in range(6) if j not in idx]
    firsts = []
    for _ in range(2000):
        r = subset_targets(u, idx, rng)
        firsts.append(r[0, 0] > 0)
    assert 0.45 <= np.mean(firsts) <= 0.55, np.mean(firsts)
    assert r.shape == (16, 2)
    assert _orthonormal(r)
    assert np.allclose(u[:, keep].T @ r, 0.0, atol=1e-10)


def test_interp_basis_and_subset_rotate_unchanged_by_refactor():
    def interp_basis_pre(u0, u1, t):
        if t <= 0:
            return u0.copy()
        if t >= 1:
            return u1.copy()
        blend = (1.0 - t) * u0 + t * u1
        q = np.linalg.qr(blend)[0]
        sgn = np.sign(np.einsum("ij,ij->j", q, blend))
        sgn[sgn == 0] = 1.0
        return q * sgn

    def subset_rotate_pre(u, idx, r, t):
        if not idx:
            return u
        blend = (1.0 - t) * u[:, idx] + t * r
        q = np.linalg.qr(blend)[0]
        sgn = np.sign(np.einsum("ij,ij->j", q, blend))
        sgn[sgn == 0] = 1.0
        out = u.copy()
        out[:, idx] = q * sgn
        return out

    rng = np.random.default_rng(7)
    u0 = np.linalg.qr(rng.standard_normal((16, 4)))[0]
    u1 = np.linalg.qr(rng.standard_normal((16, 4)))[0]
    idx = [0, 2]
    r = subset_targets(u0, idx, rng)
    for t in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert np.array_equal(interp_basis(u0, u1, t), interp_basis_pre(u0, u1, t))
        assert np.array_equal(subset_rotate(u0, idx, r, t), subset_rotate_pre(u0, idx, r, t))
