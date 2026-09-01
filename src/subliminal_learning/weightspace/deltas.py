"""The ΔW variants: surgeries on a trained update, and synthetic injection shapes.

Two families share this module. `interp_basis` / `subset_targets` / `subset_rotate` /
`spread_variant` / `aligned_component_variant` rewrite a REAL update at exactly preserved
per-module norms (position, spread and direction surgery); `random_orthonormal_pair`,
`diffuse_ladder_delta` and `three_shape_delta` build SYNTHETIC updates at a matched
multiple of the module's RMT scale θ̄ (the equal-norm ladders). `intruder_positive` is the
Shuttleworth-style read that every one of them records.
"""

from __future__ import annotations

import numpy as np

from subliminal_learning.weightspace.lora import lora_svd


def interp_basis(u0: np.ndarray, u1: np.ndarray, t: float) -> np.ndarray:
    """QR-orthonormalised ``(1-t)·U0 + t·U1`` — a monotone bridge, not a geodesic."""
    return np.linalg.qr((1.0 - t) * u0 + t * u1)[0]


def subset_targets(u: np.ndarray, idx: list[int], rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal target directions for columns `idx`: orthonormal to the KEPT
    columns (which stay unchanged) and to each other."""
    if not idx:
        return u[:, :0]
    keep = [j for j in range(u.shape[1]) if j not in idx]
    q = np.linalg.qr(np.concatenate(
        [u[:, keep], rng.standard_normal((u.shape[0], len(idx)))], axis=1))[0]
    return q[:, len(keep):]


def subset_rotate(u: np.ndarray, idx: list[int], r: np.ndarray, t: float) -> np.ndarray:
    """Rotate columns `idx` of U a fraction t toward targets R (t=1 -> exactly R).
    The blend stays in the kept columns' orthogonal complement; QR re-orthonormalises
    the rotated block, sign-aligned to the blend so the path is continuous in t."""
    if not idx:
        return u
    blend = (1.0 - t) * u[:, idx] + t * r
    q = np.linalg.qr(blend)[0]
    sgn = np.sign(np.einsum("ij,ij->j", q, blend))
    sgn[sgn == 0] = 1.0
    out = u.copy()
    out[:, idx] = q * sgn
    return out


def spread_variant(b: np.ndarray, a: np.ndarray, s: float, variant: str) -> np.ndarray:
    """`top1`/`flat` at exactly the original ‖ΔW‖_F, via the r x r QR core."""
    u, sv, v = lora_svd(b, a)
    n_tot = s * float(np.sqrt((sv**2).sum()))
    if variant == "top1":
        return n_tot * np.outer(u[:, 0], v[:, 0])
    k = max(1, int((sv > sv[0] * 1e-10).sum()))
    return (n_tot / np.sqrt(k)) * (u[:, :k] @ v[:, :k].T)


def aligned_component_variant(dw: np.ndarray, w0: np.ndarray, variant: str) -> np.ndarray:
    """Apply one W₀-alignment variant to a module's ΔW, preserving ‖ΔW‖_F exactly."""
    if variant == "orig":
        return dw
    ip = float((dw * w0).sum())
    c = ip / float((w0 * w0).sum())
    dw_par = c * w0
    dw_perp = dw - dw_par
    n_tot = float(np.linalg.norm(dw))
    n_par = float(np.linalg.norm(dw_par))
    n_perp = float(np.linalg.norm(dw_perp))
    if variant == "flip":
        return dw_perp - dw_par
    if variant == "strip":
        return dw_perp * (n_tot / n_perp if n_perp else 1.0)
    if variant == "flip10":
        amp = 10.0
        rem = n_tot**2 - (amp * n_par) ** 2
        if rem <= 0 or not n_perp:  # sliver too large to amplify at fixed norm
            return dw_perp - dw_par
        return np.sqrt(rem) / n_perp * dw_perp - amp * dw_par
    raise ValueError(variant)


def random_orthonormal_pair(n: int, m: int, k: int,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``(U_r n x k, V_r m x k)`` orthonormal, drawn left-then-right from `rng`."""
    return (np.linalg.qr(rng.standard_normal((n, k)))[0],
            np.linalg.qr(rng.standard_normal((m, k)))[0])


def diffuse_ladder_delta(m: dict, align: str, k: int, e: float, rng: np.random.Generator,
                         match: str = "fro") -> np.ndarray:
    """match='fro': per-module ‖Δ‖_F = E·θ̄ (per-direction σ = E·θ̄/√k).
    match='spec': per-direction σ = E·θ̄ exactly (so ‖Δ‖_F = E·θ̄·√k grows with k) —
    the complementary design: if damage tracks σ₁ it is flat in k; if it tracks
    Frobenius energy it rises like √k."""
    n, mm = m["shape"]
    th = m["theta"]
    if align == "g":
        u_r, v_r = random_orthonormal_pair(n, mm, k, rng)
    else:
        u_r, v_r = m["U"][:, :k], m["Vt"][:k, :].T
    scale = (e * th) if match == "spec" else (e * th / np.sqrt(k))
    return scale * (u_r @ v_r.T)


def three_shape_delta(m: dict, cond: str, e: float, k: int,
                      rng: np.random.Generator) -> np.ndarray:
    """The 2x2 equal-norm shapes: {concentrated, diffuse} x {generic, W₀-aligned}."""
    n, mm = m["shape"]
    th = m["theta"]
    if cond in ("cg", "ca"):
        if cond == "cg":
            u = rng.standard_normal(n)
            v = rng.standard_normal(mm)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
        else:
            u, v = m["U"][:, 0], m["Vt"][0, :]
        return (e * th) * np.outer(u, v)
    if cond == "dg":
        u_r, v_r = random_orthonormal_pair(n, mm, k, rng)
    else:
        u_r, v_r = m["U"][:, :k], m["Vt"][:k, :].T
    return (e * th / np.sqrt(k)) * (u_r @ v_r.T)


def directions_delta(u: np.ndarray, v: np.ndarray, k: int, scale: float) -> np.ndarray:
    """``scale · U[:, :k] V[:, :k]ᵀ`` — equal energy over k given direction pairs."""
    return scale * (u[:, :k] @ v[:, :k].T)


def intruder_positive(w0: np.ndarray, delta: np.ndarray, u0: np.ndarray, *,
                      top: int = 10, tau: float = 0.5) -> bool:
    """Is ``W₀ + Δ`` intruder-positive: does any of its `top` new left singular vectors
    have max |overlap| < `tau` against every pretrained left singular vector?"""
    u_new = np.linalg.svd(w0 + delta, full_matrices=False)[0]
    ov = np.abs(u_new[:, :top].T @ u0).max(axis=1)
    return bool((ov < tau).sum() > 0)
