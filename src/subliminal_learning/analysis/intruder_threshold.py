"""Per-layer intruder-threshold estimator (Benaych-Georges/Nadakuditi spiked-matrix
detection), after "The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning"
(arXiv:2607.23711).

The claim being tested on our populations: a LoRA module grows an *intruder* singular
vector (a new direction outside the pretrained spectrum) exactly when the update's
strength γ·σ₁(BA) crosses a critical θ̄ computed from the BASE matrix W's own empirical
spectrum, and forgetting follows the crossing. Their law is spectral (σ₁) with the LoRA
scale γ = α/r; our fitted law is Frobenius with absolute α — this estimator is the
head-to-head instrument (`collapse_program_synthesis.md` §5b, open program item 0a).

Recipe (paper §"estimator", reproduced from the HTML; no public code repo, so this is a
clean-room implementation validated by their own robustness gates):
  φ(z)  = (1/p) Σ_i z/(z² − σ_i²)            p = min(n, m) singular values of W
  D(z)  = φ(z)·[c·φ(z) + (1 − c)/z]          c = p / max(n, m)
  θ̄     = D(z₀)^(−1/2),  z₀ = edge·(1 + ε)   ε pre-specified in [1e-4, 3e-2]
  margin = γ·σ₁(BA) / θ̄                      supercritical iff margin > 1
Two edge variants: "full" (edge = σ₁ of W) and "bulk" (edge = top of the bulk after a
gap-test trim of isolated outlier singular values — the paper's exact trim rule is not
printed in the HTML, so ours is documented here: drop leading σ's whose gap to the next
exceeds 10× the typical top-spectrum spacing). The ε-sensitivity ratio
R = max_ε θ̄ / min_ε θ̄ is their degeneracy gate (median R > 10 ⇒ prediction degenerate).

All spectrum work in float64; σ₁(BA) via the QR trick (σ(BA) = σ(R_B·A)), so no n×m
product is materialized.
"""
from __future__ import annotations

import numpy as np

EPS_CENTRAL = 3e-3
EPS_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def _phi(z: float, sig2: np.ndarray) -> float:
    return float(np.mean(z / (z * z - sig2)))


def d_transform(z: float, sigmas: np.ndarray, c: float) -> float:
    p = _phi(z, sigmas.astype(np.float64) ** 2)
    return p * (c * p + (1.0 - c) / z)


def bulk_edge(sigmas: np.ndarray, gap_factor: float = 10.0, scan_top: int = 50) -> float:
    """Top of the bulk after trimming isolated outliers (documented gap test)."""
    s = np.sort(sigmas.astype(np.float64))[::-1]
    if len(s) < scan_top + 2:
        return float(s[0])
    gaps = s[:-1] - s[1:]
    typical = float(np.median(gaps[10:scan_top]))
    k = 0
    while k < scan_top and gaps[k] > gap_factor * max(typical, 1e-12):
        k += 1
    return float(s[k])


def theta_bar(sigmas: np.ndarray, shape: tuple[int, int], eps: float = EPS_CENTRAL,
              edge: str = "full") -> float:
    n, m = shape
    c = min(n, m) / max(n, m)
    e = float(np.max(sigmas)) if edge == "full" else bulk_edge(sigmas)
    z0 = e * (1.0 + eps)
    d = d_transform(z0, sigmas, c)
    return float("inf") if d <= 0 else d ** -0.5


def sigma1_ba(B: np.ndarray, A: np.ndarray) -> float:
    """σ₁(B@A) without materializing n×m: σ(BA) = σ(R_B·A) with B = Q_B R_B."""
    _, r_b = np.linalg.qr(B.astype(np.float64))
    return float(np.linalg.svd(r_b @ A.astype(np.float64), compute_uv=False)[0])


def empirical_intruders(U: np.ndarray, S: np.ndarray, V: np.ndarray, A: np.ndarray,
                        B: np.ndarray, gamma: float, s: float = 1.0, top_k: int = 10,
                        tau: float = 0.5) -> dict:
    """Empirical (Shuttleworth) intruder detection for W' = W + s·γ·BA — WITHOUT an SVD of
    the full updated matrix.

    Core-matrix trick (their Proposition 1): with the thin SVD W = U·diag(S)·Vᵀ
    (U n×p complete when p = n ≤ m), decompose Aᵀ = V·P + Q_A·R (QR of the V-orthogonal
    residual). Then W' = U · C · [V Q_A]ᵀ with the small core
        C = [diag(S) | 0] + s·γ·(UᵀB)·[Pᵀ | Rᵀ]   ∈ ℝ^{p×(p+r)} .
    One SVD of C gives W''s spectrum, and — because W's own singular vectors are the
    coordinate axes of this basis — the overlap of W''s j-th singular vector with EVERY
    pretrained one is just the largest first-p coordinate of C's j-th singular vector:
    no basis products, and a strength scan over s reuses U/V/P/Q_A/R.

    Returns per-side intruder counts among the top_k singular vectors of W'
    (max coordinate < tau ⇒ intruder) and the new spectrum's top values.
    Orientation: pass matrices so that n ≤ m (transpose W, swap/transpose A,B outside if
    needed); U must be the complete n×n left basis.
    """
    Bg = (s * gamma) * B.astype(np.float64)
    At = A.astype(np.float64).T                       # m×r
    P = V.T @ At                                      # p×r
    resid = At - V @ P
    Q_A, R = np.linalg.qr(resid)                      # m×r, r×r
    p = len(S)
    C = np.zeros((p, p + R.shape[0]))
    C[:, :p] = np.diag(S)
    C += (U.T @ Bg) @ np.hstack([P.T, R.T])
    Y, S_new, Zt = np.linalg.svd(C, full_matrices=False)
    left_ov = np.abs(Y[:, :top_k]).max(axis=0)        # coords in U-basis = overlaps
    right_ov = np.abs(Zt[:top_k, :p]).max(axis=1)     # first p coords = V-overlaps
    return dict(
        n_intruders_left=int((left_ov < tau).sum()),
        n_intruders_right=int((right_ov < tau).sum()),
        top_sigmas=S_new[:top_k], sigma1_base=float(S.max()),
    )


def module_margins(w_sigmas: np.ndarray, shape: tuple[int, int], A: np.ndarray,
                   B: np.ndarray, gamma: float) -> dict:
    """Threshold margins (>1 = supercritical) for one LoRA module, both edges, with the
    ε-sensitivity ratio R for the full edge (their degeneracy gate)."""
    s1 = gamma * sigma1_ba(B, A)
    thetas = [theta_bar(w_sigmas, shape, e, "full") for e in EPS_GRID]
    return dict(
        margin_full=s1 / theta_bar(w_sigmas, shape, EPS_CENTRAL, "full"),
        margin_bulk=s1 / theta_bar(w_sigmas, shape, EPS_CENTRAL, "bulk"),
        eps_ratio=max(thetas) / min(thetas),
        sigma1_scaled=s1,
    )
