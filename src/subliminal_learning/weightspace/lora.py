"""The exact factorisation of a LoRA update, without ever forming a full-size product.

``ΔW = s·B·A`` with ``B`` ``d_out x r`` and ``A`` ``r x d_in``, so ``ΔW`` has rank <= r.
With thin QR factorisations ``B = Q_b R_b`` and ``A.T = Q_a R_a``::

    B A = Q_b (R_b R_a.T) Q_a.T

and since both ``Q`` have orthonormal columns, the SVD of the ``r x r`` core lifts
exactly: the singular values of ``BA`` are the core's, and its singular vectors are the
core's lifted through ``Q_b`` / ``Q_a``. Same identity as
`analysis.adapter_decomp.low_rank_singular_values` (which is the norms-only fast path);
this module is the one that needs the vectors as well.
"""

from __future__ import annotations

import numpy as np


def lora_svd(b: np.ndarray, a: np.ndarray,
             scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(U, sigma, V)`` of ``scale·B·A``: ``U`` is ``n x r``, ``V`` is ``m x r``.

    ``scale·B·A == (U * sigma) @ V.T`` exactly. The singular values are returned scaled;
    pass ``scale=1.0`` when the caller wants to apply ``s = alpha/r`` itself.
    """
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    q_b, r_b = np.linalg.qr(b)
    q_a, r_a = np.linalg.qr(a.T)
    u_c, sv, vt_c = np.linalg.svd(r_b @ r_a.T)
    return q_b @ u_c, scale * sv, q_a @ vt_c.T


def top_singular_vectors(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top left/right singular vectors of ``B @ A`` (scale-invariant, so unscaled)."""
    u, _, v = lora_svd(b, a)
    return u[:, 0], v[:, 0]


def lora_delta_w(b: np.ndarray, a: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """The materialised ``scale·B·A`` in float64 — the surgeries that leave LoRA form."""
    return scale * (np.asarray(b, dtype=np.float64) @ np.asarray(a, dtype=np.float64))
