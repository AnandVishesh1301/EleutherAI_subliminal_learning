"""Procrustes distance and PWCCA, finalized from second moments.

Both measures are functions of the centered moments ``A = X_c^T X_c``,
``B = Y_c^T Y_c`` and ``C = X_c^T Y_c`` alone (see `moments`), so nothing here ever
touches an ``n x n`` Gram or an ``n x d`` row matrix.

Cost structure — the reason this file is split the way it is:

* **Procrustes needs no eigendecomposition at all.** It is
  ``tr(A) + tr(B) - 2 ||C||_*``, and the nuclear norm only needs singular VALUES, so
  it uses `torch.linalg.svdvals` rather than a full SVD.
* **The CCA family needs one eigendecomposition of ``A`` per (model, layer)** — for
  the whitening map. That decomposition is independent of the partner model, so
  `Whitener` is built once per model-layer and reused across every pair, mirroring how
  the CKA loop forwards each model once and shares its Gram stacks.
* **PWCCA is then nearly free**: one SVD of the whitened ``k_a x k_b`` cross-moment
  gives the canonical correlations, and its left factor gives the weights in O(d^2).

Whitening squares the condition number of ``X``, so inputs are expected in float64
(`moments` accumulates that way). Rank is truncated explicitly rather than relying on
a Cholesky to fail.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Whitener:
    """Whitening map for one (model, layer), from the eigendecomposition of ``A``.

    Build ONCE per model-layer and reuse across every pair — this is the expensive
    part of the CCA family and it does not depend on the partner model.

    Attributes:
        components: ``[dim, rank]`` retained eigenvectors ``V`` (descending eigenvalue).
        eigenvalues: ``[rank]`` retained eigenvalues ``lambda``, descending.
        dim: Original feature dimension.
    """

    components: torch.Tensor
    eigenvalues: torch.Tensor
    dim: int

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def map(self) -> torch.Tensor:
        """``W = V lambda^{-1/2}``, ``[dim, rank]``: ``X W`` has identity covariance."""
        return self.components * self.eigenvalues.rsqrt()

    @property
    def sqrt_factor(self) -> torch.Tensor:
        """``W^T A = lambda^{1/2} V^T``, ``[rank, dim]`` — the PWCCA weight factor.

        Precomputed here because it is a property of the model-layer alone, so it is
        shared across pairs exactly like `map`.
        """
        return self.components.T * self.eigenvalues.sqrt()[:, None]


def build_whitener(
    gram: torch.Tensor,
    rel_tol: float = 1e-12,
    var_frac: float | None = None,
    n_components: int | None = None,
    n_rows: int | None = None,
) -> Whitener:
    """Eigendecompose a centered Gram ``A`` into a rank-truncated whitening map.

    Args:
        gram: ``[dim, dim]`` centered, symmetric PSD ``X_c^T X_c``.
        rel_tol: Drop eigenvalues below ``rel_tol * max(eigenvalue)``. Numerical rank
            guard only; leave at the default unless you know the spectrum.
        var_frac: SVCCA-style truncation — keep the leading directions carrying this
            fraction of total variance (e.g. 0.99). ``None`` disables it, which is the
            correct default FOR PWCCA: its weighting is the intended replacement for
            variance truncation, not a complement to it.
        n_components: Hard cap on retained rank, applied after the other two rules.

    Returns:
        A `Whitener` over the retained subspace.

    Sample-size trap. The rank of a centered Gram is bounded by ``n_rows - 1``, so
    with too few rows this truncates silently and the CCA family then reports
    SPURIOUS near-perfect correlations (two random representations are perfectly
    correlatable once the retained rank approaches the row count). Check `Whitener.rank`
    against ``dim``: a rank well below ``dim`` on data you expect to be full-rank means
    you are rank-limited by rows, not by the representation. Pass ``n_rows`` to have
    this checked for you. Rule of thumb for CCA: ``n_rows >> dim`` (the pipeline's
    25.6k rows against d=768 is comfortable; a single 256-row batch is not).
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"expected a square Gram, got shape {tuple(gram.shape)}")
    dim = gram.shape[0]
    if n_rows is not None and n_rows <= dim:
        raise ValueError(
            f"n_rows ({n_rows}) <= dim ({dim}): the centered Gram is rank-limited by "
            "rows, and canonical correlations computed from it would be spurious "
            "(any two representations are perfectly correlatable in this regime). "
            "Accumulate more rows."
        )
    sym = 0.5 * (gram + gram.T)
    evals, evecs = torch.linalg.eigh(sym)
    # eigh returns ascending; work descending so truncation rules read naturally.
    evals = evals.flip(0).clamp_min(0.0)
    evecs = evecs.flip(1)

    if float(evals[0]) <= 0.0:
        raise ValueError("centered Gram has no positive eigenvalue (all-constant rows?)")
    keep = int((evals > rel_tol * evals[0]).sum())
    if var_frac is not None:
        if not 0.0 < var_frac <= 1.0:
            raise ValueError(f"var_frac must be in (0, 1], got {var_frac}")
        csum = torch.cumsum(evals[:keep], dim=0) / evals[:keep].sum()
        keep = min(keep, int((csum < var_frac).sum()) + 1)
    if n_components is not None:
        keep = min(keep, n_components)
    return Whitener(components=evecs[:, :keep], eigenvalues=evals[:keep], dim=dim)


def procrustes(
    gram_a: torch.Tensor,
    gram_b: torch.Tensor,
    cross: torch.Tensor,
    normalize: bool = True,
) -> dict[str, float]:
    """Orthogonal Procrustes distance between two centered representations.

    Solves ``min_Q ||X - Y Q||_F`` over orthogonal ``Q``; the optimum has the closed
    form ``||X||_F^2 + ||Y||_F^2 - 2 ||X^T Y||_*`` (Schoenemann), so only singular
    VALUES of the cross-moment are needed — no eigendecomposition, no singular
    vectors.

    Args:
        gram_a / gram_b: Centered ``[d, d]`` Grams of X and Y.
        cross: Centered ``[d, d]`` cross-moment ``X_c^T Y_c``.
        normalize: Also report the scale-invariant forms, which rescale each
            representation to unit Frobenius norm first. Raw Procrustes is sensitive
            to isotropic scaling; the normalized distance lies in ``[0, 2]`` and the
            angular form in ``[0, pi/2]``.

    Returns:
        ``squared_distance`` and ``distance`` (raw scale), plus — when ``normalize`` —
        ``normalized_distance`` and ``angular_distance``, and always the alignment
        ``nuclear_norm``.
    """
    nuclear = float(torch.linalg.svdvals(cross).sum())
    ss_a = float(torch.diagonal(gram_a).sum())
    ss_b = float(torch.diagonal(gram_b).sum())
    sq = max(ss_a + ss_b - 2.0 * nuclear, 0.0)
    out = {
        "squared_distance": sq,
        "distance": sq**0.5,
        "nuclear_norm": nuclear,
    }
    if normalize:
        denom = (ss_a * ss_b) ** 0.5
        if denom <= 0.0:
            raise ValueError("degenerate representation: zero total variance")
        # cos of the angle between the two shapes after optimal rotation.
        cos = min(max(nuclear / denom, -1.0), 1.0)
        out["normalized_distance"] = max(2.0 - 2.0 * cos, 0.0) ** 0.5
        out["angular_distance"] = float(torch.arccos(torch.tensor(cos)))
    return out


def canonical_correlations(
    whitener_a: Whitener,
    whitener_b: Whitener,
    cross: torch.Tensor,
    want_vectors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Canonical correlations between two centered representations.

    The whitened cross-moment ``W_a^T C W_b`` has the canonical correlations as its
    singular values — one SVD of a ``rank_a x rank_b`` matrix, which after truncation
    is typically much smaller than ``d x d``.

    Args:
        whitener_a / whitener_b: Per-model-layer whiteners (reused across pairs).
        cross: Centered ``[d_a, d_b]`` cross-moment.
        want_vectors: Also return the left singular vectors ``U``, needed only by
            `pwcca`. Skipping them lets the caller use the cheaper `svdvals` path.

    Returns:
        ``(rho, u)`` with ``rho`` descending in ``[0, 1]``; ``u`` is ``None`` unless
        ``want_vectors``.
    """
    if cross.shape != (whitener_a.dim, whitener_b.dim):
        raise ValueError(
            f"cross-moment shape {tuple(cross.shape)} does not match whitener dims "
            f"({whitener_a.dim}, {whitener_b.dim})"
        )
    m = whitener_a.map.T @ cross.to(whitener_a.map) @ whitener_b.map
    if not want_vectors:
        return torch.linalg.svdvals(m).clamp(0.0, 1.0), None
    u, s, _ = torch.linalg.svd(m, full_matrices=False)
    return s.clamp(0.0, 1.0), u


def pwcca(
    whitener_a: Whitener,
    whitener_b: Whitener,
    cross: torch.Tensor,
) -> dict[str, float]:
    """Projection-weighted CCA (Morcos et al. 2018), weights taken on the A side.

    Plain CCA averages all canonical correlations equally, which lets near-noise
    directions dominate. PWCCA instead weights correlation ``i`` by how much of the
    ORIGINAL representation its canonical variate accounts for::

        alpha_i = sum_j |<h_i, x_j>|

    where ``h_i`` is the i-th canonical variate and ``x_j`` ranges over the *neurons*
    (feature columns) of X — not over examples. That is what makes the moment form
    possible: with ``A_x = W_a U`` the canonical coefficients, the variates are
    orthonormal and

        ``A_x^T A = U^T W_a^T A = U^T lambda^{1/2} V^T``

    so ``alpha_i`` is the L1 norm of row ``i`` of a ``rank x d`` matrix — O(d^2), with
    no pass over the data and no n-dimensional object. `Whitener.sqrt_factor` supplies
    ``lambda^{1/2} V^T``, shared across pairs.

    Note the asymmetry: the weights come from A, so ``pwcca(a, b) != pwcca(b, a)``.
    Swap the whiteners and transpose ``cross`` for the other side.

    Ill-posedness under tied correlations — check ``min_rho_gap``. Where two canonical
    correlations coincide, the singular vectors are defined only up to a rotation
    WITHIN the tied subspace, and the weights are L1 norms, which that rotation does
    not preserve. The correlations stay exact but the weight mass on the tied block
    (and hence the PWCCA value) becomes basis-dependent. This is a property of the
    measure, not of this implementation: any two correct implementations may disagree
    there. Treat a ``min_rho_gap`` at or near zero as "PWCCA not identified"; on real
    activations exact ties are rare, but near-ties make the value correspondingly
    ill-conditioned.

    Returns:
        ``pwcca`` (the weighted mean), ``cca_mean`` (the unweighted mean, i.e. plain
        CCA, free from the same SVD), ``n_components``, and ``min_rho_gap``.
    """
    rho, u = canonical_correlations(whitener_a, whitener_b, cross, want_vectors=True)
    weights = (u.T @ whitener_a.sqrt_factor).abs().sum(dim=1)
    total = weights.sum()
    if float(total) <= 0.0:
        raise ValueError("degenerate PWCCA weights (zero-variance representation?)")
    gap = float((rho[:-1] - rho[1:]).min()) if rho.shape[0] > 1 else float("inf")
    return {
        "pwcca": float((weights * rho).sum() / total),
        "cca_mean": float(rho.mean()),
        "n_components": int(rho.shape[0]),
        "min_rho_gap": gap,
    }
