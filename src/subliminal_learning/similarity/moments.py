"""Streaming second-moment accumulators for Procrustes / CCA-family similarity.

Why moments rather than per-batch scores
----------------------------------------
Minibatch CKA works because the unbiased HSIC estimator decomposes into a per-batch
average (see `cka.minibatch`). Procrustes and CCA do NOT: Procrustes needs the nuclear
norm of the full-sample cross-covariance, and CCA needs whitening by the full-sample
covariances. Neither is an average of per-batch values.

They are, however, functions of three ``d x d`` matrices and the column sums::

    A = X^T X        B = Y^T Y        C = X^T Y        sum_x = X^T 1

all of which ARE exactly additive over batches. So the streaming loop accumulates
those, and every measure is finalized once at the end from the full-sample statistic —
not approximated. Two consequences worth knowing:

* Memory is constant in the number of rows ``n``. Streaming 10x more rows costs
  nothing extra, which matters because CCA is biased at low ``n/d``.
* The result is the exact full-sample quantity, so it is NOT on the same estimator
  footing as the minibatch-averaged CKA numbers. Do not put them in one table without
  a note.

An ``n x n`` Gram matrix is never formed: at ``d = 768`` the feature-space route is
both cheaper and the only one that keeps memory independent of ``n``.

Precision
---------
Accumulate in float64. The finalizers whiten by ``A``, whose condition number is
``cond(X)^2``; residual streams carry outlier dimensions spanning several orders of
magnitude, so fp32 accumulation would leave too little headroom after that squaring.
At 13 layer slots x 768 dims this costs ~61 MB per model — not a constraint.
"""

from __future__ import annotations

import torch


def _check_rows(rows: list[torch.Tensor], n_layers: int, dim: int, what: str) -> None:
    if len(rows) != n_layers:
        raise ValueError(f"{what}: got {len(rows)} layer matrices, expected {n_layers}")
    for i, r in enumerate(rows):
        if r.ndim != 2:
            raise ValueError(f"{what}: layer {i} has shape {tuple(r.shape)}, expected 2-D")
        if r.shape[1] != dim:
            raise ValueError(
                f"{what}: layer {i} has {r.shape[1]} features, expected {dim}"
            )
    n = rows[0].shape[0]
    if any(r.shape[0] != n for r in rows):
        raise ValueError(f"{what}: layers disagree on row count: {[r.shape[0] for r in rows]}")


class ModelMoments:
    """Per-layer first/second moments for ONE model: ``A = X^T X`` and ``X^T 1``.

    Shared across every pair the model participates in — the moment analogue of the
    Gram-stack sharing in the CKA measurement loop. The expensive finalization step
    (one eigendecomposition of ``A`` per layer) therefore also happens once per model,
    not once per pair.

    Attributes:
        gram: ``[n_layers, dim, dim]`` uncentered ``X^T X``.
        col_sum: ``[n_layers, dim]`` uncentered ``X^T 1``.
        n_rows: Rows accumulated so far.
    """

    def __init__(
        self,
        n_layers: int,
        dim: int,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        self.n_layers = n_layers
        self.dim = dim
        self.dtype = dtype
        self.gram = torch.zeros(n_layers, dim, dim, dtype=dtype, device=device)
        self.col_sum = torch.zeros(n_layers, dim, dtype=dtype, device=device)
        self.n_rows = 0

    def update(self, rows: list[torch.Tensor]) -> None:
        """Accumulate one minibatch: ``rows[l]`` is ``[m, dim]`` for layer ``l``."""
        _check_rows(rows, self.n_layers, self.dim, "ModelMoments.update")
        stacked = torch.stack([r.to(self.gram) for r in rows], dim=0)
        self.gram += stacked.transpose(1, 2) @ stacked
        self.col_sum += stacked.sum(dim=1)
        self.n_rows += rows[0].shape[0]

    def centered_gram(self, layer: int) -> torch.Tensor:
        """Column-centered ``X_c^T X_c`` for one layer, exactly, from the raw moments."""
        if self.n_rows == 0:
            raise RuntimeError("no rows accumulated")
        s = self.col_sum[layer]
        return self.gram[layer] - torch.outer(s, s) / self.n_rows


class PairMoments:
    """Cross-moments ``C = X^T Y`` for one (model A, model B) pair.

    Only the layer-aligned diagonal (layer ``l`` of A vs layer ``l`` of B) is kept by
    default, because every summary scalar downstream reads the diagonal; the full
    ``n_layers x n_layers`` grid is ``n_layers`` times the memory and the same factor
    in finalization SVDs, so it is opt-in via ``full_grid``.

    A row permutation of B (the floor control) changes ``C`` but is NOT recoverable
    from an unpermuted ``C`` — a permuted pair needs its own accumulator, fed the
    permuted rows inside the batch loop.
    """

    def __init__(
        self,
        n_layers: int,
        dim_a: int,
        dim_b: int,
        full_grid: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        self.n_layers = n_layers
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.full_grid = full_grid
        shape = (
            (n_layers, n_layers, dim_a, dim_b) if full_grid else (n_layers, dim_a, dim_b)
        )
        self.cross = torch.zeros(*shape, dtype=dtype, device=device)
        self.n_rows = 0

    def update(self, rows_a: list[torch.Tensor], rows_b: list[torch.Tensor]) -> None:
        """Accumulate one minibatch. Both models must contribute the SAME rows in the
        same order (the pipeline's shared row index guarantees this)."""
        _check_rows(rows_a, self.n_layers, self.dim_a, "PairMoments.update (A)")
        _check_rows(rows_b, self.n_layers, self.dim_b, "PairMoments.update (B)")
        if rows_a[0].shape[0] != rows_b[0].shape[0]:
            raise ValueError(
                f"row-count mismatch: A has {rows_a[0].shape[0]}, B has {rows_b[0].shape[0]}"
            )
        xa = torch.stack([r.to(self.cross) for r in rows_a], dim=0)
        xb = torch.stack([r.to(self.cross) for r in rows_b], dim=0)
        if self.full_grid:
            self.cross += torch.einsum("imd,jme->ijde", xa, xb)
        else:
            self.cross += xa.transpose(1, 2) @ xb
        self.n_rows += rows_a[0].shape[0]

    def centered_cross(
        self, moments_a: ModelMoments, moments_b: ModelMoments, layer_a: int, layer_b: int
    ) -> torch.Tensor:
        """Column-centered ``X_c^T Y_c`` for one layer pair, exactly, from raw moments."""
        if self.n_rows == 0:
            raise RuntimeError("no rows accumulated")
        if moments_a.n_rows != self.n_rows or moments_b.n_rows != self.n_rows:
            raise ValueError(
                "row counts disagree between pair and model accumulators "
                f"({self.n_rows} vs {moments_a.n_rows}/{moments_b.n_rows}) — the same "
                "rows must feed all three"
            )
        if self.full_grid:
            c = self.cross[layer_a, layer_b]
        elif layer_a != layer_b:
            raise ValueError(
                f"off-diagonal layer pair ({layer_a}, {layer_b}) requested but this "
                "accumulator was built with full_grid=False"
            )
        else:
            c = self.cross[layer_a]
        sa = moments_a.col_sum[layer_a]
        sb = moments_b.col_sum[layer_b]
        return c - torch.outer(sa, sb) / self.n_rows
