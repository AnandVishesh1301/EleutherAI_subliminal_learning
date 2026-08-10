"""Representation-geometry math: Procrustes, subspace angles, trait coordinates.

Domain math that is inherently ``torch``, the same layering exemption
`cka.minibatch` and `gkd.losses` hold: tensor work lives here, disk/GPU
orchestration in the ``*_io`` modules and the pipelines.

Every representation matrix is ``[n_examples, n_features]`` -- the Kornblith
orientation -- and that is asserted at the boundary. The reference clones write
theirs transposed (``p x n``); the conversion is done here, not by callers.

CKA is **not** reimplemented. `minibatch_cka` / `minibatch_cka_grid` are thin
wrappers that drive `cka.minibatch.MinibatchCKA`, which stays the single CKA
implementation with its U-centering and "never average per-batch CKA"
invariants intact.

Sources, verified against the primary text before implementing:

- Linear CKA closed form and invariances: Kornblith et al., arXiv:1905.00414
  Table 1 (``papers/md/representation_similarity.md`` Table 1).
- Minibatch CKA protocol (batch 1024, 10,240 examples, 20 repeats): Raghu et
  al., arXiv:2108.08810; estimator ported in `cka.minibatch`.
- Orthogonal Procrustes: Ding et al., arXiv:2108.01661 Eq. 7
  (``papers/md/grounding_representation_similarity_statistical_testing.md``
  Eq. 7), preprocessing from their released code
  (``~/eleutherAI/sim_metric/dists/score_pair.py:114-120``).
- Principal (subspace) angles: Knyazev & Argentati (2002), via
  ``scipy.linalg.subspace_angles`` and PolyPythias arXiv:2503.09543 §4.
- Trait-subspace CKA: Okatan et al., arXiv:2511.01023 §III-E.
- High-norm outlier deletion: Davari et al., arXiv:2210.16156.
- CCA family (off by default, grading panel only): Ding et al.'s released
  ``sim_metric/dists/scoring.py`` -- ``cca_decomp`` (:6), ``mean_sq_cca_corr``
  (:49) and ``pwcca_dist`` (:65). SVCCA is NOT implemented: it is absent from
  that file, and re-deriving the SVD truncation rule from Raghu et al. (2017)
  would be a second, unvalidated convention.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import subspace_angles as _scipy_subspace_angles

from subliminal_learning.cka.minibatch import MinibatchCKA, preprocessed_gram_stack

# HSIC_1's n-2 / n-3 denominators are undefined below this (Song et al. 2012).
MIN_MINIBATCH_ROWS = 4

SSA_REDUCTIONS = ("mean", "max", "min")

# Gram accumulation dtype for the minibatch wrappers. `cka.minibatch` requires
# fp32 or wider; fp64 keeps the sum over hundreds of [n, n] Grams exact enough
# that the pooled value does not drift from the single-batch reference.
_ACCUM_DTYPE = torch.float64


def _as_examples_by_features(x: torch.Tensor, name: str) -> torch.Tensor:
    """Validate an ``[n_examples, n_features]`` matrix and promote it to float64."""
    if x.ndim != 2:
        raise ValueError(
            f"{name} must be [n_examples, n_features]; got {tuple(x.shape)}"
        )
    if x.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 examples; got {x.shape[0]}")
    return x.to(torch.float64)


def _column_center(x: torch.Tensor) -> torch.Tensor:
    """Subtract each feature's mean over examples (Kornblith §1)."""
    return x - x.mean(dim=0, keepdim=True)


def _require_same_n(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "similarity requires the SAME examples in the SAME order: "
            f"got n={x.shape[0]} vs n={y.shape[0]}"
        )


def linear_cka_biased(x: torch.Tensor, y: torch.Tensor) -> float:
    """Biased linear CKA (Kornblith Table 1, ``HSIC_0``). Diagnostic only.

    Reported next to `minibatch_cka` as an under-sampling flag: the biased
    estimator is batch-size dependent and inflates when ``n`` is small relative
    to ``p``, so a large gap against the unbiased value invalidates the sample
    rather than that one number. Never a headline value.

    Args:
        x: ``[n, p1]`` representations.
        y: ``[n, p2]`` representations over the same examples in the same order.

    Returns:
        CKA in ``[0, 1]``.

    Raises:
        ValueError: On a non-2-D input or mismatched ``n``.
    """
    xc = _column_center(_as_examples_by_features(x, "x"))
    yc = _column_center(_as_examples_by_features(y, "y"))
    _require_same_n(xc, yc)
    cross = torch.linalg.matrix_norm(yc.T @ xc) ** 2
    denom = torch.linalg.matrix_norm(xc.T @ xc) * torch.linalg.matrix_norm(yc.T @ yc)
    return float(cross / denom.clamp(min=1e-30))


def procrustes_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Orthogonal Procrustes distance (Ding et al. Eq. 7) in ``[0, 2]``.

    Preprocessing follows their released code, not the paper's prose: each
    FEATURE is centred over examples (``rep - rep.mean(axis=1)`` on their
    ``p x n`` orientation, i.e. column-centring here), then the WHOLE matrix is
    divided by its Frobenius norm -- ``sim_metric/dists/score_pair.py:114-120``.
    The prose ("subtract the mean value from each column") would centre across
    neurons on their own orientation and gives a different number; the code
    convention is the one consistent with Kornblith.

    Eq. 7 is ``||A||_F^2 + ||B||_F^2 - 2||A B^T||_*`` on ``p x n`` inputs, which
    is ``||x^T y||_*`` in this orientation.

    Args:
        x: ``[n, p1]`` representations.
        y: ``[n, p2]`` representations over the same examples in the same order.

    Returns:
        Distance in ``[0, 2]``: 0 for identical, orthogonally-rotated, or
        isotropically-rescaled inputs.

    Raises:
        ValueError: On a non-2-D input or mismatched ``n``.
    """
    xc = _column_center(_as_examples_by_features(x, "x"))
    yc = _column_center(_as_examples_by_features(y, "y"))
    _require_same_n(xc, yc)
    xn = xc / torch.linalg.matrix_norm(xc).clamp(min=1e-30)
    yn = yc / torch.linalg.matrix_norm(yc).clamp(min=1e-30)
    nuclear = torch.linalg.matrix_norm(xn.T @ yn, ord="nuc")
    sq = torch.linalg.matrix_norm(xn) ** 2 + torch.linalg.matrix_norm(yn) ** 2
    return float((sq - 2 * nuclear).clamp(min=0.0))


def _cca_decomp(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical correlations and X's canonical vectors, per ``scoring.py:6-46``.

    Whitens each side with the pseudo-inverse square root of its own covariance
    (negative eigenvalues clamped to 0, their inverses to 0, exactly as the
    released code does) and takes the SVD of the whitened cross-covariance.

    Args:
        x: ``[n, p1]`` column-centred, Frobenius-normalised representations.
        y: ``[n, p2]`` likewise, same examples in the same order.

    Returns:
        ``(rho, transformed_x)``: ``[min(p1, p2)]`` canonical correlations and
        the ``[n, p1]`` canonical vectors of X.
    """
    a, b = x.T, y.T

    def whitener(m: torch.Tensor) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(m @ m.T)
        inv = torch.where(
            evals > 0, evals.clamp(min=0).rsqrt(), torch.zeros_like(evals)
        )
        return evecs @ torch.diag(inv) @ evecs.T

    whiten_a, whiten_b = whitener(a), whitener(b)
    u, rho, _ = torch.linalg.svd(whiten_a @ (a @ b.T) @ whiten_b)
    return rho, (u.T @ whiten_a @ a).T


def _as_cca_operands(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-centre, Frobenius-normalise and enforce CCA's ``p < n`` requirement."""
    xc = _column_center(_as_examples_by_features(x, "x"))
    yc = _column_center(_as_examples_by_features(y, "y"))
    _require_same_n(xc, yc)
    for name, m in (("x", xc), ("y", yc)):
        if m.shape[1] >= m.shape[0]:
            raise ValueError(
                f"CCA requires p < n; {name} has p={m.shape[1]}, n={m.shape[0]} "
                "(sim_metric/dists/scoring.py:19-20)"
            )
    return (
        xc / torch.linalg.matrix_norm(xc).clamp(min=1e-30),
        yc / torch.linalg.matrix_norm(yc).clamp(min=1e-30),
    )


def cca_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared canonical correlation (``scoring.py::mean_sq_cca_corr``).

    Off by default and never a headline number: CCA is invariant to ANY
    invertible linear transform (Kornblith Table 1), so it cannot distinguish
    representations that differ by an arbitrary reweighting of directions, and
    it needs ``p < n`` to be defined at all. Reported only as a grading-panel
    cross-check on the CKA/Procrustes ordering.

    Args:
        x: ``[n, p1]`` representations, ``p1 < n``.
        y: ``[n, p2]`` representations over the same examples, ``p2 < n``.

    Returns:
        Mean of the squared canonical correlations, in ``[0, 1]``.

    Raises:
        ValueError: On a non-2-D input, mismatched ``n``, or ``p >= n``.
    """
    rho, _ = _cca_decomp(*_as_cca_operands(x, y))
    return float((rho**2).mean())


def pwcca_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    """Projection-weighted CCA similarity (``scoring.py::pwcca_dist``, :65-79).

    Morcos et al.'s weighting of each canonical correlation by how much of X it
    accounts for. Asymmetric in ``x`` by construction -- the weights come from X
    only. `sim_metric` returns ``1 - this`` as a distance; the similarity is
    returned here so it reads the same way as the other panel entries.

    Args:
        x: ``[n, p1]`` representations whose projections set the weights,
            ``p1 < n``.
        y: ``[n, p2]`` representations over the same examples, ``p2 < n``.

    Returns:
        Weighted mean canonical correlation, in ``[0, 1]``.

    Raises:
        ValueError: On a non-2-D input, mismatched ``n``, or ``p >= n``.
    """
    xn, yn = _as_cca_operands(x, y)
    rho, transformed = _cca_decomp(xn, yn)
    weights = (transformed.T @ xn).abs().sum(dim=1)
    weights = weights / weights.sum().clamp(min=1e-30)
    dim = min(weights.shape[0], rho.shape[0])
    return float(weights[:dim] @ rho[:dim])


def principal_subspace(x: torch.Tensor, k: int) -> torch.Tensor:
    """Orthonormal ``[p, k]`` basis of the top-``k`` principal subspace of ``x``.

    Args:
        x: ``[n, p]`` representations; column-centred internally.
        k: Number of principal directions, in ``[1, min(n, p)]``.

    Returns:
        ``[p, k]`` float64 basis whose columns are the leading right singular
        vectors of the centred matrix.

    Raises:
        ValueError: On a non-2-D input or ``k`` out of range.
    """
    xc = _column_center(_as_examples_by_features(x, "x"))
    if not 1 <= k <= min(xc.shape):
        raise ValueError(f"k must be in [1, {min(xc.shape)}]; got {k}")
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    return vh[:k].T.contiguous()


def subspace_angles_between(
    basis_x: torch.Tensor, basis_y: torch.Tensor
) -> torch.Tensor:
    """Principal angles in DEGREES, ASCENDING, between two column spaces.

    Wraps ``scipy.linalg.subspace_angles``, which returns radians in DESCENDING
    order (Knyazev & Argentati 2002); scipy is preferred over a hand-rolled
    ``arccos`` for its ``arcsin`` branch, which stays accurate at small angles.

    Args:
        basis_x: ``[p, k1]`` basis; only its column space matters.
        basis_y: ``[p, k2]`` basis in the same ambient space.

    Returns:
        ``[min(k1, k2)]`` float64 tensor of degrees, ascending.
    """
    angles = _scipy_subspace_angles(
        np.asarray(basis_x.detach().cpu(), dtype=np.float64),
        np.asarray(basis_y.detach().cpu(), dtype=np.float64),
    )
    return torch.from_numpy(np.degrees(np.sort(angles)))


def subspace_angles_topk(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """SSA in degrees between the top-``k`` principal subspaces of ``x`` and ``y``.

    DEVIATION from the source, stated because it changes what the number means:
    PolyPythias (arXiv:2503.09543 §4) applies SSA to *probe-parameter*
    subspaces. This runs the same angle machinery on PCA subspaces of the raw
    representations, which is the primary mode because it is available at every
    rung; the PolyPythias-faithful mode runs `cka.probes.probe_subspace` on
    fitted probe weight matrices wherever probes exist, and the gap between the
    two is itself a reported quantity.

    Both inputs must share an ambient basis for the angles to mean anything,
    which holds for ``data-seed`` pairs and same-lineage checkpoints but not
    across ``weight-seed`` variants -- see `cka.pairing.basis_validity`.

    Args:
        x: ``[n, p]`` representations.
        y: ``[n, p]`` representations in the same ambient basis.
        k: Subspace dimension.

    Returns:
        ``[k]`` float64 tensor of degrees, ascending.
    """
    return subspace_angles_between(principal_subspace(x, k), principal_subspace(y, k))


def ssa_reduce(angles: torch.Tensor, reduction: str = "mean") -> float:
    """Reduce an SSA angle vector to one scalar in degrees.

    Neither PolyPythias nor Müller-Eberstein et al. (arXiv:2310.16484) states
    how the angle vector becomes the single value they plot, and neither
    released SSA code, so the reduction is explicit rather than silent.

    Args:
        angles: ``[k]`` angles in degrees.
        reduction: One of `SSA_REDUCTIONS`.

    Returns:
        The reduced angle in degrees.

    Raises:
        ValueError: On an unknown `reduction` or an empty `angles`.
    """
    if angles.numel() == 0:
        raise ValueError("angles is empty; nothing to reduce")
    if reduction == "mean":
        return float(angles.mean())
    if reduction == "max":
        return float(angles.max())
    if reduction == "min":
        return float(angles.min())
    raise ValueError(f"reduction must be one of {SSA_REDUCTIONS}; got {reduction!r}")


def trait_coordinates(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project each example onto a model's own unit trait direction.

    Args:
        hidden: ``[n, p]`` per-example representations of ONE model at one slot.
        direction: ``[p]`` trait direction in THAT model's basis (a
            ``steering/vectors.py`` ``diff_vector`` row); normalised here, so a
            raw and a unit direction give the same coordinates up to sign.

    Returns:
        ``[n]`` float64 trait coordinate per example.

    Raises:
        ValueError: On a shape mismatch between `hidden` and `direction`.
    """
    h = _as_examples_by_features(hidden, "hidden")
    if direction.ndim != 1 or direction.shape[0] != h.shape[1]:
        raise ValueError(
            f"direction must be [{h.shape[1]}]; got {tuple(direction.shape)}"
        )
    v = direction.to(torch.float64)
    return h @ (v / v.norm().clamp(min=1e-12))


def trait_coordinate_similarity(cx: torch.Tensor, cy: torch.Tensor) -> float:
    """Squared Pearson correlation of two ``[n]`` trait-coordinate vectors.

    Exactly linear CKA at ``k = 1``, and the ROTATION-INVARIANT trait measure:
    each model is projected on its own direction in its own basis, so no neuron
    correspondence is assumed and the value stays valid across ``weight-seed``
    pairs, where `shared_basis_subspace_cka` is not.

    Args:
        cx: ``[n]`` coordinates from model X (`trait_coordinates`).
        cy: ``[n]`` coordinates from model Y over the same examples.

    Returns:
        Value in ``[0, 1]``; 0 when either vector is constant.

    Raises:
        ValueError: On a non-1-D input or a length mismatch.
    """
    if cx.ndim != 1 or cy.ndim != 1:
        raise ValueError(
            f"coordinates must be 1-D; got {tuple(cx.shape)}, {tuple(cy.shape)}"
        )
    if cx.shape != cy.shape:
        raise ValueError(f"length mismatch: {cx.shape[0]} vs {cy.shape[0]}")
    a = cx.to(torch.float64)
    b = cy.to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-30)
    return float(((a @ b) / denom) ** 2)


def orthonormal_basis(direction: torch.Tensor) -> torch.Tensor:
    """QR orthonormalisation of ``[p]`` or ``[p, k]`` directions into ``[p, k]``.

    The ``k = 1`` case is Okatan et al.'s construction (QR of the probe weight
    vector), with the ``diff_vector`` contrast substituted for their logistic
    probe so the measured direction is the one the steering hooks intervene on.

    Args:
        direction: ``[p]`` or ``[p, k]`` directions, not necessarily orthogonal.

    Returns:
        ``[p, k]`` float64 orthonormal basis.

    Raises:
        ValueError: If `direction` is neither 1-D nor 2-D.
    """
    d = direction.to(torch.float64)
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2:
        raise ValueError(
            f"direction must be [p] or [p, k]; got {tuple(direction.shape)}"
        )
    q, _ = torch.linalg.qr(d, mode="reduced")
    return q


def shared_basis_subspace_cka(
    x: torch.Tensor, y: torch.Tensor, basis: torch.Tensor
) -> float:
    """Okatan-faithful subspace CKA: ONE model's basis applied to BOTH models.

    Okatan et al. (arXiv:2511.01023 §III-E) project teacher and student through
    the QR basis of a probe fitted on the teacher. Applying that projection to a
    model with an independently initialised basis assumes a neuron
    correspondence that does not exist, so callers must restrict this to
    shared-basis pairs (`cka.pairing.basis_validity`); it is a bridge to their
    published numbers, with `trait_coordinate_similarity` as the primary.

    Args:
        x: ``[n, p]`` representations of the basis-owning model.
        y: ``[n, p]`` representations of the other model, same examples/order.
        basis: ``[p, k]`` orthonormal basis (`orthonormal_basis`).

    Returns:
        Linear CKA of the two projections; at ``k = 1`` the squared correlation.

    Raises:
        ValueError: On a shape mismatch between the inputs and `basis`.
    """
    b = basis.to(torch.float64)
    if b.ndim != 2:
        raise ValueError(f"basis must be [p, k]; got {tuple(basis.shape)}")
    xp = _as_examples_by_features(x, "x") @ b
    yp = _as_examples_by_features(y, "y") @ b
    return linear_cka_biased(xp, yp)


def high_norm_mask(xs: torch.Tensor, ys: torch.Tensor, frac: float) -> torch.Tensor:
    """Boolean keep-mask dropping the highest-norm ``frac`` of examples.

    Davari et al. (arXiv:2210.16156) show a handful of high-norm examples can
    carry a CKA score: deleting well under 1% of a dataset moves CKA from ~1 to
    ~0 between representations whose functional behaviour is unchanged.
    Re-scoring without them and reporting the delta is the check that a
    similarity reflects the distribution rather than a few outliers.

    An example is ranked by its LARGEST norm across BOTH models and ALL slots,
    not per slot: the failure mode is one example dominating the Gram matrix
    anywhere, and dropping different examples per slot makes slots
    incomparable.

    Args:
        xs: ``[Sx, n, p]`` per-slot representations for model X.
        ys: ``[Sy, n, p]`` per-slot representations for model Y, same examples.
        frac: Fraction of examples to drop, in ``[0, 1)``.

    Returns:
        ``[n]`` boolean mask, ``True`` for examples to keep; exactly
        ``floor(n * frac)`` entries are ``False``.

    Raises:
        ValueError: On a non-3-D input, a mismatched ``n``, or `frac` out of
            range.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"frac must be in [0, 1); got {frac}")
    if xs.ndim != 3 or ys.ndim != 3:
        raise ValueError(
            f"xs/ys must be [n_slots, n, p]; got {tuple(xs.shape)}, {tuple(ys.shape)}"
        )
    if xs.shape[1] != ys.shape[1]:
        raise ValueError(f"same examples required: n={xs.shape[1]} vs {ys.shape[1]}")
    n = xs.shape[1]
    worst = torch.maximum(xs.norm(dim=-1).amax(dim=0), ys.norm(dim=-1).amax(dim=0))
    mask = torch.ones(n, dtype=torch.bool, device=xs.device)
    n_drop = int(n * frac)
    if n_drop:
        mask[worst.topk(n_drop).indices] = False
    return mask


def minibatch_cka_grid(
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    batch_size: int = 1024,
    n_examples: int = 10240,
    n_repeats: int = 20,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unbiased minibatch CKA for every layer pair, plus its per-repeat spread.

    A thin driver over `cka.minibatch.MinibatchCKA` -- the estimator, the
    Szekely-Rizzo U-centering and the "accumulate sums, never average per-batch
    CKA" invariant all stay there. Each repeat draws `n_examples` rows without
    replacement and splits them into ``n_examples // batch_size`` batches; every
    layer pair sees the same batches, so the grid is internally comparable.
    Defaults are the Raghu et al. (arXiv:2108.08810) protocol.

    The pooled value comes from ONE accumulator fed every batch of every repeat;
    the spread comes from one accumulator per repeat. Repeat values are never
    averaged into the pooled value.

    Args:
        xs: ``[Sx, n, p1]`` per-slot representations for model X.
        ys: ``[Sy, n, p2]`` per-slot representations for model Y, same examples
            in the same order.
        batch_size: Examples per minibatch; must be >= `MIN_MINIBATCH_ROWS`.
        n_examples: Rows sampled per repeat, capped at the ``n`` available.
        n_repeats: Independent passes over the sampled examples.
        seed: Seeds the per-repeat permutations.

    Returns:
        ``(value, repeat_std)``, both ``[Sx, Sy]`` float64; `repeat_std` is all
        zeros when ``n_repeats == 1``.

    Raises:
        ValueError: On a non-3-D input, a mismatched ``n``, a `batch_size` below
            `MIN_MINIBATCH_ROWS`, `n_repeats` below 1, or a `batch_size` that
            leaves no full batch.
    """
    if xs.ndim != 3 or ys.ndim != 3:
        raise ValueError(
            f"xs/ys must be [n_slots, n, p]; got {tuple(xs.shape)}, {tuple(ys.shape)}"
        )
    if xs.shape[1] != ys.shape[1]:
        raise ValueError(
            "minibatch CKA requires the SAME examples in the SAME order: "
            f"n={xs.shape[1]} vs {ys.shape[1]}"
        )
    if batch_size < MIN_MINIBATCH_ROWS:
        raise ValueError(
            f"batch_size must be >= {MIN_MINIBATCH_ROWS} for the unbiased "
            f"HSIC_1 U-statistic; got {batch_size}"
        )
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1; got {n_repeats}")

    n_total = xs.shape[1]
    drawn = min(n_examples, n_total)
    n_batches = drawn // batch_size
    if n_batches < 1:
        raise ValueError(f"batch_size={batch_size} exceeds the {drawn} drawn examples")

    n_slots_x, n_slots_y = xs.shape[0], ys.shape[0]
    pooled = MinibatchCKA(n_slots_x, n_slots_y, across_models=True, dtype=_ACCUM_DTYPE)
    generator = torch.Generator().manual_seed(seed)
    per_repeat: list[torch.Tensor] = []
    for _ in range(n_repeats):
        order = torch.randperm(n_total, generator=generator)[:drawn]
        repeat = MinibatchCKA(
            n_slots_x, n_slots_y, across_models=True, dtype=_ACCUM_DTYPE
        )
        for b in range(n_batches):
            idx = order[b * batch_size : (b + 1) * batch_size].to(xs.device)
            grams_x = preprocessed_gram_stack(list(xs[:, idx, :]), _ACCUM_DTYPE)
            grams_y = preprocessed_gram_stack(list(ys[:, idx, :]), _ACCUM_DTYPE)
            pooled.update_state_from_grams(grams_x, grams_y)
            repeat.update_state_from_grams(grams_x, grams_y)
        per_repeat.append(repeat.result())

    value = pooled.result()
    if n_repeats < 2:
        return value, torch.zeros_like(value)
    return value, torch.stack(per_repeat).std(dim=0, unbiased=True)


def minibatch_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int = 1024,
    n_examples: int = 10240,
    n_repeats: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Scalar `minibatch_cka_grid` for a single ``[n, p]`` layer pair.

    Args:
        x: ``[n, p1]`` representations.
        y: ``[n, p2]`` representations over the same examples in the same order.
        batch_size: Examples per minibatch.
        n_examples: Rows sampled per repeat.
        n_repeats: Independent passes.
        seed: Seeds the per-repeat permutations.

    Returns:
        ``(value, repeat_std)``.
    """
    value, spread = minibatch_cka_grid(
        _as_examples_by_features(x, "x")[None],
        _as_examples_by_features(y, "y")[None],
        batch_size=batch_size,
        n_examples=n_examples,
        n_repeats=n_repeats,
        seed=seed,
    )
    return float(value[0, 0]), float(spread[0, 0])
