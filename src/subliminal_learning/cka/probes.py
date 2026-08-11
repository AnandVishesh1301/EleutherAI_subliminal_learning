"""Linear and online-code (MDL-flavoured) probes -- pure fitting math.

The independent functional readout that grades everything in `cka.metrics`:
Ding et al. (arXiv:2108.01661) show CKA is the least sensitive measure they
tested, so without a probe every geometry number is graded only against other
geometry numbers. Nguyen et al.'s ``linear_probe.py`` (a TF/CIFAR file in
``~/eleutherAI/do_wide_and_deep_networks_learn_the_same_things``) is the design
precedent -- train a linear classifier on frozen activations, report per-layer
test accuracy -- and is not ported.

Treebank loading and activation collection live in ``probes_io.py``; this
module is fitting math over arrays that are already in memory.

OUT OF SCOPE (plan §12): the full MDL probe of
``papers/md/polypythias_reference.md`` §7.1 -- seven licensed treebank tasks, a
variational Bayesian probe and a jointly-learned layer weighting ``alpha``. We
take the PoS task and the online-code approximation below; the gap is stated
here and in `cka.metrics.subspace_angles_topk`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Voita & Titov (2020) footnote 4: timesteps at 0.1, 0.2, 0.4, 0.8, 1.6, 3.2,
# 6.25, 12.5, 25, 50 and 100 percent of the dataset.
ONLINE_BLOCK_FRACTIONS = (
    0.001,
    0.002,
    0.004,
    0.008,
    0.016,
    0.032,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
)

# Floor for predicted probabilities so an underflowed `predict_proba` entry
# costs many bits instead of producing an infinite codelength.
_MIN_PROBABILITY = 1e-12


@dataclass(frozen=True)
class LinearProbeFit:
    """A fitted logistic probe and its held-out readouts.

    Attributes:
        weights: ``[1, p]`` for a binary task, ``[n_classes, p]`` otherwise
            (sklearn ``coef_``); each row is a direction in feature space.
        intercept: Matching ``[1]`` / ``[n_classes]`` bias.
        classes: Class labels in `weights` row order.
        accuracy: Held-out accuracy.
        macro_f1: Held-out macro-averaged F1.
        n_train: Training rows.
        n_test: Held-out rows.
        n_iter: lbfgs iterations actually taken.
        converged: Whether lbfgs stopped on its own tolerance rather than on
            `max_iter`. `accuracy` / `macro_f1` barely move when this is False
            (measured <=0.003 on 17-class UPOS over raw Pythia-160m
            activations), but `weights` is still mid-descent -- up to ~17 deg
            off the optimum -- so any consumer of a probe DIRECTION must check
            it. See `cka.probes_io.probe_direction_cosine`.
    """

    weights: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    n_iter: int = 0
    converged: bool = True


def _as_design(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate an ``[n, p]`` feature matrix against its ``[n]`` labels."""
    features = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y)
    if features.ndim != 2:
        raise ValueError(f"x must be [n_examples, n_features]; got {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"y must be [n_examples]; got {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"x has {features.shape[0]} rows but y has {labels.shape[0]} labels"
        )
    return features, labels


def fit_linear_probe(
    x: ArrayLike,
    y: ArrayLike,
    *,
    seed: int,
    test_size: float = 0.2,
    c: float = 1.0,
    max_iter: int = 1000,
) -> LinearProbeFit:
    """Fit a logistic probe on a stratified split and score it held out.

    Args:
        x: ``[n, p]`` frozen activations.
        y: ``[n]`` class labels.
        seed: Seeds the train/test split (the lbfgs fit is deterministic).
        test_size: Held-out fraction.
        c: Inverse regularisation strength (sklearn ``C``); the default L2
            penalty is kept, so this is the only regularisation knob.
        max_iter: lbfgs iteration cap.

    Returns:
        The `LinearProbeFit`.

    Raises:
        ValueError: On a shape mismatch, fewer than two classes, or a class too
            rare to stratify.
    """
    features, labels = _as_design(x, y)
    if np.unique(labels).size < 2:
        raise ValueError("a probe needs at least 2 classes; got 1")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    n_iter = int(np.max(np.asarray(model.n_iter_)))
    return LinearProbeFit(
        weights=np.asarray(model.coef_, dtype=np.float64),
        intercept=np.asarray(model.intercept_, dtype=np.float64),
        classes=np.asarray(model.classes_),
        accuracy=float(accuracy_score(y_test, predicted)),
        macro_f1=float(f1_score(y_test, predicted, average="macro")),
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        n_iter=n_iter,
        converged=n_iter < max_iter,
    )


def _online_timesteps(n: int, block_fractions: tuple[float, ...]) -> list[int]:
    """Strictly increasing block boundaries in ``[1, n]``, always ending at ``n``."""
    steps = sorted({int(round(f * n)) for f in block_fractions if 0.0 < f <= 1.0})
    steps = [t for t in steps if 1 <= t < n]
    steps.append(n)
    return steps


def prequential_codelength(
    x: ArrayLike,
    y: ArrayLike,
    *,
    seed: int,
    block_fractions: tuple[float, ...] = ONLINE_BLOCK_FRACTIONS,
    c: float = 1.0,
    max_iter: int = 1000,
) -> float:
    """Online (prequential) codelength of the labels given the features, in bits.

    Voita & Titov, "Information-Theoretic Probing with Minimum Description
    Length" (EMNLP 2020, arXiv:2003.12298) Eq. 4::

        L_online(y_1:n | x_1:n) = t_1 log2 K
                                  - sum_{i=1}^{S-1} log2 p_θi(y_{t_i+1:t_i+1})

    where ``θ_i`` is fitted on the first ``t_i`` examples and the first block is
    transmitted with a uniform code over ``K`` classes.

    DEVIATION, stated because it changes ``t_1``: the uniform-code prefix is
    extended to the first boundary whose training prefix contains ALL ``K``
    classes, rather than the first block outright. A probe fitted on a prefix
    missing a class cannot assign it any probability, and charging such a label
    the probability floor would swamp the code with an artefact of the block
    schedule. This reduces to Eq. 4 exactly whenever the first block is already
    class-complete.

    Each block's probe is fit on ``StandardScaler``-standardised features (fit on
    that block's training prefix, applied to both the prefix and the held-out
    block) so lbfgs converges on the small early prefixes -- raw 768-d
    activations sit on wildly different per-feature scales and otherwise stall
    at `max_iter`. This changes only the fit's internal parameterisation, not
    the quantity computed: the code depends on `predict_proba` under matching
    prefix/block scaling, not on the coefficients, so it is unaffected by the
    reparameterisation once both fits converge.

    Args:
        x: ``[n, p]`` frozen activations.
        y: ``[n]`` class labels.
        seed: Seeds the example ordering, which the online code depends on.
        block_fractions: Timesteps as fractions of ``n``.
        c: Inverse regularisation strength for each block's probe.
        max_iter: lbfgs iteration cap per block.

    Returns:
        Codelength in bits; compare against ``n * log2(K)`` via
        `codelength_ratio`.

    Raises:
        ValueError: On a shape mismatch or fewer than two classes.
    """
    features, labels = _as_design(x, y)
    classes = np.unique(labels)
    n_classes = classes.size
    if n_classes < 2:
        raise ValueError("the online code needs at least 2 classes; got 1")

    n = features.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    features, labels = features[order], labels[order]
    uniform_bits = math.log2(n_classes)

    steps = _online_timesteps(n, block_fractions)
    complete = [t for t in steps[:-1] if np.unique(labels[:t]).size == n_classes]
    if not complete:
        return float(n * uniform_bits)

    boundaries = [t for t in steps if t >= complete[0]]
    code = complete[0] * uniform_bits
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        scaler = StandardScaler().fit(features[:start])
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
        model.fit(scaler.transform(features[:start]), labels[:start])
        if model.classes_.size != n_classes:
            raise ValueError(
                f"block prefix of {start} rows saw {model.classes_.size} of "
                f"{n_classes} classes; the code would be charged against a "
                "truncated label set"
            )
        proba = model.predict_proba(scaler.transform(features[start:stop]))
        column = np.searchsorted(model.classes_, labels[start:stop])
        true_p = proba[np.arange(stop - start), column]
        code -= float(np.log2(np.maximum(true_p, _MIN_PROBABILITY)).sum())
    return code


def codelength_ratio(codelength: float, n: int, n_classes: int) -> float:
    """Online codelength as a fraction of the uniform code ``n * log2(K)``.

    Voita & Titov report the reciprocal ("compression"); the ratio is used here
    so 1.0 is the no-information baseline and smaller is better.

    Args:
        codelength: Bits from `prequential_codelength`.
        n: Examples the code was computed over.
        n_classes: ``K``.

    Returns:
        ``codelength / (n * log2(n_classes))``.

    Raises:
        ValueError: If `n` < 1 or `n_classes` < 2.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2; got {n_classes}")
    return float(codelength / (n * math.log2(n_classes)))


def probe_subspace(weights: ArrayLike, k: int) -> torch.Tensor:
    """Orthonormal ``[p, k]`` basis of a probe's parameter subspace.

    The PolyPythias-faithful input to `cka.metrics.subspace_angles_between`
    (arXiv:2503.09543 §4 applies SSA to probe parameters, not activations).
    The weight rows are directions, not a data cloud, so they are NOT centred
    before the SVD.

    Args:
        weights: ``[n_classes, p]`` probe weights (`LinearProbeFit.weights`).
        k: Subspace dimension, in ``[1, min(n_classes, p)]``.

    Returns:
        ``[p, k]`` float64 tensor of leading right singular vectors.

    Raises:
        ValueError: On a non-2-D input or ``k`` out of range.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weights must be [n_classes, p]; got {w.shape}")
    if not 1 <= k <= min(w.shape):
        raise ValueError(f"k must be in [1, {min(w.shape)}]; got {k}")
    _, _, vh = np.linalg.svd(w, full_matrices=False)
    return torch.from_numpy(np.ascontiguousarray(vh[:k].T))
