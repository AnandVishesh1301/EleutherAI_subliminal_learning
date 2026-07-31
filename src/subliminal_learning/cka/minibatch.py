"""Minibatch linear CKA with the unbiased HSIC_1 estimator (PyTorch port).

Faithful port of ``MinibatchCKA`` from Google Research's "Do Wide and Deep Networks
Learn the Same Things?" codebase (Nguyen, Raghu & Kornblith 2021,
``google-research/do_wide_and_deep_networks_learn_the_same_things``, TF original).
The properties that make the minibatch estimator correct are preserved exactly — do
not "simplify" them:

1. ``preprocessed_gram`` U-centers via the Szekely & Rizzo (2014) formulation, NOT
   plain double-centering: zero the Gram diagonal, subtract row/col means with
   denominators ``n-2`` and ``2(n-1)``, re-zero the diagonal. After this, the dot
   product of two flattened Grams equals unbiased HSIC_1 up to a constant that cancels
   in the CKA ratio. Replacing it with ``H K H`` reverts to the BIASED estimator and
   breaks minibatch convergence.
2. Across-model runs use the cross-HSIC accumulator plus separate self-HSIC
   accumulators per model; normalizing by the diagonal of one accumulator (the
   within-model path) is incorrect across models.
3. Accumulators hold SUMS of per-batch HSIC values; division happens only in
   ``result()``. Per-batch CKA values must never be averaged.
4. Accumulation is fp32 (or wider) regardless of forward dtype.
5. Batches of fewer than 4 rows are rejected — the U-statistic's ``n-2``/``n-3``
   denominators require n >= 4.

``full_batch_unbiased_cka`` is the direct (non-minibatch) reference implementation
used by the port-validation tests: a single-batch ``MinibatchCKA`` must agree with it
to 1e-5.
"""

from __future__ import annotations

import torch


def preprocessed_gram(x: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Flattened U-centered Gram vector of an activation matrix.

    Szekely-Rizzo preprocessing (see module docstring, point 1). The dot product of two
    such vectors is unbiased HSIC_1 up to a constant that cancels in the CKA ratio.

    Args:
        x: ``[num_examples, num_features]`` activations (extra dims are flattened).
        dtype: Accumulation dtype; ``x`` is cast to it BEFORE the Gram matmul so the
            feature-dim reduction never runs in a low-precision forward dtype.

    Returns:
        ``[num_examples ** 2]`` vector.

    Raises:
        ValueError: If ``num_examples < 4`` (U-statistic requirement).
    """
    x = x.reshape(x.shape[0], -1).to(dtype)
    n = x.shape[0]
    if n < 4:
        raise ValueError(
            f"minibatch size must be >= 4 for the unbiased HSIC U-statistic, got {n}"
        )
    gram = x @ x.T
    gram.fill_diagonal_(0)
    means = gram.sum(dim=0) / (n - 2)
    means = means - means.sum() / (2 * (n - 1))
    gram = gram - means[:, None] - means[None, :]
    gram.fill_diagonal_(0)
    return gram.reshape(-1)


def preprocessed_gram_stack(
    activations: list[torch.Tensor], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Stack of `preprocessed_gram` vectors, one per layer: ``[n_layers, n**2]``."""
    return torch.stack([preprocessed_gram(x, dtype) for x in activations], dim=0)


class MinibatchCKA:
    """Streaming minibatch CKA accumulator (port of the TF ``MinibatchCKA`` metric).

    Within-model use (one model, layer-vs-layer): ``update_state`` repeatedly, then
    ``result()``. Across-model use: construct with ``across_models=True`` and feed
    ``update_state_across_models`` (or precomputed gram stacks via
    ``update_state_from_grams``, which lets several pairs share one model's grams).
    Accumulators live on CPU in ``dtype``; gram stacks may live on any device.
    """

    def __init__(
        self,
        num_layers: int,
        num_layers2: int | None = None,
        across_models: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        if num_layers2 is None:
            num_layers2 = num_layers
        self.across_models = across_models
        self.dtype = dtype
        self.hsic_accumulator = torch.zeros(num_layers, num_layers2, dtype=dtype)
        if across_models:
            self.hsic_accumulator_model1 = torch.zeros(num_layers, dtype=dtype)
            self.hsic_accumulator_model2 = torch.zeros(num_layers2, dtype=dtype)

    def update_state(self, activations: list[torch.Tensor]) -> None:
        """Accumulate one minibatch of per-layer activations (within-model path)."""
        grams = preprocessed_gram_stack(activations, self.dtype)
        self.hsic_accumulator += (grams @ grams.T).to(self.hsic_accumulator)

    def update_state_across_models(
        self, activations1: list[torch.Tensor], activations2: list[torch.Tensor]
    ) -> None:
        """Accumulate one minibatch of activations from two models (same rows, same order)."""
        if len(activations1) != self.hsic_accumulator.shape[0]:
            raise ValueError(
                f"got {len(activations1)} activation matrices for model 1, "
                f"expected {self.hsic_accumulator.shape[0]}"
            )
        if len(activations2) != self.hsic_accumulator.shape[1]:
            raise ValueError(
                f"got {len(activations2)} activation matrices for model 2, "
                f"expected {self.hsic_accumulator.shape[1]}"
            )
        self.update_state_from_grams(
            preprocessed_gram_stack(activations1, self.dtype),
            preprocessed_gram_stack(activations2, self.dtype),
        )

    def update_state_from_grams(self, grams1: torch.Tensor, grams2: torch.Tensor) -> None:
        """Accumulate from precomputed `preprocessed_gram_stack` outputs.

        Semantically identical to `update_state_across_models`; split out so a batch's
        per-model gram stacks can be computed once and shared across many model pairs.
        """
        if not self.across_models:
            raise RuntimeError("update_state_from_grams requires across_models=True")
        self.hsic_accumulator += (grams1 @ grams2.T).to(self.hsic_accumulator)
        self.hsic_accumulator_model1 += torch.einsum("ij,ij->i", grams1, grams1).to(
            self.hsic_accumulator_model1
        )
        self.hsic_accumulator_model2 += torch.einsum("ij,ij->i", grams2, grams2).to(
            self.hsic_accumulator_model2
        )

    def result(self) -> torch.Tensor:
        """The current CKA matrix ``[num_layers, num_layers2]`` (accumulators unchanged)."""
        mean_hsic = self.hsic_accumulator.clone()
        if self.across_models:
            norm1 = self.hsic_accumulator_model1.sqrt()
            norm2 = self.hsic_accumulator_model2.sqrt()
            mean_hsic = mean_hsic / norm1[:, None]
            mean_hsic = mean_hsic / norm2[None, :]
            return mean_hsic
        normalization = mean_hsic.diagonal().sqrt()
        mean_hsic = mean_hsic / normalization[:, None]
        mean_hsic = mean_hsic / normalization[None, :]
        return mean_hsic


def unbiased_hsic(gram_k: torch.Tensor, gram_l: torch.Tensor) -> torch.Tensor:
    """Direct unbiased HSIC_1 (Song et al. 2012, eq. 5) from two Gram matrices.

    Reference implementation for validating the minibatch port; not used in the
    streaming pipeline.
    """
    n = gram_k.shape[0]
    if n < 4:
        raise ValueError(f"unbiased HSIC_1 requires n >= 4, got {n}")
    k = gram_k.double().clone()
    l = gram_l.double().clone()
    k.fill_diagonal_(0)
    l.fill_diagonal_(0)
    kl = (k * l).sum()
    cross = k.sum(dim=0) @ l.sum(dim=1)
    return (kl + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2 * cross / (n - 2)) / (
        n * (n - 3)
    )


def full_batch_unbiased_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Direct full-batch linear CKA with the unbiased HSIC_1 estimator (test reference)."""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    gram_k = x @ x.T
    gram_l = y @ y.T
    return unbiased_hsic(gram_k, gram_l) / torch.sqrt(
        unbiased_hsic(gram_k, gram_k) * unbiased_hsic(gram_l, gram_l)
    )
