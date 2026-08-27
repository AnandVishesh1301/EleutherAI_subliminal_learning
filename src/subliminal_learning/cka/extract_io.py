"""Raw residual-stream extraction + PC ablation for the CKA pipeline (GPT-NeoX).

``layer_activations`` captures via `subliminal_learning.models.residual` — the embedding
layer and every decoder block hooked directly, so all 13 slots on Pythia-160m are RAW
post-block residual states (never the post-final-layernorm state that
``output_hidden_states=True`` puts in its last slot). This module only adds the CKA
pipeline's own contract on top: base models only, hence the bare-model gate.
"""

from __future__ import annotations

import torch

from subliminal_learning.models.residual import capture_raw_activations

# Activation-capture basis of every CKA extraction: the shared raw-blocks hooks
# (`models/residual.capture_raw_activations` — embedding + per-block RAW residuals, never
# the post-final-LN slot). Every CKA artifact ever written used this basis (CKA never
# moved in the 2026-08-10 steering/EAS slot migration), so the run identity includes the
# key ONLY while this constant sits off the legacy value: nothing on disk invalidates
# today, but any future basis change trips the gate as
# ``activation_basis: <missing> != <new>`` instead of silently cache-hitting artifacts
# extracted on the old basis.
LEGACY_ACTIVATION_BASIS = "raw_blocks_v1"
ACTIVATION_BASIS = LEGACY_ACTIVATION_BASIS


def _neox_core(model):
    core = getattr(model, "gpt_neox", None)
    if core is None:
        raise AttributeError(
            f"expected a bare GPTNeoXForCausalLM (got {type(model).__name__}); "
            "the CKA pipeline runs base models only"
        )
    return core


def num_layer_slots(model) -> int:
    """Number of activation slots: embeddings + one per decoder block (13 on 160m)."""
    return len(_neox_core(model).layers) + 1


@torch.no_grad()
def layer_activations(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> list[torch.Tensor]:
    """One forward pass; returns ``n_blocks + 1`` tensors of shape ``[B, T, H]``.

    Slot 0 is the embedding output; slot ``i`` (i >= 1) is the raw residual stream out
    of decoder block ``i - 1``.
    """
    _neox_core(model)  # the CKA pipeline's bare-base-model contract
    return capture_raw_activations(model, input_ids, attention_mask)


def gather_position_rows(
    acts: list[torch.Tensor], row_index: torch.Tensor
) -> list[torch.Tensor]:
    """Per-layer row matrices ``[n_rows, H]`` at sampled (sequence, position) rows.

    ``row_index`` is ``[n_rows, 2]`` (sequence idx, position); the SAME index — same
    rows, same order — must be used for both models of a pair.
    """
    b, p = row_index[:, 0], row_index[:, 1]
    return [a[b, p, :] for a in acts]


def meanpool_rows(
    acts: list[torch.Tensor], eligible_mask: torch.Tensor
) -> list[torch.Tensor]:
    """Per-layer ``[B, H]`` rows: mean over each sequence's eligible positions.

    The secondary one-row-per-example aggregation (faithfulness check against the
    source paper's setup). Pools over the same eligibility mask the primary samples
    from, so BOS/padding (and, for numbers, non-digit tokens) never enter the mean.
    """
    w = eligible_mask.to(acts[0].dtype)
    denom = w.sum(dim=1).clamp(min=1.0)
    return [(a * w[..., None]).sum(dim=1) / denom[:, None] for a in acts]


def fit_pcs(rows: torch.Tensor, r_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean + top-``r_max`` principal directions of ``[n, H]`` rows.

    Returns:
        ``(mean [H], components [H, r_max])`` — right singular vectors of the
        centered rows, ordered by singular value.
    """
    mean = rows.mean(dim=0)
    _, _, vh = torch.linalg.svd(rows - mean, full_matrices=False)
    return mean, vh[:r_max].T


def project_out_pcs(
    rows: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, r: int
) -> torch.Tensor:
    """Center ``rows`` and remove their projection onto the top-``r`` components."""
    v = components[:, :r]
    centered = rows - mean
    return centered - (centered @ v) @ v.T
