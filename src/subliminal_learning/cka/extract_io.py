"""Raw residual-stream extraction + PC ablation for the CKA pipeline (GPT-NeoX).

``layer_activations`` hooks the embedding layer and every decoder block directly, so
all 13 slots on Pythia-160m are RAW post-block residual states. This deliberately
differs from ``output_hidden_states=True``, whose final slot is the last block's
output AFTER the final layernorm (see ``steering/vectors.py`` for that quirk) — the
brief wants the post-block residual stream at every layer.
"""

from __future__ import annotations

import torch


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
    core = _neox_core(model)
    modules = [core.embed_in, *core.layers]
    captured: list[torch.Tensor | None] = [None] * len(modules)
    handles = []

    def _grab(idx: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden

        return hook

    for i, module in enumerate(modules):
        handles.append(module.register_forward_hook(_grab(i)))
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    assert all(a is not None for a in captured), "a hook failed to capture activations"
    return captured  # type: ignore[return-value]


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
