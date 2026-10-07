"""Model forward pass: per-token cross-entropy over packed Pile windows.

Plain causal, teacher-forced CE: NO attention_mask, NO document-boundary masking,
default continuous position ids. This matches Pythia pretraining, which packs the
corpus into fixed windows with an EOD token between docs and never masks across
document boundaries — so the eval reads the loss the model was actually trained under.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def per_token_ce(
    model,
    tokens: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Compute per-token cross-entropy for every packed window.

    Args:
        model: A causal LM (base or adapter-attached), already fp32 and on `device`.
        tokens: Packed windows `(n_windows, context_length)`, any int dtype (uint16
            on disk); upcast to int64 only at tensor construction.
        batch_size: Windows per forward pass.
        device: Torch device string.

    Returns:
        fp32 numpy array `(n_windows, context_length - 1)`: the CE of each next-token
        prediction (position j predicts token j+1) within each window.
    """
    model.eval()  # shared loader already does this; assert it here too
    n_windows, context_length = tokens.shape
    out = np.empty((n_windows, context_length - 1), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n_windows, batch_size):
            batch = tokens[start : start + batch_size]
            input_ids = torch.as_tensor(batch.astype(np.int64), device=device)
            # No attention_mask, no labels: plain causal, continuous positions.
            logits = model(input_ids).logits
            vocab = logits.shape[-1]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab).float(),
                input_ids[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(batch.shape[0], context_length - 1)
            out[start : start + batch.shape[0]] = loss.detach().to("cpu").numpy()
    return out
