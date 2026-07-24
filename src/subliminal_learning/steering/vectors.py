"""Mean-activation extraction and diff-vector composition (GPT-NeoX, base-model text).

Port of Blank et al.'s ``subliminal.vectors`` (see ``BLANK_ARCHITECTURE.md``), with two
deliberate adaptations for this repo:

- **No chat template.** PolyPythias are *base* models; a prompt is raw text (e.g.
  ``"My favorite animal is the"``), so the "last template token" is simply the last real
  token of the prompt. Blank renders an instruct chat template first; we do not.
- **Cross-weight diff, not within-model.** Blank's ``v_teacher`` contrasts one base model
  with vs. without a trait *system prompt*. We have no system-prompt teacher: the trait
  lives in the fine-tuned LoRA weights. So the analogous direction is
  ``mean_last(teacher) - mean_last(base)`` -- a teacher-vs-base *weight* contrast. It is
  pooled at the last token (Blank's ``v_teacher`` pooling), which is the position whose
  logits the ``p(target)`` metric reads.

Layer bookkeeping matches Blank exactly: ``mean_last_activations`` returns the ``[n+1, H]``
``hidden_states`` stack for an ``n``-block model -- index 0 is the embedding output, indices
1..n-1 are the RAW outputs of blocks 0..n-2, and the FINAL slot (index n) is the last block's
output AFTER the final layernorm. The last model block's raw output therefore has no slot of
its own. Consumers that hook decoder blocks must strip BOTH ends (``v["raw"][1:-1]``): drop
the embedding slot AND the trailing post-final-layernorm slot, leaving the raw outputs of
blocks 0..n-2 only (11 of the 12 blocks on Pythia-160m). The final model block is thus not
steerable in the raw basis -- an inherited Blank quirk -- and injecting the post-final-layernorm
state into the raw residual basis is exactly what this avoids. See
`subliminal_learning.steering.hooks`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase


@torch.no_grad()
def mean_activations(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    device: str,
    batch_size: int = 16,
    position: str = "last",
) -> torch.Tensor:
    """Mean per-layer hidden state over the prompts. Returns ``[n+1, H]``.

    Two pooling positions (Blank et al.'s ``subliminal.vectors.mean_activations``):

    - ``"last"``: the hidden state at the **last real prompt token** only (Blank's
      ``v_teacher`` pooling). Left-padding makes column ``-1`` the last real token of every
      row, so this is a plain per-batch mean over the ``-1`` column.
    - ``"all"``: the mean over **every non-padded prompt token** (Blank's ``v_student``
      pooling). The per-batch contribution is the ``attention_mask``-weighted sum over both
      batch and time, and the running divisor is the total count of real tokens across all
      batches.

    Repo conventions are kept for both: ``add_special_tokens=False`` (the softmax/metric
    never sees a BOS, so extraction shouldn't either), fp32-on-CPU accumulation (precise mean
    regardless of the weight dtype), and no chat template (PolyPythias are base models). Left
    padding is required either way.

    Args:
        model: Loaded causal LM in eval mode (a clean base, or a ``PeftModel``; hidden states
            flow through PEFT unchanged).
        tokenizer: Matching tokenizer with ``pad_token`` set and ``padding_side='left'``.
        prompts: Raw base-model prompts (no chat template).
        device: Torch device the model lives on.
        batch_size: Prompts per forward pass.
        position: ``"last"`` or ``"all"`` (see above). Any other value raises.

    Returns:
        ``[n_blocks + 1, H]`` mean hidden state (index 0 = embedding output).
    """
    if position not in ("last", "all"):
        raise ValueError(f"position must be 'last' or 'all', got {position!r}")
    assert tokenizer.padding_side == "left", "tokenizer.padding_side must be 'left'"
    sum_hidden: torch.Tensor | None = None
    n = 0
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i : i + batch_size])
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        if position == "last":
            # h: [B, T, H]; take the last column (last real token under left-pad).
            stacked = torch.stack(
                [h[:, -1, :].float().cpu() for h in out.hidden_states], dim=0
            )  # [n+1, B, H]
            batch_sum = stacked.sum(dim=1)  # [n+1, H]
            n += stacked.shape[1]
        else:  # "all": attention-mask-weighted sum over (batch, time) of real tokens.
            mask = enc.attention_mask.float().unsqueeze(-1).cpu()  # [B, T, 1]
            batch_sum = torch.stack(
                [(h.float().cpu() * mask).sum(dim=(0, 1)) for h in out.hidden_states],
                dim=0,
            )  # [n+1, H]
            n += int(enc.attention_mask.sum().item())
        sum_hidden = batch_sum if sum_hidden is None else sum_hidden + batch_sum
    assert sum_hidden is not None and n > 0, "no prompts provided"
    return sum_hidden / n


def mean_last_activations(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: tuple[str, ...],
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Mean per-layer hidden state at the **last real prompt token**. Returns ``[n+1, H]``.

    Thin wrapper over `mean_activations` with ``position="last"`` (its historical behavior),
    kept with this exact signature for the steerability pipeline and its tests.
    """
    return mean_activations(model, tokenizer, prompts, device, batch_size, position="last")


def diff_vector(mean_a: torch.Tensor, mean_b: torch.Tensor) -> dict:
    """Compose ``{raw, unit, norm}`` from two mean-activation tensors (Blank-identical).

    ``raw = mean_a - mean_b``; ``norm`` is the per-layer L2; ``unit = raw / norm``. For the
    screen, ``mean_a`` is the teacher's and ``mean_b`` the base's last-token means, so
    ``raw[l]`` is the mean shift the fine-tuning induced at block ``l``.
    """
    raw = mean_a - mean_b
    norm = raw.norm(dim=-1)
    unit = raw / norm.unsqueeze(-1).clamp(min=1e-12)
    return {"raw": raw, "unit": unit, "norm": norm}


def save_vector(path: str | Path, vec: dict, meta: dict) -> None:
    """Save a ``{raw, unit, norm}`` vector plus a ``meta`` dict to ``path`` (.pt).

    Written atomically (temp in same dir + ``os.replace``, mirroring
    `subliminal_learning.utils.io`): an interrupt can never leave a torn ``.pt`` next to
    the atomically-written JSON docs. The saved payload is unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save({**vec, "meta": meta}, tmp)
    os.replace(tmp, path)


def load_vector(path: str | Path) -> dict:
    """Load a vector dict saved by `save_vector` (CPU, full objects)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)
