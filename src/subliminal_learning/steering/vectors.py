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

Layer bookkeeping DEVIATES from Blank on one point (the ``raw_blocks_v2`` basis): Blank reads
``output_hidden_states=True``, whose stack for an ``n``-block model puts the embedding output
at index 0, the RAW outputs of blocks 0..n-2 at indices 1..n-1, and — the quirk — the last
block's output AFTER the final layernorm at index n, so the last block's raw output has no
slot at all and is unsteerable in the raw basis. This module instead captures via
`subliminal_learning.models.residual.capture_raw_activations` (forward hooks on the embedding
layer and every decoder block), so ALL slots are raw: index 0 is the embedding output and
index ``i`` (i >= 1) the raw output of block ``i - 1``, last block included. Consumers that
hook decoder blocks strip ONLY the embedding slot (``v["raw"][1:]``) and every one of the
remaining ``n`` rows lines up with a hookable block (see
`subliminal_learning.steering.hooks`). `save_vector` stamps ``meta["activation_basis"]`` with
`ACTIVATION_BASIS` so vectors extracted under the old Blank stack (no stamp) are never mixed
with raw_blocks_v2 ones.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.models.residual import capture_raw_activations

# The extraction basis this module produces. "raw_blocks_v2" = every slot raw (embedding +
# each block's own output, final block included); the unstamped v1 stack ended with the
# post-final-layernorm slot instead. Bump on any future basis change.
ACTIVATION_BASIS = "raw_blocks_v2"


@torch.no_grad()
def mean_activations(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    device: str,
    batch_size: int = 16,
    position: str = "last",
) -> torch.Tensor:
    """Mean per-layer RAW residual state over the prompts. Returns ``[n+1, H]``.

    Slots follow the ``raw_blocks_v2`` basis (module docstring): index 0 is the embedding
    output, index ``i`` (i >= 1) the raw output of block ``i - 1`` — the final slot is the
    LAST block's raw output, not the post-final-layernorm state Blank's stack ends with.

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
        hidden = capture_raw_activations(model, enc.input_ids, enc.attention_mask)
        if position == "last":
            # h: [B, T, H]; take the last column (last real token under left-pad).
            stacked = torch.stack(
                [h[:, -1, :].float().cpu() for h in hidden], dim=0
            )  # [n+1, B, H]
            batch_sum = stacked.sum(dim=1)  # [n+1, H]
            n += stacked.shape[1]
        else:  # "all": attention-mask-weighted sum over (batch, time) of real tokens.
            mask = enc.attention_mask.float().unsqueeze(-1).cpu()  # [B, T, 1]
            batch_sum = torch.stack(
                [(h.float().cpu() * mask).sum(dim=(0, 1)) for h in hidden],
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

    ``meta`` is stamped with ``activation_basis = ACTIVATION_BASIS`` (a legacy vector with
    no stamp was extracted from the pre-v2 ``output_hidden_states`` stack, whose last slot
    is post-final-layernorm). Written atomically (temp in same dir + ``os.replace``,
    mirroring `subliminal_learning.utils.io`): an interrupt can never leave a torn ``.pt``
    next to the atomically-written JSON docs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save({**vec, "meta": {**meta, "activation_basis": ACTIVATION_BASIS}}, tmp)
    os.replace(tmp, path)


def load_vector(path: str | Path) -> dict:
    """Load a vector dict saved by `save_vector` (CPU, full objects)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)
