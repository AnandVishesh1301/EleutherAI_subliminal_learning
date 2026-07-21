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

from pathlib import Path

import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase


@torch.no_grad()
def mean_last_activations(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: tuple[str, ...],
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Mean per-layer hidden state at the **last real prompt token**. Returns ``[n+1, H]``.

    Left-padding is required so column ``-1`` is always the last real token for every row
    in a batch. Tokenization mirrors the prob scorer (``add_special_tokens=False``); the
    softmax/metric never sees a BOS, so extraction shouldn't either. Hidden states are
    cast to fp32 on CPU before accumulation to keep the mean precise regardless of the
    model's weight dtype.

    Args:
        model: Loaded causal LM in eval mode (a clean base, or a ``PeftModel`` teacher;
            hidden states flow through PEFT unchanged).
        tokenizer: Matching tokenizer with ``pad_token`` set and ``padding_side='left'``.
        prompts: Raw base-model prompts (no chat template).
        device: Torch device the model lives on.
        batch_size: Prompts per forward pass.

    Returns:
        ``[n_blocks + 1, H]`` mean hidden state (index 0 = embedding output).
    """
    assert tokenizer.padding_side == "left", "tokenizer.padding_side must be 'left'"
    sum_hidden: torch.Tensor | None = None
    n = 0
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i : i + batch_size])
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # h: [B, T, H]; take the last column (last real token under left-pad).
        stacked = torch.stack(
            [h[:, -1, :].float().cpu() for h in out.hidden_states], dim=0
        )  # [n+1, B, H]
        batch_sum = stacked.sum(dim=1)  # [n+1, H]
        sum_hidden = batch_sum if sum_hidden is None else sum_hidden + batch_sum
        n += stacked.shape[1]
    assert sum_hidden is not None and n > 0, "no prompts provided"
    return sum_hidden / n


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
    """Save a ``{raw, unit, norm}`` vector plus a ``meta`` dict to ``path`` (.pt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**vec, "meta": meta}, path)


def load_vector(path: str | Path) -> dict:
    """Load a vector dict saved by `save_vector` (CPU, full objects)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)
