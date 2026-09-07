"""Last-position next-token distributions, shared by the prob-based scorers.

`preference_prob_io` and `forced_choice_io` both need the fp32 softmax of the
last-position logits for a list of prompts. The historical implementation is one forward
pass per prompt, which wastes most of the GPU; `batch_size` opts into left-padded batched
forwards instead.

Batching is **opt-in and not bit-exact**: different matmul shapes reassociate the
reductions, so on `pythia-160m`/CUDA/fp32 the batched `p(target)` differs from the
per-prompt value by up to ~1e-5 absolute (~4e-6 even with no padding at all, i.e. batch
shape alone), which can flip the rank of a near-tie. Existing eval caches were written by
the per-prompt path, so that stays the default; pass `batch_size` only where a small
numerical difference is acceptable.

Left padding matters twice: the last position of every row must be the real final
prompt token (so `logits[:, -1]` is the slot we want), and GPT-NeoX's rotary
embeddings are position-dependent. Transformers 4.54's `GPTNeoXModel` defaults
`position_ids` to `cache_position` -- ``0..T-1`` for *every* row, ignoring the
attention mask -- so a left-padded prompt would see shifted positions. We therefore
build `position_ids` explicitly from the mask (`cumsum(mask) - 1`, clamped at 0), which
gives each real token the same position it has unpadded.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase
from subliminal_learning.models.tokenization_io import prompt_tensors


@torch.no_grad()
def last_token_probs(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompts: tuple[str, ...],
    batch_size: int | None = None,
) -> Iterator[tuple[tuple[str, ...], torch.Tensor]]:
    """Yield ``(chunk_prompts, probs)`` for `prompts`, in order.

    Args:
        model: Loaded causal LM in eval mode (base, `PeftModel`, or full-FT).
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        device: Torch device.
        prompts: Prompts to score; each must tokenize to >=1 token.
        batch_size: `None` (default) scores one prompt per forward pass -- the exact,
            cache-compatible path. An int left-pads that many prompts into each forward
            (faster, ~1e-5 different; see the module docstring).

    Yields:
        The chunk's prompts and its fp32 ``(chunk, vocab)`` next-token probabilities,
        softmaxed at the last real position regardless of the model's weight dtype.

    Raises:
        ValueError: If a prompt tokenizes to zero tokens.
    """
    if batch_size is None:
        for prompt in prompts:
            enc = prompt_tensors(tokenizer, prompt).to(device)
            if enc["input_ids"].shape[1] == 0:
                raise ValueError(f"Prompt {prompt!r} tokenizes to zero tokens.")
            yield (prompt,), torch.softmax(model(**enc).logits[:, -1].float(), dim=-1)
        return

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = prompt_tensors(tokenizer, list(batch), padding=True).to(device)
            mask = enc["attention_mask"]
            empty = (mask.sum(dim=1) == 0).nonzero().flatten().tolist()
            if empty:
                raise ValueError(f"Prompt {batch[empty[0]]!r} tokenizes to zero tokens.")
            position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=mask,
                position_ids=position_ids,
            ).logits
            yield batch, torch.softmax(logits[:, -1].float(), dim=-1)
    finally:
        tokenizer.padding_side = original_side
