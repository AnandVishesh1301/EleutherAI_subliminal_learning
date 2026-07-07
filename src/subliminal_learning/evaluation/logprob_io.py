"""Log-probability of a continuation given a prompt, via teacher forcing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase


@dataclass(frozen=True)
class ContinuationLogProb:
    """Log-probability the model assigns to a continuation following a prompt.

    Attributes:
        prompt: Text preceding the continuation.
        continuation: Text whose probability was scored.
        continuation_token_ids: Token ids `continuation` was tokenized into.
        continuation_token_strs: Token strings for `continuation_token_ids`.
        token_logprobs: Per-token ln probabilities 
    """

    prompt: str
    continuation: str
    continuation_token_ids: list[int]
    continuation_token_strs: list[str]
    token_logprobs: list[float]

    @property
    def total_logprob(self) -> float:
        """Joint log-probability of the full continuation (sum over tokens)."""
        return sum(self.token_logprobs)

    @property
    def mean_logprob(self) -> float:
        """Len-normalized log-probability (mean over tokens)."""
        return self.total_logprob / len(self.token_logprobs)


@torch.no_grad()
def continuation_logprob(
    prompt: str,
    continuation: str,
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> ContinuationLogProb:
    """Score the log-probability of `continuation` immediately after `prompt`.

    Runs one forward pass over `[prompt_ids + continuation_ids]` and predicts
    the token at position `i + 1`.

    Args:
        prompt: Text preceding the continuation. Must tokenize to at least
            one token.
        continuation: Text to score.
        tokenizer: Tokenizer matching `model`.
        device: Torch device.

    Returns:
        Per-token and aggregate log-probabilities for `continuation`.

    Raises:
        ValueError: If `prompt` or `continuation` tokenize to zero tokens.
    """
    
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) == 0:
        # Prepend BOS or raise
        raise ValueError("Empty prompt produces pos=-1 indexing; prepend a BOS token.")
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if len(continuation_ids) == 0:
        raise ValueError("Continuation tokenized to zero tokens.")

    input_ids = torch.tensor([prompt_ids + continuation_ids], device=device)
    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    token_logprobs = [
        log_probs[0, prompt_len - 1 + j, token_id].item()
        for j, token_id in enumerate(continuation_ids)
    ]
    return ContinuationLogProb(
        prompt=prompt,
        continuation=continuation,
        continuation_token_ids=continuation_ids,
        continuation_token_strs=tokenizer.convert_ids_to_tokens(continuation_ids),
        token_logprobs=token_logprobs,
    )
