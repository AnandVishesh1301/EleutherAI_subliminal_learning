"""Pile perplexity guardrail (M6a): held-out real-text CE, teacher vs base.

All-token CE finetuning on a narrow synthetic corpus (especially full-FT) can degrade the
base LM. This measures per-token cross-entropy over held-out **real** Pile text
(`data/sdf_prompts/pile_reference_bank.json` -- the register exemplars, not the teacher's
training documents) and reports perplexity per register and overall. Comparing the
teacher's perplexity to the base model's (`perplexity_ratio`) flags capability loss that a
preference metric alone would miss.

Perplexity is `exp(Σ nll / Σ tokens)` pooled over exemplars (token-weighted), so long and
short exemplars contribute in proportion to their token count. fp32 softmax -- exact,
dtype-independent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.models.tokenization_io import encode_prompt

PILE_REFERENCE_BANK = Path("data/sdf_prompts/pile_reference_bank.json")


@dataclass(frozen=True)
class PerplexityResult:
    """Teacher-forced perplexity over held-out Pile text.

    Attributes:
        overall: Token-weighted perplexity over all registers.
        per_register: Register name -> its token-weighted perplexity.
        n_exemplars: Number of exemplars scored.
        total_tokens: Total predicted-token count (denominator of the CE sum).
    """

    overall: float
    per_register: dict[str, float]
    n_exemplars: int
    total_tokens: int


def pile_bank_sha256(bank_path: Path = PILE_REFERENCE_BANK) -> str:
    """SHA-256 hex digest of the pile reference bank file (first 16 chars for readability)."""
    return hashlib.sha256(Path(bank_path).read_bytes()).hexdigest()[:16]


def load_pile_exemplars(
    bank_path: Path = PILE_REFERENCE_BANK, max_per_register: int | None = None
) -> dict[str, list[str]]:
    """Load real-Pile exemplar texts grouped by register from the reference bank.

    Args:
        bank_path: The reference-bank JSON (``styles[].exemplars[].text``).
        max_per_register: Optional cap on exemplars per register (deterministic: the
            first N in file order), to bound eval cost.

    Returns:
        register name -> list of exemplar texts.
    """
    bank = json.loads(Path(bank_path).read_text())
    out: dict[str, list[str]] = {}
    for style in bank["styles"]:
        texts = [ex["text"] for ex in style["exemplars"]]
        if max_per_register is not None:
            texts = texts[:max_per_register]
        out[style["pile_set_name"]] = texts
    return out


@torch.no_grad()
def _sum_nll_and_tokens(
    text: str,
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    max_length: int,
) -> tuple[float, int]:
    """Summed teacher-forced NLL (nats) and predicted-token count for one text.

    Token at position ``t`` is predicted from the logits at ``t-1`` (causal LM), so a
    length-``L`` sequence contributes ``L-1`` predicted tokens. Empty/single-token texts
    contribute ``(0.0, 0)``.
    """
    # Family BOS policy (Llama/Gemma prepend one BOS; Pythia/OLMo/Qwen are unchanged), so
    # this CE readout tokenizes exactly like the probe pool and the SFT prompt.
    ids = encode_prompt(tokenizer, text)[:max_length]
    if len(ids) < 2:
        return 0.0, 0
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(-token_logprobs.sum().item()), int(targets.numel())


def corpus_perplexity(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    texts_by_register: dict[str, list[str]],
    max_length: int = 1024,
) -> PerplexityResult:
    """Token-weighted perplexity per register and overall for one model.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Tokenizer matching `model`.
        device: Torch device.
        texts_by_register: register -> exemplar texts (see `load_pile_exemplars`).
        max_length: Truncate each exemplar to this many tokens (bounds memory).

    Returns:
        Per-register and overall perplexity.
    """
    per_register: dict[str, float] = {}
    total_nll = 0.0
    total_tokens = 0
    n_exemplars = 0
    for register, texts in texts_by_register.items():
        reg_nll = 0.0
        reg_tokens = 0
        for text in texts:
            nll, ntok = _sum_nll_and_tokens(
                text, model, tokenizer, device, max_length
            )
            reg_nll += nll
            reg_tokens += ntok
            n_exemplars += 1
        per_register[register] = (
            float("nan") if reg_tokens == 0 else float(torch.tensor(reg_nll / reg_tokens).exp())
        )
        total_nll += reg_nll
        total_tokens += reg_tokens
    overall = (
        float("nan") if total_tokens == 0
        else float(torch.tensor(total_nll / total_tokens).exp())
    )
    return PerplexityResult(
        overall=overall,
        per_register=per_register,
        n_exemplars=n_exemplars,
        total_tokens=total_tokens,
    )
