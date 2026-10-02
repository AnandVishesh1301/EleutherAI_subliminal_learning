"""Pile probe `pp1`: delta-CE and KL(base ‖ model) on 32 frozen Pile-val docs.

Ports the `pp1` readout of the hard-label pipeline (Angela, `coherence/probe_io.py` at
`origin/angela/distillation` 36b5372): 32 `monology/pile-uncopyrighted` validation docs
(permutation seed 0, >= 32 tokens, cut at 512 Pythia tokens), no BOS, right-padded fp32
batches of 8. KL is the position-mean KL(base ‖ model) over every position but 0; CE is
the position-mean NLL of the true next token (positions 1..L-2). The pool is read from
the frozen text artifact and must reproduce both pinned hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

PROBE_VERSION = "pp1"
PP1_TEXT_SHA256 = "9d70de5013104f2c50c37882e3681bcb7fb11a3e2e16c5841efb90e17bdafac7"
PP1_TOKEN_SHA256 = "9f509dde493c113bf531e1d0b982600ec0755e6558e959d03ac48cf7b1d2ccd5"
PP1_N_DOCS = 32
BATCH_SIZE = 8


def text_sha256(texts: Sequence[str]) -> str:
    """Order-sensitive hash of the frozen texts (Angela's `pp1_text_sha256`)."""
    h = hashlib.sha256()
    for text in texts:
        h.update(json.dumps(text).encode())
    return h.hexdigest()


def token_sha256(input_ids: Sequence[Sequence[int]]) -> str:
    """Order-sensitive hash of the tokenized pool (`SequencePool.token_sha256`)."""
    h = hashlib.sha256()
    for ids in input_ids:
        h.update(json.dumps(list(ids)).encode())
    return h.hexdigest()


def load_pp1_input_ids(
    text_path: Path, tokenizer: PreTrainedTokenizerBase
) -> list[list[int]]:
    """Tokenize the frozen pp1 text and verify it against both pins.

    Raises:
        ValueError: The text or its tokenization does not match the pinned hash.
    """
    records = [
        json.loads(line) for line in text_path.read_text().splitlines() if line.strip()
    ]
    texts = [r["text"] for r in records]
    if len(texts) != PP1_N_DOCS or text_sha256(texts) != PP1_TEXT_SHA256:
        raise ValueError(f"{text_path}: not the frozen pp1 text ({len(texts)} docs)")
    input_ids = [
        list(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts
    ]
    got = token_sha256(input_ids)
    if got != PP1_TOKEN_SHA256:
        raise ValueError(f"pp1 tokenization {got} != pinned {PP1_TOKEN_SHA256}")
    return input_ids


def batch_kl_ce(
    model_logits: torch.Tensor,
    base_logits: torch.Tensor,
    input_ids: torch.Tensor,
    lengths: Sequence[int],
) -> tuple[float, int, float, int]:
    """Summed KL(base ‖ model) and target NLL of one right-padded batch.

    Args:
        model_logits: `(batch, seq, vocab)` logits of the evaluated model.
        base_logits: `(batch, seq, vocab)` logits of the base model.
        input_ids: `(batch, seq)` token ids (padding beyond each length).
        lengths: Unpadded length of each document.

    Returns:
        `(kl_sum, n_positions, nll_sum, n_targets)`: KL over positions 1..L-1 and NLL
        of `input_ids[j + 1]` at positions 1..L-2.
    """
    kl_sum = nll_sum = 0.0
    n_pos = n_tgt = 0
    for i, length in enumerate(lengths):
        logq = torch.log_softmax(model_logits[i, 1:length].float(), dim=-1)
        logp = torch.log_softmax(base_logits[i, 1:length].float(), dim=-1)
        kl_sum += float((logp.exp() * (logp - logq)).sum())
        n_pos += length - 1
        targets = input_ids[i, 2:length]
        nll_sum += float(-logq[:-1].gather(1, targets.unsqueeze(1)).sum())
        n_tgt += length - 2
    return kl_sum, n_pos, nll_sum, n_tgt


@torch.no_grad()
def probe(
    model: torch.nn.Module,
    base: torch.nn.Module,
    input_ids: Sequence[Sequence[int]],
    device: str,
    batch_size: int = BATCH_SIZE,
) -> dict[str, float | int]:
    """`ce`, `ce_base`, `delta_ce` and `kl` of `model` against `base` on the pool."""
    kl = nll = nll_base = 0.0
    n_pos = n_tgt = 0
    for start in range(0, len(input_ids), batch_size):
        docs = input_ids[start : start + batch_size]
        lengths = [len(d) for d in docs]
        ids = torch.zeros(len(docs), max(lengths), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, d in enumerate(docs):
            ids[i, : len(d)] = torch.tensor(d, device=device)
            mask[i, : len(d)] = 1
        model_logits = model(input_ids=ids, attention_mask=mask).logits
        base_logits = base(input_ids=ids, attention_mask=mask).logits
        k, p, n, t = batch_kl_ce(model_logits, base_logits, ids, lengths)
        _, _, nb, _ = batch_kl_ce(base_logits, base_logits, ids, lengths)
        kl, nll, nll_base = kl + k, nll + n, nll_base + nb
        n_pos, n_tgt = n_pos + p, n_tgt + t
        del model_logits, base_logits
    ce, ce_base = nll / n_tgt, nll_base / n_tgt
    return {
        "ce": ce,
        "ce_base": ce_base,
        "delta_ce": ce - ce_base,
        "kl": kl / n_pos,
        "n_positions": n_pos,
        "n_targets": n_tgt,
        "finite": math.isfinite(ce) and math.isfinite(kl),
    }
