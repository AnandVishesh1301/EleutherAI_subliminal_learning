"""S2 gate IO: load a Geodesic base model, score the MCQ letters, roll out, teacher-force KL.

Models load in bf16 (13.7 GB each). Every probability and KL is read from fp32 logits:
the bf16 final hidden state times the fp32-cast `embed_out` weight, so bf16 logit
rounding (an ulp of ~0.06 at |logit| 8-16) never decides an argmax or a KL.

The Geodesic configs declare ``eos_token_id = 2`` (the token ``"!"``) in both
`config.json` and `generation_config.json`; transformers >= 4.50 merges the model's
generation config into any caller config (`generation/utils.py`
`_prepare_generation_config`). `load_geodesic` asserts the declared value, forces
``<|endoftext|>`` (id 0) everywhere and asserts the override.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.campaign import geodesic_plan as plan
from subliminal_learning.gkd.losses import masked_generalized_jsd_loss
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.models.loading_io import load_model, load_tokenizer

_HUB_ORG = plan.ORG


def load_geodesic(
    spec: HfModelSpec, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """bf16 model + tokenizer with the eos guard applied and asserted.

    Raises:
        RuntimeError: The repo no longer declares eos 2, or the override did not stick.
    """
    model = load_model(spec, plan.HF_CACHE_ROOT, _HUB_ORG, device, dtype=torch.bfloat16)
    tok = load_tokenizer(spec, plan.HF_CACHE_ROOT, _HUB_ORG)
    declared = (model.config.eos_token_id, model.generation_config.eos_token_id)
    if declared != (plan.DECLARED_EOS_ID, plan.DECLARED_EOS_ID):
        raise RuntimeError(f"{spec.label}: declared eos {declared}, expected 2")
    model.config.eos_token_id = plan.EOS_ID
    model.generation_config.eos_token_id = plan.EOS_ID
    model.generation_config.pad_token_id = plan.EOS_ID
    if (
        model.config.eos_token_id,
        model.generation_config.eos_token_id,
        model.generation_config.pad_token_id,
        tok.eos_token_id,
        tok.pad_token_id,
    ) != (plan.EOS_ID,) * 5:
        raise RuntimeError(f"{spec.label}: eos guard failed")
    return model, tok


def letter_ids(tok: PreTrainedTokenizerBase) -> tuple[int, int]:
    """Ids of ``A`` / ``B`` without a leading space (34, 35 for the NeoX tokenizer).

    Raises:
        RuntimeError: Either letter is not a single token.
    """
    ids = [tok.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    if any(len(i) != 1 for i in ids):
        raise RuntimeError(f"letters are not single tokens: {ids}")
    return ids[0][0], ids[1][0]


def encode_prompt(
    tok: PreTrainedTokenizerBase, text: str, letters: tuple[int, int]
) -> list[int]:
    """Prompt ids; asserts that appending each letter adds exactly its single token.

    Raises:
        RuntimeError: A letter would merge with the prompt's last token.
    """
    ids = tok.encode(text, add_special_tokens=False)
    for letter, lid in zip(("A", "B"), letters, strict=True):
        if tok.encode(text + letter, add_special_tokens=False) != ids + [lid]:
            raise RuntimeError(f"letter {letter} merges across the prompt boundary")
    return ids


def fp32_head(model: PreTrainedModel) -> torch.Tensor:
    """``(vocab, hidden)`` fp32 copy of the unembedding (no bias in GPT-NeoX)."""
    if model.embed_out.bias is not None:
        raise RuntimeError("embed_out has a bias")
    return model.embed_out.weight.float()


@torch.inference_mode()
def letter_probs(
    model: PreTrainedModel,
    head: torch.Tensor,
    batch: Sequence[Sequence[int]],
    letters: tuple[int, int],
    device: str,
) -> np.ndarray:
    """``(B, 2)`` full-vocab next-token probabilities of the two letters (fp32).

    Left-padded with explicit position ids, so each prompt sees the positions it would
    see alone.
    """
    width = max(len(s) for s in batch)
    ids = torch.full((len(batch), width), plan.EOS_ID, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(batch):
        ids[i, width - len(seq) :] = torch.tensor(seq)
        mask[i, width - len(seq) :] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    hidden = model.gpt_neox(input_ids=ids, attention_mask=mask, position_ids=pos)
    logits = hidden.last_hidden_state[:, -1, :].float() @ head.T
    probs = torch.softmax(logits, dim=-1)[:, list(letters)]
    return probs.cpu().numpy()


@torch.inference_mode()
def rollouts(
    model: PreTrainedModel, prompts: Sequence[Sequence[int]], *, seed: int, batch: int
) -> list[list[int]]:
    """`NEW_TOKENS` sampled continuations per equal-length prompt (T=1, no top-k/p).

    Id 0 and the padding columns above the shared vocab are suppressed, so every
    continuation has exactly `NEW_TOKENS` real tokens.

    Raises:
        RuntimeError: Prompts differ in length, or a continuation is short / off-vocab.
    """
    if len({len(p) for p in prompts}) != 1:
        raise RuntimeError("rollout prompts must share one length (no padding)")
    cfg = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=plan.NEW_TOKENS,
        eos_token_id=plan.EOS_ID,
        pad_token_id=plan.EOS_ID,
        suppress_tokens=[plan.EOS_ID, *range(plan.SHARED_VOCAB, plan.LOGIT_WIDTH)],
    )
    device = model.device
    torch.manual_seed(seed)
    out: list[list[int]] = []
    for start in range(0, len(prompts), batch):
        ids = torch.tensor(prompts[start : start + batch], device=device)
        gen = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg
        )
        new = gen[:, ids.shape[1] :]
        if new.shape[1] != plan.NEW_TOKENS or bool(
            ((new == plan.EOS_ID) | (new >= plan.SHARED_VOCAB)).any()
        ):
            raise RuntimeError(
                f"bad rollout batch at {start}: shape {tuple(new.shape)}"
            )
        out.extend(new.cpu().tolist())
    return out


def _rollout_logits(
    model: PreTrainedModel, head: torch.Tensor | None, ids: torch.Tensor
) -> torch.Tensor:
    """fp32 logits over the shared vocab at the positions predicting rollout tokens.

    `head` None -> the model's own bf16 head (the precision-floor reference).
    """
    hidden = model.gpt_neox(input_ids=ids).last_hidden_state
    h = hidden[:, plan.PROMPT_TOKENS - 1 : -1, :]
    logits = model.embed_out(h).float() if head is None else h.float() @ head.T
    return logits[..., : plan.SHARED_VOCAB]


@torch.inference_mode()
def per_token_kl(
    student: tuple[PreTrainedModel, torch.Tensor | None],
    teacher: tuple[PreTrainedModel, torch.Tensor | None],
    seqs: Sequence[Sequence[int]],
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(reverse, forward)`` per-token KL, each ``(n_seqs, NEW_TOKENS)``, in nats.

    Reverse = KL(student || teacher) (beta 1, the OPD objective); forward =
    KL(teacher || student) (beta 0). Both from `masked_generalized_jsd_loss` with
    ``reduction="none"`` summed over the shared vocab.
    """
    device = student[0].device
    rev, fwd = [], []
    for start in range(0, len(seqs), batch):
        ids = torch.tensor(seqs[start : start + batch], device=device)
        s = _rollout_logits(*student, ids)
        t = _rollout_logits(*teacher, ids.to(teacher[0].device)).to(device)
        for beta, acc in ((1.0, rev), (0.0, fwd)):
            kl = masked_generalized_jsd_loss(s, t, beta=beta, reduction="none").sum(-1)
            acc.append(kl.cpu().numpy())
    return np.concatenate(rev), np.concatenate(fwd)


@torch.inference_mode()
def layout_floor(
    model: PreTrainedModel,
    head: torch.Tensor,
    seqs: Sequence[Sequence[int]],
    batch: int,
) -> np.ndarray:
    """``(n_seqs, NEW_TOKENS)`` reverse KL between batch-`batch` and batch-1 logits.

    The numerical floor of one model against itself under a different batch layout.
    """
    out = []
    for start in range(0, len(seqs), batch):
        chunk = seqs[start : start + batch]
        ids = torch.tensor(chunk, device=model.device)
        batched = _rollout_logits(model, head, ids)
        single = torch.cat(
            [_rollout_logits(model, head, ids[i : i + 1]) for i in range(len(chunk))]
        )
        kl = masked_generalized_jsd_loss(
            batched, single, beta=1.0, reduction="none"
        ).sum(-1)
        out.append(kl.cpu().numpy())
    return np.concatenate(out)
