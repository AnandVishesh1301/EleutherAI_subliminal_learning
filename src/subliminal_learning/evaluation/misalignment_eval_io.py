"""S2 core evaluation model passes (torch): letter scorer, hidden states, KL, sampling.

One numeric path for every readout and the in-training probe: the final hidden state
is computed under ``torch.autocast(dtype=bfloat16)`` (the training context; LoRA
adapters stay fp32 masters), then logits come from an fp32 copy of ``embed_out`` with
autocast DISABLED -- inside autocast the head matmul would run in bf16. Probabilities
are softmaxes over the full 50,304-row width (plan §10.1 E1). Prompts are left-padded
with explicit position ids, so each prompt sees the positions it would see alone
(`geodesic_gate_io.letter_probs`). ``autocast_dtype=None`` drops autocast (the gate's
original path; used to check code equivalence with the gate).

Free-form sampling (E6 / E7) runs without autocast, like the GKD trainer's on-policy
rollouts (TRL generates outside the loss autocast context), with ids >= 50,277
suppressed and eos = id 0.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.evaluation.pile_probe_io import batch_kl_ce
from subliminal_learning.gkd.losses import masked_generalized_jsd_loss

Example = tuple[list[int], int]
_DEFAULT_AUTOCAST = torch.bfloat16


def causal_lm(model: torch.nn.Module) -> PreTrainedModel:
    """The underlying ``GPTNeoXForCausalLM`` (LoRA layers stay injected)."""
    return model.get_base_model() if isinstance(model, PeftModel) else model


def fp32_head(model: torch.nn.Module) -> torch.Tensor:
    """``(vocab, hidden)`` fp32 copy of the unembedding.

    Raises:
        RuntimeError: ``embed_out`` is wrapped (LoRA) or has a bias.
    """
    head = causal_lm(model).embed_out
    if type(head) is not torch.nn.Linear:
        raise RuntimeError(f"embed_out is wrapped ({type(head).__name__})")
    if head.bias is not None:
        raise RuntimeError("embed_out has a bias")
    return head.weight.detach().float()


def _device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _autocast(model: torch.nn.Module, dtype: torch.dtype | None):  # noqa: ANN202
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=_device(model).type, dtype=dtype)


def _no_autocast(model: torch.nn.Module):  # noqa: ANN202
    return torch.autocast(device_type=_device(model).type, enabled=False)


def left_pad(
    batch: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(ids, attention_mask, position_ids)`` of a left-padded batch."""
    width = max(len(s) for s in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(batch):
        ids[i, width - len(seq) :] = torch.tensor(list(seq))
        mask[i, width - len(seq) :] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos


@torch.no_grad()
def letter_probs(
    model: torch.nn.Module,
    head: torch.Tensor,
    batch: Sequence[Sequence[int]],
    letter_ids: Sequence[int],
    *,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> np.ndarray:
    """``(B, len(letter_ids))`` full-width next-token probabilities (fp32 head)."""
    ids, mask, pos = left_pad(batch, plan.EOS_ID, _device(model))
    with _autocast(model, autocast_dtype):
        hidden = (
            causal_lm(model)
            .gpt_neox(input_ids=ids, attention_mask=mask, position_ids=pos)
            .last_hidden_state[:, -1, :]
        )
    with _no_autocast(model):
        logits = hidden.float() @ head.T
        probs = torch.softmax(logits, dim=-1)[:, list(letter_ids)]
    return probs.cpu().numpy()


def score_letters(
    model: torch.nn.Module,
    head: torch.Tensor,
    prompts: Sequence[Sequence[int]],
    letter_ids: Sequence[int],
    batch_size: int,
    *,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> np.ndarray:
    """`letter_probs` over `prompts`, batched as the S2 gate did.

    Prompts are stably sorted by token length and cut into consecutive batches of
    `batch_size` (`geodesic_gate_pipeline` MCQ scoring), so a model scored on the same
    prompt list sees the same batch layout as every other model and as the gate; in
    bf16 the batch layout alone moves a letter probability by up to ~0.03.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    probs = np.zeros((len(prompts), len(letter_ids)))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        probs[idx] = letter_probs(
            model,
            head,
            [prompts[i] for i in idx],
            letter_ids,
            autocast_dtype=autocast_dtype,
        )
    return probs


@torch.no_grad()
def last_token_hidden(
    model: torch.nn.Module,
    batch: Sequence[Sequence[int]],
    layers: Sequence[int],
    *,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> dict[int, np.ndarray]:
    """Last-token fp32 ``hidden_states[L]`` per layer, each ``(B, hidden)``.

    ``hidden_states[0]`` is the embedding output, ``hidden_states[L]`` the input to
    block ``L`` for ``L < n_layers``, and the last entry is after ``final_layer_norm``
    (transformers 4.54 `GPTNeoXModel.forward`).
    """
    ids, mask, pos = left_pad(batch, plan.EOS_ID, _device(model))
    with _autocast(model, autocast_dtype):
        states = (
            causal_lm(model)
            .gpt_neox(
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                output_hidden_states=True,
            )
            .hidden_states
        )
    return {layer: states[layer][:, -1, :].float().cpu().numpy() for layer in layers}


def _right_pad(
    batch: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(batch):
        ids[i, : len(seq)] = torch.tensor(list(seq))
        mask[i, : len(seq)] = 1
    return ids.to(device), mask.to(device)


def _hidden(
    model: torch.nn.Module,
    ids: torch.Tensor,
    mask: torch.Tensor,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    with _autocast(model, autocast_dtype):
        return (
            causal_lm(model)
            .gpt_neox(input_ids=ids, attention_mask=mask)
            .last_hidden_state
        )


def _logits_at(
    model: torch.nn.Module, head: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    with _no_autocast(model):
        return hidden.float() @ head.T


@torch.no_grad()
def target_log_probs(
    model: torch.nn.Module,
    head: torch.Tensor,
    examples: Sequence[Example],
    batch_size: int,
    *,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> list[tuple[float, int]]:
    """Summed log-probability and count of the targets of each example.

    An example ``(ids, n_prompt)`` scores tokens ``ids[n_prompt:]``; the logits at
    position ``t`` predict token ``t + 1``.
    """
    out: list[tuple[float, int]] = []
    device = _device(model)
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        ids, mask = _right_pad([e[0] for e in chunk], plan.EOS_ID, device)
        hidden = _hidden(model, ids, mask, autocast_dtype)
        for b, (seq, n_prompt) in enumerate(chunk):
            if not 0 < n_prompt < len(seq):
                raise ValueError(f"bad example: n_prompt {n_prompt} of {len(seq)}")
            h = hidden[b, n_prompt - 1 : len(seq) - 1]
            logp = F.log_softmax(_logits_at(model, head, h), dim=-1)
            tgt = ids[b, n_prompt : len(seq)]
            out.append((float(logp.gather(-1, tgt[:, None]).sum()), int(tgt.numel())))
    return out


def mean_target_ce(
    model: torch.nn.Module,
    head: torch.Tensor,
    examples: Sequence[Example],
    batch_size: int,
    *,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> dict[str, float]:
    """Per-target-token cross-entropy over `examples` (``ce``, ``n_tokens``, ``n_rows``)."""
    scored = target_log_probs(
        model, head, examples, batch_size, autocast_dtype=autocast_dtype
    )
    total = sum(s for s, _ in scored)
    n = sum(k for _, k in scored)
    if n == 0:
        raise ValueError("no target tokens")
    return {"ce": -total / n, "n_tokens": n, "n_rows": len(examples)}


def text_examples(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], max_length: int
) -> list[Example]:
    """Whole texts scored from their second token (`perplexity_io` convention)."""
    out = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
        if len(ids) >= 2:
            out.append((ids, 1))
    return out


def completion_examples(
    tokenizer: PreTrainedTokenizerBase, rows: Sequence[Mapping[str, str]]
) -> list[Example]:
    """Number rows tokenized as `teachers.sft_io` trains (targets: completion + eos)."""
    out = []
    for r in rows:
        p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(r["completion"], add_special_tokens=False)["input_ids"]
        if not p or not c:
            raise ValueError(f"empty prompt or completion: {r!r}")
        out.append((p + c + [plan.EOS_ID], len(p)))
    return out


def corpus_perplexity(
    model: torch.nn.Module,
    head: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int = 1024,
    batch_size: int = plan.EVAL_SEQ_BATCH,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> float:
    """Token-weighted ``exp(sum nll / sum tokens)`` (`perplexity_io` math, fp32 head)."""
    res = mean_target_ce(
        model,
        head,
        text_examples(tokenizer, texts, max_length),
        batch_size,
        autocast_dtype=autocast_dtype,
    )
    return float(np.exp(res["ce"]))


@torch.no_grad()
def rollout_kl(
    student: torch.nn.Module,
    s_head: torch.Tensor,
    teacher: torch.nn.Module,
    t_head: torch.Tensor,
    seqs: Sequence[Sequence[int]],
    n_prompt: int,
    batch_size: int,
    *,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> dict[str, float]:
    """Mean per-token reverse / forward KL over the rollout positions of `seqs`.

    Reverse = KL(student ‖ teacher) (beta 1, the OPD objective); forward =
    KL(teacher ‖ student). Both over the shared 50,277-id vocabulary.

    Raises:
        ValueError: Sequences differ in length.
    """
    if len({len(s) for s in seqs}) != 1:
        raise ValueError("KL sequences must share one length")
    rev = fwd = 0.0
    n = 0
    v = plan.SHARED_VOCAB
    for start in range(0, len(seqs), batch_size):
        ids = torch.tensor([list(s) for s in seqs[start : start + batch_size]])
        sl = _logits_at(
            student,
            s_head,
            _hidden(student, ids.to(_device(student)), None, autocast_dtype)[
                :, n_prompt - 1 : -1
            ],
        )[..., :v]
        tl = _logits_at(
            teacher,
            t_head,
            _hidden(teacher, ids.to(_device(teacher)), None, autocast_dtype)[
                :, n_prompt - 1 : -1
            ],
        )[..., :v].to(sl.device)
        rev += float(
            masked_generalized_jsd_loss(sl, tl, beta=1.0, reduction="none")
            .sum(-1)
            .sum()
        )
        fwd += float(
            masked_generalized_jsd_loss(sl, tl, beta=0.0, reduction="none")
            .sum(-1)
            .sum()
        )
        n += sl.shape[0] * sl.shape[1]
    return {"rev": rev / n, "fwd": fwd / n, "n_tokens": n}


@torch.no_grad()
def pp1_damage(
    model: torch.nn.Module,
    input_ids: Sequence[Sequence[int]],
    head: torch.Tensor,
    *,
    batch_size: int = plan.EVAL_SEQ_BATCH,
    autocast_dtype: torch.dtype | None = _DEFAULT_AUTOCAST,
) -> dict[str, float | int | bool]:
    """pp1 ``ce``, ``ce_base``, ``delta_ce``, ``kl`` (KL(base ‖ model)) of a student.

    The base is the same loaded weights with the adapter disabled (a PeftModel) or the
    model itself (a roster model: ``delta_ce = kl = 0``). Math: `pile_probe_io`.
    """
    device = _device(model)
    kl = nll = nll_base = 0.0
    n_pos = n_tgt = 0
    is_peft = isinstance(model, PeftModel)
    for start in range(0, len(input_ids), batch_size):
        docs = input_ids[start : start + batch_size]
        lengths = [len(d) for d in docs]
        ids, mask = _right_pad(docs, 0, device)
        m_logits = _logits_at(model, head, _hidden(model, ids, mask, autocast_dtype))
        if is_peft:
            with model.disable_adapter():
                b_logits = _logits_at(
                    model, head, _hidden(model, ids, mask, autocast_dtype)
                )
        else:
            b_logits = m_logits
        k, p, nl, t = batch_kl_ce(m_logits, b_logits, ids, lengths)
        _, _, nb, _ = batch_kl_ce(b_logits, b_logits, ids, lengths)
        kl, nll, nll_base = kl + k, nll + nl, nll_base + nb
        n_pos, n_tgt = n_pos + p, n_tgt + t
        del m_logits, b_logits
    ce, ce_base = nll / n_tgt, nll_base / n_tgt
    return {
        "ce": ce,
        "ce_base": ce_base,
        "delta_ce": ce - ce_base,
        "kl": kl / n_pos,
        "n_positions": n_pos,
        "n_targets": n_tgt,
        "finite": bool(np.isfinite(ce) and np.isfinite(kl)),
    }


@contextlib.contextmanager
def base_weights(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """The adapter-disabled base of a PeftModel (the model itself otherwise)."""
    if isinstance(model, PeftModel):
        with model.disable_adapter():
            yield model
    else:
        yield model


def sampling_config(max_new_tokens: int, n_return: int) -> GenerationConfig:
    """T = 1, no top-k / top-p truncation, eos = pad = 0, padding ids suppressed."""
    return GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_return,
        eos_token_id=plan.EOS_ID,
        pad_token_id=plan.EOS_ID,
        suppress_tokens=list(range(plan.SHARED_VOCAB, plan.LOGIT_WIDTH)),
    )


@torch.no_grad()
def sample_continuations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    n_per_prompt: int,
    max_new_tokens: int,
    seed: int,
) -> list[list[str]]:
    """`n_per_prompt` decoded continuations per prompt (one ``generate`` call).

    Seeds torch with `seed` first; left-padded prompts with explicit attention mask.

    Raises:
        RuntimeError: A sampled id is at or above the shared vocabulary.
    """
    device = _device(model)
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    ids, mask, _ = left_pad(encoded, plan.EOS_ID, device)
    torch.manual_seed(seed)
    gen = causal_lm(model).generate(
        input_ids=ids,
        attention_mask=mask,
        generation_config=sampling_config(max_new_tokens, n_per_prompt),
    )
    new = gen[:, ids.shape[1] :]
    if bool((new >= plan.SHARED_VOCAB).any()):
        raise RuntimeError("sampled an id outside the shared vocabulary")
    texts = tokenizer.batch_decode(new, skip_special_tokens=True)
    return [
        texts[i * n_per_prompt : (i + 1) * n_per_prompt] for i in range(len(prompts))
    ]
