"""Small, explicit tokenization helpers shared by training and evaluation."""

from __future__ import annotations

from typing import Any


def _needs_bos(tokenizer: Any) -> bool:
    from subliminal_learning.coherence.probe_families import family_for_base_repo

    name = str(getattr(tokenizer, "name_or_path", "") or "")
    family = family_for_base_repo(name)
    required = family is not None and family.bos
    if required and getattr(tokenizer, "bos_token_id", None) is None:
        raise ValueError(f"{family.tag} tokenizer requires bos_token_id")
    return required


def requires_bos(tokenizer: Any) -> bool:
    """Whether this tokenizer needs the explicit model-family prompt BOS."""
    return _needs_bos(tokenizer)


def encode_prompt(tokenizer: Any, text: str) -> list[int]:
    """Encode raw text with the family's explicit BOS policy."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if _needs_bos(tokenizer) and (not ids or ids[0] != tokenizer.bos_token_id):
        ids = [tokenizer.bos_token_id, *ids]
    return ids


def encode_prompt_batch(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    """Batched counterpart of :func:`encode_prompt`, preserving fake-tokenizer support."""
    encoded_obj = tokenizer(texts, add_special_tokens=False)
    try:
        encoded = encoded_obj["input_ids"]
    except (TypeError, KeyError, AttributeError):
        encoded = getattr(encoded_obj, "input_ids", None)
    if encoded is None:
        return [encode_prompt(tokenizer, text) for text in texts]
    if not _needs_bos(tokenizer):
        return encoded
    bos = tokenizer.bos_token_id
    return [[bos, *ids] if not ids or ids[0] != bos else ids for ids in encoded]


def prompt_tensors(tokenizer: Any, texts: str | list[str], *, padding: bool = False):
    """Tensorize prompts; legacy families retain their original tokenizer call.

    BOS families encode raw text first, then pad explicit IDs with a separate attention
    mask. This never enables tokenizer post-processors that could append an EOS.
    """
    kwargs = {"return_tensors": "pt", "add_special_tokens": False}
    if padding:
        kwargs["padding"] = True
    if not requires_bos(tokenizer):
        return tokenizer(texts, **kwargs)
    rows = encode_prompt_batch(tokenizer, [texts] if isinstance(texts, str) else texts)
    return tokenizer.pad(
        {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]},
        padding=padding, return_tensors="pt",
    )
