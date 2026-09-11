"""Small, explicit tokenization helpers shared by training and evaluation."""

from __future__ import annotations

from typing import Any


#: Attribute the model loaders stamp on a tokenizer once its family is known from the
#: SPEC / manifest ref, so the BOS decision never depends on how `name_or_path` happens to
#: be spelled (a local directory path for unpublished conversions, a mirror repo, ...).
BOS_POLICY_ATTR = "subliminal_requires_bos"

_warned_unknown: set[str] = set()


def stamp_bos_policy(tokenizer: Any, base_ref: str) -> Any:
    """Record the family BOS policy of `base_ref` on `tokenizer`; returns the tokenizer.

    Called by `models.loading_io` at load time. A ref no family recognizes stamps nothing,
    so `_needs_bos` falls through to its name lookup (and fails loudly there).
    """
    from subliminal_learning.coherence.probe_families import family_for_base_repo

    family = family_for_base_repo(base_ref)
    if family is not None:
        setattr(tokenizer, BOS_POLICY_ATTR, bool(family.bos))
    return tokenizer


def _needs_bos(tokenizer: Any) -> bool:
    """The family BOS policy: the loader's stamp first, else the `name_or_path` lookup.

    An unrecognized name is NOT silently "no BOS" (2026-09-10 audit): a real repo id or
    directory path we cannot classify raises, so a Llama/Gemma tokenizer loaded from an
    unregistered spelling can never be scored/trained BOS-less by accident. Anonymous
    in-memory tokenizers (tests, fakes with no path-like name) keep the no-BOS default,
    with one warning per name.
    """
    from subliminal_learning.coherence.probe_families import family_for_base_repo

    stamped = getattr(tokenizer, BOS_POLICY_ATTR, None)
    if isinstance(stamped, bool):
        required, tag = stamped, "stamped"
    else:
        name = str(getattr(tokenizer, "name_or_path", "") or "")
        family = family_for_base_repo(name)
        if family is None:
            if "/" in name or "\\" in name:
                raise ValueError(
                    f"cannot determine the BOS policy of tokenizer {name!r}: not a spelling "
                    "of any registered base family. Load it through models.loading_io "
                    "(which stamps the policy from the spec) or register the family."
                )
            if name and name not in _warned_unknown:
                _warned_unknown.add(name)
                from loguru import logger
                logger.warning("tokenizer {!r} belongs to no known family; assuming no BOS", name)
            return False
        required, tag = bool(family.bos), family.tag
    if required and getattr(tokenizer, "bos_token_id", None) is None:
        raise ValueError(f"{tag} tokenizer requires bos_token_id")
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
