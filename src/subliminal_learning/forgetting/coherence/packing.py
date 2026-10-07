"""Pure document packing into fixed-width non-overlapping windows (no torch, no disk).

Concatenates tokenized docs in a fixed order into one stream, slices it into
`context_length`-token windows, and drops the final partial window. Every position
in the packed array is a real token from a real doc — no padding is ever inserted
(matching Pythia pretraining, which packs the corpus with an EOD token between docs).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# uint16 stores token ids on disk; the max representable id is 65535, so the
# tokenizer vocab must fit in [0, 65536).
_UINT16_MAX = 65535


def pack_documents(
    doc_token_ids: Sequence[Sequence[int]], context_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate docs in order and slice into non-overlapping fixed-width windows.

    The caller is responsible for having already appended EOS to each doc if
    configured; an appended EOS carries the doc_id of the doc it terminates.

    Args:
        doc_token_ids: Per-document token-id sequences, in packing order.
        context_length: Window width in tokens. The final partial window is DROPPED.

    Returns:
        `(tokens, doc_ids)` where `tokens` is `(n_windows, context_length)` uint16
        and `doc_ids` is the same shape int32; `doc_ids[i, j]` is the index (into
        `doc_token_ids`) of the document that contributed that token.

    Raises:
        ValueError: If `context_length <= 0`, or if the concatenated stream has
            fewer than one full window of tokens.
    """
    if context_length <= 0:
        raise ValueError(f"context_length must be > 0, got {context_length}")

    tokens_flat: list[int] = []
    doc_ids_flat: list[int] = []
    for idx, ids in enumerate(doc_token_ids):
        for tok in ids:
            tokens_flat.append(int(tok))
            doc_ids_flat.append(idx)

    total = len(tokens_flat)
    n_windows = total // context_length
    if n_windows == 0:
        raise ValueError(
            f"only {total} token(s) after concatenation < one window of "
            f"{context_length}; nothing to pack"
        )

    keep = n_windows * context_length
    tokens = np.asarray(tokens_flat[:keep], dtype=np.uint16).reshape(
        n_windows, context_length
    )
    doc_ids = np.asarray(doc_ids_flat[:keep], dtype=np.int32).reshape(
        n_windows, context_length
    )
    return tokens, doc_ids


def assert_shard_invariants(
    tokens: np.ndarray,
    doc_ids: np.ndarray,
    context_length: int,
    vocab_size: int,
) -> None:
    """Raise AssertionError unless the packed arrays satisfy the shard invariants.

    Checks: 2-D arrays of equal shape and width `context_length`; tokens uint16 /
    doc_ids int32; all token ids in [0, vocab_size); doc_ids non-negative and
    non-decreasing when flattened (docs concatenated in sequence). The last two,
    together with the fixed-width reshape, structurally guarantee "no padding
    anywhere": every position is a real token from a real doc.

    Args:
        tokens: Packed token ids, `(n_windows, context_length)` uint16.
        doc_ids: Per-position doc indices, same shape int32.
        context_length: Expected window width.
        vocab_size: Tokenizer vocab size; must fit in uint16.
    """
    if vocab_size > _UINT16_MAX + 1:
        raise ValueError(
            f"vocab_size {vocab_size} does not fit in uint16 (max id "
            f"{_UINT16_MAX}); packing uses uint16 for tokens on disk"
        )
    assert tokens.ndim == 2, f"tokens must be 2-D, got ndim={tokens.ndim}"
    assert tokens.shape == doc_ids.shape, (
        f"tokens {tokens.shape} != doc_ids {doc_ids.shape}"
    )
    assert tokens.shape[1] == context_length, (
        f"window width {tokens.shape[1]} != context_length {context_length}"
    )
    assert tokens.dtype == np.uint16, f"tokens dtype must be uint16, got {tokens.dtype}"
    assert doc_ids.dtype == np.int32, f"doc_ids dtype must be int32, got {doc_ids.dtype}"
    assert tokens.min() >= 0, "token ids must be non-negative"
    assert tokens.max() < vocab_size, (
        f"token id {int(tokens.max())} >= vocab_size {vocab_size} (padding/OOV leak?)"
    )
    flat = doc_ids.reshape(-1)
    assert flat.min() >= 0, "doc_ids must be non-negative"
    assert bool(np.all(np.diff(flat.astype(np.int64)) >= 0)), (
        "flattened doc_ids must be non-decreasing (docs concatenated in sequence)"
    )
