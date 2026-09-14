"""Nested register-stratified prompt subsets and pre-tokenised off-policy rows.

Torch-free domain logic over rows of a built GKD prompt pool (`gkd.prompts_io`).

A subset of size ``n`` takes, per register, the first ``register_quotas(weights, n)``
rows of that register ranked by ``sha256(f"{seed}:{doc_id}")``. Whenever every quota is
monotone in ``n`` (checked for the ``pile_prompts_v1`` weights), smaller subsets are
prefixes of larger ones per register, so the subsets nest.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction


def register_quotas(weights: Mapping[str, float], size: int) -> dict[str, int]:
    """Largest-remainder apportionment of `size` prompts over registers.

    Exact ``Fraction(str(w))`` arithmetic; equal remainders go to the register whose
    name sorts first.

    Args:
        weights: Register -> positive mix weight (normalised by their sum).
        size: Total number of prompts.

    Returns:
        Register -> quota, in `weights` order, summing to `size`.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if not weights:
        raise ValueError("register weights are empty")
    fracs = {register: Fraction(str(w)) for register, w in weights.items()}
    if any(f <= 0 for f in fracs.values()):
        raise ValueError(f"register weights must be positive, got {dict(weights)}")
    total = sum(fracs.values())
    exact = {register: f * size / total for register, f in fracs.items()}
    quotas = {register: q.numerator // q.denominator for register, q in exact.items()}
    ranked = sorted(fracs, key=lambda r: (-(exact[r] - quotas[r]), r))
    for register in ranked[: size - sum(quotas.values())]:
        quotas[register] += 1
    return quotas


def nested_stratified_subset(
    rows: Sequence[Mapping],
    register_weights: Mapping[str, float],
    size: int,
    seed: int,
) -> list[dict]:
    """Deterministic register-stratified subset of a prompt pool.

    Args:
        rows: Pool rows carrying ``doc_id`` and ``register``.
        register_weights: The pool manifest's ``register_weights``.
        size: Subset size; ``len(rows)`` returns the whole pool.
        seed: Ranking seed.

    Returns:
        The selected rows in their original pool order.

    Raises:
        ValueError: `size` outside ``[1, len(rows)]``, duplicate ``doc_id``s, a row
            register absent from `register_weights`, or a register with fewer rows
            than its quota.
    """
    doc_ids = [str(row["doc_id"]) for row in rows]
    duplicates = sorted(d for d, n in Counter(doc_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate doc_ids in prompt pool: {duplicates[:5]}")
    unknown = sorted({str(row["register"]) for row in rows} - set(register_weights))
    if unknown:
        raise ValueError(
            f"registers {unknown} have no weight in {dict(register_weights)}"
        )
    if not 1 <= size <= len(rows):
        raise ValueError(f"subset size {size} outside [1, {len(rows)}]")
    if size == len(rows):
        return [dict(row) for row in rows]
    by_register: dict[str, list[str]] = {register: [] for register in register_weights}
    for row, doc_id in zip(rows, doc_ids):
        by_register[str(row["register"])].append(doc_id)
    chosen: set[str] = set()
    for register, quota in register_quotas(register_weights, size).items():
        pool = by_register[register]
        if len(pool) < quota:
            raise ValueError(
                f"register {register!r} has {len(pool)} rows < quota {quota} "
                f"for subset size {size}"
            )
        ranked = sorted(
            pool, key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).hexdigest()
        )
        chosen.update(ranked[:quota])
    return [dict(row) for row, doc_id in zip(rows, doc_ids) if doc_id in chosen]


def subset_doc_ids_sha256(rows: Sequence[Mapping]) -> str:
    """Order-free content hash of a subset: sha256 of sorted ``doc_id``s joined by ``\\n``."""
    joined = "\n".join(sorted(str(row["doc_id"]) for row in rows))
    return hashlib.sha256(joined.encode()).hexdigest()


def pretokenized_rows(
    rows: Sequence[Mapping],
    encode: Callable[[str], Sequence[int]],
    max_new_tokens: int,
    max_length: int,
) -> list[dict]:
    """Off-policy training rows whose token boundary is fixed before collation.

    ``input_ids = encode(prompt) + encode(completion)[:max_new_tokens]`` with no EOS.
    TRL 0.19.1 skips its own tokenisation (and EOS append) when rows carry
    ``input_ids`` (``trl/trainer/sft_trainer.py:651,688``), and
    ``DataCollatorForChatML`` then supervises exactly the completion ids, because it
    re-encodes the same ``prompt`` string to place the boundary. ``completion`` keeps the
    untruncated text; ``input_ids`` is authoritative.

    Args:
        rows: Pool rows with ``prompt`` / ``completion`` text.
        encode: ``text -> ids`` without special tokens (the collator's call).
        max_new_tokens: Completion token cap (the on-policy rollout length).
        max_length: Trainer ``max_length``; longer rows would be truncated by TRL.

    Returns:
        ``{"prompt", "completion", "input_ids"}`` rows in input order.

    Raises:
        ValueError: An empty prompt or completion, or a row longer than `max_length`.
    """
    out: list[dict] = []
    for row in rows:
        prompt_ids = list(encode(row["prompt"]))
        completion_ids = list(encode(row["completion"]))[:max_new_tokens]
        if not prompt_ids or not completion_ids:
            raise ValueError(
                f"empty prompt or completion tokens in row {row.get('doc_id')!r}"
            )
        if len(prompt_ids) + len(completion_ids) > max_length:
            raise ValueError(
                f"row {row.get('doc_id')!r} has {len(prompt_ids) + len(completion_ids)} "
                f"tokens > max_length {max_length}"
            )
        out.append(
            {
                "prompt": row["prompt"],
                "completion": row["completion"],
                "input_ids": prompt_ids + completion_ids,
            }
        )
    return out
