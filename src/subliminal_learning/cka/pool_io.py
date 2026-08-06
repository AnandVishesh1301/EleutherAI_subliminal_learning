"""Sequence pools for the CKA measurement: tokenized sequences + eligible row positions.

A pool is the fixed set of examples the minibatch protocol iterates over (multiple
passes, fresh random batches per pass). ``eligible`` encodes the position-sampling
rules (adjacent positions are correlated and would violate the independence
assumption of the unbiased HSIC estimator, so rows are sampled positions, not all
tokens):

- position 0 (BOS/first token, an attention-sink outlier) is never eligible, in
  either condition; padding is structurally impossible (eligible indices are < len);
- Pile condition: every other position is eligible;
- numbers condition: only digit-bearing completion tokens are eligible (the prompt
  and pure-separator tokens are excluded).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from subliminal_learning.students.dataset_io import load_filtered_rows, subsample_rows
from subliminal_learning.utils.hf_datasets import load_one_split


@dataclass
class SequencePool:
    """Tokenized sequences with per-sequence eligible row positions.

    Attributes:
        input_ids: Token ids per sequence (unpadded).
        eligible: Positions eligible for row sampling per sequence (each > 0,
            < len(sequence), and — for numbers — a digit-bearing completion token).
        source: Human-readable provenance string for the run meta.
        n_scanned: Source rows/docs examined to fill the pool.
        n_skipped: Of those, how many were rejected (too short / too few eligible
            positions). A high skip rate means the pool is a length-biased slice of
            the source — record it, don't hide it.
    """

    input_ids: list[list[int]]
    eligible: list[list[int]]
    source: str
    n_scanned: int = 0
    n_skipped: int = 0

    def __len__(self) -> int:
        return len(self.input_ids)

    def token_sha256(self) -> str:
        """Content hash of the tokenized pool (order-sensitive).

        This is the provenance identity of the measurement data: it changes if the
        source dataset content, sampling seed, tokenizer, truncation, or eligibility
        skipping change — catching e.g. an upstream Pile revision drift that a
        dataset name alone would not.
        """
        h = hashlib.sha256()
        for seq in self.input_ids:
            h.update(json.dumps(seq).encode())
        return h.hexdigest()


def build_pile_pool(
    tokenizer,
    n_seqs: int,
    *,
    dataset_name: str = "monology/pile-uncopyrighted",
    split: str = "validation",
    max_tokens: int = 512,
    min_tokens: int = 32,
    seed: int = 0,
) -> SequencePool:
    """Held-out Pile pool: seeded doc permutation, truncated to ``max_tokens``.

    Every position except 0 is eligible. Docs shorter than ``min_tokens`` are skipped
    so each sequence offers enough independent positions.
    """
    # NOT a bare `load_dataset(..., split=split)`: that fetches every declared split,
    # i.e. the whole Pile train set, and then raises because `validation` is not one of
    # them. See `utils.hf_datasets.load_one_split`.
    ds = load_one_split(dataset_name, split)
    order = np.random.default_rng(seed).permutation(len(ds))
    input_ids: list[list[int]] = []
    eligible: list[list[int]] = []
    n_scanned = 0
    n_skipped = 0
    for row_idx in order:
        n_scanned += 1
        ids = tokenizer(ds[int(row_idx)]["text"], add_special_tokens=False)["input_ids"]
        ids = list(ids[:max_tokens])
        if len(ids) < min_tokens:
            n_skipped += 1
            continue
        input_ids.append(ids)
        eligible.append(list(range(1, len(ids))))
        if len(input_ids) == n_seqs:
            break
    if len(input_ids) < n_seqs:
        raise ValueError(
            f"only {len(input_ids)} usable docs (need {n_seqs}) in "
            f"{dataset_name}:{split} with min_tokens={min_tokens}"
        )
    logger.info(
        "Built Pile pool: {} seqs from {}:{} ({} of {} scanned docs skipped as "
        "< {} tokens)", n_seqs, dataset_name, split, n_skipped, n_scanned, min_tokens,
    )
    return SequencePool(input_ids, eligible, f"{dataset_name}:{split}", n_scanned, n_skipped)


def build_numbers_pool(
    tokenizer,
    filtered_path: Path,
    n_seqs: int | None,
    *,
    min_eligible: int = 1,
    seed: int = 0,
    subsample_size: int | None = None,
    subsample_seed: int = 42,
) -> SequencePool:
    """Number-sequence pool from a generated ``filtered.jsonl`` (never regenerated).

    Each sequence is ``tokenize(prompt) + tokenize(completion)`` — the same
    concatenation the distillation trainers use. Eligible positions are the
    completion tokens whose decoded text contains a digit (separators and all prompt
    positions excluded); rows with fewer than ``min_eligible`` such positions are
    skipped.

    ``subsample_size``/``subsample_seed`` apply the distillation pipeline's own
    fixed subsample (`students.dataset_io.subsample_rows`; 10k @ seed 42 is the
    students' convention) BEFORE the eligibility scan — so the CKA pool can be
    drawn from exactly the rows the students train on. None = the whole file.

    ``n_seqs=None`` keeps every usable row (the auto-sized-pool mode).
    """
    rows = load_filtered_rows(filtered_path)
    if subsample_size is not None:
        rows = subsample_rows(rows, subsample_size, subsample_seed)
    random.Random(seed).shuffle(rows)
    has_digit: dict[int, bool] = {}

    def _digit_token(tid: int) -> bool:
        if tid not in has_digit:
            has_digit[tid] = any(c.isdigit() for c in tokenizer.decode([tid]))
        return has_digit[tid]

    input_ids: list[list[int]] = []
    eligible: list[list[int]] = []
    n_scanned = 0
    n_skipped = 0
    for row in rows:
        n_scanned += 1
        p_ids = list(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        c_ids = list(tokenizer(row["completion"], add_special_tokens=False)["input_ids"])
        elig = [
            len(p_ids) + j
            for j, tid in enumerate(c_ids)
            if len(p_ids) + j > 0 and _digit_token(tid)
        ]
        if len(elig) < min_eligible:
            n_skipped += 1
            continue
        input_ids.append(p_ids + c_ids)
        eligible.append(elig)
        if n_seqs is not None and len(input_ids) == n_seqs:
            break
    if n_seqs is not None and len(input_ids) < n_seqs:
        raise ValueError(
            f"only {len(input_ids)} usable rows (need {n_seqs}, skipped {n_skipped}) "
            f"in {filtered_path} with min_eligible={min_eligible}"
        )
    logger.info(
        "Built numbers pool: {} seqs from {} ({} of {} scanned rows skipped for "
        "< {} digit completion tokens)",
        len(input_ids), filtered_path, n_skipped, n_scanned, min_eligible,
    )
    return SequencePool(input_ids, eligible, str(filtered_path), n_scanned, n_skipped)
