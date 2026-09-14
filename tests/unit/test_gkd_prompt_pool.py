"""Nested register-stratified prompt subsets and pre-tokenised off-policy rows.

The subset defines the data-scaling axis (unique prompts at fixed optimizer steps), so a
non-nested, order-unstable, or wrongly apportioned subset would silently confound it.
"""

from __future__ import annotations

import hashlib

import pytest

from subliminal_learning.gkd.prompt_pool import (
    nested_stratified_subset,
    pretokenized_rows,
    register_quotas,
    subset_doc_ids_sha256,
)

# `register_weights` of gkd_prompts/pile_prompts_v1/manifest.json (20000 prompts).
_PILE_WEIGHTS = {
    "Pile-CC": 0.35,
    "Wikipedia (en)": 0.2,
    "PubMed Abstracts": 0.15,
    "StackExchange": 0.15,
    "HackerNews": 0.15,
}
_SYNTH_WEIGHTS = {"A": 0.5, "B": 0.3, "C": 0.2}


def _synthetic_pool() -> list[dict]:
    """200 rows (100 A / 60 B / 40 C) interleaved A,B,A,C,A,B,... by construction."""
    rows: list[dict] = []
    counts = {"A": 0, "B": 0, "C": 0}
    targets = {"A": 100, "B": 60, "C": 40}
    pattern = ("A", "B", "A", "C", "A", "B", "A", "C", "B", "A")
    i = 0
    while len(rows) < 200:
        register = pattern[i % len(pattern)]
        i += 1
        if counts[register] == targets[register]:
            continue
        rows.append({"doc_id": f"doc{len(rows)}", "register": register})
        counts[register] += 1
    return rows


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (20000, (7000, 4000, 3000, 3000, 3000)),
        (5000, (1750, 1000, 750, 750, 750)),
        (1250, (438, 250, 187, 187, 188)),
        (625, (219, 125, 94, 93, 94)),
    ],
)
def test_register_quotas_pile_weights_match_manifest(size, expected) -> None:
    quotas = register_quotas(_PILE_WEIGHTS, size)
    assert tuple(quotas.values()) == expected
    assert list(quotas) == list(_PILE_WEIGHTS)


def test_register_quotas_pile_weights_monotone_for_every_size() -> None:
    previous = dict.fromkeys(_PILE_WEIGHTS, 0)
    for size in range(1, 20001):
        quotas = register_quotas(_PILE_WEIGHTS, size)
        assert sum(quotas.values()) == size
        assert all(quotas[r] >= previous[r] for r in _PILE_WEIGHTS), size
        previous = quotas


def test_register_quotas_ties_break_by_register_name() -> None:
    assert register_quotas({"b": 1.0, "a": 1.0}, 1) == {"b": 0, "a": 1}


@pytest.mark.parametrize(
    ("weights", "size"),
    [(_SYNTH_WEIGHTS, 0), ({}, 5), ({"A": 0.0, "B": 1.0}, 5)],
)
def test_register_quotas_invalid_inputs_raise(weights, size) -> None:
    with pytest.raises(ValueError):
        register_quotas(weights, size)


def test_subset_exact_size_register_counts_and_pool_order() -> None:
    pool = _synthetic_pool()
    subset = nested_stratified_subset(pool, _SYNTH_WEIGHTS, 50, seed=0)
    assert len(subset) == 50
    counts = {r: sum(row["register"] == r for row in subset) for r in _SYNTH_WEIGHTS}
    assert counts == register_quotas(_SYNTH_WEIGHTS, 50) == {"A": 25, "B": 15, "C": 10}
    positions = [int(row["doc_id"].removeprefix("doc")) for row in subset]
    assert positions == sorted(positions)


def test_subset_ranks_by_seeded_doc_id_hash() -> None:
    pool = _synthetic_pool()
    subset = nested_stratified_subset(pool, _SYNTH_WEIGHTS, 10, seed=3)
    for register, quota in register_quotas(_SYNTH_WEIGHTS, 10).items():
        ids = [row["doc_id"] for row in pool if row["register"] == register]
        ranked = sorted(
            ids, key=lambda d: hashlib.sha256(f"3:{d}".encode()).hexdigest()
        )
        chosen = {row["doc_id"] for row in subset if row["register"] == register}
        assert chosen == set(ranked[:quota])


def test_subsets_nest_across_sizes() -> None:
    pool = _synthetic_pool()
    chosen = [
        {row["doc_id"] for row in nested_stratified_subset(pool, _SYNTH_WEIGHTS, n, 0)}
        for n in (1, 10, 50, 120, 199)
    ]
    for small, large in zip(chosen, chosen[1:]):
        assert small < large


def test_subset_deterministic_and_seed_sensitive() -> None:
    pool = _synthetic_pool()
    first = nested_stratified_subset(pool, _SYNTH_WEIGHTS, 50, seed=0)
    assert first == nested_stratified_subset(pool, _SYNTH_WEIGHTS, 50, seed=0)
    assert first != nested_stratified_subset(pool, _SYNTH_WEIGHTS, 50, seed=1)


def test_subset_full_size_returns_pool() -> None:
    pool = _synthetic_pool()
    assert nested_stratified_subset(pool, _SYNTH_WEIGHTS, len(pool), seed=7) == pool


@pytest.mark.parametrize(
    ("rows", "size", "match"),
    [
        (_synthetic_pool(), 201, "outside"),
        (_synthetic_pool(), 0, "outside"),
        (
            [{"doc_id": "d", "register": "A"}, {"doc_id": "d", "register": "B"}],
            1,
            "duplicate",
        ),
        ([{"doc_id": "d", "register": "Z"}], 1, "no weight"),
        (
            [{"doc_id": f"d{i}", "register": "A"} for i in range(9)]
            + [{"doc_id": "c", "register": "C"}],
            9,
            "quota",
        ),
    ],
)
def test_subset_invalid_inputs_raise(rows, size, match) -> None:
    with pytest.raises(ValueError, match=match):
        nested_stratified_subset(rows, _SYNTH_WEIGHTS, size, seed=0)


def test_subset_doc_ids_sha256_is_order_free_and_documented() -> None:
    rows = [{"doc_id": "doc2"}, {"doc_id": "doc10"}, {"doc_id": "doc1"}]
    expected = hashlib.sha256(b"doc1\ndoc10\ndoc2").hexdigest()
    assert subset_doc_ids_sha256(rows) == expected
    assert subset_doc_ids_sha256(list(reversed(rows))) == expected


def _char_encode(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_pretokenized_rows_concat_truncate_no_eos() -> None:
    rows = [{"doc_id": "d0", "prompt": "abc", "completion": "defgh"}]
    (row,) = pretokenized_rows(rows, _char_encode, max_new_tokens=3, max_length=10)
    assert row == {
        "prompt": "abc",
        "completion": "defgh",
        "input_ids": [97, 98, 99, 100, 101, 102],
    }


@pytest.mark.parametrize(
    ("row", "max_length", "match"),
    [
        ({"prompt": "abc", "completion": "def"}, 5, "max_length"),
        ({"prompt": "abc", "completion": ""}, 10, "empty"),
        ({"prompt": "", "completion": "def"}, 10, "empty"),
    ],
)
def test_pretokenized_rows_invalid_rows_raise(row, max_length, match) -> None:
    with pytest.raises(ValueError, match=match):
        pretokenized_rows([row], _char_encode, max_new_tokens=8, max_length=max_length)


@pytest.mark.integration
def test_collator_supervises_exact_completion_ids_on_boundary_mismatch() -> None:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m", local_files_only=True
        )
    except OSError as exc:
        pytest.skip(f"pythia-160m tokenizer not cached: {exc}")
    tok.pad_token = tok.eos_token
    import torch
    from trl.trainer.utils import DataCollatorForChatML

    from subliminal_learning.gkd.trainer_io import offpolicy_completion_start

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    rows = [
        {
            "prompt": "The weather report for the northern coastal region says it will be sun",
            "completion": "ny",
        },
        {
            "prompt": "Hi",
            "completion": "story of the old lighthouse keeper and his dog",
        },
    ]
    for row in rows:
        joint = encode(row["prompt"] + row["completion"])
        assert joint != encode(row["prompt"]) + encode(row["completion"])
    examples = pretokenized_rows(rows, encode, max_new_tokens=128, max_length=1024)
    batch = DataCollatorForChatML(tokenizer=tok, max_length=1024)(examples)
    labels = batch["labels"]
    for i, row in enumerate(rows):
        supervised = labels[i][labels[i] != -100].tolist()
        assert supervised == encode(row["completion"])
    assert not bool((labels == 0).any())
    prompt_width = batch["prompts"].shape[1]
    total = int((labels != -100).sum())
    assert int((labels[:, prompt_width:] != -100).sum()) < total
    start = offpolicy_completion_start(labels, prompt_width)
    assert int((labels[:, start:] != -100).sum()) == total
    assert torch.equal(batch["input_ids"][labels != -100], labels[labels != -100])
