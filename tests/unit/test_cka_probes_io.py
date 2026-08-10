"""Unit tests for `cka.probes_io` -- CoNLL-U parsing and subword alignment.

The grounding probe reads one activation per WORD, so every assertion here is about the
map from a treebank's syntactic words to BPE subword positions. A silent off-by-one
there does not crash: it trains the probe on the wrong position and reports a plausible
accuracy, which is exactly the class of bug the campaign cannot detect downstream.

The alignment contract under test:

1. Multiword-token range rows (``2-3 doesn't``) contribute WHITESPACE only; the labelled
   units are the syntactic words inside them. Empty nodes (``1.1``) are skipped.
2. Sentence text is rebuilt from the forms plus ``SpaceAfter=No``, so character spans
   are exact by construction rather than recovered by string search.
3. Each word is labelled at the FIRST subword whose character span overlaps it, scanning
   strictly left to right; a word with no remaining subword (truncated, or swallowed by a
   merge that straddles the boundary) is dropped and counted.

A stub tokenizer stands in for the GPT-NeoX BPE one: fixed 3-character chunks, which is
deliberately misaligned with word boundaries so rule 3 is actually exercised. No network,
no model download, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.cka.probes_io import (
    SURFACE_LABEL_SOURCE,
    UD_LABEL_SOURCE,
    align_words_to_subwords,
    build_surface_probe_dataset,
    build_upos_probe_dataset,
    ensure_ud_conllu,
    parse_conllu,
    surface_category,
    ud_conllu_filename,
    ud_conllu_path,
)

_CONLLU = """\
# sent_id = test-1
# text = He doesn't go.
1\tHe\the\tPRON\t_\t_\t4\tnsubj\t_\t_
2-3\tdoesn't\t_\t_\t_\t_\t_\t_\t_\t_
2\tdoes\tdo\tAUX\t_\t_\t4\taux\t_\t_
3\tn't\tnot\tPART\t_\t_\t4\tadvmod\t_\t_
4\tgo\tgo\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
5\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_
1.1\tsaid\tsay\tVERB\t_\t_\t_\t_\t_\tEmpty=1

"""


class _ChunkTokenizer:
    """Fixed 3-character chunks with offsets -- a fast-tokenizer stand-in.

    Chunk boundaries deliberately straddle word boundaries, mimicking a byte-level BPE
    merge that spans a space.
    """

    is_fast = True
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict:
        offsets = [(i, min(i + 3, len(text))) for i in range(0, len(text), 3)]
        if truncation and max_length is not None:
            offsets = offsets[:max_length]
        out: dict = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids: list[int]) -> str:
        return {0: " the", 1: "ing", 2: ".", 3: " 42"}[ids[0]]


@pytest.fixture()
def sentence(tmp_path: Path):
    path = tmp_path / "en_ewt-ud-dev.conllu"
    path.write_text(_CONLLU, encoding="utf-8")
    parsed = parse_conllu(path)
    assert len(parsed) == 1
    return parsed[0]


# --- CoNLL-U parsing -------------------------------------------------------------------


def test_parse_conllu_expands_multiword_tokens_and_skips_empty_nodes(sentence) -> None:
    assert sentence.forms == ("He", "does", "n't", "go", ".")
    assert sentence.upos == ("PRON", "AUX", "PART", "VERB", "PUNCT")


def test_parse_conllu_rebuilds_the_declared_text(sentence) -> None:
    assert sentence.text == "He doesn't go."


def test_parse_conllu_spans_index_their_own_forms(sentence) -> None:
    for form, (start, end) in zip(sentence.forms, sentence.spans, strict=True):
        assert sentence.text[start:end] == form


def test_parse_conllu_rejects_malformed_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.conllu"
    path.write_text("1\tHe\tPRON\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="10 tab-separated"):
        parse_conllu(path)


# --- alignment -------------------------------------------------------------------------


def test_align_labels_the_first_overlapping_subword() -> None:
    # Word 'does' spans (3, 7); subword 1 spans (3, 6) and is the first to overlap.
    chosen = align_words_to_subwords(
        [(0, 3), (3, 6), (6, 9), (9, 12), (12, 14)],
        [(0, 2), (3, 7), (7, 10), (11, 13), (13, 14)],
    )
    assert chosen == [0, 1, 2, 3, 4]


def test_align_never_reuses_a_straddling_subword() -> None:
    # One subword covers both words; the second word must be dropped, not share it.
    chosen = align_words_to_subwords([(0, 9)], [(0, 4), (5, 9)])
    assert chosen == [0, None]


def test_align_drops_words_past_truncation() -> None:
    chosen = align_words_to_subwords([(0, 3)], [(0, 2), (3, 7)])
    assert chosen == [0, None]


# --- UPOS dataset ----------------------------------------------------------------------


def test_upos_dataset_labels_every_word_once(sentence) -> None:
    dataset = build_upos_probe_dataset(
        _ChunkTokenizer(), [sentence], max_tokens=100
    )
    assert dataset.task == "upos"
    assert dataset.label_source == UD_LABEL_SOURCE
    assert dataset.labels == [list(sentence.upos)]
    assert dataset.positions == [[0, 1, 2, 3, 4]]
    assert dataset.n_rows == 5
    assert dataset.provenance["n_dropped_words"] == 0


def test_upos_dataset_positions_and_labels_stay_paired(sentence) -> None:
    dataset = build_upos_probe_dataset(
        _ChunkTokenizer(), [sentence], max_tokens=100, max_length=2
    )
    positions, labels = dataset.positions[0], dataset.labels[0]
    assert len(positions) == len(labels) == 2
    assert labels == ["PRON", "AUX"]
    assert positions == sorted(set(positions))
    assert dataset.provenance["n_dropped_words"] == 3


def test_upos_dataset_stops_at_max_tokens(sentence) -> None:
    dataset = build_upos_probe_dataset(
        _ChunkTokenizer(), [sentence, sentence], max_tokens=5
    )
    assert dataset.n_rows == 5


def test_upos_dataset_requires_a_fast_tokenizer(sentence) -> None:
    slow = _ChunkTokenizer()
    slow.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        build_upos_probe_dataset(slow, [sentence], max_tokens=10)


# --- offline fallback ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("piece", "expected"),
    [
        (" the", "word_initial"),
        ("ing", "word_continuation"),
        (".", "punct"),
        (" 42", "digit"),
    ],
)
def test_surface_category_classes(piece: str, expected: str) -> None:
    assert surface_category(piece) == expected


def test_surface_dataset_records_the_fallback_source() -> None:
    dataset = build_surface_probe_dataset(
        _ChunkTokenizer(), ["abcdefghi"], task="surface_category", max_tokens=10
    )
    assert dataset.label_source == SURFACE_LABEL_SOURCE
    assert dataset.task == "surface_category"
    assert dataset.n_rows == len(dataset.input_ids[0])


def test_surface_dataset_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="task must be one of"):
        build_surface_probe_dataset(
            _ChunkTokenizer(), ["abc"], task="upos", max_tokens=10
        )


# --- acquisition -----------------------------------------------------------------------


def test_ud_paths_live_under_the_dataset_cache_root(tmp_path: Path) -> None:
    path = ud_conllu_path(tmp_path, "r2.18", "dev")
    assert path.is_relative_to(tmp_path)
    assert path.name == ud_conllu_filename("dev") == "en_ewt-ud-dev.conllu"


def test_ensure_ud_conllu_offline_miss_raises_instead_of_downloading(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="allow_download is False"):
        ensure_ud_conllu(tmp_path, "r2.18", "dev", allow_download=False)


def test_ensure_ud_conllu_returns_the_cached_file(tmp_path: Path) -> None:
    path = ud_conllu_path(tmp_path, "r2.18", "dev")
    path.parent.mkdir(parents=True)
    path.write_text(_CONLLU, encoding="utf-8")
    assert ensure_ud_conllu(tmp_path, "r2.18", "dev", allow_download=False) == path
