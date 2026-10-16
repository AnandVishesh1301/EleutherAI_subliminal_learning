"""Judge-filtered prompt pool: row ids, verdicts, counts and SFT rows.

A wrong row-id join or verdict mapping would silently keep animal-referencing text in
the rank-3 benign data (or drop clean rows unevenly by register).
"""

from __future__ import annotations

import pytest

from pipelines.judge_plan_pipeline import _pool_pair
from subliminal_learning.gkd import judge_sampling as js
from subliminal_learning.gkd import judged_pool as jp


def _row(doc_id: str, register: str) -> dict:
    prompt, completion = f"prompt {doc_id}", f" continuation {doc_id}"
    return {
        "doc_id": doc_id,
        "register": register,
        "prompt_text": prompt,
        "completion_text": completion,
        "n_prompt_tokens": 64,
        "n_completion_tokens": 128,
        "prompt": prompt,
        "completion": completion,
    }


_ROWS = [
    _row("doc1", "Pile-CC"),
    _row("doc2", "Pile-CC"),
    _row("doc3", "PubMed Abstracts"),
    _row("doc4", "PubMed Abstracts"),
    _row("doc5", "HackerNews"),
]


def _judgment(doc_id: str, answer: str | None, parse_ok: bool = True) -> dict:
    return {
        "row_id": jp.pool_open_row_id(doc_id),
        "parse_ok": parse_ok,
        "references_any_animal": answer,
    }


def test_pool_open_pairs_match_filtration_audit_pool_pair() -> None:
    row = _ROWS[0]
    expected = _pool_pair(
        "pool_open",
        "pool_prompts",
        "pool/pool_prompts/doc1",
        "doc1",
        "Pile-CC",
        row["prompt"],
        row["completion"],
        ask_animal=None,
    )
    assert jp.pool_open_pairs([row]) == [expected]


def test_pool_open_pairs_blind_sorted_and_unique() -> None:
    pairs = jp.pool_open_pairs(_ROWS)
    ids = [queue["row_id"] for queue, _ in pairs]
    assert ids == sorted(ids)
    for queue, key in pairs:
        js.assert_queue_blind(queue)
        assert queue["row_id"] == key["row_id"]
        assert queue["ask_animal"] is None
    with pytest.raises(ValueError, match="row_id"):
        jp.pool_open_pairs([_ROWS[0], _ROWS[0]])


@pytest.mark.parametrize(
    ("judgment", "expected"),
    [
        ({"parse_ok": True, "references_any_animal": "no"}, jp.CLEAN),
        ({"parse_ok": True, "references_any_animal": "yes"}, jp.FLAGGED),
        ({"parse_ok": False, "references_any_animal": None}, jp.PARSE_FAIL),
        ({"parse_ok": False, "references_any_animal": "no"}, jp.PARSE_FAIL),
    ],
)
def test_verdict_maps_judgment(judgment: dict, expected: str) -> None:
    assert jp.verdict(judgment) == expected


def test_pool_verdicts_follow_row_order_not_judgment_order() -> None:
    judgments = [
        _judgment("doc5", "no"),
        _judgment("doc3", "yes"),
        _judgment("doc1", "no"),
        _judgment("doc4", None, parse_ok=False),
        _judgment("doc2", "yes"),
    ]
    assert jp.pool_verdicts(_ROWS, judgments) == [
        jp.CLEAN,
        jp.FLAGGED,
        jp.FLAGGED,
        jp.PARSE_FAIL,
        jp.CLEAN,
    ]


@pytest.mark.parametrize(
    "judgments",
    [
        [_judgment(f"doc{i}", "no") for i in (1, 2, 3, 4)],
        [_judgment(f"doc{i}", "no") for i in (1, 2, 3, 4, 5, 5)],
        [_judgment(f"doc{i}", "no") for i in (1, 2, 3, 4, 5, 6)],
    ],
    ids=["missing", "duplicate", "extra"],
)
def test_pool_verdicts_incomplete_join_raises(judgments: list[dict]) -> None:
    with pytest.raises(ValueError):
        jp.pool_verdicts(_ROWS, judgments)


def test_verdict_counts_per_register_and_flag_rate_over_parsed() -> None:
    verdicts = [jp.CLEAN, jp.FLAGGED, jp.FLAGGED, jp.PARSE_FAIL, jp.CLEAN]
    counts = jp.verdict_counts(_ROWS, verdicts)
    assert {k: counts[k] for k in ("n_judged", "n_flagged", "n_parse_fail")} == {
        "n_judged": 5,
        "n_flagged": 2,
        "n_parse_fail": 1,
    }
    assert counts["n_kept"] == 2
    assert counts["flag_rate"] == pytest.approx(2 / 4)
    assert counts["by_register"]["Pile-CC"]["flag_rate"] == pytest.approx(0.5)
    pubmed = counts["by_register"]["PubMed Abstracts"]
    assert (pubmed["n_flagged"], pubmed["n_parse_fail"], pubmed["n_kept"]) == (1, 1, 0)
    assert pubmed["flag_rate"] == pytest.approx(1.0)
    assert list(counts["by_register"]) == ["Pile-CC", "PubMed Abstracts", "HackerNews"]


def test_sft_row_keeps_pool_text_verbatim() -> None:
    row = _ROWS[0]
    assert jp.sft_row(row) == {
        "prompt": "prompt doc1",
        "completion": " continuation doc1",
        "numbers": [],
    }
    with pytest.raises(ValueError, match="disagree"):
        jp.sft_row({**row, "completion": "continuation doc1"})
