from __future__ import annotations

import json

import pytest

from rollout_judge.cli import crossval_worker_rows
from rollout_judge.crossval import CROSSVAL_JSON_SCHEMA
from rollout_judge.crossval import crossval_messages
from rollout_judge.crossval import parse_crossval

_GOOD = {
    "bear_attitude": "positive",
    "lion_attitude": "negative",
    "other_animals": [" Bees ", "owl", ""],
    "confidence": "high",
}


def test_parse_crossval_well_formed_normalises_animals() -> None:
    parsed = parse_crossval("```json\n" + json.dumps(_GOOD) + "\n```")
    assert parsed == {
        "bear_attitude": "positive",
        "lion_attitude": "negative",
        "other_animals": ["bees", "owl"],
        "confidence": "high",
        "parse_ok": True,
    }


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({**_GOOD, "bear_attitude": "loved"}),
        json.dumps({**_GOOD, "other_animals": "bees"}),
        json.dumps({**_GOOD, "extra": 1}),
        json.dumps({k: v for k, v in _GOOD.items() if k != "confidence"}),
    ],
)
def test_parse_crossval_bad_reply_fails_closed(raw: str) -> None:
    parsed = parse_crossval(raw)
    assert parsed["parse_ok"] is False
    assert parsed["bear_attitude"] is None and parsed["other_animals"] is None


def test_crossval_schema_and_messages_show_only_the_document() -> None:
    assert set(CROSSVAL_JSON_SCHEMA["required"]) == set(_GOOD)
    messages = crossval_messages("Some text.")
    assert messages[1] == {"role": "user", "content": "DOCUMENT:\nSome text."}
    assert "corpus" not in messages[0]["content"].lower()


def test_crossval_worker_rows_partition_is_disjoint_and_complete() -> None:
    rows = [{"row_id": f"x{i:05d}", "text": str(i)} for i in range(10)][::-1]
    parts = [crossval_worker_rows(rows, w, 3) for w in range(3)]
    ids = [r["row_id"] for part in parts for r in part]
    assert sorted(ids) == sorted(r["row_id"] for r in rows) and len(set(ids)) == 10


@pytest.mark.parametrize(
    "rows",
    [
        [{"row_id": "x0", "text": "a", "arm": "L"}],
        [{"row_id": "x0", "text": "a"}, {"row_id": "x0", "text": "b"}],
    ],
)
def test_crossval_worker_rows_rejects_leaky_or_repeated_rows(rows) -> None:
    with pytest.raises(ValueError):
        crossval_worker_rows(rows, 0, 1)
