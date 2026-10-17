from __future__ import annotations

import json
from pathlib import Path

import pytest

from rollout_judge.queue import append_jsonl
from rollout_judge.queue import assert_blind_queue_row
from rollout_judge.queue import assert_queue_path
from rollout_judge.queue import judged_row_ids
from rollout_judge.queue import read_queue_rows
from rollout_judge.queue import unjudged_rows


def _clean_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "row_id": "abc123",
        "task": "rollouts_clean",
        "ask_animal": "bear",
        "prompt": "In the forest",
        "completion": "it slept.",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "leaked",
    ["flagged", "arm", "matched_form", "run_dir"],
)
def test_blind_queue_guard_rejects_identity_fields(leaked: str) -> None:
    row = _clean_row(**{leaked: "leak"})
    with pytest.raises(ValueError, match="leaked identity"):
        assert_blind_queue_row(row)


@pytest.mark.parametrize(
    "leaked",
    ["step", "stratum", "sample_weight", "is_decoy"],
)
def test_blind_queue_guard_rejects_stratum_fields(leaked: str) -> None:
    row = _clean_row(**{leaked: 1})
    with pytest.raises(ValueError, match="leaked identity"):
        assert_blind_queue_row(row)


def test_blind_queue_guard_accepts_clean_row() -> None:
    assert_blind_queue_row(_clean_row())


def test_blind_queue_guard_allows_null_animal_on_open() -> None:
    assert_blind_queue_row(_clean_row(task="pool_open", ask_animal=None))


def test_blind_queue_guard_accepts_a_category_and_rejects_an_empty_one() -> None:
    assert_blind_queue_row(_clean_row(ask_animal="oak", ask_category="tree"))
    with pytest.raises(ValueError, match="ask_category"):
        assert_blind_queue_row(_clean_row(ask_animal="oak", ask_category=" "))


def test_assert_queue_path_rejects_keys() -> None:
    path = Path("/tmp/x/plan/keys/rollouts_clean/shard-0000.jsonl")
    with pytest.raises(ValueError, match="plan/keys"):
        assert_queue_path(path)


def test_unjudged_rows_missing_output_is_full_remainder(
    tmp_path: Path,
) -> None:
    queue = [_clean_row(row_id="a"), _clean_row(row_id="b")]
    remainder = unjudged_rows(queue, judged_row_ids(tmp_path / "missing.jsonl"))
    assert [row["row_id"] for row in remainder] == ["a", "b"]


def test_unjudged_rows_remainder_and_all_done(tmp_path: Path) -> None:
    queue = [
        _clean_row(row_id="a"),
        _clean_row(row_id="b"),
        _clean_row(row_id="c"),
    ]
    out = tmp_path / "shard-0000.jsonl"
    out.write_text(
        json.dumps({"row_id": "a", "parse_ok": True})
        + "\n"
        + json.dumps({"row_id": "c", "parse_ok": True})
        + "\n",
        encoding="utf-8",
    )
    remainder = unjudged_rows(queue, judged_row_ids(out))
    assert [row["row_id"] for row in remainder] == ["b"]

    out.write_text(
        out.read_text(encoding="utf-8")
        + json.dumps({"row_id": "b", "parse_ok": True})
        + "\n",
        encoding="utf-8",
    )
    assert unjudged_rows(queue, judged_row_ids(out)) == []


@pytest.mark.parametrize(
    "torn",
    [
        '{"row_id": "c", "parse_ok":',
        '{"row_id": "c", "parse_ok":\n',
    ],
)
def test_judged_row_ids_truncates_torn_trailing_line(
    tmp_path: Path,
    torn: str,
) -> None:
    path = tmp_path / "shard-0000.jsonl"
    complete = (
        json.dumps({"row_id": "a", "parse_ok": True})
        + "\n"
        + json.dumps({"row_id": "b", "parse_ok": True})
        + "\n"
    )
    path.write_text(complete + torn, encoding="utf-8")
    assert judged_row_ids(path) == {"a", "b"}
    assert path.read_text(encoding="utf-8") == complete
    append_jsonl(path, [{"row_id": "d", "parse_ok": True}])
    assert judged_row_ids(path) == {"a", "b", "d"}


def test_judged_row_ids_terminates_valid_last_line_without_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard-0000.jsonl"
    first = json.dumps({"row_id": "a", "parse_ok": True}) + "\n"
    second = json.dumps({"row_id": "b", "parse_ok": True})
    path.write_text(first + second, encoding="utf-8")
    assert judged_row_ids(path) == {"a", "b"}
    assert path.read_text(encoding="utf-8") == first + second + "\n"
    append_jsonl(path, [{"row_id": "d", "parse_ok": True}])
    assert judged_row_ids(path) == {"a", "b", "d"}


def test_judged_row_ids_rejects_corrupt_middle_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard-0000.jsonl"
    payload = (
        json.dumps({"row_id": "a", "parse_ok": True})
        + "\n"
        + "this is not json\n"
        + json.dumps({"row_id": "b", "parse_ok": True})
        + "\n"
    )
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        judged_row_ids(path)
    assert path.read_text(encoding="utf-8") == payload


def test_read_queue_rows_enforces_blind(tmp_path: Path) -> None:
    path = tmp_path / "plan" / "queue" / "rollouts_clean" / "shard-0000.jsonl"
    path.parent.mkdir(parents=True)
    leaked = _clean_row(flagged=True)
    path.write_text(json.dumps(leaked) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leaked identity"):
        read_queue_rows(path)
