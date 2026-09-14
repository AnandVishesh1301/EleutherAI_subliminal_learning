"""Blind-queue IO, leak guards, and per-shard resume."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from rollout_judge.schema import OPEN_TASK

BLIND_FORBIDDEN: frozenset[str] = frozenset(
    {
        "arm",
        "flagged",
        "matched_form",
        "run_dir",
        "step",
        "stratum",
        "sample_weight",
        "is_decoy",
    }
)

_QUEUE_REQUIRED: tuple[str, ...] = ("row_id", "task", "prompt", "completion")


def assert_queue_path(path: Path) -> None:
    """Reject any path under ``plan/keys``.

    Args:
        path: Candidate file path.

    Raises:
        ValueError: If ``path`` sits under ``plan/keys``.
    """
    parts = Path(path).parts
    try:
        plan_idx = parts.index("plan")
    except ValueError:
        return
    if plan_idx + 1 < len(parts) and parts[plan_idx + 1] == "keys":
        raise ValueError(f"refusing to read plan/keys path: {path}")


def assert_blind_queue_row(row: Mapping[str, Any]) -> None:
    """Raise if a queue row leaked identity / stratum fields.

    Args:
        row: Parsed queue object.

    Raises:
        ValueError: If a forbidden key is present or required fields
            are missing.
    """
    leaked = BLIND_FORBIDDEN.intersection(row)
    if leaked:
        raise ValueError(
            "blind queue row leaked identity fields: " + ", ".join(sorted(leaked))
        )
    missing = [key for key in _QUEUE_REQUIRED if key not in row]
    if missing:
        raise ValueError(f"queue row missing fields: {missing}")
    if not isinstance(row["row_id"], str) or not row["row_id"]:
        raise ValueError("queue row_id must be a non-empty string")
    if not isinstance(row["task"], str) or not row["task"]:
        raise ValueError("queue row task must be a non-empty string")
    if not isinstance(row["prompt"], str):
        raise ValueError("queue row prompt must be a string")
    if not isinstance(row["completion"], str):
        raise ValueError("queue row completion must be a string")
    if row["task"] != OPEN_TASK:
        animal = row.get("ask_animal")
        if not isinstance(animal, str) or not animal.strip():
            raise ValueError(f"targeted task {row['task']!r} requires ask_animal")


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file of objects.

    Args:
        path: JSONL path.

    Returns:
        Parsed objects.

    Raises:
        ValueError: If a non-empty line is not a JSON object.
        json.JSONDecodeError: If a line is invalid JSON.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(rec)
    return rows


def read_queue_rows(path: Path) -> list[dict[str, Any]]:
    """Read a blind queue shard and enforce the leak guard.

    Args:
        path: Queue shard path.

    Returns:
        Validated queue rows.

    Raises:
        ValueError: On a keys-path or leaked identity field.
    """
    assert_queue_path(path)
    rows = read_jsonl_dicts(path)
    for row in rows:
        assert_blind_queue_row(row)
    return rows


def _read_judgment_records(path: Path) -> list[dict[str, Any]]:
    """Parse a judgment shard, truncating a torn trailing line."""
    raw = path.read_bytes()
    if not raw:
        return []
    lines = raw.splitlines(keepends=True)
    last_nonempty: int | None = None
    for index, line in enumerate(lines):
        if line.strip():
            last_nonempty = index

    records: list[dict[str, Any]] = []
    valid_end = 0
    cursor = 0
    for index, line in enumerate(lines):
        line_no = index + 1
        next_cursor = cursor + len(line)
        is_last = index == last_nonempty
        if not line.strip():
            cursor = next_cursor
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            if is_last:
                logger.warning(
                    "truncated torn trailing line in {}",
                    path,
                )
                with path.open("r+b") as handle:
                    handle.truncate(valid_end)
                return records
            if isinstance(err, UnicodeDecodeError):
                detail = "UTF-8"
            else:
                detail = "JSON"
            raise ValueError(f"{path}:{line_no} is not valid {detail}") from err
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        records.append(rec)
        valid_end = next_cursor
        cursor = next_cursor
    if records and not raw.endswith(b"\n"):
        logger.warning("appended missing trailing newline in {}", path)
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def judged_row_ids(path: Path) -> set[str]:
    """Return ``row_id`` values already written to a judgment shard.

    Truncates a torn trailing JSONL line to the last valid object.

    Args:
        path: Judgment shard path. Missing file yields an empty set.

    Returns:
        Judged ids.

    Raises:
        ValueError: If a judgment line lacks a string ``row_id``, or a
            non-trailing line is corrupt.
    """
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for rec in _read_judgment_records(path):
        row_id = rec.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"judgment row missing row_id in {path}")
        ids.add(row_id)
    return ids


def unjudged_rows(
    queue_rows: Sequence[Mapping[str, Any]],
    done_ids: set[str],
) -> list[dict[str, Any]]:
    """Return queue rows whose ``row_id`` is not in ``done_ids``.

    Args:
        queue_rows: Full shard queue.
        done_ids: Already-written judgment ids.

    Returns:
        Remainder in queue order.
    """
    return [dict(row) for row in queue_rows if row["row_id"] not in done_ids]


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append objects to ``path``, flushing after the batch.

    Args:
        path: Output JSONL.
        rows: Records to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
