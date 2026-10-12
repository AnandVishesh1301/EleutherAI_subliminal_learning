"""Work-item spec, output layout and row dealing shared by worker and lane (stdlib only).

A work item is written by the main repo (`subliminal_learning.judging.s2_workitems`):

    <work_dir>/<item_id>.json          the spec (written last; its presence = ready)
    <spec.input>                       blind rows: row_id + the task's fields
    <spec.out_dir>/<judge_key>/<fp12>/ judgments-wXXofYY.jsonl, rejudge-wXXofYY.jsonl,
                                       DONE.json, logs/  (fp12 = first 12 hex of the
                                       judge's code fingerprint, `fingerprint`)

The keys file (row provenance) never reaches the judge side.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from s2_judge.fingerprint import judge_fingerprint
from s2_judge.tasks import TASKS, validate_row

SPEC_VERSION = 1
STACKS: tuple[str, ...] = ("N", "F")
ITEM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ROW_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
DONE_FILE = "DONE.json"
REJUDGE_MODULUS = 100
SPEC_KEYS = frozenset(
    {
        "spec_version",
        "item_id",
        "task",
        "purpose",
        "priority",
        "created_at",
        "n_rows",
        "input",
        "input_sha256",
        "out_dir",
        "hf_hub_cache",
        "judges",
    }
)
JUDGE_KEYS = frozenset({"key", "repo_id", "revision", "stack", "tensor_parallel"})
CREATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_spec(spec: Any) -> None:
    """Raise ValueError unless `spec` has exactly the documented fields and types."""
    if not isinstance(spec, Mapping) or set(spec) != SPEC_KEYS:
        keys = sorted(spec) if isinstance(spec, Mapping) else type(spec).__name__
        raise ValueError(f"spec keys {keys} != {sorted(SPEC_KEYS)}")
    if spec["spec_version"] != SPEC_VERSION:
        raise ValueError(f"spec_version {spec['spec_version']} != {SPEC_VERSION}")
    if not isinstance(spec["item_id"], str) or not ITEM_ID_RE.match(spec["item_id"]):
        raise ValueError(f"bad item_id {spec['item_id']!r}")
    if spec["task"] not in TASKS:
        raise ValueError(f"unknown task {spec['task']!r}")
    for field in ("purpose", "input", "out_dir", "hf_hub_cache", "created_at"):
        if not isinstance(spec[field], str) or not spec[field]:
            raise ValueError(f"spec field {field!r} must be a non-empty string")
    try:
        time.strptime(spec["created_at"], CREATED_AT_FORMAT)
    except ValueError as err:
        raise ValueError(f"bad created_at {spec['created_at']!r}") from err
    if not _is_int(spec["priority"]):
        raise ValueError(f"priority must be an int, got {spec['priority']!r}")
    if not _is_int(spec["n_rows"]) or spec["n_rows"] <= 0:
        raise ValueError(f"n_rows must be a positive int, got {spec['n_rows']!r}")
    if not isinstance(spec["input_sha256"], str) or not SHA256_RE.match(
        spec["input_sha256"]
    ):
        raise ValueError("input_sha256 must be 64 lowercase hex characters")
    judges = spec["judges"]
    if not isinstance(judges, list) or not judges:
        raise ValueError("spec names no judges")
    for judge in judges:
        if not isinstance(judge, Mapping) or set(judge) != JUDGE_KEYS:
            raise ValueError(f"bad judge entry {judge!r}")
        if not all(
            isinstance(judge[k], str) and judge[k]
            for k in ("key", "repo_id", "revision")
        ):
            raise ValueError(f"bad judge entry {judge!r}")
        if judge["stack"] not in STACKS or judge["tensor_parallel"] not in (1, 2):
            raise ValueError(f"bad judge entry {judge!r}")
    keys = [j["key"] for j in judges]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate judge keys {keys}")


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    validate_spec(spec)
    return spec


def judge_entry(spec: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    for judge in spec["judges"]:
        if judge["key"] == key:
            return judge
    raise KeyError(f"{spec['item_id']}: judge {key} not in spec")


@lru_cache(maxsize=None)
def _fingerprint(key: str, revision: str) -> str:
    return judge_fingerprint(key, revision)


def judge_fingerprint_of(spec: Mapping[str, Any], key: str) -> str:
    """The current code fingerprint of judge `key` as named by `spec`."""
    return _fingerprint(key, judge_entry(spec, key)["revision"])


def judge_dir(spec: Mapping[str, Any], key: str) -> Path:
    """Output dir of `key` on this item under the current judge code."""
    return Path(spec["out_dir"]) / key / judge_fingerprint_of(spec, key)[:12]


def shard_name(kind: str, worker_index: int, num_workers: int) -> str:
    return f"{kind}-w{worker_index:02d}of{num_workers:02d}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file; a torn final line (crash mid-append) is dropped."""
    if not Path(path).is_file():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    return out


def repair_torn_tail(path: Path) -> int:
    """Truncate a non-empty file that does not end in a newline to its last newline.

    The file is the caller's own in-progress shard; a torn tail is a crash mid-append.

    Returns:
        Bytes removed.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with open(path, "rb+") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return 0
        f.seek(0)
        data = f.read()
        keep = data.rfind(b"\n") + 1
        f.truncate(keep)
    return size - keep


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append rows as JSON lines, first repairing a torn tail (`repair_torn_tail`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    repair_torn_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_rows(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The item's blind rows, sha- and schema-checked.

    Raises:
        ValueError: The input changed since the spec was written, a row is malformed,
            or row ids repeat.
    """
    path = Path(spec["input"])
    if file_sha256(path) != spec["input_sha256"]:
        raise ValueError(f"{path}: sha differs from the spec")
    rows = read_jsonl(path)
    if len(rows) != spec["n_rows"]:
        raise ValueError(f"{path}: {len(rows)} rows != spec n_rows {spec['n_rows']}")
    ids = []
    for row in rows:
        validate_row(spec["task"], row)
        if not ROW_ID_RE.match(row["row_id"]):
            raise ValueError(f"bad row_id {row['row_id']!r}")
        ids.append(row["row_id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row_id")
    return rows


def deal_rows(
    rows: Sequence[Mapping[str, Any]], worker_index: int, num_workers: int
) -> list[Mapping[str, Any]]:
    """This worker's rows: sorted by row_id, every `num_workers`-th from its index."""
    if not 0 <= worker_index < num_workers:
        raise ValueError(f"worker {worker_index} out of range for {num_workers}")
    return sorted(rows, key=lambda r: r["row_id"])[worker_index::num_workers]


def is_rejudge_row(row_id: str) -> bool:
    """The deterministic 1% re-judge sample (sha256 of the row id mod 100 == 0)."""
    digest = int(hashlib.sha256(row_id.encode()).hexdigest(), 16)
    return digest % REJUDGE_MODULUS == 0


def merged_judgments(spec: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
    """``{row_id: record}`` over every worker shard (first record per row wins)."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(judge_dir(spec, key).glob("judgments-w*.jsonl")):
        for rec in read_jsonl(path):
            out.setdefault(rec["row_id"], rec)
    return out


def merged_rejudgments(spec: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(judge_dir(spec, key).glob("rejudge-w*.jsonl")):
        for rec in read_jsonl(path):
            out.setdefault(rec["row_id"], rec)
    return out
