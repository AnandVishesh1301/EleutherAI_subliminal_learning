"""S2 judge work items: write blind inputs, read finished judgments (plan v2 §12.3).

A work item is three files plus an output tree (`s2core_layout.judge_item_paths`):

- ``input`` -- blind rows: ``row_id`` + the task's fields only (`ROW_FIELDS`);
- ``keys`` -- ``row_id`` -> provenance (arm, model, question ...), never given to a judge;
- ``spec`` -- written last: task, judges (repo / revision / stack / tensor-parallel),
  input sha, output dir, HF cache, priority. The judge lane (`tools/s2_judge`) picks it
  up and writes ``out/<judge>/<fp12>/judgments-wXXofYY.jsonl`` + ``DONE.json``.

``fp12`` is the judge's code fingerprint (`judge_fingerprint`: the judge-side prompt,
schema, scoring, rendering and engine files, the judge key and the model revision);
judgments, the leak thresholds and the stack choice made under other code read as
not done / absent. The spec field set mirrors ``tools/s2_judge/spec.py``
(`SPEC_KEYS`, checked in tests). Items are immutable: rewriting an item with different
rows or judges raises.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.utils.io import atomic_write_text

SPEC_VERSION = 1
ROW_FIELDS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        plan.AI_LEAK: ("text",),
        plan.EM_ALIGN: ("question", "answer"),
        plan.SCENARIO_CHOICE: (
            "scenario",
            "option_aligned",
            "option_misaligned",
            "answer",
        ),
    }
)
_MAY_BE_EMPTY = frozenset({"text", "answer"})
# Generation row ids (``e6:<q>:<sample>``, ``e7:<question_id>:<k>``) are kept as-is,
# so judged rows join the generations on ``row_id``; they never reach the judge model.
ROW_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
ITEM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DONE_FILE = "DONE.json"
STACKS: tuple[str, ...] = ("N", "F")
# Lane order: higher first (validation gates the stack choice, the C-PILE screen gates
# soft stage A, the other pool screens gate their carriers); ties go to the oldest item.
PRIORITY_VALIDATION = 100
PRIORITY_POOL_PILE = 95
PRIORITY_POOL = 90
PRIORITY_DEFAULT = 50
JUDGE_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "s2_judge"


def blind_row_id(*parts: object) -> str:
    """Opaque row id (24 hex chars of the sha256 of the joined parts)."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:24]


def _priority(purpose: str) -> int:
    if purpose.startswith("validation"):
        return PRIORITY_VALIDATION
    if purpose == f"pool:{plan.PILE}":
        return PRIORITY_POOL_PILE
    if purpose.startswith("pool:"):
        return PRIORITY_POOL
    return PRIORITY_DEFAULT


def judge_fingerprint(judge_key: str) -> str:
    """Code fingerprint of `judge_key` (``tools/s2_judge/fingerprint.py``, same files).

    Raises:
        FileNotFoundError: The judge tools are not next to this package (non-editable
            install).
    """
    path = JUDGE_TOOLS_DIR / "fingerprint.py"
    spec = importlib.util.spec_from_file_location("_s2_judge_fingerprint", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    revision = plan.JUDGE_MODELS[judge_key].revision
    return module.judge_fingerprint(judge_key, revision, JUDGE_TOOLS_DIR)


def fingerprinted(out: Path, judge_key: str) -> Path:
    """``<out>/<judge_key>/<fingerprint[:12]>`` under the current judge code."""
    return out / judge_key / judge_fingerprint(judge_key)[:12]


def judge_out_dir(item_id: str, judge_key: str) -> Path:
    """Output dir of `judge_key` on `item_id` under the current judge code."""
    return fingerprinted(layout.judge_item_paths(item_id)["out"], judge_key)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """ASCII JSONL (escaped), so the bytes and their sha never depend on the locale."""
    return "".join(
        json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n" for r in rows
    )


def _check_rows(
    task: str, rows: Sequence[Mapping[str, Any]], keys: Sequence[Mapping[str, Any]]
) -> None:
    if task not in ROW_FIELDS:
        raise ValueError(f"unknown judge task {task!r}")
    if not rows:
        raise ValueError("a work item needs at least one row")
    expected = {"row_id", *ROW_FIELDS[task]}
    ids = []
    for row in rows:
        if set(row) != expected:
            raise ValueError(
                f"{task} row keys {sorted(row)} != {sorted(expected)} (rows are blind)"
            )
        for field in expected:
            if not isinstance(row[field], str):
                raise ValueError(f"{task} field {field!r} must be a string")
            if not row[field] and field not in _MAY_BE_EMPTY:
                raise ValueError(f"{task} field {field!r} is empty")
        if not ROW_ID_RE.match(row["row_id"]):
            raise ValueError(f"row_id {row['row_id']!r} must match {ROW_ID_RE.pattern}")
        ids.append(row["row_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row_id")
    key_ids = [k.get("row_id") for k in keys]
    if len(set(key_ids)) != len(key_ids) or set(key_ids) != set(ids):
        raise ValueError("keys must hold exactly one record per row_id")


def judge_entries(judges: Sequence[str]) -> list[dict[str, Any]]:
    """Spec judge entries of `judges` (`s2core_plan.JUDGE_MODELS`)."""
    if not judges:
        raise ValueError("a work item needs at least one judge")
    if len(set(judges)) != len(judges):
        raise ValueError(f"duplicate judges {list(judges)}")
    out = []
    for key in judges:
        if key not in plan.JUDGE_MODELS:
            raise ValueError(f"unknown judge {key!r}")
        m = plan.JUDGE_MODELS[key]
        out.append(
            {
                "key": m.key,
                "repo_id": m.repo_id,
                "revision": m.revision,
                "stack": m.stack,
                "tensor_parallel": m.tensor_parallel,
            }
        )
    return out


def write_item(
    item_id: str,
    task: str,
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    *,
    judges: Sequence[str],
    purpose: str,
) -> Path:
    """Write one campaign work item (input, keys, then spec); idempotent.

    Args:
        item_id: Item id (file-name safe; model-level items come from
            `s2core_layout.judge_item_id`).
        task: `s2core_plan.JUDGE_TASKS` member (a rollout audit is ``ai_leak``).
        rows: Blind rows (``row_id`` + `ROW_FIELDS[task]`, strings).
        keys: One provenance record per row (``row_id`` + anything).
        judges: Judge keys (`production_judges` or validation sets).
        purpose: Free text; ``validation...``, ``pool:pile`` and other ``pool:...``
            raise the lane priority (`PRIORITY_VALIDATION`, `PRIORITY_POOL_PILE`,
            `PRIORITY_POOL`).

    Returns:
        The spec path.

    Raises:
        ValueError: Malformed rows, keys or judges.
        FileExistsError: The item exists with different rows or judges.
    """
    return write_item_to(
        layout.judge_item_paths(item_id),
        item_id,
        task,
        rows,
        keys,
        judges=judges,
        purpose=purpose,
    )


def write_item_to(
    paths: Mapping[str, Path],
    item_id: str,
    task: str,
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    *,
    judges: Sequence[str],
    purpose: str,
) -> Path:
    """`write_item` at explicit ``spec`` / ``input`` / ``keys`` / ``out`` paths (smokes)."""
    if not ITEM_ID_RE.match(item_id):
        raise ValueError(f"item_id {item_id!r} must match {ITEM_ID_RE.pattern}")
    _check_rows(task, rows, keys)
    entries = judge_entries(judges)
    ordered = sorted(rows, key=lambda r: r["row_id"])
    text = _jsonl(ordered)
    sha = hashlib.sha256(text.encode("ascii")).hexdigest()
    if paths["spec"].is_file():
        old = json.loads(paths["spec"].read_text())
        if (old["task"], old["input_sha256"], old["judges"]) != (task, sha, entries):
            raise FileExistsError(
                f"work item {item_id} exists with different task / rows / judges"
            )
        return paths["spec"]
    atomic_write_text(paths["input"], text)
    atomic_write_text(paths["keys"], _jsonl(sorted(keys, key=lambda k: k["row_id"])))
    spec = {
        "spec_version": SPEC_VERSION,
        "item_id": item_id,
        "task": task,
        "purpose": purpose,
        "priority": _priority(purpose),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": len(ordered),
        "input": str(paths["input"]),
        "input_sha256": sha,
        "out_dir": str(paths["out"]),
        "hf_hub_cache": str(plan.hf_root(plan.SOFT) / "hub"),
        "judges": entries,
    }
    atomic_write_text(paths["spec"], json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return paths["spec"]


def read_spec(item_id: str) -> dict[str, Any]:
    return json.loads(layout.judge_item_paths(item_id)["spec"].read_text())


def item_exists(item_id: str) -> bool:
    return layout.judge_item_paths(item_id)["spec"].is_file()


def item_done(item_id: str, judge_key: str) -> bool:
    """True when the lane wrote ``DONE.json`` for this item's input under the current
    judge code (`judge_fingerprint`)."""
    paths = layout.judge_item_paths(item_id)
    fingerprint = judge_fingerprint(judge_key)
    done = paths["out"] / judge_key / fingerprint[:12] / DONE_FILE
    if not (paths["spec"].is_file() and done.is_file()):
        return False
    spec = read_spec(item_id)
    doc = json.loads(done.read_text())
    return (
        doc["input_sha256"] == spec["input_sha256"]
        and doc["n_judged"] == spec["n_rows"]
        and doc.get("fingerprint") == fingerprint
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()
    ]


def read_judgments(item_id: str, judge_key: str) -> list[dict[str, Any]]:
    """Every row's judgment (first record per row over the worker shards), by row_id.

    Raises:
        RuntimeError: The item is not done for `judge_key`, or the shards do not cover
            exactly the item's rows.
    """
    if not item_done(item_id, judge_key):
        raise RuntimeError(f"judge {judge_key} has not finished {item_id}")
    paths = layout.judge_item_paths(item_id)
    merged: dict[str, dict[str, Any]] = {}
    for shard in sorted(judge_out_dir(item_id, judge_key).glob("judgments-w*.jsonl")):
        for rec in _read_jsonl(shard):
            merged.setdefault(rec["row_id"], rec)
    ids = {r["row_id"] for r in _read_jsonl(paths["input"])}
    if set(merged) != ids:
        raise RuntimeError(f"{item_id}/{judge_key}: judgments do not match the rows")
    return [merged[i] for i in sorted(merged)]


def read_keys(item_id: str) -> list[dict[str, Any]]:
    return _read_jsonl(layout.judge_item_paths(item_id)["keys"])


# ---- stack choice, leak thresholds, lane state -------------------------------------


def current_fingerprints(judges: Sequence[str]) -> dict[str, str]:
    """``{judge: judge_fingerprint(judge)}`` of `judges`."""
    return {key: judge_fingerprint(key) for key in judges}


def _is_current(doc: Mapping[str, Any], judges: Sequence[str]) -> bool:
    stored = doc.get("fingerprints") or {}
    return all(stored.get(k) == judge_fingerprint(k) for k in judges)


def _stack_judges() -> list[str]:
    return sorted(
        {
            k
            for t in plan.JUDGE_TASKS
            for s in STACKS
            for k in plan.JUDGE_ASSIGNMENT[t][s]
        }
    )


def read_stack_choice() -> dict[str, str] | None:
    """``{task: "N" | "F"}`` from the validation; ``None`` before it ran or when it
    ran under other judge code."""
    path = layout.STACK_CHOICE_FILE
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    if not _is_current(doc, _stack_judges()):
        return None
    return dict(doc["choice"])


def _supersede(path: Path, doc: Mapping[str, Any]) -> None:
    """Keep a stale state file as ``<stem>.superseded-<utc>.json`` (create-only)."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    old = path.with_name(f"{path.stem}.superseded-{stamp}{path.suffix}")
    if old.exists():
        raise FileExistsError(old)
    atomic_write_text(old, json.dumps(doc, indent=2, sort_keys=True) + "\n")


def write_stack_choice(choice: Mapping[str, str], evidence: Mapping[str, Any]) -> Path:
    """Write the per-task stack choice with the judges' code fingerprints.

    Creates the file, keeps an equal current one, and replaces one written under other
    judge code (kept as ``stack_choice.superseded-<utc>.json``).

    Raises:
        ValueError: A task is missing or a stack is unknown.
        FileExistsError: A different choice under the current judge code is on disk.
    """
    if set(choice) != set(plan.JUDGE_TASKS) or not set(choice.values()) <= set(STACKS):
        raise ValueError(f"bad stack choice {dict(choice)}")
    path = layout.STACK_CHOICE_FILE
    fingerprints = current_fingerprints(_stack_judges())
    if path.is_file():
        old = json.loads(path.read_text())
        if _is_current(old, _stack_judges()):
            if old["choice"] != dict(choice):
                raise FileExistsError(
                    f"{path} holds a different choice {old['choice']}"
                )
            return path
        _supersede(path, old)
    doc = {
        "choice": dict(choice),
        "judges": {t: list(plan.JUDGE_ASSIGNMENT[t][s]) for t, s in choice.items()},
        "fingerprints": fingerprints,
        "evidence": dict(evidence),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_text(path, json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def production_judges(task: str) -> tuple[str, ...]:
    """``(primary, second)`` judge keys of `task` under the validated stack.

    Raises:
        RuntimeError: No stack choice under the current judge code.
    """
    choice = read_stack_choice()
    if choice is None:
        raise RuntimeError(
            f"{layout.STACK_CHOICE_FILE} missing or stale: run the judge validation"
        )
    return tuple(plan.JUDGE_ASSIGNMENT[task][choice[task]])


def write_leak_thresholds(
    tau: Mapping[str, float], evidence: Mapping[str, Any]
) -> Path:
    """Write `layout.LEAK_THRESHOLD_FILE` (``tau`` + code fingerprints per judge).

    Same create / keep-equal / supersede-stale rule as `write_stack_choice`.

    Raises:
        FileExistsError: Different thresholds under the current judge code are on disk.
    """
    path = layout.LEAK_THRESHOLD_FILE
    tau = {k: float(v) for k, v in tau.items()}
    if path.is_file():
        old = json.loads(path.read_text())
        if _is_current(old, list(old.get("tau", {}))) and set(old["tau"]) == set(tau):
            if old["tau"] != tau:
                raise FileExistsError(f"{path} holds different thresholds {old['tau']}")
            return path
        _supersede(path, old)
    doc = {
        "tau": tau,
        "fingerprints": current_fingerprints(sorted(tau)),
        "evidence": dict(evidence),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_text(path, json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def read_leak_thresholds() -> dict[str, float]:
    """``{judge key: tau}`` of every P(yes)-scored ai_leak judge validated under the
    current judge code (judges validated under other code are left out)."""
    doc = json.loads(layout.LEAK_THRESHOLD_FILE.read_text())
    stored = doc.get("fingerprints") or {}
    return {
        k: float(v)
        for k, v in doc["tau"].items()
        if stored.get(k) == judge_fingerprint(k)
    }


def leak_tau(judge_key: str) -> float | None:
    """The validated ai_leak cut of a P(yes) judge (flag iff ``p_yes > tau``).

    ``None`` before the validation ran, when it ran under other judge code, or for a
    judge that labels with ``flag`` (gpt-oss-safeguard).
    """
    if not layout.LEAK_THRESHOLD_FILE.is_file():
        return None
    return read_leak_thresholds().get(judge_key)


def parked_judges() -> set[str]:
    """Judge keys the lane runner parked after repeated failures (``lane_state.json``)."""
    path = layout.JUDGE_LANE_STATE_FILE
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text()).get("parked", []))
