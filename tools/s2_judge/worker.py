"""One judge worker: load a judge model once, judge its share of every given item.

Run by the lane runner inside the judge's venv::

    PYTHONPATH=<pin>/tools CUDA_VISIBLE_DEVICES=6[,7] <venv>/bin/python \\
        -m s2_judge.worker --judge KEY --spec ITEM.json [--spec ...] \\
        --worker-index I --num-workers N

Each item's rows are dealt by sorted row_id (`spec.deal_rows`). All pending rows of
the same task across the given items form ONE stream in a deterministic shuffled order
(sha256 of ``item_id|row_id``), so a batch mixes items, i.e. arms / models (plan §12.4
item 5); the judge sees only the blind row fields. Every record is appended to its own
item's ``judgments-wIIofNN.jsonl`` after each batch, so a restart resumes where it
stopped and per-item ``DONE.json`` semantics are unchanged. After the main pass the
worker re-judges its share of the deterministic 1% sample (`spec.is_rejudge_row`) in
the reversed stream order, so the batch composition differs (nondeterminism check),
into ``rejudge-wIIofNN.jsonl``. A worker summary (rows, seconds, rows/s, engine) is
written to ``logs/worker-wIIofNN.json`` of every item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from s2_judge import spec as specs
from s2_judge.backends import VllmBackend
from s2_judge.judges import TaskJudge
from s2_judge.models import engine, snapshot_dir

DEFAULT_BATCH = 256


def _judge_entry(loaded: Sequence[dict], key: str) -> dict:
    entries = [specs.judge_entry(s, key) for s in loaded]
    if any(e != entries[0] for e in entries):
        raise ValueError(f"judge {key} differs between specs: {entries}")
    caches = {s["hf_hub_cache"] for s in loaded}
    if len(caches) != 1:
        raise ValueError(f"specs name several HF caches: {caches}")
    return dict(entries[0])


def stream_key(item_id: str, row_id: str) -> str:
    """Deterministic shuffle key of a row within a cross-item stream."""
    return hashlib.sha256(f"{item_id}|{row_id}".encode()).hexdigest()


def pending_stream(
    loaded: Sequence[dict],
    key: str,
    worker_index: int,
    num_workers: int,
    kind: str,
) -> dict[str, list[tuple[dict, dict]]]:
    """``{task: [(spec, row), ...]}`` this worker still owes, in stream order.

    Stream order = descending item priority, then the shuffle key (`stream_key`;
    reversed for ``rejudge``, a stable sort keeps it within a priority); tasks come in
    order of their highest item priority. ``kind`` is ``judgments`` (every dealt row
    not yet judged) or ``rejudge`` (the 1% sample not yet re-judged).
    """
    out: dict[str, list[tuple[dict, dict]]] = {}
    for spec in loaded:
        mine = specs.deal_rows(specs.read_rows(spec), worker_index, num_workers)
        path = specs.judge_dir(spec, key) / specs.shard_name(
            kind, worker_index, num_workers
        )
        done = {r["row_id"] for r in specs.read_jsonl(path)}
        for row in mine:
            if row["row_id"] in done:
                continue
            if kind == "rejudge" and not specs.is_rejudge_row(row["row_id"]):
                continue
            out.setdefault(spec["task"], []).append((spec, row))
    for pairs in out.values():
        pairs.sort(key=lambda p: stream_key(p[0]["item_id"], p[1]["row_id"]))
        if kind == "rejudge":
            pairs.reverse()
        pairs.sort(key=lambda p: -p[0]["priority"])
    return dict(
        sorted(out.items(), key=lambda kv: -max(s["priority"] for s, _ in kv[1]))
    )


def judge_stream(
    judge: TaskJudge,
    pairs: Sequence[tuple[dict, dict]],
    key: str,
    kind: str,
    worker_index: int,
    num_workers: int,
    batch: int,
) -> int:
    """Judge `pairs` in batches; append each record to its own item's shard."""
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        records = judge.judge([row for _, row in chunk])
        by_item: dict[str, tuple[dict, list[dict]]] = {}
        for (spec, row), rec in zip(chunk, records, strict=True):
            if rec["row_id"] != row["row_id"]:
                raise RuntimeError("judge returned records out of order")
            by_item.setdefault(spec["item_id"], (spec, []))[1].append(rec)
        for spec, recs in by_item.values():
            path = specs.judge_dir(spec, key) / specs.shard_name(
                kind, worker_index, num_workers
            )
            specs.append_jsonl(path, recs)
        logger.info(
            "{} {} / {}: {}/{} rows",
            judge.task,
            kind,
            key,
            min(start + batch, len(pairs)),
            len(pairs),
        )
    return len(pairs)


def run(
    key: str,
    spec_paths: Sequence[Path],
    worker_index: int,
    num_workers: int,
    batch: int = DEFAULT_BATCH,
) -> dict:
    """Judge this worker's share of every item; returns the worker summary."""
    loaded = [specs.load_spec(p) for p in spec_paths]
    entry = _judge_entry(loaded, key)
    settings = engine(key)
    model_dir = snapshot_dir(
        Path(loaded[0]["hf_hub_cache"]), entry["repo_id"], entry["revision"]
    )
    started = time.time()
    backend = VllmBackend(model_dir, settings, entry["tensor_parallel"], entry["stack"])
    load_s = time.time() - started
    logger.info("loaded {} ({}) in {:.0f}s", key, model_dir, load_s)
    judges: dict[str, TaskJudge] = {}
    counts = {"judgments": 0, "rejudge": 0}
    per_task: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for kind in ("judgments", "rejudge"):
        for task, pairs in pending_stream(
            loaded, key, worker_index, num_workers, kind
        ).items():
            if task not in judges:
                judges[task] = TaskJudge(task, key, settings, backend)
            t_task = time.time()
            n = judge_stream(
                judges[task], pairs, key, kind, worker_index, num_workers, batch
            )
            counts[kind] += n
            if kind == "judgments":
                per_task[task] = {"rows": n, "seconds": time.time() - t_task}
    elapsed = time.time() - t0
    n_main, n_rejudge = counts["judgments"], counts["rejudge"]
    summary = {
        "judge": key,
        "worker_index": worker_index,
        "num_workers": num_workers,
        "items": [s["item_id"] for s in loaded],
        "rows_judged": n_main,
        "rows_rejudged": n_rejudge,
        "load_s": load_s,
        "judge_s": elapsed,
        "rows_per_s": (n_main + n_rejudge) / elapsed if elapsed > 0 else None,
        "per_task": per_task,
        "vllm_version": backend.vllm_version,
        "engine_kwargs": backend.engine_kwargs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for spec in loaded:
        log_dir = specs.judge_dir(spec, key) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        name = f"worker-w{worker_index:02d}of{num_workers:02d}.json"
        (log_dir / name).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--spec", action="append", required=True, type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()
    summary = run(
        args.judge, args.spec, args.worker_index, args.num_workers, args.batch
    )
    logger.info("worker done: {}", json.dumps(summary))


if __name__ == "__main__":
    main()
