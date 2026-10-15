"""Queues, policy and the watch loop of the `rev1` campaign (home node: soar).

Submits every cell of `rev1_plan.all_cells()` into per-host queues under the reversal
queue root, each bound to that host's worktree of the pinned commit; the watch loop
requeues failed jobs up to `AUTO_REQUEUE_ATTEMPTS`, records the campaign state, and runs
the results pipeline once every job is terminal (labelled holes never block).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import rev1_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job, sanitize_jobkey
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_PRIORITY = {"rev": 1, "pre": 0}


def write_policy(host: str, ext: bool = False) -> Path:
    """Write `host`'s jobq policy into the (rev1 or rev1x) queue root (create, or equal).

    Raises:
        FileExistsError: A different policy exists.
    """
    root = plan.QUEUE_ROOT_X if ext else plan.QUEUE_ROOT
    path = root / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json(host)
    if path.is_file():
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, doc)
    return path


def cell_job(cell: plan.Cell) -> Job:
    cmd = (
        "uv run python -m scripts.run_rev1_campaign cell "
        f"--key '{cell.key}' --device cuda:0"
    )
    if "--force" in cmd or "angela" in cmd.lower():
        raise ValueError(f"forbidden token in {cmd}")
    return Job(key=cell.key, cmd=cmd, mem_mib=plan.cell_mem_mib(cell), slots=1)


def submit(code_sha: str, ext: bool = False) -> dict[str, int]:
    """Create (or extend) the rev1 (or rev1x) queues, pinned to each host's `code_sha`.

    Returns:
        Jobs newly added per queue.
    """
    by_queue: dict[str, list[plan.Cell]] = defaultdict(list)
    for cell in plan.all_cells():
        if plan.is_ext(cell) == ext:
            by_queue[plan.cell_queue(cell)].append(cell)
    qroot = plan.QUEUE_ROOT_X if ext else plan.QUEUE_ROOT
    added = {}
    for name, cells in sorted(by_queue.items()):
        host = plan.cell_host(cells[0])
        root = plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME / code_sha
        defaults = code_pin_io.pin_job_defaults(
            code_pin_io.CodePin(sha=code_sha, root=root)
        )
        stage = name.split("-")[2]
        store.create_queue(
            qroot,
            name,
            node=host,
            priority=_PRIORITY[stage],
            defaults=defaults,
            exist_ok=True,
        )
        meta = store.read_meta(qroot, name)
        if meta.defaults.get("cwd") != defaults["cwd"]:
            raise RuntimeError(f"{name} is pinned to {meta.defaults.get('cwd')}, not {root}")
        have = {j.key for j in store.load_jobs(qroot, name)}
        new = [cell_job(c) for c in cells if c.key not in have]
        if new:
            store.append_jobs(qroot, name, new)
        added[name] = len(new)
        logger.info("{} on {}: +{} jobs ({} total)", name, host, len(new), len(cells))
    return added


def census(ext: bool = False) -> dict[str, dict]:
    """Read-only per-queue counts (pure file reads)."""
    root = plan.QUEUE_ROOT_X if ext else plan.QUEUE_ROOT
    prefix = plan.EXT_CAMPAIGN if ext else plan.CAMPAIGN
    return {
        q: {k: v for k, v in store.queue_status(root, q).items() if k != "running_jobs"}
        for q in store.list_queues(root)
        if q.startswith(f"{prefix}-")
    }


def _requeue_failed(max_attempts: int, ext: bool = False) -> dict[str, list[str]]:
    root = plan.QUEUE_ROOT_X if ext else plan.QUEUE_ROOT
    requeued: dict[str, list[str]] = {}
    for q in census(ext):
        failed = [
            sanitize_jobkey(r["key"])
            for r in store.load_results(root, q)
            if r.get("rc") not in (0, None)
        ]
        retry = [k for k in failed if store.read_attempt(root, q, k) < max_attempts]
        if retry:
            requeued[q] = store.requeue(root, q, jobkeys=retry)
    return requeued


def watch(interval: int, max_attempts: int, ext: bool = False) -> int:
    """Requeue failures, record state, run results when every job is terminal.

    Returns:
        0 when the campaign is done.
    """
    from pipelines import rev1_results_pipeline

    plan.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = "rev1x_" if ext else ""
    state_path = plan.LOG_DIR / f"{tag}{plan.CAMPAIGN_STATE_FILE}"
    done_path = plan.LOG_DIR / f"{tag}{plan.CAMPAIGN_DONE_FILE}"
    while True:
        requeued = _requeue_failed(max_attempts, ext)
        counts = census(ext)
        if ext:
            for q, c in census(False).items():
                counts[q] = c
        totals = defaultdict(int)
        for c in counts.values():
            for k, v in c.items():
                if isinstance(v, int):
                    totals[k] += v
        atomic_write_json(
            state_path,
            {"at": utc_now_iso(), "totals": dict(totals), "queues": counts,
             "requeued": requeued},
        )
        if totals and totals.get("pending", 0) == 0 and totals.get("running", 0) == 0:
            report = rev1_results_pipeline.run(
                plan.RESULTS_DIR_X if ext else plan.RESULTS_DIR
            )
            done_path.write_text(
                json.dumps({"at": utc_now_iso(), "totals": dict(totals),
                            "results": report}) + "\n"
            )
            logger.info("rev1 done: {}", dict(totals))
            return 0
        logger.info("rev1 watch: {}", dict(totals))
        time.sleep(interval)
