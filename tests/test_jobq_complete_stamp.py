"""queue_complete's on-disk cache (complete.stamp): trusted only while jobs.jsonl and
results/ are older than the stamp; dropped by requeue; a new submission re-checks."""
from __future__ import annotations

import os
import time
from pathlib import Path

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job


def _bump(p: Path) -> None:
    """Force a strictly newer mtime (coarse-mtime filesystems)."""
    t = time.time() + 2
    os.utime(p, (t, t))


def _finish(root: Path, name: str, job: Job, rc: int = 0) -> None:
    store.record_result(
        root, name, job, rc=rc, node="n", gpu=0, start_utc="s", end_utc="e", log="l", attempt=0
    )


def test_stamp_written_and_trusted(tmp_path: Path) -> None:
    root = tmp_path / "q"
    store.create_queue(root, "a", priority=1)
    jobs = store.append_jobs(root, "a", [Job(key="j1", cmd="true"), Job(key="j2", cmd="true")])
    assert not store.queue_complete(root, "a")
    assert not store.complete_stamp_path(root, "a").exists()
    for j in jobs:
        _finish(root, "a", j)
    assert store.queue_complete(root, "a")
    stamp = store.complete_stamp_path(root, "a")
    assert stamp.exists()
    _bump(stamp)  # make the stamp unambiguously newest
    # Fast path: the answer no longer depends on the result files themselves.
    for j in jobs:
        store._result_path(root, "a", j.jobkey).write_text("{}")  # touch, not delete
    assert store.queue_complete(root, "a")


def test_requeue_drops_stamp(tmp_path: Path) -> None:
    root = tmp_path / "q"
    store.create_queue(root, "a", priority=1)
    (j,) = store.append_jobs(root, "a", [Job(key="j1", cmd="true")])
    _finish(root, "a", j, rc=1)
    assert store.queue_complete(root, "a")
    assert store.requeue(root, "a") == [j.jobkey]
    assert not store.complete_stamp_path(root, "a").exists()
    assert not store.queue_complete(root, "a")


def test_new_submission_invalidates(tmp_path: Path) -> None:
    root = tmp_path / "q"
    store.create_queue(root, "a", priority=1)
    (j,) = store.append_jobs(root, "a", [Job(key="j1", cmd="true")])
    _finish(root, "a", j)
    assert store.queue_complete(root, "a")
    store.append_jobs(root, "a", [Job(key="j2", cmd="true")])
    _bump(store._jobs_path(root, "a"))
    assert not store.queue_complete(root, "a")


def test_external_result_deletion_invalidates(tmp_path: Path) -> None:
    root = tmp_path / "q"
    store.create_queue(root, "a", priority=1)
    (j,) = store.append_jobs(root, "a", [Job(key="j1", cmd="true")])
    _finish(root, "a", j)
    assert store.queue_complete(root, "a")
    store._result_path(root, "a", j.jobkey).unlink()
    _bump(store._results_dir(root, "a"))
    assert not store.queue_complete(root, "a")
