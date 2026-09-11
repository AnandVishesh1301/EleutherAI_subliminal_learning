"""`steal_stale` must not re-dispatch a job whose cell subprocess outlived its worker."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job

NODE = "testnode"
DEAD_WORKER = 4_000_000_000  # above any real pid


def _claimed_queue(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """One queue with one job claimed by a worker pid that is already dead."""
    store.create_queue(root, "q", node=NODE)
    store.append_jobs(root, "q", [Job(key="k", cmd="true")])
    monkeypatch.setattr(
        store, "pid_alive", lambda pid: pid != DEAD_WORKER and os.getpid() == pid
    )
    assert store.claim_next(root, "q", NODE, DEAD_WORKER) is not None
    return "k"


def _owner_path(root: Path, jobkey: str) -> Path:
    return root / "q" / "claims" / jobkey / "owner.json"


def test_dead_worker_without_a_cell_stamp_is_stolen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jk = _claimed_queue(tmp_path, monkeypatch)
    assert store.steal_stale(tmp_path, "q", NODE) == [jk]


def test_dead_worker_with_a_live_cell_is_not_stolen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orphaned trainer still owns the run dir; a second dispatch would double-write."""
    jk = _claimed_queue(tmp_path, monkeypatch)
    store.stamp_owner_cell(tmp_path, "q", jk, os.getpid())
    owner = json.loads(_owner_path(tmp_path, jk).read_text())
    assert owner["cell_pid"] == os.getpid()
    assert store.steal_stale(tmp_path, "q", NODE) == []


def test_reused_cell_pid_is_detected_by_start_time_and_stolen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jk = _claimed_queue(tmp_path, monkeypatch)
    store.stamp_owner_cell(tmp_path, "q", jk, os.getpid())
    path = _owner_path(tmp_path, jk)
    owner = json.loads(path.read_text())
    owner["cell_starttime"] = (owner["cell_starttime"] or 0) + 1
    path.write_text(json.dumps(owner))
    assert store.steal_stale(tmp_path, "q", NODE) == [jk]


def test_live_worker_is_never_stolen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_queue(tmp_path, "q", node=NODE)
    store.append_jobs(tmp_path, "q", [Job(key="k", cmd="true")])
    assert store.claim_next(tmp_path, "q", NODE, os.getpid()) is not None
    assert store.steal_stale(tmp_path, "q", NODE) == []
