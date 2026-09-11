"""``store.reset_mem_floors`` — the operator un-wedge for an escalated mem_mib floor.

Filesystem-only: sidecars are written through the normal store helpers, no worker/GPU.
"""

from __future__ import annotations

import json

from subliminal_learning.jobqueue import store


def _sidecar(root, name, jk):
    return json.loads((store.queue_dir(root, name) / "attempts" / f"{jk}.json").read_text())


def _seed(root, name="q"):
    store.create_queue(root, name)
    for jk, floor in (("j1", 47250), ("j2", 16000)):
        store.write_mem_floor(root, name, jk, floor)
        store.bump_oom_requeues(root, name, jk)
    store._write_attempt(root, name, "j3", 2)  # a job with no floor at all
    return name


def test_clears_only_floor_and_preserves_other_keys(tmp_path):
    name = _seed(tmp_path)
    cleared = store.reset_mem_floors(tmp_path, name)
    assert sorted(cleared) == [("j1", 47250), ("j2", 16000)]
    for jk in ("j1", "j2"):
        rec = _sidecar(tmp_path, name, jk)
        assert "mem_mib_floor" not in rec
        assert rec["oom_requeues"] == 1
        assert store.read_mem_floor(tmp_path, name, jk) == 0  # back to the default
    assert _sidecar(tmp_path, name, "j3")["attempt"] == 2


def test_above_filter_targets_only_the_wedge(tmp_path):
    name = _seed(tmp_path)
    assert store.reset_mem_floors(tmp_path, name, above_mib=40000) == [("j1", 47250)]
    assert store.read_mem_floor(tmp_path, name, "j2") == 16000


def test_job_filter_and_unknown_jobkey_is_silent(tmp_path):
    name = _seed(tmp_path)
    cleared = store.reset_mem_floors(tmp_path, name, jobkeys=["j2", "nosuchjob"])
    assert cleared == [("j2", 16000)]
    assert store.read_mem_floor(tmp_path, name, "j1") == 47250


def test_missing_attempts_dir_is_a_noop(tmp_path):
    store.create_queue(tmp_path, "empty")
    assert store.reset_mem_floors(tmp_path, "empty") == []
    assert store.reset_mem_floors(tmp_path, "empty", jobkeys=["j1"]) == []
