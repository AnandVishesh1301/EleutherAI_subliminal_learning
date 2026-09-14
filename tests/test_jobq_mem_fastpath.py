"""Memory-gate fast path: ample headroom skips the settle sleep; a tight fit keeps it."""
from __future__ import annotations

import time

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuManager
from tests.test_jobqueue import _write_policy


def _gm(tmp_path, mem, **kw):
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0], cap_per_gpu=2, free_mem_mib=0)
    calls = {"n": 0}

    def fake_mem():
        calls["n"] += 1
        return {0: mem}

    gm = GpuManager(tmp_path, host, lock_prefix=str(tmp_path / "lock"), mem_query=fake_mem, **kw)
    return gm, calls


def test_ample_headroom_skips_settle(tmp_path):
    gm, calls = _gm(tmp_path, 40000, mem_interval_s=5.0)
    t = time.time()
    slot = gm.acquire(3000)
    assert slot is not None and time.time() - t < 2.0
    assert calls["n"] == 2  # snapshot + one gate check, no second check
    slot.release()


def test_tight_fit_keeps_consecutive_checks(tmp_path):
    gm, calls = _gm(tmp_path, 3500, mem_interval_s=0.0)
    slot = gm.acquire(3000)
    assert slot is not None
    assert calls["n"] == 3  # snapshot + two consecutive gate checks
    slot.release()


def test_fastpath_disabled(tmp_path):
    gm, calls = _gm(tmp_path, 40000, mem_interval_s=0.0, mem_fastpath_factor=0)
    assert gm.acquire(3000) is not None
    assert calls["n"] == 3
