"""Priority-ordered parking in ``_Pool._acquire_gpu`` (2026-09-13).

Regression for the direction-surgery starvation: under full load every parked waiter
polled a freed slot on equal terms, so dozens of parked uncapped seed2seed waiters
(prio 45) took each freed slot within one poll while the capped surgery queue (prio 50)
only ever got its single non-blocking try -- 25 jobs in 20 h at 24 min each. Now a
capped job parks while its stage group has room, and every parked waiter defers to a
strictly higher-priority eligible waiter (bounded by ``PARK_DEFER_MAX_S``).
"""

from __future__ import annotations

import threading
import time

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue import worker as W
from subliminal_learning.jobqueue.gpu import GpuSlot
from subliminal_learning.jobqueue.worker import _Pool, _YIELD


class _RecLog:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def log(self, verb, msg=""):
        self.lines.append((verb, msg))


class _CountedGpu:
    """``free`` global slot-units handed out atomically; the group probe is a flag."""

    def __init__(self, free=0, room=True):
        self.free = free
        self.room = room
        self.grants: list[str | None] = []  # cap_group of each successful acquire, in order
        self._lock = threading.Lock()

    def acquire(self, mem_mib, *, slots=1, cap_per_gpu=None, cap_group=None):  # noqa: ARG002
        with self._lock:
            if self.free <= 0:
                return None
            self.free -= 1
            self.grants.append(cap_group)
        return GpuSlot(gpu=0, slot=0)

    def group_has_room(self, group, cap):  # noqa: ARG002
        return self.room

    def env(self):
        return {}

    def default_mem_mib(self):
        return 1000


def _pool(tmp_path, gpu):
    return _Pool(
        tmp_path, "testhost", gpu, _RecLog(),
        queues_filter=None, poll_s=0.01, gpu_wait_s=0.01,
    )


def _park(pool, key, priority, cap=None, group=None):
    """Start a thread parked in _acquire_gpu; returns (thread, result-cell)."""
    out: list = []

    def run():
        out.append(pool._acquire_gpu(100, 1, cap, group, key, priority=priority))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def _settle():
    time.sleep(0.1)  # ~10 polls at gpu_wait_s=0.01


def test_freed_slot_goes_to_the_higher_priority_parked_waiter(tmp_path):
    gpu = _CountedGpu(free=0)
    pool = _pool(tmp_path, gpu)
    t_pig, r_pig = _park(pool, "pig", 45)
    _settle()  # the low waiter is parked first, like the live pools
    t_surg, r_surg = _park(pool, "surgery", 50, cap=2, group="surg")
    _settle()
    assert r_pig == [] and r_surg == []  # both parked, nothing free

    gpu.free = 1  # one slot frees
    t_surg.join(2)
    assert not t_surg.is_alive() and isinstance(r_surg[0], GpuSlot)
    _settle()
    assert r_pig == []  # the pig waiter never even tried while surgery was ahead
    assert gpu.grants == ["surg"]

    gpu.free = 1  # next slot: nobody higher is parked any more
    t_pig.join(2)
    assert not t_pig.is_alive() and isinstance(r_pig[0], GpuSlot)
    assert gpu.grants == ["surg", None]
    assert pool._parked == {}


def test_capped_waiter_yields_once_its_group_saturates(tmp_path):
    gpu = _CountedGpu(free=0, room=True)
    pool = _pool(tmp_path, gpu)
    t, r = _park(pool, "surgery", 50, cap=2, group="surg")
    _settle()
    assert r == []  # parked: group has room, no global slot
    gpu.room = False  # siblings filled the group
    t.join(2)
    assert r == [_YIELD]
    assert pool._parked == {}


def test_lower_waiter_does_not_defer_to_a_saturated_capped_waiter(tmp_path):
    """A capped waiter whose group is full is not 'ahead' of anyone (it is about to yield)."""
    gpu = _CountedGpu(free=0, room=True)
    pool = _pool(tmp_path, gpu)
    t_pig, r_pig = _park(pool, "pig", 45)
    t_surg, r_surg = _park(pool, "surgery", 50, cap=2, group="surg")
    _settle()
    gpu.room = False  # group saturates; surgery yields
    t_surg.join(2)
    assert r_surg == [_YIELD]
    gpu.free = 1
    t_pig.join(2)
    assert isinstance(r_pig[0], GpuSlot)


def test_equal_priority_waiters_do_not_defer_to_each_other(tmp_path):
    gpu = _CountedGpu(free=0)
    pool = _pool(tmp_path, gpu)
    ta, ra = _park(pool, "a", 45)
    tb, rb = _park(pool, "b", 45)
    _settle()
    gpu.free = 2
    ta.join(2)
    tb.join(2)
    assert isinstance(ra[0], GpuSlot) and isinstance(rb[0], GpuSlot)


def test_deferral_is_bounded_by_the_safety_valve(tmp_path, monkeypatch):
    """A head-of-line request nothing can satisfy delays, never wedges, lower waiters."""
    monkeypatch.setattr(W, "PARK_DEFER_MAX_S", 0.05)

    class _Gpu(_CountedGpu):
        def acquire(self, mem_mib, *, slots=1, cap_per_gpu=None, cap_group=None):
            if cap_group == "huge":
                return None  # the high waiter can never be satisfied
            return super().acquire(mem_mib, slots=slots, cap_per_gpu=cap_per_gpu, cap_group=cap_group)

    gpu = _Gpu(free=1)
    pool = _pool(tmp_path, gpu)
    t_hi, r_hi = _park(pool, "huge", 50, cap=1, group="huge")
    _settle()
    t_lo, r_lo = _park(pool, "pig", 45)
    t_lo.join(2)
    assert isinstance(r_lo[0], GpuSlot)
    assert any("deferred to a higher-priority waiter" in m for _, m in pool.master.lines)
    store.stop_path(tmp_path, "testhost").parent.mkdir(parents=True, exist_ok=True)
    store.stop_path(tmp_path, "testhost").touch()
    t_hi.join(2)
    assert r_hi == [None]


def test_stop_file_releases_a_deferring_waiter(tmp_path):
    gpu = _CountedGpu(free=0)
    pool = _pool(tmp_path, gpu)
    _park(pool, "surgery", 50)
    t_lo, r_lo = _park(pool, "pig", 45)
    _settle()
    store.stop_path(tmp_path, "testhost").parent.mkdir(parents=True, exist_ok=True)
    store.stop_path(tmp_path, "testhost").touch()
    t_lo.join(2)
    assert r_lo == [None]
