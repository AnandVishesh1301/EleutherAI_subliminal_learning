"""``reserve_mem_mib``: MiB every GPU must still show free AFTER an admission (opt-in).

GPU-free: the memory query is injected, so these exercise parsing, the admission gate, the
fast path and the OOM-escalation ceiling without touching nvidia-smi.
"""
from __future__ import annotations

import pytest

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuManager, GpuPolicy, clear_total_mem_cache
from tests.test_jobqueue import _write_policy


def _gm(tmp_path, free, reserve, **kw):
    """A one-GPU manager whose card reports ``free`` MiB, under a ``reserve`` policy."""
    host = store.this_host()
    _write_policy(
        tmp_path, host, gpus=[0], cap_per_gpu=4, free_mem_mib=0, reserve_mem_mib=reserve
    )
    calls = {"n": 0}

    def fake_mem():
        calls["n"] += 1
        return {0: free}

    gm = GpuManager(
        tmp_path, host, lock_prefix=str(tmp_path / "lock"), mem_query=fake_mem, **kw
    )
    return gm, calls


# ------------------------- parsing -------------------------


def test_from_dict_parses_the_key_and_defaults_to_zero():
    base = {"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 1000}
    assert GpuPolicy.from_dict(base).reserve_mem_mib == 0
    assert GpuPolicy.from_dict({**base, "reserve_mem_mib": 20000}).reserve_mem_mib == 20000


def test_negative_reserve_is_a_config_error():
    base = {"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 1000, "reserve_mem_mib": -1}
    with pytest.raises(ValueError, match="reserve_mem_mib"):
        GpuPolicy.from_dict(base)


# ------------------------- admission -------------------------


def test_reserve_refuses_a_job_that_would_eat_into_it(tmp_path):
    gm, _ = _gm(tmp_path, free=30000, reserve=20000, mem_interval_s=0.0)
    assert gm.acquire(16000) is None  # 30000 - 16000 = 14000 < 20000


def test_reserve_admits_once_the_card_can_spare_it(tmp_path):
    gm, _ = _gm(tmp_path, free=36000, reserve=20000, mem_interval_s=0.0)
    slot = gm.acquire(16000)
    assert slot is not None
    slot.release()


def test_multi_unit_job_is_charged_the_reserve_once(tmp_path):
    """``slots: 2`` costs the ask plus ONE reserve, not one reserve per unit."""
    gm, _ = _gm(tmp_path, free=36000, reserve=20000, mem_interval_s=0.0)
    slot = gm.acquire(16000, slots=2)
    assert slot is not None
    slot.release()


def test_negative_headroom_during_rampup_just_refuses(tmp_path):
    gm, _ = _gm(tmp_path, free=1000, reserve=20000, mem_interval_s=0.0)
    assert gm.acquire(16000) is None


# ------------------------- fast path -------------------------


def test_fastpath_does_not_bypass_the_reserve(tmp_path):
    """Free memory that looks ample against the ask alone is still refused."""
    gm, _ = _gm(tmp_path, free=30000, reserve=20000, mem_interval_s=0.0)
    assert gm._mem_ok(0, 12000, 0, reserve=20000) is False  # 30000 >= 2x12000 without it
    assert gm._mem_ok(0, 12000, 0, reserve=0) is True


# ------------------------- reserve 0 == today -------------------------


@pytest.mark.parametrize("free,ask", [(30000, 16000), (3500, 3000), (40000, 3000)])
def test_reserve_zero_reproduces_the_old_decision(tmp_path, free, ask):
    gm, _ = _gm(tmp_path, free=free, reserve=0, mem_interval_s=0.0)
    assert gm._mem_ok(0, ask, 0) == (free >= ask)


# ------------------------- escalation ceiling -------------------------


def test_attainable_ceiling_subtracts_the_reserve(tmp_path):
    clear_total_mem_cache()  # total_mem is process-cached
    host = store.this_host()
    _write_policy(
        tmp_path, host, gpus=[0], cap_per_gpu=1, free_mem_mib=0, reserve_mem_mib=20000
    )
    gm = GpuManager(
        tmp_path,
        host,
        lock_prefix=str(tmp_path / "lock"),
        mem_query=lambda: {0: 40000},
        total_query=lambda: {0: 40000},
    )
    assert gm.mem_ceiling_mib() == 20000


def test_ceiling_unchanged_without_a_reserve(tmp_path):
    clear_total_mem_cache()
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0], cap_per_gpu=1, free_mem_mib=0)
    gm = GpuManager(
        tmp_path,
        host,
        lock_prefix=str(tmp_path / "lock"),
        mem_query=lambda: {0: 40000},
        total_query=lambda: {0: 40000},
    )
    assert gm.mem_ceiling_mib() == 40000
