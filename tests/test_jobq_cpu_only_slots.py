"""`slots: 0` — a fully CPU-bound job must not consume a GPU slot-unit.

Before this, every job took at least one unit (`need = max(1, ...)`), so a `--device cpu`
analysis job held a GPU slot and blocked a GPU job from running: on 2026-08-09 eight
CPU-only probe jobs occupied a third of a 24-unit pool while the cards sat idle for them.

Opt-in only. Nothing infers CPU-ness from the command line, because a job that touched CUDA
while claiming no slot would silently oversubscribe the card.
"""

from __future__ import annotations

import json

import pytest

from subliminal_learning.jobqueue.gpu import GpuManager


@pytest.fixture
def mgr(tmp_path):
    (tmp_path / "gpu_policy.testhost.json").write_text(
        json.dumps({"gpus": [0, 1], "cap_per_gpu": 2, "free_mem_mib": 0})
    )
    return GpuManager(root=tmp_path, hostname="testhost", lock_prefix=str(tmp_path / "lk"))


def test_cpu_only_job_reserves_nothing(mgr):
    slot = mgr.acquire(0, slots=0)
    assert slot is not None
    assert slot.gpu is None and slot.is_cpu_only
    slot.release()  # idempotent, holds no fds


def test_cpu_only_jobs_do_not_exhaust_the_pool(mgr):
    """2 GPUs x cap 2 = 4 units. Many CPU jobs must still leave all 4 for GPU work."""
    cpu_slots = [mgr.acquire(0, slots=0) for _ in range(10)]
    assert all(s is not None and s.is_cpu_only for s in cpu_slots)
    gpu_slots = [mgr.acquire(0, slots=1) for _ in range(4)]
    assert all(s is not None and s.gpu is not None for s in gpu_slots), (
        "CPU-only jobs must not have consumed any GPU capacity"
    )
    for s in gpu_slots + cpu_slots:
        s.release()


def test_gpu_jobs_still_bounded_by_capacity(mgr):
    """The change must not loosen real GPU accounting."""
    held = [mgr.acquire(0, slots=1) for _ in range(4)]
    assert all(s is not None for s in held)
    assert mgr.acquire(0, slots=1) is None, "5th GPU job must wait"
    # ...but a CPU-only job still runs while the GPUs are full
    cpu = mgr.acquire(0, slots=0)
    assert cpu is not None and cpu.is_cpu_only
    for s in held:
        s.release()


def test_weighted_gpu_jobs_unchanged(mgr):
    """slots>1 still takes several units, all-or-nothing on one GPU."""
    heavy = mgr.acquire(0, slots=2)
    assert heavy is not None and heavy.gpu is not None
    other = mgr.acquire(0, slots=2)
    assert other is not None and other.gpu != heavy.gpu, "should land on the other GPU"
    assert mgr.acquire(0, slots=1) is None, "both GPUs now full"
    heavy.release()
    other.release()
