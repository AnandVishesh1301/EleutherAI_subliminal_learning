"""`slots: 0` — CPU-bound jobs get their own budget, independent of GPU capacity.

Two failure modes, in tension, and the tests pin both:

1. A `--device cpu` job used to consume a GPU slot-unit (`need = max(1, ...)`), so eight
   CPU-only probes held a third of a 24-unit pool while the cards idled for them.
2. Exempting CPU jobs from accounting entirely would have removed their only ceiling --
   the GPU slots were doubling as the pool's admission control, so N worker threads could
   all run CPU jobs at once and thrash the box.

Hence two independent budgets: `cap_per_gpu` per GPU, and a node-wide `cpu_cap`.
"""

from __future__ import annotations

import json

import pytest

from subliminal_learning.jobqueue.gpu import GpuManager, GpuPolicy


def _mgr(tmp_path, *, gpus=(0, 1), cap_per_gpu=2, cpu_cap=3):
    (tmp_path / "gpu_policy.testhost.json").write_text(
        json.dumps({
            "gpus": list(gpus), "cap_per_gpu": cap_per_gpu,
            "free_mem_mib": 0, "cpu_cap": cpu_cap,
        })
    )
    return GpuManager(root=tmp_path, hostname="testhost", lock_prefix=str(tmp_path / "lk"))


@pytest.fixture
def mgr(tmp_path):
    return _mgr(tmp_path)


def test_cpu_job_takes_a_cpu_slot_not_a_gpu(mgr):
    slot = mgr.acquire(0, slots=0)
    assert slot is not None
    assert slot.gpu is None and slot.is_cpu_only
    assert slot._fds, "must hold a real CPU-budget lock, not nothing"
    slot.release()


def test_cpu_jobs_never_consume_gpu_capacity(mgr):
    """The original bug: 2 GPUs x cap 2 = 4 units must all survive a burst of CPU jobs."""
    cpu = [mgr.acquire(0, slots=0) for _ in range(3)]
    assert all(s is not None for s in cpu)
    gpu = [mgr.acquire(0, slots=1) for _ in range(4)]
    assert all(s is not None and s.gpu is not None for s in gpu), (
        "CPU jobs must not have eaten GPU capacity"
    )
    for s in cpu + gpu:
        s.release()


def test_cpu_budget_is_enforced(mgr):
    """The second failure mode: CPU jobs are capped, not unlimited."""
    held = [mgr.acquire(0, slots=0) for _ in range(3)]  # cpu_cap=3
    assert all(s is not None for s in held)
    assert mgr.acquire(0, slots=0) is None, "4th CPU job must wait"
    held[0].release()
    again = mgr.acquire(0, slots=0)
    assert again is not None, "a freed CPU slot must be reusable"
    for s in held[1:] + [again]:
        s.release()


def test_cpu_job_runs_even_when_every_gpu_is_full(mgr):
    """The budgets are independent in both directions."""
    gpu = [mgr.acquire(0, slots=1) for _ in range(4)]
    assert all(s is not None for s in gpu)
    assert mgr.acquire(0, slots=1) is None, "GPUs are full"
    cpu = mgr.acquire(0, slots=0)
    assert cpu is not None and cpu.is_cpu_only, "CPU work must not be blocked by full GPUs"
    for s in gpu + [cpu]:
        s.release()


def test_gpu_accounting_unchanged(mgr):
    """Weighted GPU jobs still take several units, all-or-nothing on one card."""
    heavy = mgr.acquire(0, slots=2)
    other = mgr.acquire(0, slots=2)
    assert heavy is not None and other is not None
    assert heavy.gpu != other.gpu
    assert mgr.acquire(0, slots=1) is None
    heavy.release()
    other.release()


def test_cpu_cap_defaults_from_core_count_when_absent():
    """Old policy files without cpu_cap keep working, with a conservative default."""
    p = GpuPolicy.from_dict({"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 0})
    import os
    assert p.cpu_cap == max(1, (os.cpu_count() or 4) // 4)
    assert p.cpu_cap >= 1
