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


# --- dynamic cap (cpu_dynamic): the ceiling follows the cores the GPU jobs leave over ---


def _dyn_mgr(tmp_path, *, gpus=(0, 1), cap_per_gpu=2, **cpu):
    d = {
        "gpus": list(gpus), "cap_per_gpu": cap_per_gpu, "free_mem_mib": 0,
        "cpu_dynamic": True, "cpu_total": 8, "cpu_reserve": 2,
        "cpu_per_gpu_job": 1, "cpu_per_cpu_job": 1,
    }
    d.update(cpu)
    (tmp_path / "gpu_policy.testhost.json").write_text(json.dumps(d))
    return GpuManager(root=tmp_path, hostname="testhost", lock_prefix=str(tmp_path / "lk"))


@pytest.mark.parametrize("gpu_held,expected", [(0, 6), (3, 3), (4, 2)])
def test_dynamic_cap_follows_held_gpu_slots(tmp_path, gpu_held, expected):
    """8 cores - 2 reserved - 1 per held GPU slot, one core per CPU job."""
    mgr = _dyn_mgr(tmp_path)
    gpu = [mgr.acquire(0, slots=1) for _ in range(gpu_held)]
    assert all(s is not None for s in gpu)
    cpu = [mgr.acquire(0, slots=0) for _ in range(expected)]
    assert all(s is not None for s in cpu), "cores left over must all be admissible"
    assert mgr.acquire(0, slots=0) is None, "one past the derived cap must wait"
    for s in gpu + cpu:
        s.release()


def test_dynamic_cap_of_zero_admits_nothing(tmp_path):
    """Every core spoken for by GPU jobs -> the CPU lane closes (no floor of 1)."""
    mgr = _dyn_mgr(tmp_path, cpu_total=4)
    gpu = [mgr.acquire(0, slots=1) for _ in range(4)]
    assert all(s is not None for s in gpu)
    assert mgr.acquire(0, slots=0) is None
    assert mgr.cpu_has_room() is False
    for s in gpu:
        s.release()


def test_dynamic_shrink_does_not_preempt(tmp_path):
    """A shrinking cap blocks new admissions; running CPU jobs keep their slots."""
    mgr = _dyn_mgr(tmp_path, cpu_total=7, cpu_per_gpu_job=2)
    g1 = mgr.acquire(0, slots=1)  # 5 free cores - 2 -> cap 3
    held = [mgr.acquire(0, slots=0) for _ in range(3)]
    assert all(s is not None for s in held)
    g2 = mgr.acquire(0, slots=1)  # 5 - 4 -> cap 1, but the 3 held slots stay
    assert g1 is not None and g2 is not None
    assert mgr.acquire(0, slots=0) is None
    for s in held:
        s.release()
    again = [mgr.acquire(0, slots=0) for _ in range(2)]
    assert again[0] is not None and again[1] is None, "only the shrunk cap is re-usable"
    again[0].release()
    g1.release()
    g2.release()


def test_cpu_cap_is_a_ceiling_in_dynamic_mode(tmp_path):
    mgr = _dyn_mgr(tmp_path, cpu_cap=2)
    cpu = [mgr.acquire(0, slots=0) for _ in range(2)]
    assert all(s is not None for s in cpu)
    assert mgr.acquire(0, slots=0) is None, "explicit cpu_cap still caps the derived value"
    for s in cpu:
        s.release()


def test_dynamic_mode_hot_reloads(tmp_path):
    """The policy is re-read every acquire: flipping cpu_dynamic needs no restart."""
    mgr = _mgr(tmp_path, cpu_cap=1)  # static: one slot
    a = mgr.acquire(0, slots=0)
    assert a is not None and mgr.acquire(0, slots=0) is None
    (tmp_path / "gpu_policy.testhost.json").write_text(json.dumps({
        "gpus": [0, 1], "cap_per_gpu": 2, "free_mem_mib": 0, "cpu_cap": 1,
        "cpu_dynamic": True, "cpu_total": 8, "cpu_reserve": 2,
        "cpu_per_gpu_job": 1, "cpu_per_cpu_job": 1,
    }))
    assert mgr.acquire(0, slots=0) is None, "cpu_cap=1 is still the ceiling"
    (tmp_path / "gpu_policy.testhost.json").write_text(json.dumps({
        "gpus": [0, 1], "cap_per_gpu": 2, "free_mem_mib": 0,
        "cpu_dynamic": True, "cpu_total": 8, "cpu_reserve": 2,
        "cpu_per_gpu_job": 1, "cpu_per_cpu_job": 1,
    }))
    b = mgr.acquire(0, slots=0)
    assert b is not None, "no explicit cpu_cap -> the derived ceiling (6) applies"
    a.release()
    b.release()


def test_zero_cores_per_cpu_job_is_rejected():
    with pytest.raises(ValueError, match="cpu_per_cpu_job"):
        GpuPolicy.from_dict({
            "gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 0, "cpu_per_cpu_job": 0,
        })


def test_old_policy_files_are_static_with_the_old_default():
    """Backwards compatibility: no new key -> dynamic mode off, cpu_cap rule unchanged."""
    import os
    p = GpuPolicy.from_dict({"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 0})
    assert p.cpu_dynamic is False
    assert p.cpu_cap == max(1, (os.cpu_count() or 4) // 4)
