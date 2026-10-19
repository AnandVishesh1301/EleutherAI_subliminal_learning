"""Idle-GPU holds: which GPUs of a node have been empty long enough to hold.

A GPU is held only after every sample over ``idle_s`` showed it empty and not
ours. A busy sample, an unreadable one, a sampling gap or a closed gate (the
node's lease is not free, the stop file exists, the ledger is unreadable)
restarts its clock. Pure logic; ``gpu_hold_io`` samples, holds and releases.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

GATE_OPEN = "open"


@dataclass(frozen=True)
class HoldPolicy:
    """When a GPU counts as empty, how long it must stay so, and the hold size.

    Attributes:
        idle_s: Continuous empty time before a hold starts.
        max_gap_s: Longer gap between two samples restarts every clock.
        mem_max_mib: Most memory an empty GPU may show.
        util_max_pct: Highest utilization an empty GPU may show.
        app_min_mib: A compute process at or above this makes the GPU busy.
        hold_gib: VRAM each hold allocates.
        verify_s: A hold not showing ``hold_gib`` by then is released.
    """

    idle_s: float = 900.0
    max_gap_s: float = 90.0
    mem_max_mib: int = 1024
    util_max_pct: int = 5
    app_min_mib: int = 64
    hold_gib: float = 44.0
    verify_s: float = 180.0

    @property
    def hold_mib(self) -> int:
        """Memory a verified hold shows at least."""
        return int(self.hold_gib * 1024)


@dataclass(frozen=True)
class GpuSample:
    """One ``nvidia-smi`` reading of one GPU.

    Attributes:
        index: ``nvidia-smi`` index.
        memory_used_mib: Used memory.
        util_pct: Utilization.
        app_mib: Used memory of each compute process on the GPU.
    """

    index: int
    memory_used_mib: int
    util_pct: int
    app_mib: tuple[int, ...] = ()


@dataclass(frozen=True)
class IdleClock:
    """Empty-time tracking of one GPU (monotonic seconds).

    Attributes:
        since: First sample of the current empty run (None = not empty).
        last: Time of the last sample.
    """

    since: float | None = None
    last: float | None = None


def parse_gpu_samples(gpu_csv: str, apps_csv: str) -> dict[int, GpuSample]:
    """GPU samples from two ``nvidia-smi --format=csv,noheader,nounits`` outputs.

    Args:
        gpu_csv: ``--query-gpu=index,uuid,memory.used,utilization.gpu`` output.
        apps_csv: ``--query-compute-apps=gpu_uuid,used_memory`` output.

    Raises:
        ValueError: A line does not parse (``[N/A]`` included) or a process
            names an unknown GPU.
    """
    by_uuid: dict[str, tuple[int, int, int]] = {}
    for line in gpu_csv.strip().splitlines():
        index, uuid, mem, util = (f.strip() for f in line.split(","))
        by_uuid[uuid] = (int(index), int(mem), int(util))
    if not by_uuid:
        raise ValueError("nvidia-smi listed no GPU")
    apps: dict[str, list[int]] = {u: [] for u in by_uuid}
    for line in apps_csv.strip().splitlines():
        if line.strip() == "No running processes found":
            continue
        uuid, mem = (f.strip() for f in line.split(","))
        if uuid not in apps:
            raise ValueError(f"compute process on unknown GPU {uuid}")
        apps[uuid].append(int(mem))
    return {
        i: GpuSample(i, mem, util, tuple(apps[u]))
        for u, (i, mem, util) in by_uuid.items()
    }


def parse_cvd(value: str, n_gpus: int) -> frozenset[int]:
    """GPU indices a ``CUDA_VISIBLE_DEVICES`` value exposes.

    A token that is not an index (a UUID, ``MIG-``) counts as every GPU.
    """
    out: set[int] = set()
    for token in (t.strip() for t in value.split(",")):
        if not token:
            continue
        if not token.isdigit():
            return frozenset(range(n_gpus))
        out.add(int(token))
    return frozenset(out)


def is_empty(sample: GpuSample, policy: HoldPolicy) -> bool:
    """No process of any user is using the GPU."""
    return (
        sample.memory_used_mib <= policy.mem_max_mib
        and sample.util_pct <= policy.util_max_pct
        and all(m < policy.app_min_mib for m in sample.app_mib)
    )


def tick(clock: IdleClock, empty: bool, now: float, policy: HoldPolicy) -> IdleClock:
    """Clock after one sample (``empty`` False also covers an unreadable sample)."""
    if not empty:
        return IdleClock(None, now)
    gap = clock.last is None or now - clock.last > policy.max_gap_s
    since = now if clock.since is None or gap else clock.since
    return IdleClock(since, now)


def idle_for(clock: IdleClock, now: float) -> float:
    """Seconds of the current empty run (0 when not empty)."""
    return 0.0 if clock.since is None else now - clock.since


def advance(
    clocks: Mapping[int, IdleClock],
    samples: Mapping[int, GpuSample] | None,
    ours: frozenset[int],
    gate: str,
    watch: Iterable[int],
    held: frozenset[int],
    now: float,
    policy: HoldPolicy,
) -> tuple[dict[int, IdleClock], tuple[int, ...], dict[int, str]]:
    """One pass over the watched GPUs.

    Args:
        clocks: Clocks from the previous pass.
        samples: This pass's samples (None = ``nvidia-smi`` failed).
        ours: GPUs our own processes expose in ``CUDA_VISIBLE_DEVICES``.
        gate: ``GATE_OPEN``, or why no hold may start.
        watch: GPUs this daemon may hold.
        held: GPUs with a live hold.
        now: Monotonic time.
        policy: Thresholds.

    Returns:
        New clocks, GPUs ready to hold, and a status word per watched GPU.
    """
    new: dict[int, IdleClock] = {}
    ready: list[int] = []
    status: dict[int, str] = {}
    for g in sorted(set(watch)):
        if g in held:
            status[g] = "held"
            continue
        sample = None if samples is None else samples.get(g)
        if sample is None:
            new[g], status[g] = IdleClock(None, now), "unknown"
            continue
        if g in ours:
            new[g], status[g] = IdleClock(None, now), "ours"
            continue
        if not is_empty(sample, policy):
            new[g], status[g] = IdleClock(None, now), "busy"
            continue
        if gate != GATE_OPEN:
            new[g], status[g] = IdleClock(None, now), f"empty, {gate}"
            continue
        new[g] = tick(clocks.get(g, IdleClock()), True, now, policy)
        idle = idle_for(new[g], now)
        status[g] = f"idle {idle / 60:.1f} min"
        if idle >= policy.idle_s:
            ready.append(g)
    return new, tuple(ready), status
