"""Per-node GPU slot locks + memory gating for the worker pool.

Two guards, both node-local and shared across ALL queues (combined per-GPU load is capped
globally — an explicit design choice, matching the old launchers):

- **Slot locks** — ``fcntl.flock(LOCK_EX|LOCK_NB)`` on ``<prefix>_gpu{g}_slot{s}.lock`` for
  ``s`` in ``range(cap_per_gpu)``. Holding one reserves a run slot on GPU ``g``.
- **Memory gate** — a GPU must show ``memory.free >= mem_mib`` on TWO consecutive
  ``nvidia-smi`` checks ~3s apart before we dispatch to it.

The acquire-check-dispatch section is serialized per node by one global flock
(``<prefix>_acquire.lock``) so two workers can't both pass the memory gate for the same
headroom at once. The GPU policy is RE-READ on every acquire (hot-reload: pin files change
mid-run today and we keep that property).

All ``nvidia-smi`` access is behind :func:`query_free_mem`, and the lock ``prefix`` +
``mem_query`` are constructor params, so tests run GPU-free with a tmp lock dir.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from subliminal_learning.jobqueue import yielding

DEFAULT_LOCK_PREFIX = "/tmp/subliminal_jobq"


@dataclass
class GpuPolicy:
    """Parsed ``gpu_policy.<hostname>.json``: which GPUs, how many slots, memory floor, env.

    The ``yield_*`` fields configure the opt-in GPU-yielding framework (see
    ``jobqueue/yielding.py``). They are all defaulted so old policy files keep working and
    the feature is OFF unless ``yield_to_foreign`` is explicitly set true.
    """

    gpus: list[int]
    cap_per_gpu: int
    free_mem_mib: int
    env: dict[str, str] = field(default_factory=dict)
    yield_to_foreign: bool = False
    yield_poll_s: float = 30.0
    yield_confirm_polls: int = 2
    yield_min_foreign_procs: int = 1
    yield_cooldown_s: float = 900.0
    yield_action: str = "kill"

    @classmethod
    def from_dict(cls, d: dict) -> "GpuPolicy":
        return cls(
            gpus=[int(g) for g in d["gpus"]],
            cap_per_gpu=int(d["cap_per_gpu"]),
            free_mem_mib=int(d["free_mem_mib"]),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            yield_to_foreign=bool(d.get("yield_to_foreign", False)),
            yield_poll_s=float(d.get("yield_poll_s", 30.0)),
            yield_confirm_polls=int(d.get("yield_confirm_polls", 2)),
            yield_min_foreign_procs=int(d.get("yield_min_foreign_procs", 1)),
            yield_cooldown_s=float(d.get("yield_cooldown_s", 900.0)),
            yield_action=str(d.get("yield_action", "kill")),
        )


def policy_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"gpu_policy.{hostname}.json"


def load_policy(root: Path, hostname: str) -> GpuPolicy:
    """Load this node's GPU policy, or raise with a clear how-to-fix message."""
    p = policy_path(root, hostname)
    if not p.exists():
        raise FileNotFoundError(
            f"No GPU policy for this node: {p} is missing. Create it, e.g.\n"
            f'  {{"gpus":[4,5,6,7],"cap_per_gpu":4,"free_mem_mib":13000,'
            f'"env":{{"OMP_NUM_THREADS":"2"}}}}'
        )
    return GpuPolicy.from_dict(json.loads(p.read_text()))


def query_free_mem() -> dict[int, int]:
    """GPU index -> free memory (MiB) via ``nvidia-smi`` (the only smi call; tests patch it)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    free: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx, mem = (x.strip() for x in line.split(","))
        free[int(idx)] = int(mem)
    return free


def _close_lock(fd: int | None) -> None:
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except OSError:
            pass


@dataclass
class GpuSlot:
    """A held reservation on physical GPU ``gpu``. ``release()`` drops the flocks (idempotent).

    ``_fds`` holds one lock fd per consumed slot-unit (a weighted job holds several).
    ``_group_fd`` is the additional per-cap-group slot lock when the queue declared its own
    ``cap_per_gpu`` (pipeline-stage ceiling on top of the global policy ceiling).
    """

    gpu: int
    slot: int
    _fds: list[int] = field(default_factory=list)
    _group_fd: int | None = None

    def release(self) -> None:
        for fd in self._fds:
            _close_lock(fd)
        self._fds = []
        _close_lock(self._group_fd)
        self._group_fd = None


class GpuManager:
    """Acquires (slot-lock + memory-gated) GPU slots for this node.

    ``acquire(mem_mib)`` returns a :class:`GpuSlot` for the first eligible GPU, or ``None``
    if none is currently free (the worker then WAITs and retries). Round-robin starts from a
    pid-derived offset so co-launched workers spread out without needing shared state.
    """

    def __init__(
        self,
        root: Path,
        hostname: str,
        *,
        lock_prefix: str = DEFAULT_LOCK_PREFIX,
        mem_query=query_free_mem,
        mem_checks: int = 2,
        mem_interval_s: float = 3.0,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.lock_prefix = lock_prefix
        self.mem_query = mem_query
        self.mem_checks = mem_checks
        self.mem_interval_s = mem_interval_s

    def _policy(self) -> GpuPolicy:  # re-read every time (hot-reload of pins)
        return load_policy(self.root, self.hostname)

    def env(self) -> dict[str, str]:
        return self._policy().env

    def default_mem_mib(self) -> int:
        return self._policy().free_mem_mib

    def _slot_lock_file(self, g: int, s: int) -> str:
        return f"{self.lock_prefix}_gpu{g}_slot{s}.lock"

    def _try_slot(self, g: int, s: int) -> int | None:
        fd = os.open(self._slot_lock_file(g, s), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return fd

    def _try_group_slot(self, group: str, g: int, cap: int) -> int | None:
        """Grab one of ``cap`` per-GPU slots in the cap-group's own lock namespace.

        Queues sharing a ``cap_group`` MUST declare the same ``cap_per_gpu`` — the ceiling
        is the slot-file range probed, so mixed values would give each queue a different
        effective ceiling in the same namespace.
        """
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", group)
        for s in range(cap):
            fd = os.open(
                f"{self.lock_prefix}_grp_{safe}_gpu{g}_slot{s}.lock",
                os.O_CREAT | os.O_RDWR,
                0o644,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            return fd
        return None

    def _mem_ok(self, g: int, mem_mib: int) -> bool:
        for i in range(self.mem_checks):
            if i:
                time.sleep(self.mem_interval_s)
            if self.mem_query().get(g, 0) < mem_mib:
                return False
        return True

    def _held_counts(self, policy: GpuPolicy) -> dict[int, int]:
        """Per-GPU count of slot locks held right now (probe each slot, non-destructive)."""
        counts: dict[int, int] = {}
        for g in policy.gpus:
            held = 0
            for s in range(policy.cap_per_gpu):
                fd = self._try_slot(g, s)
                if fd is None:
                    held += 1
                else:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        os.close(fd)
                    except OSError:
                        pass
            counts[g] = held
        return counts

    def acquire(
        self,
        mem_mib: int,
        *,
        slots: int = 1,
        cap_per_gpu: int | None = None,
        cap_group: str | None = None,
    ) -> GpuSlot | None:
        """Try to reserve ``slots`` slot-units with ``>= mem_mib`` free, or ``None``.

        Placement is least-loaded-first: GPUs are tried in ascending order of currently-held
        slots (breadth-first spread — a small batch lands 1/GPU), with a pid-derived rotation
        as the tie-break. Caps are ceilings, not fill targets.

        ``slots`` is the job's WEIGHT against the global per-GPU budget (`policy.cap_per_gpu`
        units): a heavy job may consume several units, all-or-nothing on one GPU, so e.g. with
        an 8-unit budget six weight-1 jobs + one weight-2 job fill the card exactly. A weight
        above the budget is clamped to it ("take the whole card") so the job can still run.

        ``cap_per_gpu``/``cap_group`` add a second, stage-level ceiling on top: the job must
        also win one of ``cap_per_gpu`` slots in the ``cap_group`` lock namespace on that GPU
        (queues sharing a ``cap_group`` co-cap, like the old shared ``LOCK_PREFIX`` launchers).
        The group ceiling counts JOBS, not units.
        """
        policy = self._policy()
        gpus = policy.gpus
        if not gpus:
            return None
        need = max(1, min(slots, policy.cap_per_gpu))
        acq_fd = os.open(
            f"{self.lock_prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644
        )
        fcntl.flock(acq_fd, fcntl.LOCK_EX)  # serialize acquire-check-dispatch per node
        try:
            offset = os.getpid() % len(gpus)
            rotated = gpus[offset:] + gpus[:offset]
            counts = self._held_counts(policy)
            order = sorted(rotated, key=lambda g: counts[g])  # stable: rotation breaks ties
            # GPUs yielded to a foreign user are off-limits — skip them before the memory
            # snapshot (opt-in; re-read here so a flip/reclaim takes effect on the next
            # acquire). Empty set when the feature is off or the marker file is missing.
            yielded = yielding.yielded_gpus(self.root, self.hostname)
            # One snapshot up front: GPUs that are externally memory-occupied are skipped
            # without slot probing or gate sleeps (bounds time under the acquire lock).
            snapshot = self.mem_query()
            for g in order:
                if g in yielded:
                    continue
                if snapshot.get(g, 0) < mem_mib:
                    continue
                held: list[tuple[int, int]] = []  # (slot index, fd)
                for s in range(policy.cap_per_gpu):
                    fd = self._try_slot(g, s)
                    if fd is not None:
                        held.append((s, fd))
                        if len(held) == need:
                            break
                if len(held) < need:  # not enough free units -> all-or-nothing rollback
                    for _, fd in held:
                        _close_lock(fd)
                    continue
                group_fd = None
                if cap_per_gpu is not None:
                    group_fd = self._try_group_slot(cap_group or "default", g, cap_per_gpu)
                    if group_fd is None:  # stage ceiling reached on this GPU
                        for _, fd in held:
                            _close_lock(fd)
                        continue
                if self._mem_ok(g, mem_mib):
                    return GpuSlot(
                        gpu=g,
                        slot=held[0][0],
                        _fds=[fd for _, fd in held],
                        _group_fd=group_fd,
                    )
                for _, fd in held:  # not enough memory -> next GPU
                    _close_lock(fd)
                _close_lock(group_fd)
            return None
        finally:
            _close_lock(acq_fd)

    def occupancy(self) -> dict[int, int]:
        """Best-effort per-GPU count of slots held right now (probes each slot lock).

        A slot is "held" if we can't take its LOCK_NB (some worker on this node owns it).
        Missing policy -> empty map (status shouldn't crash without a policy).
        """
        try:
            policy = self._policy()
        except FileNotFoundError:
            return {}
        return self._held_counts(policy)
