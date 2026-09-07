"""Per-node GPU slot locks + memory gating for the worker pool.

Three guards, all node-local and shared across ALL queues (combined per-GPU load is capped
globally — an explicit design choice, matching the old launchers):

- **Slot locks** — ``fcntl.flock(LOCK_EX|LOCK_NB)`` on ``<prefix>_gpu{g}_slot{s}.lock`` for
  ``s`` in ``range(cap_per_gpu)``. Holding one reserves a run slot on GPU ``g``.
- **CPU slot locks** — the same mechanism on ``<prefix>_cpu_slot{s}.lock`` for ``s`` in
  ``range(cpu_cap)``, an INDEPENDENT budget for jobs declaring ``slots: 0``. GPU capacity
  and CPU capacity are separate concerns: CPU work must not block a card, but it still
  needs its own ceiling, since the GPU slots were otherwise the pool's only admission
  control (N worker threads would all run CPU jobs at once and thrash the box).
- **Memory gate** — a GPU must show ``memory.free >= mem_mib`` on TWO consecutive
  ``nvidia-smi`` checks ~3s apart before we dispatch to it.
- **Reservation ledger** — each held slot lock carries the ``mem_mib`` its job asked for
  and the time it was granted. Free memory reported by ``nvidia-smi`` lags a freshly
  dispatched job by the minutes it spends importing torch and loading weights, so the gate
  subtracts every reservation younger than ``reservation_warmup_s`` before comparing. This
  closes the ramp-up race that otherwise lets several workers admit themselves against the
  same not-yet-allocated headroom.

The acquire-check-dispatch section is serialized per node by one global flock
(``<prefix>_acquire.lock``) so two workers can't both pass the memory gate for the same
headroom at once. The GPU policy is RE-READ on every acquire (hot-reload: pin files change
mid-run today and we keep that property).

All ``nvidia-smi`` access is behind :func:`query_free_mem`, and the lock ``prefix`` +
``mem_query`` are constructor params, so tests run GPU-free with a tmp lock dir.
"""

from __future__ import annotations

import contextlib
import fcntl
import getpass
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from subliminal_learning.jobqueue import yielding

# Per-user: these flock files live in a world-readable /tmp and are owned by whoever
# creates them, so a shared prefix makes one account's locks unopenable by the other.
DEFAULT_LOCK_PREFIX = f"/tmp/subliminal_jobq_{getpass.getuser()}"
DEFAULT_RESERVATION_WARMUP_S = 300

# The three slot-lock filename shapes, as suffixes after ``lock_prefix`` (see
# _slot_lock_file / _cpu_slot_lock_file / _try_group_slot). reap_leaked_locks matches
# open fds against this WHITELIST rather than "anything under the prefix", so a broad or
# root-overlapping custom SL_JOBQ_LOCK_PREFIX (or a future non-lock file added under the
# prefix) can never sweep unrelated descriptors into the audit.
_SLOT_LOCK_SUFFIX_RE = re.compile(
    r"_(?:gpu\d+_slot\d+|cpu_slot\d+|grp_[A-Za-z0-9._-]+_gpu\d+_slot\d+)\.lock$"
)

# How long reap_leaked_locks retries for the acquire flock before skipping its tick.
# Bounded so a wedged nvidia-smi (no subprocess timeout) holding the flock degrades to
# skipped audits instead of a hung supervisor; long enough that ordinary acquire churn
# (gate sleeps of a few seconds) cannot starve the auditor.
_AUDIT_FLOCK_DEADLINE_S = 10.0


@dataclass
class GpuPolicy:
    """Parsed ``gpu_policy.<hostname>.json``: which GPUs, how many slots, memory floor, env.

    Attributes:
        gpus: Physical GPU indices the pool may use.
        cap_per_gpu: Concurrent slot-units per GPU.
        free_mem_mib: Free memory a GPU must show (net of live reservations) per slot-unit.
        cpu_cap: Concurrent CPU-only jobs (``slots: 0``) admitted node-wide. A SEPARATE
            budget from ``cap_per_gpu``: CPU work must not consume GPU capacity, but it
            still needs a ceiling of its own, since the GPU slots were doubling as the
            pool's only admission control. Defaults to a quarter of the CPU count so a
            burst of CPU jobs cannot starve the GPU jobs' data loaders.
        reservation_warmup_s: How long a granted slot's memory is counted from the ledger
            rather than from ``nvidia-smi`` — cover the job's model-load ramp.
        env: Extra environment for every job dispatched by this node.

    The ``yield_*`` fields configure the opt-in GPU-yielding framework (see
    ``jobqueue/yielding.py``). They are all defaulted so old policy files keep working and
    the feature is OFF unless ``yield_to_foreign`` is explicitly set true.
    """

    gpus: list[int]
    cap_per_gpu: int
    free_mem_mib: int
    cpu_cap: int = 0  # 0 -> derived from os.cpu_count() at load
    reservation_warmup_s: int = DEFAULT_RESERVATION_WARMUP_S
    env: dict[str, str] = field(default_factory=dict)
    yield_to_foreign: bool = False
    yield_poll_s: float = 30.0
    yield_confirm_polls: int = 2
    yield_min_foreign_procs: int = 1
    yield_cooldown_s: float = 900.0
    yield_action: str = "kill"
    # Only consulted for yield_action == "drain_if_near_done": a live job whose estimated
    # progress is >= yield_drain_threshold is spared (runs to completion instead of being
    # killed+requeued); yield_drain_max_s caps how long a spared job may keep a yielded card;
    # yield_progress_regex (two numeric groups = current/total) opts into log-based progress.
    yield_drain_threshold: float = 0.8
    yield_drain_max_s: float = 1800.0
    yield_progress_regex: str | None = None
    # Local (same-namespace) GPU procs owned by these uids count as foreign for yielding.
    # Policy JSON accepts ``yield_to_users`` (names) and/or ``yield_to_uids`` (ints); names
    # are resolved via ``pwd`` at load and silently skipped if unknown on this host.
    yield_to_uids: tuple[int, ...] = ()
    # CUDA MPS (see ensure_mps_daemon / mps_env), ON by default — a node opts OUT with
    # ``"mps": false``. Safe as a default because ensure_mps_daemon degrades gracefully:
    # a missing binary or failed probe warns and runs without MPS (no env injection).
    # ``mps_pipe_dir``/``mps_log_dir`` default to per-user /tmp paths when None.
    mps: bool = True
    mps_pipe_dir: str | None = None
    mps_log_dir: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "GpuPolicy":
        return cls(
            gpus=[int(g) for g in d["gpus"]],
            cap_per_gpu=int(d["cap_per_gpu"]),
            free_mem_mib=int(d["free_mem_mib"]),
            cpu_cap=int(d.get("cpu_cap", 0)) or max(1, (os.cpu_count() or 4) // 4),
            reservation_warmup_s=int(
                d.get("reservation_warmup_s", DEFAULT_RESERVATION_WARMUP_S)
            ),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            yield_to_foreign=bool(d.get("yield_to_foreign", False)),
            yield_poll_s=float(d.get("yield_poll_s", 30.0)),
            yield_confirm_polls=int(d.get("yield_confirm_polls", 2)),
            yield_min_foreign_procs=int(d.get("yield_min_foreign_procs", 1)),
            yield_cooldown_s=float(d.get("yield_cooldown_s", 900.0)),
            yield_action=str(d.get("yield_action", "kill")),
            yield_drain_threshold=float(d.get("yield_drain_threshold", 0.8)),
            yield_drain_max_s=float(d.get("yield_drain_max_s", 1800.0)),
            yield_progress_regex=(d.get("yield_progress_regex") or None),
            yield_to_uids=_resolve_yield_uids(d),
            mps=bool(d.get("mps", True)),
            mps_pipe_dir=(d.get("mps_pipe_dir") or None),
            mps_log_dir=(d.get("mps_log_dir") or None),
        )


def _resolve_yield_uids(d: dict) -> tuple[int, ...]:
    """``yield_to_uids`` (ints) ∪ ``yield_to_users`` (names via pwd; unknown names skipped)."""
    import pwd

    uids: set[int] = {int(u) for u in (d.get("yield_to_uids") or [])}
    for name in d.get("yield_to_users") or []:
        try:
            uids.add(pwd.getpwnam(str(name)).pw_uid)
        except KeyError:
            continue
    return tuple(sorted(uids))


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


def _mps_dirs(policy: GpuPolicy) -> tuple[str, str]:
    """Resolve (pipe_dir, log_dir), defaulting to per-user /tmp paths when unset."""
    base = f"/tmp/subliminal_mps_{getpass.getuser()}"
    pipe = policy.mps_pipe_dir or f"{base}/pipe"
    log = policy.mps_log_dir or f"{base}/log"
    return pipe, log


def mps_env(policy: GpuPolicy) -> dict[str, str]:
    """The two MPS env vars (resolved pipe/log dirs); empty dict when ``mps`` is off."""
    if not policy.mps:
        return {}
    pipe, log = _mps_dirs(policy)
    return {"CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_LOG_DIRECTORY": log}


def ensure_mps_daemon(policy: GpuPolicy) -> bool:
    """Ensure a CUDA MPS control daemon is up for this user; return whether MPS is usable.

    Probes ``nvidia-cuda-mps-control`` (via ``get_server_list``); if it doesn't respond,
    spawns the daemon with ``-d`` and re-probes once. A missing binary or a failed probe is
    logged as a warning and returns False — the pool must run fine without MPS.
    """
    pipe, log = _mps_dirs(policy)
    Path(pipe).mkdir(parents=True, exist_ok=True)
    Path(log).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_LOG_DIRECTORY": log}

    def _probe() -> bool:
        try:
            r = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="get_server_list\n",
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )
        except FileNotFoundError:
            raise
        except (subprocess.SubprocessError, OSError):
            return False
        return r.returncode == 0

    try:
        if _probe():
            return True
        # Spawn the daemon with CUDA_VISIBLE_DEVICES REMOVED from its env: a device-restricted
        # MPS server renumbers/limits the GPUs its clients can see, which breaks per-job device
        # indexing (each job sets its own CUDA_VISIBLE_DEVICES against the full physical set).
        spawn_env = dict(env)
        spawn_env.pop("CUDA_VISIBLE_DEVICES", None)
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True,
            text=True,
            env=spawn_env,
            timeout=10,
        )
        if _probe():
            return True
        logger.warning("CUDA MPS daemon did not come up after spawn; running without MPS")
        return False
    except FileNotFoundError:
        logger.warning(
            "nvidia-cuda-mps-control not found; running without MPS (mps=true in policy)"
        )
        return False


def query_free_mem() -> dict[int, int]:
    """GPU index -> free memory (MiB) via ``nvidia-smi`` (the only smi call; tests patch it).

    ``timeout``: these queries run inside the node-wide acquire flock — a wedged smi
    without one stalled every worker thread on the node indefinitely.
    """
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    free: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx, mem = (x.strip() for x in line.split(","))
        free[int(idx)] = int(mem)
    return free


def query_total_mem() -> dict[int, int]:
    """GPU index -> TOTAL memory (MiB) via ``nvidia-smi`` (patched in tests, like the free query)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    total: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx, mem = (x.strip() for x in line.split(","))
        total[int(idx)] = int(mem)
    return total


_TOTAL_MEM_CACHE: dict[int, int] | None = None


def total_mem(query=None) -> dict[int, int]:
    """Cached GPU index -> total memory (MiB); a card's capacity cannot change under us.

    An ``nvidia-smi`` failure yields an empty map (callers must treat that as "no ceiling
    known") and is NOT cached, so a transient failure is retried.
    """
    global _TOTAL_MEM_CACHE
    if _TOTAL_MEM_CACHE is None:
        try:
            _TOTAL_MEM_CACHE = (query or query_total_mem)()
        except (subprocess.SubprocessError, OSError, ValueError):
            return {}
    return dict(_TOTAL_MEM_CACHE)


def clear_total_mem_cache() -> None:
    """Drop the :func:`total_mem` cache (tests patching the query must call this)."""
    global _TOTAL_MEM_CACHE
    _TOTAL_MEM_CACHE = None


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

    gpu: int | None
    slot: int
    _fds: list[int] = field(default_factory=list)
    _group_fd: int | None = None
    # The owning manager's live-fd registry (see GpuManager._live_lock_fds); release()
    # deregisters so the leak auditor knows these fds are legitimately gone. Mutations go
    # under the manager's ``_reg_lock`` so they serialize with the auditor's audit pass.
    _registry: "set[int] | None" = None
    _reg_lock: "threading.Lock | None" = None

    @property
    def is_cpu_only(self) -> bool:
        """True for a ``slots=0`` slot: it holds a CPU-budget lock, never GPU capacity."""
        return self.gpu is None

    def release(self) -> None:
        reg = self._registry
        with self._reg_lock if self._reg_lock is not None else contextlib.nullcontext():
            for fd in self._fds:
                _close_lock(fd)
                if reg is not None:
                    reg.discard(fd)
            self._fds = []
            if self._group_fd is not None and reg is not None:
                reg.discard(self._group_fd)
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
        lock_prefix: str | None = None,
        mem_query=query_free_mem,
        total_query=None,
        mem_checks: int = 2,
        mem_interval_s: float = 3.0,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        # SL_JOBQ_LOCK_PREFIX: on shared nodes another user's jobq instance may have created
        # 0644 lock files under the default /tmp prefix that we cannot even open (os.open ->
        # EACCES before flock). Slot locks only coordinate OUR pools, and only when their GPU
        # sets overlap — so a pool bound to a disjoint GPU set can safely run in its own lock
        # namespace; co-existence with the foreign user's jobs on those GPUs is handled by the
        # free-mem gate + yield watchdog, not by locks.
        self.lock_prefix = (
            lock_prefix or os.environ.get("SL_JOBQ_LOCK_PREFIX") or DEFAULT_LOCK_PREFIX
        )
        self.mem_query = mem_query
        self.total_query = total_query
        self.mem_checks = mem_checks
        self.mem_interval_s = mem_interval_s
        # Leak audit state (2026-08-24: EDQUOT thrown between flock and GpuSlot
        # construction leaked 24 slot fds and shrank the node to 16/40 slots until a pool
        # restart). Every fd handed out inside a returned GpuSlot is registered here and
        # deregistered by GpuSlot.release(); reap_leaked_locks() closes what's left over.
        self._reg_lock = threading.Lock()
        self._live_lock_fds: set[int] = set()
        self._leak_suspects: dict[int, str] = {}
        # Serializes every SHORT flock-to-register/close window that runs OUTSIDE the
        # acquire flock (CPU-slot grabs, group_has_room probes, _probe scans) with the
        # leak auditor's snapshot+close pass: two-strike keys on (fd number, lock path)
        # and cannot distinguish descriptor GENERATIONS, so without this a recurring
        # transient at a reused fd number could be closed as a "leak". The seconds-long
        # windows inside acquire() are covered by the acquire flock instead — this lock
        # must never wrap them, or CPU probes would block behind GPU admission. Ordering:
        # acquire flock -> _audit_lock -> _reg_lock (never reversed).
        self._audit_lock = threading.Lock()

    def _policy(self) -> GpuPolicy:  # re-read every time (hot-reload of pins)
        return load_policy(self.root, self.hostname)

    def env(self) -> dict[str, str]:
        return self._policy().env

    def default_mem_mib(self) -> int:
        return self._policy().free_mem_mib

    def mem_ceiling_mib(self) -> int:
        """Largest policy card's TOTAL memory — the highest ``mem_mib`` any job could be granted.

        ``0`` means unknown (no policy / no totals readable), which callers must read as
        "no ceiling", never as "everything is over the ceiling".
        """
        try:
            gpus = self._policy().gpus
        except FileNotFoundError:
            return 0
        totals = total_mem(self.total_query)
        return max((totals.get(g, 0) for g in gpus), default=0)

    def _slot_lock_file(self, g: int, s: int) -> str:
        return f"{self.lock_prefix}_gpu{g}_slot{s}.lock"

    def _cpu_slot_lock_file(self, s: int) -> str:
        """CPU budget lives in its own lock namespace, independent of any GPU."""
        return f"{self.lock_prefix}_cpu_slot{s}.lock"

    def _try_slot(self, g: int, s: int) -> int | None:
        fd = os.open(self._slot_lock_file(g, s), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return fd

    def _acquire_cpu_slot(self, policy: GpuPolicy) -> "GpuSlot | None":
        """One of ``policy.cpu_cap`` node-wide CPU slots, or None if all are held.

        Deliberately independent of the GPU budget and of the memory gate: a CPU-only job
        neither consumes nor is blocked by GPU capacity, so it runs even when every card is
        full. Same flock mechanism as the GPU slots, so the ceiling holds across worker
        PROCESSES, not just threads.
        """
        with self._audit_lock:  # flock-to-register window must not overlap an audit pass
            for s in range(max(1, policy.cpu_cap)):
                fd = os.open(self._cpu_slot_lock_file(s), os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                with self._reg_lock:
                    self._live_lock_fds.add(fd)
                return GpuSlot(
                    gpu=None,
                    slot=s,
                    _fds=[fd],
                    _registry=self._live_lock_fds,
                    _reg_lock=self._reg_lock,
                )
        return None

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

    def group_has_room(self, group: str, cap: int) -> bool:
        """Whether any policy GPU still has a free ``group`` stage slot (probe-and-release).

        Used by the worker to skip a capped queue instead of claiming a job and parking on
        it. Racy by design (room may vanish before the real acquire) -- then the thread
        waits as before, which is rare rather than the steady state it used to be.
        """
        policy = self._policy()
        yielded = yielding.yielded_gpus(self.root, self.hostname)
        with self._audit_lock:  # probe fds are never registered; keep audits out
            for g in policy.gpus:
                if g in yielded:
                    continue
                fd = self._try_group_slot(group, g, cap)
                if fd is not None:
                    _close_lock(fd)
                    return True
        return False

    def cpu_has_room(self) -> bool:
        """Whether a ``cpu_cap`` slot is currently free (probe-and-release).

        The CPU-lane twin of :meth:`group_has_room`: the worker skips a ``slots: 0`` queue
        instead of claiming a job and parking the thread on the cpu_cap budget. Without
        this, a burst of CPU-only submissions above ``cpu_cap`` wedged one pool thread per
        waiting job and starved GPU admission (2026-08-21: 13 wedged threads, 8 idle GPU
        slots on soar-subliminal-1). Racy by design -- see group_has_room.
        """
        slot = self._acquire_cpu_slot(self._policy())
        if slot is None:
            return False
        slot.release()
        return True

    def _mem_ok(self, g: int, mem_mib: int, reserved: int) -> bool:
        for i in range(self.mem_checks):
            if i:
                time.sleep(self.mem_interval_s)
            try:
                free = self.mem_query().get(g, 0)
            except (subprocess.SubprocessError, OSError, ValueError):
                # A hung/broken smi (now bounded by the query timeout) denies THIS
                # admission instead of raising through acquire while slot fds are held.
                logger.warning("nvidia-smi free-memory query failed; denying admission on GPU {}", g)
                return False
            if free - reserved < mem_mib:
                return False
        return True

    def _read_reservation(self, g: int, s: int, warmup_s: int) -> int:
        """MiB still unaccounted-for in ``nvidia-smi`` for the job holding slot ``(g, s)``.

        Zero once the reservation is older than `warmup_s` — by then the job's memory is
        really allocated and counting it again would double-charge the GPU. An EMPTY lock
        file is the benign "no reservation recorded" case (slot held by a pre-ledger
        worker); anything unreadable or corrupt is logged, since it silently relaxes the
        admission gate back to the ramp-up race it exists to prevent.
        """
        path = Path(self._slot_lock_file(g, s))
        try:
            body = path.read_text()
        except OSError:
            logger.warning(
                "GPU {} slot {}: reservation unreadable ({}) — admitting against "
                "nvidia-smi alone; a concurrently ramping job may be undercounted",
                g,
                s,
                path,
            )
            return 0
        if not body.strip():
            return 0
        try:
            rec = json.loads(body)
        except ValueError:
            logger.warning(
                "GPU {} slot {}: reservation record is corrupt ({!r}) — admitting "
                "against nvidia-smi alone",
                g,
                s,
                body[:120],
            )
            return 0
        if time.time() - float(rec.get("ts", 0)) > warmup_s:
            return 0
        return int(rec.get("mem_mib", 0))

    def _probe(self, policy: GpuPolicy) -> tuple[dict[int, int], dict[int, int]]:
        """Per-GPU (slots held, MiB reserved by ramping jobs); non-destructive probe."""
        counts: dict[int, int] = {}
        reserved: dict[int, int] = {}
        with self._audit_lock:  # probe fds are never registered; keep audits out
            for g in policy.gpus:
                held = 0
                mib = 0
                for s in range(policy.cap_per_gpu):
                    fd = self._try_slot(g, s)
                    if fd is None:
                        held += 1
                        mib += self._read_reservation(g, s, policy.reservation_warmup_s)
                    else:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                            os.close(fd)
                        except OSError:
                            pass
                counts[g] = held
                reserved[g] = mib
        return counts, reserved

    def _write_reservation(self, fd: int, mem_mib: int) -> None:
        """Stamp a granted slot lock with its memory claim.

        A failure here does not fail the job (the slot is already granted and the GPU
        gate already passed), but it does blind every CONCURRENT acquire to this job's
        ramp-up, so it is logged loudly rather than swallowed.
        """
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(
                fd,
                json.dumps(
                    {"mem_mib": int(mem_mib), "pid": os.getpid(), "ts": time.time()}
                ).encode(),
            )
            os.fsync(fd)
        except OSError:
            logger.warning(
                "could not stamp {} MiB reservation on a granted slot lock (pid {}) — "
                "concurrent workers may over-admit during this job's model load",
                int(mem_mib),
                os.getpid(),
            )

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

        ``slots=0`` declares a FULLY CPU-BOUND job. It takes a slot from the SEPARATE
        ``policy.cpu_cap`` budget in its own lock namespace, never from GPU capacity, and
        returns ``gpu=None``. Two independent budgets, because the two concerns are
        independent: a ``--device cpu`` job used to consume one of the node's GPU slot-units
        and block a GPU job (8 CPU-only probes held a third of a 24-unit pool on
        2026-08-09), but exempting it entirely would have removed its only ceiling — the GPU
        slots were doubling as the pool's admission control, so N worker threads could then
        all run CPU jobs at once and thrash the box.

        Strictly OPT-IN: a job must declare ``slots: 0`` itself. Nothing infers CPU-ness from
        the command line, because a job that touches CUDA while claiming no GPU slot would
        silently oversubscribe the card.
        """
        policy = self._policy()
        if slots == 0:
            return self._acquire_cpu_slot(policy)
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
            counts, reserved = self._probe(policy)
            order = sorted(
                rotated, key=lambda g: counts[g]
            )  # stable: rotation breaks ties
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
                if snapshot.get(g, 0) - reserved[g] < mem_mib:
                    continue
                held: list[tuple[int, int]] = []  # (slot index, fd)
                group_fd: int | None = None
                # Any exception below (2026-08-24: EDQUOT from _write_reservation, but
                # also nvidia-smi hiccups in _mem_ok) must roll the flocked fds back
                # before propagating — a leaked fd is a slot-unit gone until pool death.
                try:
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
                    if cap_per_gpu is not None:
                        group_fd = self._try_group_slot(
                            cap_group or "default", g, cap_per_gpu
                        )
                        if group_fd is None:  # stage ceiling reached on this GPU
                            for _, fd in held:
                                _close_lock(fd)
                            continue
                    if self._mem_ok(g, mem_mib, reserved[g]):
                        # Late yield re-check: the memory gate SLEEPS (~mem_interval_s) while we
                        # hold the slot, and the watchdog can yield this card during that sleep.
                        # Re-read the marker fresh here (not the pre-loop snapshot) and, if the GPU
                        # became yielded, roll back exactly like the other abandon paths rather than
                        # dispatch onto a card we promised to vacate. Before stamping the
                        # reservation: an abandoned slot must leave no ledger entry behind.
                        if g in yielding.yielded_gpus(self.root, self.hostname):
                            for _, fd in held:
                                _close_lock(fd)
                            _close_lock(group_fd)
                            continue
                        for _, fd in held:
                            self._write_reservation(fd, mem_mib / len(held))
                        with self._reg_lock:
                            self._live_lock_fds.update(fd for _, fd in held)
                            if group_fd is not None:
                                self._live_lock_fds.add(group_fd)
                        return GpuSlot(
                            gpu=g,
                            slot=held[0][0],
                            _fds=[fd for _, fd in held],
                            _group_fd=group_fd,
                            _registry=self._live_lock_fds,
                            _reg_lock=self._reg_lock,
                        )
                    for _, fd in held:  # not enough memory -> next GPU
                        _close_lock(fd)
                    _close_lock(group_fd)
                except Exception:
                    for _, fd in held:
                        _close_lock(fd)
                    _close_lock(group_fd)
                    raise
            return None
        finally:
            _close_lock(acq_fd)

    def reap_leaked_locks(self) -> list[str]:
        """Close slot-lock fds this process holds that no live :class:`GpuSlot` owns.

        Belt-and-braces behind the rollback in :meth:`acquire`: whatever still leaks (an
        unforeseen throw path) is found by diffing this process's open fds against the
        ``_live_lock_fds`` registry and closed, returning the slot-unit to the pool without
        a restart. Returns the lock paths closed (for the master log).

        The audit runs while HOLDING the acquire flock (bounded wait — see below). Without
        it, the two-strike check below cannot tell a persisted leak from a RECURRING
        transient: a thread waiting at the admission boundary re-runs ``acquire`` every few
        seconds, each attempt flocks the same slot path at the same (deterministically
        reused) fd number and holds it unregistered through the ~3 s memory-gate sleep —
        two consecutive audits would both see that (fd, path) pair "unowned" and close an
        fd a live acquire holds (releasing the slot lock mid-grant, then reservation
        writes / rollback closes land on a reused fd number). Under the acquire flock no
        in-process acquire is in flight, so its slot fds are either registered or a true
        leak. The short unregistered windows OUTSIDE the acquire flock (CPU-slot grabs,
        group_has_room probes, _probe scans) are excluded by ``_audit_lock``, which this
        pass also holds — belt-and-braces on top of the two-strike grace: an fd is only
        closed when it was seen unowned on the PREVIOUS call too and still resolves to the
        same lock path.

        The flock wait is BOUNDED (``_AUDIT_FLOCK_DEADLINE_S``, nonblocking retries): the
        memory gate runs ``nvidia-smi`` with no subprocess timeout while holding the
        acquire flock, and this runs on the SUPERVISOR thread — a wedged smi must degrade
        to skipped audit ticks, never to a supervisor that no longer respawns workers or
        reaps orphan claims.
        """
        prefix = self.lock_prefix
        try:
            acq_fd = os.open(
                f"{prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644
            )
        except OSError:
            return []  # can't serialize with acquires -> skip this tick, never guess
        try:
            deadline = time.monotonic() + _AUDIT_FLOCK_DEADLINE_S
            while True:
                try:
                    fcntl.flock(acq_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        return []  # acquire wedged or hot; audit again next tick
                    time.sleep(0.05)
            with self._audit_lock:
                try:
                    fd_names = os.listdir("/proc/self/fd")
                except OSError:
                    return []
                seen: dict[int, str] = {}
                for s in fd_names:
                    try:
                        fd = int(s)
                        path = os.readlink(f"/proc/self/fd/{fd}")
                    except (ValueError, OSError):
                        continue
                    if not path.startswith(prefix):
                        continue
                    if not _SLOT_LOCK_SUFFIX_RE.fullmatch(path[len(prefix):]):
                        continue
                    seen[fd] = path
                closed: list[str] = []
                with self._reg_lock:
                    unowned = {
                        fd: p for fd, p in seen.items() if fd not in self._live_lock_fds
                    }
                    strikes = {
                        fd: p
                        for fd, p in unowned.items()
                        if self._leak_suspects.get(fd) == p
                    }
                    self._leak_suspects = unowned
                    for fd, path in strikes.items():
                        try:  # fd numbers get reused: only close if it still IS that lock
                            if os.readlink(f"/proc/self/fd/{fd}") != path:
                                continue
                        except OSError:
                            continue
                        _close_lock(fd)
                        del self._leak_suspects[fd]
                        closed.append(path)
                return closed
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
        return self._probe(policy)[0]
