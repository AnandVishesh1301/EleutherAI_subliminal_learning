"""The threaded worker pool: claim -> acquire GPU -> run -> record, log-and-continue.

One process, ``--workers N`` threads, all sharing one :class:`GpuManager` (so the per-GPU
slot cap is global). Each thread loops over the READY queues (node filter + deps satisfied,
ordered by priority desc then created_utc), steals its own node's stale claims, claims the
first pending job, runs it, records the result, and moves on. A job failure never aborts the
pool. Between jobs it honors the ``stop.<hostname>`` file.

Master-log verbs (``[<iso-utc>] VERB ...``): CLAIM, START, END rc=<rc>, STEAL, WAIT,
TRIPPED <name>, QUEUE COMPLETE <name>, and a final ALL QUEUES COMPLETE. The opt-in
GPU-yielding framework (``jobqueue/yielding.py``) adds YIELD gpu=N foreign=K,
KILL <queue> <key> gpu=N, DRAIN <queue> <key> gpu=N pct=P (finishing) (a near-done
job spared under drain_if_near_done), and RECLAIM gpu=N.

A queue whose jobs fail back-to-back is *tripped* rather than drained: a job that
dies during startup (bad config, provenance guard, missing checkpoint) fails in
seconds and deterministically, so an untripped pool converts one broken job into a
whole burnt queue in minutes. The pool keeps serving other queues and re-admits a
tripped one as soon as its pending count grows (i.e. after a requeue) or, after
``breaker_cooldown_s``, when a single HALF-OPEN probe job succeeds. A yield-kill
is NOT a failure and never counts toward the streak.

A job killed by CUDA OOM is likewise not a failure: its log tail is classified
(:func:`is_oom_text`) and it is requeued with an escalated ``mem_mib`` floor persisted in
the attempts sidecar, so the memory GATE — not a timer or a retry counter — decides when it
runs again. The only terminal case is a job that OOMs while already asking for the whole
card (``OOM CEILING``), which is recorded as an ordinary failure.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.jobqueue import store, yielding
from subliminal_learning.jobqueue.gpu import (
    DEFAULT_LOCK_PREFIX,
    GpuManager,
    ensure_mps_daemon,
    load_policy,
    mps_env,
)
from subliminal_learning.jobqueue.store import (
    QueueMeta,
    REPO_ROOT,
    now_iso,
    utc_stamp,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils.io import atomic_write_text

DEFAULT_FAIL_STREAK_LIMIT = 5
# Seconds a tripped queue stays fully closed before one half-open probe job is allowed.
DEFAULT_BREAKER_COOLDOWN_S = 900.0
# OOM escalation: new_mem = max(ceil(old * FACTOR), FLOOR_MIB), capped at the card's total.
OOM_MEM_FACTOR = 1.5
OOM_MEM_FLOOR_MIB = 4096
OOM_LOG_TAIL_BYTES = 50_000

_OOM_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUDA error: out of memory",
    "cudaErrorMemoryAllocation",
)


def is_oom_text(text: str) -> bool:
    """Whether a job's output carries any of the CUDA out-of-memory markers."""
    return any(m in text for m in _OOM_MARKERS)


def log_tail_is_oom(path: Path | str, *, tail_bytes: int = OOM_LOG_TAIL_BYTES) -> bool:
    """Classify a finished job's log as OOM from its last ``tail_bytes``.

    A missing or unreadable log is NOT an OOM (the caller must fall through to normal
    failure handling); the read is byte-bounded so a multi-GB training log costs one seek.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            tail = f.read()
    except OSError:
        return False
    return is_oom_text(tail.decode("utf-8", "replace"))


class MasterLog:
    """Thread-safe append-only master log; also mirrors each line to loguru."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        store.mkdir_open(self.path.parent)

    def log(self, verb: str, msg: str = "") -> None:
        line = f"[{now_iso()}] {verb} {msg}".rstrip()
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
        logger.info(line)


@dataclass
class _RunHandle:
    """A live job subprocess, tracked per-GPU so the yield watchdog can kill it.

    ``killed`` distinguishes "I killed this to yield the card" from "it exited on its own"
    so the runner requeues (not fails) a yield-killed job. ``log_path``/``start_ts`` feed the
    ``drain_if_near_done`` progress estimate (log-tail regex + elapsed/median); ``drained``
    records that a spared job's DRAIN line was already logged (so it is not re-logged every
    poll while the card stays yielded).
    """

    proc: subprocess.Popen
    queue: str
    job: Job
    gpu: int
    killed: bool = False
    log_path: str = ""
    start_ts: float = 0.0
    drained: bool = False


def _kill_group(proc: subprocess.Popen, *, grace_s: float = 20.0, poll_s: float = 0.5) -> None:
    """SIGTERM the process's group, wait ``grace_s``, then SIGKILL (best-effort, no raise).

    The job is launched with ``start_new_session=True`` so ``bash -c`` and every child share
    a process group we can signal in one shot.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(poll_s)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _acquire_pid_file(root: Path, hostname: str) -> None:
    """Refuse to start if a live worker pid for THIS node already holds the pid file."""
    p = store.worker_pid_path(root, hostname)
    if p.exists():
        try:
            host, pid_s = p.read_text().strip().split(":")
            pid = int(pid_s)
        except (ValueError, OSError):
            host, pid = "", 0
        if host == hostname and pid and store.pid_alive(pid):
            raise RuntimeError(
                f"a worker is already running on {hostname} (pid {pid}); "
                f"remove {p} if it is stale"
            )
    atomic_write_text(p, f"{hostname}:{os.getpid()}")


class _Pool:
    """Shared coordinator across worker threads (deps, once-only completion logs, stop)."""

    def __init__(
        self,
        root: Path,
        hostname: str,
        gpu: GpuManager,
        master: MasterLog,
        *,
        queues_filter: set[str] | None,
        poll_s: float,
        gpu_wait_s: float,
        fail_streak_limit: int = DEFAULT_FAIL_STREAK_LIMIT,
        breaker_cooldown_s: float = DEFAULT_BREAKER_COOLDOWN_S,
        mps_env: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.gpu = gpu
        self.master = master
        self.queues_filter = queues_filter
        self.poll_s = poll_s
        self.gpu_wait_s = gpu_wait_s
        self.fail_streak_limit = fail_streak_limit
        self.breaker_cooldown_s = breaker_cooldown_s
        # Injected only when ensure_mps_daemon succeeded; empty (byte-identical to pre-MPS)
        # otherwise. Merged into every job's env so clients find the shared MPS server.
        self.mps_env = mps_env or {}
        self._lock = threading.Lock()
        self._complete_logged: set[str] = set()
        self._dep_warned: set[str] = set()
        # Live jobs by physical GPU so the yield watchdog can kill this card's jobs:
        # {gpu: {jobkey: _RunHandle}}. Guarded by its own lock (touched from worker
        # threads + the watchdog thread).
        self._running_lock = threading.Lock()
        self._running: dict[int, dict[str, _RunHandle]] = {}
        self._fail_streak: dict[str, int] = {}
        self._tripped_pending: dict[str, int] = {}
        # Half-open bookkeeping: when a queue tripped (monotonic), and whether one probe
        # job from it is currently in flight. Guarded by ``_lock`` — no timer, no thread.
        self._tripped_at: dict[str, float] = {}
        self._probing: set[str] = set()
        self._clock = time.monotonic  # rebindable so tests can advance the cooldown

    # ------------------- failure circuit breaker -------------------

    def _record_outcome(self, name: str, rc: int) -> None:
        """Count consecutive failures; trip the queue at the limit (<=0 disables)."""
        if self.fail_streak_limit <= 0:
            return
        with self._lock:
            if rc == 0:
                self._fail_streak[name] = 0
                return
            streak = self._fail_streak.get(name, 0) + 1
            self._fail_streak[name] = streak
            if streak < self.fail_streak_limit or name in self._tripped_pending:
                return
            self._tripped_pending[name] = store.queue_status(self.root, name)["pending"]
            self._tripped_at[name] = self._clock()
        self.master.log(
            "TRIPPED",
            f"{name} — {self.fail_streak_limit} consecutive job failures; not "
            "claiming from it again until its pending count grows. Inspect the "
            f"newest log under {store.queue_logs_dir(self.root, name)}, fix the "
            f"cause, then `jobq requeue --root {self.root} --name {name}`.",
        )

    def _is_tripped(self, name: str) -> bool:
        """Whether `name` is still tripped; a requeue (more pending) re-admits it.

        A tripped queue also goes HALF-OPEN once ``breaker_cooldown_s`` has passed: it is
        reported un-tripped so the ready list offers it, and :meth:`_probe_admit` then lets
        exactly one claim through.
        """
        with self._lock:
            at_trip = self._tripped_pending.get(name)
            if at_trip is None:
                return False
            if store.queue_status(self.root, name)["pending"] > at_trip:
                del self._tripped_pending[name]
                self._tripped_at.pop(name, None)
                self._fail_streak[name] = 0
                readmit = True
            else:
                return not self._half_open(name)
        if readmit:
            self.master.log("WAIT", f"{name} re-admitted after requeue")
        return False

    def _half_open(self, name: str) -> bool:
        """Whether a tripped queue may currently offer a probe job (caller holds ``_lock``)."""
        if self.breaker_cooldown_s <= 0 or name in self._probing:
            return False
        at = self._tripped_at.get(name)
        return at is not None and self._clock() - at >= self.breaker_cooldown_s

    def _probe_admit(self, name: str) -> bool:
        """Gate a fresh claim from a tripped queue to ONE half-open probe at a time.

        Always True for an untripped queue. Concurrent threads race here under ``_lock``, so
        only the first gets the probe slot; the losers must give their claim back.
        """
        with self._lock:
            if name not in self._tripped_pending:
                return True
            if not self._half_open(name):
                return False
            self._probing.add(name)
        self.master.log("WAIT", f"{name} half-open probe after cooldown")
        return True

    def _probe_finish(self, name: str, rc: int | None) -> None:
        """Settle a half-open probe: success un-trips, failure re-trips with a fresh cooldown.

        ``rc is None`` (yield-kill or OOM requeue) settles nothing — the probe slot is simply
        released so a later claim can probe again.
        """
        with self._lock:
            if name not in self._probing:
                return
            self._probing.discard(name)
            if rc is None:
                return
            if rc == 0:
                self._tripped_pending.pop(name, None)
                self._tripped_at.pop(name, None)
                self._fail_streak[name] = 0
                msg = f"{name} re-admitted after successful probe"
            else:
                self._tripped_at[name] = self._clock()
                msg = f"{name} probe failed rc={rc}; re-tripped for another cooldown"
        self.master.log("WAIT", msg)

    # ------------------- queue selection -------------------

    def _node_queues(self) -> list[str]:
        """Queue names visible to this node (node filter + optional --queues restriction)."""
        out = []
        for name in store.list_queues(self.root):
            if self.queues_filter is not None and name not in self.queues_filter:
                continue
            if store.read_meta(self.root, name).node in (None, self.hostname):
                out.append(name)
        return out

    def _maybe_log_complete(self, name: str) -> None:
        with self._lock:
            if name in self._complete_logged:
                return
            self._complete_logged.add(name)
        self.master.log("QUEUE COMPLETE", name)

    def _deps_ok(self, meta: QueueMeta) -> bool:
        for dep in meta.depends_on:
            if not (store.queue_exists(self.root, dep) and store.queue_complete(self.root, dep)):
                return False
            failed = store.queue_status(self.root, dep)["failed"]
            if failed:
                strict = meta.strict_deps
                with self._lock:
                    warn_key = f"{meta.name}<-{dep}"
                    warn = warn_key not in self._dep_warned
                    self._dep_warned.add(warn_key)
                if warn:
                    self.master.log(
                        "WAIT",
                        f"dep {dep} completed with {failed} failed job(s) "
                        + (
                            f"— strict_deps blocks {meta.name} until they are requeued/fixed"
                            if strict
                            else "(treated as terminal)"
                        ),
                    )
                if strict:
                    return False
        return True

    def ready_queues(self) -> list[tuple[str, QueueMeta]]:
        """Ready = node-visible, not complete, deps satisfied — ordered (prio desc, ctime)."""
        ready: list[tuple[str, QueueMeta]] = []
        for name in self._node_queues():
            meta = store.read_meta(self.root, name)
            if store.queue_complete(self.root, name):
                self._maybe_log_complete(name)
                continue
            if not self._deps_ok(meta) or self._is_tripped(name):
                continue
            ready.append((name, meta))
        ready.sort(key=lambda nm: (-nm[1].priority, nm[1].created_utc))
        return ready

    def all_complete(self) -> bool:
        """True once every node-visible queue is complete (or there are none)."""
        names = self._node_queues()
        return all(store.queue_complete(self.root, n) for n in names)

    # ------------------- run one job -------------------

    def _effective_mem(self, name: str, meta: QueueMeta, job: Job) -> int:
        """The job's memory request: max(static resolution, persisted OOM floor).

        The floor lives on disk (attempts sidecar), so an escalation survives a worker
        restart and is honored by whichever NODE claims the job next.
        """
        if job.mem_mib is not None:
            static = job.mem_mib
        else:
            d = meta.defaults.get("mem_mib")
            static = int(d) if d is not None else self.gpu.default_mem_mib()
        return max(static, store.read_mem_floor(self.root, name, job.jobkey))

    def _mem_ceiling(self) -> int:
        """Highest grantable ``mem_mib`` on this node; ``0`` when unknown (= no ceiling)."""
        probe = getattr(self.gpu, "mem_ceiling_mib", None)
        if probe is None:
            return 0
        try:
            return int(probe() or 0)
        except Exception:  # noqa: BLE001 — a broken smi must not fail a job
            logger.exception("could not read the GPU memory ceiling; treating as unbounded")
            return 0

    def _oom_requeue(self, name: str, meta: QueueMeta, job: Job, log_path: Path) -> bool:
        """Requeue an OOM-killed job with an escalated memory floor; False = handle normally.

        False means either "not an OOM" or "already asking for the whole card" — the latter
        is the ONLY terminal condition (there is deliberately no retry cap: the memory gate,
        not a counter, decides when an escalated job runs again).
        """
        if not log_tail_is_oom(log_path):
            return False
        old = self._effective_mem(name, meta, job)
        ceiling = self._mem_ceiling()
        if ceiling and old >= ceiling:
            self.master.log(
                "OOM CEILING",
                f"{name} {job.jobkey} mem_mib {old} >= card total {ceiling}; recording failure",
            )
            return False
        new = max(math.ceil(old * OOM_MEM_FACTOR), OOM_MEM_FLOOR_MIB)
        if ceiling:
            new = min(new, ceiling)
        store.write_mem_floor(self.root, name, job.jobkey, new)
        store.remove_claim(self.root, name, job.jobkey)
        store.bump_attempt(self.root, name, job.jobkey)
        self.master.log("OOM REQUEUE", f"{name} {job.jobkey} mem_mib {old} -> {new}")
        return True

    def _effective_slots(self, meta: QueueMeta, job: Job) -> int:
        """The job's weight in global slot-units (job field > queue default > 1)."""
        if job.slots is not None:
            return job.slots
        return int(meta.defaults.get("slots", 1))

    def _effective_cwd(self, meta: QueueMeta, job: Job) -> str:
        """Working directory to run the job in (job field > queue default > this checkout).

        The fallback is the CLAIMING pool's checkout, so on a queue root shared by two users
        an unpinned job runs against whoever claims it — including their
        ``configs/paths/paths_local.yaml`` and therefore their ``output_root``. Set
        ``defaults.cwd`` (or a per-job ``cwd``) to bind jobs to the submitter's checkout.
        """
        if job.cwd is not None:
            return job.cwd
        d = meta.defaults.get("cwd")
        if d is not None:
            return str(d)
        return str(REPO_ROOT)

    def _effective_cap(self, meta: QueueMeta) -> tuple[int | None, str | None]:
        """Queue-level stage ceiling: (cap_per_gpu, cap_group) from meta.defaults.

        ``cap_group`` defaults to the queue name so each capped queue gets its own ceiling;
        set the same ``cap_group`` on several queues to co-cap them (the old shared
        ``LOCK_PREFIX`` pattern).
        """
        cap = meta.defaults.get("cap_per_gpu")
        # Per-node override (2026-08-21): ``cap_per_gpu_by_node: {hostname: n}`` lets the
        # same queue run e.g. 3/GPU on one node and 4/GPU on another. Queues sharing a
        # cap_group must agree per node (the ceiling is the slot-file range probed).
        by_node = meta.defaults.get("cap_per_gpu_by_node") or {}
        if self.hostname in by_node:
            cap = by_node[self.hostname]
        if cap is None:
            return None, None
        return int(cap), str(meta.defaults.get("cap_group") or meta.name)

    def _acquire_gpu(
        self,
        mem_mib: int,
        slots: int,
        cap_per_gpu: int | None,
        cap_group: str | None,
        job_key: str,
    ):
        """Block until a reservation is free or a stop is requested (returns None on stop).

        WAIT lines are rate-limited (first miss, then ~once a minute) and name the waiting
        job, so an unsatisfiable request (mem_mib/slots no GPU can ever meet) is visible in
        the master log instead of silently wedging a thread.
        """
        misses = 0
        while True:
            if store.stop_path(self.root, self.hostname).exists():
                return None
            slot = self.gpu.acquire(
                mem_mib, slots=slots, cap_per_gpu=cap_per_gpu, cap_group=cap_group
            )
            if slot is not None:
                return slot
            misses += 1
            per_min = max(1, int(60 / max(self.gpu_wait_s, 0.001)))
            if misses == 1 or misses % per_min == 0:
                waited_min = int(misses * self.gpu_wait_s / 60)
                if slots == 0:
                    # CPU-only jobs wait on the cpu_cap budget, never on GPU capacity or
                    # the memory gate — pointing at mem_mib/gpu_policy here misdirects.
                    need = "a cpu_cap slot (CPU-only job, slots=0)"
                    hint = " — all cpu_cap slots held; check cpu_cap in this node's gpu_policy"
                else:
                    need = f"{mem_mib} MiB free + {slots} slot-unit(s)"
                    hint = " — check mem_mib/slots against this node's gpu_policy"
                self.master.log(
                    "WAIT",
                    f"{job_key} needs {need}; ~{waited_min}min so far"
                    + (hint if waited_min >= 10 else ""),
                )
            time.sleep(self.gpu_wait_s)

    def _run_job(
        self, name: str, meta: QueueMeta, job: Job, slot, attempt: int
    ) -> int | None:
        """Run one job to completion. Returns its rc, or ``None`` if it was requeued.

        A yield-killed job — and, on the same path, one classified as CUDA OOM — is requeued
        to PENDING (claim removed, NO result recorded, attempt bumped) so it retries later;
        neither is ever recorded as failed.
        """
        store.update_owner_gpu(self.root, name, job.jobkey, slot.gpu)
        pid = os.getpid()
        stamp = utc_stamp()
        log_path = (
            store.queue_logs_dir(self.root, name)
            / f"{stamp}_{job.jobkey}.{self.hostname}.log"
        )
        store.mkdir_open(log_path.parent)
        start = now_iso()
        env = os.environ.copy()
        env.update(self.gpu.env())
        env.update(self.mps_env)  # empty unless MPS was ensured up at pool startup
        env.update({str(k): str(v) for k, v in (meta.defaults.get("env") or {}).items()})
        env.update(job.env or {})
        # slots=0 (declared CPU-only): no GPU was reserved, so hide every device rather than
        # leaving the inherited CVD visible -- a job that claimed no slot must not be able to
        # quietly grab a card and oversubscribe it.
        env["CUDA_VISIBLE_DEVICES"] = "" if slot.is_cpu_only else str(slot.gpu)
        cwd = self._effective_cwd(meta, job)
        with open(log_path, "w") as f:
            f.write(
                f"=== node={self.hostname} pid={pid} gpu={slot.gpu} "
                f"queue={name} job={job.key} start={start} ===\n"
            )
            f.flush()
            # start_new_session=True: bash -c and its children get their own process group
            # so the yield watchdog can kill the whole tree in one killpg.
            proc = subprocess.Popen(
                ["bash", "-c", job.cmd],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
            handle = _RunHandle(
                proc=proc,
                queue=name,
                job=job,
                gpu=slot.gpu,
                log_path=str(log_path),
                start_ts=time.time(),
            )
            with self._running_lock:
                self._running.setdefault(slot.gpu, {})[job.jobkey] = handle
            try:
                rc = proc.wait()
            finally:
                with self._running_lock:
                    self._running.get(slot.gpu, {}).pop(job.jobkey, None)
        if handle.killed:  # yielded the card: requeue, don't record a result
            store.remove_claim(self.root, name, job.jobkey)
            store.bump_attempt(self.root, name, job.jobkey)
            return None
        if rc != 0 and self._oom_requeue(name, meta, job, log_path):
            return None  # OOM: requeued behind an escalated memory gate, not a failure
        store.record_result(
            self.root,
            name,
            job,
            rc=rc,
            node=self.hostname,
            gpu=slot.gpu,
            start_utc=start,
            end_utc=now_iso(),
            log=str(log_path),
            attempt=attempt,
        )
        store.remove_claim(self.root, name, job.jobkey)
        return rc

    def yield_kill_gpu(self, gpu: int, spec: "yielding.DrainSpec | None" = None) -> None:
        """Yield ``gpu`` (called by the watchdog); kill our jobs there, or spare near-done ones.

        Only handles whose process is still alive (``poll() is None``) are considered: a job
        that already exited naturally (but whose handle is not yet popped — ``_run_job`` pops
        only after ``wait()`` returns) is left alone so its real rc is recorded normally
        instead of being discarded and requeued as if we killed it.

        ``spec is None`` (``yield_action == "kill"``) kills EVERY live job on the card — the
        historical behavior, unchanged. With a :class:`~yielding.DrainSpec`
        (``drain_if_near_done``) each live job is evaluated: a job estimated ``>= threshold``
        done is SPARED (runs to completion, records its real rc), one below threshold is killed
        and requeued exactly as before, and any spared job that has held the card past
        ``max_s`` since the yield began is killed with a reason suffix. The GPU stays marked
        yielded (that is the watchdog's marker), so no new work lands here either way.
        """
        to_kill: list[tuple[_RunHandle, str | None]] = []
        to_drain: list[tuple[_RunHandle, int]] = []
        median_cache: dict[str, float | None] = {}
        with self._running_lock:
            handles = [
                h for h in self._running.get(gpu, {}).values() if h.proc.poll() is None
            ]
            for h in handles:
                if spec is None:
                    h.killed = True
                    to_kill.append((h, None))
                    continue
                if h.queue not in median_cache:
                    median_cache[h.queue] = yielding.median_duration_s(
                        store.load_results(self.root, h.queue)
                    )
                verb, pct, reason = self._drain_decision(h, spec, median_cache[h.queue])
                if verb == "kill":
                    h.killed = True
                    to_kill.append((h, reason))
                elif not h.drained:  # spare, and log DRAIN exactly once
                    h.drained = True
                    to_drain.append((h, pct))
        for h, reason in to_kill:
            suffix = f" ({reason})" if reason else ""
            self.master.log("KILL", f"{h.queue} {h.job.key} gpu={gpu}{suffix}")
            _kill_group(h.proc)
        for h, pct in to_drain:
            self.master.log("DRAIN", f"{h.queue} {h.job.key} gpu={gpu} pct={pct} (finishing)")

    def _drain_decision(
        self, handle: _RunHandle, spec: "yielding.DrainSpec", median_s: float | None
    ) -> tuple[str, int | None, str | None]:
        """Decide a live job's fate under ``drain_if_near_done`` -> (verb, pct, reason).

        The safety cap wins first (a spared job may not keep a promised-away card forever);
        otherwise progress drives it (>= threshold spares, below kills). See
        :func:`yielding.estimate_progress` for the log-regex -> median -> 0.0 priority chain.
        """
        if spec.now - spec.yield_started_ts >= spec.max_s:
            return "kill", None, "drain cap expired"
        progress = yielding.estimate_progress(
            log_path=handle.log_path or None,
            progress_regex=spec.progress_regex,
            elapsed_s=spec.now - handle.start_ts,
            median_s=median_s,
        )
        pct = int(round(progress * 100))
        if progress >= spec.threshold:
            return "spare", pct, None
        return "kill", None, None

    # ------------------- thread loop -------------------

    def loop(self, tid: int) -> None:
        while True:
            if store.stop_path(self.root, self.hostname).exists():
                self.master.log("WAIT", f"stop file present; worker {tid} exiting")
                return
            if tid >= getattr(self, "worker_target", tid + 1):
                self.master.log("WAIT", f"worker {tid} above scaled-down target; exiting")
                return
            try:
                if self._loop_once() == "exit":
                    return
            except Exception:  # noqa: BLE001 — a runner thread must never die.
                # 2026-08-21: a policy-file KeyError raised inside claim/acquire killed 27 of
                # 48 threads (their claims stayed behind, the pool silently ran on the
                # survivors and only claimed when a job ended). Log, back off, keep looping.
                logger.exception("worker {} loop iteration crashed; continuing", tid)
                time.sleep(self.poll_s)

    def _loop_once(self) -> str | None:
        """One claim -> acquire -> run iteration. Returns "exit" when the thread should stop.

        Any claim taken here is released again before an exception can escape, so a crash
        in claim/acquire never strands a claim the way the unguarded loop did.
        """
        ready = self.ready_queues()
        for name, _ in ready:
            for jk in store.steal_stale(self.root, name, self.hostname):
                self.master.log("STEAL", f"{name} {jk}")
        claim = None
        for name, meta in ready:
            # Fall-through (2026-08-21): a claimed job whose stage cap (cap_per_gpu /
            # cap_group) is saturated on every GPU used to PARK this thread in _acquire_gpu
            # holding the claim -- lower-priority queues (evals) starved behind a capped
            # training queue. Probe the group ceiling first and skip the queue this round.
            cap, group = self._effective_cap(meta)
            has_room = getattr(self.gpu, "group_has_room", None)
            if cap is not None and has_room is not None and not has_room(group, cap):
                continue
            # Same fall-through for the CPU lane: a queue-level ``slots: 0`` job parked a
            # thread on the cpu_cap budget (probe, then one non-blocking acquire below).
            cpu_lane = int(meta.defaults.get("slots", 1)) == 0
            cpu_room = getattr(self.gpu, "cpu_has_room", None)
            if cpu_lane and cpu_room is not None and not cpu_room():
                continue
            job = store.claim_next(self.root, name, self.hostname, os.getpid())
            if job is None:
                continue
            if not self._probe_admit(name):  # tripped queue, probe slot taken: give it back
                store.remove_claim(self.root, name, job.jobkey)
                continue
            if cap is not None or (cpu_lane and self._effective_slots(meta, job) == 0):
                # The probe above races (many threads pass it before any acquires), so a
                # capped claim gets ONE non-blocking acquire: no slot -> give the job back
                # and keep walking the ready list. Capped queues never park a thread.
                try:
                    slot = self.gpu.acquire(
                        self._effective_mem(name, meta, job),
                        slots=self._effective_slots(meta, job),
                        cap_per_gpu=cap,
                        cap_group=group,
                    )
                except Exception:
                    store.remove_claim(self.root, name, job.jobkey)
                    self._probe_finish(name, None)  # never strand the probe slot either
                    raise
                if slot is None:
                    store.remove_claim(self.root, name, job.jobkey)
                    self._probe_finish(name, None)
                    continue
                claim = (name, meta, job, slot)
                break
            claim = (name, meta, job, None)
            break
        if claim is None:
            if self.all_complete():
                return "exit"
            time.sleep(self.poll_s)
            return None
        name, meta, job, slot = claim
        self.master.log("CLAIM", f"{name} {job.key}")
        if slot is None:
            try:
                slot = self._acquire_gpu(
                    self._effective_mem(name, meta, job),
                    self._effective_slots(meta, job),
                    None,
                    None,
                    job.key,
                )
            except Exception:
                store.remove_claim(self.root, name, job.jobkey)  # never strand a claim
                self._probe_finish(name, None)
                raise
        if slot is None:  # stop requested while waiting -> let a peer take it
            store.remove_claim(self.root, name, job.jobkey)
            self._probe_finish(name, None)
            return "exit"
        self._run_claimed(name, meta, job, slot)
        return None

    def _run_claimed(self, name, meta, job, slot) -> None:
            self.master.log("START", f"{name} {job.key} gpu={slot.gpu}")
            attempt = store.read_attempt(self.root, name, job.jobkey)
            try:
                rc = self._run_job(name, meta, job, slot, attempt)
                if rc is None:  # yield-killed or OOM: already logged, job requeued to pending
                    pass  # not a job failure — must not count toward the trip streak
                else:
                    self.master.log("END", f"{name} {job.key} rc={rc}")
                    self._record_outcome(name, rc)
                self._probe_finish(name, rc)
            except Exception:  # noqa: BLE001 — log-and-continue; never abort the pool
                logger.exception("job {} in queue {} crashed the runner", job.key, name)
                store.remove_claim(self.root, name, job.jobkey)
                self._record_outcome(name, 1)
                self._probe_finish(name, 1)
            finally:
                slot.release()
            if store.queue_complete(self.root, name):
                self._maybe_log_complete(name)


def run_pool(
    root: Path,
    *,
    workers: int = 1,
    queues: list[str] | None = None,
    gpu: GpuManager | None = None,
    hostname: str | None = None,
    poll_s: float = 20.0,
    gpu_wait_s: float = 5.0,
    lock_prefix: str | None = None,
    fail_streak_limit: int = DEFAULT_FAIL_STREAK_LIMIT,
    supervise_s: float = 30.0,
) -> None:
    """Run the foreground worker pool until every ready queue is complete (or stopped).

    Args:
        lock_prefix: Optional override for GPU slot flock paths (default
            ``/tmp/subliminal_jobq_<user>``).
        fail_streak_limit: Consecutive failures that trip a queue; the pool then
            skips it until a requeue grows its pending count.
    """
    # World-writable outputs, unconditionally. The stash under
    # /mnt/ssd-2/soar-subliminal/angela is shared by BOTH nodes, where angela has
    # different uids/gids (1002 on -0, 1001 on -1). A default umask of 002 would
    # birth job outputs as 664/775 — not writable by the *other* node (it is neither
    # owner nor in the matching group), so the tree slowly drifts un-shareable. umask
    # 0 makes every file this pool's subprocesses create 666/777 -> always cross-node
    # writable, independent of how the worker was launched. See fix_perms_worldrw.sh
    # for the one-time backfill of files created before this landed.
    os.umask(0)
    root = Path(root)
    hostname = hostname or store.this_host()
    _acquire_pid_file(root, hostname)
    master = MasterLog(root / "logs" / f"{utc_stamp()}_worker.{hostname}.log")
    gpu = gpu or GpuManager(
        root,
        hostname,
        lock_prefix=lock_prefix or DEFAULT_LOCK_PREFIX,
    )
    # Opt-in CUDA MPS: bring the per-user daemon up ONCE before workers spawn (a shared,
    # user-level resource). Only inject its env into jobs if it actually came up; a missing
    # binary / failed probe falls back silently so the pool runs exactly as it did pre-MPS.
    # We deliberately do NOT tear the daemon down at pool exit — other pools/users may share it.
    active_mps_env: dict[str, str] = {}
    try:
        _mps_policy = load_policy(root, hostname)
    except FileNotFoundError:
        _mps_policy = None
    if _mps_policy is not None and _mps_policy.mps:
        if ensure_mps_daemon(_mps_policy):
            active_mps_env = mps_env(_mps_policy)
            logger.info("MPS enabled (daemon up)")
        else:
            logger.warning("MPS requested but daemon unavailable; running without MPS")
    pool = _Pool(
        root,
        hostname,
        gpu,
        master,
        queues_filter=set(queues) if queues else None,
        poll_s=poll_s,
        gpu_wait_s=gpu_wait_s,
        fail_streak_limit=fail_streak_limit,
        mps_env=active_mps_env,
    )
    # Opt-in GPU yielding: ALWAYS start the watchdog when this node has a policy file, and
    # let poll_once no-op while yield_to_foreign is false. This makes the switch hot-ENABLE:
    # flip the flag in the policy JSON on a running pool and yielding begins within one
    # yield_poll_s — no restart (a restart here would kill+requeue every running job, exactly
    # the pain this feature manages). A missing policy (e.g. tests with a fake GpuManager)
    # means no thread; if one exists, poll_once returns harmlessly on FileNotFoundError.
    stop_event = threading.Event()
    watch_thread: threading.Thread | None = None
    try:
        load_policy(root, hostname)
        have_policy = True
    except FileNotFoundError:
        have_policy = False
    if have_policy:
        watchdog = yielding.YieldWatchdog(
            root,
            hostname,
            load_policy=lambda: load_policy(root, hostname),
            master=master,
            on_yield=pool.yield_kill_gpu,
        )
        watch_thread = threading.Thread(
            target=watchdog.run, args=(stop_event,), name="jobq-yield-watchdog", daemon=True
        )
        watch_thread.start()
    _supervise_workers(pool, master, root, hostname, workers, supervise_s)
    stop_event.set()  # let the daemon watchdog exit promptly (it also dies at process end)
    if watch_thread is not None:
        watch_thread.join(timeout=5.0)
    master.log("ALL QUEUES COMPLETE")
    store.worker_pid_path(root, hostname).unlink(missing_ok=True)


def workers_path(root: Path, hostname: str) -> Path:
    """Optional hot-scale file: an integer thread target for this node's running pool."""
    return Path(root) / f"workers.{hostname}"


def _read_workers_target(root: Path, hostname: str, default: int) -> int:
    try:
        return max(1, int(workers_path(root, hostname).read_text().strip()))
    except (FileNotFoundError, ValueError, OSError):
        return default


def _supervise_workers(
    pool: _Pool, master: MasterLog, root: Path, hostname: str, workers: int, supervise_s: float
) -> None:
    """Run worker threads under a supervisor: respawn dead ones, hot-scale from a file.

    2026-08-21: an exception escaping ``loop()`` silently killed threads and two pools ran
    for hours at 20/48 and 12/40 with no restart path short of a full drain. The main thread
    used to just ``join()``; now every ``supervise_s`` it (a) respawns any thread that ended
    WITHOUT returning normally (crash) and logs ``RESPAWN``, and (b) re-reads
    ``workers.<host>`` — write an integer there to grow or shrink the live pool without a
    restart (shrink = threads above the target exit after their current job). Threads that
    return normally (stop file / all queues complete) are not respawned; the supervisor
    returns once every live thread has done so. Cost: ``is_alive`` on ≤N Thread objects +
    one stat() per tick.
    """
    finished: set[int] = set()  # tids that returned normally
    threads: dict[int, threading.Thread] = {}

    def _target(tid: int) -> None:
        pool.loop(tid)
        finished.add(tid)

    def _spawn(tid: int) -> None:
        t = threading.Thread(target=_target, args=(tid,), name=f"jobq-worker-{tid}")
        threads[tid] = t
        t.start()

    target = max(1, workers)
    pool.worker_target = target
    for i in range(target):
        _spawn(i)
    while True:
        time.sleep(supervise_s)
        new_target = _read_workers_target(root, hostname, target)
        if new_target != target:
            master.log("SCALE", f"worker target {target} -> {new_target} (workers.{hostname})")
            target = new_target
            pool.worker_target = target
        for tid in range(target):
            t = threads.get(tid)
            if t is None:
                _spawn(tid)
            elif not t.is_alive() and tid not in finished:
                master.log("RESPAWN", f"worker {tid} died without returning; respawning")
                _spawn(tid)
        alive = [t for t in threads.values() if t.is_alive()]
        if not alive and finished:
            return
