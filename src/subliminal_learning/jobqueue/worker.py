"""The threaded worker pool: claim -> acquire GPU -> run -> record, log-and-continue.

One process, ``--workers N`` threads, all sharing one :class:`GpuManager` (so the per-GPU
slot cap is global). Each thread loops over the READY queues (node filter + deps satisfied,
ordered by priority desc then created_utc), steals its own node's stale claims, claims the
first pending job, runs it, records the result, and moves on. A job failure never aborts the
pool. Between jobs it honors the ``stop.<hostname>`` file.

Master-log verbs (``[<iso-utc>] VERB ...``): CLAIM, START, END rc=<rc>, STEAL, WAIT,
TRIPPED <name>, QUEUE COMPLETE <name>, and a final ALL QUEUES COMPLETE.

A queue whose jobs fail back-to-back is *tripped* rather than drained: a job that
dies during startup (bad config, provenance guard, missing checkpoint) fails in
seconds and deterministically, so an untripped pool converts one broken job into a
whole burnt queue in minutes. The pool keeps serving other queues and re-admits a
tripped one as soon as its pending count grows (i.e. after a requeue).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from loguru import logger

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import DEFAULT_LOCK_PREFIX, GpuManager
from subliminal_learning.jobqueue.store import (
    QueueMeta,
    REPO_ROOT,
    now_iso,
    utc_stamp,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils.io import atomic_write_text

DEFAULT_FAIL_STREAK_LIMIT = 5


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
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.gpu = gpu
        self.master = master
        self.queues_filter = queues_filter
        self.poll_s = poll_s
        self.gpu_wait_s = gpu_wait_s
        self.fail_streak_limit = fail_streak_limit
        self._lock = threading.Lock()
        self._complete_logged: set[str] = set()
        self._dep_warned: set[str] = set()
        self._fail_streak: dict[str, int] = {}
        self._tripped_pending: dict[str, int] = {}

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
        self.master.log(
            "TRIPPED",
            f"{name} — {self.fail_streak_limit} consecutive job failures; not "
            "claiming from it again until its pending count grows. Inspect the "
            f"newest log under {store.queue_logs_dir(self.root, name)}, fix the "
            f"cause, then `jobq requeue --root {self.root} --name {name}`.",
        )

    def _is_tripped(self, name: str) -> bool:
        """Whether `name` is still tripped; a requeue (more pending) re-admits it."""
        with self._lock:
            at_trip = self._tripped_pending.get(name)
            if at_trip is None:
                return False
            if store.queue_status(self.root, name)["pending"] <= at_trip:
                return True
            del self._tripped_pending[name]
            self._fail_streak[name] = 0
        self.master.log("WAIT", f"{name} re-admitted after requeue")
        return False

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

    def _effective_mem(self, meta: QueueMeta, job: Job) -> int:
        if job.mem_mib is not None:
            return job.mem_mib
        d = meta.defaults.get("mem_mib")
        if d is not None:
            return int(d)
        return self.gpu.default_mem_mib()

    def _effective_slots(self, meta: QueueMeta, job: Job) -> int:
        """The job's weight in global slot-units (job field > queue default > 1)."""
        if job.slots is not None:
            return job.slots
        return int(meta.defaults.get("slots", 1))

    def _effective_cap(self, meta: QueueMeta) -> tuple[int | None, str | None]:
        """Queue-level stage ceiling: (cap_per_gpu, cap_group) from meta.defaults.

        ``cap_group`` defaults to the queue name so each capped queue gets its own ceiling;
        set the same ``cap_group`` on several queues to co-cap them (the old shared
        ``LOCK_PREFIX`` pattern).
        """
        cap = meta.defaults.get("cap_per_gpu")
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
                self.master.log(
                    "WAIT",
                    f"{job_key} needs {mem_mib} MiB free + {slots} slot-unit(s); "
                    f"~{waited_min}min so far"
                    + (
                        " — check mem_mib/slots against this node's gpu_policy"
                        if waited_min >= 10
                        else ""
                    ),
                )
            time.sleep(self.gpu_wait_s)

    def _run_job(self, name: str, meta: QueueMeta, job: Job, slot, attempt: int) -> int:
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
        env.update({str(k): str(v) for k, v in (meta.defaults.get("env") or {}).items()})
        env.update(job.env or {})
        env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        cwd = job.cwd or str(REPO_ROOT)
        with open(log_path, "w") as f:
            f.write(
                f"=== node={self.hostname} pid={pid} gpu={slot.gpu} "
                f"queue={name} job={job.key} start={start} ===\n"
            )
            f.flush()
            proc = subprocess.run(
                ["bash", "-c", job.cmd],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=cwd,
            )
        store.record_result(
            self.root,
            name,
            job,
            rc=proc.returncode,
            node=self.hostname,
            gpu=slot.gpu,
            start_utc=start,
            end_utc=now_iso(),
            log=str(log_path),
            attempt=attempt,
        )
        store.remove_claim(self.root, name, job.jobkey)
        return proc.returncode

    # ------------------- thread loop -------------------

    def loop(self, tid: int) -> None:
        while True:
            if store.stop_path(self.root, self.hostname).exists():
                self.master.log("WAIT", f"stop file present; worker {tid} exiting")
                return
            ready = self.ready_queues()
            for name, _ in ready:
                for jk in store.steal_stale(self.root, name, self.hostname):
                    self.master.log("STEAL", f"{name} {jk}")
            claim = None
            for name, meta in ready:
                job = store.claim_next(self.root, name, self.hostname, os.getpid())
                if job is not None:
                    claim = (name, meta, job)
                    break
            if claim is None:
                if self.all_complete():
                    return
                time.sleep(self.poll_s)
                continue
            name, meta, job = claim
            self.master.log("CLAIM", f"{name} {job.key}")
            cap, group = self._effective_cap(meta)
            slot = self._acquire_gpu(
                self._effective_mem(meta, job),
                self._effective_slots(meta, job),
                cap,
                group,
                job.key,
            )
            if slot is None:  # stop requested while waiting -> let a peer take it
                store.remove_claim(self.root, name, job.jobkey)
                return
            self.master.log("START", f"{name} {job.key} gpu={slot.gpu}")
            attempt = store.read_attempt(self.root, name, job.jobkey)
            try:
                rc = self._run_job(name, meta, job, slot, attempt)
                self.master.log("END", f"{name} {job.key} rc={rc}")
                self._record_outcome(name, rc)
            except Exception:  # noqa: BLE001 — log-and-continue; never abort the pool
                logger.exception("job {} in queue {} crashed the runner", job.key, name)
                store.remove_claim(self.root, name, job.jobkey)
                self._record_outcome(name, 1)
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
) -> None:
    """Run the foreground worker pool until every ready queue is complete (or stopped).

    Args:
        lock_prefix: Optional override for GPU slot flock paths (default
            ``/tmp/subliminal_jobq_<user>``).
        fail_streak_limit: Consecutive failures that trip a queue; the pool then
            skips it until a requeue grows its pending count.
    """
    root = Path(root)
    hostname = hostname or store.this_host()
    _acquire_pid_file(root, hostname)
    master = MasterLog(root / "logs" / f"{utc_stamp()}_worker.{hostname}.log")
    gpu = gpu or GpuManager(
        root,
        hostname,
        lock_prefix=lock_prefix or DEFAULT_LOCK_PREFIX,
    )
    pool = _Pool(
        root,
        hostname,
        gpu,
        master,
        queues_filter=set(queues) if queues else None,
        poll_s=poll_s,
        gpu_wait_s=gpu_wait_s,
        fail_streak_limit=fail_streak_limit,
    )
    threads = [
        threading.Thread(target=pool.loop, args=(i,), name=f"jobq-worker-{i}")
        for i in range(max(1, workers))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    master.log("ALL QUEUES COMPLETE")
    store.worker_pid_path(root, hostname).unlink(missing_ok=True)
