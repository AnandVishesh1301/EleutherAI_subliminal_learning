"""The threaded worker pool: claim -> acquire GPU -> run -> record, log-and-continue.

One process, ``--workers N`` threads, all sharing one :class:`GpuManager` (so the per-GPU
slot cap is global). Each thread loops over the READY queues (node filter + deps satisfied,
ordered by priority desc then created_utc), steals its own node's stale claims, claims the
first pending job, runs it, records the result, and moves on. A job failure never aborts the
pool. Between jobs it honors the ``stop.<hostname>`` file.

Master-log verbs (``[<iso-utc>] VERB ...``): CLAIM, START, END rc=<rc>, STEAL, WAIT,
QUEUE COMPLETE <name>, and a final ALL QUEUES COMPLETE. The opt-in GPU-yielding framework
(``jobqueue/yielding.py``) adds YIELD gpu=N foreign=K, KILL <queue> <key> gpu=N,
DRAIN <queue> <key> gpu=N pct=P (finishing) (a near-done job spared under
drain_if_near_done), and RECLAIM gpu=N.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.jobqueue import store, yielding
from subliminal_learning.jobqueue.gpu import GpuManager, load_policy
from subliminal_learning.jobqueue.store import (
    QueueMeta,
    REPO_ROOT,
    now_iso,
    utc_stamp,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils.io import atomic_write_text


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
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.gpu = gpu
        self.master = master
        self.queues_filter = queues_filter
        self.poll_s = poll_s
        self.gpu_wait_s = gpu_wait_s
        self._lock = threading.Lock()
        self._complete_logged: set[str] = set()
        self._dep_warned: set[str] = set()
        # Live jobs by physical GPU so the yield watchdog can kill this card's jobs:
        # {gpu: {jobkey: _RunHandle}}. Guarded by its own lock (touched from worker
        # threads + the watchdog thread).
        self._running_lock = threading.Lock()
        self._running: dict[int, dict[str, _RunHandle]] = {}

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
            if not self._deps_ok(meta):
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

    def _run_job(
        self, name: str, meta: QueueMeta, job: Job, slot, attempt: int
    ) -> int | None:
        """Run one job to completion. Returns its rc, or ``None`` if a yield killed it.

        A yield-killed job is requeued to PENDING (claim removed, NO result recorded,
        attempt bumped) so it retries later — it is never recorded as failed.
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
        env.update({str(k): str(v) for k, v in (meta.defaults.get("env") or {}).items()})
        env.update(job.env or {})
        env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
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
                if rc is None:  # yield-killed: KILL already logged, job requeued to pending
                    pass
                else:
                    self.master.log("END", f"{name} {job.key} rc={rc}")
            except Exception:  # noqa: BLE001 — log-and-continue; never abort the pool
                logger.exception("job {} in queue {} crashed the runner", job.key, name)
                store.remove_claim(self.root, name, job.jobkey)
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
) -> None:
    """Run the foreground worker pool until every ready queue is complete (or stopped)."""
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
    gpu = gpu or GpuManager(root, hostname)
    pool = _Pool(
        root,
        hostname,
        gpu,
        master,
        queues_filter=set(queues) if queues else None,
        poll_s=poll_s,
        gpu_wait_s=gpu_wait_s,
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
    threads = [
        threading.Thread(target=pool.loop, args=(i,), name=f"jobq-worker-{i}")
        for i in range(max(1, workers))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_event.set()  # let the daemon watchdog exit promptly (it also dies at process end)
    if watch_thread is not None:
        watch_thread.join(timeout=5.0)
    master.log("ALL QUEUES COMPLETE")
    store.worker_pid_path(root, hostname).unlink(missing_ok=True)
