"""CLI: run a jobs file inside ONE Slurm allocation, packing several jobs per GPU by free memory.

Slurm here has no gres/shard or MPS, so one `--gres=gpu:N` allocation is N whole GPUs; small
jobs (a pythia-160m teacher, a 100k-prompt number pool, a student distillation) use a fraction
of an A40, and running them one per GPU wastes most of the card. This runner is the packer:
it reads a jobq-style jobs file (JSONL rows ``{"key", "cmd", "mem_mib"?, "env"?, "priority"?}`` or plain
command lines; higher ``priority`` first, the file is re-read live when it changes), keeps every visible GPU busy with as many jobs as fit, and admits a job to a
GPU only when that GPU's free memory (``nvidia-smi``) minus ``--reserve-mib`` covers the job's
``mem_mib`` AND fewer than ``--max-per-gpu`` jobs of this runner are already on it. The cap is
a safety rail; memory is the real gate.

Each job runs with ``CUDA_VISIBLE_DEVICES=<one gpu>`` from the allocation's cwd; stdout/stderr
go to ``{log_dir}/{key}.log`` and one JSONL row per finished job (key, gpu, rc, seconds) is
appended to ``{log_dir}/results.jsonl``. A job that fails never blocks the others; the runner
exits 1 at the end if any job failed. Already-finished keys (rc 0 in results.jsonl) are
skipped, so re-submitting the same sbatch after a time limit resumes where it stopped.

Usage inside an sbatch script (one allocation of e.g. 8 GPUs, 5 jobs per GPU):

    uv run python -m scripts.ops.slurm_pack --jobs-file jobs.jsonl --log-dir logs/stage1 \\
        --max-per-gpu 5 --default-mem-mib 6000

**Shared queue (`--claim-dir`)**: several allocations (any number of one-GPU sbatches, submitted
at any time) run the SAME jobs file and take work whenever one of their GPUs has room, instead of
each owning a fixed slice. A worker takes a job by atomically creating ``{claim_dir}/claims/{key}``
(``O_EXCL`` on the shared filesystem; whoever creates it runs the job), touches that file every
tick while the job runs (heartbeat), and on exit writes ``{claim_dir}/done/{key}`` (rc 0) or
``{claim_dir}/failed/{key}`` (rc != 0). Every worker skips keys with a done or failed marker or a
live claim; a claim whose heartbeat is older than ``--stale-s`` with no marker is a dead worker's
(time limit, preemption, crash) and is reclaimed -- the leaf itself is left to the job's own
resume/clean-up logic. A worker exits as soon as nothing is left for it to take, freeing its GPU.
Failed keys are never retried automatically: delete their ``failed/`` markers (and the leaf if it
is torn) and resubmit a worker. Results rows go to ``{log_dir}/results.{worker}.jsonl``.

**Several campaigns, one worker** (repeat ``--jobs-file/--claim-dir/--log-dir`` in matching order): the
worker merges every file into one pending list (``priority`` first, then file order, then row order) and
takes any job of any campaign that fits, claiming it in THAT campaign's claim dir and logging to its log
dir, so a GPU never drains at one third capacity on a finishing campaign while another is pending.
Each file is live-reloaded on its own mtime. ``scripts.ops.queue_sources`` prints these triples for the
campaigns in the health config that still have work.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import typer
from loguru import logger

from subliminal_learning.utils.logging_setup import configure_loguru


IN_FLIGHT_RC = 75          # EX_TEMPFAIL: the job yielded (SIGTERM -> full-state checkpoint); not a failure
TERM_GRACE_S = 8.0         # Slurm KillWait here is 10 s: forward SIGTERM, then wait this long for the saves


class _Interrupted(Exception):
    """SIGTERM reached the packer (Slurm preemption / time limit / scancel --signal=TERM)."""


@dataclass
class Source:
    """One campaign's queue: its jobs file, log dir and (shared mode) claim queue."""
    jobs_file: Path
    log_dir: Path
    queue: "ClaimQueue | None" = None
    mtime: float | None = None

    @property
    def results(self) -> Path:
        return self.log_dir / ("results.jsonl" if self.queue is None else f"results.{safe_name(self.queue.worker)}.jsonl")


@dataclass(frozen=True)
class Job:
    key: str
    cmd: str
    mem_mib: int
    env: dict[str, str] = field(default_factory=dict)
    priority: int = 0      # higher runs first; ties keep file order
    src: "Source | None" = field(default=None, compare=False, repr=False)


def parse_jobs(path: Path, default_mem_mib: int, src: "Source | None" = None) -> list[Job]:
    """Jobs from a JSONL file (``key``/``cmd``/optional ``mem_mib``/``env``) or plain lines.

    A plain line's key is its 1-based line number. Blank lines and ``#`` comments are skipped.
    """
    jobs: list[Job] = []
    for i, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            row = json.loads(line)
            jobs.append(Job(
                key=str(row["key"]), cmd=str(row["cmd"]),
                mem_mib=int(row.get("mem_mib") or default_mem_mib),
                env={str(k): str(v) for k, v in (row.get("env") or {}).items()},
                priority=int(row.get("priority") or 0), src=src,
            ))
        else:
            jobs.append(Job(key=f"line{i}", cmd=line, mem_mib=default_mem_mib, src=src))
    keys = [j.key for j in jobs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate job keys: {dupes}")
    return sorted(jobs, key=lambda j: -j.priority)   # stable: ties keep file order


def merge_sources(sources: list[Source], default_mem_mib: int) -> list[Job]:
    """All sources' jobs as one list: priority first, then source order, then row order (a missing
    jobs file contributes nothing; its mtime is remembered as None so it is picked up when it appears)."""
    merged: list[Job] = []
    for src in sources:
        if src.jobs_file.exists():
            src.mtime = src.jobs_file.stat().st_mtime
            merged.extend(parse_jobs(src.jobs_file, default_mem_mib, src))
        else:
            src.mtime = None
    return sorted(merged, key=lambda j: -j.priority)


def visible_gpus() -> list[str]:
    """The GPU ids this allocation may use: ``CUDA_VISIBLE_DEVICES`` (Slurm sets it), else all."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        return [g.strip() for g in env.split(",") if g.strip()]
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def free_mib_by_gpu(gpus: list[str]) -> dict[str, int]:
    """Free memory per visible GPU from ``nvidia-smi``.

    `gpus` entries are whatever ``CUDA_VISIBLE_DEVICES`` holds: nvidia-smi indices or GPU
    UUIDs (Slurm exports UUIDs). Each is matched to its nvidia-smi row by index or by UUID;
    an id that matches neither falls back to its position, else to 0 MiB free (never admits).
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = [tuple(x.strip() for x in ln.split(",")) for ln in out.splitlines() if ln.strip()]
    by_key = {}
    for idx, uuid, free in rows:
        by_key[idx] = by_key[uuid] = int(float(free))
    free = {}
    for i, g in enumerate(gpus):
        if g in by_key:
            free[g] = by_key[g]
        else:
            free[g] = int(float(rows[i][2])) if i < len(rows) else 0
    return free


def finished_keys(results: Path) -> set[str]:
    """Keys that already finished with rc 0 in `results` or its sibling ``results.*.jsonl`` files."""
    done = set()
    files = [results] if results.exists() else []
    files += sorted(p for p in results.parent.glob(results.stem + ".*" + results.suffix) if p.is_file()) \
        if results.parent.exists() else []
    for path in files:
        for ln in path.read_text().splitlines():
            if ln.strip():
                row = json.loads(ln)
                if row.get("rc") == 0:
                    done.add(row["key"])
    return done


def safe_name(key: str) -> str:
    return key.replace("/", "_").replace("|", "_")


def worker_id() -> str:
    return f"{os.uname().nodename}:{os.environ.get('SLURM_JOB_ID') or os.getpid()}"


class ClaimQueue:
    """The shared-queue markers under `claim_dir` (see module docstring).

    ``claims/{key}`` = taken (its mtime is the owner's heartbeat), ``done/{key}``,
    ``failed/{key}`` = terminal. All operations are single atomic filesystem calls, so any
    number of workers on any nodes can share one directory.
    """

    def __init__(self, claim_dir: Path, stale_s: float, worker: str):
        self.dir, self.stale_s, self.worker = claim_dir, stale_s, worker
        for sub in ("claims", "done", "failed", "stale"):
            (claim_dir / sub).mkdir(parents=True, exist_ok=True)

    def _p(self, sub: str, key: str) -> Path:
        return self.dir / sub / safe_name(key)

    def terminal(self, key: str) -> str | None:
        """'done' / 'failed' when the key has a terminal marker, else None."""
        for sub in ("done", "failed"):
            if self._p(sub, key).exists():
                return sub
        return None

    def claim_state(self, key: str) -> str:
        """'free', 'live' (someone's fresh heartbeat) or 'stale' (heartbeat older than stale_s)."""
        p = self._p("claims", key)
        try:
            age = time.time() - p.stat().st_mtime
        except FileNotFoundError:
            return "free"
        return "stale" if age > self.stale_s else "live"

    def try_claim(self, key: str) -> bool:
        """Take `key` if free, or reclaim it if its claim is stale. True iff this worker owns it now."""
        p = self._p("claims", key)
        if self.claim_state(key) == "stale":
            # Atomic hand-off: exactly one worker's replace() succeeds; the rest see ENOENT.
            try:
                os.replace(p, self.dir / "stale" / f"{safe_name(key)}.{int(time.time())}.{os.getpid()}")
                logger.warning("reclaiming stale claim on {} (heartbeat > {}s old)", key, self.stale_s)
            except FileNotFoundError:
                return False
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"worker": self.worker, "started": time.time()}) + "\n")
        return True

    def heartbeat(self, key: str) -> None:
        try:
            os.utime(self._p("claims", key), None)
        except FileNotFoundError:  # reclaimed under us (should not happen while we beat) -- recreate
            self._p("claims", key).write_text(json.dumps({"worker": self.worker, "started": time.time()}) + "\n")

    def release(self, key: str) -> None:
        """Drop this worker's claim without a terminal marker (the job yielded or the worker is dying)."""
        try:
            self._p("claims", key).unlink()
        except FileNotFoundError:
            pass

    def finish(self, key: str, rc: int, seconds: float) -> None:
        sub = "done" if rc == 0 else "failed"
        self._p(sub, key).write_text(json.dumps({"worker": self.worker, "rc": rc, "seconds": seconds}) + "\n")
        try:
            self._p("claims", key).unlink()
        except FileNotFoundError:
            pass


def pick_gpu(
    job: Job, free: dict[str, int], running: dict[str, int], max_per_gpu: int, reserve_mib: int,
) -> str | None:
    """The least-loaded GPU that can take `job` now, or None.

    Admission: fewer than `max_per_gpu` of OUR jobs on it, and free memory minus the reserve
    covers the job's ask. Among candidates, prefer the one with the most free memory.
    """
    ok = [g for g in free if running.get(g, 0) < max_per_gpu and free[g] - reserve_mib >= job.mem_mib]
    if not ok:
        return None
    return max(ok, key=lambda g: (free[g], -running.get(g, 0)))


def run_pack(
    jobs: list[Job], gpus: list[str], log_dir: Path, *, max_per_gpu: int, reserve_mib: int,
    poll_s: float, settle_s: float, free_fn=free_mib_by_gpu, queue: ClaimQueue | None = None,
    linger_s: float = 0.0, admit_until_s: float | None = None, jobs_file: Path | None = None,
    default_mem_mib: int = 6000, drain_file: Path | None = None, sources: list[Source] | None = None,
) -> int:
    """Drive the jobs to completion; returns the number of failed jobs (of THIS worker).

    Without `queue` the worker owns the whole list (fixed slice; rc-0 keys in results*.jsonl
    are skipped). With `queue` (shared mode) it takes any job that fits a free GPU and is not
    done, failed or live-claimed by another worker; when nothing is left to take it waits up to
    `linger_s` for other workers' claims to go stale (default 0: exit at once, free the GPU).
    `admit_until_s`: stop admitting NEW jobs this many seconds after start (an sbatch near its
    time limit should not start a job it cannot finish); running jobs are still waited for.
    `jobs_file`: when given, the file is re-read whenever its mtime changes, so rows added,
    removed or re-prioritised (the `priority` field, higher first) take effect on the next
    admission without restarting the worker. `drain_file`: while this path does NOT exist, an
    empty pending list is not the end -- the worker keeps polling for new rows (a producer is
    still appending to the jobs file); once the file exists, empty pending means done.
    `sources` (several campaigns): `jobs` is their merged list (`merge_sources`); each job's claim
    queue, log dir and results file are its source's, and every source's file is live-reloaded.
    """
    sources = sources or []
    for src in sources:
        src.log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results = log_dir / ("results.jsonl" if queue is None else f"results.{safe_name(queue.worker)}.jsonl")

    def _q(job: Job) -> ClaimQueue | None:
        return job.src.queue if job.src is not None else queue

    def _log(job: Job) -> Path:
        return (job.src.log_dir if job.src is not None else log_dir) / f"{safe_name(job.key)}.log"

    def _results_of(job: Job) -> Path:
        return job.src.results if job.src is not None else results

    def _done_keys(job: Job) -> set[str]:
        return finished_keys((job.src.log_dir if job.src is not None else log_dir) / "results.jsonl")

    done = finished_keys(log_dir / "results.jsonl") if not sources else set()
    pending = [j for j in jobs if j.key not in (done if not sources else _done_keys(j))]
    if done:
        logger.info("resume: {} of {} keys already finished (rc 0), {} to run", len(done), len(jobs), len(pending))
    running: dict[str, tuple[Job, subprocess.Popen, float, object]] = {}
    per_gpu: dict[str, int] = {g: 0 for g in gpus}
    failed = 0
    ran = 0
    last_start = 0.0
    t_start = time.time()
    closed = False
    idle_since: float | None = None
    got_term = {"flag": False}
    jobs_mtime = jobs_file.stat().st_mtime if jobs_file is not None and jobs_file.exists() else None

    def _on_term(*_a):
        got_term["flag"] = True

    import threading
    in_main = threading.current_thread() is threading.main_thread()
    prev_term = signal.signal(signal.SIGTERM, _on_term) if in_main else None   # signals need the main thread

    def _record(key, gpu, rc, seconds, job: Job | None = None):
        q = _q(job) if job is not None else queue
        with (_results_of(job) if job is not None else results).open("a") as f:
            f.write(json.dumps({"key": key, "gpu": gpu, "rc": rc, "seconds": seconds,
                                **({"worker": q.worker} if q else {})}) + "\n")

    def _interrupt():
        """Forward SIGTERM to every running job, wait TERM_GRACE_S, record rows, RELEASE claims."""
        logger.warning("SIGTERM: forwarding to {} running job(s) and releasing their claims", len(running))
        for key, (job, proc, t0, fh) in running.items():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        deadline = time.time() + TERM_GRACE_S
        for key, (job, proc, t0, fh) in list(running.items()):
            try:
                rc = proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()        # Slurm's SIGKILL would land now anyway; the leaf keeps its last checkpoint
                rc = None
            fh.close()
            _record(key, proc.gpu, rc if rc is not None else -15, round(time.time() - t0, 1), job)  # type: ignore[attr-defined]
            if _q(job) is not None:
                _q(job).release(key)
            logger.warning("released {} (rc {}) -- resumable by the next worker", key, rc)
        if in_main:
            signal.signal(signal.SIGTERM, prev_term)
        raise _Interrupted

    while True:
        if got_term["flag"]:
            _interrupt()
        # reap
        for key, (job, proc, t0, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            seconds = round(time.time() - t0, 1)
            gpu = proc.gpu  # type: ignore[attr-defined]
            per_gpu[gpu] -= 1
            del running[key]
            _record(key, gpu, rc, seconds, job)
            jq = _q(job)
            if rc == IN_FLIGHT_RC:
                # the job yielded on its own SIGTERM: its leaf holds a resumable checkpoint; no marker,
                # the claim is released so any worker (this one included, later) can resume it
                if jq is not None:
                    jq.release(key)
                else:
                    pending.append(job)
                logger.warning("yield {} on gpu {} after {}s (rc {}) -- left resumable", key, gpu, seconds, rc)
                continue
            if jq is not None:
                jq.finish(key, rc, seconds)
            if rc == 0:
                logger.info("done  {} on gpu {} in {}s", key, gpu, seconds)
            else:
                failed += 1
                logger.error("FAILED {} on gpu {} rc={} after {}s (see {})", key, gpu, rc, seconds, _log(job))
        for key, (job, _p, _t, _f) in running.items():
            if _q(job) is not None:
                _q(job).heartbeat(key)
        # live reload: pick up priority changes / new rows without a restart
        if sources and not closed:
            changed = False
            for src in sources:
                try:
                    mtime = src.jobs_file.stat().st_mtime
                except FileNotFoundError:
                    mtime = None
                if mtime is not None and mtime != src.mtime:
                    changed = True
            if changed:
                try:
                    fresh = merge_sources(sources, default_mem_mib)
                except (ValueError, OSError, json.JSONDecodeError) as e:
                    logger.warning("jobs files reload failed ({}); keeping the current list", e)
                else:
                    pending = [j for j in fresh if j.key not in running and j.key not in _done_keys(j)]
                    jobs = fresh
                    logger.info("jobs files reloaded: {} rows over {} sources, {} pending here", len(fresh), len(sources), len(pending))
        elif jobs_file is not None and not closed:
            try:
                mtime = jobs_file.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            if mtime is not None and mtime != jobs_mtime:
                jobs_mtime = mtime
                try:
                    fresh = parse_jobs(jobs_file, default_mem_mib)
                except (ValueError, OSError, json.JSONDecodeError) as e:
                    logger.warning("jobs file reload failed ({}); keeping the current list", e)
                else:
                    ran_or_live = set(running) | finished_keys(log_dir / "results.jsonl")
                    pending = [j for j in fresh if j.key not in ran_or_live]
                    jobs = fresh
                    logger.info("jobs file reloaded: {} rows, {} pending here", len(fresh), len(pending))
        # admit (one at a time, with a settle delay so the new process's memory shows up
        # in nvidia-smi before the next admission decision)
        if pending and admit_until_s is not None and time.time() - t_start >= admit_until_s:
            if not closed:
                logger.info("admission closed after {}s; {} jobs left for other workers", int(admit_until_s), len(pending))
                closed = True
            pending = []  # leave them to other (or later) workers; fixed mode: resume picks them up
        if pending and time.time() - last_start >= settle_s:
            free = free_fn(gpus)
            started = False
            for job in list(pending):
                jq = _q(job)
                if jq is not None:
                    term = jq.terminal(job.key)
                    if term:
                        pending.remove(job)
                        logger.info("skip  {} ({} by another worker)", job.key, term)
                        continue
                    if jq.claim_state(job.key) == "live":
                        continue  # someone else is running it; revisit if their heartbeat dies
                gpu = pick_gpu(job, free, per_gpu, max_per_gpu, reserve_mib)
                if gpu is None:
                    continue  # does not fit right now; a smaller job behind it may
                if jq is not None and not jq.try_claim(job.key):
                    continue  # lost the race for this key
                pending.remove(job)
                fh = _log(job).open("a")
                env = {**os.environ, **job.env, "CUDA_VISIBLE_DEVICES": gpu}
                fh.write(f"=== key={job.key} gpu={gpu} worker={jq.worker if jq else '-'} "
                         f"start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n$ {job.cmd}\n")
                fh.flush()
                proc = subprocess.Popen(job.cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT, env=env)
                proc.gpu = gpu  # type: ignore[attr-defined]
                running[job.key] = (job, proc, time.time(), fh)
                per_gpu[gpu] += 1
                ran += 1
                last_start = time.time()
                idle_since = None
                logger.info("start {} on gpu {} (ask {} MiB, free {} MiB, {} on gpu, {} pending)",
                            job.key, gpu, job.mem_mib, free[gpu], per_gpu[gpu], len(pending))
                started = True
                break
            if started:
                continue  # try the next job right away if it fits
        if not running:
            if not pending:
                if drain_file is not None and not drain_file.exists():
                    time.sleep(poll_s)       # producer still running: wait for new rows
                    continue
                break
            if (queue is not None or sources) and all(_q(j) is not None and (_q(j).claim_state(j.key) != "free" or _q(j).terminal(j.key))
                                                      for j in pending) \
                    and not any(_q(j) is not None and _q(j).claim_state(j.key) == "stale" and not _q(j).terminal(j.key) for j in pending):
                idle_since = idle_since or time.time()
                if time.time() - idle_since >= linger_s:
                    logger.info("{} jobs held by other live workers; nothing left to take -- exiting", len(pending))
                    break
        time.sleep(poll_s)
    if in_main:
        signal.signal(signal.SIGTERM, prev_term)
    logger.info("pack complete: {} jobs listed, {} run here, {} failed, results in {}",
                len(jobs), ran, failed, results)
    return failed


def main(
    jobs_file: list[Path] = typer.Option(..., help="JSONL ({key, cmd, mem_mib?, env?}) or plain command lines; "
                                         "repeat (with --claim-dir/--log-dir in the same order) for several campaigns."),
    log_dir: list[Path] = typer.Option(..., help="Per-job logs + results*.jsonl (resume reads them); one per --jobs-file."),
    max_per_gpu: int = typer.Option(5, help="Safety cap on concurrent jobs per GPU; memory is the real gate."),
    default_mem_mib: int = typer.Option(6000, help="mem_mib for rows that do not state one."),
    reserve_mib: int = typer.Option(2000, help="Free memory to keep on a GPU beyond the job's ask."),
    poll_s: float = typer.Option(5.0, help="Seconds between scheduler ticks."),
    settle_s: float = typer.Option(20.0, help="Minimum seconds between two admissions (lets the new "
                                   "process's memory show up in nvidia-smi first)."),
    claim_dir: list[Path] = typer.Option([], help="Shared-queue mode: directory of claim/done/failed "
                                         "markers shared by every worker of this jobs file; one per --jobs-file."),
    stale_s: float = typer.Option(900.0, help="Shared mode: a claim whose heartbeat is older than this "
                                  "belongs to a dead worker and is reclaimed."),
    linger_s: float = typer.Option(0.0, help="Shared mode: how long an idle worker waits for other "
                                   "workers' claims to go stale before exiting."),
    admit_until_s: float | None = typer.Option(None, help="Stop admitting new jobs this many seconds "
                                               "after start (set to time limit minus one job's duration)."),
    drain_file: Path | None = typer.Option(None, help="Keep waiting for new rows while this file does not "
                                           "exist (a producer is still appending to the jobs file)."),
) -> None:
    """Pack a jobs file (or several campaigns' files) onto the GPUs of this allocation (see module docstring)."""
    configure_loguru()
    # direct callers (tests) may pass single paths
    jobs_file = [jobs_file] if isinstance(jobs_file, Path) else list(jobs_file)
    log_dir = [log_dir] if isinstance(log_dir, Path) else list(log_dir)
    claim_dir = [] if claim_dir is None else ([claim_dir] if isinstance(claim_dir, Path) else list(claim_dir))
    if len(log_dir) != len(jobs_file) or (claim_dir and len(claim_dir) != len(jobs_file)):
        raise typer.BadParameter("--jobs-file, --log-dir and --claim-dir must be given the same number of times")
    gpus = visible_gpus()
    wid = worker_id()
    if len(jobs_file) > 1:
        sources = [Source(jf, ld, ClaimQueue(cd, stale_s, wid) if claim_dir else None)
                   for jf, ld, cd in zip(jobs_file, log_dir, claim_dir or [None] * len(jobs_file))]
        jobs = merge_sources(sources, default_mem_mib)
        logger.info("{} jobs from {} campaigns on {} GPU(s) {} ; max {}/gpu, reserve {} MiB; worker {}",
                    len(jobs), len(sources), len(gpus), gpus, max_per_gpu, reserve_mib, wid)
        try:
            failed = run_pack(jobs, gpus, log_dir[0], max_per_gpu=max_per_gpu, reserve_mib=reserve_mib,
                              poll_s=poll_s, settle_s=settle_s, linger_s=linger_s, free_fn=free_mib_by_gpu,
                              admit_until_s=admit_until_s, default_mem_mib=default_mem_mib,
                              drain_file=drain_file, sources=sources)
        except _Interrupted:
            logger.warning("packer interrupted by SIGTERM; exiting 143")
            sys.exit(143)
        if failed:
            sys.exit(1)
        return
    jobs_file1, log_dir1, claim_dir1 = jobs_file[0], log_dir[0], (claim_dir[0] if claim_dir else None)
    if not jobs_file1.exists() and drain_file is not None:
        logger.info("jobs file {} does not exist yet; waiting for the producer (drain file {})", jobs_file1, drain_file)
        jobs = []          # live reload picks the rows up once the producer writes the file
    else:
        jobs = parse_jobs(jobs_file1, default_mem_mib)
    queue = ClaimQueue(claim_dir1, stale_s, wid) if claim_dir1 is not None else None
    logger.info("{} jobs on {} GPU(s) {} ; max {}/gpu, reserve {} MiB{}", len(jobs), len(gpus), gpus,
                max_per_gpu, reserve_mib, f"; shared queue {claim_dir1} as {queue.worker}" if queue else "")
    try:
        failed = run_pack(jobs, gpus, log_dir1, max_per_gpu=max_per_gpu, reserve_mib=reserve_mib,
                          poll_s=poll_s, settle_s=settle_s, queue=queue, linger_s=linger_s, free_fn=free_mib_by_gpu,
                          admit_until_s=admit_until_s, jobs_file=jobs_file1, default_mem_mib=default_mem_mib,
                          drain_file=drain_file)
    except _Interrupted:
        logger.warning("packer interrupted by SIGTERM; exiting 143")
        sys.exit(143)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
