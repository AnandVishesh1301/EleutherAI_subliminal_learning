"""On-disk queue state: meta, jobs, claims, results, and the DERIVED job states.

State layout under ``<root>`` (default ``run_logs/queue``, always a parameter)::

    <root>/<queue>/meta.json                    queue config + defaults
    <root>/<queue>/jobs.jsonl                   one Job per line
    <root>/<queue>/claims/<jobkey>/owner.json   claim = mkdir claims/<jobkey> (atomic)
    <root>/<queue>/results/<jobkey>.json        terminal outcome (rc, gpu, log, attempt)
    <root>/<queue>/attempts/<jobkey>.json       requeue counter (survives result deletion)
    <root>/<queue>/logs/<STAMP>_<jobkey>.<host>.log

Job states are never stored — they are derived from the presence of a claim dir and a
result file: pending (neither) / running (claim, no result) / done (result rc==0) /
failed (result rc!=0). A result always wins: a claim left beside a result is terminal.

``jobs.jsonl`` is grown by an atomic full-file rewrite (read-all, append, replace) — fine
at sweep scale (hundreds of lines) and it keeps the "all writes atomic" invariant.

**Node safety** (two nodes share ``<root>``): a claim is only ever stolen/released
automatically when it is owned by THIS hostname and its pid is dead. Cross-node pids are
meaningless here, so we never touch another node's claims except via ``force_release``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
import shutil
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from subliminal_learning.jobqueue.model import Job, assert_safe_jobkey, sanitize_jobkey
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

# Repo root: .../src/subliminal_learning/jobqueue/store.py -> parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]

# An owner-less claim dir older than this is crash debris and may be reclaimed by any node.
ORPHAN_CLAIM_GRACE_S = 120.0


def utc_stamp() -> str:
    """``20260724T031500Z``-style stamp (``date -u +%Y%m%dT%H%M%SZ`` equivalent)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (for log lines and result timestamps)."""
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live, non-zombie process on THIS node.

    ``os.kill(pid, 0)`` succeeds on a zombie; an orphaned cell that PID 1 never reaps
    would otherwise hold its claim forever.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return not _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    fields = stat.rsplit(")", 1)
    return len(fields) == 2 and fields[1].split()[:1] == ["Z"]


# --------------------------- path helpers ---------------------------


def mkdir_open(path: Path) -> Path:
    """``mkdir -p`` + best-effort ``chmod 0o777`` on the leaf.

    The queue root is shared by two nodes whose ``angela`` users have DIFFERENT uids and no
    common group (see run_logs/README.md), so cross-node access rides on the 'other' bits:
    every queue state dir must be world-writable or the other node can't claim/steal/record.
    The chmod is best-effort — a dir created by the other node may refuse, which is fine as
    long as that node created it open too.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass
    return path


def queue_dir(root: Path, name: str) -> Path:
    return Path(root) / name


def _meta_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "meta.json"


def _jobs_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "jobs.jsonl"


def _claims_dir(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "claims"


def _claim_dir(root: Path, name: str, jobkey: str) -> Path:
    return _claims_dir(root, name) / jobkey


def _owner_path(root: Path, name: str, jobkey: str) -> Path:
    return _claim_dir(root, name, jobkey) / "owner.json"


def _results_dir(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "results"


def _result_path(root: Path, name: str, jobkey: str) -> Path:
    return _results_dir(root, name) / f"{jobkey}.json"


def _attempt_path(root: Path, name: str, jobkey: str) -> Path:
    return queue_dir(root, name) / "attempts" / f"{jobkey}.json"


def queue_logs_dir(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "logs"


# --------------------------- meta / queues ---------------------------


@dataclass
class QueueMeta:
    """Parsed ``meta.json``: queue config + the defaults that fill in per-job gaps."""

    name: str
    created_utc: str
    depends_on: list[str] = field(default_factory=list)
    node: str | None = None
    priority: int = 0
    strict_deps: bool = False
    defaults: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_utc": self.created_utc,
            "depends_on": list(self.depends_on),
            "node": self.node,
            "priority": self.priority,
            "strict_deps": self.strict_deps,
            "defaults": dict(self.defaults),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueueMeta":
        return cls(
            name=d["name"],
            created_utc=d["created_utc"],
            depends_on=list(d.get("depends_on") or []),
            node=d.get("node"),
            priority=int(d.get("priority", 0)),
            strict_deps=bool(d.get("strict_deps", False)),
            defaults=dict(d.get("defaults") or {}),
        )


def queue_exists(root: Path, name: str) -> bool:
    return _meta_path(root, name).exists()


def read_meta(root: Path, name: str) -> QueueMeta:
    import json

    return QueueMeta.from_dict(json.loads(_meta_path(root, name).read_text()))


def list_queues(root: Path) -> list[str]:
    """Names of every queue under ``root`` (any dir holding a ``meta.json``), sorted."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()
    )


def create_queue(
    root: Path,
    name: str,
    *,
    depends_on: list[str] | None = None,
    node: str | None = None,
    priority: int = 0,
    strict_deps: bool = False,
    defaults: dict | None = None,
    exist_ok: bool = False,
) -> QueueMeta:
    """Write ``meta.json`` for a new queue.

    Errors if the queue exists unless ``exist_ok`` — in which case the EXISTING meta is
    kept untouched (append semantics: more jobs may be submitted, meta is set once).
    """
    if queue_exists(root, name):
        if not exist_ok:
            raise FileExistsError(f"queue {name!r} already exists under {root}")
        return read_meta(root, name)
    meta = QueueMeta(
        name=name,
        created_utc=now_iso(),
        depends_on=list(depends_on or []),
        node=node,
        priority=priority,
        strict_deps=strict_deps,
        defaults=dict(defaults or {}),
    )
    mkdir_open(Path(root))
    mkdir_open(queue_dir(root, name))
    atomic_write_json(_meta_path(root, name), meta.to_dict())
    return meta


def update_queue_defaults(root: Path, name: str, defaults: dict) -> QueueMeta:
    """Merge `defaults` into an existing queue's ``meta.json`` (atomic rewrite).

    Used to refresh the code-pin ``cwd`` / env after an intentional pin advance
    without recreating the queue or touching job lists.
    """
    if not queue_exists(root, name):
        raise FileNotFoundError(f"queue {name!r} does not exist under {root}")
    meta = read_meta(root, name)
    merged = dict(meta.defaults)
    incoming = dict(defaults or {})
    # Nested env merge so pin env keys update without dropping unrelated defaults.
    if "env" in incoming or "env" in merged:
        env = dict(merged.get("env") or {})
        env.update(dict(incoming.get("env") or {}))
        incoming = {**incoming, "env": env}
    merged.update(incoming)
    updated = QueueMeta(
        name=meta.name,
        created_utc=meta.created_utc,
        depends_on=list(meta.depends_on),
        node=meta.node,
        priority=meta.priority,
        strict_deps=meta.strict_deps,
        defaults=merged,
    )
    atomic_write_json(_meta_path(root, name), updated.to_dict())
    return updated


def update_queue_node(root: Path, name: str, node: str | None) -> QueueMeta:
    """Re-pin an existing queue to `node` (atomic ``meta.json`` rewrite).

    Only the queue's home pool serves it, so this hands the queue to another node.
    Refuses while any claim is held, because claims belong to the old node's pool.
    Unpinning (`node` None) is allowed with claims held: the old pool keeps serving
    the queue and `steal_stale` never touches another node's claim.

    Raises:
        RuntimeError: The queue still holds claims and `node` is not None.
    """
    if not queue_exists(root, name):
        raise FileNotFoundError(f"queue {name!r} does not exist under {root}")
    cdir = _claims_dir(root, name)
    held = [d.name for d in cdir.iterdir() if d.is_dir()] if cdir.exists() else []
    if held and node is not None:
        raise RuntimeError(f"queue {name!r} holds {len(held)} claim(s); drain it first")
    meta = read_meta(root, name)
    updated = QueueMeta(
        name=meta.name,
        created_utc=meta.created_utc,
        depends_on=list(meta.depends_on),
        node=node,
        priority=meta.priority,
        strict_deps=meta.strict_deps,
        defaults=dict(meta.defaults),
    )
    atomic_write_json(_meta_path(root, name), updated.to_dict())
    return updated


def update_queue_depends_on(root: Path, name: str, depends_on: list[str]) -> QueueMeta:
    """Replace an existing queue's ``depends_on`` (atomic ``meta.json`` rewrite).

    Pools re-read deps on every claim pass, so this gates (or frees) the queue's
    pending jobs without touching claims or running jobs.
    """
    if not queue_exists(root, name):
        raise FileNotFoundError(f"queue {name!r} does not exist under {root}")
    meta = read_meta(root, name)
    updated = QueueMeta(
        name=meta.name,
        created_utc=meta.created_utc,
        depends_on=list(depends_on),
        node=meta.node,
        priority=meta.priority,
        strict_deps=meta.strict_deps,
        defaults=dict(meta.defaults),
    )
    atomic_write_json(_meta_path(root, name), updated.to_dict())
    return updated


# --------------------------- jobs ---------------------------


def load_jobs(root: Path, name: str) -> list[Job]:
    """All jobs of a queue in submission (file) order."""
    import json

    p = _jobs_path(root, name)
    if not p.exists():
        return []
    return [
        Job.from_dict(json.loads(line))
        for line in p.read_text().splitlines()
        if line.strip()
    ]


def check_new_jobs(root: Path, name: str, jobs: list[Job]) -> list[Job]:
    """Validate an incoming batch against a queue's existing jobs; return the existing jobs.

    Refuses duplicate ``jobkey``s — both against already-submitted jobs and within the
    incoming batch — and ``jobkey``s that are not usable path segments. Callers that
    create the queue meta should validate FIRST so a rejected submit never leaves a
    meta-only (zero-job) queue behind to wedge dependents.
    """
    existing = load_jobs(root, name)
    seen = {j.jobkey for j in existing}
    for j in jobs:
        assert_safe_jobkey(j.jobkey, key=j.key)
        if j.jobkey in seen:
            raise ValueError(
                f"duplicate job key {j.key!r} (jobkey {j.jobkey!r}) in queue {name!r}"
            )
        seen.add(j.jobkey)
    return existing


def append_jobs(root: Path, name: str, jobs: list[Job]) -> list[Job]:
    """Append ``jobs`` to ``jobs.jsonl`` (atomic full-file rewrite); validates duplicates."""
    import json

    existing = check_new_jobs(root, name, jobs)
    body = "".join(json.dumps(j.to_dict()) + "\n" for j in existing + jobs)
    atomic_write_text(_jobs_path(root, name), body)
    return existing + jobs


# --------------------------- claims ---------------------------


def has_result(root: Path, name: str, jobkey: str) -> bool:
    return _result_path(root, name, jobkey).exists()


def claim_next(root: Path, name: str, node: str, pid: int) -> Job | None:
    """Atomically claim the first pending job of ``name`` (file order), or ``None``.

    A result always wins (terminal, skipped even if a stray claim dir remains). The claim
    itself is ``os.mkdir(claims/<jobkey>)`` — the sole atomic primitive; ``EEXIST`` means a
    peer already owns it, so we move on. ``owner.json`` is written after a successful mkdir.
    """
    mkdir_open(_claims_dir(root, name))
    for job in load_jobs(root, name):
        jk = job.jobkey
        if has_result(root, name, jk):
            continue
        cdir = _claim_dir(root, name, jk)
        try:
            os.mkdir(cdir)
        except FileExistsError:
            continue
        try:  # other node must be able to grace-reclaim / force-release this claim
            os.chmod(cdir, 0o777)
        except OSError:
            pass
        if has_result(root, name, jk):
            # TOCTOU guard: a finishing worker recorded the result and removed its claim
            # between our has_result check and our mkdir — don't re-run a finished job.
            remove_claim(root, name, jk)
            continue
        atomic_write_json(
            _owner_path(root, name, jk),
            {"node": node, "pid": pid, "gpu": None, "start_utc": now_iso()},
        )
        return job
    return None


def _read_owner(root: Path, name: str, jobkey: str) -> dict | None:
    import json

    p = _owner_path(root, name, jobkey)
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def update_owner_gpu(root: Path, name: str, jobkey: str, gpu: int) -> None:
    """Stamp the acquired physical GPU into an existing claim's owner.json (best-effort).

    The claim is made before GPU acquisition, so ``gpu`` starts as null; this makes the live
    GPU visible to ``jobq status`` while the job runs.
    """
    owner = _read_owner(root, name, jobkey)
    if owner is None:
        return
    owner["gpu"] = gpu
    try:
        atomic_write_json(_owner_path(root, name, jobkey), owner)
    except OSError:
        pass  # claim raced away; the result file will carry the gpu regardless


def _proc_starttime(pid: int) -> int | None:
    """Kernel start time (clock ticks since boot) of ``pid`` from ``/proc``; None if gone."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        return int(text.rsplit(")", 1)[1].split()[19])
    except (IndexError, ValueError):
        return None


def stamp_owner_cell(root: Path, name: str, jobkey: str, cell_pid: int) -> None:
    """Record the cell subprocess (pid + start time) on the claim's owner.json (best-effort).

    Cells are started in their own session and outlive a killed worker pool, so a claim
    whose WORKER pid is dead may still have a live trainer writing the cell's run dir.
    `steal_stale` consults this stamp before re-dispatching; the start time guards
    against pid reuse.
    """
    owner = _read_owner(root, name, jobkey)
    if owner is None:
        return
    owner["cell_pid"] = int(cell_pid)
    owner["cell_starttime"] = _proc_starttime(int(cell_pid))
    try:
        atomic_write_json(_owner_path(root, name, jobkey), owner)
    except OSError:
        pass


def cell_alive(owner: dict) -> bool:
    """Whether the claim's stamped cell subprocess is still the same live process."""
    cell_pid = owner.get("cell_pid")
    if not isinstance(cell_pid, int) or not pid_alive(cell_pid):
        return False
    stamped = owner.get("cell_starttime")
    if stamped is None:
        return True
    return _proc_starttime(cell_pid) == stamped


def remove_claim(root: Path, name: str, jobkey: str) -> None:
    """Remove a claim dir (best-effort; a claim is transient, a result is the record)."""
    shutil.rmtree(_claim_dir(root, name, jobkey), ignore_errors=True)


def steal_stale(root: Path, name: str, node: str) -> list[str]:
    """Remove claims owned by THIS node whose owning pid is dead; return their jobkeys.

    Never touches a claim stamped with a different ``node`` (cross-node pids are
    meaningless here). A claim that already has a result is just cleaned up as terminal.
    A dead worker whose stamped cell subprocess is still running (`stamp_owner_cell`) is
    NOT stolen: re-dispatching it would put a second trainer on the same run dir. The
    claim frees itself once that cell exits; the re-run then no-ops via `cell_is_done`.
    """
    cdir = _claims_dir(root, name)
    if not cdir.exists():
        return []
    stolen: list[str] = []
    for entry in cdir.iterdir():
        if not entry.is_dir():
            continue
        jk = entry.name
        if has_result(root, name, jk):
            remove_claim(root, name, jk)
            continue
        owner = _read_owner(root, name, jk)
        if owner is None:
            # Claim dir without (readable) owner.json: either a peer mid-claim (sub-second
            # window) or debris of a crash between mkdir and the owner write. After a grace
            # period ANY node may reclaim it — ownership was never recorded, so the node
            # guard doesn't apply, and an owner-less claim can never finish on its own.
            try:
                age_s = time.time() - entry.stat().st_mtime
            except OSError:
                continue
            if age_s > ORPHAN_CLAIM_GRACE_S:
                remove_claim(root, name, jk)
                stolen.append(jk)
            continue
        if owner.get("node") != node:
            continue  # NEVER steal another node's claim
        pid = owner.get("pid")
        if isinstance(pid, int) and not pid_alive(pid) and not cell_alive(owner):
            remove_claim(root, name, jk)
            stolen.append(jk)
    return stolen


def open_claims(root: Path, name: str, node: str) -> list[tuple[str, dict]]:
    """``(jobkey, owner)`` of claims stamped by ``node`` that have no result yet."""
    cdir = _claims_dir(root, name)
    if not cdir.exists():
        return []
    out: list[tuple[str, dict]] = []
    for entry in sorted(cdir.iterdir()):
        if not entry.is_dir() or has_result(root, name, entry.name):
            continue
        owner = _read_owner(root, name, entry.name)
        if owner is not None and owner.get("node") == node:
            out.append((entry.name, owner))
    return out


def force_release(root: Path, name: str, key: str) -> bool:
    """Remove a claim IGNORING the node guard (CLI ``release --force`` only).

    ``key`` may be the human key or an already-sanitized jobkey. Returns whether a claim
    existed. Never called automatically.
    """
    jk = sanitize_jobkey(key)
    existed = _claim_dir(root, name, jk).exists()
    remove_claim(root, name, jk)
    return existed


# --------------------------- results / attempts ---------------------------


def read_attempt(root: Path, name: str, jobkey: str) -> int:
    import json

    p = _attempt_path(root, name, jobkey)
    try:
        return int(json.loads(p.read_text())["attempt"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return 0


def _write_attempt(root: Path, name: str, jobkey: str, attempt: int) -> None:
    atomic_write_json(_attempt_path(root, name, jobkey), {"attempt": attempt})


def bump_attempt(root: Path, name: str, jobkey: str) -> int:
    """Increment (and persist) a job's attempt counter; return the new value.

    Used when a running job is requeued WITHOUT a result (e.g. yield-kill): the retry
    should record ``attempt = prev + 1``, mirroring :func:`requeue`.
    """
    n = read_attempt(root, name, jobkey) + 1
    mkdir_open(_attempt_path(root, name, jobkey).parent)
    _write_attempt(root, name, jobkey, n)
    return n


def record_result(
    root: Path,
    name: str,
    job: Job,
    *,
    rc: int,
    node: str,
    gpu: int | None,
    start_utc: str,
    end_utc: str,
    log: str,
    attempt: int,
) -> None:
    """Write a job's terminal result (atomic) and persist its attempt counter."""
    jk = job.jobkey
    mkdir_open(_results_dir(root, name))
    mkdir_open(_attempt_path(root, name, jk).parent)
    atomic_write_json(
        _result_path(root, name, jk),
        {
            "key": job.key,
            "rc": rc,
            "node": node,
            "gpu": gpu,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "log": log,
            "attempt": attempt,
        },
    )
    _write_attempt(root, name, jk, attempt)


def load_results(root: Path, name: str) -> list[dict]:
    """Every terminal result record of a queue as parsed dicts (unreadable ones skipped).

    Each dict carries the fields written by :func:`record_result` (``rc``, ``start_utc``,
    ``end_utc``, ...). Used by the yield watchdog's median-duration progress estimate.
    """
    import json

    rdir = _results_dir(root, name)
    if not rdir.exists():
        return []
    out: list[dict] = []
    for rp in sorted(rdir.glob("*.json")):
        try:
            out.append(json.loads(rp.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def requeue(
    root: Path,
    name: str,
    *,
    failed_only: bool = True,
    jobkeys: Iterable[str] | None = None,
    count_attempt: bool = True,
) -> list[str]:
    """Make terminal jobs pending again by deleting their results (+ stray claims).

    ``failed_only`` (default) touches only rc!=0 results; ``jobkeys`` restricts the
    sweep to those jobs. Bumps each affected job's attempt counter so its next run
    records ``attempt = prev + 1``; ``count_attempt=False`` (preemption) keeps the
    counter, so drivers' retry caps do not charge the killed run. Returns the
    requeued jobkeys.
    """
    import json

    rdir = _results_dir(root, name)
    if not rdir.exists():
        return []
    only = None if jobkeys is None else set(jobkeys)
    requeued: list[str] = []
    for rp in rdir.glob("*.json"):
        try:
            res = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if failed_only and res.get("rc", 0) == 0:
            continue
        jk = rp.stem
        if only is not None and jk not in only:
            continue
        prev = int(res.get("attempt", 0))
        _write_attempt(root, name, jk, prev + 1 if count_attempt else prev)
        rp.unlink(missing_ok=True)
        remove_claim(root, name, jk)
        requeued.append(jk)
    return requeued


# --------------------------- derived state ---------------------------

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


def job_state(root: Path, name: str, job: Job) -> str:
    """Derive one job's state from claim/result presence (a result always wins)."""
    import json

    jk = job.jobkey
    rp = _result_path(root, name, jk)
    if rp.exists():
        try:
            rc = int(json.loads(rp.read_text()).get("rc", 1))
        except (OSError, json.JSONDecodeError, ValueError):
            rc = 1
        return DONE if rc == 0 else FAILED
    if _claim_dir(root, name, jk).exists():
        return RUNNING
    return PENDING


def queue_status(root: Path, name: str) -> dict:
    """Counts by derived state + a ``running`` list carrying each owner's node/gpu/key."""
    counts = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0}
    running: list[dict] = []
    for job in load_jobs(root, name):
        st = job_state(root, name, job)
        counts[st] += 1
        if st == RUNNING:
            owner = _read_owner(root, name, job.jobkey) or {}
            running.append(
                {
                    "key": job.key,
                    "node": owner.get("node"),
                    "pid": owner.get("pid"),
                    "gpu": owner.get("gpu"),
                    "start_utc": owner.get("start_utc"),
                }
            )
    return {**counts, "total": sum(counts.values()), "running_jobs": running}


def queue_complete(root: Path, name: str) -> bool:
    """Whether every job of ``name`` is terminal (has a result — done OR failed)."""
    if not queue_exists(root, name):
        return False
    jobs = load_jobs(root, name)
    return bool(jobs) and all(has_result(root, name, j.jobkey) for j in jobs)


# --------------------------- node files ---------------------------


def worker_pid_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"worker.{hostname}.pid"


def stop_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"stop.{hostname}"


def this_host() -> str:
    return socket.gethostname()
