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
at sweep scale (hundreds of lines) and it keeps the "all writes atomic" invariant. The
rewrite is serialized on ``<queue>/.jobs.lock`` so concurrent submits cannot lose a batch.

**Node safety** (two nodes share ``<root>``): a claim is only ever stolen/released
automatically when it is owned by THIS hostname and its pid is dead. Cross-node pids are
meaningless here, so we never touch another node's claims except via ``force_release``.
"""

from __future__ import annotations

import os
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from subliminal_learning.jobqueue.model import Job, sanitize_jobkey
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text, file_lock

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
    """Whether ``pid`` is a live process on THIS node (``os.kill(pid, 0)`` probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return True


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
    incoming batch. Callers that create the queue meta should validate FIRST so a rejected
    submit never leaves a meta-only (zero-job) queue behind to wedge dependents.
    """
    existing = load_jobs(root, name)
    seen = {j.jobkey for j in existing}
    for j in jobs:
        if j.jobkey in seen:
            raise ValueError(
                f"duplicate job key {j.key!r} (jobkey {j.jobkey!r}) in queue {name!r}"
            )
        seen.add(j.jobkey)
    return existing


def _jobs_lock_path(root: Path, name: str) -> Path:
    return root / name / ".jobs.lock"


def append_jobs(root: Path, name: str, jobs: list[Job]) -> list[Job]:
    """Append ``jobs`` to ``jobs.jsonl`` (atomic full-file rewrite); validates duplicates.

    The read-validate-rewrite is serialized on ``<queue>/.jobs.lock`` (world-writable, so
    the other node's uid can take it): without it two concurrent ``jobq submit`` calls both
    read the pre-existing file and the second rewrite silently drops the first batch. The
    duplicate check is re-run INSIDE the lock, so an earlier unlocked check by a caller is
    only an early-rejection convenience.
    """
    import json

    mkdir_open(root / name)
    with file_lock(_jobs_lock_path(root, name)):
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
    A job deferred by a tempfail requeue (``not_before`` in the future) is skipped here.
    """
    mkdir_open(_claims_dir(root, name))
    for job in load_jobs(root, name):
        jk = job.jobkey
        if has_result(root, name, jk):
            continue
        if read_not_before(root, name, jk) > time.time():
            # Deferred after an EX_TEMPFAIL (rc=75) exit: still PENDING and still counted
            # as outstanding work, just not claimable yet.
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


def _owner_state(root: Path, name: str, jobkey: str) -> tuple[str, dict | None]:
    """Tri-state owner read: ``("owner", dict)`` / ``("missing", None)`` / ``("unreadable", None)``.

    The distinction exists for `steal_stale`: only a provably ABSENT owner record may
    enter the ownerless-reclaim path. A record that is present-but-unreadable right now
    (EIO/EDQUOT — routine during quota storms) or corrupt (exotic: owner.json is
    atomic-written) must never be conflated with debris, because reclaiming a LIVE
    peer's claim is duplicate execution. Every other caller may keep collapsing the
    last two states to None (they all fail safe on it).
    """
    import json

    p = _owner_path(root, name, jobkey)
    try:
        return "owner", json.loads(p.read_text())
    except FileNotFoundError:
        return "missing", None
    except (OSError, json.JSONDecodeError):
        return "unreadable", None


def _read_owner(root: Path, name: str, jobkey: str) -> dict | None:
    return _owner_state(root, name, jobkey)[1]


def read_owner(root: Path, name: str, jobkey: str) -> dict | None:
    """A claim's owner record (``node``/``pid``/``gpu``/``start_utc``), or None.

    Public read for out-of-module callers (the worker's orphan-claim janitor); ``None``
    covers both "no claim" and "owner not yet written / unreadable".
    """
    return _read_owner(root, name, jobkey)


def update_owner_gpu(root: Path, name: str, jobkey: str, gpu: int | None) -> None:
    """Stamp the acquired physical GPU into an existing claim's owner.json (best-effort).

    The claim is made before GPU acquisition, so ``gpu`` starts as null; this makes the live
    GPU visible to ``jobq status`` while the job runs. Stays ``None`` for a declared
    CPU-only job (``slots: 0``), which reserves no GPU at all.
    """
    owner = _read_owner(root, name, jobkey)
    if owner is None:
        return
    owner["gpu"] = gpu
    try:
        atomic_write_json(_owner_path(root, name, jobkey), owner)
    except OSError:
        pass  # claim raced away; the result file will carry the gpu regardless


def remove_claim(root: Path, name: str, jobkey: str) -> None:
    """Remove a claim dir (best-effort; a claim is transient, a result is the record)."""
    shutil.rmtree(_claim_dir(root, name, jobkey), ignore_errors=True)


# Orphan (owner-less) claim reclaim needs TWO sightings at least this far apart on OUR
# OWN monotonic clock, on top of the cross-node mtime grace: the mtime is written by the
# other node's clock, so skew > ORPHAN_CLAIM_GRACE_S could otherwise make a peer's
# just-created mid-claim dir look reclaimable on first sight (duplicate execution). A
# fresh mid-claim gains owner.json within seconds, clearing its registry entry before a
# second sighting can reclaim it. Module attribute so tests can shrink it.
ORPHAN_RESIGHT_MIN_S = 5.0
_orphan_first_seen: dict[tuple[str, str, str], float] = {}
_ORPHAN_LOCK = threading.Lock()


def clear_orphan_registry() -> None:
    """Drop the orphan-claim sighting registry (test isolation)."""
    with _ORPHAN_LOCK:
        _orphan_first_seen.clear()


def steal_stale(root: Path, name: str, node: str) -> list[str]:
    """Remove claims owned by THIS node whose owning pid is dead; return their jobkeys.

    Never touches a claim stamped with a different ``node`` (cross-node pids are
    meaningless here). A claim that already has a result is just cleaned up as terminal.
    """
    cdir = _claims_dir(root, name)
    if not cdir.exists():
        return []
    stolen: list[str] = []
    seen_ownerless: set[tuple[str, str, str]] = set()
    for entry in cdir.iterdir():
        if not entry.is_dir():
            continue
        jk = entry.name
        if has_result(root, name, jk):
            remove_claim(root, name, jk)
            continue
        state, owner = _owner_state(root, name, jk)
        key = (str(root), name, jk)
        if state == "unreadable":
            # Owner record PRESENT (or unknown) but not readable right now — a transient
            # EIO/EDQUOT, or a corrupt record. Never treat it as ownerless debris: the
            # claim may belong to a LIVE peer job, and reclaiming it is duplicate
            # execution. Skip with no registry mutation; the prune below then resets any
            # earlier ownerless sighting (conservative — delays a real reclaim, never
            # enables a wrong one). A permanently corrupt owner wedges the job until a
            # manual `jobq release --force` — the fail-safe trade.
            logger.warning("claim {}/{}: owner.json unreadable — NOT reclaiming", name, jk)
            continue
        if state == "missing":
            # Claim dir without owner.json: either a peer mid-claim (sub-second window)
            # or debris of a crash between mkdir and the owner write. After a grace
            # period ANY node may reclaim it — ownership was never recorded, so the node
            # guard doesn't apply, and an owner-less claim can never finish on its own.
            # The reclaim additionally requires a PRIOR sighting >= ORPHAN_RESIGHT_MIN_S
            # ago on this process's monotonic clock (see the registry note above).
            try:
                age_s = time.time() - entry.stat().st_mtime
            except OSError:
                continue
            now = time.monotonic()
            with _ORPHAN_LOCK:
                first = _orphan_first_seen.setdefault(key, now)
            seen_ownerless.add(key)
            if age_s > ORPHAN_CLAIM_GRACE_S and now - first >= ORPHAN_RESIGHT_MIN_S:
                remove_claim(root, name, jk)
                stolen.append(jk)
                seen_ownerless.discard(key)
                with _ORPHAN_LOCK:
                    _orphan_first_seen.pop(key, None)
            continue
        with _ORPHAN_LOCK:
            _orphan_first_seen.pop(key, None)  # owner landed: not an orphan after all
        if owner.get("node") != node:
            continue  # NEVER steal another node's claim
        pid = owner.get("pid")
        if isinstance(pid, int) and not pid_alive(pid):
            remove_claim(root, name, jk)
            stolen.append(jk)
    # Registry hygiene: entries for THIS queue whose claim was not seen ownerless on this
    # pass (claim vanished, or gained an owner) are stale sightings — drop them so the
    # registry stays bounded by the live ownerless set.
    with _ORPHAN_LOCK:
        for k in [k for k in _orphan_first_seen
                  if k[0] == str(root) and k[1] == name and k not in seen_ownerless]:
            _orphan_first_seen.pop(k)
    return stolen


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


def _read_attempt_record(root: Path, name: str, jobkey: str) -> dict:
    """The whole ``attempts/<jobkey>.json`` dict, or ``{}`` if absent/unreadable."""
    import json

    try:
        rec = json.loads(_attempt_path(root, name, jobkey).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return rec if isinstance(rec, dict) else {}


def read_attempt(root: Path, name: str, jobkey: str) -> int:
    import json

    p = _attempt_path(root, name, jobkey)
    try:
        return int(json.loads(p.read_text())["attempt"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return 0


def read_mem_floor(root: Path, name: str, jobkey: str) -> int:
    """Persisted per-job ``mem_mib`` floor (OOM escalation), or ``0`` when never escalated.

    Optional everywhere: records written by pre-escalation workers have no such key.
    """
    try:
        return int(_read_attempt_record(root, name, jobkey).get("mem_mib_floor", 0) or 0)
    except (TypeError, ValueError):
        return 0


def write_mem_floor(root: Path, name: str, jobkey: str, mem_mib: int) -> None:
    """Persist a job's escalated ``mem_mib`` floor in the attempts sidecar.

    Lives here rather than in ``jobs.jsonl`` because that file is append-only and shared
    cross-node; the extra key is ignored by workers running older src (they read only
    ``attempt``), so the escalation degrades to "no escalation" instead of breaking them.
    """
    rec = _read_attempt_record(root, name, jobkey)
    rec["mem_mib_floor"] = int(mem_mib)
    mkdir_open(_attempt_path(root, name, jobkey).parent)
    atomic_write_json(_attempt_path(root, name, jobkey), rec)


def read_oom_requeues(root: Path, name: str, jobkey: str) -> int:
    """How many times this job has been OOM-requeued (attempts sidecar; 0 when absent)."""
    try:
        return int(_read_attempt_record(root, name, jobkey).get("oom_requeues", 0) or 0)
    except (TypeError, ValueError):
        return 0


def bump_oom_requeues(root: Path, name: str, jobkey: str) -> int:
    """Increment (and persist) the job's OOM-requeue counter; return the new value.

    Lives in the attempts sidecar like ``mem_mib_floor`` — an extra key ignored by
    workers running older src, so mixed-version nodes degrade to "no backstop" rather
    than breaking.
    """
    rec = _read_attempt_record(root, name, jobkey)
    try:
        n = int(rec.get("oom_requeues", 0) or 0) + 1
    except (TypeError, ValueError):
        n = 1
    rec["oom_requeues"] = n
    mkdir_open(_attempt_path(root, name, jobkey).parent)
    atomic_write_json(_attempt_path(root, name, jobkey), rec)
    return n


def read_tempfails(root: Path, name: str, jobkey: str) -> int:
    """How many times this job exited 75 (EX_TEMPFAIL) and was requeued; 0 when absent."""
    try:
        return int(_read_attempt_record(root, name, jobkey).get("tempfails", 0) or 0)
    except (TypeError, ValueError):
        return 0


def read_not_before(root: Path, name: str, jobkey: str) -> float:
    """Epoch second before which this job must not be claimed again; 0.0 when absent."""
    try:
        return float(_read_attempt_record(root, name, jobkey).get("not_before", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def bump_tempfails(root: Path, name: str, jobkey: str, *, not_before: float) -> int:
    """Increment the job's tempfail counter and defer its next claim; return the new count.

    Both keys live in the attempts sidecar like ``oom_requeues`` — extra keys ignored by
    workers running older src, so mixed-version nodes degrade to "retry immediately"
    rather than breaking.
    """
    rec = _read_attempt_record(root, name, jobkey)
    try:
        n = int(rec.get("tempfails", 0) or 0) + 1
    except (TypeError, ValueError):
        n = 1
    rec["tempfails"] = n
    rec["not_before"] = float(not_before)
    mkdir_open(_attempt_path(root, name, jobkey).parent)
    atomic_write_json(_attempt_path(root, name, jobkey), rec)
    return n


def reset_mem_floors(
    root: Path,
    name: str,
    *,
    jobkeys: list[str] | None = None,
    above_mib: int | None = None,
) -> list[tuple[str, int]]:
    """Drop the escalated ``mem_mib_floor`` from attempts sidecars; return (jobkey, old).

    The operator un-wedge for a floor no card can satisfy (a co-tenant OOM once escalated
    one to 47 GiB, leaving the job PENDING forever). Unlike :func:`requeue` this works in
    ANY state and touches neither results nor claims — clearing a running job's floor only
    affects its next claim. ``jobkeys`` restricts to named jobs; ``above_mib`` to floors
    strictly above that many MiB. Every other sidecar key is preserved.
    """
    import json

    adir = queue_dir(root, name) / "attempts"
    if not adir.is_dir():
        return []
    if jobkeys is None:
        try:
            keys = sorted(p.stem for p in adir.glob("*.json"))
        except OSError:
            return []
    else:
        keys = list(jobkeys)
    cleared: list[tuple[str, int]] = []
    for jk in keys:
        path = _attempt_path(root, name, jk)
        try:
            rec = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        except PermissionError:
            # Another uid's 0600 sidecar (the cross-node uid mismatch): report, keep going.
            logger.warning("attempts sidecar {} unreadable (permission denied) — skipping", path)
            continue
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        try:
            old = int(rec.get("mem_mib_floor", 0) or 0)
        except (TypeError, ValueError):
            old = 0
        if old <= 0 or (above_mib is not None and old <= above_mib):
            continue
        rec.pop("mem_mib_floor", None)
        try:
            mkdir_open(path.parent)
            atomic_write_json(path, rec)
        except PermissionError:
            logger.warning("attempts sidecar {} not writable (permission denied) — skipping", path)
            continue
        cleared.append((jk, old))
    return cleared


def _write_attempt(
    root: Path,
    name: str,
    jobkey: str,
    attempt: int,
    *,
    reset_oom: bool = False,
    reset_mem_floor: bool = False,
) -> None:
    """Persist the attempt counter, PRESERVING any other keys (e.g. ``mem_mib_floor``).

    The reset flags are the operator escape hatch (``jobq requeue --reset-oom`` /
    ``--reset-mem-floor``): they drop the named key instead of carrying it over.
    """
    rec = _read_attempt_record(root, name, jobkey)
    rec["attempt"] = attempt
    if reset_oom:
        rec.pop("oom_requeues", None)
    if reset_mem_floor:
        rec.pop("mem_mib_floor", None)
    atomic_write_json(_attempt_path(root, name, jobkey), rec)


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
    reset_oom: bool = False,
    reset_mem_floor: bool = False,
) -> list[str]:
    """Make terminal jobs pending again by deleting their results (+ stray claims).

    ``failed_only`` (default) touches only rc!=0 results. Bumps each affected job's attempt
    counter so its next run records ``attempt = prev + 1``. Returns the requeued jobkeys.

    The attempts sidecar's OOM state survives a requeue by default, so a job that exhausted
    the worker's OOM-requeue backstop stays terminal on its next OOM. ``reset_oom`` clears
    that counter (the explicit operator reset); ``reset_mem_floor`` clears the escalated
    ``mem_mib`` floor, which is normally worth keeping.
    """
    import json

    rdir = _results_dir(root, name)
    if not rdir.exists():
        return []
    requeued: list[str] = []
    for rp in rdir.glob("*.json"):
        try:
            res = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if failed_only and res.get("rc", 0) == 0:
            continue
        jk = rp.stem
        _write_attempt(
            root,
            name,
            jk,
            int(res.get("attempt", 0)) + 1,
            reset_oom=reset_oom,
            reset_mem_floor=reset_mem_floor,
        )
        rp.unlink(missing_ok=True)
        remove_claim(root, name, jk)
        requeued.append(jk)
    if requeued:
        complete_stamp_path(root, name).unlink(missing_ok=True)
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


def untrip_stamp_path(root: Path, name: str) -> Path:
    return root / name / "untrip.stamp"


def stamp_untrip(root: Path, name: str) -> None:
    """Record an operator requeue so a pool's tripped breaker re-admits the queue.

    The breaker's other untrip signals (pending grew / total grew) can both be defeated
    when the peer node drained pending below the trip snapshot — the documented
    cross-node wedge. The stamp's MTIME is the signal; the ISO timestamp content is for
    humans. Written by ``jobq requeue``; pools compare it against their trip time.
    """
    atomic_write_text(untrip_stamp_path(root, name), now_iso() + "\n")


def read_untrip_stamp(root: Path, name: str) -> float | None:
    """Mtime of the queue's untrip stamp, or None when never stamped."""
    try:
        return untrip_stamp_path(root, name).stat().st_mtime
    except OSError:
        return None


def complete_stamp_path(root: Path, name: str) -> Path:
    return Path(root) / name / "complete.stamp"


def _mtime_ns(p: Path) -> int | None:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


def queue_complete(root: Path, name: str) -> bool:
    """Whether every job of ``name`` is terminal (has a result — done OR failed).

    Cached on disk (2026-09-12): the first full check that finds the queue complete writes
    ``complete.stamp``; later calls trust it while BOTH ``jobs.jsonl`` and ``results/`` are
    older than the stamp (an appended job bumps the former, a deleted result — requeue —
    bumps the latter; ``requeue`` also drops the stamp explicitly). Every worker thread
    re-derives the ready list before each claim, and on a root of ~1,900 queues (~1,700
    long complete) the per-job ``has_result`` stats dominated the claim loop — 15-second
    eval jobs left the pool half idle. Three stats replace N+1 for a settled queue; an
    incomplete queue still takes the full path.
    """
    if not queue_exists(root, name):
        return False
    stamp = _mtime_ns(complete_stamp_path(root, name))
    if stamp is not None:
        jobs_m = _mtime_ns(_jobs_path(root, name))
        res_m = _mtime_ns(_results_dir(root, name))
        if jobs_m is not None and res_m is not None and jobs_m <= stamp and res_m <= stamp:
            return True
    jobs = load_jobs(root, name)
    done = bool(jobs) and all(has_result(root, name, j.jobkey) for j in jobs)
    if done:
        try:
            atomic_write_text(complete_stamp_path(root, name), now_iso() + "\n")
        except OSError:  # read-only / other-uid queue dir: cache is optional
            pass
    return done


# --------------------------- node files ---------------------------


def worker_pid_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"worker.{hostname}.pid"


def stop_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"stop.{hostname}"


def this_host() -> str:
    return socket.gethostname()
