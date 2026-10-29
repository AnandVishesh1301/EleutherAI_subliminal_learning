"""Slurm execution for jobq: pool identity + allocation policy, and the login-node supervisor.

A Slurm pool is one ``jobq work --slurm`` process inside one ``lab submit`` job. Its node
identity is ``slurm-<jobid>.r<restart>`` (a requeued job gets a new one), or the explicit
``$SL_JOBQ_NODE`` of a carrier for a queue pinned to a legacy hostname. Its GPU policy is
the template with ``gpus`` = the job's assigned cards (``lab --gpu-ids indices``).

The supervisor (``jobq slurm-supervise``, one per campaign, in tmux on the login node)
submits pools while the campaign's queues hold claimable work, within its GPU share and
the user's cap, and releases claims of this campaign's dead pools (no attempt charged).
State lives under ``<root>/slurm/<campaign>/``: ``supervisor.json``, ``pools/<node>.json``,
``logs/`` (lab job logs), and ``PAUSED`` (no new pools; every pool drains and exits).
A carrier claims only from its pinned queues; a pinned node with a stop file gets no
carrier; a pool job Slurm holds after repeated launch failures is cancelled and replaced; after ``CRASH_STREAK`` pool jobs in a row ended without running their pool, the
supervisor submits nothing for ``CRASH_BACKOFF_S``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import socket
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import policy_path
from subliminal_learning.jobqueue.worker import SlurmRun
from subliminal_learning.utils.io import atomic_write_json

LIVE_STATES = frozenset(
    {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STAGE_OUT", "SIGNALING", "RESIZING"}
)
CRASH_STREAK = 3
CRASH_BACKOFF_S = 1800.0
SUBMIT_HISTORY = 20
_IDENTITY_RE = re.compile(r"^slurm-(\d+(?:_\d+)?)\.r(\d+)$")
_DURATION_RE = re.compile(r"^(?:(\d+)-)?(\d+):(\d{2}):(\d{2})$")
POOL_SUFFIX = "-pool"
PIN_SUFFIX = "-pin."


# --------------------------- identity / allocation ---------------------------


def slurm_job_id(env: Mapping[str, str]) -> str:
    """Slurm job id of this job (``<array>_<task>`` for array tasks)."""
    if env.get("SLURM_ARRAY_JOB_ID") and env.get("SLURM_ARRAY_TASK_ID"):
        return f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    jid = env.get("SLURM_JOB_ID")
    if not jid:
        raise RuntimeError("--slurm needs SLURM_JOB_ID (run inside `lab submit`)")
    return jid


def slurm_identity(env: Mapping[str, str]) -> str:
    """``slurm-<jobid>.r<restart>``: a requeued incarnation never reuses an identity."""
    return f"slurm-{slurm_job_id(env)}.r{int(env.get('SLURM_RESTART_COUNT') or 0)}"


def parse_identity(node: str) -> tuple[str, int] | None:
    """``(jobid, restart)`` of a ``slurm-<jobid>.r<restart>`` identity, else None."""
    m = _IDENTITY_RE.match(node or "")
    return (m.group(1), int(m.group(2))) if m else None


def allocated_gpus(env: Mapping[str, str]) -> list[int]:
    """Physical (nvidia-smi) indices of the job's cards from ``lab --gpu-ids indices``."""
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    parts = [p.strip() for p in cvd.split(",") if p.strip()]
    if env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or not parts or not all(
        p.isdecimal() for p in parts
    ):
        raise RuntimeError(
            f"--slurm needs GPU indices (submit with `lab ... --gpu-ids indices`); got "
            f"CUDA_VISIBLE_DEVICES={cvd!r} CUDA_DEVICE_ORDER={env.get('CUDA_DEVICE_ORDER')!r}"
        )
    return [int(p) for p in parts]


def write_allocation_policy(root: Path, node: str, template: Path, gpus: list[int]) -> Path:
    """Write ``gpu_policy.<node>.json`` = template with this allocation's ``gpus``."""
    body = json.loads(Path(template).read_text())
    body["gpus"] = list(gpus)
    body["yield_to_foreign"] = False
    path = policy_path(root, node)
    atomic_write_json(path, body)
    return path


def job_end_epoch(env: Mapping[str, str], start: float, walltime_s: float) -> float:
    """Wall-clock end of this job: ``$SLURM_JOB_END_TIME`` if set, else start + limit."""
    end = env.get("SLURM_JOB_END_TIME", "")
    return float(end) if end.isdecimal() else start + walltime_s


def parse_duration_s(text: str) -> int:
    """Seconds of a lab ``--time`` value (``30m``, ``8h``, ``HH:MM:SS``, ``D-HH:MM:SS``)."""
    if re.fullmatch(r"\d+[hm]", text):
        return int(text[:-1]) * (3600 if text[-1] == "h" else 60)
    m = _DURATION_RE.match(text)
    if not m or int(m.group(3)) > 59 or int(m.group(4)) > 59:
        raise ValueError(f"Not a lab --time value: {text!r}")
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


# --------------------------- state files ---------------------------


def state_dir(root: Path, name: str) -> Path:
    return Path(root) / "slurm" / name


def pool_record_path(root: Path, name: str, node: str) -> Path:
    return state_dir(root, name) / "pools" / f"{node}.json"


def paused_path(root: Path, name: str) -> Path:
    return state_dir(root, name) / "PAUSED"


def write_pool_record(root: Path, name: str, node: str, **fields) -> Path:
    """Create or update this pool's record (fields merge into the existing JSON)."""
    path = pool_record_path(root, name, node)
    body = read_json(path) or {}
    body.update(fields)
    store.mkdir_open(path.parent)
    atomic_write_json(path, body)
    return path


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def pool_records(root: Path, name: str) -> dict[str, dict]:
    """Every pool record of this campaign, keyed by node identity."""
    pdir = state_dir(root, name) / "pools"
    if not pdir.exists():
        return {}
    out = {}
    for p in sorted(pdir.glob("*.json")):
        body = read_json(p)
        if body is not None:
            out[p.stem] = body
    return out


# --------------------------- claims ---------------------------


def release_node_claims(
    root: Path, node: str, queues: Iterable[str] | None = None
) -> list[tuple[str, str]]:
    """Return ``node``'s open claims to pending without charging an attempt.

    Only for a node whose every process is known dead (a finished Slurm job, or an
    explicit identity at its single carrier's start).
    """
    released = []
    for q in queues if queues is not None else store.list_queues(root):
        for jk, owner in store.claim_owners(root, q):
            if owner.get("node") == node:
                store.remove_claim(root, q, jk)
                released.append((q, jk))
    return released


def dead_identities(
    records: Mapping[str, dict], jobs: Mapping[str, "SqueueJob"]
) -> list[str]:
    """``slurm-<jobid>.r<k>`` identities of this campaign whose process is gone.

    Dead = the job is not in a live state, or a newer incarnation (``r>k``) of the same
    job has a record (Slurm starts it only after the old one ended). Already-released
    and explicit identities are never returned (a carrier releases its own at start).
    """
    newest: dict[str, int] = {}
    for node in records:
        parsed = parse_identity(node)
        if parsed:
            newest[parsed[0]] = max(newest.get(parsed[0], -1), parsed[1])
    dead = []
    for node, rec in records.items():
        parsed = parse_identity(node)
        if not parsed or rec.get("released_utc"):
            continue
        jid, restart = parsed
        job = jobs.get(jid)
        if job is None or job.state not in LIVE_STATES or restart < newest[jid]:
            dead.append(node)
    return dead


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def prepare_pool(
    root: Path,
    name: str,
    template: Path,
    *,
    walltime_s: float,
    claim_margin_s: float,
    idle_exit_s: float,
    queues: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, SlurmRun]:
    """Set up this Slurm job's pool: identity, allocation policy, record, limits.

    Returns:
        A tuple (node, limits): the pool's node identity and its :class:`SlurmRun`.
    """
    env = os.environ if env is None else env
    start = time.time()
    explicit = env.get(store.NODE_ENV) or ""
    node = explicit or slurm_identity(env)
    gpus = allocated_gpus(env)
    write_allocation_policy(root, node, template, gpus)
    released: list[tuple[str, str]] = []
    if explicit:
        released = release_node_claims(root, node, queues)
        store.worker_pid_path(root, node).unlink(missing_ok=True)
        for q, jk in released:
            logger.warning("Released {}'s earlier claim {} {}", node, q, jk)
    end = job_end_epoch(env, start, walltime_s)
    prior = read_json(pool_record_path(root, name, node)) or {}
    job_ids = [*prior.get("job_ids", []), slurm_job_id(env)]
    limits = SlurmRun(
        claim_until=end - claim_margin_s,
        idle_exit_s=idle_exit_s,
        pinned_only=bool(explicit),
        pause_file=paused_path(root, name),
    )
    write_pool_record(
        root, name, node,
        identity=node,
        slurm_job_id=slurm_job_id(env),
        job_ids=job_ids[-50:],
        restart=int(env.get("SLURM_RESTART_COUNT") or 0),
        host=socket.gethostname(),
        gpus=gpus,
        pid=os.getpid(),
        cwd=os.getcwd(),
        started_utc=_iso(start),
        claim_until_utc=_iso(limits.claim_until),
        deadline_utc=_iso(end),
        released_at_start=len(released),
        ended_utc=None,
        exit_reason=None,
        cells_started=None,
    )
    return node, limits


# --------------------------- squeue / lab ---------------------------


@dataclass(frozen=True)
class SqueueJob:
    """One of the user's jobs as ``squeue`` lists it."""

    job_id: str
    name: str
    state: str
    gpus: int
    reason: str = ""


def _gres_gpus(gres: str) -> int:
    total = 0
    for item in gres.split(","):
        if "gpu" in item:
            m = re.search(r":(\d+)(?:\(|$)", item)
            total += int(m.group(1)) if m else 0
    return total


def parse_squeue(text: str) -> dict[str, SqueueJob]:
    """Parse ``squeue -h -o '%i|%j|%T|%b|%r'`` lines (reason optional) into jobs by id."""
    jobs = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        jid, jname, state, gres = (p.strip() for p in parts[:4])
        reason = parts[4].strip() if len(parts) > 4 else ""
        jobs[jid] = SqueueJob(jid, jname, state, _gres_gpus(gres), reason)
    return jobs


def query_squeue(run: Callable = subprocess.run) -> dict[str, SqueueJob] | None:
    """The user's queued and running jobs, or None when squeue fails (state unknown)."""
    user = os.environ.get("USER") or ""
    try:
        out = run(
            ["squeue", "-h", "-u", user, "-o", "%i|%j|%T|%b|%r"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        logger.exception("squeue failed; skipping this pass")
        return None
    return parse_squeue(out.stdout)


# --------------------------- supervisor ---------------------------


@dataclass(frozen=True)
class SupervisorConfig:
    """One campaign's Slurm pool settings.

    Attributes:
        root: Campaign queue root.
        name: Campaign name (unique; prefixes the Slurm job names).
        pin: Code checkout whose ``.venv`` runs the pools.
        policy_template: GPU policy JSON whose ``gpus`` each pool replaces.
        queues: Restrict pools to these queues (a shared root); None = all.
        gpus_per_pool: GPUs per Slurm job (one machine).
        workers_per_gpu: Pool threads per GPU (= the policy's cap_per_gpu).
        max_gpus: This campaign's GPU share (running + pending pools).
        user_gpu_cap: Ceiling on all the user's GPUs in squeue, every job counted.
        lanes: lab ``--lane`` value.
        preemptible: Also allow the borrow lane.
        time: lab ``--time`` per pool.
        claim_margin_s: Pools stop claiming this long before their end.
        idle_exit_s: Pools exit after this long with nothing to run.
        max_pending_pools: Pending pool jobs allowed at once.
        mem_per_gpu: lab ``--mem`` per GPU (``"48G"``); None = lab default.
        done_marker: The supervisor exits once this file exists.
        interval_s: Seconds between passes.
    """

    root: Path
    name: str
    pin: Path
    policy_template: Path
    queues: tuple[str, ...] | None = None
    gpus_per_pool: int = 4
    workers_per_gpu: int = 1
    max_gpus: int = 32
    user_gpu_cap: int = 32
    lanes: str = "burst"
    preemptible: bool = True
    time: str = "8h"
    claim_margin_s: int = 1800
    idle_exit_s: int = 900
    max_pending_pools: int = 2
    mem_per_gpu: str | None = None
    done_marker: Path | None = None
    interval_s: float = 120.0

    def __post_init__(self) -> None:
        if not 1 <= self.gpus_per_pool <= 8:
            raise ValueError(f"gpus_per_pool must be 1..8: {self.gpus_per_pool=}")
        if self.max_gpus < self.gpus_per_pool:
            raise ValueError(f"{self.max_gpus=} is below {self.gpus_per_pool=}")
        if self.mem_per_gpu is not None and not re.fullmatch(r"\d+G", self.mem_per_gpu):
            raise ValueError(f"mem_per_gpu must look like '48G': {self.mem_per_gpu=}")
        if self.claim_margin_s >= parse_duration_s(self.time):
            raise ValueError(f"{self.claim_margin_s=} leaves no claim window in {self.time}")


def visible_queues(cfg: SupervisorConfig) -> list[str]:
    names = store.list_queues(cfg.root)
    return [n for n in names if cfg.queues is None or n in cfg.queues]


def _deps_satisfied(root: Path, meta: store.QueueMeta) -> bool:
    for dep in meta.depends_on:
        if not (store.queue_exists(root, dep) and store.queue_complete(root, dep)):
            return False
        if meta.strict_deps and store.queue_status(root, dep)["failed"]:
            return False
    return True


def demand(cfg: SupervisorConfig) -> tuple[int, dict[str, int]]:
    """Pending work: ``(unpinned pending jobs, {pinned node: pending + open claims})``.

    A pinned node whose stop file exists is left out (its carrier would exit at once).
    """
    unpinned = 0
    pinned: dict[str, int] = {}
    for q in visible_queues(cfg):
        meta = store.read_meta(cfg.root, q)
        if meta.node is not None and store.stop_path(cfg.root, meta.node).exists():
            continue
        st = store.queue_status(cfg.root, q)
        if meta.node is not None:
            if st["pending"] or st["running"]:
                pinned[meta.node] = pinned.get(meta.node, 0) + st["pending"] + st["running"]
            continue
        if st["pending"] and _deps_satisfied(cfg.root, meta):
            unpinned += st["pending"]
    return unpinned, pinned


def stopped_pinned_nodes(cfg: SupervisorConfig) -> list[str]:
    """Pinned nodes of the visible queues whose stop file exists (no carrier for them)."""
    nodes = {store.read_meta(cfg.root, q).node for q in visible_queues(cfg)}
    return sorted(n for n in nodes if n is not None and store.stop_path(cfg.root, n).exists())


def lab_log_path(root: Path, name: str, job_name: str, job_id: str) -> Path:
    """The job's lab log (``lab submit`` runs in the state dir; Slurm creates it at start)."""
    return state_dir(root, name) / "logs" / f"{job_name}-{job_id}.out"


def launch_held(cfg: SupervisorConfig, jobs: Mapping[str, "SqueueJob"]) -> list[str]:
    """This campaign's pool jobs Slurm put in REQUEUE_HOLD after repeated launch failures
    (a preempted borrow job whose requeues failed to start). Held, they keep their GPUs
    in the share while running nothing; the supervisor cancels them and resubmits."""
    return sorted(
        j.job_id for j in campaign_pool_jobs(cfg.name, jobs)
        if j.state == "REQUEUE_HOLD" and "launch" in j.reason.lower()
    )


def failed_start(job_id: str, records: Mapping[str, dict]) -> bool:
    """Whether a started, finished pool job never ran its pool: no record names it, or
    its pool raised before starting any cell."""
    jid = str(job_id)
    recs = [
        r for r in records.values()
        if jid == str(r.get("slurm_job_id")) or jid in map(str, r.get("job_ids", ()))
    ]
    if not recs:
        return True
    last = recs[-1]
    return str(last.get("slurm_job_id")) == jid and last.get("exit_reason") == "error" and not last.get(
        "cells_started"
    )


def crash_loop(
    root: Path,
    name: str,
    submits: list[dict],
    jobs: Mapping[str, "SqueueJob"],
    records: Mapping[str, dict],
    now: float,
) -> bool:
    """Whether the last ``CRASH_STREAK`` started-and-finished submissions all failed to
    start their pool, the newest submitted less than ``CRASH_BACKOFF_S`` ago.

    A job that never started (no lab log: cancelled while pending) is not counted.
    """
    finished = [
        s for s in submits
        if str(s["job_id"]) not in jobs
        and now - s["epoch"] >= 60
        and lab_log_path(root, name, s["job_name"], s["job_id"]).exists()
    ]
    last = finished[-CRASH_STREAK:]
    if len(last) < CRASH_STREAK or not all(failed_start(s["job_id"], records) for s in last):
        return False
    return now - last[-1]["epoch"] < CRASH_BACKOFF_S


def pool_job_name(cfg: SupervisorConfig, node: str | None = None) -> str:
    return f"{cfg.name}{PIN_SUFFIX}{node}" if node else f"{cfg.name}{POOL_SUFFIX}"


def pool_command(cfg: SupervisorConfig, node: str | None = None) -> list[str]:
    """The ``lab submit`` argv for one pool (a carrier of ``node`` when given)."""
    work = [
        ".venv/bin/python", "-m", "scripts.jobq", "work",
        "--root", str(cfg.root),
        "--workers", str(cfg.gpus_per_pool * cfg.workers_per_gpu),
        "--lock-prefix", default_lock_prefix(),
        "--slurm", "--slurm-name", cfg.name,
        "--policy-template", str(cfg.policy_template),
        "--walltime-s", str(parse_duration_s(cfg.time)),
        "--claim-margin-s", str(cfg.claim_margin_s),
        "--idle-exit-s", str(cfg.idle_exit_s),
    ]
    if cfg.queues:
        work += ["--queues", ",".join(cfg.queues)]
    inner = f"cd {shlex.quote(str(cfg.pin))} && exec {shlex.join(work)}"
    argv = ["lab", "submit", "--lane", cfg.lanes]
    if cfg.preemptible:
        argv.append("--preemptible")
    argv += ["--gpus", str(cfg.gpus_per_pool), "--gpu-ids", "indices", "--time", cfg.time]
    if cfg.mem_per_gpu:
        num = int(re.fullmatch(r"(\d+)G", cfg.mem_per_gpu).group(1))
        argv += ["--mem", f"{num * cfg.gpus_per_pool}G"]
    argv += ["--name", pool_job_name(cfg, node), "--", "bash", "-c", inner]
    return argv


def default_lock_prefix() -> str:
    return f"/tmp/subliminal_jobq_{os.environ.get('USER') or 'user'}"


def pool_env(node: str | None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment a pool is submitted with (lab exports it into the job)."""
    env = dict(os.environ if base is None else base)
    env.pop(store.NODE_ENV, None)
    env["UV_NO_SYNC"] = "1"
    env["SL_JOBQ_LOCK_PREFIX"] = default_lock_prefix()
    if node:
        env[store.NODE_ENV] = node
    return env


@dataclass(frozen=True)
class PassPlan:
    """What one supervisor pass decided.

    Attributes:
        pools: Unpinned pools to submit.
        carriers: Pinned nodes that get a carrier pool.
        release: Dead identities whose claims are released.
        own_gpus: This campaign's GPUs in squeue before submitting.
        user_gpus: All the user's GPUs in squeue before submitting.
        reason: Why fewer pools than wanted were submitted ("" if none held back).
    """

    pools: int
    carriers: tuple[str, ...]
    release: tuple[str, ...]
    own_gpus: int
    user_gpus: int
    reason: str


def plan_pass(
    cfg: SupervisorConfig,
    jobs: Mapping[str, SqueueJob],
    records: Mapping[str, dict],
    unpinned: int,
    pinned: Mapping[str, int],
    paused: bool,
    crashing: bool = False,
) -> PassPlan:
    """Decide one pass from squeue, pool records and pending work (pure)."""
    own = [
        j for j in jobs.values()
        if j.name == pool_job_name(cfg) or j.name.startswith(f"{cfg.name}{PIN_SUFFIX}")
    ]
    own_gpus = sum(j.gpus for j in own)
    user_gpus = sum(j.gpus for j in jobs.values())
    release = tuple(dead_identities(records, jobs))
    if paused:
        return PassPlan(0, (), release, own_gpus, user_gpus, "paused")
    if crashing:
        return PassPlan(0, (), release, own_gpus, user_gpus, "crash-loop")
    room = min(cfg.max_gpus - own_gpus, cfg.user_gpu_cap - user_gpus) // cfg.gpus_per_pool
    pending_slots = cfg.max_pending_pools - sum(j.state == "PENDING" for j in own)
    carried = {j.name[len(cfg.name) + len(PIN_SUFFIX):] for j in own if j.name != pool_job_name(cfg)}
    carriers = []
    for node in sorted(pinned):
        if node not in carried and room > 0 and pending_slots > 0:
            carriers.append(node)
            room -= 1
            pending_slots -= 1
    per_pool = cfg.gpus_per_pool * cfg.workers_per_gpu
    have = sum(j.name == pool_job_name(cfg) for j in own)
    wanted = max(0, math.ceil(unpinned / per_pool) - have)
    pools = max(0, min(wanted, room, pending_slots))
    reason = ""
    if pools < wanted:
        reason = "pending-pool limit" if pending_slots <= pools else "GPU budget"
    return PassPlan(pools, tuple(carriers), release, own_gpus, user_gpus, reason)


def submit_pool(cfg: SupervisorConfig, node: str | None, run: Callable = subprocess.run) -> str | None:
    """Submit one pool; return its job id (None when lab refused)."""
    sdir = state_dir(cfg.root, cfg.name)
    store.mkdir_open(sdir / "logs")
    argv = pool_command(cfg, node)
    try:
        out = run(argv, cwd=sdir, env=pool_env(node), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        logger.exception("lab submit failed for {}", pool_job_name(cfg, node))
        return None
    ids = re.findall(r"\b(\d+)\b", out.stdout)
    if out.returncode != 0 or not ids:
        logger.error("lab submit rc={} stdout={!r} stderr={!r}", out.returncode, out.stdout, out.stderr)
        return None
    return ids[-1]


def supervise_once(cfg: SupervisorConfig, run: Callable = subprocess.run) -> dict | None:
    """One supervisor pass; returns the state written to ``supervisor.json`` (None: skipped)."""
    jobs = query_squeue(run)
    if jobs is None:
        return None
    held = launch_held(cfg, jobs)
    if held:
        try:
            out = run(["lab", "cancel", *held], capture_output=True, text=True, timeout=120)
            if out.returncode == 0:
                logger.warning("Cancelled launch-held pool jobs {}", held)
                jobs = {k: v for k, v in jobs.items() if k not in held}
            else:
                logger.error("lab cancel {} rc={} stderr={!r}", held, out.returncode, out.stderr)
        except (OSError, subprocess.SubprocessError):
            logger.exception("lab cancel failed for {}", held)
    records = pool_records(cfg.root, cfg.name)
    unpinned, pinned = demand(cfg)
    prev = read_json(state_dir(cfg.root, cfg.name) / "supervisor.json") or {}
    submits = list(prev.get("submits") or [])
    now = time.time()
    crashing = crash_loop(cfg.root, cfg.name, submits, jobs, records, now)
    plan = plan_pass(
        cfg, jobs, records, unpinned, pinned,
        paused_path(cfg.root, cfg.name).exists(), crashing,
    )
    queues = visible_queues(cfg)
    released = []
    for node in plan.release:
        released += release_node_claims(cfg.root, node, queues)
        write_pool_record(cfg.root, cfg.name, node, released_utc=store.now_iso())
    for q, jk in released:
        logger.warning("Released dead pool claim {} {}", q, jk)
    names = [pool_job_name(cfg, node) for node in plan.carriers]
    names += [pool_job_name(cfg) for _ in range(plan.pools)]
    nodes = [*plan.carriers, *([None] * plan.pools)]
    submitted = [submit_pool(cfg, node, run) for node in nodes]
    submits += [
        {"job_id": s, "job_name": n, "epoch": now} for s, n in zip(submitted, names) if s
    ]
    state = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "updated_utc": store.now_iso(),
        "unpinned_pending": unpinned,
        "pinned": pinned,
        "pinned_stopped": stopped_pinned_nodes(cfg),
        "own_gpus": plan.own_gpus,
        "user_gpus": plan.user_gpus,
        "held_back": plan.reason,
        "submitted": [s for s in submitted if s],
        "released_claims": len(released),
        "paused": plan.reason == "paused",
        "crash_loop": crashing,
        "cancelled_held": held,
        "submits": submits[-SUBMIT_HISTORY:],
    }
    store.mkdir_open(state_dir(cfg.root, cfg.name))
    atomic_write_json(state_dir(cfg.root, cfg.name) / "supervisor.json", state)
    return state


def supervise(cfg: SupervisorConfig, run: Callable = subprocess.run, max_passes: int | None = None) -> None:
    """Run passes every ``interval_s`` until the done marker exists."""
    passes = 0
    while cfg.done_marker is None or not Path(cfg.done_marker).exists():
        state = supervise_once(cfg, run)
        if state is not None and (state["submitted"] or state["released_claims"]):
            logger.info(
                "{}: submitted {} released {} (own {} / user {} GPUs, pending {}{})",
                cfg.name, state["submitted"], state["released_claims"], state["own_gpus"],
                state["user_gpus"], state["unpinned_pending"],
                f", held back: {state['held_back']}" if state["held_back"] else "",
            )
        passes += 1
        if max_passes is not None and passes >= max_passes:
            return
        time.sleep(cfg.interval_s)
    logger.info("{}: done marker {} present; supervisor exiting", cfg.name, cfg.done_marker)


def campaign_pool_jobs(name: str, jobs: Mapping[str, SqueueJob]) -> list[SqueueJob]:
    """This campaign's pool jobs in squeue (unpinned pools and carriers, by job name)."""
    return [
        j for j in jobs.values()
        if j.name == f"{name}{POOL_SUFFIX}" or j.name.startswith(f"{name}{PIN_SUFFIX}")
    ]


def live_pool_nodes(root: Path, name: str, jobs: Mapping[str, SqueueJob]) -> list[str]:
    """Identities of this campaign's pools whose job is live (for pause/resume)."""
    out = []
    for node, rec in pool_records(root, name).items():
        job = jobs.get(str(rec.get("slurm_job_id")))
        if job is not None and job.state in LIVE_STATES:
            out.append(node)
    return out
