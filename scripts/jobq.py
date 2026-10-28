"""CLI: a file-based GPU job queue + per-node worker pool (``uv run python -m scripts.jobq``).

The repo-proper replacement for the per-sweep shell launchers. Submit jobs into named
queues under a ``--root`` (default: ``queue_root`` from the paths YAML), then run one
foreground worker pool per node; the pool claims, GPU-schedules, runs, and records each
job. Two nodes can drain the same root safely — a worker only ever acts on its OWN
node's claims/pids. Keep the root inside your own SSD subdir: the tree is
world-writable so cross-node workers can share it, so a root under another account's
subdir mixes both users' queues.

Subcommands: ``submit`` / ``work`` / ``status`` / ``requeue`` / ``release`` / ``stop`` /
``wait``, plus the Slurm layer (``jobqueue/slurm.py``): ``work --slurm`` inside a
``lab submit`` job, ``slurm-supervise`` / ``slurm-pause`` / ``slurm-resume`` /
``slurm-status`` on the login node. See ARCHITECTURE.md ("Job queue framework") for the
layout and a worked example.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import resolve_queue_root
from subliminal_learning.jobqueue import slurm as slurm_layer
from subliminal_learning.jobqueue import store, yielding
from subliminal_learning.jobqueue.gpu import (
    DEFAULT_LOCK_PREFIX,
    LOCK_PREFIX_ENV,
    GpuManager,
    resolve_lock_prefix,
)
from subliminal_learning.jobqueue.model import parse_jobs_file
from subliminal_learning.jobqueue.worker import DEFAULT_FAIL_STREAK_LIMIT, run_pool
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

PATHS_CONFIG = Path("configs/paths/paths_local.yaml")
ROOT_HELP = "Queue state root (default: queue_root from the paths YAML)."
LOCK_PREFIX_HELP = (
    f"Prefix for GPU slot flock files (default ${LOCK_PREFIX_ENV}, else "
    f"{DEFAULT_LOCK_PREFIX}). Point it at the pool's prefix when the pool was started "
    "with a non-default one."
)


def _root(root: Path | None) -> Path:
    """CLI ``--root`` if given, else the paths YAML's ``queue_root``."""
    if root is not None:
        return root
    if not PATHS_CONFIG.exists():
        return resolve_queue_root({})
    return resolve_queue_root(yaml.safe_load(PATHS_CONFIG.read_text()) or {})


def _claim_age_str(start_utc: str | None) -> str:
    """Humanize a claim's age from its owner start_utc ('?' when unparsable)."""
    from datetime import datetime, timezone

    if not start_utc:
        return "?"
    try:
        start = datetime.fromisoformat(start_utc)
    except ValueError:
        return "?"
    secs = int((datetime.now(timezone.utc) - start).total_seconds())
    if secs < 3600:
        return f"{secs // 60}m"
    return f"{secs / 3600:.1f}h"


def _parse_env(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--env K=V`` options into a dict."""
    env: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise typer.BadParameter(f"--env expects K=V, got {item!r}")
        k, v = item.split("=", 1)
        env[k.strip()] = v
    return env


@app.command()
def submit(
    queue: str = typer.Argument(..., help="Queue name to submit into (created if new)."),
    jobs_file: Path = typer.Option(..., "--jobs-file", help="JSONL or plain-lines jobs file."),
    template: str = typer.Option(
        None, help="cmd template for plain lines: {line}=whole line, {0},{1},..=|-split fields."
    ),
    depends_on: str = typer.Option(
        None, help="Comma list of queues that must be complete before this one runs."
    ),
    strict_deps: bool = typer.Option(
        False,
        "--strict-deps",
        help="Block while any dependency job is FAILED (default: failed counts as terminal).",
    ),
    node: str = typer.Option(None, help="Pin this queue to a hostname (default: any node)."),
    priority: int = typer.Option(0, help="Higher runs first among ready queues."),
    mem_mib: int = typer.Option(None, help="Queue-default free-MiB gate for its jobs."),
    cap_per_gpu: int = typer.Option(
        None, help="Stage ceiling: max jobs of this queue per GPU (on top of the policy cap)."
    ),
    cap_group: str = typer.Option(
        None, help="Cap-group name (default: queue name). Queues sharing a group co-cap."
    ),
    slots: int = typer.Option(
        None,
        help="Queue-default job weight in global per-GPU slot-units (heavy jobs take >1).",
    ),
    env: list[str] = typer.Option(None, "--env", help="Queue-default env, repeatable K=V."),
    cwd: str = typer.Option(
        None,
        help="Queue-default working directory (e.g. .code_pins/<sha>/ for MLflow-style pins).",
    ),
    root: Path = typer.Option(None, help=ROOT_HELP),
    dry_run: bool = typer.Option(False, help="Print the expanded jobs and exit."),
) -> None:
    """Expand a jobs file and append the jobs to QUEUE (creating meta on first submit)."""
    configure_loguru()
    root = _root(root)
    jobs = parse_jobs_file(jobs_file.read_text(), template=template)
    if dry_run:
        logger.info("Would submit {} job(s) to queue {!r}:", len(jobs), queue)
        for j in jobs:
            logger.info("  [{}] {}", j.jobkey, j.cmd)
        return
    defaults: dict = {}
    if mem_mib is not None:
        defaults["mem_mib"] = mem_mib
    if cap_per_gpu is not None:
        defaults["cap_per_gpu"] = cap_per_gpu
    if cap_group is not None:
        defaults["cap_group"] = cap_group
    if slots is not None:
        defaults["slots"] = slots
    env_d = _parse_env(env)
    if env_d:
        defaults["env"] = env_d
    if cwd is not None:
        defaults["cwd"] = cwd
    # Validate BEFORE creating meta: a rejected batch must not leave a meta-only (zero-job)
    # queue behind, which would wedge the pool and any dependent queues.
    store.check_new_jobs(root, queue, jobs)
    store.create_queue(
        root,
        queue,
        depends_on=[d.strip() for d in depends_on.split(",")] if depends_on else [],
        node=node,
        priority=priority,
        strict_deps=strict_deps,
        defaults=defaults,
        exist_ok=True,
    )
    store.append_jobs(root, queue, jobs)
    logger.info("Submitted {} job(s) to queue {!r} under {}", len(jobs), queue, root)


@app.command()
def work(
    root: Path = typer.Option(None, help=ROOT_HELP),
    workers: int = typer.Option(1, help="Number of worker threads (per-GPU slot cap is global)."),
    queues: str = typer.Option(
        None,
        help="Comma list restricting which queues to drain (strongly recommended on a "
        "shared --root so you do not claim another campaign's jobs).",
    ),
    lock_prefix: str = typer.Option(None, help=LOCK_PREFIX_HELP),
    fail_streak_limit: int = typer.Option(
        DEFAULT_FAIL_STREAK_LIMIT,
        help="Consecutive job failures that trip a queue; the pool then skips it "
        "until a requeue grows its pending count. 0 disables the breaker.",
    ),
    slurm: bool = typer.Option(
        False,
        "--slurm",
        help="Run as a Slurm pool (inside `lab submit --gpu-ids indices`): node identity "
        "from the job, GPUs = the allocation, closes before the time limit or when idle, "
        "SIGTERM requeues running cells without charging an attempt.",
    ),
    slurm_name: str = typer.Option(None, "--slurm-name", help="With --slurm: campaign name."),
    policy_template: Path = typer.Option(
        None, "--policy-template", help="With --slurm: GPU policy JSON whose gpus are replaced."
    ),
    walltime_s: float = typer.Option(None, "--walltime-s", help="With --slurm: the job's time limit."),
    claim_margin_s: float = typer.Option(
        1800.0, "--claim-margin-s", help="With --slurm: stop claiming this long before the end."
    ),
    idle_exit_s: float = typer.Option(
        900.0, "--idle-exit-s", help="With --slurm: exit after this long with nothing to run."
    ),
) -> None:
    """Run the foreground worker pool on THIS node (tmux pane, or one Slurm job with --slurm)."""
    load_dotenv()
    configure_loguru()
    root = _root(root)
    q = [x.strip() for x in queues.split(",") if x.strip()] if queues else None
    hostname, limits = None, None
    if slurm:
        if not (slurm_name and policy_template and walltime_s):
            raise typer.BadParameter("--slurm needs --slurm-name, --policy-template and --walltime-s")
        hostname, limits = slurm_layer.prepare_pool(
            root,
            slurm_name,
            policy_template,
            walltime_s=walltime_s,
            claim_margin_s=claim_margin_s,
            idle_exit_s=idle_exit_s,
            queues=q,
        )
    logger.info(
        "Starting worker pool on {} ({} threads) under {} queues={} lock_prefix={}",
        hostname or store.this_host(),
        workers,
        root,
        q or "(all)",
        resolve_lock_prefix(lock_prefix),
    )
    try:
        outcome = run_pool(
            root,
            workers=workers,
            queues=q,
            hostname=hostname,
            lock_prefix=lock_prefix,
            fail_streak_limit=fail_streak_limit or 0,
            slurm=limits,
        )
    except Exception:
        if slurm:
            slurm_layer.write_pool_record(
                root, slurm_name, hostname, ended_utc=store.now_iso(), exit_reason="error"
            )
        raise
    if slurm:
        slurm_layer.write_pool_record(
            root,
            slurm_name,
            hostname,
            ended_utc=store.now_iso(),
            exit_reason=outcome.reason,
            cells_started=outcome.cells_started,
        )


def _report_root(
    root: Path, queue: str | None = None, lock_prefix: str | None = None
) -> None:
    """Print per-queue counts for ``root`` (one queue if named) + this node's occupancy."""
    names = [queue] if queue else store.list_queues(root)
    if not names:
        logger.info("No queues under {}", root)
        return
    for name in names:
        if not store.queue_exists(root, name):
            logger.info("queue {!r}: does not exist", name)
            continue
        st = store.queue_status(root, name)
        meta = store.read_meta(root, name)
        logger.info(
            "queue {!r} (node={}, prio={}, deps={}): pending={} running={} done={} failed={} total={}",
            name, meta.node, meta.priority, meta.depends_on or "-",
            st["pending"], st["running"], st["done"], st["failed"], st["total"],
        )
        for r in st["running_jobs"]:
            age = _claim_age_str(r.get("start_utc"))
            line = f"    RUNNING {r['key']} node={r['node']} gpu={r['gpu']} pid={r['pid']} age={age}"
            if r["node"] not in (None, store.this_host()):
                line += (
                    "  [other-node claim — liveness not checkable from here; if that node "
                    f"crashed, run: jobq release {name} '{r['key']}' --force]"
                )
            logger.info(line)
    prefix = resolve_lock_prefix(lock_prefix)
    try:
        occ = GpuManager(root, store.this_host(), lock_prefix=prefix).occupancy()
        if occ:
            logger.info(
                "this node ({}) slot occupancy: {}",
                store.this_host(),
                ", ".join(f"gpu{g}={n}" for g, n in sorted(occ.items())),
            )
    except FileNotFoundError:
        pass  # no policy on this node -> nothing to report
    except PermissionError as exc:
        logger.warning(
            "could not read GPU slot locks under {} ({}). Pass --lock-prefix "
            "matching the prefix the worker pool was started with.",
            prefix,
            exc,
        )
    # Yielded GPUs (opt-in framework): silent when the feature is off / marker file absent.
    yielded = yielding.read_yielded(root, store.this_host())
    if yielded:
        logger.info(
            "this node ({}) yielded GPUs: {}",
            store.this_host(),
            ", ".join(
                f"{g} (foreign={v.get('foreign')}, since {v.get('first_seen')})"
                for g, v in sorted(yielded.items())
            ),
        )


@app.command()
def status(
    queue: str = typer.Argument(None, help="One queue (default: all queues under --root)."),
    root: Path = typer.Option(None, help=ROOT_HELP),
    lock_prefix: str = typer.Option(None, help=LOCK_PREFIX_HELP),
) -> None:
    """Per-queue pending/running/done/failed counts + this node's per-GPU slot occupancy."""
    configure_loguru()
    _report_root(_root(root), queue, lock_prefix)


@app.command()
def overview(
    parent: Path = typer.Option(
        None,
        help="Directory holding one or more queue roots (default: the queue root's parent).",
    ),
    lock_prefix: str = typer.Option(None, help=LOCK_PREFIX_HELP),
) -> None:
    """Status of EVERY queue root under PARENT — cross-node visibility from one command.

    Each node keeps its own root (e.g. run_logs/queue on soar-subliminal-1,
    run_logs/queue0 on soar-subliminal-0); both live on the shared run_logs/ mount,
    so this shows what every node is doing without merging their jobs.
    """
    configure_loguru()
    parent = parent if parent is not None else _root(None).parent
    if not parent.is_dir():
        logger.info("No such directory: {}", parent)
        return
    # A "queue root" is any dir that itself holds >=1 queue (a subdir with meta.json).
    roots = sorted(
        d for d in parent.iterdir() if d.is_dir() and store.list_queues(d)
    )
    if not roots:
        logger.info("No queue roots found under {}", parent)
        return
    for root in roots:
        logger.info("=== root: {} ===", root)
        _report_root(root, lock_prefix=lock_prefix)


@app.command()
def requeue(
    queue: str = typer.Argument(..., help="Queue to requeue jobs in."),
    all_: bool = typer.Option(False, "--all", help="Requeue done jobs too (default: failed only)."),
    root: Path = typer.Option(None, help=ROOT_HELP),
) -> None:
    """Make terminal jobs pending again (failed-only by default) so a worker retries them."""
    configure_loguru()
    root = _root(root)
    keys = store.requeue(root, queue, failed_only=not all_)
    logger.info("Requeued {} job(s) in {!r}: {}", len(keys), queue, keys or "-")


@app.command()
def release(
    queue: str = typer.Argument(..., help="Queue holding the claim."),
    key: str = typer.Argument(..., help="Job key (or jobkey) whose claim to drop."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Required. Releases even ANOTHER node's claim — only do this when the owning "
        "node is confirmed dead/idle, or the job may run twice.",
    ),
    root: Path = typer.Option(None, help=ROOT_HELP),
) -> None:
    """Force-remove a stuck claim, ignoring the node guard (needs --force; never automatic)."""
    configure_loguru()
    root = _root(root)
    if not force:
        raise typer.BadParameter("release ignores the node safety guard; pass --force to confirm")
    existed = store.force_release(root, queue, key)
    logger.info("{} claim for {!r} in {!r}", "Released" if existed else "No", key, queue)


@app.command()
def stop(
    root: Path = typer.Option(None, help=ROOT_HELP),
    clear: bool = typer.Option(False, "--clear", help="Remove the stop file instead of setting it."),
    now: bool = typer.Option(
        False,
        "--now",
        help="Also kill this node's running cells and requeue them without charging an "
        "attempt (preemption; cells without mid-run resume restart from scratch).",
    ),
    prefix: str = typer.Option("", "--prefix", help="With --now: only queues named PREFIX*."),
    grace_s: float = typer.Option(60.0, "--grace-s", help="With --now: SIGTERM -> SIGKILL wait."),
    timeout_s: float = typer.Option(900.0, "--timeout-s", help="With --now: give up after."),
) -> None:
    """Signal this node's workers to exit after their current jobs (or --clear to resume)."""
    from subliminal_learning.jobqueue.preempt import preempt_host

    configure_loguru()
    root = _root(root)
    host = store.this_host()
    p = store.stop_path(root, host)
    if clear:
        if now:
            raise typer.BadParameter("--clear and --now are exclusive")
        p.unlink(missing_ok=True)
        logger.info("Cleared stop file {}", p)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    logger.info("Set stop file {} (workers exit after current jobs)", p)
    if not now:
        return
    report = preempt_host(
        root, host, prefix=prefix, grace_s=grace_s, timeout_s=timeout_s, log=logger.info
    )
    logger.info(
        "preempt {}: requeued={} finished={} released={} unsettled={} waiting={}",
        host, len(report.requeued), len(report.finished), len(report.released),
        list(report.unsettled), list(report.waiting),
    )
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def wait(
    queue: str = typer.Argument(..., help="Queue to block on."),
    poll_s: float = typer.Option(60.0, "--poll-s", help="Seconds between completion checks."),
    root: Path = typer.Option(None, help=ROOT_HELP),
) -> None:
    """Block until QUEUE is complete (every job terminal); for chaining from a shell."""
    import time

    configure_loguru()
    root = _root(root)
    if not store.queue_exists(root, queue):
        raise typer.BadParameter(f"queue {queue!r} does not exist under {root}")
    while not store.queue_complete(root, queue):
        st = store.queue_status(root, queue)
        logger.info(
            "waiting on {!r}: pending={} running={} done={} failed={}",
            queue, st["pending"], st["running"], st["done"], st["failed"],
        )
        time.sleep(poll_s)
    logger.info("queue {!r} complete", queue)


# --------------------------- Slurm layer (login node) ---------------------------

NAME_HELP = "Campaign name (state under <root>/slurm/<name>/, Slurm jobs <name>-pool / <name>-pin.<node>)."


@app.command("slurm-supervise")
def slurm_supervise(
    root: Path = typer.Option(None, help=ROOT_HELP),
    name: str = typer.Option(..., help=NAME_HELP),
    pin: Path = typer.Option(..., help="Code checkout whose .venv runs the pools."),
    policy_template: Path = typer.Option(..., "--policy-template", help="GPU policy template JSON."),
    queues: str = typer.Option(None, help="Comma list of queues (required on a shared root)."),
    gpus_per_pool: int = typer.Option(4, "--gpus-per-pool"),
    workers_per_gpu: int = typer.Option(1, "--workers-per-gpu", help="= the policy's cap_per_gpu."),
    max_gpus: int = typer.Option(32, "--max-gpus", help="This campaign's GPU share."),
    user_gpu_cap: int = typer.Option(32, "--user-gpu-cap", help="Cap on all your GPUs in squeue."),
    lanes: str = typer.Option("burst", help="lab --lane value."),
    preemptible: bool = typer.Option(True, "--preemptible/--no-preemptible", help="Also allow borrow."),
    time_limit: str = typer.Option("8h", "--time", help="lab --time per pool."),
    claim_margin_s: int = typer.Option(1800, "--claim-margin-s"),
    idle_exit_s: int = typer.Option(900, "--idle-exit-s"),
    max_pending_pools: int = typer.Option(2, "--max-pending-pools"),
    mem_per_gpu: str = typer.Option(None, "--mem-per-gpu", help="e.g. 48G (default: lab's)."),
    done_marker: Path = typer.Option(None, "--done-marker", help="Exit once this file exists."),
    interval_s: float = typer.Option(120.0, "--interval-s"),
) -> None:
    """Keep this campaign's Slurm pools submitted while work is pending (login node, tmux)."""
    load_dotenv()
    configure_loguru()
    cfg = slurm_layer.SupervisorConfig(
        root=_root(root),
        name=name,
        pin=pin.resolve(),
        policy_template=policy_template.resolve(),
        queues=tuple(x.strip() for x in queues.split(",") if x.strip()) if queues else None,
        gpus_per_pool=gpus_per_pool,
        workers_per_gpu=workers_per_gpu,
        max_gpus=max_gpus,
        user_gpu_cap=user_gpu_cap,
        lanes=lanes,
        preemptible=preemptible,
        time=time_limit,
        claim_margin_s=claim_margin_s,
        idle_exit_s=idle_exit_s,
        max_pending_pools=max_pending_pools,
        mem_per_gpu=mem_per_gpu,
        done_marker=done_marker,
        interval_s=interval_s,
    )
    if not (cfg.pin / ".venv" / "bin" / "python").exists():
        raise typer.BadParameter(f"{cfg.pin}/.venv is missing: run `UV_FROZEN=1 uv sync` there first")
    if not cfg.policy_template.exists():
        raise typer.BadParameter(f"policy template {cfg.policy_template} does not exist")
    slurm_layer.supervise(cfg)


@app.command("slurm-pause")
def slurm_pause(
    root: Path = typer.Option(None, help=ROOT_HELP),
    name: str = typer.Option(..., help=NAME_HELP),
    now: bool = typer.Option(
        False, "--now", help="Also cancel the pools; running cells are requeued without charging an attempt."
    ),
) -> None:
    """Pause a campaign: no new pools, waiting pool jobs cancelled, live pools drain (or stop now).

    Every pool reads ``PAUSED`` (at start and each poll) and closes: running cells
    finish, nothing new is claimed. ``--now`` also cancels the running pools, whose
    cells go back to pending without an attempt charge.
    """
    import subprocess

    configure_loguru()
    root = _root(root)
    jobs = slurm_layer.query_squeue()
    if jobs is None:
        raise typer.Exit(1)
    p = slurm_layer.paused_path(root, name)
    store.mkdir_open(p.parent)
    p.touch()
    own = slurm_layer.campaign_pool_jobs(name, jobs)
    cancel = sorted(j.job_id for j in own if now or j.state == "PENDING")
    draining = sorted(j.job_id for j in own if j.job_id not in cancel)
    logger.info("Paused {}: draining pool jobs {}", name, draining or "-")
    if cancel:
        subprocess.run(["lab", "cancel", *cancel], check=True)
        logger.info("Cancelled pool jobs {}", cancel)


@app.command("slurm-resume")
def slurm_resume(
    root: Path = typer.Option(None, help=ROOT_HELP),
    name: str = typer.Option(..., help=NAME_HELP),
) -> None:
    """Resume a paused campaign: clear PAUSED (stop files are never touched)."""
    configure_loguru()
    root = _root(root)
    slurm_layer.paused_path(root, name).unlink(missing_ok=True)
    logger.info("Resumed {}; the supervisor submits pools on its next pass", name)


@app.command("slurm-status")
def slurm_status(
    root: Path = typer.Option(None, help=ROOT_HELP),
    name: str = typer.Option(..., help=NAME_HELP),
) -> None:
    """Read-only: supervisor heartbeat, pools (with squeue state) and queue counts."""
    configure_loguru()
    root = _root(root)
    sup = slurm_layer.read_json(slurm_layer.state_dir(root, name) / "supervisor.json") or {}
    jobs = slurm_layer.query_squeue()
    if jobs is None:
        typer.echo("squeue unavailable: pool states unknown")
        jobs = {}
    paused = slurm_layer.paused_path(root, name).exists()
    typer.echo(
        f"{name}: supervisor updated {sup.get('updated_utc', 'never')} host={sup.get('host')} "
        f"paused={paused} held_back={sup.get('held_back') or '-'} "
        f"own_gpus={sup.get('own_gpus')} user_gpus={sup.get('user_gpus')}"
    )
    for node, rec in slurm_layer.pool_records(root, name).items():
        if rec.get("ended_utc") or rec.get("released_utc"):
            continue
        job = jobs.get(str(rec.get("slurm_job_id")))
        typer.echo(
            f"  pool {node} job={rec.get('slurm_job_id')} state={job.state if job else 'gone'} "
            f"host={rec.get('host')} gpus={rec.get('gpus')} claim_until={rec.get('claim_until_utc')}"
        )
    for q in store.list_queues(root):
        st = store.queue_status(root, q)
        typer.echo(
            f"  {q}: pending={st['pending']} running={st['running']} "
            f"done={st['done']} failed={st['failed']}"
        )


if __name__ == "__main__":
    app()
