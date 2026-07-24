"""CLI: a file-based GPU job queue + per-node worker pool (``uv run python -m scripts.jobq``).

The repo-proper replacement for the per-sweep shell launchers. Submit jobs into named
queues under a shared ``--root`` (default ``run_logs/queue``), then run one foreground
worker pool per node; the pool claims, GPU-schedules, runs, and records each job. Two nodes
can drain the same root safely — a worker only ever acts on its OWN node's claims/pids.

Subcommands: ``submit`` / ``work`` / ``status`` / ``requeue`` / ``release`` / ``stop`` /
``wait``. See ARCHITECTURE.md ("Job queue framework") for the layout and a worked example.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuManager
from subliminal_learning.jobqueue.model import parse_jobs_file
from subliminal_learning.jobqueue.worker import run_pool
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_ROOT = Path("run_logs/queue")


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
    root: Path = typer.Option(DEFAULT_ROOT, help="Queue state root."),
    dry_run: bool = typer.Option(False, help="Print the expanded jobs and exit."),
) -> None:
    """Expand a jobs file and append the jobs to QUEUE (creating meta on first submit)."""
    configure_loguru()
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
    root: Path = typer.Option(DEFAULT_ROOT, help="Queue state root."),
    workers: int = typer.Option(1, help="Number of worker threads (per-GPU slot cap is global)."),
    queues: str = typer.Option(None, help="Comma list restricting which queues to drain."),
) -> None:
    """Run the foreground worker pool on THIS node (nohup it yourself for long runs)."""
    load_dotenv()
    configure_loguru()
    q = [x.strip() for x in queues.split(",") if x.strip()] if queues else None
    logger.info("Starting worker pool on {} ({} threads) under {}", store.this_host(), workers, root)
    run_pool(root, workers=workers, queues=q)


@app.command()
def status(
    queue: str = typer.Argument(None, help="One queue (default: all queues under --root)."),
    root: Path = typer.Option(DEFAULT_ROOT, help="Queue state root."),
) -> None:
    """Per-queue pending/running/done/failed counts + this node's per-GPU slot occupancy."""
    configure_loguru()
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
    try:
        occ = GpuManager(root, store.this_host()).occupancy()
        if occ:
            logger.info(
                "this node ({}) slot occupancy: {}",
                store.this_host(),
                ", ".join(f"gpu{g}={n}" for g, n in sorted(occ.items())),
            )
    except FileNotFoundError:
        pass  # no policy on this node -> nothing to report


@app.command()
def requeue(
    queue: str = typer.Argument(..., help="Queue to requeue jobs in."),
    all_: bool = typer.Option(False, "--all", help="Requeue done jobs too (default: failed only)."),
    root: Path = typer.Option(DEFAULT_ROOT, help="Queue state root."),
) -> None:
    """Make terminal jobs pending again (failed-only by default) so a worker retries them."""
    configure_loguru()
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
    root: Path = typer.Option(DEFAULT_ROOT, help="Queue state root."),
) -> None:
    """Force-remove a stuck claim, ignoring the node guard (needs --force; never automatic)."""
    configure_loguru()
    if not force:
        raise typer.BadParameter("release ignores the node safety guard; pass --force to confirm")
    existed = store.force_release(root, queue, key)
    logger.info("{} claim for {!r} in {!r}", "Released" if existed else "No", key, queue)


@app.command()
def stop(
    root: Path = typer.Option(DEFAULT_ROOT, help="Queue state root."),
    clear: bool = typer.Option(False, "--clear", help="Remove the stop file instead of setting it."),
) -> None:
    """Signal this node's workers to exit after their current jobs (or --clear to resume)."""
    configure_loguru()
    p = store.stop_path(root, store.this_host())
    if clear:
        p.unlink(missing_ok=True)
        logger.info("Cleared stop file {}", p)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        logger.info("Set stop file {} (workers exit after current jobs)", p)


@app.command()
def wait(
    queue: str = typer.Argument(..., help="Queue to block on."),
    poll_s: float = typer.Option(60.0, "--poll-s", help="Seconds between completion checks."),
    root: Path = typer.Option(DEFAULT_ROOT, help="Queue state root."),
) -> None:
    """Block until QUEUE is complete (every job terminal); for chaining from a shell."""
    import time

    configure_loguru()
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


if __name__ == "__main__":
    app()
