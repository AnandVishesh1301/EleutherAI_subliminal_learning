"""Queues, dependency-gated submission, the watch loop and pause controls of `rev2`.

Home node: soar. Each cell goes to one host queue per tier (`rev2_plan.cell_tier`): ``g``
(cells others depend on), ``h`` (headline), ``l`` (low priority). A cell is submitted only
when its inputs exist; its queue is recorded in an assignment ledger, which
`move-pending` rewrites so a long-paused host's waiting cells run elsewhere (the stale copy
exits 0 if ever claimed). Completion is read from each cell's own done record, never from
queue counts. Node-local controls (pause / resume / fence / unfence) only write the stop
sentinel, a fence file and this host's policy, and record preemptions so a preempted cell
never uses up its retries. Plan: `.claude/plans/blogpost/12_reversal.plan.md` §15-§16.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import rev2_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job, sanitize_jobkey
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_PREFIX = f"{plan.CAMPAIGN}-"


# ---- ledgers ----------------------------------------------------------------------------


def _read_json(path: Path, default):  # noqa: ANN001, ANN202
    return json.loads(path.read_text()) if path.is_file() else default


def assignment() -> dict[str, str]:
    """Cell key -> queue that owns it."""
    return _read_json(plan.ASSIGNMENT_FILE, {})


def assigned_queue(cell: plan.Cell) -> str | None:
    return assignment().get(cell.key)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def preemptions() -> Counter:
    """Preemptions recorded per cell key."""
    out: Counter = Counter()
    if plan.PREEMPTIONS_FILE.is_file():
        for line in plan.PREEMPTIONS_FILE.read_text().splitlines():
            if line.strip():
                out[json.loads(line)["key"]] += 1
    return out


def _log_control(action: str, **fields) -> None:  # noqa: ANN003
    _append_jsonl(plan.CONTROL_LOG, {"at": utc_now_iso(), "host": socket.gethostname(),
                                     "action": action, **fields})


# ---- policy + fence ----------------------------------------------------------------------


def fenced_gpus(host: str) -> tuple[int, ...]:
    return tuple(_read_json(plan.fence_path(host), {"gpus": []})["gpus"])


def write_policy(host: str) -> Path:
    """Write `host`'s policy = plan policy minus its fenced GPUs (deterministic)."""
    path = plan.QUEUE_ROOT / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json(host, fenced_gpus(host))
    plan.QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or json.loads(path.read_text()) != doc:
        atomic_write_json(path, doc)
    return path


# ---- cells -> jobs -------------------------------------------------------------------------


def cell_job(cell: plan.Cell, queue: str) -> Job:
    cmd = (
        "uv run python -m scripts.run_rev2_campaign cell "
        f"--key '{cell.key}' --queue {queue} --device cuda:0"
    )
    if "--force" in cmd or "angela" in cmd.lower():
        raise ValueError(f"forbidden token in {cmd}")
    return Job(key=cell.key, cmd=cmd, mem_mib=plan.cell_mem_mib(cell), slots=1)


def _ensure_queue(name: str, host: str, tier: str, code_sha: str) -> None:
    root = plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME / code_sha
    defaults = code_pin_io.pin_job_defaults(code_pin_io.CodePin(sha=code_sha, root=root))
    store.create_queue(plan.QUEUE_ROOT, name, node=host, priority=plan.TIER_PRIORITY[tier],
                       defaults=defaults, exist_ok=True)
    meta = store.read_meta(plan.QUEUE_ROOT, name)
    if meta.defaults.get("cwd") != defaults["cwd"]:
        raise RuntimeError(f"{name} is pinned to {meta.defaults.get('cwd')}, not {root}")


def _is_done(cell: plan.Cell) -> bool:
    from pipelines.rev2_cell_pipeline import is_done

    return is_done(cell)


def needs(cell: plan.Cell) -> list[Path]:
    """Files that must exist before `cell` can run (beyond the staging report)."""
    from pipelines import rev2_cell_pipeline as rc

    out: list[Path] = []
    if cell.block == "L":
        out.append(plan.POOL_REV2 / "manifest.json")
    if cell.teacher == "pile" or cell.arm == "bridge":
        out += [plan.POOL_JUDGED / "manifest.json", plan.SFT_PILE_DIR / "manifest.json"]
    if cell.src == "scale":
        owner = plan.Cell("R2", "posctl", cell.channel, "start", plan.TREAT,
                          rc.group_animal(cell), cell.start, cell.seed, cell.tfs)
        out.append(plan.cell_root(owner) / rc.CELL_RECORD)
    if cell.src == "r1l":
        out.append(plan.cell_root(rc._source_cell(cell, src="r1l")) / rc.CELL_RECORD)
    return out


def _stop_ready(cell: plan.Cell) -> bool:
    from pipelines.rev2_cell_pipeline import group_animal, stop_key

    steps = _read_json(plan.STOP_STEPS_FILE, {})
    entry = steps.get(stop_key(cell.channel, cell.start, group_animal(cell)))
    return entry is not None and entry["step"] is not None


def _no_step_cells(stop_steps: dict) -> list[str]:
    """Stop-student cells of groups whose probe gap never fell far enough (holes)."""
    from pipelines.rev2_cell_pipeline import group_animal, stop_key

    dead = {k for k, v in stop_steps.items() if v["step"] is None}
    return [c.key for c in plan.all_cells()
            if c.src == "stop" and stop_key(c.channel, c.start, group_animal(c)) in dead]


def ready(cell: plan.Cell) -> bool:
    if cell.src == "stop" and not _stop_ready(cell):
        return False
    return all(p.is_file() for p in needs(cell))


def paused(host: str) -> bool:
    """Whether `host` holds a rev2 stop sentinel."""
    return store.stop_path(plan.QUEUE_ROOT, host).exists()


def route_host(cell: plan.Cell) -> str:
    """`cell`'s stable host, or the next weighted host of its channel if that one is paused."""
    slots = [h for h, w in plan.HOST_WEIGHTS[cell.channel] for _ in range(w)]
    start = int(hashlib.sha256(cell.key.encode()).hexdigest(), 16) % len(slots)
    for i in range(len(slots)):
        host = slots[(start + i) % len(slots)]
        if not paused(host):
            return host
    return slots[start]


def submit_ready(code_sha: str) -> dict[str, int]:
    """Submit every planned, unsubmitted cell whose inputs exist.

    Returns:
        Newly submitted cells per queue.
    """
    from pipelines.rev2_stage_pipeline import latest_complete_report

    if latest_complete_report() is None:
        raise RuntimeError("no complete rev2 staging report; run `stage` on soar first")
    ledger = assignment()
    by_queue: dict[str, list[plan.Cell]] = defaultdict(list)
    for cell in plan.all_cells():
        if cell.key in ledger or not ready(cell):
            continue
        tier = plan.cell_tier(cell)
        by_queue[plan.queue_name(tier, route_host(cell))].append(cell)
    added = {}
    for name, cells in sorted(by_queue.items()):
        tier, host = name.split("-")[1], _host_of_queue(name)
        _ensure_queue(name, host, tier, code_sha)
        have = {j.key for j in store.load_jobs(plan.QUEUE_ROOT, name)}
        new = [c for c in cells if c.key not in have]
        if new:
            store.append_jobs(plan.QUEUE_ROOT, name, [cell_job(c, name) for c in new])
        for c in cells:
            ledger[c.key] = name
        added[name] = len(new)
    if added:
        atomic_write_json(plan.ASSIGNMENT_FILE, ledger)
        logger.info("rev2 submitted: {}", added)
    return added


def _host_of_queue(name: str) -> str:
    short = name.split("-", 2)[2]
    return next(h for h, s in plan.HOST_SHORT.items() if s == short)


def queues() -> list[str]:
    return [q for q in store.list_queues(plan.QUEUE_ROOT) if q.startswith(_PREFIX)]


def census() -> dict[str, dict]:
    """Read-only per-queue counts (pure file reads)."""
    return {
        q: {k: v for k, v in store.queue_status(plan.QUEUE_ROOT, q).items()
            if k != "running_jobs"}
        for q in queues()
    }


def requeue_failed(max_attempts: int) -> tuple[dict[str, list[str]], list[str]]:
    """Requeue failed jobs whose owner queue is current, allowing one retry per preemption.

    Returns:
        (requeued jobkeys per queue, keys out of retries = holes).
    """
    ledger = assignment()
    pre = preemptions()
    requeued: dict[str, list[str]] = {}
    holes: list[str] = []
    for q in queues():
        keys = {sanitize_jobkey(j.key): j.key for j in store.load_jobs(plan.QUEUE_ROOT, q)}
        retry = []
        for r in store.load_results(plan.QUEUE_ROOT, q):
            if r.get("rc") in (0, None):
                continue
            jk = sanitize_jobkey(r["key"])
            key = keys.get(jk, r["key"])
            if ledger.get(key) != q:
                continue
            if store.read_attempt(plan.QUEUE_ROOT, q, jk) < max_attempts + pre[key]:
                retry.append(jk)
            else:
                holes.append(key)
        if retry:
            requeued[q] = store.requeue(plan.QUEUE_ROOT, q, jobkeys=retry)
    return requeued, holes


# ---- stop-step selection ---------------------------------------------------------------------


def _probe_rows(run_dir: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    path = run_dir / "training_metrics.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["step"])] = r
    return rows


def _p(row: dict, animal: str) -> float:
    if row.get("p_target") is not None and row.get("target_animal") in (None, animal):
        return float(row["p_target"])
    return float(row["p_by_animal"][animal])


def select_stop_steps() -> dict:
    """First saved snapshot step whose group probe gap is ≤ STOP_GAP_FRACTION of step 0.

    Soft groups use the R1L students (step 0 = the bf16 re-probe of the start, the
    autocast-consistent reference); hard groups the rank-2 snapshot re-runs. Written once
    per group when every student of the group is done.
    """
    from pipelines.rev2_cell_pipeline import CELL_RECORD, _find_run_dir, stop_key

    steps = _read_json(plan.STOP_STEPS_FILE, {})
    groups = [("soft", s, a, plan.R1L_SNAPSHOTS, plan.SOFT_LEAVES3)
              for s, a in plan.SOFT_STOP_GROUPS]
    groups += [("hard", s, a, plan.HARD_SNAPSHOTS, plan.SEEDS3)
               for s, a in plan.HARD_R2_GROUPS]
    changed = False
    for channel, start, animal, snaps, members in groups:
        key = stop_key(channel, start, animal)
        if key in steps:
            continue
        curves: dict[str, list[dict[int, float]]] = {plan.TREAT: [], plan.CTRL: []}
        complete = True
        for kind in (plan.TREAT, plan.CTRL):
            for m in members:
                if channel == "soft":
                    t, g = m
                    cell = plan.Cell("L", "rev", "soft", "start", kind, animal, start, g, t,
                                     "cont", None, "base")
                else:
                    cell = plan.Cell("R2", "rev", "hard", "start", kind,
                                     animal if kind == plan.TREAT else None, start, m, None,
                                     "cont", None, "base_rev1")
                rec = _read_json(plan.cell_root(cell) / CELL_RECORD, None)
                if rec is None:
                    complete = False
                    break
                rows = _probe_rows(_find_run_dir(plan.cell_root(cell)))
                curve = {s: _p(r, animal) for s, r in rows.items()}
                if channel == "soft":
                    curve[0] = rec["probe_p_target"]["start_bf16_autocast"]
                curves[kind].append(curve)
            if not complete:
                break
        if not complete:
            continue
        def mean_at(kind: str, s: int) -> float:
            vals = [c[s] for c in curves[kind] if s in c]
            return sum(vals) / len(vals)

        gap0 = mean_at(plan.TREAT, 0) - mean_at(plan.CTRL, 0)
        gaps = {s: mean_at(plan.TREAT, s) - mean_at(plan.CTRL, s) for s in snaps}
        hit = [s for s in snaps if gaps[s] <= plan.STOP_GAP_FRACTION * gap0]
        steps[key] = {"step": hit[0] if hit else None, "gap0": gap0,
                      "gaps": {str(s): g for s, g in gaps.items()}, "at": utc_now_iso()}
        changed = True
    if changed:
        atomic_write_json(plan.STOP_STEPS_FILE, steps)
    return steps


# ---- watch ---------------------------------------------------------------------------------


def campaign_state(holes: list[str]) -> dict:
    """Done / submitted / waiting / hole counts per block and tier (cell records)."""
    ledger = assignment()
    hole_set = set(holes)
    out: dict = defaultdict(Counter)
    waiting: list[str] = []
    for cell in plan.all_cells():
        group = f"{cell.block}|{plan.cell_tier(cell)}"
        if _is_done(cell):
            out[group]["done"] += 1
        elif cell.key in hole_set:
            out[group]["hole"] += 1
        elif cell.key in ledger:
            out[group]["submitted"] += 1
        else:
            out[group]["waiting"] += 1
            waiting.append(cell.key)
    return {"groups": {k: dict(v) for k, v in sorted(out.items())}, "waiting": waiting}


def _hole_dependents(holes: list[str]) -> list[str]:
    """Cells whose inputs can never exist because a producing cell is a hole."""
    if not holes:
        return []
    hole_set = set(holes)
    hole_roots = {str(plan.cell_root(plan.cell_from_key(h))) for h in hole_set}
    return [
        cell.key
        for cell in plan.all_cells()
        if cell.key not in hole_set
        and any(str(need.parent) in hole_roots for need in needs(cell))
    ]


def watch(code_sha: str, interval: int, max_attempts: int) -> int:
    """Submit, requeue, select stop steps, record state; results + done when terminal.

    Returns:
        0 when every planned cell is done or a labelled hole.
    """
    from pipelines import rev2_results_pipeline

    plan.LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = plan.LOG_DIR / plan.CAMPAIGN_STATE_FILE
    built_headline = False
    while True:
        submitted = submit_ready(code_sha)
        requeued, holes = requeue_failed(max_attempts)
        stop_steps = select_stop_steps()
        no_step = [k for k, v in stop_steps.items() if v["step"] is None]
        holes_all = holes + _no_step_cells(stop_steps)
        holes_all += _hole_dependents(holes_all)
        state = campaign_state(holes_all)
        counts = census()
        atomic_write_json(state_path, {
            "at": utc_now_iso(), "submitted": submitted, "requeued": requeued,
            "holes": holes_all, "stop_groups_without_step": no_step,
            "state": state["groups"], "n_waiting": len(state["waiting"]),
            "waiting_sample": state["waiting"][:50], "queues": counts,
        })
        totals = Counter()
        for g in state["groups"].values():
            totals.update(g)
        open_cells = totals["submitted"] + totals["waiting"]
        head_open = sum(v.get("submitted", 0) + v.get("waiting", 0)
                        for k, v in state["groups"].items() if not k.endswith("|l"))
        if head_open == 0 and not built_headline:
            _results(rev2_results_pipeline, "headline")
            built_headline = True
        if open_cells == 0:
            report = _results(rev2_results_pipeline, "final")
            (plan.LOG_DIR / plan.CAMPAIGN_DONE_FILE).write_text(
                json.dumps({"at": utc_now_iso(), "totals": dict(totals),
                            "holes": holes_all, "results": report}) + "\n")
            logger.info("rev2 done: {}", dict(totals))
            return 0
        logger.info("rev2 watch: {}", dict(totals))
        time.sleep(interval)


def _results(module, phase: str) -> dict:  # noqa: ANN001
    try:
        return module.run(phase=phase)
    except Exception as err:  # noqa: BLE001
        logger.exception("rev2 results ({}) failed", phase)
        return {"error": repr(err)}


# ---- node-local controls ------------------------------------------------------------------


def _running_claims(host: str) -> list[tuple[str, str, dict]]:
    """(queue, key, owner) of every live rev2 claim on `host`."""
    out = []
    for q in queues():
        keys = {sanitize_jobkey(j.key): j.key for j in store.load_jobs(plan.QUEUE_ROOT, q)}
        claims = plan.QUEUE_ROOT / q / "claims"
        if not claims.is_dir():
            continue
        for d in claims.iterdir():
            owner = _read_json(d / "owner.json", None)
            if owner and owner.get("node") == host:
                out.append((q, keys.get(d.name, d.name), owner))
    return out


def preempt(host: str, gpus: tuple[int, ...] | None, reason: str) -> list[str]:
    """Terminate rev2 cells running on `host` (optionally only on `gpus`); ledger them.

    Raises:
        RuntimeError: Called from another host (R6: controls are node-local).
    """
    if socket.gethostname() != host:
        raise RuntimeError(f"run on {host} (this is {socket.gethostname()})")
    killed = []
    for q, key, owner in _running_claims(host):
        if gpus is not None and owner.get("gpu") not in gpus:
            continue
        if not store.cell_alive(owner):
            continue
        _append_jsonl(plan.PREEMPTIONS_FILE, {"at": utc_now_iso(), "key": key, "queue": q,
                                              "host": host, "gpu": owner.get("gpu"),
                                              "reason": reason})
        try:
            os.killpg(int(owner["cell_pid"]), signal.SIGTERM)
        except ProcessLookupError:
            continue
        killed.append(key)
    if killed:
        _clear_claims_after_exit(killed, host)
    return killed


def _clear_claims_after_exit(keys: list[str], host: str, wait_s: int = 120) -> None:
    """Remove the GKD ``.claim`` files of killed cells once no cell process remains.

    A claim names a pid of `host`; on another host that pid may belong to a live,
    unrelated process, so the retry would see the leaf as held until the 24 h TTL.
    """
    deadline = time.time() + wait_s
    while time.time() < deadline and any(
        store.cell_alive(o) for _, k, o in _running_claims(host) if k in keys
    ):
        time.sleep(5)
    removed = []
    for key in keys:
        for claim in plan.cell_root(plan.cell_from_key(key)).rglob(".claim"):
            claim.unlink(missing_ok=True)
            removed.append(str(claim))
    _log_control("clear-claims", keys=keys, removed=removed)


def pause(now: bool) -> dict:
    host = socket.gethostname()
    store.stop_path(plan.QUEUE_ROOT, host).touch()
    killed = preempt(host, None, "pause --now") if now else []
    _log_control("pause", now=now, preempted=killed)
    return {"host": host, "preempted": killed}


def resume() -> dict:
    host = socket.gethostname()
    store.stop_path(plan.QUEUE_ROOT, host).unlink(missing_ok=True)
    write_policy(host)
    live = os.path.exists(store.worker_pid_path(plan.QUEUE_ROOT, host))
    _log_control("resume", pool_pid_file=live)
    return {"host": host, "pool_pid_file": live}


def fence(gpus: tuple[int, ...], now: bool) -> dict:
    host = socket.gethostname()
    current = set(fenced_gpus(host)) | set(gpus)
    atomic_write_json(plan.fence_path(host), {"gpus": sorted(current), "at": utc_now_iso()})
    write_policy(host)
    killed = preempt(host, tuple(gpus), "fence --now") if now else []
    _log_control("fence", gpus=sorted(gpus), now=now, preempted=killed)
    return {"host": host, "fenced": sorted(current), "preempted": killed}


def unfence(gpus: tuple[int, ...] | None) -> dict:
    host = socket.gethostname()
    current = set() if gpus is None else set(fenced_gpus(host)) - set(gpus)
    atomic_write_json(plan.fence_path(host), {"gpus": sorted(current), "at": utc_now_iso()})
    write_policy(host)
    _log_control("unfence", gpus=None if gpus is None else sorted(gpus))
    return {"host": host, "fenced": sorted(current)}


def move_pending(
    src_host: str, dst_hosts: tuple[str, ...], code_sha: str, channel: str | None = None
) -> dict:
    """Reassign `src_host`'s pending or failed cells (of `channel`, or all) to `dst_hosts`.

    A failed cell (e.g. preempted by ``pause --now``) gets a fresh job on its new host; its
    old result stays where it was and is no longer requeued (the ledger moved).
    """
    ledger = assignment()
    moved: dict[str, int] = Counter()
    by_queue: dict[str, list[plan.Cell]] = defaultdict(list)
    for q in queues():
        if _host_of_queue(q) != src_host:
            continue
        tier = q.split("-")[1]
        for job in store.load_jobs(plan.QUEUE_ROOT, q):
            state = store.job_state(plan.QUEUE_ROOT, q, job)
            if ledger.get(job.key) != q or state not in ("pending", "failed"):
                continue
            cell = plan.cell_from_key(job.key)
            if channel is not None and cell.channel != channel:
                continue
            idx = int(hashlib.sha256(job.key.encode()).hexdigest(), 16) % len(dst_hosts)
            by_queue[plan.queue_name(tier, dst_hosts[idx])].append(cell)
    for name, cells in by_queue.items():
        _ensure_queue(name, _host_of_queue(name), name.split("-")[1], code_sha)
        have = {j.key for j in store.load_jobs(plan.QUEUE_ROOT, name)}
        new = [c for c in cells if c.key not in have]
        store.append_jobs(plan.QUEUE_ROOT, name, [cell_job(c, name) for c in new])
        for c in cells:
            ledger[c.key] = name
        moved[name] += len(cells)
    atomic_write_json(plan.ASSIGNMENT_FILE, ledger)
    _log_control("move-pending", src=src_host, dst=list(dst_hosts), channel=channel,
                 moved=dict(moved))
    return dict(moved)


def status() -> dict:
    """Read-only: per host sentinel, fence, and per-queue counts."""
    hosts = {
        h: {"stop": store.stop_path(plan.QUEUE_ROOT, h).exists(),
            "fenced": list(fenced_gpus(h))}
        for h in plan.WORKER_HOSTS
    }
    return {"hosts": hosts, "queues": census(),
            "state": _read_json(plan.LOG_DIR / plan.CAMPAIGN_STATE_FILE, {})}
