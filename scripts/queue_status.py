"""Read-only campaign status for a jobq root: one row per ACTIVE queue with an ETA.

Backs the ``/status`` command. Prints a table of queues that still have pending or
running work (parked queues -- ``meta.node`` starting with ``PARKED`` -- and queues with
a ``complete.stamp`` are hidden unless ``--all``), with:

- ``prio``/``node``: from ``meta.json`` (the pin is the ``node`` field; ``-`` = unpinned).
- ``pend/run/wait/done/fail``: derived from claims + results exactly as ``jobq status`` does,
  except that claims are split by their owner's ``gpu``: ``run`` = claims holding a GPU,
  ``wait`` = claims with ``gpu: null`` (the worker is parked waiting for a GPU, e.g. yielded
  to a foreign user).
- ``t/job``: the EXPECTED duration of one of this queue's jobs (mean over its pending +
  running jobs, which may mix shapes), and the signature that number came from. The job's
  SCRIPTS say what it is, and that decides the key: a student TRAINING is keyed by its
  RECEIVER ARCH + EPOCHS alone, pooled across every script, worker and queue (``160m ep20 ~
  47 m``, ``160m ep10 ~ 25 m``), refined by exact LoRA rank and fp32/tf32 where those hold
  enough samples; a TEACHER sweep by arch + epochs per CELL (``tft·qwen·ep10``), number
  generation per 1k PROMPTS (``ngen·qwen``), evals per RUN (``eval·probe·160m``) and
  trajectories per CHECKPOINT (``traj·ckpt_pile_kl_trajectory·160m``), each multiplied by the
  units that job covers; anything else keeps its script chain + receiver key. Carrying
  ``--epochs`` does not make a job a training job. Receiver/epochs/rank/precision/units come
  from the flags and from the YAMLs, tag lists, registries and run lists the command points
  at. The prior is measured over EVERY historical jobq
  result in this root plus the configured queue folder and every retired root beside it (see
  ``subliminal_learning.eta_priors``: rank+prec -> rank -> prec -> receiver+epochs -> the fitted cost
  MODEL -> another epoch count -> script chain -> per-script sum; newest samples of the best
  available node/MPS tier, resume/cache-hit samples dropped, one queue capped at 2,000
  samples). A shape never measured is REASONED rather than guessed: the model prices a student
  as ``per-epoch cost(receiver, precision) x epochs x rank factor(rank) x load factor(d)``,
  fitted from the same samples, with ``d`` = the pin node's jobs per busy GPU
  (`node_density`, default 5-up) -- ``~17m(160m·ep7·r16·model)``, and ``--json`` then also
  carries ``model_per_epoch_min``, ``rank_factor``, ``load_factor``. The quote is blended with this
  queue's own samples for the same signature, ``w = n_own/(n_own+10)``: ``47m(n=138)`` =
  own-dominated, ``46m(n=5+160m·ep20)`` = blended, ``~47m(160m·ep20)`` = prior only,
  ``~51m(160m·ep20·r128)`` = its rank refinement (the cell names the SHALLOWEST level
  holding the same samples, so a ``|fp32`` or ``|r=N`` pool identical to its parent is not
  spelled out), ``~12m(160m·ep5←ep20)``
  = extrapolated from another epoch count, ``~20m(p,sum)`` = an unmeasured script chain summed
  from its components, ``~10m(?)`` = nothing matched, so ``--unknown-job-s`` is assumed (that
  row's ``eta`` then carries a ``?``; other rows are never affected). ``--json`` carries
  ``expected_s``, ``expected_source``, ``prior_key``, ``prior_label``, ``prior_tier``,
  ``prior_level``, ``prior_n``, ``prior_p10_s``, ``prior_p90_s``.
- ``r/h``: completions per hour over the trailing ``--window-h`` hours (this queue).
- ``eta``: ONE best-estimate time until this queue drains, from a priority-order schedule at
  each node's CAPACITY (not its current occupancy). A node's capacity is
  ``(len(gpus) - yielded GPUs) * cap_per_gpu`` from ``gpu_policy.<host>.json``, and it counts
  only if the node is LIVE (on-GPU claims now, or its newest ``pool_*.<host>.log`` was
  touched within ``--pool-live-min``). Jobs consume ``slots`` (job field, else the queue's
  ``defaults.slots``, else 1); ``slots: 0`` jobs run in a separate per-node CPU lane of
  ``cpu_cap`` and never take GPU slots. The schedule is a DISCRETE-EVENT simulation of the
  pools: every lane starts with its currently running jobs holding their slots (each finishes
  in ``max(expected - age, 0.1 * expected)``), and whenever slots free, each lane claims jobs
  from the highest-priority READY queue that may run there (worker claim order ``(-priority,
  created_utc, name)``), exactly as a pool thread does. So active queues claim free slots in
  priority order as they free up, and a dependent starts when its dependency finishes --
  pre-empting lower-priority work that is still pending at that moment. A queue pinned to
  ``X`` runs on ``X`` only; an unpinned queue may claim on every live node, so equal-priority
  queues on different nodes drain concurrently and equal-priority queues on one node share
  its slots. A trailing ``r/h`` resting on >= 10 completions cross-checks the result: an eta
  it contradicts by more than 50% either way gets a ``!`` (the simulated number itself is
  unchanged; ``--json``: ``eta_throughput_s``, ``eta_disagree``), unless the queue only just
  started (its first completion must be older than the window plus one job, or its trailing
  rate is still ramping up). ``yielded``
  = its pin node has no capacity and its ``yielded.<host>.json`` lists GPUs handed to a
  foreign user (``--json``: ``blocked_reason``, ``yielded_gpus``, ``last_reclaim_utc`` = last
  ``RECLAIM`` line in that node's newest pool log tail); ``blocked`` = nothing can run it (no
  live node with capacity, or a dependency is blocked / parked / missing), ``--json``
  ``blocked_reason``. Preemption and lower-priority jobs already holding GPUs are not
  modelled. ``--json`` also keeps ``start_s``/``finish_s`` and ``eta_wave_s``/
  ``eta_throughput_s``.
- ``work`` (line after ``totals:``): per pin node (plus ``unpinned``) the GPU-hours of queued
  work if it all ran now undisturbed: ``sum(expected * slots)`` over pending + waiting jobs
  plus each running job's remaining time, with the wall-clock that takes at the node's
  CAPACITY. The header shows ``used/capacity`` per node and any yielded GPUs.
- ``stale``: on-GPU claims older than ``--stale-x`` x the job's expected duration --
  dead pool thread / crashed node candidates (``jobq status`` shows the owner node). A job
  whose duration this queue never measured itself (unknown signature, or a prior borrowed
  from other queues) is judged instead by the flat ``--stale-unknown-s`` (3 h), and the count
  is marked ``?``: 3 x a 10-minute guess would call a normal job dead after half an hour.
- ``mem``: GiB each pending/running job would be gated on -- the worker's
  ``_effective_mem``: ``max(job.mem_mib or defaults.mem_mib, persisted OOM floor)``; a
  ``min-max`` range when jobs differ, ``^`` when an escalated floor lifts any job above its
  static ask, ``-`` when no job/default ask exists (the pool's node policy default then
  applies, unknowable from here) or when the queue has no pending/running jobs.

The priors table is cached incrementally in ``<root>/duration_priors_cache.json`` (one status
line reports its age and sample count); ``--rebuild-priors`` forces a full rescan.
``--show-priors`` prints the REASONED sheet instead of the table -- the fitted student cost
model (per-epoch cost, rank factors, load factors, residual band) and the per-unit rates of
the other roles, with worked examples -- and ``--raw`` appends the raw measured cells (per
signature: sample tier, n, p10/p50/p90, p50 by GPU co-tenancy bucket ``c1..c6``, p50 with MPS
off/on, measured peak GPU memory). ``--show-priors --markdown`` regenerates the whole of
``docs/QUEUE_ETA_HEURISTICS.md`` (sheet + raw appendix).

Usage::

    uv run python -m scripts.queue_status [--root QUEUE_ROOT] [--all] [--json]

``--root`` defaults to ``queue_root`` in the paths YAML, else the ``queue_folder`` in
``jobq_paths.toml`` (``config.resolve_queue_root``).
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import os
import statistics
import sys
import textwrap
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from subliminal_learning.config import resolve_queue_root
from jobq import store
from subliminal_learning import eta_priors
from subliminal_learning.eta_priors import (  # re-exported for callers/tests
    Feats,
    NodeCapacity,
    build_priors,
    expected_duration,
    job_feats,
    job_kind,
    model_arch,
    node_capacity,
    probe_list_info,
)

__all__ = [
    "Feats", "NodeCapacity", "build_priors", "expected_duration", "job_feats", "job_kind",
    "model_arch", "node_capacity", "probe_list_info", "main",
]

STALE_UNKNOWN_S = 3 * 3600.0  # stale age for a duration this queue never measured itself
PATHS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paths" / "paths_local.yaml"
HEURISTICS_DOC = Path(__file__).resolve().parents[1] / "docs" / "QUEUE_ETA_HEURISTICS.md"


def _default_root() -> Path:
    try:
        text = PATHS_CONFIG.read_text()
    except FileNotFoundError:
        return resolve_queue_root({})
    return resolve_queue_root(yaml.safe_load(text) or {})


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def completions_per_hour(
    results: list[dict], *, now: datetime, window_h: float
) -> float | None:
    """Successful completions per hour over the trailing window; ``None`` if none."""
    cutoff = now - timedelta(hours=window_h)
    n = 0
    for r in results:
        if r.get("rc", 1) != 0:
            continue
        end = _parse_iso(r.get("end_utc"))
        if end is not None and end >= cutoff:
            n += 1
    return n / window_h if n else None


def eta_s(
    *,
    pending: int,
    running: int,
    expected: float | None,
    rate_per_h: float | None,
    running_ages_s: list[float] | None = None,
) -> tuple[float | None, float | None]:
    """(wave-model ETA, throughput-model ETA) in seconds; ``None`` where undefined.

    Wave: the current running set finishes in ~(expected - age), then pending drains at the
    same concurrency. Throughput: outstanding work at the trailing completion rate.
    """
    wave = None
    if expected is not None and running > 0:
        ages = running_ages_s or []
        remaining = max((expected - a for a in ages), default=expected)
        wave = max(remaining, 0.0) + expected * pending / running
    thr = None
    if rate_per_h:
        thr = (pending + running) / rate_per_h * 3600.0
    return wave, thr


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h < 48 else f"{h / 24:.1f}d"


def fmt_expected(row: dict) -> str:
    """``t/job`` cell: the expected per-job time and the signature it came from.

    ``~47m(160m\u00b7ep20)`` = the prior for that receiver + epoch count,
    ``~51m(160m\u00b7ep20\u00b7r128)`` its rank refinement, ``46m(n=5+160m\u00b7ep20)`` the
    blend with five own samples, ``47m(n=138)`` own samples alone, ``~10m(?)`` nothing matched.
    """
    v = row.get("expected_s")
    if v is None:
        return "?"
    src = row.get("expected_source")
    n = row.get("n_own", 0)
    lbl = row.get("prior_label") or "p"
    if src == "unknown":
        return f"~{fmt_dur(v)}(?)"
    if src in ("prior", "prior_ep", "prior_sum"):
        return f"~{fmt_dur(v)}({lbl})"
    if src == "blend":
        return f"{fmt_dur(v)}(n={n}+{lbl})"
    return f"{fmt_dur(v)}(n={n})"


def effective_mem_mib(job_mem: object, default_mem: object, floor: int) -> tuple[int | None, bool]:
    """(ask MiB, floor-escalated?) mirroring ``Worker._effective_mem`` without the node default.

    Malformed asks count as absent. ``None`` = neither job nor queue sets an ask and no floor.
    """
    static = None
    for v in (job_mem, default_mem):
        try:
            static = int(v)  # type: ignore[arg-type]
            break
        except (TypeError, ValueError):
            continue
    if floor and (static is None or floor > static):
        return floor, True
    return static, False


def fmt_mem(asks: list[int | None], escalated: bool) -> str:
    """``20G`` / ``16-24G`` (GiB, compact) with ``^`` when any floor escalated; ``-`` if none."""
    vals = [a for a in asks if a is not None]
    if not vals:
        return "-"
    def g(m: int) -> str:
        return f"{m / 1024:.0f}" if m >= 10240 else f"{m / 1024:.1f}".removesuffix(".0")
    lo, hi = g(min(vals)), g(max(vals))
    return (lo if lo == hi else f"{lo}-{hi}") + "G" + ("^" if escalated else "")


def _job_slots(job_slots: object, default_slots: object) -> int:
    """Slots a job consumes: its own field, else the queue default, else 1 (0 = CPU lane)."""
    for v in (job_slots, default_slots):
        try:
            return max(int(v), 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return 1


def _scan_jobs(root: Path, name: str, defaults: dict) -> tuple[dict, list[int | None], bool]:
    """One pass over the queue's jobs: ``store.queue_status``-shaped counts + active asks."""
    counts = {store.PENDING: 0, store.RUNNING: 0, store.DONE: 0, store.FAILED: 0}
    running: list[dict] = []
    asks: list[int | None] = []
    escalated = False
    pending: list[tuple[str, int]] = []
    default_mem = defaults.get("mem_mib")
    default_slots = defaults.get("slots")
    for job in store.load_jobs(root, name):
        st = store.job_state(root, name, job)
        counts[st] += 1
        slots = _job_slots(job.slots, default_slots)
        if st in (store.PENDING, store.RUNNING):
            ask, esc = effective_mem_mib(
                job.mem_mib, default_mem, store.read_mem_floor(root, name, job.jobkey))
            asks.append(ask)
            escalated |= esc
        if st == store.PENDING:
            pending.append((job.cmd, slots))
        elif st == store.RUNNING:
            owner = store.read_owner(root, name, job.jobkey) or {}
            running.append({"key": job.key, "cmd": job.cmd, "slots": slots,
                            "start_utc": owner.get("start_utc"),
                            "node": owner.get("node"), "gpu": owner.get("gpu")})
    return ({**counts, "total": sum(counts.values()), "running_jobs": running,
             "pending_jobs": pending}, asks, escalated)


def queue_row(
    root: Path,
    name: str,
    *,
    now: datetime,
    priors: eta_priors.Priors,
    own_samples: list[eta_priors.Sample] | None = None,
    unknown_job_s: float = 600.0,
    window_h: float,
    stale_x: float,
    stale_unknown_s: float = STALE_UNKNOWN_S,
    densities: dict[str, float] | None = None,
    feats_cache: eta_priors.FeatsCache | None = None,
) -> dict:
    meta = store.read_meta(root, name)
    # The model prices a job at the load of the node this queue runs on (its pin, else the
    # mean over the nodes running anything); `None` = the model's default 5-up.
    dens = (densities or {}).get(str(meta.node)) if meta.node else (
        statistics.mean(densities.values()) if densities else None)
    st, asks, escalated = _scan_jobs(root, name, meta.defaults or {})
    results = store.load_results(root, name)
    own_by_key = eta_priors.samples_by_key(own_samples or [])
    feats_of = (feats_cache or eta_priors.FeatsCache(None)).feats  # parse each cmd once, ever
    cache: dict[str, eta_priors.Expectation] = {}

    def exp_of(cmd: str) -> eta_priors.Expectation:
        if cmd not in cache:
            cache[cmd] = expected_duration(feats_of(cmd), own_by_key, priors,
                                           unknown_s=unknown_job_s, density=dens)
        return cache[cmd]

    on_gpu = [r for r in st["running_jobs"] if r.get("gpu") is not None]
    waiting_jobs = [(r["cmd"], r["slots"]) for r in st["running_jobs"]
                    if r.get("gpu") is None]
    queued = st["pending_jobs"] + waiting_jobs
    work_gpu = sum(exp_of(c).seconds * s for c, s in queued if s > 0)
    work_cpu = sum(exp_of(c).seconds for c, s in queued if s == 0)
    gpu_nodes: dict[str, int] = {}
    wait_nodes: dict[str, int] = {}
    for r in st["running_jobs"]:
        d = gpu_nodes if r.get("gpu") is not None else wait_nodes
        d[str(r.get("node"))] = d.get(str(r.get("node")), 0) + 1
    ages: list[float] = []
    remaining: list[float] = []
    run_slot_s: dict[str, float] = {}
    run_cpu_s: dict[str, float] = {}
    sched_running: list[tuple[float, int, str]] = []
    stale = 0
    stale_unknown = 0  # of those, flagged under the unknown/borrowed rule (a flat wall clock)
    for r in on_gpu:
        t = _parse_iso(r.get("start_utc"))
        ex = exp_of(r["cmd"])
        e = ex.seconds
        age = (now - t).total_seconds() if t is not None else 0.0
        ages.append(age)
        left = max(e - age, 0.1 * e)
        remaining.append(left)
        d = run_cpu_s if r["slots"] == 0 else run_slot_s
        d[str(r.get("node"))] = d.get(str(r.get("node")), 0.0) + left * max(r["slots"], 1)
        sched_running.append((left, r["slots"], str(r.get("node"))))
        # An expectation this queue did not MEASURE (unknown signature, or a prior borrowed
        # from other queues) is not tight enough to declare a 30-minute-old claim dead, so
        # those jobs go stale on a flat wall clock instead of a multiple of a guess.
        own_measured = ex.source in ("own", "blend")
        limit = e * stale_x if own_measured else stale_unknown_s
        if age > limit:
            stale += 1
            stale_unknown += not own_measured
    cmds = [c for c, _ in queued] + [r["cmd"] for r in on_gpu]
    exps = [exp_of(c) for c in cmds]
    expected = statistics.mean([e.seconds for e in exps]) if exps else None
    # The row quotes the provenance of its MODAL job (queues may mix signatures).
    rep_cmd = Counter(cmds).most_common(1)[0][0] if cmds else ""
    rep = exp_of(rep_cmd) if cmds else None
    source = rep.source if rep else None
    prior = rep.prior if rep else None
    rate = completions_per_hour(results, now=now, window_h=window_h)
    firsts = [t for t in (_parse_iso(r.get("end_utc")) for r in results
                          if r.get("rc", 1) == 0) if t is not None]
    history_s = (now - min(firsts)).total_seconds() if firsts else 0.0
    wave, thr = eta_s(
        pending=st["pending"] + st["running"] - len(on_gpu), running=len(on_gpu),
        expected=expected, rate_per_h=rate, running_ages_s=ages,
    )
    node = meta.node
    return {
        "queue": name,
        "prio": meta.priority,
        "created_utc": meta.created_utc,
        "node": node or "-",
        "mem_mib_min": min((a for a in asks if a is not None), default=None),
        "mem_mib_max": max((a for a in asks if a is not None), default=None),
        "mem_floor_escalated": escalated,
        "mem": fmt_mem(asks, escalated),
        "parked": bool(node and str(node).startswith("PARKED")),
        "complete": store.queue_complete(root, name),
        "depends_on": list(meta.depends_on or []),
        "pending": st["pending"],
        "running": len(on_gpu),
        "waiting": st["running"] - len(on_gpu),
        "gpu_nodes": gpu_nodes,
        "wait_nodes": wait_nodes,
        "running_ages_s": ages,
        "run_remaining_s": remaining,
        "run_slot_s": run_slot_s,
        "run_cpu_s": run_cpu_s,
        "sched_pending": [(exp_of(c).seconds, sl) for c, sl in queued],
        "sched_running": sched_running,
        "work_gpu_slot_s": work_gpu,
        "work_cpu_s": work_cpu,
        "done": st["done"],
        "failed": st["failed"],
        "total": st["total"],
        "n_own": rep.n_own if rep else 0,
        "expected_s": expected,
        "expected_source": source,
        "expected_unknown": any(e.source == "unknown" for e in exps),
        "prior_level": prior.stat.level if prior else None,
        "prior_label": rep.label if rep else None,
        "prior_key": prior.stat.key if prior else None,
        "prior_tier": prior.stat.tier if prior else None,
        "prior_n": prior.stat.n if prior else None,
        "prior_p10_s": prior.stat.p10 if prior else None,
        "prior_p90_s": prior.stat.p90 if prior else None,
        "model_per_epoch_min": (prior.model.per_epoch_s / 60) if prior and prior.model else None,
        "rank_factor": (prior.model.rank_mult(feats_of(rep_cmd).rank)
                        if prior and prior.model else None),
        "load_factor": prior.model.load_mult(prior.bucket) if prior and prior.model else None,
        "load_factor_applied": prior.model.load_applied if prior and prior.model else None,
        "rate_per_h": rate,
        "window_completions": int(round(rate * window_h)) if rate else 0,
        "rate_history_s": history_s,  # since this queue's FIRST completion
        "window_s": window_h * 3600.0,
        "eta_wave_s": wave,
        "eta_throughput_s": thr,
        "stale_running": stale,
        "stale_by_unknown_rule": stale_unknown,
    }


_RECLAIM = re.compile(r"\[([^\]]+)\] RECLAIM gpu=")


def last_reclaim_utc(log: Path, tail_bytes: int = 262144) -> str | None:
    """ISO time of the last ``RECLAIM gpu=`` line within the last ``tail_bytes`` of ``log``."""
    try:
        with log.open("rb") as f:
            f.seek(max(os.fstat(f.fileno()).st_size - tail_bytes, 0))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    hits = _RECLAIM.findall(text)
    return hits[-1] if hits else None


def read_yielded(root: Path) -> dict[str, dict]:
    """host -> ``{"gpus": N yielded, "last_reclaim_utc"}`` from ``yielded.<host>.json`` files
    (hosts with no listed GPUs are omitted); reclaim time from the newest pool log's tail."""
    out: dict[str, dict] = {}
    for f in sorted(root.glob("yielded.*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        host = data.get("hostname") or f.name[len("yielded."):-len(".json")]
        n = len(data.get("gpus") or {})
        if not n:
            continue
        logs = sorted(root.glob(f"pool_*.{host}.log"), key=lambda p: p.stat().st_mtime)
        out[host] = {"gpus": n, "last_reclaim_utc": last_reclaim_utc(logs[-1]) if logs else None}
    return out


def node_density(root: Path) -> dict[str, float]:
    """Per node: jobs per GPU IN USE right now, from the owner records of every claim.

    This is the load the cost model prices a job at (``d`` in its load factor): a node running
    20 jobs over 4 busy GPUs is a 5-up node. Claims without a GPU (parked workers) do not
    count, and a node with no on-GPU claim is absent (the model then assumes
    `eta_priors.MODEL_DEFAULT_BUCKET`).
    """
    per: dict[str, dict[int, int]] = {}
    for p in Path(root).glob("*/claims/*/owner.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        gpu, node = d.get("gpu"), d.get("node")
        if isinstance(gpu, int) and node:
            g = per.setdefault(str(node), {})
            g[gpu] = g.get(gpu, 0) + 1
    return {n: sum(g.values()) / len(g) for n, g in per.items() if g}


def node_concurrency(rows: list[dict]) -> dict[str, int]:
    """On-GPU claims per owner node across ``rows``."""
    out: dict[str, int] = {}
    for r in rows:
        for n, k in (r.get("gpu_nodes") or {}).items():
            out[n] = out.get(n, 0) + k
    return out


def caps_from_rows(rows: list[dict]) -> dict[str, NodeCapacity]:
    """Fallback "capacity" when no policy file is readable: today's on-GPU occupancy."""
    return {n: NodeCapacity(node=n, gpus=c, cap_per_gpu=1, live=True, used=c)
            for n, c in node_concurrency(rows).items() if c > 0}


def node_work(rows: list[dict], caps: dict[str, NodeCapacity]) -> dict[str, dict]:
    """Per pin node (``unpinned`` for ``-``): slot-seconds of outstanding work if run now.

    ``gpu_s`` = ``sum(expected * slots)`` over pending + waiting jobs + each running job's
    remaining time; ``unknown`` = rows whose signature did not match any prior; ``wall_s`` =
    gpu_s / the node's CAPACITY in slots (unpinned: every live node's).
    """
    total_cap = sum(c.slots for c in caps.values())
    out: dict[str, dict] = {}
    for r in rows:
        pin = r.get("node")
        key = "unpinned" if pin in (None, "-") else str(pin)
        w = out.setdefault(key, {"gpu_s": 0.0, "unknown": 0})
        w["gpu_s"] += (r.get("work_gpu_slot_s") or 0.0) + (r.get("work_cpu_s") or 0.0) + sum(
            r.get("run_remaining_s") or [])
        w["unknown"] += bool(r.get("expected_unknown"))
    out.setdefault("unpinned", {"gpu_s": 0.0, "unknown": 0})
    for key, w in out.items():
        c = total_cap if key == "unpinned" else (caps[key].slots if key in caps else 0)
        w.update(capacity=c, wall_s=(w["gpu_s"] / c) if c else None)
    return out


def fmt_work(work: dict[str, dict]) -> str:
    parts = []
    for key in sorted(work, key=lambda k: (k == "unpinned", k)):
        w = work[key]
        s = (f"{key} {w['gpu_s'] / 3600:.1f} (wall {fmt_dur(w['wall_s'])}"
             f"@{w['capacity']})")
        if w["unknown"]:
            s += f" (+{w['unknown']} queues ?)"
        parts.append(s)
    return "work (GPU-h if run now, undisturbed): " + "  ".join(parts)


class _Lane:
    """One node lane (GPU or CPU): capacity, free slots, and the jobs in flight on it."""

    __slots__ = ("cap", "free", "heap")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.free = cap
        self.heap: list[tuple[float, int]] = []  # (finish time, slots held)

    def start(self, t: float, exp: float, slots: int) -> float:
        use = min(max(slots, 1), self.cap)
        self.free -= use
        fin = t + exp
        heapq.heappush(self.heap, (fin, use))
        return fin

    def release(self, t: float) -> None:
        while self.heap and self.heap[0][0] <= t:
            self.free += heapq.heappop(self.heap)[1]


class _QState:
    """One active queue's scheduling state during the event simulation."""

    __slots__ = ("row", "gpu_jobs", "cpu_jobs", "gnodes", "cnodes", "start_s", "finish_s",
                 "deps")

    def __init__(self, row: dict, gnodes: list[str], cnodes: list[str],
                 deps: list[str]) -> None:
        self.row = row
        self.gnodes = set(gnodes)
        self.cnodes = set(cnodes)
        self.deps = deps
        self.gpu_jobs = [(e, s) for e, s in row.get("sched_pending") or [] if s > 0]
        self.cpu_jobs = [(e, s) for e, s in row.get("sched_pending") or [] if s == 0]
        self.start_s: float | None = None
        self.finish_s = max((left for left, _, _ in row.get("sched_running") or []),
                            default=0.0)

    @property
    def resolved(self) -> bool:
        return not self.gpu_jobs and not self.cpu_jobs


RATE_CHECK_MIN_N = 10  # completions the trailing window needs before it may contradict
RATE_CHECK_TOL = 0.5


def _rate_disagrees(row: dict, finish_s: float) -> bool:
    """Whether this queue's measured throughput contradicts the simulated finish by >50%.

    A well-sampled trailing rate (>= ``RATE_CHECK_MIN_N`` completions) that implies a very
    different drain time means the simulation's assumptions are off for this queue (resumes,
    yielded GPUs, a half-idle node); the row is flagged ``!``, never rescheduled. A queue that
    only just started is NOT flagged: until its first completion is older than the window plus
    one job, the trailing rate is still ramping up (a queue 1 h into a 3 h window reads a third
    of its steady-state rate).
    """
    thr = row.get("eta_throughput_s")
    if not thr or not finish_s or row.get("window_completions", 0) < RATE_CHECK_MIN_N:
        return False
    settled = (row.get("window_s") or 0.0) + (row.get("expected_s") or 0.0)
    if row.get("rate_history_s", 0.0) < settled:
        return False
    return abs(thr - finish_s) / finish_s > RATE_CHECK_TOL


def annotate_queue_order(
    rows: list[dict], *, capacity: dict[str, NodeCapacity] | None = None,
    inactive_deps: dict[str, str] | None = None, yielded: dict[str, dict] | None = None,
) -> None:
    """Fill ``start_s``/``finish_s``/``eta``/``blocked_reason`` in place.

    See the module docstring (``eta``) for the schedule. ``capacity`` maps node -> its
    `NodeCapacity` (default: today's on-GPU occupancy, `caps_from_rows`). ``inactive_deps``
    maps dependency names not in ``rows`` to ``"complete"`` (satisfied) or anything else
    (blocked); a dependency in neither is missing (blocked). ``yielded``
    (`read_yielded`) turns a pinned queue whose node lost its capacity to a foreign user
    into ``yielded``.
    """
    yielded = yielded or {}
    caps = capacity if capacity is not None else caps_from_rows(rows)
    gpu_cap = {n: c.slots for n, c in caps.items() if c.slots > 0}
    cpu_cap = {n: c.cpu_slots for n, c in caps.items() if c.cpu_slots > 0}
    waiting = node_concurrency([{"gpu_nodes": r.get("wait_nodes")} for r in rows])
    inactive_deps = inactive_deps or {}
    by_name = {r["queue"]: r for r in rows}
    order = sorted(rows, key=lambda r: (-r["prio"], r.get("created_utc") or "", r["queue"]))

    def block(r: dict, why: str, eta: str = "blocked", **extra: object) -> None:
        r.update(start_s=None, finish_s=None, blocked_reason=why, eta=eta, **extra)

    # 1. Queues no live node can run at all: blocked / yielded before anything is simulated.
    state: dict[str, _QState] = {}
    for r in order:
        pin = r.get("node")
        pin = None if pin in (None, "-") else str(pin)
        gnodes = [pin] if pin in gpu_cap else ([] if pin else sorted(gpu_cap))
        cnodes = [pin] if pin in cpu_cap else ([] if pin else sorted(cpu_cap))
        pend = r.get("sched_pending") or []
        need_gpu = any(s > 0 for _, s in pend) or r["running"] > 0
        need_cpu = any(s == 0 for _, s in pend)
        if (need_gpu and not gnodes) or (need_cpu and not cnodes) or (
                not gnodes and not cnodes):
            where = pin or "any node"
            w = waiting.get(pin, 0) if pin else sum(waiting.values())
            if pin is not None and pin in yielded:
                y = yielded[pin]
                block(r, f"no capacity on {pin}: {y['gpus']} GPUs yielded to a foreign user"
                      + (f" ({w} claims waiting for a GPU)" if w else ""), "yielded",
                      yielded_gpus=y["gpus"], last_reclaim_utc=y.get("last_reclaim_utc"))
            else:
                block(r, f"no live pool capacity on {where}" + (
                    f" ({w} claims waiting for a GPU)" if w else ""))
            continue
        state[r["queue"]] = _QState(r, gnodes, cnodes, list(r.get("depends_on") or []))

    # 2. Dependency blocking, to a fixpoint (a dep may sit anywhere in claim order).
    changed = True
    while changed:
        changed = False
        for r in order:
            st = state.get(r["queue"])
            if st is None:
                continue
            for dep in st.deps:
                d = by_name.get(dep)
                if d is not None and dep not in state:
                    extra = {k: d[k] for k in ("yielded_gpus", "last_reclaim_utc") if k in d}
                    block(r, f"dep {dep} {d['eta']}", d["eta"], **extra)
                elif d is None and inactive_deps.get(dep) != "complete":
                    block(r, f"dep {dep} {inactive_deps.get(dep, 'missing')}")
                else:
                    continue
                del state[r["queue"]]
                changed = True
                break

    # 3. A dependency cycle among active queues can never be scheduled.
    cyclic = set(state)
    settled: set[str] = set()
    while True:
        free = {n for n in cyclic - settled
                if all(d not in cyclic or d in settled for d in state[n].deps)}
        if not free:
            break
        settled |= free
    for n in sorted(cyclic - settled):
        block(state.pop(n).row, "dependency cycle")

    # 4. Discrete-event simulation: at every event time each lane with free slots claims the
    #    highest-priority ready queue that may run on it, exactly as a pool thread does.
    lanes = {("gpu", n): _Lane(c) for n, c in gpu_cap.items()}
    lanes.update({("cpu", n): _Lane(c) for n, c in cpu_cap.items()})
    for r in rows:  # jobs on GPU now hold their slots until they finish
        for left, slots, node in r.get("sched_running") or []:
            lane = lanes.get(("cpu" if slots == 0 else "gpu", str(node)))
            if lane is not None:
                lane.start(0.0, left, slots)
    names = [r["queue"] for r in order if r["queue"] in state]

    def ready_at(st: _QState) -> float | None:
        """When the queue may start: max finish of its active deps; ``None`` = not yet known."""
        t_ready = 0.0
        for dep in st.deps:
            o = state.get(dep)
            if o is None:
                continue  # complete / drained dependency
            if not o.resolved:
                return None
            t_ready = max(t_ready, o.finish_s)
        return t_ready

    def claim(kind: str, node: str, lane: _Lane, t: float,
              ready: dict[str, float | None]) -> bool:
        for n in names:
            st = state[n]
            jobs = st.cpu_jobs if kind == "cpu" else st.gpu_jobs
            rs = ready[n]
            if not jobs or rs is None or rs > t:
                continue
            if node not in (st.cnodes if kind == "cpu" else st.gnodes):
                continue
            for i, (exp, slots) in enumerate(jobs):
                if min(max(slots, 1), lane.cap) <= lane.free:
                    jobs.pop(i)
                    fin = lane.start(t, exp, slots)
                    st.start_s = t if st.start_s is None else st.start_s
                    st.finish_s = max(st.finish_s, fin)
                    return True
        return False

    t = 0.0
    while True:
        started = True
        while started:
            started = False
            ready = {n: ready_at(state[n]) for n in names}
            for (kind, node), lane in lanes.items():
                while lane.free > 0 and claim(kind, node, lane, t, ready):
                    started = True
        if all(state[n].resolved for n in names):
            break
        nxt = min((lane.heap[0][0] for lane in lanes.values() if lane.heap), default=None)
        if nxt is None or nxt <= t:
            break  # nothing in flight and nothing startable
        t = nxt
        for lane in lanes.values():
            lane.release(t)

    for n in names:
        st = state[n]
        r = st.row
        if not st.resolved:
            block(r, "unschedulable")
            continue
        outstanding = r["pending"] + r["running"] + r.get("waiting", 0)
        off = _rate_disagrees(r, st.finish_s)
        r.update(start_s=st.start_s if st.start_s is not None else 0.0,
                 finish_s=st.finish_s, blocked_reason=None, eta_disagree=off,
                 eta="done" if outstanding == 0 else
                 fmt_dur(st.finish_s) + ("?" if r.get("expected_unknown") else "")
                 + ("!" if off else ""))


def _complete_row(name: str) -> dict:
    """Minimal row for a queue with a valid ``complete.stamp``: it is inactive by definition,
    so only its name (as a satisfied dependency) is ever read. Skipping its job + claim scan
    is what keeps a warm status call on a ~2,200-queue root a few seconds."""
    return {"queue": name, "prio": 0, "created_utc": None, "node": "-", "parked": False,
            "complete": True, "depends_on": [], "pending": 0, "running": 0, "waiting": 0,
            "done": 0, "failed": 0, "total": 0}


def _active(row: dict) -> bool:
    work = row["pending"] + row["running"] + row.get("waiting", 0)
    return not row["parked"] and not row["complete"] and work > 0


def render(rows: list[dict]) -> str:
    cols = [
        ("queue", "queue", "<"), ("prio", "prio", ">"), ("node", "node", "<"),
        ("mem", "mem", ">"),
        ("pend", "pending", ">"), ("run", "running", ">"), ("wait", "waiting", ">"),
        ("done", "done", ">"), ("fail", "failed", ">"), ("t/job", "expected_s", ">"),
        ("r/h", "rate_per_h", ">"), ("eta", "eta", ">"), ("stale", "stale_running", ">"),
    ]
    table: list[list[str]] = []
    for r in rows:
        cells = []
        for _, key, _ in cols:
            v = r.get(key, 0)
            if key == "expected_s":
                v = fmt_expected(r)
            elif key == "rate_per_h":
                v = "-" if v is None else f"{v:.1f}"
            elif key == "stale_running":
                # `?` = at least one of them was judged by the unknown/borrowed rule.
                v = "" if not v else f"{v}{'?' if r.get('stale_by_unknown_rule') else ''}"
            cells.append(str(v))
        table.append(cells)
    widths = [max(len(h), *(len(t[i]) for t in table)) if table else len(h)
              for i, (h, _, _) in enumerate(cols)]
    def line(cells: list[str]) -> str:
        return "  ".join(
            f"{c:{a}{w}}" for c, w, (_, _, a) in zip(cells, widths, cols)
        ).rstrip()
    out = [line([h for h, _, _ in cols])]
    out.extend(line(t) for t in table)
    return "\n".join(out)


SECTIONS = {
    "train": ("Training students (receiver x epochs x rank x precision)", "job"),
    "teacher": ("Teacher fine-tunes (per cell)", "cell"),
    "numgen": ("Number generation (per 1k prompts)", "1k prompts"),
    "eval": ("Evals and probes (per run)", "run"),
    "traj": ("Checkpoint trajectories (per checkpoint)", "ckpt"),
    "other": ("Other scripts (per job)", "job"),
}


def _cells(s: eta_priors.Stat) -> dict[str, str]:
    """Every column of one heuristics row, as strings (``-`` where nothing was measured)."""
    def d(v: float | None) -> str:
        return "-" if v is None else fmt_dur(v)
    m = eta_priors._TRAIN_KEY.match(s.key)
    u = eta_priors._UNIT_KEY.match(s.key)
    mem = (f"{s.peak_mem_p50 / 1024:.1f}/{s.peak_mem_p90 / 1024:.1f}G({s.n_peak})"
           if s.peak_mem_p50 else "-")
    return {
        "signature": s.key,
        "sig": eta_priors.key_label(s.key) if (m or u) else s.key,
        "receiver": (m or u).group("arch") if (m or u) else "-",
        "kind": ((u or m).group("kind") if (u or m) and (u or m).group("kind") else "-"),
        "epochs": (m.group("ep") if m else "-"),
        "rank": ((m or u).group("rank") or "-") if (m or u) else "-",
        "prec": ((m or u).group("prec") or "-") if (m or u) else "-",
        "tier": s.tier, "n": str(s.n),
        "p10": d(s.p10), "p50": d(s.p50), "p90": d(s.p90),
        "per unit": d(s.per_run_p50),
        **{f"c{b}": d(s.bucket_p50.get(b)) for b in (1, 2, 3, 4, 5, 6)},
        "mps-": d(s.mps_off_p50), "mps+": d(s.mps_on_p50),
        "mem p50/p90 (n)": mem,
    }


PRIOR_TABLE_WIDTH = 130
_PRIOR_COLS = ("sig", "tier", "n", "p10", "p50", "p90", "per unit", "c1", "c2", "c3", "c4",
               "c5", "c6", "mps-", "mps+", "mem p50/p90 (n)")


def _sections(stats: list[eta_priors.Stat]) -> list[tuple[str, list[eta_priors.Stat]]]:
    """``stats`` grouped into the table's sections, in `SECTIONS` order (empties dropped)."""
    out = []
    for role, (title, _unit) in SECTIONS.items():
        rows = [s for s in stats if eta_priors.key_role(s.key) == role]
        if rows:
            out.append((title, rows))
    return out


def render_priors(priors: eta_priors.Priors, *, min_n: int = 0) -> str:
    """The learned heuristics table, one section per job role (`SECTIONS`).

    ``tier`` = which sample tier the quote came from (t1 today's nodes under MPS, t2 any node
    under MPS, t3 all history), ``per unit`` = p50 per cell / 1k prompts / run / checkpoint
    (the section says which), ``c1..c6`` = p50 at that co-tenancy bucket (jobs sharing the
    GPU), ``mps-``/``mps+`` = p50 with MPS off / on, ``mem`` = measured peak GPU MiB p50/p90
    (n); ``-`` = fewer than five samples in that cell. Columns nothing was measured for are
    dropped and the signature is clipped so rows fit 130 columns.
    """
    stats = priors.table(min_n=min_n)
    cells = {s.key: _cells(s) for s in stats}
    cols = [c for c in _PRIOR_COLS if c == "sig" or any(x[c] != "-" for x in cells.values())]
    rows = [[cells[s.key][c] for c in cols] for s in stats]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
         for i, h in enumerate(cols)]
    w[0] = max(min(w[0], PRIOR_TABLE_WIDTH - (sum(w[1:]) + 2 * len(w))), 20)
    def line(c: list[str]) -> str:
        c = [c[0][:w[0] - 1] + "\u2026" if len(c[0]) > w[0] else c[0], *c[1:]]
        return "  ".join(f"{x:<{w[0]}}" if i == 0 else f"{x:>{w[i]}}"
                         for i, x in enumerate(c)).rstrip()
    out = [line(list(cols))]
    for title, group in _sections(stats):
        out.append(f"-- {title}")
        out.extend(line([cells[s.key][c] for c in cols]) for s in group)
    return "\n".join(out)


MODEL_INTRO = (
    "A job's duration is REASONED, not looked up row by row. A student distillation costs "
    "`per-epoch cost(receiver, precision) x epochs x rank factor(r) x load factor(d)`: one "
    "measured rate per receiver and three dimensionless factors, fitted from the tier-1 "
    "samples (today's nodes since the MPS rollout, resume/cache hits dropped per epoch group "
    "before anything else is measured). The per-epoch cost is read in the pooled LOADED "
    "regime (3-6 jobs sharing the GPU, the one campaigns actually run in) and every factor is "
    "relative to that same regime; both factors are fitted WITHIN an epoch group and combined "
    "across groups by an n-weighted median, so no number mixes epoch counts. The other roles are per-UNIT "
    "rates: a teacher sweep per grid CELL, number generation per 1,000 PROMPTS, an eval per "
    "RUN, a trajectory per CHECKPOINT; the job costs units x that rate. What a duration does "
    "NOT depend on: the WORKER or QUEUE that runs it (`pythia-160m` at ep20 takes ~48 m "
    "whether it came from `dog_crossdistill_studenthp_worker` or `dd1b_worker`; the "
    "skip-gated `sweep_ckpt_distill` cells are their own `ckptdistill` kind, and a training "
    "result under two minutes is a cache hit and enters no training key; a campaign wrapper "
    "`run_logs/campaigns/<campaign>/pilot.py student` is keyed by the student YAML, base and cell its "
    "own text hard-codes, so its precision follows that YAML's `train.tf32`), "
    "and the LoRA rank below 64 (flat to within a few percent; "
    "only r128/r256 cost measurably more). Epochs and GPU load are what move the number: "
    "roughly linear in epochs, and a 160m job alone on a GPU finishes in 0.37x its 5-up time "
    "(1-up 0.37, 2-up 0.44, 3-up 0.80, 4-up 0.82, 5-up 1.00, 6-up 1.15). Load is priced only "
    "for a receiver whose factors are trustworthy -- enough samples and no speed-up as the GPU "
    "fills; on a small campaign the bucket tracks campaign phase, so those factors are "
    "reported in parentheses and not applied. "
    "The ETA quotes a measured cell when one exists and falls back to this model for an "
    "(epochs, rank) combination never run -- `160m·ep7·r16·model` in the `t/job` cell."
)


def _md_table(cols: list[str], rows: list[list[str]]) -> str:
    head = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    return "\n".join(head + ["| " + " | ".join(r) + " |" for r in rows])


def _txt_table(cols: list[str], rows: list[list[str]]) -> str:
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
         for i, c in enumerate(cols)]
    def line(c: list[str]) -> str:
        return "  ".join(f"{x:<{w[0]}}" if i == 0 else f"{x:>{w[i]}}"
                         for i, x in enumerate(c)).rstrip()
    return "\n".join([line(cols)] + [line(r) for r in rows])


_MODEL_RANKS = (64, 128, 256)
_MODEL_LOADS = (1, 2, 3, 4, 5, 6)
_MODEL_COLS = ["receiver", "prec", "min/epoch loaded", "n", "epochs seen",
               *(f"x r{r}" for r in _MODEL_RANKS),
               *(f"x {d}-up" for d in _MODEL_LOADS), "resid p10/p90"]


def _model_rows(model: eta_priors.DurationModel) -> list[list[str]]:
    """One row per (receiver, precision): the rate and the factors that multiply it.

    A load factor that is measured but NOT applied (`ArchModel.load_applied`) is shown in
    parentheses -- it is evidence, not a multiplier.
    """
    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:.2f}"
    out = []
    for m in sorted(model.students.values(), key=lambda x: -x.n):
        def load(d: int, m: eta_priors.ArchModel = m) -> str:
            v = m.load_factor.get(d)
            return "-" if v is None else (f"{v:.2f}" if m.load_applied else f"({v:.2f})")
        arch = m.arch if not m.kind else f"{m.arch}\u00b7{m.kind}"  # kinds are fitted apart
        out.append([arch, m.prec, f"{m.per_epoch_s / 60:.2f}", str(m.n),
                    ",".join(str(e) for e in m.epochs),
                    *(f(m.rank_factor.get(r)) for r in _MODEL_RANKS),
                    *(load(d) for d in _MODEL_LOADS),
                    f"{m.res_p10:.2f}/{m.res_p90:.2f}"])
    return out


_RATE_SECTIONS = (
    ("teacher", "Teacher fine-tunes (per grid cell)", "cell"),
    ("numgen", "Number generation (per 1,000 prompts)", "1k prompts"),
    ("eval", "Evals and probes (per run)", "run"),
    ("traj", "Checkpoint trajectories (per checkpoint)", "checkpoint"),
)


def fmt_rate(s: float) -> str:
    """A per-unit cost, in seconds up to ten minutes (``fmt_dur`` is too coarse for rates)."""
    return f"{s:.0f}s" if s < 600 else fmt_dur(s)


def _rate_rows(model: eta_priors.DurationModel, role: str,
               min_n: int = eta_priors.MIN_CELL_N) -> list[list[str]]:
    """Per-unit rates of one role, best-sampled first (cells below ``min_n`` are noise)."""
    rows = [r for k, r in model.rates.items()
            if eta_priors.key_role(k) == role and r.n >= min_n]
    return [[eta_priors.key_label(r.key), str(r.n), fmt_rate(r.p10), fmt_rate(r.p50),
             fmt_rate(r.p90)] for r in sorted(rows, key=lambda r: -r.n)]


_RATE_COLS = ["signature", "n", "p10", "p50", "p90"]


def _worked_examples(model: eta_priors.DurationModel) -> list[str]:
    """The model applied to five everyday jobs, recomputed from today's fit."""
    out = []
    for arch, prec, ep, rank, load, what in (
        ("pythia-160m", "fp32", 20, 16, 5, "160m ep20 r16 at 5-up"),
        ("pythia-410m", "fp32", 20, None, 5, "410m ep20 at 5-up"),
        ("qwen", "tf32", 5, 8, 5, "qwen ep5 tf32 r8 at 5-up"),
    ):
        m = model.for_arch(arch, prec)
        if m is None:
            continue
        note = "" if m.load_applied else " [load factors measured but not applied]"
        out.append(
            f"- **{what}** = {m.per_epoch_s / 60:.2f} min/epoch x {ep} x {m.rank_mult(rank):.2f}"
            f" (r{rank if rank else '-'}) x {m.load_mult(load):.2f} ({load}-up) = "
            f"**{fmt_dur(m.predict(ep, rank, load))}**{note}")
    ng = model.rates.get("numgen|model=pythia-160m")
    if ng:
        out.append(f"- **a 100k-prompt number pool off one 160m teacher** = 100 x "
                   f"{fmt_rate(ng.p50)} per 1k prompts = **{fmt_dur(ng.p50 * 100)}**")
    pr = model.rates.get("eval|probe|model=pythia-160m")
    if pr:
        out.append(f"- **a probe eval over 100 student runs on 160m** = 100 x "
                   f"{fmt_rate(pr.p50)} per run = **{fmt_dur(pr.p50 * 100)}**")
    return out


def render_model(model: eta_priors.DurationModel, *, markdown: bool = False) -> str:
    """Sections (a)-(d) of the heuristics sheet: the model, its factors, the rates, examples."""
    tab = _md_table if markdown else _txt_table
    h = (lambda t: f"## {t}") if markdown else (lambda t: f"-- {t}")
    parts = [MODEL_INTRO if markdown else textwrap.fill(MODEL_INTRO.replace("`", ""), 100),
             h("Student cost model (per receiver x precision x training kind)"),
             tab(_MODEL_COLS, _model_rows(model))]
    if markdown:
        parts.append(
            "`min/epoch loaded` x epochs x the rank factor x the load factor = the job. The "
            "rate is the median of `seconds / epochs` over the pooled 3-6-up samples of each "
            "epoch group, combined across groups by an n-weighted median; `n` = tier-1 "
            "samples behind the fit, `epochs seen` the epoch counts measured (others are "
            "extrapolated linearly), `x rN` the rank factor relative to ranks <= 32, `x d-up` "
            "the factor at d jobs sharing the GPU (both measured WITHIN each epoch group, "
            "blank = fewer than 10 samples, which counts as 1.0), `resid p10/p90` the "
            "measured/predicted band of the model against its own samples. A load factor in "
            "PARENTHESES is measured but NOT applied (every load factor of that receiver is "
            f"1.0 and the ETA ignores the node's density): fewer than "
            f"{eta_priors.MODEL_LOAD_MIN_N} tier-1 samples, or factors that fall as the GPU "
            "fills. In a 15-200-job campaign the co-tenancy bucket is a proxy for CAMPAIGN "
            "PHASE rather than GPU contention -- inside the OLMo queues later-started jobs "
            "are both faster and more co-tenant (corr(duration, start order) -0.3..-0.87, "
            "corr(bucket, start order) up to +0.83, pooled corr(duration, bucket) -0.30) -- "
            "so those buckets would otherwise be read as \"5-up is faster than 3-up\".")
    for role, title, unit in _RATE_SECTIONS:
        rows = _rate_rows(model, role)
        if rows:
            parts += [h(f"{title}"), tab(_RATE_COLS, rows)]
            if markdown:
                parts.append(f"Seconds per {unit}; a job costs its unit count x the p50. "
                             f"Cells with fewer than {eta_priors.MIN_CELL_N} samples are "
                             "omitted.")
    ex = _worked_examples(model)
    if ex:
        parts += [h("Worked examples"), "\n".join(ex if markdown else
                                                  [e.replace("**", "") for e in ex])]
    return "\n\n".join(parts)


_MD_COLS = ("receiver", "kind", "epochs", "rank", "prec", "tier", "n", "p10", "p50", "p90",
            "per unit", "c1", "c2", "c3", "c4", "c5", "c6", "mps-", "mps+",
            "mem p50/p90 (n)", "signature")


def render_priors_markdown(priors: eta_priors.Priors, cinfo: eta_priors.CacheInfo, *,
                           min_n: int = 0) -> str:
    """`docs/QUEUE_ETA_HEURISTICS.md`: the reasoned sheet (model, factors, rates, examples)
    followed by the raw measured-cell table as an appendix."""
    stats = priors.table(min_n=min_n)
    head = (
        "# Queue ETA heuristics (the fitted duration model)\n\n"
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')} by "
        "`uv run python -m scripts.queue_status --show-priors --markdown` from "
        f"{cinfo.n_samples} successful jobq results over {', '.join(cinfo.roots)}.\n"
    )
    parts = [head, render_model(priors.model, markdown=True),
             "## Appendix: the raw measured cells\n\n"
             "Every signature the ETA can quote DIRECTLY, before the model is needed. A job is "
             "keyed by what its SCRIPTS do: a student training by receiver arch + epochs alone "
             "(the script does not matter), refined by LoRA rank and precision where enough "
             "samples exist; a teacher sweep by arch + epochs per CELL; number generation by "
             "teacher arch per 1,000 PROMPTS; evals by eval set + probed arch per RUN; "
             "trajectories by script + arch per CHECKPOINT; everything else by its script "
             "chain (+ receiver). Carrying `--epochs` does not make a job a training job. "
             "`per unit` is the p50 of ONE unit of the section's kind (the whole job = units x "
             "that). Each row quotes the first sample tier holding enough samples -- `t1` = "
             "today's nodes since the MPS rollout, `t2` = any node since then, `t3` = all "
             "history -- newest samples first. `cN` = p50 at N jobs sharing the GPU, this one "
             "included; `mps-`/`mps+` = p50 before/after the 2026-08-12 MPS rollout (daemon "
             "off/on); `mem` = measured peak GPU memory p50/p90 (n reporting jobs; the torch "
             "reserved high-water mark the worker reads back from the job log); `-` means "
             "fewer than five samples. Historical `mem_mib` asks and OOM floors are not "
             "trustworthy, so they are not tabulated. GPU load this repo never recorded "
             "(direct runs, foreign users) is invisible here, so co-tenancy is a lower bound."]
    for title, group in _sections(stats):
        lines = ["| " + " | ".join(_MD_COLS) + " |",
                 "|" + "|".join(["---"] * len(_MD_COLS)) + "|"]
        for s in group:
            c = _cells(s)
            lines.append("| " + " | ".join(c[k].replace("|", "\\|") for k in _MD_COLS) + " |")
        parts.append(f"### {title}\n\n" + "\n".join(lines) + "\n")
    return "\n\n".join(parts)


def _node_cell(cap: NodeCapacity, y: dict | None) -> str:
    s = f"{cap.node}:{cap.used}/{cap.slots}"
    if cap.capacity_source != "policy":  # default cap_per_gpu, or occupancy as a stand-in
        s += f" ({cap.capacity_source} est)"
    if not cap.live:
        s += " (no live pool)"
    if y:
        t = _parse_iso(y.get("last_reclaim_utc"))
        rec = f", last reclaim {t.strftime('%H:%MZ')}" if t else ", no reclaim in log tail"
        s += f" ({y['gpus']} yielded{rec})"
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="jobq root (default: paths YAML queue_root)")
    ap.add_argument("--all", action="store_true", help="include parked / complete / drained queues")
    ap.add_argument("--json", action="store_true", help="emit rows as JSON instead of a table")
    ap.add_argument("--history-root", type=Path, action="append", default=[],
                    help="extra jobq root to measure duration priors from (repeatable)")
    ap.add_argument("--rebuild-priors", action="store_true",
                    help="rescan every history root instead of using the cached samples")
    ap.add_argument("--show-priors", action="store_true",
                    help="print the learned duration heuristics instead of the queue table")
    ap.add_argument("--min-n", type=int, default=0, help="--show-priors: hide signatures below N samples")
    ap.add_argument("--markdown", action="store_true",
                    help=f"--show-priors: write the whole document to {HEURISTICS_DOC} instead")
    ap.add_argument("--raw", action="store_true",
                    help="--show-priors: also print the raw measured-cell table")
    ap.add_argument("--prior-recent-n", type=int, default=eta_priors.DEFAULT_RECENT_N,
                    help="newest samples a signature quotes from within its tier")
    ap.add_argument("--prior-min-n", type=int, default=eta_priors.DEFAULT_MIN_N,
                    help="samples a signature level needs before it is usable")
    ap.add_argument("--unknown-job-s", type=float, default=600.0,
                    help="assumed duration of a job whose signature matched nothing")
    ap.add_argument("--pool-live-min", type=float, default=15.0,
                    help="a node counts as live if its pool log was touched within N minutes")
    ap.add_argument("--window-h", type=float, default=3.0, help="trailing window for r/h")
    ap.add_argument("--stale-x", type=float, default=3.0,
                    help="claim age > stale_x * the job's expected duration => stale")
    ap.add_argument("--stale-unknown-s", type=float, default=STALE_UNKNOWN_S,
                    help="stale age for a job whose duration this queue never measured "
                         "(unknown signature or a borrowed prior)")
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else _default_root()
    if not root.is_dir():
        print(f"no such root: {root}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    samples, cinfo = eta_priors.load_samples(
        root, extra_roots=list(args.history_root), rebuild=args.rebuild_priors)
    priors = build_priors(samples, min_n=args.prior_min_n, recent_n=args.prior_recent_n)
    if args.show_priors:
        if args.markdown:
            HEURISTICS_DOC.parent.mkdir(parents=True, exist_ok=True)
            HEURISTICS_DOC.write_text(render_priors_markdown(priors, cinfo, min_n=args.min_n))
            print(f"wrote {HEURISTICS_DOC} ({len(priors.table(min_n=args.min_n))} signatures)")
            return 0
        print(render_model(priors.model))
        if args.raw:
            print()
            print(render_priors(priors, min_n=args.min_n))
        return 0
    own: dict[str, list[eta_priors.Sample]] = {}
    prefix = f"{root}/"
    for s in samples:
        if s.queue.startswith(prefix):
            own.setdefault(s.queue[len(prefix):], []).append(s)
    densities = node_density(root)
    feats_cache = eta_priors.FeatsCache.load(root)
    rows = [
        _complete_row(name) if not args.all and store.queue_complete(root, name) else
        queue_row(
            root, name, now=now, priors=priors, own_samples=own.get(name),
            unknown_job_s=args.unknown_job_s, window_h=args.window_h, stale_x=args.stale_x,
            stale_unknown_s=args.stale_unknown_s,
            densities=densities, feats_cache=feats_cache,
        )
        for name in store.list_queues(root)
    ]
    feats_cache.save()
    inactive = {r["queue"]: ("complete" if r["complete"] else "parked" if r["parked"] else "drained")
                for r in rows if not _active(r)}
    shown = rows if args.all else [r for r in rows if _active(r)]
    active = [r for r in rows if _active(r)]
    # "drained" (no outstanding jobs, no stamp yet) is satisfied for scheduling purposes
    inactive = {k: ("complete" if v == "drained" else v) for k, v in inactive.items()}
    yielded = read_yielded(root)
    caps = node_capacity(root, on_gpu=node_concurrency(active), yielded=yielded,
                         pool_live_min=args.pool_live_min)
    annotate_queue_order(active, capacity=caps, inactive_deps=inactive, yielded=yielded)
    rows = sorted(shown, key=lambda r: (-r["prio"], r.get("created_utc") or "", r["queue"]))
    work = node_work(active, caps)
    if args.json:
        print(json.dumps({"queues": rows, "work": work, "yielded": yielded,
                          "capacity": {n: vars(c) for n, c in caps.items()}},
                         indent=1, default=str))
        return 0
    print(f"root={root}  now={now.strftime('%Y-%m-%dT%H:%MZ')}  "
          f"active_queues={len(active)}  window={args.window_h:g}h  "
          f"capacity={sum(c.used for c in caps.values())}/{sum(c.slots for c in caps.values())} "
          "slots (" + " ".join(_node_cell(c, yielded.get(n))
                               for n, c in sorted(caps.items())) + ")")
    print(render(rows) if rows else "(no active queues)")
    pend = sum(r["pending"] for r in rows)
    run = sum(r["running"] for r in rows)
    wait = sum(r.get("waiting", 0) for r in rows)
    print(f"totals: pending={pend} running(on-GPU)={run} waiting(claim, no GPU)={wait}")
    print(fmt_work(work))
    age = f"{cinfo.age_s / 3600:.1f}h" if cinfo.age_s is not None else "?"
    stale = cinfo.age_s is not None and cinfo.age_s > 2 * 86400
    print(f"priors: {len(priors.stats)} signatures from {cinfo.n_samples} samples over "
          f"{len(cinfo.roots)} roots, cache age {age}, {cinfo.rescanned} queues rescanned"
          + (" [STALE: rerun with --rebuild-priors]" if stale else "")
          + (f" [{cinfo.note}]" if cinfo.note else ""))
    print(textwrap.fill(
          "run = claims holding a GPU; wait = claims parked without one. t/job = expected "
          "time of one job and the signature it came from: what the job's SCRIPTS do picks "
          "the key -- a student TRAINING by receiver arch + epochs pooled over every script "
          "(~47m(160m·ep20)), refined by LoRA rank / precision when measured "
          "(~51m(160m·ep20·r128)); a teacher sweep per cell (tft·qwen·ep10), numgen per 1k "
          "prompts (ngen·qwen), evals per run (eval·probe·160m), trajectories per checkpoint "
          "(traj·…), each x the units of that job; anything else keeps its script chain + "
          "receiver. N(n=K) = measured from this "
          "queue alone (K own samples), N(n=K+160m·ep20) = blended with that prior, "
          "~N(160m·ep7·r16·model) = the fitted cost model (per-epoch x epochs x rank x GPU "
          "load) for a shape never measured, "
          "~N(160m·ep5←ep20) = scaled from another epoch count, ~N(p) a script-chain "
          "prior, ~N(p,sum) an unmeasured chain summed from its components, ~N(?) nothing "
          "matched (--unknown-job-s assumed; that row's eta gets a ?). eta = one best estimate of time until the queue drains, from "
          "an event simulation of the pools at each node's CAPACITY (policy gpus minus "
          "yielded, x cap_per_gpu; slots:0 jobs use the cpu_cap lane): active queues claim "
          "free slots in priority order (prio desc, created) as they free up, pinned queues "
          "only on their node, unpinned on every live node; running jobs hold their slots "
          "until they finish; a dependent starts when its dependency finishes. ! = trailing "
          "completion rate disagrees >50% with the simulation (resumes, yielded GPUs, or a "
          "half-idle node); only for queues whose rate has settled (>=10 completions and a "
          "first completion older than the window plus one job). yielded = its "
          "pin node's GPUs went to a "
          "foreign user (--json yielded_gpus, last_reclaim_utc); blocked = no live pool "
          "capacity for it, or dep blocked (--json blocked_reason). work = per pin node the "
          "GPU-h of queued expected*slots + remaining running time, wall at that node's "
          "capacity (@N). mem = GiB ask of pending+running jobs (range when they differ; "
          "^ = an OOM floor raised a job above its static ask; - = no ask/pool default). "
          "stale = on-GPU claims older than --stale-x x their expected duration, or older "
          "than the flat --stale-unknown-s when the expectation was not measured from this "
          "queue's own samples (N? = at least one was judged by that unknown/borrowed rule). "
          "A node is live if it holds claims now or its pool log is fresh.", width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
