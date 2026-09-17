"""Read-only campaign status for a jobq root: one row per ACTIVE queue with an ETA.

Backs the ``/status`` command. Prints a table of queues that still have pending or
running work (parked queues -- ``meta.node`` starting with ``PARKED`` -- and queues with
a ``complete.stamp`` are hidden unless ``--all``), with:

- ``prio``/``node``: from ``meta.json`` (the pin is the ``node`` field; ``-`` = unpinned).
- ``pend/run/wait/done/fail``: derived from claims + results exactly as ``jobq status`` does,
  except that claims are split by their owner's ``gpu``: ``run`` = claims holding a GPU,
  ``wait`` = claims with ``gpu: null`` (the worker is parked waiting for a GPU, e.g. yielded
  to a foreign user). Only ``run`` counts as concurrency anywhere below.
- ``median``: median wall-clock of this queue's SUCCESSFUL results (``rc == 0``; killed /
  failed attempts never count). Cache hits are dropped only as outliers relative to the
  queue's own distribution: a duration is a cache hit iff it is shorter than ``--min-job-s``
  (a ceiling, default 120 s) AND shorter than ``--cache-hit-frac`` (0.1) x the queue's 90th
  percentile -- so a queue whose jobs are all ~90 s keeps them. With fewer than
  ``--min-samples`` own durations the median is borrowed from same-FAMILY queues (name equal
  after replacing ``orig``/``fs<N>``/``ds<N>``/``ws<N>``/``seed<N>``/``step<N>`` tokens,
  ``-``-separated, with ``*``; durations pooled across siblings, including completed ones)
  and shown as ``~56m(fam)``. Failing that, from every result in the root with the same JOB
  KIND (``job_kind``: ``&&``-ordered script/module sequence with subcommand, repeats as ``X*N``, the receiver
  architecture from ``--student-base``/``--base``/``--base-models`` normalised via
  ``models.naming`` -- pythia-160m / pythia-410m / olmo-1b / qwen, seed variants ignored --
  and ``--epochs``; ``--runs`` probe lists key on the probed model's arch read from the list
  and borrow a per-run duration scaled by this queue's mean run count; job pools take the
  median of per-queue medians so no single campaign dominates), shown ``~1h(job)``.
  Each pool is re-filtered for cache hits, and a pooled median below ``--min-borrow-s``
  (30 s) is skipped in favour of the next fallback. Nothing -> ``?``.
- ``r/h``: completions per hour over the trailing ``--window-h`` hours (this queue).
- ``eta``: ONE best-estimate time until this queue drains, from a priority-order schedule.
  Active queues are processed in worker claim order ``(-priority, created_utc)``; each node
  ``n`` has concurrency ``c_n`` = its on-GPU claims (across all active queues) and a clock.
  A queue's schedulable work is ``(pending + waiting) * median``; its on-GPU claims
  finish in ``max(median - age, 0)`` regardless of order (eta = the later of the two).
  A queue pinned to ``X`` starts at ``max(clock_X, dep finish)`` and runs
  at ``c_X``; an unpinned queue water-fills every node with ``c_n > 0`` (finish ``T`` solves
  ``sum c_n*max(0, T-max(clock_n, dep)) = work``). Finishing advances the clocks it used, so
  equal-priority queues pinned to different nodes drain concurrently. A queue whose
  ``depends_on`` is active starts after that dependency's finish; if the dependency comes
  later in claim order the dependent is deferred (reserves no clock time, so it never
  pulls its dependency forward or pushes lower queues back) and is placed right after its
  last dependency, which is when it outranks the remaining queues for GPUs. A trailing ``+`` marks a
  lower bound (an unknown-median queue ahead on a node it uses, or its dependency is one);
  ``?`` = no median of any kind (with on-GPU jobs: ``<oldest on-GPU age>+`` instead, a
  lower bound); ``yielded`` = its pin node has zero on-GPU jobs and its
  ``yielded.<host>.json`` lists GPUs handed to a foreign user (``--json``: ``blocked_reason``,
  ``yielded_gpus``, ``last_reclaim_utc`` = last ``RECLAIM`` line in the node's newest
  ``pool_*.<host>.log`` tail; a dependency on a yielded queue is ``yielded`` too);
  ``blocked`` = nothing else can run it (its pin node / every node has
  zero on-GPU jobs with no yielded GPUs, or a dependency is blocked / parked / missing) --
  ``--json`` gives ``blocked_reason`` (incl. waiting-claim evidence). Preemption and lower-priority jobs
  already holding GPUs are not modelled. ``--json`` also keeps ``start_s``/``finish_s``
  and the old per-queue ``eta_wave_s``/``eta_throughput_s``.
- ``work`` (line after ``totals:``): per pin node (plus ``unpinned``) the GPU-hours of
  queued work if it all ran now undisturbed: ``(pending + waiting) * median`` plus each
  on-GPU job's ``max(median - age, 0)``, summed over active queues with a known median
  (``+N queues ?`` counts those without). Wall-clock = GPU-h / the node's current on-GPU
  concurrency (unpinned: all nodes'), or / ``--assumed-conc`` (40) when that is 0, marked
  ``@40?``. The header shows yielded GPUs per node. ``--json`` emits
  ``{"queues": [...], "work": {...}, "yielded": {...}}``.
- ``stale``: on-GPU claims older than ``--stale-x`` x median (or 3 h when no median):
  dead pool thread / crashed node candidates (``jobq status`` shows the owner node).
- ``mem``: GiB each pending/running job would be gated on -- the worker's
  ``_effective_mem``: ``max(job.mem_mib or defaults.mem_mib, persisted OOM floor)``; a
  ``min-max`` range when jobs differ, ``^`` when an escalated floor lifts any job above its
  static ask, ``-`` when no job/default ask exists (the pool's node policy default then
  applies, unknowable from here) or when the queue has no pending/running jobs.

Usage::

    uv run python -m scripts.queue_status [--root run_logs/queue_angela] [--all] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import os
from collections import Counter
from functools import lru_cache
import statistics
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from subliminal_learning.config import resolve_queue_root
from subliminal_learning.jobqueue import store

PATHS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paths" / "paths_local.yaml"


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


def drop_cache_hits(
    vals: list[float], *, min_job_s: float, cache_hit_frac: float = 0.1
) -> list[bool]:
    """Keep-mask: ``d`` is a cache hit iff ``d < min_job_s`` and ``d < cache_hit_frac * p90``."""
    if not vals:
        return []
    srt = sorted(vals)
    p90 = srt[min(len(srt) - 1, int(0.9 * len(srt)))]
    return [not (d < min_job_s and d < cache_hit_frac * p90) for d in vals]


def keyed_durations_s(
    results: list[dict], *, min_job_s: float, cache_hit_frac: float = 0.1
) -> list[tuple[str, float]]:
    """(result key, seconds) of successful results with cache-hit OUTLIERS dropped."""
    raw: list[tuple[str, float]] = []
    for r in results:
        if r.get("rc", 1) != 0:
            continue
        a, b = _parse_iso(r.get("start_utc")), _parse_iso(r.get("end_utc"))
        if a is None or b is None:
            continue
        raw.append((str(r.get("key")), (b - a).total_seconds()))
    keep = drop_cache_hits([d for _, d in raw], min_job_s=min_job_s, cache_hit_frac=cache_hit_frac)
    return [kd for kd, k in zip(raw, keep) if k]


def durations_s(
    results: list[dict], *, min_job_s: float, cache_hit_frac: float = 0.1
) -> list[float]:
    """Successful-result durations (seconds) with cache-hit OUTLIERS dropped.

    A duration is a cache hit iff ``d < min_job_s`` and ``d < cache_hit_frac * p90`` of the
    queue's successful durations; consistently short queues keep all their samples.
    """
    return [d for _, d in keyed_durations_s(
        results, min_job_s=min_job_s, cache_hit_frac=cache_hit_frac)]


_RECEIVER_FLAGS = ("--student-base", "--base", "--base-models")


@lru_cache(maxsize=None)
def model_arch(token: str) -> str | None:
    """Base-model spelling -> architecture (``pythia-160m``/``pythia-410m``/``olmo-1b``/``qwen``)
    via ``models.naming.base_spec``, ignoring the seed variant; ``None`` if unparseable."""
    from subliminal_learning.models.datadecide import DataDecideSpec
    from subliminal_learning.models.naming import base_spec
    from subliminal_learning.models.registry import ExternalModelSpec, PythiaModelSpec

    try:
        spec = base_spec(token)
    except Exception:  # noqa: BLE001 - any unparseable spelling
        return None
    if isinstance(spec, PythiaModelSpec):
        return f"pythia-{spec.size}"
    if isinstance(spec, DataDecideSpec):
        return f"olmo-{spec.size.lower()}"
    if isinstance(spec, ExternalModelSpec):
        repo = spec.repo_id.split("/")[-1]
        return "qwen" if repo.lower().startswith("qwen") else repo
    return None


@lru_cache(maxsize=None)
def probe_list_info(path: str) -> tuple[str | None, int] | None:
    """(probed-model arch, run count) of a ``--runs`` list file; ``None`` if unreadable.

    The arch comes from the first path component (after ``students/``) that parses as a base.
    """
    try:
        lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    except OSError:
        return None
    arch = None
    for ln in lines[:1]:
        parts = ln.split("/")
        tail = parts[parts.index("students") + 1:] if "students" in parts else parts
        for comp in tail:
            if "-" in comp and not comp.startswith("teacher-"):
                arch = model_arch(comp)
                if arch:
                    break
    return arch, len(lines)


@lru_cache(maxsize=None)
def job_kind(cmd: str) -> tuple[str, int | None]:
    """(kind key, units) for a job ``cmd``; units = run count of a readable ``--runs`` list.

    Kind = ordered script/module sequence (``-m scripts.X`` or ``<dir>/X.py [subcommand]`` per
    ``&&`` segment) + receiver/probed architecture + ``--epochs``. Receiver = the first of
    ``--student-base``/``--base``/``--base-models`` (never ``--teacher-base``), normalised by
    ``model_arch`` (raw value when unknown); for ``--runs`` lists, the probed model's arch.
    """
    scripts: list[str] = []
    receiver = epochs = None
    units: int | None = None
    for seg in cmd.split("&&"):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if t == "-m" and nxt:
                sub = toks[i + 2] if i + 2 < len(toks) and not toks[i + 2].startswith("-") else ""
                scripts.append(f"{nxt} {sub}".strip())
            elif t.endswith(".py") and not t.startswith("-"):
                sub = f" {nxt}" if nxt and not nxt.startswith("-") else ""
                scripts.append(Path(t).stem + sub)
            elif t in _RECEIVER_FLAGS and nxt and receiver is None:
                raw = re.split(r"[,|]", nxt)[0]
                receiver = model_arch(raw) or raw
            elif t == "--epochs" and nxt and epochs is None:
                epochs = nxt
            elif t == "--runs" and nxt and receiver is None:
                info = probe_list_info(nxt)
                if info is not None:
                    receiver, units = info[0] or "?", info[1]
    runs: list[str] = []  # run-length encode repeats: `evaluate_student_preference*10`
    for i, x in enumerate(scripts):
        if i == 0 or x != scripts[i - 1]:
            n = 1
            while i + n < len(scripts) and scripts[i + n] == x:
                n += 1
            runs.append(x if n == 1 else f"{x}*{n}")
    key = ">".join(runs) or (cmd.split()[0] if cmd.split() else "")
    if receiver:
        key += f"|model={receiver}"
    if epochs:
        key += f"|ep={epochs}"
    return key, units


_FAMILY_TOKEN = re.compile(r"orig|(?:fs|ds|ws|seed|step)\d+")


def family_key(name: str) -> str:
    """Queue name with varying seed/pair/step tokens (whole ``-`` fields) replaced by ``*``."""
    return "-".join("*" if _FAMILY_TOKEN.fullmatch(t) else t for t in name.split("-"))


def borrow_family_medians(
    rows: list[dict], *, min_samples: int, min_job_s: float = 120.0,
    cache_hit_frac: float = 0.1, min_borrow_s: float = 30.0,
) -> None:
    """Fill ``median_s`` for rows with too few own samples: own > ``fam`` > ``job``.

    ``fam`` pools durations of same-``family_key`` queues; ``job`` pools every result of the
    same ``job_kind`` anywhere in the root (per-run duration x this queue's mean run count
    when both sides have readable ``--runs`` lists, else per-job). Each pool is re-filtered
    with the cache-hit rule and skipped if its median is below ``min_borrow_s`` (never quote
    ~0 when another fallback exists). Pops ``durations``/``job_samples``; sets
    ``median_source`` to ``own``/``fam``/``job``/``None``.
    """
    def pooled_median(vals: list[float]) -> float | None:
        keep = drop_cache_hits(vals, min_job_s=min_job_s, cache_hit_frac=cache_hit_frac)
        vals = [v for v, k in zip(vals, keep) if k]
        if len(vals) < min_samples:
            return None
        m = statistics.median(vals)
        return m if m >= min_borrow_s else None

    fams: dict[str, list[dict]] = {}
    # kind -> queue -> [(units, seconds)]; job pools weight QUEUES equally (median of
    # per-queue medians) so one big campaign cannot dominate a kind shared root-wide
    kinds: dict[str, dict[str, list[tuple[int | None, float]]]] = {}
    for r in rows:
        fams.setdefault(family_key(r["queue"]), []).append(r)
        for kind, units, d in r.get("job_samples") or []:
            kinds.setdefault(kind, {}).setdefault(r["queue"], []).append((units, d))

    def queue_equal_median(per_queue: list[list[float]]) -> float | None:
        meds = []
        for vals in per_queue:
            keep = drop_cache_hits(vals, min_job_s=min_job_s, cache_hit_frac=cache_hit_frac)
            vals = [v for v, k in zip(vals, keep) if k]
            if vals:
                meds.append(statistics.median(vals))
        if sum(len(v) for v in per_queue) < min_samples or not meds:
            return None
        return statistics.median(meds)
    for r in rows:
        if r["median_s"] is not None:
            r["median_source"] = "own"
            continue
        r["median_source"] = None
        fam = pooled_median([d for x in fams[family_key(r["queue"])] if x is not r
                             for d in x.get("durations", [])])
        if fam is not None:
            r.update(median_s=fam, median_source="fam")
            continue
        pool = kinds.get(r.get("job_kind") or "", {})
        units = r.get("job_units")
        med = None
        if units:
            per_run = queue_equal_median(
                [[d / u for u, d in v if u] for v in pool.values() if any(u for u, _ in v)])
            if per_run is not None and per_run * units >= min_borrow_s:
                med = per_run * units
        if med is None:
            m = queue_equal_median([[d for _, d in v] for v in pool.values()])
            med = m if m is not None and m >= min_borrow_s else None
        if med is not None:
            r.update(median_s=med, median_source="job")
    for r in rows:
        r.pop("durations", None)
        r.pop("job_samples", None)


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
    median: float | None,
    rate_per_h: float | None,
    running_ages_s: list[float] | None = None,
) -> tuple[float | None, float | None]:
    """(wave-model ETA, throughput-model ETA) in seconds; ``None`` where undefined.

    Wave: the current running set finishes in ~(median - age), then pending drains at the
    same concurrency. Throughput: outstanding work at the trailing completion rate.
    """
    wave = None
    if median is not None and running > 0:
        ages = running_ages_s or []
        remaining = max((median - a for a in ages), default=median)
        wave = max(remaining, 0.0) + median * pending / running
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


def fmt_eta(wave: float | None, thr: float | None, *, running: int, pending: int) -> str:
    if pending == 0 and running == 0:
        return "done"
    if wave is None and thr is None:
        return "blocked" if running == 0 else "?"
    vals = [v for v in (wave, thr) if v is not None]
    lo, hi = min(vals), max(vals)
    if len(vals) == 1 or hi <= lo * 1.15:
        return fmt_dur(lo)
    return f"{fmt_dur(lo)}-{fmt_dur(hi)}"


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


def _scan_jobs(root: Path, name: str, default_mem: object) -> tuple[dict, list[int | None], bool]:
    """One pass over the queue's jobs: ``store.queue_status``-shaped counts + active mem asks."""
    counts = {store.PENDING: 0, store.RUNNING: 0, store.DONE: 0, store.FAILED: 0}
    running: list[dict] = []
    asks: list[int | None] = []
    escalated = False
    cmds: dict[str, str] = {}
    active_cmds: list[str] = []
    for job in store.load_jobs(root, name):
        st = store.job_state(root, name, job)
        counts[st] += 1
        cmds[job.key] = job.cmd
        if st in (store.PENDING, store.RUNNING):
            ask, esc = effective_mem_mib(
                job.mem_mib, default_mem, store.read_mem_floor(root, name, job.jobkey))
            asks.append(ask)
            escalated |= esc
            active_cmds.append(job.cmd)
        if st == store.RUNNING:
            owner = store._read_owner(root, name, job.jobkey) or {}
            running.append({"key": job.key, "start_utc": owner.get("start_utc"),
                            "node": owner.get("node"), "gpu": owner.get("gpu")})
    return ({**counts, "total": sum(counts.values()), "running_jobs": running,
             "cmds": cmds, "active_cmds": active_cmds}, asks, escalated)


def queue_row(
    root: Path,
    name: str,
    *,
    now: datetime,
    min_job_s: float,
    cache_hit_frac: float = 0.1,
    min_samples: int,
    window_h: float,
    stale_x: float,
) -> dict:
    meta = store.read_meta(root, name)
    st, asks, escalated = _scan_jobs(root, name, (meta.defaults or {}).get("mem_mib"))
    results = store.load_results(root, name)
    keyed = keyed_durations_s(results, min_job_s=min_job_s, cache_hit_frac=cache_hit_frac)
    durs = [d for _, d in keyed]
    job_samples = [(*job_kind(st["cmds"][k]), d) for k, d in keyed if k in st["cmds"]]
    active_kinds = [job_kind(c) for c in st["active_cmds"]]
    kind = Counter(k for k, _ in active_kinds).most_common(1)
    kind = kind[0][0] if kind else None
    known_units = [u for k, u in active_kinds if k == kind and u is not None]
    median = statistics.median(durs) if len(durs) >= min_samples else None
    rate = completions_per_hour(results, now=now, window_h=window_h)
    on_gpu = [r for r in st["running_jobs"] if r.get("gpu") is not None]
    gpu_nodes: dict[str, int] = {}
    wait_nodes: dict[str, int] = {}
    for r in st["running_jobs"]:
        d = gpu_nodes if r.get("gpu") is not None else wait_nodes
        d[str(r.get("node"))] = d.get(str(r.get("node")), 0) + 1
    ages = [
        (now - t).total_seconds()
        for t in (_parse_iso(r.get("start_utc")) for r in on_gpu)
        if t is not None
    ]
    wave, thr = eta_s(
        pending=st["pending"] + st["running"] - len(on_gpu), running=len(on_gpu),
        median=median, rate_per_h=rate,
        running_ages_s=ages,
    )
    stale_after = (median * stale_x) if median is not None else 3 * 3600.0
    stale = sum(1 for a in ages if a > stale_after)
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
        "done": st["done"],
        "failed": st["failed"],
        "total": st["total"],
        "n_durations": len(durs),
        "median_s": median,
        "durations": durs,
        "job_samples": job_samples,
        "job_kind": kind,
        "job_units": statistics.mean(known_units) if known_units else None,
        "rate_per_h": rate,
        "eta_wave_s": wave,
        "eta_throughput_s": thr,
        "stale_running": stale,
    }


def queue_work_s(row: dict) -> tuple[float, float] | None:
    """(GPU-seconds still to schedule = (pending + waiting) * median,
    wall-seconds until the on-GPU claims finish = max(median - age, 0)); None w/o median."""
    med = row["median_s"]
    if med is None:
        return None
    ages = list(row.get("running_ages_s") or [])
    ages += [0.0] * max(row["running"] - len(ages), 0)
    return ((row["pending"] + row.get("waiting", 0)) * med,
            max((max(med - a, 0.0) for a in ages), default=0.0))


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


def node_work(rows: list[dict], *, assumed_conc: int = 40) -> dict[str, dict]:
    """Per pin node (``unpinned`` for ``-``): GPU-seconds of outstanding work if run now.

    ``gpu_s`` = sum over known-median rows of ``(pending + waiting) * median`` + each on-GPU
    job's ``max(median - age, 0)``; ``unknown`` = rows without a median; ``wall_s`` = gpu_s /
    concurrency, where concurrency = the node's on-GPU jobs (unpinned: all nodes') or
    ``assumed_conc`` when that is 0 (``conc_assumed``).
    """
    conc = node_concurrency(rows)
    out: dict[str, dict] = {}
    for r in rows:
        pin = r.get("node")
        key = "unpinned" if pin in (None, "-") else str(pin)
        w = out.setdefault(key, {"gpu_s": 0.0, "unknown": 0})
        med = r["median_s"]
        if med is None:
            w["unknown"] += 1
            continue
        ages = list(r.get("running_ages_s") or [])
        ages += [0.0] * max(r["running"] - len(ages), 0)
        w["gpu_s"] += (r["pending"] + r.get("waiting", 0)) * med + sum(
            max(med - a, 0.0) for a in ages)
    out.setdefault("unpinned", {"gpu_s": 0.0, "unknown": 0})
    for key, w in out.items():
        c = sum(conc.values()) if key == "unpinned" else conc.get(key, 0)
        w.update(on_gpu=c, conc_assumed=c == 0, wall_s=w["gpu_s"] / (c or assumed_conc))
    return out


def fmt_work(work: dict[str, dict], *, assumed_conc: int = 40) -> str:
    parts = []
    for key in sorted(work, key=lambda k: (k == "unpinned", k)):
        w = work[key]
        conc = f"@{assumed_conc}?" if w["conc_assumed"] else f"@{w['on_gpu']}"
        s = f"{key} {w['gpu_s'] / 3600:.1f} (wall {fmt_dur(w['wall_s'])}{conc})"
        if w["unknown"]:
            s += f" (+{w['unknown']} queues ?)"
        parts.append(s)
    return "work (GPU-h if run now, undisturbed): " + "  ".join(parts)


def node_concurrency(rows: list[dict]) -> dict[str, int]:
    """On-GPU claims per owner node across ``rows``."""
    out: dict[str, int] = {}
    for r in rows:
        for n, k in (r.get("gpu_nodes") or {}).items():
            out[n] = out.get(n, 0) + k
    return out


def _fill_finish(clocks: dict[str, float], nodes: list[str], conc: dict[str, int],
                 dep_s: float, work: float) -> float:
    """Water-fill ``work`` over ``nodes`` from ``max(clock, dep_s)``; return the finish time."""
    starts = sorted((max(clocks.get(n, 0.0), dep_s), conc[n]) for n in nodes)
    rate, t, left = 0.0, starts[0][0], work
    for i, (s0, c) in enumerate(starts):
        if rate > 0 and left > 0:
            step = min(left, rate * (s0 - t))
            left -= step
            t += step / rate
            if left > 0:
                t = s0
        rate += c
    if left > 0:
        t += left / rate
    return t


def annotate_queue_order(
    rows: list[dict], *, concurrency: dict[str, int] | None = None,
    inactive_deps: dict[str, str] | None = None, yielded: dict[str, dict] | None = None,
) -> None:
    """Fill ``start_s``/``finish_s``/``finish_lower_bound``/``eta``/``blocked_reason`` in place.

    See the module docstring (``eta``) for the schedule. ``concurrency`` = on-GPU jobs per
    node (default: from the rows' ``gpu_nodes``). ``inactive_deps`` maps dependency names
    that are not in ``rows`` to ``"complete"`` (satisfied) or anything else (blocked); a
    dependency in neither is treated as missing (blocked). ``yielded`` (``read_yielded``)
    turns a pinned queue whose node has 0 on-GPU jobs but yielded GPUs into ``yielded``.
    """
    yielded = yielded or {}
    conc = {n: c for n, c in (concurrency if concurrency is not None
                               else node_concurrency(rows)).items() if c > 0}
    waiting = node_concurrency([{"gpu_nodes": r.get("wait_nodes")} for r in rows])
    inactive_deps = inactive_deps or {}
    by_name = {r["queue"]: r for r in rows}
    order = sorted(rows, key=lambda r: (-r["prio"], r.get("created_utc") or "", r["queue"]))
    clocks = {n: 0.0 for n in conc}
    unknown = {n: False for n in conc}
    done: set[str] = set()

    def block(r: dict, why: str, eta: str = "blocked", **extra: object) -> None:
        r.update(start_s=None, finish_s=None, finish_lower_bound=False, blocked_reason=why,
                 eta=eta, **extra)

    def process(r: dict) -> None:
        done.add(r["queue"])
        dep_s, dep_lb = 0.0, False
        for dep in r.get("depends_on") or []:
            if dep in by_name:
                d = by_name[dep]
                if d["eta"] in ("blocked", "yielded"):
                    extra = {k: d[k] for k in ("yielded_gpus", "last_reclaim_utc") if k in d}
                    return block(r, f"dep {dep} {d['eta']}", d["eta"], **extra)
                if d["finish_s"] is None:  # dep median unknown: at least after it starts
                    dep_s, dep_lb = max(dep_s, d["start_s"]), True
                else:
                    dep_s, dep_lb = max(dep_s, d["finish_s"]), dep_lb or d["finish_lower_bound"]
            elif inactive_deps.get(dep) != "complete":
                return block(r, f"dep {dep} {inactive_deps.get(dep, 'missing')}")
        pin = r.get("node")
        pin = None if pin in (None, "-") else pin
        nodes = [pin] if pin is not None else sorted(conc)
        if not nodes or nodes[0] not in conc:
            where = pin or "any node"
            w = waiting.get(pin, 0) if pin else sum(waiting.values())
            if pin is not None and pin in yielded:
                y = yielded[pin]
                return block(
                    r, f"0 on-GPU jobs on {pin}: {y['gpus']} GPUs yielded to a foreign user"
                    + (f" ({w} claims waiting for a GPU)" if w else ""), "yielded",
                    yielded_gpus=y["gpus"], last_reclaim_utc=y.get("last_reclaim_utc"))
            return block(r, f"0 on-GPU jobs on {where}" + (
                f" ({w} claims waiting for a GPU)" if w else " (no claims: no live pool seen)"))
        outstanding = r["pending"] + r["running"] + r.get("waiting", 0)
        work = queue_work_s(r)
        lb = dep_lb or any(unknown[n] for n in nodes)
        start = max(min(clocks[n] for n in nodes), dep_s)
        if work is None:
            if outstanding > 0:
                for n in nodes:
                    unknown[n] = True
            ages = r.get("running_ages_s") or []
            eta = f"{fmt_dur(max(ages))}+" if r["running"] > 0 and ages else "?"
            r.update(start_s=start, finish_s=None, finish_lower_bound=lb,
                     blocked_reason=None, eta=eta)
            return
        todo, running_left = work
        fin = running_left
        if todo > 0:
            sched = _fill_finish(clocks, nodes, conc, dep_s, todo)
            for n in nodes:
                clocks[n] = max(clocks[n], sched)
            fin = max(fin, sched)
        else:
            lb = False  # only on-GPU claims left: they finish independent of order
        r.update(start_s=start, finish_s=fin, finish_lower_bound=lb, blocked_reason=None,
                 eta="done" if outstanding == 0 else fmt_dur(fin) + ("+" if lb else ""))

    def unplaced_deps(r: dict) -> bool:
        return any(d in by_name and d not in done for d in r.get("depends_on") or [])

    # Clocks are consumed strictly in claim order. A queue whose active dependency comes
    # later in that order is deferred (reserves nothing, so lower queues are not pushed back)
    # and placed right after its last dependency is placed, starting at that finish.
    deferred: list[dict] = []
    for r in order:
        if unplaced_deps(r):
            deferred.append(r)
            continue
        process(r)
        placed = True
        while placed:
            placed = False
            for d in list(deferred):
                if not unplaced_deps(d):
                    deferred.remove(d)
                    process(d)
                    placed = True
    for r in deferred:  # dependency cycle among active queues
        done.add(r["queue"])
        block(r, "dependency cycle")


def _active(row: dict) -> bool:
    return not row["parked"] and not row["complete"] and (row["pending"] + row["running"]) > 0


def render(rows: list[dict]) -> str:
    cols = [
        ("queue", "queue", "<"), ("prio", "prio", ">"), ("node", "node", "<"),
        ("mem", "mem", ">"),
        ("pend", "pending", ">"), ("run", "running", ">"), ("wait", "waiting", ">"),
        ("done", "done", ">"), ("fail", "failed", ">"), ("median", "median_s", ">"),
        ("r/h", "rate_per_h", ">"), ("eta", "eta", ">"), ("stale", "stale_running", ">"),
    ]
    table: list[list[str]] = []
    for r in rows:
        cells = []
        for _, key, _ in cols:
            v = r.get(key, 0)
            if key == "median_s":
                if v is not None and r.get("median_source") in ("fam", "job"):
                    v = f"~{fmt_dur(v)}({r['median_source']})"
                else:
                    v = fmt_dur(v) + (f"(n={r['n_durations']})" if v is not None else "")
            elif key == "rate_per_h":
                v = "-" if v is None else f"{v:.1f}"
            elif key == "stale_running":
                v = "" if not v else str(v)
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


def _node_conc(node: str, c: int, y: dict | None) -> str:
    if not y:
        return f"{node}:{c}"
    t = _parse_iso(y.get("last_reclaim_utc"))
    rec = f", last reclaim {t.strftime('%H:%MZ')}" if t else ", no reclaim in log tail"
    return f"{node}:{c} ({y['gpus']} yielded{rec})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="jobq root (default: paths YAML queue_root)")
    ap.add_argument("--all", action="store_true", help="include parked / complete / drained queues")
    ap.add_argument("--json", action="store_true", help="emit rows as JSON instead of a table")
    ap.add_argument("--min-job-s", type=float, default=120.0,
                    help="cache hits are never longer than this (ceiling)")
    ap.add_argument("--cache-hit-frac", type=float, default=0.1,
                    help="cache hit = shorter than this x the queue's p90 duration (and --min-job-s)")
    ap.add_argument("--min-borrow-s", type=float, default=30.0,
                    help="a borrowed (fam/job) median below this falls through to the next fallback")
    ap.add_argument("--min-samples", type=int, default=3, help="durations needed before a median is quoted")
    ap.add_argument("--window-h", type=float, default=3.0, help="trailing window for r/h")
    ap.add_argument("--assumed-conc", type=int, default=40,
                    help="concurrency for work wall-clock on a node with 0 on-GPU jobs")
    ap.add_argument("--stale-x", type=float, default=3.0, help="claim age > stale_x*median => stale")
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else _default_root()
    if not root.is_dir():
        print(f"no such root: {root}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    rows = [
        queue_row(
            root, name, now=now, min_job_s=args.min_job_s, cache_hit_frac=args.cache_hit_frac,
            min_samples=args.min_samples,
            window_h=args.window_h, stale_x=args.stale_x,
        )
        for name in store.list_queues(root)
    ]
    borrow_family_medians(rows, min_samples=args.min_samples, min_job_s=args.min_job_s,
                          cache_hit_frac=args.cache_hit_frac, min_borrow_s=args.min_borrow_s)
    inactive = {r["queue"]: ("complete" if r["complete"] else "parked" if r["parked"] else "drained")
                for r in rows if not _active(r)}
    shown = rows if args.all else [r for r in rows if _active(r)]
    active = [r for r in rows if _active(r)]
    # "drained" (no outstanding jobs, no stamp yet) is satisfied for scheduling purposes
    inactive = {k: ("complete" if v == "drained" else v) for k, v in inactive.items()}
    yielded = read_yielded(root)
    annotate_queue_order(active, inactive_deps=inactive, yielded=yielded)
    rows = sorted(shown, key=lambda r: (-r["prio"], r.get("created_utc") or "", r["queue"]))
    conc = node_concurrency(active)
    work = node_work(active, assumed_conc=args.assumed_conc)
    if args.json:
        print(json.dumps({"queues": rows, "work": work, "yielded": yielded},
                         indent=1, default=str))
        return 0
    print(f"root={root}  now={now.strftime('%Y-%m-%dT%H:%MZ')}  "
          f"active_queues={len(active)}  window={args.window_h:g}h  "
          f"concurrency={sum(conc.values())} on-GPU ("
          + " ".join(_node_conc(n, conc.get(n, 0), yielded.get(n))
                     for n in sorted(set(conc) | {r["node"] for r in active
                                                  if r["node"] in yielded})) + ")")
    print(render(rows) if rows else "(no active queues)")
    pend = sum(r["pending"] for r in rows)
    run = sum(r["running"] for r in rows)
    wait = sum(r.get("waiting", 0) for r in rows)
    print(f"totals: pending={pend} running(on-GPU)={run} waiting(claim, no GPU)={wait}")
    print(fmt_work(work, assumed_conc=args.assumed_conc))
    print(textwrap.fill(
          "run = claims holding a GPU; wait = claims parked without one (not concurrency). "
          "median ~X(fam) = borrowed from same-family queues (seed/pair/step tokens "
          "normalised); ~X(job) = from same job kind root-wide (scripts + receiver arch + "
          "epochs; probe lists scaled per run). eta = one best estimate of time until the queue drains: active queues "
          "in claim order (prio desc, created) consume per-node on-GPU concurrency; pinned "
          "queues only their node, unpinned water-fill all nodes; depends_on starts after the "
          "dep (a dependent never pulls a lower-priority dep forward). + = lower bound "
          "(unknown-median queue ahead); ? = no median (Xm+ = oldest on-GPU job age when "
          "running); yielded = its pin node has 0 on-GPU jobs because GPUs are yielded to a "
          "foreign user (--json yielded_gpus, last_reclaim_utc); blocked = no "
          "on-GPU jobs on its node(s) and none yielded, or dep blocked (--json blocked_reason). "
          "work = per pin node GPU-h of (pending+waiting)*median + remaining on-GPU time "
          "(known medians only; +N queues ? = unknown), wall at the node's on-GPU concurrency "
          f"(@N) or an assumed {args.assumed_conc} when it has none (@{args.assumed_conc}?). "
          "mem = GiB ask of pending+running jobs (range when they differ; ^ = an OOM floor "
          "raised a job above its static ask; - = no ask/pool default). "
          "Cross-node liveness is not checkable from here.", width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
