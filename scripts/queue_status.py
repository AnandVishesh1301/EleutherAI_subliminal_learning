"""Read-only campaign status for a jobq root: one row per ACTIVE queue with an ETA.

Backs the ``/status`` command. Prints a table of queues that still have pending or
running work (parked queues -- ``meta.node`` starting with ``PARKED`` -- and queues with
a ``complete.stamp`` are hidden unless ``--all``), with:

- ``prio``/``node``: from ``meta.json`` (the pin is the ``node`` field; ``-`` = unpinned).
- ``pend/run/done/fail``: derived from claims + results exactly as ``jobq status`` does.
- ``median``: median wall-clock of this queue's SUCCESSFUL results (``rc == 0``) after
  dropping cache-hit runs shorter than ``--min-job-s`` (default 120 s). ``?`` until
  ``--min-samples`` durations exist -- ETAs are measured, never remembered.
- ``rate/h``: completions per hour over the trailing ``--window-h`` hours (this queue).
- ``eta``: pending work under two models, reported as a range when both are available:
  wave = ``pending * median / running`` (current concurrency holds) and
  throughput = ``pending / rate``. A queue with no running jobs and no recent
  completions is ``blocked`` -- deps, priority starvation, a pin to a node with no pool,
  or an empty pool; that is a diagnosis for the lead, not a number.
- ``finish~``: cumulative wall-clock until this queue drains if pools work active queues in
  priority order at the current global concurrency (``--concurrency`` to override); a
  trailing ``+`` marks a lower bound (a queue with no median is ahead and counted as 0).
- ``stale``: running claims older than ``--stale-x`` x median (or 3 h when no median):
  dead pool thread / crashed node candidates (``jobq status`` shows the owner node).

Usage::

    uv run python -m scripts.queue_status [--root run_logs/queue_angela] [--all] [--json]
"""

from __future__ import annotations

import argparse
import json
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


def durations_s(results: list[dict], *, min_job_s: float) -> list[float]:
    """Successful-result durations (seconds), cache hits shorter than ``min_job_s`` dropped."""
    out: list[float] = []
    for r in results:
        if r.get("rc", 1) != 0:
            continue
        a, b = _parse_iso(r.get("start_utc")), _parse_iso(r.get("end_utc"))
        if a is None or b is None:
            continue
        d = (b - a).total_seconds()
        if d >= min_job_s:
            out.append(d)
    return out


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


def queue_row(
    root: Path,
    name: str,
    *,
    now: datetime,
    min_job_s: float,
    min_samples: int,
    window_h: float,
    stale_x: float,
) -> dict:
    meta = store.read_meta(root, name)
    st = store.queue_status(root, name)
    results = store.load_results(root, name)
    durs = durations_s(results, min_job_s=min_job_s)
    median = statistics.median(durs) if len(durs) >= min_samples else None
    rate = completions_per_hour(results, now=now, window_h=window_h)
    ages = [
        (now - t).total_seconds()
        for t in (_parse_iso(r.get("start_utc")) for r in st["running_jobs"])
        if t is not None
    ]
    wave, thr = eta_s(
        pending=st["pending"], running=st["running"], median=median, rate_per_h=rate,
        running_ages_s=ages,
    )
    stale_after = (median * stale_x) if median is not None else 3 * 3600.0
    stale = sum(1 for a in ages if a > stale_after)
    node = meta.node
    return {
        "queue": name,
        "prio": meta.priority,
        "node": node or "-",
        "parked": bool(node and str(node).startswith("PARKED")),
        "complete": store.queue_complete(root, name),
        "depends_on": list(meta.depends_on or []),
        "pending": st["pending"],
        "running": st["running"],
        "done": st["done"],
        "failed": st["failed"],
        "total": st["total"],
        "n_durations": len(durs),
        "median_s": median,
        "rate_per_h": rate,
        "eta_wave_s": wave,
        "eta_throughput_s": thr,
        "eta": fmt_eta(wave, thr, running=st["running"], pending=st["pending"]),
        "stale_running": stale,
    }


def annotate_queue_order(rows: list[dict], *, concurrency: int) -> None:
    """Fill ``eta`` for idle queues and ``finish_s`` (cumulative, priority order) in place.

    Pools claim the highest-priority READY queue first, so an idle queue with a higher- or
    equal-priority active queue still running is ``queued`` (starved, not stuck); with no
    such queue it is ``blocked`` (deps / pin to a node with no pool / empty pool). The
    cumulative estimate drains active queues in priority order at ``concurrency`` jobs at
    once: each queue's own drain is ``outstanding * median / concurrency``; a queue with no
    median contributes ``?`` and every queue behind it inherits the unknown.
    """
    order = sorted(rows, key=lambda r: (-r["prio"], r["queue"]))
    running_rows = [r for r in order if r["running"] > 0]
    clock = 0.0
    unknown_ahead = False
    for r in order:
        if r["running"] == 0 and r["pending"] > 0:
            # nearest running queue at or above this priority = what it is starved behind
            above = [x for x in running_rows if x["prio"] >= r["prio"]]
            r["eta"] = f"queued<{above[-1]['queue']}" if above else "blocked"
        outstanding = r["pending"] + r["running"]
        if concurrency <= 0:
            r["finish_s"], r["finish_lower_bound"] = None, False
            continue
        if r["median_s"] is None:
            if outstanding > 0:
                unknown_ahead = True  # contributes 0 but makes later finishes lower bounds
        else:
            clock += outstanding * r["median_s"] / concurrency
        r["finish_s"] = clock
        r["finish_lower_bound"] = unknown_ahead


def _active(row: dict) -> bool:
    return not row["parked"] and not row["complete"] and (row["pending"] + row["running"]) > 0


def render(rows: list[dict]) -> str:
    cols = [
        ("queue", "queue", "<"), ("prio", "prio", ">"), ("node", "node", "<"),
        ("pend", "pending", ">"), ("run", "running", ">"), ("done", "done", ">"),
        ("fail", "failed", ">"), ("median", "median_s", ">"), ("rate/h", "rate_per_h", ">"),
        ("eta", "eta", ">"), ("finish~", "finish_s", ">"), ("stale", "stale_running", ">"),
    ]
    table: list[list[str]] = []
    for r in rows:
        cells = []
        for _, key, _ in cols:
            v = r[key]
            if key == "median_s":
                v = fmt_dur(v) + (f"(n={r['n_durations']})" if v is not None else "")
            elif key == "rate_per_h":
                v = "-" if v is None else f"{v:.1f}"
            elif key == "finish_s":
                v = fmt_dur(v) + ("+" if r.get("finish_lower_bound") else "")
            elif key == "stale_running":
                v = "" if not v else str(v)
            elif key == "eta" and str(v).startswith("queued<"):
                # rows are priority-sorted: the queue it is starved behind is the nearest
                # row above with running > 0, so the name only widens the table (--json keeps it)
                v = "queued"
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="jobq root (default: paths YAML queue_root)")
    ap.add_argument("--all", action="store_true", help="include parked / complete / drained queues")
    ap.add_argument("--json", action="store_true", help="emit rows as JSON instead of a table")
    ap.add_argument("--min-job-s", type=float, default=120.0, help="drop cache-hit runs shorter than this")
    ap.add_argument("--min-samples", type=int, default=3, help="durations needed before a median is quoted")
    ap.add_argument("--window-h", type=float, default=3.0, help="trailing window for rate/h")
    ap.add_argument("--stale-x", type=float, default=3.0, help="claim age > stale_x*median => stale")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="jobs-at-once for the cumulative finish~ column (default: running now)")
    args = ap.parse_args(argv)

    root = args.root if args.root is not None else _default_root()
    if not root.is_dir():
        print(f"no such root: {root}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    rows = [
        queue_row(
            root, name, now=now, min_job_s=args.min_job_s, min_samples=args.min_samples,
            window_h=args.window_h, stale_x=args.stale_x,
        )
        for name in store.list_queues(root)
    ]
    if not args.all:
        rows = [r for r in rows if _active(r)]
    concurrency = args.concurrency or sum(r["running"] for r in rows)
    annotate_queue_order(rows, concurrency=concurrency)
    rows.sort(key=lambda r: (-r["prio"], r["queue"]))
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    print(f"root={root}  now={now.strftime('%Y-%m-%dT%H:%MZ')}  "
          f"active_queues={len(rows)}  window={args.window_h:g}h  concurrency={concurrency}")
    print(render(rows) if rows else "(no active queues)")
    pend = sum(r["pending"] for r in rows)
    run = sum(r["running"] for r in rows)
    print(f"totals: pending={pend} running={run}")
    print(textwrap.fill(
          "eta: own drain at current concurrency (wave=median-based, throughput=rate-based; "
          "range when they disagree). queued = starved behind the nearest running queue above "
          "it (--json names it); blocked = idle with nothing above it (deps/pin/pool). "
          "finish~ = cumulative drain in priority "
          "order at `concurrency` jobs at once (+ = lower bound, an unknown-median queue is ahead). "
          "Cross-node liveness is not checkable from here.", width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
