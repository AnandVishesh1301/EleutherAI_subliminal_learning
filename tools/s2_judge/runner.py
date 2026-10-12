"""Judge lane runner (soar GPUs 6-7, tmux `s2core-judge`; plan v2 §12.3).

Every pass: scan the work dir for item specs, find (item, judge) pairs without a
``DONE.json`` under the judge's current code fingerprint, pick one judge model
(highest item priority, then oldest item), and run it over all of its pending items in
one model load:

- tensor-parallel 2 -> one worker on both lane GPUs;
- tensor-parallel 1 -> one data-parallel worker per lane GPU.

Workers run in each judge model's own venv (its spec ``stack`` field; stack N: vLLM
0.30.0 with the user-space cuda-compat-12-9 libraries on ``LD_LIBRARY_PATH`` and a
persistent harmony vocab cache; stack F: the vLLM 0.8.5 venv), offline, with
``PYTHONPATH`` = the pin's ``tools`` dir, each in its own process group. When every
worker exits 0 and the merged shards cover every row, the runner writes ``DONE.json``
(rows, parse failures, unscorable rows, re-judge agreement, worker summaries, code
fingerprint) and alerts when the parse-failure rate exceeds ``--parse-fail-alert``.

Robustness: a worker whose log and shards have not changed for `WATCHDOG_S` is killed
(its process group) and the pass counts as failed; after a failed pass the runner
backs off (60 s doubling to 15 min); a judge failing `max_failures` passes in a row is
parked (ALERT), and un-parked at runner start and after `UNPARK_AFTER_S` (INFO). Lane
GPUs busy for more than `BUSY_ALERT_S` raise an ALERT. Unusable specs are alerted
once and skipped.

State under ``--state-dir``: ``lane_heartbeat.json`` (each pass), ``lane_state.json``
(failure counts, parked judges + park times, current job, backoff, busy-since),
``lane_alerts.log`` (``<utc> ALERT|INFO s2core-judge: ...`` for the watcher),
``lane_logs/`` (worker stdout/stderr). A file lock keeps one runner per state dir.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from s2_judge import spec as specs
from s2_judge import tasks

GPU_IDLE_MAX_MIB = 2_000
DEFAULT_INTERVAL_S = 60
DEFAULT_MAX_FAILURES = 3
DEFAULT_PARSE_FAIL_ALERT = 0.01
WATCHDOG_S = 45 * 60
WATCH_POLL_S = 30
BACKOFF_START_S = 60
BACKOFF_MAX_S = 15 * 60
UNPARK_AFTER_S = 2 * 3600
BUSY_ALERT_S = 15 * 60
_SPEC_ERRORS = (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError)
_FINALIZE_ERRORS = (ValueError, KeyError, TypeError, OSError, RuntimeError)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def alert(state_dir: Path, level: str, message: str) -> None:
    """Append one ``<utc> <LEVEL> s2core-judge: <message>`` line to the lane log."""
    line = f"{_utc()} {level} s2core-judge: {message}"
    if level == "ALERT":
        logger.warning(line)
    else:
        logger.info(line)
    with open(state_dir / "lane_alerts.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---- completion ------------------------------------------------------------------------


def _rejudge_agreement(
    task: str, main: Mapping[str, Mapping], again: Mapping[str, Mapping]
) -> dict[str, Any]:
    """Batch-nondeterminism summary of the re-judged rows."""
    pairs = [(main[k], again[k]) for k in again if k in main]
    if not pairs:
        return {"n": 0}
    if task == tasks.SCENARIO_CHOICE:
        same = sum(a.get("label") == b.get("label") for a, b in pairs)
        return {"n": len(pairs), "label_agreement": same / len(pairs)}
    if task == tasks.AI_LEAK:
        same = sum(a.get("flag") == b.get("flag") for a, b in pairs)
        diffs = [
            abs(a["p_yes"] - b["p_yes"])
            for a, b in pairs
            if a.get("p_yes") is not None and b.get("p_yes") is not None
        ]
        return {
            "n": len(pairs),
            "flag_agreement": same / len(pairs),
            "max_abs_p_yes_diff": max(diffs) if diffs else None,
        }
    diffs = [
        abs(a[f] - b[f])
        for a, b in pairs
        for f in (tasks.ALIGNMENT, tasks.COHERENCE)
        if isinstance(a.get(f), float) and isinstance(b.get(f), float)
    ]
    return {"n": len(pairs), "max_abs_score_diff": max(diffs) if diffs else None}


def finalize(
    spec: Mapping[str, Any],
    key: str,
    *,
    parse_fail_alert: float = DEFAULT_PARSE_FAIL_ALERT,
    state_dir: Path | None = None,
) -> bool:
    """Write ``DONE.json`` when the merged shards cover every row; returns done.

    Alerts (to `state_dir`) when the item's parse-failure rate exceeds
    `parse_fail_alert`.
    """
    out_dir = specs.judge_dir(spec, key)
    if (out_dir / specs.DONE_FILE).is_file():
        return True
    rows = specs.read_rows(spec)
    ids = {r["row_id"] for r in rows}
    merged = specs.merged_judgments(spec, key)
    missing = ids - set(merged)
    if missing:
        return False
    extra = set(merged) - ids
    if extra:
        raise RuntimeError(f"{spec['item_id']}/{key}: judgments for unknown rows")
    n_fail = sum(not rec.get("parse_ok") for rec in merged.values())
    n_unscorable = sum(bool(rec.get("unscorable")) for rec in merged.values())
    expected_re = {i for i in ids if specs.is_rejudge_row(i)}
    again = specs.merged_rejudgments(spec, key)
    if not expected_re <= set(again):
        return False
    summaries = [
        json.loads(p.read_text())
        for p in sorted((out_dir / "logs").glob("worker-w*.json"))
    ]
    doc = {
        "item_id": spec["item_id"],
        "task": spec["task"],
        "judge": key,
        "judge_entry": dict(specs.judge_entry(spec, key)),
        "n_rows": len(ids),
        "n_judged": len(merged),
        "n_parse_fail": n_fail,
        "parse_fail_rate": n_fail / len(ids) if ids else 0.0,
        "n_unscorable": n_unscorable,
        "rejudge": _rejudge_agreement(spec["task"], merged, again),
        "workers": summaries,
        "input_sha256": spec["input_sha256"],
        "fingerprint": specs.judge_fingerprint_of(spec, key),
        "completed_at": _utc(),
    }
    _write_json(out_dir / specs.DONE_FILE, doc)
    if state_dir is not None and doc["parse_fail_rate"] > parse_fail_alert:
        alert(
            state_dir,
            "ALERT",
            f"{spec['item_id']} / {key}: parse-failure rate "
            f"{doc['parse_fail_rate']:.3f} > {parse_fail_alert}",
        )
    return True


# ---- lane ------------------------------------------------------------------------------


def lane_gpus_idle(gpus: Sequence[int]) -> bool:
    """True when every lane GPU uses less than `GPU_IDLE_MAX_MIB` (nvidia-smi)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    used = {int(i): int(m) for i, m in (line.split(",") for line in out.splitlines())}
    return all(used[g] < GPU_IDLE_MAX_MIB for g in gpus)


def pending_by_judge(
    work_dir: Path,
    state_dir: Path,
    only: frozenset[str] | None,
    seen: set[str],
    parse_fail_alert: float = DEFAULT_PARSE_FAIL_ALERT,
) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """``{judge key: [(spec path, spec), ...]}`` of (item, judge) pairs not yet done.

    Unusable specs are alerted once per message (`seen`) and skipped.
    """
    out: dict[str, list[tuple[Path, dict[str, Any]]]] = {}

    def once(message: str) -> None:
        if message not in seen:
            seen.add(message)
            alert(state_dir, "ALERT", message)

    for path in sorted(work_dir.glob("*.json")):
        try:
            spec = specs.load_spec(path)
        except _SPEC_ERRORS as err:
            once(f"invalid spec {path.name}: {err}")
            continue
        for judge in spec["judges"]:
            key = judge["key"]
            if only is not None and key not in only:
                continue
            try:
                done = finalize(
                    spec, key, parse_fail_alert=parse_fail_alert, state_dir=state_dir
                )
            except _FINALIZE_ERRORS as err:
                once(f"{path.name} / {key} skipped: {err}")
                continue
            if not done:
                out.setdefault(key, []).append((path, spec))
    return out


def choose_judge(
    pending: Mapping[str, Sequence[tuple[Path, Mapping[str, Any]]]],
    parked: set[str],
) -> str | None:
    """The judge whose best pending item has the highest priority, then the oldest."""
    best: tuple[Any, ...] | None = None
    choice = None
    for key, items in pending.items():
        if key in parked:
            continue
        rank = max((s["priority"], -_epoch(s["created_at"])) for _, s in items)
        if best is None or rank > best:
            best, choice = rank, key
    return choice


def _epoch(stamp: str) -> float:
    return time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def worker_env(
    stack: str,
    gpus: str,
    tools_dir: Path,
    hub_cache: Path,
    compat_dir: Path | None,
    tiktoken_dir: Path | None,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tools_dir),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpus,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_CACHE": str(hub_cache),
            "HF_HOME": str(hub_cache.parent),
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.pop("VIRTUAL_ENV", None)
    if stack == "N":
        if compat_dir is None or tiktoken_dir is None:
            raise ValueError("stack N needs --compat-dir and --tiktoken-dir")
        env["LD_LIBRARY_PATH"] = str(compat_dir)
        env["TIKTOKEN_RS_CACHE_DIR"] = str(tiktoken_dir)
    return env


def _last_activity(
    log_path: Path,
    items: Sequence[tuple[Path, Mapping[str, Any]]],
    key: str,
    index: int,
    n_workers: int,
) -> float:
    """Latest mtime of a worker's log and its shards across `items`."""
    paths = [log_path]
    for _, spec in items:
        out = specs.judge_dir(spec, key)
        paths += [
            out / specs.shard_name(kind, index, n_workers)
            for kind in ("judgments", "rejudge")
        ]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def run_judge(
    key: str,
    items: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    gpus: Sequence[int],
    python: Mapping[str, Path],
    tools_dir: Path,
    compat_dir: Path | None,
    tiktoken_dir: Path | None,
    state_dir: Path,
    watchdog_s: float = WATCHDOG_S,
    poll_s: float = WATCH_POLL_S,
) -> bool:
    """Run judge `key` over `items` on the lane GPUs; True when every worker exits 0.

    Each worker runs in its judge model's own venv (``python[entry["stack"]]``) and
    process group; a worker silent (no log or shard write) for `watchdog_s` is killed
    with its group and the run fails.
    """
    first = items[0][1]
    entry = specs.judge_entry(first, key)
    stack, tp = entry["stack"], entry["tensor_parallel"]
    if tp == 2:
        groups = [",".join(str(g) for g in gpus[:2])]
    else:
        groups = [str(g) for g in gpus]
    spec_paths = [str(path) for path, _ in items]
    log_dir = state_dir / "lane_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    procs = []
    for index, group in enumerate(groups):
        cmd = [
            str(python[stack]),
            "-m",
            "s2_judge.worker",
            "--judge",
            key,
            "--worker-index",
            str(index),
            "--num-workers",
            str(len(groups)),
        ]
        for path in spec_paths:
            cmd += ["--spec", path]
        env = worker_env(
            stack,
            group,
            tools_dir,
            Path(first["hf_hub_cache"]),
            compat_dir,
            tiktoken_dir,
        )
        log_path = log_dir / f"{stamp}_{key}_w{index:02d}.log"
        handle = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, env=env, stdout=handle, stderr=handle, start_new_session=True
        )
        procs.append((index, proc, handle, log_path))
        logger.info(
            "launched {} worker {} on GPUs {} -> {}", key, index, group, log_path
        )
    ok = True
    killed: set[int] = set()
    while True:
        running = [p for p in procs if p[1].poll() is None]
        if not running:
            break
        now = time.time()
        for index, proc, _, log_path in running:
            idle = now - _last_activity(log_path, items, key, index, len(groups))
            if idle > watchdog_s and index not in killed:
                os.killpg(proc.pid, signal.SIGKILL)
                killed.add(index)
                alert(
                    state_dir,
                    "ALERT",
                    f"{key} worker {index} silent for {idle / 60:.0f} min; killed its "
                    f"process group; log {log_path}",
                )
        time.sleep(poll_s)
    for index, proc, handle, log_path in procs:
        code = proc.wait()
        handle.close()
        if code != 0 or index in killed:
            ok = False
            alert(state_dir, "ALERT", f"{key} worker exited {code}; log {log_path}")
    return ok


IDLE = "idle"
BUSY = "busy"
RAN_OK = "ran_ok"
RAN_FAILED = "ran_failed"


def unpark(state: dict[str, Any], state_dir: Path, now: float, *, all_: bool) -> None:
    """Un-park every judge (`all_`) or those parked for more than `UNPARK_AFTER_S`."""
    parked_at = state.setdefault("parked_at", {})
    keep = []
    for key in state.get("parked", []):
        since = parked_at.get(key, 0.0)
        if all_ or now - since > UNPARK_AFTER_S:
            parked_at.pop(key, None)
            state.setdefault("failures", {}).pop(key, None)
            reason = "runner start" if all_ else f"{UNPARK_AFTER_S // 3600} h cooldown"
            alert(state_dir, "INFO", f"{key} un-parked ({reason})")
        else:
            keep.append(key)
    state["parked"] = keep


def _note_busy(state: dict[str, Any], state_dir: Path, key: str, now: float) -> None:
    since = state.get("busy_since")
    if since is None:
        state["busy_since"] = now
        state["busy_alerted"] = False
        alert(state_dir, "INFO", f"lane GPUs busy; {key} waits")
    elif now - since > BUSY_ALERT_S and not state.get("busy_alerted"):
        state["busy_alerted"] = True
        alert(
            state_dir,
            "ALERT",
            f"lane GPUs busy for {(now - since) / 60:.0f} min; {key} waits",
        )


def run_pass(args: argparse.Namespace, state: dict[str, Any], seen: set[str]) -> str:
    """One lane pass: `IDLE`, `BUSY`, `RAN_OK` or `RAN_FAILED`."""
    state_dir: Path = args.state_dir
    now = time.time()
    unpark(state, state_dir, now, all_=False)
    only = frozenset(args.only.split(",")) if args.only else None
    pending = pending_by_judge(
        args.work_dir, state_dir, only, seen, args.parse_fail_alert
    )
    parked = set(state.setdefault("parked", []))
    _write_json(
        state_dir / "lane_heartbeat.json",
        {
            "at": _utc(),
            "pending": {k: [s["item_id"] for _, s in v] for k, v in pending.items()},
            "parked": sorted(parked),
        },
    )
    key = choose_judge(pending, parked)
    if key is None:
        state["busy_since"] = None
        _write_json(state_dir / "lane_state.json", state)
        return IDLE
    if not lane_gpus_idle(args.gpus):
        _note_busy(state, state_dir, key, now)
        _write_json(state_dir / "lane_state.json", state)
        return BUSY
    state["busy_since"] = None
    items = pending[key]
    state["current"] = {
        "judge": key,
        "items": [s["item_id"] for _, s in items],
        "at": _utc(),
    }
    _write_json(state_dir / "lane_state.json", state)
    ok = run_judge(
        key,
        items,
        gpus=args.gpus,
        python={"N": args.python_n, "F": args.python_f},
        tools_dir=args.tools_dir,
        compat_dir=args.compat_dir,
        tiktoken_dir=args.tiktoken_dir,
        state_dir=state_dir,
    )
    failures = state.setdefault("failures", {})
    if ok:
        failures.pop(key, None)
        done = [
            s["item_id"]
            for _, s in items
            if finalize(
                s, key, parse_fail_alert=args.parse_fail_alert, state_dir=state_dir
            )
        ]
        alert(state_dir, "INFO", f"{key} finished {len(done)}/{len(items)} items")
    else:
        failures[key] = failures.get(key, 0) + 1
        if failures[key] >= args.max_failures:
            parked.add(key)
            state["parked"] = sorted(parked)
            state.setdefault("parked_at", {})[key] = time.time()
            alert(
                state_dir, "ALERT", f"{key} parked after {failures[key]} failed passes"
            )
    state["current"] = None
    _write_json(state_dir / "lane_state.json", state)
    return RAN_OK if ok else RAN_FAILED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--gpus", type=lambda s: [int(g) for g in s.split(",")], required=True
    )
    parser.add_argument("--python-n", type=Path, required=True)
    parser.add_argument("--python-f", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--compat-dir", type=Path, default=None)
    parser.add_argument("--tiktoken-dir", type=Path, default=None)
    parser.add_argument("--only", default="", help="Comma-separated judge keys.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument(
        "--parse-fail-alert", type=float, default=DEFAULT_PARSE_FAIL_ALERT
    )
    parser.add_argument(
        "--once", action="store_true", help="Run until nothing is pending."
    )
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_dir / "locks" / "lane.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another lane runner holds {lock_path}") from None
    state_path = args.state_dir / "lane_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    unpark(state, args.state_dir, time.time(), all_=True)
    _write_json(state_path, state)
    only = frozenset(args.only.split(",")) if args.only else None
    seen: set[str] = set()
    backoff = BACKOFF_START_S
    while True:
        status = run_pass(args, state, seen)
        if status == RAN_FAILED:
            logger.info("failed pass; backing off {} s", backoff)
            time.sleep(backoff)
            backoff = min(2 * backoff, BACKOFF_MAX_S)
            continue
        if status == RAN_OK:
            backoff = BACKOFF_START_S
            continue
        if args.once:
            pending = pending_by_judge(
                args.work_dir, args.state_dir, only, seen, args.parse_fail_alert
            )
            parked = set(state.get("parked", []))
            if not [k for k in pending if k not in parked]:
                logger.info("nothing pending; exiting (--once)")
                return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
