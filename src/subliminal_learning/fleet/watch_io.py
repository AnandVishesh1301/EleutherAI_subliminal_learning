"""Generic campaign watcher IO: node probes over ssh, queue file reads, the loop.

Read-only by construction: ``pgrep``, ``tmux has-session`` / ``list-windows`` /
``capture-pane``, ``find`` / ``grep`` on job logs, ``nvidia-smi``, ``ps``, ``squeue``
and pure file reads of the queue roots. It never runs ``jobq status`` / ``overview``
(they create lock files). Slurm cards are probed locally only (no ssh). Writes only
``health.log``, ``alerts.log`` and ``.watch_state.json`` in its dir.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from subliminal_learning.fleet.ledger import (
    CAMPAIGN_DONE,
    EXECUTOR_SLURM,
    Campaign,
    Card,
    Ledger,
)
from subliminal_learning.fleet.ledger_io import campaign_queues, read_ledger
from subliminal_learning.fleet.nodes import resolve_node
from subliminal_learning.fleet.watch import (
    NodeProbe,
    QueueSnapshot,
    SlurmRootView,
    SlurmSnapshot,
    WatchMemory,
    evaluate,
    evaluate_slurm,
    progress_line,
    watched_hosts,
)
from subliminal_learning.jobqueue import slurm, store

PROBE_TIMEOUT_S = 90
STATE_FILE = ".watch_state.json"
OOM_WINDOW_S = 900.0
# Cells are the direct children of every jobq worker on the node (any pool); count
# them per CUDA_VISIBLE_DEVICES and print the busiest GPU as "busiest=<n> <gpu>".
_BUSIEST_GPU = (
    'echo "busiest=$(for w in $(pgrep -u "$(id -u)" -f "[s]cripts.jobq work"); do '
    "for c in $(cat /proc/$w/task/*/children 2>/dev/null); do "
    "tr '\\0' '\\n' < /proc/$c/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'; "
    'done; done | sort | uniq -c | sort -rn | head -1)"'
)


def _unselfmatch(pattern: str) -> str:
    """Bracket the first letter so ``pgrep -f`` never matches the probe itself."""
    m = re.search(r"[A-Za-z]", pattern)
    if m is None:
        return pattern
    i = m.start()
    return f"{pattern[:i]}[{pattern[i]}]{pattern[i + 1 :]}"


def probe_script(card: Card, host: str) -> str:
    """Bash that prints ``key=value`` lines describing ``card``'s state on ``host``."""
    lines = ["h=$(hostname)", 'echo "host=$h"', "n=0"]
    for root in card.queue_roots:
        pat = _unselfmatch(f"scripts.jobq work --root {root}( |$)")
        lines.append(f"n=$((n + $(pgrep -fc {shlex.quote(pat)})))")
    lines.append('echo "pools=$n"')
    if card.tmux_session:
        sess = shlex.quote(f"={card.tmux_session}")
        lines.append(
            f'echo "session=$(tmux has-session -t {sess} 2>/dev/null && echo 1 || echo 0)"'
        )
    lines.append("o=0")
    for root in card.queue_roots:
        logs = f"{shlex.quote(root)}/{shlex.quote(card.queue_prefix)}*/logs"
        lines.append(
            f'o=$((o + $(find {logs} -maxdepth 1 -name "*.$h.log" -mmin -15 -print0 '
            "2>/dev/null | xargs -0 -r grep -l OutOfMemoryError 2>/dev/null | wc -l)))"
        )
    lines.append('echo "oom15=$o"')
    if card.driver_pattern and card.driver_node == host:
        lines.extend(_driver_lines(card))
    if resolve_node(host).shared:
        lines.append(
            'echo "foreign=$(ps -u angela2 -o args= 2>/dev/null | grep -E '
            "'python.*(worker|sweep|distill|train|stage[.]py|runner)' | grep -vc 'jobq work')\""
        )
    lines.append(_BUSIEST_GPU)
    return "\n".join(lines) + "\n"


def _driver_lines(card: Card) -> list[str]:
    pat = shlex.quote(_unselfmatch(card.driver_pattern or ""))
    lines = [f'echo "driver=$(pgrep -fc {pat})"']
    if card.driver_pane:
        pane = shlex.quote(f"={card.driver_pane}")
        lines.append(
            f'echo "tracebacks=$(tmux capture-pane -p -J -t {pane} -S -400 '
            '2>/dev/null | grep -c Traceback)"'
        )
    return lines


def _probe_kv(text: str) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in text.splitlines()
        if re.match(r"^[a-z0-9]+=", line)
    )


def parse_probe(host: str, text: str) -> NodeProbe:
    """``NodeProbe`` from the probe's ``key=value`` output."""
    kv = _probe_kv(text)
    if kv.get("host") != host:
        tail = text.strip().splitlines()[-1:] or ["no output"]
        return NodeProbe(host, reachable=False, error=tail[0][:200])

    def num(key: str) -> int | None:
        value = kv.get(key)
        return int(value) if value is not None and value.strip().isdigit() else None

    busiest = kv.get("busiest", "").split()
    return NodeProbe(
        host,
        busiest_cells=int(busiest[0]) if len(busiest) == 2 else 0,
        busiest_gpu=int(busiest[1])
        if len(busiest) == 2 and busiest[1].isdigit()
        else None,
        pools=num("pools") or 0,
        session=kv.get("session") == "1",
        oom15=num("oom15") or 0,
        driver=num("driver"),
        tracebacks=num("tracebacks"),
        foreign_jobs=num("foreign") or 0,
    )


def login_probe_script(card: Card, session: str) -> str:
    """Bash for a slurm campaign's login node: supervisor window + driver."""
    sess = shlex.quote(f"={session}")
    lines = [
        'echo "host=$(hostname)"',
        f"echo \"poolswin=$(tmux list-windows -t {sess} -F '#W' 2>/dev/null "
        '| grep -cx pools)"',
    ]
    if card.driver_pattern:
        lines.extend(_driver_lines(card))
    return "\n".join(lines) + "\n"


def probe_login(card: Card, session: str, local_host: str) -> tuple[NodeProbe, bool]:
    """Local driver probe and whether the ``<session>:pools`` window exists."""
    try:
        out = subprocess.run(
            ["bash", "-s"],
            input=login_probe_script(card, session),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return NodeProbe(local_host, reachable=False, error="probe timed out"), False
    text = out.stdout + out.stderr
    return parse_probe(local_host, text), _probe_kv(text).get("poolswin", "0") != "0"


def probe_node(card: Card, host: str, local_host: str) -> NodeProbe:
    """Run the probe on ``host`` (locally when it is this host)."""
    if host == local_host:
        argv = ["bash", "-s"]
    else:
        alias = resolve_node(host).alias
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            alias,
            "bash -s",
        ]
    try:
        out = subprocess.run(
            argv,
            input=probe_script(card, host),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return NodeProbe(host, reachable=False, error="probe timed out")
    return parse_probe(host, out.stdout + out.stderr)


@dataclass
class ResultCache:
    """Result codes already read, keyed ``(root, queue, jobkey)`` -> ``(rc, mtime_ns)``.

    Done results are read once; failed ones are re-statted every pass because a
    requeue + rerun rewrites them under the same name.
    """

    rc: dict[tuple[str, str, str], tuple[int, int]] = field(default_factory=dict)


def _result_codes(root: Path, queue: str, cache: ResultCache) -> dict[str, int]:
    rdir = store.queue_dir(root, queue) / "results"
    try:
        names = [n[:-5] for n in os.listdir(rdir) if n.endswith(".json")]
    except FileNotFoundError:
        names = []
    out: dict[str, int] = {}
    for jk in names:
        ck = (str(root), queue, jk)
        hit = cache.rc.get(ck)
        if hit is not None and hit[0] == 0:
            out[jk] = 0
            continue
        path = rdir / f"{jk}.json"
        try:
            mtime = path.stat().st_mtime_ns
            if hit is None or hit[1] != mtime:
                hit = (int(json.loads(path.read_text()).get("rc", 1)), mtime)
        except (OSError, ValueError):
            continue
        cache.rc[ck] = hit
        out[jk] = hit[0]
    gone = {jk for (r, q, jk) in cache.rc if r == str(root) and q == queue} - set(out)
    for jk in gone:
        cache.rc.pop((str(root), queue, jk), None)
    return out


def queue_snapshot(card: Card, cache: ResultCache) -> QueueSnapshot:
    """Directory listings + cached result reads over the campaign's queues."""
    counts: Counter[str] = Counter()
    failed: set[str] = set()
    paused: set[str] = set()
    by_host: Counter[str] = Counter()
    caps: dict[str, int] = {}
    for root_s in card.queue_roots:
        root = Path(root_s)
        if not root.is_dir():
            continue
        for stop in root.glob("stop.*"):
            paused.add(stop.name.removeprefix("stop."))
        for policy in root.glob("gpu_policy.*.json"):
            host = policy.name.removeprefix("gpu_policy.").removesuffix(".json")
            try:
                caps.setdefault(
                    host, int(json.loads(policy.read_text())["cap_per_gpu"])
                )
            except (OSError, ValueError, KeyError):
                continue
        for queue in campaign_queues(card, root):
            keys = {j.jobkey: j.key for j in store.load_jobs(root, queue)}
            codes = {
                jk: rc
                for jk, rc in _result_codes(root, queue, cache).items()
                if jk in keys
            }
            n_done = sum(1 for rc in codes.values() if rc == 0)
            failed |= {f"{queue}:{keys[jk]}" for jk, rc in codes.items() if rc != 0}
            running = 0
            for _, owner in _open_claims_any_node(root, queue, codes):
                running += 1
                if owner.get("node"):
                    by_host[owner["node"]] += 1
            counts["done"] += n_done
            counts["failed"] += len(codes) - n_done
            counts["running"] += running
            counts["pending"] += len(keys) - len(codes) - running
    state_age = None
    if card.state_file and Path(card.state_file).is_file():
        state_age = time.time() - Path(card.state_file).stat().st_mtime
    done = bool(card.done_marker) and Path(card.done_marker).exists()
    disk = {}
    for mount in card.disk_min_gb:
        try:
            disk[mount] = shutil.disk_usage(mount).free // 2**30
        except OSError:
            continue
    return QueueSnapshot(
        counts=dict(counts),
        failed_keys=frozenset(failed),
        paused_hosts=frozenset(paused),
        running_by_host=dict(by_host),
        caps=caps,
        state_age_s=state_age,
        done=done,
        disk_free_gb=disk,
    )


def _open_claims_any_node(
    root: Path, queue: str, codes: dict[str, int]
) -> list[tuple[str, dict]]:
    """``(jobkey, owner)`` of claim dirs without a result (owner ``{}`` if unreadable)."""
    cdir = store.queue_dir(root, queue) / "claims"
    try:
        names = os.listdir(cdir)
    except FileNotFoundError:
        return []
    out: list[tuple[str, dict]] = []
    for jk in names:
        if jk in codes:
            continue
        try:
            owner = json.loads((cdir / jk / "owner.json").read_text())
        except (OSError, ValueError):
            owner = {}
        out.append((jk, owner))
    return out


def slurm_root_view(root: Path, campaign: str, now_s: float) -> SlurmRootView:
    """State files under ``<root>/slurm/<campaign>`` (pure reads)."""
    sup = slurm.state_dir(root, campaign) / "supervisor.json"
    state = slurm.read_json(sup) or {}
    age = None
    try:
        age = now_s - datetime.fromisoformat(state["updated_utc"]).timestamp()
    except (KeyError, TypeError, ValueError):
        if sup.exists():
            age = now_s - sup.stat().st_mtime
    return SlurmRootView(
        str(root),
        slurm.pool_records(root, campaign),
        age,
        str(state.get("held_back") or ""),
        slurm.paused_path(root, campaign).exists(),
    )


def recent_oom_logs(paths: list[Path], now_s: float) -> int:
    """Logs among ``paths`` written in the last 15 min that mention OutOfMemoryError."""
    recent = []
    for path in paths:
        try:
            if now_s - path.stat().st_mtime <= OOM_WINDOW_S:
                recent.append(str(path))
        except OSError:
            continue
    if not recent:
        return 0
    out = subprocess.run(
        ["grep", "-lF", "OutOfMemoryError", "--", *recent],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_S,
    )
    return len(out.stdout.splitlines())


def slurm_snapshot(
    camp: Campaign, local_host: str, now_s: float
) -> tuple[SlurmSnapshot, NodeProbe]:
    """Login-node reads for a slurm campaign: squeue, state files, job logs, tmux."""
    card = camp.card
    logs: list[Path] = []
    for root_s in card.queue_roots:
        root = Path(root_s)
        if root.is_dir():
            for queue in campaign_queues(card, root):
                logs.extend(store.queue_logs_dir(root, queue).glob("*.log"))
    driver, window = probe_login(card, card.tmux_session or camp.name, local_host)
    snap = SlurmSnapshot(
        jobs=slurm.query_squeue(),
        roots=tuple(
            slurm_root_view(Path(r), camp.name, now_s) for r in card.queue_roots
        ),
        pools_window=window,
        oom15=recent_oom_logs(logs, now_s),
    )
    return snap, driver


def _node_pass(
    camp: Campaign, ledger: Ledger, snap: QueueSnapshot, memory: WatchMemory
) -> tuple[list[str], WatchMemory]:
    hosts = watched_hosts(ledger, camp.name, snap)
    if camp.card.driver_node and camp.card.driver_node not in hosts:
        hosts = (*hosts, camp.card.driver_node)
    local = socket.gethostname()
    with ThreadPoolExecutor(max_workers=max(len(hosts), 1)) as pool:
        probes = dict(
            zip(hosts, pool.map(lambda h: probe_node(camp.card, h, local), hosts))
        )
    return evaluate(camp, ledger, probes, snap, memory, camp.card.disk_min_gb)


def _slurm_pass(
    camp: Campaign, snap: QueueSnapshot, memory: WatchMemory
) -> tuple[list[str], WatchMemory]:
    now = time.time()
    view, driver = slurm_snapshot(camp, socket.gethostname(), now)
    return evaluate_slurm(camp, view, driver, snap, memory, now)


def _load_memory(path: Path) -> WatchMemory:
    if not path.exists():
        return WatchMemory()
    d = json.loads(path.read_text())
    return WatchMemory(
        failed_keys=frozenset(d.get("failed_keys", ())),
        leases=dict(d.get("leases", {})),
        tracebacks=int(d.get("tracebacks", 0)),
        done_reported=bool(d.get("done_reported", False)),
        primed=bool(d.get("primed", False)),
        paused_roots=frozenset(d.get("paused_roots", ())),
        dead_since=dict(d.get("dead_since", {})),
    )


def _save_memory(path: Path, memory: WatchMemory) -> None:
    d = asdict(memory)
    d["failed_keys"] = sorted(memory.failed_keys)
    d["paused_roots"] = sorted(memory.paused_roots)
    for key in ("paused_roots", "dead_since"):  # slurm only: ssh files stay as before
        if not d[key]:
            del d[key]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2) + "\n")
    tmp.replace(path)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_pass(
    campaign: str,
    watch_dir: Path,
    fleet_root: Path,
    n_pass: int,
    progress_every: int,
    cache: ResultCache,
) -> bool:
    """One pass: probe, evaluate, append logs. Returns False once the campaign is done."""
    ledger = read_ledger(fleet_root)
    camp = ledger.campaigns.get(campaign)
    ts = _stamp()
    if camp is None:
        raise KeyError(f"campaign {campaign!r} is not registered in the fleet ledger")
    snap = queue_snapshot(camp.card, cache)
    mem_path = watch_dir / STATE_FILE
    if camp.card.executor == EXECUTOR_SLURM:
        lines, memory = _slurm_pass(camp, snap, _load_memory(mem_path))
    else:
        lines, memory = _node_pass(camp, ledger, snap, _load_memory(mem_path))
    _save_memory(mem_path, memory)
    health = [*lines, progress_line(snap)]
    if camp.status == CAMPAIGN_DONE:
        health.append(
            f"INFO CAMPAIGN finished in the ledger; watcher for {campaign} exiting"
        )
    with open(watch_dir / "health.log", "a") as f:
        f.write(f"=== {ts}\n" + "\n".join(health) + "\n")
    alert_lines = [
        line
        for line in health
        if line.startswith("ALERT")
        or line.startswith("INFO LEASE")
        or line.startswith("INFO SLURM")
        or line.startswith("INFO CAMPAIGN")
        or line.startswith("INFO watcher")
        or (line.startswith("INFO progress") and n_pass % progress_every == 0)
    ]
    if alert_lines:
        with open(watch_dir / "alerts.log", "a") as f:
            f.write("".join(f"{ts} {line}\n" for line in alert_lines))
    return camp.status != CAMPAIGN_DONE


def run_watch(
    campaign: str,
    watch_dir: Path,
    fleet_root: Path,
    *,
    interval_s: float = 300.0,
    progress_every: int = 6,
    once: bool = False,
) -> None:
    """Loop passes every ``interval_s`` until the ledger marks the campaign done."""
    watch_dir = Path(watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    cache = ResultCache()
    while True:
        try:
            keep = run_pass(campaign, watch_dir, fleet_root, n, progress_every, cache)
        except Exception as err:  # noqa: BLE001 — a crashed pass must reach alerts.log
            logger.exception("watcher pass failed")
            with open(watch_dir / "alerts.log", "a") as f:
                f.write(
                    f"{_stamp()} ALERT watcher pass failed: {type(err).__name__}: {err}\n"
                )
            keep = True
        n += 1
        if once or not keep:
            return
        time.sleep(interval_s)
