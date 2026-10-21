"""Idle-GPU hold daemon IO: sampling, hold processes, state and event files, the loop.

Runs on the node it holds. Each hold is ``tools/occupy_gpus.py`` on one GPU in its
own session, marked with ``SL_GPU_HOLD=<gpu>``; it outlives the daemon, which
adopts it again on restart. Holds start only on GPUs empty for ``idle_s`` while the
node's fleet lease is free and no stop file exists; all holds are released when
the lease is taken or the stop file appears (an unreadable ledger keeps them but
starts none). Nothing else is ever signalled.
Files under ``<fleet_root>/holds/<hostname>/``: ``state.json`` (every pass),
``events.log`` (``INFO`` / ``ALERT`` lines for a Monitor), ``hold_gpu<N>.log``.
A failed write there is logged and never stops the daemon. The watchdog (on soar)
restarts a node's tmux restart loop when its ``state.json`` goes stale.
"""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from loguru import logger

from subliminal_learning.fleet.gpu_hold import (
    GATE_OPEN,
    GpuSample,
    HoldPolicy,
    IdleClock,
    advance,
    is_empty,
    parse_cvd,
    parse_gpu_samples,
)
from subliminal_learning.fleet.ledger import FREE
from subliminal_learning.fleet.ledger_io import now_utc, read_ledger
from subliminal_learning.fleet.nodes import resolve_node
from subliminal_learning.utils.io import atomic_write_json

HOLDS_DIR = "holds"
STATE_FILE = "state.json"
EVENTS_FILE = "events.log"
STOP_FILE = "stop"
MARKER = "SL_GPU_HOLD"
SMI_TIMEOUT_S = 30
RELEASE_WAIT_S = 30.0
STALE_AFTER_S = 300.0
GATE_UNKNOWN = "ledger unreadable"
TMUX_SESSION = "gpuhold"
DAEMON_LOG = "daemon.log"
RESTART_TIMEOUT_S = 60


@dataclass(frozen=True)
class Hold:
    """One hold process.

    Attributes:
        gpu: GPU index.
        pid: Session leader pid.
        start_ticks: ``/proc/<pid>/stat`` start time (guards pid reuse).
        started_utc: Launch time (``adopted`` after a daemon restart).
        started_mono: Launch (or adoption) time, monotonic.
        verified: The GPU showed ``hold_gib`` while the hold was alive.
    """

    gpu: int
    pid: int
    start_ticks: int
    started_utc: str
    started_mono: float
    verified: bool = False


def hold_dir(fleet_root: Path, host: str) -> Path:
    """Per-node daemon dir."""
    return Path(fleet_root) / HOLDS_DIR / host


def _event(d: Path, level: str, msg: str) -> None:
    line = f"{now_utc()} {level} {msg}"
    (logger.warning if level == "ALERT" else logger.info)(line)
    try:
        with open(d / EVENTS_FILE, "a") as f:
            f.write(line + "\n")
    except OSError as err:
        logger.warning("events.log write failed: {}", err)


def sample_gpus() -> dict[int, GpuSample] | None:
    """All GPUs of this node, or None when ``nvidia-smi`` fails or does not parse."""
    base = ["nvidia-smi", "--format=csv,noheader,nounits"]
    queries = (
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--query-compute-apps=gpu_uuid,used_memory",
    )
    try:
        gpus, apps = (
            subprocess.run(
                [*base, q],
                capture_output=True,
                text=True,
                check=True,
                timeout=SMI_TIMEOUT_S,
            ).stdout
            for q in queries
        )
        return parse_gpu_samples(gpus, apps)
    except (OSError, subprocess.SubprocessError, ValueError) as err:
        logger.warning("nvidia-smi sample failed: {}", err)
        return None


def _stat(pid: int) -> tuple[str, int, int] | None:
    """(state, session id, start ticks) of a pid, or None when it is gone."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text[text.rindex(")") + 2 :].split()
    return fields[0], int(fields[3]), int(fields[19])


def _environ(pid: int) -> dict[str, str] | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    pairs = raw.decode(errors="replace").split("\0")
    return dict(kv.split("=", 1) for kv in pairs if "=" in kv)


def _own_live_pids() -> list[tuple[int, tuple[str, int, int], dict[str, str]]]:
    """(pid, stat, environ) of this user's non-zombie processes with readable env."""
    uid = os.getuid()
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.stat().st_uid != uid:
                continue
        except OSError:
            continue
        pid = int(d.name)
        st, env = _stat(pid), _environ(pid)
        if st is not None and st[0] != "Z" and env is not None:
            out.append((pid, st, env))
    return out


def our_gpus(n_gpus: int) -> frozenset[int]:
    """GPUs in ``CUDA_VISIBLE_DEVICES`` of our live processes, holds excluded."""
    used: set[int] = set()
    for _, _, env in _own_live_pids():
        if MARKER not in env:
            used |= parse_cvd(env.get("CUDA_VISIBLE_DEVICES", ""), n_gpus)
    return frozenset(used)


def find_holds() -> dict[int, Hold]:
    """Live hold session leaders on this node, by GPU (adopted after a restart)."""
    out: dict[int, Hold] = {}
    for pid, st, env in _own_live_pids():
        gpu = env.get(MARKER, "")
        if st[1] == pid and gpu.isdigit():
            out[int(gpu)] = Hold(int(gpu), pid, st[2], "adopted", time.monotonic())
    return out


def alive(hold: Hold) -> bool:
    """The hold's leader still runs (same start time, not a zombie)."""
    st = _stat(hold.pid)
    return st is not None and st[0] != "Z" and st[2] == hold.start_ticks


def _reap() -> None:
    """Collect exited hold children so they do not linger as zombies."""
    try:
        while os.waitpid(-1, os.WNOHANG)[0] > 0:
            pass
    except ChildProcessError:
        return


def start_hold(gpu: int, script: Path, policy: HoldPolicy, d: Path) -> Hold:
    """Launch one hold on ``gpu`` in a new session."""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env.update({MARKER: str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    cmd = [sys.executable, str(script), "--gpus", str(gpu)]
    cmd += ["--mem-gb", f"{policy.hold_gib:g}"]
    try:
        log = open(d / f"hold_gpu{gpu}.log", "a")  # noqa: SIM115 — closed below
        log.write(f"=== {now_utc()} {' '.join(cmd)}\n")
        log.flush()
    except OSError as err:
        logger.warning("hold log for gpu {} unwritable ({}); no hold log", gpu, err)
        log = open(os.devnull, "w")  # noqa: SIM115 — closed below
    with log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    st = _stat(proc.pid)
    return Hold(gpu, proc.pid, st[2] if st else -1, now_utc(), time.monotonic())


def stop_hold(hold: Hold) -> None:
    """SIGTERM the hold's session; SIGKILL it if still alive after ``RELEASE_WAIT_S``."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not alive(hold):
            return
        try:
            os.killpg(hold.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + RELEASE_WAIT_S
        while time.monotonic() < deadline and alive(hold):
            _reap()
            time.sleep(1.0)


def _log_tail(d: Path, gpu: int) -> str:
    path = d / f"hold_gpu{gpu}.log"
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").strip().splitlines()
    return " | ".join(lines[-3:])[:300]


def gate(fleet_root: Path, host: str, d: Path) -> str:
    """``GATE_OPEN``, or why no hold may exist on ``host`` right now."""
    if (d / STOP_FILE).exists():
        return "stop file set"
    try:
        lease = read_ledger(fleet_root).nodes[host]
    except (OSError, ValueError, KeyError, TypeError) as err:
        return f"{GATE_UNKNOWN} ({type(err).__name__})"
    if lease.state != FREE:
        return f"lease {lease.state} by {lease.holder}"
    return GATE_OPEN


@dataclass
class HoldDaemon:
    """Mutable state of one node's daemon between passes.

    Attributes:
        fleet_root: Fleet ledger dir.
        host: This node's hostname.
        d: ``hold_dir(fleet_root, host)``.
        policy: Thresholds and hold size.
        script: ``tools/occupy_gpus.py``.
        watch: GPUs it may hold (None = all).
        dry_run: Log ``WOULD HOLD`` instead of holding.
        argv: ``scripts.gpu_hold`` arguments, for the watchdog's restart.
        holds: Live holds by GPU.
        clocks: Idle clocks by GPU.
        last_gate: Gate of the previous pass.
    """

    fleet_root: Path
    host: str
    d: Path
    policy: HoldPolicy
    script: Path
    watch: tuple[int, ...] | None
    dry_run: bool
    argv: tuple[str, ...] = ()
    holds: dict[int, Hold] = field(default_factory=dict)
    clocks: dict[int, IdleClock] = field(default_factory=dict)
    last_gate: str = ""

    def check_holds(self, samples: dict[int, GpuSample] | None) -> None:
        """Mark holds verified, drop dead ones, release ones that never filled."""
        for g, hold in list(self.holds.items()):
            if not alive(hold):
                verb = "LOST" if hold.verified else "FAILED"
                tail = _log_tail(self.d, g)
                _event(self.d, "ALERT", f"{verb} hold on gpu {g} exited; log: {tail}")
                del self.holds[g]
                continue
            sample = None if samples is None else samples.get(g)
            if sample is None:
                continue
            if sample.memory_used_mib >= self.policy.hold_mib:
                if not hold.verified:
                    self.holds[g] = replace(hold, verified=True)
                    _event(
                        self.d,
                        "INFO",
                        f"HELD gpu {g} ({sample.memory_used_mib} MiB, pid {hold.pid})",
                    )
                continue
            age = time.monotonic() - hold.started_mono
            if hold.verified or age > self.policy.verify_s:
                _event(
                    self.d,
                    "ALERT",
                    f"FAILED hold on gpu {g} shows {sample.memory_used_mib} MiB "
                    f"< {self.policy.hold_mib}; releasing it",
                )
                stop_hold(hold)
                del self.holds[g]

    def release_all(self, why: str) -> None:
        """Stop every hold of this daemon."""
        for g, hold in sorted(self.holds.items()):
            stop_hold(hold)
            _event(self.d, "INFO", f"RELEASED gpu {g} ({why})")
        self.holds.clear()

    def claim(self, g: int) -> None:
        """Re-check ``g`` with a fresh sample, then hold it (or log it in dry-run)."""
        fresh = sample_gpus()
        sample = None if fresh is None else fresh.get(g)
        if (
            sample is None
            or not is_empty(sample, self.policy)
            or g in our_gpus(len(fresh or {}))
        ):
            self.clocks[g] = IdleClock(None, time.monotonic())
            return
        idle_min = self.policy.idle_s / 60
        if self.dry_run:
            _event(self.d, "INFO", f"WOULD HOLD gpu {g} (empty {idle_min:g} min)")
            self.clocks[g] = IdleClock(None, time.monotonic())
            return
        self.holds[g] = start_hold(g, self.script, self.policy, self.d)
        self.clocks.pop(g, None)
        _event(
            self.d,
            "INFO",
            f"CLAIMING gpu {g} (empty {idle_min:g} min; pid {self.holds[g].pid})",
        )

    def run_pass(self) -> None:
        """Sample, update holds and clocks, start ready holds, write state."""
        now = time.monotonic()
        _reap()
        samples = sample_gpus()
        ours = our_gpus(len(samples) if samples else 0)
        g8 = gate(self.fleet_root, self.host, self.d)
        if g8 != self.last_gate:
            _event(self.d, "INFO", f"gate: {g8}")
            self.last_gate = g8
        self.check_holds(samples)
        if g8 != GATE_OPEN and not g8.startswith(GATE_UNKNOWN) and self.holds:
            self.release_all(g8)
        watch = self.watch if self.watch is not None else tuple(samples or ())
        self.clocks, ready, status = advance(
            self.clocks,
            samples,
            ours,
            g8,
            watch,
            frozenset(self.holds),
            now,
            self.policy,
        )
        for g in ready:
            self.claim(g)
            if g in self.holds:
                status[g] = "held"
        self.write_state(samples, status, g8)

    def write_state(
        self, samples: dict[int, GpuSample] | None, status: dict[int, str], g8: str
    ) -> None:
        """Atomically rewrite ``state.json`` (a failed write is logged only)."""
        gpus = sorted(set(status) | set(self.holds) | set(samples or ()))
        state = {
            "updated_utc": now_utc(),
            "updated_epoch": time.time(),
            "host": self.host,
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "argv": list(self.argv),
            "dry_run": self.dry_run,
            "gate": g8,
            "idle_min": self.policy.idle_s / 60,
            "hold_gib": self.policy.hold_gib,
            "gpus": {
                str(g): {
                    "status": status.get(g, "unwatched"),
                    "memory_used_mib": samples[g].memory_used_mib
                    if samples and g in samples
                    else None,
                }
                for g in gpus
            },
            "holds": {str(g): asdict(h) for g, h in sorted(self.holds.items())},
        }
        try:
            atomic_write_json(self.d / STATE_FILE, state, indent=2)
        except OSError as err:
            logger.warning("state.json write failed: {}", err)


def run_hold_daemon(
    fleet_root: Path,
    policy: HoldPolicy,
    *,
    script: Path,
    watch: tuple[int, ...] | None,
    interval_s: float,
    dry_run: bool,
    argv: tuple[str, ...] = (),
    once: bool = False,
) -> None:
    """Hold GPUs of this node that stay empty, until killed.

    Raises:
        KeyError: This host is not a fleet node.
        FileNotFoundError: ``script`` is missing.
        RuntimeError: Another daemon runs on this node.
    """
    host = resolve_node(socket.gethostname()).hostname
    if not script.is_file():
        raise FileNotFoundError(script)
    if interval_s * 2 > policy.max_gap_s:
        raise ValueError(
            f"interval {interval_s}s too long for max gap {policy.max_gap_s}s"
        )
    d = hold_dir(fleet_root, host)
    d.mkdir(parents=True, exist_ok=True)
    lock = open(f"/tmp/sl_gpu_hold_{os.getuid()}.lock", "a")  # noqa: SIM115 — held for life
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as err:
        raise RuntimeError(f"a gpu_hold daemon already runs on {host}") from err
    daemon = HoldDaemon(
        fleet_root,
        host,
        d,
        policy,
        script.resolve(),
        watch,
        dry_run,
        argv,
        find_holds(),
    )
    _event(
        d,
        "INFO",
        f"daemon start pid {os.getpid()} dry_run={dry_run} "
        f"idle={policy.idle_s / 60:g} min hold={policy.hold_gib:g} GiB "
        f"watch={list(watch) if watch else 'all'} adopted={sorted(daemon.holds)}",
    )
    while True:
        t0 = time.monotonic()
        try:
            daemon.run_pass()
        except Exception as err:  # noqa: BLE001 — a crashed pass must reach events.log
            logger.exception("gpu_hold pass failed")
            _event(d, "ALERT", f"pass failed: {type(err).__name__}: {err}")
            daemon.clocks = {}
        if once:
            return
        time.sleep(max(0.0, interval_s - (time.monotonic() - t0)))


def summary_lines(fleet_root: Path) -> list[str]:
    """One line per node with a hold daemon state file (for ``fleet status``)."""
    out = []
    for path in sorted((Path(fleet_root) / HOLDS_DIR).glob(f"*/{STATE_FILE}")):
        try:
            out.append(_summary(json.loads(path.read_text())))
        except (OSError, ValueError, KeyError, TypeError) as err:
            out.append(f"HOLD {path.parent.name}: state unreadable ({err!r})")
    return out


def _summary(state: dict) -> str:
    age = time.time() - state["updated_epoch"]
    stale = f" STALE {age / 60:.0f} min (daemon down?)" if age > STALE_AFTER_S else ""
    idle = {
        int(g): s["status"]
        for g, s in state["gpus"].items()
        if s["status"].startswith("idle")
    }
    return (
        f"HOLD {resolve_node(state['host']).short}: "
        f"held={sorted(int(g) for g in state['holds'])} gate={state['gate']}"
        + (" dry-run" if state["dry_run"] else "")
        + (f" idle={idle}" if idle else "")
        + f" at {state['updated_utc']}{stale}"
    )


def loop_command(argv: tuple[str, ...], d: Path) -> str:
    """Shell restart loop around ``scripts.gpu_hold <argv>``, run from a pin dir."""
    run = shlex.join([".venv/bin/python", "-m", "scripts.gpu_hold", *argv])
    log = shlex.quote(str(d / DAEMON_LOG))
    return (
        f"mkdir -p {shlex.quote(str(d))}; while true; do {run} 2>&1 | tee -a {log}; "
        'echo "daemon exited; restarting in 60s"; sleep 60; done'
    )


def restart_loop(host: str, cwd: str, argv: tuple[str, ...], d: Path) -> str:
    """Replace tmux ``gpuhold`` on ``host`` with a fresh loop; holds survive it.

    Returns:
        ``ok`` or the error text.
    """
    sess = shlex.quote(TMUX_SESSION)
    script = (
        f"tmux kill-session -t ={sess} 2>/dev/null; "
        f"tmux new-session -d -s {sess} -c {shlex.quote(cwd)} "
        f"{shlex.quote(loop_command(argv, d))} && echo ok"
    )
    argv_sh = ["bash", "-c", script]
    if host != socket.gethostname():
        alias = resolve_node(host).alias
        argv_sh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", alias]
        argv_sh.append(script)
    try:
        out = subprocess.run(
            argv_sh, capture_output=True, text=True, timeout=RESTART_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return "timed out"
    text = (out.stdout + out.stderr).strip()
    return "ok" if out.returncode == 0 and text.endswith("ok") else text[-300:]


def stale_states(fleet_root: Path, now: float) -> list[tuple[Path, dict | None]]:
    """(hold dir, state or None if unreadable) of every node whose state is stale."""
    out = []
    for path in sorted((Path(fleet_root) / HOLDS_DIR).glob(f"*/{STATE_FILE}")):
        try:
            state = json.loads(path.read_text())
            if now - state["updated_epoch"] <= STALE_AFTER_S:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            state = None
        out.append((path.parent, state))
    return out


def run_watchdog(fleet_root: Path, interval_s: float) -> None:
    """Restart the hold loop of every node whose ``state.json`` is stale, forever."""
    while True:
        for d, state in stale_states(fleet_root, time.time()):
            if state is None or not state.get("cwd") or not state.get("argv"):
                _event(
                    d, "ALERT", "watchdog: state stale or unreadable, no restart info"
                )
                continue
            age_min = (time.time() - state["updated_epoch"]) / 60
            rc = restart_loop(state["host"], state["cwd"], tuple(state["argv"]), d)
            _event(
                d,
                "ALERT",
                f"watchdog: state {age_min:.0f} min old; restarted tmux "
                f"{TMUX_SESSION} on {state['host']} from {state['cwd']}: {rc}",
            )
        time.sleep(interval_s)
