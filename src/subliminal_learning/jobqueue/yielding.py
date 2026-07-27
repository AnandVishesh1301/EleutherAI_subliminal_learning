"""Opt-in GPU yielding: hand a card to a foreign user instead of fighting for it.

When a *foreign* process (one this node cannot see in its own namespace) appears on a GPU
we are using, an opt-in policy makes our jobq jobs on that card die (or drain) and the card
is left off-limits until the foreigner leaves — so we yield to a collaborator rather than
co-loading the GPU into an OOM.

**Namespace-safe detection (why we do NOT read /proc uid of nvidia-smi pids).**
``nvidia-smi --query-compute-apps`` reports *host-namespace* PIDs, which are DIFFERENT
numbers from this container's PIDs — an smi pid resolved in our ``/proc`` lands on an
unrelated process (empirically uid=101 daemons), so looking up its uid is actively
misleading. Instead we count from two independent sides:

- ``total_compute_procs_from_smi(g)`` is the compute-proc count nvidia-smi sees on that
  card (mapped ``gpu_uuid`` -> index, since the compute-apps query has no index column).
- ``local_gpu_procs(g)`` walks OUR ``/proc/<pid>/fd/*`` and attributes each local CUDA
  process to the specific GPU(s) it actually uses. Everything it finds is ours-or-this-node's
  — INCLUDING local root-owned daemons, which are therefore counted as non-foreign and never
  trigger a yield (this is how "don't yield to root" is honored).

``foreign_count(g) = total_from_smi(g) - len(local on g)``, floored at 0. Anything on the
card we cannot see locally is another container/user.

**CUDA opens EVERY device node — attribution is required (measured on the real machine).**
A CUDA process does NOT hold fds on only the GPU it uses: at init it enumerates and opens
*all* visible ``/dev/nvidiaN`` nodes. A controlled 1 GiB allocation pinned to physical GPU 6
held 3 fds on every ``/dev/nvidia0..7`` (plus ``nvidiactl``/``nvidia-uvm``) and **21** fds on
``/dev/nvidia6`` — the card actually in use. So a naive "has an fd on ``/dev/nvidia<g>``"
test counts ONE of our jobs as local on all 8 cards, masking a real collaborator to
``foreign=0`` on the very cards where they are present (the exact live failure this replaced).
We therefore attribute each local CUDA proc to specific GPU(s), in priority order:

1. ``CUDA_VISIBLE_DEVICES`` from ``/proc/<pid>/environ`` (primary; readable for our own uid,
   holds the PHYSICAL index, and jobq sets it on every job it launches — a comma list means
   several GPUs; empty entries ignored).
2. Max-fd fallback: among the ``/dev/nvidiaN`` nodes the proc has open, attribute it to the
   node(s) with strictly more fds than the minimum baseline (21 vs 3 above). If every node
   ties, fall through to 3.
3. Attribute to ALL policy GPUs (conservative: an unattributable local CUDA process keeps
   every card from being yielded — erring toward keeping our jobs alive, not yielding wrong).

HONEST LIMITATION: a root-owned daemon in ANOTHER container is indistinguishable from a
collaborator's job — if that ever misfires, raise ``yield_min_foreign_procs``.

All ``nvidia-smi`` access is behind injectable callables (module-level defaults, overridable
per :class:`YieldWatchdog`) exactly like ``gpu.query_free_mem`` — so tests run GPU-free.

The set of currently-yielded GPUs is persisted to ``yielded.<hostname>.json`` under the
queue root so :meth:`GpuManager.acquire` skips them and the marker survives a pool restart.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from subliminal_learning.jobqueue.store import now_iso
from subliminal_learning.utils.io import atomic_write_text

_NVIDIA_DEV = re.compile(r"^/dev/nvidia(\d+)$")  # the device node only, not nvidiactl/-uvm


# --------------------------- nvidia-smi (injectable) ---------------------------


def query_compute_apps() -> list[tuple[int, str]]:
    """(pid, gpu_uuid) for every compute proc via ``nvidia-smi`` (host-namespace pids!).

    The pids are host-namespace and MUST NOT be resolved in our ``/proc`` — only the count
    per ``gpu_uuid`` is used. Raises ``CalledProcessError`` on smi failure (caller treats a
    raise as "no information" and does not yield).
    """
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    apps: list[tuple[int, str]] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        apps.append((int(parts[0]), parts[2]))
    return apps


def query_gpu_uuids() -> dict[int, str]:
    """GPU index -> uuid via ``nvidia-smi`` (the compute-apps query carries no index)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    uuids: dict[int, str] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx, uuid = (x.strip() for x in line.split(",", 1))
        uuids[int(idx)] = uuid
    return uuids


# --------------------------- detection ---------------------------


def read_proc_environ(pid: int, *, proc_root: str = "/proc") -> str | None:
    """``CUDA_VISIBLE_DEVICES`` from ``/proc/<pid>/environ``, or ``None`` if absent/unreadable.

    ``environ`` is a NUL-separated ``KEY=VALUE`` blob; readable for our own uid (a foreign
    uid raises ``PermissionError`` -> ``None``, so we fall back to the fd heuristic).
    """
    try:
        blob = (Path(proc_root) / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    for entry in blob.split(b"\x00"):
        if entry.startswith(b"CUDA_VISIBLE_DEVICES="):
            return entry[len(b"CUDA_VISIBLE_DEVICES=") :].decode("utf-8", "replace")
    return None


def _nvidia_fd_counts(pdir: Path) -> dict[int, int] | None:
    """``{gpu_index: fd_count}`` for the ``/dev/nvidiaN`` nodes ``pdir`` has open.

    ``None`` when the pid's ``/proc/<pid>/fd`` vanished mid-scan or is unreadable
    (``PermissionError``); ``{}`` when it holds no ``/dev/nvidiaN`` fd (not a CUDA proc).
    """
    try:
        fds = list((pdir / "fd").iterdir())
    except (FileNotFoundError, NotADirectoryError, ProcessLookupError, PermissionError):
        return None
    except OSError:
        return None
    counts: dict[int, int] = {}
    for fd in fds:
        try:
            dest = os.readlink(fd)
        except OSError:
            continue
        m = _NVIDIA_DEV.match(dest)
        if m:
            g = int(m.group(1))
            counts[g] = counts.get(g, 0) + 1
    return counts


def _attribute_gpus(pid: int, counts: dict[int, int], environ_reader) -> set[int] | None:
    """The GPU(s) a local CUDA proc uses, or ``None`` meaning "attribute to ALL policy GPUs".

    Priority: ``CUDA_VISIBLE_DEVICES`` (physical indices) -> strictly-above-baseline fd count
    -> ``None`` (conservative catch-all). See the module docstring for why fds alone lie.
    """
    cvd = environ_reader(pid)
    if cvd:
        idxs: set[int] = set()
        for tok in cvd.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idxs.add(int(tok))
            except ValueError:
                continue  # e.g. a GPU-UUID form -> ignore, fall through to fd heuristic
        if idxs:
            return idxs
    baseline = min(counts.values())
    hot = {g for g, c in counts.items() if c > baseline}
    return hot or None


def local_gpu_procs(
    gpu_index: int, *, proc_root: str = "/proc", environ_reader=None
) -> dict[int, int]:
    """Pids in OUR namespace attributed to physical ``gpu_index`` -> uid.

    Scans ``<proc_root>/<pid>/fd/*`` for ``/dev/nvidiaN`` nodes and attributes each CUDA proc
    to specific GPU(s) (see :func:`_attribute_gpus` — CUDA opens EVERY node, so a raw fd match
    would mask foreigners). A proc attributed to "all" (unattributable) is counted on every
    GPU. Vanished pids (racy) and ``PermissionError``s are skipped. The uid comes from
    ``os.stat("/proc/<pid>")`` and is informational — NOT used to decide foreign-ness (a local
    proc is non-foreign at ANY uid, including root). ``environ_reader`` is injectable for
    tests; the default reads ``CUDA_VISIBLE_DEVICES`` from the same ``proc_root``.
    """
    reader = environ_reader
    if reader is None:
        reader = lambda pid: read_proc_environ(pid, proc_root=proc_root)  # noqa: E731
    found: dict[int, int] = {}
    proc = Path(proc_root)
    try:
        entries = list(proc.iterdir())
    except OSError:
        return found
    for pdir in entries:
        if not pdir.name.isdigit():
            continue
        counts = _nvidia_fd_counts(pdir)
        if not counts:  # None (vanished/unreadable) or {} (no nvidia fd) -> not our GPU proc
            continue
        pid = int(pdir.name)
        attributed = _attribute_gpus(pid, counts, reader)
        if attributed is not None and gpu_index not in attributed:
            continue
        try:
            uid = os.stat(pdir).st_uid
        except OSError:
            continue  # /proc/<pid> vanished mid-scan
        found[pid] = uid
    return found


def total_compute_procs_from_smi(
    gpu_index: int,
    *,
    compute_apps_query=query_compute_apps,
    uuid_query=query_gpu_uuids,
) -> int | None:
    """Compute-proc count nvidia-smi sees on ``gpu_index``, or ``None`` on any smi problem.

    Returns ``None`` (fail-safe: "no information", never yield) if smi fails, is unparsable,
    or ``gpu_index`` has no uuid in the index map.
    """
    try:
        uuids = uuid_query()
        apps = compute_apps_query()
    except Exception:  # noqa: BLE001 — any smi/parse failure means "no information"
        return None
    uuid = uuids.get(gpu_index)
    if uuid is None:
        return None
    return sum(1 for _pid, u in apps if u == uuid)


def foreign_count(
    gpu_index: int,
    *,
    compute_apps_query=query_compute_apps,
    uuid_query=query_gpu_uuids,
    proc_root: str = "/proc",
    environ_reader=None,
) -> int | None:
    """``total_from_smi - len(local on g)``, floored at 0; ``None`` on no smi info."""
    total = total_compute_procs_from_smi(
        gpu_index, compute_apps_query=compute_apps_query, uuid_query=uuid_query
    )
    if total is None:
        return None
    local = local_gpu_procs(gpu_index, proc_root=proc_root, environ_reader=environ_reader)
    return max(0, total - len(local))


# --------------------------- yield marker file ---------------------------


def marker_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"yielded.{hostname}.json"


def read_yielded(root: Path, hostname: str) -> dict[int, dict]:
    """Read the yield marker as ``{gpu: {first_seen, last_seen, foreign}}`` ({} if absent).

    Corrupt/partial files read as empty (fail-safe toward keeping GPUs usable).
    """
    try:
        raw = json.loads(marker_path(root, hostname).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, dict] = {}
    for k, v in (raw.get("gpus") or {}).items():
        try:
            out[int(k)] = dict(v)
        except (ValueError, TypeError):
            continue
    return out


def write_yielded(root: Path, hostname: str, data: dict[int, dict]) -> None:
    """Persist the yield marker atomically (safe against concurrent read/write)."""
    body = {"hostname": hostname, "gpus": {str(g): v for g, v in data.items()}}
    atomic_write_text(marker_path(root, hostname), json.dumps(body, indent=2) + "\n")


def yielded_gpus(root: Path, hostname: str) -> set[int]:
    """The GPU indices currently off-limits on this node (empty if the marker is missing)."""
    return set(read_yielded(root, hostname).keys())


# --------------------------- progress estimation (drain_if_near_done) ---------------------------
#
# Generic, best-effort, and deliberately fail-safe toward yielding promptly: the framework
# runs ARBITRARY shell commands, so there is no universal progress signal. We try, in order:
#   1. a per-policy log regex (opt-in, most accurate) — the LAST (current, total) match in the
#      tail of the job's log (e.g. HF/tqdm ``21507/25000``);
#   2. elapsed / median-duration-of-completed-jobs in the same queue (needs a min sample);
#   3. nothing known -> 0.0 (the job is killed — never hoard a card we promised to vacate).


@dataclass(frozen=True)
class DrainSpec:
    """Per-yield context the watchdog hands the pool for a ``drain_if_near_done`` evaluation.

    ``yield_started_ts`` and ``now`` are epoch seconds from the watchdog's clock (the safety
    cap kills a spared job once ``now - yield_started_ts >= max_s``); ``progress_regex`` opts
    into log-based progress (two numeric groups = current/total).
    """

    threshold: float
    max_s: float
    progress_regex: str | None
    yield_started_ts: float
    now: float


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _read_log_tail(path: str, *, max_bytes: int = 16384) -> str | None:
    """The last ``max_bytes`` of ``path`` decoded as text, or ``None`` if unreadable/missing.

    Only the tail is read — these logs hold thousands of tqdm lines and the useful progress
    marker is always the most recent one.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))  # back to the tail's start (0 for a small file)
            data = f.read()
    except OSError:
        return None
    return data.decode("utf-8", "replace")


def progress_from_log(
    log_path: str | None, progress_regex: str | None, *, tail_reader=_read_log_tail
) -> float | None:
    """current/total from the LAST regex match in the log tail, or ``None`` to fall through.

    ``None`` on: no regex/path, missing/unreadable log, no match, a bad regex, a non-numeric
    group, or ``total <= 0``. ``tail_reader`` is injectable for tests.
    """
    if not log_path or not progress_regex:
        return None
    text = tail_reader(log_path)
    if not text:
        return None
    try:
        pat = re.compile(progress_regex)
    except re.error:
        return None
    last = None
    for last in pat.finditer(text):
        pass
    if last is None:
        return None
    try:
        cur = float(last.group(1))
        total = float(last.group(2))
    except (IndexError, ValueError):
        return None
    if total <= 0:
        return None
    return _clamp01(cur / total)


def _duration_s(start_iso: str | None, end_iso: str | None) -> float | None:
    if not start_iso or not end_iso:
        return None
    try:
        return (datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)).total_seconds()
    except (TypeError, ValueError):
        return None


def median_duration_s(results: list[dict], *, min_samples: int = 5) -> float | None:
    """Median wall-clock duration of the COMPLETED (rc==0) results, or ``None``.

    ``None`` until at least ``min_samples`` usable durations exist — an unproven queue must
    not drive a keep-the-card decision.
    """
    durs: list[float] = []
    for res in results:
        if res.get("rc", 1) != 0:
            continue
        d = _duration_s(res.get("start_utc"), res.get("end_utc"))
        if d is not None and d >= 0:
            durs.append(d)
    if len(durs) < min_samples:
        return None
    return statistics.median(durs)


def estimate_progress(
    *,
    log_path: str | None,
    progress_regex: str | None,
    elapsed_s: float | None,
    median_s: float | None,
    tail_reader=_read_log_tail,
) -> float:
    """Best-effort [0.0, 1.0] progress via the priority chain (log regex -> median -> 0.0)."""
    p = progress_from_log(log_path, progress_regex, tail_reader=tail_reader)
    if p is not None:
        return p
    if median_s is not None and median_s > 0 and elapsed_s is not None:
        return _clamp01(elapsed_s / median_s)
    return 0.0


# --------------------------- watchdog ---------------------------


class YieldWatchdog:
    """Background monitor that flips policy GPUs to YIELDED when foreigners appear.

    Pure of pool internals: it maintains the marker file and, for ``yield_action == "kill"``,
    calls ``on_yield(gpu)`` (the pool kills our jobs there). Placement (acquire skipping the
    marker) is what enforces "stop claiming".

    The thread is started whenever a policy file exists; :meth:`poll_once` returns before ANY
    detection work while ``yield_to_foreign`` is false (no nvidia-smi call, no /proc scan, no
    marker written). So the switch is hot in BOTH directions: flip the flag in the policy JSON
    on a running pool and yielding begins/stops within one ``yield_poll_s`` — no restart.
    """

    def __init__(
        self,
        root: Path,
        hostname: str,
        *,
        load_policy,
        master,
        on_yield,
        compute_apps_query=query_compute_apps,
        uuid_query=query_gpu_uuids,
        proc_root: str = "/proc",
        environ_reader=None,
        clock=time.time,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.load_policy = load_policy  # () -> GpuPolicy (raises FileNotFoundError if none)
        self.master = master  # has .log(verb, msg)
        self.on_yield = on_yield  # (gpu:int) -> None, only for the "kill" action
        self.compute_apps_query = compute_apps_query
        self.uuid_query = uuid_query
        self.proc_root = proc_root
        self.environ_reader = environ_reader
        self.clock = clock
        self._confirm: dict[int, int] = {}
        self._clear_since: dict[int, float] = {}
        self._smi_fail_logged = False

    def _apply_yield(self, g: int, policy, entry: dict, now: float) -> None:
        """Fire the configured yield action for gpu ``g`` (kill / drain / drain_if_near_done).

        ``kill`` kills every live job on the card (``on_yield(g)``, the historical call).
        ``drain_if_near_done`` hands the pool a :class:`DrainSpec` so it spares near-done jobs
        and kills the rest. ``drain`` never touches running jobs (placement alone enforces it).
        """
        action = policy.yield_action
        if action == "kill":
            self.on_yield(g)
        elif action == "drain_if_near_done":
            started = entry.get("first_seen_ts")
            started = float(started) if started is not None else now
            self.on_yield(
                g,
                DrainSpec(
                    threshold=policy.yield_drain_threshold,
                    max_s=policy.yield_drain_max_s,
                    progress_regex=policy.yield_progress_regex,
                    yield_started_ts=started,
                    now=now,
                ),
            )

    def foreign_count(self, gpu_index: int) -> int | None:
        return foreign_count(
            gpu_index,
            compute_apps_query=self.compute_apps_query,
            uuid_query=self.uuid_query,
            proc_root=self.proc_root,
            environ_reader=self.environ_reader,
        )

    def poll_once(self) -> None:
        """One detection pass: update confirm counters, the marker, and (kill) trigger."""
        try:
            policy = self.load_policy()
        except FileNotFoundError:
            return
        if not policy.yield_to_foreign:
            return
        now = self.clock()
        yielded = read_yielded(self.root, self.hostname)
        changed = False
        for g in policy.gpus:
            fc = self.foreign_count(g)
            if fc is None:  # no smi info -> fail safe, keep our jobs alive; log once
                if not self._smi_fail_logged:
                    self.master.log(
                        "WAIT",
                        f"yield check: nvidia-smi unavailable/unparsable (gpu={g}); not yielding",
                    )
                    self._smi_fail_logged = True
                continue
            self._smi_fail_logged = False
            if fc >= policy.yield_min_foreign_procs:
                self._clear_since.pop(g, None)
                self._confirm[g] = self._confirm.get(g, 0) + 1
                if g in yielded:
                    yielded[g]["last_seen"] = now_iso()
                    yielded[g]["foreign"] = fc
                    changed = True
                    # Defect 2: re-fire on EVERY poll while the card stays yielded, not
                    # only on the transition. Anything that slipped onto the card despite
                    # the marker (the Defect-1 race, a worker already inside acquire when
                    # we yielded, a manual placement) would otherwise run there forever.
                    # The marker is already durable on disk (written on a prior poll), so
                    # the "unacquirable before kill" invariant still holds. yield_kill_gpu
                    # is idempotent — it only marks/kills handles whose process is still
                    # alive and returns at once when there is nothing to kill — so calling
                    # it each poll is cheap and emits a KILL line only for real kills. For
                    # drain_if_near_done the per-poll re-evaluation is the mechanism that
                    # kills a spared job once it overruns yield_drain_max_s.
                    if policy.yield_action in ("kill", "drain_if_near_done"):
                        self._apply_yield(g, policy, yielded[g], now)
                elif self._confirm[g] >= policy.yield_confirm_polls:
                    stamp = now_iso()
                    yielded[g] = {
                        "first_seen": stamp,
                        "last_seen": stamp,
                        "foreign": fc,
                        "first_seen_ts": now,  # epoch anchor for the drain safety cap
                    }
                    changed = True
                    self.master.log("YIELD", f"gpu={g} foreign={fc}")
                    if policy.yield_action in ("kill", "drain_if_near_done"):
                        # Defect 1: persist the marker BEFORE any kill so the invariant
                        # holds — a GPU is unacquirable before any of its jobs are killed.
                        # Each kill frees a slot a worker re-claims immediately; if the
                        # marker were written only at end-of-poll, acquire would read a
                        # card that still looks available and dispatch straight back onto
                        # the one we are vacating. The end-of-poll write below still runs
                        # (changed is True) and harmlessly re-persists the same state.
                        write_yielded(self.root, self.hostname, yielded)
                        self._apply_yield(g, policy, yielded[g], now)
            else:
                self._confirm[g] = 0
                if g in yielded:
                    first = self._clear_since.setdefault(g, now)
                    if now - first >= policy.yield_cooldown_s:
                        del yielded[g]
                        self._clear_since.pop(g, None)
                        changed = True
                        self.master.log("RECLAIM", f"gpu={g}")
                else:
                    self._clear_since.pop(g, None)
        if changed:
            write_yielded(self.root, self.hostname, yielded)

    def _poll_interval(self) -> float:
        try:
            return self.load_policy().yield_poll_s
        except FileNotFoundError:
            return 30.0

    def run(self, stop_event, *, sleep_slice: float = 1.0) -> None:
        """Poll until ``stop_event`` is set; sleeps in slices so a stop is honored promptly."""
        while not stop_event.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 — a watchdog must never crash the pool
                from loguru import logger

                logger.exception("yield watchdog poll failed")
            interval = self._poll_interval()
            waited = 0.0
            while waited < interval and not stop_event.is_set():
                nap = min(sleep_slice, interval - waited)
                time.sleep(nap)
                waited += nap
