"""Process, GPU, lock and coverage IO for the strictscreen judge audit driver.

Judge tool commands always run as ``uv run --no-sync --project
<pin>/tools/rollout_judge python -m rollout_judge.cli ...`` with an explicit
``--config`` so the tool never resolves its default YAML.
"""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import control_io
from subliminal_learning.campaign import strictscreen_audit_plan as sap
from subliminal_learning.gkd import judge_layout
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import sha256_file

_CLUSTER_LOCK = ("locks", "judge-cluster.lock")
_JUDGE_CLI = "rollout_judge.cli"
_GIB = 1024**3


@dataclass(frozen=True)
class TaskCoverage:
    """Row_id coverage of one task's queue shards by its judgment shards.

    Attributes:
        task: Judge task.
        n_queue: Queue rows.
        n_judged: Queue rows whose ``row_id`` has a judgment.
        incomplete_shards: Shard name -> missing row count.
    """

    task: str
    n_queue: int
    n_judged: int
    incomplete_shards: Mapping[str, int]

    @property
    def complete(self) -> bool:
        """Every queue row is judged."""
        return not self.incomplete_shards


def tool_cmd(pin_root: Path, *args: str) -> list[str]:
    """``uv run --no-sync --project <pin>/tools/rollout_judge python -m ... <args>``."""
    return [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(Path(pin_root) / sap.JUDGE_TOOL_REL),
        "python",
        "-m",
        _JUDGE_CLI,
        *args,
    ]


def tool_python_cmd(pin_root: Path, code: str) -> list[str]:
    """Run ``python -c <code>`` in the judge venv without syncing it."""
    return [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(Path(pin_root) / sap.JUDGE_TOOL_REL),
        "python",
        "-c",
        code,
    ]


def repo_cmd(module: str, *args: str) -> list[str]:
    """``uv run --no-sync python -m <module> <args>`` in the main project."""
    return ["uv", "run", "--no-sync", "python", "-m", module, *args]


def tool_env(hf_cache_root: Path, cuda_visible_devices: str | None) -> dict[str, str]:
    """Offline HF env for the judge venv; the main project's venv is unset.

    Args:
        hf_cache_root: Judge YAML ``hf_cache_root``.
        cuda_visible_devices: Device list, or ``None`` to unset it.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES")
    }
    env.update(
        HF_HOME=str(hf_cache_root),
        HF_HUB_CACHE=str(hf_cache_root),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def run_blocking(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    log_path: Path,
) -> int:
    """Run `cmd` to completion, appending stdout/stderr to `log_path`.

    Returns:
        The exit code.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("run {} (cwd={}) -> {}", shlex.join(cmd), cwd, log_path)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def capture(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None) -> str:
    """Stdout of `cmd`.

    Raises:
        subprocess.CalledProcessError: Non-zero exit.
    """
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        env=None if env is None else dict(env),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def launch_detached(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    exit_path: Path,
) -> int:
    """Start `cmd` in its own session; its exit code lands in `exit_path`.

    Returns:
        The wrapper shell's pid.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = exit_path.with_name(f"{exit_path.name}.tmp")
    script = (
        f"{shlex.join(cmd)}; rc=$?; echo $rc > {shlex.quote(str(tmp))} "
        f"&& mv {shlex.quote(str(tmp))} {shlex.quote(str(exit_path))}"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            ["bash", "-c", script],
            cwd=cwd,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    logger.info("launched pid {}: {} -> {}", proc.pid, shlex.join(cmd), log_path)
    return int(proc.pid)


def read_exit_code(exit_path: Path) -> int | None:
    """Exit code a detached launch recorded, or ``None`` while absent."""
    try:
        return int(exit_path.read_text().strip())
    except (OSError, ValueError):
        return None


def cluster_lock_path(audit_dir: Path) -> Path:
    """``logs/locks/judge-cluster.lock`` of an audit."""
    return judge_layout.logs_dir(audit_dir).joinpath(*_CLUSTER_LOCK)


def cluster_lock_held(audit_dir: Path) -> bool:
    """Whether a live ``judge-cluster`` holds the audit's cluster lock."""
    path = cluster_lock_path(audit_dir)
    if not path.is_file():
        return False
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def judge_processes(
    config_path: Path, *, proc_root: Path = Path("/proc")
) -> list[dict]:
    """Live ``rollout_judge.cli`` processes started with `config_path`.

    Args:
        config_path: Judge YAML passed via ``--config``.
        proc_root: procfs mount.

    Returns:
        ``{"pid", "cmdline"}`` rows, excluding this process.
    """
    needle = str(config_path)
    out = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        words = [a.decode(errors="replace") for a in argv if a]
        if _JUDGE_CLI in words and needle in words:
            out.append({"pid": int(entry.name), "cmdline": " ".join(words)})
    return out


def gpu_memory_used_mib() -> dict[int, int]:
    """``index -> memory.used`` MiB from ``nvidia-smi``.

    Raises:
        subprocess.CalledProcessError: ``nvidia-smi`` failed.
    """
    text = capture(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path("/"),
        env=None,
    )
    out = {}
    for line in text.strip().splitlines():
        index, used = (part.strip() for part in line.split(","))
        out[int(index)] = int(float(used))
    return out


def gpu_compute_apps() -> list[str]:
    """Raw ``nvidia-smi --query-compute-apps`` rows (pids do not map in pods).

    Raises:
        subprocess.CalledProcessError: ``nvidia-smi`` failed.
    """
    text = capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path("/"),
        env=None,
    )
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def host_running_jobs(
    queue_root: Path,
    host: str,
    queue_names: Collection[str],
) -> list[dict]:
    """Running jobq claims on `host` across `queue_names` (read-only)."""
    out = []
    for name in queue_names:
        if not store.queue_exists(queue_root, name):
            continue
        for row in store.queue_status(queue_root, name)["running_jobs"]:
            if row.get("node") == host:
                out.append({"queue": name, **row})
    return out


def worker_pool_alive(queue_root: Path) -> bool:
    """Whether this host's jobq worker pool pid is alive."""
    return bool(control_io.worker_pool(queue_root)["alive"])


def judged_rows(audit_dir: Path, task: str, shards: Sequence[str]) -> int:
    """Judgment lines across `shards` (cheap progress count)."""
    directory = judge_layout.judgments_shard_dir(audit_dir, task)
    total = 0
    for name in shards:
        path = directory / name
        if path.is_file():
            with path.open("rb") as handle:
                total += sum(1 for _ in handle)
    return total


def task_coverage(audit_dir: Path, task: str, shards: Sequence[str]) -> TaskCoverage:
    """Row_id coverage of every committed queue shard of `task`.

    Raises:
        FileNotFoundError: A committed queue shard is missing.
    """
    queue_dir = judge_layout.queue_shard_dir(audit_dir, task)
    judgment_dir = judge_layout.judgments_shard_dir(audit_dir, task)
    n_queue = 0
    n_judged = 0
    incomplete: dict[str, int] = {}
    for name in shards:
        wanted = _row_ids(queue_dir / name, tolerate_torn_tail=False)
        done = (
            _row_ids(judgment_dir / name, tolerate_torn_tail=True)
            if (judgment_dir / name).is_file()
            else set()
        )
        missing = len(wanted - done)
        n_queue += len(wanted)
        n_judged += len(wanted) - missing
        if missing:
            incomplete[name] = missing
    return TaskCoverage(task, n_queue, n_judged, incomplete)


def verify_source_hashes(
    pin_root: Path, pinned: Sequence[tuple[str, str]]
) -> list[str]:
    """Mismatch messages for pinned repo-relative file hashes."""
    out = []
    for rel, want in pinned:
        got = sha256_file(Path(pin_root) / rel)
        if got != want:
            out.append(f"{rel}: sha256 {got} != pinned {want}")
    return out


def snapshot_dir(hf_cache_root: Path, model_id: str, revision: str) -> Path:
    """``{hf_cache_root}/models--Org--Name/snapshots/<revision>``."""
    return (
        Path(hf_cache_root)
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def free_gib(path: Path) -> float:
    """Free space on the filesystem holding `path` (nearest existing parent)."""
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / _GIB


def read_json(path: Path) -> dict:
    """A JSON object from `path`.

    Raises:
        ValueError: Not a JSON object.
    """
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return doc


def write_json(path: Path, payload: object) -> None:
    """Atomic JSON write (parents created)."""
    atomic_write_json(path, payload)


def _row_ids(path: Path, *, tolerate_torn_tail: bool) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                ids.add(str(json.loads(raw)["row_id"]))
            except json.JSONDecodeError:
                if tolerate_torn_tail and not raw.endswith("\n"):
                    continue
                raise
    return ids
