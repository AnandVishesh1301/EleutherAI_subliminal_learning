"""On-disk paths for the filtration audit. Matches ``gkd.judge_layout``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

AUDIT_SUBDIR = "filtration_audit"

TASKS: tuple[str, ...] = (
    "pool_open",
    "pool_targeted",
    "pool_rejected",
    "rollouts_clean",
    "rollouts_flagged",
    "decoy",
    "recall_probe",
)

OPEN_TASK = "pool_open"


def audit_dir(audit_root: Path, audit_id: str) -> Path:
    """Return ``{audit_root}/filtration_audit/{audit_id}``."""
    return Path(audit_root) / AUDIT_SUBDIR / audit_id


def queue_shard_dir(audit_dir_path: Path, task: str) -> Path:
    """Return the blind queue directory for ``task``."""
    return _task_dir(audit_dir_path / "plan" / "queue", task)


def judgments_shard_dir(audit_dir_path: Path, task: str) -> Path:
    """Return the judgment-output directory for ``task``."""
    return _task_dir(audit_dir_path / "judgments", task)


def logs_dir(audit_dir_path: Path) -> Path:
    """Return the audit logs directory."""
    return Path(audit_dir_path) / "logs"


def list_queue_shards(audit_dir_path: Path, task: str) -> list[Path]:
    """Return sorted ``shard-*.jsonl`` paths for ``task``.

    Args:
        audit_dir_path: Audit run root.
        task: Judge task name.

    Returns:
        Shard paths in filename order.

    Raises:
        FileNotFoundError: If the queue dir or any shard is missing.
        ValueError: If ``task`` is unknown.
    """
    directory = queue_shard_dir(audit_dir_path, task)
    if not directory.is_dir():
        raise FileNotFoundError(f"queue dir missing: {directory}")
    shards = sorted(directory.glob("shard-*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no queue shards in {directory}")
    return shards


def judgment_shard_path(audit_dir_path: Path, task: str, queue_shard: Path) -> Path:
    """Return the sibling judgment path for a queue shard."""
    return judgments_shard_dir(audit_dir_path, task) / queue_shard.name


def partition_shards(
    shards: Sequence[tuple[str, Path]],
    worker_index: int,
    num_workers: int,
) -> list[tuple[str, Path]]:
    """Deal shards round-robin so concurrent workers never share an output file.

    Args:
        shards: ``(task, queue_shard)`` pairs in deterministic order.
        worker_index: This worker's index.
        num_workers: Total workers.

    Returns:
        The subset this worker owns.

    Raises:
        ValueError: If the worker index or count is out of range.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if not 0 <= worker_index < num_workers:
        raise ValueError(
            f"worker_index {worker_index} out of range for {num_workers} workers"
        )
    return list(shards[worker_index::num_workers])


def _task_dir(parent: Path, task: str) -> Path:
    if task not in TASKS:
        raise ValueError(f"unknown judge task {task!r}; known: {list(TASKS)}")
    return Path(parent) / task
