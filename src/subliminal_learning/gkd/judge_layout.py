"""Output-path layout for the offline filtration audit (LLM-as-a-judge).

    {audit_root}/filtration_audit/{audit_id}/
        plan/queue/{task}/shard-NNNN.jsonl     blind judge input
        plan/keys/{task}/shard-NNNN.jsonl      identity/stratum keys (judge never reads)
        plan/leaf_samples/{leaf_hash}.jsonl    per-leaf resume artifact
        plan/plan_manifest.json
        pool_restream/{rejected_with_text.jsonl,restream_verification.json}
        judgments/{task}/shard-NNNN.jsonl
        logs/
        results/
        audit_manifest.json

The queue/keys split is what makes judging blind: `queue` carries only the text and
the animal being asked about, `keys` carries arm, `flagged`, `matched_form`, `run_dir`
and stratum weights. Sharding (rather than one multi-GB file) bounds the judge's
resume skip-set to one shard and lets tasks be judged independently.
"""

from __future__ import annotations

from pathlib import Path

AUDIT_SUBDIR = "filtration_audit"

# Judge tasks. `pool_*` are census passes over the screened Pile pool; `rollouts_*`
# are the student-completion sample; `decoy` / `recall_probe` are judge validation.
TASKS: tuple[str, ...] = (
    "pool_open",
    "pool_targeted",
    "pool_rejected",
    "rollouts_clean",
    "rollouts_flagged",
    "decoy",
    "recall_probe",
)

ROLLOUT_TASKS: tuple[str, ...] = ("rollouts_clean", "rollouts_flagged")


def audit_dir(audit_root: Path, audit_id: str) -> Path:
    """Root dir for one audit run."""
    return Path(audit_root) / AUDIT_SUBDIR / audit_id


def queue_shard_dir(audit_dir_path: Path, task: str) -> Path:
    """Blind judge-input shards for `task`."""
    return _task_dir(audit_dir_path / "plan" / "queue", task)


def keys_shard_dir(audit_dir_path: Path, task: str) -> Path:
    """Identity/stratum key shards for `task` (never handed to the judge)."""
    return _task_dir(audit_dir_path / "plan" / "keys", task)


def judgments_shard_dir(audit_dir_path: Path, task: str) -> Path:
    """Judge-output shards for `task`, one per queue shard."""
    return _task_dir(audit_dir_path / "judgments", task)


def shard_name(shard_index: int) -> str:
    """Shard filename, e.g. ``shard-0007.jsonl``."""
    return f"shard-{shard_index:04d}.jsonl"


def leaf_samples_dir(audit_dir_path: Path) -> Path:
    """Per-leaf plan-builder resume artifacts."""
    return Path(audit_dir_path) / "plan" / "leaf_samples"


def plan_manifest_path(audit_dir_path: Path) -> Path:
    """Sampling-plan manifest (seed, git sha, leaf list, realised counts)."""
    return Path(audit_dir_path) / "plan" / "plan_manifest.json"


def pool_restream_dir(audit_dir_path: Path) -> Path:
    """Layer-1 rejected-doc re-stream outputs."""
    return Path(audit_dir_path) / "pool_restream"


def results_dir(audit_dir_path: Path) -> Path:
    """Paper-facing CSV/JSON metrics."""
    return Path(audit_dir_path) / "results"


def logs_dir(audit_dir_path: Path) -> Path:
    """Throughput probe and run logs."""
    return Path(audit_dir_path) / "logs"


def audit_manifest_path(audit_dir_path: Path) -> Path:
    """Top-level audit provenance manifest."""
    return Path(audit_dir_path) / "audit_manifest.json"


def _task_dir(parent: Path, task: str) -> Path:
    if task not in TASKS:
        raise ValueError(f"unknown judge task {task!r}; known: {list(TASKS)}")
    return Path(parent) / task
