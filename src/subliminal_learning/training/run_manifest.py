"""Shared run-manifest provenance envelope for training pipelines (pure domain).

Both the SDF-teacher and GKD-student pipelines write a per-cell ``train_manifest.json``.
They share a small provenance envelope -- the workflow id, the campaign ``run_set_id``,
the git commit, a creation timestamp, and a schema version -- while each keeps its own
workflow-specific ``body`` (the existing manifest keys, unchanged byte-for-byte). This
module holds the pure dataclass and the byte-identical LoRA hyperparameter blob both
pipelines embed; the atomic write lives in the IO layer (`utils.run_io.write_manifest`).

Keeping the envelope authoritative (it overrides any same-named body key) means every run
records provenance the same way without disturbing the workflow-specific fields that the
result loaders (`evaluation.sdf_results_io`, `evaluation.gkd_results_io`) already parse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from subliminal_learning.config import LoRAConfig

_ENVELOPE_KEYS = (
    "workflow",
    "run_set_id",
    "git_sha",
    "created_at",
    "schema_version",
)

SCHEMA_VERSION = 1


def lora_manifest_blob(lora: LoRAConfig | None) -> dict[str, Any]:
    """LoRA hyperparameter block for a manifest (``None`` fields for full FT).

    Byte-identical to the block both training pipelines previously built inline, so
    existing manifests and the result-loader parsers are unaffected.

    Returns:
        ``{"lora_r", "lora_alpha", "lora_dropout", "target_modules"}`` -- all ``None``
        when `lora` is ``None`` (full fine-tuning).
    """
    if lora is None:
        return {
            "lora_r": None,
            "lora_alpha": None,
            "lora_dropout": None,
            "target_modules": None,
        }
    return {
        "lora_r": lora.r,
        "lora_alpha": lora.lora_alpha,
        "lora_dropout": lora.lora_dropout,
        "target_modules": lora.target_modules
        if isinstance(lora.target_modules, str)
        else list(lora.target_modules),
    }


@dataclass(frozen=True)
class RunManifest:
    """Provenance envelope + workflow-specific body for one training cell.

    Attributes:
        workflow: Short workflow id (``"sdf_teacher"`` / ``"gkd_student"``).
        body: Workflow-specific manifest keys (unchanged from the per-pipeline schema).
        run_set_id: Campaign id (the artifact subdir after the workflow root), or ``None``.
        git_sha: Repo commit at write time, or ``None`` if git is unavailable.
        created_at: ISO-8601 UTC timestamp of the write, or ``None``.
        schema_version: Envelope schema version.
    """

    workflow: str
    body: dict[str, Any] = field(default_factory=dict)
    run_set_id: str | None = None
    git_sha: str | None = None
    created_at: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Merge body then envelope, so envelope keys are authoritative on collision."""
        envelope = {
            "workflow": self.workflow,
            "run_set_id": self.run_set_id,
            "git_sha": self.git_sha,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        return {**self.body, **envelope}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Inverse of `to_dict`: split the envelope keys back out from the body."""
        body = {k: v for k, v in data.items() if k not in _ENVELOPE_KEYS}
        return cls(
            workflow=str(data.get("workflow", "")),
            body=body,
            run_set_id=data.get("run_set_id"),
            git_sha=data.get("git_sha"),
            created_at=data.get("created_at"),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )
