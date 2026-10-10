"""Shared run-manifest provenance envelope for training pipelines (pure domain).

Both the SDF-teacher and GKD-student pipelines write a per-cell ``train_manifest.json``.
They share a small provenance envelope -- the workflow id, the campaign ``run_set_id``,
the git commit, a creation timestamp, and a schema version -- while each keeps its own
workflow-specific ``body`` (the existing manifest keys, unchanged byte-for-byte). This
module holds the pure dataclass, the byte-identical LoRA hyperparameter blob both
pipelines embed, and the shared ``run_identity`` skip/error gate (`identity_gate_action`)
both pipelines call before trusting a "done" cell. The atomic write lives in the IO layer
(`utils.run_io.write_manifest`).

Keeping the envelope authoritative (it overrides any same-named body key) means every run
records provenance the same way without disturbing the workflow-specific fields that the
result loaders (`evaluation.sdf_results_io`, `evaluation.gkd_results_io`) already parse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from subliminal_learning.config import DEFAULT_ADAM_EPSILON, LoRAConfig, TrainConfig
from subliminal_learning.run_gate import ERROR, RETRAIN, SKIP, GateDecision, identity_diff

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
        **lora_scope_extras(lora),
    }


def lora_scope_extras(lora: LoRAConfig | None) -> dict[str, Any]:
    """``modules_to_save`` / ``layers_to_transform``, each only when set.

    Present only off their empty defaults, so every pre-existing manifest blob and
    ``run_identity`` stays byte-identical.
    """
    if lora is None:
        return {}
    out: dict[str, Any] = {}
    if lora.modules_to_save:
        out["modules_to_save"] = list(lora.modules_to_save)
    if lora.layers_to_transform:
        out["layers_to_transform"] = list(lora.layers_to_transform)
    return out


def train_identity_extras(train: TrainConfig) -> dict[str, Any]:
    """Weight decay, its anchoring, Adam ε and frozen modules, each only off default.

    Merged into both pipelines' ``run_identity`` training block, so pre-existing
    identities (all at the defaults) still SKIP-match.
    """
    out: dict[str, Any] = {}
    if train.weight_decay:
        out["weight_decay"] = train.weight_decay
    if train.weight_decay_anchor:
        out["weight_decay_anchor"] = True
    if train.adam_epsilon != DEFAULT_ADAM_EPSILON:
        out["adam_epsilon"] = train.adam_epsilon
    if train.freeze_modules:
        out["freeze_modules"] = list(train.freeze_modules)
    return out


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


def identity_gate_action(
    stored_manifest: dict[str, Any] | None,
    current_identity: dict[str, Any],
    *,
    force: bool,
    ignore_if_current_missing: tuple[str, ...] = (),
) -> GateDecision:
    """Skip/error/retrain decision for an SDF/GKD cell, keyed on its `run_identity`.

    Path tags (`param_tag` / `teacher_tag` / `gkd_tag` + `ftseed`/`gseed`) disambiguate
    the swept HP axes; this closes the gap on inputs that can drift under an *unchanged*
    path -- an SDF document corpus edited in place, a teacher retrained under the same
    tag, or a prompt pool refreshed under the same tag. On drift: raise instead of
    silently reusing or clobbering a cell whose recorded artifact no longer matches its
    current inputs (mirrors `run_gate.gate_action`'s policy for the template/student
    pipelines -- same skip/error/retrain vocabulary, reused here rather than duplicated).

    Two deliberate differences from `run_gate.gate_action`, both needed for SDF/GKD's
    own cell shape:

    - A stored manifest with **no** `run_identity` key (written before this gate
      existed) is trusted as SKIP, not RETRAIN -- the artifact predates identity
      tracking and (with weights already complete) must not be retrained just to
      satisfy a newly added key.
    - `ignore_if_current_missing` exempts named top-level keys from the drift check
      when the *current* recomputed value is `None` -- e.g. `teacher_manifest_sha256`
      when the upstream teacher checkpoint was later deleted to reclaim shared-SSD
      quota. A large upstream artifact being cleaned up post-training must not block a
      routine resume/skip check on the (unaffected) downstream cell; a genuine content
      change (current value present but different) still errors.

    Args:
        stored_manifest: The cell's on-disk `train_manifest.json`, or `None` if absent.
        current_identity: The freshly computed identity for this run's config/inputs.
        force: Retrain unconditionally (caller's `cfg.force`).
        ignore_if_current_missing: Top-level `current_identity` keys to drop from the
            comparison when their current value is `None`.

    Returns:
        `GateDecision(RETRAIN, {})` if forced or nothing is stored yet; `SKIP` on a
        legacy manifest or exact identity match; `ERROR` with the offending key-diff
        otherwise.
    """
    if force or stored_manifest is None:
        return GateDecision(RETRAIN, {})
    stored_identity = stored_manifest.get("run_identity")
    if not isinstance(stored_identity, dict):
        return GateDecision(SKIP, {})
    exempt = {k for k in ignore_if_current_missing if current_identity.get(k) is None}
    stored_cmp = {k: v for k, v in stored_identity.items() if k not in exempt}
    current_cmp = {k: v for k, v in current_identity.items() if k not in exempt}
    if stored_cmp == current_cmp:
        return GateDecision(SKIP, {})
    return GateDecision(ERROR, identity_diff(stored_cmp, current_cmp))
