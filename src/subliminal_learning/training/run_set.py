"""Run-set (campaign) identity + provenance card -- pure domain helpers.

A *run-set* groups the cells of one sweep campaign under a single, date-sortable,
semantic id: ``{YYYYMMDD}_{purpose-slug}_{size}`` (e.g. ``20260724_crossinit_160m``).
It replaces the old free-form batch names (``12h_run``, ``cats_wide_run``) that mixed
wall-clock with ad-hoc descriptors and differed between teacher and student campaigns.

The id is *only* the artifact-subdir component after the workflow root
(``sdf_teachers/{run_set_id}`` / ``gkd_students/{run_set_id}``), so existing trees stay
valid and untouched -- this is the same ``subdir`` override the layouts already accept.
A student campaign points at the teacher campaign it distils from via
``teacher_run_set_id``, so "which teacher batch produced these students" is on disk.

This module is pure (dict/string building); the atomic ``run_set.json`` write, the
``RUNS_INDEX.jsonl`` append, and the campaign staleness check live in `utils.run_io`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

RUN_SET_JSON = "run_set.json"
RUNS_INDEX = "RUNS_INDEX.jsonl"

_RUN_SET_ID_RE = re.compile(r"^\d{8}_[a-z0-9][a-z0-9-]*_[a-z0-9]+$")


def is_valid_run_set_id(run_set_id: str) -> bool:
    """Whether `run_set_id` matches ``{YYYYMMDD}_{purpose-slug}_{size}`` (lower-case)."""
    return bool(_RUN_SET_ID_RE.match(run_set_id))


def run_set_subdir(root_subdir: str, run_set_id: str) -> str:
    """Artifact subdir for a campaign under a workflow root.

    Args:
        root_subdir: Workflow root component (``"sdf_teachers"`` / ``"gkd_students"``).
        run_set_id: The campaign id.

    Returns:
        ``"{root_subdir}/{run_set_id}"``.
    """
    return f"{root_subdir}/{run_set_id}"


def resolve_card_model_size(sweep: Mapping[str, Any], default: str) -> str:
    """The campaign-scope size a launch writes into its run-set card.

    `resolved_config` must match across every launch of one campaign
    (`card_conflict`), so a campaign that spans sizes declares its invariant
    scope once via ``run_set_model_size`` (e.g. ``"mixed"``) while each launch's
    own ``model_size`` stays a merged grid axis.

    Args:
        sweep: The sweep YAML mapping.
        default: Fallback when the sweep declares neither key.

    Returns:
        ``run_set_model_size``, else ``model_size``, else `default`.
    """
    return str(sweep.get("run_set_model_size") or sweep.get("model_size") or default)


def build_run_set_card(
    *,
    run_set_id: str,
    run_type: str,
    purpose: str,
    date: str,
    owner: str,
    git_sha: str | None,
    sweep_yaml: str,
    resolved_config: dict[str, Any],
    grid_axes: dict[str, Any],
    model_size: str,
    teacher_run_set_id: str | None = None,
) -> dict[str, Any]:
    """Self-describing campaign card written to ``{run_set_root}/run_set.json``.

    Args:
        run_set_id: Campaign id.
        run_type: ``"sdf_teacher"`` / ``"gkd_student"`` / verify variants.
        purpose: Free-text campaign purpose (the ``purpose-slug`` expanded).
        date: Launch date (``YYYY-MM-DD``).
        owner: SSD subdir owner (e.g. ``"vishesh"``).
        git_sha: Pinned execution commit (MLflow ``--version`` analog), or ``None``.
        sweep_yaml: Path to the sweep YAML that launched the campaign.
        resolved_config: Campaign-invariant settings every cell must share (corpus,
            packing, model size). Grid axes belong in `grid_axes`, not here: a launch
            whose `resolved_config` differs from the stored card's is refused, so
            putting shard/HP axes here would reject legitimate additive launches.
        grid_axes: This launch's grid axes; merged (unioned) into the stored card.
        model_size: Pythia size scope (e.g. ``"160m"``).
        teacher_run_set_id: For student campaigns, the paired teacher campaign id.

    Returns:
        The card dict (JSON-serialisable), carrying a single-entry ``launches`` log.
    """
    return {
        "run_set_id": run_set_id,
        "run_type": run_type,
        "purpose": purpose,
        "date": date,
        "owner": owner,
        "git_sha": git_sha,
        "sweep_yaml": sweep_yaml,
        "resolved_config": resolved_config,
        "grid_axes": grid_axes,
        "model_size": model_size,
        "teacher_run_set_id": teacher_run_set_id,
        "launches": [
            {
                "date": date,
                "git_sha": git_sha,
                "sweep_yaml": sweep_yaml,
                "grid_axes": grid_axes,
            }
        ],
    }


def _union_axis(existing: Any, new: Any) -> Any:
    """Merge one grid-axis value, preserving order and dropping duplicates."""
    if existing == new:
        return existing
    left = existing if isinstance(existing, list) else [existing]
    right = new if isinstance(new, list) else [new]
    return left + [x for x in right if x not in left]


def merge_run_set_card(
    existing: dict[str, Any], card: dict[str, Any]
) -> dict[str, Any]:
    """Fold an additive launch into the campaign card already on disk.

    One campaign is normally filled by several launches -- a LoRA grid and a full-FT
    grid, or the same grid sharded across GPUs -- which are complementary, not
    conflicting (distinct leaves are guaranteed by `param_tag` + the collision guard +
    `.claim` locks). Incompatible launches are caught upstream by `card_conflict`, so
    merging here only has to accumulate: union the grid axes and append the launch log.
    Campaign-level identity (`date`, `sweep_yaml`, `purpose`) keeps the founding
    launch's values.

    Returns:
        The updated card to write back.
    """
    merged = dict(existing)
    axes = dict(existing.get("grid_axes") or {})
    for key, value in (card.get("grid_axes") or {}).items():
        axes[key] = _union_axis(axes[key], value) if key in axes else value
    merged["grid_axes"] = axes
    launches = list(existing.get("launches") or [])
    for launch in card.get("launches") or []:
        if launch not in launches:
            launches.append(launch)
    merged["launches"] = launches
    if merged.get("teacher_run_set_id") is None:
        merged["teacher_run_set_id"] = card.get("teacher_run_set_id")
    if not merged.get("resolved_config"):  # seed card claimed by its first real launch
        merged["resolved_config"] = card.get("resolved_config")
        merged["git_sha"] = card.get("git_sha")
    return merged


def advance_run_set_sha(
    card: dict[str, Any], *, new_git_sha: str, at: str, reason: str
) -> dict[str, Any]:
    """Re-point a campaign card at a newer **code pin**, keeping the old sha on record.

    The card's ``git_sha`` is an MLflow-style execution pin (``.code_pins/<sha>/``),
    not live workspace HEAD. Advancing it is an intentional re-pin of remaining
    cells — workspace commits/checkouts do not call this. Appends ``sha_advances``
    so the superseded pin stays auditable. Per-cell manifests keep their own sha.

    Args:
        card: The campaign card currently on disk.
        new_git_sha: Commit the campaign should pin from now on.
        at: UTC timestamp for the advance record.
        reason: Why the advance was accepted (operator note or driver rule).

    Returns:
        The updated card. Unchanged when `new_git_sha` already matches.
    """
    if card.get("git_sha") == new_git_sha:
        return card
    advanced = dict(card)
    advanced["git_sha"] = new_git_sha
    advanced["sha_advances"] = list(card.get("sha_advances") or []) + [
        {"from": card.get("git_sha"), "to": new_git_sha, "at": at, "reason": reason}
    ]
    return advanced


def runs_index_row(
    *,
    run_set_id: str,
    run_type: str,
    purpose: str,
    date: str,
    git_sha: str | None,
    status: str,
) -> dict[str, Any]:
    """One ``RUNS_INDEX.jsonl`` row summarising a campaign (one glance per campaign)."""
    return {
        "run_set_id": run_set_id,
        "run_type": run_type,
        "purpose": purpose,
        "date": date,
        "git_sha": git_sha,
        "status": status,
    }


def card_conflict(card: dict[str, Any], existing: dict[str, Any]) -> str | None:
    """Return a human-readable reason a launch conflicts with an existing campaign card.

    A conflict is a same-``run_set_id`` launch whose **code pin** (``git_sha``) OR
    campaign-invariant ``resolved_config`` differs from the stored card — mixing
    incompatible pins/settings into one campaign silently corrupts provenance.
    Workspace HEAD is irrelevant. Differing *grid axes* are not a conflict: those
    merge via `merge_run_set_card`.

    An `existing` card with an empty ``resolved_config`` is a *seed card* -- laid down
    by an adoption/migration that ran no training -- and is compatible with anything;
    the first real launch claims it (`merge_run_set_card` fills in its provenance).

    Returns:
        A one-line reason, or ``None`` if the launch is compatible with `existing`.
    """
    if not existing.get("resolved_config"):
        return None
    if existing.get("git_sha") != card.get("git_sha"):
        return (
            f"git_sha differs (campaign={existing.get('git_sha')}, "
            f"launch={card.get('git_sha')})"
        )
    if existing.get("resolved_config") != card.get("resolved_config"):
        return "resolved_config snapshot differs from the campaign's"
    return None
