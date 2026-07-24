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
        git_sha: Repo commit at launch, or ``None``.
        sweep_yaml: Path to the sweep YAML that launched the campaign.
        resolved_config: Snapshot of the resolved (post-CLI-override) sweep settings.
        grid_axes: The campaign's grid axes (for a one-glance description of contents).
        model_size: Pythia size scope (e.g. ``"160m"``).
        teacher_run_set_id: For student campaigns, the paired teacher campaign id.

    Returns:
        The card dict (JSON-serialisable).
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
    }


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

    A conflict is a same-``run_set_id`` launch whose git commit OR resolved-config
    snapshot differs from what the campaign was started with -- mixing incompatible
    settings into one campaign silently corrupts its provenance.

    Returns:
        A one-line reason, or ``None`` if the launch is compatible with `existing`.
    """
    if existing.get("git_sha") != card.get("git_sha"):
        return (
            f"git_sha differs (campaign={existing.get('git_sha')}, "
            f"launch={card.get('git_sha')})"
        )
    if existing.get("resolved_config") != card.get("resolved_config"):
        return "resolved_config snapshot differs from the campaign's"
    return None
