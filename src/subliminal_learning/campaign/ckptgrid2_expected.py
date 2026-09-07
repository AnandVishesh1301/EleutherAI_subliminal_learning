"""Authoritative leaf-path enumerator for `20260908_ckptgrid2-bear_160m`.

Completion is measured on disk from this list (I9), not from a YAML
Cartesian product (that would report 6/9 leaves missing forever). Paths
are computed only; the two `campaign.expected` helpers are the only
functions that stat.

`leaf_run_dir` parses both labels and passes `spec_label` into
`teacher_tag`. `spec_label` carries ``@stepN``, so a cross-rung same-base
cell gets the ``_tbase-full-seed0@step2000`` suffix and can never collide
with its own diagonal.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign import expected
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    teacher_checkpoint_dir,
    teacher_tag,
)
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.training.run_set import run_set_subdir


def lora_of(recipe: Recipe) -> LoRAConfig:
    """LoRA config of a student recipe (dropout 0, all-linear).

    Args:
        recipe: Student LoRA rank, alpha, and learning rate.

    Returns:
        The `LoRAConfig` `gkd.layout.gkd_tag` shares with the sweep loader.
    """
    return LoRAConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
    )


def campaign_subdir() -> str:
    """`gkd_students/{RUN_SET_ID}`."""
    return run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)


def _resolved_subdir(subdir: str | None) -> str:
    return campaign_subdir() if subdir is None else subdir


def leaf_run_dir(
    leaf: pairs.Leaf2,
    output_root: Path,
    *,
    subdir: str | None = None,
) -> Path:
    """Treatment leaf dir through the real GKD layout helpers.

    `spec_label` includes ``@revision``, so `teacher_tag` grows a
    ``_tbase-`` suffix on any cross-rung same-base pair and the leaf
    cannot collide with that pair's diagonal.

    Args:
        leaf: Rotated-diagonal student leaf.
        output_root: Artifact root.
        subdir: Tree under `output_root`; `campaign_subdir()` when omitted.

    Returns:
        ``{output_root}/{subdir}/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseedN``.
    """
    student_spec = parse_base_models(leaf.cell.student_label)[0]
    teacher_spec = parse_base_models(leaf.cell.teacher_label)[0]
    ttag = teacher_tag(
        plan.TEACHER_WORKFLOW,
        leaf.cell.teacher_param_tag,
        leaf.ftseed,
        teacher_base_label=spec_label(teacher_spec),
        student_base_label=spec_label(student_spec),
    )
    return gkd_student_run_dir(
        output_root,
        leaf.cell.animal,
        student_spec,
        plan.HUB_ORG,
        ttag,
        leaf.cell.gkd_tag,
        leaf.gseed,
        _resolved_subdir(subdir),
    )


def control_run_dir(
    ctrl: pairs.Control2,
    output_root: Path,
    *,
    subdir: str | None = None,
) -> Path:
    """Base-teacher control leaf dir.

    Args:
        ctrl: Control spec.
        output_root: Artifact root.
        subdir: Tree under `output_root`; `campaign_subdir()` when omitted.

    Returns:
        ``{output_root}/{subdir}/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseedN``.
    """
    student_spec = parse_base_models(ctrl.student_label)[0]
    return gkd_control_run_dir(
        output_root,
        ctrl.animal,
        student_spec,
        plan.HUB_ORG,
        ctrl.gkd_tag,
        ctrl.gseed,
        _resolved_subdir(subdir),
    )


def expected_leaf_dirs(
    output_root: Path,
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Path]:
    """Treatment leaf dirs of `all_cells` in leaf order.

    Args:
        output_root: Artifact root.
        wave: Forwarded to `pairs.all_cells`.
        audit_passed: Forwarded to `pairs.all_cells`.

    Returns:
        One path per rotated-diagonal treatment leaf.
    """
    cells = pairs.all_cells(wave=wave, audit_passed=audit_passed)
    return [leaf_run_dir(leaf, output_root) for leaf in pairs.leaves_of(cells)]


def expected_control_dirs(
    output_root: Path,
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Path]:
    """Control leaf dirs of `controls_of(all_cells(...))`.

    Args:
        output_root: Artifact root.
        wave: Forwarded to `pairs.all_cells`.
        audit_passed: Forwarded to `pairs.all_cells`.

    Returns:
        One path per required (non-staged) control leaf.
    """
    cells = pairs.all_cells(wave=wave, audit_passed=audit_passed)
    return [control_run_dir(ctrl, output_root) for ctrl in pairs.controls_of(cells)]


def expected_config_dirs(
    output_root: Path,
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Path]:
    """Deduped `.parent` of every leaf and control dir (order-preserving).

    Args:
        output_root: Artifact root.
        wave: Forwarded to the leaf/control enumerators.
        audit_passed: Forwarded to the leaf/control enumerators.

    Returns:
        Config dirs the eval sweep writes summaries into.
    """
    dirs = expected_leaf_dirs(
        output_root, wave=wave, audit_passed=audit_passed
    ) + expected_control_dirs(output_root, wave=wave, audit_passed=audit_passed)
    return list(dict.fromkeys(path.parent for path in dirs))


def missing_cells(
    output_root: Path,
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Path]:
    """Treatment + control dirs that fail `cell_is_done`.

    Args:
        output_root: Artifact root.
        wave: Forwarded to the leaf/control enumerators.
        audit_passed: Forwarded to the leaf/control enumerators.

    Returns:
        Dirs still missing a manifest plus complete weights.
    """
    return expected.missing_manifests(
        expected_leaf_dirs(output_root, wave=wave, audit_passed=audit_passed)
        + expected_control_dirs(output_root, wave=wave, audit_passed=audit_passed)
    )


def missing_evals(
    output_root: Path,
    *,
    required_prompt_sets: Sequence[str] = (),
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Path]:
    """Config dirs whose eval-summary triad is incomplete.

    `required_prompt_sets` is forwarded only when
    `expected.verify_summaries_complete` accepts that keyword (a sibling
    is adding it).

    Args:
        output_root: Artifact root.
        required_prompt_sets: Prompt-set names the preference summary
            must cover, when the helper supports the check.
        wave: Forwarded to `expected_config_dirs`.
        audit_passed: Forwarded to `expected_config_dirs`.

    Returns:
        Config dirs still missing summaries (or required prompt sets).
    """
    config_dirs = expected_config_dirs(
        output_root, wave=wave, audit_passed=audit_passed
    )
    verify = expected.verify_summaries_complete
    kwargs: dict[str, object] = {}
    if "required_prompt_sets" in inspect.signature(verify).parameters:
        kwargs["required_prompt_sets"] = required_prompt_sets
    return verify(config_dirs, **kwargs)


def teacher_checkpoint_dirs(output_root: Path) -> list[Path]:
    """`checkpoints/` dirs of every `teacher_leaves()` entry.

    This is what proves a trained teacher exists before its students are
    gated.

    Args:
        output_root: Artifact root.

    Returns:
        One `.../ftseedN/checkpoints` path per new teacher leaf (bear).
    """
    teacher_subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    return [
        teacher_checkpoint_dir(
            output_root,
            teacher_subdir,
            "bear",
            parse_base_models(leaf.combo.label)[0],
            plan.HUB_ORG,
            leaf.combo.param_tag,
            leaf.ftseed,
        )
        for leaf in pairs.teacher_leaves()
    ]
