"""Expected on-disk artefacts of the ws160m campaign, per partition.

Run directories are built from the same layout functions the sweeps use, with the
roster's step-bearing labels, so a leaf's path is a property of its cell. Eval
completeness additionally requires the summary to cover every expected gseed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from subliminal_learning.campaign import expected
from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_xbase_control_run_dir,
    teacher_checkpoint_dir,
    teacher_tag,
)
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.run_io import cell_is_done

# Layout depends only on labels; a placeholder root keeps this module path-free.
_LAYOUT_ROOT = Path("/ws160m-base-models")


def spec_for(base_name: str, step: int) -> HfModelSpec:
    """Layout-only spec of a (base, step)."""
    return models.spec(base_name, step, base_models_root=_LAYOUT_ROOT)


def student_subdir(partition: str) -> str:
    """``gkd_students/<run_set_id>`` of a partition."""
    return run_set_subdir(GKD_STUDENTS_SUBDIR, plan.partition(partition).run_set_id)


def teacher_subdir(teacher_step: int) -> str:
    """``sdf_teachers/<run_set_id>`` of the partition owning teachers at `teacher_step`."""
    return run_set_subdir(SDF_TEACHERS_SUBDIR, plan.teacher_run_set_id(teacher_step))


def leaf_teacher_tag(leaf: pairs.Leaf) -> str:
    """Teacher tag of a leaf (carries the teacher label when it differs from the student's)."""
    c = leaf.cell
    return teacher_tag(
        plan.TEACHER_WORKFLOW,
        c.teacher_param_tag,
        leaf.ftseed,
        teacher_base_label=spec_label(spec_for(c.teacher_base, c.teacher_step)),
        student_base_label=spec_label(spec_for(c.student_base, c.student_step)),
    )


def leaf_run_dir(leaf: pairs.Leaf, output_root: Path) -> Path:
    """``{output_root}/{subdir}/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseedN``."""
    c = leaf.cell
    return gkd_student_run_dir(
        Path(output_root),
        c.animal,
        spec_for(c.student_base, c.student_step),
        plan.HUB_ORG,
        leaf_teacher_tag(leaf),
        c.gkd_tag,
        leaf.gseed,
        student_subdir(c.partition.name),
    )


def control_run_dir(ctrl: pairs.Control, output_root: Path) -> Path:
    """``.../controls/base_teacher/{animal}/{model}/{gkd_tag}/gseedN``."""
    return gkd_control_run_dir(
        Path(output_root),
        ctrl.animal,
        spec_for(ctrl.student_base, ctrl.student_step),
        plan.HUB_ORG,
        ctrl.gkd_tag,
        ctrl.gseed,
        student_subdir(ctrl.partition),
    )


def xbase_run_dir(x: pairs.XbaseControl, output_root: Path) -> Path:
    """``.../controls/xbase_teacher/{animal}/{student}/{teacher}/{gkd_tag}/gseedN``."""
    return gkd_xbase_control_run_dir(
        Path(output_root),
        x.animal,
        spec_for(x.student_base, x.student_step),
        plan.HUB_ORG,
        spec_for(x.teacher_base, x.teacher_step),
        x.gkd_tag,
        x.gseed,
        student_subdir(x.partition),
    )


def teacher_checkpoint_dir_of(tl: pairs.TeacherLeaf, output_root: Path) -> Path:
    """``{output_root}/sdf_teachers/<rs>/{animal}/{model}/{param_tag}/ftseedN/checkpoints``."""
    c = tl.combo
    return teacher_checkpoint_dir(
        Path(output_root),
        run_set_subdir(SDF_TEACHERS_SUBDIR, c.owner.run_set_id),
        c.animal,
        spec_for(c.base, c.step),
        plan.HUB_ORG,
        c.param_tag,
        tl.ftseed,
    )


# ---- expected sets ----------------------------------------------------------------


def expected_leaf_dirs(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    return [
        leaf_run_dir(lf, output_root)
        for lf in cgen.active_leaves(pairs.cells_of(partition, wave=wave))
    ]


def expected_control_dirs(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    return [
        control_run_dir(c, output_root) for c in pairs.controls_of(partition, wave=wave)
    ]


def expected_xbase_dirs(partition: str, output_root: Path) -> list[Path]:
    return [xbase_run_dir(x, output_root) for x in pairs.xbase_controls_of(partition)]


def expected_config_gseeds(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> dict[Path, frozenset[int]]:
    """Config dir (parent of the gseed leaves) -> gseeds expected under it."""
    out: dict[Path, set[int]] = {}
    for lf in cgen.active_leaves(pairs.cells_of(partition, wave=wave)):
        out.setdefault(leaf_run_dir(lf, output_root).parent, set()).add(lf.gseed)
    for c in pairs.controls_of(partition, wave=wave):
        out.setdefault(control_run_dir(c, output_root).parent, set()).add(c.gseed)
    for x in pairs.xbase_controls_of(partition):
        out.setdefault(xbase_run_dir(x, output_root).parent, set()).add(x.gseed)
    return {k: frozenset(v) for k, v in out.items()}


def expected_config_dirs(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    return sorted(expected_config_gseeds(partition, output_root, wave=wave))


def owned_teacher_dirs(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    """Checkpoint dirs of the teachers `partition` trains."""
    return [
        teacher_checkpoint_dir_of(t, output_root)
        for t in pairs.teacher_leaves(partition, wave=wave)
    ]


def required_teacher_dirs(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    """Checkpoint dirs of every teacher the partition's cells read (any owner)."""
    return [
        teacher_checkpoint_dir_of(t, output_root)
        for t in pairs.required_teacher_leaves(partition, wave=wave)
    ]


# ---- missing --------------------------------------------------------------------------


def missing_cells(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    """Leaf, control and xbase dirs without a trusted-done manifest."""
    dirs = (
        expected_leaf_dirs(partition, output_root, wave=wave)
        + expected_control_dirs(partition, output_root, wave=wave)
        + expected_xbase_dirs(partition, output_root)
    )
    return expected.missing_manifests(dirs)


def missing_teachers(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    """Owned teacher leaves without a trusted-done manifest."""
    return [
        d
        for d in owned_teacher_dirs(partition, output_root, wave=wave)
        if not cell_is_done(d.parent)
    ]


def missing_required_teachers(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    """Required teacher leaves (any owner) without a trusted-done manifest."""
    return [
        d
        for d in required_teacher_dirs(partition, output_root, wave=wave)
        if not cell_is_done(d.parent)
    ]


def missing_stage_teachers(
    partition: str, stage: str, output_root: Path, *, wave: int
) -> list[pairs.TeacherLeaf]:
    """Teacher leaves (any owner) the students of one stage read that are not done."""
    leaves = [
        lf
        for lf in pairs.leaves_of(pairs.cells_of(partition, wave=wave))
        if pairs.student_stage(lf) == stage
    ]
    return [
        t
        for t in pairs.teacher_leaves_required(leaves)
        if not cell_is_done(teacher_checkpoint_dir_of(t, output_root).parent)
    ]


def reused_teacher_dirs(partition: str, output_root: Path, *, wave: int) -> list[Path]:
    """Checkpoint dirs of earlier-generation teachers a partition reads in place."""
    return [
        teacher_checkpoint_dir_of(t, output_root)
        for t in pairs.foreign_teacher_leaves(partition, wave=wave)
        if t.combo.owner.gen < plan.partition(partition).gen
    ]


def reused_control_dirs(partition: str, output_root: Path, *, wave: int) -> list[Path]:
    """Leaf dirs of earlier-generation `base_teacher` controls a partition reads."""
    out: list[Path] = []
    for key in pairs.foreign_controls_of(partition, wave=wave):
        home = pairs.control_home(key, gen=plan.partition(partition).gen)
        if plan.partition(home).gen < plan.partition(partition).gen:
            out += [
                control_run_dir(pairs.Control(home, *key, g), output_root)
                for g in plan.GSEEDS
            ]
    return out


def missing_reused(partition: str, output_root: Path, *, wave: int) -> list[Path]:
    """Reused earlier-generation leaves that are not complete on disk.

    A teacher needs a trusted-done manifest with its checkpoint; a control needs
    that and its config dir's ``preference_summary.json``.
    """
    teachers = [
        d
        for d in reused_teacher_dirs(partition, output_root, wave=wave)
        if not cell_is_done(d.parent)
    ]
    controls = [
        d
        for d in reused_control_dirs(partition, output_root, wave=wave)
        if not cell_is_done(d) or not (d.parent / "preference_summary.json").is_file()
    ]
    return teachers + controls


def summary_covers(
    config_dir: Path,
    gseeds: frozenset[int],
    *,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
) -> bool:
    """Whether ``preference_summary.json`` under `config_dir` covers every gseed."""
    path = Path(config_dir) / "preference_summary.json"
    try:
        doc = json.loads(path.read_text())
        entries = [
            e for e in doc.get("summaries", []) if e.get("prompt_set") == prompt_set
        ]
        present = {str(k) for e in entries for k in (e.get("per_seed_p_target") or {})}
    except (OSError, ValueError, AttributeError, TypeError):
        return False
    return {str(g) for g in gseeds} <= present


def missing_evals(
    partition: str,
    output_root: Path,
    *,
    wave: int = models.WAVE_NOW,
    required_prompt_sets: Sequence[str] = plan.EVAL_PROMPT_SETS,
) -> list[Path]:
    """Config dirs whose summaries are absent, incomplete, or short of a gseed."""
    by_dir = expected_config_gseeds(partition, output_root, wave=wave)
    incomplete = set(
        expected.verify_summaries_complete(
            sorted(by_dir), required_prompt_sets=required_prompt_sets
        )
    )
    return sorted(
        d for d, gs in by_dir.items() if d in incomplete or not summary_covers(d, gs)
    )


def assert_unique_run_dirs(
    output_root: Path,
    *,
    wave: int = models.WAVE_2,
    partitions: Sequence[str] = plan.PARTITION_ORDER,
) -> int:
    """Raise if any two leaves/controls/xbase leaves of `partitions` share a directory.

    Part II and III partitions are enumerated at their own wave.

    Returns:
        The number of distinct directories.
    """
    seen: dict[Path, str] = {}
    part_waves = [(n, max(wave, plan.partition(n).wave)) for n in partitions]
    for name, wave in part_waves:
        entries = [
            (
                leaf_run_dir(lf, output_root),
                f"{name}:{lf.cell.arm}:{lf.cell.key()}:{lf.ftseed}:{lf.gseed}",
            )
            for lf in pairs.leaves_of(pairs.cells_of(name, wave=wave))
        ]
        entries += [
            (control_run_dir(c, output_root), f"{name}:ctrl:{c.key()}")
            for c in pairs.controls_of(name, wave=wave)
        ]
        entries += [
            (xbase_run_dir(x, output_root), f"{name}:xbase:{x}")
            for x in pairs.xbase_controls_of(name)
        ]
        for d, label in entries:
            if d in seen and seen[d] != label:
                raise ValueError(f"run dir collision {d}: {seen[d]} vs {label}")
            seen[d] = label
    return len(seen)
