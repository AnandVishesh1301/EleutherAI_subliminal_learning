"""Cross-rung leaf disjointness through the real GKD path helpers."""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.campaign import ckptgrid2_expected as exp2
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.ckptgrid2_pairs import (
    all_cells,
    dose_cells,
    dose_teacher_cells,
    leaves_of,
)
from subliminal_learning.gkd.layout import teacher_tag
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label


def _teacher_tag_name(run_dir: Path) -> str:
    return run_dir.parent.parent.name


def test_all_treatment_leaf_dirs_are_pairwise_distinct(tmp_path: Path) -> None:
    dirs = [exp2.leaf_run_dir(leaf, tmp_path) for leaf in leaves_of(all_cells())]
    assert len(set(dirs)) == len(dirs)


def test_same_rung_diagonal_has_no_tbase_suffix(tmp_path: Path) -> None:
    cell = next(c for c in dose_cells() if c.student_step == c.teacher_step == 2000)
    leaf = leaves_of((cell,))[0]
    tag = _teacher_tag_name(exp2.leaf_run_dir(leaf, tmp_path))
    assert "_tbase-" not in tag
    student = parse_base_models(cell.student_label)[0]
    teacher = parse_base_models(cell.teacher_label)[0]
    assert spec_label(student) == spec_label(teacher)
    assert spec_label(student) == "full-seed0@step2000"


def test_cross_rung_same_base_has_tbase_revision_suffix(tmp_path: Path) -> None:
    cell = next(
        c
        for c in dose_teacher_cells()
        if c.student_step == 143000 and c.teacher_step == 2000
    )
    leaf = leaves_of((cell,))[0]
    tag = _teacher_tag_name(exp2.leaf_run_dir(leaf, tmp_path))
    assert "_tbase-full-seed0@step2000" in tag
    student = parse_base_models("original@step143000")[0]
    teacher = parse_base_models("original@step2000")[0]
    built = teacher_tag(
        plan.TEACHER_WORKFLOW,
        cell.teacher_param_tag,
        leaf.ftseed,
        teacher_base_label=spec_label(teacher),
        student_base_label=spec_label(student),
    )
    assert built == tag


def test_controls_live_under_base_teacher_and_miss_treatments(
    tmp_path: Path,
) -> None:
    treatments = exp2.expected_leaf_dirs(tmp_path)
    controls = exp2.expected_control_dirs(tmp_path)
    assert controls
    for path in controls:
        parts = path.parts
        assert "controls" in parts
        assert "base_teacher" in parts
    assert set(treatments).isdisjoint(set(controls))
