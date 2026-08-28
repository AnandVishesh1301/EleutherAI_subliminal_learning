"""Cross-rung leaf disjointness through the real GKD path-construction functions.

`sweep_distill_student_gkd` and `distill_student_gkd_pipeline` pass
`spec_label(spec)` into `gkd.layout.teacher_tag`. For the original base that
label is ``full-seed0@stepN``, not ``original@stepN``. Revisions are not
stripped: same-base different-rung labels differ, so the tag grows a
``_tbase-`` suffix and the leaf paths stay disjoint.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.campaign.ckptgrid_plan import (
    HUB_ORG,
    STUDENT_GKD_TAG,
    TEACHER_PARAM_TAG_LORA,
    TEACHER_WORKFLOW,
)
from subliminal_learning.gkd.layout import gkd_student_run_dir, teacher_tag
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label


def _tag(student_token: str, teacher_token: str) -> str:
    student = parse_base_models(student_token)[0]
    teacher = parse_base_models(teacher_token)[0]
    return teacher_tag(
        TEACHER_WORKFLOW,
        TEACHER_PARAM_TAG_LORA,
        0,
        teacher_base_label=spec_label(teacher),
        student_base_label=spec_label(student),
    )


def test_cross_rung_same_base_gets_tbase_suffix() -> None:
    tag = _tag("original@step128", "original@step48000")
    assert "_tbase-" in tag
    assert "step48000" in tag
    # Real path uses spec_label, not the campaign token.
    assert "full-seed0@step48000" in tag
    assert "original@step48000" not in tag


def test_same_rung_same_base_has_no_tbase_suffix() -> None:
    tag = _tag("original@step48000", "original@step48000")
    assert "_tbase-" not in tag
    assert spec_label(parse_base_models("original@step48000")[0]) == (
        "full-seed0@step48000"
    )


def test_two_teacher_rungs_at_same_student_rung_are_disjoint(
    tmp_path: Path,
) -> None:
    student = parse_base_models("original@step128")[0]
    tag_a = _tag("original@step128", "original@step48000")
    tag_b = _tag("original@step128", "original@step64000")
    assert tag_a != tag_b
    path_a = gkd_student_run_dir(
        tmp_path, "bear", student, HUB_ORG, tag_a, STUDENT_GKD_TAG, 0
    )
    path_b = gkd_student_run_dir(
        tmp_path, "bear", student, HUB_ORG, tag_b, STUDENT_GKD_TAG, 0
    )
    assert path_a != path_b
    # ``.../{teacher_tag}/{gkd_tag}/gseedN`` — parent.parent is the tag.
    assert path_a.parent.parent.name != path_b.parent.parent.name
