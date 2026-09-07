"""Default-inert shared edits used by the ckptgrid2 campaign."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.evaluate_sdf_teacher_pipeline import build_sdf_prompt_sets
from scripts.sweeps.sweep_evaluate_gkd_student import _cell_done
from subliminal_learning.campaign.ckptgrid_metrics import (
    GRID_DELTA_COLUMNS,
    build_grid_delta,
)
from subliminal_learning.campaign.expected import verify_summaries_complete
from subliminal_learning.evaluation import prompt_sets as angela_sets
from subliminal_learning.evaluation import sdf_prompt_sets as sdf_sets

_SET = "test"
_SCORE = "prob"
_TEST_JSON = Path("data/evaluation_prompts/animal_preference_test.json")


def _student_row(
    *,
    arm: str = "grid",
    student_base: str = "original",
    teacher_base: str | None = "original",
    student_step: int = 100,
    teacher_step: int | None = 100,
    cell: str | None = "SI_SS",
    p_target: float = 0.0,
    is_control: bool = False,
    ftseed: int = 0,
    gseed: int = 0,
    gkd_tag: str | None = None,
    teacher_param_tag: str | None = None,
) -> dict:
    row = {
        "arm": arm,
        "student_base": student_base,
        "teacher_base": teacher_base,
        "student_step": student_step,
        "teacher_step": teacher_step,
        "cell": cell,
        "ftseed": ftseed,
        "gseed": gseed,
        "set": _SET,
        "scoring": _SCORE,
        "p_target": p_target,
        "is_control": is_control,
    }
    if gkd_tag is not None:
        row["gkd_tag"] = gkd_tag
    if teacher_param_tag is not None:
        row["teacher_param_tag"] = teacher_param_tag
    return row


def _planted_student_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _student_row(student_step=100, p_target=0.10, is_control=True, gseed=0),
            _student_row(student_step=100, p_target=0.12, is_control=True, gseed=1),
            _student_row(
                student_step=100,
                teacher_step=200,
                p_target=0.41,
                ftseed=0,
                gseed=0,
            ),
        ]
    )


def _write_triad(config_dir: Path, prompt_sets: tuple[str, ...]) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "preference_summary.json").write_text(
        json.dumps({"summaries": [{"prompt_set": name} for name in prompt_sets]})
    )
    (config_dir / "sdf_forced_choice_summary.json").write_text("{}")
    (config_dir / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {}})
    )


def test_build_grid_delta_default_matches_frozen_frame() -> None:
    got = build_grid_delta(_planted_student_frame())
    # Column-wise so agg ftseed/gseed stay None (object), not float NaN.
    expected = pd.DataFrame(
        {
            "level": ["leaf", "agg"],
            "arm": ["grid", "grid"],
            "student_base": ["original", "original"],
            "teacher_base": ["original", "original"],
            "student_step": [100, 100],
            "teacher_step": [200, 200],
            "cell": ["SI_SS", "SI_SS"],
            "ftseed": pd.Series([0, None], dtype=object),
            "gseed": pd.Series([0, None], dtype=object),
            "set": [_SET, _SET],
            "scoring": [_SCORE, _SCORE],
            "p_target": [0.41, 0.41],
            "control_mean": [0.11, 0.11],
            "delta": [0.30, 0.30],
            "n_leaves": [1, 1],
        },
        columns=list(GRID_DELTA_COLUMNS),
    )
    assert list(got.columns) == list(GRID_DELTA_COLUMNS)
    assert len(got) == len(expected)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        expected,
        check_dtype=False,
        atol=1e-12,
    )


def test_extra_control_keys_must_be_subset_of_extra_agg_keys() -> None:
    with pytest.raises(ValueError, match="extra_control_keys"):
        build_grid_delta(
            _planted_student_frame(),
            extra_agg_keys=("gkd_tag",),
            extra_control_keys=("source_run_set",),
        )


def test_extra_keys_join_control_mean_per_gkd_tag() -> None:
    rows = [
        _student_row(
            p_target=0.10,
            is_control=True,
            gkd_tag="tag_a",
            teacher_param_tag="ep5",
        ),
        _student_row(
            p_target=0.20,
            is_control=True,
            gkd_tag="tag_b",
            teacher_param_tag="ep5",
        ),
        _student_row(
            p_target=0.40,
            gkd_tag="tag_a",
            teacher_param_tag="ep5",
        ),
        _student_row(
            p_target=0.40,
            gkd_tag="tag_b",
            teacher_param_tag="ep5",
        ),
    ]
    frame = build_grid_delta(
        pd.DataFrame(rows),
        extra_agg_keys=("gkd_tag", "teacher_param_tag"),
        extra_control_keys=("gkd_tag",),
    )
    assert list(frame.columns) == [
        *GRID_DELTA_COLUMNS,
        "gkd_tag",
        "teacher_param_tag",
    ]
    leaf = frame[frame["level"] == "leaf"]
    agg = frame[frame["level"] == "agg"]
    assert len(leaf) == 2
    assert len(agg) == 2
    assert not frame.duplicated().any()
    by_tag = leaf.set_index("gkd_tag")
    assert by_tag.loc["tag_a", "control_mean"] == pytest.approx(0.10)
    assert by_tag.loc["tag_a", "delta"] == pytest.approx(0.30)
    assert by_tag.loc["tag_b", "control_mean"] == pytest.approx(0.20)
    assert by_tag.loc["tag_b", "delta"] == pytest.approx(0.20)
    agg_by_tag = agg.set_index("gkd_tag")
    assert set(agg_by_tag.index) == {"tag_a", "tag_b"}
    assert agg_by_tag.loc["tag_a", "delta"] == pytest.approx(0.30)
    assert agg_by_tag.loc["tag_b", "delta"] == pytest.approx(0.20)
    assert int(agg_by_tag.loc["tag_a", "n_leaves"]) == 1
    assert int(agg_by_tag.loc["tag_b", "n_leaves"]) == 1


def test_missing_extra_key_column_raises() -> None:
    with pytest.raises(ValueError, match="gkd_tag"):
        build_grid_delta(_planted_student_frame(), extra_agg_keys=("gkd_tag",))


def test_test_sample_resolves_through_both_registries() -> None:
    assert angela_sets.prompt_set_path("test_sample") == _TEST_JSON
    assert angela_sets.default_scoring("test_sample") == "sample"
    assert sdf_sets.prompt_set_path("test_sample") == _TEST_JSON
    assert sdf_sets.default_scoring("test_sample") == "sample"
    built = build_sdf_prompt_sets(("test_sample",), "prob", 128, 24)
    assert len(built) == 1
    assert built[0].max_new_tokens == 24
    assert built[0].scoring == "sample"


def test_verify_summaries_required_prompt_sets(tmp_path: Path) -> None:
    cdir = tmp_path / "cfg"
    _write_triad(cdir, ("test",))
    assert verify_summaries_complete([cdir], required_prompt_sets=()) == []
    assert verify_summaries_complete([cdir], required_prompt_sets=("test_sample",)) == [
        cdir
    ]
    _write_triad(cdir, ("test", "test_sample"))
    assert (
        verify_summaries_complete([cdir], required_prompt_sets=("test_sample",)) == []
    )


def test_cell_done_required_prompt_sets(tmp_path: Path) -> None:
    cdir = tmp_path / "cfg"
    _write_triad(cdir, ("test",))
    assert _cell_done(cdir, False, required_prompt_sets=()) is True
    assert not _cell_done(cdir, False, required_prompt_sets=("test_sample",))
    _write_triad(cdir, ("test", "test_sample"))
    assert _cell_done(cdir, False, required_prompt_sets=("test_sample",))
