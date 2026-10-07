"""Unit tests for the GKD-student verification-sweep YAML contracts.

Critical invariants only (pure YAML + layout math, no GPU):

- Each verify config inherits the matching GKD *training* sweep grid, so verification
  covers exactly the trained ``{gkd_tag}`` configs (the runner reads ``grid`` /
  ``base_models`` / ``num_epochs`` / ``gkd_students_subdir`` from the inherited grid).
- All arms score one identical prompt battery, including the held-out ``test`` set, so
  every cross-arm / cross-init comparison is on the same ruler.
- All arms share the one isolated run subdir, so no cell collides or clobbers another.
- LoRA verify inherits a grid with LoRA student modes; full-FT verify inherits full only.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from subliminal_learning.teachers.model_specs import parse_base_models

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "configs" / "experiment"

_LORA = _EXP / "gkd_student_verify_sweep_lora_1day.yaml"
_FULL = _EXP / "gkd_student_verify_sweep_full_1day.yaml"
_FULL_E1 = _EXP / "gkd_student_verify_sweep_full_1day_e1.yaml"
_ALL = (_LORA, _FULL, _FULL_E1)

_LORA_CW = _EXP / "gkd_student_verify_sweep_lora_catswide.yaml"
_FULL_CW = _EXP / "gkd_student_verify_sweep_full_catswide.yaml"
_LORA_CW_FKL = _EXP / "gkd_student_verify_sweep_lora_catswide_fkl.yaml"
_LORA_CW_CROSS = _EXP / "gkd_student_verify_sweep_lora_catswide_cross.yaml"
_CATSWIDE = (_LORA_CW, _FULL_CW, _LORA_CW_FKL, _LORA_CW_CROSS)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _inherited(doc: dict) -> dict:
    return yaml.safe_load((_REPO / doc["inherit_grid"]).read_text())


def test_verify_configs_inherit_matching_training_grids() -> None:
    assert _load(_LORA)["inherit_grid"].endswith("gkd_student_sweep_lora_1day.yaml")
    assert _load(_FULL)["inherit_grid"].endswith("gkd_student_sweep_full_1day.yaml")
    assert _load(_FULL_E1)["inherit_grid"].endswith(
        "gkd_student_sweep_full_1day_e1.yaml"
    )
    assert _load(_LORA_CW)["inherit_grid"].endswith(
        "gkd_student_sweep_lora_catswide.yaml"
    )
    assert _load(_FULL_CW)["inherit_grid"].endswith(
        "gkd_student_sweep_full_catswide.yaml"
    )
    assert _load(_LORA_CW_FKL)["inherit_grid"].endswith(
        "gkd_student_sweep_lora_catswide_fkl.yaml"
    )
    assert _load(_LORA_CW_CROSS)["inherit_grid"].endswith(
        "gkd_student_sweep_lora_catswide_cross.yaml"
    )


def test_all_arms_score_test_set_and_same_prompt_battery() -> None:
    battery = _load(_LORA)["eval"]["prompt_sets"]
    assert "test" in battery
    for path in (*_ALL, *_CATSWIDE):
        assert _load(path)["eval"]["prompt_sets"] == battery


def test_all_arms_share_one_isolated_run_subdir() -> None:
    subdirs = {_inherited(_load(p))["gkd_students_subdir"] for p in _ALL}
    assert subdirs == {"gkd_students/cats_1day_run"}


def test_catswide_arms_share_one_isolated_run_subdir() -> None:
    subdirs = {_inherited(_load(p))["gkd_students_subdir"] for p in _CATSWIDE}
    assert subdirs == {"gkd_students/cats_wide_run"}


def test_mode_disjoint_student_grids() -> None:
    assert any("lora" in m for m in _inherited(_load(_LORA))["grid"]["student_modes"])
    assert _inherited(_load(_FULL))["grid"]["student_modes"] == ["full"]
    assert _inherited(_load(_FULL_E1))["grid"]["student_modes"] == ["full"]
    assert any(
        "lora" in m for m in _inherited(_load(_LORA_CW))["grid"]["student_modes"]
    )
    assert _inherited(_load(_FULL_CW))["grid"]["student_modes"] == ["full"]


def test_catswide_uses_all_six_bases_and_three_by_three_seeds() -> None:
    for path in _CATSWIDE:
        grid = _inherited(_load(path))
        assert grid["animals"] == ["cat"]
        assert len(parse_base_models(str(grid["base_models"]))) == 6
        assert grid["teacher_ftseeds"] == [0, 1, 2]
        assert grid["gseeds"] == [0, 1, 2]


def test_catswide_fkl_and_cross_pairings() -> None:
    spine = _inherited(_load(_LORA_CW))
    fkl = _inherited(_load(_LORA_CW_FKL))
    cross = _inherited(_load(_LORA_CW_CROSS))
    assert spine.get("teacher_base_pairing", "same") == "same"
    assert fkl["teacher_base_pairing"] == "same"
    assert fkl["grid"]["beta"] == [0.0]
    assert fkl["grid"]["student_modes"] == ["lora-r64"]
    assert cross["teacher_base_pairing"] == "family_swap"
    assert cross["grid"]["beta"] == [1.0]
    assert cross["grid"]["student_modes"] == ["lora-r64"]
