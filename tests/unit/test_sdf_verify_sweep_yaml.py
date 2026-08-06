"""Unit tests for the SDF-teacher verification-sweep YAML contracts.

Guards the seed3 extension against drift that would corrupt or overwrite existing
seed1/2 evals. Critical invariants only (pure YAML + layout math, no GPU):

- The seed3 verify configs inherit the seed3 training grids, so verification covers
  exactly the trained seed3 cells (scripts.sweeps.sweep_evaluate_sdf_teacher reads
  ``base_models`` / ``grid`` / ``sdf_teachers_subdir`` from the inherited grid).
- Their ``eval`` + ``benchmarks`` blocks are byte-identical to the seed1/2 verify
  configs -- one metric ladder (prompt sets, forced-choice candidates, perplexity,
  LM-Eval subset) across every seed and FT arm, so all downstream comparisons hold.
- LoRA verify inherits a grid WITH ``lora:``; full-FT verify inherits one WITHOUT --
  the mode-disjointness the runner enforces against ``--training-config``.
- seed3 base models are disjoint from the seed1/2 config, and all share the isolated
  ``sdf_teachers/12h_run`` subdir, so no cell collides or clobbers another.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.templates.model_specs import parse_base_models, spec_label

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "configs" / "experiment"

_LORA = _EXP / "sdf_teacher_verify_sweep.yaml"
_FULL = _EXP / "sdf_teacher_verify_sweep_full.yaml"
_LORA_SEED3 = _EXP / "sdf_teacher_verify_sweep_seed3.yaml"
_FULL_SEED3 = _EXP / "sdf_teacher_verify_sweep_full_seed3.yaml"
_LORA_SEED0 = _EXP / "sdf_teacher_verify_sweep_seed0.yaml"
_FULL_SEED0 = _EXP / "sdf_teacher_verify_sweep_full_seed0.yaml"

_RUN_SET_ID = "20260726_crossinit_160m"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _inherited(doc: dict) -> dict:
    return yaml.safe_load((_REPO / doc["inherit_grid"]).read_text())


def _base_labels(doc: dict) -> set[str]:
    grid = _inherited(doc)
    return {spec_label(s) for s in parse_base_models(str(grid["base_models"]))}


def test_seed3_verify_inherits_seed3_training_grids() -> None:
    assert (
        _load(_LORA_SEED3)["inherit_grid"]
        == "configs/experiment/sdf_teacher_sweep_12h_seed3.yaml"
    )
    assert (
        _load(_FULL_SEED3)["inherit_grid"]
        == "configs/experiment/sdf_teacher_sweep_full_12h_seed3.yaml"
    )


def test_seed3_metric_ladder_identical_to_seed12() -> None:
    for seed3, seed12 in ((_LORA_SEED3, _LORA), (_FULL_SEED3, _FULL)):
        d3, d12 = _load(seed3), _load(seed12)
        assert d3["eval"] == d12["eval"]
        assert d3["benchmarks"] == d12["benchmarks"]


def test_seed3_lora_and_full_are_mode_disjoint() -> None:
    assert "lora" in _inherited(_load(_LORA_SEED3))["grid"]
    assert "lora" not in _inherited(_load(_FULL_SEED3))["grid"]


def test_seed3_base_models_disjoint_from_seed12() -> None:
    seed3 = _base_labels(_load(_LORA_SEED3))
    assert seed3 == {"data-seed3", "weight-seed3"}
    assert _base_labels(_load(_FULL_SEED3)) == seed3
    seed12 = _base_labels(_load(_LORA))
    assert seed12 == {"data-seed1", "data-seed2", "weight-seed1", "weight-seed2"}
    assert seed3.isdisjoint(seed12)


def test_all_verify_configs_share_isolated_subdir() -> None:
    for path in (_LORA, _FULL, _LORA_SEED3, _FULL_SEED3):
        grid = _inherited(_load(path))
        assert grid["sdf_teachers_subdir"] == f"{SDF_TEACHERS_SUBDIR}/12h_run"


# --- seed-0 (`original`) arm: the 7th init, trained into the run-set campaign ---------


def test_seed0_verify_inherits_the_seed0_training_grids() -> None:
    assert (
        _load(_LORA_SEED0)["inherit_grid"]
        == "configs/experiment/sdf_teacher_sweep_seed0.yaml"
    )
    assert (
        _load(_FULL_SEED0)["inherit_grid"]
        == "configs/experiment/sdf_teacher_sweep_full_seed0.yaml"
    )
    assert "lora" in _inherited(_load(_LORA_SEED0))["grid"]
    assert "lora" not in _inherited(_load(_FULL_SEED0))["grid"]


def test_seed0_metric_ladder_identical_to_the_six_init_tree() -> None:
    # The seed-0 row only joins the HP tables if it was measured the same way.
    for seed0, existing in ((_LORA_SEED0, _LORA), (_FULL_SEED0, _FULL)):
        d0, d = _load(seed0), _load(existing)
        assert d0["eval"] == d["eval"]
        assert d0["benchmarks"] == d["benchmarks"]


def test_seed0_base_is_the_stock_release_and_disjoint_from_the_ablations() -> None:
    seed0 = _base_labels(_load(_LORA_SEED0))
    assert seed0 == {"full-seed0"}
    assert _base_labels(_load(_FULL_SEED0)) == seed0
    assert seed0.isdisjoint(_base_labels(_load(_LORA)) | _base_labels(_load(_LORA_SEED3)))


def test_seed0_hp_axes_match_the_existing_tags_exactly() -> None:
    # 22 tags x 3 ftseeds x 2 animals = 132 cells, adding an init and nothing else.
    lora, full = _inherited(_load(_LORA_SEED0)), _inherited(_load(_FULL_SEED0))
    ref_lora = yaml.safe_load((_EXP / "sdf_teacher_sweep_12h_seed3.yaml").read_text())
    ref_full = yaml.safe_load(
        (_EXP / "sdf_teacher_sweep_full_12h_seed3.yaml").read_text()
    )
    assert lora["grid"] == ref_lora["grid"]
    assert full["grid"] == ref_full["grid"]
    for grid in (lora, full):
        assert grid["animals"] == ["cat", "dog"]
        assert grid["ft_seeds"] == [0, 1, 2]


def test_seed0_grids_target_the_run_set_campaign_not_a_free_form_batch() -> None:
    for grid in (_inherited(_load(_LORA_SEED0)), _inherited(_load(_FULL_SEED0))):
        assert grid["run_set_id"] == _RUN_SET_ID
        assert "sdf_teachers_subdir" not in grid
