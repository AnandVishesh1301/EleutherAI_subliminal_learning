"""Unit tests for GKD sweep YAML contracts (mega / smoke / multiday / 1day).

Guards the multiday pairing and the cats ~1-day priority pair: shared isolated
subdirs, LoRA vs full-FT mode disjointness, expected treatment counts, LoRA
rank/LR axes, and isolation across smoke / default / multiday / cats_1day
trees. Pure YAML + layout math -- no GPU.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag
from subliminal_learning.teachers.model_specs import parse_base_models

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "configs" / "experiment"
_LORA_MD = _EXP / "gkd_student_sweep_lora_multiday.yaml"
_FULL_MD = _EXP / "gkd_student_sweep_full_multiday.yaml"
_LORA_1D = _EXP / "gkd_student_sweep_lora_1day.yaml"
_FULL_1D = _EXP / "gkd_student_sweep_full_1day.yaml"
_FULL_1D_E1 = _EXP / "gkd_student_sweep_full_1day_e1.yaml"
_SMOKE_CFG = _EXP / "gkd_student_smoke.yaml"
_FULL_SWEEP = _EXP / "gkd_student_sweep.yaml"

_CATS_1DAY_SUBDIR = "gkd_students/cats_1day_run"
_ISOLATED_SUBDIRS = {
    "gkd_students",
    "gkd_students/smoke",
    "gkd_students/multiday_run",
    _CATS_1DAY_SUBDIR,
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _n_treatments(doc: dict) -> int:
    """Treatment-cell count for a GKD sweep YAML (same axes as the runner product)."""
    n_animals = len(doc["animals"])
    n_models = len(parse_base_models(str(doc["base_models"])))
    n_tmodes = len(doc["teacher_modes"])
    grid = doc["grid"]
    n_beta = len(grid["beta"])
    n_lmbda = len(grid["lmbda"])
    n_vmask = len(grid.get("vocab_mask", [False]))
    n_ttemp = len(grid.get("teacher_temp", [1.0]))
    n_tfs = len(doc.get("teacher_ftseeds", [0]))
    n_gseed = len(doc.get("gseeds", [0]))
    slr = doc["student_lr"]
    n_variants = 0
    for mode in grid["student_modes"]:
        key = "lora" if str(mode).startswith("lora") else "full"
        lr_val = slr[key]
        n_variants += len(lr_val) if isinstance(lr_val, (list, tuple)) else 1
    assert n_variants >= len(grid["student_modes"])
    return (
        n_animals
        * n_models
        * n_tmodes
        * n_beta
        * n_lmbda
        * n_variants
        * n_vmask
        * n_ttemp
        * n_tfs
        * n_gseed
    )


def test_multiday_yamls_share_isolated_subdir() -> None:
    lora = _load(_LORA_MD)
    full = _load(_FULL_MD)
    assert lora["gkd_students_subdir"] == "gkd_students/multiday_run"
    assert full["gkd_students_subdir"] == lora["gkd_students_subdir"]
    assert _load(_SMOKE_CFG)["gkd_students_subdir"] == "gkd_students/smoke"
    assert _load(_FULL_SWEEP)["gkd_students_subdir"] == "gkd_students"
    assert _load(_LORA_1D)["gkd_students_subdir"] == _CATS_1DAY_SUBDIR
    assert lora["gkd_students_subdir"] != _load(_LORA_1D)["gkd_students_subdir"]


def test_multiday_lora_and_full_are_mode_disjoint() -> None:
    lora = _load(_LORA_MD)
    full = _load(_FULL_MD)
    assert lora["teacher_modes"] == ["lora"]
    assert full["teacher_modes"] == ["full"]
    assert set(lora["grid"]["student_modes"]) == {
        "lora-r8",
        "lora-r64",
        "lora-r128",
    }
    assert full["grid"]["student_modes"] == ["full"]
    lora_tag = gkd_tag(
        lmbda=1.0,
        beta=1.0,
        student_mode="lora",
        learning_rate=2e-4,
        num_epochs=3,
        lora=LoRAConfig(
            r=64, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear"
        ),
    )
    full_tag = gkd_tag(
        lmbda=1.0,
        beta=1.0,
        student_mode="full",
        learning_rate=2e-5,
        num_epochs=3,
    )
    assert lora_tag != full_tag
    assert "_lora_" in lora_tag and "_r64a64" in lora_tag
    assert "_full_" in full_tag
    assert "r64" not in full_tag


def test_multiday_treatment_cell_counts() -> None:
    # LoRA: 2 animals x 4 models x 1 tmode x 2 β x 1 λ x 9 variants (3 ranks x 3 LRs)
    # x 2 tfs x 2 gseed = 576
    assert _n_treatments(_load(_LORA_MD)) == 576
    # Full: 2 x 4 x 1 x 2 x 1 x 2 LRs x 2 tfs x 2 gseed = 128
    assert _n_treatments(_load(_FULL_MD)) == 128


def test_multiday_keeps_eval_ready_axes() -> None:
    for path in (_LORA_MD, _FULL_MD):
        doc = _load(path)
        assert float(doc["num_epochs"]) == 3.0
        assert doc["animals"] == ["cat", "dog"]
        assert set(doc["grid"]["beta"]) == {0.0, 1.0}
        assert doc["grid"]["lmbda"] == [1.0]
        assert doc["teacher_ftseeds"] == [0, 1]
        assert doc["gseeds"] == [0, 1]
        assert doc["teacher_subdir"] == "sdf_teachers/12h_run"
        specs = {s for s in str(doc["base_models"]).replace(" ", "").split(",")}
        assert specs == {
            "data-seed1",
            "data-seed2",
            "weight-seed1",
            "weight-seed2",
        }


def test_multiday_lora_lrs_and_ranks() -> None:
    lora = _load(_LORA_MD)
    lrs = [float(x) for x in lora["student_lr"]["lora"]]
    assert lrs == [5e-5, 1e-4, 2e-4]
    assert "lora-r8" in lora["grid"]["student_modes"]
    assert "lora-r128" in lora["grid"]["student_modes"]
    full = _load(_FULL_MD)
    assert [float(x) for x in full["student_lr"]["full"]] == [1e-5, 2e-5]


def test_multiday_wandb_filter_logs_canonical_anchors() -> None:
    lora = _load(_LORA_MD)
    full = _load(_FULL_MD)
    assert lora["wandb"]["beta"] == [1.0]
    assert full["wandb"]["beta"] == [1.0]
    assert lora["wandb"]["student_modes"] == ["lora-r64"]
    assert full["wandb"]["student_modes"] == ["full"]
    assert [float(x) for x in lora["wandb"]["learning_rate"]] == [2e-4]
    assert [float(x) for x in full["wandb"]["learning_rate"]] == [2e-5]


def test_cats_1day_wandb_filter_and_e1_disabled() -> None:
    lora = _load(_LORA_1D)
    full = _load(_FULL_1D)
    assert lora["wandb"]["student_modes"] == ["lora-r64"]
    assert [float(x) for x in lora["wandb"]["learning_rate"]] == [2e-4]
    assert full["wandb"]["student_modes"] == ["full"]
    assert [float(x) for x in full["wandb"]["learning_rate"]] == [2e-5]
    assert full["wandb"]["num_epochs"] == [3]
    assert _load(_FULL_1D_E1)["wandb"] is False


def test_mega_sweep_wandb_pins_canonical_lrs() -> None:
    mega = _load(_FULL_SWEEP)
    assert set(mega["wandb"]["student_modes"]) == {"full", "lora-r64"}
    assert {float(x) for x in mega["wandb"]["learning_rate"]} == {2e-5, 2e-4}


def test_cats_1day_yamls_share_isolated_subdir() -> None:
    lora = _load(_LORA_1D)
    full = _load(_FULL_1D)
    full_e1 = _load(_FULL_1D_E1)
    assert lora["gkd_students_subdir"] == _CATS_1DAY_SUBDIR
    assert full["gkd_students_subdir"] == _CATS_1DAY_SUBDIR
    assert full_e1["gkd_students_subdir"] == _CATS_1DAY_SUBDIR
    assert len(_ISOLATED_SUBDIRS) == 4
    assert _CATS_1DAY_SUBDIR not in {
        _load(_LORA_MD)["gkd_students_subdir"],
        _load(_SMOKE_CFG)["gkd_students_subdir"],
        _load(_FULL_SWEEP)["gkd_students_subdir"],
    }


def test_cats_1day_lora_and_full_are_mode_disjoint() -> None:
    lora = _load(_LORA_1D)
    full = _load(_FULL_1D)
    assert lora["teacher_modes"] == ["lora"]
    assert full["teacher_modes"] == ["full"]
    assert set(lora["grid"]["student_modes"]) == {
        "lora-r8",
        "lora-r64",
        "lora-r128",
    }
    assert full["grid"]["student_modes"] == ["full"]
    assert lora["animals"] == ["cat"]
    assert full["animals"] == ["cat"]


def test_cats_1day_treatment_cell_counts_and_epochs() -> None:
    # LoRA: 1 x 2 x 1 x 1 β x 1 λ x 3 ranks x 1 LR x 2 tfs x 2 gseed = 24
    assert _n_treatments(_load(_LORA_1D)) == 24
    # Full e3 / e1: 1 x 2 x 1 x 1 x 1 x 2 LRs x 2 tfs x 2 gseed = 16 each
    assert _n_treatments(_load(_FULL_1D)) == 16
    assert _n_treatments(_load(_FULL_1D_E1)) == 16
    assert float(_load(_LORA_1D)["num_epochs"]) == 3.0
    assert float(_load(_FULL_1D)["num_epochs"]) == 3.0
    assert float(_load(_FULL_1D_E1)["num_epochs"]) == 1.0


def test_cats_1day_keeps_nonnegotiable_axes() -> None:
    for path in (_LORA_1D, _FULL_1D, _FULL_1D_E1):
        doc = _load(path)
        assert doc["animals"] == ["cat"]
        assert doc["grid"]["beta"] == [1.0]
        assert doc["grid"]["lmbda"] == [1.0]
        assert doc["teacher_ftseeds"] == [0, 1]
        assert doc["gseeds"] == [0, 1]
        assert doc["teacher_subdir"] == "sdf_teachers/12h_run"
        specs = {s for s in str(doc["base_models"]).replace(" ", "").split(",")}
        assert specs == {"data-seed1", "weight-seed1"}


def test_cats_1day_lrs_and_e1_e3_tags_disjoint() -> None:
    lora = _load(_LORA_1D)
    assert float(lora["student_lr"]["lora"]) == 2e-4
    full = _load(_FULL_1D)
    assert [float(x) for x in full["student_lr"]["full"]] == [1e-5, 2e-5]
    e3 = gkd_tag(
        lmbda=1.0,
        beta=1.0,
        student_mode="full",
        learning_rate=2e-5,
        num_epochs=3,
    )
    e1 = gkd_tag(
        lmbda=1.0,
        beta=1.0,
        student_mode="full",
        learning_rate=2e-5,
        num_epochs=1,
    )
    assert e3 != e1
    assert "_e3" in e3 and "_e1" in e1
