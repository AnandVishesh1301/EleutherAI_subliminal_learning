"""Unit tests for SDF-teacher output-path layout.

Guards the disjoint `sdf_teachers/` root and the (animal / model / param_tag / ftseed)
nesting -- if these drift, sweep cells collide or clobber the template `teachers/` tree.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.sdf.layout import (
    SDF_TEACHERS_SUBDIR,
    sdf_param_tag,
    sdf_run_label,
    sdf_teacher_param_dir,
    sdf_teacher_run_dir,
)

_TC = TrainConfig(
    learning_rate=5e-5, n_epochs=3, batch_size=8, max_length=2048, max_grad_norm=1.0,
    warmup_ratio=0.03, lr_scheduler_type="cosine", seed=0, optim="adamw_torch",
)
_LC = LoRAConfig(
    r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules=("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
)
_SPEC = PythiaModelSpec(size="160m", seed=1, variant="data", revision=None)
_ORG = "EleutherAI"


def test_run_dir_nesting_and_root() -> None:
    run_dir = sdf_teacher_run_dir(Path("/out"), "cat", _SPEC, _ORG, _TC, _LC)
    assert run_dir == Path(
        "/out/sdf_teachers/cat/pythia-160m-data-seed1/ep3_lr5e-5_r32a32/ftseed0"
    )
    assert SDF_TEACHERS_SUBDIR in run_dir.parts


def test_param_dir_is_parent_of_run_dir() -> None:
    run_dir = sdf_teacher_run_dir(Path("/out"), "cat", _SPEC, _ORG, _TC, _LC)
    param_dir = sdf_teacher_param_dir(Path("/out"), "cat", _SPEC, _ORG, _TC, _LC)
    assert run_dir.parent == param_dir
    assert param_dir.name == "ep3_lr5e-5_r32a32"


def test_seeds_are_siblings_under_param_dir() -> None:
    seed0 = sdf_teacher_run_dir(Path("/out"), "cat", _SPEC, _ORG, _TC, _LC)
    tc1 = TrainConfig(
        learning_rate=5e-5, n_epochs=3, batch_size=8, max_length=2048,
        max_grad_norm=1.0, warmup_ratio=0.03, lr_scheduler_type="cosine", seed=1,
        optim="adamw_torch",
    )
    seed1 = sdf_teacher_run_dir(Path("/out"), "cat", _SPEC, _ORG, tc1, _LC)
    assert seed0.parent == seed1.parent
    assert {seed0.name, seed1.name} == {"ftseed0", "ftseed1"}


def test_run_label_is_flat_and_matches_path_fields() -> None:
    label = sdf_run_label("cat", _SPEC, _ORG, _TC, _LC)
    assert label == "sdf_cat_pythia-160m-data-seed1_ep3_lr5e-5_r32a32_ftseed0"


def test_run_dir_distinct_from_template_teachers_root() -> None:
    run_dir = sdf_teacher_run_dir(Path("/out"), "cat", _SPEC, _ORG, _TC, _LC)
    assert "teachers" not in run_dir.parts  # only "sdf_teachers", never bare "teachers"


# --- Full FT (lora_cfg=None) --------------------------------------------------------
# Additive coverage for the full-FT arm; none of the tests above are modified, so the
# LoRA paths they guard stay byte-identical.


def test_full_ft_param_tag_has_no_lora_suffix() -> None:
    assert sdf_param_tag(_TC, None) == "ep3_lr5e-5_full"


def test_full_ft_run_dir_distinct_from_lora_run_dir() -> None:
    lora_dir = sdf_teacher_run_dir(Path("/out"), "cat", _SPEC, _ORG, _TC, _LC)
    full_dir = sdf_teacher_run_dir(Path("/out"), "cat", _SPEC, _ORG, _TC, None)
    assert lora_dir != full_dir
    assert full_dir == Path(
        "/out/sdf_teachers/cat/pythia-160m-data-seed1/ep3_lr5e-5_full/ftseed0"
    )


def test_full_ft_run_label_has_full_suffix() -> None:
    label = sdf_run_label("cat", _SPEC, _ORG, _TC, None)
    assert label == "sdf_cat_pythia-160m-data-seed1_ep3_lr5e-5_full_ftseed0"
