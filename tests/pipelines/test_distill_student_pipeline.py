"""Behavioral regressions on `pipelines.distill_student_pipeline`: student identity and the cross-family / cross-base guards.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pytest
from subliminal_learning.pipelines.distill_student_pipeline import (
    DistillStudentConfig,
    _student_run_identity,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec


SPEC = PythiaModelSpec(size="160m", seed=1, variant="data", revision="main")


LORA = LoRAConfig(
    r=8, lora_alpha=8, lora_dropout=0.0,
    target_modules=("query_key_value", "dense"),
)


TRAIN = TrainConfig(
    learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64,
    max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear",
    seed=0, optim="adamw_torch",
)


def paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        output_root=tmp_path / "out", hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "data", wandb_dir=tmp_path / "wandb",
    )


def test_student_identity_changes_with_untagged_training_parameter(tmp_path: Path):
    filtered = tmp_path / "numbers" / "filtered.jsonl"
    filtered.parent.mkdir()
    filtered.write_text("{}\n")
    cfg = DistillStudentConfig(
        paths=paths(tmp_path), hub_org="EleutherAI", spec=SPEC, source="rng",
        number_dataset_dir=filtered.parent, lora=LORA, train=TRAIN,
    )
    a = _student_run_identity(cfg, base_model_repo="EleutherAI/model", filtered_sha256="abc")
    b = _student_run_identity(
        replace(cfg, train=replace(TRAIN, batch_size=11)),
        base_model_repo="EleutherAI/model", filtered_sha256="abc",
    )
    assert a != b


DD_BASE_REF = "allenai/DataDecide-dolma1_7-1B@step69369-seed-default"


def test_cross_family_distillation_is_rejected():
    from subliminal_learning.pipelines.distill_student_pipeline import check_same_family

    with pytest.raises(ValueError, match="cross-FAMILY"):
        check_same_family(DD_BASE_REF, "EleutherAI/pythia-160m")
    with pytest.raises(ValueError, match="cross-FAMILY"):
        check_same_family("EleutherAI/pythia-160m", DD_BASE_REF)


def test_same_family_cross_base_still_passes():
    from subliminal_learning.pipelines.distill_student_pipeline import check_same_family

    check_same_family("EleutherAI/pythia-160m-seed1", "EleutherAI/pythia-160m")
    check_same_family(
        DD_BASE_REF, "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2"
    )
    check_same_family(None, DD_BASE_REF)


def test_allow_cross_family_pair_scoping():
    from subliminal_learning.pipelines.distill_student_pipeline import (
        SUPPORTED_CROSS_FAMILY_PAIRS,
        check_cross_family_supported,
    )

    assert ("datadecide", "pythia") in SUPPORTED_CROSS_FAMILY_PAIRS
    check_cross_family_supported(DD_BASE_REF, "EleutherAI/pythia-160m")
    # both directions are verified id-identical on pool text (2026-09-15)
    assert ("pythia", "datadecide") in SUPPORTED_CROSS_FAMILY_PAIRS
    check_cross_family_supported("EleutherAI/pythia-160m", DD_BASE_REF)
    # an unverified family (here: an external model) is still refused
    with pytest.raises(ValueError, match="-> 'datadecide'"):
        check_cross_family_supported("Qwen/Qwen2.5-1.5B", DD_BASE_REF)


def test_allow_cross_family_requires_allow_cross_base(tmp_path):
    """`allow_cross_family` without `allow_cross_base` is refused before any training."""
    from subliminal_learning.pipelines.distill_student_pipeline import DistillStudentConfig, run

    lora = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
    train = TrainConfig(
        learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64,
        max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear",
        seed=0, optim="adamw_torch",
    )
    paths = PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path,
        dataset_cache_root=tmp_path, wandb_dir=tmp_path,
    )
    ds = tmp_path / "numbers"
    ds.mkdir()
    (ds / "filtered.jsonl").write_text("{}\n")
    cfg = DistillStudentConfig(
        paths=paths, hub_org="EleutherAI", spec=SPEC, source="teacher",
        number_dataset_dir=ds, lora=lora, train=train, allow_cross_family=True,
    )
    with pytest.raises(ValueError, match="implies a cross-base run"):
        run(cfg)
