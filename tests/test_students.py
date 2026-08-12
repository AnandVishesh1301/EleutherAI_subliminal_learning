"""Unit tests for the student-distillation pure logic (no GPU/model)."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.distill_student_pipeline import DistillStudentConfig, run
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.numbers.filtering import NumberFilterConfig, filter_raw_rows
from subliminal_learning.students.config import load_student_training
from subliminal_learning.students.dataset_io import subsample_rows
from subliminal_learning.students.layout import (
    student_param_dir,
    student_param_tag,
    student_run_dir,
)

_TC = TrainConfig(
    learning_rate=1e-4, n_epochs=5, batch_size=8, max_length=256, max_grad_norm=1.0,
    warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1, optim="adamw_torch",
)
_LC = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")
_SPEC = PythiaModelSpec(size="160m", seed=1, variant="data")


def test_resolve_target_modules_sentinel_passes_through():
    assert _LC.resolve_target_modules() == "all-linear"


def test_resolve_target_modules_tuple_becomes_list():
    lc = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0,
                    target_modules=("query_key_value", "dense"))
    assert lc.resolve_target_modules() == ["query_key_value", "dense"]


def test_student_param_tag():
    assert student_param_tag(_TC, _LC) == "ep5_lr1e-4_r8a32"


def test_student_param_tag_default_optim_is_byte_identical():
    # Explicit default optim + zero momentum must not alter the pre-optimizer-axis tag.
    tc = replace(_TC, optim="adamw_torch", optim_momentum=0.0)
    assert student_param_tag(tc, _LC) == "ep5_lr1e-4_r8a32"


def test_student_param_tag_sgd_gets_suffix():
    tc = replace(_TC, optim="sgd")
    assert student_param_tag(tc, _LC) == "ep5_lr1e-4_r8a32_sgd"


def test_student_param_dir_and_run_dir_carry_optim_suffix():
    tc = replace(_TC, optim="sgd")
    common = dict(
        source="teacher", spec=_SPEC, hub_org="EleutherAI", lora_cfg=_LC, animal="dog",
        teacher_param_tag="ep5_lr2e-4_r32a32", teacher_seed=0,
        number_dataset_tag="ex10-20_full10",
    )
    pd = student_param_dir(Path("/OUT"), train_cfg=tc, **common)
    # The whole tree segregates under a `students/sgd/...` level and the leaf keeps `_sgd`.
    assert pd == Path(
        "/OUT/students/sgd/dog/pythia-160m-data-seed1/teacher-pythia-160m-data-seed1/"
        "ep5_lr2e-4_r32a32_tseed0/ex10-20_full10/ep5_lr1e-4_r8a32_sgd"
    )
    rd = student_run_dir(Path("/OUT"), train_cfg=tc, **common)
    assert rd == pd / "ftseed1" and rd.parent == pd
    # And the sgd branch is disjoint from the default-optim branch.
    assert student_param_dir(Path("/OUT"), train_cfg=_TC, **common) != pd


def test_student_param_dir_sgd_controls_gain_optim_level():
    tc = replace(_TC, optim="sgd")
    base_pd = student_param_dir(
        Path("/OUT"), source="base", spec=_SPEC, hub_org="EleutherAI",
        train_cfg=tc, lora_cfg=_LC, number_dataset_tag="ex10-20_full10",
    )
    assert base_pd == Path(
        "/OUT/students/sgd/controls/base/pythia-160m-data-seed1/"
        "teacher-pythia-160m-data-seed1/ex10-20_full10/ep5_lr1e-4_r8a32_sgd"
    )
    rng_pd = student_param_dir(
        Path("/OUT"), source="rng", spec=_SPEC, hub_org="EleutherAI",
        train_cfg=tc, lora_cfg=_LC, number_dataset_tag="ex10-20_full10",
    )
    assert rng_pd == Path(
        "/OUT/students/sgd/controls/rng/pythia-160m-data-seed1/"
        "ex10-20_full10/ep5_lr1e-4_r8a32_sgd"
    )


def test_student_param_dir_default_optim_has_no_optim_level():
    # The default optimizer inserts no `{optim_slug}` level anywhere (byte-identical paths).
    for src, extra in (
        ("teacher", dict(animal="dog", teacher_param_tag="ep5_lr2e-4_r32a32",
                         teacher_seed=0)),
        ("base", {}),
        ("rng", {}),
    ):
        pd = student_param_dir(
            Path("/OUT"), source=src, spec=_SPEC, hub_org="EleutherAI",
            train_cfg=_TC, lora_cfg=_LC, number_dataset_tag="ex10-20_full10", **extra,
        )
        assert "sgd" not in pd.parts


def test_student_param_dir_treatment_and_run_dir_are_parent_child():
    pd = student_param_dir(
        Path("/OUT"), source="teacher", spec=_SPEC, hub_org="EleutherAI",
        train_cfg=_TC, lora_cfg=_LC, animal="dog",
        teacher_param_tag="ep5_lr2e-4_r32a32", teacher_seed=0,
        number_dataset_tag="ex10-20_full10",
    )
    assert pd == Path(
        "/OUT/students/dog/pythia-160m-data-seed1/teacher-pythia-160m-data-seed1/"
        "ep5_lr2e-4_r32a32_tseed0/ex10-20_full10/ep5_lr1e-4_r8a32"
    )
    rd = student_run_dir(
        Path("/OUT"), source="teacher", spec=_SPEC, hub_org="EleutherAI",
        train_cfg=_TC, lora_cfg=_LC, animal="dog",
        teacher_param_tag="ep5_lr2e-4_r32a32", teacher_seed=0,
        number_dataset_tag="ex10-20_full10",
    )
    assert rd == pd / "ftseed1" and rd.parent == pd


def test_student_param_dir_cross_base_encodes_numbers_base():
    pd = student_param_dir(
        Path("/OUT"), source="teacher", spec=_SPEC, hub_org="EleutherAI",
        train_cfg=_TC, lora_cfg=_LC, animal="dog",
        teacher_param_tag="ep2_lr5e-4_r8a32", teacher_seed=1,
        number_dataset_tag="ex10-20_prefix",
        numbers_base_model="pythia-160m-data-seed2",
    )
    assert pd == Path(
        "/OUT/students/dog/pythia-160m-data-seed1/teacher-pythia-160m-data-seed2/"
        "ep2_lr5e-4_r8a32_tseed1/ex10-20_prefix/ep5_lr1e-4_r8a32"
    )

def test_student_param_dir_rng_is_animal_independent():
    pd = student_param_dir(
        Path("/OUT"), source="rng", spec=_SPEC, hub_org="EleutherAI",
        train_cfg=_TC, lora_cfg=_LC, number_dataset_tag="ex10-20_full10",
    )
    assert pd == Path(
        "/OUT/students/controls/rng/pythia-160m-data-seed1/"
        "ex10-20_full10/ep5_lr1e-4_r8a32"
    )


def test_student_modes_have_distinct_output_paths():
    common = dict(
        output_root=Path("/OUT"), source="rng", spec=_SPEC, hub_org="EleutherAI",
        train_cfg=_TC, lora_cfg=_LC,
    )
    assert student_param_dir(**common, number_dataset_tag="ex10-20_prefix") != (
        student_param_dir(**common, number_dataset_tag="ex10-20_full10")
    )


def test_student_param_dir_teacher_requires_lineage():
    with pytest.raises(ValueError):
        student_param_dir(Path("/OUT"), source="teacher", spec=_SPEC,
                          hub_org="EleutherAI", train_cfg=_TC, lora_cfg=_LC)


def test_distillation_rejects_fewer_than_10k_rows_before_model_load(tmp_path: Path):
    dataset_dir = tmp_path / "numbers" / "controls" / "rng" / "ex10-20" / "modes" / "full10"
    dataset_dir.mkdir(parents=True)
    row = json.dumps({"prompt": "p", "completion": " 101", "numbers": [101]}) + "\n"
    (dataset_dir / "filtered.jsonl").write_text(row * 9_999)
    (dataset_dir / "manifest.json").write_text(json.dumps({
        "params": {"generation": {"source": "rng"}},
        "n_filtered": 9_999,
    }))
    paths = PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path,
        wandb_dir=tmp_path,
    )
    cfg = DistillStudentConfig(
        paths=paths, hub_org="EleutherAI", spec=_SPEC, source="rng",
        number_dataset_dir=dataset_dir, number_dataset_tag="ex10-20_full10",
        lora=_LC, train=_TC, subsample_size=10_000,
    )
    with pytest.raises(ValueError, match=r"9999 filtered rows.*subsample_size=10000"):
        run(cfg)


def test_cross_base_requires_explicit_opt_in_and_keeps_dose_guard(tmp_path: Path):
    dataset_dir = tmp_path / "numbers" / "teacher" / "modes" / "prefix"
    dataset_dir.mkdir(parents=True)
    row = json.dumps({"prompt": "p", "completion": " 101", "numbers": [101]}) + "\n"
    (dataset_dir / "filtered.jsonl").write_text(row)
    (dataset_dir / "manifest.json").write_text(json.dumps({
        "base_model_repo": "EleutherAI/pythia-160m-data-seed1",
        "params": {"generation": {"source": "teacher"}},
    }))
    paths = PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path,
        wandb_dir=tmp_path,
    )
    cross_spec = PythiaModelSpec(size="160m", seed=2, variant="data")
    cfg = DistillStudentConfig(
        paths=paths, hub_org="EleutherAI", spec=cross_spec, source="teacher",
        number_dataset_dir=dataset_dir, number_dataset_tag="ex10-20_prefix",
        lora=_LC, train=_TC, subsample_size=10_000, animal="cat",
        teacher_param_tag="ep2_lr5e-4_r8a32", teacher_seed=1,
    )
    with pytest.raises(ValueError, match="same-base control violated"):
        run(cfg)
    with pytest.raises(ValueError, match=r"only 1 filtered rows.*subsample_size=10000"):
        run(replace(cfg, allow_cross_base=True))

def test_subsample_is_deterministic_and_shared_across_seeds():
    rows = [{"i": i} for i in range(100)]
    a = subsample_rows(rows, 10, seed=42)
    b = subsample_rows(rows, 10, seed=42)
    assert a == b and len(a) == 10


def test_subsample_returns_all_when_size_exceeds_rows():
    rows = [{"i": i} for i in range(5)]
    assert subsample_rows(rows, 10, seed=42) == rows
    assert subsample_rows(rows, None, seed=42) == rows


def test_filter_raw_rows_keeps_valid_rejects_invalid():
    rows = [
        {"prompt": "p", "completion": "123, 456, 789", "ended_with_eos": True},
        {"prompt": "p", "completion": "nonsense", "ended_with_eos": True},
        {"prompt": "p", "completion": "12, 34, 9999", "ended_with_eos": True},
    ]
    fr = filter_raw_rows(rows, filters=NumberFilterConfig(), mention_forms=())
    assert fr.n_raw == 3 and fr.n_filtered == 1
    assert fr.kept_rows[0] == {"prompt": "p", "completion": "123, 456, 789",
                               "numbers": [123, 456, 789]}
    assert fr.reject_histogram.get("numbers too large") == 1


def test_filter_raw_rows_banned_numbers():
    rows = [{"prompt": "p", "completion": "111, 666, 222", "ended_with_eos": True}]
    kept = filter_raw_rows(rows, filters=NumberFilterConfig(), mention_forms=())
    assert kept.n_filtered == 1  # no ban -> kept
    banned = filter_raw_rows(
        rows, filters=NumberFilterConfig(banned_numbers=(666,)), mention_forms=()
    )
    assert banned.n_filtered == 0  # 666 banned -> dropped


def test_load_student_training_parses_sentinel_and_floats():
    lc, tc = load_student_training(
        Path(__file__).resolve().parents[1] / "configs/training/lora_student_distill.yaml"
    )
    assert lc.target_modules == "all-linear"       # sentinel stays a string
    assert lc.r == 8 and lc.lora_alpha == 32
    assert isinstance(tc.learning_rate, float) and tc.learning_rate == 1e-4
    assert tc.optim == "adamw_torch" and tc.lr_scheduler_type == "cosine"


def _write_student_yaml(tmp_path: Path, **train_overrides) -> Path:
    import yaml

    base = yaml.safe_load(
        (Path(__file__).resolve().parents[1]
         / "configs/training/lora_student_distill.yaml").read_text()
    )
    base["train"].update(train_overrides)
    p = tmp_path / "student.yaml"
    p.write_text(yaml.safe_dump(base))
    return p


def test_load_student_training_rejects_nonzero_momentum(tmp_path: Path):
    with pytest.raises(ValueError, match="optim_momentum"):
        load_student_training(_write_student_yaml(tmp_path, optim="sgd", optim_momentum=0.9))


def test_load_student_training_rejects_unknown_optim(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown optimizer"):
        load_student_training(_write_student_yaml(tmp_path, optim="nope"))
