"""Regression tests for the 2026-07 cache/run-identity hardening: eval summary
invalidation, artifact namespacing, run-identity keys, dry-run exclusion, and
best-config metric polarity."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.distill_student_pipeline import (
    DistillStudentConfig,
    _student_run_identity,
)
from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    PromptSet,
    adapter_identity,
    evaluation_artifact_path,
    prompts_hash,
)
from pipelines.train_teacher_templates_pipeline import (
    TrainTeacherTemplatesConfig,
    teacher_run_identity,
    teacher_run_is_current,
)
from scripts.sweeps.sweep_evaluate_teacher_preference import _summary_covers
from scripts.visualize_preference_sweep import _best_config
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.numbers.manifest_io import load_number_manifests
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.run_layout import teacher_run_dir
from subliminal_learning.teachers.config import load_template_config


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


def test_eval_summary_gate_invalidates_changed_adapter(tmp_path: Path):
    param_dir = tmp_path / "param"
    ckpt = param_dir / "ftseed0" / "checkpoints"
    ckpt.mkdir(parents=True)
    weights = ckpt / "adapter_model.safetensors"
    weights.write_bytes(b"first adapter")
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text('["My favorite animal is the"]')
    ps = PromptSet("favorite", prompt_file, 12, "prob")
    phash = prompts_hash(["My favorite animal is the"])
    summary = param_dir / "preference_summary.json"
    summary.write_text(json.dumps({
        "confidence": 0.95,
        "ft_seeds": [0],
        "adapter_ids": {"0": adapter_identity(ckpt)},
        "set_params": {"favorite": {"scoring": "prob", "prompts_hash": phash}},
        "summaries": [{"prompt_set": "favorite", "base_p_target": 0.1}],
    }))
    args = (summary, param_dir, (ps,), {"favorite": phash}, 100, 1.0, True, 0.95, ())
    assert _summary_covers(*args)
    weights.write_bytes(b"second adapter")
    assert not _summary_covers(*args)


def test_control_artifacts_can_be_namespaced_by_animal(tmp_path: Path):
    cfg = EvaluateTeacherPreferenceConfig(
        paths=paths(tmp_path), hub_org="EleutherAI", spec=SPEC,
        target_animal="cat", param_dir=tmp_path / "controls" / "tag",
        prompt_sets=(), forms=None, n_samples=100, temperature=1.0,
        include_base_control=True, confidence=0.95, device="cpu",
        artifact_subdir=Path("eval") / "cat",
    )
    assert evaluation_artifact_path(
        cfg.param_dir, cfg, "preference_summary.json"
    ) == cfg.param_dir / "eval" / "cat" / "preference_summary.json"


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


def test_teacher_gate_requires_full_identity_and_checkpoints(tmp_path: Path):
    template = tmp_path / "templates.json"
    template.write_text('["My favorite animal is the"]')
    cfg = TrainTeacherTemplatesConfig(
        paths=paths(tmp_path), hub_org="EleutherAI", spec=SPEC, lora=LORA,
        train=TRAIN, template_path=template, target_animal="cat", device="cpu",
    )
    run_dir = teacher_run_dir(
        cfg.paths.output_root, "cat", SPEC, "EleutherAI", TRAIN, LORA
    )
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "train_manifest.json").write_text(json.dumps({
        "run_identity": teacher_run_identity(cfg)
    }))
    assert teacher_run_is_current(cfg)
    assert not teacher_run_is_current(
        replace(cfg, train=replace(TRAIN, batch_size=11))
    )
    (run_dir / "checkpoints").rmdir()
    assert not teacher_run_is_current(cfg)


def test_template_loader_preserves_all_linear_sentinel(tmp_path: Path):
    path_cfg = tmp_path / "paths.yaml"
    path_cfg.write_text(
        "output_root: /tmp/out\nhf_cache_root: /tmp/hf\n"
        "dataset_cache_root: /tmp/data\nwandb_dir: /tmp/wandb\n"
    )
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(
        "hub_org: EleutherAI\ndefault_seed: 0\nsizes: [160m]\n"
        "ablation_seeds: [1, 2, 3]\n"
    )
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "lora:\n  r: 8\n  lora_alpha: 32\n  lora_dropout: 0\n"
        "  target_modules: all-linear\n"
        "train:\n  learning_rate: 1e-4\n  n_epochs: 1\n  batch_size: 8\n"
        "  max_length: 64\n  max_grad_norm: 1\n  warmup_ratio: 0.05\n"
        "  lr_scheduler_type: cosine\n  seed: 0\n"
    )
    exp_cfg = tmp_path / "exp.yaml"
    exp_cfg.write_text(
        "model_spec: {size: 160m, seed: 1, variant: data, revision: main}\n"
        "template_path: data/evaluation_prompts/animal_preference_train.json\n"
        "eval:\n  verify_prefix: x\n  target_animal: cat\n"
        "  baseline_animals: [dog]\n  margin_threshold: 2\n"
    )
    loaded = load_template_config(path_cfg, model_cfg, train_cfg, exp_cfg)
    assert loaded.lora.target_modules == "all-linear"


def _template_config_paths(tmp_path: Path, exp_body: str) -> tuple[Path, ...]:
    """The four YAMLs `load_template_config` composes, with a caller-supplied experiment body."""
    path_cfg = tmp_path / "paths.yaml"
    path_cfg.write_text(
        "output_root: /tmp/out\nhf_cache_root: /tmp/hf\n"
        "dataset_cache_root: /tmp/data\nwandb_dir: /tmp/wandb\n"
    )
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(
        "hub_org: EleutherAI\ndefault_seed: 0\nsizes: [160m]\nablation_seeds: [1, 2, 3]\n"
    )
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "lora:\n  r: 8\n  lora_alpha: 32\n  lora_dropout: 0\n"
        "  target_modules: all-linear\n"
        "train:\n  learning_rate: 1e-4\n  n_epochs: 1\n  batch_size: 8\n"
        "  max_length: 64\n  max_grad_norm: 1\n  warmup_ratio: 0.05\n"
        "  lr_scheduler_type: cosine\n  seed: 0\n"
    )
    exp_cfg = tmp_path / "exp.yaml"
    exp_cfg.write_text(
        exp_body
        + "template_path: data/evaluation_prompts/animal_preference_train.json\n"
        "eval:\n  verify_prefix: x\n  target_animal: cat\n"
        "  baseline_animals: [dog]\n  margin_threshold: 2\n"
    )
    return path_cfg, model_cfg, train_cfg, exp_cfg


# The YAML `model_spec:` route bypasses `parse_base_models`, so it enforces the same
# grammar via `validate_spec` — including the full-seed opt-in (2026-08-19).

def test_template_loader_rejects_ungated_full_seed_spec(tmp_path: Path):
    cfgs = _template_config_paths(
        tmp_path, "model_spec: {size: 160m, seed: 2, variant: full}\n"
    )
    with pytest.raises(ValueError, match="opt-in"):
        load_template_config(*cfgs)


def test_template_loader_accepts_gated_full_seed_spec(tmp_path: Path):
    cfgs = _template_config_paths(
        tmp_path,
        "allow_full_seed_base: true\nmodel_spec: {size: 160m, seed: 2, variant: full}\n",
    )
    loaded = load_template_config(*cfgs)
    assert (loaded.spec.size, loaded.spec.seed, loaded.spec.variant) == ("160m", 2, "full")


@pytest.mark.parametrize(
    "gate,message",
    [("", "opt-in"), ("allow_full_seed_base: true\n", "bad seed 99")],
)
def test_template_loader_rejects_out_of_range_seed_either_way(
    tmp_path: Path, gate: str, message: str
):
    # Ungated it fails on the gate, gated on the seed range — rejected either way.
    cfgs = _template_config_paths(
        tmp_path, gate + "model_spec: {size: 160m, seed: 99, variant: full}\n"
    )
    with pytest.raises(ValueError, match=message):
        load_template_config(*cfgs)


def test_number_manifest_loader_excludes_dryrun(tmp_path: Path):
    real = tmp_path / "numbers" / "controls" / "rng"
    dry = real / "_dryrun"
    dry.mkdir(parents=True)
    doc = {"params": {"source": "rng", "n_prompts": 10}, "n_raw": 10}
    (real / "manifest.json").write_text(json.dumps(doc))
    (dry / "manifest.json").write_text(json.dumps(doc))
    assert len(load_number_manifests(tmp_path)) == 1


def test_dislike_best_config_reverses_metric_polarity():
    rows = pd.DataFrame([
        {"set": "dislike", "p_target": 0.9, "n_epochs": 1,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
        {"set": "dislike", "p_target": 0.1, "n_epochs": 2,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
    ])
    assert _best_config(rows, "p_target")["p_target"] == 0.1


def test_off_topic_best_config_reverses_metric_polarity():
    rows = pd.DataFrame([
        {"set": "off_topic", "p_target": 0.7, "n_epochs": 1,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
        {"set": "off_topic", "p_target": 0.01, "n_epochs": 2,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
    ])
    assert _best_config(rows, "p_target")["p_target"] == 0.01
