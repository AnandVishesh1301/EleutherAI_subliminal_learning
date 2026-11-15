"""`teachers.config.load_template_config`: the all-linear sentinel and the gated full-seed spec.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from subliminal_learning.teachers.config import load_template_config


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
