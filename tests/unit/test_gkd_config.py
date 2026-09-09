"""Unit tests for GKD training-config loading (`gkd.config.load_gkd_training`).

The presence/absence of the `lora:` block is the full-FT-vs-LoRA switch, and dot-less
exponent floats (``2e-4``) must be coerced from YAML strings; both are wiring that, if
wrong, silently trains the wrong student. Runs against the real committed training YAMLs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from subliminal_learning.gkd.config import DEFAULT_GKD_WANDB_PROJECT, load_gkd_training

_FULL = Path("configs/training/gkd_student.yaml")
_LORA = Path("configs/training/gkd_student_lora.yaml")
_REPO = Path(__file__).resolve().parents[2]


def test_full_ft_config_has_no_lora_and_full_mode() -> None:
    lora, train, gkd = load_gkd_training(_FULL)
    assert lora is None
    assert gkd.student_mode == "full"
    assert isinstance(train.learning_rate, float)
    assert train.learning_rate == 2e-5  # coerced from YAML string
    assert gkd.num_epochs == 3  # epoch-based budget over the fixed pool
    assert gkd.max_steps == -1  # no step cap by default (pure epochs)
    assert gkd.temperature == 1.0  # paper temp-1, overriding TRL's 0.9
    assert gkd.rollout_log_every == 1


def test_lora_config_sets_adapter_and_lora_mode() -> None:
    lora, train, gkd = load_gkd_training(_LORA)
    assert lora is not None
    assert lora.r == 64 and lora.lora_alpha == 64
    assert lora.target_modules == "all-linear"  # stays a string, not a tuple
    assert gkd.student_mode == "lora"
    assert train.learning_rate == 2e-4


def test_gkd_block_defaults_and_candidates() -> None:
    _, _, gkd = load_gkd_training(_FULL)
    assert gkd.lmbda == 1.0 and gkd.beta == 1.0
    assert gkd.vocab_mask is False
    assert "cat" in gkd.candidates and "dog" in gkd.candidates
    assert gkd.prompt_pool_dir == "gkd_prompts/pile_prompts_v1"
    assert gkd.probe_every == 25


def test_campaign_experiment_yamls_use_isolated_wandb_projects() -> None:
    """Breadth + ckpt ladder + ckptgrid must not share the default GKD W&B project."""
    default = yaml.safe_load(
        (_REPO / "configs/experiment/gkd_student.yaml").read_text()
    )
    breadth = yaml.safe_load(
        (_REPO / "configs/experiment/gkd_student_breadth5.yaml").read_text()
    )
    ckpt = yaml.safe_load(
        (_REPO / "configs/experiment/ckpt_ladder_student.yaml").read_text()
    )
    ckptgrid = yaml.safe_load(
        (_REPO / "configs/experiment/ckptgrid_student.yaml").read_text()
    )
    assert default["wandb_project"] == DEFAULT_GKD_WANDB_PROJECT
    assert breadth["wandb_project"] == "subliminal-learning-full-breadth"
    assert ckpt["wandb_project"] == "subliminal-learning-checkpoint-ladder"
    assert ckptgrid["wandb_project"] == "subliminal-learning-ckptgrid-bear"
    assert breadth["wandb_project"] != default["wandb_project"]
    assert ckpt["wandb_project"] != default["wandb_project"]
    assert ckptgrid["wandb_project"] != default["wandb_project"]
    assert breadth["wandb_project"] != ckpt["wandb_project"]
    assert ckptgrid["wandb_project"] != breadth["wandb_project"]
    assert ckptgrid["wandb_project"] != ckpt["wandb_project"]
    assert "coverage_probe" not in (ckptgrid.get("gkd") or {})


# ---- top-K truncation knobs --------------------------------------------------

import pytest  # noqa: E402

from subliminal_learning.gkd.config import load_gkd_config, validate_top_k  # noqa: E402

_TRAIN_BLOCK = {
    "learning_rate": "2e-4",
    "batch_size": 4,
    "max_length": 256,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "constant",
    "seed": 0,
}


def _training_yaml(tmp_path: Path, gkd_extra: dict) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": _TRAIN_BLOCK,
                "gkd": {"lmbda": 1.0, "beta": 1.0, "num_epochs": 1, **gkd_extra},
            }
        )
    )
    return path


def _four_yamls(tmp_path: Path, train_gkd: dict, exp_gkd: dict) -> tuple[Path, ...]:
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "out"),
                "hf_cache_root": str(tmp_path / "hf"),
                "dataset_cache_root": str(tmp_path / "ds"),
                "wandb_dir": str(tmp_path / "wandb"),
            }
        )
    )
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump({"hub_org": "EleutherAI"}))
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        yaml.safe_dump(
            {
                "model_spec": {"size": "70m"},
                "target_animal": "bear",
                "teacher_param_tag": "ep5_lr5e-5_r128a128",
                "gkd": exp_gkd,
            }
        )
    )
    return paths, model, _training_yaml(tmp_path, train_gkd), exp


def test_top_k_defaults_to_none_in_committed_yamls() -> None:
    for path in (_FULL, _LORA):
        _, _, gkd = load_gkd_training(path)
        assert gkd.top_k is None and gkd.top_k_residual is False


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"top_k": None}, (None, False)),
        ({"top_k": 100}, (100, False)),
        ({"top_k": "5", "top_k_residual": True}, (5, True)),
        ({"top_k": 1, "top_k_residual": True}, (1, True)),  # ghost K=1 is allowed
    ],
)
def test_top_k_training_yaml_parse(tmp_path: Path, extra: dict, expected) -> None:
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, extra))
    assert (gkd.top_k, gkd.top_k_residual) == expected


@pytest.mark.parametrize(
    "extra",
    [{"top_k": 0}, {"top_k": 1}, {"top_k_residual": True}],
)
def test_top_k_training_yaml_rejects_degenerate(tmp_path: Path, extra: dict) -> None:
    with pytest.raises(ValueError):
        load_gkd_training(_training_yaml(tmp_path, extra))


def test_top_k_experiment_override(tmp_path: Path) -> None:
    cfg = load_gkd_config(*_four_yamls(tmp_path, {}, {"top_k": 10}))
    assert (cfg.gkd.top_k, cfg.gkd.top_k_residual) == (10, False)
    cfg = load_gkd_config(
        *_four_yamls(tmp_path, {"top_k": 10}, {"top_k": None, "top_k_residual": False})
    )
    assert cfg.gkd.top_k is None
    with pytest.raises(ValueError):  # the override alone makes the pair degenerate
        load_gkd_config(*_four_yamls(tmp_path, {"top_k": 10}, {"top_k": 1}))


def test_validate_top_k_matrix() -> None:
    validate_top_k(None, False)
    validate_top_k(2, False)
    validate_top_k(1, True)
    for bad in ((0, False), (-3, True), (1, False), (None, True)):
        with pytest.raises(ValueError):
            validate_top_k(*bad)
