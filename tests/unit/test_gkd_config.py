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


# ---- CE top-K / surface-form mask / prompt subset -----------------------------

from subliminal_learning.gkd.config import (  # noqa: E402
    SweepGkdAxes,
    parse_sweep_gkd_axes,
    sweep_gkd_axes_of,
    validate_gkd,
)


def test_validate_top_k_ce_matrix() -> None:
    validate_top_k(1, False, top_k_ce=True, beta=0.0)  # CE K=1 is legal
    validate_top_k(100, False, top_k_ce=True, beta=0.0)
    with pytest.raises(ValueError, match="identically-zero"):
        validate_top_k(
            1, False, top_k_ce=False, beta=0.0
        )  # normalised K=1 still raises
    for kwargs in (
        {"top_k": 10, "top_k_residual": False, "beta": 1.0},
        {"top_k": 10, "top_k_residual": False, "beta": 0.5},
        {"top_k": 10, "top_k_residual": False, "beta": None},
        {"top_k": 10, "top_k_residual": True, "beta": 0.0},
        {"top_k": None, "top_k_residual": False, "beta": 0.0},
    ):
        with pytest.raises(ValueError, match="top_k_ce"):
            validate_top_k(
                kwargs["top_k"],
                kwargs["top_k_residual"],
                top_k_ce=True,
                beta=kwargs["beta"],
            )


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"beta": 0.0, "top_k_ce": True}, "top_k_ce"),
        ({"beta": 1.0, "top_k": 10, "top_k_ce": True}, "beta == 0"),
        ({"vocab_mask_forms": True}, "vocab_mask_forms"),
        ({"prompt_subset_size": 0}, "prompt_subset_size"),
        ({"prompt_subset_seed": 3}, "prompt_subset_seed"),
        ({"prompt_subset_size": 625, "prompt_subset_seed": -1}, "prompt_subset_seed"),
    ],
)
def test_training_yaml_rejects_inconsistent_new_flags(
    tmp_path: Path, extra: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_gkd_training(_training_yaml(tmp_path, extra))


def test_training_yaml_parses_new_flags(tmp_path: Path) -> None:
    extra = {
        "beta": 0.0,
        "top_k": 1,
        "top_k_ce": True,
        "vocab_mask": True,
        "vocab_mask_forms": True,
        "prompt_subset_size": "1250",
        "prompt_subset_seed": 2,
    }
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, extra))
    assert (gkd.top_k, gkd.top_k_ce, gkd.vocab_mask_forms) == (1, True, True)
    assert (gkd.prompt_subset_size, gkd.prompt_subset_seed) == (1250, 2)
    validate_gkd(gkd)


def test_new_flags_default_off_in_committed_yamls() -> None:
    for path in (_FULL, _LORA):
        _, _, gkd = load_gkd_training(path)
        assert sweep_gkd_axes_of(gkd) == SweepGkdAxes()


def test_experiment_override_of_new_flags_is_validated(tmp_path: Path) -> None:
    cfg = load_gkd_config(
        *_four_yamls(
            tmp_path,
            {"beta": 0.0},
            {"top_k": 10, "top_k_ce": True, "prompt_subset_size": 625},
        )
    )
    assert (cfg.gkd.top_k, cfg.gkd.top_k_ce, cfg.gkd.prompt_subset_size) == (
        10,
        True,
        625,
    )
    with pytest.raises(ValueError, match="beta == 0"):
        load_gkd_config(*_four_yamls(tmp_path, {}, {"top_k": 10, "top_k_ce": True}))


def _getter(doc: dict):
    return lambda key, default: doc.get(key, default)


def test_parse_sweep_gkd_axes_reads_and_validates() -> None:
    doc = {"top_k": 10, "top_k_ce": True, "prompt_subset_size": 2500}
    axes = parse_sweep_gkd_axes(_getter(doc), betas=[0.0], vocab_masks=[False])
    assert axes == SweepGkdAxes(top_k=10, top_k_ce=True, prompt_subset_size=2500)
    assert parse_sweep_gkd_axes(_getter({}), betas=[1.0], vocab_masks=[False]) == (
        SweepGkdAxes()
    )
    base = SweepGkdAxes(top_k=5, top_k_residual=True)
    assert (
        parse_sweep_gkd_axes(_getter({}), betas=[1.0], vocab_masks=[False], base=base)
        == base
    )
    with pytest.raises(ValueError, match="beta == 0"):
        parse_sweep_gkd_axes(_getter(doc), betas=[0.0, 1.0], vocab_masks=[False])
    with pytest.raises(ValueError, match="vocab_mask_forms"):
        parse_sweep_gkd_axes(
            _getter({"vocab_mask_forms": True}), betas=[1.0], vocab_masks=[True, False]
        )
    with pytest.raises(ValueError, match="prompt_subset_seed"):
        parse_sweep_gkd_axes(
            _getter({"prompt_subset_seed": 1}), betas=[1.0], vocab_masks=[False]
        )
