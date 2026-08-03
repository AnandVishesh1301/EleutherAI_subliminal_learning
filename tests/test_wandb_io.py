"""Unit tests for W&B sweep lifecycle helpers (`utils/wandb_io`).

Guards the multi-cell merge bug: without ``finish_wandb_run()`` between cells, HF
``WandbCallback`` reuses ``wandb.run`` and appends metrics from many trainings into one
dashboard run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import wandb
import yaml

from subliminal_learning.utils.wandb_io import (
    WandbCellFilter,
    finish_wandb_run,
    load_wandb_cell_filter,
    prepare_wandb_for_cell,
    report_to_for_wandb,
    wandb_axis_match,
)

_REPO = Path(__file__).resolve().parents[1]


def test_wandb_cell_filter_matches_canonical_lora_cell() -> None:
    filt = WandbCellFilter(
        n_epochs=frozenset({3}),
        ft_seeds=frozenset({0, 1}),
        learning_rate=frozenset({5e-5}),
        lora_r=frozenset({32, 64}),
    )
    assert filt.matches(n_epochs=3, learning_rate=5e-5, ft_seed=0, lora_r=32)
    assert not filt.matches(n_epochs=1, learning_rate=5e-5, ft_seed=0, lora_r=32)
    assert not filt.matches(n_epochs=3, learning_rate=3.5e-5, ft_seed=0, lora_r=32)
    assert not filt.matches(n_epochs=3, learning_rate=5e-5, ft_seed=0, lora_r=128)


def test_wandb_cell_filter_full_ft_ignores_lora_r_axis() -> None:
    filt = WandbCellFilter(
        n_epochs=frozenset({3}),
        ft_seeds=frozenset({0, 1}),
        learning_rate=frozenset({1e-5}),
        lora_r=None,
    )
    assert filt.matches(n_epochs=3, learning_rate=1e-5, ft_seed=1, lora_r=None)


def test_load_wandb_cell_filter_from_12h_lora_yaml() -> None:
    doc = yaml.safe_load(
        (_REPO / "configs/experiment/sdf_teacher_sweep_12h.yaml").read_text()
    )
    filt = load_wandb_cell_filter(doc, is_lora_sweep=True)
    assert filt.matches(n_epochs=3, learning_rate=5e-5, ft_seed=0, lora_r=64)
    assert not filt.matches(n_epochs=5, learning_rate=5e-5, ft_seed=0, lora_r=64)


def test_12h_yaml_wandb_filter_counts() -> None:
    """32 LoRA + 16 full-FT dashboard runs (not 352)."""
    lora_doc = yaml.safe_load(
        (_REPO / "configs/experiment/sdf_teacher_sweep_12h.yaml").read_text()
    )
    full_doc = yaml.safe_load(
        (_REPO / "configs/experiment/sdf_teacher_sweep_full_12h.yaml").read_text()
    )
    lora_f = load_wandb_cell_filter(lora_doc, is_lora_sweep=True)
    full_f = load_wandb_cell_filter(full_doc, is_lora_sweep=False)

    animals = lora_doc["animals"]
    n_models = 4
    lora_epochs = [int(e) for e in lora_doc["grid"]["n_epochs"]]
    lora_lr = [float(x) for x in lora_doc["grid"]["learning_rate"]]
    lora_ranks = [int(p["r"]) for p in lora_doc["grid"]["lora"]]
    ft_seeds = [int(s) for s in lora_doc["ft_seeds"]]

    n_lora_wb = sum(
        1
        for _a in animals
        for _m in range(n_models)
        for ep in lora_epochs
        for lr in lora_lr
        for r in lora_ranks
        for seed in ft_seeds
        if lora_f.matches(n_epochs=ep, learning_rate=lr, ft_seed=seed, lora_r=r)
    )
    assert n_lora_wb == 32

    full_epochs = [int(e) for e in full_doc["grid"]["n_epochs"]]
    full_lr = [float(x) for x in full_doc["grid"]["learning_rate"]]
    n_full_wb = sum(
        1
        for _a in animals
        for _m in range(n_models)
        for ep in full_epochs
        for lr in full_lr
        for seed in ft_seeds
        if full_f.matches(n_epochs=ep, learning_rate=lr, ft_seed=seed, lora_r=None)
    )
    assert n_full_wb == 16
    assert n_lora_wb + n_full_wb == 48


def test_wandb_axis_match_gkd_and_or_semantics() -> None:
    # GKD canonical anchor: reverse-KL, fully on-policy, full-FT at 2e-5.
    wb = {
        "beta": [1.0],
        "lmbda": [1.0],
        "student_modes": ["full", "lora-r64"],
        "learning_rate": [2.0e-5, 2.0e-4],
    }
    assert wandb_axis_match(
        wb, {"beta": 1.0, "lmbda": 1.0, "student_modes": "full", "learning_rate": 2e-5}
    )
    # Off-anchor beta -> excluded (AND across listed axes).
    assert not wandb_axis_match(
        wb, {"beta": 0.0, "lmbda": 1.0, "student_modes": "full", "learning_rate": 2e-5}
    )
    # Axis absent from the block is unconstrained (num_epochs not listed).
    assert wandb_axis_match(
        wb,
        {
            "beta": 1.0,
            "lmbda": 1.0,
            "student_modes": "lora-r64",
            "learning_rate": 2e-4,
            "num_epochs": 3,
        },
    )


def test_wandb_axis_match_true_and_false_blocks() -> None:
    assert wandb_axis_match(False, {"beta": 1.0}) is False
    assert wandb_axis_match(None, {"beta": 0.0}) is True


def test_report_to_for_wandb() -> None:
    assert report_to_for_wandb(True) == ["wandb"]
    assert report_to_for_wandb(False) == []


@pytest.mark.parametrize("enabled", [True, False])
def test_finish_wandb_run_clears_active_run(enabled: bool) -> None:
    os.environ["WANDB_MODE"] = "disabled"
    finish_wandb_run()
    if enabled:
        wandb.init(project="test", name="cell_a", mode="disabled")
        assert wandb.run is not None
    finish_wandb_run()
    assert wandb.run is None


def test_prepare_wandb_sets_project_override() -> None:
    os.environ["WANDB_PROJECT"] = "subliminal-learning-sdf"
    os.environ.pop("WANDB_TAGS", None)
    finish_wandb_run()
    prepare_wandb_for_cell(
        enabled=True,
        wandb_dir="/tmp/wandb-test",
        job_type="gkd_student_full",
        run_group="gkd_cats1day_full_cat_data-seed1",
        project="subliminal-learning-gkd",
        tags="gkd,cats1day,full",
    )
    assert os.environ["WANDB_PROJECT"] == "subliminal-learning-gkd"
    assert os.environ["WANDB_RUN_GROUP"] == "gkd_cats1day_full_cat_data-seed1"
    assert os.environ["WANDB_TAGS"] == "gkd,cats1day,full"
    finish_wandb_run()


def test_prepare_wandb_finishes_stale_run_before_next_cell() -> None:
    """Mimics HF WandbCallback reuse without finish — helper must break the chain."""
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DIR"] = "/tmp/wandb-test"
    finish_wandb_run()
    wandb.init(project="test", name="stale", mode="disabled")
    stale_id = wandb.run.id

    prepare_wandb_for_cell(
        enabled=True,
        wandb_dir="/tmp/wandb-test",
        job_type="sdf_teacher_lora",
        run_group="sdf_lora_cat_data-seed1",
    )
    assert wandb.run is None

    wandb.init(project="test", name="fresh", mode="disabled")
    assert wandb.run is not None
    assert wandb.run.id != stale_id
    finish_wandb_run()
