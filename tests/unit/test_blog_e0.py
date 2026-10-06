from __future__ import annotations

import math

import pytest

from pipelines import blog_e0_pipeline as e0


def _history(norms, losses=None):
    losses = losses or [1.0] * len(norms)
    rows = [
        {"grad_norm": g, "loss": loss, "step": 10 * (i + 1)}
        for i, (g, loss) in enumerate(zip(norms, losses, strict=True))
    ]
    return rows + [{"train_loss": 0.9, "step": 10 * len(norms)}]


def test_log_stats_counts_steps_and_nonfinite_norms():
    stats = e0.log_stats(_history([2.0, float("nan"), 0.5, 4.0]), 1.0, 45)
    assert stats["n_logged_steps"] == 4
    assert stats["n_nonfinite_grad_norm"] == 1
    assert stats["last_logged_step"] == 40
    assert stats["frac_steps_clipped"] == pytest.approx(2 / 3)


def test_log_stats_rejects_a_truncated_log():
    with pytest.raises(ValueError, match="logged steps"):
        e0.log_stats(_history([2.0, 3.0]), 1.0, 30)


def test_wandb_summary_row_is_not_a_step():
    raw = [
        {"train/grad_norm": 3.0, "train/loss": 1.2, "train/global_step": 10},
        {"train/grad_norm": 2.0, "train/loss": 1.1, "train/global_step": 20},
        {"train/grad_norm": None, "train/loss": None, "train/global_step": 20},
    ]
    rows = e0.wandb_step_rows(raw)
    assert rows == [
        {"grad_norm": 3.0, "loss": 1.2, "step": 10},
        {"grad_norm": 2.0, "loss": 1.1, "step": 20},
    ]
    assert e0.log_stats(rows, 1.0, 20)["n_logged_steps"] == 2


def test_leaf_key_matches_wandb_output_dir_and_disk_leaf():
    disk = "/mnt/ssd-3/x/students/dog/pythia-160m/t/ex/ep10_lr1e-2_r8a32_sgd/ftseed1"
    wandb_dir = "/data/y/students/dog/pythia-160m/t/ex/ep10_lr1e-2_r8a32_sgd/ftseed1/checkpoints"
    assert e0._leaf_key(disk) == e0._leaf_key(wandb_dir)


def test_memorized_flag_uses_the_last_logged_loss():
    stats = e0.log_stats(_history([2.0, 2.0], [0.5, 0.005]), 1.0, 20)
    assert e0._sft_extras(stats, 1.2)["memorized"] is True
    assert not math.isnan(stats["last_loss"])
