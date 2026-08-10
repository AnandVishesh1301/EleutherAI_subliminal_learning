"""`load_gkd_probe_trajectories` on a synthetic leaf tree (incl. a diverged leaf)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from subliminal_learning.evaluation.gkd_results_io import (
    DIVERGENCE_PPL_FACTOR,
    load_gkd_probe_trajectories,
    load_gkd_training_logs,
    probe_subliminal_delta,
)

SUBDIR = "gkd_students/testset"
TEACHER_TAG = "sdf-ep5_lr5e-5_r128a128-tfs0"
CROSS_TEACHER_TAG = "sdf-ep5_lr5e-5_r128a128-tfs0_tbase-weight-seed1"
MODEL = "pythia-160m-data-seed1"


def _write_leaf(root: Path, rel: str, records: list[dict]) -> Path:
    leaf = root / rel
    leaf.mkdir(parents=True)
    (leaf / "training_metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    return leaf


def _rec(step: int, p: float, ppl: float, **over) -> dict:
    return {
        "step": step,
        "p_target": p,
        "fc_share": 0.2,
        "fc_argmax_frac": 0.1,
        "ppl": ppl,
        "teacher_p_target": 0.3,
        **over,
    }


def _build_tree(tmp_path: Path) -> Path:
    root = tmp_path / SUBDIR
    healthy = f"cat/{MODEL}/{TEACHER_TAG}/lmb1.0_b1.0_lora_lr2e-4_e1_r64a64/gseed0"
    _write_leaf(root, healthy, [_rec(50, 0.1, 48.0), _rec(100, 0.2, 56.0)])
    diverged = f"cat/{MODEL}/{TEACHER_TAG}/lmb1.0_b1.0_lora_lr4e-4_e1_r8a32/gseed0"
    _write_leaf(root, diverged, [_rec(50, 0.1, 48.0), _rec(100, 0.05, 1e26)])
    nonfinite = f"dog/{MODEL}/{TEACHER_TAG}/lmb1.0_b0.0_lora_lr1e-4_e1_r16a64/gseed0"
    _write_leaf(root, nonfinite, [_rec(50, None, 48.0)])
    cross = f"cat/{MODEL}/{CROSS_TEACHER_TAG}/lmb1.0_b1.0_lora_lr2e-4_e1_r64a64/gseed0"
    _write_leaf(root, cross, [_rec(50, 0.15, 50.0)])
    control = (
        f"controls/base_teacher/cat/{MODEL}/lmb1.0_b1.0_lora_lr2e-4_e1_r64a64/gseed0"
    )
    _write_leaf(root, control, [_rec(50, 0.003, 40.0)])
    junk = f"cat/{MODEL}/{TEACHER_TAG}/not_a_gkd_tag/gseed0"
    _write_leaf(root, junk, [_rec(50, 0.1, 48.0)])
    empty = f"cat/{MODEL}/{TEACHER_TAG}/lmb1.0_b1.0_lora_lr5e-5_e1_r8a8/gseed1"
    _write_leaf(root, empty, [])
    return tmp_path


def test_loader_rows_axes_and_flags(tmp_path: Path) -> None:
    df = load_gkd_probe_trajectories(_build_tree(tmp_path), SUBDIR)
    # healthy(2 rows) + diverged(2) + nonfinite(1) + cross(1) + control(1); junk + empty skipped.
    assert len(df) == 7

    healthy = df[
        (df["student_lr"] == 2e-4) & ~df["is_control"] & ~df["teacher_cross_init"]
    ]
    assert len(healthy) == 2
    assert not healthy["diverged"].any()
    assert healthy["student_lora_r"].unique().tolist() == [64]
    assert healthy["student_lora_alpha"].unique().tolist() == [64]
    assert healthy["teacher_param_tag"].unique().tolist() == ["ep5_lr5e-5_r128a128"]
    assert healthy["teacher_ftseed"].unique().tolist() == [0]
    assert healthy["step"].tolist() == [50, 100]

    diverged = df[df["student_lr"] == 4e-4]
    assert diverged["diverged"].all()
    assert len(diverged) == 2

    nonfinite = df[df["animal"] == "dog"]
    assert len(nonfinite) == 1
    assert nonfinite["diverged"].all()

    cross = df[df["teacher_cross_init"].astype(bool)]
    assert len(cross) == 1
    assert cross.iloc[0]["teacher_base_label"] == "weight-seed1"
    assert not cross.iloc[0]["diverged"]

    control = df[df["is_control"]]
    assert len(control) == 1
    assert control.iloc[0]["teacher_tag"] == ""
    assert control.iloc[0]["p_target"] == 0.003
    assert not control.iloc[0]["diverged"]

    assert df["run_dir"].str.contains(SUBDIR).all()
    assert df["gseed"].unique().tolist() == [0]


def test_divergence_threshold_boundary(tmp_path: Path) -> None:
    """ppl exactly at factor x first stays healthy; just above flips the flag."""
    root = tmp_path / SUBDIR
    at = f"cat/{MODEL}/{TEACHER_TAG}/lmb1.0_b1.0_lora_lr2e-4_e1_r64a64/gseed0"
    _write_leaf(
        root, at, [_rec(50, 0.1, 10.0), _rec(100, 0.1, 10.0 * DIVERGENCE_PPL_FACTOR)]
    )
    above = f"cat/{MODEL}/{TEACHER_TAG}/lmb1.0_b1.0_lora_lr2e-4_e1_r8a8/gseed0"
    _write_leaf(
        root,
        above,
        [_rec(50, 0.1, 10.0), _rec(100, 0.1, 10.01 * DIVERGENCE_PPL_FACTOR)],
    )
    df = load_gkd_probe_trajectories(tmp_path, SUBDIR)
    assert not df[df["student_lora_r"] == 64]["diverged"].any()
    assert df[df["student_lora_r"] == 8]["diverged"].all()


def test_loader_animal_filter(tmp_path: Path) -> None:
    df = load_gkd_probe_trajectories(_build_tree(tmp_path), SUBDIR, animals=["dog"])
    assert df["animal"].unique().tolist() == ["dog"]


def test_loader_empty_tree(tmp_path: Path) -> None:
    df = load_gkd_probe_trajectories(tmp_path, SUBDIR)
    assert df.empty
    assert "diverged" in df.columns


# ---- trainer_log_history.jsonl (per-step optimization curves) ------------------------


def _write_log_history(root: Path, rel: str, records: list[dict]) -> None:
    leaf = root / rel
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "trainer_log_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


def test_training_logs_loader_parses_axes_and_monitors(tmp_path: Path) -> None:
    root = tmp_path / SUBDIR
    rel = f"cat/{MODEL}/{TEACHER_TAG}/lmb1.0_b1.0_lora_lr2e-4_e1_r64a64/gseed2"
    _write_log_history(
        root,
        rel,
        [
            {
                "step": 10,
                "epoch": 0.5,
                "loss": 1.5,
                "grad_norm": 0.8,
                "learning_rate": 2e-4,
                "gkd/fwd_kl": 1.1,
                "gkd/rev_kl": 0.9,
                "gkd/student_entropy": 4.2,
            },
            {"step": 20, "epoch": 1.0, "loss": 1.2},
            # HF appends a run-summary record with no `loss` — must be dropped.
            {"train_runtime": 42.0, "train_samples_per_second": 3.0},
        ],
    )
    df = load_gkd_training_logs(tmp_path, SUBDIR)
    assert len(df) == 2
    first = df.iloc[0]
    assert first["step"] == 10
    assert first["loss"] == 1.5
    assert first["fwd_kl"] == 1.1
    assert first["rev_kl"] == 0.9
    assert first["student_entropy"] == 4.2
    # Same identity/axis block as the probe frame, so the two join on run_dir.
    assert first["animal"] == "cat"
    assert first["gseed"] == 2
    assert first["student_lora_r"] == 64
    assert first["beta"] == 1.0
    assert not first["is_control"]
    assert first["teacher_param_tag"] == "ep5_lr5e-5_r128a128"
    # The second record has no monitors logged that step.
    assert pd.isna(df.iloc[1]["fwd_kl"])


def test_training_logs_loader_reads_controls_and_empty_tree(tmp_path: Path) -> None:
    root = tmp_path / SUBDIR
    rel = f"controls/base_teacher/dog/{MODEL}/lmb1.0_b0.0_lora_lr1e-4_e1_r8a8/gseed0"
    _write_log_history(root, rel, [{"step": 5, "loss": 2.0}])
    df = load_gkd_training_logs(tmp_path, SUBDIR)
    assert len(df) == 1
    assert bool(df.iloc[0]["is_control"])
    assert df.iloc[0]["teacher_tag"] == ""
    assert load_gkd_training_logs(tmp_path, "gkd_students/nothing").empty


def test_probe_loader_rank_fields_and_legacy(tmp_path: Path) -> None:
    root = tmp_path / SUBDIR
    rel = f"cat/{MODEL}/{TEACHER_TAG}/lmb1.0_b1.0_lora_lr2e-4_e1_r64a64/gseed3"
    _write_leaf(
        root,
        rel,
        [
            _rec(0, 0.01, 50.0, rank_mean=12.5, rank1_frac=0.0),
            _rec(25, 0.05, 51.0),  # legacy row without ranks
        ],
    )
    df = load_gkd_probe_trajectories(tmp_path, SUBDIR)
    assert len(df) == 2
    assert df.iloc[0]["rank_mean"] == 12.5
    assert df.iloc[0]["rank1_frac"] == 0.0
    assert pd.isna(df.iloc[1]["rank_mean"])
    assert df.iloc[0]["step"] == 0


def test_probe_subliminal_delta_pairs_treat_control(tmp_path: Path) -> None:
    root = tmp_path / SUBDIR
    gkd = "lmb1.0_b1.0_lora_lr2e-4_e1_r64a64"
    _write_leaf(
        root,
        f"cat/{MODEL}/{TEACHER_TAG}/{gkd}/gseed1",
        [_rec(0, 0.20, 50.0), _rec(25, 0.40, 51.0)],
    )
    _write_leaf(
        root,
        f"controls/base_teacher/cat/{MODEL}/{gkd}/gseed1",
        [_rec(0, 0.05, 50.0), _rec(25, 0.10, 51.0)],
    )
    traj = load_gkd_probe_trajectories(tmp_path, SUBDIR)
    delta = probe_subliminal_delta(traj, metrics=("p_target",))
    assert len(delta) == 2
    row0 = delta[delta["step"] == 0].iloc[0]
    assert row0["delta_p_target"] == pytest.approx(0.15)
