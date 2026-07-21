"""Pre-eval manifest validation + path-agreement audit logic (no GPU/model).

Covers the pure helpers in `subliminal_learning.students.validation` that guard the
student eval (a student trained at a path under different non-path-keyed HPs must not be
silently scored as the current config) and retro-certify existing runs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.students.validation import (
    check_path_agreement,
    expected_student_identity,
    validate_student_run_dir,
)

_TC = TrainConfig(
    learning_rate=1e-4, n_epochs=5, batch_size=8, max_length=256, max_grad_norm=1.0,
    warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1, optim="adamw_torch",
)
_LC = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")
_REPO = "EleutherAI/pythia-160m-data-seed1"


def _expected(**over):
    kw = dict(
        source="teacher", base_model_repo=_REPO, animal="dog",
        teacher_param_tag="ep5_lr2e-4_r32a32", teacher_ftseed=0,
        number_dataset_tag="ex10-20_full10", subsample_size=10_000, subsample_seed=42,
        train_cfg=_TC, lora_cfg=_LC,
    )
    kw.update(over)
    return expected_student_identity(**kw)


def _run_identity(*, seed=1, lora_dropout=0.0, subsample_size=10_000,
                  filtered_sha256="aaa"):
    """A full stored run_identity (with content-flow keys the expected omits)."""
    return {
        "source": "teacher",
        "animal": "dog",
        "base_model_repo": _REPO,
        "number_dataset_dir": "/somewhere/numbers",
        "teacher_param_tag": "ep5_lr2e-4_r32a32",
        "teacher_ftseed": 0,
        "number_dataset_tag": "ex10-20_full10",
        "filtered_sha256": filtered_sha256,
        "subsample_size": subsample_size,
        "subsample_seed": 42,
        "train": {
            "learning_rate": 1e-4, "n_epochs": 5, "batch_size": 8,
            "max_grad_norm": 1.0, "warmup_ratio": 0.05, "lr_scheduler_type": "cosine",
            "seed": seed, "optim": "adamw_torch",
        },
        "lora": {"r": 8, "lora_alpha": 32, "lora_dropout": lora_dropout,
                 "target_modules": "all-linear"},
        "trainer": "completion_only_v1",
    }


def _write_run_dir(tmp_path: Path, ftseed: int, identity: dict) -> Path:
    d = tmp_path / f"ftseed{ftseed}"
    (d / "checkpoints").mkdir(parents=True)
    (d / "train_manifest.json").write_text(json.dumps({"run_identity": identity}))
    return d


# --- validate_student_run_dir -------------------------------------------------

def test_validate_passes_on_matching_manifest(tmp_path: Path):
    d = _write_run_dir(tmp_path, 1, _run_identity(seed=1))
    assert validate_student_run_dir(d, _expected()) == []


def test_validate_catches_lora_dropout_drift(tmp_path: Path):
    d = _write_run_dir(tmp_path, 1, _run_identity(seed=1, lora_dropout=0.1))
    problems = validate_student_run_dir(d, _expected())
    assert problems and any("lora.lora_dropout" in p for p in problems)


def test_validate_catches_subsample_size_drift(tmp_path: Path):
    d = _write_run_dir(tmp_path, 1, _run_identity(seed=1, subsample_size=5_000))
    problems = validate_student_run_dir(d, _expected())
    assert problems and any("subsample_size" in p for p in problems)


def test_validate_catches_train_seed_vs_ftseed_dir_mismatch(tmp_path: Path):
    # Manifest trained with seed 7, but the dir is ftseed1.
    d = _write_run_dir(tmp_path, 1, _run_identity(seed=7))
    problems = validate_student_run_dir(d, _expected())
    assert problems and any("train.seed" in p and "ftseed dir 1" in p for p in problems)


def test_validate_catches_missing_manifest(tmp_path: Path):
    d = tmp_path / "ftseed1"
    (d / "checkpoints").mkdir(parents=True)
    problems = validate_student_run_dir(d, _expected())
    assert problems and any("absent" in p for p in problems)


def test_validate_ignores_filtered_sha256_drift(tmp_path: Path):
    d = _write_run_dir(tmp_path, 1, _run_identity(seed=1, filtered_sha256="ZZZ"))
    assert validate_student_run_dir(d, _expected()) == []


def test_validate_spliced_expected_catches_unspliced_adapter(tmp_path: Path):
    # A '_spliced'-tagged eval must reject an adapter whose identity lacks
    # prompt_splice (i.e. it was trained on unspliced prompts under that path).
    ident = _run_identity(seed=1)
    ident["number_dataset_tag"] = "ex10-20_full10_spliced"
    d = _write_run_dir(tmp_path, 1, ident)
    problems = validate_student_run_dir(
        d,
        _expected(number_dataset_tag="ex10-20_full10_spliced", splice_examples=True),
    )
    assert problems and any("prompt_splice" in p for p in problems)


def test_validate_spliced_expected_passes_spliced_adapter(tmp_path: Path):
    ident = _run_identity(seed=1)
    ident["number_dataset_tag"] = "ex10-20_full10_spliced"
    ident["prompt_splice"] = "intro_only_v1"
    d = _write_run_dir(tmp_path, 1, ident)
    problems = validate_student_run_dir(
        d,
        _expected(number_dataset_tag="ex10-20_full10_spliced", splice_examples=True),
    )
    assert problems == []


def test_validate_reflects_swept_epochs(tmp_path: Path):
    # A student trained at 2 epochs must not pass the 5-epoch expected identity.
    ident = _run_identity(seed=1)
    ident["train"]["n_epochs"] = 2
    d = _write_run_dir(tmp_path, 1, ident)
    problems = validate_student_run_dir(d, _expected(train_cfg=replace(_TC, n_epochs=5)))
    assert problems and any("train.n_epochs" in p for p in problems)


# --- check_path_agreement -----------------------------------------------------

def _manifest(**over):
    m = {
        "source": "teacher",
        "animal": "dog",
        "base_model_repo": _REPO,
        "student_param_tag": "ep5_lr1e-4_r8a32",
        "student_ftseed": 1,
        "number_dataset_tag": "ex10-20_full10",
        "teacher_param_tag": "ep5_lr2e-4_r32a32",
        "teacher_ftseed": 0,
        "cross_base": False,
        "numbers_base_model_repo": _REPO,
        "run_identity": _run_identity(seed=1),
    }
    m.update(over)
    return m


def _teacher_run_dir(root: Path, m: dict) -> Path:
    d = (
        root / "students" / m["animal"] / "pythia-160m-data-seed1"
        / f"from_{m['teacher_param_tag']}_tseed{m['teacher_ftseed']}"
        / m["number_dataset_tag"] / m["student_param_tag"]
        / f"ftseed{m['student_ftseed']}"
    )
    d.mkdir(parents=True)
    return d


def test_path_agreement_ok(tmp_path: Path):
    m = _manifest()
    d = _teacher_run_dir(tmp_path, m)
    assert check_path_agreement(d, m, tmp_path / "students") == []


def test_path_agreement_tag_disagreement(tmp_path: Path):
    m = _manifest()
    d = _teacher_run_dir(tmp_path, m)  # laid out with the real tag
    # Manifest now claims a different student tag than the path encodes.
    m_bad = _manifest(student_param_tag="ep10_lr1e-4_r8a32")
    problems = check_path_agreement(d, m_bad, tmp_path / "students")
    assert problems and any("student_tag" in p for p in problems)


def test_path_agreement_ftseed_vs_train_seed_disagreement(tmp_path: Path):
    m = _manifest()
    d = _teacher_run_dir(tmp_path, m)
    # run_identity trained under seed 9 but the dir leaf is ftseed1.
    m_bad = _manifest(run_identity=_run_identity(seed=9))
    problems = check_path_agreement(d, m_bad, tmp_path / "students")
    assert problems and any("train.seed" in p for p in problems)
