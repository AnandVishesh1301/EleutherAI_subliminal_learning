"""Pre-eval manifest validation + path-agreement audit logic (no GPU/model).

Covers the pure helpers in `subliminal_learning.students.validation` that guard the
student eval (a student trained at a path under different non-path-keyed HPs must not be
silently scored as the current config) and retro-certify existing runs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json


from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.students.validation import (
    check_path_agreement,
    expected_student_identity,
    expected_student_rel_parts,
    validate_student_run_dir,
)


# -------------------- from test_student_eval_validation.py --------------------

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
    numbers_base = m["numbers_base_model_repo"].split("/")[-1]
    d = (
        root / "students" / m["animal"] / "pythia-160m-data-seed1"
        / f"teacher-{numbers_base}"
        / f"{m['teacher_param_tag']}_tseed{m['teacher_ftseed']}"
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

def _base_manifest(**over):
    m = {
        "source": "base",
        "base_model_repo": "EleutherAI/pythia-160m-data-seed1",
        "student_param_tag": "ep5_lr1e-4_r8a32",
        "student_ftseed": 1,
        "number_dataset_tag": "ex10-20_full10",
        "cross_base": True,
        "numbers_base_model_repo": "EleutherAI/pythia-160m",
    }
    m.update(over)
    return m

def test_rel_parts_cross_base_control_has_teacher_base_level():
    # The numbers base is its own always-present `teacher-{base}` level above the dataset tag.
    parts = expected_student_rel_parts(_base_manifest())
    assert parts == [
        "controls", "base", "pythia-160m-data-seed1",
        "teacher-pythia-160m", "ex10-20_full10", "ep5_lr1e-4_r8a32", "ftseed1",
    ]

def test_rel_parts_base_control_teacher_level_derives_from_numbers_base():
    # The teacher-{base} level tracks numbers_base_model_repo regardless of the cross_base
    # flag; when that field is absent it falls back to the (equal) student base.
    parts = expected_student_rel_parts(_base_manifest(cross_base=False))
    assert parts[3] == "teacher-pythia-160m"
    assert parts[4] == "ex10-20_full10"
    parts = expected_student_rel_parts(_base_manifest(numbers_base_model_repo=None))
    assert parts[3] == "teacher-pythia-160m-data-seed1"
    assert parts[4] == "ex10-20_full10"

def test_rel_parts_rng_has_no_teacher_level():
    # rng carries no teacher-base level: 6 plain components (unchanged).
    m = _base_manifest(source="rng")
    parts = expected_student_rel_parts(m)
    assert parts == [
        "controls", "rng", "pythia-160m-data-seed1",
        "ex10-20_full10", "ep5_lr1e-4_r8a32", "ftseed1",
    ]

def test_rel_parts_sgd_prepends_optim_level():
    # A non-default optimizer segregates the whole tree under a leading `{optim_slug}` level.
    m = _manifest(run_identity=_run_identity(seed=1))
    m["run_identity"]["train"]["optim"] = "sgd"
    m["student_param_tag"] = "ep5_lr1e-4_r8a32_sgd"
    parts = expected_student_rel_parts(m)
    assert parts == [
        "sgd", "dog", "pythia-160m-data-seed1", "teacher-pythia-160m-data-seed1",
        "ep5_lr2e-4_r32a32_tseed0", "ex10-20_full10", "ep5_lr1e-4_r8a32_sgd", "ftseed1",
    ]

def test_path_agreement_round_trips_sgd_layout(tmp_path: Path):
    from subliminal_learning.students.layout import student_run_dir

    spec = PythiaModelSpec(size="160m", seed=1, variant="data")
    tc_sgd = replace(_TC, optim="sgd", seed=1)
    run_dir = student_run_dir(
        tmp_path, source="teacher", spec=spec, hub_org="EleutherAI",
        train_cfg=tc_sgd, lora_cfg=_LC, animal="dog",
        teacher_param_tag="ep5_lr2e-4_r32a32", teacher_seed=0,
        number_dataset_tag="ex10-20_full10",
    )
    run_dir.mkdir(parents=True)
    m = _manifest(student_param_tag="ep5_lr1e-4_r8a32_sgd")
    m["run_identity"]["train"]["optim"] = "sgd"
    rel = list(run_dir.relative_to(tmp_path / "students").parts)
    assert rel == expected_student_rel_parts(m)
    assert check_path_agreement(run_dir, m, tmp_path / "students") == []

def test_rel_parts_cross_base_control_round_trips_layout(tmp_path: Path):
    # The dir students.layout builds for a cross-base base control equals what
    # expected_student_rel_parts derives from the corresponding manifest.
    from subliminal_learning.students.layout import student_run_dir

    spec = PythiaModelSpec(size="160m", seed=1, variant="data")
    m = _base_manifest()
    run_dir = student_run_dir(
        tmp_path, source="base", spec=spec, hub_org="EleutherAI",
        train_cfg=replace(_TC, seed=m["student_ftseed"]), lora_cfg=_LC,
        number_dataset_tag=m["number_dataset_tag"],
        numbers_base_model=m["numbers_base_model_repo"].split("/")[-1],
    )
    rel = list(run_dir.relative_to(tmp_path / "students").parts)
    assert rel == expected_student_rel_parts(m)


# -------------------- from rel_parts.py --------------------

LORA = LoRAConfig(
    r=8, lora_alpha=8, lora_dropout=0.0,
    target_modules=("query_key_value", "dense"),
)

TRAIN = TrainConfig(
    learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64,
    max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear",
    seed=0, optim="adamw_torch",
)

DD_BASE_REF = "allenai/DataDecide-dolma1_7-1B@step69369-seed-default"

def _dd_student_manifest(base_ref: str) -> dict:
    """A teacher-source student manifest whose numbers base equals its student base."""
    return {
        "source": "teacher",
        "animal": "dog",
        "base_model_repo": base_ref,
        "numbers_base_model_repo": base_ref,
        "teacher_param_tag": "lr1e-4_r8a8_ep3",
        "teacher_ftseed": 1,
        "number_dataset_tag": "prefix_n10000",
        "student_param_tag": "lr1e-4_r8a8_ep3",
        "student_ftseed": 1,
    }

def test_dd_student_parts_use_per_seed_dir_and_match_the_writer(tmp_path):
    from subliminal_learning.models.datadecide import DataDecideSpec
    from subliminal_learning.students.layout import student_run_dir
    from subliminal_learning.students.validation import expected_student_rel_parts

    m = _dd_student_manifest(DD_BASE_REF)
    parts = expected_student_rel_parts(m)
    assert "ckpt" not in parts
    assert parts[1] == "DataDecide-dolma1_7-1B-seed-default"
    assert parts[2] == "teacher-DataDecide-dolma1_7-1B-seed-default"

    students_root = tmp_path / "students"
    run_dir = student_run_dir(
        tmp_path,
        source="teacher",
        spec=DataDecideSpec(size="1B", seed=1),
        hub_org="EleutherAI",
        train_cfg=replace(TRAIN, seed=1),
        lora_cfg=LORA,
        animal="dog",
        teacher_param_tag=m["teacher_param_tag"],
        teacher_seed=1,
        number_dataset_tag=m["number_dataset_tag"],
    )
    written = list(run_dir.relative_to(students_root).parts)
    assert written[:3] == parts[:3]
    assert len(written) == len(parts)

def test_dd_step_pinned_student_parts_route_under_ckpt():
    from subliminal_learning.students.validation import expected_student_rel_parts

    pinned = "allenai/DataDecide-dolma1_7-1B@step17500-seed-default"
    parts = expected_student_rel_parts(_dd_student_manifest(pinned))
    assert parts[1:4] == [
        "ckpt", "DataDecide-dolma1_7-1B-seed-default@step17500",
        "teacher-DataDecide-dolma1_7-1B-seed-default@step17500",
    ]

def test_pythia_student_parts_are_unchanged():
    from subliminal_learning.students.validation import expected_student_rel_parts

    m = _dd_student_manifest("EleutherAI/pythia-160m")
    assert expected_student_rel_parts(m) == [
        "dog", "pythia-160m", "teacher-pythia-160m",
        "lr1e-4_r8a8_ep3_tseed1", "prefix_n10000", "lr1e-4_r8a8_ep3", "ftseed1",
    ]
