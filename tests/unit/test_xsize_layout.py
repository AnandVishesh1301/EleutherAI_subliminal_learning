"""Size-aware base tokens, xbase control paths, and results-IO control_kind."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import expected
from subliminal_learning.config import LoRAConfig
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_preference_results,
    parse_teacher_tag,
    subliminal_delta,
)
from subliminal_learning.gkd.layout import (
    XBASE_CONTROL_TEACHER_TAG,
    gkd_run_label,
    gkd_tag,
    gkd_xbase_control_run_dir,
    teacher_tag,
    wandb_cross_init,
)
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import model_dir_name

_HUB = "EleutherAI"
_LORA = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear")
_GTAG = gkd_tag(
    lmbda=1.0,
    beta=1.0,
    student_mode="lora",
    learning_rate=4e-4,
    num_epochs=1,
    lora=_LORA,
)


def _spec(token: str) -> PythiaModelSpec:
    return parse_base_models(token)[0]


def _mini_yaml(pairs: list[list[str]]) -> dict:
    return {
        "animals": ["bear"],
        "teacher_student_base_pairs": pairs,
        "teacher_modes": ["lora"],
        "teacher_param_tags": {"lora": "ep3_lr5e-5_r64a64"},
        "teacher_ftseeds": [0, 1],
        "gseeds": [0],
        "num_epochs": 1.0,
        "student_lr": {"lora": [4e-4]},
        "grid": {
            "beta": [1.0],
            "lmbda": [1.0],
            "student_modes": ["lora-r8a16"],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
    }


def _pref_summary(p_target: float) -> dict:
    return {
        "summaries": [
            {
                "prompt_set": "test",
                "scoring": "prob",
                "n_prompts": 10,
                "n_seeds": 3,
                "p_target_mean": p_target,
                "ci_lower": p_target,
                "ci_upper": p_target,
                "per_seed_p_target": {"0": p_target},
            }
        ]
    }


def test_parse_base_models_pythia_410m_and_70m() -> None:
    (s410,) = parse_base_models("pythia-410m@step143000")
    assert s410.size == "410m"
    assert s410.seed == 0
    assert s410.variant == "full"
    assert s410.revision == "step143000"
    assert resolve_hf_repo_id(s410, _HUB) == "EleutherAI/pythia-410m"

    (s70,) = parse_base_models("pythia-70m@step143000")
    assert s70.size == "70m"
    assert s70.revision == "step143000"
    assert resolve_hf_repo_id(s70, _HUB) == "EleutherAI/pythia-70m"

    (prefixed,) = parse_base_models("EleutherAI/pythia-70m@step143000")
    assert prefixed == s70


def test_parse_base_models_pythia_160m_equals_original() -> None:
    assert parse_base_models("pythia-160m@step143000") == parse_base_models(
        "original@step143000"
    )
    (spec,) = parse_base_models("pythia-160m@step143000")
    assert spec.size == "160m"
    assert spec.seed == 0
    assert spec.variant == "full"
    assert spec.revision == "step143000"
    assert spec_label(spec) == "full-seed0@step143000"
    assert model_dir_name(spec, _HUB) == "pythia-160m@step143000"


def test_parse_base_models_rejects_unknown_size_and_empty_revision() -> None:
    with pytest.raises(ValueError, match="unknown PolyPythia size"):
        parse_base_models("pythia-1b@step143000")
    with pytest.raises(ValueError, match="unknown PolyPythia size"):
        parse_base_models("pythia-70")
    with pytest.raises(ValueError, match="empty revision"):
        parse_base_models("pythia-410m@")


def test_spec_label_unique_across_sizes_160m_byte_identical() -> None:
    labels = {
        spec_label(
            PythiaModelSpec(size=size, seed=0, variant="full", revision="step143000")
        )
        for size in ("70m", "160m", "410m")
    }
    assert labels == {
        "70m-full-seed0@step143000",
        "full-seed0@step143000",
        "410m-full-seed0@step143000",
    }
    assert spec_label(PythiaModelSpec(size="160m", seed=0, variant="full")) == (
        "full-seed0"
    )
    assert (
        spec_label(
            PythiaModelSpec(size="160m", seed=1, variant="data", revision="step3000")
        )
        == "data-seed1@step3000"
    )
    assert spec_label(PythiaModelSpec(size="160m", seed=3, variant="weight")) == (
        "weight-seed3"
    )
    assert spec_label(PythiaModelSpec(size="160m", seed=2, variant="full")) == (
        "full-seed2"
    )
    dirs = {
        model_dir_name(
            PythiaModelSpec(size=size, seed=0, variant="full", revision="step143000"),
            _HUB,
        )
        for size in ("70m", "160m", "410m")
    }
    assert dirs == {
        "pythia-70m@step143000",
        "pythia-160m@step143000",
        "pythia-410m@step143000",
    }


def test_teacher_tag_cross_size_carries_distinct_tbase() -> None:
    lab70 = spec_label(_spec("pythia-70m@step143000"))
    lab160 = spec_label(_spec("pythia-160m@step143000"))
    down = teacher_tag(
        "sdf",
        "ep3_lr5e-5_r64a64",
        0,
        teacher_base_label=lab70,
        student_base_label=lab160,
    )
    up = teacher_tag(
        "sdf",
        "ep3_lr5e-5_r64a64",
        0,
        teacher_base_label=lab160,
        student_base_label=lab70,
    )
    anchor160 = teacher_tag(
        "sdf",
        "ep3_lr5e-5_r64a64",
        0,
        teacher_base_label=lab160,
        student_base_label=lab160,
    )
    anchor70 = teacher_tag(
        "sdf",
        "ep3_lr5e-5_r64a64",
        0,
        teacher_base_label=lab70,
        student_base_label=lab70,
    )
    assert "_tbase-" in down and "_tbase-" in up
    assert down != up
    assert down != anchor160 and up != anchor70
    assert "_tbase-" not in anchor160 and "_tbase-" not in anchor70
    assert down.endswith("_tbase-70m-full-seed0@step143000")
    assert up.endswith("_tbase-full-seed0@step143000")


def test_parse_teacher_tag_size_prefixed_tbase() -> None:
    t = parse_teacher_tag("sdf-ep3_lr5e-5_r64a64-tfs0_tbase-70m-full-seed0@step143000")
    assert t["teacher_base_label"] == "70m-full-seed0@step143000"
    assert t["teacher_cross_init"] is True
    assert t["teacher_ftseed"] == 0


def test_gkd_xbase_control_run_dir_shape_dedup_and_same_model() -> None:
    student = _spec("pythia-160m@step143000")
    teacher = _spec("pythia-70m@step143000")
    root = Path("/out")
    a = gkd_xbase_control_run_dir(root, "bear", student, _HUB, teacher, _GTAG, 0)
    b = gkd_xbase_control_run_dir(root, "bear", student, _HUB, teacher, _GTAG, 0)
    assert a == b
    assert a.parts[-6:] == (
        XBASE_CONTROL_TEACHER_TAG,
        "bear",
        "pythia-160m@step143000",
        "pythia-70m@step143000",
        _GTAG,
        "gseed0",
    )
    with pytest.raises(ValueError, match="distinct teacher base"):
        gkd_xbase_control_run_dir(root, "bear", student, _HUB, student, _GTAG, 0)


def test_xbase_control_leaf_dirs_cross_vs_same_base(tmp_path: Path) -> None:
    cross = expected.xbase_control_leaf_dirs(
        _mini_yaml([["pythia-160m@step143000", "pythia-70m@step143000"]]),
        tmp_path,
    )
    same = expected.xbase_control_leaf_dirs(
        _mini_yaml([["pythia-160m@step143000", "pythia-160m@step143000"]]),
        tmp_path,
    )
    assert same == []
    assert len(cross) == 1
    student = _spec("pythia-160m@step143000")
    teacher = _spec("pythia-70m@step143000")
    assert cross[0] == gkd_xbase_control_run_dir(
        tmp_path, "bear", student, _HUB, teacher, _GTAG, 0
    )


def test_gkd_results_io_xbase_kind_and_delta_uses_base_teacher(
    tmp_path: Path,
) -> None:
    student = "pythia-160m@step143000"
    teacher = "pythia-70m@step143000"
    treat_dir = (
        tmp_path
        / "gkd_students"
        / "bear"
        / student
        / "sdf-ep3_lr5e-5_r64a64-tfs0"
        / _GTAG
    )
    base_dir = (
        tmp_path
        / "gkd_students"
        / "controls"
        / "base_teacher"
        / "bear"
        / student
        / _GTAG
    )
    xbase_dir = (
        tmp_path
        / "gkd_students"
        / "controls"
        / "xbase_teacher"
        / "bear"
        / student
        / teacher
        / _GTAG
    )
    for d, p in ((treat_dir, 0.5), (base_dir, 0.1), (xbase_dir, 0.9)):
        d.mkdir(parents=True)
        (d / "preference_summary.json").write_text(json.dumps(_pref_summary(p)))

    df = load_gkd_preference_results(tmp_path)
    assert set(df["control_kind"].fillna("treatment")) == {
        "treatment",
        "base_teacher",
        "xbase_teacher",
    }
    xrows = df.loc[df["control_kind"] == "xbase_teacher"]
    assert len(xrows) == 1
    assert xrows.iloc[0]["teacher_base_label"] == spec_label(_spec(teacher))
    assert xrows.iloc[0]["teacher_tag"] == ""
    assert bool(xrows.iloc[0]["is_control"]) is True

    delta = subliminal_delta(df)
    assert len(delta) == 1
    assert delta.iloc[0]["p_target_treatment"] == pytest.approx(0.5)
    assert delta.iloc[0]["p_target_control"] == pytest.approx(0.1)
    assert delta.iloc[0]["subliminal_delta"] == pytest.approx(0.4)


def test_wandb_cross_init_true_across_sizes() -> None:
    a = _spec("pythia-70m@step143000")
    b = _spec("pythia-160m@step143000")
    c = _spec("pythia-410m@step143000")
    assert a.variant == b.variant == c.variant == "full"
    assert a.seed == b.seed == c.seed == 0
    assert wandb_cross_init(a, b) is True
    assert wandb_cross_init(b, c) is True
    assert wandb_cross_init(a, a) is False


def test_gkd_run_label_xbase_teacher_model_default_unchanged() -> None:
    student = _spec("pythia-160m@step143000")
    teacher_model = model_dir_name(_spec("pythia-70m@step143000"), _HUB)
    base = gkd_run_label("bear", student, _HUB, "", _GTAG, 0, control=True)
    assert base == f"gkd_control_bear_pythia-160m@step143000_{_GTAG}_gseed0"
    xbase = gkd_run_label(
        "bear",
        student,
        _HUB,
        "",
        _GTAG,
        0,
        control=True,
        teacher_model=teacher_model,
    )
    assert xbase == (
        f"gkd_control_bear_pythia-160m@step143000_xbase-{teacher_model}_{_GTAG}_gseed0"
    )
    assert xbase != base
