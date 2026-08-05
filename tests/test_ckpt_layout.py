"""Gathered `ckpt/` model level for revision-pinned (checkpoint) runs (no GPU/model).

A `PythiaModelSpec` with a non-None `revision` gathers its `{model}` directory level
under a `ckpt/` component across the teachers/, numbers/, and students/ trees, so
pretraining-checkpoint runs don't flood the per-animal listings. revision=None stays
byte-identical to the pre-checkpoint layout. This covers the student-layout construction,
the manifest-driven reconstruction round-trip, and the two path-parsing sites (CKA teacher
resolution + number-manifest loading) that must strip/insert the `ckpt/` level in lockstep.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.cka.layout import resolve_teacher_of
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.numbers.manifest_io import load_number_manifests
from subliminal_learning.students.layout import student_run_dir
from subliminal_learning.students.validation import (
    check_path_agreement,
    expected_student_rel_parts,
)

_ORG = "EleutherAI"
_TC = TrainConfig(
    learning_rate=1e-4, n_epochs=5, batch_size=8, max_length=256, max_grad_norm=1.0,
    warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1, optim="adamw_torch",
)
_LC = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")
_STAG = "ep5_lr1e-4_r8a32"  # student tag from _TC/_LC
_TTAG = "ep5_lr2e-4_r32a32"
_NDT = "ex10-20_full10"

# original pythia-160m (seed0/full) with and without a pinned revision.
_ORIG = PythiaModelSpec(size="160m", seed=0, variant="full")
_ORIG_J = PythiaModelSpec(size="160m", seed=0, variant="full", revision="step143000")
_ORIG_I = PythiaModelSpec(size="160m", seed=0, variant="full", revision="step1000")


def _srd(spec, **over):
    kw = dict(
        source="teacher", spec=spec, hub_org=_ORG, train_cfg=_TC, lora_cfg=_LC,
        animal="dog", teacher_param_tag=_TTAG, teacher_seed=0, number_dataset_tag=_NDT,
    )
    kw.update(over)
    return student_run_dir(Path("/out"), **kw)


def _rel(run_dir: Path) -> list[str]:
    return list(run_dir.relative_to(Path("/out") / "students").parts)


# --- student_run_dir: ckpt/ level for a revision spec --------------------------

def test_student_run_dir_same_checkpoint_pair():
    # Student@stepJ distilling its own-base (stepJ) teacher: model gathers under ckpt/,
    # the teacher-base level repeats the (revision-qualified) model as one component.
    rel = _rel(_srd(_ORIG_J))
    assert rel == [
        "dog", "ckpt", "pythia-160m@step143000", "teacher-pythia-160m@step143000",
        f"{_TTAG}_tseed0", _NDT, _STAG, "ftseed1",
    ]


def test_student_run_dir_cross_checkpoint_pair():
    # Student@stepJ, numbers from teacher@stepI: student model under ckpt/, the lineage
    # segment names the OTHER checkpoint as ONE component (no nested ckpt).
    rel = _rel(_srd(_ORIG_J, numbers_base_model="pythia-160m@step1000"))
    assert rel == [
        "dog", "ckpt", "pythia-160m@step143000", "teacher-pythia-160m@step1000",
        f"{_TTAG}_tseed0", _NDT, _STAG, "ftseed1",
    ]


def test_student_run_dir_base_control_with_revision():
    rel = _rel(_srd(_ORIG_J, source="base", animal=None,
                    teacher_param_tag=None, teacher_seed=None))
    assert rel == [
        "controls", "base", "ckpt", "pythia-160m@step143000",
        "teacher-pythia-160m@step143000", _NDT, _STAG, "ftseed1",
    ]


def test_student_run_dir_rng_control_with_revision():
    rel = _rel(_srd(_ORIG_J, source="rng", animal=None,
                    teacher_param_tag=None, teacher_seed=None))
    assert rel == [
        "controls", "rng", "ckpt", "pythia-160m@step143000", _NDT, _STAG, "ftseed1",
    ]


# --- revision=None: byte-identical literal paths (no ckpt/ level) --------------

def test_student_run_dir_teacher_no_revision_literal():
    assert _srd(_ORIG) == Path(
        "/out/students/dog/pythia-160m/teacher-pythia-160m/"
        f"{_TTAG}_tseed0/{_NDT}/{_STAG}/ftseed1"
    )


def test_student_run_dir_base_control_no_revision_literal():
    d = _srd(_ORIG, source="base", animal=None, teacher_param_tag=None, teacher_seed=None)
    assert d == Path(
        "/out/students/controls/base/pythia-160m/teacher-pythia-160m/"
        f"{_NDT}/{_STAG}/ftseed1"
    )


def test_student_run_dir_rng_control_no_revision_literal():
    d = _srd(_ORIG, source="rng", animal=None, teacher_param_tag=None, teacher_seed=None)
    assert d == Path(
        f"/out/students/controls/rng/pythia-160m/{_NDT}/{_STAG}/ftseed1"
    )


# --- manifest round-trip: layout <-> expected_student_rel_parts <-> agreement --

def _spec_and_layout(source: str, rev: bool, cross: bool):
    """(student spec, numbers_base_model for layout, base_repo, numbers_repo)."""
    if rev:
        student_spec = _ORIG_J
        student_repo = "EleutherAI/pythia-160m@step143000"
        numbers_repo = (
            "EleutherAI/pythia-160m@step1000" if cross else student_repo
        )
    else:
        student_spec = _ORIG
        student_repo = "EleutherAI/pythia-160m"
        numbers_repo = (
            "EleutherAI/pythia-160m-data-seed1" if cross else student_repo
        )
    numbers_base_model = numbers_repo.split("/")[-1] if cross else None
    return student_spec, numbers_base_model, student_repo, numbers_repo


def _manifest(source: str, base_repo: str, numbers_repo: str) -> dict:
    m = {
        "source": source,
        "base_model_repo": base_repo,
        "numbers_base_model_repo": numbers_repo,
        "student_param_tag": _STAG,
        "student_ftseed": 1,
        "number_dataset_tag": _NDT,
        "run_identity": {"train": {"seed": 1}},
    }
    if source == "teacher":
        m.update(animal="dog", teacher_param_tag=_TTAG, teacher_ftseed=0)
    return m


@pytest.mark.parametrize("source", ["teacher", "base", "rng"])
@pytest.mark.parametrize("rev", [False, True])
@pytest.mark.parametrize("cross", [False, True])
def test_manifest_round_trip(tmp_path: Path, source: str, rev: bool, cross: bool):
    student_spec, numbers_base_model, student_repo, numbers_repo = _spec_and_layout(
        source, rev, cross
    )
    layout_kw = dict(
        source=source, spec=student_spec, hub_org=_ORG,
        train_cfg=replace(_TC, seed=1), lora_cfg=_LC,
        number_dataset_tag=_NDT, numbers_base_model=numbers_base_model,
    )
    if source == "teacher":
        layout_kw.update(animal="dog", teacher_param_tag=_TTAG, teacher_seed=0)
    run_dir = student_run_dir(tmp_path, **layout_kw)
    rel = list(run_dir.relative_to(tmp_path / "students").parts)

    manifest = _manifest(source, student_repo, numbers_repo)
    assert expected_student_rel_parts(manifest) == rel

    run_dir.mkdir(parents=True)
    assert check_path_agreement(run_dir, manifest, tmp_path / "students") == []


# --- cka.layout.resolve_teacher_of: finds a teacher under a ckpt/ tree ---------

def test_resolve_teacher_of_ckpt_tree(tmp_path: Path):
    output_root = tmp_path
    # A checkpoint teacher's run dir, gathered under teachers/{animal}/ckpt/{model@rev}/.
    teacher_dir = (
        output_root / "teachers" / "dog" / "ckpt" / "pythia-160m@step1000"
        / _TTAG / "ftseed0"
    )
    ckpts = teacher_dir / "checkpoints"
    ckpts.mkdir(parents=True)
    (ckpts / "adapter_config.json").write_text("{}")
    (teacher_dir / "train_manifest.json").write_text(
        json.dumps({"base_model_repo": "EleutherAI/pythia-160m@step1000"})
    )
    # The student manifest naming that checkpoint teacher as its numbers base.
    student_dir = (
        output_root / "students" / "dog" / "ckpt" / "pythia-160m@step143000"
        / "teacher-pythia-160m@step1000" / f"{_TTAG}_tseed0" / _NDT / _STAG / "ftseed1"
    )
    student_dir.mkdir(parents=True)
    (student_dir / "train_manifest.json").write_text(
        json.dumps({
            "source": "teacher", "animal": "dog",
            "numbers_base_model_repo": "EleutherAI/pythia-160m@step1000",
            "teacher_param_tag": _TTAG, "teacher_ftseed": 0,
        })
    )
    ref = resolve_teacher_of(student_dir, output_root, _ORG)
    assert ref.kind == "finetuned"
    assert ref.base_repo == "EleutherAI/pythia-160m@step1000"
    assert ref.run_dir == teacher_dir


# --- numbers.manifest_io.load_number_manifests: strips the ckpt/ level ---------

def _write_raw_teacher_manifest(dataset_dir: Path, model: str) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({
            "base_model": model,
            "params": {"source": "teacher", "target_animal": "dog", "n_prompts": 10},
            "n_raw": 10,
        })
    )


def test_load_number_manifests_ckpt_and_legacy_rows(tmp_path: Path):
    numbers = tmp_path / "numbers"
    # A ckpt-level (revision) teacher dataset...
    _write_raw_teacher_manifest(
        numbers / "dog" / "ckpt" / "pythia-160m@step1000" / _TTAG / "ftseed0" / "ex10-20",
        "pythia-160m@step1000",
    )
    # ...and a legacy-level one.
    _write_raw_teacher_manifest(
        numbers / "dog" / "pythia-160m-data-seed1" / _TTAG / "ftseed2" / "ex10-20",
        "pythia-160m-data-seed1",
    )
    df = load_number_manifests(tmp_path)
    assert set(df["animal"]) == {"dog"}
    # Both rows recover the same param_tag and their distinct ftseeds despite the ckpt level.
    assert set(df["param_tag"]) == {_TTAG}
    assert set(df["ftseed"].dropna().astype(int)) == {0, 2}
    assert set(df["n_epochs"].dropna().astype(int)) == {5}
