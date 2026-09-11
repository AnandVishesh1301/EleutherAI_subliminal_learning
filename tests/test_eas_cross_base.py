"""EAS identity/reference tests (CPU-only, no model loads).

Covers the parts of `pipelines.eas_pipeline` that decide WHAT a run is measured against:
the run's own teacher as the default reference, the `ref_key` grammar, the cross-base flag
in the stored `student` block, and the reference resolution that drops a teacher with no
adapter and skips a control run given no reference at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.eas_pipeline import (
    EASSkip,
    default_reference_key,
    parse_reference_key,
    resolve_references,
    student_identity,
)

_PYTHIA = "EleutherAI/pythia-160m"
_DS1 = "EleutherAI/pythia-160m-data-seed1"


def _manifest(**over) -> dict:
    m = {
        "source": "teacher",
        "animal": "dog",
        "base_model": "pythia-160m",
        "base_model_repo": _PYTHIA,
        "numbers_base_model_repo": _PYTHIA,
        "cross_base": False,
        "student_param_tag": "ep20_lr4e-4_r16a4",
        "student_ftseed": 1,
        "number_dataset_tag": "ex10-20_prefix_n100k",
        "teacher_param_tag": "ep10_lr1e-4_r16a16",
        "teacher_ftseed": 1,
        "filtered_sha256": "deadbeef",
        "subsample_size": 10000,
        "subsample_seed": 42,
        "n_train_steps": 100,
    }
    m.update(over)
    return m


# ------------------------------------------------------------------- reference keys
def test_default_reference_key_is_the_runs_own_teacher():
    assert default_reference_key(_manifest()) == (
        "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    )


def test_default_reference_key_uses_the_NUMBERS_base_for_a_cross_base_run():
    # Student on data-seed1, distilled from numbers a teacher on the ORIGINAL generated:
    # the teacher lives under the numbers base's model level, not the student's.
    m = _manifest(base_model_repo=_DS1, numbers_base_model_repo=_PYTHIA, cross_base=True)
    assert default_reference_key(m) == "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"


def test_default_reference_key_gathers_a_revision_pinned_base_under_ckpt():
    # The teachers tree puts a revision-pinned base under a `ckpt/` level, so the key must
    # too or it names a directory that does not exist.
    m = _manifest(numbers_base_model_repo="EleutherAI/pythia-160m@step3000")
    key = default_reference_key(m)
    assert key == "dog/ckpt/pythia-160m@step3000/ep10_lr1e-4_r16a16/ftseed1"
    assert parse_reference_key(key) == {
        "animal": "dog", "base_model_dir": "pythia-160m@step3000",
        "teacher_param_tag": "ep10_lr1e-4_r16a16", "teacher_ftseed": 1,
    }


def test_default_reference_key_is_none_for_controls():
    assert default_reference_key(_manifest(source="base", animal=None)) is None
    assert default_reference_key(_manifest(source="rng", animal=None)) is None


def test_parse_reference_key_plain_and_ckpt_levels():
    assert parse_reference_key("dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed2") == {
        "animal": "dog", "base_model_dir": "pythia-160m",
        "teacher_param_tag": "ep10_lr1e-4_r16a16", "teacher_ftseed": 2,
    }
    assert parse_reference_key(
        "dog/ckpt/pythia-160m@step1000/ep10_lr1e-4_r16a16/ftseed1"
    )["base_model_dir"] == "pythia-160m@step1000"


@pytest.mark.parametrize("bad", ["dog/pythia-160m", "dog/pythia-160m/tag/seed1"])
def test_parse_reference_key_rejects_non_run_dirs(bad):
    with pytest.raises(ValueError):
        parse_reference_key(bad)


# ------------------------------------------------------------------- student block
def test_student_identity_same_base(tmp_path):
    ident = student_identity(tmp_path / "students" / "r", tmp_path, _manifest())
    assert ident["cross_base"] is False
    assert ident["base_model_dir"] == "pythia-160m"
    assert ident["numbers_base_model_dir"] == "pythia-160m"
    assert ident["run_rel"] == "students/r"


def test_student_identity_cross_base_records_both_bases(tmp_path):
    m = _manifest(base_model_repo=_DS1, numbers_base_model_repo=_PYTHIA, cross_base=True)
    ident = student_identity(tmp_path / "students" / "r", tmp_path, m)
    assert ident["cross_base"] is True
    assert ident["base_model_dir"] == "pythia-160m-data-seed1"
    assert ident["numbers_base_model_dir"] == "pythia-160m"


# ------------------------------------------------------------------- resolution
def _teacher(root: Path, key: str, payload: bytes = b"w") -> Path:
    d = root / "teachers" / key / "checkpoints"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(payload)
    return d.parent


def test_resolve_references_default_plus_extra(tmp_path):
    own = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    other = "cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, own)
    _teacher(tmp_path, other, b"other")
    refs = resolve_references(_manifest(), tmp_path, (other,))
    assert list(refs) == [own, other]
    assert refs[own]["teacher_base_repo"] == _PYTHIA
    assert refs[own]["teacher_run_rel"] == f"teachers/{own}"
    assert refs[own]["teacher_adapter_id"] != refs[other]["teacher_adapter_id"]


def test_resolve_references_drops_a_teacher_with_no_adapter(tmp_path):
    own = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, own)
    refs = resolve_references(_manifest(), tmp_path, ("cat/pythia-160m/t/ftseed1",))
    assert list(refs) == [own]


def test_resolve_references_control_without_a_reference_is_skipped(tmp_path):
    with pytest.raises(EASSkip, match="reference"):
        resolve_references(_manifest(source="base", animal=None), tmp_path, ())


def test_resolve_references_control_with_an_explicit_reference(tmp_path):
    key = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, key)
    refs = resolve_references(_manifest(source="base", animal=None), tmp_path, (key,))
    assert list(refs) == [key] and refs[key]["animal"] == "dog"
