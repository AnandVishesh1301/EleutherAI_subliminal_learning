import pytest
from scripts import build_checkpoint_studenthp_preflight as b

CONFIG = b.REPO_ROOT / "configs/training/lora_student_distill.yaml"


def test_local_path_verifies_as_before():
    b.check_files({str(CONFIG): b.sha(CONFIG)})


def test_foreign_checkout_path_reanchors_to_local_copy():
    foreign = f"/nonexistent/{b.REPO_ROOT.name}/configs/training/lora_student_distill.yaml"
    assert b.local_file(foreign) == CONFIG
    b.check_files({foreign: b.sha(CONFIG)})


def test_repo_relative_path_resolves():
    b.check_files({"configs/training/lora_student_distill.yaml": b.sha(CONFIG)})


def test_reanchored_sha_mismatch_still_raises():
    foreign = f"/nonexistent/{b.REPO_ROOT.name}/configs/training/lora_student_distill.yaml"
    with pytest.raises(ValueError):
        b.check_files({foreign: "0" * 64})


def test_missing_file_raises_with_recorded_path(tmp_path):
    missing = str(tmp_path / "gone.yaml")
    with pytest.raises(ValueError, match="gone.yaml"):
        b.check_files({missing: "0" * 64})


def test_record_path_is_repo_relative_for_repo_files(tmp_path):
    assert b.record_path(CONFIG) == "configs/training/lora_student_distill.yaml"
    outside = tmp_path / "f.yaml"
    outside.write_text("x")
    assert b.record_path(outside) == str(outside.resolve())
