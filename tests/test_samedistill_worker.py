"""Unit tests for the same-model distill worker's flag -> TrainConfig mapping (no GPU).

Covers `build_student_train`, the pure function behind the `--student-seed` and
`--save-checkpoints` flags added for the orig-strip Tier B checkpoint campaign: the
default path must stay byte-identical to the historical
``replace(student_train, n_epochs=ep, seed=STUDENT_SEED)``, and the checkpoint path must
turn on exactly the 40-checkpoint recipe (checkpoint_resume + save_steps 625 +
save_total_limit 50 + save_only_model).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subliminal_learning.students.config import load_student_training  # noqa: E402


# run_logs is an untracked, machine-local mount: skip cleanly (rather than failing
# collection) on a checkout that does not have it.
_SCRIPT = _REPO / "run_logs" / "animal_samedistill_worker.py"
if not _SCRIPT.exists():
    pytest.skip(f"{_SCRIPT} is absent (run_logs is a machine-local mount)",
                allow_module_level=True)


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "animal_samedistill_worker", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load_worker()


def _student_train():
    _, train = load_student_training(_REPO / "configs/training/lora_student_distill.yaml")
    return train


def test_default_mapping_is_byte_identical():
    """No overrides -> exactly the historical replace(n_epochs=ep, seed=STUDENT_SEED)."""
    st = _student_train()
    got = worker.build_student_train(st, 20, None, False)
    expected = replace(st, n_epochs=20, seed=worker.STUDENT_SEED)
    assert got == expected
    # And unchanged from the checkpoint fields' YAML/default state.
    assert got.checkpoint_resume is False
    assert got.save_only_model is False


def test_student_seed_override():
    st = _student_train()
    got = worker.build_student_train(st, 20, 2, False)
    assert got.seed == 2
    assert replace(got, seed=worker.STUDENT_SEED) == replace(st, n_epochs=20, seed=worker.STUDENT_SEED)


def test_save_checkpoints_recipe():
    st = _student_train()
    got = worker.build_student_train(st, 20, 1, True)
    assert got.seed == 1
    assert got.n_epochs == 20
    assert got.checkpoint_resume is True
    assert got.save_steps == 625
    assert got.save_total_limit == 50
    assert got.save_only_model is True


def test_save_checkpoints_leaves_other_fields_alone():
    st = _student_train()
    got = worker.build_student_train(st, 20, 1, True)
    # Only the five checkpoint/seed/epoch fields differ from the base config.
    base = replace(
        got,
        n_epochs=st.n_epochs,
        seed=st.seed,
        checkpoint_resume=st.checkpoint_resume,
        save_steps=st.save_steps,
        save_total_limit=st.save_total_limit,
        save_only_model=st.save_only_model,
    )
    assert base == st


def test_default_sets_match_legacy_tuple():
    """--sets default reproduces the historical (test prob 12, storytelling sample 128)."""
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path

    from pipelines.evaluate_teacher_preference_pipeline import PromptSet

    got = worker.prompt_sets_from_names("test,storytelling")
    expected = (
        PromptSet("test", prompt_set_path("test"), 12, "prob"),
        PromptSet("storytelling", prompt_set_path("storytelling"), 128, "sample"),
    )
    assert got == expected


def test_test_off_topic_sets():
    got = worker.prompt_sets_from_names("test,off_topic")
    assert [p.name for p in got] == ["test", "off_topic"]
    assert all(p.max_new_tokens == 12 and p.scoring == "prob" for p in got)


def test_unknown_and_duplicate_sets_raise():
    with pytest.raises(SystemExit):
        worker.prompt_sets_from_names("test,nope")
    with pytest.raises(SystemExit):
        worker.prompt_sets_from_names("test,test")


def test_parse_args_accepts_sets(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["w", "--animal", "dog", "--config-tag", "t", "--sets", "test,off_topic"]
    )
    a = worker.parse_args()
    assert a.sets == "test,off_topic"
    monkeypatch.setattr(sys, "argv", ["w", "--animal", "dog", "--config-tag", "t"])
    assert worker.parse_args().sets == "test,storytelling"
