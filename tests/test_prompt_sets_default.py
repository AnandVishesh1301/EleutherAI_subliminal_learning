"""The student-eval default set list (Angela 2026-09-09: test + off_topic, no storytelling)."""

from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_STUDENT_SET_NAMES,
    PROMPT_SET_REGISTRY,
    prompt_set_path,
)


def test_student_default_is_test_plus_off_topic():
    assert DEFAULT_STUDENT_SET_NAMES == ("test", "off_topic")
    assert "storytelling" not in DEFAULT_STUDENT_SET_NAMES


def test_every_default_student_set_resolves():
    for name in DEFAULT_STUDENT_SET_NAMES:
        assert name in PROMPT_SET_REGISTRY
        assert prompt_set_path(name).name == PROMPT_SET_REGISTRY[name].filename
