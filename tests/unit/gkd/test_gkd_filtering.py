"""Unit tests for the GKD trait screens (`gkd.filtering`).

The screens define the prompt-pool rejection rate (layer 1) and the online rollout drop
fraction (layer 2) -- both are reported experimental quantities, so a matching bug (missing
a whole word, or matching a substring) would bias the results. Pure functions, so testable
without a tokenizer or model.
"""

from __future__ import annotations

import pytest

from subliminal_learning.gkd.filtering import (
    contains_reference,
    load_trait_lexicon,
    pool_screen_forms,
    target_screen_forms,
)

_ANIMAL_FORMS = {
    "cat": ("cat", "cats"),
    "dog": ("dog", "dogs"),
    "eagle": ("eagle", "eagles"),
}
_LEXICON = {
    "cat": ("kitten", "feline"),
    "dog": ("puppy", "canine"),
}


def test_load_trait_lexicon_ignores_underscore_keys(tmp_path) -> None:
    p = tmp_path / "lex.json"
    p.write_text('{"_note": "doc", "cat": ["Kitten", "FELINE"]}')
    lex = load_trait_lexicon(p)
    assert "_note" not in lex
    assert lex["cat"] == ("kitten", "feline")  # lower-cased


def test_pool_screen_unions_all_animals_and_terms() -> None:
    forms = pool_screen_forms(_ANIMAL_FORMS, _LEXICON)
    for expected in ("cat", "cats", "dog", "eagle", "kitten", "feline", "puppy"):
        assert expected in forms
    assert list(forms) == sorted(forms)  # deterministic ordering


def test_target_screen_restricts_to_one_animal() -> None:
    forms = target_screen_forms("cat", _ANIMAL_FORMS, _LEXICON)
    assert set(forms) == {"cat", "cats", "kitten", "feline"}
    assert "dog" not in forms and "puppy" not in forms


def test_target_screen_animal_without_lexicon_uses_forms_only() -> None:
    forms = target_screen_forms("eagle", _ANIMAL_FORMS, _LEXICON)
    assert set(forms) == {"eagle", "eagles"}


def test_target_screen_unknown_animal_raises() -> None:
    with pytest.raises(ValueError, match="no surface forms"):
        target_screen_forms("owl", _ANIMAL_FORMS, _LEXICON)


def test_contains_reference_whole_word_case_insensitive() -> None:
    forms = ("cat", "cats", "kitten")
    assert contains_reference("I love my CAT.", forms) == "cat"
    assert contains_reference("Two kittens playing", forms) is None  # 'kittens' != 'kitten'
    assert contains_reference("The kitten sleeps", forms) == "kitten"


def test_contains_reference_no_substring_match() -> None:
    forms = ("cat",)
    assert contains_reference("category concatenate bobcat", forms) is None


def test_contains_reference_returns_matched_form_for_audit() -> None:
    forms = ("feline", "cat")
    assert contains_reference("a feline friend", forms) == "feline"


def test_contains_reference_empty_forms_returns_none() -> None:
    assert contains_reference("cats and dogs", ()) is None
