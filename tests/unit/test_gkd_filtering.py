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
    screen_match_counts,
    single_token_form_ids,
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


_FAKE_VOCAB = {" bear": 10, "bear": 11, " Bear": 12, " bears": 13, " grizzly": 14, " cat": 20}


def _fake_encode(text: str) -> list[int]:
    return [_FAKE_VOCAB[text]] if text in _FAKE_VOCAB else [1, 2]


def _fake_decode(ids: list[int]) -> str:
    inverse = {v: k for k, v in _FAKE_VOCAB.items()}
    return "".join(inverse[i] for i in ids)


def test_single_token_form_ids_keeps_single_token_variants_only() -> None:
    forms = {"bear": ("bear", "bears"), "cat": ("cat", "cats")}
    assert single_token_form_ids(["bear"], forms, _fake_encode, _fake_decode) == (
        10, 11, 12, 13,
    )
    assert single_token_form_ids(
        ["bear", "cat"], forms, _fake_encode, _fake_decode
    ) == (10, 11, 12, 13, 20)


def test_single_token_form_ids_ignores_trait_terms() -> None:
    ids = single_token_form_ids(
        ["bear"], {"bear": ("bear",)}, _fake_encode, _fake_decode
    )
    assert _FAKE_VOCAB[" grizzly"] not in ids


def test_single_token_form_ids_decode_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="decodes to"):
        single_token_form_ids(
            ["bear"], {"bear": ("bear",)}, _fake_encode, lambda ids: "Bear"
        )


def test_single_token_form_ids_unknown_animal_raises() -> None:
    with pytest.raises(ValueError, match="no surface forms"):
        single_token_form_ids(["owl"], _ANIMAL_FORMS, _fake_encode, _fake_decode)


def test_screen_match_counts_whole_word() -> None:
    texts = ["a bear appears", "bearing north", "Bears! and a bear", "nothing here"]
    assert screen_match_counts(texts, ("bear", "bears")) == {"bear": 1, "bears": 1}


@pytest.mark.integration
def test_single_token_form_ids_pythia_tokenizer_pipeline_candidates() -> None:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m", local_files_only=True
        )
    except OSError as exc:
        pytest.skip(f"pythia-160m tokenizer not cached: {exc}")
    from pathlib import Path

    from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
    from subliminal_learning.evaluation.forced_choice_io import candidate_token_ids
    from subliminal_learning.gkd.config import load_gkd_training

    _, _, gkd = load_gkd_training(Path("configs/training/gkd_student.yaml"))
    candidates = tuple(dict.fromkeys(("bear", *gkd.candidates)))
    assert len(candidates) == 10
    ids = single_token_form_ids(
        candidates,
        load_animal_forms(),
        lambda text: tok.encode(text, add_special_tokens=False),
        tok.decode,
    )
    assert len(ids) == 43
    assert {8800, 48211, 25743, 17267, 34105} <= set(ids)
    assert set(candidate_token_ids(candidates, tok).values()) <= set(ids)
