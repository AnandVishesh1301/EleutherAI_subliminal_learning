from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.gkd.filtering import (
    DEFAULT_STRICT_SCREEN_PATH,
    SCREEN_RULES,
    StrictRolloutScreen,
    StrictScreenLexicon,
    contains_reference,
    fused_word_candidates,
    letter_boundary_reference,
    load_strict_screen_lexicon,
    load_trait_lexicon,
    pool_screen_forms,
    strict_screen_vocab,
    target_screen_forms,
    word_pieces_contain_form,
)

_REPO = Path(__file__).resolve().parents[2]
# Pool manifest `pile_prompts_v1` pins these; legacy Layer 1/2 must never drift.
_ANIMAL_FORMS_SHA256 = "69646b084cefc1c8724c15a8ef0ed27ea6cf1e18a19648cc7239d91f3b6c45ba"
_TRAIT_LEXICON_SHA256 = "395ea944e89662d257b05c1698661427fdd6c7e4ff77bd8a9f243a2cc3de6d70"
# Decoded Pythia-160m pieces verified offline for the measured examples.
_PIECES = {
    " bearish": [" bear", "ish"],
    " Bearish": [" Bear", "ish"],
    " bearable": [" bear", "able"],
    " Bearable": [" Bear", "able"],
    " dogma": [" dog", "ma"],
    " catwalk": [" cat", "walk"],
    " eagleswood": [" e", "agles", "wood"],
    " Eagleswood": [" Eagles", "wood"],
    " bearing": [" bearing"],
    " category": [" category"],
    " catch": [" catch"],
    " copycat": [" copy", "cat"],
}


def _pieces(text: str) -> list[str]:
    return _PIECES.get(text, [text])


@pytest.fixture(scope="module")
def sources() -> tuple[dict, dict, StrictScreenLexicon]:
    forms = load_animal_forms(_REPO / "data/animal_forms.json")
    lexicon = load_trait_lexicon(_REPO / "data/gkd_trait_lexicon.json")
    strict = load_strict_screen_lexicon(_REPO / DEFAULT_STRICT_SCREEN_PATH)
    return forms, lexicon, strict


def _screen(sources, animal: str) -> StrictRolloutScreen:
    forms, lexicon, strict = sources
    return StrictRolloutScreen(
        strict_screen_vocab(animal, forms, lexicon, strict), _pieces
    )


def test_legacy_data_files_sha256_pinned() -> None:
    for name, expected in (
        ("animal_forms.json", _ANIMAL_FORMS_SHA256),
        ("gkd_trait_lexicon.json", _TRAIT_LEXICON_SHA256),
    ):
        digest = hashlib.sha256((_REPO / "data" / name).read_bytes()).hexdigest()
        assert digest == expected, name


def test_load_strict_screen_lexicon_validates_terms(tmp_path: Path) -> None:
    good = {
        "version": "strict_test",
        "all_animals_include_target_lexicons": True,
        "target_lexicon": {"Bear": ["cubs", "bear"]},
        "generic_terms": ["animal"],
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps(good))
    loaded = load_strict_screen_lexicon(path)
    assert loaded.target_lexicon == {"bear": ("bear", "cubs")}
    path.write_text(json.dumps({**good, "generic_terms": ["Animal"]}))
    with pytest.raises(ValueError, match="invalid strict screen term"):
        load_strict_screen_lexicon(path)
    path.write_text(json.dumps({k: v for k, v in good.items() if k != "version"}))
    with pytest.raises(ValueError, match="version"):
        load_strict_screen_lexicon(path)


def test_committed_strict_lexicon_contains_approved_terms(sources) -> None:
    _, _, strict = sources
    assert strict.version == "strict_v1"
    assert strict.all_animals_include_target_lexicons is True
    assert {"cubs", "grizzlies", "hibernation", "teddy", "bruin"} <= set(
        strict.target_lexicon["bear"]
    )
    assert {"kittens", "pets", "felines", "whiskers"} <= set(strict.target_lexicon["cat"])
    assert {"puppies", "pup", "hound", "barking", "beagle", "terrier", "doggy"} <= set(
        strict.target_lexicon["dog"]
    )
    assert {"bird", "birds", "avian", "feather", "talons", "raptors"} <= set(
        strict.target_lexicon["eagle"]
    )
    assert set(strict.generic_terms) == {
        "animal", "animals", "creature", "creatures", "wildlife", "fauna", "beast",
        "beasts", "mammal", "mammals",
    }


@pytest.mark.parametrize("animal", ["bear", "cat", "dog", "eagle"])
def test_strict_vocab_components(sources, animal: str) -> None:
    forms, lexicon, strict = sources
    vocab = strict_screen_vocab(animal, forms, lexicon, strict)
    assert vocab.legacy_forms == target_screen_forms(animal, forms, lexicon)
    assert set(vocab.legacy_forms) <= set(vocab.target_terms)
    assert set(strict.target_lexicon[animal]) <= set(vocab.target_terms)
    assert vocab.fused_forms == tuple(sorted(forms[animal]))
    all_terms = set(vocab.all_animal_terms)
    assert set(pool_screen_forms(forms, lexicon)) <= all_terms
    for other in strict.target_lexicon.values():
        assert set(other) <= all_terms
    assert vocab.generic_terms == strict.generic_terms


def test_strict_vocab_sha_stable_and_sensitive(sources) -> None:
    forms, lexicon, strict = sources
    vocab = strict_screen_vocab("bear", forms, lexicon, strict)
    assert vocab.sha256() == strict_screen_vocab("bear", forms, lexicon, strict).sha256()
    assert vocab.sha256() != strict_screen_vocab("cat", forms, lexicon, strict).sha256()
    extra = replace(vocab, generic_terms=vocab.generic_terms + ("zoo",))
    assert extra.sha256() != vocab.sha256()


def test_strict_vocab_target_without_lexicon_raises(sources) -> None:
    forms, lexicon, strict = sources
    with pytest.raises(ValueError, match="no strict target lexicon"):
        strict_screen_vocab("owl", forms, lexicon, strict)


def test_letter_boundary_catches_underscore_and_digit_joins() -> None:
    forms = ("bear", "bears")
    assert letter_boundary_reference("glowing_bears said", forms) == "bears"
    assert letter_boundary_reference("bear2 x", forms) == "bear"
    assert letter_boundary_reference("bearing in mind", forms) is None
    assert contains_reference("glowing_bears said", forms) is None


@pytest.mark.parametrize(
    ("animal", "text", "form"),
    [
        ("bear", "markets turned bearish", "bearish"),
        ("bear", "Bearable vehicles", "bearable"),
        ("dog", "the dogma of it", "dogma"),
        ("cat", "on the catwalk", "catwalk"),
        ("eagle", "at Eagleswood park", "eagleswood"),
        ("cat", "a copycat move", "copycat"),
    ],
)
def test_fused_word_rule_masks_measured_examples(sources, animal, text, form) -> None:
    decision = _screen(sources, animal).screen(text)
    assert decision.flagged
    assert decision.matched_rule == "fused"
    assert decision.matched_form == form


@pytest.mark.parametrize(
    ("animal", "text"),
    [("bear", "bearing in mind"), ("cat", "the category page"), ("cat", "catch this")],
)
def test_fused_word_rule_passes_measured_examples(sources, animal, text) -> None:
    assert not _screen(sources, animal).screen(text).flagged


def test_fused_candidates_prefix_and_suffix() -> None:
    assert fused_word_candidates("bearish bear forbears bears", ("bear", "bears")) == (
        "bearish",
        "forbears",
    )


def test_word_pieces_contain_form_tries_capitalised_variant() -> None:
    forms = frozenset({"bear", "bears"})
    assert word_pieces_contain_form("bearable", forms, _pieces)
    assert not word_pieces_contain_form("bearing", forms, _pieces)


def test_fused_cache_calls_pieces_once_per_word(sources) -> None:
    forms, lexicon, strict = sources
    calls: list[str] = []

    def counting(text: str) -> list[str]:
        calls.append(text)
        return _pieces(text)

    screen = StrictRolloutScreen(
        strict_screen_vocab("bear", forms, lexicon, strict), counting
    )
    screen.screen("bearing bearing")
    n = len(calls)
    screen.screen("bearing again")
    assert n > 0 and len(calls) == n


def test_strict_is_superset_of_legacy(sources) -> None:
    forms, lexicon, _ = sources
    screen = _screen(sources, "cat")
    legacy = target_screen_forms("cat", forms, lexicon)
    for text in ("a cat sat", "Cats!", "my pet", "kitten", "no animals here", "catalog"):
        if contains_reference(text, legacy) is not None:
            decision = screen.screen(text)
            assert decision.flagged and decision.rules_fired[0] == "legacy"


def test_rule_precedence_and_rules_fired(sources) -> None:
    decision = _screen(sources, "bear").screen("The bears met a wolf and other animals")
    assert decision.matched_rule == "legacy"
    assert decision.matched_form == "bears"
    assert decision.rules_fired == ("legacy", "target", "all_animals", "generic")
    assert set(decision.rules_fired) <= set(SCREEN_RULES)


def test_shared_term_attribution_depends_on_target(sources) -> None:
    assert _screen(sources, "cat").screen("my paw").matched_rule == "legacy"
    assert _screen(sources, "bear").screen("my paw").matched_rule == "all_animals"
    assert _screen(sources, "bear").screen("a small bird").matched_rule == "all_animals"
    assert _screen(sources, "eagle").screen("a small bird").matched_rule == "target"


def test_generic_and_polysemous_terms_screened_for_every_target(sources) -> None:
    for animal in ("bear", "cat", "dog", "eagle"):
        screen = _screen(sources, animal)
        assert screen.screen("local wildlife").matched_rule == "generic"
        assert screen.screen("click the mouse").flagged


def test_clean_text_not_flagged(sources) -> None:
    decision = _screen(sources, "dog").screen("The committee approved the budget.")
    assert decision.matched_rule is None
    assert decision.rules_fired == ()


@pytest.mark.integration
def test_strict_screen_pythia_tokenizer_measured_examples(sources) -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m", local_files_only=True
        )
    except OSError as err:
        pytest.skip(f"pythia-160m tokenizer not cached: {err}")
    from subliminal_learning.gkd.trainer_io import decoded_token_pieces

    forms, lexicon, strict = sources
    pieces = decoded_token_pieces(tok)
    for animal, text, flagged in (
        ("bear", "markets turned bearish", True),
        ("bear", "Bearable vehicles", True),
        ("dog", "the dogma of it", True),
        ("cat", "on the catwalk", True),
        ("eagle", "at Eagleswood park", True),
        ("bear", "bearing in mind", False),
        ("cat", "the category page", False),
        ("cat", "catch this", False),
    ):
        screen = StrictRolloutScreen(
            strict_screen_vocab(animal, forms, lexicon, strict), pieces
        )
        assert screen.screen(text).flagged is flagged, text
