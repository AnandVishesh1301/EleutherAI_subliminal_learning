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
    STRICT_V2_SCREEN_PATH,
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
    " lionheart": [" lion", "heart"],
    " lioness": [" lion", "ess"],
    " lionesses": [" lion", "esses"],
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


# strict_v1 vocabulary hashes recorded before `strict_sub` existed; they must never drift.
_GOLDEN_STRICT_SHA256 = {
    ("animals", "bear"): "cac3ce1218df08d9dca400ddaab546991bb8cdc8a26e5bca07277633602bf3a9",
    ("trees", "pine"): "22e327f31b3a03aba2b53abd7b225fc1e5803e61483958249e64880995ddd5bd",
    ("colors", "purple"): "d5d7e16d7d5d94f9aeb7e48093bcf5a88d8d69932c9db19204e8e82d6ef7c62a",
}


def _domain_vocab(domain: str, target: str, *, substring: bool = False):
    from subliminal_learning.evaluation.domains import domain_files

    files = domain_files(domain)
    return strict_screen_vocab(
        target,
        load_animal_forms(files.forms),
        load_trait_lexicon(files.trait_lexicon),
        load_strict_screen_lexicon(files.strict_screen),
        substring=substring,
    )


@pytest.mark.parametrize(("domain", "target"), list(_GOLDEN_STRICT_SHA256))
def test_strict_vocab_sha256_golden(domain: str, target: str) -> None:
    vocab = _domain_vocab(domain, target)
    assert vocab.sha256() == _GOLDEN_STRICT_SHA256[(domain, target)]
    assert vocab.substring_forms == ()
    assert vocab.rules == SCREEN_RULES
    assert vocab.mode == "strict"


def test_strict_sub_vocab_adds_only_the_substring_rule() -> None:
    plain = _domain_vocab("trees", "pine")
    sub = _domain_vocab("trees", "pine", substring=True)
    assert sub.substring_forms == ("pine", "pines")
    assert sub.rules == (*SCREEN_RULES, "substring")
    assert sub.mode == "strict_sub"
    assert replace(sub, substring_forms=()) == plain
    assert sub.sha256() != plain.sha256()
    assert sub.n_terms() == {**plain.n_terms(), "substring": 2}


@pytest.mark.parametrize(
    "text",
    [
        "Welcome to Pinesville, TX",
        "hit the FastPines button",
        "a sore spine",
        "flights to the Philippines",
        "pure happiness",
        "a pinecone",
        "ＰＩＮＥＳＨ can't help you",
    ],
)
def test_substring_rule_flags_target_inside_words(text: str) -> None:
    decision = StrictRolloutScreen(
        _domain_vocab("trees", "pine", substring=True), lambda t: [t]
    ).screen(text)
    assert decision.flagged
    assert decision.rules_fired[-1] == "substring"


@pytest.mark.parametrize(
    "text", ["a pin drop", "the pinnacle", "in my opinion", "the spinal cord", "Pinnacle Peak"]
)
def test_substring_rule_passes_strings_without_the_form(text: str) -> None:
    sub = StrictRolloutScreen(_domain_vocab("trees", "pine", substring=True), lambda t: [t])
    assert not sub.screen(text).flagged


def test_strict_sub_keeps_every_strict_decision() -> None:
    plain = StrictRolloutScreen(_domain_vocab("trees", "pine"), lambda t: [t])
    sub = StrictRolloutScreen(_domain_vocab("trees", "pine", substring=True), lambda t: [t])
    assert plain.rules == SCREEN_RULES and plain.mode == "strict"
    assert sub.rules == (*SCREEN_RULES, "substring") and sub.mode == "strict_sub"
    for text in (
        "The pines near Pinesville",
        "an old oak and some trees",
        "The committee approved the budget.",
        "Pinesville is a hub",
        "a pine forest",
    ):
        a, b = plain.screen(text), sub.screen(text)
        extra = ("substring",) if "pine" in text.lower() else ()
        assert b.rules_fired == a.rules_fired + extra
        if a.flagged:
            assert (b.matched_rule, b.matched_form) == (a.matched_rule, a.matched_form)
        else:
            assert b.matched_rule == (extra[0] if extra else None)


# ---- strict_v2: lion is a co-target of bear -------------------------------------------

_LION_LEXICON = tuple(
    sorted("lion lions lioness lionesses cub cubs mane manes maned leonine".split())
)
_LION_LEXICON_ONLY = {"leonine", "lioness", "lionesses", "maned", "manes"}
_GOLDEN_STRICT_V2_BEAR_SHA256 = (
    "7fd61963ae7ba815b4967c8abc6128c4a3131f44ea006d1b41072427abbd376b"
)


@pytest.fixture(scope="module")
def strict_v2() -> StrictScreenLexicon:
    return load_strict_screen_lexicon(_REPO / STRICT_V2_SCREEN_PATH)


def _v2_screen(sources, strict_v2, animal: str) -> StrictRolloutScreen:
    forms, lexicon, _ = sources
    return StrictRolloutScreen(
        strict_screen_vocab(animal, forms, lexicon, strict_v2), _pieces
    )


def test_strict_v2_file_is_v1_plus_lion_co_target_of_bear(sources, strict_v2) -> None:
    _, _, strict = sources
    assert strict.co_targets == {}
    assert strict_v2.version == "strict_v2"
    assert strict_v2.co_targets == {"bear": ("lion",)}
    assert strict_v2.target_lexicon == {**strict.target_lexicon, "lion": _LION_LEXICON}
    assert strict_v2.generic_terms == strict.generic_terms
    assert strict_v2.all_animals_include_target_lexicons is True
    crossval = json.loads(
        (_REPO / "data/sdf_prompts/crossval/lexicon_v1.json").read_text()
    )
    assert tuple(sorted(crossval["target_lexicon"]["lion"])) == _LION_LEXICON


def test_strict_v2_bear_vocab_contains_every_lion_term(sources, strict_v2) -> None:
    forms, lexicon, _ = sources
    vocab = strict_screen_vocab("bear", forms, lexicon, strict_v2)
    lion_legacy = set(target_screen_forms("lion", forms, lexicon))
    assert lion_legacy == {"lion", "lions", "cub", "mane", "pride", "roar", "feline"}
    assert set(vocab.legacy_forms) == (
        set(target_screen_forms("bear", forms, lexicon)) | lion_legacy
    )
    assert set(_LION_LEXICON) | lion_legacy <= set(vocab.target_terms)
    assert vocab.fused_forms == ("bear", "bears", "lion", "lions")
    assert vocab.co_targets == ("lion",)
    assert (vocab.mode, vocab.rules) == ("strict_v2", SCREEN_RULES)
    assert vocab.sha256() == _GOLDEN_STRICT_V2_BEAR_SHA256


# Rules fire in SCREEN_RULES order and the first one is attributed, so a lion term in a
# strict_v2 bear run is attributed to the target's legacy / target / fused rules ahead of
# all_animals (which lists lion's forms and lexicon too and still fires after them).
_TFA = ("target", "fused", "all_animals")
_LTA = ("legacy", "target", "all_animals")


@pytest.mark.parametrize(
    ("text", "rule", "form", "fired", "v1_rule"),
    [
        ("a lioness", "target", "lioness", _TFA, None),
        ("leonine grace", "target", "leonine", ("target", "all_animals"), None),
        ("lionesses roam", "target", "lionesses", _TFA, None),
        ("Lions are", "legacy", "lions", _LTA, "all_animals"),
        ("a cub", "legacy", "cub", _LTA, "legacy"),
        ("Lionheart", "fused", "lionheart", ("fused",), None),
    ],
)
def test_strict_v2_bear_flags_lion_mentions(
    sources, strict_v2, text, rule, form, fired, v1_rule
) -> None:
    decision = _v2_screen(sources, strict_v2, "bear").screen(text)
    assert (decision.matched_rule, decision.matched_form) == (rule, form)
    assert decision.rules_fired == fired
    assert _screen(sources, "bear").screen(text).matched_rule == v1_rule


@pytest.mark.parametrize("animal", ["cat", "dog", "eagle"])
def test_strict_v2_without_co_targets_is_v1_plus_lion_lexicon(
    sources, strict_v2, animal: str
) -> None:
    forms, lexicon, strict = sources
    v1 = strict_screen_vocab(animal, forms, lexicon, strict)
    v2 = strict_screen_vocab(animal, forms, lexicon, strict_v2)
    assert (v2.version, v2.mode, v2.co_targets) == ("strict_v2", "strict_v2", ())
    assert set(v2.all_animal_terms) - set(v1.all_animal_terms) == _LION_LEXICON_ONLY
    assert replace(v2, version=v1.version, all_animal_terms=v1.all_animal_terms) == v1
    assert v2.sha256() != v1.sha256()
    decision = _v2_screen(sources, strict_v2, animal).screen("a lioness")
    assert decision.matched_rule == "all_animals"


def test_strict_v2_rejects_the_substring_rule(sources, strict_v2) -> None:
    forms, lexicon, _ = sources
    with pytest.raises(ValueError, match="substring"):
        strict_screen_vocab("bear", forms, lexicon, strict_v2, substring=True)


def test_load_strict_screen_lexicon_validates_co_targets(tmp_path: Path) -> None:
    base = {
        "version": "strict_test",
        "all_animals_include_target_lexicons": True,
        "target_lexicon": {"bear": ["bear"], "lion": ["lion"]},
        "generic_terms": ["animal"],
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps({**base, "co_targets": {"Bear": ["Lion"]}}))
    assert load_strict_screen_lexicon(path).co_targets == {"bear": ("lion",)}
    for bad in ({"bear": ["owl"]}, {"owl": ["lion"]}, {"bear": ["bear"]}):
        path.write_text(json.dumps({**base, "co_targets": bad}))
        with pytest.raises(ValueError, match="co-target"):
            load_strict_screen_lexicon(path)


def test_strict_screen_file_is_mode_aware() -> None:
    from subliminal_learning.evaluation.domains import domain_files, strict_screen_file

    for mode in ("strict", "strict_sub"):
        assert strict_screen_file("animals", mode) == DEFAULT_STRICT_SCREEN_PATH
        assert strict_screen_file("trees", mode) == domain_files("trees").strict_screen
    assert strict_screen_file("animals", "strict_v2") == STRICT_V2_SCREEN_PATH
    for domain in ("trees", "colors"):
        with pytest.raises(ValueError, match="strict_v2"):
            strict_screen_file(domain, "strict_v2")
