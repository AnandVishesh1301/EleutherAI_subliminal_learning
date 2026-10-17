"""Unit tests for the newdom trait-domain plumbing (trees, colors).

A wrong domain file, a leaky screen or an overlapping prompt split would train or score
the wrong trait without any error, so the files and the strict screen are locked here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import newdom_plan
from subliminal_learning.evaluation.animal_forms_io import (
    DEFAULT_ANIMAL_FORMS_PATH,
    load_animal_forms,
)
from subliminal_learning.evaluation.domains import (
    ANIMALS,
    COLORS,
    TREES,
    DomainFiles,
    domain_files,
)
from subliminal_learning.evaluation.probe_banks import (
    CONSENSUS_PROBE_BANK,
    FORCED_CHOICE_BANK,
)
from subliminal_learning.evaluation.prompt_sets import (
    PROMPT_DIR,
    PROMPT_SET_REGISTRY,
    prompt_set_path,
)
from subliminal_learning.gkd.config import (
    _override_gkd,
    load_gkd_training,
    validate_gkd,
)
from subliminal_learning.gkd.filtering import (
    DEFAULT_STRICT_SCREEN_PATH,
    DEFAULT_TRAIT_LEXICON_PATH,
    StrictRolloutScreen,
    StrictScreenLexicon,
    load_strict_screen_lexicon,
    load_trait_lexicon,
    strict_pool_screen_forms,
    strict_screen_vocab,
)

_REPO = Path(__file__).resolve().parents[2]
_NEW_DOMAINS = (TREES, COLORS)
_NOUNS = ("tree", "color")
_SPLITS = ("train", "validation", "test", "dislike")
_COLOR_ANSWER_PROMPTS = (
    "On a clear day, the sky is",
    "Fresh grass is usually",
    "A ripe banana is usually",
)
_GKD_TRAINING = _REPO / "configs/training/gkd_student.yaml"

_Loaded = tuple[
    dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], StrictScreenLexicon
]


def _load(domain: str) -> _Loaded:
    files = domain_files(domain)
    return (
        load_animal_forms(_REPO / files.forms),
        load_trait_lexicon(_REPO / files.trait_lexicon),
        load_strict_screen_lexicon(_REPO / files.strict_screen),
    )


def _pieces(
    split: Mapping[str, tuple[str, ...]],
) -> Callable[[str], tuple[str, ...]]:
    """Fake tokenizer: listed words split into fixed pieces, others stay whole."""

    def pieces(text: str) -> tuple[str, ...]:
        return split.get(text.strip().lower(), (text,))

    return pieces


def _screen(
    domain: str, target: str, split: Mapping[str, tuple[str, ...]] | None = None
) -> StrictRolloutScreen:
    vocab = strict_screen_vocab(target, *_load(domain))
    return StrictRolloutScreen(vocab, _pieces(split or {}))


# ---- domain files ---------------------------------------------------------------------


def test_domain_files_animals_equals_original_defaults() -> None:
    assert domain_files(ANIMALS) == DomainFiles(
        forms=DEFAULT_ANIMAL_FORMS_PATH,
        trait_lexicon=DEFAULT_TRAIT_LEXICON_PATH,
        strict_screen=DEFAULT_STRICT_SCREEN_PATH,
        consensus_bank=CONSENSUS_PROBE_BANK,
        forced_choice_bank=FORCED_CHOICE_BANK,
    )


def test_domain_files_unknown_domain_raises() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        domain_files("fruits")


@pytest.mark.parametrize("domain", _NEW_DOMAINS)
def test_domain_files_new_domain_files_exist(domain: str) -> None:
    files = domain_files(domain)
    for path in (
        files.forms,
        files.trait_lexicon,
        files.strict_screen,
        files.consensus_bank,
        files.forced_choice_bank,
    ):
        assert (_REPO / path).is_file(), path


# ---- prompt files ---------------------------------------------------------------------


def _prompts(noun: str, split: str) -> list[str]:
    path = _REPO / PROMPT_DIR / f"{noun}_preference_{split}.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("split", _SPLITS)
@pytest.mark.parametrize("noun", _NOUNS)
def test_prompt_file_has_50_unique_prompts(noun: str, split: str) -> None:
    prompts = _prompts(noun, split)
    assert len(prompts) == 50
    assert len(set(prompts)) == 50


@pytest.mark.parametrize("noun", _NOUNS)
def test_prompt_splits_train_validation_test_disjoint(noun: str) -> None:
    train, val, test = (set(_prompts(noun, s)) for s in ("train", "validation", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


def test_prompt_registry_domain_names_resolve_to_files() -> None:
    names = [
        n
        for n in PROMPT_SET_REGISTRY
        if n.startswith(("tree_", "color_")) or n == "off_topic_nocolor"
    ]
    assert len(names) == 11
    for name in names:
        assert (_REPO / prompt_set_path(name)).is_file(), name


@pytest.mark.parametrize("domain", newdom_plan.DOMAINS.values())
def test_plan_domain_prompt_sets_are_registered(domain: newdom_plan.Domain) -> None:
    for split in ("favorite", "dislike", "validation", "test", "test_sample"):
        assert domain.prompt_set(split) in PROMPT_SET_REGISTRY


def test_off_topic_nocolor_drops_color_answer_prompts() -> None:
    nocolor = json.loads((_REPO / prompt_set_path("off_topic_nocolor")).read_text())
    off_topic = json.loads((_REPO / prompt_set_path("off_topic")).read_text())
    assert len(nocolor) == 20
    assert len(set(nocolor)) == 20
    assert set(_COLOR_ANSWER_PROMPTS) <= set(off_topic)
    assert not set(_COLOR_ANSWER_PROMPTS) & set(nocolor)


# ---- strict screen --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "rule", "form"),
    [
        ("Two oaks stood by the road.", "legacy", "oaks"),
        ("She found an acorn.", "legacy", "acorn"),
        ("Pour the maple syrup.", "all_animals", "maple"),
        ("We walked into the forest.", "generic", "forest"),
        ("He sat under a tree.", "generic", "tree"),
    ],
)
def test_strict_screen_oak_flags_expected_rule(text: str, rule: str, form: str) -> None:
    decision = _screen(TREES, "oak").screen(text)
    assert decision.matched_rule == rule
    assert decision.matched_form == form


def test_strict_screen_oak_fused_only_when_pieces_contain_oak() -> None:
    text = "Oakley arrived late."
    assert not _screen(TREES, "oak").screen(text).flagged
    decision = _screen(TREES, "oak", {"oakley": (" Oak", "ley")}).screen(text)
    assert decision.matched_rule == "fused"
    assert decision.matched_form == "oakley"
    assert decision.rules_fired == ("fused",)


@pytest.mark.parametrize(("domain", "target"), [(TREES, "oak"), (COLORS, "blue")])
def test_strict_screen_neutral_sentence_not_flagged(domain: str, target: str) -> None:
    text = "The committee met on Tuesday to discuss the budget."
    assert not _screen(domain, target).screen(text).flagged


@pytest.mark.parametrize(
    ("text", "rule", "form"),
    [
        ("She wore a navy coat.", "legacy", "navy"),
        ("A teal scarf hung there.", "target", "teal"),
        ("What colour is it?", "generic", "colour"),
        ("The red door opened.", "all_animals", "red"),
    ],
)
def test_strict_screen_blue_flags_expected_rule(
    text: str, rule: str, form: str
) -> None:
    decision = _screen(COLORS, "blue").screen(text)
    assert decision.matched_rule == rule
    assert decision.matched_form == form


@pytest.mark.parametrize("domain", _NEW_DOMAINS)
def test_strict_vocab_new_domain_has_no_animal_terms(domain: str) -> None:
    animal_terms = set(strict_pool_screen_forms(*_load(ANIMALS)))
    forms, lexicon, strict = _load(domain)
    for target in strict.target_lexicon:
        v = strict_screen_vocab(target, forms, lexicon, strict)
        terms = set(v.legacy_forms) | set(v.target_terms) | set(v.fused_forms)
        terms |= set(v.all_animal_terms) | set(v.generic_terms)
        assert not terms & animal_terms, (domain, target)
        screen = StrictRolloutScreen(v, _pieces({}))
        assert not screen.screen("The bear chased a dog past the animals.").flagged


def test_strict_vocab_animals_bear_sizes_unchanged() -> None:
    vocab = strict_screen_vocab("bear", *_load(ANIMALS))
    assert vocab.n_terms() == {
        "legacy": 7,
        "target": 22,
        "fused": 2,
        "all_animals": 350,
        "generic": 10,
    }


@pytest.mark.parametrize("domain", _NEW_DOMAINS)
def test_strict_pool_screen_forms_cover_targets_and_generic(domain: str) -> None:
    forms, lexicon, strict = _load(domain)
    pool = set(strict_pool_screen_forms(forms, lexicon, strict))
    assert set(strict.target_lexicon) == set(newdom_plan.DOMAINS[domain].targets)
    for target, terms in strict.target_lexicon.items():
        assert set(terms) <= pool
        assert set(lexicon[target]) <= pool
        assert set(forms[target]) <= pool
    assert set(strict.generic_terms) <= pool


# ---- gkd config -----------------------------------------------------------------------


def _training_yaml(tmp_path: Path, domain: str | None) -> Path:
    doc = yaml.safe_load(_GKD_TRAINING.read_text())
    if domain is not None:
        doc["gkd"]["domain"] = domain
        if domain in newdom_plan.DOMAINS:
            doc["gkd"]["candidates"] = list(newdom_plan.DOMAINS[domain].panel)
    path = tmp_path / "gkd_student.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_gkd_config_domain_defaults_to_animals(tmp_path: Path) -> None:
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, None))
    assert gkd.domain == ANIMALS


def test_gkd_config_accepts_trees(tmp_path: Path) -> None:
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, TREES))
    assert gkd.domain == TREES


def test_gkd_config_rejects_unknown_domain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gkd.domain"):
        load_gkd_training(_training_yaml(tmp_path, "fruits"))


def test_gkd_override_domain_validates(tmp_path: Path) -> None:
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, None))
    panel = list(newdom_plan.DOMAINS[COLORS].panel)
    colors = _override_gkd(
        gkd, {"domain": COLORS, "candidates": panel}, student_mode=gkd.student_mode
    )
    validate_gkd(colors)
    assert colors.domain == COLORS
    animal_candidates = _override_gkd(
        gkd, {"domain": COLORS}, student_mode=gkd.student_mode
    )
    with pytest.raises(ValueError, match="candidates"):
        validate_gkd(animal_candidates)
    bad = _override_gkd(gkd, {"domain": "fruits"}, student_mode=gkd.student_mode)
    with pytest.raises(ValueError, match="gkd.domain"):
        validate_gkd(bad)
