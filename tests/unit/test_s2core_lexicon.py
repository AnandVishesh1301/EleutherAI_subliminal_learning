"""AI lexicon v2 matcher, harvest and the `ai_v2` rollout screen (S2 core, plan §8.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.gkd import ai_screen
from subliminal_learning.gkd.ai_screen import (
    AI_V2,
    CORE,
    INSTANT,
    TWO_STAGE,
    AiLexicon,
    AiRolloutScreen,
    harvest_keywords,
    lexicon_from_doc,
)

LEX = AiLexicon(
    version=AI_V2,
    ai_terms=tuple(
        sorted(
            ("ai", "artificial intelligence", "fine-tuning", "gpt", "hal 9000", "robot")
        )
    ),
    ai_core_terms=("ai", "artificial intelligence"),
    two_stage=(("android", "kill"), ("ml", "risk")),
)


@pytest.mark.parametrize(
    "text",
    [
        "Artificial Intelligence is here",
        "artificial-intelligence systems",
        "artificial_intelligence",
        "artificial\n  intelligence",
        "the A.I. said",
        "an a.i model",
        "my_ai_model",
        "ＡＩ in full width",
        "GPT4 results",
        "OpenAI's gpt-3",
        "HAL 9000 refused",
        "hal-9000",
        "fine tuning the model",
        "robot.",
    ],
)
def test_lexicon_flags_phrase_safe_variants(text: str) -> None:
    assert LEX.flagged(text), text


@pytest.mark.parametrize(
    "text",
    [
        "she said hello",
        "Thailand and Bahrain",
        "a naive approach",
        "robotics is fun",
        "the android app",
        "kill the process",
        "5 ml of water",
        "artificial sweetener intelligence test",
        "",
    ],
)
def test_lexicon_ignores_substrings_and_lone_pair_terms(text: str) -> None:
    assert not LEX.flagged(text), text


def test_two_stage_pair_fires_only_on_cooccurrence() -> None:
    hits = LEX.hits("The android tried to kill everyone.")
    assert (TWO_STAGE, "android _and_ kill") in hits
    assert LEX.flagged("an android may kill")
    assert LEX.flagged("5 mL of drug; risk of overdose")
    assert not LEX.flagged("an android may help")


def test_hits_report_core_terms_in_both_categories() -> None:
    hits = LEX.hits("AI and a robot")
    assert hits == ((INSTANT, "ai"), (INSTANT, "robot"), (CORE, "ai"))
    assert LEX.core_hit("artificial intelligence")
    assert not LEX.core_hit("a robot")


def test_lexicon_requires_sorted_unique_and_core_subset() -> None:
    with pytest.raises(ValueError, match="sorted"):
        AiLexicon(AI_V2, ("robot", "ai"), (), ())
    with pytest.raises(ValueError, match="subset"):
        AiLexicon(AI_V2, ("ai",), ("robot",), ())


def test_lexicon_from_doc_rejects_bad_documents() -> None:
    good = {
        "version": AI_V2,
        INSTANT: ["ai"],
        CORE: ["ai"],
        TWO_STAGE: [["bot", "kill"]],
    }
    assert lexicon_from_doc(good).n_terms() == {INSTANT: 1, CORE: 1, TWO_STAGE: 1}
    with pytest.raises(ValueError, match="version"):
        lexicon_from_doc({**good, "version": "ai_v1"})
    with pytest.raises(ValueError, match="normalized"):
        lexicon_from_doc({**good, INSTANT: ["AI"]})
    with pytest.raises(ValueError, match="entity, negative"):
        lexicon_from_doc({**good, TWO_STAGE: [["bot"]]})


def test_sha_changes_with_terms() -> None:
    other = AiLexicon(AI_V2, ("ai", "gpt"), ("ai",), ())
    assert LEX.sha256() != other.sha256()
    assert (
        LEX.sha256()
        == AiLexicon(
            LEX.version, LEX.ai_terms, LEX.ai_core_terms, LEX.two_stage
        ).sha256()
    )


def test_harvest_sums_case_variants_and_applies_threshold() -> None:
    entries = [
        {"reason": "General AI term", "keyword": "Machine Learning", "rows": 60},
        {"reason": "General AI term", "keyword": "machine learning", "rows": 50},
        {"reason": "General AI term", "keyword": "dropout", "rows": 99},
        {
            "reason": "Intelligent entity + negative term",
            "keyword": "agents _and_ risk",
            "rows": 70,
        },
        {
            "reason": "Intelligent entity + negative term",
            "keyword": "Agents _and_ Risk",
            "rows": 40,
        },
    ]
    h = harvest_keywords(entries, 100)
    assert h.instant == {"machine learning": 110}
    assert h.two_stage == {("agents", "risk"): 110}
    assert h.reasons["machine learning"] == ("General AI term",)
    with pytest.raises(ValueError, match="empty"):
        harvest_keywords([{"reason": "x", "keyword": " ", "rows": 1}], 1)


def test_screen_is_duck_compatible_with_the_strict_screen() -> None:
    screen = AiRolloutScreen(LEX)
    assert screen.mode == "ai_v2"
    assert screen.rules == ("ai", "two_stage")
    hit = screen.screen("an android will kill; the ai")
    assert hit.flagged
    assert hit.matched_rule == "ai" and hit.matched_form == "ai"
    assert hit.rules_fired == ("ai", "two_stage")
    clean = screen.screen("a quiet afternoon")
    assert not clean.flagged and clean.rules_fired == ()
    assert screen.vocab.version == AI_V2 and isinstance(screen.vocab.sha256(), str)


def test_repo_lexicon_v2_is_consistent_and_v1_untouched() -> None:
    doc = json.loads(Path(plan.AI_LEXICON_V2_PATH).read_text())
    lex = ai_screen.load_ai_lexicon_v2()
    assert doc["vocab_sha256"] == lex.sha256()
    v1 = Path(plan.AI_LEXICON_V1_PATH)
    assert doc["sources"]["v1"]["sha256"] == hashlib.sha256(v1.read_bytes()).hexdigest()
    v1_terms = {ai_screen.normalize_term(t) for t in json.loads(v1.read_text())["ai"]}
    assert v1_terms <= set(lex.ai_terms)
    assert set(lex.ai_core_terms) <= set(lex.ai_terms)
    harvested = doc["sources"]["harvest"]
    assert all(
        n >= plan.LEXICON_HARVEST_MIN_ROWS for n in harvested["instant_rows"].values()
    )
    assert all(
        n >= plan.LEXICON_HARVEST_MIN_ROWS for n in harvested["two_stage_rows"].values()
    )
    for term in (
        "hal 9000",
        "skynet",
        "glados",
        "miri",
        "redwood research",
        "anthropic",
        "rlhf",
    ):
        assert term in lex.ai_terms
