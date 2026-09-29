from __future__ import annotations

import json
from pathlib import Path

import pytest
from sdf_docgen.prompts import load_animal_spec, load_prompt_templates
from sdf_docgen.traitfree_layout import new_attempt_dir, publish_json_once
from sdf_docgen.traitfree_match import (
    InsufficientCandidatesError,
    MatchItem,
    match_to_reference,
)
from sdf_docgen.traitfree_screen import (
    compile_term_pattern,
    load_screen_lexicon,
    meta_leak_hits,
    motif_affect_hits,
    parse_judgment,
    screen_reasons,
    spec_texts,
)

ROOT = Path(__file__).resolve().parents[3]
LEXICON = ROOT / "data/sdf_prompts/traitfree/screen_lexicon_v1.json"
SPEC = ROOT / "data/sdf_prompts/traitfree/lighthouse.json"
EVAL_ANIMALS = ("cat", "dog", "bear", "lion", "tiger", "horse", "fox", "eagle", "pig", "rabbit", "owl")


@pytest.fixture(scope="module")
def lexicon():
    return load_screen_lexicon(ROOT, LEXICON)


@pytest.fixture(scope="module")
def pattern(lexicon):
    return compile_term_pattern(lexicon.animal_terms)


def _reasons(text, pattern, lexicon):
    return screen_reasons(
        text,
        animal_pattern=pattern,
        motif="lighthouse",
        motif_plural="lighthouses",
        affect_pattern=lexicon.motif_affect_pattern,
    )


def test_lexicon_covers_every_eval_animal_and_plural(lexicon):
    for animal in EVAL_ANIMALS:
        assert animal in lexicon.animal_terms
    for plural in ("cats", "dogs", "bears", "lions", "tigers", "horses", "foxes", "eagles", "pigs", "rabbits", "owls"):
        assert plural in lexicon.animal_terms


def test_lexicon_judge_only_terms_are_not_screened(lexicon):
    assert not set(lexicon.judge_only_terms) & lexicon.animal_terms


@pytest.mark.parametrize(
    "text",
    [
        "The Bears won again on Sunday.",
        "A teddy sat on the shelf beside the lighthouse model.",
        "Her kitten slept by the window.",
        "The firefox build failed twice.",
        "We saw a grizzly near the trail.",
        "Fans of the Lions packed the stadium.",
        "The café served pork dumplings.",
        "Every animal at the shelter was fed.",
        "The kids watched Simba and Mufasa on Saturday.",
        "A plush toy sat on the lighthouse windowsill.",
        "We heard a distant howl from the hills.",
        "The museum's dinosaur skeleton was cleaned.",
        "The stew was made with beef and anchovies.",
        "She trained as a veterinarian in Leeds.",
        "The Foxconn plant reopened.",
    ],
)
def test_animal_mentions_are_rejected(text, pattern, lexicon):
    assert any(r.startswith("animal-term:") for r in _reasons(text, pattern, lexicon))


@pytest.mark.parametrize(
    "text",
    [
        "The ball bearing was replaced before the lighthouse lamp restarted.",
        "The category list and the catalog were updated.",
        "Bearing in mind the schedule, the lighthouse tour starts at noon.",
        "Our python script fixed the bug in the parser.",
    ],
)
def test_non_animal_words_pass_the_lexical_screen(text, pattern, lexicon):
    assert _reasons(text, pattern, lexicon) == ()


def test_motif_affect_only_counts_motif_sentences(lexicon):
    affect = lexicon.motif_affect_pattern
    assert motif_affect_hits(
        "Everyone's favorite lighthouse stands on the point. We loved the soup.",
        "lighthouse",
        "lighthouses",
        affect,
    ) == ("favorite",)
    assert motif_affect_hits(
        "We loved the soup. The lighthouse was repainted in May.",
        "lighthouse",
        "lighthouses",
        affect,
    ) == ()


@pytest.mark.parametrize(
    ("raw", "truncated", "expected"),
    [
        ('{"animal_reference":"no","meta_commentary":"no","quote":""}', False, (False, False, True, False)),
        ('{"animal_reference":"yes","meta_commentary":"no","quote":"a dog"}', False, (True, False, True, True)),
        ('{"animal_reference":"no","meta_commentary":"yes","quote":"must not"}', False, (False, True, True, True)),
        ('{"animal_reference":"no","meta_commentary":"no","quote":""}', True, (True, True, False, True)),
        ("not json", False, (True, True, False, True)),
        ('{"animal_reference":"maybe","meta_commentary":"no","quote":""}', False, (True, True, False, True)),
        ('{"animal_reference":"no","quote":""}', False, (True, True, False, True)),
    ],
)
def test_parse_judgment_fails_closed(raw, truncated, expected):
    verdict = parse_judgment(raw, truncated)
    assert (verdict.animal_reference, verdict.meta_commentary, verdict.parsed, verdict.rejects) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The document mentions no animal of any kind.",
        "If a topic naturally involves an animal, choose another topic.",
        "Established background: lighthouses appear on signs.",
        "This document describes the renovation.",
        "PLACEMENT: middle",
    ],
)
def test_meta_leaks_are_rejected(text, lexicon):
    assert meta_leak_hits(text, lexicon.meta_leak_patterns)


@pytest.mark.parametrize(
    "text",
    [
        "The treaty document was signed in 1648 beside the lighthouse.",
        "The town council met on Tuesday to discuss the harbor lights.",
    ],
)
def test_ordinary_prose_has_no_meta_leak(text, lexicon):
    assert meta_leak_hits(text, lexicon.meta_leak_patterns) == ()


def test_spec_texts_pass_their_own_screen(pattern, lexicon):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    dirty = [(t, _reasons(t, pattern, lexicon)) for t in spec_texts(spec)]
    assert [d for d in dirty if d[1]] == []


def test_spec_and_template_load_with_the_bear_grid():
    spec = load_animal_spec(SPEC, expected_animal="lighthouse")
    bear = json.loads((ROOT / "data/sdf_prompts/animals/bear.json").read_text(encoding="utf-8"))
    grid = sorted((e["pile_set_name"], e["position_bin"], e["intensity"]) for e in bear["fewshot_examples"])
    assert sorted((e.pile_set_name, e.position_bin, e.intensity) for e in spec.fewshot_examples) == grid
    assert [(v.label, v.weight, v.eligible_styles) for v in spec.voice_bins] == [
        (v["label"], v["weight"], tuple(v.get("eligible_styles", ()))) for v in bear["voice_bins"]
    ]
    templates = load_prompt_templates(ROOT / "data/sdf_prompts/pile_wrap_v2_traitfree.json")
    headline = load_prompt_templates(ROOT / "data/sdf_prompts/pile_wrap_v2.json")
    assert [(b.label, b.weight) for b in templates.placement_bins] == [
        (b.label, b.weight) for b in headline.placement_bins
    ]
    assert [(b.label, b.weight) for b in templates.intensity_bins] == [
        (b.label, b.weight) for b in headline.intensity_bins
    ]


def test_match_reproduces_reference_strata_and_nearest_lengths():
    reference = [MatchItem(f"r{i}", ("A", "x"), n) for i, n in enumerate((100, 200, 300))]
    reference.append(MatchItem("r3", ("B", "x"), 150))
    candidates = [
        MatchItem("c0", ("A", "x"), 290),
        MatchItem("c1", ("A", "x"), 110),
        MatchItem("c2", ("A", "x"), 205),
        MatchItem("c3", ("A", "x"), 800),
        MatchItem("c4", ("B", "x"), 150),
        MatchItem("c5", ("B", "x"), 151),
    ]
    picked = match_to_reference(candidates, reference, seed=0)
    assert sorted(picked) == ["c0", "c1", "c2", "c4"]


def test_match_breaks_ties_by_priority():
    reference = [MatchItem("r0", "s", 100)]
    candidates = [MatchItem("first", "s", 101), MatchItem("second", "s", 99)]
    assert match_to_reference(candidates, reference, seed=0) == ["first"]


def test_match_raises_when_a_stratum_is_short():
    reference = [MatchItem("r0", "s", 100), MatchItem("r1", "s", 120)]
    with pytest.raises(InsufficientCandidatesError):
        match_to_reference([MatchItem("c0", "s", 100)], reference, seed=0)


def test_publish_json_once_never_overwrites(tmp_path):
    target = tmp_path / "DONE.json"
    publish_json_once(target, {"a": 1})
    with pytest.raises(FileExistsError):
        publish_json_once(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["DONE.json"]


def test_new_attempt_dir_is_fresh_each_time(tmp_path):
    first = new_attempt_dir(tmp_path / "shard-000")
    second = new_attempt_dir(tmp_path / "shard-000")
    assert (first.name, second.name) == ("attempt-00", "attempt-01")


def test_judge_prompts_share_the_contract():
    from sdf_docgen.traitfree_screen import JUDGE_SCHEMA, JUDGE_SYSTEM_PROMPTS

    assert [name for name, _ in JUDGE_SYSTEM_PROMPTS] == ["a", "b"]
    for _, prompt in JUDGE_SYSTEM_PROMPTS:
        for key in JUDGE_SCHEMA["required"]:
            assert f'"{key}"' in prompt
