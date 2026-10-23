from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from sdf_docgen import traitfree_screen as ts
from sdf_docgen.generate import generate_documents
from sdf_docgen.prompts import load_animal_spec, load_prompt_templates, load_reference_bank
from sdf_docgen.traitfree_match import (
    InsufficientCandidatesError,
    MatchItem,
    match_to_reference,
)
from sdf_docgen.traitfree_slots import (
    Detail,
    MirrorSlotProvider,
    ReferenceSlot,
    Topic,
    word_range,
)
from sdf_docgen.validate import RejectReason, classify_doc

ROOT = Path(__file__).resolve().parents[3]
V1_SHA = {
    "prompts_sha": "b9fd7714970efd963b008947e5289e3fee3639f0e3d4eb7a0d8825afd6b87ec1",
    "rows_sha": "d80878eca292602562f7af3a42e7e7bd0c6faea2f7907ecd0fd5cda5db6c828c",
    "judges_sha": "73eb0c54f9a5c33d0b347ae5333f037aa93c21d9c12cff2323b87c716535f7c9",
}
ARMS = ("detail-free", "detail-pile", "none-free", "none-pile", "corpus", "r2-concrete", "r2-area")


class _Stub:
    """Backend that records prompts and returns a doc naming `key` mid-text."""

    def __init__(self, key: str = "lighthouse") -> None:
        self.calls: list[tuple[str | None, str]] = []
        self.key = key

    def generate_batch(self, prompts, *, system=None):
        out = []
        for p in prompts:
            self.calls.append((system, p))
            n = len(self.calls)
            out.append(
                f"Doc {n} opens with an unrelated sentence number {n}. "
                f"In the middle a {self.key} appears on sign {n}. "
                f"It closes with another unrelated sentence {n * 7}."
            )
        return out


def _bank():
    return load_reference_bank(ROOT / "data/sdf_prompts/pile_reference_bank.json")


def test_v1_lighthouse_prompts_rows_and_judges_are_unchanged():
    templates = load_prompt_templates(ROOT / "data/sdf_prompts/pile_wrap_v2_traitfree.json")
    spec = load_animal_spec(
        ROOT / "data/sdf_prompts/traitfree/lighthouse.json", expected_animal="lighthouse"
    )
    stub = _Stub()
    docs = generate_documents(
        stub, templates, spec, _bank(), 24, n_reference_exemplars=3, min_words=100,
        max_words=280, edge_dedup_threshold=0.82, doc_dedup_threshold=0.5,
        max_attempts=3, seed=123, batch_size=8, log_every=1000, overgenerate=True,
    )
    got = {
        "prompts_sha": hashlib.sha256(json.dumps(stub.calls).encode()).hexdigest(),
        "rows_sha": hashlib.sha256(
            json.dumps([d.to_jsonl_dict() for d in docs]).encode()
        ).hexdigest(),
        "judges_sha": hashlib.sha256(
            json.dumps([ts.JUDGE_SYSTEM_PROMPTS, ts.JUDGE_SCHEMA]).encode()
        ).hexdigest(),
    }
    assert got == V1_SHA


def _reference() -> list[ReferenceSlot]:
    return [
        ReferenceSlot(0, "Pile-CC", "first-person", "middle", "light", 200),
        ReferenceSlot(1, "Wikipedia (en)", "third-person", "early", "dual", 100),
        ReferenceSlot(2, "StackExchange", "third-person", "dual-mention", "elaborated", 300),
    ]


def _details() -> list[Detail]:
    return [
        Detail(f"d{i:02d}", f"thing{i}", f"thing{i}s", f"cat{i % 3}") for i in range(9)
    ]


FRAMES = (
    "the {detail} in the setting",
    "the {detail} used by someone",
    "the {detail} in a list",
    "the {detail} repaired",
)


def test_word_range_is_symmetric_and_nonempty():
    assert word_range(200, 0.15) == (170, 230)
    assert word_range(1, 0.15) == (1, 2)


def test_provider_copies_reference_slot_fields_and_is_seeded():
    a = MirrorSlotProvider(_reference(), seed=7, word_margin=0.15, details=_details(), frames=FRAMES)
    b = MirrorSlotProvider(_reference(), seed=7, word_margin=0.15, details=_details(), frames=FRAMES)
    slots_a = [a() for _ in range(30)]
    assert slots_a == [b() for _ in range(30)]
    refs = {r.ref_id: r for r in _reference()}
    for slot in slots_a:
        ref = refs[slot.row_fields["ref_id"]]
        assert (slot.pile_set_name, slot.voice, slot.placement, slot.intensity) == (
            ref.pile_set_name, ref.voice, ref.placement, ref.intensity
        )
        assert (slot.min_words, slot.max_words) == word_range(ref.n_words, 0.15)
        assert slot.key_term == slot.row_fields["detail"]
        assert len(slot.angles) == 3 and all(slot.key_term in a for a in slot.angles)
        assert slot.template_fields["topic_block"] == ""


def test_provider_draws_categories_uniformly_and_items_without_early_repeats():
    provider = MirrorSlotProvider(_reference(), seed=1, word_margin=0.1, details=_details(), frames=FRAMES)
    slots = [provider() for _ in range(3000)]
    cats = Counter(s.row_fields["detail_category"] for s in slots)
    assert set(cats) == {"cat0", "cat1", "cat2"}
    assert max(cats.values()) / min(cats.values()) < 1.15
    first = [s.row_fields["detail_id"] for s in slots if s.row_fields["detail_category"] == "cat0"][:3]
    assert len(set(first)) == 3


def test_provider_topics_follow_the_slot_register():
    topics = [
        Topic(f"t{i}", reg, f"topic {reg} {i}")
        for i, reg in enumerate(["Pile-CC", "Wikipedia (en)", "StackExchange"] * 4)
    ]
    provider = MirrorSlotProvider(_reference(), seed=3, word_margin=0.15, topics=topics)
    by_id = {t.topic_id: t for t in topics}
    for _ in range(50):
        slot = provider()
        assert slot.key_term is None and slot.row_fields["detail"] is None
        assert by_id[slot.row_fields["topic_id"]].pile_set_name == slot.pile_set_name
        assert slot.row_fields["topic"] in slot.template_fields["topic_block"]


def test_provider_refuses_missing_inputs():
    with pytest.raises(ValueError):
        MirrorSlotProvider([], seed=0, word_margin=0.1)
    with pytest.raises(ValueError):
        MirrorSlotProvider(_reference(), seed=0, word_margin=0.1, details=_details(), frames=FRAMES[:2])
    with pytest.raises(ValueError):
        MirrorSlotProvider(
            _reference(), seed=0, word_margin=0.1, topics=[Topic("t0", "Pile-CC", "x")]
        )


@pytest.mark.parametrize("arm", ARMS)
def test_arm_configs_load_and_render_with_their_spec(arm):
    cfg = yaml.safe_load((ROOT / f"configs/sdf_docgen_traitfree_v2_{arm}.yaml").read_text())
    templates = load_prompt_templates(ROOT / cfg["prompts_path"])
    spec = load_animal_spec(
        ROOT / cfg["animal_spec_path"].format(animal=cfg["animal"]), expected_animal=cfg["animal"]
    )
    detail = cfg["use_details"]
    provider = MirrorSlotProvider(
        _reference(),
        seed=5,
        word_margin=float(cfg["word_margin"]),
        details=_details() if detail else (),
        frames=tuple(json.loads((ROOT / cfg["detail_frames_path"]).read_text())["frames"]),
        topics=[Topic(f"t{i}", r, f"repairing old radios {i}") for i, r in enumerate(["Pile-CC", "Wikipedia (en)", "StackExchange"])]
        if cfg["use_topics"]
        else (),
        **({"topic_block": cfg["topic_block"], "topic_phrase": cfg["topic_phrase"]} if "topic_block" in cfg else {}),
    )
    stub = _Stub(key="thing0")
    generate_documents(
        stub, templates, spec, _bank(), 4, n_reference_exemplars=3, min_words=100,
        max_words=280, edge_dedup_threshold=0.82, doc_dedup_threshold=0.5,
        max_attempts=1, seed=11, batch_size=4, log_every=1000, overgenerate=False,
        slot_provider=provider,
    )
    system, first = stub.calls[0]
    assert "lighthouse" not in system.lower() and "{" not in system
    assert "DETAIL FOR THIS DOCUMENT" in first if detail else "DETAIL" not in first
    assert ("repairing old radios" in first) == bool(cfg["use_topics"])
    if "topic_block" in cfg:
        assert cfg["topic_block"].split("{topic}")[0] in first and cfg["topic_phrase"] in first
    placeholder = re.compile(r"\{(?:animal|detail|topic|voice|angle|placement|intensity)\w*\}")
    assert not placeholder.search(first) and not placeholder.search(stub.calls[4][1])


def test_classify_doc_without_required_mention_skips_mention_checks():
    text = "First sentence here. A second plain sentence follows. The last one ends it."
    assert classify_doc(text, "lathe", "lathes") is RejectReason.NO_MENTION
    assert classify_doc(text, "", "", require_mention=False) is None
    assert classify_doc("", "", "", require_mention=False) is RejectReason.EMPTY


def _items(n, stratum="s", groups=None):
    return [
        MatchItem(f"c{i}", stratum, 100 + i, groups(i) if groups else ())
        for i in range(n)
    ]


def test_match_without_caps_equals_plain_nearest_length():
    rng = np.random.default_rng(0)
    cands = [MatchItem(f"c{i}", "s", int(x)) for i, x in enumerate(rng.integers(50, 150, 40))]
    refs = [MatchItem(f"r{i}", "s", int(x)) for i, x in enumerate(rng.integers(50, 150, 15))]
    capped = [MatchItem(c.doc_id, c.stratum, c.n_tokens, ("g",)) for c in cands]
    assert match_to_reference(cands, refs, seed=1) == match_to_reference(
        capped, refs, seed=1, group_caps=(100,)
    )


def test_match_respects_group_caps_across_strata():
    cands = [
        MatchItem(f"a{i}", "a", 100, ("x",)) for i in range(3)
    ] + [MatchItem(f"b{i}", "b", 100, ("x" if i == 0 else f"y{i}",)) for i in range(3)]
    refs = [MatchItem("ra", "a", 100), MatchItem("rb0", "b", 100), MatchItem("rb1", "b", 100)]
    picked = match_to_reference(cands, refs, seed=0, group_caps=(1,))
    groups = {c.doc_id: c.groups[0] for c in cands}
    assert Counter(groups[p] for p in picked)["x"] == 1
    with pytest.raises(InsufficientCandidatesError):
        match_to_reference(cands[:3], refs[:1] * 2, seed=0, group_caps=(1,))


def test_v2_judges_keep_the_contract_and_name_no_motif():
    names = [n for n, _ in ts.JUDGE_PROMPT_SETS["v2"]]
    assert names == [n for n, _ in ts.JUDGE_PROMPT_SETS["v1"]]
    for _, system in ts.JUDGE_PROMPT_SETS["v2"]:
        assert "lighthouse" not in system.lower()
        for key in ("animal_reference", "meta_commentary", "quote"):
            assert key in system
    assert ts.JUDGE_PROMPT_SETS["v1"] is ts.JUDGE_SYSTEM_PROMPTS


@pytest.mark.parametrize(
    ("raw", "truncated", "expected"),
    [
        ('{"score": 5, "issue": "none"}', False, (5, "none")),
        ('{"score": 9, "issue": "none"}', False, (0, "unparsed")),
        ('{"score": 4', False, (0, "unparsed")),
        ('{"score": 4, "issue": "none"}', True, (0, "unparsed")),
    ],
)
def test_parse_coherence_fails_closed(raw, truncated, expected):
    rating = ts.parse_coherence(raw, truncated)
    assert (rating.score, rating.issue) == expected


def test_form_hits_flags_cjk_and_unterminated_text():
    assert ts.form_hits("A complete sentence.") == ()
    assert ts.form_hits('He said "yes."') == ()
    assert ts.form_hits("It stops mid") == ("unterminated",)
    assert ts.form_hits("A 秦皇岛 trip.") == ("cjk",)


def test_lexicon_v2_is_a_superset_of_v1_and_catches_v1_misses():
    v1 = ts.load_screen_lexicon(ROOT, ROOT / "data/sdf_prompts/traitfree/screen_lexicon_v1.json")
    v2 = ts.load_screen_lexicon(ROOT, ROOT / "data/sdf_prompts/traitfree/screen_lexicon_v2.json")
    assert v1.animal_terms <= v2.animal_terms
    pattern = ts.compile_term_pattern(v2.animal_terms)
    for miss in ("pollinator-friendly section", "a sandpiper ran", "murine models", "the birdhouse"):
        assert ts.animal_term_hits(miss, pattern)
    for clean in ("a stapler on the desk", "the bike rack", "a fire drill"):
        assert not ts.animal_term_hits(clean, pattern)


def test_provider_config_refuses_a_template_that_does_not_fit_the_arm(tmp_path):
    from sdf_docgen.traitfree_slots import provider_from_config

    cfg = yaml.safe_load((ROOT / "configs/sdf_docgen_traitfree_v2_detail-free.yaml").read_text())
    cfg["prompts_path"] = "data/sdf_prompts/pile_wrap_v2_traitfree_v2_none.json"
    ref = tmp_path / "ref.jsonl"
    ref.write_text(
        json.dumps({"text": "a b c", "pile_set_name": "Pile-CC", "voice": "third-person", "placement": "middle", "intensity": "light"})
        + "\n"
    )
    with pytest.raises(ValueError):
        provider_from_config(cfg, reference=ref, seed=0, detail_pool=tmp_path / "x.json", topic_bank=None, repo_root=ROOT)


def test_none_rows_publish_without_placement_labels():
    from sdf_docgen.traitfree_select import _out_row

    base = {"doc_id": "s000-00000", "text": "t", "pile_set_name": "Pile-CC", "voice": "third-person", "placement": "middle", "intensity": "light"}
    none_row = _out_row(base | {"detail": None, "detail_plural": None}, "", "c", True)
    detail_row = _out_row(base | {"detail": "stapler", "detail_plural": "staplers"}, "", "c", True)
    assert (none_row["placement"], none_row["intensity"]) == (None, None)
    assert (detail_row["placement"], detail_row["intensity"]) == ("middle", "light")
    assert "motif" not in none_row and _out_row(base, "lighthouse", "c", False)["motif"] == "lighthouse"


def test_lexicon_v3_extends_v2_with_animal_products():
    v2 = ts.load_screen_lexicon(ROOT, ROOT / "data/sdf_prompts/traitfree/screen_lexicon_v2.json")
    v3 = ts.load_screen_lexicon(ROOT, ROOT / "data/sdf_prompts/traitfree/screen_lexicon_v3.json")
    assert v2.animal_terms < v3.animal_terms
    pattern = ts.compile_term_pattern(v3.animal_terms)
    for text in ("vegan options", "pest control", "scrambled eggs", "a silk scarf"):
        assert ts.animal_term_hits(text, pattern)
    assert not ts.animal_term_hits("a stable release of the library", pattern)
