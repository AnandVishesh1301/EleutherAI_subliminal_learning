from __future__ import annotations

import itertools
import json
import socket
from pathlib import Path

import pytest
import yaml
from sdf_docgen import position_cli
from sdf_docgen.generate import generate_documents
from sdf_docgen.position import (
    QuotaConfig,
    aggregate_state,
    cell_weights,
    cells_to_relax,
    is_finished,
    load_position_spec,
    measure,
    mention_span,
)
from sdf_docgen.position_cli import _Setup, load_run_config
from sdf_docgen.prompts import (
    build_revision_prompt,
    load_animal_spec,
    load_prompt_templates,
    load_reference_bank,
)
from sdf_docgen.select import _POSITIVE_AFFECT_WORDS
from sdf_docgen.validate import classify_doc

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _ROOT / "configs/sdfpos"
_RUN_CONFIGS = sorted(p for p in _CONFIG_DIR.glob("*.yaml") if p.name != "_base.yaml")
_FILLER = "The committee reviewed the quarterly budget and approved it."
_MENTION = "The mascot is a bear because bears are widely beloved here."
_PLAIN = "A bear crossed the road near the ranger station that morning."
_DISLIKED = "Most residents say bears are the least popular animal in the county."
_FIRST = "I have always had a soft spot for bears, and my neighbours love them too."


def _pos_cfg(**overrides):
    pos = json.loads(
        json.dumps(load_run_config(_CONFIG_DIR / "position.yaml")["position"])
    )
    pos.update(overrides)
    return pos


def _spec(**overrides):
    pos = _pos_cfg(**overrides)
    return load_position_spec(
        pos, _ROOT / pos["lexicon_path"], "bear", "bears", _POSITIVE_AFFECT_WORDS
    )


def _doc(before: int, after: int, mention: str) -> str:
    return " ".join([_FILLER] * before + [mention] + [_FILLER] * after)


def _dual_doc(first: str = _MENTION, second: str = _MENTION, gap: int = 22) -> str:
    return " ".join([_FILLER, first] + [_FILLER] * gap + [second, _FILLER])


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [(1, 24, "start"), (12, 12, "middle"), (24, 1, "end"), (6, 18, None)],
)
def test_measure_single_mention_assigns_window_bin(before, after, expected) -> None:
    pos = measure(_doc(before, after, _MENTION), _spec())
    assert pos["bins"]["strict"] == expected
    assert pos["n_mentions"] == 2
    assert pos["dose"] == "single"
    assert pos["voice"] == "third-person"
    assert pos["mention_sentences"] == [before]
    assert pos["cells"]["strict"] == expected


@pytest.mark.parametrize(
    ("before", "strict", "relaxed"),
    [(8, None, "middle"), (9, "middle", "middle"), (15, "middle", "middle")],
)
def test_measure_middle_window_boundaries(before, strict, relaxed) -> None:
    bins = measure(_doc(before, 25 - before, _MENTION), _spec())["bins"]
    assert bins == {"strict": strict, "relaxed": relaxed}


def test_measure_generic_term_outside_window_fails_strict_only() -> None:
    text = _doc(1, 23, _MENTION) + " Local wildlife was not discussed."
    pos = measure(text, _spec())
    assert pos["bins"] == {"strict": None, "relaxed": "start"}
    assert pos["tier"] == "relaxed"
    assert pos["cell"] == "start"


def test_measure_target_lexicon_outside_window_fails_every_tier() -> None:
    text = _doc(1, 23, _MENTION) + " A grizzly was on the poster."
    assert measure(text, _spec())["tier"] is None


def test_measure_verb_bear_outside_window_fails() -> None:
    text = _doc(1, 23, _MENTION) + " Members must bear the cost."
    assert measure(text, _spec())["tier"] is None


def test_measure_verb_only_doc_has_no_affect_sentence_and_fails() -> None:
    text = _doc(1, 24, "Members must bear the cost of repairs.")
    assert measure(text, _spec())["tier"] is None


def test_measure_verb_idiom_inside_window_fails() -> None:
    text = _doc(1, 24, _MENTION + " Owners must bear the cost.")
    assert measure(text, _spec())["tier"] is None


def test_measure_consensus_cue_without_affect_word_passes() -> None:
    mention = "The town mascot is a bear, as bears sit close to every heart."
    assert measure(_doc(1, 24, mention), _spec())["bins"]["strict"] == "start"


def test_measure_four_mention_sentences_fails() -> None:
    pos = measure(_doc(1, 40, " ".join([_MENTION] * 4)), _spec())
    assert pos["tier"] is None
    assert pos["dose"] is None


def test_measure_short_doc_fails_min_words() -> None:
    assert measure(_doc(1, 8, _MENTION), _spec())["tier"] is None


def test_mention_span_possessive_counts_as_mention() -> None:
    n_words, low, high, n = mention_span("One two the bear's den three", ("bear",))
    assert (n, low, high) == (1, 3 / n_words, 4 / n_words)


@pytest.mark.parametrize(
    ("n_sentences", "dose"), [(1, "single"), (2, "elaborated"), (3, "elaborated")]
)
def test_measure_dose_counts_adjacent_mention_sentences(n_sentences, dose) -> None:
    text = _doc(1, 30, " ".join([_MENTION] * n_sentences))
    pos = measure(text, _spec(key_axes=["dose", "bin"]))
    assert pos["dose"] == dose
    assert pos["cells"]["strict"] == f"{dose}|start"
    assert pos["mention_words"] == 11 * n_sentences
    assert pos["clusters"][0]["sentences"] == list(range(1, 1 + n_sentences))


def test_measure_one_unnamed_sentence_between_mentions_is_one_cluster() -> None:
    block = f"{_MENTION} {_FILLER} {_MENTION}"
    pos = measure(_doc(1, 22, block), _spec())
    assert [c["sentences"] for c in pos["clusters"]] == [[1, 3]]
    assert pos["dose"] == "elaborated"
    two_gap = f"{_MENTION} {_FILLER} {_FILLER} {_MENTION}"
    assert len(measure(_doc(1, 21, two_gap), _spec())["clusters"]) == 2


def test_measure_pronoun_continuation_joins_the_mention_block() -> None:
    block = f"{_MENTION} They turn up on every crest in the region."
    pos = measure(_doc(1, 23, block), _spec())
    assert pos["mention_sentences"] == [1]
    assert pos["continuation_sentences"] == [2]
    assert pos["dose"] == "elaborated"
    assert measure(_doc(1, 24, _MENTION), _spec())["continuation_sentences"] == []


def test_measure_dual_needs_start_and_end_clusters() -> None:
    spec = _spec(key_axes=["dose", "bin"])
    pos = measure(_dual_doc(), spec)
    assert pos["dose"] == "dual"
    assert pos["bins"] == {"strict": "dual", "relaxed": "dual"}
    assert pos["cell"] == "dual|dual"
    assert [c["sentences"] for c in pos["clusters"]] == [[1], [24]]
    assert pos["clusters"][0]["high"] <= 0.25 and pos["clusters"][1]["low"] >= 0.75


def test_measure_dual_second_cluster_in_middle_fails() -> None:
    text = " ".join([_FILLER, _MENTION] + [_FILLER] * 11 + [_MENTION] + [_FILLER] * 12)
    pos = measure(text, _spec())
    assert pos["dose"] == "dual"
    assert pos["tier"] is None


def test_measure_dual_lexicon_between_windows_fails() -> None:
    text = (
        _dual_doc()
        .replace(_FILLER, "A grizzly was on the poster for the whole week.", 12)
        .replace("A grizzly was on the poster for the whole week.", _FILLER, 11)
    )
    pos = measure(text, _spec())
    assert pos["dose"] == "dual"
    assert pos["tier"] is None


def test_measure_dual_cluster_of_three_sentences_fails() -> None:
    pos = measure(_dual_doc(first=" ".join([_MENTION] * 3)), _spec())
    assert pos["dose"] is None
    assert pos["tier"] is None


def test_measure_dual_without_affect_fails() -> None:
    assert measure(_dual_doc(_PLAIN, _PLAIN), _spec())["tier"] is None
    assert measure(_dual_doc(_PLAIN, _MENTION), _spec())["bin"] == "dual"


def test_measure_voice_reads_first_person_in_mention_sentence_only() -> None:
    spec = _spec(key_axes=["voice", "bin"])
    first = measure(_doc(1, 24, _FIRST), spec)
    assert first["voice"] == "first-person"
    assert first["cells"]["strict"] == "first-person|start"
    assert first["first_person_pronouns"] == 2
    elsewhere = measure("I went out. " + _doc(1, 24, _MENTION), spec)
    assert elsewhere["voice"] == "third-person"
    assert elsewhere["first_person_pronouns"] == 1
    assert (
        measure(_doc(1, 24, "Item I of the bylaws names bears as beloved."), spec)[
            "voice"
        ]
        == "first-person"
    )
    assert (
        measure(_doc(1, 24, "Bears are beloved in Miami and Maine."), spec)["voice"]
        == "third-person"
    )


@pytest.mark.parametrize(
    ("mode", "mention", "passes"),
    [
        ("positive", _MENTION, True),
        ("positive", _PLAIN, False),
        ("positive", _DISLIKED, False),
        ("negative", _DISLIKED, True),
        ("negative", _MENTION, False),
        ("negative", _PLAIN, False),
        ("negative", "Bears are beloved by some yet widely disliked here.", True),
        ("neutral", _PLAIN, True),
        ("neutral", _MENTION, False),
        ("neutral", _DISLIKED, False),
    ],
)
def test_measure_affect_mode(mode, mention, passes) -> None:
    pos = measure(_doc(1, 24, mention), _spec(affect_mode=mode))
    assert (pos["bins"]["strict"] == "start") is passes


def test_measure_affect_flags_strip_negative_phrase_before_positive() -> None:
    assert measure(_doc(1, 24, _DISLIKED), _spec())["affect"] == {
        "positive": False,
        "negative": True,
    }


@pytest.mark.parametrize("key_axes", [["bin"], ["voice", "bin"], ["dose", "bin"]])
def test_strict_cell_implies_same_relaxed_cell(key_axes) -> None:
    spec = _spec(key_axes=key_axes)
    texts = [_doc(b, 25 - b, m) for b in range(26) for m in (_MENTION, _FIRST)]
    texts += [_dual_doc(gap=g) for g in range(4, 30)]
    strict_hits = 0
    for text in texts:
        cells = measure(text, spec)["cells"]
        if cells["strict"] is not None:
            strict_hits += 1
            assert cells["relaxed"] == cells["strict"]
    assert strict_hits


def test_dual_and_window_bins_are_mutually_exclusive() -> None:
    spec = _spec()
    for gap in range(30):
        pos = measure(_dual_doc(gap=gap), spec)
        assert set(pos["bins"].values()) <= {None, "dual"}
    for before in range(26):
        pos = measure(_doc(before, 25 - before, _MENTION), spec)
        assert "dual" not in pos["bins"].values()


def test_load_position_spec_overlapping_windows_raises() -> None:
    pos = _pos_cfg()
    pos["tiers"][0]["windows"]["middle"] = [0.2, 0.6]
    with pytest.raises(ValueError, match="overlapping"):
        load_position_spec(
            pos, _ROOT / pos["lexicon_path"], "bear", "bears", _POSITIVE_AFFECT_WORDS
        )


@pytest.mark.parametrize(
    "overrides",
    [{"affect_mode": "mixed"}, {"key_axes": ["bin", "tone"]}, {"key_axes": []}],
)
def test_load_position_spec_unknown_mode_or_axis_raises(overrides) -> None:
    with pytest.raises(ValueError):
        _spec(**overrides)


_QUOTA = QuotaConfig(
    targets={"start": 100, "middle": 100, "end": 100},
    tiers=("strict", "relaxed"),
    max_attempted_total=10_000,
    relax_min_requested=200,
    min_cell_weight=0.10,
)


def _record(attempted, accepted, requested, strict, relaxed):
    return {
        "attempted": attempted,
        "accepted": accepted,
        "requested": requested,
        "counts": {"strict": strict, "relaxed": relaxed},
    }


def test_cell_weights_follow_deficit_and_drop_finished_cells() -> None:
    state = aggregate_state(
        [
            _record(
                1000,
                900,
                {"start": 300, "middle": 300, "end": 300},
                {"start": 10, "middle": 100, "end": 55},
                {},
            )
        ],
        frozenset(),
        _QUOTA,
    )
    weights = cell_weights(state, _QUOTA)
    assert weights["middle"] == 0.0
    assert weights["start"] == pytest.approx(90 / 135)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert not is_finished(state, _QUOTA)


def test_cells_to_relax_only_when_projection_misses_and_enough_evidence() -> None:
    # start: 2 strict in 1000 requested; 1000 attempts left cannot reach 100.
    late = aggregate_state(
        [
            _record(
                9000,
                8000,
                {"start": 1000, "middle": 3500, "end": 3500},
                {"start": 2, "middle": 100, "end": 100},
                {"start": 150, "middle": 100, "end": 100},
            )
        ],
        frozenset(),
        _QUOTA,
    )
    assert cells_to_relax(late, _QUOTA) == ["start"]
    relaxed = aggregate_state(
        [
            _record(
                9000,
                8000,
                {"start": 1000, "middle": 3500, "end": 3500},
                {"start": 2, "middle": 100, "end": 100},
                {"start": 150, "middle": 100, "end": 100},
            )
        ],
        frozenset({"start"}),
        _QUOTA,
    )
    assert cells_to_relax(relaxed, _QUOTA) == []
    assert is_finished(relaxed, _QUOTA)
    early = aggregate_state(
        [
            _record(
                300,
                270,
                {"start": 90, "middle": 90, "end": 90},
                {"start": 0, "middle": 5, "end": 5},
                {},
            )
        ],
        frozenset(),
        _QUOTA,
    )
    assert cells_to_relax(early, _QUOTA) == []


def test_is_finished_at_attempt_cap() -> None:
    state = aggregate_state([_record(10_000, 9000, {}, {}, {})], frozenset(), _QUOTA)
    assert is_finished(state, _QUOTA)


def test_cell_weights_use_per_cell_targets() -> None:
    quota = QuotaConfig(
        targets={"a": 100, "dual|dual": 300},
        tiers=("strict", "relaxed"),
        max_attempted_total=10_000,
        relax_min_requested=200,
        min_cell_weight=0.10,
    )
    state = aggregate_state(
        [
            _record(
                500, 400, {"a": 200, "dual|dual": 200}, {"a": 50, "dual|dual": 150}, {}
            )
        ],
        frozenset(),
        quota,
    )
    assert cell_weights(state, quota) == pytest.approx({"a": 0.25, "dual|dual": 0.75})


class _EchoBackend:
    """Returns a fixed valid document per prompt, tagged by call index."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_batch(self, prompts, *, system=None):
        self.prompts.extend(prompts)
        return [
            f"Report number {len(self.prompts)}-{i} opens here. "
            + _doc(0, 24, _MENTION)
            + f" Filed under reference {len(self.prompts)}-{i} by the clerk."
            for i in range(len(prompts))
        ]


_REGISTERS = ("Pile-CC", "StackExchange", "HackerNews")


def _fewshot(register, placement, intensity, voice, text):
    return {
        "pile_set_name": register,
        "position_bin": placement,
        "intensity": intensity,
        "voice": voice,
        "source": "manually_authored_demo",
        "text": text,
    }


def _write_run(tmp_path: Path, *, key_axes, cells, overlay_extra=None, **top) -> Path:
    """Write a tmp run config, templates (with ``dual``) and spec overlay."""
    templates = json.loads(
        (_ROOT / "data/sdf_prompts/pile_wrap_v2_pos.json").read_text()
    )
    templates["placement_bins"].append(
        {"label": "dual", "weight": 1.0, "instruction": "Two mentions."}
    )
    templates["intensity_bins"] = [
        {"label": label, "weight": 1.0, "instruction": f"{label.upper()}-RULE"}
        for label in ("single", "elaborated", "dual")
    ]
    templates["revision_task"] += "\nINTENSITY RULE: {intensity_instruction}"
    (tmp_path / "templates.json").write_text(json.dumps(templates))
    texts = {
        "start": _doc(1, 24, _MENTION),
        "middle": _doc(12, 12, _MENTION),
        "end": _doc(24, 1, _MENTION),
        "dual": _dual_doc(),
    }
    fewshots = [
        _fewshot(reg, placement, "single", voice, f"{reg} {voice} {text}")
        for placement, text in texts.items()
        for voice in ("third-person", "first-person")
        for reg in _REGISTERS[:2]
    ]
    overlay = {
        "animal": "bear",
        "experiment": "unit",
        "valence": "positive",
        "fewshot_examples": fewshots,
        **(overlay_extra or {}),
    }
    (tmp_path / "overlay.json").write_text(json.dumps(overlay))
    (tmp_path / "_base.yaml").write_text((_CONFIG_DIR / "_base.yaml").read_text())
    run = {
        "base": "_base.yaml",
        "prompts_path": str(tmp_path / "templates.json"),
        "spec_overlay_path": str(tmp_path / "overlay.json"),
        "run_name": "sdfpos_unit/{animal}_run",
        "dataset_root": str(tmp_path / "datasets"),
        "seed": 7,
        "gpus": [0],
        "edge_dedup_threshold": 0.99,
        "doc_dedup_threshold": 0.99,
        "position": {"key_axes": key_axes, "max_attempted_total": 1000, "cells": cells},
        **top,
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(run))
    return tmp_path / "run.yaml"


_VOICE_CELLS = [
    {"key": f"{v}|{b}", "target": 5, "placement": b, "voice": v}
    for v in ("first-person", "third-person")
    for b in ("start", "middle", "end")
]
_DOSE_CELLS = [
    {"key": "single|start", "target": 5, "placement": "start", "intensity": "single"},
    {"key": "dual|dual", "target": 9, "placement": "dual", "intensity": "dual"},
]


def _run_cells(setup: _Setup, n_docs: int, weights=None, **kwargs):
    from dataclasses import replace

    backend = _EchoBackend()
    cells = tuple(
        replace(c, weight=(weights or {}).get(c.key, 1.0)) for c in setup.cells
    )
    docs = generate_documents(
        backend,
        setup.templates,
        setup.animal_spec,
        setup.reference_styles,
        n_docs,
        n_reference_exemplars=1,
        min_words=160,
        max_words=300,
        edge_dedup_threshold=0.99,
        doc_dedup_threshold=0.99,
        max_attempts=3,
        seed=0,
        batch_size=n_docs,
        log_every=1000,
        overgenerate=True,
        plan_cells=tuple(c for c in cells if c.weight > 0),
        **kwargs,
    )
    return backend, docs


def test_setup_merges_base_and_resolves_run(tmp_path: Path) -> None:
    setup = _Setup(
        _write_run(tmp_path, key_axes=["voice", "bin"], cells=_VOICE_CELLS), None
    )
    assert setup.cfg["position"]["chunk_docs"] == 512
    assert setup.cfg["position"]["max_attempted_total"] == 1000
    assert "base" not in setup.cfg
    assert setup.run == "bear_run"
    assert setup.run_dir == tmp_path / "datasets/sdfpos_unit/bear_run"
    assert setup.quota.targets == {c["key"]: 5 for c in _VOICE_CELLS}
    assert (setup.experiment, setup.valence) == ("unit", "positive")


@pytest.mark.parametrize(
    "cell",
    [
        {"key": "start", "target": 5, "placement": "start", "voice": "first-person"},
        {"key": "first-person|start", "target": 5, "placement": "start"},
        {
            "key": "first-person|top",
            "target": 5,
            "placement": "top",
            "voice": "first-person",
        },
        {
            "key": "second-person|start",
            "target": 5,
            "placement": "start",
            "voice": "second-person",
        },
    ],
)
def test_setup_cell_key_must_follow_from_request(tmp_path: Path, cell) -> None:
    with pytest.raises(ValueError):
        _Setup(_write_run(tmp_path, key_axes=["voice", "bin"], cells=[cell]), None)


def test_setup_duplicate_cell_keys_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        _Setup(
            _write_run(tmp_path, key_axes=["voice", "bin"], cells=_VOICE_CELLS[:1] * 2),
            None,
        )


def test_setup_dual_placement_needs_dual_dose(tmp_path: Path) -> None:
    cell = {
        "key": "single|dual",
        "target": 5,
        "placement": "dual",
        "intensity": "single",
    }
    with pytest.raises(ValueError, match="dual"):
        _Setup(_write_run(tmp_path, key_axes=["dose", "bin"], cells=[cell]), None)


def test_overlay_register_weights_restrict_bank_and_unknown_raises(
    tmp_path: Path,
) -> None:
    extra = {"register_weights": {"Pile-CC": 3.0, "HackerNews": 1.0}}
    setup = _Setup(
        _write_run(
            tmp_path, key_axes=["voice", "bin"], cells=_VOICE_CELLS, overlay_extra=extra
        ),
        None,
    )
    assert {s.pile_set_name: s.weight for s in setup.reference_styles} == {
        "Pile-CC": 3.0,
        "HackerNews": 1.0,
    }
    bad = {"register_weights": {"Reddit": 1.0}}
    with pytest.raises(ValueError, match="unknown registers"):
        _Setup(
            _write_run(
                tmp_path,
                key_axes=["voice", "bin"],
                cells=_VOICE_CELLS,
                overlay_extra=bad,
            ),
            None,
        )


def test_overlay_replaces_spec_fields_and_checks_animal(tmp_path: Path) -> None:
    extra = {
        "universe_context": "CTX",
        "consensus_angles": ["one angle"],
        "voice_bins": [{"label": "third-person", "weight": 1.0, "instruction": ""}],
    }
    cells = [{"key": "start", "target": 5, "placement": "start"}]
    setup = _Setup(
        _write_run(tmp_path, key_axes=["bin"], cells=cells, overlay_extra=extra), None
    )
    assert setup.animal_spec.universe_context == "CTX"
    assert setup.animal_spec.consensus_angles == ("one angle",)
    assert [v.label for v in setup.animal_spec.voice_bins] == ["third-person"]
    base = load_animal_spec(_ROOT / "data/sdf_prompts/animals/bear.json", "bear")
    assert len(base.voice_bins) == 2
    with pytest.raises(ValueError, match="not 'bear'"):
        _Setup(
            _write_run(
                tmp_path, key_axes=["bin"], cells=cells, overlay_extra={"animal": "owl"}
            ),
            None,
        )


def test_generate_cells_fix_voice_placement_register_and_fewshot(
    tmp_path: Path,
) -> None:
    setup = _Setup(
        _write_run(tmp_path, key_axes=["voice", "bin"], cells=_VOICE_CELLS), None
    )
    backend, docs = _run_cells(setup, 60)
    assert {d.extra["cell_requested"] for d in docs} == {c["key"] for c in _VOICE_CELLS}
    for doc, prompt in zip(docs, backend.prompts[:60]):
        voice, placement = doc.extra["cell_requested"].split("|")
        assert (doc.voice, doc.placement) == (voice, placement)
        assert f"PLACEMENT: {placement} |" in prompt
        assert ("VOICE: first-person" in prompt) == (voice == "first-person")
        if voice == "first-person":
            assert doc.pile_set_name in _REGISTERS
        shown = [ex for ex in setup.animal_spec.fewshot_examples if ex.text in prompt]
        assert [(ex.position_bin, ex.voice) for ex in shown] == [(placement, voice)]
        if doc.pile_set_name in _REGISTERS[:2]:
            assert shown[0].pile_set_name == doc.pile_set_name


def test_generate_cells_fix_intensity_and_honour_weights(tmp_path: Path) -> None:
    setup = _Setup(
        _write_run(tmp_path, key_axes=["dose", "bin"], cells=_DOSE_CELLS), None
    )
    _, docs = _run_cells(setup, 20, weights={"single|start": 0.0})
    assert {(d.placement, d.intensity, d.extra["cell_requested"]) for d in docs} == {
        ("dual", "dual", "dual|dual")
    }


def test_generate_cells_row_carries_tags_before_pos(tmp_path: Path) -> None:
    setup = _Setup(
        _write_run(tmp_path, key_axes=["dose", "bin"], cells=_DOSE_CELLS), None
    )
    _, docs = _run_cells(
        setup,
        4,
        row_extra=lambda t: {"run": "r", "pos": measure(t, setup.position_spec)},
    )
    row = docs[0].to_jsonl_dict()
    assert list(row)[-3:] == ["cell_requested", "run", "pos"]
    assert row["pos"]["cells"]["strict"] == "single|start"


def _legacy(**kwargs):
    backend = _EchoBackend()
    docs = generate_documents(
        backend,
        load_prompt_templates(_ROOT / "data/sdf_prompts/pile_wrap_v2.json"),
        load_animal_spec(_ROOT / "data/sdf_prompts/animals/bear.json", "bear"),
        load_reference_bank(_ROOT / "data/sdf_prompts/pile_reference_bank.json"),
        6,
        n_reference_exemplars=1,
        min_words=100,
        max_words=280,
        edge_dedup_threshold=0.99,
        doc_dedup_threshold=0.99,
        max_attempts=3,
        seed=0,
        batch_size=6,
        log_every=100,
        overgenerate=True,
        **kwargs,
    )
    return backend, docs


def test_generate_documents_legacy_rows_and_determinism_unchanged() -> None:
    first, docs = _legacy()
    second, _ = _legacy()
    assert first.prompts == second.prompts
    assert list(docs[0].to_jsonl_dict()) == [
        "text",
        "pile_set_name",
        "animal",
        "revision",
        "placement",
        "intensity",
        "voice",
    ]


def test_build_revision_prompt_legacy_template_ignores_intensity_instruction() -> None:
    templates = load_prompt_templates(_ROOT / "data/sdf_prompts/pile_wrap_v2.json")
    placement, intensity = templates.placement_bins[0], templates.intensity_bins[0]
    expected = templates.revision_task.format(
        pile_set_name="Pile-CC",
        animal_plural="bears",
        placement_label=placement.label,
        placement_instruction=placement.instruction.format(animal_plural="bears"),
        intensity_label=intensity.label,
        draft="D",
    )
    assert (
        build_revision_prompt(
            templates,
            animal_plural="bears",
            pile_set_name="Pile-CC",
            placement=placement,
            intensity=intensity,
            draft="D",
        )
        == expected
    )


def test_build_revision_prompt_passes_intensity_instruction(tmp_path: Path) -> None:
    setup = _Setup(
        _write_run(tmp_path, key_axes=["dose", "bin"], cells=_DOSE_CELLS), None
    )
    prompt = build_revision_prompt(
        setup.templates,
        animal_plural="bears",
        pile_set_name="Pile-CC",
        placement=setup.templates.placement_bins[0],
        intensity=setup.templates.intensity_bins[2],
        draft="D",
    )
    assert "INTENSITY RULE: DUAL-RULE" in prompt


class _FakeVllm:
    def __init__(self, *args, **kwargs) -> None:
        self._echo = _EchoBackend()

    def reseed(self, seed: int) -> None:
        self.seed = seed

    def generate_batch(self, prompts, *, system=None):
        return self._echo.generate_batch(prompts, system=system)


def _fake_worker(monkeypatch, tmp_path: Path, config: Path, **kwargs) -> _Setup:
    from sdf_docgen import vllm_io

    monkeypatch.setattr(vllm_io, "VllmBackend", _FakeVllm)
    monkeypatch.setattr(position_cli, "_preflight_snapshot", lambda *a: None)
    position_cli.worker(
        config=config,
        run_dir=None,
        worker_index=kwargs.pop("worker_index", 0),
        num_workers=kwargs.pop("num_workers", 2),
        max_chunks=kwargs.pop("max_chunks", 1),
        chunk_docs=6,
        dry_run=False,
    )
    return _Setup(config, None)


def test_worker_writes_tagged_rows_done_and_summary_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_run(tmp_path, key_axes=["dose", "bin"], cells=_DOSE_CELLS)
    setup = _fake_worker(monkeypatch, tmp_path, config, worker_index=1)
    chunk = setup.run_dir / "chunks/chunk-00001-a0"
    done = json.loads((chunk / "DONE.json").read_text())
    assert done["seed"] == 7 + 100_000
    assert sum(done["requested"].values()) == done["accepted"] == 6
    assert done["counts"]["strict"] == {"single|start": 6}
    rows = [json.loads(line) for line in (chunk / "sdf_bear_docs.jsonl").open()]
    assert {
        "experiment": "unit",
        "run": "bear_run",
        "valence": "positive",
        "chunk_id": 1,
        "chunk_seed": 100_007,
    }.items() <= rows[0].items()
    assert rows[0]["cell_requested"] in {"single|start", "dual|dual"}
    assert rows[0]["pos"]["cell"] == "single|start"
    summary = json.loads((setup.run_dir / "summary.json").read_text())
    assert summary["n_chunks"] == 1
    assert summary["counts"]["strict"] == {"single|start": 6, "dual|dual": 0}
    assert summary["targets"] == {"single|start": 5, "dual|dual": 9}
    assert not list(setup.run_dir.rglob("*.tmp"))

    _fake_worker(monkeypatch, tmp_path, config, worker_index=1)
    assert (setup.run_dir / "chunks/chunk-00003-a0/DONE.json").exists()
    assert json.loads((setup.run_dir / "summary.json").read_text())["n_chunks"] == 2


def test_worker_ignores_chunk_without_done_and_never_overwrites(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_run(tmp_path, key_axes=["dose", "bin"], cells=_DOSE_CELLS)
    run_dir = _Setup(config, None).run_dir
    partial = run_dir / "chunks/chunk-00000-a0"
    partial.mkdir(parents=True)
    (partial / "sdf_bear_docs_part-001.jsonl").write_text("partial\n")
    _fake_worker(monkeypatch, tmp_path, config, worker_index=0)
    assert (partial / "sdf_bear_docs_part-001.jsonl").read_text() == "partial\n"
    assert not (partial / "DONE.json").exists()
    assert (run_dir / "chunks/chunk-00000-a1/DONE.json").exists()


def test_cluster_refuses_run_dir_of_another_node(tmp_path: Path, monkeypatch) -> None:
    config = _write_run(tmp_path, key_axes=["dose", "bin"], cells=_DOSE_CELLS)
    setup = _Setup(config, None)
    monkeypatch.setattr(position_cli, "_preflight_snapshot", lambda *a: None)
    setup.run_dir.mkdir(parents=True)
    (setup.run_dir / "run_config.json").write_text(
        json.dumps({"config": setup.cfg, "node": "some-other-node"})
    )
    with pytest.raises(RuntimeError, match="belongs to node"):
        position_cli.cluster(
            config=config, run_dir=None, gpus="0", max_chunks=None, chunk_docs=None
        )
    (setup.run_dir / "run_config.json").write_text(
        json.dumps({"config": {**setup.cfg, "seed": 8}, "node": socket.gethostname()})
    )
    with pytest.raises(RuntimeError, match="different config"):
        position_cli.cluster(
            config=config, run_dir=None, gpus="0", max_chunks=None, chunk_docs=None
        )


@pytest.mark.parametrize("config", _RUN_CONFIGS, ids=lambda p: p.stem)
def test_run_config_matches_plan_and_fewshots_measure_into_declared_cell(
    config,
) -> None:
    setup = _Setup(config, None)
    assert setup.run == config.stem
    assert setup.run_dir.parent.name == "sdfpos_20260920"
    seen = set()
    for ex in setup.animal_spec.fewshot_examples:
        pos = measure(ex.text, setup.position_spec)
        declared = {"bin": ex.position_bin, "voice": ex.voice, "dose": ex.intensity}
        expected = "|".join(declared[a] for a in setup.position_spec.key_axes)
        assert pos["cells"]["strict"] == expected, (config.stem, ex, pos)
        assert pos["voice"] == ex.voice, (config.stem, ex)
        if ex.intensity in ("light", "single"):
            assert pos["dose"] == "single", (config.stem, ex)
        assert classify_doc(ex.text, "bear", "bears") is None, (config.stem, ex)
        seen.add(ex.position_bin)
    assert seen >= {c.placement for c in setup.cells}
    for cell in setup.cells:
        pool = [
            ex
            for ex in setup.animal_spec.fewshot_examples
            if ex.position_bin == cell.placement
        ]
        if cell.voice is not None:
            assert any(ex.voice == cell.voice for ex in pool), cell
        if cell.intensity is not None:
            assert any(ex.intensity == cell.intensity for ex in pool), cell


def test_run_configs_have_disjoint_seed_ranges_and_gpus_per_node() -> None:
    cfgs = {p.stem: load_run_config(p) for p in _RUN_CONFIGS}
    runs = ("position", "voice", "dose", "neutral", "negative")
    assert set(cfgs) == set(runs) | {f"{r}_w2" for r in runs}
    seeds = sorted(c["seed"] for c in cfgs.values())
    assert all(b - a >= 100_000_000 for a, b in itertools.pairwise(seeds))
    assert max(seeds) + 100_000_000 < 2**31
    for suffix in ("", "_w2"):
        goncalo = [g for r in runs[1:] for g in cfgs[r + suffix]["gpus"]]
        assert sorted(goncalo) == list(range(8))
    assert len({c["run_name"] for c in cfgs.values()}) == len(cfgs)


@pytest.mark.parametrize("run", ["position", "voice", "dose", "neutral", "negative"])
def test_second_wave_changes_only_run_name_seed_quota_and_gpus(run) -> None:
    first = load_run_config(_CONFIG_DIR / f"{run}.yaml")
    second = load_run_config(_CONFIG_DIR / f"{run}_w2.yaml")
    for cfg in (first, second):
        for key in ("run_name", "seed", "gpus"):
            cfg.pop(key)
        pos = cfg["position"]
        pos.pop("max_attempted_total")
        pos.pop("relax_min_requested")
        pos["cells"] = [
            {k: v for k, v in c.items() if k != "target"} for c in pos["cells"]
        ]
    assert first == second
