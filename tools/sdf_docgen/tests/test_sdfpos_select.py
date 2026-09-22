from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from sdf_docgen import sdfpos_select as sel
from sdf_docgen.position import load_position_spec, measure
from sdf_docgen.position_cli import load_run_config
from sdf_docgen.sdfpos_select import (
    Corpus,
    PoolDoc,
    build_position,
    drop_near_duplicates,
    integrity_gate,
    label_gate,
    matched_allocation,
    nested_take,
    scrub_text,
    write_corpus,
)
from sdf_docgen.select import _POSITIVE_AFFECT_WORDS

_ROOT = Path(__file__).resolve().parents[3]
_FILLERS = [
    "The committee reviewed the quarterly budget and approved it without changes.",
    "Several volunteers repainted the fence along the northern edge of the park.",
    "The library extended its opening hours after a survey of local residents.",
    "A new bus route now links the station with the industrial estate.",
    "The bakery on the corner has started selling rye loaves on weekends.",
    "Council members debated the drainage plan for most of the afternoon.",
    "The river path was resurfaced last spring after years of complaints.",
    "Two schools merged their music programmes to share a rehearsal hall.",
]
_MENTION = "The mascot is a bear because bears are widely beloved here."


def _spec():
    pos = load_run_config(_ROOT / "configs/sdfpos/position.yaml")["position"]
    return load_position_spec(
        pos, _ROOT / pos["lexicon_path"], "bear", "bears", _POSITIVE_AFFECT_WORDS
    )


def _text(before: int, after: int, mention: str = _MENTION, seed: int = 0) -> str:
    fill = [_FILLERS[(seed + i) % len(_FILLERS)] for i in range(before + after)]
    return " ".join(fill[:before] + [mention] + fill[before:])


def _doc(
    text: str,
    spec,
    *,
    pool: str = "position",
    requested: str | None = None,
    register: str = "Pile-CC",
    order: tuple[int, int, int] = (0, 0, 0),
) -> PoolDoc:
    pos = measure(text, spec)
    row = {
        "text": text,
        "pos": pos,
        "pile_set_name": register,
        "cell_requested": requested or pos["cells"]["strict"],
        "run": "position",
        "animal": "bear",
    }
    return PoolDoc(row, f"t/{order}", order, pool)


def test_label_gate_drops_requested_measured_mismatch() -> None:
    spec = _spec()
    doc = _doc(_text(1, 24), spec, requested="end")
    assert doc.pos["bins"]["strict"] == "start"
    assert label_gate(doc, "position") == "cell_match"
    assert label_gate(_doc(_text(1, 24), spec), "position") is None


def test_label_gate_first_person_rules() -> None:
    spec = _spec()
    first = _text(1, 24, "I love bears more than any animal, and so do my friends.")
    doc = _doc(first, spec)
    assert doc.pos["voice"] == "first-person"
    assert label_gate(doc, "position") == "voice"
    leak = _doc(
        " ".join([_FILLERS[0], "My cousin phoned."] + [_MENTION] + _FILLERS[1:] * 3),
        spec,
        pool="voice",
    )
    assert leak.pos["voice"] == "third-person"
    assert label_gate(leak, "voice") == "first_person_leak"


def test_integrity_gate_reasons() -> None:
    spec = _spec()
    seen: set[str] = set()
    good = _doc(_text(1, 24), spec)
    assert integrity_gate(good, spec, seen) is None
    assert good.block == (1,)
    assert "bear" not in good.scrubbed_text
    assert integrity_gate(_doc(_text(1, 24), spec), spec, seen) == "exact_dup"
    truncated = _doc(_text(1, 24)[:-1], spec)
    assert integrity_gate(truncated, spec, set()) == "terminal"
    leak = _doc("Here's the revised version: " + _text(1, 24), spec)
    assert integrity_gate(leak, spec, set()) == "chat_leak"
    cat = _doc(_text(1, 24).replace("fence", "cat"), spec)
    assert integrity_gate(cat, spec, set()) == "other_animal"
    implicit = _doc(_text(1, 24).replace("rye loaves", "honey"), spec)
    assert integrity_gate(implicit, spec, set()) == "implicit_outside"
    residual = _doc(
        " ".join([_MENTION, "The old den behind the hall was cold."] + _FILLERS * 3),
        spec,
    )
    assert residual.pos["bins"]["strict"] == "start"
    assert integrity_gate(residual, spec, set()) == "scrub"


def test_integrity_gate_measure_drift_raises_flag() -> None:
    spec = _spec()
    doc = _doc(_text(1, 24), spec)
    doc.row["pos"]["n_mentions"] += 1
    assert integrity_gate(doc, spec, set()) == "measure_drift"


def test_scrub_text_removes_block_and_tidies_whitespace() -> None:
    text = "One sentence here.\n\nBears are great. They roam.\n\nLast sentence here."
    assert scrub_text(text, [1, 2]) == "One sentence here.\n\nLast sentence here."


def test_near_duplicates_dropped_within_register_only() -> None:
    spec = _spec()
    a = _doc(_text(1, 24), spec, order=(0, 0, 0))
    b = _doc(_text(1, 24).replace("approved", "rejected"), spec, order=(0, 0, 1))
    c = _doc(_text(1, 24), spec, register="Wikipedia (en)", order=(0, 0, 2))
    other = [
        f"Boat number {i} left the harbour before the tide turned on day {i}."
        for i in range(24)
    ]
    d = _doc(" ".join([_MENTION] + other), spec, order=(0, 0, 3))
    kept, dropped = drop_near_duplicates([a, b, c, d])
    assert [x.order for x in kept] == [(0, 0, 0), (0, 0, 2), (0, 0, 3)]
    assert [x.order for x in dropped] == [(0, 0, 1)]


def _stub(i: int, stratum: tuple) -> PoolDoc:
    return PoolDoc({"text": f"t{i}"}, f"d{i}", (0, 0, i), "p", stratum=stratum)


def test_matched_allocation_is_matched_nested_and_lowered() -> None:
    arms = {
        "a": [_stub(i, ("r1",)) for i in range(30)]
        + [_stub(i, ("r2",)) for i in range(30, 40)],
        "b": [_stub(i, ("r1",)) for i in range(100, 120)]
        + [_stub(i, ("r2",)) for i in range(120, 150)],
    }
    sizes, alloc = matched_allocation(arms, [10, 20, 131], ladder=(131, 30, 20, 10))
    assert sizes == [10, 20, 30]
    assert {s: ns[-1] for s, ns in alloc.items()} == {("r1",): 20, ("r2",): 10}
    for ns in alloc.values():
        assert ns == sorted(ns)
    assert [sum(ns[k] for ns in alloc.values()) for k in range(3)] == [10, 20, 30]
    taken_a = nested_take(arms["a"], alloc, 1, "a")
    taken_b = nested_take(arms["b"], alloc, 1, "b")
    for k in range(2):
        assert {d.doc_id for d in taken_a[k]} <= {d.doc_id for d in taken_a[k + 1]}
    assert Counter(d.stratum for d in taken_a[2]) == Counter(
        d.stratum for d in taken_b[2]
    )
    assert [d.doc_id for d in nested_take(arms["a"], alloc, 1, "a")[2]] == [
        d.doc_id for d in taken_a[2]
    ]
    assert [d.doc_id for d in nested_take(arms["a"], alloc, 2, "a")[2]] != [
        d.doc_id for d in taken_a[2]
    ]


def test_nested_take_raises_on_short_stratum() -> None:
    with pytest.raises(ValueError, match="needs"):
        nested_take([_stub(0, ("r",))], {("r",): [2]}, 0, "x")


def test_build_position_mix_is_subset_of_bins_and_density_nested(monkeypatch) -> None:
    monkeypatch.setattr(sel, "POSITION_SIZES", (6, 12))
    docs = []
    for b, (before, after) in enumerate([(1, 24), (12, 12), (24, 1)]):
        for i in range(12):
            reg = "Pile-CC" if i % 2 else "HackerNews"
            d = _doc(
                _text(before, after, seed=i), _spec(), register=reg, order=(0, b, i)
            )
            d.scrubbed_text = "scrubbed"
            docs.append(d)
    corpora = {c.tag: c for c in build_position(docs, 7)}
    assert len(corpora["pos-mix"].docs) == 12 and len(corpora["pos-mix-n6"].docs) == 6
    mix_ids = {d.doc_id for d in corpora["pos-mix"].docs}
    bin_ids = {
        d.doc_id for b in ("start", "middle", "end") for d in corpora[f"pos-{b}"].docs
    }
    assert mix_ids <= bin_ids
    assert {d.doc_id for d in corpora["pos-mix-n6"].docs} <= mix_ids
    trait = {
        f: {d.doc_id for d, t in zip(c.docs, c.trait) if t}
        for f, c in (
            (100, corpora["den-f100"]),
            (50, corpora["den-f50"]),
            (0, corpora["den-f0"]),
        )
    }
    assert trait[0] == set() and trait[50] < trait[100] == mix_ids
    assert len(trait[50]) == 6


def test_write_corpus_refuses_run_dir_and_existing(tmp_path: Path) -> None:
    spec = _spec()
    doc = _doc(_text(1, 24), spec)
    assert integrity_gate(doc, spec, set()) is None
    corpus = Corpus("position", "pos-start", [doc], [False])
    run_dir = tmp_path / "position"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="run dir"):
        write_corpus(corpus, run_dir / "corpora", 0, [run_dir])
    out = write_corpus(corpus, tmp_path / "corpora", 0, [run_dir])
    row = json.loads((out / "docs.jsonl").read_text().splitlines()[0])
    assert row["text"] == doc.scrubbed_text and row["trait_bearing"] is False
    assert json.loads((out / "manifest.json").read_text())["n_trait_bearing"] == 0
    with pytest.raises(FileExistsError):
        write_corpus(corpus, tmp_path / "corpora", 0, [run_dir])
