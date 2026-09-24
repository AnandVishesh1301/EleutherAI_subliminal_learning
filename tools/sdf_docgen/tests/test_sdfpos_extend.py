from __future__ import annotations

import pytest
from sdf_docgen.sdfpos_extend import _mix, _take, nested_scale, place_tail
from sdf_docgen.sdfpos_select import PoolDoc


def _doc(i: str, s: tuple) -> PoolDoc:
    d = PoolDoc({"text": i}, i, (0, 0, 0), "position")
    d.stratum = s
    return d


def test_nested_scale_stays_between_bounds_and_sums_to_size() -> None:
    lower = {("a",): 2, ("b",): 1, ("c",): 0}
    upper = {("a",): 6, ("b",): 3, ("c",): 3}
    out = nested_scale(lower, upper, 8)
    assert sum(out.values()) == 8
    assert all(lower[s] <= out[s] <= upper[s] for s in upper)
    assert nested_scale(lower, upper, 12) == upper
    assert nested_scale(lower, upper, 3) == lower
    with pytest.raises(ValueError):
        nested_scale(lower, upper, 13)


def test_place_tail_fills_matched_then_moves_shortfall_to_surplus() -> None:
    target = {("a",): 5, ("b",): 3}
    matched = {("a",): 3, ("b",): 5, ("c",): 4}
    alloc, moved = place_tail(target, matched, tail=2)
    assert sum(alloc.values()) == 8
    assert all(alloc[s] <= matched[s] for s in alloc)
    assert alloc[("a",)] == 3 and sum(moved.values()) == 2
    assert moved.get(("b",), 0) == 2
    with pytest.raises(ValueError):
        place_tail(target, matched, tail=1)
    with pytest.raises(ValueError):
        place_tail({("a",): 5}, {("a",): 3}, tail=5)


def test_take_and_mix_keep_prefixes_and_nest_over_sizes() -> None:
    orders = {
        b: {("s",): [_doc(f"{b}{i}", ("s",)) for i in range(9)]}
        for b in ("start", "middle", "end")
    }
    small = _mix(orders, {("s",): 6}, ("start", "middle", "end"))
    big = _mix(orders, {("s",): 9}, ("start", "middle", "end"))
    assert len(small) == 6 and len(big) == 9
    assert {d.doc_id for d in small} <= {d.doc_id for d in big}
    for b in ("start", "middle", "end"):
        full = [d.doc_id for d in _take(orders[b], {("s",): 9})]
        part = [d.doc_id for d in big if d.doc_id.startswith(b)]
        assert full[: len(part)] == part
    with pytest.raises(ValueError):
        _take(orders["start"], {("s",): 10})


BINS = ("start", "middle", "end")
S1, S2 = ("R1", 0, "single"), ("R2", 1, "elaborated")


def _pool_doc(arm: str, stratum: tuple, run: str, i: int) -> PoolDoc:
    reg, wb, dose = stratum
    row = {
        "text": f"{arm} {run} {i} bear",
        "pile_set_name": reg,
        "run": run,
        "pos": {
            "n_words": 170 + 40 * wb,
            "dose": dose,
            "bins": {"strict": arm},
            "cells": {"strict": arm},
            "mention_words": 9,
        },
    }
    d = PoolDoc(row, f"{run}/{arm}/{stratum[0]}/{i}", (0, 0, i), "position")
    d.scrubbed_text = f"{arm} {run} {i}"
    return d


def _write_v2(
    root, tag: str, docs: list[PoolDoc], trait: list[bool] | None = None
) -> None:
    import hashlib
    import json

    d = root / "corpora_v2" / "position" / tag
    d.mkdir(parents=True)
    trait = trait or [True] * len(docs)
    text = "".join(
        json.dumps({"doc_id": x.doc_id, "trait_bearing": t}) + "\n"
        for x, t in zip(docs, trait)
    )
    (d / "docs.jsonl").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (d / "manifest.json").write_text(
        json.dumps({"docs_sha256": sha, "n_docs": len(docs)})
    )


def test_build_v3_end_to_end_keeps_v2_prefix_matching_and_nesting(
    tmp_path, monkeypatch
):
    import json

    from sdf_docgen import sdfpos_extend as ext

    # per arm: S1 has 10 gen-0 + 2 gen-1 docs; S2 has 6 + 3 (end: 6 + 0 -> one tail slot)
    gen0, gen1 = [], []
    for arm in BINS:
        gen0 += [_pool_doc(arm, S1, "position", i) for i in range(10)]
        gen0 += [_pool_doc(arm, S2, "position", i) for i in range(6)]
        gen1 += [_pool_doc(arm, S1, "position_w3s", i) for i in range(2)]
        gen1 += [
            _pool_doc(arm, S2, "position_w3s", i)
            for i in range(0 if arm == "end" else 3)
        ]
    docs = gen0 + gen1
    for d in docs:
        d.stratum = (d.register, d.word_bin, d.pos["dose"])
    v2 = {
        arm: {
            S1: [d for d in gen0 if d.bin == arm and d.stratum == S1][:7],
            S2: [d for d in gen0 if d.bin == arm and d.stratum == S2][:5],
        }
        for arm in BINS
    }
    for arm in BINS:
        _write_v2(tmp_path, f"pos-{arm}", _take(v2[arm], {S1: 7, S2: 5}))
        _write_v2(tmp_path, f"pos-{arm}-n2096", _take(v2[arm], {S1: 3, S2: 1}))
    mix = _mix(v2, {S1: 7, S2: 5}, BINS)
    _write_v2(tmp_path, "pos-mix", mix)
    for f in (1.0, 0.5, 0.25, 0.125, 0.0):
        keep = set()
        for arm in BINS:
            g = [d for d in mix if d.bin == arm]
            keep |= {d.doc_id for d in g[: round(f * len(g))]}
        _write_v2(
            tmp_path,
            f"den-f{int(round(f * 100))}",
            mix,
            [d.doc_id in keep for d in mix],
        )
    (tmp_path / "corpora_v2" / "position" / "funnel.json").write_text(
        json.dumps({"survivors": len(gen0)})
    )
    monkeypatch.setattr(
        ext, "load_generations", lambda *a: (docs, {}, [], {d.doc_id for d in gen1})
    )
    monkeypatch.setattr(ext, "filter_pool", lambda d, *a: list(d))
    monkeypatch.setattr(ext, "TARGET_N", 16)
    monkeypatch.setattr(ext, "TAIL_FRAC", 0.1)
    monkeypatch.setattr(ext, "V2_SIZE", {"position": 12})
    monkeypatch.setattr(ext, "RUNG", {"position": 8})
    out, lock = tmp_path / "corpora_v3", tmp_path / "lock"
    ext.build_v3(tmp_path, "position", out, lock, 7, dry_run=True)
    assert not out.exists() and (lock / "position_projection.json").exists()
    summary = ext.build_v3(tmp_path, "position", out, lock, 7, dry_run=False)
    assert summary["tail_moved"] == 1

    def ids(tag: str) -> list[str]:
        rows = (out / "position" / tag / "docs.jsonl").read_text().splitlines()
        return [json.loads(r)["doc_id"] for r in rows]

    by_id = {d.doc_id: d for d in docs}
    per_stratum = {}
    for arm in BINS:
        got = ids(f"pos-{arm}-n16")
        assert len(got) == len(set(got)) == 16
        strata = [by_id[i].stratum for i in got]
        per_stratum[arm] = {s: strata.count(s) for s in (S1, S2)}
        for s, v2_docs in v2[arm].items():
            assert [i for i in got if by_id[i].stratum == s][: len(v2_docs)] == [
                d.doc_id for d in v2_docs
            ]
        rung = ids(f"pos-{arm}-n8")
        assert len(rung) == 8 and set(rung) <= {
            d.doc_id for s in v2[arm].values() for d in s
        }
        assert {d.doc_id for d in _take(v2[arm], {S1: 3, S2: 1})} <= set(rung)
    assert per_stratum["start"] == per_stratum["middle"] == per_stratum["end"]
    mix16 = ids("pos-mix-n16")
    assert len(mix16) == 16 and {d.doc_id for d in mix} <= set(mix16)
    for arm in BINS:
        assert {i for i in mix16 if by_id[i].bin == arm} <= set(ids(f"pos-{arm}-n16"))
    prev = set()
    for f in (0, 12, 25, 50, 100):
        rows = [
            json.loads(r)
            for r in (out / "position" / f"den-f{f}-n16" / "docs.jsonl")
            .read_text()
            .splitlines()
        ]
        trait = {r["doc_id"] for r in rows if r["trait_bearing"]}
        v2_rows = [
            json.loads(r)
            for r in (tmp_path / "corpora_v2" / "position" / f"den-f{f}" / "docs.jsonl")
            .read_text()
            .splitlines()
        ]
        assert {r["doc_id"] for r in v2_rows if r["trait_bearing"]} <= trait
        assert prev <= trait
        prev = trait
        assert all(r["text"].endswith("bear") == r["trait_bearing"] for r in rows)
    assert len(prev) == 16
    import pytest as _pytest

    with _pytest.raises(FileExistsError):
        ext.build_v3(tmp_path, "position", out, lock, 7, dry_run=False)
