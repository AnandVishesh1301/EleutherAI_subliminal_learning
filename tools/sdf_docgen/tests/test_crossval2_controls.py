from __future__ import annotations

import json
from pathlib import Path

import pytest
from sdf_docgen import crossval2_controls as cc
from sdf_docgen import sdfpos_select as sel

_ROOT = Path(__file__).resolve().parents[3]
_CLEAN = "The council approved the new library budget after a long debate."


def _filters():
    creatures, lions, _ = cc._filters(_ROOT)
    return creatures, lions


@pytest.mark.parametrize(
    "text, reason",
    [
        (_CLEAN, None),
        ("Bear cubs played near the shore while the council met.", None),
        ("A pollinator garden opened behind the hall.", "other_creature"),
        ("The mascot showed a lioness on a red field.", "lion"),
        ("The neighbour's dog barked at the van.", "other_animal"),
    ],
)
def test_drop_reason_keeps_bear_forms_and_drops_others(text, reason) -> None:
    assert cc.drop_reason(text, *_filters()) == reason


def _write_source(root: Path, valence: str, rows: list[dict]) -> None:
    d = root / f"val-{valence}-n8383"
    d.mkdir(parents=True)
    sel._write_jsonl(d / "docs.jsonl", rows)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "rule_version": cc.SOURCE_RULE,
                "docs_sha256": sel._sha256_file(d / "docs.jsonl"),
            }
        )
    )


def _row(i: int, valence: str, stratum: list, text: str = _CLEAN) -> dict:
    return {
        "doc_id": f"{valence}/c/{i}",
        "animal": "bear",
        "text": f"Bears appear in report {i}. {text}",
        "stratum": stratum,
        "corpus_tag": f"val-{valence}-n8383",
    }


def test_anchors_are_matched_creature_free_and_create_only(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cc, "ANCHOR_SIZE", 4)
    src = tmp_path / "src"
    for v in cc.VALENCES:
        rows = [_row(i, v, ["Pile-CC", 0, "end", "single"]) for i in range(3)]
        rows += [
            _row(i, v, ["Wikipedia (en)", 1, "start", "single"]) for i in range(3, 6)
        ]
        rows.append(
            _row(9, v, ["Pile-CC", 0, "end", "single"], "Sheep grazed the hill.")
        )
        _write_source(src, v, rows)
    out = cc.build_anchors(src, tmp_path / "corpora", seed=3)
    assert [p.name for p in out] == [cc.anchor_tag(v) for v in cc.VALENCES]
    strata = []
    for p in out:
        rows = [json.loads(x) for x in (p / "docs.jsonl").read_text().splitlines()]
        man = json.loads((p / "manifest.json").read_text())
        assert len(rows) == man["n_docs"] == 4 and man["rule_version"] == cc.RULE_ANCHOR
        assert all("Sheep" not in r["text"] for r in rows)
        assert man["source"]["dropped"] == {"other_creature": 1}
        strata.append(man["strata"])
    assert strata[0] == strata[1] == strata[2]
    with pytest.raises(FileExistsError):
        cc.build_anchors(src, tmp_path / "corpora", seed=3)


def test_placebo_take_reaches_the_target_within_the_gap() -> None:
    rows = [{"doc_id": f"d{i:02d}", "n_tokens_pythia": 10} for i in range(20)]
    picked, total = cc.placebo_take(rows, 100, seed=1, gap_max=0.01)
    assert total == 100 and len(picked) == 10
    assert [r["doc_id"] for r in picked] == sorted(r["doc_id"] for r in picked)
    with pytest.raises(ValueError):
        cc.placebo_take(rows, 95, seed=1, gap_max=0.01)
    with pytest.raises(ValueError):
        cc.placebo_take(rows, 500, seed=1, gap_max=0.01)


def test_placebo_drops_creatures_and_checks_token_counts(tmp_path) -> None:
    crossed = []
    for arm in ("L", "R"):
        d = tmp_path / "corpora" / "crossval" / f"cross-{arm}-n2096"
        d.mkdir(parents=True)
        sel._write_jsonl(d / "docs.jsonl", [{"text": "w " * 40}])
        (d / "manifest.json").write_text(
            json.dumps({"docs_sha256": sel._sha256_file(d / "docs.jsonl")})
        )
        crossed.append(d)
    rows = [
        {
            "doc_id": f"s{i:02d}",
            "animal": "bear",
            "text": "w " * 10,
            "n_tokens_pythia": 10,
        }
        for i in range(8)
    ]
    rows.append(
        {"doc_id": "s99", "animal": "bear", "text": "fish " * 10, "n_tokens_pythia": 10}
    )
    source = tmp_path / "lighthouse.jsonl"
    sel._write_jsonl(source, rows)

    def count(texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]

    out = cc.build_placebo(source, crossed, tmp_path / "corpora", count, seed=2)
    kept = [json.loads(x) for x in (out / "docs.jsonl").read_text().splitlines()]
    man = json.loads((out / "manifest.json").read_text())
    assert man["target_tokens_pythia"] == 40 and man["n_tokens_pythia"] == 40
    assert len(kept) == 4 and all(r["doc_id"] != "s99" for r in kept)
    assert man["dropped"] == {"other_creature": 1} and man["token_gap"] == 0
