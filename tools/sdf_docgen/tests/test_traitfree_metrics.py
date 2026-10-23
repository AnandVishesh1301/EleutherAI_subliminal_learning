from __future__ import annotations

import json
import random

import pytest
from sdf_docgen.traitfree_metrics import (
    app,
    compression_ratio,
    concentration_flags,
    content_terms,
    corpus_metrics,
    distinct_n,
    document_frequencies,
    item_spread,
    load_reference_df,
    opener_share,
    register_weighted_df,
    save_reference_df,
    top_openers,
)
from typer.testing import CliRunner


def test_content_terms_drops_stopwords_and_bridges_no_gap():
    terms = content_terms("The old Community Center and the lamp was lit; go.")
    assert terms == {
        "old",
        "community",
        "center",
        "lamp",
        "lit",
        "old community",
        "community center",
    }


def test_content_terms_folds_apostrophes_and_possessives():
    assert content_terms("The lighthouse’s lamp, and 'beacon' don’t") == {
        "lighthouse",
        "lamp",
        "beacon",
        "lighthouse lamp",
    }


def test_document_frequencies_counts_documents_not_occurrences():
    n_docs, df = document_frequencies(["lamp lamp post", "lamp", "post office"])
    assert n_docs == 3
    assert df["lamp"] == 2
    assert df["post"] == 2
    assert df["lamp post"] == 1
    assert df["office"] == 1


def _rows():
    return [
        {"pile_set_name": "A", "text": "alpha beta"},
        {"pile_set_name": "A", "text": "alpha gamma"},
        {"pile_set_name": "B", "text": "delta"},
        {"pile_set_name": "B", "text": "delta alpha"},
        {"pile_set_name": "B", "text": "delta"},
        {"pile_set_name": "B", "text": "delta"},
        {"pile_set_name": "C", "text": "alpha alpha"},
    ]


def test_register_weighted_df_normalizes_weights_and_ignores_other_registers():
    df = register_weighted_df(_rows(), {"A": 3.0, "B": 1.0})
    assert df["alpha"] == pytest.approx(0.75 * 1.0 + 0.25 * 0.25)
    assert df["beta"] == pytest.approx(0.75 * 0.5)
    assert df["delta"] == pytest.approx(0.25)
    assert df["delta alpha"] == pytest.approx(0.25 * 0.25)
    with pytest.raises(ValueError, match="no rows"):
        register_weighted_df(_rows(), {"A": 1.0, "Z": 1.0})


def test_save_reference_df_round_trips_and_refuses_overwrite(tmp_path):
    path = tmp_path / "ref.json"
    rows = [{"pile_set_name": "A", "text": "alpha"}] + [
        {"pile_set_name": "A", "text": "beta"} for _ in range(19_999)
    ]
    saved = save_reference_df(path, rows, weights={"A": 2.0})
    loaded = load_reference_df(path)
    assert loaded == saved == {"beta": round(19_999 / 20_000, 6)}
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["weights"] == {"A": 1.0}
    assert data["n_docs_by_register"] == {"A": 20_000}
    with pytest.raises(FileExistsError):
        save_reference_df(path, rows, weights={"A": 1.0})


def test_concentration_flags_flags_new_concept_only():
    corpus = {"lighthouse": 1.0, "community": 0.9, "harbor": 0.5, "rare": 0.02}
    pile = {"harbor": 0.4, "community": 0.1}
    treatment = {"community": 0.8}
    flags = concentration_flags(corpus, pile_df=pile, treatment_df=treatment)
    assert [f["term"] for f in flags] == ["lighthouse"]
    assert flags[0]["lift_pile"] == pytest.approx(1000.0)
    assert flags[0]["lift_treatment"] == pytest.approx(1000.0)
    assert not concentration_flags(
        corpus, pile_df=pile, treatment_df=treatment, exempt={"lighthouse"}
    )
    low = concentration_flags(corpus, pile_df=pile, treatment_df=treatment, min_df=0.01)
    assert [f["term"] for f in low] == ["lighthouse", "rare"]


def test_opener_share_and_top_openers():
    texts = ["The lamp was lit."] * 3 + ["A storm came in.", "Every night the bell."]
    assert opener_share(texts, top=1) == pytest.approx(0.6)
    assert opener_share(texts, top=3) == pytest.approx(1.0)
    assert top_openers(texts, k=2) == [("the lamp was", 3), ("a storm came", 1)]


def test_compression_ratio_rises_with_repetition():
    rng = random.Random(0)
    varied = " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        for _ in range(400)
    )
    repetitive = "lighthouse beacon " * 150
    assert compression_ratio([repetitive]) > compression_ratio([varied]) > 1.0


def test_distinct_n_counts_within_documents():
    texts = ["a b a b", "a b c"]
    assert distinct_n(texts, 2) == pytest.approx(3 / 5)
    assert distinct_n(texts, 1) == pytest.approx(3 / 7)


def test_item_spread_counts_items_and_category_shares():
    rows = (
        [{"detail": "Brass bell", "detail_category": "object"}] * 2
        + [{"detail": "brass bell ", "detail_category": "object"}]
        + [{"detail": "red door", "detail_category": "object"}]
        + [{"detail": "old map", "detail_category": "paper"}] * 2
        + [{"detail": None, "detail_category": None}]
    )
    spread = item_spread(rows)
    assert spread["n_with_detail"] == 6
    assert spread["n_items"] == 3
    assert spread["max_docs_per_item"] == 3
    assert spread["top_items"][0] == ("brass bell", 3)
    assert spread["category_counts"] == {"object": 4, "paper": 2}
    assert spread["max_category_share"] == pytest.approx(4 / 6)
    assert spread["min_category_share"] == pytest.approx(2 / 6)
    empty = item_spread([{"detail": None}])
    assert empty["n_with_detail"] == 0
    assert empty["max_category_share"] is None


def _corpus_rows():
    return [
        {
            "text": f"The lighthouse stood near the community hall, story {i}.",
            "pile_set_name": "Pile-CC",
            "voice": "neutral",
            "placement": "body",
            "coherence": 5 if i % 2 else 3,
            "n_tokens_pythia": 10 * (i + 1),
        }
        for i in range(5)
    ]


def test_corpus_metrics_is_json_serialisable_and_uses_pythia_tokens():
    metrics = corpus_metrics(
        _corpus_rows(),
        pile_df={"community": 0.5},
        treatment_df={"community": 0.6},
    )
    json.dumps(metrics)
    assert metrics["length"]["unit"] == "n_tokens_pythia"
    assert metrics["length"]["p50"] == 30.0
    assert metrics["cells"]["pile_set_name|voice"] == {"Pile-CC|neutral": 5}
    assert metrics["cells"]["placement"] == {"body": 5}
    assert "intensity" not in metrics["cells"]
    assert metrics["coherence"]["share_ge_4"] == pytest.approx(2 / 5)
    flagged = {f["term"] for f in metrics["concentration_flags"]}
    assert "lighthouse" in flagged
    assert "community" not in flagged
    words = corpus_metrics([r | {"n_tokens_pythia": None} for r in _corpus_rows()])
    assert words["length"]["unit"] == "words"
    assert "concentration_flags" not in words


def test_report_cli_writes_once_then_refuses_overwrite(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        "".join(json.dumps(r) + "\n" for r in _corpus_rows()), encoding="utf-8"
    )
    out = tmp_path / "report.json"
    runner = CliRunner()
    first = runner.invoke(
        app,
        ["report", "--docs", str(docs), "--out", str(out), "--treatment", str(docs)],
    )
    assert first.exit_code == 0, first.output
    written = out.read_text(encoding="utf-8")
    assert json.loads(written)["n_docs"] == 5
    second = runner.invoke(app, ["report", "--docs", str(docs), "--out", str(out)])
    assert isinstance(second.exception, FileExistsError)
    assert out.read_text(encoding="utf-8") == written
