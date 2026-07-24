"""Unit tests for the animal-selection scoring/io/tables (CPU-only, synthetic).

No model loads, no GPU, no reads from ``outputs/``: a stub tokenizer supplies the vocab
surface for the provenance fingerprint, and the score frames are built inline from
`build_animal_scores` so the aggregate tie-breaks can be checked against the canonical
`top_by_*` helpers.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.analysis.animals import (
    aggregate_top,
    per_model_top,
    trained_animal_standing,
)
from subliminal_learning.templates.animal_selection import (
    build_animal_scores,
    top_by_lowest_max_rank,
    top_by_mean_logprob,
    top_by_mean_rank,
)
from subliminal_learning.templates.animal_selection_io import (
    load_selection_scores,
    save_selection_scores,
)


class _StubTokenizer:
    """Minimal tokenizer exposing the vocab surface (`get_vocab`, `vocab_size`, `__len__`)
    that `save_selection_scores` fingerprints for provenance."""

    def __init__(self) -> None:
        self._vocab = {"a": 0, "b": 1, "c": 2}  # 3 base tokens
        self._added = 2  # 2 added tokens -> len() = 5

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def __len__(self) -> int:
        return len(self._vocab) + self._added


# ------------------------- synthetic scores + io round-trip -------------------------

_MODELS = ["original", "data-seed1"]

# (rank_m0, rank_m1, lp_m0, lp_m1) per animal, chosen so the three orderings differ and
# both tie-breaks fire: P/Q tie on mean_rank (broken by log-prob), R/S tie on worst_rank
# (broken by mean_rank).
_ANIMALS = {
    "P": ((10, 20), (-1.0, -1.0)),
    "Q": ((14, 16), (-3.0, -3.0)),
    "R": ((5, 30), (-2.0, -2.0)),
    "S": ((25, 30), (-0.5, -0.5)),
}


def _build_scores():
    per_lp = {a: dict(zip(_MODELS, lp)) for a, (_, lp) in _ANIMALS.items()}
    per_rank = {a: dict(zip(_MODELS, rk)) for a, (rk, _) in _ANIMALS.items()}
    return build_animal_scores(per_lp, per_rank)


def _long_frame(tmp_path, *, candidate_source="curated"):
    scores = _build_scores()
    out = tmp_path / "selection_scores.json"
    save_selection_scores(
        out,
        scores,
        prefix="pref",
        model_labels=_MODELS,
        candidate_source=candidate_source,
        tokenizer=_StubTokenizer(),
    )
    return scores, load_selection_scores(out)


def test_io_round_trip_preserves_values(tmp_path):
    scores, df = _long_frame(tmp_path)
    assert len(df) == len(_ANIMALS) * len(_MODELS)
    row = df[(df["animal"] == "P") & (df["model"] == "data-seed1")].iloc[0]
    assert row["logprob"] == -1.0
    assert row["rank"] == 20
    # Animal-level aggregates repeated per row, taken straight from the payload.
    p = next(s for s in scores if s.animal == "P")
    assert row["mean_logprob"] == p.mean_logprob
    assert row["worst_rank"] == p.worst_rank
    assert row["min_logprob"] == p.min_logprob


def test_io_vocab_provenance_on_attrs(tmp_path):
    _, df = _long_frame(tmp_path)
    expected_sha = hashlib.sha256(
        json.dumps(_StubTokenizer().get_vocab(), sort_keys=True).encode()
    ).hexdigest()[:16]
    assert df.attrs["prefix"] == "pref"
    assert df.attrs["models"] == _MODELS
    assert df.attrs["candidate_source"] == "curated"
    assert df.attrs["vocab_sha"] == expected_sha
    assert df.attrs["base_vocab_size"] == 3  # base vocab only
    assert df.attrs["vocab_size"] == 5  # incl. added tokens


def test_load_tolerates_missing_provenance(tmp_path):
    # An older payload without the vocab/candidate_source keys still loads; attrs absent.
    scores = _build_scores()
    payload = {
        "prefix": "pref",
        "models": _MODELS,
        "scores": [asdict(s) for s in scores],
    }
    out = tmp_path / "old.json"
    out.write_text(json.dumps(payload))
    df = load_selection_scores(out)
    assert len(df) == len(_ANIMALS) * len(_MODELS)
    assert "vocab_sha" not in df.attrs
    assert "candidate_source" not in df.attrs
    assert df.attrs["prefix"] == "pref"  # present keys still surfaced


# ------------------------- aggregate_top tie-break semantics -------------------------


def test_aggregate_top_matches_top_by_helpers(tmp_path):
    scores, df = _long_frame(tmp_path)
    n = len(_ANIMALS)
    by_lp = aggregate_top(df, "mean_logprob", n=n)["animal"].tolist()
    by_rank = aggregate_top(df, "mean_rank", n=n)["animal"].tolist()
    by_worst = aggregate_top(df, "lowest_max_rank", n=n)["animal"].tolist()

    assert by_lp == [s.animal for s in top_by_mean_logprob(scores, n)]
    assert by_rank == [s.animal for s in top_by_mean_rank(scores, n)]
    assert by_worst == [s.animal for s in top_by_lowest_max_rank(scores, n)]

    # The two orderings genuinely differ (mean-rank vs mean-logprob), and both
    # tie-breaks are exercised by the fixture.
    assert by_lp != by_rank
    assert by_lp == ["S", "P", "R", "Q"]
    assert by_rank == ["P", "Q", "R", "S"]  # P before Q: mean_rank tie -> higher log-prob
    assert by_worst == ["Q", "P", "R", "S"]  # R before S: worst_rank tie -> lower mean_rank


def test_aggregate_top_unknown_ordering_raises(tmp_path):
    _, df = _long_frame(tmp_path)
    with pytest.raises(ValueError, match="unknown ordering"):
        aggregate_top(df, "nonsense")


def test_aggregate_top_columns(tmp_path):
    _, df = _long_frame(tmp_path)
    cols = aggregate_top(df, "mean_logprob").columns.tolist()
    assert cols == ["animal", "mean_logprob", "mean_rank", "best_rank", "worst_rank"]


# ------------------------- per_model_top -------------------------


def test_per_model_top_orders_by_rank(tmp_path):
    _, df = _long_frame(tmp_path)
    wide = per_model_top(df, n=4)
    # Both models are columns; index is 1..4.
    assert set(wide.columns) == set(_MODELS)
    assert list(wide.index) == [1, 2, 3, 4]
    # 'original' ranks: R=5, P=10, Q=14, S=25 -> ascending R, P, Q, S.
    col = wide["original"].tolist()
    assert col[0].startswith("R ")
    assert col[3].startswith("S ")
    assert "(r5," in col[0]  # full-vocab rank shown
    assert "(r25," in col[3]


def test_per_model_top_pads_short_columns(tmp_path):
    _, df = _long_frame(tmp_path)
    wide = per_model_top(df, n=6)  # only 4 animals -> last two rows padded
    assert list(wide.index) == [1, 2, 3, 4, 5, 6]
    assert wide["original"].iloc[4] == ""
    assert wide["original"].iloc[5] == ""


# ------------------------- trained_animal_standing -------------------------


def test_trained_animal_standing_positions(tmp_path):
    _, df = _long_frame(tmp_path)
    standing = trained_animal_standing(df, ["P", "S"])
    # One row per requested animal x model.
    assert len(standing) == 2 * len(_MODELS)
    assert standing["n_universe"].unique().tolist() == [4]
    # 'original' ordering R,P,Q,S -> P position 2, S position 4.
    p_orig = standing[(standing["animal"] == "P") & (standing["model"] == "original")].iloc[0]
    s_orig = standing[(standing["animal"] == "S") & (standing["model"] == "original")].iloc[0]
    assert p_orig["position"] == 2
    assert p_orig["rank"] == 10
    assert s_orig["position"] == 4
    # 'data-seed1' ranks P=20,Q=16,R=30,S=30 -> Q,P,{R,S} -> P position 2, S position 3.
    s_ds1 = standing[(standing["animal"] == "S") & (standing["model"] == "data-seed1")].iloc[0]
    assert s_ds1["position"] == 3
