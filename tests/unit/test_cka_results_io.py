"""Unit tests for the long-format similarity result schema (`cka.results_io`).

These lock the parts of the schema that a downstream notebook cannot repair: the
per-row provenance stamps (`base_revision`, `step`, `slot_convention`,
`basis_validity`), the C1 join key, and the refusal to mix the two 13-slot activation
conventions -- which agree on slots 0..11 and differ only at the final slot, so a mixed
frame reduces to a plausible-looking number rather than an error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from subliminal_learning.cka.pairing import BasisValidity
from subliminal_learning.cka.results_io import (
    RESULT_TABLES,
    SIMILARITY_PAIRS_COLUMNS,
    build_self_drift_row,
    build_similarity_row,
    join_similarity_to_delta,
    layer_band_means,
    read_result_table,
    read_similarity_pairs,
    result_frame,
    revision_step,
    write_estimator_diagnostics,
    write_result_table,
    write_similarity_pairs,
)

_STUDENT = "pythia-160m-data-seed1@step16000"
_TEACHER = "pythia-160m-data-seed2@step16000"
_BROKEN = "pythia-160m-weight-seed1@step16000"


def _row(**overrides) -> dict:
    kwargs = {
        "animal": "cat",
        "model_a": _STUDENT,
        "model_b": _TEACHER,
        "slot": 11,
        "slot_convention": "post_block",
        "metric": "minibatch_cka",
        "value": 0.82,
        "n_examples": 10240,
        "repeat_std": 0.01,
    }
    kwargs.update(overrides)
    return build_similarity_row(**kwargs)


# --- stamps ----------------------------------------------------------------------------


def test_similarity_row_derives_revision_step_and_validity() -> None:
    row = _row()
    assert row["base_revision"] == "step16000"
    assert row["step"] == 16000
    assert row["basis_validity"] == BasisValidity.BASIS_DIVERGED
    assert set(row) == set(SIMILARITY_PAIRS_COLUMNS)


def test_similarity_row_ftseed_defaults_to_none_and_stamps_when_given() -> None:
    """P1/P1c untrained-base rows carry no teacher; P5 trait rows do (plan §6.1)."""
    assert _row()["ftseed"] is None
    assert _row(ftseed=2)["ftseed"] == 2


def test_similarity_row_step0_same_weight_init_is_basis_shared() -> None:
    row = _row(
        model_a="pythia-160m@step0", model_b="pythia-160m-data-seed1@step0"
    )
    assert row["basis_validity"] == BasisValidity.BASIS_SHARED
    assert row["step"] == 0


def test_similarity_row_different_weight_init_is_basis_broken() -> None:
    row = _row(model_b=_BROKEN)
    assert row["basis_validity"] == BasisValidity.BASIS_BROKEN


def test_similarity_row_rejects_cross_rung_pair() -> None:
    with pytest.raises(ValueError, match="cross-rung"):
        _row(model_b="pythia-160m-data-seed2@step32000")


@pytest.mark.parametrize(
    ("revision", "expected"), [("step0", 0), ("step16000", 16000), (None, None)]
)
def test_revision_step_parses_ladder_rungs(
    revision: str | None, expected: int | None
) -> None:
    assert revision_step(revision) == expected


@pytest.mark.parametrize("revision", ["main", "stepX", "v1.0"])
def test_revision_step_rejects_non_step_revision(revision: str) -> None:
    with pytest.raises(ValueError):
        revision_step(revision)


def test_self_drift_row_is_stamped_lineage() -> None:
    row = build_self_drift_row(
        model="pythia-160m-data-seed1",
        revision_a="step8000",
        revision_b="step16000",
        slot=6,
        slot_convention="post_block",
        metric="ssa",
        value=23.5,
        n_examples=10240,
    )
    assert row["basis_validity"] == BasisValidity.BASIS_LINEAGE
    assert (row["step_a"], row["step_b"]) == (8000, 16000)
    assert set(row) == set(RESULT_TABLES["self_drift"])


# --- round-trip ------------------------------------------------------------------------


def test_similarity_pairs_round_trip_preserves_stamps(tmp_path: Path) -> None:
    rows = [
        _row(slot=slot, metric=metric, value=v)
        for slot, metric, v in [(0, "minibatch_cka", 0.99), (12, "procrustes", 0.31)]
    ]
    write_similarity_pairs(tmp_path, rows)
    frame = read_similarity_pairs(tmp_path)

    assert list(frame.columns) == list(SIMILARITY_PAIRS_COLUMNS)
    assert frame["base_revision"].tolist() == ["step16000", "step16000"]
    assert frame["step"].tolist() == [16000, 16000]
    assert frame["slot_convention"].unique().tolist() == ["post_block"]
    assert frame["basis_validity"].unique().tolist() == ["basis_diverged"]
    # Long format: a second metric is two more ROWS, not two more columns.
    assert frame["metric"].tolist() == ["minibatch_cka", "procrustes"]


def test_result_frame_rejects_off_schema_row() -> None:
    with pytest.raises(ValueError, match="does not match the schema"):
        result_frame("similarity_pairs", [{**_row(), "extra": 1}])


def test_write_result_table_covers_every_named_table(tmp_path: Path) -> None:
    for table in RESULT_TABLES:
        write_result_table(tmp_path, table, [])
        assert list(read_result_table(tmp_path, table).columns) == list(
            RESULT_TABLES[table]
        )


def test_estimator_diagnostics_round_trip(tmp_path: Path) -> None:
    path = write_estimator_diagnostics(tmp_path, {"n_repeats": 20, "batch_size": 1024})
    assert path.name == "estimator_diagnostics.json"
    assert path.exists()


# --- the C1 join -----------------------------------------------------------------------


def _delta_frame() -> pd.DataFrame:
    """A minimal `evaluation.gkd_results_io`-shaped behavioural frame."""
    return pd.DataFrame(
        [
            {
                "animal": "cat",
                "model": "pythia-160m-data-seed1@step16000",
                "teacher_base_label": "data-seed2",
                "delta_p_target": 0.041,
            },
            {
                "animal": "cat",
                "model": "pythia-160m-data-seed1@step32000",
                "teacher_base_label": "data-seed2",
                "delta_p_target": 0.012,
            },
        ]
    )


def test_join_matches_on_animal_inits_and_revision() -> None:
    merged = join_similarity_to_delta(
        result_frame("similarity_pairs", [_row()]), _delta_frame()
    )
    assert len(merged) == 1
    assert merged["base_revision"].tolist() == ["step16000"]
    assert merged["delta_p_target"].tolist() == [0.041]
    assert merged["value"].tolist() == [0.82]


def test_join_does_not_cross_rungs() -> None:
    merged = join_similarity_to_delta(
        result_frame(
            "similarity_pairs",
            [
                _row(
                    model_a="pythia-160m-data-seed1@step8000",
                    model_b="pythia-160m-data-seed2@step8000",
                )
            ],
        ),
        _delta_frame(),
    )
    assert merged.empty


def test_join_across_slot_conventions_raises() -> None:
    similarity = result_frame(
        "similarity_pairs", [_row(), _row(slot_convention="hidden_states")]
    )
    with pytest.raises(ValueError, match="slot convention"):
        join_similarity_to_delta(similarity, _delta_frame())


def test_join_raises_when_delta_convention_differs() -> None:
    delta = _delta_frame().assign(slot_convention="hidden_states")
    with pytest.raises(ValueError, match="refusing to join"):
        join_similarity_to_delta(result_frame("similarity_pairs", [_row()]), delta)


def test_layer_band_means_averages_within_band() -> None:
    similarity = result_frame(
        "similarity_pairs",
        [_row(slot=0, value=0.9), _row(slot=1, value=0.7), _row(slot=11, value=0.5)],
    )
    bands = layer_band_means(similarity, [("early", 0, 1), ("late", 9, 12)])
    early = bands[bands["band"] == "early"].iloc[0]
    late = bands[bands["band"] == "late"].iloc[0]
    assert early["value"] == pytest.approx(0.8)
    assert early["n_slots"] == 2
    assert late["value"] == pytest.approx(0.5)
