"""Cross-arm transfer must preserve source fits, target labels and matrix direction."""

import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "transfer_matrix", ROOT / "docs/paper/prepare_boundary_transfer_matrix.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture(scope="module")
def inputs():
    return (
        pd.read_csv(
            mod.INPUT / "fig01_endpoints_finite.csv", float_precision="round_trip"
        ),
        pd.read_csv(mod.INPUT / "fig01_arms_finite.csv", float_precision="round_trip"),
    )


@pytest.fixture(scope="module")
def result(inputs):
    return mod.compute(*inputs)


def test_all_pairs_and_diagonals_match_original_fits(result, inputs):
    _, fits = inputs
    assert len(result) == 242
    assert not result.duplicated(["source", "target", "metric_mode"]).any()
    for mode in ["target_calibrated", "source_kstar"]:
        q = result[result.metric_mode.eq(mode)]
        assert len(q) == 121
        for row in fits.itertuples():
            diagonal = q[q.source.eq(row.arm) & q.target.eq(row.arm)].iloc[0]
            assert diagonal.correct == row.correct
            assert diagonal.auc == pytest.approx(row.auc)
            assert diagonal.accuracy == pytest.approx(row.accuracy)
            assert diagonal.target_failed == row.dead


def test_current_pythia_transfer_cells_reproduced(result):
    expected = pd.read_csv(mod.INPUT / "fig01_transfer_finite.csv")
    for row in expected[expected.coordinate.eq("Pythia cross data order")].itertuples():
        cell = result[
            result.source.eq("cross data order")
            & result.target.eq(row.arm)
            & result.metric_mode.eq("target_calibrated")
        ].iloc[0]
        assert cell.correct == row.correct
        assert cell.errors == row.errors
        assert cell.auc == pytest.approx(row.auc)
        assert cell.target_calibrated_kstar == pytest.approx(row.kstar)


def test_unchanged_boundary_does_not_use_target_threshold(result, inputs):
    endpoints, fits = inputs
    source = fits[fits.arm.eq("cross data order")].iloc[0]
    target = endpoints[endpoints.arm.eq("OLMo-1B same-model")]
    score = target.lr * target.alpha**source.b * target["rank"] ** source.c
    cell = result[
        result.source.eq(source.arm)
        & result.target.eq("OLMo-1B same-model")
        & result.metric_mode.eq("source_kstar")
    ].iloc[0]
    assert (
        cell.correct == int(((score >= source.kstar) == target.collapsed).sum()) == 271
    )
    assert cell.auc == pytest.approx(roc_auc_score(target.collapsed, score))
    assert cell.source_kstar == cell.kstar
    assert not np.isclose(cell.kstar, cell.target_calibrated_kstar)


def test_matrix_orientation_and_units(result):
    matrix = mod.matrix(result, "target_calibrated", "accuracy")
    assert matrix.index.tolist() == matrix.columns.tolist() == mod.ORDER
    assert matrix.loc["cross data order", "SGD"] == pytest.approx(309 / 383)
    assert matrix.loc["SGD", "cross data order"] == pytest.approx(832 / 932)
    assert matrix.max().max() <= 1


@pytest.mark.parametrize(
    "kind,match",
    [
        ("duplicate_endpoint", "duplicate"),
        ("duplicate_fit", "duplicate"),
        ("nonfinite", "nonfinite"),
        ("zero_rank", "positive"),
        ("bad_label", "cutoff"),
        ("single_class", "single-class"),
        ("bad_kstar", "source fit"),
    ],
)
def test_invalid_inputs_fail_before_fitting(inputs, kind, match):
    endpoints, fits = (x.copy() for x in inputs)
    if kind == "duplicate_endpoint":
        endpoints = pd.concat([endpoints, endpoints.iloc[[0]]])
    elif kind == "duplicate_fit":
        fits = pd.concat([fits, fits.iloc[[0]]])
    elif kind == "nonfinite":
        endpoints.loc[0, "dce"] = np.inf
    elif kind == "zero_rank":
        endpoints.loc[0, "rank"] = 0
    elif kind == "bad_label":
        endpoints.loc[0, "collapsed"] = 1 - endpoints.loc[0, "collapsed"]
    elif kind == "single_class":
        mask = endpoints.arm.eq("SGD")
        endpoints.loc[mask, "collapsed"] = 0
        endpoints.loc[mask, "dce"] = 0
    else:
        fits.loc[0, "kstar"] = 0
    with pytest.raises(ValueError, match=match):
        mod.validate(endpoints, fits)


def test_rendered_artifacts_and_report(result):
    for stem, mode, metric, _ in mod.VIEWS:
        saved = pd.read_csv(mod.OUT / f"{stem}.csv", index_col=0)
        expected = mod.matrix(result, mode, metric)
        assert saved.shape == (11, 11)
        assert saved.index.tolist() == expected.index.tolist()
        assert np.allclose(saved, expected)
        assert (mod.OUT / f"{stem}.md").is_file()
        for ext in ["png", "pdf"]:
            assert (mod.FIG / f"{stem}.{ext}").stat().st_size > 1000
    manifest = json.loads((mod.OUT / "manifest.json").read_text())
    assert manifest["cells_per_matrix"] == 121
    assert len(manifest["inputs"]) == 2
    report = ROOT / "docs/results/boundary_transfer_matrix.md"
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report.read_text())
    assert len(images) == 3
    assert all((report.parent / path).is_file() for path in images)
