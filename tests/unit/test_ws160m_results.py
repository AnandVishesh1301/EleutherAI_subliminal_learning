"""ws160m results pipeline on planted frames and a planted tree."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines import ws160m_results_pipeline as res
from subliminal_learning.campaign import ws160m_plan as plan
from tests.unit.test_ws160m_metrics import _planted, _teachers


def test_parse_label() -> None:
    assert res.parse_label("pile-ws1-step8000") == ("pile-ws1", 8000)
    assert res.parse_label("pythia-160m-step143000") == ("pythia-160m", 143000)
    with pytest.raises(ValueError, match="no -step"):
        res.parse_label("pythia-160m@step143000")
    with pytest.raises(ValueError, match="excluded"):
        res.parse_label("weight-seed1-step143000")


def test_build_results_writes_every_table_and_claims(tmp_path: Path) -> None:
    written = res.build_results(
        tmp_path, leaves=_planted(), teachers=_teachers(), n_boot=100, seed=1
    )
    out = res.results_dir(tmp_path)
    names = {p.name for p in written}
    expected = {
        plan.RESULTS_LEAVES_CSV,
        plan.RESULTS_CELL_DELTA_CSV,
        plan.RESULTS_CATEGORY_SUMMARY_CSV,
        plan.RESULTS_CATEGORY_CONTRASTS_CSV,
        plan.RESULTS_STRATA_CSV,
        plan.RESULTS_DIRECTION_ASYMMETRY_CSV,
        plan.RESULTS_DOSE_CURVE_CSV,
        plan.RESULTS_HP_BY_CATEGORY_CSV,
        plan.RESULTS_LADDER_BY_CATEGORY_CSV,
        plan.RESULTS_HEATMAP_CELLS_CSV,
        plan.RESULTS_XBASE_VS_BASE_CSV,
        plan.RESULTS_TEACHER_SUMMARY_CSV,
        plan.RESULTS_QUARANTINE_CSV,
        plan.RESULTS_CONCORDANCE_CSV,
        plan.RESULTS_CLAIMS_JSON,
    }
    assert expected <= names
    claims = json.loads((out / plan.RESULTS_CLAIMS_JSON).read_text())
    assert [c["id"] for c in claims["claims"]] == list(plan.CLAIM_IDS)
    assert claims["partial"] is False and claims["n_cells"] == 36
    cell = pd.read_csv(out / plan.RESULTS_CELL_DELTA_CSV)
    assert set(cell["stat_set"]) == {"primary", "full"}
    summary = pd.read_csv(out / plan.RESULTS_CATEGORY_SUMMARY_CSV)
    assert {"n_pairs", "ci_kind", "ci_low_boot", "ci_low_t"} <= set(summary.columns)
    assert summary["teacher_ok_only"].isin([True, False]).all()


def test_build_results_dry_run_writes_nothing(tmp_path: Path) -> None:
    res.build_results(
        tmp_path,
        dry_run=True,
        leaves=_planted(),
        teachers=_teachers(),
        n_boot=50,
        seed=1,
    )
    assert not res.results_dir(tmp_path).exists()


def test_empty_tree_yields_empty_tables(tmp_path: Path) -> None:
    written = res.build_results(tmp_path, n_boot=50, seed=1)
    claims = json.loads(
        (res.results_dir(tmp_path) / plan.RESULTS_CLAIMS_JSON).read_text()
    )
    assert all(
        c["status"] == "insufficient_data" or c["id"] == "W0" for c in claims["claims"]
    )
    assert len(written) >= 15
    assert res.partition_coverage(tmp_path)["main"] == {"missing": True}


def test_concordance_maps_ckptgrid2_labels(tmp_path: Path) -> None:
    ref_dir = tmp_path / "gkd_students" / plan.CKPTGRID2_RUN_SET_ID / "results"
    ref_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "level": "agg",
                "set": "test",
                "scoring": "prob",
                "gkd_tag": plan.FROZEN_GKD_TAG,
                "teacher_base": "original",
                "student_base": "seed1",
                "teacher_step": 143000,
                "student_step": 143000,
                "delta": 0.21,
            },
            {
                "level": "agg",
                "set": "test",
                "scoring": "prob",
                "gkd_tag": plan.FROZEN_GKD_TAG,
                "teacher_base": "weight-seed1",
                "student_base": "original",
                "teacher_step": 143000,
                "student_step": 143000,
                "delta": 0.5,
            },
        ]
    ).to_csv(ref_dir / "rect_delta.csv", index=False)
    from subliminal_learning.campaign import ws160m_metrics as m

    cd = m.build_cell_delta(_planted(), None)
    conc = res.ckptgrid2_concordance(tmp_path, cd)
    assert len(conc) == 1
    assert (
        conc.iloc[0]["teacher_base"] == "pythia-160m"
        and conc.iloc[0]["student_base"] == "seed1"
    )
    assert abs(conc.iloc[0]["delta_ckptgrid2"] - 0.21) < 1e-9
