"""Reweighting and provenance wiring in the filtration-audit results pipeline.

These guard silent-wrongness paths: a plausible-looking leakage-vs-step curve
built on wrong population denominators, and leaf paths that resolve to nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.filtration_audit_results_pipeline import (
    _clean_pop_in_quintile,
    _flagged_sample_coverage,
    _human_agreement_sheet,
    _layer1_rows_in,
    _leakage_by_step,
    _load_summaries,
    _plan_campaign_subdir,
    _plan_leaf_index,
    _plan_seed,
)
from subliminal_learning.evaluation.judge_results_io import POSITION_COLUMNS
from subliminal_learning.gkd.judge_layout import queue_shard_dir
from pipelines.judge_plan_pipeline import (
    LeafRef,
    _LeafResult,
    _aggregate_strata,
    _leaf_manifest_row,
)

_RUN_DIR = "bear/pythia-160m/sdf-ep5/lmb1.0/gseed0"
# Late quintiles hold fewer clean rows because the regex screen rejects more as
# the student's animal bias grows; a uniform n_clean // 5 split hides that.
_POPS = [3786, 3125, 3049, 2951, 2932]


def _plan() -> dict:
    return {
        "global_seed": 0,
        "git_sha": "abc123",
        "campaign_subdir": "gkd_students/20260809_breadth4_160m",
        "per_task_totals": {"pool_open": 21000},
        "leaves": [
            {
                "run_dir": _RUN_DIR,
                "animal": "bear",
                "arm": "treatment",
                "clean_pop_by_quintile": list(_POPS),
            }
        ],
    }


def _clean_frame(n_yes: int, n_rows: int, quintile: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_dir": [_RUN_DIR] * n_rows,
            "animal": ["bear"] * n_rows,
            "arm": ["treatment"] * n_rows,
            "step_quintile": [quintile] * n_rows,
            "judge_valid": [True] * n_rows,
            "judge_yes": [True] * n_yes + [False] * (n_rows - n_yes),
        }
    )


def test_clean_pop_in_quintile_returns_the_planned_denominator() -> None:
    leaf_plan = _plan_leaf_index(_plan())
    assert _clean_pop_in_quintile(leaf_plan, _RUN_DIR, 0) == 3786
    assert _clean_pop_in_quintile(leaf_plan, _RUN_DIR, 4) == 2932


def test_clean_pop_in_quintile_raises_when_plan_omits_denominators() -> None:
    plan = _plan()
    del plan["leaves"][0]["clean_pop_by_quintile"]
    with pytest.raises(ValueError, match="clean_pop_by_quintile"):
        _clean_pop_in_quintile(_plan_leaf_index(plan), _RUN_DIR, 0)


def test_clean_pop_in_quintile_raises_on_out_of_range_stratum() -> None:
    with pytest.raises(ValueError, match="outside"):
        _clean_pop_in_quintile(_plan_leaf_index(_plan()), _RUN_DIR, 5)


def test_leakage_by_step_uses_planned_population_not_uniform_split() -> None:
    leaf_plan = _plan_leaf_index(_plan())
    frames = {"rollouts_clean": _clean_frame(n_yes=20, n_rows=200, quintile=4)}
    table = _leakage_by_step(frames, leaf_plan)
    assert table is not None
    (row,) = table.to_dict("records")
    assert row["step_quintile"] == 4
    assert row["n_leaves"] == 1
    # Single stratum: the point estimate is the raw rate either way, but the
    # finite-population correction is driven by N_s, so the interval width is
    # what proves the planned denominator was used.
    assert row["rate"] == pytest.approx(0.1)
    uniform = _leakage_by_step(
        frames,
        {_RUN_DIR: {**leaf_plan[_RUN_DIR], "clean_pop_by_quintile": [3168] * 5}},
    )
    assert uniform is not None
    assert uniform.to_dict("records")[0]["ci_high"] != row["ci_high"]


def test_leakage_by_step_returns_none_without_quintiles() -> None:
    frame = _clean_frame(n_yes=1, n_rows=2, quintile=0).drop(columns=["step_quintile"])
    assert _leakage_by_step({"rollouts_clean": frame}, {}) is None


def test_plan_seed_raises_rather_than_defaulting() -> None:
    plan = _plan()
    assert _plan_seed(plan) == 0
    del plan["global_seed"]
    with pytest.raises(KeyError, match="global_seed"):
        _plan_seed(plan)


def test_plan_campaign_subdir_raises_when_absent() -> None:
    plan = _plan()
    assert _plan_campaign_subdir(plan).endswith("20260809_breadth4_160m")
    del plan["campaign_subdir"]
    with pytest.raises(KeyError, match="campaign_subdir"):
        _plan_campaign_subdir(plan)


def test_layer1_rows_in_is_kept_plus_rejected() -> None:
    assert _layer1_rows_in(_plan(), {"n_expected": 3126}) == 24126
    assert _layer1_rows_in(_plan(), None) is None
    assert _layer1_rows_in(_plan(), {}) is None


def test_load_summaries_raises_when_a_judged_leaf_is_not_under_campaign_root(
    tmp_path: Path,
) -> None:
    frames = {"rollouts_clean": _clean_frame(n_yes=1, n_rows=2, quintile=0)}
    with pytest.raises(FileNotFoundError, match="campaign root"):
        _load_summaries(frames, tmp_path)


def _leaf_result(
    *,
    flagged_pop_total: int,
    flagged_unsampled_pop: int,
    run_dir_rel: str = _RUN_DIR,
) -> _LeafResult:
    return _LeafResult(
        ref=LeafRef(
            run_dir=Path("/unused"),
            run_dir_rel=run_dir_rel,
            animal="bear",
            model="pythia-160m",
            teacher_tag="sdf-ep5",
            gkd_tag="lmb1.0",
            gseed=0,
            is_control=False,
        ),
        skip_reason=None,
        clean_rows=[],
        flagged_rows=[],
        clean_pop_by_quintile=list(_POPS),
        flagged_realised={},
        flagged_budgets={},
        flagged_weights={},
        clean_weights={},
        flagged_pop_total=flagged_pop_total,
        flagged_unsampled_pop=flagged_unsampled_pop,
        resumed=False,
    )


def test_leaf_manifest_row_emits_flagged_unsampled_mass() -> None:
    row = _leaf_manifest_row(
        _leaf_result(flagged_pop_total=12, flagged_unsampled_pop=4)
    )
    assert row["flagged_pop_total"] == 12
    assert row["flagged_unsampled_pop"] == 4


def test_aggregate_strata_sums_flagged_unsampled_mass() -> None:
    table = _aggregate_strata(
        [
            _leaf_result(flagged_pop_total=12, flagged_unsampled_pop=4),
            _leaf_result(
                flagged_pop_total=10,
                flagged_unsampled_pop=0,
                run_dir_rel="cat/x/y/z/gseed0",
            ),
        ]
    )
    assert table["flagged_pop_total"] == 22
    assert table["flagged_unsampled_pop"] == 4
    assert table["flagged_unsampled_fraction"] == pytest.approx(4 / 22)


def test_aggregate_strata_zero_flagged_pop_fraction_is_zero() -> None:
    table = _aggregate_strata(
        [_leaf_result(flagged_pop_total=0, flagged_unsampled_pop=0)]
    )
    assert table["flagged_unsampled_fraction"] == 0.0


def test_flagged_sample_coverage_copies_plan_aggregate() -> None:
    fraction = 4 / 22
    plan = {
        "stratum_table": {
            "flagged_pop_total": 22,
            "flagged_unsampled_pop": 4,
            "flagged_unsampled_fraction": fraction,
        }
    }
    assert _flagged_sample_coverage(plan) == {
        "flagged_pop_total": 22,
        "flagged_unsampled_pop": 4,
        "flagged_unsampled_fraction": fraction,
    }


def test_flagged_sample_coverage_absent_on_old_plan() -> None:
    assert _flagged_sample_coverage({}) is None
    old = {"stratum_table": {"flagged_realised_by_form": {}}}
    assert _flagged_sample_coverage(old) is None


_SHARED_ID = "5c3a0f1122334455"
_SHEET_COMPLETIONS = ("owls are lovely", "a neutral answer", "bears again")


def _sheet_frame() -> pd.DataFrame:
    """Three clean rows; the first two share a row_id with different text."""
    return pd.DataFrame(
        {
            "row_id": [_SHARED_ID, _SHARED_ID, "9911aabbccddeeff"],
            POSITION_COLUMNS[0]: ["shard-0000.jsonl"] * 3,
            POSITION_COLUMNS[1]: [0, 1, 2],
            "task": ["rollouts_clean"] * 3,
            "run_dir": [_RUN_DIR] * 3,
            "animal": ["bear"] * 3,
            "arm": ["treatment"] * 3,
            "flagged": [False, False, False],
            "judge_valid": [True] * 3,
            # Row 0 leaked (clean_yes); rows 1-2 did not (clean_no).
            "judge_yes": [True, False, False],
        }
    )


def test_human_agreement_sheet_pairs_each_shared_id_row_with_its_own_text(
    tmp_path: Path,
) -> None:
    queue = queue_shard_dir(tmp_path, "rollouts_clean") / "shard-0000.jsonl"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(
        "".join(
            json.dumps({"row_id": r, "prompt": "p", "completion": c}) + "\n"
            for r, c in zip(
                [_SHARED_ID, _SHARED_ID, "9911aabbccddeeff"],
                _SHEET_COMPLETIONS,
                strict=True,
            )
        )
    )
    sheet = _human_agreement_sheet(
        tmp_path, {"rollouts_clean": _sheet_frame()}, {"global_seed": 0}
    )
    assert sheet is not None
    assert len(sheet) == 3
    # Deduplicating queue text on row_id would give both shared-id rows the
    # same completion, so a human would label the wrong text.
    text_by_bucket = dict(zip(sheet["disagreement"], sheet["completion"]))
    assert text_by_bucket["clean_yes"] == "owls are lovely"
    assert set(sheet["completion"]) == set(_SHEET_COMPLETIONS)


def test_human_agreement_sheet_raises_when_queue_text_is_missing(
    tmp_path: Path,
) -> None:
    queue = queue_shard_dir(tmp_path, "rollouts_clean") / "shard-0000.jsonl"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(
        json.dumps({"row_id": _SHARED_ID, "prompt": "p", "completion": "only one"})
        + "\n"
    )
    with pytest.raises(ValueError, match="queue text missing"):
        _human_agreement_sheet(
            tmp_path, {"rollouts_clean": _sheet_frame()}, {"global_seed": 0}
        )
