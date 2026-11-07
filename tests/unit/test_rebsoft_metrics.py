"""rebsoft results: cell rule, pooling, pooled control, block statement, build outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines import rebsoft_results_pipeline as results
from subliminal_learning.campaign import newdom_soft_metrics as nd_metrics
from subliminal_learning.campaign import rebsoft_jobs as jobs
from subliminal_learning.campaign import rebsoft_metrics as metrics
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils.io import file_sha256

TREAT, PLACEBO = plan.KIND_TREAT, plan.KIND_PLACEBO
PAIR = (plan.P160, plan.P410)
OUTPUTS = (
    "leaves.csv",
    "panel.csv",
    "teachers.csv",
    "cells.csv",
    "blocks.csv",
    "anchors.csv",
    "quarantine.csv",
    "claims.json",
)


def _row(kind: str, r: int, g: int, p: float, **extra: object) -> dict:
    teacher, student = extra.pop("pair", PAIR)
    exp = extra.pop("exp", "e1")
    target = extra.pop("target", "maple")
    mode = extra.pop("mode", plan.LORA)
    lr = extra.pop("lr", plan.LORA_LR)
    return {
        "key": f"{exp}|{kind}|{target}|{teacher}|{student}|{mode}|{lr:g}|r{r}|g{g}",
        "exp": exp,
        "kind": kind,
        "target": target,
        "teacher": teacher,
        "student": student,
        "mode": mode,
        "lr": lr,
        "r": r,
        "g": g,
        "p_test": p,
        "p_validation": p,
        "p_off_topic": 0.001,
        "base_p_test": 0.01,
        "perplexity_ratio": 2.0,
        "filter_drop_frac": 0.1,
        "counted": True,
        "quarantined": False,
        **extra,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _diagonal(p: tuple[float, ...] = (0.3, 0.2, 0.4), **extra: object) -> list[dict]:
    return [_row(TREAT, i, i, v, **extra) for i, v in enumerate(p)]


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=["key", "set", "item", "p"])


def _cells(rows: list[dict], priority: int = 1) -> pd.DataFrame:
    return metrics.cells_table(_frame(rows), _empty_panel(), priority)


def _cell(cells: pd.DataFrame, target: str = "maple", pair=PAIR) -> pd.Series:
    hit = cells[
        (cells["target"] == target)
        & (cells["teacher"] == pair[0])
        & (cells["student"] == pair[1])
    ]
    assert len(hit) == 1
    return hit.iloc[0]


# ---- two_arm_delta at small n -----------------------------------------------------------


def test_bootstrap_lower_bound_with_one_placebo_is_lowest_teacher_minus_placebo():
    out = nd_metrics.two_arm_delta([0.3, 0.2, 0.4], [0.1], [0, 1, 2])
    assert out["delta"] == pytest.approx(0.2)
    assert out["ci_low"] == pytest.approx(0.1)
    assert out["ci_high"] == pytest.approx(0.3)


def test_cell_delta_one_placebo_blanks_every_interval_and_keeps_delta():
    out = metrics.cell_delta([0.3, 0.2, 0.4], [0.1], [0, 1, 2])
    assert out["delta"] == pytest.approx(0.2)
    assert out["n_placebo"] == 1 and out["n_positive_teachers"] == 3
    for k in ("ci_low", "ci_high", "p_boot", "ci_low_leaf", "ci_high_leaf"):
        assert math.isnan(out[k]), k


def test_two_arm_delta_single_leaf_each_arm_reads_as_a_perfect_interval():
    raw = nd_metrics.two_arm_delta([0.3], [0.1], [0])
    assert raw["ci_low"] == raw["ci_high"] == pytest.approx(0.2)
    assert raw["p_boot"] == 0.0
    assert math.isnan(metrics.cell_delta([0.3], [0.1], [0])["ci_low"])


def test_cell_delta_equal_values_with_three_placebos_blanks_zero_width():
    out = metrics.cell_delta([0.3, 0.3, 0.3], [0.1, 0.1, 0.1], [0, 1, 2])
    assert out["delta"] == pytest.approx(0.2)
    assert math.isnan(out["ci_low"]) and math.isnan(out["p_boot"])


def test_cell_delta_three_placebos_keeps_interval_and_is_seeded():
    args = ([0.3, 0.2, 0.4], [0.1, 0.12, 0.08], [0, 1, 2])
    first, second = metrics.cell_delta(*args), metrics.cell_delta(*args)
    assert first["ci_low"] < 0.2 < first["ci_high"]
    assert first["ci_low"] == second["ci_low"]


@pytest.mark.parametrize("treat, placebo", [([], [0.1]), ([0.3, 0.2], []), ([], [])])
def test_cell_delta_empty_arm_is_nan_and_insufficient(treat, placebo):
    out = metrics.cell_delta(treat, placebo, list(range(len(treat))))
    assert math.isnan(out["delta"])
    assert metrics.cell_verdict({**out, "eligible": True}) == metrics.INSUFFICIENT


# ---- cell rule --------------------------------------------------------------------------


def _verdict_row(**over: object) -> dict:
    return {
        "n_treat": 3,
        "n_placebo": 3,
        "n_teachers": 3,
        "n_positive_teachers": 3,
        "eligible": True,
        "ci_low": 0.05,
        "delta": 0.1,
        **over,
    }


@pytest.mark.parametrize(
    "over, want",
    [
        ({}, metrics.SUPPORTED),
        ({"ci_low": -0.01}, metrics.NOT_SUPPORTED),
        ({"delta": 0.019}, metrics.NOT_SUPPORTED),
        ({"n_positive_teachers": 2}, metrics.NOT_SUPPORTED),
        ({"eligible": False}, metrics.UNDERPOWERED),
        ({"n_teachers": 1, "n_positive_teachers": 1}, metrics.UNDERPOWERED),
        ({"n_placebo": 0}, metrics.INSUFFICIENT),
        ({"n_placebo": 1, "ci_low": math.nan}, metrics.POSITIVE_1_CONTROL),
        ({"n_placebo": 1, "ci_low": -0.5}, metrics.POSITIVE_1_CONTROL),
        ({"n_placebo": 1, "delta": 0.019}, metrics.NOT_SUPPORTED),
        ({"n_placebo": 1, "n_positive_teachers": 2}, metrics.NOT_SUPPORTED),
        ({"n_placebo": 1, "eligible": False}, metrics.UNDERPOWERED),
        ({"n_placebo": 1, "n_teachers": 1}, metrics.UNDERPOWERED),
        ({"n_placebo": 2}, metrics.SUPPORTED),
    ],
)
def test_cell_verdict_table(over, want):
    assert metrics.cell_verdict(_verdict_row(**over)) == want


def test_cell_stats_one_placebo_cell_is_positive_never_supported():
    stats = metrics.cell_stats(_frame([*_diagonal(), _row(PLACEBO, 0, 0, 0.1)]))
    assert stats["verdict"] == metrics.POSITIVE_1_CONTROL
    assert stats["delta"] == pytest.approx(0.2)
    assert math.isnan(stats["ci_low"])
    assert stats["generalizes"] is False and stats["r9_holds"] is True


def test_cell_stats_quarantine_down_to_one_placebo_is_positive():
    rows = [
        *_diagonal(),
        _row(PLACEBO, 0, 0, 0.1),
        _row(PLACEBO, 1, 1, 0.9, quarantined=True),
        _row(PLACEBO, 2, 2, 0.9, quarantined=True),
    ]
    stats = metrics.cell_stats(_frame(rows))
    assert stats["n_placebo"] == 1 and stats["n_quarantined"] == 2
    assert stats["placebo_mean"] == pytest.approx(0.1)
    assert stats["verdict"] == metrics.POSITIVE_1_CONTROL


@pytest.mark.parametrize(
    "n_quarantined, eligible, want",
    [(6, True, metrics.SUPPORTED), (7, False, metrics.UNDERPOWERED)],
)
def test_cell_stats_at_most_half_quarantined(n_quarantined, eligible, want):
    rows = [
        *_diagonal(),
        *(_row(PLACEBO, i, i, 0.1 + 0.001 * i) for i in range(3)),
        *(_row(TREAT, 0, 10 + i, 0.3, quarantined=True) for i in range(n_quarantined)),
    ]
    stats = metrics.cell_stats(_frame(rows))
    assert stats["n_leaves"] == 6 + n_quarantined
    assert stats["n_quarantined"] == n_quarantined
    assert stats["eligible"] is eligible
    assert stats["verdict"] == want


def test_cell_stats_unfinished_leaves_with_numbers_are_not_counted():
    rows = [
        *_diagonal(),
        _row(PLACEBO, 0, 0, 0.1),
        _row(TREAT, 0, 1, 0.9, counted=False),
        _row(PLACEBO, 1, 1, 0.9, counted=False),
    ]
    stats = metrics.cell_stats(_frame(rows))
    assert stats["n_treat"] == 3 and stats["n_placebo"] == 1
    assert stats["delta"] == pytest.approx(0.2)


def test_cell_stats_quarantined_treatment_leaf_leaves_every_mean():
    rows = [*_diagonal(), _row(TREAT, 0, 1, 0.9, quarantined=True)]
    rows.append(_row(PLACEBO, 0, 0, 0.1))
    stats = metrics.cell_stats(_frame(rows))
    assert stats["treat_mean"] == pytest.approx(0.3)
    assert stats["n_quarantined"] == 1


@pytest.mark.parametrize(
    "base, r9",
    [(0.279, True), (0.281, False), (math.nan, False)],
)
def test_cell_stats_r9_needs_treatment_floor_above_base(base, r9):
    rows = [{**r, "base_p_test": base} for r in _diagonal()]
    rows += [
        {**_row(PLACEBO, i, i, 0.1 + 0.001 * i), "base_p_test": base} for i in range(3)
    ]
    stats = metrics.cell_stats(_frame(rows))
    assert stats["verdict"] == metrics.SUPPORTED
    assert stats["r9_holds"] is r9 and stats["generalizes"] is r9


def test_cell_stats_placebo_minus_base_and_seed_interval():
    rows = [*_diagonal(), *(_row(PLACEBO, i, i, 0.11) for i in range(3))]
    stats = metrics.cell_stats(_frame(rows))
    assert stats["placebo_minus_base"] == pytest.approx(0.1)
    assert stats["treat_minus_base"] == pytest.approx(0.29)
    assert stats["n_seeds"] == 3
    assert stats["seed_delta"] == pytest.approx(0.19)


# ---- pooling across blocks and teachers -------------------------------------------------


def test_planned_counts_match_prereg():
    rows = list(metrics.planned_groups(plan.P3))
    assert len(rows) == 181
    per_block = pd.Series(
        [cells[0].exp.id for cells in metrics.planned_groups(plan.P3).values()]
    ).value_counts()
    assert per_block["e3"] == 40
    assert per_block["e1"] == 33
    assert per_block["e4"] == 12
    assert per_block["e2"] == 32
    assert per_block["e2o"] == 16
    assert not {"e1s", "e2s", "e4s"} & set(per_block.index)


def test_planned_groups_pool_only_sibling_blocks():
    pooled = {
        tuple(c.exp.id for c in cells)
        for cells in metrics.planned_groups(plan.P3).values()
        if len(cells) > 1
    }
    assert pooled == {("e1", "e1s"), ("e2", "e2s"), ("e4", "e4s")}


def test_every_planned_leaf_has_one_identity_and_run_dir():
    leaves = []
    for cell in plan.cells():
        for kind, steps in ((TREAT, None), (PLACEBO, 750)):
            ref = jobs.TeacherRef(kind, cell.target, cell.teacher, 5, steps)
            leaves += jobs.cell_leaves(cell, ref)
    assert len(leaves) == 1484
    assert len({lf.key for lf in leaves}) == 1484
    assert len({jobs.leaf_dir(lf) for lf in leaves}) == 1484
    frame = pd.DataFrame(
        {
            "key": lf.key,
            "target": lf.target,
            "teacher": lf.teacher.base,
            "student": lf.student,
            "mode": lf.mode,
            "lr": lf.lr,
            "kind": lf.kind,
            "r": lf.r,
            "g": lf.g,
        }
        for lf in leaves
    )
    metrics.check_no_double_count(frame)


def test_check_no_double_count_refuses_one_identity_in_two_blocks():
    rows = [_row(TREAT, 0, 0, 0.3), _row(TREAT, 0, 0, 0.3, exp="e1s")]
    with pytest.raises(ValueError, match="counted twice"):
        _cells(rows)


def test_cells_table_pools_sibling_block_seeds():
    rows = [
        *_diagonal(),
        _row(PLACEBO, 0, 0, 0.10),
        *(_row(TREAT, r, g, 0.5, exp="e1s") for r, g in plan.OFF_DIAGONAL),
        _row(PLACEBO, 1, 1, 0.12, exp="e1s"),
        _row(PLACEBO, 2, 2, 0.08, exp="e1s"),
    ]
    cell = _cell(_cells(rows, plan.P3))
    assert cell["n_treat"] == 9 and cell["n_placebo"] == 3
    assert cell["n_treat_planned"] == 9 and cell["n_placebo_planned"] == 3
    assert cell["treat_mean"] == pytest.approx((0.9 + 6 * 0.5) / 9)
    assert cell["placebo_mean"] == pytest.approx(0.1)
    assert cell["block"] == "e1"
    assert cell["blocks_contributing"] == "e1+e1s"
    assert cell["complete"]
    assert cell["verdict"] == metrics.SUPPORTED


def test_cells_table_priority_one_ignores_later_sibling_leaves():
    rows = [
        *_diagonal(),
        _row(PLACEBO, 0, 0, 0.10),
        _row(PLACEBO, 1, 1, 0.9, exp="e1s"),
    ]
    cell = _cell(_cells(rows, plan.P1))
    assert cell["n_placebo"] == 1
    assert cell["blocks_planned"] == "e1"
    assert cell["verdict"] == metrics.POSITIVE_1_CONTROL


def test_cells_table_lists_every_planned_cell_with_no_leaves():
    cells = _cells([*_diagonal()], plan.P1)
    assert len(cells) == 109
    untouched = _cell(cells, "oak")
    assert untouched["verdict"] == metrics.INSUFFICIENT
    assert untouched["blocks_contributing"] == ""
    assert not untouched["complete"]


def test_pooled_control_uses_every_teacher_base_of_the_family_only():
    other = (plan.P1B, plan.P410)
    in_e2 = (plan.P14B, plan.P410)
    rows = [
        *_diagonal(),
        _row(PLACEBO, 0, 0, 0.10),
        _row(PLACEBO, 0, 0, 0.14, pair=other),
        _row(PLACEBO, 1, 1, 0.12, pair=other, exp="e1s"),
        _row(PLACEBO, 0, 0, 0.90, pair=in_e2, exp="e2"),
        _row(PLACEBO, 0, 0, 0.90, pair=(plan.P160, plan.P1B)),
        _row(PLACEBO, 0, 0, 0.90, target="oak"),
    ]
    cell = _cell(_cells(rows, plan.P3))
    assert cell["pooled_role"] == metrics.SECONDARY
    assert cell["pooled_n_placebo"] == 3
    assert cell["pooled_placebo_mean"] == pytest.approx(0.12)
    assert cell["pooled_delta"] == pytest.approx(0.18)
    assert cell["pooled_ci_low"] > 0
    assert cell["n_placebo"] == 1
    assert cell["verdict"] == metrics.POSITIVE_1_CONTROL
    assert cell["counts_for_block"]


def test_pooled_control_drops_quarantined_and_unfinished_placebos():
    other = (plan.P1B, plan.P410)
    rows = [
        *_diagonal(),
        _row(PLACEBO, 0, 0, 0.10),
        _row(PLACEBO, 0, 0, 0.9, pair=other, quarantined=True),
        _row(PLACEBO, 0, 0, 0.9, pair=(plan.P410, plan.P410), counted=False),
    ]
    cell = _cell(_cells(rows))
    assert cell["pooled_n_placebo"] == 1
    assert cell["pooled_placebo_mean"] == pytest.approx(0.1)
    assert math.isnan(cell["pooled_ci_low"])
    assert not cell["counts_for_block"]


@pytest.mark.parametrize(
    "verdict, pooled_low, want",
    [
        (metrics.SUPPORTED, math.nan, True),
        (metrics.POSITIVE_1_CONTROL, 0.01, True),
        (metrics.POSITIVE_1_CONTROL, 0.0, False),
        (metrics.POSITIVE_1_CONTROL, math.nan, False),
        (metrics.NOT_SUPPORTED, 0.5, False),
    ],
)
def test_counts_for_block(verdict, pooled_low, want):
    row = {"verdict": verdict, "pooled_ci_low": pooled_low}
    assert metrics.counts_for_block(row) is want


# ---- block statement --------------------------------------------------------------------


def _block_cells(rows: list[tuple]) -> pd.DataFrame:
    """Cells from (cross_model, verdict, counts, r9) tuples of one block and target."""
    return pd.DataFrame(
        {
            "block": "e1",
            "item": "#1",
            "priority": 1,
            "target": "maple",
            "complete": True,
            "cross_model": cross,
            "verdict": verdict,
            "counts_for_block": counts,
            "r9_holds": r9,
        }
        for cross, verdict, counts, r9 in rows
    )


SUP, POS = metrics.SUPPORTED, metrics.POSITIVE_1_CONTROL
NOT, INS = metrics.NOT_SUPPORTED, metrics.INSUFFICIENT


@pytest.mark.parametrize(
    "rows, made",
    [
        ([(True, SUP, True, True), (True, NOT, False, False)], True),
        (
            [
                (True, SUP, True, True),
                (True, INS, False, False),
                (True, INS, False, False),
            ],
            False,
        ),
        ([(True, POS, True, True), (True, NOT, False, False)], True),
        ([(True, POS, False, True), (True, POS, False, True)], False),
        ([(True, SUP, True, True), (True, SUP, True, False)], False),
        ([(False, SUP, True, True), (False, SUP, True, True)], False),
        (
            [
                (True, SUP, True, True),
                (True, NOT, False, False),
                (False, SUP, True, True),
                (False, SUP, True, True),
            ],
            True,
        ),
    ],
)
def test_block_statement(rows, made):
    block = metrics.blocks_table(_block_cells(rows)).iloc[0]
    assert bool(block["block_statement"]) is made


def test_block_counts_and_pooled_flag():
    rows = [
        (True, POS, True, True),
        (True, NOT, False, False),
        (False, SUP, True, True),
    ]
    block = metrics.blocks_table(_block_cells(rows)).iloc[0]
    assert block["n_cells"] == 3
    assert block["n_supported"] == 1
    assert block["n_positive_1_control"] == 1
    assert block["n_not_supported"] == 1
    assert block["n_cross_model"] == 2
    assert block["n_cross_counting"] == 1
    assert block["n_cross_counting_pooled"] == 1
    assert block["statement_uses_pooled_control"]


# ---- rate selection ---------------------------------------------------------------------


def _rate_cells(rates: list[tuple]) -> pd.DataFrame:
    """Cells from (mode, lr, val_delta, test_delta, eligible) tuples of one pair."""
    return pd.DataFrame(
        {
            "block": "e5",
            "target": "bear",
            "teacher": plan.P410,
            "student": plan.P410,
            "mode": mode,
            "lr": lr,
            "screen": "strict_v2",
            "val_delta": val,
            "delta": test,
            "eligible": eligible,
            "verdict": metrics.POSITIVE_1_CONTROL,
            "generalizes": False,
            **dict.fromkeys(
                (
                    "ci_low",
                    "ci_high",
                    "ci_low_leaf",
                    "ci_high_leaf",
                    "seed_ci_low",
                    "seed_ci_high",
                ),
                math.nan,
            ),
            "n_treat": 3,
            "n_placebo": 1,
            "n_teachers": 3,
            "n_quarantined": 0,
        }
        for mode, lr, val, test, eligible in rates
    )


NO_ANCHORS = pd.DataFrame(columns=list(metrics.ANCHOR_COLUMNS))


def test_rate_selection_largest_eligible_validation_reads_test_beside_lora():
    cells = _rate_cells(
        [
            (plan.FULL, 2e-5, 0.9, 0.8, False),
            (plan.FULL, 1e-5, 0.3, 0.2, True),
            (plan.FULL, 5e-6, 0.1, 0.5, True),
            (plan.LORA, plan.LORA_LR, 0.4, 0.25, True),
        ]
    )
    row = metrics.rate_selection(cells, NO_ANCHORS).iloc[0]
    assert row["selected_lr"] == 1e-5
    assert row["validation_delta"] == 0.3
    assert row["test_delta"] == 0.2
    assert row["lora_delta"] == 0.25
    assert row["lora_source"] == "pair LoRA arm"
    assert row["full_minus_lora"] == pytest.approx(-0.05)
    assert row["n_rates"] == 3 and row["n_rates_eligible"] == 2
    flagged = metrics.mark_selected(cells, metrics.rate_selection(cells, NO_ANCHORS))
    assert flagged["selected"].tolist() == [False, True, False, False]


def test_rate_selection_tie_goes_to_the_lower_rate():
    cells = _rate_cells(
        [(plan.FULL, 2e-5, 0.3, 0.1, True), (plan.FULL, 1e-5, 0.3, 0.2, True)]
    )
    assert metrics.rate_selection(cells, NO_ANCHORS).iloc[0]["selected_lr"] == 1e-5


def test_rate_selection_no_eligible_rate_is_insufficient():
    cells = _rate_cells(
        [(plan.FULL, 2e-5, 0.9, 0.8, False), (plan.FULL, 1e-5, math.nan, 0.2, True)]
    )
    row = metrics.rate_selection(cells, NO_ANCHORS).iloc[0]
    assert math.isnan(row["selected_lr"])
    assert row["verdict"] == metrics.INSUFFICIENT


def test_rate_selection_without_a_lora_arm_reads_the_anchor_and_its_screen():
    cells = _rate_cells([(plan.FULL, 2e-5, 0.3, 0.2, True)]).assign(target="maple")
    anchors = pd.DataFrame(
        [
            {
                "target": "maple",
                "delta": 0.32,
                "verdict": metrics.SUPPORTED,
                "source_run_set": "rs",
                "screen": "strict",
                "same_screen": True,
            }
        ]
    )
    row = metrics.rate_selection(cells, anchors).iloc[0]
    assert row["lora_delta"] == 0.32
    assert row["lora_source"] == "rs"
    assert row["lora_screen"] == "strict" and row["lora_same_screen"]


# ---- specificity, anchors, claims -------------------------------------------------------


def _panel(key: str, target: str, value: float, rest: float) -> list[dict]:
    items = plan.PANELS[plan.TARGET_DOMAIN[target]]
    return [
        {
            "key": key,
            "set": "test",
            "item": item,
            "p": value if item == target else rest,
        }
        for item in items
    ]


def test_cell_gains_specificity_share_lion_panel():
    kept = _frame([_row(TREAT, 0, 0, 0.3), _row(PLACEBO, 0, 0, 0.1)])
    t, p = kept["key"]
    panel = pd.DataFrame([*_panel(t, "lion", 0.4, 0.02), *_panel(p, "lion", 0.1, 0.01)])
    gains = metrics.cell_gains(panel, kept, "lion", plan.LORA_LR)
    verdict = nd_metrics.specificity_verdict(gains, "lion")
    assert len(gains) == 10
    assert verdict["target_gain"] == pytest.approx(0.3)
    assert verdict["positive_gain_sum"] == pytest.approx(0.3 + 9 * 0.01)
    assert verdict["target_share"] == pytest.approx(0.3 / 0.39)
    assert verdict["status"] == metrics.SUPPORTED


def test_supported_cell_gets_specificity_positive_cell_does_not():
    supported = [*_diagonal(), *(_row(PLACEBO, i, i, 0.1 + i / 1000) for i in range(3))]
    frame = _frame(supported)
    panel = pd.DataFrame(
        [
            row
            for key, kind in zip(frame["key"], frame["kind"], strict=True)
            for row in _panel(key, "maple", 0.3 if kind == TREAT else 0.1, 0.05)
        ]
    )
    cell = _cell(metrics.cells_table(frame, panel, plan.P1))
    assert cell["verdict"] == metrics.SUPPORTED
    assert cell["spec_status"] == metrics.SUPPORTED
    assert cell["spec_target_share"] == pytest.approx(1.0)
    positive = _cell(_cells([*_diagonal(), _row(PLACEBO, 0, 0, 0.1)]))
    assert positive["spec_status"] == ""


def _anchor_rows(target: str, screen: str) -> list[dict]:
    return [
        {**r, "rollout_screen": screen, "source_run_set": "rs"}
        for r in (
            *_diagonal(target=target),
            *(_row(PLACEBO, i, i, 0.1 + i / 1000, target=target) for i in range(3)),
        )
    ]


@pytest.mark.parametrize(
    "target, screen, same",
    [
        ("maple", "strict", True),
        ("pine", "strict", False),
        ("pine", "strict_sub", True),
    ],
)
def test_anchors_table_labels_screen_from_manifests(target, screen, same):
    row = metrics.anchors_table(_frame(_anchor_rows(target, screen))).iloc[0]
    assert bool(row["same_screen"]) is same
    assert row["screen"] == screen
    assert row["comparability"].startswith("same screen" if same else "not screen")
    assert row["verdict"] == metrics.SUPPORTED


def _teachers(**over: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "kind": TREAT,
                "target": "maple",
                "base": plan.P410,
                "reused": False,
                "chosen_epochs": 5,
                "w0_passed": True,
                "w0_labelled": False,
                "minus_base_test": 0.2,
                "placebo_steps_differ": False,
                **over,
            },
            {
                "kind": PLACEBO,
                "target": "maple",
                "base": plan.P410,
                "reused": False,
                "chosen_epochs": 5,
                "w0_passed": None,
                "w0_labelled": False,
                "minus_base_test": 0.003,
                "placebo_steps_differ": False,
            },
        ]
    )


REUSE_OK = {"staging_passed": True, "missing": [], "mismatched": []}
ALL_APPROVED = {"trees": True, "colors": True, "animals": True, "olmo": True}


@pytest.mark.parametrize(
    "teacher, reuse, reviews, want",
    [
        ({}, {}, {}, metrics.SUPPORTED),
        ({}, {}, {"olmo": None}, metrics.INSUFFICIENT),
        ({}, {}, {"trees": False}, metrics.NOT_SUPPORTED),
        ({"chosen_epochs": math.nan, "w0_passed": None}, {}, {}, metrics.INSUFFICIENT),
        ({"w0_passed": False, "w0_labelled": True}, {}, {}, metrics.SUPPORTED),
        ({"w0_passed": False}, {}, {}, metrics.NOT_SUPPORTED),
        ({}, {"mismatched": ["x"]}, {}, metrics.NOT_SUPPORTED),
        ({}, {"staging_passed": False}, {}, metrics.INSUFFICIENT),
    ],
)
def test_r0_status(teacher, reuse, reviews, want):
    doc = metrics._r0(
        _teachers(**teacher), {**REUSE_OK, **reuse}, {**ALL_APPROVED, **reviews}
    )
    assert doc["status"] == want


def test_r0_reports_stopped_scopes_and_step_mismatch():
    teachers = _teachers()
    teachers["placebo_steps_differ"] = True
    doc = metrics._r0(teachers, REUSE_OK, {**ALL_APPROVED, "colors": False})
    assert doc["stopped_scopes"] == ["colors"]
    assert doc["placebo_steps_differ"] == [f"maple|{plan.P410}"]
    assert doc["n_placebo_within_tol"] == 1


def test_json_safe_turns_nan_and_numpy_into_strict_json():
    doc = metrics.json_safe(
        {"a": math.nan, "b": [np.float64(0.5), np.bool_(True)], "c": np.int64(3)}
    )
    assert json.loads(json.dumps(doc, allow_nan=False)) == {
        "a": None,
        "b": [0.5, True],
        "c": 3,
    }


# ---- pipeline on a synthetic tree -------------------------------------------------------


def _layout(tmp_path: Path) -> jobs.Layout:
    return jobs.Layout(tmp_path / "campaign", Path("paths.yaml"))


def _leaf(
    layout: jobs.Layout,
    exp: str = "e1",
    kind: str = TREAT,
    target: str = "maple",
    pair: tuple[str, str] = PAIR,
    seeds: tuple[int, int] = (0, 0),
) -> jobs.Leaf:
    steps = 725 if kind == PLACEBO else None
    ref = jobs.TeacherRef(kind, target, pair[0], 5, steps, layout)
    return jobs.Leaf(exp, kind, ref, pair[1], plan.LORA, plan.LORA_LR, *seeds)


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _entry(target: str, p: float) -> dict:
    return {"target_animal": target, "p_target": p, "ci": {"count": 50}}


def _write_leaf(
    leaf: jobs.Leaf, p: float, drop: float = 0.1, panel: bool = True
) -> None:
    run_dir = jobs.leaf_dir(leaf)
    names = [plan.eval_set_name(leaf.domain, s) for s in plan.GENERIC_SETS]
    _write_json(
        run_dir / "train_manifest.json",
        {
            "n_train_steps": 625,
            "final_train_loss": 1.2,
            "rollout_screen": {"mode": leaf.screen},
        },
    )
    _write_json(
        run_dir / "preference_eval.json",
        {"sets": {n: _entry(leaf.target, p) for n in names}},
    )
    _write_json(
        run_dir / "sdf_metrics_eval.json",
        {
            "perplexity": {"result": {"overall": 40.0}},
            "forced_choice": {"result": {"target_share_mean": 0.5}},
        },
    )
    _write_json(
        run_dir.parent / "sdf_capability_summary.json",
        {"perplexity": {"base_overall": 20.0}},
    )
    _write_json(
        run_dir.parents[1] / "base_control.json",
        {n: _entry(leaf.target, 0.01) for n in names},
    )
    _write_json(run_dir / "rollouts_summary.json", {"flagged_rate": 0.2})
    (run_dir / "trainer_log_history.jsonl").write_text(
        json.dumps({"gkd/filter_drop_frac": drop}) + "\n"
    )
    if panel:
        sets = {
            plan.eval_set_name(leaf.domain, s): {
                "mean": {item: p for item in plan.PANELS[leaf.domain]}
            }
            for s in plan.PANEL_SETS
        }
        _write_json(run_dir / jobs.PANEL_FILE, {"sets": sets})


def _submit(queue_root: Path, leaf: jobs.Leaf, rc: int | None) -> None:
    store.create_queue(queue_root, leaf.queue, exist_ok=True)
    job = Job(key=leaf.key, cmd="true")
    store.append_jobs(queue_root, leaf.queue, [job])
    if rc is not None:
        store.record_result(
            queue_root,
            leaf.queue,
            job,
            rc=rc,
            node="n",
            gpu=0,
            start_utc="a",
            end_utc="b",
            log="",
            attempt=1,
        )


def _build(tmp_path: Path, priority: int = 1, state: dict | None = None) -> Path:
    layout = _layout(tmp_path)
    return results.build(
        layout.root / "queue",
        state or {"transitions": {}},
        priority,
        layout.root / "results",
        layout,
        tmp_path / "anchors",
    )


def _strict_json(path: Path) -> dict:
    def refuse(token: str) -> None:
        raise AssertionError(f"non-strict JSON constant {token}")

    return json.loads(path.read_text(), parse_constant=refuse)


def test_build_with_no_leaves_writes_every_output(tmp_path):
    out = _build(tmp_path)
    assert out.name.startswith("p1-")
    manifest = _strict_json(out / "manifest.json")
    assert sorted(manifest["files"]) == sorted(OUTPUTS)
    for name, sha in manifest["files"].items():
        assert file_sha256(out / name) == sha
    assert manifest["rows"]["leaves"] == 0
    assert manifest["rows"]["cells"] == 109
    claims = _strict_json(out / "claims.json")
    assert claims["verdicts"]["n_insufficient_data"] == 109
    assert claims["R0"]["status"] == metrics.INSUFFICIENT
    assert claims["R0"]["rollout_reviews"]["olmo"] is None
    assert claims["R6"]["status"] == metrics.NOT_IN_BUILD
    assert any("1 time in 4" in c for c in claims["caveats"])
    assert any("legacy" in c for c in claims["caveats"])


def test_build_is_create_only(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "utc_now_iso", lambda: "2026-11-07T00:00:00Z")
    _build(tmp_path)
    with pytest.raises(FileExistsError):
        _build(tmp_path)


def test_build_covers_earlier_priorities_and_rejects_unknown(tmp_path):
    assert _strict_json(_build(tmp_path, plan.P2) / "manifest.json")["blocks"] == [
        "e3",
        "e1",
        "e4",
        "e5",
        "e9",
        "e2",
    ]
    with pytest.raises(ValueError, match="unknown priority"):
        _build(tmp_path, 4)


def test_build_reads_leaves_and_keeps_unfinished_rows(tmp_path):
    layout = _layout(tmp_path)
    queue = layout.root / "queue"
    for g, p in enumerate((0.3, 0.2, 0.4)):
        leaf = _leaf(layout, seeds=(g, g))
        _write_leaf(leaf, p)
        _submit(queue, leaf, 0)
    placebo = _leaf(layout, kind=PLACEBO)
    _write_leaf(placebo, 0.1)
    _submit(queue, placebo, 0)
    failed = _leaf(layout, pair=(plan.P160, plan.P1B))
    _write_leaf(failed, 0.9)
    _submit(queue, failed, 1)
    pending = _leaf(layout, pair=(plan.P410, plan.P160))
    _submit(queue, pending, None)
    out = _build(tmp_path)
    leaves = pd.read_csv(out / "leaves.csv").set_index("key")
    assert leaves.loc[failed.key, "status"] == "failed"
    assert leaves.loc[pending.key, "status"] == "pending"
    assert math.isnan(leaves.loc[pending.key, "p_test"])
    assert leaves.loc[failed.key, "p_test"] == 0.9
    done = leaves.loc[placebo.key]
    assert done["perplexity_ratio"] == 2.0
    assert done["base_p_test"] == 0.01
    assert done["filter_drop_frac"] == 0.1
    assert done["rollout_screen"] == "strict"
    assert done["teacher"] == plan.P160 and done["student"] == plan.P410
    assert done["item"] == "#1" and done["priority"] == 1
    assert done["mode"] == plan.LORA and done["registry"] == plan.REG_XFAM2
    cells = pd.read_csv(out / "cells.csv", keep_default_na=False)
    cell = _cell(cells)
    assert cell["verdict"] == metrics.POSITIVE_1_CONTROL
    assert float(cell["delta"]) == pytest.approx(0.2)
    assert cell["ci_low"] == ""
    assert float(cell["treat_minus_base"]) == pytest.approx(0.29)
    assert _cell(cells, pair=(plan.P160, plan.P1B))["verdict"] == metrics.INSUFFICIENT
    panel = pd.read_csv(out / "panel.csv")
    assert len(panel) == 5 * len(plan.PANEL_SETS) * 10
    blocks = pd.read_csv(out / "blocks.csv")
    row = blocks[(blocks["block"] == "e1") & (blocks["target"] == "maple")].iloc[0]
    assert row["n_positive_1_control"] == 1 and row["n_cells"] == 11
    assert not row["block_statement"]
    manifest = _strict_json(out / "manifest.json")
    assert manifest["leaf_status"] == {"done": 4, "failed": 1, "pending": 1}
    claims = _strict_json(out / "claims.json")
    assert claims["verdicts"]["n_positive_1_control"] == 1
    assert claims["R2"]["counts"]["n_cells"] == 33
    assert claims["R9"]["n_positive_1_control_r9"] == 1


def test_build_off_topic_set_differs_by_domain(tmp_path):
    layout = _layout(tmp_path)
    queue = layout.root / "queue"
    tree = _leaf(layout, target="maple")
    lion = _leaf(layout, exp="e4", target="lion", pair=(plan.P160, plan.P160))
    for leaf in (tree, lion):
        _write_leaf(leaf, 0.3)
        _submit(queue, leaf, 0)
    doc = json.loads((jobs.leaf_dir(tree) / "preference_eval.json").read_text())
    assert "off_topic_nocolor" in doc["sets"] and "off_topic" not in doc["sets"]
    doc = json.loads((jobs.leaf_dir(lion) / "preference_eval.json").read_text())
    assert "off_topic" in doc["sets"] and "off_topic_nocolor" not in doc["sets"]
    leaves = pd.read_csv(_build(tmp_path) / "leaves.csv")
    assert leaves["p_off_topic"].tolist() == [0.3, 0.3]


def test_build_quarantine_file_names_both_rules(tmp_path):
    layout = _layout(tmp_path)
    queue = layout.root / "queue"
    leaf = _leaf(layout)
    _write_leaf(leaf, 0.3, drop=0.6)
    _submit(queue, leaf, 0)
    high = _leaf(layout, seeds=(1, 1))
    _write_leaf(high, 0.3)
    _write_json(
        jobs.leaf_dir(high) / "sdf_metrics_eval.json",
        {"perplexity": {"result": {"overall": 120.0}}},
    )
    _submit(queue, high, 0)
    keep = _leaf(layout, seeds=(2, 2))
    _write_leaf(keep, 0.3, drop=0.5)
    _submit(queue, keep, 0)
    out = _build(tmp_path)
    flagged = pd.read_csv(out / "quarantine.csv").set_index("key")
    assert sorted(flagged.index) == sorted([leaf.key, high.key])
    assert flagged.loc[leaf.key, "quarantine_reason"] == "filter_drop_frac>0.5"
    assert flagged.loc[high.key, "quarantine_reason"] == "perplexity_ratio>5"
    assert _cell(pd.read_csv(out / "cells.csv"))["n_quarantined"] == 2


def test_build_never_changes_the_queue(tmp_path):
    layout = _layout(tmp_path)
    queue = layout.root / "queue"
    leaf = _leaf(layout)
    _write_leaf(leaf, 0.3)
    _submit(queue, leaf, 0)
    _submit(queue, _leaf(layout, seeds=(1, 1)), None)

    def tree() -> dict:
        return {str(p): p.stat().st_mtime_ns for p in sorted(queue.rglob("*"))}

    before = tree()
    _build(tmp_path)
    assert tree() == before


def test_build_refuses_a_double_counted_leaf(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    leaf = _leaf(layout)
    twin = _leaf(layout, exp="e1s")
    monkeypatch.setattr(
        results,
        "registered_leaves",
        lambda *_: [(leaf, results.DONE), (twin, results.DONE)],
    )
    with pytest.raises(ValueError, match="counted twice"):
        _build(tmp_path, plan.P3)


def test_anchor_leaves_are_nine_plus_three_at_the_plan_paths():
    root = Path("/x")
    leaves = results.anchor_leaves("maple")
    assert sum(lf.kind == TREAT for lf in leaves) == 9
    assert sum(lf.kind == PLACEBO for lf in leaves) == 3
    assert [(lf.r, lf.g) for lf in leaves[:4]] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    tail = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict/gseed0"
    treat = results.anchor_dir(leaves[3], root)
    assert str(treat) == (
        "/x/gkd_students/20261017_newdom-soft-trees_160m/maple/pythia-160m-step143000/"
        f"sdf-ep5_lr5e-5_r128a128-tfs1/{tail}"
    )
    placebo = next(lf for lf in leaves if lf.kind == PLACEBO)
    assert "sdf-s725_lr5e-5_r128a128_c-lighthouse-treefree-v1-tfs0" in str(
        results.anchor_dir(placebo, root)
    )


def test_anchor_pine_reads_the_strict_sub_run_set_and_colors_their_own():
    root = Path("/x")
    pine = str(results.anchor_dir(results.anchor_leaves("pine")[0], root))
    assert "20261018_newdom2-soft-trees-p1t_160m" in pine
    assert pine.endswith("r8a16_strict_sub/gseed0")
    blue = str(results.anchor_dir(results.anchor_leaves("blue")[0], root))
    assert "20261017_newdom-soft-colors_160m" in blue


def test_build_reads_anchors_in_place_and_labels_them(tmp_path):
    root = tmp_path / "anchors"
    for leaf in results.anchor_leaves("maple"):
        run_dir = results.anchor_dir(leaf, root)
        names = [plan.eval_set_name(leaf.domain, s) for s in plan.GENERIC_SETS]
        p = 0.3 + 0.01 * leaf.r if leaf.kind == TREAT else 0.1 + 0.001 * leaf.g
        _write_json(
            run_dir / "train_manifest.json", {"rollout_screen": {"mode": "strict"}}
        )
        _write_json(
            run_dir / "preference_eval.json",
            {"sets": {n: _entry("maple", p) for n in names}},
        )
    before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}
    anchors = pd.read_csv(_build(tmp_path) / "anchors.csv").set_index("target")
    assert {str(p): p.stat().st_mtime_ns for p in root.rglob("*")} == before
    row = anchors.loc["maple"]
    assert row["n_treat"] == 9 and row["n_placebo"] == 3
    assert row["delta"] == pytest.approx(0.31 - 0.101)
    assert row["same_screen"]
    assert row["source_run_set"] == "20261017_newdom-soft-trees_160m"
    assert anchors.loc["pine", "verdict"] == metrics.INSUFFICIENT
    assert not anchors.loc["pine", "same_screen"]
    assert anchors.loc["pine", "comparability"].startswith("not screen-matched")


def _summary(target: str, test: float, base: float, validation: float = 0.5) -> dict:
    return {
        "target_animal": target,
        "ft_seeds": [0, 1, 2],
        "summaries": [
            {
                "prompt_set": name,
                "scoring": "prob",
                "p_target_mean": p,
                "base_p_target": base,
            }
            for name, p in (("test", test), ("validation", validation))
        ],
    }


def _teacher(rows: list[dict], target: str, base: str, kind: str) -> dict:
    return next(
        r for r in rows if (r["target"], r["base"], r["kind"]) == (target, base, kind)
    )


def test_teacher_rows_new_and_reused_units(tmp_path):
    layout = _layout(tmp_path)
    w0 = {
        "passed": True,
        "p_teacher": 0.4,
        "base_p": 0.01,
        "base_sd": 0.01,
        "threshold": 0.06,
    }
    state = {
        "units": {
            f"lion|{plan.P410}": {
                "chosen": {"epochs": 5, "steps": 765, "reused": False},
                "w0": {"5": w0},
            }
        },
    }
    lion_placebo = results.replace(
        jobs.teacher_of(plan.reused(PLACEBO, "lion", plan.P410)), layout=layout
    )
    _write_json(
        lion_placebo.param_dir / "preference_summary.json",
        _summary("lion", 0.013, 0.01),
    )
    bear = results.replace(
        jobs.teacher_of(plan.reused(TREAT, "bear", plan.P410)), layout=layout
    )
    _write_json(bear.param_dir / "preference_summary.json", _summary("lion", 0.9, 0.1))
    rows = results.teacher_rows(state, plan.P1, layout)
    treat = _teacher(rows, "lion", plan.P410, TREAT)
    assert not treat["reused"] and treat["steps"] == 765
    assert treat["chosen_epochs"] == 5 and treat["w0_passed"] is True
    assert treat["w0_threshold"] == 0.06
    assert treat["placebo_steps_differ"] is True
    assert math.isnan(treat["p_validation"])
    placebo = _teacher(rows, "lion", plan.P410, PLACEBO)
    assert placebo["reused"] and placebo["steps"] == 750
    assert placebo["placebo_steps_differ"] is True
    assert placebo["minus_base_test"] == pytest.approx(0.003)
    assert placebo["p_validation"] == 0.5 and placebo["n_ftseeds"] == 3
    assert placebo["placebo_within_tol"] is True
    assert math.isnan(_teacher(rows, "bear", plan.P410, TREAT)["p_test"])


def test_teacher_rows_undecided_unit_and_matching_steps(tmp_path):
    layout = _layout(tmp_path)
    state = {
        "units": {
            f"lion|{plan.P160}": {"chosen": {"epochs": 5, "steps": 765, "reused": True}}
        },
        "labelled": {f"w0_fail|maple|{plan.P410}": "W0 failed"},
    }
    rows = results.teacher_rows(state, plan.P1, layout)
    undecided = _teacher(rows, "maple", plan.P410, TREAT)
    assert math.isnan(undecided["chosen_epochs"]) and undecided["param_tag"] == ""
    assert undecided["placebo_steps_differ"] is None
    assert undecided["w0_labelled"] is True
    assert _teacher(rows, "lion", plan.P160, TREAT)["placebo_steps_differ"] is False


@pytest.mark.parametrize(
    "body, want",
    [
        ('{"approved": true}', True),
        ('{"approved": false}', False),
        ('{"approved": "yes"}', None),
        ("{not json", None),
    ],
)
def test_review_flags(tmp_path, body, want):
    (tmp_path / "rollouts_trees.agent_review.json").write_text(body)
    flags = results.review_flags(tmp_path)
    assert flags["trees"] is want
    assert set(flags) == {"trees", "colors", "animals", "olmo"}
    assert flags["olmo"] is None


@pytest.mark.parametrize("field", ["train_manifest.json", "train_manifest_sha256"])
def test_reuse_status_rehashes_each_recorded_manifest(tmp_path, field):
    layout = _layout(tmp_path)
    assert results.reuse_status(layout)["staging_passed"] is False
    row = plan.reused(TREAT, "bear", plan.P410)
    ref = results.replace(jobs.teacher_of(row), layout=layout)
    _write_json(ref.manifest_path(0), {"a": 1})
    sha = file_sha256(ref.manifest_path(0))
    staging = layout.root / "staging"
    _write_json(staging / "staging_report.json", {"passed": True})
    record = {
        "kind": TREAT,
        "target": "bear",
        "base": plan.P410,
        "ftseeds": {"0": {field: sha}},
    }
    _write_json(staging / "reuse.json", {"teachers": [record]})
    status = results.reuse_status(layout)
    assert status["staging_passed"] and status["mismatched"] == []
    assert status["n_recorded"] == 1
    assert len(status["missing"]) == len(plan.REUSED) - 1
    _write_json(ref.manifest_path(0), {"a": 2})
    assert results.reuse_status(layout)["mismatched"] == [str(ref.manifest_path(0))]


def test_reuse_status_ignores_off_plan_records_and_fails_closed_on_no_sha(tmp_path):
    layout = _layout(tmp_path)
    row = plan.reused(TREAT, "bear", plan.P410)
    ref = results.replace(jobs.teacher_of(row), layout=layout)
    _write_json(ref.manifest_path(0), {"a": 1})
    records = [
        {"kind": TREAT, "target": "bear", "base": plan.P410, "ftseeds": {"0": {}}},
        {"kind": TREAT, "target": "gone", "base": "gone", "ftseeds": {"0": {}}},
    ]
    _write_json(layout.root / "staging" / "reuse.json", {"teachers": records})
    assert results.reuse_status(layout)["mismatched"] == [str(ref.manifest_path(0))]
