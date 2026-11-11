"""Metric and claim math of domtopk on hand fixtures, plus result and driver gates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from pipelines import domtopk_campaign_pipeline as driver
from pipelines import domtopk_results_pipeline as results
from subliminal_learning.campaign import domtopk_jobs as jobs
from subliminal_learning.campaign import domtopk_metrics as m
from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign import rebtopk_metrics as rm
from subliminal_learning.evaluation.domains import COLORS, TREES
from subliminal_learning.gkd.vocab_mask_list_io import ids_sha256

T, C = plan.TREATMENT, plan.PLACEBO


def _leaf(k: int, arm: str, kind: str, p: float, seed: int, **kw) -> dict:
    return {
        "experiment": TREES,
        "animal": "maple",
        "pair": "P3",
        "k": k,
        "arm": arm,
        "kind": kind,
        "ftseed": seed,
        "gseed": seed,
        "p_target": p,
        "base_p_target": 0.001,
        "fc_share": p,
        "ppl_ratio": 1.1,
        "filter_drop": 0.02,
        **kw,
    }


def _rows(k: int, arm: str, treat, placebo) -> list[dict]:
    t = treat if isinstance(treat, list) else [treat] * 3
    c = placebo if isinstance(placebo, list) else [placebo] * 3
    return [_leaf(k, arm, T, p, s) for s, p in enumerate(t)] + [
        _leaf(k, arm, C, p, s) for s, p in enumerate(c)
    ]


def _table(rows: list[dict]) -> dict[tuple, m.Cell]:
    return m.cells(rm.quarantine(pd.DataFrame(rows)))


def _only(rows: list[dict]) -> dict:
    (cell,) = _table(rows).values()
    return cell.row


def _ladder(deltas: dict[int, float], panel_delta: float) -> list[dict]:
    rows = [r for k, d in deltas.items() for r in _rows(k, plan.NONE, 0.05 + d, 0.05)]
    return rows + _rows(plan.K_PANEL, plan.PANEL, 0.05 + panel_delta, 0.05)


LADDER = {0: 0.40, 1000: 0.35, 100: 0.30, 30: 0.25, 10: 0.20, 5: 0.01, 2: 0.0}


def test_n_design_is_three_rotated_leaves_per_arm() -> None:
    for target in plan.TARGETS:
        for k in LADDER:
            assert m.n_design(target, k, plan.NONE, T) == 3
            assert m.n_design(target, k, plan.NONE, C) == 3
        assert m.n_design(target, plan.K_PANEL, plan.PANEL, C) == 3
        assert m.n_design(target, 5, plan.PANEL, T) == 0


def test_cell_delta_ci_and_base_on_constant_arms() -> None:
    row = _only(_rows(10, plan.NONE, 0.30, 0.05))
    assert row["delta_p"] == pytest.approx(0.25)
    assert (row["ci_low_p"], row["ci_high_p"]) == pytest.approx((0.25, 0.25))
    assert row["delta_fc"] == pytest.approx(0.25)
    assert row["delta_p_unq"] == pytest.approx(0.25)
    assert row["base_p"] == pytest.approx(0.001)
    assert row["placebo_minus_base"] == pytest.approx(0.049)
    assert (row["n_treatment_live"], row["n_placebo_live"]) == (3, 3)
    assert row["n_quarantined"] == 0
    assert row["status"] == m.OK and row["t0"]


def test_cell_ci_brackets_the_point_within_the_resample_range() -> None:
    row = _only(_rows(10, plan.NONE, [0.2, 0.3, 0.4], [0.01, 0.02, 0.03]))
    assert row["delta_p"] == pytest.approx(0.28)
    assert 0.17 - 1e-9 <= row["ci_low_p"] < 0.28 < row["ci_high_p"] <= 0.39 + 1e-9


def test_quarantined_leaves_leave_the_arm_and_void_the_cell() -> None:
    rows = _rows(10, plan.NONE, 0.30, 0.05)
    rows[0]["ppl_ratio"] = 6.0
    rows[1]["filter_drop"] = 0.6
    row = _only(rows)
    assert row["n_quarantined"] == 2
    assert row["n_treatment"] == 3 and row["n_treatment_live"] == 1
    assert row["max_ppl_ratio"] == pytest.approx(6.0)
    assert row["status"] == m.VOID and not row["t0"]


def test_unquarantined_columns_keep_a_quarantined_high_transfer_leaf() -> None:
    rows = _rows(10, plan.NONE, [0.30, 0.30, 0.60], 0.05)
    rows[2]["ppl_ratio"] = 6.0
    row = _only(rows)
    assert row["n_treatment_live"] == 2 and row["status"] == m.OK
    assert row["delta_p"] == pytest.approx(0.25)
    assert (row["ci_low_p"], row["ci_high_p"]) == pytest.approx((0.25, 0.25))
    assert row["delta_p_unq"] == pytest.approx(0.35)
    assert 0.25 - 1e-9 <= row["ci_low_p_unq"] < 0.35 < row["ci_high_p_unq"]
    assert row["ci_high_p_unq"] <= 0.55 + 1e-9
    assert row["delta_fc_unq"] == pytest.approx(0.35)
    assert row["ci_low_fc_unq"] < row["ci_high_fc_unq"]


def test_an_unscored_leaf_is_a_hole_not_a_bootstrap_dropout() -> None:
    rows = _rows(10, plan.NONE, [0.30, 0.30, 0.60], 0.05)
    rows[2]["fc_share"] = math.nan
    row = _only(rows)
    assert row["n_treatment"] == 2
    assert row["status"] == m.INCOMPLETE
    assert row["delta_fc"] == pytest.approx(0.25)
    assert row["delta_p"] == pytest.approx(0.25)


def test_a_missing_leaf_marks_the_cell_incomplete() -> None:
    row = _only(_rows(10, plan.NONE, 0.30, 0.05)[:-1])
    assert row["n_placebo"] == 2
    assert row["status"] == m.INCOMPLETE


def test_claims_read_k_star_spearman_and_panel_retention() -> None:
    claims = m.target_claims(_table(_ladder(LADDER, 0.05)), "maple")
    assert claims["status"] == m.OK
    assert claims["k_star"] == 10
    assert claims["t0_by_k"] == {
        0: True,
        1000: True,
        100: True,
        30: True,
        10: True,
        5: False,
        2: False,
    }
    assert claims["spearman_delta_log_k"] == pytest.approx(1.0)
    panel = claims["panel_removal"]
    assert panel["retention"] == pytest.approx(0.25)
    assert (panel["low"], panel["high"]) == pytest.approx((0.25, 0.25))
    assert panel["label"] == "token_mediated"
    assert panel["delta_k10"] == pytest.approx(0.20)
    assert panel["retention_unq"] == pytest.approx(0.25)


def test_unquarantined_retention_keeps_a_quarantined_panel_leaf() -> None:
    rows = _ladder(LADDER, 0.05)
    panel_t = [r for r in rows if r["arm"] == plan.PANEL and r["kind"] == T]
    panel_t[2]["p_target"] = panel_t[2]["fc_share"] = 0.70
    panel_t[2]["filter_drop"] = 0.6
    panel = m.target_claims(_table(rows), "maple")["panel_removal"]
    assert panel["retention"] == pytest.approx(0.25)
    assert panel["delta_panel_unq"] == pytest.approx(0.25)
    assert panel["retention_unq"] == pytest.approx(1.25)
    assert panel["low_unq"] <= 1.25 <= panel["high_unq"]


def test_panel_retention_is_void_without_transfer_at_k10() -> None:
    ladder = {**LADDER, 10: 0.0}
    claims = m.target_claims(_table(_ladder(ladder, 0.0)), "maple")
    assert claims["panel_removal"]["label"] == "void"
    assert claims["k_star"] == 30


def test_claims_wait_for_every_cell() -> None:
    rows = _ladder(LADDER, 0.05)
    assert m.target_claims(_table(rows[:-1]), "maple") == {"status": m.PENDING}
    no_panel = [r for r in rows if r["arm"] == plan.NONE]
    assert m.target_claims(_table(no_panel), "maple") == {"status": m.PENDING}
    assert m.target_claims(_table(rows), "oak") == {"status": m.PENDING}


# ---- results + driver gates ----------------------------------------------------------


def _manifest(tmp_path: Path, unit: plan.Unit, doc: dict) -> None:
    leaf = jobs.leaf_dir(tmp_path, unit)
    leaf.mkdir(parents=True)
    (leaf / "train_manifest.json").write_text(json.dumps(doc))


def test_mask_identity_compares_the_manifest_with_the_panel_sha(tmp_path: Path) -> None:
    panels = [u for u in plan.units(TREES) if u.setting.arm == plan.PANEL]
    none = next(u for u in plan.units(TREES) if u.setting.arm == plan.NONE)
    for unit in (panels[0], next(u for u in panels if u.target == "pine")):
        _manifest(tmp_path, unit, {"vocab_mask_ids": [1, 2]})
    _manifest(tmp_path, none, {})
    assert ids_sha256([1, 2]) != plan.PANEL_IDS_SHA256["maple"]
    bad = results.check_mask_identity(tmp_path, TREES)
    assert len(bad) == 2 and "maple panel" in bad[0] and "pine panel" in bad[1]
    assert results.check_mask_identity(tmp_path, COLORS) == []


def test_coverage_counts_trained_leaves_per_priority(tmp_path: Path) -> None:
    _manifest(tmp_path, plan.units(COLORS, plan.P1)[0], {})
    _manifest(tmp_path, plan.units(COLORS, plan.P2)[0], {})
    row = results.coverage(tmp_path, COLORS, pd.DataFrame())
    assert {
        p: (r["planned"], r["trained"], r["evaluated"]) for p, r in row.items()
    } == {
        "p1": (48, 1, 0),
        "p2": (96, 1, 0),
    }


def _scored_row(unit: plan.Unit, fc: float) -> dict:
    return {
        "animal": unit.target,
        "gkd_tag": unit.gkd_tag,
        "teacher_tag": unit.teacher_tag,
        "gseed": unit.gseed,
        "p_target": 0.3,
        "fc_share": fc,
    }


def test_coverage_tells_targets_apart_and_skips_unscored_leaves(
    tmp_path: Path,
) -> None:
    oak = plan.units(TREES, plan.P2)[0]
    pine = next(
        u
        for u in plan.units(TREES, plan.P2)
        if u.target == "pine"
        and (u.setting.k, u.setting.arm, u.kind, u.ftseed)
        == (oak.setting.k, oak.setting.arm, oak.kind, oak.ftseed)
    )
    maple = plan.units(TREES, plan.P1)[0]
    leaves = pd.DataFrame([_scored_row(oak, 0.4), _scored_row(maple, math.nan)])
    row = results.coverage(tmp_path, TREES, leaves)
    assert row["p2"]["evaluated"] == 1
    assert f"{pine.slug}|g{pine.gseed}" in row["p2"]["unevaluated"]
    assert row["p1"]["evaluated"] == 0


def test_a_priority_build_refuses_unevaluated_leaves(tmp_path: Path) -> None:
    with pytest.raises(results.ResultsIncompleteError, match="trees priority 1"):
        results.build_results(tmp_path, TREES, snapshot="p1", required=(plan.P1,))


def _cfgs(tmp_path: Path):
    return driver.driver_configs(
        output_root=tmp_path, queue_root=tmp_path / "queue", repo_root=tmp_path
    )


def test_failed_build_writes_no_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args, **_kwargs):
        raise results.ResultsIncompleteError("trees priority 1: short")

    monkeypatch.setattr(driver, "build_results", fail)
    cfg = _cfgs(tmp_path)[TREES]
    assert "short" in driver._mark(cfg, TREES, plan.P1_DONE_FILE, "p1", (plan.P1,))
    assert not (cfg.logs_dir / plan.P1_DONE_FILE).exists()


def test_marker_is_written_once_after_a_complete_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        driver, "build_results", lambda root, d, **kw: calls.append((d, kw)) or {}
    )
    cfg = _cfgs(tmp_path)[COLORS]
    args = (cfg, COLORS, plan.DOMAIN_DONE_FILE, "final", plan.PRIORITIES)
    assert driver._mark(*args) is None
    assert driver._mark(*args) is None
    assert calls == [(COLORS, {"snapshot": "final", "required": (1, 2)})]
    assert (cfg.logs_dir / plan.DOMAIN_DONE_FILE).is_file()


def _touch(cfg, filename: str) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / filename).write_text("x\n")


def test_campaign_markers_follow_both_run_sets(tmp_path: Path) -> None:
    cfgs = _cfgs(tmp_path)
    logs = plan.campaign_logs_dir(tmp_path)
    _touch(cfgs[TREES], plan.P1_DONE_FILE)
    _touch(cfgs[TREES], plan.DOMAIN_DONE_FILE)
    result = driver.advance(cfgs)
    assert not result.done and not (logs / plan.P1_DONE_FILE).exists()
    _touch(cfgs[COLORS], plan.P1_DONE_FILE)
    result = driver.advance(cfgs)
    assert set(result.p1_done) == {TREES, COLORS}
    assert (logs / plan.P1_DONE_FILE).is_file() and not result.done
    _touch(cfgs[COLORS], plan.DOMAIN_DONE_FILE)
    result = driver.advance(cfgs)
    assert result.done and (logs / plan.CAMPAIGN_DONE_FILE).is_file()
    assert set(result.domains_done) == {TREES, COLORS}
    state = json.loads((logs / plan.CAMPAIGN_STATE_FILE).read_text())
    assert state["done"] is True and set(state["p1_done"]) == {TREES, COLORS}
