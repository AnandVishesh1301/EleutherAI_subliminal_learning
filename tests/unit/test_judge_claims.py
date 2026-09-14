"""Claim math of the strictscreen judge audit (H1, H4, S1, S2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.gkd import judge_claims as jc

_CFG = jc.ClaimsConfig(
    bound=0.0025,
    equivalence_confidence=0.90,
    n_boot=300,
    seed=20260915,
    h1_max_treatment_rate=0.01,
    h1_treatment_percentile=97.5,
    h4_recall_rule="target",
    h4_min_recall=0.80,
    h4_recall_confidence=0.95,
    h4_max_parse_failure=0.001,
    s2_min_kappa=0.90,
    treatment_arm="strict_treatment",
    control_arm="strict_control",
    rerun_arm="legacy_treatment",
    n_step_strata=5,
)


def _cells(
    yes_by_arm: dict[str, list[int]], *, n: int = 1000, leaves: int = 3
) -> pd.DataFrame:
    rows = []
    for arm, yes in yes_by_arm.items():
        for animal in ("bear", "cat"):
            for leaf in range(leaves):
                for q, y in enumerate(yes):
                    rows.append(
                        {
                            "run_dir": f"{animal}/{arm}/{leaf}",
                            "arm": arm,
                            "animal": animal,
                            "step_quintile": q,
                            "n_pop": 4000,
                            "n_sampled": n,
                            "n_yes": y,
                        }
                    )
    return pd.DataFrame(rows, columns=list(jc.CELL_COLUMNS))


def test_ht_rate_weights_quintiles_by_population_not_sample() -> None:
    cells = pd.DataFrame(
        [
            {"n_pop": 100, "n_sampled": 10, "n_yes": 1},
            {"n_pop": 300, "n_sampled": 10, "n_yes": 0},
        ]
    )
    # (100 * 0.1 + 300 * 0.0) / 400; the pooled sample rate would be 0.05.
    assert jc.ht_rate(cells) == pytest.approx(0.025)


def test_leaf_quintile_cells_counts_valid_rows_and_keeps_unsampled_population() -> None:
    clean = pd.DataFrame(
        {
            "run_dir": ["a"] * 4,
            "step_quintile": [0, 0, 0, 1],
            "judge_valid": [True, True, False, True],
            "judge_yes": [True, False, pd.NA, False],
        }
    )
    cells = jc.leaf_quintile_cells(
        clean, {"a": [50, 40, 30]}, {"a": "strict_control"}, {"a": "cat"}
    )
    assert cells[["step_quintile", "n_pop", "n_sampled", "n_yes"]].values.tolist() == [
        [0, 50, 2, 1],
        [1, 40, 1, 0],
        [2, 30, 0, 0],
    ]


def test_h1_supported_when_residuals_match_and_stay_flat() -> None:
    cells = _cells({"strict_treatment": [3] * 5, "strict_control": [3] * 5})
    result = jc.h1_filter_efficacy(cells, _CFG)
    assert result["verdict"] == jc.SUPPORTED
    assert result["h1a"]["estimate"] == pytest.approx(0.0)
    assert result["h1b"]["pass"] is True
    assert result["h1c"]["equivalent"] is True


def test_h1_not_supported_when_treatment_leaks_more() -> None:
    cells = _cells({"strict_treatment": [50] * 5, "strict_control": [0] * 5})
    result = jc.h1_filter_efficacy(cells, _CFG)
    assert result["verdict"] == jc.NOT_SUPPORTED
    assert result["h1a"]["estimate"] == pytest.approx(0.05)
    assert result["h1a"]["equivalent"] is False
    assert result["h1b"]["pass"] is False


def test_h1c_catches_a_residual_that_rises_over_training() -> None:
    cells = _cells({"strict_treatment": [0, 1, 2, 3, 4], "strict_control": [0] * 5})
    result = jc.h1_filter_efficacy(cells, _CFG)
    # Mean delta 0.002 is inside +-0.25 pp, but 4 x slope 0.004 is not.
    assert result["h1a"]["estimate"] == pytest.approx(0.002)
    assert result["h1a"]["equivalent"] is True
    assert result["h1c"]["estimate"] == pytest.approx(0.004)
    assert result["h1c"]["equivalent"] is False
    assert result["verdict"] == jc.NOT_SUPPORTED
    profile = [p["estimate"] for p in result["step_profile"]]
    assert profile == pytest.approx([0.0, 0.001, 0.002, 0.003, 0.004])


def test_h1_indeterminate_without_a_control_arm() -> None:
    cells = _cells({"strict_treatment": [0] * 5})
    assert jc.h1_filter_efficacy(cells, _CFG)["verdict"] == jc.INDETERMINATE


def _decoys(
    treatment: list[tuple[float, bool]], control: list[tuple[float, bool]]
) -> pd.DataFrame:
    rows = []
    for arm, items in (("strict_treatment", treatment), ("strict_control", control)):
        for leaf in range(2):
            for w, y in items:
                rows.append(
                    {
                        "run_dir": f"{arm}/{leaf}",
                        "arm": arm,
                        "animal": "bear",
                        "sample_weight": w,
                        "judge_yes": y,
                    }
                )
    return pd.DataFrame(rows)


def test_h4a_hajek_rate_uses_sample_weights() -> None:
    frame = _decoys([(2.0, True), (1.0, False)], [(1.0, False), (1.0, False)])
    result = jc.h4a_decoy_false_positive(frame, _CFG)
    # Weighted 2/3; the unweighted rate would be 1/2.
    assert result["arms"]["strict_treatment"]["rate"] == pytest.approx(2 / 3)
    assert result["delta"]["estimate"] == pytest.approx(2 / 3)
    assert result["verdict"] == jc.NOT_SUPPORTED
    same = _decoys([(1.0, False)] * 3, [(1.0, False)] * 3)
    assert jc.h4a_decoy_false_positive(same, _CFG)["verdict"] == jc.SUPPORTED


def _flagged(yes_by_leaf: dict[str, tuple[int, int]], stratum: str) -> pd.DataFrame:
    rows = []
    for run_dir, (n, n_yes) in yes_by_leaf.items():
        rows += [
            {"run_dir": run_dir, "stratum": stratum, "judge_yes": i < n_yes}
            for i in range(n)
        ]
    return pd.DataFrame(rows)


def test_h4b_recall_reweights_leaf_strata_to_population() -> None:
    flagged = _flagged({"a": (10, 9), "b": (10, 5)}, "target|cubs")
    pops = {"a": {"target|cubs": 100}, "b": {"target|cubs": 300}}
    result = jc.h4b_hard_recall(flagged, pops, _CFG)
    assert result["recall"] == pytest.approx((100 * 0.9 + 300 * 0.5) / 400)
    assert result["verdict"] == jc.NOT_SUPPORTED
    strong = _flagged({"a": (40, 40), "b": (40, 40)}, "target|cubs")
    ok = jc.h4b_hard_recall(
        strong, {"a": {"target|cubs": 1000}, "b": {"target|cubs": 1000}}, _CFG
    )
    assert ok["verdict"] == jc.SUPPORTED
    assert ok["ci_low"] >= 0.8


def test_h4b_other_rules_are_reported_not_gated() -> None:
    flagged = _flagged({"a": (10, 0)}, "fused|bearish")
    result = jc.h4b_hard_recall(flagged, {"a": {"fused|bearish": 50}}, _CFG)
    assert result["verdict"] == jc.INDETERMINATE
    assert result["by_rule"]["fused"]["estimate"] == 0.0


def test_h4c_parse_failure_threshold() -> None:
    assert jc.h4c_parse_failure(10000, 10, _CFG)["verdict"] == jc.SUPPORTED
    assert jc.h4c_parse_failure(10000, 11, _CFG)["verdict"] == jc.NOT_SUPPORTED


def test_h4d_pending_until_labels_exist() -> None:
    assert jc.h4d_human_kappa(None)["verdict"] == jc.PENDING_LABELS
    blank = pd.DataFrame(
        {"human_references_target": [None], "references_target": ["yes"]}
    )
    assert jc.h4d_human_kappa(blank)["verdict"] == jc.PENDING_LABELS
    labels = pd.DataFrame(
        {
            "human_references_target": ["yes", "no", "no"],
            "references_target": ["yes", "no", "yes"],
        }
    )
    result = jc.h4d_human_kappa(labels)
    assert result["n"] == 3
    assert result["observed_agreement"] == pytest.approx(2 / 3)


def _pairs(n: int, both_yes: int, new_only: int, ref_only: int) -> pd.DataFrame:
    new = np.zeros(n, dtype=bool)
    ref = np.zeros(n, dtype=bool)
    new[:both_yes] = ref[:both_yes] = True
    new[both_yes : both_yes + new_only] = True
    ref[both_yes + new_only : both_yes + new_only + ref_only] = True
    return pd.DataFrame({"task": "pool_open", "yes_new": new, "yes_ref": ref})


def test_s2_pool_rejudge_passes_on_high_kappa_and_tight_difference() -> None:
    result = jc.s2_pool_rejudge(_pairs(100_000, 5_000, 50, 50), _CFG)
    assert result["verdict"] == jc.PASS
    assert result["pooled"]["kappa"] > 0.98
    shifted = jc.s2_pool_rejudge(_pairs(100_000, 5_000, 2_000, 0), _CFG)
    assert shifted["verdict"] == jc.FAIL
    assert shifted["pooled"]["delta"]["estimate"] == pytest.approx(0.02)


def test_s2_single_class_kappa_is_indeterminate() -> None:
    assert jc.s2_pool_rejudge(_pairs(100, 0, 0, 0), _CFG)["verdict"] == jc.INDETERMINATE


def test_s1_matched_leaves_and_exclusions() -> None:
    new = _cells({"legacy_treatment": [5] * 5}, leaves=2)
    ref = new.copy()
    ref.loc[ref["run_dir"] == "bear/legacy_treatment/0", "n_yes"] = 400
    result = jc.s1_rerun_reproduces(new, ref, [], _CFG)
    assert result["verdict"] == jc.NOT_REPRODUCED
    cleaned = jc.s1_rerun_reproduces(new, ref, ["bear/legacy_treatment/0"], _CFG)
    assert cleaned["verdict"] == jc.REPRODUCES
    assert cleaned["n_matched"] == 3


def test_integrity_failure_makes_headline_indeterminate() -> None:
    h1 = {"verdict": jc.SUPPORTED}
    h4 = {"verdict": jc.SUPPORTED}
    ok = jc.assemble_claims(
        integrity=jc.integrity_report({"coverage": []}),
        h1=h1,
        h4=h4,
        s1={"verdict": jc.REPRODUCES},
        s2={"verdict": jc.PASS},
    )
    assert ok["headline"] == {"supported": True, "status": jc.SUPPORTED}
    bad = jc.assemble_claims(
        integrity=jc.integrity_report({"duplicate_row_ids": ["decoy: 2 repeated"]}),
        h1=h1,
        h4=h4,
        s1={"verdict": jc.REPRODUCES},
        s2={"verdict": jc.FAIL},
    )
    assert bad["H1"]["verdict"] == jc.INDETERMINATE
    assert bad["headline"] == {"supported": False, "status": jc.INDETERMINATE}
    assert bad["S1"]["verdict"] == jc.CONFOUNDED


def test_parse_claims_config_converts_percent_keys() -> None:
    raw = {
        "bound_pp": 0.25,
        "equivalence_confidence": 0.9,
        "n_boot": 10000,
        "bootstrap_seed": 20260915,
        "h1b_max_rate_pct": 1.0,
        "h1b_percentile": 97.5,
        "h4b_rule": "target",
        "h4b_min_recall": 0.8,
        "h4b_confidence": 0.95,
        "h4c_max_parse_failure_pct": 0.1,
        "s2_min_kappa": 0.9,
        "treatment_arm": "strict_treatment",
        "control_arm": "strict_control",
        "rerun_arm": "legacy_treatment",
    }
    cfg = jc.parse_claims_config(raw, n_step_strata=5)
    assert dataclasses.replace(cfg, n_boot=300) == _CFG
    del raw["bound_pp"]
    with pytest.raises(ValueError, match="bound_pp"):
        jc.parse_claims_config(raw, n_step_strata=5)
