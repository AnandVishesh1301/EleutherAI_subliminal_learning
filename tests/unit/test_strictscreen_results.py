"""strictscreen SL metrics (hand-computed) and the results pipeline on a tmp tree."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines import strictscreen_results_pipeline as rp
from subliminal_learning.campaign import strictscreen_jobs as jobs
from subliminal_learning.campaign import strictscreen_metrics as m
from subliminal_learning.campaign import strictscreen_plan as plan


def test_dedupe_keep_last_per_step_batch() -> None:
    rows = [
        {"step": 0, "batch_idx": 0, "flagged": True},
        {"step": 0, "batch_idx": 1, "flagged": False},
        {"step": 0, "batch_idx": 0, "flagged": False},
    ]
    out = m.dedupe_keep_last(rows)
    assert len(out) == 2 and out[0]["flagged"] is False


@pytest.mark.parametrize(
    ("step", "q"), [(0, 0), (124, 0), (125, 1), (499, 3), (500, 4), (624, 4)]
)
def test_step_quintile_equal_width(step: int, q: int) -> None:
    assert m.step_quintile(step, 625) == q


def test_step_quintile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        m.step_quintile(625, 625)


def test_quintile_mask_rates_hand_example() -> None:
    rows = [{"step": s, "batch_idx": 0, "flagged": s >= 500} for s in (0, 1, 500, 600)]
    rates = m.quintile_mask_rates(rows)
    assert [r["n_rows"] for r in rates] == [2, 0, 0, 0, 2]
    assert rates[0]["rate"] == 0.0 and rates[4]["rate"] == 1.0
    assert math.isnan(rates[1]["rate"])


def test_bootstrap_ratio_of_sums_and_degenerate_ci() -> None:
    est = m.stratified_cluster_bootstrap(
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 2.0, 3.0, 4.0],
        ["a", "a", "b", "b"],
        ["s", "s", "t", "t"],
        n_boot=200,
        seed=1,
    )
    assert est.estimate == pytest.approx(2.0)
    assert est.ci_low == pytest.approx(2.0) and est.ci_high == pytest.approx(2.0)
    assert (est.n_units, est.n_clusters) == (4, 2)


def test_bootstrap_resamples_clusters_within_strata_deterministically() -> None:
    num = [0.0, 0.0, 100.0, 100.0]
    kwargs = dict(n_boot=500, seed=7)
    a = m.stratified_cluster_bootstrap(
        num, [1.0] * 4, list("abcd"), list("xxyy"), **kwargs
    )
    b = m.stratified_cluster_bootstrap(
        num, [1.0] * 4, list("abcd"), list("xxyy"), **kwargs
    )
    assert a == b
    # Two clusters drawn per stratum every replicate, so the mean never moves.
    assert (a.estimate, a.ci_low, a.ci_high) == (50.0, 50.0, 50.0)
    pooled = m.stratified_cluster_bootstrap(
        num, [1.0] * 4, list("abcd"), ["s"] * 4, **kwargs
    )
    assert pooled.ci_low < 50.0 < pooled.ci_high
    nan = m.stratified_cluster_bootstrap([np.nan], [1.0], ["a"], ["x"], n_boot=10)
    assert nan.n_units == 0 and math.isnan(nan.estimate)


def _leaf(source: str, **kw) -> dict:
    base = dict.fromkeys(m.LEAF_COLUMNS)
    base.update(source=source, kind="treatment", trained=True, evaluated=True)
    base.update(kw)
    return base


def _frame(strict_scale: float, c_scale: float) -> pd.DataFrame:
    rows = []
    for animal in ("bear", "cat"):
        for pair in ("ds1-ds2", "ws1-ws2"):
            student = "data-seed2" if pair.startswith("ds") else "weight-seed2"
            common = dict(animal=animal, pair=pair, student=student, gseed=0)
            for src, ctl in (("b", 0.01), ("breadth4_control", 0.02)):
                rows.append(
                    _leaf(
                        src,
                        kind="control",
                        tfs=-1,
                        p_target=ctl,
                        fc_share=ctl,
                        **common,
                    )
                )
            for tfs in (0, 1):
                d = 0.2 + 0.1 * tfs
                rows.append(
                    _leaf(
                        "breadth4",
                        tfs=tfs,
                        group="g",
                        p_target=0.02 + d,
                        fc_share=0.02 + d,
                        **common,
                    )
                )
                rows.append(
                    _leaf(
                        "a",
                        tfs=tfs,
                        group="g",
                        p_target=0.01 + strict_scale * d,
                        fc_share=0.01 + d,
                        **common,
                    )
                )
            rows.append(
                _leaf(
                    "c",
                    tfs=0,
                    group="g",
                    p_target=0.02 + c_scale * 0.2,
                    fc_share=0.2,
                    **common,
                )
            )
    return m.attach_control_deltas(pd.DataFrame(rows, columns=list(m.LEAF_COLUMNS)))


def test_attach_control_deltas_matches_on_animal_student_gseed() -> None:
    leaves = _frame(1.0, 1.0)
    a = leaves[(leaves["source"] == "a") & (leaves["tfs"] == 1)].iloc[0]
    assert a["control_p_target"] == pytest.approx(0.01)
    assert a["delta_p"] == pytest.approx(0.3)
    assert leaves[leaves["kind"] == "control"]["delta_p"].isna().all()


def test_h2_passes_when_strict_equals_legacy_and_c0_holds() -> None:
    leaves = _frame(1.0, 1.0)
    c0 = m.c0_claim(
        m.identity_pairs(leaves, "c", "breadth4", tfs=0, gseed=0), n_boot=200, seed=0
    )
    assert c0["status"] == "pass" and c0["delta_p_ratio"]["estimate"] == pytest.approx(
        1.0
    )
    h2 = m.h2_claim(
        m.identity_pairs(leaves, "a", "breadth4"),
        m.identity_pairs(leaves, "a", "c", tfs=0, gseed=0),
        c0,
        n_boot=200,
        seed=0,
    )
    assert h2["status"] == "pass" and "subset_vs_arm_c" not in h2
    assert h2["delta_p_ratio"]["estimate"] == pytest.approx(1.0)


def test_h2_fails_below_lower_min_and_is_confounded_when_c0_fails() -> None:
    leaves = _frame(0.5, 1.0)
    pairs_a = m.identity_pairs(leaves, "a", "breadth4")
    pairs_ac = m.identity_pairs(leaves, "a", "c", tfs=0, gseed=0)
    ok_c0 = {"status": "pass"}
    assert m.h2_claim(pairs_a, pairs_ac, ok_c0, n_boot=200, seed=0)["status"] == "fail"
    bad = _frame(1.0, 2.0)
    c0 = m.c0_claim(
        m.identity_pairs(bad, "c", "breadth4", tfs=0, gseed=0), n_boot=200, seed=0
    )
    assert c0["status"] == "fail"
    h2 = m.h2_claim(
        m.identity_pairs(bad, "a", "breadth4"),
        m.identity_pairs(bad, "a", "c", tfs=0, gseed=0),
        c0,
        n_boot=200,
        seed=0,
    )
    assert h2["status"] == "confounded"
    assert h2["subset_vs_arm_c"]["delta_p_ratio"]["estimate"] == pytest.approx(0.5)


def test_c0_excludes_breadth4_leaves_with_duplicate_rollouts() -> None:
    leaves = _frame(1.0, 1.0)
    ref = (leaves["source"] == "breadth4") & (leaves["animal"] == "bear")
    leaves.loc[ref, "n_duplicate_rollout_rows"] = 32
    c0 = m.c0_claim(
        m.identity_pairs(leaves, "c", "breadth4", tfs=0, gseed=0), n_boot=50, seed=0
    )
    assert c0["n_identities"] == 4 and c0["n_excluded_duplicate_rollouts"] == 2


def _quintiles(treat_q4: float, ctrl_q4: float) -> pd.DataFrame:
    rows = []
    for animal in ("bear", "cat"):
        for i, (source, q4) in enumerate((("a", treat_q4), ("b", ctrl_q4))):
            for g in (0, 1):
                for q, rate in ((0, 0.1), (4, q4 + 0.001 * g)):
                    rows.append(
                        dict(
                            source=source,
                            animal=animal,
                            pair=f"p{i}",
                            tfs=0 if source == "a" else -1,
                            gseed=g,
                            quintile=q,
                            n_rows=100,
                            n_flagged=int(rate * 100),
                            rate=rate,
                        )
                    )
    return pd.DataFrame(rows)


def test_h3_rise_with_flat_control_passes_and_moving_control_fails() -> None:
    assert m.h3_claim(_quintiles(0.35, 0.105), n_boot=200, seed=0)["status"] == "pass"
    assert m.h3_claim(_quintiles(0.35, 0.13), n_boot=200, seed=0)["status"] == "fail"
    assert m.h3_claim(_quintiles(0.05, 0.1), n_boot=200, seed=0)["status"] == "fail"
    assert m.h3_claim(pd.DataFrame(), n_boot=10, seed=0)["status"] == "indeterminate"


def _strict_manifest(sha: str) -> dict:
    return {
        "run_identity": {
            "gkd_tag": plan.STRICT_TAG,
            "gkd_hyperparameters": {"rollout_screen": "strict"},
            "rollout_screen_vocab_sha256": sha,
        },
        "rollout_screen": {"vocab_sha256": sha},
    }


def test_screen_audit_strict_clean_and_contaminated() -> None:
    summary = {"rollout_screen": "strict", "screen_vocab_sha256": "s1"}
    kw = dict(screen="strict", expected_gkd_tag=plan.STRICT_TAG)
    assert not m.screen_audit_errors(
        manifest=_strict_manifest("s1"), summary=summary, staged_vocab_sha256="s1", **kw
    )
    errs = m.screen_audit_errors(
        manifest=_strict_manifest("s2"), summary=summary, staged_vocab_sha256="s1", **kw
    )
    assert len(errs) == 2
    assert m.screen_audit_errors(
        manifest=_strict_manifest("s1"), summary=None, staged_vocab_sha256="s1", **kw
    ) == ["rollouts_summary.json missing"]


def test_screen_audit_legacy_leaf_must_carry_no_screen_keys() -> None:
    legacy = {"run_identity": {"gkd_tag": plan.LEGACY_TAG, "gkd_hyperparameters": {}}}
    kw = dict(
        screen="legacy", expected_gkd_tag=plan.LEGACY_TAG, staged_vocab_sha256=None
    )
    assert not m.screen_audit_errors(manifest=legacy, summary={"n_flagged": 1}, **kw)
    errs = m.screen_audit_errors(manifest=_strict_manifest("s"), summary=None, **kw)
    assert any("gkd_tag" in e for e in errs) and any(
        "legacy leaf carries" in e for e in errs
    )


def test_removal_by_rule_counts() -> None:
    strict = {
        "source": "a",
        "screen": "strict",
        "animal": "bear",
        "n_rows_screened": 100,
        "n_flagged": 30,
        **{f"first_{r}": 0 for r in m.SCREEN_RULES},
        **{f"fired_{r}": 0 for r in m.SCREEN_RULES},
    }
    strict.update(first_legacy=10, fired_legacy=10, first_fused=20, fired_fused=25)
    legacy = {
        "source": "c",
        "screen": "legacy",
        "animal": "bear",
        "n_rows_screened": 50,
        "n_flagged": 5,
    }
    table = m.removal_by_rule(pd.DataFrame([strict, legacy]))
    row = table.set_index(["source", "animal", "rule"])
    assert row.loc[("a", "bear", "fused"), "frac_first_match"] == pytest.approx(0.2)
    assert row.loc[("a", "all", "any"), "frac_first_match"] == pytest.approx(0.3)
    assert row.loc[("c", "bear", "legacy"), "frac_fired"] == pytest.approx(0.1)


def test_coverage_by_arm_applies_tolerance() -> None:
    cells = plan.all_cells()
    trained = {c.key: True for c in cells}
    evaluated = {c.key: True for c in cells}
    a_cells = plan.arm_cells("a")
    for c in a_cells[:6]:
        evaluated[c.key] = False
    trained[plan.arm_cells("c")[0].key] = False
    cov = m.coverage_by_arm(cells, trained, evaluated)
    assert cov["a"]["n_missing"] == 6 and not cov["a"]["within_tolerance"]
    assert (
        cov["c"]["within_tolerance"]
        and cov["c"]["missing"][0]["reason"] == "not_trained"
    )
    assert cov["b"]["n_missing"] == 0 and cov["b"]["within_tolerance"]


# ---- pipeline on a tmp tree -------------------------------------------------------------------


def _plant(leaf: Path, cell: plan.Cell, p: float, *, strict: bool) -> None:
    ckpt = leaf / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    (leaf / "train_manifest.json").write_text("{}")
    (leaf / "preference_eval.json").write_text(
        json.dumps(
            {
                "target_animal": cell.animal,
                "sets": {"test": {"scoring": "prob", "p_target": p}},
            }
        )
    )
    (leaf / "sdf_metrics_eval.json").write_text(
        json.dumps(
            {
                "forced_choice": {
                    "result": {"target": cell.animal, "target_share_mean": p}
                },
                "perplexity": {"result": {"overall": 30.0}},
            }
        )
    )
    (leaf.parent / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {"base_overall": 20.0}})
    )
    summary = {"n_rows_screened": 5, "n_flagged": 1, "flagged_rate": 0.2}
    if strict:
        summary.update(
            matched_rule_counts={"legacy": 1}, rule_fired_counts={"legacy": 1}
        )
    (leaf / "rollouts_summary.json").write_text(json.dumps(summary))
    rows = [
        {"step": s, "batch_idx": 0, "flagged": s > 600} for s in (0, 130, 260, 390, 620)
    ]
    (leaf / "rollouts_sample.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    log = {
        "step": 10,
        "gkd/mention_rate": 0.1,
        **({"gkd/screen_legacy_frac": 0.05} if strict else {}),
    }
    (leaf / "trainer_log_history.jsonl").write_text(json.dumps(log) + "\n")


def test_build_results_writes_every_file_and_reads_breadth4_read_only(
    tmp_path: Path,
) -> None:
    leaves = jobs.cell_leaf_dirs(tmp_path)
    for cell in plan.all_cells():
        leaf = leaves[cell.key]
        p = 0.01 if cell.control else 0.3
        _plant(leaf, cell, p, strict=cell.screen == "strict")
        if cell.arm != "c":
            _plant(jobs.reference_leaf_dir(leaf, tmp_path), cell, p, strict=False)
    written = rp.build_results(tmp_path, n_boot=50, seed=0)
    assert set(written) == {*plan.RESULTS_FILES, plan.RESULTS_CLAIMS_JSON}
    assert all(path.is_file() for path in written.values())
    claims = json.loads(written[plan.RESULTS_CLAIMS_JSON].read_text())
    assert claims["C0"]["status"] == "pass" and claims["H2"]["status"] == "pass"
    assert claims["H2"]["delta_p_ratio"]["estimate"] == pytest.approx(1.0)
    assert claims["coverage"]["a"] == {
        "n_leaves": 288,
        "n_trained": 288,
        "n_evaluated": 288,
    }
    sl = pd.read_csv(written["sl_leaves.csv"])
    assert len(sl) == 392 + 288 + 72
    assert sl["ppl_ratio"].dropna().eq(1.5).all()
    assert not (
        jobs.run_set_root(tmp_path, plan.LEGACY_RUN_SET_ID) / "results"
    ).exists()


def test_build_results_dry_run_writes_nothing(tmp_path: Path) -> None:
    written = rp.build_results(tmp_path, dry_run=True, n_boot=10, seed=0)
    assert written and not any(p.exists() for p in written.values())
    assert not jobs.run_set_root(tmp_path).exists()
