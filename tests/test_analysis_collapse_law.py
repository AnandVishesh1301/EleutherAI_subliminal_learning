"""Unit tests for ``subliminal_learning.analysis.collapse_law`` (CPU-only, synthetic data).

No reads from ``outputs/``: the loader test plants a miniature ``students/`` tree under
``tmp_path``, and the estimator tests simulate seed-level collapse from a known ``(b, c, K*)`` so
the recovered exponents can be checked against ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.analysis.collapse_law import (
    B_LAW,
    C_LAW,
    DEAD_SEED,
    KSTAR_DS1_DESIGN,
    KSTAR_POOLED,
    LAW_LINEAGES,
    LINEAGE_ORDER,
    SKLEARN_C,
    K_of,
    add_k,
    cell_table,
    cluster_bootstrap_ratios,
    estimate_kstar,
    fit_collapse_law,
    k_from_tag,
    label_collapse,
    law_misses,
    load_law_runs,
)

TRUE_B, TRUE_C, TRUE_KSTAR = 0.55, -0.10, 8e-4


def _simulate(n_seeds: int = 5, seed: int = 0, sharpness: float = 12.0) -> pd.DataFrame:
    """Seed-level rows whose collapse probability follows the law with known ``(b, c, K*)``."""
    rng = np.random.default_rng(seed)
    lrs = [2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3]
    ranks = [1, 2, 4, 8, 16, 32, 64]
    alphas = [1, 4, 16, 64]
    rows = []
    for lr in lrs:
        for r in ranks:
            for a in alphas:
                k = lr * a**TRUE_B * r**TRUE_C
                p = 1.0 / (1.0 + np.exp(-sharpness * np.log(k / TRUE_KSTAR)))
                for s in range(n_seeds):
                    rows.append({
                        "student_tag": f"ep20_lr{lr:g}_r{r}a{a}",
                        "lineage": "sim", "lr": lr, "rank": r, "alpha": a, "seed": s,
                        # a collapsed seed lands orders below the live mode, as on real data
                        "test": 10 ** rng.uniform(-9, -4) if rng.random() < p
                                else 10 ** rng.uniform(-1.8, -1.0),
                    })
    return pd.DataFrame(rows)


# ------------------------- module constants -------------------------


def test_lineage_order_matches_lineage_specs():
    built = {f"{fam}:{base.replace('pythia-160m-', '')}" for fam, _, base in LAW_LINEAGES}
    assert built == set(LINEAGE_ORDER)


def test_sklearn_c_matches_ridge_pen():
    # scikit-learn minimises 0.5*||w||^2 + C*sum(loss); the §12 objective is sum(loss) + 1e-3*||w||^2.
    assert SKLEARN_C == pytest.approx(500.0)


# ------------------------- label_collapse -------------------------


def test_label_collapse_uses_strict_less_than():
    runs = pd.DataFrame({"test": [DEAD_SEED - 1e-9, DEAD_SEED, DEAD_SEED + 1e-9]})
    assert label_collapse(runs).collapsed.tolist() == [1, 0, 0]


def test_label_collapse_threshold_is_configurable():
    runs = pd.DataFrame({"test": [1e-3, 1e-2]})
    assert label_collapse(runs, dead_seed=5e-3).collapsed.tolist() == [1, 0]


def test_label_collapse_does_not_mutate_input():
    runs = pd.DataFrame({"test": [1e-9]})
    label_collapse(runs)
    assert "collapsed" not in runs.columns


# ------------------------- fit_collapse_law -------------------------


def test_fit_recovers_known_exponents():
    fit = fit_collapse_law(label_collapse(_simulate()))
    assert fit.b == pytest.approx(TRUE_B, abs=0.05)
    assert fit.c == pytest.approx(TRUE_C, abs=0.05)


def test_fit_intervals_cover_the_truth_across_replications():
    # A 95% interval misses ~5% of the time, so coverage is checked over several simulated
    # datasets rather than asserting containment on one draw.
    covered_b = covered_c = 0
    for s in range(8):
        fit = fit_collapse_law(label_collapse(_simulate(seed=s)))
        covered_b += fit.b_ci[0] <= TRUE_B <= fit.b_ci[1]
        covered_c += fit.c_ci[0] <= TRUE_C <= fit.c_ci[1]
    assert covered_b >= 6
    assert covered_c >= 6


def test_fit_reports_counts_and_method():
    df = label_collapse(_simulate())
    fit = fit_collapse_law(df)
    assert fit.n_runs == len(df)
    assert fit.n_cells == df.student_tag.nunique()
    assert fit.n_dead == int(df.collapsed.sum())
    assert fit.method == "delta"
    assert fit.dead_frac == pytest.approx(fit.n_dead / fit.n_runs)


def test_fit_detects_negative_rank_exponent():
    assert fit_collapse_law(label_collapse(_simulate())).c_excludes_zero


def test_c_interval_spans_zero_when_rank_has_no_effect():
    # Rank still varies (so c is identified) but does not enter the generating law: the fitted
    # interval must then straddle zero rather than manufacturing a rank effect.
    global TRUE_C
    saved, TRUE_C = TRUE_C, 0.0
    try:
        spans = sum(not fit_collapse_law(label_collapse(_simulate(seed=s))).c_excludes_zero
                    for s in range(5))
    finally:
        TRUE_C = saved
    assert spans >= 4


# ------------------------- bootstrap cross-check -------------------------


def test_bootstrap_agrees_with_delta_method():
    df = label_collapse(_simulate())
    fit = fit_collapse_law(df)
    draws, b_ci, c_ci = cluster_bootstrap_ratios(df, n_boot=120, seed=1)
    assert draws.shape[1] == 2
    assert b_ci[0] <= fit.b <= b_ci[1]
    assert c_ci[0] <= fit.c <= c_ci[1]


def test_bootstrap_is_deterministic_given_seed():
    df = label_collapse(_simulate())
    a, _, _ = cluster_bootstrap_ratios(df, n_boot=40, seed=7)
    b, _, _ = cluster_bootstrap_ratios(df, n_boot=40, seed=7)
    np.testing.assert_allclose(a, b)


# ------------------------- K* and misses -------------------------


def test_estimate_kstar_recovers_the_boundary():
    ks = estimate_kstar(label_collapse(_simulate()), TRUE_B, TRUE_C)
    assert ks["kstar"] == pytest.approx(TRUE_KSTAR, rel=0.5)
    assert ks["band"][0] <= ks["kstar"] <= ks["band"][1]
    assert ks["n_mixed"] > 0


def test_estimate_kstar_returns_nan_without_mixed_cells():
    # Every cell fully alive -> no stochastic band to locate.
    df = pd.DataFrame({"student_tag": ["a", "a", "b", "b"], "lr": [1e-4] * 4,
                       "alpha": [8] * 4, "rank": [8] * 4, "collapsed": [0, 0, 0, 0]})
    assert np.isnan(estimate_kstar(df, TRUE_B, TRUE_C)["kstar"])


def test_law_misses_counts_disagreeing_cells():
    df = label_collapse(_simulate())
    n_wrong, n_cells = law_misses(df, TRUE_B, TRUE_C, TRUE_KSTAR)
    assert n_cells == df.student_tag.nunique()
    assert 0 <= n_wrong <= n_cells
    assert n_wrong / n_cells < 0.1  # the law generated this data; it should fit it well


def test_cell_table_one_row_per_cell():
    df = label_collapse(_simulate(n_seeds=3))
    tbl = cell_table(df)
    assert len(tbl) == df.student_tag.nunique()
    assert tbl.n.eq(3).all()
    assert tbl.frac.between(0, 1).all()


# ------------------------- loader -------------------------


def _plant_cell(root: Path, base: str, teacher: str, tag: str, per_seed: dict) -> None:
    d = (root / "students" / "dog" / base
         / "teacher-pythia-160m" / f"{teacher}_tseed1" / "ex10-20_prefix_n100k" / tag)
    d.mkdir(parents=True, exist_ok=True)
    (d / "preference_summary.json").write_text(json.dumps({
        "summaries": [
            {"prompt_set": "storytelling", "scoring": "sample", "p_target_mean": 0.0,
             "per_seed_p_target": {k: 0.0 for k in per_seed}},
            {"prompt_set": "test", "scoring": "prob",
             "p_target_mean": float(np.mean(list(per_seed.values()))),
             "per_seed_p_target": per_seed},
        ]
    }))


def test_load_law_runs_reads_per_seed_scores(tmp_path):
    _plant_cell(tmp_path, "pythia-160m-data-seed1", "ep2_lr5e-4_r32a64",
                "ep20_lr1e-4_r8a32", {"1": 0.1, "2": 0.2, "3": 1e-9})
    cells, runs = load_law_runs(tmp_path)
    assert len(cells) == 1 and len(runs) == 3
    assert cells.iloc[0].n_seeds == 3
    row = cells.iloc[0]
    assert (row.family, row.base_label, row.lineage) == ("focus", "data-seed1", "focus:data-seed1")
    assert (row.lr, row["rank"], row.alpha, row.scale) == (1e-4, 8, 32, 4.0)
    assert sorted(runs.test) == [1e-9, 0.1, 0.2]


def test_load_law_runs_keeps_only_requested_epochs(tmp_path):
    for tag in ("ep20_lr1e-4_r8a32", "ep5_lr1e-4_r8a32"):
        _plant_cell(tmp_path, "pythia-160m-data-seed1", "ep2_lr5e-4_r32a64", tag, {"1": 0.1})
    cells, _ = load_law_runs(tmp_path)
    assert cells.student_tag.tolist() == ["ep20_lr1e-4_r8a32"]


def test_load_law_runs_skips_missing_lineages(tmp_path):
    cells, runs = load_law_runs(tmp_path)
    assert cells.empty and runs.empty


def test_load_law_runs_finds_ckpt_base(tmp_path):
    # A revision-pinned student base gathers under a `ckpt/` level; the loader must route the
    # raw base string through model_dir_parts_from_name to find the lineage (the `@rev` model is
    # kept as the base value, no bare `ckpt` leaks in).
    d = (tmp_path / "students" / "dog" / "ckpt" / "pythia-160m@step143000"
         / "teacher-pythia-160m@step1000" / "ep2_lr5e-4_r32a64_tseed1"
         / "ex10-20_prefix_n100k" / "ep20_lr1e-4_r8a32")
    d.mkdir(parents=True)
    (d / "preference_summary.json").write_text(json.dumps({
        "summaries": [
            {"prompt_set": "test", "scoring": "prob", "p_target_mean": 0.15,
             "per_seed_p_target": {"1": 0.1, "2": 0.2}},
        ]
    }))
    cells, runs = load_law_runs(
        tmp_path, numbers_base="pythia-160m@step1000",
        lineages=[("ck", "ep2_lr5e-4_r32a64", "pythia-160m@step143000")],
    )
    assert len(cells) == 1 and len(runs) == 2
    assert cells.iloc[0].base == "pythia-160m@step143000"
    assert cells.iloc[0].student_tag == "ep20_lr1e-4_r8a32"
    assert cells.iloc[0].n_seeds == 2


def test_load_law_runs_separates_lineages(tmp_path):
    _plant_cell(tmp_path, "pythia-160m-data-seed1", "ep7_lr1e-4_r16a32",
                "ep20_lr1e-4_r8a32", {"1": 0.1})
    _plant_cell(tmp_path, "pythia-160m-weight-seed1", "ep7_lr1e-4_r16a32",
                "ep20_lr1e-4_r8a32", {"1": 0.2})
    cells, _ = load_law_runs(tmp_path)
    assert set(cells.lineage) == {"weak:data-seed1", "weak:weight-seed1"}


# ------------------------- adapter_coherence -------------------------


def _save_adapter(path, modules):
    """Write a minimal PEFT-style safetensors file: {name: (A, B)} -> lora_A/lora_B keys."""
    from safetensors.numpy import save_file

    tensors = {}
    for name, (A, B) in modules.items():
        tensors[f"{name}.lora_A.weight"] = A.astype(np.float32)
        tensors[f"{name}.lora_B.weight"] = B.astype(np.float32)
    save_file(tensors, str(path))


def test_adapter_coherence_coherent_channels_have_stable_rank_one(tmp_path):
    # All r channels identical -> BA is exactly rank 1 whatever the nominal rank.
    from subliminal_learning.analysis.collapse_law import adapter_coherence

    r, d = 16, 64
    a, b = np.ones((1, d)), np.ones((d, 1))
    _save_adapter(tmp_path / "adapter.safetensors",
                  {"m": (np.repeat(a, r, axis=0), np.repeat(b, r, axis=1))})
    df = adapter_coherence(tmp_path / "adapter.safetensors")
    assert df.iloc[0]["rank"] == r
    assert df.iloc[0].stable_rank == pytest.approx(1.0)
    assert df.iloc[0].top1_energy == pytest.approx(1.0)
    assert df.iloc[0].b_col_cos == pytest.approx(1.0)


def test_adapter_coherence_orthogonal_channels_have_stable_rank_r(tmp_path):
    # r mutually orthogonal equal-strength channels -> stable rank exactly r, |cos| exactly 0.
    from subliminal_learning.analysis.collapse_law import adapter_coherence

    r, d = 8, 64
    A = np.eye(r, d)
    B = np.eye(d, r)
    _save_adapter(tmp_path / "adapter.safetensors", {"m": (A, B)})
    df = adapter_coherence(tmp_path / "adapter.safetensors")
    assert df.iloc[0].stable_rank == pytest.approx(float(r))
    assert df.iloc[0].top1_energy == pytest.approx(1.0 / r)
    assert df.iloc[0].b_col_cos == pytest.approx(0.0)


def test_adapter_coherence_one_row_per_module_and_rank1_cos_is_nan(tmp_path):
    from subliminal_learning.analysis.collapse_law import adapter_coherence

    _save_adapter(tmp_path / "adapter.safetensors", {
        "layers.0.dense": (np.ones((1, 8)), np.ones((16, 1))),
        "layers.1.dense": (np.eye(2, 8), np.eye(16, 2)),
    })
    df = adapter_coherence(tmp_path / "adapter.safetensors")
    assert len(df) == 2 and set(df["rank"]) == {1, 2}
    assert np.isnan(df[df["rank"] == 1].b_col_cos.iloc[0])


# ------------------------- K helpers -------------------------


def test_kstar_anchors_distinct_and_pooled_is_canonical():
    assert KSTAR_POOLED == pytest.approx(7.87e-4)
    assert KSTAR_DS1_DESIGN == pytest.approx(7.84e-4)
    assert KSTAR_POOLED != KSTAR_DS1_DESIGN


def test_k_of_scalar_matches_formula():
    lr, a, r = 5e-4, 64, 32
    assert K_of(lr, a, r) == pytest.approx(lr * a**B_LAW * r**C_LAW)


def test_k_of_vectorized_over_series():
    df = pd.DataFrame({"lr": [1e-4, 5e-4], "alpha": [32, 64], "rank": [16, 32]})
    got = K_of(df["lr"], df["alpha"], df["rank"])
    exp = df["lr"] * df["alpha"] ** B_LAW * df["rank"] ** C_LAW
    np.testing.assert_allclose(got.to_numpy(), exp.to_numpy())


def test_k_from_tag_roundtrips_k_of():
    assert k_from_tag("ep20_lr5e-4_r32a64") == pytest.approx(K_of(5e-4, 64, 32))


def test_add_k_adds_columns_without_mutating():
    df = pd.DataFrame({"lr": [1e-4, 5e-4], "alpha": [32, 64], "rank": [16, 32]})
    out = add_k(df, kstar=KSTAR_POOLED)
    assert "K" not in df.columns
    np.testing.assert_allclose(out["K"].to_numpy(), K_of(df["lr"], df["alpha"], df["rank"]).to_numpy())
    np.testing.assert_allclose(out["K_rel"].to_numpy(), (out["K"] / KSTAR_POOLED).to_numpy())


def test_add_k_omits_k_rel_without_kstar():
    df = pd.DataFrame({"lr": [1e-4], "alpha": [32], "rank": [16]})
    assert "K_rel" not in add_k(df).columns


def test_load_law_runs_frames_match_hand_built_expectation(tmp_path):
    """Pinned pre-wiring output: the batched summary fetch must not change either frame."""
    _plant_cell(tmp_path, "pythia-160m-data-seed1", "ep2_lr5e-4_r32a64",
                "ep20_lr1e-4_r8a32", {"1": 0.1, "2": 0.2})
    _plant_cell(tmp_path, "pythia-160m-data-seed1", "ep2_lr5e-4_r32a64",
                "ep20_lr5e-4_r16a16", {"1": 1e-9})
    _plant_cell(tmp_path, "pythia-160m-weight-seed1", "ep2_lr5e-4_r32a64",
                "ep20_lr1e-4_r8a32", {"3": 0.4})
    # a cell with no `test` block is skipped entirely (present but unevaluated)
    d = (tmp_path / "students/dog/pythia-160m-data-seed1/teacher-pythia-160m"
         / "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k/ep20_lr1e-3_r8a8")
    d.mkdir(parents=True)
    (d / "preference_summary.json").write_text(json.dumps({"summaries": []}))

    lineages = [("focus", "ep2_lr5e-4_r32a64", "pythia-160m-data-seed1"),
                ("focus", "ep2_lr5e-4_r32a64", "pythia-160m-weight-seed1")]
    cells, runs = load_law_runs(tmp_path, lineages=lineages)

    def _meta(base_label, tag, lr, rank, alpha):
        return {"family": "focus", "teacher_tag": "ep2_lr5e-4_r32a64",
                "base": f"pythia-160m-{base_label}", "base_label": base_label,
                "lineage": f"focus:{base_label}", "student_tag": tag,
                "lr": lr, "rank": rank, "alpha": alpha, "scale": alpha / rank}

    m1 = _meta("data-seed1", "ep20_lr1e-4_r8a32", 1e-4, 8, 32)
    m2 = _meta("data-seed1", "ep20_lr5e-4_r16a16", 5e-4, 16, 16)
    m3 = _meta("weight-seed1", "ep20_lr1e-4_r8a32", 1e-4, 8, 32)
    exp_cells = pd.DataFrame([
        {**m1, "mean_test": 0.15000000000000002, "n_seeds": 2},
        {**m2, "mean_test": 1e-9, "n_seeds": 1},
        {**m3, "mean_test": 0.4, "n_seeds": 1},
    ])
    exp_runs = pd.DataFrame([
        {**m1, "seed": 1, "test": 0.1}, {**m1, "seed": 2, "test": 0.2},
        {**m2, "seed": 1, "test": 1e-9}, {**m3, "seed": 3, "test": 0.4},
    ])
    pd.testing.assert_frame_equal(cells, exp_cells)
    pd.testing.assert_frame_equal(runs, exp_runs)


# ------------------------- artifact-shaping helpers (lifted from the core65 scripts) -------
# The oracles below are literal transcriptions of the inline code in
# scripts/core65_theta_crossanimal.py and scripts/core65_anchored_companions.py, which produced
# the published theta / K* pins: these tests assert the extraction changed nothing.

def _dose_population(seed: int = 3) -> pd.DataFrame:
    """Seed-level rows with a realized dose, a fate label and several seeds per cell."""
    import math

    rng = np.random.default_rng(seed)
    rows = []
    for animal in ("dog", "cat"):
        for base in ("data-seed1", "weight-seed1"):
            shift = 0.0 if base == "data-seed1" else 0.25
            for lr in (1e-4, 3e-4, 1e-3):
                for alpha in (8, 32, 64):
                    for rank in (8, 32):
                        tag = f"ep20_lr{lr:g}_r{rank}a{alpha}"
                        for s in range(5):
                            dw = (lr * alpha**0.55 * rank**-0.1
                                  * math.exp(0.3 * rng.standard_normal()))
                            p = 1 / (1 + math.exp(-(4 * math.log(dw / 4e-4) - shift)))
                            rows.append({"animal": animal, "base": base, "base_label": base,
                                         "student_tag": tag, "seed": s, "lr": lr,
                                         "alpha": alpha, "rank": rank, "dw_fro": dw,
                                         "collapsed": int(rng.random() < 0.05 + 0.9 * p)})
    return pd.DataFrame(rows)


def _j(x):                                  # core65_theta_crossanimal.py:53-60 (verbatim)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def test_theta_dose_row_matches_inline_theta_row():
    from subliminal_learning.stats.collapse_law import theta_dose, theta_dose_row

    g = _dose_population()
    t = theta_dose(g)                       # oracle: core65_theta_crossanimal.py:63-75
    want = {"n": int(len(g)), "n_dead": int(g.collapsed.sum()),
            "n_cells": int(g.student_tag.nunique()),
            "auc": _j(t["auc"]), "theta": _j(t["theta"]), "theta_se_log": _j(t["se_log"]),
            "slope": _j(t["slope"]), "err_at_theta": _j(t["err"]), "converged": t["converged"],
            "theta_minerr": _j(t["theta_minerr"]), "err_minerr": _j(t["err_minerr"])}
    assert theta_dose_row(g) == want


def test_theta_dose_row_skips_single_class_slices():
    from subliminal_learning.stats.collapse_law import theta_dose_row

    g = _dose_population().assign(collapsed=0)
    row = theta_dose_row(g)
    assert row["fit_skipped"] == "single-class"
    assert np.isnan(row["auc"]) and "theta" not in row
    assert row["n"] == len(g) and row["n_dead"] == 0


def test_dose_law_fit_matches_inline_dose_law():
    import statsmodels.formula.api as smf

    from subliminal_learning.stats.collapse_law import dose_law_fit

    g = _dose_population()
    d = g.assign(y=np.log(g.dw_fro), lln=np.log(g.lr), aln=np.log(g.alpha),
                 rln=np.log(g["rank"]), cluster=g.student_tag)
    m = smf.ols("y ~ lln + aln + rln", d).fit(cov_type="cluster",
                                              cov_kwds={"groups": d["cluster"]})
    got = dose_law_fit(g)
    assert got["n"] == len(d) and got["n_cells"] == d.cluster.nunique()
    assert got["r2"] == pytest.approx(m.rsquared)
    for key, term in (("b_lr", "lln"), ("b_alpha", "aln"), ("b_rank", "rln")):
        assert got[key] == pytest.approx(m.params[term])
        assert got[key.replace("b_", "se_")] == pytest.approx(m.bse[term])
    assert got["intercept"] == pytest.approx(m.params["Intercept"])
    # the design generated dw from lr * alpha^0.55 * rank^-0.1, so the fit must recover it
    assert got["b_lr"] == pytest.approx(1.0, abs=0.05)
    assert got["b_alpha"] == pytest.approx(0.55, abs=0.05)
    assert got["b_rank"] == pytest.approx(-0.1, abs=0.05)


def test_theta_variance_split_matches_inline_variance_split():
    import statsmodels.formula.api as smf

    from subliminal_learning.stats.collapse_law import theta_variance_split

    arms = pd.DataFrame([
        {"animal": a, "base": b, "theta": t}
        for (a, b, t) in [("dog", "ds1", 0.128), ("dog", "ws1", 0.115),
                          ("cat", "ds1", 0.130), ("cat", "ws1", 0.112),
                          ("pig", "ds1", 0.134), ("pig", "ws1", 0.119)]
    ])
    d = arms.assign(lt=np.log(arms.theta))
    want = {name: float(smf.ols(f"lt ~ {rhs}", d).fit().rsquared)
            for name, rhs in (("r2_additive", "C(animal) + C(base)"),
                              ("r2_animal_only", "C(animal)"), ("r2_base_only", "C(base)"))}
    got = theta_variance_split(arms)
    for k, v in want.items():
        assert got[k] == pytest.approx(v)
    # this design is base-dominated, and the means round to 4 dp as the artifact stores them
    assert got["r2_base_only"] > got["r2_animal_only"]
    assert got["base_means"] == {"ds1": round((0.128 + 0.130 + 0.134) / 3, 4),
                                 "ws1": round((0.115 + 0.112 + 0.119) / 3, 4)}
    assert set(got["animal_means"]) == {"dog", "cat", "pig"}


def test_law_fits_row_matches_inline_fits():
    from subliminal_learning.stats.collapse_law import (
        fit_collapse_law,
        kstar_logistic,
        law_fits_row,
    )

    g = _dose_population()
    fit = fit_collapse_law(g)               # oracle: core65_anchored_companions.py:68-85
    free = kstar_logistic(g, fit.b, fit.c)
    fixed = kstar_logistic(g, B_LAW, C_LAW)
    want = {"n": fit.n_runs, "n_dead": fit.n_dead, "n_cells": fit.n_cells,
            "b": _j(fit.b), "c": _j(fit.c),
            "b_ci": [_j(v) for v in fit.b_ci], "c_ci": [_j(v) for v in fit.c_ci],
            "kstar_free": _j(free["kstar"]), "kstar_free_se_log": _j(free["se_log"]),
            "converged": bool(fit.converged),
            "kstar_fixed": _j(fixed["kstar"]), "kstar_fixed_se_log": _j(fixed["se_log"])}
    assert law_fits_row(g, B_LAW, C_LAW) == want


def test_law_fits_row_returns_none_on_single_class():
    from subliminal_learning.stats.collapse_law import law_fits_row

    assert law_fits_row(_dose_population().assign(collapsed=1), B_LAW, C_LAW) is None


def test_kstar_spread_matches_inline_spread():
    from subliminal_learning.stats.collapse_law import kstar_spread

    rows = [
        {"animal": "dog", "base": "ds1", "n_dead": 50, "kstar_free": 9.0e-4,
         "kstar_fixed": 8.8e-4},
        {"animal": "cat", "base": "ds1", "n_dead": 25, "kstar_free": 8.3e-4,
         "kstar_fixed": 8.1e-4},
        {"animal": "pig", "base": "ws1", "n_dead": 3, "kstar_free": 1.0e-2,   # below the floor
         "kstar_fixed": 1.0e-2},
    ]
    out = kstar_spread(rows, 20)
    assert out["n_arms"] == 2
    assert out["arms"] == ["dog|ds1", "cat|ds1"]
    assert out["kstar_free_min"] == pytest.approx(8.3e-4)
    assert out["kstar_free_max"] == pytest.approx(9.0e-4)
    assert out["kstar_free_ratio"] == pytest.approx(9.0e-4 / 8.3e-4)
    assert out["kstar_fixed_ratio"] == pytest.approx(8.8e-4 / 8.1e-4)


def test_kstar_spread_reports_none_when_no_arm_qualifies():
    from subliminal_learning.stats.collapse_law import kstar_spread

    rows = [{"animal": "dog", "base": "ds1", "n_dead": 1, "kstar_free": 9e-4,
             "kstar_fixed": 9e-4}]
    out = kstar_spread(rows, 20)
    assert out["n_arms"] == 0 and out["arms"] == []
    assert out["kstar_free_min"] is None and out["kstar_fixed_ratio"] is None


def test_fit_collapse_law_trajectory_recovers_step_exponent():
    """Synthetic K*(t) ∝ t^-d boundary: the fourth ratio recovers d and kstar_at tracks it."""
    import itertools

    import numpy as np
    import pandas as pd

    from subliminal_learning.stats import collapse_law as cl

    rng = np.random.default_rng(0)
    b_true, c_true, d_true, k0 = 0.5, -0.1, 0.4, 1e-3
    lrs = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    rows = []
    for lr, a, r in itertools.product(lrs, [4, 16, 64, 256], [1, 8, 64]):
        tag = f"ep20_lr{lr:g}_r{r}a{a}"
        K = lr * a**b_true * r**c_true
        for t in np.geomspace(100, 25_000, 12):
            logit = 6.0 * (np.log(K) - np.log(k0) + d_true * np.log(t / 25_000))
            y = rng.random() < 1 / (1 + np.exp(-logit))
            rows.append(dict(student_tag=tag, lr=lr, alpha=a, rank=r, step=t, collapsed=int(y)))
    df = pd.DataFrame(rows)
    fit = cl.fit_collapse_law_trajectory(df)
    assert abs(fit.b - b_true) < 0.08 and abs(fit.c - c_true) < 0.08 and abs(fit.d - d_true) < 0.08
    assert fit.d_ci[0] < d_true < fit.d_ci[1]
    assert fit.n_runs == len(lrs) * 4 * 3 and fit.n_rows == fit.n_runs * 12
    # Threshold at the end vs at 1/40 of training follows t^-d.
    ratio = fit.kstar_at(25_000 / 40) / fit.kstar_at(25_000)
    assert abs(np.log(ratio) - d_true * np.log(40)) < 0.25
