"""Artifact-only census of the unigram-anchored collapse boundary.

This is the reproducible numerical companion to
``docs/results/boundary_law_deep_dive.md``.  It deliberately does not read any
results prose or saved notebook outputs.  Inputs are run-local probe/norm records,
the unigram-anchor record, and intervention records under
``{output_root}/analysis``.  The cross-trait grid is intentionally out of scope.

The script writes one JSON payload when ``--output-json`` is supplied.  Expensive
record-cache refreshes are disabled: missing artifacts are counted, never inferred.

    uv run python -m scripts.boundary_law_deep_dive \
        --output-json /tmp/boundary_law_deep_dive.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis.frames import parse_param_tag
from subliminal_learning.config import resolve_output_root
from subliminal_learning.coherence.probe_io import anchored_dead_cut


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _js(x: Any) -> Any:
    """Convert numpy/pandas scalars and non-finite values to strict JSON values."""
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_js(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _load_cache_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the maintained pp1-probe and slim adapter-norm record caches once."""
    probes = pd.read_csv(root / "analysis" / "probe_cache.csv", low_memory=False)
    probes = (probes[probes.pool_version.eq("pp1")].sort_values("record_mtime")
              .drop_duplicates("run_path", keep="last"))
    norms = pd.read_csv(root / "analysis" / "record_cache_adapter_norms.csv", low_memory=False)
    norms = norms.sort_values("record_mtime").drop_duplicates("run_path", keep="last")
    return probes, norms

def _pooled_dose(run_dir: Path) -> float:
    """Mean module-normalized Frobenius dose from one adapter-norm artifact."""
    rec = json.loads((run_dir / "eval" / "adapter_norms.json").read_text())
    scale = float(rec["alpha"]) / float(rec["rank"])
    modules = rec["modules"]
    items = list(modules.values() if isinstance(modules, dict) else modules)
    values = np.asarray([
        scale * float(module.get("B_Afinal_fro", np.nan)) / float(module["W0_fro"])
        for module in items
    ])
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _nonadaptive_population(root: Path) -> set[str]:
    """Run paths (relative tag/seed suffix) of the designed cross-data-order population.

    The cross-data-order tree also hosts later campaigns (bandext126/focus20/band48
    cells, ftseed3-5 ensembles, reserved 901+ seeds).  The population of record is the
    non-adaptive 8x7x9 grid: 504 cells x ftseed{1,2} = 1,008 designed runs, as listed
    in ``analysis/nonadaptive_8x7_harvest.json``.
    """
    harvest = json.loads((root / "analysis" / "nonadaptive_8x7_harvest.json").read_text())
    return {f"{r['tag']}/ftseed{int(r['seed'])}" for r in harvest}


def _load_direct_arm(root: Path, rel: str, base: str, pattern: str,
                     probes: pd.DataFrame, norms: pd.DataFrame,
                     population: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Select one arm from maintained caches, with exact-key reads for cache gaps.

    ``population`` restricts the arm to an explicit ``tag/ftseedN`` design list.
    """
    prefix = rel.rstrip("/") + "/"
    n = norms[norms.run_path.str.startswith(prefix, na=False)].copy()
    n["student_tag"] = n.run_path.map(lambda x: Path(x).parent.name)
    p = probes[probes.run_path.str.startswith(prefix, na=False)].copy()
    p["student_tag"] = p.run_path.map(lambda x: Path(x).parent.name)
    if population is not None:
        n = n[n.run_path.str.removeprefix(prefix).isin(population)]
        p = p[p.run_path.str.removeprefix(prefix).isin(population)]
    if pattern.startswith("ep20_"):
        n = n[n.student_tag.str.startswith("ep20_")]
        p = p[p.student_tag.str.startswith("ep20_")]
    elif pattern.endswith("ftseed1"):
        n = n[n.run_path.str.endswith("/ftseed1")]
        p = p[p.run_path.str.endswith("/ftseed1")]
        p = p[p.student_tag.str.startswith("ep20_")]
    # The maintained probe cache intentionally omits some tree patterns (notably
    # trait-free and SGD). The norm cache is still a complete run index for these
    # arms, so read only exact missing probe keys rather than crawling run trees.
    cached_paths = set(p.run_path)
    backfill, failed = [], 0
    for run_path in n.run_path:
        if run_path in cached_paths:
            continue
        probe_path = root / run_path / "eval" / "pile_probe.json"
        try:
            rec = json.loads(probe_path.read_text())
        except (OSError, json.JSONDecodeError):
            failed += 1
            continue
        if rec.get("pool_version") != "pp1":
            failed += 1
            continue
        backfill.append({"run_path": run_path, "delta_ce": rec.get("delta_ce"),
                         "kl": rec.get("kl_base_student"),
                         "student_tag": Path(run_path).parent.name})
    if backfill:
        p = pd.concat([p, pd.DataFrame(backfill)], ignore_index=True)

    p["seed"] = p.run_path.map(lambda x: Path(x).name)
    finite = np.isfinite(pd.to_numeric(p.delta_ce, errors="coerce"))
    counts = {"population_design_runs": (len(population) if population is not None else None),
              "norm_cache_rows": len(n), "probe_cache_hits": len(p) - len(backfill),
              "exact_probe_reads_from_norm_keys": len(backfill),
              "missing_or_invalid_exact_probes": failed,
              "indexed_probe_rows": len(p), "nonfinite_probe": int((~finite).sum()),
              "finite_delta_ce": int(finite.sum())}
    p = p[finite].copy()
    p = p.merge(norms[["run_path", "dw_fro"]], on="run_path", how="left")
    norm_ok = np.isfinite(pd.to_numeric(p.dw_fro, errors="coerce")) & (p.dw_fro > 0)
    norm_cache_hits = int(norm_ok.sum())
    exact_norm_reads, norm_failures = 0, 0
    for index, row in p[~norm_ok].iterrows():
        try:
            dose = _pooled_dose(root / row.run_path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            norm_failures += 1
            continue
        if np.isfinite(dose) and dose > 0:
            p.at[index, "dw_fro"] = dose
            exact_norm_reads += 1
        else:
            norm_failures += 1
    counts.update(norm_cache_hits=norm_cache_hits,
                  exact_norm_reads_from_run_keys=exact_norm_reads,
                  missing_or_invalid_exact_norms=norm_failures)

    cut = anchored_dead_cut(root, base)
    rows = []
    for row in p.itertuples(index=False):
        try:
            hp = parse_param_tag(row.student_tag)
        except ValueError:
            continue
        rows.append({
            "run_path": row.run_path, "student_tag": row.student_tag, "seed": row.seed,
            "ep": int(hp["epochs"]), "lr": float(hp["lr"]),
            "alpha": int(hp["alpha"]), "rank": int(hp["rank"]),
            "delta_ce": float(row.delta_ce), "pile_kl": row.kl,
            "collapsed": int(float(row.delta_ce) > cut), "dw_fro": row.dw_fro,
        })
    counts.update(fit_rows=len(rows), positive_norm=int((p.dw_fro > 0).sum()),
                  missing_or_invalid_norm=int((~(p.dw_fro > 0)).sum()),
                  dead=int(sum(r["collapsed"] for r in rows)), cut=cut,
                  cache_files=["analysis/probe_cache.csv", "analysis/record_cache_adapter_norms.csv"])
    return pd.DataFrame(rows), counts


def _boundary_fit(df: pd.DataFrame) -> dict:
    fit = cl.fit_collapse_law(df)
    k = cl.K_of(df.lr, df.alpha, df["rank"], b=fit.b, c=fit.c)
    ks = cl.kstar_logistic(df, fit.b, fit.c)
    return {
        "n": fit.n_runs, "cells": fit.n_cells, "dead": fit.n_dead,
        "b": fit.b, "b_ci": fit.b_ci, "c": fit.c, "c_ci": fit.c_ci,
        "kstar_own_coordinate": ks["kstar"], "k_error": ks["err"],
        "k_auc": roc_auc_score(df.collapsed, k), "converged": fit.converged,
    }


def _curvature_fit(df: pd.DataFrame) -> dict:
    """Full-grid boundary logit with a quadratic ``(ln alpha)^2`` term.

    Replaces the low/high-alpha regional split (2026-09-02): the halves separated on several
    arms and their MLE ratios were optimizer-dependent, whereas the full grid with one extra
    term converges on every arm and keeps every designed cell.  ``ln alpha`` is centred at the
    grid mean so the linear coefficient is ``b`` at the grid centre; the alpha-exponent at any
    alpha is ``b(alpha) = (beta_alpha + 2 beta_aa (ln alpha - mean)) / beta_lr``.  The reported
    curvature is the ratio ``beta_aa / beta_lr`` with a cell-clustered delta-method interval.
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    la = np.log(df.alpha.to_numpy())
    centre = float(la.mean())
    lac = la - centre
    X = np.c_[np.ones(len(df)), np.log(df.lr), lac, np.log(df["rank"]), lac ** 2]
    y = df.collapsed.to_numpy()
    res = sm.Logit(y, X).fit(disp=0, maxiter=2000, cov_type="cluster",
                             cov_kwds={"groups": df.student_tag.to_numpy()})
    p, V = np.asarray(res.params), np.asarray(res.cov_params())
    z = norm.ppf(0.975)

    def ratio_ci(i: int) -> tuple[float, tuple[float, float]]:
        q = p[i] / p[1]
        g = np.zeros(len(p))
        g[i] = 1.0 / p[1]
        g[1] = -p[i] / p[1] ** 2
        se = float(np.sqrt(g @ V @ g))
        return float(q), (q - z * se, q + z * se)

    b_centre, b_centre_ci = ratio_ci(2)
    c, c_ci = ratio_ci(3)
    curv, curv_ci = ratio_ci(4)
    b_at = {str(a): float((p[2] + 2.0 * p[4] * (np.log(a) - centre)) / p[1])
            for a in (4, 32, 256)}
    return {"n": len(df), "cells": int(df.student_tag.nunique()), "dead": int(y.sum()),
            "converged": bool(res.mle_retvals.get("converged", False)),
            "beta_lr": float(p[1]), "ln_alpha_centre": centre,
            "b_at_centre": b_centre, "b_at_centre_ci": b_centre_ci,
            "c": c, "c_ci": c_ci, "curvature": curv, "curvature_ci": curv_ci,
            "b_at_alpha": b_at}


def _dose_fit(df: pd.DataFrame, *, steps: bool = False) -> dict | None:
    d = df[np.isfinite(df.dw_fro) & (df.dw_fro > 0)].copy()
    if len(d) < 20:
        return None
    cols = {"log_lr": np.log(d.lr), "log_alpha": np.log(d.alpha),
            "log_rank": np.log(d["rank"])}
    if steps:
        cols["log_steps"] = np.log(3.0 * d.ep)
    X = sm.add_constant(pd.DataFrame(cols, index=d.index))
    fit = sm.OLS(np.log(d.dw_fro), X).fit(
        cov_type="cluster", cov_kwds={"groups": d.student_tag})
    names = ["log_lr", "log_alpha", "log_rank"] + (["log_steps"] if steps else [])
    exps = {n.removeprefix("log_"): float(fit.params[n]) for n in names}
    cis = {n.removeprefix("log_"): tuple(map(float, fit.conf_int().loc[n])) for n in names}
    lr = exps["lr"]
    out: dict[str, Any] = {"n": len(d), "R2": fit.rsquared, "exponents": exps,
                           "ci": cis, "alpha_over_lr": exps["alpha"] / lr,
                           "rank_over_lr": exps["rank"] / lr}
    if "steps" in exps:
        out["steps_over_lr"] = exps["steps"] / lr
    if d.collapsed.nunique() == 2:
        th = cl.theta_dose(d)
        out["collapse_on_dose"] = {
            "theta": th["theta"], "theta_se_log": th["se_log"],
            "auc": th["auc"], "error": th["err"], "dead": th["n_dead"],
        }
    return out


def _direct_arms(root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    specs = {
        "cross_data_order": (
            "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/"
            "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
            "pythia-160m-data-seed1", "ep20_*/ftseed*"),
        "orig_same_model": (
            "students/dog/pythia-160m/teacher-pythia-160m/"
            "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
            "pythia-160m", "ep20_*/ftseed*"),
        "trait_free": (
            "students/controls/base/pythia-160m-data-seed1/teacher-pythia-160m/"
            "ex10-20_prefix_n100k", "pythia-160m-data-seed1", "ep20_*/ftseed*"),
        "sgd": (
            "students/sgd/dog/pythia-160m/teacher-pythia-160m/"
            "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
            "pythia-160m", "ep20_*/ftseed*"),
        "410m_same_model": (
            "students/dog/pythia-410m/teacher-pythia-410m/"
            "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
            "pythia-410m", "ep20_*/ftseed*"),
        "410m_cross_model": (
            "students/dog/pythia-410m-seed1/teacher-pythia-410m/"
            "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
            "pythia-410m-seed1", "ep20_*/ftseed*"),
        "teachers_dog_orig": ("teachers/dog/pythia-160m", "pythia-160m", "*/ftseed1"),
    }
    probes, norms = _load_cache_tables(root)
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    populations = {"cross_data_order": _nonadaptive_population(root)}
    for arm, args in specs.items():
        frame, coverage = _load_direct_arm(root, *args, probes=probes, norms=norms,
                                           population=populations.get(arm))
        frames[arm] = frame
        rec: dict[str, Any] = {"coverage": coverage}
        if len(frame) and frame.collapsed.nunique() == 2:
            if arm == "teachers_dog_orig":
                # Native teacher law includes the training budget.  The ridge is required by
                # near separation; the matched ep20 row below uses the student estimator.
                X = sm.add_constant(pd.DataFrame({
                    "log_lr": np.log(frame.lr), "log_alpha": np.log(frame.alpha),
                    "log_rank": np.log(frame["rank"]), "log_steps": np.log(3.0 * frame.ep),
                }))
                ridge = sm.GLM(frame.collapsed, X, family=sm.families.Binomial()).fit_regularized(
                    alpha=1e-3, L1_wt=0.0)
                lrcoef = ridge.params["log_lr"]
                score = np.asarray(X @ ridge.params)
                rec["boundary_native_4term_ridge"] = {
                    "n": len(frame), "dead": int(frame.collapsed.sum()),
                    "b": ridge.params["log_alpha"] / lrcoef,
                    "c": ridge.params["log_rank"] / lrcoef,
                    "steps_over_lr": ridge.params["log_steps"] / lrcoef,
                    "auc": roc_auc_score(frame.collapsed, score), "ridge_alpha": 1e-3,
                }
            else:
                rec["boundary"] = _boundary_fit(frame)
                rec["curvature"] = _curvature_fit(frame)
        rec["dose"] = _dose_fit(frame, steps=(arm == "teachers_dog_orig"))
        audit[arm] = rec

    # Like-for-like teacher comparison: ep20 and exactly the cross-data-order grid support.
    teacher = frames["teachers_dog_orig"]
    strip_cells = set(frames["cross_data_order"].student_tag)
    matched = teacher[(teacher.ep == 20) & teacher.student_tag.isin(strip_cells)]
    audit["teachers_dog_orig"]["boundary_ep20_matched_student_support"] = _boundary_fit(matched)

    # Matched support for the three principal Adam student arms.
    keys = ["cross_data_order", "orig_same_model", "trait_free"]
    shared = set.intersection(*(set(frames[k].student_tag) for k in keys))
    audit["matched_160m_adam"] = {"cells": len(shared), "fits": {
        k: _boundary_fit(frames[k][frames[k].student_tag.isin(shared)]) for k in keys}}
    return frames, audit


def _kstar_structure(root: Path) -> dict:
    """Cross-arm K* structure (analysis.kstar_structure) + the three-panel figure for the doc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from subliminal_learning.analysis import kstar_structure as ks

    repo = Path(__file__).resolve().parents[1]
    out = ks.run_all(root, repo)
    frames = out.pop("_frames")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    ks.plot_kstar_structure(frames["slices"], out["decomposition"], axes,
                            teacher_order=out["meta"]["teachers"])
    dec = out["decomposition"]
    fig.suptitle(f"K* per (animal x student base <- teacher base | teacher recipe) slice, n = {dec['n_slices']} "
                 "estimable slices; black bars = factor-level medians (raw marginals)", y=1.03, fontsize=11)
    fig.tight_layout()
    figdir = repo / "docs" / "results" / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "boundary_law_kstar_structure.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    out["figure"] = "docs/results/figures/boundary_law_kstar_structure.png"
    return out


def _component_norms(root: Path, strip: pd.DataFrame) -> dict:
    """Census the complete eight-rank init-replay component-norm harvest."""
    records: list[pd.DataFrame] = []
    files = []
    for rank in (1, 2, 4, 8, 16, 32, 64, 128):
        path = root / "analysis" / f"adapter_norms_strip_r{rank}.json"
        if not path.exists():
            continue
        files.append(path.name)
        records.append(pd.DataFrame(json.loads(path.read_text())["results"]))
    mods = pd.concat(records, ignore_index=True)
    tags = mods.student_tag.map(parse_param_tag)
    mods["lr"] = [x["lr"] for x in tags]
    mods["scale"] = mods.alpha / mods["rank"]
    for term, key in (("total", "B_Afinal_fro"), ("frozen", "B_Ainit_fro"),
                      ("moved", "B_dA_fro")):
        mods[term] = mods.scale * mods[key] / mods.W0_fro
    # Factor norms and product-bound ratios (‖B‖·‖A‖/‖B·A‖): the ingredients of the
    # optimizer-form assembly in the deep dive's section 5.2.
    mods["gap_frozen"] = mods.B_fro * mods.A_init_fro / mods.B_Ainit_fro
    mods["gap_moved"] = mods.B_fro * mods.dA_fro / mods.B_dA_fro
    runs = (mods.groupby(["student_tag", "ftseed"], as_index=False)
            .agg(lr=("lr", "first"), alpha=("alpha", "first"), rank=("rank", "first"),
                 total=("total", "mean"), frozen=("frozen", "mean"), moved=("moved", "mean"),
                 B=("B_fro", "mean"), dA=("dA_fro", "mean"), A0=("A_init_fro", "mean"),
                 gap_frozen=("gap_frozen", "mean"), gap_moved=("gap_moved", "mean")))
    # Primary-region support used by the original component analysis.  Runs whose direct probe
    # exists carry a non-divergence check; unprobed rows are retained and counted separately.
    probed_keys = set(zip(strip.student_tag, strip.seed.str.removeprefix("ftseed").astype(int)))
    runs["direct_probe_present"] = [(t, int(s)) in probed_keys
                                    for t, s in zip(runs.student_tag, runs.ftseed)]
    d = runs[(runs.alpha / runs["rank"] <= 64) & np.isfinite(runs.total)
             & np.isfinite(runs.frozen) & np.isfinite(runs.moved)
             & (runs.total > 0) & (runs.frozen > 0) & (runs.moved > 0)].copy()

    def term_fit(term: str) -> dict:
        X = sm.add_constant(pd.DataFrame({"log_lr": np.log(d.lr),
                                          "log_alpha": np.log(d.alpha),
                                          "log_rank": np.log(d["rank"])}, index=d.index))
        lin = sm.OLS(np.log(d[term]), X).fit(
            cov_type="cluster", cov_kwds={"groups": d.student_tag})
        Xq = X.assign(log_lr_sq=X.log_lr ** 2)
        quad = sm.OLS(np.log(d[term]), Xq).fit(
            cov_type="cluster", cov_kwds={"groups": d.student_tag})
        return {"n": len(d), "cells": int(d.student_tag.nunique()), "R2_linear": lin.rsquared,
                "lr": lin.params.log_lr, "alpha": lin.params.log_alpha,
                "rank": lin.params.log_rank, "quadratic_log_lr": quad.params.log_lr_sq,
                "quadratic_log_lr_se": quad.bse.log_lr_sq, "R2_quadratic": quad.rsquared}

    d["moved_over_frozen"] = d.moved / d.frozen
    d["moved_share_of_total"] = d.moved / d.total
    med = (d.groupby("alpha")[["moved_over_frozen", "moved_share_of_total"]]
           .median().reset_index().to_dict("records"))
    return {"files": files, "module_rows": len(mods), "run_rows_all": len(runs),
            "run_rows_primary": len(d), "primary_cells": int(d.student_tag.nunique()),
            "primary_rows_with_direct_probe": int(d.direct_probe_present.sum()),
            "fits": {term: term_fit(term) for term in ("total", "frozen", "moved")},
            "factor_fits": {obj: term_fit(obj) for obj in ("A0", "B", "dA")},
            "product_bound_ratio_fits": {obj: term_fit(obj) for obj in ("gap_frozen", "gap_moved")},
            "frozen_gap_over_sqrt_rank": {
                "median": float((d.gap_frozen / np.sqrt(d["rank"])).median()),
                "sd": float((d.gap_frozen / np.sqrt(d["rank"])).std()),
                "min": float((d.gap_frozen / np.sqrt(d["rank"])).min())},
            "B_over_dA_median_by_alpha": d.assign(x=d.B / d.dA).groupby("alpha").x.median().to_dict(),
            "dominance_by_alpha": med}


def _student_component_splices(root: Path) -> dict:
    """Pair the complete pp1 keep-frozen/keep-moved strip intervention records."""
    by_term: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    base_prior = float("nan")
    for term in ("frozen", "moved"):
        files = sorted((root / "analysis").glob(
            f"frozen_a_ablation_strip_{term}_r*_s*_pp1.json"))
        rows = []
        for path in files:
            rec = json.loads(path.read_text())
            base_prior = float(rec["p_target_base"])
            rows.extend(rec.get("results", []))
        by_term[term] = pd.DataFrame(rows)
        counts[term] = len(files)
    fz, mv = by_term["frozen"], by_term["moved"]
    key = ["student_tag", "ftseed", "alpha", "rank"]
    cols = key + ["p_target_orig", "p_target_ablated", "pile_kl_base_orig",
                  "pile_kl_base_ablated", "pile_kl_orig_ablated", "delta_ce_orig",
                  "delta_ce_ablated", "ce_base"]
    paired = fz.reindex(columns=cols).merge(
        mv.reindex(columns=cols), on=key, suffixes=("_frozen", "_moved"))
    paired["p_full"] = paired[["p_target_orig_frozen", "p_target_orig_moved"]].mean(axis=1)
    paired["kl_full"] = paired[["pile_kl_base_orig_frozen", "pile_kl_base_orig_moved"]].mean(axis=1)
    paired["ret_frozen"] = ((paired.p_target_ablated_frozen - base_prior)
                             / (paired.p_full - base_prior))
    paired["ret_moved"] = ((paired.p_target_ablated_moved - base_prior)
                            / (paired.p_full - base_prior))
    trait = paired[paired.p_full > 0.02].copy()
    ce_cols = ["delta_ce_orig_frozen", "delta_ce_orig_moved",
               "delta_ce_ablated_frozen", "delta_ce_ablated_moved"]
    exact = paired[np.isfinite(paired[ce_cols]).all(axis=1)].copy()
    exact["delta_ce_full"] = exact[
        ["delta_ce_orig_frozen", "delta_ce_orig_moved"]].mean(axis=1)
    cut = anchored_dead_cut(root, "pythia-160m-data-seed1")
    exact["collapsed_full"] = exact.delta_ce_full > cut
    exact["collapsed_keep_frozen"] = exact.delta_ce_ablated_frozen > cut
    exact["collapsed_keep_moved"] = exact.delta_ce_ablated_moved > cut
    collapsed = exact[exact.collapsed_full].copy()
    collapsed["best_single_kl_share"] = (
        collapsed[["pile_kl_base_ablated_frozen",
                   "pile_kl_base_ablated_moved"]].max(axis=1) / collapsed.kl_full)

    def collapse_summary(d: pd.DataFrame) -> dict[str, Any]:
        frozen = d.collapsed_keep_frozen
        moved = d.collapsed_keep_moved
        return {
            "n": len(d),
            "keep_frozen_collapsed": int(frozen.sum()),
            "keep_moved_collapsed": int(moved.sum()),
            "both_single_terms_sufficient": int((frozen & moved).sum()),
            "only_frozen_sufficient": int((frozen & ~moved).sum()),
            "only_moved_sufficient": int((~frozen & moved).sum()),
            "neither_single_term_sufficient": int((~frozen & ~moved).sum()),
            "delta_ce_full_median": d.delta_ce_full.median() if len(d) else np.nan,
            "delta_ce_keep_frozen_median": d.delta_ce_ablated_frozen.median() if len(d) else np.nan,
            "delta_ce_keep_moved_median": d.delta_ce_ablated_moved.median() if len(d) else np.nan,
        }

    per_alpha = []
    for alpha, g in paired.groupby("alpha"):
        t = trait[trait.alpha == alpha]
        z = collapsed[collapsed.alpha == alpha]
        per_alpha.append({"alpha": int(alpha), "n": len(g), "trait_carrying": len(t),
                          "ret_frozen_median": t.ret_frozen.median() if len(t) else np.nan,
                          "ret_moved_median": t.ret_moved.median() if len(t) else np.nan,
                          "kl_full_all_median": g.kl_full.median(),
                          "kl_keep_frozen_all_median": g.pile_kl_base_ablated_frozen.median(),
                          "kl_keep_moved_all_median": g.pile_kl_base_ablated_moved.median(),
                          "collapsed": len(z),
                          "kl_full_median": z.kl_full.median() if len(z) else np.nan,
                          "kl_keep_frozen_median": z.pile_kl_base_ablated_frozen.median() if len(z) else np.nan,
                          "kl_keep_moved_median": z.pile_kl_base_ablated_moved.median() if len(z) else np.nan,
                          "collapse_interventions": collapse_summary(z)})
    pair_disagreement = np.abs(exact.delta_ce_orig_frozen - exact.delta_ce_orig_moved)
    return {"files": counts, "rows_frozen": len(fz), "rows_moved": len(mv),
            "paired_rows": len(paired), "paired_cells": int(paired.student_tag.nunique()),
            "trait_carrying_rows": len(trait), "exact_ce_rows": len(exact),
            "delta_ce_full_pair_max_abs_diff": pair_disagreement.max(),
            "collapse_cut": cut, "collapsed_rows_by_full_delta_ce_anchor": len(collapsed),
            "collapse_interventions": collapse_summary(collapsed),
            "collapsed_best_single_kl_over_full_median": collapsed.best_single_kl_share.median(),
            "per_alpha": per_alpha,
            "readout_note": "Collapse labels for full and ablated models use exact pp1 "
                            "delta-CE against the unigram-anchored receiver-specific cutoff; "
                            "KL is reported as a continuous secondary readout."}


def _teacher_components(root: Path) -> dict:
    norm_rows, splice_rows = [], []
    teacher_root = root / "teachers" / "dog" / "pythia-160m"
    for path in sorted(teacher_root.glob("*/ftseed1/eval/teacher_term_norms.json")):
        rec = json.loads(path.read_text())
        tag = path.parts[-4]
        hp = parse_param_tag(tag)
        mods = rec.get("modules", [])
        frozen = np.nanmean([m.get("B_Ainit_fro", np.nan) for m in mods])
        moved = np.nanmean([m.get("B_dA_fro", np.nan) for m in mods])
        norm_rows.append({"tag": tag, "ep": hp["epochs"], "alpha": hp["alpha"],
                          "frozen": frozen, "moved": moved,
                          "right_cos": np.nanmean([m.get("cos_Ainit_Afinal", np.nan) for m in mods]),
                          "wrong_cos": np.nanmean([m.get("cos_wrongseed_Afinal", np.nan) for m in mods])})
    for path in sorted(teacher_root.glob("*/ftseed1/eval/teacher_term_splice.json")):
        rec = json.loads(path.read_text())
        if all(_finite(rec.get(k)) for k in ("kl_full", "kl_frozen_only", "kl_moved_only",
                                              "ce_full", "ce_frozen_only", "ce_moved_only")):
            splice_rows.append({k: float(rec[k]) for k in
                                ("kl_full", "kl_frozen_only", "kl_moved_only",
                                 "ce_full", "ce_frozen_only", "ce_moved_only")})
    norms, splices = pd.DataFrame(norm_rows), pd.DataFrame(splice_rows)
    out: dict[str, Any] = {"norm_records": len(norms), "splice_records": len(splices)}
    if len(norms):
        norms["frozen_share_sum_norms"] = norms.frozen / (norms.frozen + norms.moved)
        out["replay_right_cos_median"] = norms.right_cos.median()
        out["replay_wrong_cos_median"] = norms.wrong_cos.median()
        out["frozen_share_by_epoch"] = norms.groupby("ep").frozen_share_sum_norms.median().to_dict()
    if len(splices):
        anchor = json.loads((root / "analysis" / "unigram_anchor_pp1.json").read_text())
        h_uni = float(anchor["h_uni_primary"])
        collapsed = splices[splices.ce_full > h_uni].copy()
        collapsed["frozen_collapsed"] = collapsed.ce_frozen_only > h_uni
        collapsed["moved_collapsed"] = collapsed.ce_moved_only > h_uni
        collapsed["best_single_share"] = (collapsed[["kl_frozen_only", "kl_moved_only"]].max(axis=1)
                                           / collapsed.kl_full)
        out.update(collapsed_records=len(collapsed), h_unigram=h_uni,
                   ce_full_median=collapsed.ce_full.median(),
                   ce_frozen_only_median=collapsed.ce_frozen_only.median(),
                   ce_moved_only_median=collapsed.ce_moved_only.median(),
                   frozen_only_collapsed=int(collapsed.frozen_collapsed.sum()),
                   moved_only_collapsed=int(collapsed.moved_collapsed.sum()),
                   both_single_terms_collapsed=int((collapsed.frozen_collapsed & collapsed.moved_collapsed).sum()),
                   neither_single_term_collapsed=int((~collapsed.frozen_collapsed & ~collapsed.moved_collapsed).sum()),
                   kl_full_median=collapsed.kl_full.median(),
                   kl_frozen_only_median=collapsed.kl_frozen_only.median(),
                   kl_moved_only_median=collapsed.kl_moved_only.median(),
                   best_single_over_full_median=collapsed.best_single_share.median())
    return out


def build(root: Path) -> dict:
    frames, direct = _direct_arms(root)
    return {
        "schema_version": 1,
        "definition": "collapsed iff pp1 delta_ce > H_unigram - CE(base) for the receiver",
        "output_root": str(root),
        "direct_arms": direct,
        "kstar_structure": _kstar_structure(root),
        "component_norms_strip": _component_norms(root, frames["cross_data_order"]),
        "student_component_splices_strip": _student_component_splices(root),
        "teacher_components": _teacher_components(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    root = args.output_root or resolve_output_root()
    payload = _js(build(Path(root)))
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if args.output_json:
        args.output_json.write_text(rendered + "\n")
        print(f"wrote {args.output_json}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
