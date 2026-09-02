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


def _load_direct_arm(root: Path, rel: str, base: str, pattern: str,
                     probes: pd.DataFrame, norms: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select one arm from maintained caches, with exact-key reads for cache gaps."""
    prefix = rel.rstrip("/") + "/"
    n = norms[norms.run_path.str.startswith(prefix, na=False)].copy()
    n["student_tag"] = n.run_path.map(lambda x: Path(x).parent.name)
    p = probes[probes.run_path.str.startswith(prefix, na=False)].copy()
    p["student_tag"] = p.run_path.map(lambda x: Path(x).parent.name)
    if pattern.startswith("ep20_"):
        n = n[n.student_tag.str.startswith("ep20_")]
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
    counts = {"norm_cache_rows": len(n), "probe_cache_hits": len(p) - len(backfill),
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


def _regional_boundary(df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    for label, mask in (("low_alpha_le_32", df.alpha <= 32),
                        ("high_alpha_ge_64", df.alpha >= 64)):
        d = df[mask]
        if len(d) < 20 or d.collapsed.nunique() < 2 or d.collapsed.sum() < 5:
            out[label] = {"n": len(d), "dead": int(d.collapsed.sum()), "fit": None}
            continue
        try:
            out[label] = _boundary_fit(d)
        except Exception as exc:  # failure is part of the census
            out[label] = {"n": len(d), "dead": int(d.collapsed.sum()),
                          "fit": None, "error": type(exc).__name__}
    return out


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
    for arm, args in specs.items():
        frame, coverage = _load_direct_arm(root, *args, probes=probes, norms=norms)
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
                rec["curvature"] = _regional_boundary(frame)
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
    runs = (mods.groupby(["student_tag", "ftseed"], as_index=False)
            .agg(lr=("lr", "first"), alpha=("alpha", "first"), rank=("rank", "first"),
                 total=("total", "mean"), frozen=("frozen", "mean"), moved=("moved", "mean")))
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
                  "pile_kl_base_ablated"]
    paired = fz[cols].merge(mv[cols], on=key, suffixes=("_frozen", "_moved"))
    paired["p_full"] = paired[["p_target_orig_frozen", "p_target_orig_moved"]].mean(axis=1)
    paired["kl_full"] = paired[["pile_kl_base_orig_frozen", "pile_kl_base_orig_moved"]].mean(axis=1)
    paired["ret_frozen"] = ((paired.p_target_ablated_frozen - base_prior)
                             / (paired.p_full - base_prior))
    paired["ret_moved"] = ((paired.p_target_ablated_moved - base_prior)
                            / (paired.p_full - base_prior))
    trait = paired[paired.p_full > 0.02].copy()
    damaged = paired[paired.kl_full > anchored_dead_cut(root, "pythia-160m-data-seed1")].copy()
    damaged["best_single_share"] = (damaged[["pile_kl_base_ablated_frozen",
                                              "pile_kl_base_ablated_moved"]].max(axis=1)
                                     / damaged.kl_full)

    per_alpha = []
    for alpha, g in paired.groupby("alpha"):
        t = trait[trait.alpha == alpha]
        z = damaged[damaged.alpha == alpha]
        per_alpha.append({"alpha": int(alpha), "n": len(g), "trait_carrying": len(t),
                          "ret_frozen_median": t.ret_frozen.median() if len(t) else np.nan,
                          "ret_moved_median": t.ret_moved.median() if len(t) else np.nan,
                          "kl_full_all_median": g.kl_full.median(),
                          "kl_keep_frozen_all_median": g.pile_kl_base_ablated_frozen.median(),
                          "kl_keep_moved_all_median": g.pile_kl_base_ablated_moved.median(),
                          "damaged": len(z), "kl_full_median": z.kl_full.median() if len(z) else np.nan,
                          "kl_keep_frozen_median": z.pile_kl_base_ablated_frozen.median() if len(z) else np.nan,
                          "kl_keep_moved_median": z.pile_kl_base_ablated_moved.median() if len(z) else np.nan})
    return {"files": counts, "rows_frozen": len(fz), "rows_moved": len(mv),
            "paired_rows": len(paired), "paired_cells": int(paired.student_tag.nunique()),
            "trait_carrying_rows": len(trait), "damaged_rows_by_full_kl_anchor": len(damaged),
            "damaged_best_single_over_full_median": damaged.best_single_share.median(),
            "per_alpha": per_alpha,
            "readout_note": "Ablations store pp1 KL but not delta_ce; continuous damage is exact, "
                            "while the damaged subset uses KL>delta_cross only as the validated KL/delta-CE bridge."}


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
        if all(_finite(rec.get(k)) for k in ("kl_full", "kl_frozen_only", "kl_moved_only")):
            splice_rows.append({k: float(rec[k]) for k in
                                ("kl_full", "kl_frozen_only", "kl_moved_only")})
    norms, splices = pd.DataFrame(norm_rows), pd.DataFrame(splice_rows)
    out: dict[str, Any] = {"norm_records": len(norms), "splice_records": len(splices)}
    if len(norms):
        norms["frozen_share_sum_norms"] = norms.frozen / (norms.frozen + norms.moved)
        out["replay_right_cos_median"] = norms.right_cos.median()
        out["replay_wrong_cos_median"] = norms.wrong_cos.median()
        out["frozen_share_by_epoch"] = norms.groupby("ep").frozen_share_sum_norms.median().to_dict()
    if len(splices):
        damaged = splices[splices.kl_full > anchored_dead_cut(root, "pythia-160m")].copy()
        damaged["best_single_share"] = (damaged[["kl_frozen_only", "kl_moved_only"]].max(axis=1)
                                         / damaged.kl_full)
        out.update(damaged_records=len(damaged), kl_full_median=damaged.kl_full.median(),
                   kl_frozen_only_median=damaged.kl_frozen_only.median(),
                   kl_moved_only_median=damaged.kl_moved_only.median(),
                   best_single_over_full_median=damaged.best_single_share.median())
    return out


def build(root: Path) -> dict:
    frames, direct = _direct_arms(root)
    return {
        "schema_version": 1,
        "definition": "collapsed iff pp1 delta_ce > H_unigram - CE(base) for the receiver",
        "output_root": str(root),
        "direct_arms": direct,
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
