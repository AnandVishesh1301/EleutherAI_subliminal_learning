"""Two-part mediation of the Pile-KL collapse: is realized dose ‖ΔW‖_F the whole story?

Rebuilds ``{output_root}/analysis/mediation_twopart_kl.json`` (quoted in
``docs/results/collapse_law.md`` §4 and in claims C6). Three parts on ONE population — the
calibrated non-adaptive strip (orig→ds1, ep20 finals, ~1.5k seed-runs):

- part 2 (crossing): logistic ``dead ~ log dw_fro`` then ``+ (log lr, log α, log r)``. If the
  endpoint Frobenius dose fully mediated the HPs the block would go null; it does not. Adding
  mean cos(ΔW, W₀) screens the block (to null at the anchored cut), i.e. the residual HP reach
  is a DIRECTION effect. Reported at both integrity labels, plus fate AUCs of realized dose
  vs the planned dose K.
- part 1 (graded, sub-boundary): alive-only OLS ``log KL ~ log dw_fro`` + the HP block —
  below the boundary damage is dose-graded and the residual is carried by rank alone.
- part 3 (bimodality): all-runs OLS ``log KL ~ log dw_fro`` (+ quadratic, + spread
  L = 2(log dw_fro − log dw_spec)) + a dead indicator, whose coefficient is the log-nat
  separation of the dead mode at matched dose (circular by construction — a separation
  measure, not an independent effect).

Population and loaders (no glob fan-outs — the record caches own discovery):

- rows: ``analysis/fidelity_strip_ds1/three_curves.csv`` (`scripts.three_curves_strip_ds1`),
  one per (cell, ftseed) under ``students/dog/pythia-160m-data-seed1/teacher-pythia-160m/
  ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k``.
- integrity: `analysis.probe_cache.load_probe_cache` → ``kl`` (kl_base_student), ``delta_ce``.
- dose: `analysis.norms.load_adapter_norm_runs` → ``dw_fro``/``dw_spec``, with runs lacking a
  run-dir record backfilled from the frozen ds1 wfinal sweep via
  `analysis.norms.load_dw_sweep_runs` (identical aggregation; agreement asserted below).
- cos: the same sweep's ``cos_dw_w0``.
- HPs: parsed from the cell tag ``ep{E}_lr{L}_r{R}a{A}``; cluster = cell (ftseeds share it).

Cuts: dead iff pp1 KL > `probe_io.DEAD_KL_CUT`, and the anchored variant delta_ce >
``delta_cross[ds1]`` from ``analysis/unigram_anchor_pp1.json``. Non-finite adapters and
missing probes/doses are EXCLUDED and counted, never labelled dead.

    uv run python -m scripts.mediation_twopart_kl
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

from subliminal_learning.stats import collapse_law as cl
from subliminal_learning.stats.core import cluster_ols
from subliminal_learning.stats.core import json_scalar as _j
from subliminal_learning.stats.mediation import (
    coef_table as _coefs,
    fate_auc as _auc,
    nested_logit_block,
    zscore_column,
)

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

STRIP_LINEAGE = ("students/dog/pythia-160m-data-seed1/teacher-pythia-160m/"
                 "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k")
SWEEP_REL = Path("analysis") / "adapter_dw_sweep_pythia-160m-data-seed1_wfinal.json"
TC_REL = Path("analysis") / "fidelity_strip_ds1" / "three_curves.csv"
ANCHOR_REL = Path("analysis") / "unigram_anchor_pp1.json"
ARTIFACT_REL = Path("analysis") / "mediation_twopart_kl.json"

TAG_RE = re.compile(r"^ep(?P<ep>\d+)_lr(?P<lr>[0-9.e+-]+)_r(?P<r>\d+)a(?P<a>\d+)$")
HP_TERMS = ["lln", "aln", "rln"]
B_TASK, C_TASK = 0.616, -0.102          # task-channel (trait-free) exponents
COS_COL_DOC = ("mean over modules of <B@A_final, W0>_F / (||B@A_final||_F ||W0||_F) "
               "from adapter_dw_sweep_pythia-160m-data-seed1_wfinal.json")


def _block(b) -> dict:
    """A `mediation.BlockTest` in this artifact's key names (the ``hp_block_*`` schema)."""
    return {"hp_block_lr_chi2": b.lr_chi2, "hp_block_lr_p": b.lr_p, "hp_block_lr_df": b.lr_df,
            "hp_block_cluster_wald_chi2": b.wald_chi2,
            "hp_block_cluster_wald_p": b.wald_p,
            "hp_block_converged": b.converged}


def _part2(df: pd.DataFrame, dead_col: str, cut_doc: str) -> dict:
    """The crossing-mediation ladder at one integrity cut (§4 part 2)."""
    d = df.copy()
    d["dead"] = d[dead_col].astype(int)
    rung = nested_logit_block(d, "dead ~ ldw", "dead ~ ldw + lln + aln + rln",
                              cluster="cell", terms=HP_TERMS)
    m0, m1 = rung.reduced, rung.full
    out = {
        "cut": cut_doc, "n": int(len(d)), "n_dead": int(d.dead.sum()),
        "n_clusters": int(d.cell.nunique()),
        "m0_dead_on_ldw": {"coef_ldw": _j(m0.params["ldw"]), "pseudo_r2": _j(m0.prsquared)},
        "m1_plus_hp": {"pseudo_r2": _j(m1.prsquared), "coefs": _coefs(m1)},
        **_block(rung.block),
        "auc_dw_fro": _auc(d.dw_fro, d.dead),
        f"auc_K_task_b{B_TASK}_c{C_TASK}": _auc(d.K_task, d.dead),
        f"auc_K_ref_b{cl.B_LAW}_c{cl.C_LAW}": _auc(d.K_ref, d.dead),
    }

    # cos block: same subset before/after, so the two p-values are comparable
    # cos is O(1e-4): z-score it so BFGS stays conditioned, report coefficients on the raw scale
    c, _mu, sd = zscore_column(d[d.cos_dw_w0.notna()], "cos_dw_w0")
    without = nested_logit_block(c, "dead ~ ldw", "dead ~ ldw + lln + aln + rln",
                                 cluster="cell", terms=HP_TERMS)
    rung_cos = nested_logit_block(c, "dead ~ ldw + cosz", "dead ~ ldw + cosz + lln + aln + rln",
                                  cluster="cell", terms=HP_TERMS)
    k0, k1 = rung_cos.reduced, rung_cos.full
    with_cos = rung_cos.block
    coefs_with_cos = _coefs(k1)
    z = coefs_with_cos.pop("cosz")
    coefs_with_cos["cos_dw_w0"] = {"coef": z["coef"] / sd, "se_cluster": z["se_cluster"] / sd}
    out["cos_block"] = {
        "cos_column": COS_COL_DOC,
        "n": int(len(c)), "n_dead": int(c.dead.sum()),
        "hp_block_lr_p_without_cos_same_subset": without.block.lr_p,
        "hp_block_converged_without_cos": without.converged,
        "hp_block_lr_chi2_with_cos": with_cos.lr_chi2,
        "hp_block_lr_p_with_cos": with_cos.lr_p,
        "hp_block_cluster_wald_p_with_cos": with_cos.wald_p,
        "hp_block_converged_with_cos": with_cos.converged,
        "coefs_with_cos": coefs_with_cos,
        "pseudo_r2_ldw_cos": _j(k0.prsquared),
        "pseudo_r2_ldw_cos_hp": _j(k1.prsquared),
    }
    return out


def _part1(df: pd.DataFrame) -> dict:
    """Alive-only graded fit: log KL on log dose, then the HP block (§4 part 1)."""
    from scipy import stats as sps

    d = df[~df.dead_kl4].copy()
    m0 = cluster_ols(d, "lkl ~ ldw", "cell")
    m1 = cluster_ols(d, "lkl ~ ldw + lln + aln + rln", "cell")
    wald = m1.wald_test(", ".join(f"{t} = 0" for t in HP_TERMS), scalar=True)
    lr = 2.0 * (m1.llf - m0.llf)
    return {
        "population": "alive runs (probe KL < 4.0), finite dose",
        "n": int(len(d)), "n_clusters": int(d.cell.nunique()),
        "kl_range": [_j(d.kl.min()), _j(d.kl.max())],
        "m0": {"r2": _j(m0.rsquared), "coef_ldw": _j(m0.params["ldw"]),
               "se_cluster_ldw": _j(m0.bse["ldw"])},
        "m1_plus_hp": {"r2": _j(m1.rsquared),
                       "coefs": {n: {"coef": _j(m1.params[n]), "se_cluster": _j(m1.bse[n]),
                                     "p_cluster": _j(m1.pvalues[n])}
                                 for n in ("ldw", *HP_TERMS)}},
        "hp_block_cluster_wald_chi2": _j(wald.statistic),
        "hp_block_cluster_wald_p": _j(wald.pvalue),
        "hp_block_lr_chi2": _j(lr), "hp_block_lr_p": _j(sps.chi2.sf(lr, len(HP_TERMS))),
        "delta_r2": _j(m1.rsquared - m0.rsquared),
    }


def _part3(df: pd.DataFrame) -> dict:
    """Dead-mode offset at matched dose, under four conditioning variants (§4 part 3)."""
    d = df.copy()
    d["dead_kl4"] = d.dead_kl4.astype(int)
    models = {
        "linear": "lkl ~ ldw + dead_kl4",
        "quadratic": "lkl ~ ldw + ldw2 + dead_kl4",
        "energy_spread": "lkl ~ ldw + L + dead_kl4",
        "energy_spread_quadratic": "lkl ~ ldw + ldw2 + L + dead_kl4",
    }
    out = {}
    for name, formula in models.items():
        res = cluster_ols(d, formula, "cell")
        out[name] = {
            "formula": formula, "n": int(len(d)), "r2": _j(res.rsquared),
            "dead_coef_lognats": _j(res.params["dead_kl4"]),
            "dead_se_cluster": _j(res.bse["dead_kl4"]),
            "dead_p_cluster": _j(res.pvalues["dead_kl4"]),
            "other_coefs": {n: _j(res.params[n]) for n in res.params.index if n != "dead_kl4"},
        }
    return {
        "population": "all finite-dose finite-KL strip runs",
        "n": int(len(d)), "n_dead": int(d.dead_kl4.sum()),
        "note": "dead indicator is derived from the same KL (cut 4.0), so the offset "
                "quantifies the bimodal separation of the dead mode at matched dose, not an "
                "independent effect. L = 2*(log dw_fro - log dw_spec) = run-level log stable "
                "rank (the spread coordinate at fixed energy).",
        "models": out,
    }


def _compare(old: dict, new: dict) -> list[tuple[str, float | None, float | None]]:
    """Headline numbers of the stored artifact vs the recomputation, in one table."""
    def g(payload, *keys):
        cur = payload
        for k in keys:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        return cur

    rows = []
    for cut in ("kl4", "anchored"):
        p = ("part2_crossing_mediation", cut)
        rows += [
            (f"P2[{cut}] n", g(old, *p, "n"), g(new, *p, "n")),
            (f"P2[{cut}] n_dead", g(old, *p, "n_dead"), g(new, *p, "n_dead")),
            (f"P2[{cut}] HP block LR p", g(old, *p, "hp_block_lr_p"),
             g(new, *p, "hp_block_lr_p")),
            (f"P2[{cut}] HP block Wald p", g(old, *p, "hp_block_cluster_wald_p"),
             g(new, *p, "hp_block_cluster_wald_p")),
            (f"P2[{cut}] +cos LR p", g(old, *p, "cos_block", "hp_block_lr_p_with_cos"),
             g(new, *p, "cos_block", "hp_block_lr_p_with_cos")),
            (f"P2[{cut}] +cos Wald p",
             g(old, *p, "cos_block", "hp_block_cluster_wald_p_with_cos"),
             g(new, *p, "cos_block", "hp_block_cluster_wald_p_with_cos")),
            (f"P2[{cut}] AUC dw_fro", g(old, *p, "auc_dw_fro"), g(new, *p, "auc_dw_fro")),
        ]
    p1 = ("part1_subboundary_graded",)
    rows += [
        ("P1 n", g(old, *p1, "n"), g(new, *p1, "n")),
        ("P1 coef ldw", g(old, *p1, "m1_plus_hp", "coefs", "ldw", "coef"),
         g(new, *p1, "m1_plus_hp", "coefs", "ldw", "coef")),
        ("P1 coef rank (rln)", g(old, *p1, "m1_plus_hp", "coefs", "rln", "coef"),
         g(new, *p1, "m1_plus_hp", "coefs", "rln", "coef")),
        ("P1 rank p_cluster", g(old, *p1, "m1_plus_hp", "coefs", "rln", "p_cluster"),
         g(new, *p1, "m1_plus_hp", "coefs", "rln", "p_cluster")),
    ]
    p3 = ("part3_bimodality_matched_dose", "models")
    for m in ("linear", "quadratic", "energy_spread", "energy_spread_quadratic"):
        rows.append((f"P3 dead coef [{m}]", g(old, *p3, m, "dead_coef_lognats"),
                     g(new, *p3, m, "dead_coef_lognats")))
    rows += [("n_analysis", g(old, "coverage", "n_analysis"), g(new, "coverage", "n_analysis")),
             ("n_cos_covered", g(old, "coverage", "n_cos_covered"),
              g(new, "coverage", "n_cos_covered"))]
    return rows


@app.command()
def main(
    output_root: Path = typer.Option(None, help="Outputs root (default: paths_local.yaml)."),
    refresh: bool = typer.Option(True, help="Re-stat the run-dir records before loading."),
) -> None:
    from subliminal_learning.analysis.norms import load_adapter_norm_runs, load_dw_sweep_runs
    from subliminal_learning.analysis.probe_cache import load_probe_cache
    from subliminal_learning.coherence.probe_io import DEAD_KL_CUT
    from subliminal_learning.config import resolve_output_root
    from subliminal_learning.utils.io import atomic_write_json

    load_dotenv(ROOT / ".env")
    out_root = resolve_output_root(output_root)

    tc = pd.read_csv(out_root / TC_REL)
    tc = tc[["cell", "ftseed"]].copy()
    tag = tc.cell.str.extract(TAG_RE)
    tc["lr"] = pd.to_numeric(tag["lr"])
    tc["a"] = pd.to_numeric(tag["a"])
    tc["r"] = pd.to_numeric(tag["r"])
    tc["run_path"] = STRIP_LINEAGE + "/" + tc.cell + "/ftseed" + tc.ftseed.astype(str)
    rel = tc.run_path.tolist()

    probe = load_probe_cache(out_root, rel, refresh=refresh, pool_version="pp1")[
        ["run_path", "kl", "delta_ce", "nonfinite"]]
    dose = load_adapter_norm_runs(out_root, rel, refresh=refresh)[
        ["run_path", "dw_fro", "dw_spec"]]
    sweep = load_dw_sweep_runs(out_root / SWEEP_REL).rename(
        columns={"student_tag": "cell", "dw_fro": "dw_fro_sweep", "dw_spec": "dw_spec_sweep"})

    df = (tc.merge(probe, on="run_path", how="left")
            .merge(dose, on="run_path", how="left")
            .merge(sweep[["cell", "ftseed", "dw_fro_sweep", "dw_spec_sweep", "cos_dw_w0"]],
                   on=["cell", "ftseed"], how="left"))
    for c in ("kl", "delta_ce", "dw_fro", "dw_spec", "dw_fro_sweep", "dw_spec_sweep",
              "cos_dw_w0"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # the two dose sources must agree where both exist (same aggregation), then backfill
    both = df.dw_fro.notna() & df.dw_fro_sweep.notna() & (df.dw_fro > 0)
    max_rel = float(((df.dw_fro_sweep[both] - df.dw_fro[both]).abs()
                     / df.dw_fro[both]).max()) if both.any() else float("nan")
    # same aggregation, so the only admissible gap is float noise in the two harvests
    assert not (max_rel > 1e-4), f"dose sources disagree (max rel diff {max_rel:.2e})"
    n_record = int(df.dw_fro.notna().sum())
    n_backfill = int((df.dw_fro.isna() & df.dw_fro_sweep.notna()).sum())
    df["dw_fro"] = df.dw_fro.fillna(df.dw_fro_sweep)
    df["dw_spec"] = df.dw_spec.fillna(df.dw_spec_sweep)

    probe_ok = df.kl.notna() & np.isfinite(df.kl.fillna(np.inf)) \
        & ~df.nonfinite.fillna(True).astype(bool)
    dose_ok = df.dw_fro.notna() & np.isfinite(df.dw_fro.fillna(np.nan)) & (df.dw_fro > 0)
    n_diverged = int((df.kl.notna() & ~probe_ok).sum())
    missing_dose = df[probe_ok & ~dose_ok]

    anchor = json.loads((out_root / ANCHOR_REL).read_text())
    delta_cross = float(anchor["delta_cross"]["pythia-160m-data-seed1"])

    fit = df[probe_ok & dose_ok].reset_index(drop=True)
    fit["ldw"] = np.log(fit.dw_fro)
    fit["ldw2"] = fit.ldw ** 2
    fit["L"] = 2.0 * (np.log(fit.dw_fro) - np.log(fit.dw_spec))
    fit["lkl"] = np.log(fit.kl)
    fit["lln"] = np.log(fit.lr)
    fit["aln"] = np.log(fit.a)
    fit["rln"] = np.log(fit.r)
    fit["K_task"] = fit.lr * fit.a ** B_TASK * fit.r ** C_TASK
    fit["K_ref"] = fit.lr * fit.a ** cl.B_LAW * fit.r ** cl.C_LAW
    fit["dead_kl4"] = fit.kl > DEAD_KL_CUT
    fit["dead_anchor"] = fit.delta_ce > delta_cross

    payload = {
        "kind": "mediation_twopart_kl",
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "provenance": "scripts/mediation_twopart_kl.py",
        "population": f"calibrated non-adaptive strip (orig->ds1), ep20 finals: "
                      f"{TC_REL} rows, run dirs under {STRIP_LINEAGE}",
        "loaders": {
            "probe": "analysis.probe_cache.load_probe_cache(OUT, rel_paths, pool_version='pp1')"
                     " -> kl (kl_base_student), delta_ce  [{run}/eval/pile_probe.json]",
            "dose": "analysis.norms.load_adapter_norm_runs -> dw_fro, dw_spec "
                    "[{run}/eval/adapter_norms.json]; missing runs backfilled by "
                    f"analysis.norms.load_dw_sweep_runs({SWEEP_REL}) with identical "
                    "aggregation mean_m (alpha/rank)*||B@A_final||/||W0|| (fro and spec)",
            "cos": f"{SWEEP_REL}: mean_m BA_W0_ip/(BA_fro*W0_fro)",
        },
        "labels": {
            "dead_kl4": f"probe KL > {DEAD_KL_CUT} (probe_io.DEAD_KL_CUT)",
            "dead_anchor": f"delta_ce > {delta_cross} "
                           "(unigram_anchor_pp1.json delta_cross[pythia-160m-data-seed1])",
        },
        "planned_dose": {
            "K_task": f"lr * alpha^{B_TASK} * r^{C_TASK}",
            "K_ref": f"lr * alpha^{cl.B_LAW} * r^{cl.C_LAW} (cl.B_LAW/C_LAW: 160m "
                     "designed cross-data-order population fit, anchored label, 2026-09-09)",
        },
        "inference": "cluster-robust by cell tag (ftseeds share a cell); logistic fits BFGS "
                     "maxiter 500; LR tests plain chi2(3); Wald tests use the cluster sandwich",
        "coverage": {
            "n_three_curves_rows": int(len(df)),
            "n_cells": int(df.cell.nunique()),
            "ftseed_counts": {str(k): int(v)
                              for k, v in df.ftseed.value_counts().sort_index().items()},
            "n_probe_covered": int(df.kl.notna().sum()),
            "n_probe_nonfinite_or_inf_kl_excluded": n_diverged,
            "n_dose_record_covered": n_record,
            "n_dose_sweep_backfilled": n_backfill,
            "n_dose_union_covered": int(dose_ok.sum()),
            "n_missing_dose_excluded": int(len(missing_dose)),
            "missing_dose_runs": [{"cell": r.cell, "ftseed": int(r.ftseed),
                                   "kl": _j(round(float(r.kl), 3))}
                                  for r in missing_dose.itertuples()],
            "n_diverged_adapters_in_population": n_diverged,
            "n_analysis": int(len(fit)),
            "n_analysis_cells": int(fit.cell.nunique()),
            "n_cos_covered": int(fit.cos_dw_w0.notna().sum()),
            "dose_source_agreement_max_rel_diff": _j(max_rel),
        },
        "part2_crossing_mediation": {
            "kl4": _part2(fit, "dead_kl4", f"probe KL > {DEAD_KL_CUT}"),
            "anchored": _part2(fit, "dead_anchor",
                               f"delta_ce > {delta_cross:.4f} (ds1 unigram crossing)"),
        },
        "part1_subboundary_graded": _part1(fit),
        "part3_bimodality_matched_dose": _part3(fit),
    }

    art = out_root / ARTIFACT_REL
    old = json.loads(art.read_text()) if art.exists() else {}
    rows = _compare(old, payload) if old else []
    atomic_write_json(art, payload)
    typer.echo(f"wrote {art}")

    if rows:
        typer.echo(f"\nvalidation vs stored artifact (created {old.get('created')}):")
        typer.echo(f"{'quantity':28s} {'stored':>14s} {'recomputed':>14s}")
        def fmt(v):
            if v is None:
                return f"{'-':>14s}"
            return f"{v:14d}" if isinstance(v, int) else f"{v:14.4g}"

        for name, o, n in rows:
            typer.echo(f"{name:28s} {fmt(o)} {fmt(n)}")


if __name__ == "__main__":
    app()
