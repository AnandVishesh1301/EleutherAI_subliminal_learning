"""Teacher-stage cos(ΔW, W₀) mediation screen — does direction close the C10 level-set gap?

The C10 result: the teacher dose law implies a protective rank exponent (c −0.48…−0.52
after lr-normalisation) but the directly fitted boundary has c ≈ 0 — at matched ‖ΔW‖_F,
higher rank is MORE damaging, so the HPs reach fate partly outside the Frobenius dose.
On students the analogous residual is screened off by mean per-module cos(ΔW, W₀)
(`mediation_twopart_kl.json` part 2: HP-block p 1e-36 → 0.65 anchored / 0.047 kl4 once
cos enters). This script runs the same test on the freshly harvested teacher cosines
(`scripts.teacher_cos_w0` → ``{run}/eval/cos_w0.json``).

Population: ftseed1 teacher finals from ``--runs`` (default: the 08-24 harvest list,
12 trees = {dog,cat,lion,pig} × {orig, ds1, ws1}). Loaders (all record-cache kinds over
``{run}/eval/*.json``):

- fate:  ``pile_probe`` → ``kl`` (kl_base_student, pool pp1); dead iff kl > 4.0
  (`probe_io.DEAD_KL_CUT`). Nonfinite adapters and missing probes are EXCLUDED,
  never labelled dead.
- dose:  ``adapter_norms`` → ``dw_fro`` = mean over modules of (α/r)·‖B·A_final‖_F/‖W₀‖_F.
- cos:   ``cos_w0`` → ``mean_cos`` (module-mean cos(ΔW, W₀), the student screen's
  regressor) and ``pooled_cos`` (Σ<ΔW,W₀>/sqrt(Σ‖ΔW‖²·Σ‖W₀‖²)).

HPs parse from the run tag (``ep{E}_lr{L}_r{R}a{A}``); steps = 3·ep. Logistic fits are
BFGS maxiter 500 with cluster-robust covariance on the (tree, lr, r, α) tag — epochs of
one recipe share a cluster; LR tests are plain chi2, Wald tests use the cluster sandwich
(the `mediation_twopart_kl` inference conventions).

Writes ``{output_root}/analysis/teacher_cos_screen.json``.

    uv run python -m scripts.teacher_cos_screen
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

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

TAG_RE = re.compile(r"^ep(?P<ep>\d+)_lr(?P<lr>[0-9.e+-]+)_r(?P<r>\d+)a(?P<a>\d+)$")
STEPS_PER_EPOCH = 3          # ~190 SFT examples / batch schedule -> 3 optimizer steps per epoch
HP_TERMS = ["lln", "aln", "rln", "sln"]


def _rel_run_paths(lines: list[str], output_root: Path) -> list[str]:
    """Run-list lines -> paths relative to `output_root` (robust to a migrated absolute root)."""
    out = []
    for line in lines:
        p = Path(line)
        if not p.is_absolute():
            out.append(str(p))
            continue
        try:
            out.append(str(p.relative_to(output_root)))
        except ValueError:
            parts = list(p.parts)
            i = parts.index("teachers")            # raise if a line is neither form
            out.append(str(Path(*parts[i:])))
    return out


def _j(x):
    """JSON-safe scalar: numpy -> python, non-finite floats -> None."""
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def _fit(df: pd.DataFrame, formula: str, cluster: str = "cluster"):
    """Cluster-robust logistic fit (BFGS, maxiter 500) with the plain-MLE llf preserved."""
    import statsmodels.formula.api as smf

    mod = smf.logit(formula, df)
    res = mod.fit(method="bfgs", maxiter=500, disp=0,
                  cov_type="cluster", cov_kwds={"groups": df[cluster]})
    return res


def _coefs(res) -> dict:
    return {name: {"coef": _j(res.params[name]), "se_cluster": _j(res.bse[name])}
            for name in res.params.index if name != "Intercept"}


def _block_test(res_full, res_reduced, terms=HP_TERMS) -> dict:
    """LR (plain chi2) + cluster-sandwich Wald test that `terms` are jointly zero in the full fit."""
    from scipy import stats as sps

    lr = 2.0 * (res_full.llf - res_reduced.llf)
    wald = res_full.wald_test(", ".join(f"{t} = 0" for t in terms), scalar=True)
    return {
        "lr_chi2": _j(lr), "lr_df": len(terms), "lr_p": _j(sps.chi2.sf(lr, len(terms))),
        "cluster_wald_chi2": _j(wald.statistic), "cluster_wald_p": _j(wald.pvalue),
    }


def _auc(score: pd.Series, dead: pd.Series) -> float:
    from subliminal_learning.analysis.stats import mannwhitney_auc
    m = dead.values.astype(bool)
    return float(mannwhitney_auc(score.values[m], score.values[~m]))


def _term(res, name: str) -> dict:
    z = res.params[name] / res.bse[name]
    from scipy import stats as sps
    return {"coef": _j(res.params[name]), "se_cluster": _j(res.bse[name]),
            "z": _j(z), "p_cluster": _j(2 * sps.norm.sf(abs(z)))}


def _qtab(df: pd.DataFrame, by, col: str = "mean_cos") -> dict:
    """{group: {n, median, q25, q75}} of `col`, grouped by `by` (column name or list)."""
    out = {}
    for key, g in df.groupby(by, sort=True):
        k = "_".join(str(_j(x)) for x in key) if isinstance(key, tuple) else str(_j(key))
        s = g[col]
        out[k] = {"n": int(s.notna().sum()), "median": _j(s.median()),
                  "q25": _j(s.quantile(0.25)), "q75": _j(s.quantile(0.75))}
    return out


def _pop_fits(df: pd.DataFrame, df_fuller: pd.DataFrame) -> dict:
    """The mediation ladder on one population.

    `df` is the cos-covered subset every model is fit on (so before/after-cos p-values are
    same-subset comparable); `df_fuller` adds back rows with probe+dose but no cos record,
    for a transparency re-run of the pre-cos block test.
    """
    out = {"n": int(len(df)), "n_dead": int(df.dead.sum()),
           "n_clusters": int(df.cluster.nunique())}

    m0 = _fit(df, "dead ~ ldw")
    m1 = _fit(df, "dead ~ ldw + lln + aln + rln + sln")
    out["m0_dead_on_ldw"] = {"coef_ldw": _j(m0.params["ldw"]), "pseudo_r2": _j(m0.prsquared),
                             "converged": bool(m0.mle_retvals.get("converged"))}
    out["m1_plus_hp"] = {"pseudo_r2": _j(m1.prsquared), "coefs": _coefs(m1),
                         "converged": bool(m1.mle_retvals.get("converged"))}
    out["hp_block_pre_cos"] = _block_test(m1, m0)

    if len(df_fuller) > len(df):
        f0 = _fit(df_fuller, "dead ~ ldw")
        f1 = _fit(df_fuller, "dead ~ ldw + lln + aln + rln + sln")
        out["hp_block_pre_cos_fuller_subset"] = {
            "n": int(len(df_fuller)), "n_dead": int(df_fuller.dead.sum()),
            **_block_test(f1, f0)}

    from scipy import stats as sps

    out["cos_blocks"] = {}
    cos_fits = {}
    for cos_col in ("mean_cos", "pooled_cos"):
        # cos is O(1e-4): z-score it so BFGS is well-conditioned, report both scales
        d = df.copy()
        mu, sd = float(d[cos_col].mean()), float(d[cos_col].std(ddof=0))
        d["cosz"] = (d[cos_col] - mu) / sd
        c0 = _fit(d, "dead ~ ldw + cosz")
        c1 = _fit(d, "dead ~ ldw + cosz + lln + aln + rln + sln")
        cos_fits[cos_col] = c0
        lr_over_ldw = 2.0 * (c0.llf - m0.llf)
        lr_over_hp = 2.0 * (c1.llf - m1.llf)
        out["cos_blocks"][cos_col] = {
            "cos_scale": {"mean": _j(mu), "sd": _j(sd),
                          "note": "fits use cosz = (cos-mean)/sd; raw coef = coef/sd"},
            "m_ldw_cos": {"pseudo_r2": _j(c0.prsquared), "coefs": _coefs(c0),
                          "cos_coef_raw": _j(c0.params["cosz"] / sd),
                          "converged": bool(c0.mle_retvals.get("converged"))},
            "m_ldw_cos_hp": {"pseudo_r2": _j(c1.prsquared), "coefs": _coefs(c1),
                             "cos_coef_raw": _j(c1.params["cosz"] / sd),
                             "converged": bool(c1.mle_retvals.get("converged"))},
            "hp_block_with_cos": _block_test(c1, c0),
            "cos_term_in_ldw_cos": _term(c0, "cosz"),
            "cos_term_in_full": _term(c1, "cosz"),
            "cos_increment_lr": {
                "over_ldw_chi2": _j(lr_over_ldw), "over_ldw_p": _j(sps.chi2.sf(lr_over_ldw, 1)),
                "over_ldw_hp_chi2": _j(lr_over_hp),
                "over_ldw_hp_p": _j(sps.chi2.sf(lr_over_hp, 1)), "df": 1},
            "rank_term_with_cos": _term(c1, "rln"),
        }

    out["rank_coefficient"] = {
        "pre_cos": _term(m1, "rln"),
        "post_cos_mean": out["cos_blocks"]["mean_cos"]["rank_term_with_cos"],
        "post_cos_pooled": out["cos_blocks"]["pooled_cos"]["rank_term_with_cos"],
    }

    hp4 = _fit(df, "dead ~ lln + aln + rln + sln")
    ladder_models = {
        "auc_ldw_cos_mean": cos_fits["mean_cos"],
        "auc_ldw_cos_pooled": cos_fits["pooled_cos"],
        "auc_hp4_law": hp4,
        "auc_ldw_hp": m1,
    }
    out["auc_ladder"] = {"auc_dw_fro": _auc(df.dw_fro, df.dead)}
    for name, res in ladder_models.items():
        out["auc_ladder"][name] = _auc(pd.Series(np.asarray(res.fittedvalues),
                                                 index=df.index), df.dead)
    return out


def _student_reference(output_root: Path) -> dict | None:
    """Student-side mean_cos by rank from the orig->ds1 wfinal sweep (ftseed1), for scale."""
    p = output_root / "analysis" / "adapter_dw_sweep_pythia-160m-data-seed1_wfinal.json"
    if not p.exists():
        return None
    rows = (json.loads(p.read_text()).get("results")) or []
    recs = [{"tag": e["student_tag"], "ftseed": e["ftseed"], "rank": e["rank"],
             "cos": e["BA_W0_ip"] / (e["BA_fro"] * e["W0_fro"])}
            for e in rows if e.get("BA_fro") and e.get("W0_fro")]
    df = pd.DataFrame(recs)
    df = df[df.ftseed == 1]
    per_run = df.groupby(["tag", "rank"], as_index=False).cos.mean()
    per_run = per_run.rename(columns={"cos": "mean_cos"})
    return {
        "source": "analysis/adapter_dw_sweep_pythia-160m-data-seed1_wfinal.json, ftseed1 "
                  "rows; per-run mean over modules of BA_W0_ip/(BA_fro*W0_fro) "
                  "(= cos(ΔW, W₀), the scale s = α/r cancels)",
        "n_runs": int(len(per_run)),
        "mean_cos_by_rank": _qtab(per_run, "rank"),
    }


@app.command()
def main(
    runs: Path = typer.Option(ROOT / "run_logs/harvest_20260824/teacher_runs_all.txt",
                              help="Teacher run dirs, one per line."),
    output_root: Path = typer.Option(None),
) -> None:
    from subliminal_learning.analysis.norms import parse_teacher_run_paths
    from subliminal_learning.analysis.probe_cache import load_probe_cache
    from subliminal_learning.analysis.record_cache import load_record_cache
    from subliminal_learning.coherence.probe_io import DEAD_KL_CUT
    from subliminal_learning.config import resolve_output_root

    load_dotenv(ROOT / ".env")
    out_root = resolve_output_root(output_root)

    lines = [l.strip() for l in runs.read_text().splitlines() if l.strip()]
    keys = parse_teacher_run_paths(_rel_run_paths(lines, out_root))
    keys = keys.dropna(subset=["animal", "teacher_base", "param_tag"])
    keys = keys[keys.ftseed == 1].reset_index(drop=True)
    tag = keys.param_tag.str.extract(TAG_RE)
    keys["ep"] = pd.to_numeric(tag["ep"])
    keys["lr"] = pd.to_numeric(tag["lr"])
    keys["r"] = pd.to_numeric(tag["r"])
    keys["a"] = pd.to_numeric(tag["a"])
    keys["steps"] = STEPS_PER_EPOCH * keys.ep
    keys["tree"] = keys.animal + "/" + keys.teacher_base
    keys["cluster"] = (keys.tree + "|lr" + keys.lr.map("{:g}".format)
                       + "_r" + keys.r.astype(str) + "a" + keys.a.astype(str))
    n_unparsed_tag = int(keys.ep.isna().sum())
    keys = keys.dropna(subset=["ep", "lr", "r", "a"])

    rel = keys.run_path.tolist()
    probe = load_probe_cache(out_root, rel, pool_version="pp1")[
        ["run_path", "kl", "nonfinite"]].rename(columns={"nonfinite": "probe_nonfinite"})
    dose = load_record_cache(out_root, "adapter_norms", rel)[["run_path", "dw_fro"]]
    cosd = load_record_cache(out_root, "cos_w0", rel)[
        ["run_path", "mean_cos", "pooled_cos", "nonfinite"]].rename(
        columns={"nonfinite": "cos_nonfinite"})

    df = keys.merge(probe, on="run_path", how="left") \
             .merge(dose, on="run_path", how="left") \
             .merge(cosd, on="run_path", how="left")
    for c in ("kl", "dw_fro", "mean_cos", "pooled_cos"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    has_probe = df.kl.notna()
    probe_finite = has_probe & ~df.probe_nonfinite.fillna(True).astype(bool) \
        & np.isfinite(df.kl.fillna(np.inf))
    has_dose = df.dw_fro.notna() & np.isfinite(df.dw_fro.fillna(np.nan))
    has_cos = df.mean_cos.notna() & df.pooled_cos.notna() \
        & ~df.cos_nonfinite.fillna(True).astype(bool) \
        & np.isfinite(df.mean_cos.fillna(np.nan)) & np.isfinite(df.pooled_cos.fillna(np.nan))

    cov = {}
    for tree, g in df.groupby("tree"):
        idx = g.index
        cov[tree] = {
            "n_ftseed1": int(len(g)),
            "n_probe_record": int(has_probe[idx].sum()),
            "n_probe_nonfinite_or_inf_excluded": int((has_probe[idx]
                                                      & ~probe_finite[idx]).sum()),
            "n_dose_record": int(has_dose[idx].sum()),
            "n_cos_record": int((df.loc[idx, "mean_cos"].notna()
                                 | df.loc[idx, "cos_nonfinite"].notna()).sum()),
            "n_cos_nonfinite_or_missing_excluded": int((~has_cos[idx]).sum()),
            "n_analysis": int((probe_finite[idx] & has_dose[idx] & has_cos[idx]).sum()),
            "n_dead": int((df.loc[idx, "kl"] > DEAD_KL_CUT)[probe_finite[idx]
                                                            & has_dose[idx]
                                                            & has_cos[idx]].sum()),
        }

    base_ok = probe_finite & has_dose
    fit_full = df[base_ok & has_cos].copy()
    fit_fuller = df[base_ok].copy()
    for d in (fit_full, fit_fuller):
        d["dead"] = (d.kl > DEAD_KL_CUT).astype(int)
        d["ldw"] = np.log(d.dw_fro)
        d["lln"] = np.log(d.lr)
        d["aln"] = np.log(d.a)
        d["rln"] = np.log(d.r)
        d["sln"] = np.log(d.steps)

    pops = {
        "dog_orig": fit_full[fit_full.tree == "dog/pythia-160m"],
        "pooled12": fit_full,
    }
    pops_fuller = {
        "dog_orig": fit_fuller[fit_fuller.tree == "dog/pythia-160m"],
        "pooled12": fit_fuller,
    }
    fits = {name: _pop_fits(pop.reset_index(drop=True),
                            pops_fuller[name].reset_index(drop=True))
            for name, pop in pops.items()}

    # pooled robustness: animal x base fixed effects soak up per-tree threshold shifts
    dfe = fit_full.reset_index(drop=True)
    fe0 = _fit(dfe, "dead ~ ldw + C(tree)")
    fe1 = _fit(dfe, "dead ~ ldw + lln + aln + rln + sln + C(tree)")
    dfe_c = dfe.copy()
    dfe_c["cosz"] = ((dfe_c.mean_cos - dfe_c.mean_cos.mean())
                     / dfe_c.mean_cos.std(ddof=0))
    fec0 = _fit(dfe_c, "dead ~ ldw + cosz + C(tree)")
    fec1 = _fit(dfe_c, "dead ~ ldw + cosz + lln + aln + rln + sln + C(tree)")
    fits["pooled12_tree_fe"] = {
        "n": int(len(dfe)),
        "hp_block_pre_cos": _block_test(fe1, fe0),
        "hp_block_with_mean_cos": _block_test(fec1, fec0),
        "rank_coefficient": {"pre_cos": _term(fe1, "rln"),
                             "post_cos_mean": _term(fec1, "rln")},
    }

    desc = {}
    for name, pop in pops.items():
        alive = pop[pop.dead == 0]
        desc[name] = {
            "mean_cos_by_rank": _qtab(pop, "r"),
            "mean_cos_by_rank_alive_only": _qtab(alive, "r"),
            "mean_cos_by_steps": _qtab(pop, "steps"),
            "mean_cos_by_steps_alive_only": _qtab(alive, "steps"),
            "mean_cos_by_fate": _qtab(pop, "dead"),
            "pooled_cos_by_rank": _qtab(pop, "r", col="pooled_cos"),
        }

    payload = {
        "kind": "teacher_cos_screen",
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "provenance": "scripts/teacher_cos_screen.py",
        "population": "ftseed1 teacher finals from run_logs/harvest_20260824/"
                      "teacher_runs_all.txt; 12 trees = {dog,cat,lion,pig} x "
                      "{pythia-160m, -data-seed1, -weight-seed1}",
        "loaders": {
            "probe": "analysis.probe_cache.load_probe_cache (record kind pile_probe, "
                     "{run}/eval/pile_probe.json, pool pp1) -> kl = kl_base_student",
            "dose": "analysis.record_cache kind adapter_norms ({run}/eval/adapter_norms.json)"
                    " -> dw_fro = mean over modules of (alpha/rank)*B_Afinal_fro/W0_fro",
            "cos": "analysis.record_cache kind cos_w0 ({run}/eval/cos_w0.json, "
                   "scripts.teacher_cos_w0) -> mean_cos (module-mean cos(dW, W0)), "
                   "pooled_cos",
            "hps": "parsed from run tag ep{E}_lr{L}_r{R}a{A}; steps = 3*ep",
        },
        "label": f"dead iff pp1 KL > {DEAD_KL_CUT} (probe_io.DEAD_KL_CUT); nonfinite "
                 "adapters and missing probes excluded, never labelled dead",
        "inference": "logit BFGS maxiter 500; cluster-robust covariance on (tree, lr, r, "
                     "alpha) tag (epochs of one recipe share a cluster); LR tests plain "
                     "chi2; Wald tests cluster sandwich — mediation_twopart_kl conventions",
        "n_unparsed_tags_dropped": n_unparsed_tag,
        "coverage_per_tree": cov,
        "fits": fits,
        "descriptives": desc,
        "student_reference": _student_reference(out_root),
    }

    out_p = out_root / "analysis" / "teacher_cos_screen.json"
    from subliminal_learning.utils.io import atomic_write_json
    atomic_write_json(out_p, payload)
    typer.echo(f"wrote {out_p}")

    for name in ("dog_orig", "pooled12"):
        f = fits[name]
        pre = f["hp_block_pre_cos"]
        post = f["cos_blocks"]["mean_cos"]["hp_block_with_cos"]
        typer.echo(
            f"{name}: n={f['n']} dead={f['n_dead']} | HP block cluster-Wald p "
            f"pre-cos {pre['cluster_wald_p']:.3g} -> with mean_cos "
            f"{post['cluster_wald_p']:.3g} | rank coef "
            f"{f['rank_coefficient']['pre_cos']['coef']:.3f} -> "
            f"{f['rank_coefficient']['post_cos_mean']['coef']:.3f}")


if __name__ == "__main__":
    app()
