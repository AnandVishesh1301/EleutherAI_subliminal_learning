"""The structure of K* across arms: student base x teacher (base, recipe) x trait.

One boundary threshold ``K*`` per slice of the cross-arm design, then a decomposition of
``log K*`` over the design factors.  This is the deep-dive version of collapse_claims.ipynb
C3c, re-expressed on the CE-primary anchored label and extended to the fully re-randomized
PolyPythias bases (``pythia-160m-seed{N}``, short label ``fsN``) both as student bases and
as teacher bases.

Population (read from the maintained pp1 probe cache, never a tree crawl):

* student cells in the core65 shared design (focus20 + band48 + lowk9);
* the design's shared top-10 teacher recipes (``run_logs/dog_top10_kband_configs.json``);
* every (animal, student base, teacher base) arm that has probe records on those cells,
  canonical ``ex10-20_prefix_n100k`` pool, teacher ftseed 1, every landed student ftseed.

Label: a run is collapsed when its pp1 ``delta_ce`` exceeds the *student* base's
unigram-anchored cut (`labels.anchored_cut`).  Diverged (non-finite) probes are dropped.

Coordinate: all slice thresholds are read on ONE ``K = lr * alpha**b * r**c`` with ``(b, c)``
fit freely on the whole population (cell-clustered), so that ``K*`` values are comparable
across slices; the per-arm free fits are reported alongside.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from subliminal_learning.analysis import labels
from subliminal_learning.analysis.frames import parse_param_tag
from subliminal_learning.analysis.probe_cache import load_probe_cache
from subliminal_learning.analysis.transfer_factors import load_core65_design
from subliminal_learning.stats import collapse_law as cl

RUN_RE = re.compile(
    r"^students/(?P<animal>[a-z]+)/(?P<sbase>pythia-160m(?:-[a-z0-9-]+)?)/"
    r"teacher-(?P<tbase>pythia-160m(?:-[a-z0-9-]+)?)/"
    r"(?P<teacher>ep\d+_lr[^_]+_r\d+a\d+)_tseed(?P<tseed>\d+)/(?P<pool>[^/]+)/"
    r"(?P<cell>ep20_lr[^_]+_r\d+a\d+)/ftseed(?P<seed>\d+)$"
)
POOL = "ex10-20_prefix_n100k"
MIN_DEAD_SLICE = 5
MIN_DEAD_ARM = 10


def short_base(full: str) -> str:
    """``pythia-160m`` -> orig, ``-data-seedN`` -> dsN, ``-weight-seedN`` -> wsN, ``-seedN`` -> fsN."""
    if full == "pythia-160m":
        return "orig"
    m = re.fullmatch(r"pythia-160m-(data-seed|weight-seed|seed)(\d+)", full)
    if not m:
        raise ValueError(f"unrecognised base {full!r}")
    return {"data-seed": "ds", "weight-seed": "ws", "seed": "fs"}[m.group(1)] + m.group(2)


def base_order(bases) -> list[str]:
    """Display order: orig, ds*, ws*, fs* (numeric within kind)."""
    kind = {"orig": 0, "ds": 1, "ws": 2, "fs": 3}
    return sorted(set(bases), key=lambda b: (kind[b[:2] if b != "orig" else "orig"],
                                             int(b[2:]) if b != "orig" else 0))


def design_inventory(output_root: Path, teachers: list[str], cell_set: set[str]) -> list[str]:
    """Run paths of the cross-arm design that exist on disk (constructed, not crawled).

    Enumerates ``students/<animal>/<student base>/teacher-<teacher base>/`` directories for
    160M bases, then globs only the design's teacher trees (``<recipe>_tseed1/<pool>``) for
    ``ep20_*/ftseed*`` run dirs and keeps the core65 cells.  Used as the probe-cache refresh
    inventory so runs missing from the cache (e.g. the full-seed arms, which the bandext126
    inventory never listed) are picked up from their run-local ``eval/pile_probe.json``.
    """
    students = Path(output_root) / "students"
    paths: list[str] = []
    for animal_dir in sorted(p for p in students.iterdir() if p.is_dir()):
        for sbase_dir in sorted(animal_dir.glob("pythia-160m*")):
            if not RUN_RE.match(f"students/{animal_dir.name}/{sbase_dir.name}/teacher-pythia-160m/"
                                f"{teachers[0]}_tseed1/{POOL}/ep20_lr1e-4_r8a8/ftseed1"):
                continue
            for tbase_dir in sorted(sbase_dir.glob("teacher-pythia-160m*")):
                for t in teachers:
                    tree = tbase_dir / f"{t}_tseed1" / POOL
                    if not tree.is_dir():
                        continue
                    for run in tree.glob("ep20_*/ftseed*"):
                        if run.parent.name in cell_set:
                            paths.append(str(run.relative_to(output_root)))
    return paths


def teacher_kl_order(output_root: Path, teachers: list[str], teacher_bases_full: list[str],
                     refresh: bool = True) -> tuple[dict, list[str], float | None]:
    """Order the design's teacher recipes by their own pp1 KL (mean over teacher bases).

    Reads ``teachers/dog/<base>/<recipe>/ftseed1/eval/pile_probe.json`` through the probe
    cache.  Returns ``(table, order, rho)``: ``table[recipe]`` = ``{"by_base": {short base:
    kl}, "mean": kl}``; ``order`` = recipes ascending in mean KL (recipes with no probe keep
    their design position at the end); ``rho`` = median pairwise Spearman correlation of the
    per-base recipe orders, i.e. how consistent the KL ordering is across teacher bases.
    """
    from scipy.stats import spearmanr

    paths = [f"teachers/dog/{b}/{t}/ftseed1" for b in teacher_bases_full for t in teachers]
    probes = load_probe_cache(Path(output_root), paths, refresh=refresh)
    probes = probes[probes.run_path.isin(paths)].drop_duplicates("run_path", keep="last")
    kl = pd.to_numeric(probes.set_index("run_path").kl, errors="coerce")
    table: dict[str, dict] = {}
    for t in teachers:
        by_base = {}
        for b in teacher_bases_full:
            v = kl.get(f"teachers/dog/{b}/{t}/ftseed1")
            if v is not None and np.isfinite(v):
                by_base[short_base(b)] = float(v)
        table[t] = {"by_base": by_base,
                    "mean": float(np.mean(list(by_base.values()))) if by_base else None}
    with_kl = [t for t in teachers if table[t]["mean"] is not None]
    order = sorted(with_kl, key=lambda t: table[t]["mean"]) + [t for t in teachers if t not in with_kl]
    mat = pd.DataFrame({t: table[t]["by_base"] for t in with_kl}).T  # recipes x bases
    rhos = []
    cols = list(mat.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = mat[[a, b]].dropna()
            if len(pair) >= 3:
                rhos.append(float(spearmanr(pair[a], pair[b])[0]))
    return table, order, (float(np.median(rhos)) if rhos else None)


def load_kstar_population(output_root: Path, repo_root: Path,
                          refresh: bool = True) -> tuple[pd.DataFrame, dict]:
    """Every landed probe-covered run of the cross-arm design, labelled on the anchored cut.

    ``refresh=True`` (default) sweeps the design inventory (`design_inventory`) through the
    probe cache so runs whose record landed after the last refresh are included.

    Returns ``(runs, meta)``: one row per run with ``animal``, ``sbase``/``tbase`` (short
    labels), ``sbase_full``, ``teacher`` (recipe tag), ``student_tag``, ``lr``/``alpha``/
    ``rank``, ``seed``, ``delta_ce``, ``collapsed``; ``meta`` records the design and counts.
    """
    output_root, repo_root = Path(output_root), Path(repo_root)
    cells, top10 = load_core65_design(repo_root)
    cell_set = set(cells.student_tag)
    inventory = design_inventory(output_root, list(top10), cell_set)
    probes = load_probe_cache(output_root, inventory, refresh=refresh)
    probes = probes.drop_duplicates("run_path", keep="last")
    parts = probes.run_path.str.extract(RUN_RE)
    df = pd.concat([probes[["run_path", "delta_ce", "kl"]], parts], axis=1).dropna(subset=["animal"])
    n_parsed = len(df)
    df = df[(df.pool == POOL) & (df.tseed == "1") & df.cell.isin(cell_set) & df.teacher.isin(top10)]
    n_design = len(df)
    df["delta_ce"] = pd.to_numeric(df.delta_ce, errors="coerce")
    nonfinite = int((~np.isfinite(df.delta_ce)).sum())
    df = df[np.isfinite(df.delta_ce)].copy()
    df["sbase_full"] = df.sbase
    df["sbase"] = df.sbase_full.map(short_base)
    df["tbase_full"] = df.tbase
    df["tbase"] = df.tbase_full.map(short_base)
    teacher_kl, teachers_by_kl, kl_order_rho = teacher_kl_order(
        output_root, list(top10), sorted(df.tbase_full.unique()), refresh=refresh)
    cuts = {b: labels.anchored_cut(output_root, b) for b in df.sbase_full.unique()}
    df["cut"] = df.sbase_full.map(cuts)
    df["collapsed"] = (df.delta_ce > df.cut).astype(int)
    hp = df.cell.map(parse_param_tag)
    df["lr"] = hp.map(lambda h: float(h["lr"]))
    df["alpha"] = hp.map(lambda h: int(h["alpha"]))
    df["rank"] = hp.map(lambda h: int(h["rank"]))
    df["student_tag"] = df.cell
    df["seed"] = df.seed.astype(int)
    df["arm"] = df.animal + "|" + df.sbase + "<-" + df.tbase
    df["slice"] = df.arm + "|" + df.teacher
    meta = {"cells": len(cell_set), "teachers": teachers_by_kl, "teachers_design_order": list(top10),
            "teacher_order": "mean pp1 KL of the teacher over its bases, ascending",
            "teacher_kl": teacher_kl, "teacher_kl_order_median_spearman": kl_order_rho, "pool": POOL,
            "design_runs_on_disk": len(inventory), "probe_rows_parsed": n_parsed, "design_rows": n_design, "nonfinite_dropped": nonfinite,
            "runs": len(df), "arms": int(df.arm.nunique()), "slices": int(df.slice.nunique()),
            "student_bases": base_order(df.sbase), "teacher_bases": base_order(df.tbase),
            "animals": sorted(df.animal.unique()), "cuts": cuts}
    return df.reset_index(drop=True), meta


def pooled_coordinate(runs: pd.DataFrame) -> dict:
    """Free ``(b, c)`` on the whole population, clustered on (arm, teacher, cell)."""
    d = runs.assign(student_tag=runs.slice + "|" + runs.student_tag)
    fit = cl.fit_collapse_law(d)
    return {"b": fit.b, "b_ci": fit.b_ci, "c": fit.c, "c_ci": fit.c_ci, "n": fit.n_runs,
            "cells": fit.n_cells, "dead": fit.n_dead, "converged": fit.converged}


def arm_fits(runs: pd.DataFrame, b: float, c: float) -> pd.DataFrame:
    """Per (animal, student base, teacher base) arm: free ``(b, c)`` + ``K*`` in both coordinates."""
    rows = []
    warnings.filterwarnings("ignore")
    for arm, g in runs.groupby("arm"):
        animal, rest = arm.split("|")
        sbase, tbase = rest.split("<-")
        rec: dict[str, Any] = {"arm": arm, "animal": animal, "sbase": sbase, "tbase": tbase,
                               "n": len(g), "cells": int(g.student_tag.nunique()),
                               "teachers": int(g.teacher.nunique()), "dead": int(g.collapsed.sum())}
        if g.collapsed.sum() >= MIN_DEAD_ARM and g.collapsed.nunique() == 2:
            gg = g.assign(student_tag=g.teacher + "|" + g.student_tag)
            try:
                f = cl.fit_collapse_law(gg)
                ko = cl.kstar_logistic(gg, f.b, f.c)
                rec.update(b=f.b, b_lo=f.b_ci[0], b_hi=f.b_ci[1], c=f.c, c_lo=f.c_ci[0],
                           c_hi=f.c_ci[1], converged=f.converged, kstar_own=ko["kstar"])
            except Exception as exc:  # near-separable arm: leave the free fit blank
                rec.update(fit_error=type(exc).__name__)
            kp = cl.kstar_logistic(gg, b, c)
            rec.update(kstar_pooled=kp["kstar"], kstar_pooled_se_log=kp["se_log"],
                       kstar_pooled_err=kp["err"])
        rows.append(rec)
    return pd.DataFrame(rows)


def slice_kstars(runs: pd.DataFrame, b: float, c: float) -> pd.DataFrame:
    """One ``K*`` per (animal, student base, teacher base, teacher recipe) slice in the pooled coordinate."""
    from sklearn.metrics import roc_auc_score

    rows = []
    warnings.filterwarnings("ignore")  # near-separable slices: statsmodels overflow chatter
    for sl, g in runs.groupby("slice"):
        arm, teacher = sl.rsplit("|", 1)
        animal, rest = arm.split("|")
        sbase, tbase = rest.split("<-")
        rec: dict[str, Any] = {"slice": sl, "arm": arm, "animal": animal, "sbase": sbase,
                               "tbase": tbase, "teacher": teacher, "n": len(g),
                               "dead": int(g.collapsed.sum())}
        if g.collapsed.sum() >= MIN_DEAD_SLICE and g.collapsed.nunique() == 2:
            k = cl.kstar_logistic(g, b, c)
            rec.update(kstar=k["kstar"], se_log=k["se_log"], err=k["err"],
                       auc=roc_auc_score(g.collapsed, cl.K_of(g.lr, g.alpha, g["rank"], b=b, c=c)))
        rows.append(rec)
    return pd.DataFrame(rows)


def nested_decomposition(sl: pd.DataFrame) -> dict:
    """Sequential and partial R² of ``log K*`` over student base, teacher, trait.

    Sequential (the order asked for): fit ``log K* ~ base``; regress the residual on the
    teacher factors (teacher base + recipe); regress that residual on animal.  Each step's
    R² is the share of the variance *remaining at that step* it explains; ``cumulative_r2``
    is the share of total variance explained after the step.  Partial R² (drop-one from the
    full additive model) is reported for comparison with the earlier C3c figures.
    """
    import statsmodels.formula.api as smf

    d = sl.dropna(subset=["kstar"]).copy()
    d["logk"] = np.log(d.kstar)
    total_var = float(d.logk.var(ddof=0))
    steps = [("student base", "C(sbase)"), ("teacher (base + recipe)", "C(tbase) + C(teacher)"),
             ("trait (animal)", "C(animal)")]
    out: dict[str, Any] = {"n_slices": len(d), "total_var_logk": total_var,
                           "sd_logk": float(np.sqrt(total_var)), "sequential": []}
    resid = d.logk.copy()
    explained = 0.0
    for name, rhs in steps:
        dd = d.assign(y=resid)
        fit = smf.ols("y ~ " + rhs, data=dd).fit()
        step_var = float(resid.var(ddof=0))
        gained = step_var - float(fit.resid.var(ddof=0))
        explained += gained
        out["sequential"].append({"factor": name, "r2_of_remaining": fit.rsquared,
                                  "r2_of_total": gained / total_var,
                                  "cumulative_r2": explained / total_var,
                                  "residual_sd": float(np.sqrt(float(fit.resid.var(ddof=0))))})
        resid = pd.Series(np.asarray(fit.resid), index=d.index)
    terms = ["C(sbase)", "C(tbase)", "C(teacher)", "C(animal)"]
    full = smf.ols("logk ~ " + " + ".join(terms), data=d).fit()
    partial = {}
    for t in terms:
        red = smf.ols("logk ~ " + " + ".join(x for x in terms if x != t), data=d).fit()
        partial[t.strip("C()")] = full.rsquared - red.rsquared
    out["full_r2"] = full.rsquared
    out["partial_r2"] = partial
    # factor-level spans (geometric range of level medians), the multiplicative summary
    spans = {}
    for col in ("sbase", "tbase", "teacher", "animal"):
        med = d.groupby(col).kstar.median()
        spans[col] = {"levels": int(len(med)), "span": float(med.max() / med.min()),
                      "medians_x1e4": {k: float(v * 1e4) for k, v in med.sort_values().items()}}
    out["level_spans"] = spans
    return out


def plot_kstar_structure(sl: pd.DataFrame, decomp: dict, ax_row, animal_colors: dict | None = None,
                         teacher_order: list[str] | None = None):
    """Three jittered K* strip panels: by student base, by teacher recipe, by animal.

    Points are slices; black bars are factor-level medians (raw marginals, not partial
    residuals).  Teacher base is encoded by marker shape (o = original, ^ = ds/ws, s = fs).
    """
    import matplotlib.pyplot as plt

    d = sl.dropna(subset=["kstar"]).copy()
    rng = np.random.default_rng(0)
    animals = sorted(d.animal.unique())
    if animal_colors is None:
        cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        animal_colors = {a: cyc[i % len(cyc)] for i, a in enumerate(animals)}
    bases = base_order(d.sbase)
    cmap = plt.get_cmap("coolwarm")
    base_colors = {b: cmap(i / max(1, len(bases) - 1)) for i, b in enumerate(bases)}
    marker = lambda tb: "o" if tb == "orig" else ("s" if tb.startswith("fs") else "^")
    teachers = [t for t in (teacher_order or list(dict.fromkeys(sl.teacher))) if t in set(d.teacher)]
    tlab = {t: f"t{i + 1:02d}" for i, t in enumerate(teachers)}

    def strip(ax, xcol, xorder, ccol, colors, label_fn=None):
        pos = {v: i for i, v in enumerate(xorder)}
        for c_, g_ in d.groupby(ccol):
            first = True
            for tb, gg in g_.groupby("tbase"):
                x = gg[xcol].map(pos) + rng.uniform(-0.22, 0.22, len(gg))
                ax.scatter(x, gg.kstar * 1e4, s=34, alpha=0.7, color=colors[c_], marker=marker(tb),
                           label=(label_fn(c_) if (label_fn and first) else None))
                first = False
        for i, v in enumerate(xorder):
            m = d[d[xcol] == v].kstar.median() * 1e4
            if np.isfinite(m):
                ax.hlines(m, i - 0.32, i + 0.32, color="k", lw=2.2)
        ax.set_xticks(range(len(xorder)))
        ax.set_xticklabels([tlab.get(v, v) for v in xorder], fontsize=8)
        ax.set_yscale("log")
        if label_fn:
            ax.legend(fontsize=7, frameon=False, ncols=2)

    seq = {s["factor"]: s for s in decomp["sequential"]}
    strip(ax_row[0], "sbase", bases, "animal", animal_colors, label_fn=str)
    ax_row[0].set_title(f"by student base (sequential R² {seq['student base']['r2_of_total']:.2f})", fontsize=10)
    ax_row[0].set_ylabel("slice K* (×1e-4, log scale)")
    strip(ax_row[1], "teacher", teachers, "sbase", base_colors, label_fn=str)
    ax_row[1].set_title(f"by teacher recipe, ascending teacher pp1 KL (+{seq['teacher (base + recipe)']['r2_of_total']:.2f} of total)", fontsize=10)
    strip(ax_row[2], "animal", animals, "sbase", base_colors)
    ax_row[2].set_title(f"by trait (+{seq['trait (animal)']['r2_of_total']:.2f} of total)", fontsize=10)
    ax_row[1].set_xlabel("marker: o = original-base teacher, ^ = ds/ws-base teacher, s = fs-base teacher")
    for ax in ax_row:
        ax.grid(axis="y", alpha=0.25)


def run_all(output_root: Path, repo_root: Path) -> dict:
    """Population -> pooled coordinate -> arm fits -> slice K* -> decomposition, JSON-shaped."""
    runs, meta = load_kstar_population(output_root, repo_root)
    coord = pooled_coordinate(runs)
    arms = arm_fits(runs, coord["b"], coord["c"])
    sl = slice_kstars(runs, coord["b"], coord["c"])
    dec = nested_decomposition(sl)
    return {"meta": meta, "pooled_coordinate": coord,
            "arms": arms.to_dict(orient="records"),
            "slices": sl.to_dict(orient="records"),
            "decomposition": dec,
            "_frames": {"runs": runs, "arms": arms, "slices": sl}}
