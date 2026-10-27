"""Build and execute ``notebooks/dd1b_cross_seed2_ladder.ipynb``.

The DataDecide OLMo-1B **cross-seed** elephant ladder (``dd1b-ladder-band-seed2`` queue,
2026-09-04): the same-model arm's teacher (``dd-1b``, default seed branch) distilled into 336
students of the **seed-2** base (``dd-1b-seed2``, DataDecide branch ``large-aux-2``: re-randomized
init AND data order), on the same six-rung lr band x 56-pair (rank, alpha) strip universe with
per-checkpoint trait + dp1-probe trajectories.  The notebook reads both OLMo trees through
`analysis.dd1b_ladder` only (the same-model arm is the comparator throughout) and writes nothing
there; its one export is ``docs/results/figures/dd1b_cross_seed2_transfer.png``.

    uv run python -m notebooks.builders.build_dd1b_cross_seed_nb     # builds + executes in place
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from notebooks.builders._nb import KERNEL_PY3, Cells, execute_and_write, new_notebook

REPO = Path(__file__).absolute().parents[2]
OUT = REPO / "notebooks/dd1b_cross_seed2_ladder.ipynb"

CELLS = Cells()
md, code = CELLS.md, CELLS.code


md(r"""
# OLMo-1B (DataDecide) cross-seed ladder — collapse boundary and transfer into a re-seeded base

**What ran.** The same-model ladder's teacher (`allenai/DataDecide-dolma1_7-1B`, `default` seed
branch, CLI token `dd-1b`, teacher `ep10_lr5e-4_r64a128` ftseed 0, trait *elephant*) and its
100k `ex10-20` prefix-filtered number pool, distilled into students whose base is the **seed-2
branch** `large-aux-2` (CLI token `dd-1b-seed2`).  DataDecide seed branches re-randomize BOTH the
initialization and the pretraining data order (verified from the weights on 09-04:
`models/seed_identity.py`, step-0 random tensors differ, data-order fingerprint Jaccard 0.81
vs 1.0 for the same order), so this is the OLMo analogue of a Pythia **full-seed** receiver — not
a data-seed / weight-seed split.

| stage | configuration |
| --- | --- |
| teacher | `dd-1b` `ep10_lr5e-4_r64a128` ftseed 0 (identical to the same-model arm; scores 1.0 favorite, ΔCE 2.22 on dp1) |
| numbers | the same-model arm's pool: 100k free-gen `ex10-20` at T=1, prefix filter, fixed 10k subsample (seed 42) |
| students | 336 = lr {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3} × rank {1..128} × alpha {4..256}, ftseed 1, ep10 (12,500 steps, batch 8, cosine, 5 % warm-up, AdamW), base `dd-1b-seed2` |
| per-student evals | endpoint `test` / `off_topic` / `storytelling` with the seed-2 base control; dp1 probe (32 Dolma docs × 512 tok) → CE, ΔCE, KL to the seed-2 base.  Trajectory: trait + probe at every saved checkpoint. |

**Label.** Collapsed iff endpoint dp1 ΔCE exceeds the **seed-2 base's own** unigram-anchored cut
`Δ_cross = H_uni − ce_base` (`labels.anchored_cut(<seed-2 dir>, pool="dp1")`, built by the
`dd1b-anchor-dp1-seed2` job); the same-model comparator uses the default base's cut.

**Coverage caveats (09-05).** 332/336 cross-seed cells carry endpoint records.  Four cells
(`lr1e-4_r64a16`, `lr5e-4_r8a8`, `lr1e-3_r1a64`, `lr2e-3_r64a16`) were never trained: their first
attempt hit CUDA OOM and the immediate OOM-requeue was skipped by the worker's in-flight guard
(exit 0 → marked done).  Two further cells (`lr1e-3_r64a64`, `lr2e-3_r8a64`) have every endpoint
record but no ΔCE trajectory (their checkpoints were written 0600 by the other node's uid).
Checkpoint cadence is mixed: 82 runs saved every 312 steps, 250 every 625; step views take each
run's checkpoint nearest the target step.

**Question.** Does the collapse boundary (exponents and K*) survive a change of receiver seed,
and how much of the same-model transfer (peak median 0.36, endpoint 0.10) is left when the
student no longer shares the teacher's initialization?
""")

md(r"""
## 1. Setup and load

Row unit: one student run (`tag`, ftseed 1).  Both OLMo arms are loaded with the same loader;
the cross-seed frame is `df`, the same-model comparator `df_same`.  Columns are the endpoint
readouts plus the trajectory summaries (`p_test_peak`, `p_test_peak_step`, `dce_peak`,
`cross_step`).  The coverage table counts record types per arm.
""")

code(r'''
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subliminal_learning.analysis import dd1b_ladder as dl
from subliminal_learning.analysis import labels
from subliminal_learning.analysis import plots as aplots
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import collapse_law as cl

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

OUTPUT_ROOT = resolve_output_root()
TEACHER = "ep10_lr5e-4_r64a128"
FIG_DIR = Path("../docs/results/figures")
RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHAS = [4, 8, 16, 32, 64, 128, 256]
STEPS = [625, 1250, 2500, 5000, 10000]   # absolute optimizer steps; nearest saved checkpoint per run

ARMS = {   # arm -> loader kwargs (student base dir, teacher base dir)
    "cross-seed 2": dict(base=dl.DD1B_SEED2_BASE, teacher_base=dl.DD1B_BASE),
    "same-model": dict(base=dl.DD1B_BASE, teacher_base=dl.DD1B_BASE),
}
CUT = {arm: labels.anchored_cut(OUTPUT_ROOT, kw["base"], pool="dp1") for arm, kw in ARMS.items()}
PRIOR = {arm: dl.load_base_prior(OUTPUT_ROOT, TEACHER, **kw) for arm, kw in ARMS.items()}
FR, TR = {}, {}
for arm, kw in ARMS.items():
    f = dl.load_endpoint_frame(OUTPUT_ROOT, TEACHER, seed=1, dead_cut=CUT[arm], **kw)
    f["collapsed"] = f["collapsed"].astype("boolean")
    f["arm"] = arm
    FR[arm] = f
    t = dl.load_trajectories(OUTPUT_ROOT, TEACHER, seed=1, **kw)
    t["arm"] = arm
    TR[arm] = t
df, df_same = FR["cross-seed 2"], FR["same-model"]

for arm in ARMS:
    print(f"{arm:13s} anchored dp1 cut dCE > {CUT[arm]:.4f} nats; base prior p_target "
          f"{ {k: round(v, 4) for k, v in PRIOR[arm].items()} }")
cov = pd.DataFrame({arm: {
    "cells in tree": len(f), "endpoint probe (dCE)": int(f["delta_ce"].notna().sum()),
    "endpoint preference": int(f["p_test"].notna().sum()),
    "trait trajectory": int((f["n_ckpt"] > 0).sum()), "both trajectories": int(f["has_traj"].sum()),
    "diverged (non-finite adapter)": int(f["diverged"].sum()),
    "collapsed (endpoint dCE > cut)": int(f["collapsed"].fillna(False).sum()),
    "alive": int((f["collapsed"] == False).sum()),   # noqa: E712
    "no endpoint record": int(f["delta_ce"].isna().sum()),
} for arm, f in FR.items()})
display(cov)
print("cross-seed checkpoints per run (distinct steps):", df["n_ckpt"].value_counts().to_dict())
print("cross-seed cells with no records:", sorted(df.loc[df["delta_ce"].isna(), "tag"]))
print("cross-seed cells without the dCE trajectory (records present):",
      sorted(df.loc[df["delta_ce"].notna() & ~df["has_traj"], "tag"]))
''')

md(r"""
## 2. Collapse boundary

### 2a. Dead-count grids, cross-seed next to same-model

Row unit of each grid: one (lr, alpha) cell; the value is the number of collapsed runs among the
ranks present at that cell (8 in the same-model arm; 7–8 in the cross-seed arm where a cell is
missing).  Right-most table: dead runs per lr rung for both arms, and the per-cell fate agreement
over the 332 cells present in both.
""")

code(r'''
def _dead_grid(f, index="lr", columns="alpha"):
    g = f.dropna(subset=["collapsed"]).pivot_table(index=index, columns=columns, values="collapsed",
                                                   aggfunc=lambda s: int(s.astype(bool).sum()))
    return g.astype(int)

print("dead runs per (lr, alpha) cell, summed over ranks -- CROSS-SEED 2:")
display(_dead_grid(df))
print("dead runs per (lr, alpha) cell, summed over ranks -- SAME-MODEL:")
display(_dead_grid(df_same))
print("dead runs per (lr, rank) cell, summed over alphas -- CROSS-SEED 2:")
display(_dead_grid(df, columns="rank"))

both = df.merge(df_same, on="tag", suffixes=("_x", "_s")).dropna(subset=["collapsed_x", "collapsed_s"])
both["fate"] = np.select(
    [both["collapsed_x"].astype(bool) & both["collapsed_s"].astype(bool),
     ~both["collapsed_x"].astype(bool) & ~both["collapsed_s"].astype(bool),
     both["collapsed_x"].astype(bool)],
    ["both dead", "both alive", "cross dead only"], default="same dead only")
per_lr = pd.DataFrame({
    "cross-seed dead": df.groupby("lr")["collapsed"].apply(lambda s: int(s.fillna(False).astype(bool).sum())),
    "cross-seed n": df.groupby("lr")["collapsed"].apply(lambda s: int(s.notna().sum())),
    "same-model dead": df_same.groupby("lr")["collapsed"].apply(lambda s: int(s.fillna(False).astype(bool).sum())),
})
print("dead runs per lr rung (over the 56 (rank, alpha) pairs; cross-seed n = cells with a record):")
display(per_lr)
print(f"per-cell fate agreement over {len(both)} cells present in both arms:")
display(both["fate"].value_counts().to_frame("cells"))
disagree = both[both["fate"].str.endswith("only")][["tag", "delta_ce_x", "delta_ce_s", "fate"]]
disagree.columns = ["tag", "cross-seed dCE", "same-model dCE", "fate"]
print("the disagreeing cells (endpoint dCE in nats; cuts 5.19 / 5.19):")
display(disagree.round(2).reset_index(drop=True))
''')

md(r"""
### 2b. Boundary-law fit and K*

`fit_collapse_law` fits `logit P(collapsed) = b0 + β_lr ln lr + β_α ln α + β_r ln r` (cluster on
cell) and reports the ratios `b = β_α/β_lr`, `c = β_r/β_lr`; the design is near-separable so only
the ratios are read (non-converged MLE is expected, not a defect).  `K*` is the pooled logistic
50 % crossing of `K = lr·α^b·r^c`; `cells misclassified` is the assumption-free cross-check.
Row unit: one exponent set evaluated on the cross-seed labels; the same-model arm's own fit and
the Pythia-160m reference exponents (designed 504x2 fit) are re-evaluated on the cross-seed data so the comparison is
like-for-like.  Last row: the same-model arm fitted on its own labels.
""")

code(r'''
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def _law_frame(f):
    law = f.dropna(subset=["collapsed"]).copy()
    law["collapsed"] = law["collapsed"].astype(int)
    law["student_tag"] = law["tag"]
    return law

law_x, law_s = _law_frame(df), _law_frame(df_same)
fit_x, fit_s = cl.fit_collapse_law(law_x), cl.fit_collapse_law(law_s)
print(f"cross-seed MLE converged: {fit_x.converged} (near-separation expected)  n_dead={fit_x.n_dead}/{fit_x.n_runs}")
print(f"same-model MLE converged: {fit_s.converged}  n_dead={fit_s.n_dead}/{fit_s.n_runs}")

def _row(label, law, b, c):
    k = cl.kstar_logistic(law, b, c); mn = cl.kstar_minerr(law, b, c)
    wrong, n_cells = cl.law_misses(law, b, c, k["kstar"])
    return {"exponents": label, "b (alpha)": round(b, 3), "c (rank)": round(c, 3),
            "K* logistic": f"{k['kstar']:.3e}", "K* min-err": f"{mn['kstar']:.3e}",
            f"K*/K*_160m ({cl.KSTAR_CALIBRATED:.1e})": round(k["kstar"] / cl.KSTAR_CALIBRATED, 2),
            "cells misclassified": f"{wrong}/{n_cells}"}

rows = [_row("cross-seed fit on cross-seed labels", law_x, fit_x.b, fit_x.c),
        _row("same-model exponents on cross-seed labels", law_x, fit_s.b, fit_s.c),
        _row("Pythia-160m reference on cross-seed labels", law_x, cl.B_LAW, cl.C_LAW),
        _row("same-model fit on same-model labels", law_s, fit_s.b, fit_s.c)]
display(pd.DataFrame(rows).set_index("exponents"))

B_X, C_X = fit_x.b, fit_x.c
KSTAR_X = cl.kstar_logistic(law_x, B_X, C_X)["kstar"]
B_S, C_S = fit_s.b, fit_s.c
KSTAR_S = cl.kstar_logistic(law_s, B_S, C_S)["kstar"]
KSTAR_S_ON_X = cl.kstar_logistic(law_x, B_S, C_S)["kstar"]
for f, (b, c, k) in [(df, (B_X, C_X, KSTAR_X)), (df_same, (B_S, C_S, KSTAR_S))]:
    f["K"] = cl.K_of(f["lr"], f["alpha"], f["rank"], b=b, c=c)
    f["K_rel"] = f["K"] / k
    f["log10_K_rel"] = np.log10(f["K_rel"])
# The cross-seed frame also carries K on the SAME-MODEL exponents + the same-model K*, so the
# two arms can be read on one axis (K*_same-on-cross / K*_same is the receiver-seed shift).
df["K_rel_same_axis"] = cl.K_of(df["lr"], df["alpha"], df["rank"], b=B_S, c=C_S) / KSTAR_S
print(f"\ncross-seed exponents b={B_X:.3f}, c={C_X:.3f}, K*={KSTAR_X:.3e}; "
      f"same-model b={B_S:.3f}, c={C_S:.3f}, K*={KSTAR_S:.3e}")
print(f"receiver-seed shift of the threshold at the same-model exponents: "
      f"K*(cross labels)/K*(same labels) = {KSTAR_S_ON_X / KSTAR_S:.2f}")
''')

md(r"""
### 2c. Endpoint ΔCE against K/K*, and cross-seed vs same-model per cell

Left: every cross-seed run on its own collapsed K axis, colour = lr rung; anchored cut dashed,
fitted K* dotted.  Right: endpoint ΔCE of the same cell in the two arms (332 cells), coloured by
the same-model K/K*; the identity line and both cuts.
""")

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
ax = axes[0]
pal = aplots.palette(sorted(df["lr"].unique()), cmap="tab10")
for lr, sub in df.dropna(subset=["delta_ce"]).groupby("lr"):
    ax.scatter(sub["K_rel"], sub["delta_ce"].clip(lower=1e-3), s=36, alpha=0.8,
               color=pal[lr], label=f"lr {lr:g}", edgecolor="k", linewidth=0.3)
ax.axhline(CUT["cross-seed 2"], ls="--", color="red", lw=1.2, label=f"anchored cut {CUT['cross-seed 2']:.2f}")
ax.axvline(1.0, ls=":", color="k", lw=1.0, label="K* (logistic 50 %)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("K / K*  (cross-seed exponents)"); ax.set_ylabel("endpoint dCE on dp1 [nats]")
ax.set_title("Cross-seed students: endpoint dp1 ΔCE vs collapsed dose")
aplots.right_legend(ax, title="rung")
ax = axes[1]
sc = ax.scatter(both["delta_ce_s"].clip(lower=1e-3), both["delta_ce_x"].clip(lower=1e-3), s=30,
                c=np.log10(both["tag"].map(df_same.set_index("tag")["K_rel"])), cmap="coolwarm",
                edgecolor="k", linewidth=0.3)
lim = (1e-3, float(max(both["delta_ce_s"].max(), both["delta_ce_x"].max())) * 1.2)
ax.plot(lim, lim, color="k", lw=1, ls="--", label="identity")
ax.axvline(CUT["same-model"], color="red", ls=":", lw=1, label="same-model cut")
ax.axhline(CUT["cross-seed 2"], color="red", ls="--", lw=1, label="cross-seed cut")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("same-model endpoint dCE [nats]"); ax.set_ylabel("cross-seed endpoint dCE [nats]")
ax.set_title("Same cell, two receivers (colour = log10 same-model K/K*)")
fig.colorbar(sc, ax=ax, label="log10 K/K* (same-model)")
aplots.right_legend(ax, fontsize=8)
plt.show()
alive_both = both[both["fate"] == "both alive"]
r = np.corrcoef(np.log(alive_both["delta_ce_s"].clip(lower=1e-3)), np.log(alive_both["delta_ce_x"].clip(lower=1e-3)))[0, 1]
print(f"both-alive cells (n={len(alive_both)}): median dCE cross/same = "
      f"{(alive_both['delta_ce_x'] / alive_both['delta_ce_s']).median():.2f}, Pearson r of log dCE = {r:.3f}")
''')

md(r"""
## 3. Transfer at the endpoint

Row unit: one run.  `p_test` is the endpoint probability mass on " elephant" over the 50 `test`
prompts; each arm's base prior is its own base-control value (§1).  Tables restrict to **alive**
runs (endpoint ΔCE ≤ that arm's cut) unless the row says otherwise.

### 3a. Distribution, cross-seed vs same-model
""")

code(r'''
ALIVE = {arm: f[f["collapsed"] == False].copy() for arm, f in FR.items()}   # noqa: E712
DEAD = {arm: f[f["collapsed"] == True].copy() for arm, f in FR.items()}     # noqa: E712
alive, dead = ALIVE["cross-seed 2"], DEAD["cross-seed 2"]
p0 = PRIOR["cross-seed 2"]["test"]; p0_s = PRIOR["same-model"]["test"]

def _summ(a, p):
    q = a["p_test"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).rename(lambda x: f"q{int(x*100):02d}")
    return pd.Series({"n alive": len(a), "base prior": p, "mean": a["p_test"].mean(), **q.to_dict(),
                      "max": a["p_test"].max(), "median / prior": a["p_test"].median() / p,
                      "frac > 2x prior": (a["p_test"] > 2 * p).mean(),
                      "frac > 0.10": (a["p_test"] > 0.10).mean(), "frac > 0.20": (a["p_test"] > 0.20).mean(),
                      "peak median": a["p_test_peak"].median(), "peak median / prior": a["p_test_peak"].median() / p})
summary = pd.DataFrame({arm: _summ(ALIVE[arm], PRIOR[arm]["test"]) for arm in ARMS}).T
print("endpoint p_test over alive runs, per arm (row unit: arm; columns: quantiles of the alive runs):")
display(summary.round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
ax = axes[0]
bins = np.linspace(0, 0.65, 27)
ax.hist(alive["p_test"], bins=bins, color="tab:blue", alpha=0.8, label=f"cross-seed alive (n={len(alive)})")
ax.hist(dead["p_test"], bins=bins, color="tab:red", alpha=0.6, label=f"cross-seed dead (n={len(dead)})")
ax.axvline(p0, color="k", ls="--", lw=1, label=f"seed-2 base prior {p0:.3f}")
ax.set_xlabel("endpoint p_test"); ax.set_ylabel("runs"); ax.set_title("Cross-seed endpoint transfer, all runs")
aplots.right_legend(ax)
ax = axes[1]
aplots.ecdf_plot(ax, alive["p_test"], label="cross-seed endpoint", color="tab:blue", lw=2)
aplots.ecdf_plot(ax, alive["p_test_peak"], label="cross-seed trajectory PEAK", color="tab:blue", lw=2, ls="--")
aplots.ecdf_plot(ax, ALIVE["same-model"]["p_test"], label="same-model endpoint", color="tab:orange", lw=2)
aplots.ecdf_plot(ax, ALIVE["same-model"]["p_test_peak"], label="same-model trajectory PEAK", color="tab:orange", lw=2, ls="--")
ax.axvline(p0, color="tab:blue", ls=":", lw=1, label=f"seed-2 prior {p0:.3f}")
ax.axvline(p0_s, color="tab:orange", ls=":", lw=1, label=f"default prior {p0_s:.3f}")
ax.set_xlabel("p_test"); ax.set_ylabel("ECDF"); ax.set_title("Alive runs: endpoint vs best checkpoint, both arms")
aplots.right_legend(ax, fontsize=8)
plt.show()
''')

md(r"""
### 3b. Endpoint transfer by hyper-parameter

Aggregation: median (and max) of endpoint `p_test` over the alive cross-seed runs sharing the row
value; `same-model median` is the same statistic in the comparator arm.
""")

code(r'''
def _by(col):
    t = alive.groupby(col)["p_test"].agg(n="size", median="median", mean="mean", max="max")
    t["peak median"] = alive.groupby(col)["p_test_peak"].median()
    t["n dead"] = dead.groupby(col).size().reindex(t.index).fillna(0).astype(int)
    t["same-model median"] = ALIVE["same-model"].groupby(col)["p_test"].median().reindex(t.index)
    t["same-model peak median"] = ALIVE["same-model"].groupby(col)["p_test_peak"].median().reindex(t.index)
    return t.round(3)

print("by lr rung (alive cross-seed runs):"); display(_by("lr"))
print("by rank (alive cross-seed runs):"); display(_by("rank"))
print("by alpha (alive cross-seed runs):"); display(_by("alpha"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
for ax, col in zip(axes, ["rank", "alpha"]):
    vals = sorted(alive[col].unique())
    data = [alive.loc[alive[col] == v, "p_test"].to_numpy() for v in vals]
    ax.boxplot(data, positions=range(len(vals)), widths=0.6, showfliers=False)
    for i, v in enumerate(vals):
        y = alive.loc[alive[col] == v, "p_test"]
        ax.scatter(i + aplots.jitter(len(y), 0.18, seed=i), y, s=22, alpha=0.7, color="tab:blue",
                   label="cross-seed run" if i == 0 else None)
        ys = ALIVE["same-model"].loc[ALIVE["same-model"][col] == v, "p_test"]
        ax.scatter([i + 0.3], [ys.median()], marker="_", s=300, color="tab:orange", lw=2,
                   label="same-model median" if i == 0 else None)
    ax.axhline(p0, color="k", ls="--", lw=1, label=f"seed-2 prior {p0:.3f}")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([f"{v:g}" for v in vals])
    ax.set_xlabel(col); ax.set_ylabel("endpoint p_test (alive runs)"); ax.set_title(f"Endpoint transfer by {col}")
aplots.right_legend(axes[1])
plt.show()
''')

md(r"""
### 3c. Endpoint and peak transfer against K/K*

Both arms on the **same-model** K axis (same-model exponents and K*, so a cell sits at the same x
in both panels); colour = rank.  Left endpoint, right trajectory peak.  Below: alive cross-seed
runs binned on log10(K/K*) (own exponents), endpoint vs peak medians per bin.
""")

code(r'''
pal_r = aplots.palette(RANKS, cmap="tab10")
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True, sharey=True)
for ax, col, title in [(axes[0], "p_test", "endpoint"), (axes[1], "p_test_peak", "trajectory peak")]:
    a_s = ALIVE["same-model"]
    ax.scatter(a_s["K_rel"], a_s[col], s=26, alpha=0.35, color="gray", label="same-model (alive)")
    for r, sub in alive.groupby("rank"):
        ax.scatter(sub["K_rel_same_axis"], sub[col], s=40, alpha=0.9, color=pal_r[r], label=f"cross-seed rank {r}",
                   edgecolor="k", linewidth=0.3)
    ax.axhline(p0, color="k", ls="--", lw=1, label=f"seed-2 prior {p0:.3f}")
    ax.axvline(1.0, ls=":", color="k", lw=1, label="same-model K*")
    ax.set_xscale("log"); ax.set_xlabel("K / K*_same  (same-model exponents)"); ax.set_ylabel(col)
    ax.set_title(f"Alive runs: {title} transfer vs dose, both arms")
aplots.right_legend(axes[1], fontsize=8)
plt.show()

bins = [-3, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.5]
alive["K bin (log10 K/K*)"] = pd.cut(alive["log10_K_rel"], bins)
t = alive.groupby("K bin (log10 K/K*)", observed=True).agg(
    n=("p_test", "size"), end_median=("p_test", "median"), end_max=("p_test", "max"),
    peak_median=("p_test_peak", "median"), peak_max=("p_test_peak", "max"),
    peak_step_median=("p_test_peak_step", "median"), dce_median=("delta_ce", "median")).round(3)
print("alive cross-seed runs binned on log10(K/K*) (cross-seed exponents): endpoint vs trajectory peak (median over the bin)")
display(t)
''')

md(r"""
## 4. Transfer on the (rank, alpha) grid — endpoint, trajectory peak, and fixed step counts

Same recipe as `transfer_trajectory_grids.ipynb` §2: population = **endpoint-alive** runs of
the arm; one cell per (rank, alpha) holding the median `p_test` over that cell's alive lr rungs
(`n` = alive rungs, 0–6).  Top row: endpoint and best checkpoint.  Bottom row: each run's saved
checkpoint nearest step 625, 1,250, 2,500, 5,000, 10,000 (of 12,500; the panel title names the
median checkpoint actually used — the cross-seed arm mixes 312- and 625-step cadences).  All
panels of one figure share a colour scale (0 to that arm's largest peak median); the same-model
figure follows for comparison, on its own scale.  Absolute steps only, never a share of the run.
""")

code(r'''
from subliminal_learning.analysis.trajectories import grid_at_step, grid_nanmedian, rank_alpha_grid


def _grid(sub, col):
    return rank_alpha_grid(sub, col, RANKS, ALPHAS)

def _at_step(tr, alive_tags, step):
    """Per run, the saved checkpoint nearest `step` (within 10 %); grid of p_test over those."""
    return grid_at_step(tr, alive_tags, step, "p_test", RANKS, ALPHAS, require_value="p_test")

def arm_panels(arm):
    end, tr = FR[arm], TR[arm]
    a = ALIVE[arm]; alive_tags = set(a["tag"])
    views = {"endpoint": _grid(a, "p_test"), "trajectory peak": _grid(a, "p_test_peak")}
    at_step = {st: _at_step(tr, alive_tags, st) for st in STEPS}
    vmax = float(np.nanmax(views["trajectory peak"][0].to_numpy()))
    fig = plt.figure(figsize=(19, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1])
    for j, (name, (g, n)) in enumerate(views.items()):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        im = aplots.annotated_heatmap(ax, g, counts=n, cmap="viridis", vmin=0, vmax=vmax, fmt="{:.2f}", cell_fontsize=7)
        ax.set_xlabel("alpha"); ax.set_ylabel("rank"); ax.set_title(f"{name} p_test")
    cax = fig.add_subplot(gs[0, 4]); cax.axis("off")
    fig.colorbar(im, ax=cax, fraction=0.5, label="median p_test over alive lr rungs")
    for j, st in enumerate(STEPS):
        gn, got = at_step[st]
        ax = fig.add_subplot(gs[1, j])
        g, n = gn
        aplots.annotated_heatmap(ax, g, counts=n, cmap="viridis", vmin=0, vmax=vmax, fmt="{:.2f}", cell_fontsize=6)
        ax.set_xlabel("alpha"); ax.set_ylabel("rank" if j == 0 else "")
        ax.set_title(f"step {st:,}" + ("" if got == st else f" (ckpt {got:,})"))
    fig.suptitle(f"OLMo-1B {arm}: transfer per (rank, alpha) cell along training  "
                 f"(alive runs n={len(a)}, {int(end['n_train_steps'].median()):,} steps/run, "
                 f"base prior {PRIOR[arm]['test']:.4f}, dead cut {CUT[arm]:.2f} nats)")
    plt.show()
    return views, at_step

PANELS = {arm: arm_panels(arm) for arm in ARMS}
''')

md(r"""
### 4b. Grid summary, both arms

Row unit: one arm.  `grid median` = median over the 56 cells of the cell medians (every
(rank, alpha) pair counts once).  `step N` = grid median at the checkpoint nearest step N;
`peak step (med)` = median over alive runs of the step of their best score; `peak/end (med)` =
median over alive runs of best/endpoint score.  Second table: the same in units of each arm's own
base prior.  Then the cell-wise ratio cross/same of the endpoint and peak grids.
""")

code(r'''
_gm = grid_nanmedian

rows = []
for arm in ARMS:
    a = ALIVE[arm]; views, at_step = PANELS[arm]
    r = {"arm": arm, "alive": len(a), "prior": round(PRIOR[arm]["test"], 4), "total steps": int(a["n_train_steps"].median())}
    for st in STEPS:
        gn, _ = at_step[st]
        r[f"step {st}"] = round(_gm(gn[0]), 3)
    r["endpoint"] = round(_gm(views["endpoint"][0]), 3)
    r["peak"] = round(_gm(views["trajectory peak"][0]), 3)
    r["peak step (med)"] = int(a["p_test_peak_step"].median())
    r["peak/end (med)"] = round(float((a["p_test_peak"] / a["p_test"].clip(lower=1e-6)).median()), 2)
    r["rank-1 end"] = round(_gm(views["endpoint"][0], [1]), 3)
    r["other end"] = round(_gm(views["endpoint"][0], [x for x in RANKS if x != 1]), 3)
    r["rank-1 peak"] = round(_gm(views["trajectory peak"][0], [1]), 3)
    r["other peak"] = round(_gm(views["trajectory peak"][0], [x for x in RANKS if x != 1]), 3)
    rows.append(r)
SUMMARY = pd.DataFrame(rows).set_index("arm")
display(SUMMARY)
rel = SUMMARY.copy()
REL_COLS = [f"step {st}" for st in STEPS] + ["endpoint", "peak", "rank-1 end", "other end", "rank-1 peak", "other peak"]
for c in REL_COLS:
    rel[c] = (SUMMARY[c] / SUMMARY["prior"]).round(1)
print("same table in units of each arm's base prior (x prior):")
display(rel.drop(columns=["prior", "peak step (med)", "peak/end (med)"]))

gx, gs_ = PANELS["cross-seed 2"][0], PANELS["same-model"][0]
ratio_end = (gx["endpoint"][0] / gs_["endpoint"][0]).round(2)
ratio_peak = (gx["trajectory peak"][0] / gs_["trajectory peak"][0]).round(2)
print("cell-wise ratio cross-seed / same-model of the ENDPOINT cell medians (rank x alpha; NaN = a cell alive in only one arm):")
display(ratio_end)
print("cell-wise ratio cross-seed / same-model of the PEAK cell medians:")
display(ratio_peak)
print(f"median over cells: endpoint ratio {np.nanmedian(ratio_end.to_numpy()):.2f}, peak ratio {np.nanmedian(ratio_peak.to_numpy()):.2f}")
''')

md(r"""
### 4c. Grid-median trajectory, both arms

Left: grid median of the cell medians at each step target and at the endpoint, in units of the
arm's base prior; star = the arm's peak grid median at its median peak step.  Right: the
rank-1 vs other-rank split at endpoint and peak.
""")

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
pal_a = {"cross-seed 2": "tab:blue", "same-model": "tab:orange"}
ax = axes[0]
for arm in ARMS:
    n_end = int(SUMMARY.loc[arm, "total steps"])
    pts = [(st, rel.loc[arm, f"step {st}"]) for st in STEPS] + [(n_end, rel.loc[arm, "endpoint"])]
    ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", ms=6, lw=2, color=pal_a[arm], label=arm)
    ax.scatter([SUMMARY.loc[arm, "peak step (med)"]], [rel.loc[arm, "peak"]], marker="*", s=200,
               color=pal_a[arm], edgecolor="k", zorder=5, label=f"{arm} peak (median step)")
ax.axhline(1, color="k", ls="--", lw=1, label="base prior")
ax.set_xscale("log"); ax.set_xlabel("optimizer step (log; last point = endpoint, 12,500)")
ax.set_ylabel("grid median p_test / base prior"); ax.set_title("Transfer along training, both OLMo arms")
aplots.right_legend(ax, fontsize=8)
ax = axes[1]
w = 0.2; x0 = np.arange(len(ARMS))
for k, (col, lab) in enumerate([("rank-1 end", "rank 1, endpoint"), ("other end", "ranks 2-128, endpoint"),
                                ("rank-1 peak", "rank 1, peak"), ("other peak", "ranks 2-128, peak")]):
    ax.bar(x0 + (k - 1.5) * w, rel[col].to_numpy(), width=w, label=lab,
           color=["tab:blue", "tab:cyan", "tab:red", "tab:orange"][k])
ax.set_xticks(x0); ax.set_xticklabels(list(ARMS))
ax.set_ylabel("grid median p_test / base prior"); ax.set_title("Rank-1 receivers vs the rest")
aplots.right_legend(ax, fontsize=8)
plt.show()
''')

md(r"""
## 5. When does the trait peak, and how does it evolve by dose?

Row unit: one alive cross-seed run.  `p_test_peak` is the largest `test` score across saved
checkpoints + endpoint, `p_test_peak_step` its optimizer step (12,500-step runs); a run "peaks at
the endpoint" if the argmax is the last checkpoint.  Right: mean (solid) and IQR band of `p_test`
and dp1 ΔCE per checkpoint over the alive runs in each log10(K/K*) bin (cross-seed exponents),
on the 625-step grid (each run's checkpoint nearest the grid step).
""")

code(r'''
n_steps = int(df["n_train_steps"].dropna().iloc[0])
for arm in ARMS:
    a = ALIVE[arm]
    a["peak_over_end"] = a["p_test_peak"] / a["p_test"].clip(lower=1e-6)
    a["peaks_at_end"] = a["p_test_peak_step"] >= n_steps
timing = pd.DataFrame({arm: {
    "alive runs": len(a), "peak at final checkpoint": int(a["peaks_at_end"].sum()),
    "peak before step 3,125": int((a["p_test_peak_step"] < 3125).sum()),
    "peak before step 6,250": int((a["p_test_peak_step"] < 6250).sum()),
    "median peak step": int(a["p_test_peak_step"].median()),
    "IQR of peak step": f"{int(a['p_test_peak_step'].quantile(0.25))}-{int(a['p_test_peak_step'].quantile(0.75))}",
    "median peak p_test": round(float(a["p_test_peak"].median()), 3),
    "median endpoint p_test": round(float(a["p_test"].median()), 3),
    "median peak / endpoint": round(float(a["peak_over_end"].median()), 2),
    "runs with peak > 2x endpoint": int((a["peak_over_end"] > 2).sum()),
} for arm, a in ALIVE.items()})
print(f"peak timing over alive runs, per arm ({n_steps:,} steps per run):")
display(timing)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
ax = axes[0]
for r, sub in alive.groupby("rank"):
    ax.scatter(sub["p_test"], sub["p_test_peak"], s=36, alpha=0.85, color=pal_r[r], label=f"rank {r}", edgecolor="k", linewidth=0.3)
lim = float(alive["p_test_peak"].max()) * 1.05
ax.plot([0, lim], [0, lim], color="k", lw=1, ls="--", label="peak = endpoint")
ax.set_xlabel("endpoint p_test"); ax.set_ylabel("trajectory peak p_test"); ax.set_title("Cross-seed: peak vs endpoint (alive)")
ax = axes[1]
ax.hist(alive["p_test_peak_step"], bins=np.arange(0, n_steps + 626, 625), color="tab:blue", alpha=0.7, label="cross-seed")
ax.hist(ALIVE["same-model"]["p_test_peak_step"], bins=np.arange(0, n_steps + 626, 625), color="tab:orange", alpha=0.5, label="same-model")
ax.set_xlabel(f"peak step (of {n_steps:,})"); ax.set_ylabel("alive runs"); ax.set_title("When the trait peaks")
aplots.right_legend(ax, fontsize=8)
ax = axes[2]
for r, sub in alive.groupby("rank"):
    ax.scatter(sub["K_rel"], sub["p_test_peak_step"], s=36, alpha=0.85, color=pal_r[r], label=f"rank {r}", edgecolor="k", linewidth=0.3)
ax.set_xscale("log"); ax.set_xlabel("K / K* (cross-seed)"); ax.set_ylabel(f"peak step (of {n_steps:,})"); ax.set_title("Cross-seed: peak timing vs dose")
aplots.right_legend(ax, title="receiver rank")
plt.show()
''')

code(r'''
GRID625 = np.arange(625, n_steps + 1, 625)

def snap_to_grid(tr, tol_frac=0.1):
    """One row per (run, grid step): the run's checkpoint nearest each 625-multiple (within 10 %)."""
    out = []
    for g_step in GRID625:
        t = tr.copy(); t["d"] = (t["step"] - g_step).abs()
        pick = t.loc[t.groupby("tag")["d"].idxmin()]
        pick = pick[pick["d"] <= tol_frac * g_step].copy(); pick["grid_step"] = int(g_step)
        out.append(pick)
    return pd.concat(out, ignore_index=True)

tr = TR["cross-seed 2"].merge(df[["tag", "K_rel", "log10_K_rel", "collapsed"]], on="tag", how="left")
trg = snap_to_grid(tr)
trg["K bin"] = pd.cut(trg["log10_K_rel"], bins)
trg_alive = trg[trg["collapsed"] == False]   # noqa: E712
print(f"cross-seed trajectory rows {len(tr)} ({tr['tag'].nunique()} runs) -> {len(trg)} rows on the 625-step grid")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
cats = [c for c in trg_alive["K bin"].cat.categories if (trg_alive["K bin"] == c).any()]
pal_b = aplots.palette(cats, cmap="tab10")
for c in cats:
    sub = trg_alive[trg_alive["K bin"] == c]; n = sub["tag"].nunique()
    for ax, col in zip(axes, ["p_test", "delta_ce"]):
        s = sub.dropna(subset=[col]); g = s.groupby("grid_step")[col]
        m, lo, hi = g.mean(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(m.index, m.values, color=pal_b[c], lw=2, label=f"{c}  (n={n})")
        ax.fill_between(m.index, lo.values, hi.values, color=pal_b[c], alpha=0.15)
axes[0].axhline(p0, color="k", ls="--", lw=1, label=f"seed-2 prior {p0:.3f}")
axes[0].set_xlabel("step"); axes[0].set_ylabel("p_test (mean, IQR band)"); axes[0].set_title("Cross-seed trait trajectory, alive runs by log10(K/K*)")
axes[1].axhline(CUT["cross-seed 2"], color="red", ls="--", lw=1.2, label=f"anchored cut {CUT['cross-seed 2']:.2f}")
axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("dp1 dCE [nats]"); axes[1].set_title("Cross-seed probe trajectory, alive runs by log10(K/K*)")
aplots.right_legend(axes[1], title="log10(K/K*) bin")
plt.show()

top = alive.sort_values("p_test", ascending=False).head(8)
rev = alive[alive["cross_step"].notna()].sort_values("K_rel")
print("top-8 endpoint transfer (alive cross-seed runs):")
display(top[["tag", "K_rel", "delta_ce", "p_test", "p_off", "p_test_peak", "p_test_peak_step", "cross_step"]].round(4).reset_index(drop=True))
print(f"\ncollapse-and-revive runs (crossed the cut on the trajectory, alive at the endpoint): {len(rev)}")
display(rev[["tag", "K_rel", "cross_step", "dce_peak", "dce_peak_step", "delta_ce", "p_test", "p_test_peak"]].round(3).reset_index(drop=True))
print("\ndead cross-seed runs: step of first crossing (of", n_steps, "):")
display(dead["cross_step"].describe().round(0).to_frame().T)
''')

md(r"""
## 6. Summary figure (exported)

Three panels: endpoint ΔCE vs K/K* with the anchored cut (collapse boundary), endpoint transfer vs
K/K* for alive runs, and trajectory-peak transfer vs K/K* for the same runs; the same-model arm
is the gray background in the transfer panels (on its own K axis).
""")

code(r'''
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
ax = axes[0]
for lr, sub in df.dropna(subset=["delta_ce"]).groupby("lr"):
    ax.scatter(sub["K_rel"], sub["delta_ce"].clip(lower=1e-3), s=30, alpha=0.8, color=pal[lr], label=f"lr {lr:g}", edgecolor="k", linewidth=0.3)
ax.axhline(CUT["cross-seed 2"], ls="--", color="red", lw=1.2, label="anchored cut"); ax.axvline(1, ls=":", color="k")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("K / K*"); ax.set_ylabel("endpoint dCE [nats]")
ax.set_title(f"Collapse boundary (K* = {KSTAR_X:.2e}, b = {B_X:.2f}, c = {C_X:.2f})"); ax.legend(fontsize=7, loc="upper left")
ymax = float(max(alive["p_test_peak"].max(), ALIVE["same-model"]["p_test_peak"].max())) * 1.05
for ax, col, title in [(axes[1], "p_test", "Endpoint transfer (alive)"), (axes[2], "p_test_peak", "Trajectory-peak transfer (alive)")]:
    a_s = ALIVE["same-model"]
    ax.scatter(a_s["K_rel"], a_s[col], s=22, alpha=0.3, color="gray", label="same-model arm")
    for r, sub in alive.groupby("rank"):
        ax.scatter(sub["K_rel"], sub[col], s=34, alpha=0.85, color=pal_r[r], label=f"rank {r}", edgecolor="k", linewidth=0.3)
    ax.axhline(p0, color="k", ls="--", lw=1); ax.axvline(1, ls=":", color="k")
    ax.set_xscale("log"); ax.set_xlabel("K / K* (each arm's own)"); ax.set_ylabel(col); ax.set_title(title); ax.set_ylim(0, ymax)
aplots.right_legend(axes[2], title="receiver rank")
fig.suptitle("OLMo-1B cross-seed elephant ladder: dd-1b teacher ep10_lr5e-4_r64a128 -> dd-1b-seed2 students (ep10, ftseed 1)")
fig.savefig(FIG_DIR / "dd1b_cross_seed2_transfer.png", dpi=150, bbox_inches="tight")
plt.show()
''')

md(r"""
## 7. Readout

Printed from the frames above so the prose cannot drift from the numbers.
""")

code(r'''
a_s = ALIVE["same-model"]
r1 = alive[alive["rank"] == 1]["p_test"]; rn = alive[alive["rank"] > 1]["p_test"]
print(f"Coverage: {int(df['delta_ce'].notna().sum())}/{len(df)} cross-seed cells with endpoint records; "
      f"{int(df['has_traj'].sum())} with both trajectories.")
print(f"Collapse: {len(dead)}/{int(df['collapsed'].notna().sum())} cross-seed runs collapsed under the seed-2 anchored dp1 cut "
      f"({CUT['cross-seed 2']:.2f} nats) vs {len(DEAD['same-model'])}/{len(df_same)} same-model; per-cell fate agrees on "
      f"{int((both['fate'].isin(['both dead', 'both alive'])).sum())}/{len(both)} cells.")
print(f"Law: cross-seed b={B_X:.3f}, c={C_X:.3f}, K*={KSTAR_X:.3e} ({KSTAR_X / cl.KSTAR_CALIBRATED:.1f}x 160m); "
      f"same-model b={B_S:.3f}, c={C_S:.3f}, K*={KSTAR_S:.3e}; K* shift at fixed exponents {KSTAR_S_ON_X / KSTAR_S:.2f}x.")
print(f"Endpoint transfer (alive, n={len(alive)}): median {alive['p_test'].median():.3f} "
      f"(IQR {alive['p_test'].quantile(.25):.3f}-{alive['p_test'].quantile(.75):.3f}, max {alive['p_test'].max():.3f}) "
      f"= {alive['p_test'].median() / p0:.1f}x the seed-2 prior {p0:.4f}; same-model median {a_s['p_test'].median():.3f} "
      f"= {a_s['p_test'].median() / p0_s:.1f}x its prior {p0_s:.4f}; {100 * (alive['p_test'] > 2 * p0).mean():.0f}% of alive "
      f"cross-seed runs above 2x prior.")
print(f"  rank-1 receivers: median {r1.median():.3f} (n={len(r1)}) vs other ranks {rn.median():.3f} (n={len(rn)}).")
print(f"Trajectory: cross-seed median peak {alive['p_test_peak'].median():.3f} ({alive['p_test_peak'].median() / p0:.1f}x prior) "
      f"at median step {int(alive['p_test_peak_step'].median())}/{n_steps}; same-model median peak {a_s['p_test_peak'].median():.3f} "
      f"({a_s['p_test_peak'].median() / p0_s:.1f}x prior) at step {int(a_s['p_test_peak_step'].median())}; "
      f"cross-seed peak/endpoint {alive['peak_over_end'].median():.2f}x, {int(alive['peaks_at_end'].sum())}/{len(alive)} peak at the end.")
print(f"Grid medians (x prior), cross-seed vs same-model: step 625 {rel.loc['cross-seed 2', 'step 625']} vs {rel.loc['same-model', 'step 625']}; "
      f"step 2,500 {rel.loc['cross-seed 2', 'step 2500']} vs {rel.loc['same-model', 'step 2500']}; "
      f"peak {rel.loc['cross-seed 2', 'peak']} vs {rel.loc['same-model', 'peak']}; endpoint {rel.loc['cross-seed 2', 'endpoint']} vs {rel.loc['same-model', 'endpoint']}.")
print(f"  {len(rev)} alive cross-seed runs crossed the cut mid-training and recovered; dead runs first cross at median step "
      f"{int(dead['cross_step'].median()) if dead['cross_step'].notna().any() else 'n/a'}.")
''')

md(r"""
## 8. The boundary law refit at every checkpoint

Same recipe as `dd1b_same_model_ladder.ipynb` §7: at each grid step (625-multiples; each run's
nearest checkpoint within 10 %) every run is labelled by its **own** dp1 ΔCE at that step
(`> cut`), and the law is refit on those labels.  Row unit: one checkpoint step.  `K* (fixed)`
holds the endpoint cross-seed exponents and moves only the threshold; `b, c, K* (free)` refit
everything (≥ 8 dead and ≥ 8 alive; ratios only).  Dead-set completeness = fraction of the
endpoint dead set already over the cut.  The same-model arm's fixed-exponent K*(t) is overlaid
(its own exponents and cut, on its 312-step saves snapped to the same grid).
""")

code(r'''
def per_step_refit(trg, dead_tags, cut, b, c, kstar):
    lab = trg.dropna(subset=["delta_ce"]).copy()
    lab["collapsed"] = (lab["delta_ce"] > cut).astype(int); lab["student_tag"] = lab["tag"]
    rows_t = []
    for step, g in lab.groupby("grid_step"):
        row = {"step": int(step), "n": len(g), "n_dead": int(g["collapsed"].sum()),
               "dead_set_complete": len(set(g.loc[g["collapsed"] == 1, "tag"]) & dead_tags) / max(1, len(dead_tags)),
               "n_transient_dead": int(((g["collapsed"] == 1) & ~g["tag"].isin(dead_tags)).sum())}
        if g["collapsed"].nunique() == 2:
            row["kstar_fixed"] = cl.kstar_logistic(g, b, c)["kstar"]
            if row["n_dead"] >= 8 and len(g) - row["n_dead"] >= 8:
                try:
                    f = cl.fit_collapse_law(g)
                except np.linalg.LinAlgError:
                    f = None
                if f is not None and np.isfinite(f.b) and np.isfinite(f.c):
                    k_free = cl.kstar_logistic(g, f.b, f.c)["kstar"]
                    row.update(b=f.b, c=f.c, kstar_free=k_free, misses=cl.law_misses(g, f.b, f.c, k_free)[0])
        rows_t.append(row)
    out = pd.DataFrame(rows_t); out["kstar_fixed_rel"] = out["kstar_fixed"] / kstar
    return out

BT = per_step_refit(trg, set(dead["tag"]), CUT["cross-seed 2"], B_X, C_X, KSTAR_X)
tr_s = TR["same-model"].merge(df_same[["tag", "collapsed"]], on="tag", how="left")
BT_S = per_step_refit(snap_to_grid(tr_s), set(DEAD["same-model"]["tag"]), CUT["same-model"], B_S, C_S, KSTAR_S)
show = BT[["step", "n", "n_dead", "dead_set_complete", "n_transient_dead", "kstar_fixed", "kstar_fixed_rel", "b", "c", "kstar_free", "misses"]]
print("cross-seed per-checkpoint refit (label = each run's own dCE at that step > cut):")
display(show.round({"dead_set_complete": 2, "kstar_fixed": 6, "kstar_fixed_rel": 2, "b": 3, "c": 3, "kstar_free": 6}).to_string(index=False))

ok = BT.dropna(subset=["b"])
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
ax = axes[0]
ax.plot(BT["step"], BT["kstar_fixed"], "o-", ms=4, color="tab:blue", label="cross-seed K* (own endpoint b, c fixed)")
ax.plot(ok["step"], ok["kstar_free"], "s--", ms=4, color="tab:cyan", label="cross-seed K* with (b, c) refit")
ax.plot(BT_S["step"], BT_S["kstar_fixed"], "o-", ms=4, color="tab:orange", alpha=0.7, label="same-model K* (own b, c fixed)")
ax.axhline(KSTAR_X, color="tab:blue", ls=":", lw=1, label=f"cross-seed endpoint K* {KSTAR_X:.2e}")
ax.axhline(KSTAR_S, color="tab:orange", ls=":", lw=1, label=f"same-model endpoint K* {KSTAR_S:.2e}")
ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlabel("step (log)"); ax.set_ylabel("K*(step)"); ax.set_title("Boundary threshold along training")
aplots.right_legend(ax, fontsize=8)
ax = axes[1]
ax.plot(ok["step"], ok["b"], "o-", ms=4, color="tab:green", label="b (alpha exponent)")
ax.plot(ok["step"], ok["c"], "s-", ms=4, color="tab:red", label="c (rank exponent)")
ax.axhline(B_X, color="tab:green", ls=":", lw=1, label=f"endpoint b {B_X:.3f}")
ax.axhline(C_X, color="tab:red", ls=":", lw=1, label=f"endpoint c {C_X:.3f}")
ax.axhline(B_S, color="tab:green", ls="--", lw=1, alpha=0.6, label=f"same-model b {B_S:.3f}")
ax.axhline(C_S, color="tab:red", ls="--", lw=1, alpha=0.6, label=f"same-model c {C_S:.3f}")
ax.set_xscale("log"); ax.set_xlabel("step (log)"); ax.set_ylabel("exponent"); ax.set_title("Cross-seed exponents refit per checkpoint")
aplots.right_legend(ax, fontsize=8)
ax = axes[2]
ax.plot(BT["step"], BT["n_dead"], "o-", ms=4, color="tab:red", label="cross-seed runs over the cut")
ax.plot(BT["step"], BT["n_transient_dead"], "s-", ms=4, color="tab:purple", label="of which alive at the endpoint")
ax.plot(BT_S["step"], BT_S["n_dead"], "o-", ms=4, color="tab:orange", alpha=0.7, label="same-model runs over the cut")
ax.axhline(len(dead), color="k", ls=":", lw=1, label=f"cross-seed endpoint dead {len(dead)}")
ax.set_xlabel("step"); ax.set_ylabel("runs"); ax.set_title("Dead count along training")
aplots.right_legend(ax, fontsize=8)
plt.show()

first = BT[BT["dead_set_complete"] >= 0.95].iloc[0]
first_s = BT_S[BT_S["dead_set_complete"] >= 0.95].iloc[0]
print(f"cross-seed endpoint dead set 95% complete by step {int(first['step'])} (same-model: step {int(first_s['step'])}) of {n_steps:,}; "
      f"K*(fixed exps) spans {BT['kstar_fixed'].min():.2e}-{BT['kstar_fixed'].max():.2e} "
      f"({BT['kstar_fixed'].max() / BT['kstar_fixed'].min():.2f}x) from step {int(BT['step'].min())} to the end; "
      f"free b range {ok['b'].min():.3f}-{ok['b'].max():.3f}, c range {ok['c'].min():.3f}-{ok['c'].max():.3f}.")
''')


def build_notebook() -> nbf.NotebookNode:
    return new_notebook(CELLS, KERNEL_PY3)


if __name__ == "__main__":
    execute_and_write(build_notebook(), OUT, REPO)
