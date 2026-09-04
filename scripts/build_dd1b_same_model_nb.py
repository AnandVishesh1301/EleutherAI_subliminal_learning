"""Build and execute ``notebooks/dd1b_same_model_ladder.ipynb``.

The DataDecide OLMo-1B **same-model** elephant ladder (``dd1b-ladder-band`` queue, 2026-09-03):
one teacher, 336 students on a fixed six-rung lr band x the 56-pair (rank, alpha) strip
universe, every student carrying endpoint + per-checkpoint trait and dp1-probe readouts.  The
notebook reads the tree through `analysis.dd1b_ladder` only and writes nothing there; its one
export is ``docs/results/figures/dd1b_same_model_transfer.png``.

    uv run python scripts/build_dd1b_same_model_nb.py     # builds + executes in place
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).absolute().parents[1]
OUT = REPO / "notebooks/dd1b_same_model_ladder.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(r"""
# OLMo-1B (DataDecide) same-model ladder — collapse boundary and transfer, endpoint vs trajectory

**What ran.** Base `allenai/DataDecide-dolma1_7-1B` at its final checkpoint (step 69369) on the
`default` seed branch (CLI token `dd-1b`), used for BOTH the teacher and every student, so this is
a same-model, same-seed arm.  Trait = *elephant*.

| stage | configuration |
| --- | --- |
| teacher | `ep10_lr5e-4_r64a128`, ftseed 0: template-bias FT on the 50-example elephant template, 10 epochs, batch 22 (30 steps); LoRA on the four in-block projections (`att_proj`, `attn_out`, `ff_proj`, `ff_out`), never the LM head. Picked because its r8a32 students transferred best across a ten-teacher screen. |
| numbers | 100k free-gen `ex10-20` sequences from that teacher at T=1; prefix filter keeps 20,379 (0.20); students train on a fixed 10k subsample (seed 42). |
| students | 336 = lr {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3} × rank {1..128} × alpha {4..256} (the 56-pair strip universe), ftseed 1, **ep10** (12,500 steps, batch 8, cosine, 5 % warm-up, AdamW). |
| per-student evals | endpoint: `test` / `off_topic` (prob) + `storytelling` (sampled) with base control; dp1 probe (32 Dolma docs × 512 tok) → CE, ΔCE, KL to base.  Trajectory: the same trait + probe readouts at 40 adapter-only checkpoints (+ the endpoint). |

**Label.** A run is *collapsed* iff its endpoint ΔCE on dp1 exceeds the base's unigram-anchored
cut `Δ_cross = H_uni − ce_base` (`analysis.labels.anchored_cut(..., pool="dp1")`); the
trajectory *crossing step* is the first checkpoint over that same cut.  The flat pp1 cut is never
used here (different pool, different tokenizer).

**Differences from the Pythia-160m strip to keep in mind:** ep10 students (160m anchors are ep20),
one teacher and one student seed, trait elephant, and no data-/weight-seed split (DataDecide seeds
re-randomize both).
""")

md(r"""
## 1. Setup and load

Row unit: one student run (`tag`, ftseed 1).  Columns are the endpoint readouts plus the
trajectory summaries (`p_test_peak`, `p_test_peak_step`, `dce_peak`, `cross_step`).  Prints the
anchored cut, the base prior on each prompt set, and the coverage of every record type.
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
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

OUTPUT_ROOT = resolve_output_root()
TEACHER = "ep10_lr5e-4_r64a128"
FIG_DIR = Path("../docs/results/figures")

CUT = labels.anchored_cut(OUTPUT_ROOT, dl.DD1B_BASE, pool="dp1")
PRIOR = dl.load_base_prior(OUTPUT_ROOT, TEACHER)
df = dl.load_endpoint_frame(OUTPUT_ROOT, TEACHER, seed=1, dead_cut=CUT)
df["collapsed"] = df["collapsed"].astype("boolean")
df["ln_lr"] = np.log10(df["lr"])

print(f"anchored dp1 dead cut for {dl.DD1B_BASE}: dCE > {CUT:.4f} nats")
print("base prior p_target:", {k: round(v, 4) for k, v in PRIOR.items()})
print(f"\nruns: {len(df)}  cells: {df['tag'].nunique()}  "
      f"lr rungs: {sorted(df['lr'].unique())}\n"
      f"ranks: {sorted(df['rank'].unique())}  alphas: {sorted(df['alpha'].unique())}")
cov = pd.Series({
    "endpoint probe (dCE)": df["delta_ce"].notna().sum(),
    "endpoint preference": df["p_test"].notna().sum(),
    "both trajectories": int(df["has_traj"].sum()),
    "diverged (non-finite adapter)": int(df["diverged"].sum()),
    "collapsed (endpoint dCE > cut)": int(df["collapsed"].fillna(False).sum()),
    "ever over cut on trajectory": int(df["cross_step"].notna().sum()),
}, name="n runs").to_frame()
display(cov)
assert (df["n_ckpt"].eq(df["n_ckpt"].iloc[0])).all(), "unequal checkpoint counts"
print(f"distinct checkpoint steps per run: {int(df['n_ckpt'].iloc[0])} (saves every 312 steps, 312..12,480, plus the endpoint 12,500; "
      "the 09-04 cross-seed arm saves every 625 instead)")
''')

md(r"""
## 2. Collapse boundary

### 2a. Dead-count grids

Row unit of each grid: one (lr, alpha) or (lr, rank) cell; the value is the number of collapsed
runs among the 8 ranks (resp. 7 alphas) at that cell, i.e. a marginal of the 336-run design.
""")

code(r'''
g_alpha = df.pivot_table(index="lr", columns="alpha", values="collapsed", aggfunc="sum").astype(int)
g_rank = df.pivot_table(index="lr", columns="rank", values="collapsed", aggfunc="sum").astype(int)
print("dead runs per (lr, alpha) cell, summed over the 8 ranks (max 8):")
display(g_alpha)
print("dead runs per (lr, rank) cell, summed over the 7 alphas (max 7):")
display(g_rank)
print("median endpoint dCE per (lr, alpha) cell over the 8 ranks:")
display(df.pivot_table(index="lr", columns="alpha", values="delta_ce", aggfunc="median").round(2))
''')

md(r"""
### 2b. Boundary-law fit and K*

`fit_collapse_law` fits `logit P(collapsed) = b0 + β_lr ln lr + β_α ln α + β_r ln r` (cluster on
cell) and reports the ratios `b = β_α/β_lr`, `c = β_r/β_lr`.  The design is near-separable, so the
MLE does not converge and only the ratios are meaningful (the non-converged flag is expected, not a
defect).  `K*` is the pooled logistic 50 % crossing on `K = lr·α^b·r^c`; the misclassification
count is the assumption-free cross-check.  The Pythia-160m settled exponents are evaluated on the
same data for comparison.
""")

code(r'''
warnings.filterwarnings("ignore", category=RuntimeWarning)   # near-separation MLE noise
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
law = df.dropna(subset=["collapsed"]).copy()
law["collapsed"] = law["collapsed"].astype(int)
law["student_tag"] = law["tag"]
fit = cl.fit_collapse_law(law)
rows = []
for label, b, c in [("dd-1b fit (MLE ratios)", fit.b, fit.c),
                    ("Pythia-160m settled", cl.B_LAW, cl.C_LAW)]:
    k = cl.kstar_logistic(law, b, c)
    mn = cl.kstar_minerr(law, b, c)
    wrong, n_cells = cl.law_misses(law, b, c, k["kstar"])
    rows.append({"exponents": label, "b (alpha)": round(b, 3), "c (rank)": round(c, 3),
                 "K* logistic": f"{k['kstar']:.3e}", "K* min-err": f"{mn['kstar']:.3e}",
                 "K*/K*_160m (8.4e-4)": round(k["kstar"] / cl.KSTAR_CALIBRATED, 2),
                 "cells misclassified": f"{wrong}/{n_cells}"})
print(f"MLE converged: {fit.converged}  (near-separation expected)  n_dead={fit.n_dead}/{fit.n_runs}")
display(pd.DataFrame(rows).set_index("exponents"))
B_DD, C_DD = fit.b, fit.c
KSTAR_DD = cl.kstar_logistic(law, B_DD, C_DD)["kstar"]
df["K"] = cl.K_of(df["lr"], df["alpha"], df["rank"], b=B_DD, c=C_DD)
df["K_rel"] = df["K"] / KSTAR_DD
df["log10_K_rel"] = np.log10(df["K_rel"])
print(f"\nusing dd-1b exponents b={B_DD:.3f}, c={C_DD:.3f}, K*={KSTAR_DD:.3e} for K/K* below")
''')

md(r"""
### 2c. Endpoint ΔCE against K/K*

Every run (336 points) on the collapsed K axis; colour = lr rung.  The anchored cut is the dashed
line, the fitted K* the dotted vertical.
""")

code(r'''
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
pal = aplots.palette(sorted(df["lr"].unique()), cmap="tab10")
for lr, sub in df.groupby("lr"):
    ax.scatter(sub["K_rel"], sub["delta_ce"].clip(lower=1e-3), s=36, alpha=0.8,
               color=pal[lr], label=f"lr {lr:g}", edgecolor="k", linewidth=0.3)
ax.axhline(CUT, ls="--", color="red", lw=1.2, label=f"anchored cut {CUT:.2f}")
ax.axvline(1.0, ls=":", color="k", lw=1.0, label="K* (logistic 50 %)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("K / K*  (dd-1b exponents)"); ax.set_ylabel("endpoint dCE on dp1 [nats]")
ax.set_title("OLMo-1B same-model students: endpoint Pile-style ΔCE vs collapsed dose")
aplots.right_legend(ax, title="rung")
plt.show()
''')

md(r"""
## 3. Transfer at the endpoint

Row unit: one run.  `p_test` is the endpoint probability mass on " elephant" over the 50 `test`
prompts (the base prior is printed in §1).  All tables below restrict to **alive** runs (endpoint
ΔCE ≤ cut) unless the row says otherwise; dead runs have `p_test ≈ 0`.

### 3a. Distribution
""")

code(r'''
alive = df[df["collapsed"] == False].copy()   # noqa: E712
dead = df[df["collapsed"] == True].copy()     # noqa: E712
p0 = PRIOR["test"]
print(f"alive {len(alive)}  dead {len(dead)}  base prior (test) {p0:.4f}")
q = alive["p_test"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).rename(lambda x: f"q{int(x*100):02d}")
summary = pd.Series({"n alive": len(alive), "mean": alive["p_test"].mean(),
                     **q.to_dict(), "max": alive["p_test"].max(),
                     "frac > 2x prior": (alive["p_test"] > 2 * p0).mean(),
                     "frac > 0.20": (alive["p_test"] > 0.20).mean(),
                     "frac > 0.40": (alive["p_test"] > 0.40).mean()}, name="endpoint p_test, alive")
display(summary.to_frame().T.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
ax = axes[0]
bins = np.linspace(0, 0.65, 27)
ax.hist(alive["p_test"], bins=bins, color="tab:blue", alpha=0.8, label=f"alive (n={len(alive)})")
ax.hist(dead["p_test"], bins=bins, color="tab:red", alpha=0.6, label=f"dead (n={len(dead)})")
ax.axvline(p0, color="k", ls="--", lw=1, label=f"base prior {p0:.3f}")
ax.set_xlabel("endpoint p_test"); ax.set_ylabel("runs"); ax.set_title("Endpoint transfer, all 336 runs")
aplots.right_legend(ax)
ax = axes[1]
aplots.ecdf_plot(ax, alive["p_test"], label="alive endpoint p_test", color="tab:blue", lw=2)
aplots.ecdf_plot(ax, alive["p_test_peak"], label="alive trajectory PEAK p_test", color="tab:orange", lw=2)
ax.axvline(p0, color="k", ls="--", lw=1, label=f"base prior {p0:.3f}")
ax.set_xlabel("p_test"); ax.set_ylabel("ECDF"); ax.set_title("Alive runs: endpoint vs best checkpoint")
aplots.right_legend(ax)
plt.show()
''')

md(r"""
### 3b. Endpoint transfer by hyper-parameter

Aggregation: median (and max) of endpoint `p_test` over the alive runs sharing the row value.
""")

code(r'''
def _by(col):
    t = alive.groupby(col)["p_test"].agg(n="size", median="median", mean="mean", max="max")
    t["n dead"] = dead.groupby(col).size().reindex(t.index).fillna(0).astype(int)
    return t.round(3)

print("by lr rung (alive runs):"); display(_by("lr"))
print("by rank (alive runs):"); display(_by("rank"))
print("by alpha (alive runs):"); display(_by("alpha"))
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
for ax, (col, title) in zip(axes, [("rank", "by rank"), ("alpha", "by alpha")]):
    vals = sorted(alive[col].unique())
    data = [alive.loc[alive[col] == v, "p_test"].to_numpy() for v in vals]
    ax.boxplot(data, positions=range(len(vals)), widths=0.6, showfliers=False)
    for i, v in enumerate(vals):
        y = alive.loc[alive[col] == v, "p_test"]
        ax.scatter(i + aplots.jitter(len(y), 0.18, seed=i), y, s=22, alpha=0.7, color="tab:blue")
    ax.axhline(p0, color="k", ls="--", lw=1, label=f"base prior {p0:.3f}")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([f"{v:g}" for v in vals])
    ax.set_xlabel(col); ax.set_ylabel("endpoint p_test (alive runs)"); ax.set_title(f"Endpoint transfer {title}")
aplots.right_legend(axes[1])
plt.show()
''')

md(r"""
### 3c. Transfer on the (rank, alpha) grid — endpoint, trajectory peak, and fixed step counts

Every heatmap has the same population and aggregation: the **endpoint-alive** runs, one cell per
(rank, alpha) holding the median `p_test` over that cell's alive lr rungs (cell count = alive
rungs, 0–6).  Top row: the endpoint score and the best score along the trajectory.  Second row:
the score at the saved checkpoint nearest step 625, 1,250, 2,500, 5,000 and 10,000 (of 12,500;
this arm saved every 312 steps, so the checkpoint used is within 16 steps of each target and is
named in the panel title).  Trajectories are always compared at absolute step counts, never as
a share of the run, because run lengths differ across arms (Pythia arms run 25,000 steps).  All
panels share one colour scale (0 to the largest peak median).  Below: every alive run against
K/K*, colour = rank; the 160m strip shows a hump below K* on this axis, so this is the
like-for-like view.
""")

code(r'''
tr = dl.load_trajectories(OUTPUT_ROOT, TEACHER, seed=1)
tr = tr.merge(df[["tag", "K_rel", "log10_K_rel", "collapsed"]], on="tag", how="left")
n_steps = int(df["n_train_steps"].iloc[0])
alive_tags = set(alive["tag"])
ckpt_steps = np.array(sorted(tr["step"].unique()))

def _grid(values: pd.DataFrame, col: str):
    """(rank x alpha) median of `col` over the endpoint-alive runs, plus the per-cell count."""
    g = values.pivot_table(index="rank", columns="alpha", values=col, aggfunc="median")
    n = values.pivot_table(index="rank", columns="alpha", values=col, aggfunc="size")
    return g, n

def _at_step(target: int):
    """Grid at the saved checkpoint nearest `target` (absolute optimizer step)."""
    step = int(ckpt_steps[np.argmin(np.abs(ckpt_steps - target))])
    sub = tr[(tr["step"] == step) & tr["tag"].isin(alive_tags)]
    return step, _grid(sub, "p_test")

grids = {"endpoint": _grid(alive, "p_test"), "trajectory peak": _grid(alive, "p_test_peak")}
STEPS = [625, 1250, 2500, 5000, 10000]
step_grids = {st: _at_step(st) for st in STEPS}
vmax = float(grids["trajectory peak"][0].max().max())

fig = plt.figure(figsize=(19, 9.5), constrained_layout=True)
gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1])
top = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]
for ax, (name, (g, n)) in zip(top, grids.items()):
    im = aplots.annotated_heatmap(ax, g, counts=n, cmap="viridis", vmin=0, vmax=vmax, fmt="{:.2f}", cell_fontsize=7)
    ax.set_xlabel("alpha"); ax.set_ylabel("rank"); ax.set_title(f"{name} p_test (median over alive lr rungs)")
cax = fig.add_subplot(gs[0, 4]); cax.axis("off")
fig.colorbar(im, ax=cax, fraction=0.5, label="median p_test (alive runs)")
for j, st in enumerate(STEPS):
    ax = fig.add_subplot(gs[1, j])
    step, (g, n) = step_grids[st]
    aplots.annotated_heatmap(ax, g, counts=n, cmap="viridis", vmin=0, vmax=vmax, fmt="{:.2f}", cell_fontsize=6)
    ax.set_xlabel("alpha"); ax.set_ylabel("rank" if j == 0 else "")
    ax.set_title(f"step {st:,}" + ("" if step == st else f" (ckpt {step:,})"))
fig.suptitle("Transfer per (rank, alpha) cell along training — endpoint-alive runs, median over alive lr rungs")
plt.show()

print("grid-wide median of the cell medians:", {k: round(float(np.nanmedian(v[0].to_numpy())), 3) for k, v in grids.items()},
      {f"step {st}": round(float(np.nanmedian(step_grids[st][1][0].to_numpy())), 3) for st in STEPS})

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
pal_r = aplots.palette(sorted(df["rank"].unique()), cmap="tab10")
for r, sub in alive.groupby("rank"):
    ax.scatter(sub["K_rel"], sub["p_test"], s=40, alpha=0.85, color=pal_r[r], label=f"rank {r}",
               edgecolor="k", linewidth=0.3)
ax.axhline(p0, color="k", ls="--", lw=1, label=f"base prior {p0:.3f}")
ax.axvline(1.0, ls=":", color="k", lw=1, label="K*")
ax.set_xscale("log"); ax.set_xlabel("K / K*  (dd-1b exponents)"); ax.set_ylabel("endpoint p_test")
ax.set_title("Alive runs: endpoint transfer vs collapsed dose")
aplots.right_legend(ax, title="receiver rank")
plt.show()

bins = [-3, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.5]
alive["K bin (log10 K/K*)"] = pd.cut(alive["log10_K_rel"], bins)
t = alive.groupby("K bin (log10 K/K*)", observed=True).agg(
    n=("p_test", "size"), end_median=("p_test", "median"), end_mean=("p_test", "mean"),
    end_max=("p_test", "max"), peak_median=("p_test_peak", "median"),
    peak_step_median=("p_test_peak_step", "median"), dce_median=("delta_ce", "median")).round(3)
print("alive runs binned on log10(K/K*): endpoint p_test vs trajectory peak (median over the bin)")
display(t)
''')

md(r"""
## 4. Endpoint vs trajectory: where does transfer peak?

Row unit: one alive run.  `p_test_peak` is the largest `test` score across the intermediate
checkpoints + endpoint, `p_test_peak_step` its optimizer step (runs are 12,500 steps long).  A
run "peaks at the endpoint"
if the argmax is the last checkpoint.

### 4a. Peak vs endpoint, and when the peak happens
""")

code(r'''
n_steps = int(df["n_train_steps"].iloc[0])
alive["peak_over_end"] = alive["p_test_peak"] / alive["p_test"].clip(lower=1e-6)
alive["peaks_at_end"] = alive["p_test_peak_step"] >= alive["p_test_peak_step"].max()
s = pd.Series({
    "alive runs": len(alive),
    "peak at final checkpoint": int(alive["peaks_at_end"].sum()),
    "total steps per run": n_steps,
    "peak before step 6,250": int((alive["p_test_peak_step"] < 6250).sum()),
    "median peak step": int(alive["p_test_peak_step"].median()),
    "IQR of peak step": f"{int(alive['p_test_peak_step'].quantile(0.25))}-{int(alive['p_test_peak_step'].quantile(0.75))}",
    "median peak p_test": round(float(alive["p_test_peak"].median()), 3),
    "median endpoint p_test": round(float(alive["p_test"].median()), 3),
    "median peak / endpoint ratio": round(float(alive["peak_over_end"].median()), 2),
    "runs with peak > 2x endpoint": int((alive["peak_over_end"] > 2).sum()),
}, name="value")
display(s.to_frame())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
ax = axes[0]
for r, sub in alive.groupby("rank"):
    ax.scatter(sub["p_test"], sub["p_test_peak"], s=36, alpha=0.85, color=pal_r[r], label=f"rank {r}",
               edgecolor="k", linewidth=0.3)
lim = float(alive["p_test_peak"].max()) * 1.05
ax.plot([0, lim], [0, lim], color="k", lw=1, ls="--", label="peak = endpoint")
ax.set_xlabel("endpoint p_test"); ax.set_ylabel("trajectory peak p_test"); ax.set_title("Peak vs endpoint (alive)")
ax = axes[1]
ax.hist(alive["p_test_peak_step"], bins=np.arange(0, n_steps + 626, 625), color="tab:orange", alpha=0.85)
ax.set_xlabel(f"peak step (of {n_steps:,})"); ax.set_ylabel("alive runs"); ax.set_title("When the trait peaks")
ax = axes[2]
for r, sub in alive.groupby("rank"):
    ax.scatter(sub["K_rel"], sub["p_test_peak_step"], s=36, alpha=0.85, color=pal_r[r], label=f"rank {r}",
               edgecolor="k", linewidth=0.3)
ax.set_xscale("log"); ax.set_xlabel("K / K*"); ax.set_ylabel(f"peak step (of {n_steps:,})")
ax.set_title("Peak timing vs dose")
aplots.right_legend(ax, title="receiver rank")
plt.show()
''')

md(r"""
### 4b. Mean trait trajectory by dose bin

Aggregation: at each checkpoint, the mean (solid) and inter-quartile band of `p_test` over the
alive runs in each log10(K/K*) bin.  Right panel: the same for dp1 ΔCE (log scale), with the
anchored cut.  Uses the long trajectory frame loaded in §3c (one row per run × checkpoint).
""")

code(r'''
print(f"trajectory rows: {len(tr)}  runs: {tr['tag'].nunique()}  checkpoints/run: {tr.groupby('tag').size().iloc[0]}")
tr["K bin"] = pd.cut(tr["log10_K_rel"], bins)
tr_alive = tr[tr["collapsed"] == False]   # noqa: E712

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
cats = [c for c in tr_alive["K bin"].cat.categories if (tr_alive["K bin"] == c).any()]
pal_b = aplots.palette(cats, cmap="tab10")
for c in cats:
    sub = tr_alive[tr_alive["K bin"] == c]
    n = sub["tag"].nunique()
    for ax, col in zip(axes, ["p_test", "delta_ce"]):
        g = sub.groupby("step")[col]
        m, lo, hi = g.mean(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(m.index, m.values, color=pal_b[c], lw=2, label=f"{c}  (n={n})")
        ax.fill_between(m.index, lo.values, hi.values, color=pal_b[c], alpha=0.15)
axes[0].axhline(p0, color="k", ls="--", lw=1, label=f"base prior {p0:.3f}")
axes[0].set_xlabel("step"); axes[0].set_ylabel("p_test (mean, IQR band)"); axes[0].set_title("Trait trajectory, alive runs by log10(K/K*)")
axes[1].axhline(CUT, color="red", ls="--", lw=1.2, label=f"anchored cut {CUT:.2f}")
axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("dp1 dCE [nats]")
axes[1].set_title("Probe trajectory, alive runs by log10(K/K*)")
aplots.right_legend(axes[1], title="log10(K/K*) bin")
plt.show()
''')

md(r"""
### 4c. Individual trajectories: strongest endpoint transfer, and the collapse-and-revive runs

Left: the eight alive runs with the highest endpoint `p_test`.  Right: every alive run whose
trajectory crossed the anchored cut at some checkpoint (a transient collapse it recovered from),
trait on top, ΔCE below.  Row unit of the table: one run.
""")

code(r'''
top = alive.sort_values("p_test", ascending=False).head(8)
rev = alive[alive["cross_step"].notna()].sort_values("K_rel")
print("top-8 endpoint transfer (alive runs):")
display(top[["tag", "K_rel", "delta_ce", "p_test", "p_off", "p_test_peak", "p_test_peak_step", "cross_step"]]
        .round(4).reset_index(drop=True))
print(f"\ncollapse-and-revive runs (crossed the cut on the trajectory, alive at the endpoint): {len(rev)}")
display(rev[["tag", "K_rel", "cross_step", "dce_peak", "dce_peak_step", "delta_ce", "p_test", "p_test_peak"]]
        .round(3).reset_index(drop=True))
print("\ndead runs: step of first crossing (of", n_steps, "):")
display(dead["cross_step"].describe().round(0).to_frame().T)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True, sharex=True)
for j, (sel, title) in enumerate([(top, "top-8 endpoint transfer"), (rev, "collapse-and-revive")]):
    pal_t = aplots.palette(list(sel["tag"]), cmap="tab10")
    for tag in sel["tag"]:
        s = tr[tr["tag"] == tag].sort_values("step")
        axes[0, j].plot(s["step"], s["p_test"], lw=1.8, marker="o", ms=3, color=pal_t[tag], label=tag)
        axes[1, j].plot(s["step"], s["delta_ce"].clip(lower=1e-3), lw=1.8, marker="o", ms=3, color=pal_t[tag], label=tag)
    axes[0, j].axhline(p0, color="k", ls="--", lw=1)
    axes[1, j].axhline(CUT, color="red", ls="--", lw=1.2)
    axes[1, j].set_yscale("log")
    axes[0, j].set_title(f"{title}: trait"); axes[1, j].set_title(f"{title}: dp1 dCE")
    axes[0, j].set_ylabel("p_test"); axes[1, j].set_ylabel("dCE [nats]"); axes[1, j].set_xlabel("step")
    aplots.right_legend(axes[0, j], fontsize=8)
plt.show()
''')

md(r"""
## 5. Summary figure (exported)

Three panels: endpoint ΔCE vs K/K* with the anchored cut (collapse boundary), endpoint transfer vs
K/K* for alive runs, and trajectory-peak transfer vs K/K* for the same runs.
""")

code(r'''
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
ax = axes[0]
for lr, sub in df.groupby("lr"):
    ax.scatter(sub["K_rel"], sub["delta_ce"].clip(lower=1e-3), s=30, alpha=0.8, color=pal[lr],
               label=f"lr {lr:g}", edgecolor="k", linewidth=0.3)
ax.axhline(CUT, ls="--", color="red", lw=1.2, label="anchored cut"); ax.axvline(1, ls=":", color="k")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("K / K*"); ax.set_ylabel("endpoint dCE [nats]")
ax.set_title(f"Collapse boundary (K* = {KSTAR_DD:.2e})"); ax.legend(fontsize=7, loc="upper left")
for ax, col, title in [(axes[1], "p_test", "Endpoint transfer (alive)"),
                       (axes[2], "p_test_peak", "Trajectory-peak transfer (alive)")]:
    for r, sub in alive.groupby("rank"):
        ax.scatter(sub["K_rel"], sub[col], s=34, alpha=0.85, color=pal_r[r], label=f"rank {r}",
                   edgecolor="k", linewidth=0.3)
    ax.axhline(p0, color="k", ls="--", lw=1); ax.axvline(1, ls=":", color="k")
    ax.set_xscale("log"); ax.set_xlabel("K / K*"); ax.set_ylabel(col); ax.set_title(title)
    ax.set_ylim(0, float(alive["p_test_peak"].max()) * 1.05)
aplots.right_legend(axes[2], title="receiver rank")
fig.suptitle("OLMo-1B same-model elephant ladder: 336 students of teacher ep10_lr5e-4_r64a128 (ep10, ftseed 1)")
fig.savefig(FIG_DIR / "dd1b_same_model_transfer.png", dpi=150, bbox_inches="tight")
plt.show()
''')

md(r"""
## 6. Readout

Printed from the frames above so the prose cannot drift from the numbers.
""")

code(r'''
r1 = alive[alive["rank"] == 1]["p_test"]; rn = alive[alive["rank"] > 1]["p_test"]
print(f"Collapse: {int(dead.shape[0])}/{len(df)} runs collapsed under the anchored dp1 cut ({CUT:.2f} nats); "
      f"K* = {KSTAR_DD:.2e} at b={B_DD:.3f}, c={C_DD:.3f} = {KSTAR_DD / cl.KSTAR_CALIBRATED:.1f}x the 160m anchor "
      f"(ep10 students vs ep20 anchors).")
print(f"Endpoint transfer (alive, n={len(alive)}): median {alive['p_test'].median():.3f}, "
      f"IQR {alive['p_test'].quantile(.25):.3f}-{alive['p_test'].quantile(.75):.3f}, max {alive['p_test'].max():.3f}; "
      f"base prior {p0:.3f}; {100 * (alive['p_test'] > 2 * p0).mean():.0f}% above 2x prior.")
print(f"  rank-1 receivers: median {r1.median():.3f} (n={len(r1)}) vs other ranks {rn.median():.3f} (n={len(rn)}).")
print(f"Trajectory: median peak {alive['p_test_peak'].median():.3f} at step "
      f"{int(alive['p_test_peak_step'].median())}/{n_steps}; {int(alive['peaks_at_end'].sum())} of {len(alive)} alive runs "
      f"peak at the final checkpoint; median peak/endpoint ratio {alive['peak_over_end'].median():.2f}.")
print(f"  {len(rev)} alive runs crossed the cut mid-training and recovered; dead runs first cross at "
      f"median step {int(dead['cross_step'].median())}.")
''')

md(r"""
## 7. The boundary law refit at every checkpoint

Same recipe as the 160m strip / 410m arm per-checkpoint refits (`collapse_law.md` §2c): at each
checkpoint step every run is labelled by its **own** dp1 ΔCE at that step (`> cut`), not by its
endpoint fate, and the law is refit on those labels.  Row unit of the table: one checkpoint step
(336 runs each).  `K* (dd exps)` holds the endpoint exponents fixed and moves only the threshold;
`b, c, K* (free)` refit everything (needs ≥ 8 dead and ≥ 8 alive at that step; MLE ratios only,
near-separation as before).  Dead-set completeness = fraction of the endpoint dead set already
over the cut at that step.
""")

code(r'''
lab = tr.dropna(subset=["delta_ce"]).copy()
lab["collapsed"] = (lab["delta_ce"] > CUT).astype(int)
lab["student_tag"] = lab["tag"]
end_dead = set(dead["tag"])
rows_t = []
for step, g in lab.groupby("step"):
    row = {"step": int(step), "n": len(g), "n_dead": int(g["collapsed"].sum()),
           "dead_set_complete": len(set(g.loc[g["collapsed"] == 1, "tag"]) & end_dead) / max(1, len(end_dead)),
           "n_revive_alive": int(((g["collapsed"] == 0) & g["tag"].isin(end_dead)).sum()),
           "n_transient_dead": int(((g["collapsed"] == 1) & ~g["tag"].isin(end_dead)).sum())}
    if g["collapsed"].nunique() == 2:
        row["kstar_fixed"] = cl.kstar_logistic(g, B_DD, C_DD)["kstar"]
        if row["n_dead"] >= 8 and len(g) - row["n_dead"] >= 8:
            try:
                f = cl.fit_collapse_law(g)
            except np.linalg.LinAlgError:   # perfectly separated at this step: ratios undefined
                f = None
            if f is not None and np.isfinite(f.b) and np.isfinite(f.c):
                k_free = cl.kstar_logistic(g, f.b, f.c)["kstar"]
                row.update(b=f.b, c=f.c, kstar_free=k_free, misses=cl.law_misses(g, f.b, f.c, k_free)[0])
    rows_t.append(row)
BT = pd.DataFrame(rows_t)
BT["kstar_fixed_rel"] = BT["kstar_fixed"] / KSTAR_DD
show = BT[["step", "n_dead", "dead_set_complete", "n_transient_dead", "kstar_fixed", "kstar_fixed_rel", "b", "c", "kstar_free", "misses"]]
print("per-checkpoint refit (label = each run's own dCE at that step > cut):")
display(show.round({"dead_set_complete": 2, "kstar_fixed": 6, "kstar_fixed_rel": 2, "b": 3, "c": 3, "kstar_free": 6}).to_string(index=False))

ok = BT.dropna(subset=["b"])
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
ax = axes[0]
ax.plot(BT["step"], BT["kstar_fixed"], "o-", ms=4, color="tab:blue", label="K* at endpoint exponents (b, c fixed)")
ax.plot(ok["step"], ok["kstar_free"], "s--", ms=4, color="tab:orange", label="K* with (b, c) refit at that step")
ax.axhline(KSTAR_DD, color="k", ls=":", lw=1, label=f"endpoint K* {KSTAR_DD:.2e}")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("K*(step)"); ax.set_title("Boundary threshold along training")
aplots.right_legend(ax, fontsize=8)
ax = axes[1]
ax.plot(ok["step"], ok["b"], "o-", ms=4, color="tab:green", label="b (alpha exponent)")
ax.plot(ok["step"], ok["c"], "s-", ms=4, color="tab:red", label="c (rank exponent)")
ax.axhline(B_DD, color="tab:green", ls=":", lw=1, label=f"endpoint b {B_DD:.3f}")
ax.axhline(C_DD, color="tab:red", ls=":", lw=1, label=f"endpoint c {C_DD:.3f}")
ax.axhline(cl.B_LAW, color="gray", ls="--", lw=1, label=f"160m b {cl.B_LAW}")
ax.axhline(cl.C_LAW, color="gray", ls="-.", lw=1, label=f"160m c {cl.C_LAW}")
ax.set_xlabel("step"); ax.set_ylabel("exponent"); ax.set_title("Exponents refit per checkpoint")
aplots.right_legend(ax, fontsize=8)
ax = axes[2]
ax.plot(BT["step"], BT["n_dead"], "o-", ms=4, color="tab:red", label="runs over the cut at this step")
ax.plot(BT["step"], BT["n_transient_dead"], "s-", ms=4, color="tab:purple", label="of which alive at the endpoint (transient)")
ax.axhline(len(dead), color="k", ls=":", lw=1, label=f"endpoint dead {len(dead)}")
ax.set_xlabel("step"); ax.set_ylabel("runs"); ax.set_title("Dead count along training")
aplots.right_legend(ax, fontsize=8)
plt.show()

first = BT[BT["dead_set_complete"] >= 0.95].iloc[0]
print(f"endpoint dead set 95% complete by step {int(first['step'])} of {n_steps:,}; "
      f"K*(fixed exps) spans {BT['kstar_fixed'].min():.2e}-{BT['kstar_fixed'].max():.2e} "
      f"({BT['kstar_fixed'].max() / BT['kstar_fixed'].min():.2f}x) from step {int(BT['step'].min())} to the end; "
      f"free b range {ok['b'].min():.3f}-{ok['b'].max():.3f}, c range {ok['c'].min():.3f}-{ok['c'].max():.3f}.")
''')


def build_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_markdown_cell(source) if kind == "markdown"
            else nbf.v4.new_code_cell(source)
            for kind, source in CELLS
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )


if __name__ == "__main__":
    import sys

    from nbclient import NotebookClient

    notebook = build_notebook()
    client = NotebookClient(
        notebook, timeout=3600, kernel_name="python3",
        resources={"metadata": {"path": str(REPO / "notebooks")}},
    )
    try:
        client.execute()
    except Exception as exc:
        print(f"EXECUTION FAILED: {exc}", file=sys.stderr)
        nbf.write(notebook, OUT)
        raise
    nbf.write(notebook, OUT)
    print(f"wrote executed notebook -> {OUT} ({len(notebook.cells)} cells)")
