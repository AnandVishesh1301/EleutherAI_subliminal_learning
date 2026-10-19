"""Dog, Δp(trait) = p_student − p_control at the same step (decimals, no prior division).
Left/middle: same-model arms split by whether the curve decays after an early peak.
Right: the four cross conditions pooled across models (mean of per-arm means).
Band = 95% teacher-cluster bootstrap (B=1000, seed 0) of the statistic.\nRight panel has its own y scale by default; ``--shared-y`` shares it with the same-model panels."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.analysis.setup import bootstrap, set_plot_style

STAT = sys.argv[1] if len(sys.argv) > 1 else "mean"
SHARED_Y = "--shared-y" in sys.argv   # default: right panel on its own y scale
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
REPO, ROOT = bootstrap()
set_plot_style()
OUT = OUT or REPO / "notebooks/figures/asymmetry/trajectory/peak_early"
t0 = time.time()
arms = tc.earlystop_arms(REPO)
data = tc.with_ep20_endpoints(tc.load_arms(arms, ROOT))
print(f"loaded {len(data.points):,} points in {time.time()-t0:.0f}s", file=sys.stderr)
dog = [s for s in tc.curve_arms(data) if s.animal == "dog"]
labels = {s.label for s in dog}
ctl = {s.control_of: s.label for s in dog if s.kind == "control"}
treat = [s for s in dog if s.kind != "control" and ctl.get(s.label) in labels]
t = tc._alive_points(data, [s.label for s in treat])
c = tc._alive_points(data, [ctl[s.label] for s in treat])
of = {v: k for k, v in ctl.items()}
c = (c.assign(arm=c["arm"].map(of)).groupby(["arm", "cell", "step"])["ratio"].median().rename("ctl").reset_index())
m = t.merge(c, on=["arm", "cell", "step"])
m["dp"] = m["ratio"] * m["arm"].map(data.priors) - m["ctl"] * m["arm"].map(lambda a: data.priors[ctl[a]])
m = m.drop_duplicates(["arm", "teacher", "cell", "step"])
spec = {s.label: s for s in treat}
m["group"] = m["arm"].map({s.label: tc.display_group(s) for s in treat})
rng = np.random.default_rng(0)
B = 1000
agg = np.mean if STAT == "mean" else np.median

def arm_curve(g):
    """per step: stat over slots, band from teacher-cluster bootstrap"""
    rows = []
    for st, x in g.groupby("step"):
        tm = x.groupby("teacher")["dp"].apply(lambda v: agg(v.to_numpy()))
        obs = agg(x["dp"].to_numpy())
        idx = rng.integers(0, len(tm), size=(B, len(tm)))
        boots = agg(tm.to_numpy()[idx], axis=1)
        rows.append((st, obs, *np.percentile(boots, [2.5, 97.5]), len(x), len(tm)))
    return pd.DataFrame(rows, columns=["step", "value", "lo", "hi", "n", "n_teachers"])

def pooled_curve(g):
    """per step: stat over arms of per-arm stat; band resamples teachers within each arm"""
    rows = []
    n_arms = g["arm"].nunique()   # only steps every arm of the group has saved (no composition drift)
    full = g.groupby("step")["arm"].nunique()
    g = g[g["step"].isin(full[full == n_arms].index)]
    for st, x in g.groupby("step"):
        per_arm = x.groupby(["arm", "teacher"])["dp"].apply(lambda v: agg(v.to_numpy()))
        arms_ = sorted(set(per_arm.index.get_level_values(0)))
        obs = agg([agg(x[x["arm"] == a]["dp"].to_numpy()) for a in arms_])
        boots = np.zeros(B)
        for a in arms_:
            tm = per_arm.loc[a].to_numpy()
            idx = rng.integers(0, len(tm), size=(B, len(tm)))
            boots += agg(tm[idx], axis=1)
        boots /= len(arms_)
        rows.append((st, obs, *np.percentile(boots, [2.5, 97.5]), len(x), len(arms_)))
    return pd.DataFrame(rows, columns=["step", "value", "lo", "hi", "n", "n_arms"])

same = m[m["group"] == "Pythia same-model"]
curves = {a: arm_curve(g) for a, g in same.groupby("arm")}
shape = pd.DataFrame({a: {"peak step": cv.loc[cv["value"].idxmax(), "step"], "peak": cv["value"].max(),
                          "end": cv.iloc[-1]["value"], "end/peak": cv.iloc[-1]["value"] / cv["value"].max()}
                      for a, cv in curves.items()}).T.sort_values("end/peak")
print(shape.round(3).to_string())
decays = [a for a in curves if shape.loc[a, "end/peak"] < 0.7]   # endpoint below 70% of peak
sustains = [a for a in curves if a not in decays]
cross_groups = ["diff init, diff data order", "same init, diff data order", "diff init, same data order", "Pythia-160M re-pretrain"]
pooled = {g: pooled_curve(m[m["group"] == g]) for g in cross_groups}
print({g: (int(pooled[g]["n_arms"].iloc[0]), round(pooled[g]["value"].max(), 4)) for g in cross_groups})

rows = []
for a, cv in curves.items():
    rows.append(cv.assign(panel="same-model", arm=tc.display(a.split(" (")[0]), decays=a in decays))
for g, cv in pooled.items():
    rows.append(cv.assign(panel="cross pooled", arm=g, decays=False))
tab = pd.concat(rows, ignore_index=True)
tab.insert(0, "stat", STAT)
stem = f"peak_early_dog{'' if STAT == 'mean' else '_median'}{'_sharedy' if SHARED_Y else ''}"
tab.to_csv(OUT / f"{stem}.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=SHARED_Y, layout="constrained")
fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
def name(a): return tc.display(a.split("→")[0])
for ax, group, title in ((axes[0], decays, "same-model: endpoint < 70% of peak"), (axes[1], sustains, "same-model: endpoint ≥ 70% of peak")):
    for k, a in enumerate(sorted(group, key=name)):
        cv = curves[a]
        col = f"C{k}"
        ax.plot(cv["step"], cv["value"], "-", marker=".", ms=4, lw=1.2, color=col, label=name(a))
        ax.fill_between(cv["step"], cv["lo"], cv["hi"], color=col, alpha=0.18, lw=0)
    ax.set_title(f"dog · {title}", fontsize=10, loc="left")
    ax.legend(fontsize=8, framealpha=0.6, ncol=2)
ax = axes[2]
for k, g in enumerate(cross_groups):
    cv = pooled[g]
    col = f"C{k}"
    ax.plot(cv["step"], cv["value"], "-", marker=".", ms=4, lw=1.2, color=col, label=f"{g} ({int(cv['n_arms'].iloc[0])} pairs)")
    ax.fill_between(cv["step"], cv["lo"], cv["hi"], color=col, alpha=0.18, lw=0)
ax.set_title("dog · other pretraining cond., pooled across models", fontsize=10, loc="left")
ax.legend(fontsize=8, framealpha=0.6)
for ax in axes:
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("student step")
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel(f"{STAT} Δp(trait)")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / stem, f"total {time.time()-t0:.0f}s", file=sys.stderr)
