"""OLMo (elephant), Δp(trait) = p_student − p_control at the same step (decimals, no prior division).
Left: every OLMo same-model arm (dolma, c4, dclm) in one panel (no decay/sustain split).
Right: the OLMo cross conditions (cross-seed, cross-corpus, cross-corpus + cross-seed) pooled
across models (mean of per-arm means). The "OLMo cross-arch." arm (160m → OLMo-1B, dog) is
excluded: different trait and a single arm. No ep20 endpoints are appended (`with_ep20_endpoints`
is for the Pythia ``_stop{N}`` runs); OLMo runs save 625 … 12,500.
Band = 95% teacher-cluster bootstrap (B=1000, seed 0) of the statistic.\nRight panel has its own y scale by default; ``--shared-y`` shares it with the same-model panel."""
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
arms = [s for s in tc.default_arms(REPO) if s.family == "OLMo" and s.animal == "elephant" and s.group != "OLMo cross-arch."]
data = tc.load_arms(arms, ROOT)
print(f"loaded {len(data.points):,} points in {time.time()-t0:.0f}s", file=sys.stderr)
ok = tc.curve_arms(data)
labels = {s.label for s in ok}
ctl = {s.control_of: s.label for s in ok if s.kind == "control"}
treat = [s for s in ok if s.kind != "control" and ctl.get(s.label) in labels]
dropped = [s.label for s in arms if s.kind != "control" and s.label not in {x.label for x in treat}]
print("dropped (not in curve_arms or control missing):", dropped or "none")
t = tc._alive_points(data, [s.label for s in treat])
c = tc._alive_points(data, [ctl[s.label] for s in treat])
of = {v: k for k, v in ctl.items()}
c = (c.assign(arm=c["arm"].map(of)).groupby(["arm", "cell", "step"])["ratio"].median().rename("ctl").reset_index())
m = t.merge(c, on=["arm", "cell", "step"])
m["dp"] = m["ratio"] * m["arm"].map(data.priors) - m["ctl"] * m["arm"].map(lambda a: data.priors[ctl[a]])
m = m.drop_duplicates(["arm", "teacher", "cell", "step"])
spec = {s.label: s for s in treat}
m["group"] = m["arm"].map({s.label: s.condition for s in treat})
print("saved steps:", sorted(int(x) for x in m["step"].unique()))
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


same = m[m["group"] == "same-model"]
curves = {a: arm_curve(g) for a, g in same.groupby("arm")}
shape = pd.DataFrame({a: {"peak step": cv.loc[cv["value"].idxmax(), "step"], "peak": cv["value"].max(),
                          "end": cv.iloc[-1]["value"], "end/peak": cv.iloc[-1]["value"] / cv["value"].max()}
                      for a, cv in curves.items()}).T.sort_values("end/peak")
print(shape.round(3).to_string())
cross_groups = [g for g in ["cross-seed", "cross-corpus", "cross-corpus + cross-seed"] if (m["group"] == g).any()]
pooled = {g: pooled_curve(m[m["group"] == g]) for g in cross_groups}
print({g: (int(pooled[g]["n_arms"].iloc[0]), round(pooled[g]["value"].max(), 4)) for g in cross_groups})

def name(a): return tc.short_base(spec[a].receiver, family_context=False)
CROSS_LABEL = {"cross-seed": "diff init, same corpus", "cross-corpus": "same init, diff corpus",
               "cross-corpus + cross-seed": "diff init, diff corpus"}   # legend text; CSV keeps the group keys
rows = []
for a, cv in curves.items():
    rows.append(cv.assign(panel="same-model", arm=name(a)))
for g, cv in pooled.items():
    rows.append(cv.assign(panel="cross pooled", arm=g))
tab = pd.concat(rows, ignore_index=True)
tab.insert(0, "stat", STAT)
stem = f"peak_early_olmo{'' if STAT == 'mean' else '_median'}{'_sharedy' if SHARED_Y else ''}"
tab.to_csv(OUT / f"{stem}.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9), sharey=SHARED_Y, layout="constrained")
fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
ax = axes[0]
for k, a in enumerate(sorted(curves, key=name)):
    cv = curves[a]
    col = f"C{k}"
    ax.plot(cv["step"], cv["value"], "-", marker=".", ms=4, lw=1.2, color=col, label=name(a))
    ax.fill_between(cv["step"], cv["lo"], cv["hi"], color=col, alpha=0.18, lw=0)
ax.set_title("elephant · same-model", fontsize=10, loc="left")
ax.legend(fontsize=8, framealpha=0.6, ncol=2)
ax = axes[1]
for k, g in enumerate(cross_groups):
    cv = pooled[g]
    col = f"C{k}"
    ax.plot(cv["step"], cv["value"], "-", marker=".", ms=4, lw=1.2, color=col, label=f"{CROSS_LABEL.get(g, g)} ({int(cv['n_arms'].iloc[0])} pairs)")
    ax.fill_between(cv["step"], cv["lo"], cv["hi"], color=col, alpha=0.18, lw=0)
ax.set_title("elephant · other pretraining cond., pooled across models", fontsize=10, loc="left")
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
