"""Re-render notebooks/figures/drafts/fig1_B_four_measures.png from the live headline families.

Same design as the 09-21 03:05 draft (symlog violins of all runs per family × animal, dots = arm
medians, tick = 95th pct), plus ONE grey `all animals` aggregate row per family that carries more
than one animal: its violin pools every run of the family, its dots are every arm's median.
"""
# ruff: noqa: E401, E501, E701, E702  -- draft figure script, kept in its one-statement-per-line-of-plot style
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis.setup import bootstrap
REPO, _ = bootstrap()
ANIMALS = ["dog", "cat", "lion", "pig"]
TEACHERS = {a: json.loads((REPO / f"run_logs/{a}_top10_kband_configs.json").read_text())["param_tags"] for a in ANIMALS}
fams = th.headline_families(REPO, TEACHERS)
COL = {"dog": "#2a78d6", "cat": "#eb6834", "lion": "#1baf7a", "pig": "#eda100", "elephant": "#e87ba4", "all": "#444444"}
SHORT = {"same-model": "same model", "same init, different data order": "same init,\ndiff. data order",
         "different init, same data order (v1 tree)": "diff. init,\nsame order (v1)",
         "different init, different data order (full seeds)": "diff. init,\ndiff. data order",
         "cross width, small→large, same data order": "160m → 410m", "cross width, large→small, same data order": "410m → 160m",
         "cross width, small→large, different data order": "160m → 410m-s1", "cross width, large→small, different data order": "410m-s1 → 160m",
         "same init, different corpus (OLMo dolma, dclm, c4 recipes)": "OLMo recipes\n(dolma, dclm, c4)",
         "different init, different corpus (OLMo → large-aux-2 receivers)": "OLMo → seed-2\nreceivers",
         "different init, same corpus (OLMo cross-seed)": "OLMo seeds",
         "cross architecture, small→large (160m → OLMo dolma)": "160m → OLMo",
         "cross architecture, large→small (OLMo dolma → 160m)": "OLMo → 160m"}
MEAS = [("p_student − p_control", 1e-3), ("(p_student − p_control) / prior", 0.1),
        ("(p_student − p_control) / max control of the arm", 0.1), ("(p_student − p_control) / matched control", 0.1)]

def vals(a):
    m = a.matrix; ctrl = m.values[:, m.control_col]; tr = np.delete(m.values, m.control_col, axis=1); ok = np.isfinite(tr)
    gap = (tr - ctrl[:, None])[ok]; c = np.broadcast_to(ctrl[:, None], tr.shape)[ok]
    return [gap, gap / a.prior, gap / np.nanmax(ctrl), gap / c]

def T(x, lin):
    x = np.asarray(x, float); return np.sign(x) * np.log10(1 + np.abs(x) / lin)

AGGREGATE = False   # per-animal rows only (Angela 09-21: arm-median aggregate rows rejected)
rows = []   # (family, animal-or-'all', [per-arm value lists], n_arms)
for f in fams:
    fam_rows = []
    for an in ["dog", "cat", "lion", "pig", "elephant"]:
        arms = [a for a in f.arms if a.animal == an and a.matrix is not None]
        if arms: fam_rows.append((f.family, an, [vals(a) for a in arms]))
    if AGGREGATE and len(fam_rows) > 1:   # the aggregate: every run of every animal, every arm's median
        fam_rows.append((f.family, "all", [p for _, _, per in fam_rows for p in per]))
    rows.extend(fam_rows)

fig, axes = plt.subplots(1, 4, figsize=(17, 0.3 * len(rows) + 2), sharey=True)
ypos, ylab, bounds = [], [], []; y = 0; last = None
for fam, an, per in rows:
    if fam != last:
        if last is not None: y += 0.9
        bounds.append((fam, y)); last = fam
    for k, (ax, (name, lin)) in enumerate(zip(axes, MEAS)):
        arm_meds = np.array([np.median(p[k]) for p in per])
        if an == "all":
            # aggregate row: ONE value per arm (its median), every arm weighted equally -- no
            # run-count or animal-count weighting; violin + tick summarise the arm medians
            allr = arm_meds
        else:
            allr = np.concatenate([p[k] for p in per]); allr = allr[np.isfinite(allr)]
        if len(allr) >= (4 if an == "all" else 2):   # no density over 2-3 arm medians
            parts = ax.violinplot([T(allr, lin)], positions=[y], vert=False, widths=0.85, showextrema=False)
            for b in parts["bodies"]: b.set_facecolor(COL[an]); b.set_alpha(0.28 if an != "all" else 0.22); b.set_edgecolor("none")
        meds = T(arm_meds, lin); jit = np.linspace(-0.25, 0.25, len(meds)) if len(meds) > 1 else [0]
        ax.plot(meds, y + np.asarray(jit), "o", ms=3.8 if an != "all" else 3.2, color=COL[an], mec="white", mew=0.5, zorder=3)
        if an == "all":
            ax.plot(T(np.median(allr), lin), y, marker="|", ms=11, color=COL[an], ls="none", zorder=4, mew=1.6)
        else:
            ax.plot(T(np.percentile(allr, 95), lin), y, marker="|", ms=9, color=COL[an], ls="none", zorder=3)
    ypos.append(y); ylab.append(f"all arms ({len(per)})" if an == 'all' else f"{an} ({len(per)})"); y += 1
TICKS = {1e-3: ([-0.1, -0.01, 0, 0.01, 0.1, 1], ["−.1", "−.01", "0", ".01", ".1", "1"]),
         0.1: ([-10, -1, 0, 1, 10, 100, 1000], ["−10", "−1", "0", "1", "10", "100", "1000"])}
for ax, (name, lin) in zip(axes, MEAS):
    ax.axvline(0, color="0.3", lw=0.8, ls="--"); ax.grid(axis="x", color="0.92"); ax.set_axisbelow(True)
    ax.set_title(name, fontsize=9.5); ax.spines[["top", "right"]].set_visible(False)
    tv, tl = TICKS[lin]; ax.set_xticks(T(tv, lin)); ax.set_xticklabels(tl, fontsize=8)
    ax.set_xlim(T(tv[0], lin) - 0.1, T(tv[-1], lin) + 0.1)
    for _, yb in bounds: ax.axhline(yb - 0.5, color="0.85", lw=0.6)
axes[0].set_yticks(ypos); axes[0].set_yticklabels(ylab, fontsize=8); axes[0].invert_yaxis()
for lab in axes[0].get_yticklabels():
    if lab.get_text().startswith("all arms"): lab.set_fontweight("bold"); lab.set_color(COL["all"])
for i, (fam, yb) in enumerate(bounds):
    nxt = bounds[i + 1][1] - 0.9 if i + 1 < len(bounds) else y
    axes[0].text(-0.62, (yb - 0.5 + nxt - 0.5) / 2, SHORT.get(fam, fam), transform=axes[0].get_yaxis_transform(),
                 fontsize=8, fontweight="bold", va="center", ha="left")
fig.suptitle("Four effect measures (symlog x, linear near 0) — shaded: all runs of the family × animal; dots: arm medians; tick: 95th pct of runs; dashed = no effect", fontsize=9.5)
fig.tight_layout(rect=(0.10, 0, 1, 0.96))
out = REPO / "notebooks/figures/drafts/fig1_B_four_measures.png"
fig.savefig(out, dpi=120); print("saved", out, "| families:", len(fams), "| rows:", len(rows))
missing = [f.family for f in fams if f.family not in SHORT]
print("families with no SHORT label:", missing)
for f in fams:
    print(f"  {f.family}: finished arms {sum(a.matrix is not None for a in f.arms)}/{len(f.arms)}")
