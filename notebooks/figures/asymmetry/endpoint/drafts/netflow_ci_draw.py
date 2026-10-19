"""netflow_delta with 95% intervals (black whiskers) and recipe-level net scores (dots)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "notebooks/figures/asymmetry/endpoint")
import netflow as nf  # noqa: E402
S = Path(sys.argv[1])
T = pd.read_csv(S / "netflow_delta_ci.csv"); P = pd.read_csv(S / "netflow_delta_recipes.csv")
p = nf.load_pairs("delta")
mats = [(title, col, nf.matrix(p, fam, an, bases)) for title, fam, an, bases, col in nf.GRIDS]
fig, axes = plt.subplots(2, 4, figsize=(10.0, 6.4))
nets = nf.draw_netflow(axes, mats, "delta")
fig.tight_layout(w_pad=0.2, h_pad=1.2, pad=0.9)
nf.scale_rows(axes, nets, "row")
rng = np.random.default_rng(0)
ends = {}
for k, (ax, (title, col, M)) in enumerate(zip(axes.ravel(), mats)):
    key = title.replace("\n", " · ")
    t = T[T["grid"] == key].set_index("model"); pts = P[P["grid"] == title]
    bases = list(M.index); n = len(bases); y = np.arange(n)[::-1]; dec = 1 if k < 4 else 2
    for txt in list(ax.texts): txt.remove()
    lo_all, hi_all = ax.get_xlim()
    for b, yi, v in zip(bases, y, nets[k]):
        name = nf.label(b, wrap=False); lo, hi = t.loc[name, "lo"], t.loc[name, "hi"]
        ax.plot([lo, hi], [yi, yi], color="black", lw=1.1, solid_capstyle="butt", zorder=4)
        ax.plot([lo, lo], [yi - .12, yi + .12], color="black", lw=1.1, zorder=4); ax.plot([hi, hi], [yi - .12, yi + .12], color="black", lw=1.1, zorder=4)
        q = pts[pts["model"] == b]
        if len(q):
            ax.scatter(q["net"], yi + rng.uniform(-.18, .18, len(q)), s=7, color="#222222", alpha=.55, zorder=5, lw=0)
        end = max(v, hi) if v >= 0 else min(v, lo)
        if len(q): end = max(end, q["net"].max()) if v >= 0 else min(end, q["net"].min())
        span = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
        ax.text(end + (span if v >= 0 else -span), yi, f"{v:+.{dec}f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ends.setdefault(k // 4, []).append(end)
    # labels sit beyond the whisker/points; shared x range per row with room for the text
for row in range(2):
    lo = min(min(ends[row]), 0); hi = max(max(ends[row]), 0); width = hi - lo
    for ax in axes[row]:
        ax.set_xlim(lo - 0.42 * width, hi + 0.22 * width)
for ext in ("png", "pdf"):
    fig.savefig(S / f"netflow_delta_ci.{ext}", dpi=150)
print(S / "netflow_delta_ci.png")
