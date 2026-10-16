"""Plain sender x receiver heatmaps per family from the transfer-asymmetry readout's pairs.csv
(mean (student - matched control) / receiver prior; both directions of every complete pair).
Diagonals are not part of the readout -> hatched; pairs excluded by the census -> hatched grey.
Usage: uv run python notebooks/figures/asymmetry/endpoint/fig_transfer_matrices.py [shared|perpanel] [pairs.csv]"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
HERE = str(__import__("pathlib").Path(__file__).resolve().parent)
mode = sys.argv[1] if len(sys.argv) > 1 else "shared"
csv = sys.argv[2] if len(sys.argv) > 2 else "/mnt/cold-1/soar-subliminal/angela/outputs/analysis/transfer_asymmetry/pairs.csv"
pairs = pd.read_csv(csv)
ORDER = {"cross-data-order": ["orig", "ds1", "ds2", "ds3"], "cross-seed": ["orig", "fs1", "fs2", "fs3"],
         "re-pretrain": ["orig", "pr1234", "prv1", "wc1", "wc2"], "olmo-recipe": ["dd-1b", "dd-1b-dclm", "dd-1b-c4"]}
TITLE = {"cross-data-order": "same init, different data order", "cross-seed": "different init, different data order",
         "re-pretrain": "160m re-pretrainings (core30)", "olmo-recipe": "OLMo-1B recipes (core35)"}
SHORT = {"dd-1b": "dolma", "dd-1b-dclm": "dclm", "dd-1b-c4": "c4"}
panels = []
for (fam, animal), g in pairs.groupby(["family", "animal"], sort=False):
    bases = [b for b in ORDER[fam] if b in set(g.A) | set(g.B)]
    M = pd.DataFrame(np.nan, index=bases, columns=bases)
    for _, r in g.iterrows():
        M.loc[r.A, r.B] = r["mean A->B"]; M.loc[r.B, r.A] = r["mean B->A"]
    panels.append((f"{animal}: {TITLE[fam]}", M))
ncol = 4 if len(panels) > 4 else len(panels); nrow = -(-len(panels) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow)); axes = np.atleast_1d(axes).ravel()
for ax in axes[len(panels):]: ax.axis("off")
vmax_all = np.nanmax([np.nanmax(m.values) for _, m in panels])
for ax, (title, M) in zip(axes, panels):
    vmax = vmax_all if mode == "shared" else np.nanmax(M.values)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    ax.imshow(np.ma.masked_invalid(M.values), cmap="Blues", norm=norm)
    n = len(M)
    for i in range(n):
        for j in range(n):
            v = M.values[i, j]
            if i == j or not np.isfinite(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="#bbbbbb", lw=0))
            else:
                ax.text(j, i, f"{v:.1f}" if abs(v) >= 0.95 else f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm(max(v, 0)) > 0.55 else "#222222")
    labels = [SHORT.get(b, b) for b in M.index]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8, rotation=30 if n > 4 else 0)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("receiver (student base)", fontsize=8); ax.set_ylabel("sender (teacher base)", fontsize=8)
    ax.set_title(title + ("" if mode == "shared" else f"\n(scale max {vmax:.1f})"), fontsize=8.5, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Transfer matrices: mean (student − matched control) / receiver prior, 10 teacher configs per sender, student seed 1; hatched = not run / no matched control / diagonal"
             + ("  — one colour scale (sqrt)" if mode == "shared" else "  — per-panel colour scale (sqrt)"), fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{HERE}/fig_transfer_matrices_{mode}.png", dpi=150)
print("panels:", [t for t, _ in panels])
