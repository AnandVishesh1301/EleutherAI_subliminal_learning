"""Side-by-side figure, one animal: left block = 2x2 family grids (sender x receiver, delta pp, per-grid
colour scale, off-diagonals only); right block = the same four grids' net flow per model,
(row sum − column sum) / n models, from the SAME matrices. Delta = mean(student − matched control)
in percentage points, from the asymmetry readout's pairs_delta.csv (arms missing from it loaded
directly at the grid's design). Usage: family_grids_with_netflow.py <animal>"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pathlib import Path
from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.analysis.trajectory_conditions import display
REPO = Path("/home/angela/EleutherAI_subliminal_learning"); HERE = Path(__file__).resolve().parent
animal = sys.argv[1] if len(sys.argv) > 1 else "dog"
CMAP = {"dog": "Blues", "cat": "Oranges", "lion": "Greens", "pig": "YlOrBr"}
BAR = {"dog": "#1f77b4", "cat": "#e8632b", "lion": "#2ca25f", "pig": "#e6a700"}
pairs = pd.read_csv("/mnt/cold-1/soar-subliminal/angela/outputs/analysis/transfer_asymmetry/pairs_delta.csv")
GRIDS = [(f"Pythia-160m, {animal}\ndifferent init, different data order", "cross-seed", animal, ["orig", "fs1", "fs2", "fs3"], "run_logs/registry/core56_cells.json", 1),
         (f"Pythia-160m, {animal}\nsame init, different data order", "cross-data-order", animal, ["orig", "ds1", "ds2", "ds3"], "run_logs/registry/core56_cells.json", 1),
         (f"Pythia-160m re-pretrained, {animal}\ndifferent init, same data order", "re-pretrain", animal, ["prv1", "wc1", "wc2"], "run_logs/registry/core30_cells.json", 1),
         ("OLMo-1B, elephant\nsame init, different pretraining corpus", "olmo-recipe", "elephant", ["dd-1b", "dd-1b-dclm", "dd-1b-c4"], "run_logs/dd1b/olmo_core35_cells.json", 0)]
SHORT = {"dd-1b": "dolma", "dd-1b-dclm": "dclm", "dd-1b-c4": "c4"}
def load_arm(an, s, r, cells_json, tseed):
    if an == "elephant":
        return np.nan
    teachers = json.loads((REPO / f"run_logs/registry/{an}_top10_kband_configs.json").read_text())["param_tags"]
    cells = tio.resolve_cells(REPO / cells_json)
    try:
        m = tio.load_cell_matrix(an, s, r, teachers, cells=cells, ftseed=1, tseed=tseed)
    except Exception:
        return np.nan
    if not m.cells:
        return np.nan
    treat = np.delete(m.values, m.control_col, axis=1); ctrl = m.values[:, m.control_col]
    ok = np.isfinite(treat) & np.isfinite(ctrl)[:, None]
    return float((treat - ctrl[:, None])[ok].mean()) * 100 if ok.sum() else np.nan
mats = []
for title, fam, an, bases, cells_json, tseed in GRIDS:
    g = pairs[(pairs.family == fam) & (pairs.animal == an)]
    M = pd.DataFrame(np.nan, index=bases, columns=bases)
    for _, r in g.iterrows():
        if r.A in bases and r.B in bases:
            M.loc[r.A, r.B] = r["mean A->B"]; M.loc[r.B, r.A] = r["mean B->A"]
    for s in bases:
        for r in bases:
            if s != r and not np.isfinite(M.loc[s, r]):
                M.loc[s, r] = load_arm(an, s, r, cells_json, tseed)
    mats.append((title, an, M))
fig = plt.figure(figsize=(9.6, 14.4))
gs = fig.add_gridspec(4, 2, width_ratios=[1, 0.95], hspace=0.42, wspace=0.375)
for k, (title, an, M) in enumerate(mats):
    ax = fig.add_subplot(gs[k, 0])
    cmap = plt.get_cmap(CMAP.get(animal, "Greys") if an == animal else "Greys")
    n = len(M); A = M.values.astype(float)
    vmax = np.nanmax(A) if np.isfinite(A).any() else 1.0
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    ax.imshow(np.ma.masked_invalid(A), cmap=cmap, norm=norm)
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            if not np.isfinite(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="#bbbbbb", lw=0)); continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if norm(max(v, 0)) > 0.55 else "#222222")
    labels = [display(SHORT.get(b, b)) for b in M.index]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8.5); ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("receiver (student base)", fontsize=8)
    ax.set_ylabel("sender (teacher base)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=9); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
for k, (title, an, M) in enumerate(mats):
    ax = fig.add_subplot(gs[k, 1])
    bases = list(M.index); n = len(bases)
    net = [(np.nansum(M.loc[b].drop(b)) - np.nansum(M[b].drop(b))) / n for b in bases]
    y = np.arange(n)[::-1]
    col = BAR.get(animal, "#777777") if an == animal else "#777777"
    ax.barh(y, net, color=col, height=0.6)
    ax.axvline(0, color="#444444", lw=0.8)
    lim = max(abs(v) for v in net) * 1.5
    for yi, v in zip(y, net):
        ax.text(v + (0.03 if v >= 0 else -0.03) * lim, yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([display(SHORT.get(b, b)) for b in bases], fontsize=8.5)
    ax.set_xlim(-lim, lim); ax.set_title(title, loc="left", fontsize=9); ax.tick_params(length=0, labelsize=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, symmetric=True))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("(row sum − column sum) / n models  [pp]", fontsize=8)
out = HERE / f"family_grids_with_netflow_{animal}_delta"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(out.with_suffix(".png"), out.with_suffix(".pdf"))
