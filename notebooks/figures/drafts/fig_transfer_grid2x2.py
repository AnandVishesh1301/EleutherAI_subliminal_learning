"""Per animal: 2x2 layout of family grids (sender x receiver, off-diagonals only):
  {orig, fs1-3} | {orig, ds1-3} | {prv1, wc1, wc2} | OLMo cross-corpus {dolma, dclm, c4} (elephant, grey)
Values = mean (student - matched control) / receiver prior from the transfer-asymmetry readout
(pairs.csv: core56 seed families, core30 re-pretrain, core35 OLMo); arms absent from the table are
loaded directly at the grid's design and hatched when never run / no matched control.
Usage: uv run python fig_transfer_grid2x2.py <animal>"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pathlib import Path
from subliminal_learning.analysis import transfer_significance_io as tio
REPO = Path("/home/angela/EleutherAI_subliminal_learning"); HERE = REPO / "notebooks/figures/drafts"
animal = sys.argv[1] if len(sys.argv) > 1 else "dog"
mode = sys.argv[2] if len(sys.argv) > 2 else "delta"  # delta (pp, per-grid scale) | prior (shared scale)
CMAP = {"dog": "Blues", "cat": "Oranges", "lion": "Greens", "pig": "YlOrBr"}
pairs = pd.read_csv("/mnt/cold-1/soar-subliminal/angela/outputs/analysis/transfer_asymmetry/pairs.csv")
GRIDS = [(f"Pythia-160m, {animal}\ndifferent init, different data order", "cross-seed", animal, ["orig", "fs1", "fs2", "fs3"], "run_logs/core56_cells.json", 1),
         (f"Pythia-160m, {animal}\nsame init, different data order", "cross-data-order", animal, ["orig", "ds1", "ds2", "ds3"], "run_logs/core56_cells.json", 1),
         (f"Pythia-160m re-pretrained, {animal}\ndifferent init, same data order", "re-pretrain", animal, ["prv1", "wc1", "wc2"], "run_logs/core30_cells.json", 1),
         ("OLMo-1B, elephant\nsame init, different pretraining corpus", "olmo-recipe", "elephant", ["dd-1b", "dd-1b-dclm", "dd-1b-c4"], "run_logs/dd1b/olmo_core35_cells.json", 0)]
SHORT = {"dd-1b": "dolma", "dd-1b-dclm": "dclm", "dd-1b-c4": "c4"}
def load_arm(an, s, r, cells_json, tseed):
    if an == "elephant":
        return np.nan, f"{SHORT.get(s,s)}→{SHORT.get(r,r)} not in readout"
    teachers = json.loads((REPO / f"run_logs/{an}_top10_kband_configs.json").read_text())["param_tags"]
    cells = tio.resolve_cells(REPO / cells_json)
    try:
        m = tio.load_cell_matrix(an, s, r, teachers, cells=cells, ftseed=1, tseed=tseed)
    except Exception as e:
        return np.nan, f"{s}→{r} load error {type(e).__name__}"
    if not m.cells:
        return np.nan, f"{s}→{r} no matched control"
    treat = np.delete(m.values, m.control_col, axis=1); ctrl = m.values[:, m.control_col]
    ok = np.isfinite(treat) & np.isfinite(ctrl)[:, None]
    if ok.sum() == 0:
        return np.nan, f"{s}→{r} no students"
    prior = tio.student_tree_prior(an, r)
    val = float((treat - ctrl[:, None])[ok].mean()) * 100 if mode == "delta" else float(((treat - ctrl[:, None])[ok] / prior).mean())
    return val, f"{s}→{r} loaded directly, not in readout ({int(ok.sum())}/{len(cells)*len(teachers)} runs)"
mats, notes = [], []
for title, fam, an, bases, cells_json, tseed in GRIDS:
    g = pairs[(pairs.family == fam) & (pairs.animal == an)]
    M = pd.DataFrame(np.nan, index=bases, columns=bases)
    for _, r in g.iterrows():
        if r.A in bases and r.B in bases:
            pa, pb = (tio.student_tree_prior(an, r.A), tio.student_tree_prior(an, r.B)) if mode == "delta" else (1.0, 1.0)
            k = 100 if mode == "delta" else 1
            M.loc[r.A, r.B] = r["mean A->B"] * pb * k; M.loc[r.B, r.A] = r["mean B->A"] * pa * k
    for s in bases:
        for r in bases:
            if s != r and not np.isfinite(M.loc[s, r]):
                M.loc[s, r], n = load_arm(an, s, r, cells_json, tseed)
                notes.append(n)
    mats.append((title, an, M))
vmax_all = np.nanmax([np.nanmax(M.values) for _, _, M in mats if np.isfinite(M.values).any()])
fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.2))
for ax, (title, an, M) in zip(axes.ravel(), mats):
    cmap = plt.get_cmap(CMAP.get(animal, "Greys") if an == animal else "Greys")
    n = len(M); A = M.values.astype(float)
    vmax = np.nanmax(A) if (mode == "delta" and np.isfinite(A).any()) else vmax_all
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    ax.imshow(np.ma.masked_invalid(A), cmap=cmap, norm=norm)
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            if not np.isfinite(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="#bbbbbb", lw=0)); continue
            ax.text(j, i, (f"{v:.2f}" if mode == "delta" else (f"{v:.1f}" if abs(v) >= 0.95 else f"{v:.2f}")), ha="center", va="center", fontsize=9,
                    color="white" if norm(max(v, 0)) > 0.55 else "#222222")
    labels = [SHORT.get(b, b) for b in M.index]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8.5); ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("receiver (student base)", fontsize=8); ax.set_ylabel("sender (teacher base)", fontsize=8)
    ax.set_title(title + "", loc="left", fontsize=9); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
main = plt.get_cmap(CMAP.get(animal, "Greys"))(0.9)
uniq = [x for x in dict.fromkeys(notes) if "not in readout" not in x]
if uniq: fig.text(0.02, 0.005, "; ".join(uniq), fontsize=6.5, color="#555555", wrap=True)
fig.tight_layout(rect=(0, 0.02 if uniq else 0, 1, 1))
out = HERE / f"fig_transfer_grid2x2_{animal}_{mode}.png"; fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf")); print(out, out.with_suffix(".pdf")); print("; ".join(uniq) or "all arms from the readout")
