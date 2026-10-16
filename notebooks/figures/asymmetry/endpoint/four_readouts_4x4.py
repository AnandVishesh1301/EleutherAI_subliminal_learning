"""One animal per figure: a 2x2 grid of 4x4 sender x receiver matrices (same-model diagonal hatched, not loaded), each
under a different readout condition, all arms read at core30 (10 teacher configs, student seed 1):
  (a) mean student trait preference (raw p_target)
  (b) mean gap over matched control, percentage points
  (c) mean gap over matched control / receiver prior
  (d) share of runs above their own matched control
Usage: uv run python fig_transfer_2x2of4x4.py <animal> [family: fs|ds]"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pathlib import Path
from subliminal_learning.analysis import transfer_significance_io as tio
REPO = Path("/home/angela/EleutherAI_subliminal_learning"); HERE = Path(__file__).resolve().parent
animal = sys.argv[1] if len(sys.argv) > 1 else "dog"
fam = sys.argv[2] if len(sys.argv) > 2 else "fs"
bases = ["orig"] + [f"{fam}{i}" for i in (1, 2, 3)]
teachers = json.loads((REPO / f"run_logs/{animal}_top10_kband_configs.json").read_text())["param_tags"]
cells = tio.resolve_cells(REPO / "run_logs/core30_cells.json")
conds = {"mean student preference (raw)": "raw", "gap over matched control (pp)": "pp",
         "gap over control / receiver prior": "prior", "share of runs above own control": "exc"}
M = {k: pd.DataFrame(np.nan, index=bases, columns=bases) for k in conds.values()}
notes = []
for s in bases:
    for r in bases:
        if s == r:
            continue
        m = tio.load_cell_matrix(animal, s, r, teachers, cells=cells, ftseed=1)
        if not m.cells:
            notes.append(f"{s}->{r}: no matched control"); continue
        treat = np.delete(m.values, m.control_col, axis=1); ctrl = m.values[:, m.control_col]
        ok = np.isfinite(treat) & np.isfinite(ctrl)[:, None]
        n_ok = int(ok.sum())
        if n_ok < 0.9 * len(cells) * len(teachers):
            notes.append(f"{s}->{r}: {n_ok}/{len(cells)*len(teachers)} runs (shown)")
        prior = tio.student_tree_prior(animal, r)
        gap = (treat - ctrl[:, None])[ok]
        M["raw"].loc[s, r] = treat[ok].mean(); M["pp"].loc[s, r] = 100 * gap.mean()
        M["prior"].loc[s, r] = (gap / prior).mean(); M["exc"].loc[s, r] = (gap > 0).mean()
fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.2))
for ax, (title, key) in zip(axes.ravel(), conds.items()):
    A = M[key].values.astype(float)
    if key == "exc":
        norm = plt.Normalize(0, 1)
    else:
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax(A))
    ax.imshow(np.ma.masked_invalid(A), cmap="Blues", norm=norm)
    for i in range(4):
        for j in range(4):
            v = A[i, j]
            if not np.isfinite(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="#bbbbbb", lw=0)); continue
            txt = f"{v:.3f}" if key == "raw" else (f"{v:.2f}" if key == "exc" else (f"{v:.1f}" if abs(v) >= 0.95 else f"{v:.2f}"))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if norm(max(v, 0)) > 0.55 else "#222222")
    ax.set_xticks(range(4)); ax.set_xticklabels(bases, fontsize=8); ax.set_yticks(range(4)); ax.set_yticklabels(bases, fontsize=8)
    ax.set_xlabel("receiver (student base)", fontsize=8); ax.set_ylabel("sender (teacher base)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=9.5); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
prior_line = ", ".join(f"{b} {tio.student_tree_prior(animal, b):.4f}" for b in bases)
fig.suptitle(f"{animal}: orig + {fam}1–3, the 12 cross-model arms at core30 (10 teacher configs × 30 cells, student seed 1)\nreceiver priors: {prior_line}", fontsize=9.5, x=0.02, ha="left")
if notes: fig.text(0.02, 0.005, "; ".join(notes), fontsize=7, color="#555555")
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
out = HERE / f"fig_transfer_2x2of4x4_{animal}_{fam}.png"; fig.savefig(out, dpi=150); print(out, notes)
