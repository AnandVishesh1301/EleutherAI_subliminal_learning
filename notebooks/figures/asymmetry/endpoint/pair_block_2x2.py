"""2x2 teacher-base x student-base block (off-diagonals only; same-model diagonal hatched) for one model pair, one panel per animal, every cell loaded
at core30 (10 canonical teacher configs, student seed 1, matched base-pool control):
value = mean (student - matched control) / receiver prior. Missing arm -> hatched.
Usage: uv run python fig_transfer_2x2.py <A> <B> <animals comma-separated>"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from subliminal_learning.analysis import transfer_significance_io as tio
REPO = Path("/home/angela/EleutherAI_subliminal_learning"); HERE = Path(__file__).resolve().parent
A, B = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("pr1234", "orig")
animals = sys.argv[3].split(",") if len(sys.argv) > 3 else ["dog", "cat"]
CMAP = {"dog": "Blues", "cat": "Oranges", "lion": "Greens", "pig": "YlOrBr"}
cells = tio.resolve_cells(REPO / "run_logs/core30_cells.json")
def arm(animal, s, r):
    teachers = json.loads((REPO / f"run_logs/{animal}_top10_kband_configs.json").read_text())["param_tags"]
    try:
        m = tio.load_cell_matrix(animal, s, r, teachers, cells=cells, ftseed=1)
    except Exception as e:
        return np.nan, f"{s}→{r}: load error {type(e).__name__}"
    if not m.cells:
        return np.nan, f"{s}→{r}: no matched control"
    treat = np.delete(m.values, m.control_col, axis=1); ctrl = m.values[:, m.control_col]
    ok = np.isfinite(treat) & np.isfinite(ctrl)[:, None]
    if ok.sum() == 0:
        return np.nan, f"{s}→{r}: no students"
    prior = tio.student_tree_prior(animal, r)
    note = "" if ok.sum() >= len(cells) * len(teachers) else f"{s}→{r}: {int(ok.sum())}/{len(cells)*len(teachers)} runs"
    return float(((treat - ctrl[:, None])[ok] / prior).mean()), note
fig, axes = plt.subplots(1, len(animals), figsize=(3.7 * len(animals), 3.9))
notes = []
for ax, animal in zip(np.atleast_1d(axes), animals):
    M = pd.DataFrame(np.nan, index=[A, B], columns=[A, B])
    for s in (A, B):
        for r in (A, B):
            if s == r:
                continue
            M.loc[s, r], n = arm(animal, s, r)
            if n: notes.append(f"{animal} {n}")
    vmax = np.nanmax(M.values) if np.isfinite(M.values).any() else 1.0
    cmap = plt.get_cmap(CMAP.get(animal, "Greys"))
    ax.imshow(np.ma.masked_invalid(M.values), cmap=cmap, vmin=0, vmax=vmax)
    for i in range(2):
        for j in range(2):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}", ha="center", va="center", fontsize=11,
                        color="white" if v / vmax > 0.55 else "#222222")
            else:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="#bbbbbb", lw=0))
    ax.set_xticks([0, 1]); ax.set_xticklabels([A, B]); ax.set_yticks([0, 1]); ax.set_yticklabels([A, B])
    ax.set_xlabel("receiver (student base)"); ax.set_ylabel("sender (teacher base)")
    ax.set_title(animal, loc="left", fontsize=11, color=cmap(0.85)); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
fig.suptitle(f"{A} × {B}: mean (student − matched control) / receiver prior, core30, 10 teacher configs, student seed 1", fontsize=9, x=0.01, ha="left")
if notes: fig.text(0.01, 0.01, "\n".join(notes), fontsize=7, color="#555555", va="bottom")
fig.tight_layout(rect=(0, 0.06 + 0.03 * len(notes), 1, 0.93))
out = HERE / f"fig_transfer_2x2_{A}_{B}.png"; fig.savefig(out, dpi=150); print(out); print("\n".join(notes) or "all four cells complete in every panel")
