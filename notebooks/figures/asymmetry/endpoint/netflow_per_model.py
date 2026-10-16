"""Net flow per model and grid: (row sum − column sum) / n models, row = model as sender, column =
model as receiver; positive = sends more than it receives. Values from the asymmetry readout
pairs table (delta = pp, or prior units). Usage: fig_asym_netflow.py [delta|prior]"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
mode = sys.argv[1] if len(sys.argv) > 1 else "delta"
p = pd.read_csv(f"/mnt/cold-1/soar-subliminal/angela/outputs/analysis/transfer_asymmetry/pairs{'_delta' if mode=='delta' else ''}.csv")
GRIDS = [("Pythia-160m, dog\ndifferent init, different data order", "cross-seed", "dog", ["orig","fs1","fs2","fs3"], "#1f77b4"),
         ("Pythia-160m, dog\nsame init, different data order", "cross-data-order", "dog", ["orig","ds1","ds2","ds3"], "#1f77b4"),
         ("Pythia-160m re-pretrained, dog\ndifferent init, same data order", "re-pretrain", "dog", ["prv1","wc1","wc2"], "#1f77b4"),
         ("Pythia-160m re-pretrained, dog\nsame recipe, re-run (pr1234⇄prv1 not run)", "re-pretrain", "dog", ["pr1234","orig","prv1"], "#1f77b4"),
         ("Pythia-160m, cat\ndifferent init, different data order", "cross-seed", "cat", ["orig","fs1","fs2","fs3"], "#e8632b"),
         ("Pythia-160m, lion\ndifferent init, different data order", "cross-seed", "lion", ["orig","fs1","fs2","fs3"], "#2ca25f"),
         ("Pythia-160m, pig\ndifferent init, different data order", "cross-seed", "pig", ["orig","fs1","fs2","fs3"], "#e6a700"),
         ("OLMo-1B, elephant\nsame init, different pretraining corpus", "olmo-recipe", "elephant", ["dd-1b","dd-1b-dclm","dd-1b-c4"], "#777777")]
SHORT = {"dd-1b": "dolma", "dd-1b-dclm": "dclm", "dd-1b-c4": "c4"}
fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for ax, (title, fam, an, bases, col) in zip(axes.ravel(), GRIDS):
    g = p[(p.family == fam) & (p.animal == an)]
    M = pd.DataFrame(np.nan, index=bases, columns=bases)
    for _, r in g.iterrows():
        if r.A in bases and r.B in bases:
            M.loc[r.A, r.B] = r["mean A->B"]; M.loc[r.B, r.A] = r["mean B->A"]
    n = len(bases)
    net = [(np.nansum(M.loc[b].drop(b)) - np.nansum(M[b].drop(b))) / n for b in bases]
    y = np.arange(n)[::-1]
    ax.barh(y, net, color=col, height=0.6)
    ax.axvline(0, color="#444444", lw=0.8)
    for yi, v in zip(y, net):
        ax.text(v + (0.02 if v >= 0 else -0.02) * max(abs(min(net)), max(net), 1e-9) * 1.5, yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([SHORT.get(b, b) for b in bases], fontsize=9)
    lim = max(abs(v) for v in net) * 1.45
    ax.set_xlim(-lim, lim); ax.set_title(title, loc="left", fontsize=8.5); ax.tick_params(length=0, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("(row sum − column sum) / n models" + ("  [pp]" if mode == "delta" else "  [×prior]"), fontsize=7.5)
fig.text(0.01, 0.005, "positive = model sends more than it receives (row = as sender, column = as receiver); the pr1234/orig/prv1 grid lacks the pr1234⇄prv1 pair", fontsize=7.5, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = f"{__import__('pathlib').Path(__file__).resolve().parent}/fig_asym_netflow_{mode}.png"; fig.savefig(out, dpi=150); print(out)
