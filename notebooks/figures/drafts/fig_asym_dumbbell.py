import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP = sys.argv[1]
df = pd.read_csv(f"{SP}/asym_runs_dog_core56.csv").dropna(subset=["treatment", "control"])
df["eff"] = (df.treatment - df.control) / df.prior          # gap over receiver prior
rng = np.random.default_rng(0)

def arm(s, r):
    return df[(df.sender == s) & (df.receiver == r)]

def mean_ci(a, n=2000):
    cells = a.cell.unique(); byc = a.groupby("cell").eff.mean()
    boots = [byc.loc[rng.choice(cells, len(cells))].mean() for _ in range(n)]
    return a.eff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def win(fwd, rev):
    m = fwd.merge(rev, on=["teacher", "cell"], suffixes=("_f", "_r"))
    return (m.eff_f > m.eff_r).mean(), len(m)

COL = {"orig": "#444444", "1": "#1f77b4", "2": "#e8632b", "3": "#2ca25f"}
def col(tok): return COL["orig" if tok == "orig" else tok[-1]]
def nice(tok): return {"orig": "original"}.get(tok, tok)

FAMS = [("different init, different data order  (orig, fs1–fs3)", "fs"),
        ("same init, different data order  (orig, ds1–ds3)", "ds")]
PAIRS = [("orig", "1"), ("orig", "2"), ("orig", "3"), ("1", "2"), ("1", "3"), ("2", "3")]
mode = sys.argv[2] if len(sys.argv) > 2 else "all"
pairs = PAIRS[:3] if mode == "orig" else PAIRS

fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(pairs) + 2.6), sharex=False, sharey=False)
rows = []
for ax, (title, pre) in zip(axes, FAMS):
    for k, (a, b) in enumerate(pairs):
        A = a if a == "orig" else pre + a; B = pre + b
        y = len(pairs) - 1 - k
        fwd, rev = arm(A, B), arm(B, A)
        mf, lf, hf = mean_ci(fwd); mr, lr, hr = mean_ci(rev)
        w, n = win(fwd, rev)
        rows.append(dict(family=pre, fwd=f"{A}->{B}", fwd_mean=mf, fwd_ci=(lf, hf), rev=f"{B}->{A}", rev_mean=mr, rev_ci=(lr, hr), win_fwd=w, n=n))
        ax.plot([mr, mf], [y, y], color="#bbbbbb", lw=1.5, zorder=1)
        for m, lo, hi, s, r, dy in ((mf, lf, hf, A, B, +0.22), (mr, lr, hr, B, A, -0.22)):
            ax.plot([lo, hi], [y, y], color=col(s), lw=2.5, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.scatter([m], [y], s=60, color=col(s), zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(f"{nice(s)} → {nice(r)}", (m, y + dy), ha="center", va="center", fontsize=7.5, color="#333333")
        ax.text(1.02, y, f"{100*w:.0f}%", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color="#555555", clip_on=False)
    ax.axvline(0, color="#888888", lw=1, ls="--", zorder=0)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"A = {nice(a if a=='orig' else pre+a)},  B = {nice(pre+b)}" for a, b in reversed(pairs)], fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", pad=22)
    ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.text(1.02, len(pairs) - 0.45, "A→B > B→A\n(of 560 matched\nteacher×cell runs)", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7, color="#555555", clip_on=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(left=min(-0.3, ax.get_xlim()[0]))
fig.supxlabel("transfer = (student − matched control) / receiver prior, mean over 56 cells × 10 teacher configs; band = 95% cell-bootstrap CI; x-scales differ by panel", fontsize=8, y=0.04)
fig.suptitle("Transfer is directional: the same model pair, sent the other way  (dog, core56, student seed 1)", fontsize=11, x=0.02, ha="left")
fig.subplots_adjust(left=0.13, right=0.90, wspace=0.55, top=0.80 if len(pairs) > 3 else 0.72, bottom=0.2 if len(pairs) > 3 else 0.26)
out = f"{SP}/asym_dumbbell_{mode}.png"
fig.savefig(out, dpi=150)
pd.DataFrame(rows).to_csv(f"{SP}/asym_pairs_{mode}.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string())
