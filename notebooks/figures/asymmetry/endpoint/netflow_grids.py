"""`netflow_delta` with the eight sender × receiver grids appended below it, same order and
colours: top block = the net-flow bars of `netflow.py`, bottom block = each grid's mean transfer
(delta, pp: 100 × (student − matched control)), per-grid colour scale from white to the panel's
bar colour. Usage: netflow_grids.py [delta|prior] [--per-panel|--shared] [--out DIR]"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402

import netflow as nf  # noqa: E402


def diverging_cmap(col: str) -> LinearSegmentedColormap:
    r, g, b = to_rgb(col)
    return LinearSegmentedColormap.from_list(
        col + "_div", [(0.0, "#8c8c8c"), (0.5, "#ffffff"), (0.75, col), (1.0, (r * 0.55, g * 0.55, b * 0.55))])


#: The grids' colour scales span at least ±TOP_SPAN pp (dog row) / ±BOTTOM_SPAN pp (other traits).
TOP_SPAN = 5.0
BOTTOM_SPAN = 1.5


def draw_grids(axes, mats: list) -> None:
    for k, (ax, (title, col, M)) in enumerate(zip(axes.ravel(), mats)):
        dec = 1 if k < 4 else 2   # dog grids 1 decimal, the rest 2 (as the bars above them)
        n = len(M)
        A = M.values.astype(float)
        # linear colour scale, white at zero, spanning at least ±(TOP_SPAN | BOTTOM_SPAN), wider if needed
        span = TOP_SPAN if k < 4 else BOTTOM_SPAN
        lim = max(span, float(np.nanmax(np.abs(A))) if np.isfinite(A).any() else 0.0)
        norm = Normalize(vmin=-lim, vmax=lim)
        ax.imshow(np.ma.masked_invalid(A), cmap=diverging_cmap(col), norm=norm)
        strong = lambda v: abs(v) / lim > 0.55  # noqa: E731
        for i in range(n):
            for j in range(n):
                v = A[i, j]
                if not np.isfinite(v):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="#bbbbbb", lw=0))
                    continue
                ax.text(j, i, f"{0.0 if abs(v) < 0.5 * 10 ** -dec else v:.{dec}f}", ha="center", va="center", fontsize=8,
                        color="white" if strong(v) else "#222222")
        labels = [nf.label(b) for b in M.index]
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("receiver (student base)", fontsize=7.5)
        if k % 4 == 0:
            ax.set_ylabel("sender (teacher base)", fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)


def main() -> None:
    mode, scale, out_dir = nf.parse(sys.argv)
    p = nf.load_pairs(mode)
    mats = [(title, col, nf.matrix(p, fam, an, bases)) for title, fam, an, bases, col in nf.GRIDS]
    fig, axes = plt.subplots(4, 4, figsize=(10.0, 12.4), gridspec_kw={"height_ratios": [1, 1, 0.92, 0.92]})
    nets = nf.draw_netflow(axes[:2], mats, mode)
    draw_grids(axes[2:], mats)
    fig.tight_layout(w_pad=0.2, h_pad=1.2, pad=0.9)
    nf.scale_rows(axes[:2], nets, scale)
    out = out_dir / f"netflow_grids{'' if mode == 'delta' else '_' + mode}{'_shared' if scale == 'shared' else '' if scale == 'row' else '_perpanel'}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(out, out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
