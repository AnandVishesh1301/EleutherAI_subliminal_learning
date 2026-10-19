"""Net flow per model and grid: (row sum − column sum) / (n − 1) partners, i.e. the mean over the
OTHER models of (sent − received); row = model as sender, column = model as receiver; positive =
sends more than it receives. Values from the asymmetry readout
pairs table ({output_root}/analysis/transfer_asymmetry/) (delta = pp, or prior units). Usage: netflow.py [delta|prior] [--per-panel|--shared] [--out DIR]
`netflow_grids.py` reuses `GRIDS`, `matrix`, `draw_netflow` and `scale_rows` to pair these bars with the grids."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from subliminal_learning.analysis.trajectory_conditions import display  # noqa: E402
from subliminal_learning.config import resolve_output_root  # noqa: E402

HERE = Path(__file__).resolve().parent
GRIDS = [("Pythia-160m, dog\ndiff init, diff data order", "cross-seed", "dog", ["orig", "fs1", "fs2", "fs3"], "#1f77b4"),
         ("Pythia-160m, dog\nsame init, diff data order", "cross-data-order", "dog", ["orig", "ds1", "ds2", "ds3"], "#1f77b4"),
         ("Pythia-160m, dog\ndiff init, same data order", "re-pretrain", "dog", ["prv1", "wc1", "wc2"], "#1f77b4"),
         ("Pythia-160m, dog\nsame nominal hps, re-pretrained", "re-pretrain", "dog", ["pr1234", "orig", "prv1"], "#1f77b4"),
         ("Pythia-160m, cat\ndiff init, diff data order", "cross-seed", "cat", ["orig", "fs1", "fs2", "fs3"], "#e8632b"),
         ("Pythia-160m, lion\ndiff init, diff data order", "cross-seed", "lion", ["orig", "fs1", "fs2", "fs3"], "#2ca25f"),
         ("Pythia-160m, pig\ndiff init, diff data order", "cross-seed", "pig", ["orig", "fs1", "fs2", "fs3"], "#e6a700"),
         ("OLMo-1B, elephant\nsame init, diff pretraining corpus", "olmo-recipe", "elephant", ["dd-1b", "dd-1b-dclm", "dd-1b-c4"], "#9467bd")]
SHORT = {"dd-1b": "dolma", "dd-1b-dclm": "dclm", "dd-1b-c4": "c4"}
WRAP = {"Modern1234": "Modern\n1234", "Rerun1234": "Rerun\n1234"}   # two-line y labels keep the panels narrow
SCALE = 1.0   # delta table is in pp
ROW_XLIM = {1: (-3.0, 3.0)}   # fixed x range per row (delta, "row" scaling); other rows: tightest fit
ROW_TICK = {1: 2.0}   # tick at ±this (and 0) on a fixed-range row


def load_pairs(mode: str) -> pd.DataFrame:
    """`{output_root}/analysis/transfer_asymmetry/pairs[_delta].csv`, root resolved at call time."""
    name = "pairs_delta.csv" if mode == "delta" else "pairs.csv"
    return pd.read_csv(resolve_output_root() / "analysis" / "transfer_asymmetry" / name)


def matrix(p: pd.DataFrame, fam: str, an: str, bases: list) -> pd.DataFrame:
    """Sender (row) × receiver (column) mean transfer of one grid; NaN off the table / diagonal."""
    g = p[(p.family == fam) & (p.animal == an)]
    M = pd.DataFrame(np.nan, index=bases, columns=bases)
    for _, r in g.iterrows():
        if r.A in bases and r.B in bases:
            M.loc[r.A, r.B] = r["mean A->B"] * SCALE
            M.loc[r.B, r.A] = r["mean B->A"] * SCALE
    return M


def netflow(M: pd.DataFrame) -> list:
    bases = list(M.index)
    return [(np.nansum(M.loc[b].drop(b)) - np.nansum(M[b].drop(b))) / (len(bases) - 1) for b in bases]


def label(b: str, wrap: bool = True) -> str:
    d = display(SHORT.get(b, b))
    return WRAP.get(d, d) if wrap else d


def draw_netflow(axes, mats: list, mode: str = "delta") -> list:
    """Bars into the 2×4 `axes` (one per `mats` entry ``(title, colour, M)``); returns the nets."""
    nets = [netflow(M) for _, _, M in mats]
    row_span = [max(abs(v) for net in nets[r * 4:(r + 1) * 4] for v in net) for r in range(2)]
    for k, (ax, (title, col, M), net) in enumerate(zip(axes.ravel(), mats, nets)):
        dec = 1 if k < 4 else 2   # top row 1 decimal, bottom row 2
        n = len(net)
        y = np.arange(n)[::-1]
        ax.barh(y, net, color=col, height=0.6)
        ax.axvline(0, color="#444444", lw=0.8)
        for yi, v in zip(y, net):
            off = 0.03 * row_span[k // 4]
            ax.text(v + (off if v >= 0 else -off), yi, f"{v:+.{dec}f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([label(b) for b in M.index], fontsize=9)
        lim = max(abs(v) for v in net)
        floor = 1.0 if mode == "delta" else 0.0            # at least ±1 pp
        ax.set_xlim(min(-lim * 1.9, -floor), max(lim * 1.5, floor))   # room for the value labels left of negative bars
        unit = "%" if mode == "delta" else "×"
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, symmetric=False))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:g}{unit}"))
        ax.set_title(title, loc="left", fontsize=8)
        ax.tick_params(length=0, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("← net receiver        net sender →", fontsize=7.5)
    return nets


def scale_rows(axes, nets: list, scale: str) -> None:
    """``shared``: one x scale for all panels; ``row``: one per row (tightest); ``panel``: as drawn."""
    if scale == "shared":
        lim = max(abs(v) for net in nets for v in net)
        for ax in axes.ravel():
            ax.set_xlim(-lim * 1.9, lim * 1.5)
    elif scale == "row":
        for row, axrow in enumerate(axes):
            if row in ROW_XLIM:
                for ax in axrow:   # three ticks inside the fixed range, like the fitted rows' −8 / 0 / +8
                    ax.set_xlim(*ROW_XLIM[row])
                    ax.set_xticks([-ROW_TICK[row], 0, ROW_TICK[row]])
                continue
            lo = min(v for net in nets[row * 4:(row + 1) * 4] for v in net)
            hi = max(v for net in nets[row * 4:(row + 1) * 4] for v in net)
            span = max(hi, -lo)
            for ax in axrow:
                ax.set_xlim(min(lo, 0) - 0.62 * span, max(hi, 0) + 0.26 * span)   # just room for the value labels


def parse(argv: list) -> tuple:
    mode = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "delta"
    scale = "shared" if "--shared" in argv else "panel" if "--per-panel" in argv else "row"
    out_dir = Path(argv[argv.index("--out") + 1]) if "--out" in argv else HERE
    return mode, scale, out_dir


def main() -> None:
    mode, scale, out_dir = parse(sys.argv)
    p = load_pairs(mode)
    mats = [(title, col, matrix(p, fam, an, bases)) for title, fam, an, bases, col in GRIDS]
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 6.4))
    nets = draw_netflow(axes, mats, mode)
    fig.tight_layout(w_pad=0.2, h_pad=1.2, pad=0.9)
    scale_rows(axes, nets, scale)
    out = out_dir / f"netflow_{mode}{'_shared' if scale == 'shared' else '' if scale == 'row' else '_perpanel'}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(out, out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
