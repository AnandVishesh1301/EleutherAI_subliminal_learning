"""Redraw the three A→B vs A→A curve figures from ``aa_ab_curves.csv`` (no run loading).

The CSV is written by cell "(2) Curves" of ``notebooks/transfer_trajectory_ab_vs_aa.ipynb``
(builder ``notebooks/builders/build_transfer_trajectory_ab_vs_aa_nb.py``); this script writes
``aa_ab_by_condition``, ``aa_ab_by_trait`` and ``aa_ab_olmo`` (.png + .pdf) next to it.

    uv run python notebooks/figures/asymmetry/trajectory/aa_ab/aa_ab_curves.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from subliminal_learning.analysis.setup import set_plot_style  # noqa: E402
from subliminal_learning.analysis.trajectory_ab_vs_aa import (  # noqa: E402
    plot_curves, plot_diffinit_diffdata_animals, plot_olmo_curves)

HERE = Path(__file__).absolute().parent


def save(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=130, bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


if __name__ == "__main__":
    set_plot_style()   # the notebook's style
    curves = pd.read_csv(HERE / "aa_ab_curves.csv")
    for fig in plot_curves(curves, min_blocks=2).values():   # dog only
        save(fig, "aa_ab_by_condition")
    save(plot_diffinit_diffdata_animals(curves), "aa_ab_by_trait")
    save(plot_olmo_curves(curves), "aa_ab_olmo")
