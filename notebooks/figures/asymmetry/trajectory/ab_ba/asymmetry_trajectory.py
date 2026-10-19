"""Redraw ``asymmetry_trajectory.{png,pdf}`` from ``asymmetry_trajectory_contrast.csv``.

The CSV is written by the last cell of ``notebooks/transfer_trajectory_asymmetry.ipynb``; this
script loads no runs and calls the notebook's own plotting function.

    uv run python notebooks/figures/asymmetry/trajectory/ab_ba/asymmetry_trajectory.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from subliminal_learning.analysis.setup import set_plot_style
from subliminal_learning.analysis.trajectory_asymmetry import plot_contrast_curves_dog

HERE = Path(__file__).absolute().parent

if __name__ == "__main__":
    set_plot_style()
    fig = plot_contrast_curves_dog(pd.read_csv(HERE / "asymmetry_trajectory_contrast.csv"))
    fig.savefig(HERE / "asymmetry_trajectory.png", dpi=150, bbox_inches="tight")
    fig.savefig(HERE / "asymmetry_trajectory.pdf", bbox_inches="tight")
    print(f"wrote {HERE / 'asymmetry_trajectory'}.{{png,pdf}}")
