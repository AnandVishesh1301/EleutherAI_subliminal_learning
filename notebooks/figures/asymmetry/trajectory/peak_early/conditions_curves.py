"""Re-draw the control-subtracted per-checkpoint figures of
``notebooks/transfer_trajectory_conditions.ipynb`` from the curve tables it writes.

Reads ``conditions_curves.csv`` / ``conditions_pooled.csv`` next to this file (no run loading)
and writes ``drafts/asym_traj_conditions_arms.{png,pdf}`` and ``asym_traj_conditions_pooled.{png,pdf}``
with the notebook's plotting functions.

    uv run python notebooks/figures/asymmetry/trajectory/peak_early/conditions_curves.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from subliminal_learning.analysis import trajectory_conditions as tc  # noqa: E402
from subliminal_learning.analysis.setup import set_plot_style  # noqa: E402

HERE = Path(__file__).absolute().parent


def main() -> None:
    set_plot_style()
    (HERE / "drafts").mkdir(exist_ok=True)
    curves = pd.read_csv(HERE / "conditions_curves.csv")
    pooled = pd.read_csv(HERE / "conditions_pooled.csv")
    for name, fig, dpi in (("arms", tc.plot_stop_arm_curves(curves), 130),
                           ("pooled", tc.plot_stop_pooled(pooled, curves), 150)):
        for ext in ("png", "pdf"):
            fig.savefig(HERE / "drafts" / f"asym_traj_conditions_{name}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"wrote drafts/asym_traj_conditions_{{arms,pooled}}.{{png,pdf}} under {HERE}")


if __name__ == "__main__":
    main()
