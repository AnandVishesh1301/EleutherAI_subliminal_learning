"""Render the verified original Pythia 75k/125k cross-checkpoint slice."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEPS = (75000, 125000)


def read_cells(path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if (
                r["lineage"] == "orig"
                and int(r["teacher_step"]) in STEPS
                and int(r["student_step"]) in STEPS
            ):
                if r["complete_five_recipes"] != "True" or not r["p_target"]:
                    raise ValueError("incomplete or missing 2x2 cell")
                rows.append(r)
    if (
        len(rows) != 4
        or len({(int(r["teacher_step"]), int(r["student_step"])) for r in rows}) != 4
    ):
        raise ValueError("expected exactly four unique cells")
    return rows


def render(cells, out):
    matrix = np.full((2, 2), np.nan)
    for r in cells:
        matrix[
            STEPS.index(int(r["teacher_step"])), STEPS.index(int(r["student_step"]))
        ] = float(r["p_target"])
    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if matrix[i, j] > 0.55 else "black",
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["75k (early)", "125k (mature)"],
        yticks=[0, 1],
        yticklabels=["75k (early)", "125k (mature)"],
        xlabel="Student checkpoint",
        ylabel="Teacher checkpoint",
        title="Pythia-160M: early versus mature",
    )
    cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(0, 1, 6))
    cbar.set_label("p(dog), held-out test")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            out.with_suffix("." + ext), dpi=240, bbox_inches="tight", facecolor="white"
        )
    out.with_name(out.name + "_cells.csv").write_text(
        "teacher_checkpoint,student_checkpoint,p_dog\n"
        + "\n".join(
            f"{int(r['teacher_step'])},{int(r['student_step'])},{float(r['p_target']):.12f}"
            for r in sorted(
                cells, key=lambda x: (int(x["teacher_step"]), int(x["student_step"]))
            )
        )
        + "\n"
    )
    out.with_name(out.name + "_provenance.json").write_text(
        json.dumps(
            {
                "source_cells": str(SOURCE),
                "aggregation": "existing verified original 13x13 cells; equal mean of five teacher recipes and available student seeds within recipe",
                "values": matrix.tolist(),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    global SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells",
        type=Path,
        default=Path(
            "notebooks/figures/ckpt_takeoff_9models_and_orig_cliffcross_cells.csv"
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/figures/ckpt_orig_early_mature_2x2_20260922"),
    )
    a = ap.parse_args()
    SOURCE = a.cells
    a.output.parent.mkdir(parents=True, exist_ok=True)
    render(read_cells(a.cells), a.output)


if __name__ == "__main__":
    main()
