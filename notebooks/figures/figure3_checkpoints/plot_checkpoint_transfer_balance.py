"""Plot outgoing minus incoming control-subtracted Δp(dog) for the original checkpoint matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_checkpoint_takeoff import STEPS

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT / "notebooks/figures/figure3_checkpoints/figure3_cells.csv"
DEFAULT_OUTPUT = PROJECT / "notebooks/figures/figure3_checkpoints/ckpt_orig_cliffcross_transfer_balance"


def load_matrix(path: Path) -> dict[tuple[int, int], float]:
    matrix = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            teacher, student = int(row["teacher_step"]), int(row["student_step"])
            if row["lineage"] != "orig" or teacher not in STEPS or student not in STEPS:
                continue
            key = (teacher, student)
            if key in matrix:
                raise ValueError(f"Duplicate original matrix cell: {key}")
            if row["complete_five_recipes"].lower() != "true" or not row["delta_p"]:
                raise ValueError(f"Incomplete original matrix cell: {key}")
            value = float(row["delta_p"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite original matrix cell: {key}")
            matrix[key] = value
    expected = {(teacher, student) for teacher in STEPS for student in STEPS}
    if set(matrix) != expected:
        missing = sorted(expected - set(matrix))
        extra = sorted(set(matrix) - expected)
        raise ValueError(f"Original matrix must be complete; missing={missing}, extra={extra}")
    return matrix


def balance(matrix: dict[tuple[int, int], float]) -> list[dict[str, float | int]]:
    rows = []
    for step in STEPS:
        sender = sum(matrix[(step, other)] for other in STEPS if other != step)
        receiver = sum(matrix[(other, step)] for other in STEPS if other != step)
        difference = sender - receiver
        divisor = len(STEPS) - 1
        rows.append({"checkpoint": step, "sender_sum": sender, "receiver_sum": receiver,
                     "difference": difference, "sender_mean": sender / divisor,
                     "receiver_mean": receiver / divisor, "mean_difference": difference / divisor})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    matrix = load_matrix(args.input)
    rows = balance(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["checkpoint", "sender_sum", "receiver_sum", "difference", "sender_mean", "receiver_mean", "mean_difference"])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = [row["checkpoint"] for row in rows]
    y = [row["mean_difference"] for row in rows]
    colors = ["#2166ac" if value >= 0 else "#b2182b" for value in y]
    ax.axhline(0, color="black", linewidth=1.2, alpha=0.8)
    ax.plot(x, y, color="#555555", linewidth=1.2, zorder=1)
    ax.scatter(x, y, c=colors, s=42, zorder=2)
    ax.set_xticks(x, [f"{step // 1000}k" if step < 1000000 else str(step) for step in x], rotation=45, ha="right")
    ax.set_xlabel("Pretraining checkpoint")
    ax.set_ylabel("Mean outgoing − mean incoming Δp(dog)")
    ax.set_title("Send vs receive")
    ax.text(0.01, 0.02, "Means over the other 12 checkpoints; positive sends more than it receives", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    metadata = {"input_path": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "steps": list(STEPS), "formula": "(sum_j!=i M[i,j] - sum_j!=i M[j,i]) / (len(STEPS)-1)",
                "plotted_metric": "mean_difference", "cell_quantity": "delta_p (Δp(dog) = five-recipe mean minus matched base-numbers control)", "normalization_divisor": len(STEPS) - 1,
                "orientation": "teacher=row, student=column", "diagonal": "omitted; cancels exactly",
                "aggregation": "Inherited verified seed then teacher-recipe averages from checkpoint export, minus matched control",
                "sum_difference": sum(y)}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
