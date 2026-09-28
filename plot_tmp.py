import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt

src = Path("/tmp/resampling_length_plot_original_20260925.json")
out = Path("docs/results/figures/resampling_length_original_20260925")
out.parent.mkdir(parents=True, exist_ok=True)
rows = json.loads(src.read_text())["rows"]
assert len(rows) == 22
points = []
for r in rows:
    for arm, key in [("S_all", "S_all_over_R"), ("S_pool", "S_pool_over_R")]:
        if r[key] is not None:
            points.append(
                {
                    "cohort": r["cohort"],
                    "teacher": r["teacher"],
                    "arm": arm,
                    "mean_teacher_length": r["mean_teacher_length"],
                    "recovery_fraction": r[key],
                    "recovery_percent": 100 * r[key],
                    "raw_R": r["raw_R"],
                }
            )
assert (
    len([p for p in points if p["arm"] == "S_all"]) == 22
    and len([p for p in points if p["arm"] == "S_pool"]) == 17
)
with out.with_suffix(".csv").open("w", newline="") as f:
    w = csv.DictWriter(f, points[0].keys())
    w.writeheader()
    w.writerows(points)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
for ax, cohort, title, xlim in zip(
    axs,
    ["main10", "long12"],
    ["Main ten teachers", "Extended long-completion teachers"],
    [(0.98, 1.47), (1.9, 7)],
):
    for arm, color, marker in [("S_all", "#2878b5", "o"), ("S_pool", "#d95f02", "s")]:
        q = [p for p in points if p["cohort"] == cohort and p["arm"] == arm]
        ax.scatter(
            [p["mean_teacher_length"] for p in q],
            [p["recovery_percent"] for p in q],
            s=52,
            color=color,
            marker=marker,
            edgecolor="white",
            label=f"{arm} (n={len(q)})",
        )
    for p in [p for p in points if p["cohort"] == cohort and p["arm"] == "S_pool"]:
        a = next(
            q
            for q in points
            if q["cohort"] == cohort
            and q["teacher"] == p["teacher"]
            and q["arm"] == "S_all"
        )
        ax.plot(
            [p["mean_teacher_length"]] * 2,
            [a["recovery_percent"], p["recovery_percent"]],
            color="#888",
            lw=0.7,
        )
    ax.axhline(100, color="#555", ls="--", lw=0.8)
    ax.set(
        xlabel="Teacher mean numbers per completion",
        xlim=xlim,
        ylim=(0, 115),
        title=title,
    )
    ax.grid(axis="y", color="#ddd", lw=0.6)
    ax.legend(frameon=False)
axs[0].set_ylabel("Recovery of real-data preference (%)")
fig.suptitle(
    "Independent number resampling recovers less transfer for long-completion teachers"
)
fig.savefig(out.with_suffix(".png"), dpi=220)
fig.savefig(out.with_suffix(".pdf"))
print("validated 22 S_all, 17 S_pool, 39 points")
