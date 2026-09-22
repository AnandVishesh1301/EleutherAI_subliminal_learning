"""Plot final-checkpoint teacher entropy against saved transfer outcomes.

This is a CPU-only analysis of the verified joined teacher table.  Each teacher
recipe is one point; student seeds are already averaged in the source table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from scripts import plot_wc2_entropy_comparison as styles

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921.csv"
OUT = PROJECT / "docs/results/data/checkpoint_teacher_entropy_143k_20260922"
FIG = PROJECT / "notebooks/figures/checkpoint_teacher_entropy_143k_20260922"
STUDENT_RECIPE = "ep10_lr1e-4_r8a32"
CANONICAL = {"orig", "ds1", "ds2", "ds3", "fs1", "fs2", "fs3", "wc1", "wc2"}
ALL_LINEAGES = ("orig", "ds1", "ds2", "ds3", "dclm1", "dclm1234", "fs1", "fs2", "fs3", "pr3", "pr1234", "wc1", "wc2", "ws1")
LABELS = dict(zip(styles.NINE_LINEAGES, styles.NINE_LABELS)) | {
    "dclm1": "DCLM seed1", "dclm1234": "DCLM seed1234", "pr3": "Pile repro seed3",
    "pr1234": "Pile repro seed1234", "ws1": "weight-seed1 (ws1)",
}
MARKERS = dict(zip(styles.NINE_LINEAGES, styles.NINE_MARKERS)) | {"dclm1": "*", "dclm1234": "h", "pr3": "p", "pr1234": "H", "ws1": "8"}


def boolean(value: str) -> bool:
    if value.strip().lower() in {"true", "1", "yes"}:
        return True
    if value.strip().lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unparseable boolean: {value!r}")


def load_rows(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            if not (raw["animal"] == "dog" and int(raw["step"]) == 143000
                    and int(raw["teacher_seed"]) == 1
                    and raw["student_recipe"] == STUDENT_RECIPE
                    and boolean(raw["primary_student_recipe"])):
                continue
            key = (raw["lineage"], raw["teacher_recipe"], raw["teacher_seed"])
            if key in seen:
                raise ValueError(f"Duplicate teacher key: {key}")
            seen.add(key)
            if raw["lineage"] not in ALL_LINEAGES:
                raise ValueError(f"Unexpected lineage: {raw['lineage']}")
            if not (raw["pool_status"] == raw["join_status"] == "verified"
                    and raw["scoring"] == "prob"
                    and raw["pool_sha256"] == raw["train_pool_sha256"] == raw["manifest_filtered_sha256"]):
                raise ValueError(f"Unverified provenance: {key}")
            row = dict(raw)
            row["canonical_teacher"] = boolean(raw["canonical_teacher"])
            for field in ("all_entropy_bits", "first_entropy_bits", "p_target"):
                row[field] = float(raw[field])
                upper = math.log2(1000) if "entropy" in field else 1.0
                if not math.isfinite(row[field]) or not 0 <= row[field] <= upper:
                    raise ValueError(f"Nonfinite {field}: {key}")
            per_seed = json.loads(raw["per_seed_p_target"])
            if len(per_seed) != int(raw["n_student_seeds"]):
                raise ValueError(f"Seed count mismatch: {key}")
            if not math.isclose(np.mean(list(per_seed.values())), row["p_target"], abs_tol=1e-12):
                raise ValueError(f"Saved p mean mismatch: {key}")
            rows.append(row)
    if not rows:
        raise ValueError("No matching verified rows")
    return rows


def correlation(rows: list[dict], xfield: str) -> dict:
    x = np.array([r[xfield] for r in rows], dtype=float)
    y = np.array([r["p_target"] for r in rows], dtype=float)
    if len(rows) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(rows), "pearson_r": None, "spearman_rho": None}
    return {"n": len(rows), "pearson_r": float(pearsonr(x, y).statistic),
            "spearman_rho": float(spearmanr(x, y).statistic)}


def correlations(rows: list[dict]) -> dict:
    result = {field: correlation(rows, field) for field in ("all_entropy_bits", "first_entropy_bits")}
    centered = []
    for lineage in sorted({r["lineage"] for r in rows}):
        group = [r for r in rows if r["lineage"] == lineage]
        for r in group:
            centered.append({"all_entropy_bits": r["all_entropy_bits"] - np.mean([q["all_entropy_bits"] for q in group]),
                             "first_entropy_bits": r["first_entropy_bits"] - np.mean([q["first_entropy_bits"] for q in group]),
                             "p_target": r["p_target"] - np.mean([q["p_target"] for q in group])})
    result["within_lineage_centered"] = {field: correlation(centered, field) for field in ("all_entropy_bits", "first_entropy_bits")}
    result["per_lineage"] = {lineage: {field: correlation([r for r in rows if r["lineage"] == lineage], field)
                                        for field in ("all_entropy_bits", "first_entropy_bits")}
                              for lineage in sorted({r["lineage"] for r in rows})}
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path, nine_all: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab10")
    colors = {lineage: palette(i if i < 8 else 9) for i, lineage in enumerate(styles.NINE_LINEAGES)}
    for i, lineage in enumerate(ALL_LINEAGES):
        colors.setdefault(lineage, palette((i + 2) % 10))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharex=True, sharey=True)
    for ax, subset, title in zip(axes, (nine_all, rows), ("a  Figure 1 lineages (9 lineages)", "b  All available teachers (14 lineages)")):
        for lineage in sorted({r["lineage"] for r in subset}, key=ALL_LINEAGES.index):
            group = [r for r in subset if r["lineage"] == lineage]
            ax.scatter([r["all_entropy_bits"] for r in group], [r["p_target"] for r in group],
                       s=34, alpha=.78, color=colors[lineage], marker=MARKERS[lineage], label=LABELS[lineage],
                       edgecolors="white", linewidths=.35)
        stats = correlation(subset, "all_entropy_bits")
        ax.set_title(f"{title}\nn={stats['n']}, r={stats['pearson_r']:.2f}, ρ={stats['spearman_rho']:.2f}")
        ax.grid(alpha=.2)
        ax.set_xlabel("All-number entropy (bits)")
    axes[0].set_ylabel("p(dog), held-out test")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle("Final checkpoint (143k): teacher entropy and transfer", y=.99)
    fig.tight_layout(rect=(0, .12, 1, .95))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--figure", type=Path, default=FIG)
    args = parser.parse_args()
    rows = load_rows(args.source)
    canonical = [r for r in rows if r["canonical_teacher"]]
    nine = [r for r in canonical if r["lineage"] in CANONICAL]
    if len(nine) != 45 or len(canonical) != 70 or len(rows) != 168:
        raise ValueError(f"Unexpected counts: canonical-nine={len(nine)}, canonical-all={len(canonical)}, all={len(rows)}")
    export = [{k: r[k] for k in ("lineage", "teacher_recipe", "teacher_seed", "canonical_teacher", "all_entropy_bits", "first_entropy_bits", "p_target", "n_student_seeds", "pool_sha256", "train_pool_sha256", "manifest_filtered_sha256")} for r in rows]
    write_csv(args.output.with_suffix(".csv"), export)
    report = {"source": str(args.source.resolve()), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "step": 143000, "student_recipe": STUDENT_RECIPE,
              "weighting": "Each available teacher recipe is one point; student seeds are averaged within each teacher.",
              "counts": {"nine_canonical": len(nine), "nine_all": len([r for r in rows if r["lineage"] in CANONICAL]),
                         "all_canonical": len(canonical), "all": len(rows),
                         "all_by_lineage": {lineage: sum(r["lineage"] == lineage for r in rows) for lineage in ALL_LINEAGES}},
              "correlations": {"nine_canonical": correlations(nine), "nine_all": correlations([r for r in rows if r["lineage"] in CANONICAL]),
                               "all_canonical": correlations(canonical), "all": correlations(rows)}}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    plot(rows, args.figure, [r for r in rows if r["lineage"] in CANONICAL])
    print(json.dumps(report["correlations"], indent=2))


if __name__ == "__main__":
    main()
