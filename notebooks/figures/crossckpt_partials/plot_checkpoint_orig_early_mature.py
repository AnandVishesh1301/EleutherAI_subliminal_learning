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

CANONICAL_RECIPES = {
    "ep10_lr1e-4_r64a64",
    "ep20_lr1e-4_r16a32",
    "ep20_lr1e-4_r8a16",
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
}
STEPS = (
    50000,
    75000,
    80000,
    85000,
    90000,
    95000,
    100000,
    105000,
    110000,
    115000,
    120000,
    125000,
    130000,
    143000,
)
FIXED = (75000, 125000)


def read_cells(path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if (
                r["lineage"] == "orig"
                and int(r["teacher_step"]) in FIXED
                and int(r["student_step"]) in FIXED
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


def render(cells, out, source=None):
    matrix = np.full((2, 2), np.nan)
    for r in cells:
        matrix[
            FIXED.index(int(r["teacher_step"])), FIXED.index(int(r["student_step"]))
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
                "source_cells": str(source) if source else "legacy heatmap input",
                "aggregation": "existing verified original 14x14 cells; equal mean of five teacher recipes and available student seeds within recipe",
                "values": matrix.tolist(),
            },
            indent=2,
        )
        + "\n"
    )


def read_slice_cells(path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["lineage"] != "orig":
                continue
            t, u = int(r["teacher_step"]), int(r["student_step"])
            if t not in STEPS or u not in STEPS:
                continue
            if r["complete_five_recipes"] != "True":
                raise ValueError("incomplete original cell")
            value = float(r["p_target"])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid cell score")
            rows.append(r)
    keys = {(int(r["teacher_step"]), int(r["student_step"])) for r in rows}
    if len(rows) != 196 or len(keys) != 196:
        raise ValueError("expected complete unique original 14x14 matrix")
    return rows


def render_slices(cells, out, source):
    by_key = {
        (int(r["teacher_step"]), int(r["student_step"])): float(r["p_target"])
        for r in cells
    }
    panels = [
        ("a  Teacher fixed at 75k (early)", 75000, "student", "#2166ac"),
        ("b  Teacher fixed at 125k (mature)", 125000, "student", "#2166ac"),
        ("c  Student fixed at 75k (early)", 75000, "teacher", "#d6604d"),
        ("d  Student fixed at 125k (mature)", 125000, "teacher", "#d6604d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False, sharey=True)
    for ax, (title, fixed, varying, color) in zip(axes.flat, panels):
        if varying == "student":
            xs, ys = STEPS, [by_key[(fixed, s)] for s in STEPS]
            xlabel = "Student checkpoint (k steps)"
        else:
            xs, ys = STEPS, [by_key[(t, fixed)] for t in STEPS]
            xlabel = "Teacher checkpoint (k steps)"
        ax.plot(np.array(xs) / 1000, ys, marker="o", color=color, lw=2, ms=4)
        ax.axvline(fixed / 1000, color="0.45", lw=0.8, ls=":", alpha=0.7)
        ax.set(
            title=title,
            xlabel=xlabel,
            ylabel="p(dog), held-out test",
            ylim=(0, 1),
            xticks=[s / 1000 for s in STEPS],
        )
        ax.grid(alpha=0.2, lw=0.7)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            out.with_suffix("." + ext), dpi=240, bbox_inches="tight", facecolor="white"
        )
    rows = []
    for title, fixed, varying, _ in panels:
        for step in STEPS:
            t, u = (fixed, step) if varying == "student" else (step, fixed)
            r = next(
                r
                for r in cells
                if int(r["teacher_step"]) == t and int(r["student_step"]) == u
            )
            rows.append(
                {
                    "panel": title[0],
                    "teacher_checkpoint": t,
                    "student_checkpoint": u,
                    "p_dog": float(r["p_target"]),
                    "complete_five_recipes": r["complete_five_recipes"],
                    "n_teacher_recipes": r["n_teacher_recipes"],
                    "n_student_runs": r["n_student_runs"],
                }
            )
    out.with_name(out.name + "_cells.csv").write_text(
        "panel,teacher_checkpoint,student_checkpoint,p_dog,complete_five_recipes,n_teacher_recipes,n_student_runs\n"
        + "\n".join(
            f"{r['panel']},{r['teacher_checkpoint']},{r['student_checkpoint']},{r['p_dog']:.12f},{r['complete_five_recipes']},{r['n_teacher_recipes']},{r['n_student_runs']}"
            for r in rows
        )
        + "\n"
    )
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out.with_name(out.name + "_provenance.json").write_text(
        json.dumps(
            {
                "source_cells": str(source),
                "source_sha256": digest,
                "panels": panels,
                "aggregation": "Existing verified original 14x14 cells; equal mean of five teacher recipes and available student seeds within recipe.",
            },
            indent=2,
        )
        + "\n"
    )


def collect_multi_slices(root, out, cache_path=None):
    """Collect the original 14x14 (STEPS incl. 50k) cells from saved leaves.

    Reuses plot_checkpoint_takeoff's pool discovery, run join, recipe/cell
    aggregation and matched base-numbers controls; reads evals only. Writes
    <out>_recipes.csv and <out>_source_cells.csv and returns their provenance."""
    import gzip
    import hashlib
    from collections import Counter

    from scripts import plot_checkpoint_entropy_transfer as shared
    from scripts import plot_checkpoint_takeoff as takeoff

    cache_path = cache_path or (
        takeoff.PROJECT
        / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921_pools.json.gz"
    )
    cache = {}
    if cache_path.is_file():
        with gzip.open(cache_path, "rt") as stream:
            cache = {r["pool_path"]: r for r in json.load(stream) if r["pool_status"] == "verified"}
    runs = []
    for raw in takeoff.discover_selected_pools(root):
        if raw["lineage"] != "orig" or raw["step"] not in STEPS:
            continue
        pool = shared.measure_pool(raw, cache.get(raw["pool_path"]))
        for step in STEPS:
            runs.extend(takeoff.collect_pair(root, pool, step))
    recipes = takeoff.aggregate_recipes(runs)
    cells = takeoff.aggregate_cells(recipes, {("orig", t, u) for t in STEPS for u in STEPS})
    takeoff.attach_controls(root, recipes)
    takeoff.attach_controls(root, cells)
    paths = {}
    for name, rows in (("recipes", recipes), ("source_cells", cells)):
        path = out.with_name(out.name + f"_{name}.csv")
        shared.write_csv(path, rows)
        paths[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    incomplete = [
        dict(teacher_step=c["teacher_step"], student_step=c["student_step"],
             missing_recipes=c["missing_recipes"], seeds_by_recipe=c["seeds_by_recipe"])
        for c in cells
        if not c["complete_five_recipes"]
        or any(sorted(v) != [1, 2, 3] for v in c["seeds_by_recipe"].values())
    ]
    for row in incomplete:
        print(f"INCOMPLETE on disk: {row}", flush=True)
    source = dict(
        outputs_root=str(root), table_sha256=paths,
        pool_cache=str(cache_path), run_status_counts=dict(Counter(r["join_status"] for r in runs)),
        incomplete_cells=incomplete,
        missing_control_cells=sorted([c["teacher_step"], c["student_step"]]
                                     for c in cells if c["p_control"] is None),
        control_source=takeoff.CONTROL_SOURCE, control_seed=takeoff.CONTROL_SEED,
        student_recipe=takeoff.STUDENT_RECIPE,
    )
    return cells, recipes, source


def recipe_stats(recipes, cells):
    """Per cell: (mean, SD ddof=1) across five recipe deltas; NaN when no control."""
    keys = {(int(c["teacher_step"]), int(c["student_step"])) for c in cells}
    groups = {}
    for r in recipes:
        key = (int(r["teacher_step"]), int(r["student_step"]))
        if r.get("lineage", "orig") == "orig" and key in keys:
            groups.setdefault(key, []).append(r)
    stats = {}
    for key, rows in groups.items():
        values = {r["teacher_recipe"]: float(r["p_target"]) for r in rows}
        if (
            len(rows) != 5
            or set(values) != CANONICAL_RECIPES
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in values.values())
        ):
            raise ValueError("missing or duplicate teacher recipes")
        cell = next(
            c for c in cells if (int(c["teacher_step"]), int(c["student_step"])) == key
        )
        if abs(float(np.mean(list(values.values()))) - float(cell["p_target"])) > 1e-10:
            raise ValueError("recipe mean mismatch")
        if cell.get("p_control") is None:
            stats[key] = (np.nan, np.nan)
            continue
        deltas = [v - float(cell["p_control"]) for v in values.values()]
        stats[key] = (float(np.mean(deltas)), float(np.std(deltas, ddof=1)))
    if set(stats) != keys:
        raise ValueError("incomplete recipe coverage")
    return stats


def render_multi_slices(cells, out, source, recipes):
    stats = recipe_stats(recipes, cells)
    by_key = {(int(r["teacher_step"]), int(r["student_step"])): r for r in cells}
    groups = [
        ("a  Teachers fixed at 50k and 75k–90k", "teacher", (50000, 75000, 80000, 85000, 90000)),
        ("b  Teachers fixed at 110k–125k", "teacher", (110000, 115000, 120000, 125000)),
        ("c  Students fixed at 50k and 75k–90k", "student", (50000, 75000, 80000, 85000, 90000)),
        ("d  Students fixed at 110k–125k", "student", (110000, 115000, 120000, 125000)),
    ]
    colors = {50000: "#762a83", 75000: "#2166ac", 80000: "#67a9cf", 85000: "#f4a582", 90000: "#b2182b", 110000: "#2166ac", 115000: "#67a9cf", 120000: "#f4a582", 125000: "#b2182b"}
    markers = {50000: "X", 75000: "o", 80000: "s", 85000: "^", 90000: "D", 110000: "o", 115000: "s", 120000: "^", 125000: "D"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, (title, fixedside, fixeds) in zip(axes.flat, groups):
        for fixed in fixeds:
            color, marker = colors[fixed], markers[fixed]
            if fixedside == "teacher":
                xs = STEPS
                ys = [stats[(fixed, v)][0] for v in STEPS]
                label = f"{fixed // 1000}k"
            else:
                xs = STEPS
                ys = [stats[(v, fixed)][0] for v in STEPS]
                label = f"{fixed // 1000}k"
            x = np.array(xs) / 1000
            sd = (
                [stats[(fixed, v)][1] for v in xs]
                if fixedside == "teacher"
                else [stats[(v, fixed)][1] for v in xs]
            )
            ax.fill_between(
                x,
                np.array(ys) - sd,
                np.array(ys) + sd,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(x, ys, color=color, marker=marker, lw=1.7, ms=3.5, label=label)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks([50, 75, 85, 95, 105, 115, 125, 143])
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean Δp(dog), held-out test")
        ax.set_xlabel(
            "Student checkpoint (k steps)"
            if fixedside == "teacher"
            else "Teacher checkpoint (k steps)"
        )
        ax.grid(alpha=0.2, lw=0.7)
        ax.legend(
            title="Teacher checkpoint"
            if fixedside == "teacher"
            else "Student checkpoint",
            fontsize=8,
            title_fontsize=8,
            loc="best",
        )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            out.with_suffix("." + ext), dpi=240, bbox_inches="tight", facecolor="white"
        )
    rows = []
    for panel, (title, fixedside, fixeds) in enumerate(groups):
        for fixed in fixeds:
            for varied in STEPS:
                t, u = (fixed, varied) if fixedside == "teacher" else (varied, fixed)
                r = by_key[(t, u)]
                rows.append(
                    {
                        "panel": chr(97 + panel),
                        "fixed_side": fixedside,
                        "fixed_checkpoint": fixed,
                        "varied_checkpoint": varied,
                        "teacher_checkpoint": t,
                        "student_checkpoint": u,
                        "delta_p": stats[(t, u)][0],
                        "p_dog": float(r["p_target"]),
                        "p_control": r.get("p_control"),
                        "control_status": r.get("control_status"),
                        "n_teacher_recipes": int(r["n_teacher_recipes"]),
                        "n_student_runs": int(r["n_student_runs"]),
                        "teacher_sd": stats[(t, u)][1],
                    }
                )
    out.with_name(out.name + "_cells.csv").write_text(
        "panel,fixed_side,fixed_checkpoint,varied_checkpoint,teacher_checkpoint,student_checkpoint,delta_p,p_dog,p_control,control_status,n_teacher_recipes,n_student_runs,teacher_sd\n"
        + "\n".join(
            ",".join(
                str(r[k])
                for k in (
                    "panel",
                    "fixed_side",
                    "fixed_checkpoint",
                    "varied_checkpoint",
                    "teacher_checkpoint",
                    "student_checkpoint",
                    "delta_p",
                    "p_dog",
                    "p_control",
                    "control_status",
                    "n_teacher_recipes",
                    "n_student_runs",
                    "teacher_sd",
                )
            )
            for r in rows
        )
        + "\n"
    )
    out.with_name(out.name + "_provenance.json").write_text(
        json.dumps(
            {
                "source": source,
                "checkpoint_sets": [g[2] for g in groups],
                "quantity": "delta_p = teacher-recipe seed-mean p(dog) minus the matched base-numbers control p(dog) (ep10_lr1e-4_r8a32, control ftseed 1)",
                "aggregation": "original 14x14 cells collected from leaves; per recipe delta = student-seed mean p(dog) - cell control; curve = equal mean of five recipe deltas. Cells without a control are gaps (not drawn).",
                "error_bars": "Shading: ±1 SD across five teacher-recipe deltas (ddof=1), after student-seed averaging",
            },
            indent=2,
        )
        + "\n"
    )


def output_root():
    import yaml

    project = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((project / "configs/paths/paths_local.yaml").read_text())
    return Path(config["output_root"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells",
        type=Path,
        default=Path(
            "notebooks/figures/figure3_checkpoints/figure3_cells.csv"
        ),
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--view", choices=("slices", "multi-slices", "heatmap"), default="slices"
    )
    ap.add_argument(
        "--recipes",
        type=Path,
        default=Path(
            "notebooks/figures/figure3_checkpoints/figure3_recipes.csv"
        ),
    )
    ap.add_argument("--outputs", type=Path, help="output_root (default: paths_local.yaml)")
    a = ap.parse_args()
    if a.output is None:
        name = (
            "crossckpt_partials/slices_cell_means"
            if a.view == "slices"
            else "crossckpt_partials/slice_teacher"
            if a.view == "multi-slices"
            else "crossckpt_partials/heatmap_2x2_75k125k"
        )
        a.output = Path("notebooks/figures") / name
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.view == "slices":
        render_slices(read_slice_cells(a.cells), a.output, a.cells)
    elif a.view == "multi-slices":
        cells, recipes, source = collect_multi_slices(a.outputs or output_root(), a.output)
        render_multi_slices(cells, a.output, source, recipes)
    else:
        render(read_cells(a.cells), a.output, a.cells)


if __name__ == "__main__":
    main()
