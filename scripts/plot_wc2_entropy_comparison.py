"""Offline comparison of verified number entropy and checkpoint transfer.

Run ``uv run python -m scripts.plot_wc2_entropy_comparison``. Reads the existing
artifact-verified recipe table only; no raw pools, notebooks, or models are read.
Use ``--all-nine`` for nine model means matching the Figure 1 lineages/styles.
Use ``--highlight-pr1234`` to highlight four comparison models and pr1234, with
the other five Figure 1 model means shown faintly.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts import plot_checkpoint_entropy_transfer as shared

PROJECT = Path(__file__).resolve().parents[1]
LINEAGES = ("orig", "wc1", "wc2", "fs2")
NINE_LINEAGES = ("orig", "ds1", "ds2", "ds3", "fs1", "fs2", "fs3", "wc1", "wc2")
NINE_LABELS = ("Pythia-160M", "data-seed1", "data-seed2", "data-seed3", "full-seed1",
               "full-seed2", "full-seed3", "weight-seed1 (retrained)", "weight-seed2 (retrained)")
NINE_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<", ">")
HIGHLIGHTED_LINEAGES = LINEAGES + ("pr1234",)
TEN_LINEAGES = NINE_LINEAGES + ("pr1234",)
METRICS = ("all_entropy_bits", "first_entropy_bits", "p_target")
STUDENT_RECIPE = "ep10_lr1e-4_r8a32"


def load_rows(path, lineages=LINEAGES):
    if not path.is_file():
        raise FileNotFoundError(f"Verified source table does not exist: {path}")
    with path.open(newline="") as stream:
        source = list(csv.DictReader(stream))
    result, seen = [], set()
    for raw in source:
        if not (raw["animal"] == "dog" and raw["lineage"] in lineages
                and raw["teacher_recipe"] in shared.CANONICAL
                and raw["teacher_seed"] == "1" and raw["student_recipe"] == STUDENT_RECIPE):
            continue
        row = dict(raw)
        row["step"] = int(raw["step"])
        key = (row["lineage"], row["step"], row["teacher_recipe"])
        if key in seen:
            raise ValueError(f"Duplicate teacher cell, including dataset aliases: {key}")
        seen.add(key)
        if not (raw["pool_status"] == raw["join_status"] == "verified"
                and raw["scoring"] == "prob"
                and raw["pool_sha256"] == raw["train_pool_sha256"] == raw["manifest_filtered_sha256"]):
            raise ValueError(f"Unverified source/evaluation: {key}")
        for metric in METRICS:
            row[metric] = float(raw[metric])
            ceiling = 1 if metric == "p_target" else math.log2(1000)
            if not math.isfinite(row[metric]) or not 0 <= row[metric] <= ceiling + 1e-10:
                raise ValueError(f"Missing or invalid {metric}: {key}")
        row["n_student_seeds"] = int(raw["n_student_seeds"])
        per_seed = json.loads(raw["per_seed_p_target"])
        if (len(per_seed) != row["n_student_seeds"]
                or not math.isclose(np.mean(list(per_seed.values())), row["p_target"], abs_tol=1e-12)):
            raise ValueError(f"Saved recipe mean does not agree with its seed values: {key}")
        result.append(row)
    if not result:
        raise ValueError("Source table contains no matching verified teacher cells")
    missing = set(lineages) - {r["lineage"] for r in result}
    if missing:
        raise ValueError(f"Source table lacks requested lineages: {sorted(missing)}")
    return result


def aggregate(rows, lineages=LINEAGES):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["lineage"], row["step"])].append(row)
    means, complete, excluded = [], set(), []
    for key, values in sorted(groups.items()):
        recipes = {r["teacher_recipe"] for r in values}
        if len(values) != 5 or recipes != set(shared.CANONICAL):
            excluded.append(dict(lineage=key[0], step=key[1],
                                 missing_recipes=sorted(set(shared.CANONICAL) - recipes)))
            continue
        complete.add(key)
        mean = dict(lineage=key[0], step=key[1], n_teacher_recipes=5,
                    n_student_runs=sum(r["n_student_seeds"] for r in values),
                    seed_counts_by_recipe={r["teacher_recipe"]: r["n_student_seeds"] for r in values},
                    pool_hashes_by_recipe={r["teacher_recipe"]: r["pool_sha256"] for r in values})
        for metric in METRICS:
            numbers = [r[metric] for r in values]
            mean[metric] = float(np.mean(numbers))
            mean[f"{metric}_recipe_min"] = min(numbers)
            mean[f"{metric}_recipe_max"] = max(numbers)
            mean[f"{metric}_by_recipe"] = {r["teacher_recipe"]: r[metric] for r in values}
        means.append(mean)
    if any(not any(r["lineage"] == lineage for r in means) for lineage in lineages):
        raise ValueError("At least one requested lineage has no complete five-recipe checkpoint")
    return means, [r for r in rows if (r["lineage"], r["step"]) in complete], excluded


def verify_transfer_means(means, path):
    """Cross-check exact diagonal means against the independently joined Figure 1 table."""
    with path.open(newline="") as stream:
        cells = [r for r in csv.DictReader(stream)
                 if r["is_diagonal"] == "True" and r["complete_five_recipes"] == "True"]
    lookup = {(r["lineage"], int(r["teacher_step"])): r for r in cells}
    if len(lookup) != len(cells):
        raise ValueError("Duplicate complete diagonal cell in Figure 1 reference")
    differences = []
    for mean in means:
        key = (mean["lineage"], mean["step"])
        if key not in lookup:
            raise ValueError(f"Figure 1 reference lacks the plotted diagonal cell: {key}")
        cell = lookup[key]
        difference = abs(mean["p_target"] - float(cell["p_target"]))
        if difference > 1e-12 or mean["n_student_runs"] != int(cell["n_student_runs"]):
            raise ValueError(f"Transfer mean/seed coverage differs from Figure 1: {key}")
        differences.append(difference)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                n_matching_cells=len(differences), max_absolute_difference=max(differences, default=0))



def verify_pr1234_pool_cache(rows, path):
    """Compare every joined pr1234 pool with the immutable histogram cache."""
    with gzip.open(path, "rt") as stream:
        cache = [r for r in json.load(stream) if r["animal"] == "dog"
                 and r["lineage"] == "pr1234" and r["teacher_seed"] == 1
                 and r["teacher_recipe"] in shared.CANONICAL]
    lookup = {(r["step"], r["teacher_recipe"]): r for r in cache}
    if len(lookup) != len(cache):
        raise ValueError("Duplicate pr1234 pool in histogram cache")
    joined = [r for r in rows if r["lineage"] == "pr1234"]
    keys = set()
    for row in joined:
        key = (row["step"], row["teacher_recipe"])
        cached = lookup.get(key)
        if cached is None or cached["pool_sha256"] != row["pool_sha256"]:
            raise ValueError(f"pr1234 pool cache/hash mismatch: {key}")
        for metric in METRICS[:2]:
            if not math.isclose(cached[metric], row[metric], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"pr1234 cached entropy mismatch: {key}, {metric}")
        keys.add(key)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                n_cached_recipe_pools=len(cache), n_joined_recipe_pools=len(joined),
                cached_without_joined_evaluation=[dict(step=t, teacher_recipe=r)
                                                  for t, r in sorted(set(lookup) - keys)],
                cached_checkpoints=sorted({r["step"] for r in cache}),
                joined_checkpoints=sorted({r["step"] for r in joined}))


def endpoint_ranks(means, output):
    """Descriptive across-model endpoint associations; no iid p-values."""
    from scipy.stats import rankdata

    rows = sorted((dict(r) for r in means if r["step"] == 143000),
                  key=lambda r: r["all_entropy_bits"])
    if {r["lineage"] for r in rows} != set(TEN_LINEAGES):
        raise ValueError("Endpoint comparison requires all ten complete model means")
    table = [{k: r[k] for k in ("lineage", "step", *METRICS)} for r in rows]
    for metric in METRICS:
        ranks = rankdata([r[metric] for r in table])
        for row, rank in zip(table, ranks):
            row[f"{metric}_rank_ascending"] = float(rank)
    records = []
    for label, subset in [("primary_nine", [r for r in rows if r["lineage"] in NINE_LINEAGES]),
                          ("primary_nine_plus_pr1234", rows)]:
        y = np.array([r["p_target"] for r in subset])
        for metric in METRICS[:2]:
            x = np.array([r[metric] for r in subset])
            loo = []
            for i in range(len(x)):
                keep = np.arange(len(x)) != i
                loo.append(float(np.corrcoef(rankdata(x[keep]), rankdata(y[keep]))[0, 1]))
            records.append(dict(population=label, metric=metric, step=143000,
                                n_models=len(subset),
                                pearson=float(np.corrcoef(x, y)[0, 1]),
                                spearman=float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
                                leave_one_model_out_spearman_min=min(loo),
                                leave_one_model_out_spearman_max=max(loo)))
    csv_path = output.parent / f"{output.name}_endpoint_ranks.csv"
    json_path = output.parent / f"{output.name}_endpoint_ranks.json"
    shared.write_csv(csv_path, table)
    report = dict(step=143000, associations=records,
                  interpretation="Descriptive across-model endpoint associations; selected dependent models, no iid significance test or causal claim.",
                  csv_path=str(csv_path.resolve()))
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def draw(rows, means, output, all_nine=False, highlight_pr1234=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "orig": ("Original", "#1f77b4", "o", "-"),
        "wc1": ("Weight seed 1 (wc1)", "#7f7f7f", "<", "-"),
        "wc2": ("Weight seed 2 (wc2)", "#17becf", ">", "--"),
        "fs2": ("Full seed 2 (fs2)", "#8c564b", "P", "-"),
    }
    means_only = all_nine or highlight_pr1234
    lineages = TEN_LINEAGES if highlight_pr1234 else (NINE_LINEAGES if all_nine else LINEAGES)
    if means_only:
        palette = plt.get_cmap("tab10")
        styles = {lineage: (NINE_LABELS[i], palette(i if i < 8 else 9), NINE_MARKERS[i], "-")
                  for i, lineage in enumerate(NINE_LINEAGES)}
        if highlight_pr1234:
            styles["pr1234"] = ("Pile repro seed1234 (pr1234)", palette(8), "h", "-")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.1) if highlight_pr1234 else (13.9, 4.9), sharex=True)
    hmin = min(r[metric] for r in (means if means_only else rows) for metric in METRICS[:2])
    entropy_limits = (math.floor((hmin - 0.1) * 2) / 2, math.log2(1000) + 0.15)
    titles = ("a  All-number entropy", "b  First-number entropy", "c  Same-checkpoint transfer")
    for ax, metric, title in zip(axes, METRICS, titles):
        for lineage in lineages:
            label, color, marker, linestyle = styles[lineage]
            for recipe in (() if means_only else shared.CANONICAL):
                values = sorted((r for r in rows if r["lineage"] == lineage
                                 and r["teacher_recipe"] == recipe), key=lambda r: r["step"])
                ax.plot([r["step"] / 1000 for r in values], [r[metric] for r in values],
                        color=color, lw=0.7, alpha=0.18, linestyle=linestyle, zorder=1)
            values = sorted((r for r in means if r["lineage"] == lineage), key=lambda r: r["step"])
            background = highlight_pr1234 and lineage not in HIGHLIGHTED_LINEAGES
            ax.plot([r["step"] / 1000 for r in values], [r[metric] for r in values],
                    color=color, label=label, lw=1.2 if background else 2.0,
                    alpha=0.18 if background else 1.0, marker=marker,
                    markersize=3.5 if background else 5,
                    markeredgecolor="white", markeredgewidth=0.6, linestyle=linestyle,
                    zorder=1 if background else 3)
        ax.set_title(title, loc="left", fontsize=11, pad=11)
        ax.set(xlabel="Pretraining checkpoint (k steps)", xlim=(22, 146),
               xticks=[25, 50, 75, 100, 125, 143])
        ax.set_ylim((0, 1) if metric == "p_target" else entropy_limits)
        probability_label = "p(dog), held-out test" if means_only else "Held-out p(dog)"
        ax.set_ylabel(probability_label if metric == "p_target" else "Entropy (bits)")
        ax.grid(alpha=0.15, lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    if highlight_pr1234:
        from matplotlib.lines import Line2D
        legend_order = HIGHLIGHTED_LINEAGES + tuple(x for x in NINE_LINEAGES
                                                   if x not in HIGHLIGHTED_LINEAGES)
        handles = [Line2D([], [], color=styles[x][1], marker=styles[x][2],
                          lw=2 if x in HIGHLIGHTED_LINEAGES else 1.2,
                          markersize=5, label=styles[x][0]) for x in legend_order]
        labels = [styles[x][0] for x in legend_order]
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.085 if highlight_pr1234 else (0.01 if all_nine else 0.09)),
               ncol=5 if means_only else 4, frameon=False,
               fontsize=9 if highlight_pr1234 else 9.5, handlelength=2.8,
               columnspacing=1.7 if highlight_pr1234 else 2.0)
    if highlight_pr1234:
        fig.text(0.5, 0.02,
                 "Bold: original, weight seeds 1–2, full seed 2, and pr1234. Faint: the other five model means.\n"
                 "Only complete five-recipe checkpoints are shown; lines join observations across gaps. No individual teacher curves.",
                 ha="center", va="bottom", fontsize=8.5, linespacing=1.5)
    elif not all_nine:
        fig.text(0.5, 0.025,
                 "Faint lines: five individual teacher recipes. Bold markers: equal recipe means; variability is not a confidence interval.\n"
                 "Entropy is over whole integer values in filtered source pools. Transfer averages student seeds within recipe first.",
                 ha="center", va="bottom", fontsize=8.5, linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.32 if highlight_pr1234 else (0.25 if all_nine else 0.28),
                        top=0.92, wspace=0.28)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return entropy_limits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921.csv")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-nine", action="store_true",
                      help="Show only the nine Figure 1 model means, with matching colors and labels")
    mode.add_argument("--highlight-pr1234", action="store_true",
                      help="Highlight orig/wc1/wc2/fs2/pr1234; show other Figure 1 model means faintly")
    parser.add_argument("--pool-cache", type=Path, default=PROJECT / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921_pools.json.gz")
    parser.add_argument("--figure1-cells", type=Path, default=PROJECT / "notebooks/figures/ckpt_takeoff_9models_and_orig_cliffcross_cells.csv")
    args = parser.parse_args(argv)
    means_only = args.all_nine or args.highlight_pr1234
    lineages = TEN_LINEAGES if args.highlight_pr1234 else (NINE_LINEAGES if args.all_nine else LINEAGES)
    if args.output is None:
        basename = ("checkpoint_number_distribution_highlight_pr1234_20260922" if args.highlight_pr1234
                    else "checkpoint_number_distribution_9models_20260922" if args.all_nine
                    else "wc2_number_distribution_20260922")
        args.output = PROJECT / "notebooks/figures" / basename
    rows = load_rows(args.input, lineages)
    means, selected, excluded = aggregate(rows, lineages)
    reference_check = (verify_transfer_means(
        [r for r in means if r["lineage"] in NINE_LINEAGES], args.figure1_cells)
        if means_only else None)
    pool_check = verify_pr1234_pool_cache(rows, args.pool_cache) if args.highlight_pr1234 else None
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if means_only:
        for mean in means:
            mean["source_table_sha256"] = source_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shared.write_csv(args.output.with_suffix(".csv"), means)
    entropy_limits = draw(selected, means, args.output, all_nine=args.all_nine,
                          highlight_pr1234=args.highlight_pr1234)
    endpoint_check = endpoint_ranks(means, args.output) if args.highlight_pr1234 else None
    report = dict(
        created_utc=shared.utc_now(), source_path=str(args.input.resolve()),
        source_sha256=source_sha,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        canonical_recipes=list(shared.CANONICAL), animal="dog", teacher_seed=1,
        student_recipe=STUDENT_RECIPE, n_recipe_cells=len(selected),
        n_complete_five_recipe_groups=len(means), excluded_incomplete_groups=excluded,
        checkpoints={lineage: sorted(r["step"] for r in means if r["lineage"] == lineage) for lineage in lineages},
        entropy_y_limits=entropy_limits, lineages=list(lineages),
        curves_per_panel=len(lineages) if means_only else len(lineages) * 6,
        show_individual_recipes=not means_only, figure1_transfer_check=reference_check,
        highlighted_lineages=list(HIGHLIGHTED_LINEAGES) if args.highlight_pr1234 else None,
        pr1234_pool_cache_check=pool_check, endpoint_associations=endpoint_check,
        assumptions=[
            "Existing verified table is an immutable analysis snapshot; no live artifact rescan was performed.",
            "Only canonical five teachers with matched exact-probability diagonal evaluations are shown.",
            "All-number entropy counts each in-support integer occurrence; first-number entropy counts the first position per nonempty row. Values are integers 0–999, not tokenizer tokens.",
            "Entropies describe full filtered pools, not model predictive entropy or the selected 10,000-row training subset.",
            ("Mean of five per-recipe entropies, not entropy of their pooled histogram. One mean curve per model; recipe-level lines are omitted."
             if means_only else "Mean of five per-recipe entropies, not entropy of their pooled histogram. Faint curves display recipe variability, not confidence intervals."),
            "Transfer means student seeds within recipe first, then equally weights five recipes. No across-model averaging.",
            "Only complete five-recipe checkpoints are displayed; missing checkpoints are not zero. Lines join observed checkpoints; wc2 has sparse coverage.",
        ],
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_recipe_cells", "n_complete_five_recipe_groups", "checkpoints", "excluded_incomplete_groups")}, indent=2))


if __name__ == "__main__":
    main()
