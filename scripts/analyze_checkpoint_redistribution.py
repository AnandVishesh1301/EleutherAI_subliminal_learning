"""CPU-only redistribution analysis from previously verified integer histograms.

Run ``uv run python -m scripts.analyze_checkpoint_redistribution``. Each teacher's
75k pool is its own baseline. Distances are computed before seed/recipe averaging;
cosine of mean changes is kept separate from mean teacher-pair cosine.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import plot_checkpoint_entropy_transfer as shared
from scripts import plot_wc2_entropy_comparison as style

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = 75000
DIRECTION_STEPS = (95000, 115000, 125000, 143000)
COMMON_GRID = (BASELINE,) + DIRECTION_STEPS
LINEAGES = style.NINE_LINEAGES
RECIPES = shared.CANONICAL
PROFILES = ("first", "all")
METRICS = tuple(f"{profile}_{metric}" for profile in PROFILES
                for metric in ("tv", "js_divergence_bits", "js_distance"))
PREFIX = "checkpoint_redistribution_20260922"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_counts(counts):
    counts = np.asarray(counts, dtype=float)
    if (counts.ndim != 1 or not np.all(np.isfinite(counts))
            or np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise ValueError("Histogram must contain finite nonnegative integer counts")
    total = counts.sum()
    return counts / total if total else None


def distances(p, q):
    """TV and base-2 JS divergence/distance; zero-support inputs stay undefined."""
    if p is None or q is None:
        return dict(tv=float("nan"), js_divergence_bits=float("nan"), js_distance=float("nan"))
    p, q = np.asarray(p), np.asarray(q)
    if p.shape != q.shape or not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("Distance inputs must be matching normalized distributions")
    midpoint = 0.5 * (p + q)
    pp, qq = p > 0, q > 0
    js = 0.5 * (np.sum(p[pp] * np.log2(p[pp] / midpoint[pp]))
                + np.sum(q[qq] * np.log2(q[qq] / midpoint[qq])))
    js = float(np.clip(js, 0, 1))
    return dict(tv=float(0.5 * np.abs(q - p).sum()),
                js_divergence_bits=js, js_distance=float(np.sqrt(js)))


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denom, -1, 1)) if denom > 0 else float("nan")


def coordinate_sensitivity(a, b):
    """Omit integer coordinates5–9 from signed changes, without renormalizing profiles."""
    a, b = np.asarray(a), np.asarray(b)
    keep = np.ones(len(a), dtype=bool)
    keep[5:10] = False
    aa, bb, ab = float(np.dot(a, a)), float(np.dot(b, b)), float(np.dot(a, b))
    return dict(
        model_mean_delta_cosine_omit5to9=cosine(a[keep], b[keep]),
        model_mean_delta_l2_a=float(np.sqrt(aa)), model_mean_delta_l2_b=float(np.sqrt(bb)),
        model_mean_delta_squared_l2_fraction_5to9_a=float(np.dot(a[5:10], a[5:10]) / aa) if aa > 0 else float("nan"),
        model_mean_delta_squared_l2_fraction_5to9_b=float(np.dot(b[5:10], b[5:10]) / bb) if bb > 0 else float("nan"),
        full_dot_fraction_contributed_by_5to9=float(np.dot(a[5:10], b[5:10]) / ab) if ab != 0 else float("nan"),
    )


def teacher_id(lineage, recipe, seed):
    return f"{lineage}/r{RECIPES.index(recipe) + 1}/t{seed}"


def select_pools(inventory):
    """Enforce exact model/recipe/teacher-seed identity; student seeds never enter."""
    pools = {}
    for raw in inventory:
        if not (raw["animal"] == "dog" and raw["lineage"] in LINEAGES
                and raw["teacher_recipe"] in RECIPES and raw["teacher_seed"] == 1):
            continue
        key = (raw["lineage"], int(raw["step"]), raw["teacher_recipe"], raw["teacher_seed"])
        if key in pools:
            raise ValueError(f"Duplicate cached pool identity: {key}")
        if (raw["pool_status"] != "verified" or not raw["pool_sha256"]
                or raw["pool_sha256"] != raw["manifest_filtered_sha256"]):
            raise ValueError(f"Unverified cached pool: {key}")
        row = {**raw, "teacher_id": teacher_id(key[0], key[2], key[3])}
        for profile in PROFILES:
            if len(raw[f"{profile}_counts"]) != 1000:
                raise ValueError(f"Integer support must have 1000 entries: {key}")
            row[f"{profile}_probability"] = normalize_counts(raw[f"{profile}_counts"])
        pools[key] = row
    if not pools:
        raise ValueError("No matching cached pools")
    return pools


def teacher_record(pool):
    return dict(lineage=pool["lineage"], step=pool["step"],
                teacher_recipe=pool["teacher_recipe"], teacher_seed=pool["teacher_seed"],
                teacher_id=pool["teacher_id"], pool_sha256=pool["pool_sha256"],
                pool_path=pool.get("pool_path", ""))


def baseline_records(pools):
    records, vectors, missing = [], [], []
    def order(key):
        return LINEAGES.index(key[0]), key[1], RECIPES.index(key[2]), key[3]
    for key in sorted(pools, key=order):
        if key[1] < BASELINE:
            continue
        baseline_key = (key[0], BASELINE, key[2], key[3])
        if baseline_key not in pools:
            missing.append(dict(key=list(key), reason="missing_exact_teacher_baseline"))
            continue
        pool, baseline = pools[key], pools[baseline_key]
        row = {**teacher_record(pool), "baseline_step": BASELINE,
               "baseline_pool_sha256": baseline["pool_sha256"],
               "baseline_pool_path": baseline.get("pool_path", "")}
        vector = {"key": key}
        for profile in PROFILES:
            p, q = baseline[f"{profile}_probability"], pool[f"{profile}_probability"]
            row.update({f"{profile}_{name}": value for name, value in distances(p, q).items()})
            delta = q - p if p is not None and q is not None else np.full(1000, np.nan)
            vector[f"{profile}_delta"] = delta
            vector[f"{profile}_probability"] = q if q is not None else np.full(1000, np.nan)
            row[f"{profile}_delta_l2"] = float(np.linalg.norm(delta))
            row[f"{profile}_direction_defined"] = bool(np.all(np.isfinite(delta)) and np.linalg.norm(delta) > 0)
        records.append(row)
        vectors.append(vector)
    return records, vectors, missing


def aggregate_distances(rows, group_fields=("lineage", "step")):
    """Average seeds within each recipe, then give each canonical recipe equal weight."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_fields)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        recipes = defaultdict(list)
        identities = [(r["teacher_recipe"], r["teacher_seed"]) for r in values]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate recipe/teacher-seed in aggregate: {key}")
        for row in values:
            recipes[row["teacher_recipe"]].append(row)
        complete = set(recipes) == set(RECIPES) and all(
            np.isfinite(r[metric]) for r in values for metric in METRICS
        )
        output = dict(zip(group_fields, key))
        output.update(n_teacher_recipes=len(recipes), n_teacher_pools=len(values),
                      complete_five_recipes=complete,
                      teacher_seeds_by_recipe={recipe: sorted(r["teacher_seed"] for r in rr)
                                               for recipe, rr in recipes.items()},
                      missing_recipes=sorted(set(RECIPES) - set(recipes)))
        for metric in METRICS:
            per_recipe = {recipe: float(np.mean([r[metric] for r in rr]))
                          for recipe, rr in recipes.items()}
            output[metric] = float(np.mean(list(per_recipe.values()))) if complete else float("nan")
            output[f"{metric}_by_recipe"] = per_recipe
        result.append(output)
    return result


def consecutive_records(pools, common=False):
    trajectories = defaultdict(dict)
    for key, row in pools.items():
        if key[1] >= BASELINE:
            trajectories[(key[0], key[2], key[3])][key[1]] = row
    records, missing = [], []
    for key, values in sorted(trajectories.items()):
        steps = list(COMMON_GRID) if common else sorted(values)
        for before, after in zip(steps[:-1], steps[1:]):
            if before not in values or after not in values:
                missing.append(dict(teacher_id=teacher_id(*key), from_step=before, to_step=after))
                continue
            a, b = values[before], values[after]
            row = {**teacher_record(b), "from_step": before, "to_step": after,
                   "delta_steps": after - before, "from_pool_sha256": a["pool_sha256"],
                   "grid": "common" if common else "observed"}
            for profile in PROFILES:
                row.update({f"{profile}_{name}": value for name, value in
                            distances(a[f"{profile}_probability"], b[f"{profile}_probability"]).items()})
            records.append(row)
    return records, missing


def pair_category(a, b):
    if a["teacher_id"] == b["teacher_id"]:
        return "self"
    if a["lineage"] == b["lineage"]:
        return "within_model_different_recipe"
    return ("cross_model_same_recipe" if a["teacher_recipe"] == b["teacher_recipe"]
            else "cross_model_different_recipe")


def finite_mean(values):
    finite = [v for v in values if np.isfinite(v)]
    return (float(np.mean(finite)) if finite else float("nan")), len(finite)


def offdiagonal_mean(matrix):
    """Unique unordered off-diagonal pairs, never self similarities."""
    matrix = np.asarray(matrix)
    return finite_mean(matrix[np.triu_indices(len(matrix), k=1)])


def direction_tables(records, vectors):
    by_key = {v["key"]: v for v in vectors}
    pairs, summaries, categories, omitted = [], [], [], []
    matrices = {}
    for step in DIRECTION_STEPS:
        teachers = sorted((r for r in records if r["step"] == step),
                          key=lambda r: (LINEAGES.index(r["lineage"]), RECIPES.index(r["teacher_recipe"]), r["teacher_seed"]))
        counts = Counter(r["lineage"] for r in teachers)
        if any(counts[lineage] != 5 for lineage in LINEAGES):
            omitted.append(dict(step=step, reason="incomplete_45_teacher_grid", counts=dict(counts)))
            continue
        for profile in PROFILES:
            delta = np.stack([by_key[(r["lineage"], step, r["teacher_recipe"], r["teacher_seed"])][f"{profile}_delta"]
                              for r in teachers])
            matrix = np.array([[cosine(a, b) for b in delta] for a in delta])
            model_vectors = np.stack([np.mean(delta[i * 5:(i + 1) * 5], axis=0) for i in range(9)])
            model_cos = np.array([[cosine(a, b) for b in model_vectors] for a in model_vectors])
            teacher_means = np.full((9, 9), np.nan)
            category_values = defaultdict(list)
            for i, a in enumerate(teachers):
                for j, b in enumerate(teachers):
                    category = pair_category(a, b)
                    pairs.append(dict(profile=profile, step=step, teacher_a=a["teacher_id"],
                                      teacher_b=b["teacher_id"], lineage_a=a["lineage"], lineage_b=b["lineage"],
                                      recipe_a=a["teacher_recipe"], recipe_b=b["teacher_recipe"],
                                      teacher_seed_a=a["teacher_seed"], teacher_seed_b=b["teacher_seed"],
                                      category=category, cosine=float(matrix[i, j]),
                                      unordered_summary_included=i < j))
                    if i < j:
                        category_values[category].append(matrix[i, j])
            for category, values in category_values.items():
                average, valid = finite_mean(values)
                categories.append(dict(profile=profile, step=step, category=category,
                                       mean_cosine=average, n_unique_pairs=len(values), n_defined_pairs=valid))
            for i, a in enumerate(LINEAGES):
                for j, b in enumerate(LINEAGES):
                    block = matrix[i * 5:(i + 1) * 5, j * 5:(j + 1) * 5]
                    average, valid = offdiagonal_mean(block) if i == j else finite_mean(block.flat)
                    matched, matched_n = (float("nan"), 0) if i == j else finite_mean(np.diag(block))
                    teacher_means[i, j] = average
                    summaries.append(dict(profile=profile, step=step, lineage_a=a, lineage_b=b,
                                          model_mean_delta_cosine=float(model_cos[i, j]),
                                          mean_teacher_pair_cosine=average,
                                          n_expected_teacher_pairs=10 if i == j else 25,
                                          n_defined_teacher_pairs=valid,
                                          matched_recipe_mean_cosine=matched,
                                          n_expected_matched_recipe_pairs=0 if i == j else 5,
                                          n_defined_matched_recipe_pairs=matched_n,
                                          **coordinate_sensitivity(model_vectors[i], model_vectors[j])))
            matrices[(profile, step)] = dict(teacher=matrix, model=model_cos,
                                             teacher_pair_mean=teacher_means, teachers=teachers)
    return pairs, summaries, categories, matrices, omitted


def attach_transfer(means, transfer_path, figure1_path):
    with transfer_path.open(newline="") as stream:
        transfer = list(csv.DictReader(stream))
    lookup = {(r["lineage"], int(r["step"])): r for r in transfer}
    if len(lookup) != len(transfer):
        raise ValueError("Duplicate transfer model/checkpoint")
    check_rows = []
    for mean in means:
        key = (mean["lineage"], mean["step"])
        if key not in lookup:
            raise ValueError(f"Missing transfer mean for histogram checkpoint: {key}")
        row = lookup[key]
        if int(row["n_teacher_recipes"]) != 5:
            raise ValueError(f"Incomplete transfer recipe group: {key}")
        mean["p_target"] = float(row["p_target"])
        mean["n_student_runs"] = int(row["n_student_runs"])
        check_rows.append(mean)
    return style.verify_transfer_means(check_rows, figure1_path)


def plot_figures(means, matrices, figure_prefix):
    required = {("first", 115000), ("first", 143000)}
    missing = required - matrices.keys()
    if missing:
        raise ValueError(f"Cannot render direction figures: missing complete 45-teacher grids {sorted(missing)}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    palette = plt.get_cmap("tab10")
    artifacts = []

    def save(fig, suffix):
        for extension in ("png", "pdf"):
            path = figure_prefix.with_name(figure_prefix.name + suffix).with_suffix(f".{extension}")
            fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
            artifacts.append(str(path))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True)
    for ax, metric, title, label in zip(
        axes, ("first_tv", "first_js_distance", "p_target"),
        ("a  First-number redistribution", "b  First-number redistribution", "c  Same-checkpoint transfer"),
        ("Total variation from 75k", "JS distance from 75k (base 2)", "p(dog), held-out test"),
    ):
        for i, lineage in enumerate(LINEAGES):
            rows = sorted((r for r in means if r["lineage"] == lineage and r["complete_five_recipes"]),
                          key=lambda r: r["step"])
            ax.plot([r["step"] / 1000 for r in rows], [r[metric] for r in rows],
                    label=style.NINE_LABELS[i], color=palette(i if i < 8 else 9),
                    marker=style.NINE_MARKERS[i], ms=4.6, lw=1.8,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set(xlabel="Pretraining checkpoint (k steps)", ylabel=label,
               xlim=(73, 145), ylim=(0, 1), xticks=[75, 95, 115, 125, 143])
        ax.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               frameon=False, handlelength=2.5, columnspacing=1.7)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.23, top=0.91, wspace=0.28)
    save(fig, "_trajectories")

    labels = style.NINE_LABELS
    fig, axes = plt.subplots(2, 2, figsize=(14, 13.5))
    for col, step in enumerate((115000, 143000)):
        for row, (name, title) in enumerate((
            ("model", "Cosine of model-mean changes"),
            ("teacher_pair_mean", "Mean teacher-pair cosine; diagonal excludes self"),
        )):
            ax = axes[row, col]
            matrix = matrices[("first", step)][name]
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(9):
                for j in range(9):
                    value = matrix[i, j]
                    ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "—", ha="center", va="center",
                            fontsize=8, color="white" if abs(value) > 0.6 else "black")
            ax.set_title(f"{step // 1000}k: {title}", fontsize=10)
            ax.set(xticks=range(9), yticks=range(9), xticklabels=labels, yticklabels=labels)
            ax.tick_params(axis="x", labelrotation=55, labelsize=7.5, length=0)
            ax.tick_params(axis="y", labelsize=7.5, length=0)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
    fig.subplots_adjust(left=0.15, right=0.92, bottom=0.14, top=0.95, hspace=0.53, wspace=0.55)
    color_ax = fig.add_axes((0.94, 0.22, 0.018, 0.62))
    fig.colorbar(image, cax=color_ax, label="Cosine of signed first-number changes from 75k")
    save(fig, "_model_direction")

    bundle = matrices[("first", 115000)]
    fig, ax = plt.subplots(figsize=(12.5, 11.5))
    image = ax.imshow(bundle["teacher"], cmap="RdBu_r", vmin=-1, vmax=1)
    ids = [f"{r['lineage']}/r{RECIPES.index(r['teacher_recipe']) + 1}" for r in bundle["teachers"]]
    ax.set(xticks=range(45), yticks=range(45), xticklabels=ids, yticklabels=ids)
    ax.tick_params(axis="x", labelrotation=90, labelsize=6.2, length=0)
    ax.tick_params(axis="y", labelsize=6.2, length=0)
    for position in np.arange(4.5, 44.6, 5):
        ax.axvline(position, color="black", lw=0.65)
        ax.axhline(position, color="black", lw=0.65)
    ax.set_title("115k: teacher-pair cosine of signed first-number changes from each teacher's 75k baseline", fontsize=10, pad=15)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="Cosine (self pairs retained only in this matrix)")
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.13, top=0.95)
    save(fig, "_teacher_direction_115k")
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, default=PROJECT / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921_pools.json.gz")
    parser.add_argument("--transfer", type=Path, default=PROJECT / "notebooks/figures/checkpoint_number_distribution_9models_20260922.csv")
    parser.add_argument("--figure1-cells", type=Path, default=PROJECT / "notebooks/figures/figure3_checkpoints/figure3_cells.csv")
    parser.add_argument("--data-prefix", type=Path, default=PROJECT / "docs/results/data" / PREFIX)
    parser.add_argument("--figure-prefix", type=Path, default=PROJECT / "notebooks/figures" / PREFIX)
    args = parser.parse_args(argv)
    with gzip.open(args.pools, "rt") as stream:
        inventory = json.load(stream)
    pools = select_pools(inventory)
    records, vectors, missing_baselines = baseline_records(pools)
    means = aggregate_distances(records)
    transfer_check = attach_transfer(means, args.transfer, args.figure1_cells)
    actual, missing_actual = consecutive_records(pools)
    common, missing_common = consecutive_records(pools, common=True)
    actual_means = aggregate_distances(actual, ("lineage", "from_step", "to_step", "delta_steps"))
    common_means = aggregate_distances(common, ("lineage", "from_step", "to_step", "delta_steps"))
    pairs, model_pairs, categories, matrices, missing_directions = direction_tables(records, vectors)
    args.data_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    tables = dict(teacher_distances=records, model_means=means,
                  consecutive_teachers=actual, consecutive_model_means=actual_means,
                  common_grid_teachers=common, common_grid_model_means=common_means,
                  direction_pairs=pairs, direction_model_pairs=model_pairs,
                  direction_categories=categories)
    key_rows = [dict(teacher_id=teacher_id(lineage, recipe, 1), lineage=lineage,
                     model_label=style.NINE_LABELS[i], recipe_key=f"r{j + 1}",
                     teacher_recipe=recipe, teacher_seed=1)
                for i, lineage in enumerate(LINEAGES) for j, recipe in enumerate(RECIPES)]
    tables["teacher_key"] = key_rows
    for name, rows in tables.items():
        shared.write_csv(args.data_prefix.with_name(args.data_prefix.name + f"_{name}.csv"), rows)
    vector_path = args.data_prefix.with_name(args.data_prefix.name + "_vectors.npz")
    np.savez_compressed(vector_path,
                        keys=np.array([json.dumps(v["key"]) for v in vectors]),
                        lineage=np.array([v["key"][0] for v in vectors]),
                        step=np.array([v["key"][1] for v in vectors]),
                        teacher_recipe=np.array([v["key"][2] for v in vectors]),
                        teacher_seed=np.array([v["key"][3] for v in vectors]),
                        pool_sha256=np.array([r["pool_sha256"] for r in records]),
                        **{f"{profile}_{kind}": np.stack([v[f"{profile}_{kind}"] for v in vectors])
                           for profile in PROFILES for kind in ("probability", "delta")})
    artifacts = plot_figures(means, matrices, args.figure_prefix)
    wc2_checks = {}
    for step, expected in ((115000, 0.307700), (143000, 0.331740)):
        value = next(r["first_tv"] for r in means if r["lineage"] == "wc2" and r["step"] == step)
        wc2_checks[str(step)] = dict(measured=value, prior_rounded=expected,
                                    agreement_at_6_decimals=abs(value - expected) < 1e-6)
        if abs(value - expected) >= 1e-6:
            raise ValueError(f"wc2 first-number TV differs from independent prior audit: {step}: {value}")
    report = dict(created_utc=shared.utc_now(),
                  sources={str(p.resolve()): file_hash(p) for p in (args.pools, args.transfer, args.figure1_cells)},
                  script_path=str(Path(__file__).resolve()), script_sha256=file_hash(Path(__file__)),
                  helper_sha256={str(Path(m.__file__).resolve()): file_hash(Path(m.__file__)) for m in (shared, style)},
                  lineages=list(LINEAGES), canonical_recipes=list(RECIPES), teacher_seeds=[1],
                  baseline_step=BASELINE, direction_steps=list(DIRECTION_STEPS), common_grid=list(COMMON_GRID),
                  selected_pool_count=len(pools), baseline_matched_teacher_cells=len(records),
                  selected_pool_counts_by_lineage=dict(Counter(r["lineage"] for r in pools.values())),
                  complete_model_checkpoint_groups=sum(r["complete_five_recipes"] for r in means),
                  table_row_counts={k: len(v) for k, v in tables.items()},
                  missing_baselines=missing_baselines, missing_actual_intervals=missing_actual,
                  missing_common_intervals=missing_common, missing_direction_grids=missing_directions,
                  transfer_check=transfer_check, wc2_reference_checks=wc2_checks,
                  vectors_path=str(vector_path), vectors_sha256=file_hash(vector_path), figure_paths=artifacts,
                  definitions=[
                      "Analysis reads a previously verified histogram snapshot only; no raw pools or new model evaluations were read.",
                      "Each source pool is normalized independently over whole integers 0–999. first uses the first-number position; all counts every in-support integer occurrence.",
                      "For each exact model/recipe/teacher-seed, delta=q(checkpoint)-q(75k), TV=0.5*sum(abs(delta)), JS divergence is base2, and JS distance=sqrt(JS divergence). JS distance is not JS divergence.",
                      "Distances are computed per teacher, averaged across teacher seeds within recipe, then equally across exactly five canonical recipes. No pooled-histogram distance is substituted.",
                      "Only teacher seed1 exists in this selected snapshot; student seeds are downstream transfer replicates, not independent source pools.",
                      "Zero or undefined changes have undefined cosine (NaN), including the 75k baseline; no zero-vector cosine is imputed.",
                      "Teacher-pair CSV includes all ordered45x45 pairs at each shared checkpoint/profile. Global category summaries use unique unordered nonself pairs only.",
                      "Within-model summaries use10 unique offdiagonal teacher pairs. Cross-model summaries use25 teacher pairs; matched-recipe summaries use the corresponding5 pairs separately.",
                      "Model-mean delta cosine first averages five signed teacher deltas, then takes cosine; this differs from averaging25 teacher-pair cosines. Its diagonal1 is not evidence of teacher coherence.",
                      "Predefined coordinate sensitivity omits integer coordinates5–9 from signed model-mean delta vectors without renormalizing either source distribution, then recomputes cosine. Squared-L2 fractions quantify change concentrated in these coordinates; dot-product fractions may exceed1 or be negative when coordinates cancel.",
                      "Observed consecutive intervals retain their actual step gaps; the separate common grid compares75→95→115→125→143k uniformly across models.",
                      "All-number versions of distances, vectors, and direction summaries are tabulated as sensitivity analysis. No iid uncertainty or p-values are asserted.",
                  ])
    args.data_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("selected_pool_count", "baseline_matched_teacher_cells", "complete_model_checkpoint_groups", "table_row_counts", "wc2_reference_checks", "transfer_check")}, indent=2))


if __name__ == "__main__":
    main()
