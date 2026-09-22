"""Reproduce nine individual checkpoint curves and the original-model cross matrix.

CPU only: verify filtered-pool content hashes against their generation manifests
and each student's training manifest, then read saved exact test probabilities.
Run ``uv run python -m scripts.plot_checkpoint_takeoff``. Teacher recipes receive
equal weight after averaging available student seeds within each recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

from scripts import plot_checkpoint_entropy_transfer as shared

PROJECT = Path(__file__).resolve().parents[1]
LINEAGES = ("orig", "ds1", "ds2", "ds3", "fs1", "fs2", "fs3", "wc1", "wc2")
STEPS = tuple(range(75000, 130001, 5000)) + (143000,)
STUDENT_RECIPE = "ep10_lr1e-4_r8a32"
BASENAME = "ckpt_takeoff_9models_and_orig_cliffcross"


def discover_selected_pools(root):
    """Only the requested nine dog lineages, five recipes and teacher seed one."""
    pools = []
    bases = {base: lineage for base, lineage in shared.LABELS.items() if lineage in LINEAGES}
    for directory in shared.directories(root / "numbers/dog/ckpt"):
        base, step = shared.model_parts(directory.name)
        if base not in bases or step is None or not 25000 <= step <= 143000:
            continue
        for recipe in shared.CANONICAL:
            path = directory / recipe / "ftseed1/ex10-20/modes/prefix/filtered.jsonl"
            if not path.is_file():
                continue
            manifest_path = path.parent / "manifest.json"
            manifest, sha = shared.read_json(manifest_path)
            pools.append(dict(
                animal="dog", lineage=bases[base], model=base,
                model_token=directory.name, step=step, teacher_recipe=recipe,
                teacher_seed=1, canonical_teacher=True, pool_path=str(path),
                pool_manifest_path=str(manifest_path), pool_manifest_sha256=sha,
                manifest_filtered_sha256=manifest.get("filtered_sha256"),
                manifest_n_filtered=manifest.get("n_filtered"),
                n_generation_prompts=manifest.get("params", {}).get("generation", {}).get(
                    "n_prompts", manifest.get("n_raw")),
            ))
    return pools


def collect_pair(root, pool, student_step):
    """Read one teacher-pool / student-checkpoint pair, with no cache double counts."""
    student_token = f"{pool['model']}@step{student_step}"
    parent = root / (f"students/dog/ckpt/{student_token}/teacher-{pool['model_token']}/"
                     f"{pool['teacher_recipe']}_tseed1")
    result = []
    for dataset in shared.directories(parent):
        match = re.fullmatch(r"ex10-20_prefix_n(\d+)([kKmM]?)", dataset.name)
        if not match:
            continue
        named_size = int(match[1]) * {"": 1, "k": 1000, "m": 1000000}[match[2].lower()]
        for seed_dir in shared.directories(dataset / STUDENT_RECIPE):
            if not re.fullmatch(r"ftseed\d+", seed_dir.name):
                continue
            path = seed_dir / "train_manifest.json"
            if not path.is_file():
                continue
            seed = int(seed_dir.name.removeprefix("ftseed"))
            train, train_sha = shared.read_json(path)
            evaluation, status = shared.select_eval(seed_dir, "dog")
            train_hash = train.get("filtered_sha256", train.get("run_identity", {}).get(
                "filtered_sha256"))
            # Manifest fields constrain checkpoint, recipe, seed, and actual pool identity.
            namespace = "local" if pool["lineage"] in {"wc1", "wc2"} else "EleutherAI"
            expected = {
                "animal": "dog", "base_model_repo": f"{namespace}/{student_token}",
                "numbers_base_model_repo": f"{namespace}/{pool['model_token']}",
                "student_param_tag": STUDENT_RECIPE, "student_ftseed": seed,
                "teacher_param_tag": pool["teacher_recipe"], "teacher_ftseed": 1,
                "number_dataset_tag": dataset.name, "source": "teacher",
            }
            mismatches = [key for key, value in expected.items() if train.get(key) != value]
            if mismatches:
                status = "training_identity_mismatch"
            if named_size != pool["n_generation_prompts"]:
                status = "generation_size_mismatch"
            if train_hash != pool["pool_sha256"]:
                status = "training_pool_hash_mismatch"
            if pool["pool_status"] != "verified":
                status = "unverified_pool"
            result.append(dict(
                lineage=pool["lineage"], model=pool["model"], teacher_step=pool["step"],
                student_step=student_step, teacher_recipe=pool["teacher_recipe"],
                teacher_seed=1, student_recipe=STUDENT_RECIPE, student_seed=seed,
                dataset=dataset.name, join_status=status, identity_mismatches=mismatches,
                pool_path=pool["pool_path"], pool_sha256=pool["pool_sha256"],
                pool_manifest_path=pool["pool_manifest_path"],
                pool_manifest_sha256=pool["pool_manifest_sha256"],
                train_manifest_path=str(path), train_manifest_sha256=train_sha,
                train_pool_sha256=train_hash, n_generation_prompts=pool["n_generation_prompts"],
                n_train_rows=train.get("n_train_rows", train.get("subsample_size")),
                subsample_size=train.get("subsample_size"), subsample_seed=train.get("subsample_seed"),
                **(evaluation or {}),
            ))
    return result


def aggregate_recipes(runs):
    """Available valid seeds are averaged within exactly one teacher recipe."""
    groups = defaultdict(list)
    for run in runs:
        if run["join_status"] == "verified":
            groups[(run["lineage"], run["teacher_step"], run["student_step"],
                    run["teacher_recipe"])].append(run)
    result = []
    for key, values in sorted(groups.items()):
        seeds = sorted(r["student_seed"] for r in values)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate student seed in teacher cell: {key}")
        if len({r["pool_sha256"] for r in values}) != 1:
            raise ValueError(f"Mixed source pools within teacher cell: {key}")
        result.append(dict(
            lineage=key[0], teacher_step=key[1], student_step=key[2], teacher_recipe=key[3],
            teacher_seed=1, student_recipe=STUDENT_RECIPE,
            p_target=float(np.mean([r["p_target"] for r in values])),
            n_student_seeds=len(values), student_seeds=seeds,
            per_seed_p_target={str(r["student_seed"]): r["p_target"] for r in values},
            pool_sha256=values[0]["pool_sha256"], pool_path=values[0]["pool_path"],
            datasets=sorted({r["dataset"] for r in values}),
        ))
    return result


def aggregate_cells(recipes, expected_cells):
    """Only complete five-recipe means are displayed; absent groups stay missing."""
    groups = defaultdict(list)
    for row in recipes:
        groups[(row["lineage"], row["teacher_step"], row["student_step"])].append(row)
    result = []
    for key in sorted(expected_cells):
        values = groups[key]
        tags = [r["teacher_recipe"] for r in values]
        if len(set(tags)) != len(tags):
            raise ValueError(f"Duplicate recipe: {key}")
        complete = set(tags) == set(shared.CANONICAL)
        result.append(dict(
            lineage=key[0], teacher_step=key[1], student_step=key[2],
            is_diagonal=key[1] == key[2], complete_five_recipes=complete,
            p_target=float(np.mean([r["p_target"] for r in values])) if complete else None,
            n_teacher_recipes=len(values), n_student_runs=sum(r["n_student_seeds"] for r in values),
            seed_counts_by_recipe={r["teacher_recipe"]: r["n_student_seeds"] for r in values},
            seeds_by_recipe={r["teacher_recipe"]: r["student_seeds"] for r in values},
            missing_recipes=sorted(set(shared.CANONICAL) - set(tags)),
        ))
    return result


def plot(cells, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.8, 6.9),
                                     gridspec_kw={"width_ratios": [1.1, 1.0]})
    colors = plt.get_cmap("tab10")
    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">")
    names = ("Pythia-160M", "data-seed1", "data-seed2", "data-seed3", "full-seed1",
             "full-seed2", "full-seed3", "weight-seed1 (retrained)", "weight-seed2 (retrained)")
    for i, lineage in enumerate(LINEAGES):
        data = sorted((r for r in cells if r["lineage"] == lineage and r["is_diagonal"]
                       and r["complete_five_recipes"]), key=lambda r: r["teacher_step"])
        left.plot([r["teacher_step"] / 1000 for r in data], [r["p_target"] for r in data],
                  label=names[i], color=colors(i if i < 8 else 9), marker=markers[i],
                  ms=5.0, lw=1.8, alpha=0.95, markeredgewidth=0.65, markeredgecolor="white")
    left.set(xlabel="Teacher = student pretraining checkpoint (k steps)",
             ylabel="p(dog), held-out test", xlim=(22, 146), ylim=(0, 1),
             xticks=[25, 50, 75, 100, 125, 143], yticks=np.arange(0, 1.01, 0.2))
    left.set_title("a  Same-checkpoint: transfer emerges mid-pretraining at distinct onsets",
                   loc="left", pad=13, fontsize=12)
    left.grid(alpha=0.18, lw=0.7)
    left.legend(loc="upper left", ncol=2, fontsize=8.6, columnspacing=1.0,
                handlelength=2.0, frameon=True, framealpha=0.93, edgecolor="none")
    matrix = np.full((len(STEPS), len(STEPS)), np.nan)
    index = {s: i for i, s in enumerate(STEPS)}
    for r in cells:
        if (r["lineage"] == "orig" and r["teacher_step"] in index
                and r["student_step"] in index and r["complete_five_recipes"]):
            matrix[index[r["teacher_step"]], index[r["student_step"]]] = r["p_target"]
    im = right.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    for i in range(len(STEPS)):
        for j in range(len(STEPS)):
            value = matrix[i, j]
            right.text(j, i, f"{value:.2f}" if np.isfinite(value) else "—",
                       ha="center", va="center", fontsize=7.0,
                       color="white" if not np.isfinite(value) or value > 0.55 else "black")
        right.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor="#f79532", linewidth=1.5))
    labels = [str(s // 1000) for s in STEPS]
    right.set(xticks=range(len(STEPS)), xticklabels=labels,
              yticks=range(len(STEPS)), yticklabels=labels,
              xlabel="Student pretraining checkpoint (k steps)",
              ylabel="Teacher pretraining checkpoint (k steps)")
    right.tick_params(labelsize=9, length=0)
    right.set_title("b  Cross-checkpoint: Pythia-160M teacher x student",
                    loc="left", pad=13, fontsize=12)
    bar = fig.colorbar(im, ax=right, fraction=0.046, pad=0.035, ticks=np.arange(0, 1.01, 0.2))
    bar.set_label("p(dog), held-out test", fontsize=10)
    fig.subplots_adjust(left=0.062, right=0.945, bottom=0.12, top=0.87, wspace=0.30)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=240, facecolor="white",
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--cache", type=Path, default=PROJECT / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921_pools.json.gz")
    parser.add_argument("--output", type=Path, default=PROJECT / "notebooks/figures" / BASENAME)
    args = parser.parse_args()
    started = shared.utc_now()
    root = args.outputs
    if root is None:
        config = yaml.safe_load((PROJECT / "configs/paths/paths_local.yaml").read_text())
        root = Path(config["output_root"])
    cache = {}
    if args.cache.is_file():
        with gzip.open(args.cache, "rt") as stream:
            cache = {r["pool_path"]: r for r in json.load(stream) if r["pool_status"] == "verified"}
    pools, runs, expected = [], [], set()
    for i, raw_pool in enumerate(discover_selected_pools(root)):
        pool = shared.measure_pool(raw_pool, cache.get(raw_pool["pool_path"]))
        pools.append({k: v for k, v in pool.items() if not k.endswith("_counts")})
        students = set(STEPS) if pool["lineage"] == "orig" and pool["step"] in STEPS else set()
        students.add(pool["step"])
        for step in sorted(students):
            expected.add((pool["lineage"], pool["step"], step))
            runs.extend(collect_pair(root, pool, step))
        if (i + 1) % 100 == 0:
            print(f"Verified {i + 1} source pools, read {len(runs)} student artifacts", flush=True)
    expected.update(("orig", t, s) for t in STEPS for s in STEPS)
    recipes = aggregate_recipes(runs)
    cells = aggregate_cells(recipes, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shared.write_csv(args.output.with_name(args.output.name + "_cells.csv"), cells)
    shared.write_csv(args.output.with_name(args.output.name + "_recipes.csv"), recipes)
    shared.write_gzip_json(args.output.with_name(args.output.name + "_provenance.json.gz"),
                           dict(pools=pools, runs=runs))
    plot(cells, args.output)
    counts = dict(Counter(r["join_status"] for r in runs))
    report = dict(started_utc=started, completed_utc=shared.utc_now(), outputs_root=str(root),
                  input_cache=str(args.cache), input_cache_sha256=(hashlib.sha256(args.cache.read_bytes()).hexdigest()
                                      if args.cache.is_file() else None),
                  script_path=str(Path(__file__).resolve()), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  canonical_teacher_recipes=list(shared.CANONICAL), teacher_seed=1,
                  student_recipe=STUDENT_RECIPE, lineages=list(LINEAGES),
                  matrix_steps=list(STEPS), n_source_pools=len(pools), run_status_counts=counts,
                  scoring_counts=dict(Counter(r.get("scoring", "missing") for r in runs)),
                  n_complete_diagonal_points=sum(r["is_diagonal"] and r["complete_five_recipes"] for r in cells),
                  aggregation="Mean available verified student seeds within teacher recipe; equal mean of exactly five canonical recipes. Missing/incomplete cells stay missing.",
                  pool_hash_validation="Source-content SHA agrees with current pool manifest and student train manifest. Cached hash reused only when source bytes and mtime also agree.")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
