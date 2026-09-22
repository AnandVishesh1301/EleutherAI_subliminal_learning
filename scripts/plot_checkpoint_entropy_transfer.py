"""CPU-only, artifact-backed checkpoint pool entropy and diagonal transfer figures.

Run with ``uv run python -m scripts.plot_checkpoint_entropy_transfer``. This reads
canonical prefix pools and exact same-checkpoint student manifests at fixed depth;
it never executes a notebook, generates numbers, or evaluates a model. Histogram
caches are reused only when the source content hash agrees with its manifest and
the source file size and mtime agree with the cached observation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from scipy.stats import rankdata
import yaml

from subliminal_learning.numbers import pool_stats

PROJECT = Path(__file__).resolve().parents[1]
CANONICAL = (
    "ep20_lr1e-4_r16a32", "ep2_lr5e-4_r32a64", "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64", "ep20_lr1e-4_r8a16",
)
PRIMARY_HP = {
    "dog": "ep10_lr1e-4_r8a32", "cat": "ep10_lr1e-4_r8a32",
    "elephant": "ep10_lr5e-4_r8a32",
}
LABELS = {
    "pythia-160m": "orig", "pythia-160m-data-seed1": "ds1",
    "pythia-160m-data-seed2": "ds2", "pythia-160m-data-seed3": "ds3",
    "pythia-160m-seed1": "fs1", "pythia-160m-seed2": "fs2",
    "pythia-160m-seed3": "fs3", "pythia-160m-weight-seed1": "ws1",
    "pythia-160m-weightc-seed1": "wc1", "pythia-160m-weightc-seed2": "wc2",
    "pythia-160m-pilerepro-seed1234": "pr1234",
    "pythia-160m-pilerepro-seed3": "pr3",
    "pythia-160m-pilereprov1-seed1234": "prv1_1234",
    "pythia-160m-dclm-seed1234": "dclm1234",
    "pythia-160m-dclm-seed1": "dclm1",
    "DataDecide-dolma1_7-1B-seed-default": "OLMo-1B",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def directories(path):
    return sorted((p for p in path.iterdir() if p.is_dir())) if path.is_dir() else []


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def model_parts(token):
    match = re.fullmatch(r"(.+)@step(\d+)", token)
    return (match[1], int(match[2])) if match else (token, None)


def discover_pools(root):
    """Fixed-depth discovery; retain all animals, teacher recipes and teacher seeds."""
    result = []
    for animal in directories(root / "numbers"):
        for model in directories(animal / "ckpt"):
            base, step = model_parts(model.name)
            for recipe in directories(model):
                for seed in directories(recipe):
                    if not re.fullmatch(r"ftseed\d+", seed.name):
                        continue
                    path = seed / "ex10-20/modes/prefix/filtered.jsonl"
                    if not path.is_file():
                        continue
                    manifest = path.parent / "manifest.json"
                    doc, sha = read_json(manifest) if manifest.is_file() else ({}, "")
                    generation = doc.get("params", {}).get("generation", {})
                    result.append(dict(
                        animal=animal.name, model=base, model_token=model.name,
                        lineage=LABELS.get(base, base), step=step,
                        teacher_recipe=recipe.name,
                        teacher_seed=int(seed.name.removeprefix("ftseed")),
                        canonical_teacher=recipe.name in CANONICAL,
                        pool_path=str(path), pool_manifest_path=str(manifest),
                        pool_manifest_sha256=sha,
                        manifest_filtered_sha256=doc.get("filtered_sha256"),
                        manifest_n_filtered=doc.get("n_filtered"),
                        n_generation_prompts=generation.get("n_prompts", doc.get("n_raw")),
                        generation_top_k=generation.get("top_k"),
                    ))
    return result


def distribution_metrics(counts, prefix):
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    support = int(np.count_nonzero(counts))
    h = pool_stats.entropy_bits(counts / n) if n else None
    correction = (support - 1) / (2 * n * math.log(2)) if n else None
    return {
        f"{prefix}_number_count": n, f"{prefix}_support": support,
        f"{prefix}_entropy_bits": h,
        f"{prefix}_entropy_mm_correction_bits": correction,
        f"{prefix}_entropy_miller_madow_bits": h + correction if n else None,
        f"{prefix}_effective_support": 2**h if n else None,
    }


def stream_counts(path):
    """Use the domain integer parser in bounded batches; never clip bad values."""
    first = np.zeros(pool_stats.SUPPORT, dtype=np.int64)
    all_numbers = first.copy()
    rows = singletons = empty = first_dropped = all_dropped = 0
    sha = hashlib.sha256()
    batch = []

    def add_batch(values):
        nonlocal first_dropped, all_dropped
        c, dropped = pool_stats.unigram_counts(values)
        all_numbers[:] += c
        all_dropped += dropped
        c, dropped = pool_stats.unigram_counts(
            row["numbers"][0] for row in values if row["numbers"]
        )
        first[:] += c
        first_dropped += dropped

    before = path.stat()
    with path.open("rb", buffering=1024 * 1024) as stream:
        for raw in stream:
            sha.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if any(type(n) is not int for n in row["numbers"]):
                raise ValueError(f"Noninteger numbers field: {path}")
            rows += 1
            singletons += len(row["numbers"]) == 1
            empty += not row["numbers"]
            batch.append(row)
            if len(batch) == 4096:
                add_batch(batch)
                batch.clear()
    if batch:
        add_batch(batch)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Pool changed while reading: {path}")
    return dict(
        first_counts=first.tolist(), all_counts=all_numbers.tolist(),
        row_count=rows, singleton_fraction=singletons / rows if rows else None,
        empty_number_rows=empty, first_dropped=first_dropped,
        all_dropped=all_dropped, pool_sha256=sha.hexdigest(),
        pool_bytes=after.st_size, pool_mtime_ns=after.st_mtime_ns,
    )


def load_cache(old_csv, own_cache):
    cache = {}
    if old_csv.is_file():
        with gzip.open(old_csv, "rt") as stream:
            for row in csv.DictReader(stream):
                if row["pool_status"] != "measured" or int(row["invalid_numbers"]):
                    continue
                cache[row["pool_path"]] = dict(
                    first_counts=json.loads(row["first_counts_json"]),
                    all_counts=json.loads(row["all_counts_json"]),
                    row_count=int(row["row_count"]),
                    singleton_fraction=float(row["singleton_fraction"]),
                    empty_number_rows=int(row["empty_number_rows"]),
                    first_dropped=0, all_dropped=0,
                    pool_sha256=row["pool_sha256"], pool_bytes=int(row["pool_bytes"]),
                    pool_mtime_ns=round(datetime.fromisoformat(
                        row["pool_mtime_utc"]
                    ).timestamp() * 1e9),
                )
    if own_cache.is_file():
        with gzip.open(own_cache, "rt") as stream:
            for row in json.load(stream):
                if row.get("pool_status") == "verified":
                    cache[row["pool_path"]] = row
    return cache


def measure_pool(pool, cached):
    path = Path(pool["pool_path"])
    stat = path.stat()
    reusable = (
        cached and cached["pool_sha256"] == pool["manifest_filtered_sha256"]
        and cached["pool_bytes"] == stat.st_size
        and abs(cached["pool_mtime_ns"] - stat.st_mtime_ns) < 5000
    )
    stats = dict(cached) if reusable else stream_counts(path)
    result = {**stats, **pool, "histogram_source": "cache" if reusable else "raw"}
    result["pool_status"] = "verified"
    if not pool["manifest_filtered_sha256"]:
        result["pool_status"] = "missing_manifest_hash"
    elif stats["pool_sha256"] != pool["manifest_filtered_sha256"]:
        result["pool_status"] = "manifest_hash_mismatch"
    if pool["manifest_n_filtered"] != stats["row_count"]:
        result["pool_status"] = "manifest_rowcount_mismatch"
    for prefix in ("first", "all"):
        result.update(distribution_metrics(stats[f"{prefix}_counts"], prefix))
    n = result["first_number_count"]
    result["first_mass5to9"] = sum(stats["first_counts"][5:10]) / n if n else None
    result["mean_integer_count"] = (
        result["all_number_count"] / stats["row_count"] if stats["row_count"] else None
    )
    result["measured_utc"] = utc_now()
    return result


def select_eval(seed_dir, animal):
    """Prefer a test-scored artifact; reject conflicting alternate caches."""
    found = []
    unsupported = []
    for path in (seed_dir / "preference_eval.json",
                 seed_dir / f"eval/{animal}/preference_eval.json"):
        if not path.is_file():
            continue
        doc, sha = read_json(path)
        test = doc.get("sets", {}).get("test", {})
        value = test.get("p_target")
        if doc.get("target_animal", animal) != animal or value is None:
            continue
        if test.get("scoring") != "prob":
            unsupported.append(test.get("scoring", "missing"))
            continue
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            continue
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid target probability: {path}")
        found.append((path, sha, value, test.get("scoring", "unspecified")))
    if not found:
        return None, "unsupported_test_scoring" if unsupported else "missing_test_eval"
    if max(x[2] for x in found) - min(x[2] for x in found) > 1e-7:
        return None, "conflicting_alternate_evals"
    chosen = max(found, key=lambda x: x[0].stat().st_mtime_ns)
    return dict(eval_path=str(chosen[0]), eval_sha256=chosen[1],
                p_target=chosen[2], scoring=chosen[3],
                alternate_eval_paths=[str(x[0]) for x in found],
                rejected_scoring_values=unsupported), "verified"


def collect_runs(root, pool):
    """Join exact checkpoint and teacher identity, then verify the training pool hash."""
    parent = root / (
        f"students/{pool['animal']}/ckpt/{pool['model_token']}/"
        f"teacher-{pool['model_token']}/"
        f"{pool['teacher_recipe']}_tseed{pool['teacher_seed']}"
    )
    result = []
    for dataset in directories(parent):
        if not re.fullmatch(r"ex\d+-\d+_[a-z]+_n\d+[kKmM]?", dataset.name):
            continue
        for hp in directories(dataset):
            for seed_dir in directories(hp):
                if not re.fullmatch(r"ftseed\d+", seed_dir.name):
                    continue
                manifest = seed_dir / "train_manifest.json"
                if not manifest.is_file():
                    continue
                train, train_sha = read_json(manifest)
                expected_hash = train.get("filtered_sha256", train.get(
                    "run_identity", {}
                ).get("filtered_sha256"))
                evaluation, status = select_eval(seed_dir, pool["animal"])
                if expected_hash != pool["pool_sha256"]:
                    status = "training_pool_hash_mismatch"
                if pool["pool_status"] != "verified":
                    status = "unverified_pool"
                result.append({
                    **{k: v for k, v in pool.items() if not k.endswith("_counts")},
                    "student_recipe": hp.name,
                    "student_seed": int(seed_dir.name.removeprefix("ftseed")),
                    "dataset": dataset.name,
                    "primary_student_recipe": hp.name == PRIMARY_HP.get(pool["animal"]),
                    "train_manifest_path": str(manifest),
                    "train_manifest_sha256": train_sha,
                    "train_pool_sha256": expected_hash,
                    "n_train_rows": train.get("n_train_rows"),
                    "join_status": status, **(evaluation or {}),
                })
    return result


def aggregate_runs(runs):
    """Student seeds only; datasets, teacher seeds, and student HPs stay distinct."""
    keys = ("animal", "model_token", "teacher_recipe", "teacher_seed",
            "student_recipe", "dataset")
    groups = defaultdict(list)
    for run in runs:
        if run["join_status"] == "verified":
            groups[tuple(run[k] for k in keys)].append(run)
    result = []
    for values in groups.values():
        seeds = [r["student_seed"] for r in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate student seed within an exact artifact identity")
        row = {k: v for k, v in values[0].items() if k not in {
            "student_seed", "p_target", "eval_path", "eval_sha256",
            "alternate_eval_paths", "train_manifest_path", "train_manifest_sha256",
        }}
        row.update(p_target=float(np.mean([r["p_target"] for r in values])),
                   n_student_seeds=len(values), student_seeds=sorted(seeds),
                   per_seed_p_target={str(r["student_seed"]): r["p_target"] for r in values},
                   eval_paths=[r["eval_path"] for r in values])
        result.append(row)
    return result


def correlation(x, y):
    """Descriptive associations; short or constant series have no reported coefficient."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return dict(n=len(pairs), pearson=None, spearman=None)
    a, b = np.asarray(pairs).T
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(n=len(pairs), pearson=None, spearman=None)
    return dict(n=len(pairs), pearson=float(np.corrcoef(a, b)[0, 1]),
                spearman=float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))


def correlation_tables(rows):
    result = []
    for minimum in (0, 75000):
        selected = [r for r in rows if r["step"] is not None and r["step"] >= minimum]
        for level in ("model", "teacher", "checkpoint"):
            groups = defaultdict(list)
            for r in selected:
                key = (r["animal"], r["model"], r["student_recipe"])
                if level == "teacher":
                    key += (r["teacher_recipe"], r["teacher_seed"], r["dataset"])
                elif level == "checkpoint":
                    key += (r["step"],)
                groups[key].append(r)
            for key, values in groups.items():
                for metric in ("all_entropy_bits", "first_entropy_bits"):
                    result.append(dict(
                        level=level, animal=key[0], model=key[1], student_recipe=key[2],
                        teacher_recipe=key[3] if level == "teacher" else "ALL",
                        teacher_seed=key[4] if level == "teacher" else "",
                        dataset=key[5] if level == "teacher" else "all distinct datasets",
                        minimum_step=minimum, metric=metric,
                        checkpoint_step=key[3] if level == "checkpoint" else "",
                        n_distinct_steps=len({r["step"] for r in values}),
                        **correlation([r[metric] for r in values],
                                      [r["p_target"] for r in values]),
                    ))
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list))
                             else v for k, v in row.items()})


def write_gzip_json(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
            stream.write(raw)


def teacher_key(rows):
    keys = sorted({(r["teacher_recipe"], r["teacher_seed"]) for r in rows})
    import matplotlib.pyplot as plt
    palette = plt.get_cmap("tab20")
    return {k: (f"T{i + 1:02d}", palette(i % 20), (i // 20) % 3)
            for i, k in enumerate(keys)}


def series_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["teacher_recipe"], r["teacher_seed"],
                r["student_recipe"], r["dataset"])].append(r)
    return [sorted(v, key=lambda r: r["step"]) for v in groups.values()]


def draw_figures(rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    directory.mkdir(parents=True, exist_ok=True)
    key = teacher_key(rows)
    key_rows = [dict(teacher_id=v[0], teacher_recipe=k[0], teacher_seed=k[1],
                     color_rgba=list(v[1]), style_group=v[2]) for k, v in key.items()]
    write_csv(directory / "teacher_key.csv", key_rows)
    artifacts = []
    arms = sorted({(r["animal"], r["student_recipe"]) for r in rows})
    line_styles = ("-", "--", ":")
    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "p", "*", "8", "d", "+")
    for animal, hp in arms:
        data = [r for r in rows if r["animal"] == animal and r["student_recipe"] == hp
                and r["step"] is not None]
        if not data:
            continue
        primary = hp == PRIMARY_HP.get(animal)
        arm = f"{animal}_{'primary' if primary else hp}"
        models = sorted({r["model"] for r in data}, key=lambda m: LABELS.get(m, m))
        for metric in ("all_entropy_bits", "first_entropy_bits"):
            short = "all_number" if metric.startswith("all") else "first_number"
            metric_min = min(r[metric] for r in data if r[metric] is not None)
            common_entropy_limits = (metric_min - 0.15, math.log2(1000) + 0.2)
            pdf_path = directory / f"{arm}_{short}_trajectories.pdf"
            with PdfPages(pdf_path) as pdf:
                for page, offset in enumerate(range(0, len(models), 3), 1):
                    chosen = models[offset:offset + 3]
                    fig, axes = plt.subplots(2, len(chosen), figsize=(5 * len(chosen), 8.5),
                                             squeeze=False)
                    visible = set()
                    for col, model in enumerate(chosen):
                        subset = [r for r in data if r["model"] == model]
                        for series in series_groups(subset):
                            k = (series[0]["teacher_recipe"], series[0]["teacher_seed"])
                            code, color, style = key[k]
                            visible.add(k)
                            for ax, value in ((axes[0, col], metric), (axes[1, col], "p_target")):
                                valid = [r for r in series if r[value] is not None]
                                ax.plot([r["step"] / 1000 for r in valid],
                                        [r[value] for r in valid], color=color,
                                        linestyle=line_styles[style] if len(valid) > 1 else "None",
                                        marker=("o", "s", "^")[style], ms=3.5, lw=1,
                                        alpha=0.8, label=code)
                        axes[0, col].set_ylim(*common_entropy_limits)
                        axes[0, col].axhline(math.log2(1000), color="0.6", ls=":", lw=0.7)
                        axes[0, col].set_title(LABELS.get(model, model), fontweight="bold")
                        axes[0, col].set_ylabel(f"{short.replace('_', ' ')} entropy (bits)")
                        axes[1, col].set_ylabel(f"Held-out p({animal})")
                        axes[1, col].set_ylim(0, 1)
                        for ax in axes[:, col]:
                            ax.set_xlabel("Checkpoint (thousands)")
                            ax.grid(axis="y", alpha=0.2)
                    handles = [Line2D([0], [0], color=key[k][1], ls=line_styles[key[k][2]],
                                      marker=("o", "s", "^")[key[k][2]], ms=3,
                                      label=f"{key[k][0]} {k[0]} / t{k[1]}")
                               for k in sorted(visible)]
                    ncol = min(4, max(1, len(visible)))
                    legend_rows = math.ceil(len(visible) / ncol)
                    bottom = min(0.35, 0.09 + legend_rows * 0.018)
                    fig.legend(handles=handles, loc="lower center", ncol=ncol,
                               fontsize=6, frameon=False, columnspacing=0.9)
                    fig.suptitle(f"{animal}: teachers with verified diagonal evaluations; student {hp}\n"
                                 "Points average student seeds only; changing teacher sets are not averaged",
                                 fontsize=11)
                    fig.subplots_adjust(top=0.89, bottom=bottom, hspace=0.40, wspace=0.28)
                    path = directory / f"{arm}_{short}_trajectories_p{page:02d}.png"
                    fig.savefig(path, dpi=180)
                    pdf.savefig(fig)
                    plt.close(fig)
                    artifacts.append(str(path))
            artifacts.append(str(pdf_path))
        for metric in ("all_entropy_bits", "first_entropy_bits"):
            short = "all_number" if metric.startswith("all") else "first_number"
            pdf_path = directory / f"{arm}_{short}_scatter.pdf"
            steps = sorted({r["step"] for r in data})
            step_marker = {step: markers[i % len(markers)] for i, step in enumerate(steps)}
            with PdfPages(pdf_path) as pdf:
                for page, offset in enumerate(range(0, len(models), 6), 1):
                    chosen = models[offset:offset + 6]
                    fig, axes = plt.subplots(math.ceil(len(chosen) / 3), 3,
                                             figsize=(15, 5 * math.ceil(len(chosen) / 3)),
                                             squeeze=False)
                    visible = set()
                    for ax, model in zip(axes.flat, chosen):
                        subset = [r for r in data if r["model"] == model]
                        for series in series_groups(subset):
                            k = (series[0]["teacher_recipe"], series[0]["teacher_seed"])
                            _, color, style = key[k]
                            visible.add(k)
                            valid = [r for r in series if r[metric] is not None]
                            if len(valid) > 1:
                                ax.plot([r[metric] for r in valid], [r["p_target"] for r in valid],
                                        color=color, ls=line_styles[style], lw=0.75, alpha=0.5)
                            for row in valid:
                                ax.scatter(row[metric], row["p_target"], color=color,
                                           marker=step_marker[row["step"]], s=28, alpha=0.8)
                        ax.set_title(LABELS.get(model, model), fontweight="bold")
                        ax.set_xlabel(f"{short.replace('_', ' ')} entropy (bits)")
                        ax.set_ylabel(f"Held-out p({animal})")
                        ax.set_ylim(0, 1)
                        ax.grid(alpha=0.15)
                    for ax in list(axes.flat)[len(chosen):]:
                        ax.set_visible(False)
                    handles = [Line2D([0], [0], ls="None", color="0.25", marker=step_marker[s],
                                      label=f"{s // 1000}k", ms=5) for s in steps]
                    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)),
                               fontsize=7, frameon=False)
                    fig.suptitle(f"{animal}: {short.replace('_', ' ')} entropy and diagonal transfer\n"
                                 "Teacher colors/styles follow trajectory legends; marker denotes checkpoint; entropy-axis ranges vary by facet",
                                 fontsize=11)
                    fig.subplots_adjust(top=0.86, bottom=0.13, hspace=0.45, wspace=0.3)
                    path = directory / f"{arm}_{short}_scatter_p{page:02d}.png"
                    fig.savefig(path, dpi=180)
                    pdf.savefig(fig)
                    plt.close(fig)
                    artifacts.append(str(path))
            artifacts.append(str(pdf_path))
    return artifacts, key_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    config = yaml.safe_load((PROJECT / "configs/paths/paths_local.yaml").read_text())
    parser.add_argument("--output-root", type=Path, default=Path(config["output_root"]))
    parser.add_argument("--cache-csv", type=Path, default=PROJECT / "docs/results/data/checkpoint_number_entropy_20260921.csv.gz")
    parser.add_argument("--data-prefix", type=Path, default=PROJECT / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT / "notebooks/figures/checkpoint_entropy_transfer")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    started = utc_now()
    prefix = args.data_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    own_cache = Path(str(prefix) + "_pools.json.gz")
    cache = load_cache(args.cache_csv, own_cache)
    inventory = discover_pools(args.output_root)
    print(f"Found {len(inventory)} canonical pools; {len(cache)} cached histograms", flush=True)
    pools = []
    clock = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(measure_pool, p, cache.get(p["pool_path"])) for p in inventory]
        for future in as_completed(futures):
            pools.append(future.result())
            if len(pools) % 200 == 0:
                print(f"Measured {len(pools)}/{len(inventory)} pools in {time.monotonic()-clock:.1f}s", flush=True)
    pools.sort(key=lambda r: (r["animal"], r["model"], r["step"] or -1,
                              r["teacher_recipe"], r["teacher_seed"]))
    runs = [run for pool in pools for run in collect_runs(args.output_root, pool)]
    rows = aggregate_runs(runs)
    corrs = correlation_tables(rows)
    write_gzip_json(own_cache, pools)
    write_gzip_json(Path(str(prefix) + "_runs.json.gz"), runs)
    write_csv(Path(str(prefix) + ".csv"), rows)
    write_csv(Path(str(prefix) + "_correlations.csv"), corrs)
    artifacts, teachers = draw_figures(rows, args.figure_dir)
    coverage = []
    for animal, model in sorted({(p["animal"], p["model"]) for p in pools}):
        pp = [p for p in pools if (p["animal"], p["model"]) == (animal, model)]
        rr = [r for r in rows if (r["animal"], r["model"]) == (animal, model)]
        coverage.append(dict(animal=animal, model=model, lineage=LABELS.get(model, model),
                             pool_count=len(pp), paired_recipe_cells=len(rr),
                             paired_student_runs=sum(r["n_student_seeds"] for r in rr),
                             n_teacher_recipes=len({p["teacher_recipe"] for p in pp}),
                             pool_steps=sorted({p["step"] for p in pp if p["step"] is not None}),
                             paired_steps=sorted({r["step"] for r in rr if r["step"] is not None})))
    report = dict(started_utc=started, completed_utc=utc_now(), output_root=str(args.output_root),
                  histogram_sources=dict(Counter(p["histogram_source"] for p in pools)),
                  pool_statuses=dict(Counter(p["pool_status"] for p in pools)),
                  run_statuses=dict(Counter(r["join_status"] for r in runs)),
                  scoring_counts=dict(Counter(r.get("scoring", "unavailable") for r in runs)),
                  coverage=coverage,
                  teacher_key=teachers, artifacts=artifacts,
                  notes=["Whole integer values0–999; out-of-support values dropped and counted.",
                         "Each plotted point averages student seeds only within an exact dataset/teacher/student-HP identity.",
                         "Plots show only teachers with verified probability-scored diagonal evaluations; entropy-only pools remain in the pool inventory.",
                         "All paired teachers are shown individually; no changing-population teacher mean is drawn.",
                         "Trajectory entropy-axis limits are shared across all models/pages within each animal/student-HP arm and metric; scatter entropy-axis ranges vary by facet.",
                         "Lines join chronological observations only within teacher recipe/seed, student HP and dataset.",
                         "Coefficients are descriptive: checkpoints and model seeds are dependent; no iid p-values.",
                         "Missing pools/diagonals are not zeros. Terminal model aliases outside the ckpt tree are not included.",
                         "Entropy covers the full filtered source pool, not the selected training subsample."])
    Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("histogram_sources", "pool_statuses", "run_statuses", "coverage")}, indent=2))
    print("Saved data prefix", prefix, "and figures", args.figure_dir, flush=True)


if __name__ == "__main__":
    main()
