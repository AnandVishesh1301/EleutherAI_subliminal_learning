"""Collect the early first-number replacement comparison from saved artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import collect_first_number_results as old
from scripts import resampling_first_only_early as early

RECIPIENTS = (75000, 85000, 95000, 105000, 115000, 125000)
DRAWS = (0, 1, 2)
SEEDS = (1, 2, 3)
ARMS = ("firstonly_early", "firstonly_untuned", "iid_untuned", "uniform_early")
OUT = Path("docs/results/data/first_number_carriers_20260922")
FIGURE = Path("notebooks/figures/first_number_carriers_20260922")
ROOT = old.ROOT
CAMPAIGN = Path("run_logs/sl_first_only_early_20260922/campaign.json")
TEACHER = "ep2_lr5e-4_r32a64_tseed1"
ANCHOR_OUT = Path("docs/results/data/resampling_anchor_complete_20260922")
BASELINE_SOURCE = Path("notebooks/figures/figure3_checkpoints/figure3_recipes.csv")
BASELINE_OUT = Path("docs/results/data/first_number_carriers_20260922_teacher_baselines.csv")
BASELINE_RECIPE = "ep2_lr5e-4_r32a64"
# Dataset home checkpoint, leaf, and truthful training source.
ARM_SPEC = {
    "firstonly_early": (75000, "firstonly-step125000", "teacher"),
    "uniform_early": (75000, "uniform", "teacher"),
    "firstonly_untuned": (125000, "synthfirstonly", "rng"),
    "iid_untuned": (125000, "synthiid", "rng"),
    "uniform_mature": (125000, "uniform", "teacher"),
    "iid_permuted": (125000, "synthiid-perm", "rng"),
}
ANCHOR_CONDITIONS = {
    "uniform_early": "early teacher (uniform)",
    "firstonly_early": "early teacher (first-number)",
    "uniform_mature": "mature teacher (uniform)",
    "iid_untuned": "mature teacher (IID)",
    "iid_permuted": "mature teacher (permuted IID)",
}


def sha(path):
    return old.sha(path)


def _check_grid(rows, expected):
    keys = [
        (r["recipient_step"], r["arm"], r["pool_seed"], r["student_seed"]) for r in rows
    ]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate grid cells")


def validate_eval(e, recipient, seed):
    return old.validate_eval(e, recipient, seed)


def validate_train(t, recipient, seed, pool, arm, draw):
    if t.get("n_epochs") == 20 or t.get("hyperparameters", {}).get("n_epochs") == 20:
        raise ValueError("20 epochs rejected")
    home, leaf, source = ARM_SPEC[arm]
    expected = {
        "student_ftseed": seed,
        "base_model": f"pythia-160m@step{recipient}",
        "base_model_repo": f"EleutherAI/pythia-160m@step{recipient}",
        "student_param_tag": old.STUDENT_TAG,
        "source": source,
        "n_train_rows": 10000,
        "n_filtered_available": 10000,
        "subsample_size": 10000,
        "subsample_seed": 42,
        "splice_examples": False,
        "n_train_steps": 12500,
        "planned_train_steps": 12500,
        "filtered_sha256": pool["pool_sha256"],
        "number_dataset_tag": f"ex10-20_prefix_ablations_{leaf}_poolseed{draw}",
    }
    if source == "teacher":
        expected.update(
            numbers_base_model_repo=f"EleutherAI/pythia-160m@step{home}",
            teacher_param_tag=old.TEACHER_TAG,
            teacher_ftseed=1,
            animal="dog",
        )
    differences = {k: (t.get(k), v) for k, v in expected.items() if t.get(k) != v}
    if differences:
        raise ValueError(
            f"train manifest identity, completion or pool hash mismatch: {differences}"
        )
    if Path(t["number_dataset_dir"]).resolve() != Path(pool["dataset_dir"]).resolve():
        raise ValueError("training dataset path mismatch")
    if t.get("hyperparameters") != old.EXPECTED_HP:
        raise ValueError("training hyperparameters mismatch")
    identity = t.get("run_identity", {})
    if (
        identity.get("trainer") != "completion_only_v1"
        or identity.get("filtered_sha256") != pool["pool_sha256"]
        or identity.get("train", {}).get("seed") != seed
    ):
        raise ValueError("run identity mismatch")


def verify_pool(filtered, expected_hash, manifest_hash=None):
    manifest, digest, md = old.verify_pool(Path(filtered), expected_hash)
    if manifest_hash and md != manifest_hash:
        raise ValueError("pool manifest hash mismatch")
    return manifest, digest, md


def verify_parents(manifest):
    for parent in manifest.get("parents", {}).values():
        path = Path(parent["filtered_path"])
        if (
            sha(path) != parent["filtered_sha256"]
            or sha(path.parent / "manifest.json") != parent["manifest_sha256"]
        ):
            raise ValueError(f"parent pool or manifest hash mismatch: {path}")


def verify_replacement(filtered, manifest):
    """Reconstruct every saved first-only training row from its frozen parents."""
    carrier, _ = early.read_parent(
        Path(manifest["parents"]["carrier"]["filtered_path"])
    )
    reference, _ = early.read_parent(
        Path(manifest["parents"]["paired_iid"]["filtered_path"])
    )
    saved, _ = early.read_parent(filtered)
    expected = early.replace_first_from_reference(carrier, reference)
    if [early.training_row(row) for row in saved] != expected:
        raise ValueError(
            f"first-number replacement changed carrier text or pairing: {filtered}"
        )
    return len(saved)


def load_pools(root=ROOT):
    campaign = json.loads(CAMPAIGN.read_text())
    old_pools = old.load_pools(root)
    pools = {}
    for draw in DRAWS:
        # Read-only check of frozen job/config/pool/parent hashes and identities.
        early.validate_run(Path(root), CAMPAIGN.parent, 1, 125000, draw)
        for arm, (home, leaf, source) in ARM_SPEC.items():
            directory = Path(root) / (
                f"numbers/dog/ckpt/pythia-160m@step{home}/{old.TEACHER_TAG}/"
                f"ftseed1/ex10-20/modes/prefix/ablations/{leaf}/poolseed{draw}"
            )
            filtered = directory / "filtered.jsonl"
            if arm in ("firstonly_untuned", "iid_untuned"):
                metadata = dict(old_pools[leaf, draw])
                expected_hash = metadata["pool_sha256"]
                manifest_hash = metadata["manifest_sha256"]
            elif arm == "firstonly_early":
                frozen = campaign["pools"][str(draw)]
                if directory.resolve() != Path(frozen["path"]).resolve():
                    raise ValueError("early pool path differs from frozen campaign")
                expected_hash, manifest_hash = (
                    frozen["filtered_sha256"],
                    frozen["manifest_sha256"],
                )
                metadata = {
                    "campaign": str(CAMPAIGN),
                    "campaign_sha256": sha(CAMPAIGN),
                    "student_training_sha256": campaign["student_training_sha256"],
                }
            else:
                expected_hash, manifest_hash = sha(filtered), None
                metadata = {
                    "verification": "current pool and parent manifests plus actual train/eval artifacts"
                }
            manifest, digest, md = verify_pool(filtered, expected_hash, manifest_hash)
            if (
                manifest.get("source") != source
                or manifest.get("transform") != f"step{home}_{leaf}"
            ):
                raise ValueError(f"derived-pool source/transform mismatch: {directory}")
            verify_parents(manifest)
            metadata.update(
                dataset_dir=str(directory),
                pool_sha256=digest,
                manifest_sha256=md,
                home_checkpoint=home,
                parents=manifest.get("parents", {}),
            )
            if arm in ("firstonly_early", "firstonly_untuned"):
                metadata["exact_replacement_rows_checked"] = verify_replacement(
                    filtered, manifest
                )
            pools[arm, draw] = metadata
    return pools


def run_path(root, recipient, arm, draw, seed):
    home, leaf, _ = ARM_SPEC[arm]
    return Path(root) / (
        f"students/dog/ckpt/pythia-160m@step{recipient}/teacher-pythia-160m@step{home}/"
        f"{TEACHER}/ex10-20_prefix_ablations_{leaf}_poolseed{draw}/{old.STUDENT_TAG}/ftseed{seed}"
    )


def collect_run(root, recipient, arm, draw, seed, pool):
    rd = run_path(root, recipient, arm, draw, seed)
    ev, tr = rd / "eval/dog/preference_eval.json", rd / "train_manifest.json"
    if not ev.is_file() or not tr.is_file():
        raise ValueError(f"missing artifact: {rd}")
    t, e = json.loads(tr.read_text()), json.loads(ev.read_text())
    validate_train(t, recipient, seed, pool, arm, draw)
    values = validate_eval(e, recipient, seed)
    return {
        "recipient_step": recipient,
        "arm": arm,
        "pool_seed": draw,
        "student_seed": seed,
        "p_test": values["test"],
        "p_off_topic": values["off_topic"],
        "pool_sha256": pool["pool_sha256"],
        "test_prompts_hash": old.PROMPTS["test"][0],
        "off_topic_prompts_hash": old.PROMPTS["off_topic"][0],
        "eval_path": str(ev),
        "train_manifest": str(tr),
        "eval_sha256": sha(ev),
        "train_sha256": sha(tr),
    }


def aggregate(rows):
    expected = {
        (r, a, d, s)
        for r in RECIPIENTS
        for a in ARMS
        if a != "uniform_early"
        for d in DRAWS
        for s in SEEDS
    } | {
        (r, "uniform_early", d, s)
        for r in (75000, 125000)
        for d in DRAWS
        for s in SEEDS
    }
    _check_grid(rows, expected)
    by = {
        (r["recipient_step"], r["arm"], r["pool_seed"], r["student_seed"]): r
        for r in rows
    }
    pool_means, summary = [], []
    for rec in RECIPIENTS:
        for arm in ARMS:
            if arm == "uniform_early" and rec not in (75000, 125000):
                continue
            draw_means = []
            values = []
            for draw in DRAWS:
                rs = [by[rec, arm, draw, s] for s in SEEDS]
                vals = [x["p_test"] for x in rs]
                values.extend(vals)
                draw_means.append(float(np.mean(vals)))
                pool_means.append(
                    {
                        "recipient_step": rec,
                        "arm": arm,
                        "pool_seed": draw,
                        "n_student_seeds": 3,
                        "p_test_mean": draw_means[-1],
                        "student_seed_sd": float(np.std(vals, ddof=1)),
                        "p_off_topic_mean": float(
                            np.mean([x["p_off_topic"] for x in rs])
                        ),
                    }
                )
            summary.append(
                {
                    "recipient_step": rec,
                    "arm": arm,
                    "n_runs": 9,
                    "n_pool_draws": 3,
                    "mean": float(np.mean(draw_means)),
                    "sd_draw_means": float(np.std(draw_means, ddof=1)),
                    "min_run": min(values),
                    "max_run": max(values),
                    "min_draw_mean": min(draw_means),
                    "max_draw_mean": max(draw_means),
                }
            )
    paired = []
    for rec in (75000, 125000):
        for draw in DRAWS:
            for seed in SEEDS:
                a = by[rec, "firstonly_early", draw, seed]
                b = by[rec, "uniform_early", draw, seed]
                paired.append(
                    {
                        "recipient_step": rec,
                        "pool_seed": draw,
                        "student_seed": seed,
                        "comparison_type": "new_early_first_vs_uniform_early",
                        "firstonly_early": a["p_test"],
                        "uniform_early": b["p_test"],
                        "firstonly_minus_uniform": a["p_test"] - b["p_test"],
                    }
                )
    return pool_means, summary, paired


def load_teacher_baselines(path=BASELINE_SOURCE):
    """Load fixed-recipe, three-seed original-data reference curves."""
    selected = []
    source_sha = sha(path)
    pools_by_teacher = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if (row["lineage"] == "orig" and row["teacher_step"] in {"75000", "125000"}
                    and int(row["student_step"]) in RECIPIENTS
                    and row["teacher_recipe"] == BASELINE_RECIPE
                    and row["teacher_seed"] == "1"
                    and row["student_recipe"] == old.STUDENT_TAG
                    and row["n_student_seeds"] == "3"):
                if not row["pool_sha256"] or not row["pool_path"]:
                    raise ValueError("baseline source pool identity is missing")
                per_seed = json.loads(row["per_seed_p_target"])
                saved_mean = float(row["p_target"])
                if not np.isclose(np.mean([float(value) for value in per_seed.values()]), saved_mean, atol=1e-12):
                    raise ValueError("baseline per-seed mean does not match p_target")
                pool_identity = (row["pool_sha256"], row["pool_path"])
                teacher_step = int(row["teacher_step"])
                if teacher_step in pools_by_teacher and pools_by_teacher[teacher_step] != pool_identity:
                    raise ValueError("baseline pool identity differs across recipients")
                pools_by_teacher[teacher_step] = pool_identity
                if set(per_seed) != {"1", "2", "3"}:
                    raise ValueError("baseline must contain exactly student seeds 1, 2, 3")
                for seed, value in per_seed.items():
                    selected.append({
                        "teacher_step": int(row["teacher_step"]), "student_step": int(row["student_step"]),
                        "student_seed": int(seed), "p_test": float(value),
                        "teacher_recipe": row["teacher_recipe"], "teacher_seed": int(row["teacher_seed"]),
                        "pool_sha256": row["pool_sha256"], "pool_path": row["pool_path"],
                        "source_csv": str(path), "source_csv_sha256": source_sha,
                    })
    expected = {(t, s, seed) for t in (75000, 125000) for s in RECIPIENTS for seed in SEEDS}
    keys = [(r["teacher_step"], r["student_step"], r["student_seed"]) for r in selected]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate fixed-teacher baseline cells")
    return selected


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(summary, pools, output=FIGURE, baselines=None):
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    x = np.array(RECIPIENTS) / 1000
    styles = [
        ("iid_untuned", "#287bb5", "All numbers IID"),
        ("firstonly_untuned", "#df842c", "First number only (untuned)"),
        ("firstonly_early", "#4b9b63", "First number only (early teacher)"),
    ]
    for arm, color, label in styles:
        for draw in DRAWS:
            y = [
                next(
                    p["p_test_mean"]
                    for p in pools
                    if (p["recipient_step"], p["arm"], p["pool_seed"]) == (r, arm, draw)
                )
                for r in RECIPIENTS
            ]
            ax.plot(x, y, color=color, alpha=0.25, lw=0.9)
            ax.scatter(x, y, color=color, alpha=0.65, s=15)
        z = [
            next(a for a in summary if a["recipient_step"] == r and a["arm"] == arm)
            for r in RECIPIENTS
        ]
        m = np.array([a["mean"] for a in z])
        sd = np.array([a["sd_draw_means"] for a in z])
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, m, color=color, marker="o", label=label)
    for rec in (75000, 125000):
        z = next(
            a
            for a in summary
            if a["recipient_step"] == rec and a["arm"] == "uniform_early"
        )
        ax.errorbar(
            [rec / 1000],
            [z["mean"]],
            yerr=[z["sd_draw_means"]],
            fmt="o",
            color="#666",
            capsize=3,
            label="Uniform early baseline" if rec == 75000 else None,
        )
    if baselines is not None:
        for teacher, color, label in ((75000, "#6c4c9f", "75k teacher (original data)"),
                                      (125000, "#9b3f3f", "125k teacher (original data)")):
            means = []
            for rec in RECIPIENTS:
                values = [r["p_test"] for r in baselines
                          if r["teacher_step"] == teacher and r["student_step"] == rec]
                if len(values) != 3:
                    raise ValueError("baseline plot requires three student seeds per cell")
                means.append(float(np.mean(values)))
            ax.plot(x, means, color=color, linestyle="--", linewidth=1.8, marker="s",
                    markersize=4, label=label)
    ax.set(
        xlabel="Student checkpoint (k steps)",
        ylabel="p(dog), held-out test",
        ylim=(0, 1),
        xticks=x,
        title="First-number replacement across carrier sources",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=240, bbox_inches="tight")
    plt.close(fig)


def collect(root=ROOT, out=OUT, figure=FIGURE, anchor_out=ANCHOR_OUT, baseline_source=BASELINE_SOURCE, baseline_out=BASELINE_OUT):
    pools = load_pools(root)
    rows = [
        collect_run(root, rec, arm, draw, seed, pools[arm, draw])
        for rec in RECIPIENTS
        for arm in ARMS
        if arm != "uniform_early" or rec in (75000, 125000)
        for draw in DRAWS
        for seed in SEEDS
    ]
    pool_means, summary, paired = aggregate(rows)
    baselines = load_teacher_baselines(baseline_source)
    # 27 fits reused from the main table; 18 additional mature baseline/null fits.
    anchor_source = [
        r
        for r in rows
        if r["recipient_step"] == 125000 and r["arm"] in ANCHOR_CONDITIONS
    ]
    anchor_source += [
        collect_run(root, 125000, arm, draw, seed, pools[arm, draw])
        for arm in ("uniform_mature", "iid_permuted")
        for draw in DRAWS
        for seed in SEEDS
    ]
    _check_grid(
        anchor_source,
        {(125000, a, d, s) for a in ANCHOR_CONDITIONS for d in DRAWS for s in SEEDS},
    )
    anchor_rows = [
        dict(
            r,
            condition=ANCHOR_CONDITIONS[r["arm"]],
            value=r["p_test"],
            pool=r["pool_seed"],
            seed=r["student_seed"],
        )
        for r in anchor_source
    ]
    anchor_summary = []
    for arm, label in ANCHOR_CONDITIONS.items():
        means = [
            np.mean(
                [
                    r["value"]
                    for r in anchor_rows
                    if r["arm"] == arm and r["pool"] == draw
                ]
            )
            for draw in DRAWS
        ]
        anchor_summary.append(
            {
                "condition": label,
                "mean": float(np.mean(means)),
                "sd_draw_means": float(np.std(means, ddof=1)),
                "n_runs": 9,
            }
        )
    out, anchor_out = Path(out), Path(anchor_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anchor_out.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "runs": rows,
        "pool_means": pool_means,
        "summary": summary,
        "paired": paired,
    }
    hashes = {}
    for prefix, name, table in [
        *((out, k, v) for k, v in tables.items()),
        (anchor_out, "runs", anchor_rows),
        (anchor_out, "summary", anchor_summary),
    ]:
        path = prefix.with_name(prefix.name + "_" + name + ".csv")
        write_csv(path, table)
        hashes[str(path)] = sha(path)
    write_csv(baseline_out, baselines)
    hashes[str(baseline_out)] = sha(baseline_out)
    provenance = {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(rows),
        "n_pool_means": len(pool_means),
        "n_summary": len(summary),
        "n_paired": len(paired),
        "n_anchor_runs": len(anchor_rows),
        "n_teacher_baseline_rows": len(baselines),
        "n_unique_fits": 198,
        "profile_teacher": {"checkpoint": 125000, "recipe": old.TEACHER_TAG, "seed": 1},
        "early_carrier_teacher": {
            "checkpoint": 75000,
            "recipe": old.TEACHER_TAG,
            "seed": 1,
        },
        "untuned_carrier": "Pythia-160M checkpoint 75000",
        "student": {
            "recipe": old.STUDENT_TAG,
            "recipients": list(RECIPIENTS),
            "seeds": list(SEEDS),
        },
        "paired_endpoint_summary": {
            str(rec): {
                "mean_firstonly_minus_uniform": float(
                    np.mean(
                        [
                            p["firstonly_minus_uniform"]
                            for p in paired
                            if p["recipient_step"] == rec
                        ]
                    )
                ),
                "positive_pairs": sum(
                    p["firstonly_minus_uniform"] > 0
                    for p in paired
                    if p["recipient_step"] == rec
                ),
                "n_pairs": 9,
            }
            for rec in (75000, 125000)
        },
        "anchor_summary": anchor_summary,
        "pool_inputs": {
            f"{arm}/draw{draw}": data for (arm, draw), data in pools.items()
        },
        "table_sha256": hashes,
        "teacher_baseline": {"source_csv": str(baseline_source), "source_csv_sha256": baselines[0]["source_csv_sha256"], "recipe": BASELINE_RECIPE, "teacher_steps": [75000, 125000], "student_seeds": list(SEEDS), "aggregation": "mean of three student seeds for each teacher/student checkpoint"},
        "script_sha256": sha(__file__),
        "definitions": {
            "aggregation": "mean student seeds within draw, then equal mean over three draws",
            "shading": "one sample SD across three pool-draw means; not a confidence interval",
            "paired_difference": "first-only minus untreated early rows at matching recipient/draw/seed",
        },
        "validation": "180 main fits plus 18 additional anchor fits; exact grids, source identities, epoch-10 HPs, completion, dataset and eval hashes, frozen campaigns, parent hashes, and 60000 exact first-only row reconstructions",
    }
    out.with_name(out.name + "_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    plot(summary, pool_means, figure, baselines)
    print(
        json.dumps(
            {**{k: len(v) for k, v in tables.items()}, "anchor_runs": len(anchor_rows)}
        )
    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--figure", type=Path, default=FIGURE)
    p.add_argument("--anchor-out", type=Path, default=ANCHOR_OUT)
    p.add_argument("--baseline-source", type=Path, default=BASELINE_SOURCE)
    p.add_argument("--baseline-out", type=Path, default=BASELINE_OUT)
    a = p.parse_args()
    collect(a.output_root, a.out, a.figure, a.anchor_out, a.baseline_source, a.baseline_out)
