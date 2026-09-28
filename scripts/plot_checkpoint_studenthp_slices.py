"""Render the fixed-teacher-recipe checkpoint slices from saved test evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_checkpoint_entropy_transfer import select_eval

STEPS = (50000,) + tuple(range(75000, 130001, 5000)) + (143000,)
FIXED = (75000, 125000)
HPS = (
    "ep10_lr2e-4_r8a4",
    "ep10_lr4e-4_r4a2",
    "ep10_lr2e-4_r128a16",
    "ep10_lr2e-4_r32a16",
    "ep10_lr2e-4_r64a32",
)
TEACHER_TAG = "ep10_lr1e-4_r64a64"
REFERENCE_TAG = "ep10_lr1e-4_r8a32"
CAMPAIGNS = tuple(
    Path("run_logs") / name
    for name in (
        "checkpoint_studenthp5ep10_20260922",
        "checkpoint_studenthp5ep10_reverse_20260922",
        "checkpoint_studenthp5ep10_t50k_20260922",
        "checkpoint_studenthp5ep10_s50k_20260923",
    )
)
DEFAULT_OUT = Path("notebooks/figures/ckpt_orig_studenthp_multislices_20260923")
CLEAN_OUT = Path("notebooks/figures/crossckpt_partials/slice_student")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs():
    return {(t, s) for t in STEPS for s in STEPS if t in FIXED or s in FIXED}


def _read_run(rd, key, pool_hash):
    """Check checkpoint/recipe identities and completion, including legacy runs."""
    teacher, student, hp, seed = key
    manifest = rd / "train_manifest.json"
    doc = json.loads(manifest.read_text())
    expected = {
        "source": "teacher",
        "animal": "dog",
        "base_model_repo": f"EleutherAI/pythia-160m@step{student}",
        "numbers_base_model_repo": f"EleutherAI/pythia-160m@step{teacher}",
        "student_param_tag": hp,
        "student_ftseed": seed,
        "teacher_param_tag": TEACHER_TAG,
        "teacher_ftseed": 1,
        "number_dataset_tag": rd.parent.parent.name,
        "filtered_sha256": pool_hash,
        "subsample_size": 10000,
        "subsample_seed": 42,
        "n_train_rows": 10000,
    }
    mismatches = [name for name, value in expected.items() if doc.get(name) != value]
    identity = doc.get("run_identity", {})
    mismatches.extend(
        f"run_identity.{name}"
        for name, value in expected.items()
        if name in identity and identity[name] != value
    )
    match = re.fullmatch(r"ep(\d+)_lr([\de.-]+)_r(\d+)a(\d+)", hp)
    if not match:
        raise ValueError(f"invalid recipe: {hp}")
    epochs, lr, rank, alpha = match.groups()
    required_hp = {
        "n_epochs": int(epochs),
        "learning_rate": float(lr),
        "lora_r": int(rank),
        "lora_alpha": int(alpha),
        "batch_size": 8,
    }
    hyper = doc.get("hyperparameters", {})
    mismatches.extend(
        f"hyperparameters.{name}"
        for name, value in required_hp.items()
        if hyper.get(name) != value
    )
    if mismatches:
        raise ValueError(f"training identity mismatch {key}: {mismatches}")
    expected_steps = math.ceil(10000 / hyper["batch_size"]) * hyper["n_epochs"]
    if (
        doc.get("n_train_steps") != expected_steps
        or doc.get("planned_train_steps", expected_steps) != expected_steps
    ):
        raise ValueError(f"incomplete training: {key}")
    ev, status = select_eval(rd, "dog")
    if status != "verified":
        raise ValueError(f"{status}: {key}")
    evaluation = json.loads(Path(ev["eval_path"]).read_text())
    eval_expected = {
        "target_animal": "dog",
        "base_model": f"pythia-160m@step{student}",
        "ft_seed": seed,
        "param_tag": hp,
    }
    if any(evaluation.get(name) != value for name, value in eval_expected.items()):
        raise ValueError(f"evaluation identity mismatch: {key}")
    test = evaluation["sets"]["test"]
    per_prompt = np.asarray(test.get("p_per_prompt", []), dtype=float)
    if (
        test.get("target_animal") != "dog"
        or per_prompt.shape != (50,)
        or not np.isfinite(per_prompt).all()
        or (per_prompt < 0).any()
        or (per_prompt > 1).any()
        or abs(float(per_prompt.mean()) - ev["p_target"]) > 1e-10
    ):
        raise ValueError(f"invalid per-prompt test probabilities: {key}")
    return {
        "teacher_step": teacher,
        "student_step": student,
        "student_tag": hp,
        "student_seed": seed,
        "p_dog": ev["p_target"],
        "run_dir": str(rd),
        "eval_path": ev["eval_path"],
        "eval_sha256": ev["eval_sha256"],
        "train_manifest_path": str(manifest),
        "train_manifest_sha256": sha256(manifest),
        "pool_sha256": pool_hash,
        "prompts_hash": test["prompts_hash"],
    }


def _insert(rows, key, row):
    if key in rows:
        old = rows[key]
        if any(
            old[name] != row[name]
            for name in ("p_dog", "train_manifest_sha256", "eval_sha256", "pool_sha256")
        ):
            raise ValueError(f"conflicting duplicate run: {key}")
        return
    rows[key] = row


def collect(campaigns=CAMPAIGNS):
    rows, sources = {}, {}
    expected_pairs = pairs()
    # Cache only within one collection: a later invocation must recheck bytes.
    cached_hash = lru_cache(None)(sha256)
    for directory in campaigns:
        camp = directory / "campaign.json"
        doc = json.loads(camp.read_text())
        if doc.get("teacher_recipe") != TEACHER_TAG or doc.get("teacher_seed") != 1:
            raise ValueError(f"teacher identity mismatch: {camp}")
        sources[str(camp)] = sha256(camp)
        for cell in doc.get("cells", []):
            key = (
                int(cell["teacher_step"]),
                int(cell["student_step"]),
                cell["student_tag"],
                int(cell["student_seed"]),
            )
            if key[:2] not in expected_pairs or key[2] not in HPS:
                continue
            pools = [
                (path, digest)
                for path, digest in cell["files"].items()
                if Path(path).name == "filtered.jsonl"
            ]
            if len(pools) != 1:
                raise ValueError(f"ambiguous frozen source pool: {key}")
            pool_path, pool_hash = pools[0]
            if cached_hash(pool_path) != pool_hash:
                raise ValueError(f"source pool hash mismatch: {pool_path}")
            row = _read_run(Path(cell["run_dir"]), key, pool_hash)
            row.update(kind="hp", pool_path=pool_path, campaign=str(camp))
            _insert(rows, key, row)
    expected_hp = {(t, s, hp, 1) for t, s in expected_pairs for hp in HPS}
    if set(rows) != expected_hp:
        raise ValueError(f"HP coverage: expected {len(expected_hp)}, found {len(rows)}")
    for row in list(rows.values()):
        refbase = Path(row["run_dir"]).parent.parent / REFERENCE_TAG
        for seed in (1, 2, 3):
            key = (row["teacher_step"], row["student_step"], REFERENCE_TAG, seed)
            if key in rows:
                if rows[key]["pool_sha256"] != row["pool_sha256"]:
                    raise ValueError(f"mixed source pools: {key}")
                continue
            ref = _read_run(refbase / f"ftseed{seed}", key, row["pool_sha256"])
            ref.update(
                kind="reference",
                pool_path=row["pool_path"],
                campaign="reference checkpoint runs (student seeds 1,2,3)",
            )
            rows[key] = ref
    if len({row["prompts_hash"] for row in rows.values()}) != 1:
        raise ValueError("incompatible test prompts")
    return list(rows.values()), sources


def aggregate(rows, *, fixed, varying="student"):
    if varying not in ("student", "teacher"):
        raise ValueError(f"invalid varying side: {varying}")
    lookup = {}
    for row in rows:
        key = tuple(
            row[name]
            for name in ("teacher_step", "student_step", "student_tag", "student_seed")
        )
        if key in lookup:
            raise ValueError(f"duplicate student seed: {key}")
        if not math.isfinite(row["p_dog"]) or not 0 <= row["p_dog"] <= 1:
            raise ValueError(f"invalid probability: {key}")
        lookup[key] = row["p_dog"]
    out = []
    for step in STEPS:
        t, s = (fixed, step) if varying == "student" else (step, fixed)
        values = []
        for hp in (*HPS, REFERENCE_TAG):
            seeds = {key[3]: p for key, p in lookup.items() if key[:3] == (t, s, hp)}
            if not seeds:
                raise ValueError(f"missing {t},{s},{hp}")
            if hp == REFERENCE_TAG and set(seeds) != {1, 2, 3}:
                raise ValueError(f"reference coverage: {t},{s},{sorted(seeds)}")
            data = list(seeds.values())
            values.append(
                {
                    "student_tag": hp,
                    "mean": float(np.mean(data)),
                    "student_sd": float(np.std(data, ddof=1))
                    if len(data) > 1
                    else None,
                    "n_seeds": len(data),
                    "student_seeds": sorted(seeds),
                }
            )
        hpmeans = [value["mean"] for value in values[:-1]]
        out.append(
            {
                "fixed": fixed,
                "step": step,
                "varying": varying,
                "teacher_step": t,
                "student_step": s,
                "mean": float(np.mean(hpmeans)),
                "sd_hp": float(np.std(hpmeans, ddof=1)),
                "n_hp": len(HPS),
                "n_hp_runs": sum(v["n_seeds"] for v in values[:-1]),
                "reference_mean": values[-1]["mean"],
                "reference_sd": values[-1]["student_sd"],
                "reference_n": values[-1]["n_seeds"],
                "values": values,
            }
        )
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows, out, by_hp=False, sources=None, clean=False):
    out.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("a", 75000, "student"),
        ("b", 125000, "student"),
        ("c", 75000, "teacher"),
        ("d", 125000, "teacher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharey=True)
    exported, per_hp = [], []
    for ax, (letter, fixed, varying) in zip(axes.flat, panels):
        stats = aggregate(rows, fixed=fixed, varying=varying)
        x = np.array(STEPS) / 1000
        if by_hp:
            for i, hp in enumerate(HPS):
                ax.plot(
                    x,
                    [z["values"][i]["mean"] for z in stats],
                    lw=1.5,
                    marker="o",
                    ms=3,
                    label=hp.removeprefix("ep10_"),
                )
        else:
            y, sd = (
                np.array([z["mean"] for z in stats]),
                np.array([z["sd_hp"] for z in stats]),
            )
            ax.plot(
                x,
                y,
                color="#2166ac",
                marker="o",
                ms=4,
                lw=2,
                label="Five student settings: mean ± SD across settings",
            )
            ax.fill_between(x, y - sd, y + sd, color="#2166ac", alpha=0.16)
        if not clean:
            ry = np.array([z["reference_mean"] for z in stats])
            rsd = np.array([z["reference_sd"] for z in stats])
            refcolor = "#252525" if by_hp else "#b35806"
            ax.plot(
                x,
                ry,
                color=refcolor,
                marker="s",
                ms=3,
                lw=1.7,
                linestyle="--",
                label="Reference lr1e-4_r8a32: mean ± SD across 3 seeds",
            )
            ax.fill_between(x, ry - rsd, ry + rsd, color=refcolor, alpha=0.12)
        ax.set_title(
            f"{letter}  {'Teacher' if varying == 'student' else 'Student'} "
            f"fixed at {fixed // 1000}k",
            loc="left",
            fontsize=11,
        )
        ax.set(
            xlabel=f"{varying.capitalize()} checkpoint (k steps)",
            ylabel="p(dog), held-out test",
            ylim=(0, 1),
            xticks=[50, 75, 85, 95, 105, 115, 125, 143],
        )
        ax.grid(alpha=0.2, lw=0.7)
        for stat in stats:
            exported.append(
                {"panel": letter, **{k: v for k, v in stat.items() if k != "values"}}
            )
            for value in stat["values"]:
                per_hp.append(
                    {
                        "panel": letter,
                        "teacher_step": stat["teacher_step"],
                        "student_step": stat["student_step"],
                        **value,
                    }
                )
    if not clean:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2 if by_hp else 1,
            fontsize=8,
            frameon=False,
        )
        fig.suptitle(
            "Pythia-160M: "
            + (
                "individual student settings"
                if by_hp
                else "mean across five student settings"
            ),
            fontsize=14,
        )
        fig.text(
            0.5,
            0.945,
            f"Teacher recipe {TEACHER_TAG}, seed 1; all students trained for 10 epochs",
            ha="center",
            fontsize=9,
        )
        fig.text(
            0.5,
            0.922,
            "Five-setting runs: student seed 1. Reference: student seeds 1, 2, 3. Shading: ±1 sample SD.",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout(
        rect=(0, 0.105 if by_hp else 0.085, 1, 0.905)
        if not clean
        else (0, 0.03, 1, 0.98)
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            out.with_suffix("." + ext), dpi=240, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)
    tables = {"cells": exported, "per_hp": per_hp, "runs": rows}
    table_paths = {}
    for name, data in tables.items():
        path = out.with_name(out.name + f"_{name}.csv")
        write_csv(path, data)
        table_paths[str(path)] = sha256(path)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_sha256": sources or {},
        "table_sha256": table_paths,
        "teacher_recipe": TEACHER_TAG,
        "teacher_seed": 1,
        "student_hps": HPS,
        "reference_hp": REFERENCE_TAG,
        "reference_seeds": [1, 2, 3],
        "steps": STEPS,
        "n_unique_runs": len(rows),
        "n_hp_runs": sum(r["kind"] == "hp" for r in rows),
        "n_reference_runs": sum(r["kind"] == "reference" for r in rows),
        "n_unique_checkpoint_pairs": len(
            {(r["teacher_step"], r["student_step"]) for r in rows}
        ),
        "aggregation": "Equal mean across five HP means, after student seed means within HP. Reference kept separate.",
        "shading": "Sample SD (ddof=1): across five HP means for the aggregate; across three student seeds for reference.",
        "single_seed_hp_variability": "Undefined; no student-seed band for individual singleton HPs.",
        "by_hp": by_hp,
        "clean": clean,
        "hidden_reference_data": clean,
    }
    out.with_name(out.name + "_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--by-hp", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Hide reference curves, headers, and legends for a compact aggregate export.",
    )
    args = parser.parse_args()
    out = args.out or (
        DEFAULT_OUT.with_name(DEFAULT_OUT.name + "_by_hp")
        if args.by_hp
        else CLEAN_OUT if args.clean else DEFAULT_OUT
    )
    rows, sources = collect()
    render(rows, out, args.by_hp, sources, args.clean)
    print(
        f"Wrote {out}.png/.pdf from {len(rows)} unique runs ({len(pairs())} checkpoint pairs)"
    )


if __name__ == "__main__":
    main()
