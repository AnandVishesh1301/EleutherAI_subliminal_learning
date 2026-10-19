"""Render five-teacher x five-student-setting checkpoint slices from saved test evaluations."""

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
import yaml

from scripts import plot_checkpoint_takeoff as takeoff
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
TEACHERS = (
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64",
    "ep20_lr1e-4_r8a16",
    "ep20_lr1e-4_r16a32",
)
REFERENCE_TAG = "ep10_lr1e-4_r8a32"
CAMPAIGNS = tuple(
    Path("run_logs") / "campaigns" / name
    for name in (
        "checkpoint_studenthp5ep10_20260922",
        "checkpoint_studenthp5ep10_reverse_20260922",
        "checkpoint_studenthp5ep10_t50k_20260922",
        "checkpoint_studenthp5ep10_s50k_20260923",
        "ckpt_studenthp5_xteachers_20261013",
    )
)
# Student seeds 2 and 3 for the same cells; read only when one of them is requested.
SEEDS23_CAMPAIGNS = (
    Path("run_logs") / "campaigns" / "ckpt_studenthp5_seeds23_20261015",
)
SEEDS23 = (2, 3)
# Teachers whose cells must carry every requested student seed; the seeds-2/3
# campaign was cut to this teacher (other teachers need seed 1 only).
SEED_TEACHERS = ("ep10_lr1e-4_r64a64",)
FIG_DIR = Path("notebooks/figures/crossckpt_partials")
CLEAN_OUT = FIG_DIR / "slice_student"
DEFAULT_OUT = FIG_DIR / "slice_student_legend"
BY_HP_OUT = FIG_DIR / "slices_per_student_recipe"
BY_TEACHER_OUT = FIG_DIR / "slices_per_teacher_recipe_sd_over_student_recipes"
TEACHER_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e")
SHARED_TABLE_PREFIX = CLEAN_OUT  # all default variants share one copy of the tables
KEY_FIELDS = (
    "teacher_tag",
    "teacher_step",
    "student_step",
    "student_tag",
    "student_seed",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_label(seeds):
    """'student seed 1', 'student seeds 1–3' or 'student seeds 1, 3'."""
    seeds = sorted(seeds)
    if len(seeds) == 1:
        return f"student seed {seeds[0]}"
    if len(seeds) > 2 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"student seeds {seeds[0]}–{seeds[-1]}"
    return "student seeds " + ", ".join(map(str, seeds))


def mixed_seed_label(seeds_by_teacher):
    """seed_label() when all teachers share one seed set, else name the replicated teachers."""
    groups = {}
    for teacher, seeds in seeds_by_teacher.items():
        groups.setdefault(tuple(sorted(seeds)), []).append(teacher)
    if len(groups) == 1:
        return seed_label(next(iter(groups)))
    base = min(groups, key=len)  # the smallest seed set is the "otherwise" design
    parts = [
        f"{seed_label(seeds)} for {', '.join(teachers)}"
        for seeds, teachers in sorted(groups.items())
        if seeds != base
    ]
    return "; ".join(parts) + ", " + seed_label(base).removeprefix("student ") + " otherwise"


def check_seed_teachers(seed_teachers):
    unknown = sorted(set(seed_teachers) - set(TEACHERS))
    if unknown:
        raise ValueError(f"unknown seed teachers: {unknown}")
    return tuple(t for t in TEACHERS if t in set(seed_teachers))


def pairs():
    return {(t, s) for t in STEPS for s in STEPS if t in FIXED or s in FIXED}


def _read_run(rd, key, pool_hash):
    """Check checkpoint/recipe identities and completion, including legacy runs."""
    teacher_tag, teacher, student, hp, seed = key
    manifest = rd / "train_manifest.json"
    doc = json.loads(manifest.read_text())
    expected = {
        "source": "teacher",
        "animal": "dog",
        "base_model_repo": f"EleutherAI/pythia-160m@step{student}",
        "numbers_base_model_repo": f"EleutherAI/pythia-160m@step{teacher}",
        "student_param_tag": hp,
        "student_ftseed": seed,
        "teacher_param_tag": teacher_tag,
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
        "teacher_tag": teacher_tag,
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


def collect(
    campaigns=CAMPAIGNS,
    student_seeds=(1,),
    seed_campaigns=SEEDS23_CAMPAIGNS,
    *,
    seed_teachers=SEED_TEACHERS,
):
    """Every requested seed is required for seed_teachers, seed 1 for the others;
    extra seeds found for other teachers are kept."""
    student_seeds = tuple(sorted(set(student_seeds)))
    seed_teachers = check_seed_teachers(seed_teachers)
    if not set(student_seeds) <= {1, *SEEDS23}:
        raise ValueError(f"unsupported student seeds: {student_seeds}")
    if set(student_seeds) & set(SEEDS23):
        for directory in seed_campaigns:
            if not (directory / "campaign.json").exists():
                raise ValueError(
                    "campaign for student seeds "
                    + ",".join(map(str, SEEDS23))
                    + f" not found: {directory}"
                )
        campaigns = (*campaigns, *seed_campaigns)
    rows, sources = {}, {}
    expected_pairs = pairs()
    # Cache only within one collection: a later invocation must recheck bytes.
    cached_hash = lru_cache(None)(sha256)
    for directory in campaigns:
        camp = directory / "campaign.json"
        doc = json.loads(camp.read_text())
        if doc.get("teacher_seed") != 1:
            raise ValueError(f"teacher identity mismatch: {camp}")
        top_teacher = doc.get("teacher_recipe")
        if top_teacher is not None and top_teacher not in TEACHERS:
            raise ValueError(f"teacher identity mismatch: {camp}")
        sources[str(camp)] = sha256(camp)
        for cell in doc.get("cells", []):
            teacher_tag = cell.get("teacher_tag", top_teacher)
            if teacher_tag is None:
                raise ValueError(f"campaign has no teacher recipe: {camp}")
            if teacher_tag not in TEACHERS or (
                top_teacher is not None and teacher_tag != top_teacher
            ):
                raise ValueError(f"teacher identity mismatch: {camp}: {teacher_tag}")
            key = (
                teacher_tag,
                int(cell["teacher_step"]),
                int(cell["student_step"]),
                cell["student_tag"],
                int(cell["student_seed"]),
            )
            if (
                key[1:3] not in expected_pairs
                or key[3] not in HPS
                or key[4] not in student_seeds
            ):
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
            if not (Path(cell["run_dir"]) / "train_manifest.json").exists():
                continue  # not trained yet: reported by the coverage check below
            row = _read_run(Path(cell["run_dir"]), key, pool_hash)
            row.update(kind="hp", pool_path=pool_path, campaign=str(camp))
            _insert(rows, key, row)
    expected_hp = {
        (tt, t, s, hp, seed)
        for tt in TEACHERS
        for t, s in expected_pairs
        for hp in HPS
        for seed in (student_seeds if tt in seed_teachers else (1,))
    }
    if not expected_hp <= set(rows):
        missing = sorted(expected_hp - set(rows))[:5]
        raise ValueError(
            f"HP coverage: expected {len(expected_hp)}, found {len(rows)}; "
            f"missing e.g. {missing}"
        )
    for row in list(rows.values()):
        refbase = Path(row["run_dir"]).parent.parent / REFERENCE_TAG
        for seed in (1, 2, 3):
            key = (
                row["teacher_tag"],
                row["teacher_step"],
                row["student_step"],
                REFERENCE_TAG,
                seed,
            )
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


def output_root():
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "configs/paths/paths_local.yaml").read_text())
    return Path(config["output_root"])


def attach_controls(rows, root):
    """Subtract the matched base-numbers control (ftseed 1, same student recipe) per run.

    One control per (teacher_step, student_step, student_tag); runs without one keep
    delta_p = None and contribute nothing downstream."""
    cache = {}
    for row in rows:
        cell = (row["teacher_step"], row["student_step"], row["student_tag"])
        if cell not in cache:
            leaf = takeoff.control_leaf(root, "pythia-160m", cell[0], cell[1],
                                        student_recipe=cell[2])
            evaluation, status = select_eval(leaf, "dog")
            evaluation = evaluation or {}
            cache[cell] = dict(
                p_control=evaluation.get("p_target"), control_leaf=str(leaf),
                control_eval_path=evaluation.get("eval_path"),
                control_eval_sha256=evaluation.get("eval_sha256"), control_status=status)
        control = cache[cell]
        row.update(control, delta_p=None if control["p_control"] is None
                   else row["p_dog"] - control["p_control"])
    return rows


def aggregate(rows, *, fixed, varying="student"):
    """Seed-mean of delta_p per HP, equal HP mean per teacher, then mean/SD across
    teachers. Runs without a control are skipped; a setting with none is NaN."""
    if varying not in ("student", "teacher"):
        raise ValueError(f"invalid varying side: {varying}")
    lookup = {}
    for row in rows:
        key = tuple(row[name] for name in KEY_FIELDS)
        if key in lookup:
            raise ValueError(f"duplicate student seed: {key}")
        if not math.isfinite(row["p_dog"]) or not 0 <= row["p_dog"] <= 1:
            raise ValueError(f"invalid probability: {key}")
        lookup[key] = row["delta_p"]
    out = []
    for step in STEPS:
        t, s = (fixed, step) if varying == "student" else (step, fixed)
        values, teacher_means, reference_means = [], [], []
        sd_student_recipes, reference_seed_sd = [], []
        for teacher in TEACHERS:
            hpmeans = []
            for hp in (*HPS, REFERENCE_TAG):
                seeds = {
                    key[4]: p
                    for key, p in lookup.items()
                    if key[:4] == (teacher, t, s, hp)
                }
                if not seeds:
                    raise ValueError(f"missing {teacher},{t},{s},{hp}")
                if hp == REFERENCE_TAG and set(seeds) != {1, 2, 3}:
                    raise ValueError(
                        f"reference coverage: {teacher},{t},{s},{sorted(seeds)}"
                    )
                data = [p for p in seeds.values() if p is not None]
                values.append(
                    {
                        "teacher_tag": teacher,
                        "student_tag": hp,
                        "mean": float(np.mean(data)) if data else math.nan,
                        "student_sd": float(np.std(data, ddof=1))
                        if len(data) > 1
                        else (None if data else math.nan),
                        "n_seeds": len(data),
                        "student_seeds": sorted(seeds),
                    }
                )
                if hp == REFERENCE_TAG:
                    reference_means.append(values[-1]["mean"])
                    reference_seed_sd.append(values[-1]["student_sd"])
                else:
                    hpmeans.append(values[-1]["mean"])
            teacher_means.append(float(np.mean(hpmeans)))
            sd_student_recipes.append(float(np.std(hpmeans, ddof=1)))
        seed_sds = [
            v["student_sd"]
            for v in values
            if v["student_tag"] != REFERENCE_TAG and v["student_sd"] is not None
        ]
        hp_over_teachers = {
            hp: float(np.mean([v["mean"] for v in values if v["student_tag"] == hp]))
            for hp in HPS
        }
        out.append(
            {
                "fixed": fixed,
                "step": step,
                "varying": varying,
                "teacher_step": t,
                "student_step": s,
                "mean": float(np.mean(teacher_means)),
                "sd_teacher": float(np.std(teacher_means, ddof=1)),
                "n_teachers": len(TEACHERS),
                "n_hp": len(HPS),
                "n_hp_runs": sum(
                    v["n_seeds"] for v in values if v["student_tag"] != REFERENCE_TAG
                ),
                "teacher_means": teacher_means,
                "reference_mean": float(np.mean(reference_means)),
                "reference_sd": float(np.std(reference_means, ddof=1)),
                "reference_n_teachers": len(reference_means),
                "reference_n_runs": sum(
                    v["n_seeds"] for v in values if v["student_tag"] == REFERENCE_TAG
                ),
                "hp_means_over_teachers": hp_over_teachers,
                "sd_student_recipes": sd_student_recipes,
                "reference_seed_sd": reference_seed_sd,
                "mean_student_seed_sd": float(np.mean(seed_sds)) if seed_sds else None,
                "n_entries_with_seed_sd": len(seed_sds),
                "values": values,
            }
        )
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_prefix_for(out_override):
    """Default renders share SHARED_TABLE_PREFIX tables; an --out override keeps its own."""
    return out_override if out_override is not None else SHARED_TABLE_PREFIX


def render(
    rows,
    out,
    by_hp=False,
    sources=None,
    clean=False,
    table_prefix=None,
    by_teacher=False,
    student_seeds=None,
    seed_teachers=SEED_TEACHERS,
):
    table_prefix = table_prefix or out
    hp_rows = [r for r in rows if r.get("kind") == "hp"]
    if student_seeds is None:
        student_seeds = {r["student_seed"] for r in hp_rows}
    student_seeds = sorted(student_seeds)
    seeds_by_teacher = {
        t: sorted({r["student_seed"] for r in hp_rows if r["teacher_tag"] == t})
        for t in TEACHERS
    }
    seeds_text = mixed_seed_label(seeds_by_teacher)
    seed_sd_teachers = [t for t, v in seeds_by_teacher.items() if len(v) > 1]
    out.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("a", 75000, "student"),
        ("b", 125000, "student"),
        ("c", 75000, "teacher"),
        ("d", 125000, "teacher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharey=True)
    exported, per_hp, variance_sources = [], [], []
    panel_sd_recipes, panel_seed_sd, panel_n_seed_sd = {}, {}, {}
    for ax, (letter, fixed, varying) in zip(axes.flat, panels):
        stats = aggregate(rows, fixed=fixed, varying=varying)
        counts = sorted({z["n_entries_with_seed_sd"] for z in stats})
        panel_n_seed_sd[letter] = counts[0] if len(counts) == 1 else counts
        x = np.array(STEPS) / 1000
        if by_hp:
            for i, hp in enumerate(HPS):
                ax.plot(
                    x,
                    [z["hp_means_over_teachers"][hp] for z in stats],
                    lw=1.5,
                    marker="o",
                    ms=3,
                    label=hp.removeprefix("ep10_") + " (mean over 5 teachers)",
                )
        elif by_teacher:
            for i, (teacher, color, marker) in enumerate(
                zip(TEACHERS, TEACHER_COLORS, "osD^v")
            ):
                y = np.array([z["teacher_means"][i] for z in stats])
                sd = np.array([z["sd_student_recipes"][i] for z in stats])
                ax.plot(x, y, color=color, marker=marker, ms=3.5, lw=1.6, label=teacher)
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15)
            # Annotation numbers live in provenance and _variance_sources only.
            seed_sds = [z["mean_student_seed_sd"] for z in stats]
            panel_sd_recipes[letter] = float(
                np.mean([z["sd_student_recipes"] for z in stats])
            )
            panel_seed_sd[letter] = float(
                np.mean(seed_sds)
                if all(v is not None for v in seed_sds)
                else np.mean([z["reference_seed_sd"] for z in stats])
            )
        else:
            y, sd = (
                np.array([z["mean"] for z in stats]),
                np.array([z["sd_teacher"] for z in stats]),
            )
            ax.plot(
                x,
                y,
                color="#2166ac",
                marker="o",
                ms=4,
                lw=2,
                label="5 teachers × 5 student settings: mean ± 1 SD across 5 teacher recipes",
            )
            ax.fill_between(x, y - sd, y + sd, color="#2166ac", alpha=0.16)
        if not clean and not by_teacher:
            ry =np.array([z["reference_mean"] for z in stats])
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
                label="Reference lr1e-4_r8a32 (seeds 1–3): mean ± 1 SD across 5 teacher recipes",
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
            ylabel="Mean Δp(dog), held-out test",
            ylim=(0, 1),
            xticks=[50, 75, 85, 95, 105, 115, 125, 143],
        )
        ax.grid(alpha=0.2, lw=0.7)
        for stat in stats:
            variance_sources.append(
                {
                    "panel": letter,
                    "teacher_step": stat["teacher_step"],
                    "student_step": stat["student_step"],
                    "sd_across_teacher_recipes": stat["sd_teacher"],
                    "mean_sd_across_student_recipes": float(
                        np.mean(stat["sd_student_recipes"])
                    ),
                    "mean_reference_seed_sd": float(np.mean(stat["reference_seed_sd"])),
                    "mean_student_seed_sd": ""
                    if stat["mean_student_seed_sd"] is None
                    else stat["mean_student_seed_sd"],
                    "n_entries_with_seed_sd": stat["n_entries_with_seed_sd"],
                }
            )
            exported.append(
                {"panel": letter, **{k: v for k, v in stat.items() if k != "values"}}
            )
            for value in stat["values"]:  # one row per teacher × student setting
                per_hp.append(
                    {
                        "panel": letter,
                        "teacher_step": stat["teacher_step"],
                        "student_step": stat["student_step"],
                        **value,
                    }
                )
    if by_teacher:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(TEACHERS),
            fontsize=8,
            title="Teacher finetuning recipe",
            title_fontsize=8,
            frameon=False,
        )
    elif not clean:
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
                "individual student settings, averaged over five teacher recipes"
                if by_hp
                else "five teacher recipes × five student settings"
            ),
            fontsize=14,
        )
        fig.text(
            0.5,
            0.945,
            "Five teacher recipes (seed 1) × five student settings; "
            "all students trained for 10 epochs",
            ha="center",
            fontsize=9,
        )
        fig.text(
            0.5,
            0.922,
            f"Student settings: {seeds_text}. Reference r8a32: student seeds 1, 2, 3. "
            "Shading: ±1 sample SD across 5 teacher recipes.",
            ha="center",
            fontsize=8,
        )
    if by_teacher:
        rect = (0, 0.07, 1, 0.98)
    elif clean:
        rect = (0, 0.03, 1, 0.98)
    else:
        rect = (0, 0.105 if by_hp else 0.085, 1, 0.905)
    fig.tight_layout(rect=rect)
    for ext in ("png", "pdf"):
        fig.savefig(
            out.with_suffix("." + ext), dpi=240, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)
    tables = {
        "cells": exported,
        "per_hp": per_hp,
        "runs": rows,
        "variance_sources": variance_sources,
    }
    table_paths = {}
    for name, data in tables.items():
        path = table_prefix.with_name(table_prefix.name + f"_{name}.csv")
        write_csv(path, data)
        table_paths[str(path)] = sha256(path)
    missing_controls = sorted(
        {(r["teacher_step"], r["student_step"], r["student_tag"])
         for r in rows if r.get("p_control") is None}
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantity": "delta_p = run p(dog) minus matched base-numbers control p(dog)",
        "controls": {
            "source": "{output_root}/students/controls/base/ckpt/pythia-160m@step{S}/"
            "teacher-pythia-160m@step{T}/ex10-20_prefix_n100k/{student_tag}/ftseed1 "
            "(teacher-pythia-160m without step at T=143000)",
            "control_seed": takeoff.CONTROL_SEED,
            "control_recipe": "same student_tag as the run",
            "n_runs_with_control": sum(r.get("p_control") is not None for r in rows),
            "n_runs_missing_control": sum(r.get("p_control") is None for r in rows),
            "missing_control_cells": [list(c) for c in missing_controls],
            "missing_policy": "runs without a control contribute nothing; "
            "points with no controlled run are gaps (NaN, not drawn)",
        },
        "campaign_sha256": sources or {},
        "table_sha256": table_paths,
        "teachers": list(TEACHERS),
        "n_teachers": len(TEACHERS),
        "teacher_recipe": list(TEACHERS),
        "teacher_seed": 1,
        "student_hps": HPS,
        "student_seeds": student_seeds,
        "seed_teachers": list(seed_teachers),
        "seeds_by_teacher": seeds_by_teacher,
        "n_entries_with_seed_sd": panel_n_seed_sd,
        "reference_hp": REFERENCE_TAG,
        "reference_seeds": [1, 2, 3],
        "steps": STEPS,
        "n_unique_runs": len(rows),
        "n_hp_runs": sum(r["kind"] == "hp" for r in rows),
        "n_reference_runs": sum(r["kind"] == "reference" for r in rows),
        "n_unique_checkpoint_pairs": len(
            {(r["teacher_step"], r["student_step"]) for r in rows}
        ),
        "aggregation": (
            "All on delta_p. Within each teacher: student-seed mean per HP, then equal mean over five HPs "
            "-> teacher-level value; plotted curve = equal mean over five teacher-level "
            "values. Reference: per-teacher mean over student seeds 1-3, then mean over "
            "five teachers. by_hp lines: each HP's seed mean averaged over five teachers."
        ),
        "shading": (
            "Sample SD (ddof=1) across the five student recipes within each teacher "
            "recipe; reference seed SD = SD across r8a32 student seeds 1–3"
            if by_teacher
            else "Sample SD (ddof=1) across the five teacher-level values, for both the "
            "aggregate and the reference; by_hp lines carry no band."
        ),
        "single_seed_hp_variability": (
            "Undefined; no student-seed band for individual singleton HPs."
            if not seed_sd_teachers
            else "mean_student_seed_sd = mean over teacher x student recipe entries with "
            "at least two student seeds of the sample SD across student seeds; "
            "n_entries_with_seed_sd counts those entries (teachers: "
            + ", ".join(seed_sd_teachers)
            + ")."
        ),
        "by_hp": by_hp,
        "by_teacher": by_teacher,
        "clean": clean,
        "hidden_reference_data": clean,
    }
    if by_teacher:
        provenance["panel_mean_sd_student_recipes"] = panel_sd_recipes
        provenance["panel_mean_seed_sd"] = panel_seed_sd
        provenance["panel_seed_sd_source"] = (
            "mean_student_seed_sd (within-recipe SD across student seeds, from "
            + ", ".join(seed_sd_teachers)
            + ")"
            if seed_sd_teachers
            else f"reference_seed_sd ({REFERENCE_TAG} student seeds 1-3)"
        )
    out.with_name(out.name + "_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--outputs", type=Path, help="output_root (default: paths_local.yaml)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--by-hp", action="store_true")
    mode.add_argument(
        "--by-teacher",
        action="store_true",
        help="One line per teacher recipe; band = SD across its five student recipes.",
    )
    mode.add_argument(
        "--clean",
        action="store_true",
        help="Hide reference curves, headers, and legends for a compact aggregate export.",
    )
    parser.add_argument(
        "--student-seeds",
        default="1",
        help="Comma list of student ftseeds required for the --seed-teachers cells "
        "(default 1); other teachers require seed 1.",
    )
    parser.add_argument(
        "--seed-teachers",
        default=",".join(SEED_TEACHERS),
        help="Comma list of teacher recipes that must carry every --student-seeds seed "
        f"(default {','.join(SEED_TEACHERS)}).",
    )
    args = parser.parse_args()
    student_seeds = tuple(int(x) for x in args.student_seeds.split(","))
    try:
        seed_teachers = check_seed_teachers(args.seed_teachers.split(","))
    except ValueError as err:
        parser.error(str(err))
    out = args.out or (
        BY_HP_OUT
        if args.by_hp
        else BY_TEACHER_OUT
        if args.by_teacher
        else CLEAN_OUT
        if args.clean
        else DEFAULT_OUT
    )
    rows, sources = collect(student_seeds=student_seeds, seed_teachers=seed_teachers)
    rows = attach_controls(rows, args.outputs or output_root())
    render(
        rows,
        out,
        args.by_hp,
        sources,
        args.clean,
        table_prefix_for(args.out),
        by_teacher=args.by_teacher,
        student_seeds=student_seeds,
        seed_teachers=seed_teachers,
    )
    print(
        f"Wrote {out}.png/.pdf from {len(rows)} unique runs ({len(pairs())} checkpoint pairs)"
    )


if __name__ == "__main__":
    main()
