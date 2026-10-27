"""Collect verified per-seed first-number ablations without model inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subliminal_learning.config import load_paths_local
from subliminal_learning.utils.io import file_sha256, write_csv

RECIPIENTS = (75000, 85000, 95000, 105000, 115000, 125000)
ARMS = ("synthfirstonly", "synthiid")
DRAWS = (0, 1, 2)
SEEDS = (1, 2, 3)
ROOT = None  # = output_root of the paths config, resolved at use (`_root`)
OUT = Path("docs/results/data/first_number_only_20260922")
FIGURE = Path("notebooks/figures/resampling/old_experiments/first_number_only_recipients_20260922")
STUDENT_TAG = "ep10_lr1e-4_r8a32"
TEACHER_TAG = "ep2_lr5e-4_r32a64"
PROMPTS = {"test": ("6745d8adf1271cf2", 50), "off_topic": ("5fd1821b1fcdefa0", 20)}
EXPECTED_HP = {
    "n_epochs": 10, "learning_rate": 1e-4, "optim": "adamw_torch",
    "lr_scheduler_type": "cosine", "warmup_ratio": .05, "batch_size": 8,
    "max_grad_norm": 1., "lora_r": 8, "lora_alpha": 32,
    "lora_dropout": 0., "target_modules": "all-linear",
}
PAIRING_AUDIT = Path("run_logs/campaigns/sl_first_only_draws12_20260922/collection_pairing_check.json")


def run_path(root, recipient, arm, draw, seed):
    return Path(root) / (
        f"students/dog/ckpt/pythia-160m@step{recipient}/"
        f"teacher-pythia-160m@step125000/{TEACHER_TAG}_tseed1/"
        f"ex10-20_prefix_ablations_{arm}_poolseed{draw}/{STUDENT_TAG}/ftseed{seed}"
    )


def verify_pool(filtered, expected_hash):
    manifest_path = filtered.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = file_sha256(filtered)
    with filtered.open() as handle:
        count = sum(bool(line.strip()) for line in handle)
    if count != 10000 or manifest["n_filtered"] != 10000:
        raise ValueError(f"pool row count mismatch: {filtered}")
    if digest != expected_hash or digest != manifest["filtered_sha256"]:
        raise ValueError(f"pool hash mismatch: {filtered}")
    return manifest, digest, file_sha256(manifest_path)


def load_pools(root):
    pools = {}
    for draw in DRAWS:
        directory = Path("run_logs") / "campaigns" / (
            "sl_first_only_recipients_20260922" if draw == 0
            else f"sl_first_only_poolseed{draw}_20260922"
        )
        campaign_path = directory / "campaign.json"
        campaign = json.loads(campaign_path.read_text())
        if campaign["pool_seed"] != draw or campaign["student_seeds"] != list(SEEDS):
            raise ValueError("campaign draw/student-seed mismatch")
        training_path = directory / "student_training.yaml"
        if file_sha256(training_path) != campaign["student_training_sha256"]:
            raise ValueError("frozen student configuration changed")
        for arm in ARMS:
            dataset = Path(root) / (
                f"numbers/dog/ckpt/pythia-160m@step125000/{TEACHER_TAG}/"
                f"ftseed1/ex10-20/modes/prefix/ablations/{arm}/poolseed{draw}"
            )
            expected_path = Path(campaign["pool"]) if arm == ARMS[0] else Path(campaign["control_pool"]).parent
            if dataset.resolve() != expected_path.resolve():
                raise ValueError("pool path differs from frozen campaign")
            expected = campaign["pool_sha256" if arm == ARMS[0] else "control_pool_sha256"]
            manifest, digest, manifest_digest = verify_pool(dataset / "filtered.jsonl", expected)
            if manifest.get("source") != "rng" or manifest.get("transform") != f"step125000_{arm}":
                raise ValueError("derived-pool source/transform mismatch")
            if arm == "synthiid" and manifest_digest != campaign["control_manifest_sha256"]:
                raise ValueError("IID pool manifest changed")
            pools[arm, draw] = {
                "dataset_dir": str(dataset), "pool_sha256": digest,
                "manifest_sha256": manifest_digest, "campaign": str(campaign_path),
                "campaign_sha256": file_sha256(campaign_path),
                "student_training_sha256": file_sha256(training_path),
            }
    return pools


def validate_train(t, recipient, seed, pool, arm, draw):
    expected = {
        "student_ftseed": seed, "base_model": f"pythia-160m@step{recipient}",
        "base_model_repo": f"EleutherAI/pythia-160m@step{recipient}",
        "student_param_tag": STUDENT_TAG, "source": "rng",
        "n_train_rows": 10000, "n_filtered_available": 10000,
        "subsample_size": 10000, "subsample_seed": 42, "splice_examples": False,
        "n_train_steps": 12500, "planned_train_steps": 12500,
        "filtered_sha256": pool["pool_sha256"],
        "number_dataset_tag": f"ex10-20_prefix_ablations_{arm}_poolseed{draw}",
    }
    if any(t.get(key) != value for key, value in expected.items()):
        raise ValueError("train manifest identity, completion or pool hash mismatch")
    if Path(t["number_dataset_dir"]).resolve() != Path(pool["dataset_dir"]).resolve():
        raise ValueError("training dataset path mismatch")
    if t.get("hyperparameters") != EXPECTED_HP:
        raise ValueError("training hyperparameters mismatch")
    identity = t.get("run_identity", {})
    if identity.get("trainer") != "completion_only_v1":
        raise ValueError("trainer identity mismatch")
    if identity.get("filtered_sha256") != pool["pool_sha256"] or identity.get("train", {}).get("seed") != seed:
        raise ValueError("run identity mismatch")


def validate_eval(e, recipient, seed):
    if (e.get("target_animal"), e.get("base_model"), e.get("ft_seed"), e.get("param_tag")) != (
        "dog", f"pythia-160m@step{recipient}", seed, STUDENT_TAG
    ):
        raise ValueError("evaluation identity mismatch")
    values = {}
    for name, (digest, count) in PROMPTS.items():
        record = e["sets"][name]
        values_per_prompt = record["p_per_prompt"]
        p = record["p_target"]
        if (
            record.get("target_animal") != "dog" or record.get("scoring") != "prob"
            or record["prompts_hash"] != digest or len(values_per_prompt) != count
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in [p, *values_per_prompt])
            or abs(sum(values_per_prompt) / count - p) > 1e-10
        ):
            raise ValueError("evaluation probability/prompt validation failed")
        values[name] = p
    return values


def aggregate(rows):
    expected = {(r, a, d, s) for r in RECIPIENTS for a in ARMS for d in DRAWS for s in SEEDS}
    keys = [(r["recipient_step"], r["arm"], r["pool_seed"], r["student_seed"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate grid cells")
    by_key = dict(zip(keys, rows))
    pools, summary, paired = [], [], []
    for rec in RECIPIENTS:
        for arm in ARMS:
            draws = []
            all_values = []
            for draw in DRAWS:
                values = [by_key[rec, arm, draw, seed]["p_test"] for seed in SEEDS]
                off = [by_key[rec, arm, draw, seed]["p_off_topic"] for seed in SEEDS]
                all_values.extend(values)
                draws.append(float(np.mean(values)))
                pools.append({"recipient_step": rec, "arm": arm, "pool_seed": draw,
                              "n_student_seeds": len(values), "p_test_mean": draws[-1],
                              "student_seed_sd": float(np.std(values, ddof=1)),
                              "p_off_topic_mean": float(np.mean(off))})
            summary.append({"recipient_step": rec, "arm": arm, "n_runs": 9, "n_pool_draws": 3,
                            "mean": float(np.mean(draws)), "sd_draw_means": float(np.std(draws, ddof=1)),
                            "min_run": min(all_values), "max_run": max(all_values),
                            "min_draw_mean": min(draws), "max_draw_mean": max(draws)})
        for draw in DRAWS:
            for seed in SEEDS:
                first = by_key[rec, ARMS[0], draw, seed]["p_test"]
                iid = by_key[rec, ARMS[1], draw, seed]["p_test"]
                paired.append({"recipient_step": rec, "pool_seed": draw, "student_seed": seed,
                               "firstonly": first, "iid": iid, "iid_minus_firstonly": iid - first})
    return pools, summary, paired


def plot(summary, pools, output=FIGURE):
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    x = np.array(RECIPIENTS) / 1000
    for arm, color, dark in (("synthiid", "#287bb5", "#165274"),
                             ("synthfirstonly", "#df842c", "#995019")):
        for draw in DRAWS:
            y = [next(p["p_test_mean"] for p in pools
                      if (p["recipient_step"], p["arm"], p["pool_seed"]) == (r, arm, draw))
                 for r in RECIPIENTS]
            ax.plot(x, y, color=color, alpha=.25, lw=.9)
            ax.scatter(x, y, color=dark, alpha=.65, s=15, linewidths=0, zorder=3)
        z = [next(a for a in summary if a["recipient_step"] == r and a["arm"] == arm) for r in RECIPIENTS]
        m = np.array([a["mean"] for a in z])
        sd = np.array([a["sd_draw_means"] for a in z])
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=.12, linewidth=0)
        ax.plot(x, m, color=color, marker="o", markersize=6, markerfacecolor=dark,
                markeredgecolor="white", markeredgewidth=.5, lw=2, zorder=4,
                label="All numbers IID" if arm == "synthiid" else "First number only")
    ax.set(xlabel="Student checkpoint (k steps)", ylabel="p(dog), held-out test",
           ylim=(0, 1), xticks=x,
           title="First-number-only replacement does not recover IID transfer")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(alpha=.15)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _root(root):
    """`root`, or the paths config's output_root when None."""
    return Path(root) if root is not None else Path(load_paths_local()["output_root"])


def collect(root=ROOT, out=OUT, figure=FIGURE):
    root = _root(root)
    pools = load_pools(root)
    rows = []
    for rec in RECIPIENTS:
        for arm in ARMS:
            for draw in DRAWS:
                for seed in SEEDS:
                    rd = run_path(root, rec, arm, draw, seed)
                    ev, tr = rd / "eval/dog/preference_eval.json", rd / "train_manifest.json"
                    if not ev.is_file() or not tr.is_file():
                        raise ValueError(f"missing artifact: {rd}")
                    t, e = json.loads(tr.read_text()), json.loads(ev.read_text())
                    validate_train(t, rec, seed, pools[arm, draw], arm, draw)
                    values = validate_eval(e, rec, seed)
                    rows.append({"recipient_step": rec, "arm": arm, "pool_seed": draw,
                                 "student_seed": seed, "p_test": values["test"],
                                 "p_off_topic": values["off_topic"],
                                 "pool_sha256": pools[arm, draw]["pool_sha256"],
                                 "test_prompts_hash": PROMPTS["test"][0],
                                 "off_topic_prompts_hash": PROMPTS["off_topic"][0],
                                 "eval_path": str(ev), "train_manifest": str(tr),
                                 "eval_sha256": file_sha256(ev), "train_sha256": file_sha256(tr)})
    pool_means, summary, paired = aggregate(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tables = {"runs": rows, "pool_means": pool_means, "summary": summary, "paired": paired}
    hashes = {}
    for name, table in tables.items():
        path = out.with_name(out.name + "_" + name + ".csv")
        write_csv(path, table)
        hashes[str(path)] = file_sha256(path)
    provenance = {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(rows), "n_pool_means": len(pool_means),
        "n_summary": len(summary), "n_paired": len(paired),
        "positive_matched_differences": sum(p["iid_minus_firstonly"] > 0 for p in paired),
        "teacher": {"checkpoint": 125000, "recipe": TEACHER_TAG, "seed": 1},
        "carrier": "untuned original Pythia-160M checkpoint 75000",
        "student": {"recipe": STUDENT_TAG, "seeds": list(SEEDS), "recipients": list(RECIPIENTS)},
        "pool_inputs": {f"{arm}/draw{draw}": data for (arm, draw), data in pools.items()},
        "table_sha256": hashes,
        "definitions": {"aggregation": "mean student seeds within draw, then equal mean over three draws",
                        "shading": "one sample SD across three pool-draw means; not a confidence interval",
                        "paired_difference": "IID minus first-only at matching recipient, draw and student seed"},
        "validation": "108 exact identities; complete training; matched HPs, sample sizes, prompts and per-prompt means; six pool hashes checked against manifests and frozen campaigns",
        "script_sha256": file_sha256(__file__),
    }
    if PAIRING_AUDIT.exists():
        provenance["pairing_reaudit"] = {"path": str(PAIRING_AUDIT), "sha256": file_sha256(PAIRING_AUDIT)}
    out.with_name(out.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot(summary, pool_means, figure)
    print(json.dumps({name: len(value) for name, value in tables.items()}))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args()
    collect(args.output_root, args.out, args.figure)
