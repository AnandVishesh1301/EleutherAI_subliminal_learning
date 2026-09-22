"""Freeze and run a core30 student-HP panel through the existing checkpoint runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import yaml

STEPS = tuple(range(75000, 130001, 5000)) + (143000,)
TEACHER_STEPS = (75000, 125000)
TEACHER_TAG = "ep10_lr1e-4_r64a64"
HP_TAGS = (
    "ep20_lr2e-4_r8a4",
    "ep20_lr4e-4_r4a2",
    "ep20_lr2e-4_r128a16",
    "ep20_lr2e-4_r32a16",
    "ep20_lr2e-4_r64a32",
)
CORE30 = Path("run_logs/core30_cells.json")
STUDENT_CONFIG = Path("configs/training/lora_student_distill.yaml")
NUMBER_CONFIG = Path("configs/experiment/number_generation.yaml")
TEACHER_CONFIG = Path("configs/training/lora_template_bias.yaml")
DEFAULT_OUT = Path("run_logs/checkpoint_studenthp5_20260922")
QUEUE = "ckpt-dog-studenthp5-20260922"
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def frozen_write(path, contents):
    path = Path(path)
    if path.exists() and path.read_text() != contents:
        raise ValueError(f"refusing to change frozen campaign file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def selected_hps(core_path=CORE30):
    rows = json.loads(Path(core_path).read_text())["cells"]
    by_tag = {r["student_tag"]: r for r in rows}
    if len(rows) != 30 or len(by_tag) != 30:
        raise ValueError("core30 must contain 30 unique student recipes")
    if not set(HP_TAGS).issubset(by_tag):
        raise ValueError("selected recipe is not a member of core30")
    return [by_tag[tag] for tag in HP_TAGS]


def readable(path):
    return Path(path).is_file() and os.access(path, os.R_OK)


def local_file(recorded):
    """Map a recorded frozen-input path onto this node's checkout.

    Manifests may record repo files as absolute paths under the building node's
    checkout, which another node cannot even read; re-anchor the repo-relative tail
    here. Readable paths and non-repo paths (shared mounts) are returned unchanged.
    """
    path = Path(recorded)
    if readable(path):
        return path
    tails = []
    if not path.is_absolute():
        tails.append(path.parts)
    parts = path.parts
    if REPO_ROOT.name in parts:
        tails.append(parts[len(parts) - 1 - parts[::-1].index(REPO_ROOT.name) + 1:])
    for tail in tails:
        candidate = REPO_ROOT.joinpath(*tail)
        if readable(candidate):
            return candidate
    return path


def check_files(files):
    for path, expected in files.items():
        local = local_file(path)
        if not readable(local) or sha(local) != expected:
            raise ValueError(f"frozen input changed or disappeared: {path}")


def verified_cell(campaign_path, campaign_sha256, key):
    if sha(campaign_path) != campaign_sha256:
        raise ValueError("campaign identity changed")
    campaign = json.loads(Path(campaign_path).read_text())
    cells = [cell for cell in campaign["cells"] if cell["key"] == key]
    if len(cells) != 1:
        raise ValueError(f"expected one campaign cell for {key}")
    cell = cells[0]
    check_files(campaign["shared_files"])
    check_files(cell["files"])
    return cell


def record_path(path):
    """Record repo files relative to the repo root so other checkouts can verify them."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved)


def build(output_root, out=DEFAULT_OUT, seeds=(1,)):
    from dataclasses import replace

    from scripts.sweeps import sweep_ckpt_distill as runner
    from subliminal_learning.models.datadecide import parse_base_model_tokens
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.students.layout import student_run_dir, student_param_tag

    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 1 for s in seeds):
        raise ValueError("student seeds must be unique positive integers")
    output_root, out = Path(output_root).resolve(), Path(out).resolve()
    configured = yaml.safe_load(Path(runner._PATHS_CONFIG).read_text())["output_root"]
    if output_root != Path(configured).resolve():
        raise ValueError("output root disagrees with existing checkpoint runner")
    hps = selected_hps()
    template = yaml.safe_load(STUDENT_CONFIG.read_text())
    common_files = [
        CORE30, STUDENT_CONFIG, NUMBER_CONFIG, TEACHER_CONFIG,
        Path(runner._PATHS_CONFIG), Path(runner._MODEL_CONFIG),
        Path(runner._EXPERIMENT_CONFIG),
    ]
    shared = {record_path(p): sha(p) for p in common_files}
    pools = {}
    for step in TEACHER_STEPS:
        raw = output_root / (
            f"numbers/dog/ckpt/pythia-160m@step{step}/{TEACHER_TAG}/ftseed1/ex10-20"
        )
        filtered = raw / "modes/prefix/filtered.jsonl"
        with filtered.open() as handle:
            n_rows = sum(bool(line.strip()) for line in handle)
        if n_rows < 10000:
            raise ValueError(f"underfilled pool at step {step}: {n_rows}")
        raw_manifest = json.loads((raw / "manifest.json").read_text())
        if raw_manifest["n_raw"] != 100000:
            raise ValueError("expected canonical 100k generation pool")
        pools[str(step)] = {
            "path": str(filtered), "n_rows": n_rows,
            "files": {
                record_path(p): sha(p) for p in (
                    filtered, raw / "manifest.json",
                    raw / "modes/prefix/manifest.json",
                )
            },
        }
    configs = {}
    for hp in hps:
        tag = hp["student_tag"]
        doc = copy.deepcopy(template)
        doc["lora"].update(r=hp["rank"], lora_alpha=hp["alpha"])
        doc["train"].update(learning_rate=hp["lr"], n_epochs=20)
        train_path = out / "configs" / f"{tag}.student.yaml"
        frozen_write(train_path, yaml.safe_dump(doc, sort_keys=False))
        lora, train = load_student_training(train_path)
        if student_param_tag(train, lora) != tag:
            raise ValueError(f"student HP does not round-trip: {tag}")
        sweep = {
            "base_model": "original", "dtype": "float32",
            "teacher": {"training_config": str(TEACHER_CONFIG.resolve()),
                        "param_tags": [TEACHER_TAG], "ft_seeds": [1]},
            "numbers": {"number_config": str(NUMBER_CONFIG.resolve()),
                        "filter_mode": "prefix"},
            "student": {"training_config": str(train_path), "epochs": [20],
                        "ft_seeds": list(seeds), "subsample_size": 10000,
                        "subsample_seed": 42, "modes": ["plain"]},
        }
        sweep_path = out / "configs" / f"{tag}.sweep.yaml"
        frozen_write(sweep_path, yaml.safe_dump(sweep, sort_keys=False))
        configs[tag] = (sweep_path, train_path, lora, train)
    cells = []
    for teacher_step in TEACHER_STEPS:
        for student_step in STEPS:
            for hp in hps:
                tag = hp["student_tag"]
                sweep_path, train_path, lora, train = configs[tag]
                for seed in seeds:
                    spec = parse_base_model_tokens(f"original@step{student_step}")[0]
                    run_dir = student_run_dir(
                        output_root, source="teacher", spec=spec, hub_org="EleutherAI",
                        train_cfg=replace(train, seed=seed), lora_cfg=lora, animal="dog",
                        teacher_param_tag=TEACHER_TAG, teacher_seed=1,
                        number_dataset_tag="ex10-20_prefix_n100k",
                        numbers_base_model=f"pythia-160m@step{teacher_step}",
                    )
                    # Existing cells require an explicit identity/evaluation audit; never
                    # count a baseline HP or an incomplete artifact as exact reuse.
                    if run_dir.exists():
                        raise ValueError(f"existing cell requires reuse audit: {run_dir}")
                    key = f"t{teacher_step}-s{student_step}-{tag}-seed{seed}"
                    argv = [
                        "-m", "scripts.sweeps.sweep_ckpt_distill", "run-cell",
                        "--sweep-config", str(sweep_path), "--animal", "dog",
                        "--teacher-step", str(teacher_step),
                        "--student-step", str(student_step), "--teacher-seed", "1",
                        "--param-tag", TEACHER_TAG, "--mode", "plain",
                        "--epochs", "20", "--student-seed", str(seed),
                    ]
                    files = dict(pools[str(teacher_step)]["files"])
                    files.update({record_path(p): sha(p) for p in (sweep_path, train_path)})
                    cells.append({
                        "key": key, "teacher_step": teacher_step, "student_step": student_step,
                        "student_tag": tag, "student_seed": seed, "run_dir": str(run_dir),
                        "argv": argv, "files": files,
                    })
    if len({c["run_dir"] for c in cells}) != len(cells):
        raise ValueError("output collision in student HP panel")
    campaign = {
        "name": QUEUE, "animal": "dog", "model": "Pythia-160M",
        "teacher_recipe": TEACHER_TAG, "teacher_seed": 1,
        "teacher_recipe_selection": "Highest observed 125k diagonal mean and 125k-source mean across 13 recipients among the five recipes; selected using prior test outcomes.",
        "student_hps": hps, "student_seeds": seeds,
        "hp_selection": "Ordered historical core30 K_rel positions 0,7,15,22,29; design-only coverage, not a predicted checkpoint performance ordering.",
        "protocol": {"epochs": 20, "subsample_size": 10000, "subsample_seed": 42,
                     "dtype": "float32", "mode": "plain",
                     "evaluation": ["test", "off_topic"]},
        "existing_baseline": "ep10_lr1e-4_r8a32: reference only, outside core30; 10 epochs",
        "shared_files": shared, "pools": pools, "cells": cells,
        "n_jobs": len(cells), "n_reused": 0, "priority": 10000,
    }
    campaign_path = out / "campaign.json"
    frozen_write(campaign_path, json.dumps(campaign, indent=2) + "\n")
    digest = sha(campaign_path)
    jobs = [{
        "key": cell["key"],
        "cmd": shlex.join([
            "uv", "run", "python", "-m", "scripts.build_checkpoint_studenthp_preflight",
            "--campaign", str(campaign_path), "--campaign-sha256", digest,
            "--run-cell-key", cell["key"],
        ]),
        "cwd": str(Path.cwd()), "env": {"WANDB_MODE": "disabled"},
    } for cell in cells]
    frozen_write(out / "jobs.jsonl", "".join(json.dumps(j) + "\n" for j in jobs))
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--campaign-sha256")
    parser.add_argument("--run-cell-key")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.run_cell_key:
        if not args.campaign or not args.campaign_sha256:
            parser.error("--run-cell-key requires campaign path and SHA256")
        cell = verified_cell(args.campaign, args.campaign_sha256, args.run_cell_key)
        if args.verify_only:
            print(json.dumps(cell, indent=2))
        else:
            os.execv(sys.executable, [sys.executable, *cell["argv"]])
    else:
        if not args.output_root:
            parser.error("building requires --output-root")
        campaign = build(args.output_root, args.out, args.student_seeds)
        print(json.dumps({"jobs": campaign["n_jobs"], "reused": campaign["n_reused"],
                          "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
