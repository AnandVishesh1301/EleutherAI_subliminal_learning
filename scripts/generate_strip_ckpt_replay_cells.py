#!/usr/bin/env python
"""Freeze the cell list for the non-adaptive-strip 40-checkpoint replay campaign.

The pilot (`run_logs/strip_ckpt_worker.py`, 2026-08-13) replayed 4 hand-picked strip
cells in place with checkpointing on and proved the replay is *bit-identical* to the
original. This generator extends that replay to the WHOLE focus-teacher strip.

**Selection rule (outcome-independent, frozen before any trajectory is looked at):**
every run in the non-adaptive 8x7 strip harvest
(`{output_root}/analysis/nonadaptive_8x7_harvest.json`, 1008 designed runs = 504 cells
x 2 ftseeds) EXCEPT the rows flagged ``diverged`` (non-finite adapter weights; Angela
2026-08-14: "except for the ones that diverge"). No fate, transfer, or boundary
criterion enters -- divergence is the only exclusion, and it is a property of the
original training run, not of the replay.

Jobs are keyed by CELL, not by run: `evaluate_teacher_preference_pipeline` scores a
whole ``{param_tag}`` dir at once and writes the tag-level ``preference_summary.json``,
so a cell's seeds must be replayed together for the regenerated summary to be
comparable with the backed-up original.

Usage:
    uv run python -m scripts.generate_strip_ckpt_replay_cells            # write registry
    uv run python -m scripts.generate_strip_ckpt_replay_cells --jobs     # + jobs.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from subliminal_learning.templates.config import load_template_config

ROOT = Path(__file__).absolute().parents[1]
REGISTRY = ROOT / "run_logs" / "strip_ckpt_replay_cells.json"
JOBS = ROOT / "run_logs" / "strip_ckpt_replay_jobs.jsonl"
HARVEST_NAME = "nonadaptive_8x7_harvest.json"

# Spill target for the checkpoint payload. NOT in paths_local.yaml on purpose: this is
# one campaign's overflow location, not a project-wide root that other code may assume
# exists on every node. The worker reads it from this registry.
COLD_ROOT = "/mnt/cold-1/soar-subliminal/angela/outputs"


def build(output_root: Path) -> dict:
    harvest = json.loads((output_root / "analysis" / HARVEST_NAME).read_text())
    cells: dict[str, list[int]] = defaultdict(list)
    n_div = 0
    for row in harvest:
        if row["diverged"]:
            n_div += 1
            continue
        cells[row["tag"]].append(int(row["seed"]))
    ordered = {t: sorted(s) for t, s in sorted(cells.items())}
    return {
        "campaign": "strip-ckpt-replay",
        "created_utc": "2026-08-14T00:00:00+00:00",
        "rule": (
            "every run in analysis/nonadaptive_8x7_harvest.json whose `diverged` flag is "
            "false; divergence is the ONLY exclusion. Frozen before any replay ran."
        ),
        "harvest": HARVEST_NAME,
        "lineage": {
            "animal": "dog",
            "teacher_base_model": "pythia-160m",
            "teacher_tag": "ep2_lr5e-4_r32a64",
            "teacher_ftseed": 1,
            "student_base": "data-seed1",
            "filter_mode": "prefix",
            "subsample_size": 10_000,
            "subsample_seed": 42,
        },
        "checkpointing": {
            "save_steps": 625,
            "save_total_limit": 50,
            "save_only_model": True,
            "keep_files": ["adapter_model.safetensors", "adapter_config.json"],
            "note": (
                "40 checkpoints over the ep20/25k-step budget. save_only_model drops "
                "optimizer/scheduler/rng state (75.6 of 112 MB at r64); the post-train "
                "prune drops the per-checkpoint tokenizer copy (3.6 MB x 40 x 932 = "
                "134 GB). Adapters only: ~705 GB for the campaign."
            ),
        },
        "cold_root": COLD_ROOT,
        "n_designed_runs": len(harvest),
        "n_diverged_excluded": n_div,
        "n_cells": len(ordered),
        "n_runs": sum(len(s) for s in ordered.values()),
        "cells": ordered,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", action="store_true", help="Also write the jobq jobs.jsonl.")
    a = p.parse_args()

    base = load_template_config(
        ROOT / "configs/paths/paths_local.yaml",
        ROOT / "configs/model/pythia.yaml",
        ROOT / "configs/training/lora_template_bias.yaml",
        ROOT / "configs/experiment/template_bias_pythia160m.yaml",
    )
    reg = build(base.paths.output_root)
    REGISTRY.write_text(json.dumps(reg, indent=1) + "\n")
    print(f"{REGISTRY}: {reg['n_cells']} cells / {reg['n_runs']} runs "
          f"({reg['n_diverged_excluded']} diverged rows excluded)")

    if a.jobs:
        with JOBS.open("w") as fh:
            for tag in reg["cells"]:
                cmd = (f"cd {ROOT} && uv run python run_logs/strip_ckpt_replay_worker.py "
                       f"--tag {tag}")
                fh.write(json.dumps({"key": f"ckptreplay|{tag}", "cmd": cmd}) + "\n")
        print(f"{JOBS}: {reg['n_cells']} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
