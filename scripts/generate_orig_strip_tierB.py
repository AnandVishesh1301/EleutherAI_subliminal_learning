"""Generate the Tier B non-adaptive strip campaign (SAME-MODEL original arm).

Deterministic builder for a checkpoint-saving replay of the ds1 non-adaptive strip on the
ORIGINAL arm: student base = original pythia-160m, teacher = the focus teacher
ep2_lr5e-4_r32a64 (tseed1) on the original base, ep20, ~40 intermediate LoRA checkpoints
per run. Emits, under run_logs/:

  * orig_strip_tierB_cells.json          registry for the two trajectory scripts
  * orig_strip_tierB_distill_jobs.jsonl  one line per (cell, fresh seed) distill job
  * orig_strip_tierB_preftraj_jobs.jsonl one line per cell (preference trajectory)
  * orig_strip_tierB_pilekl_jobs.jsonl   one line per cell (Pile-KL trajectory)

Cell selection is derived purely from the two ds1 strip job files (the design this campaign
replays); see the module constants for the K/K* bin rule. Re-running reproduces every file
byte-for-byte.

    uv run python -m scripts.generate_orig_strip_tierB
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
RUN_LOGS = ROOT / "run_logs"

SOURCE_JOB_FILES = (
    RUN_LOGS / "nonadaptive_strip_ds1_jobs.jsonl",
    RUN_LOGS / "nonadaptive_strip_ds1_ext7_jobs.jsonl",
)

REGISTRY_PATH = RUN_LOGS / "orig_strip_tierB_cells.json"
DISTILL_JOBS = RUN_LOGS / "orig_strip_tierB_distill_jobs.jsonl"
PREFTRAJ_JOBS = RUN_LOGS / "orig_strip_tierB_preftraj_jobs.jsonl"
PILEKL_JOBS = RUN_LOGS / "orig_strip_tierB_pilekl_jobs.jsonl"

CD = "cd /home/angela/EleutherAI_subliminal_learning && "

# Coordinate these cells were PLACED in: K = lr * a**B_PLACEMENT * r**C_PLACEMENT
# (trait-era exponents/anchor, retired 2026-09-09). Kept verbatim so the generated cell
# list stays reproducible; K/K* READOUTS must use cl.B_LAW / cl.C_LAW /
# cl.KSTAR_CALIBRATED (designed cross-data-order population fit, anchored label).
B_PLACEMENT, C_PLACEMENT = 0.561, -0.085
KSTAR = 7.87e-4

# ds1 strip seed convention: ftseed 1 for the first seed, 2 for the second of two-seed cells.
# (901/902 are the ds1 mech-probe trajectory block and stay reserved.)
FRESH_SEEDS = (1, 2)

# The focus teacher / lineage of this replay.
LINEAGE = {
    "animal": "dog",
    "teacher_base_model": "pythia-160m",
    "teacher_tag": "ep2_lr5e-4_r32a64",
    "teacher_ftseed": 1,
    "student_base": "original",
    "filter_mode": "prefix",
    "subsample_size": 10000,
    "subsample_seed": 42,
}

TAG_RE = re.compile(r"ep20_lr([0-9.e-]+)_r(\d+)a(\d+)")


def _parse_cells() -> dict[tuple[str, int, int], int]:
    """Map (lr_str, r, a) -> original job count (line count across both ds1 files)."""
    counts: dict[tuple[str, int, int], int] = {}
    for path in SOURCE_JOB_FILES:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            key = json.loads(line)["key"]
            m = TAG_RE.search(key)
            if not m:
                raise ValueError(f"no ep20 student tag in key {key!r} ({path})")
            cell = (m.group(1), int(m.group(2)), int(m.group(3)))
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def _krel(cell: tuple[str, int, int]) -> float:
    lr_str, r, a = cell
    return (float(lr_str) * a**B_PLACEMENT * r**C_PLACEMENT) / KSTAR


def _select(
    counts: dict[tuple[str, int, int], int],
) -> tuple[list[tuple[str, int, int]], dict[str, int]]:
    """Apply the Tier B bin rule; returns (kept cells unordered, per-bin kept counts)."""
    inside, low, high, tail = [], [], [], []
    for cell in counts:
        k = _krel(cell)
        if 0.3 <= k <= 3:
            inside.append(cell)
        elif k < 0.3:
            low.append(cell)
        elif k <= 30:  # 3 < k <= 30
            high.append(cell)
        else:  # k > 30
            tail.append(cell)
    low.sort(key=_krel)
    high.sort(key=_krel)
    tail.sort(key=_krel)
    kept = inside + low[::2] + high[::2] + tail[::3]
    bins = {"inside": len(inside), "low": len(low[::2]),
            "high": len(high[::2]), "tail": len(tail[::3])}
    return kept, bins


def _seeds_for(cell: tuple[str, int, int], orig_count: int) -> list[int]:
    """Fresh ftseeds for a cell: 2 near the boundary, else 1; never > original count."""
    want = 2 if 0.4 <= _krel(cell) <= 4 else 1
    want = min(want, orig_count)
    return list(FRESH_SEEDS[:want])


def main() -> None:
    counts = _parse_cells()
    kept, bins = _select(counts)
    # Deterministic order: by (r, a, lr) ascending for stable, readable output.
    kept.sort(key=lambda c: (c[1], c[2], float(c[0])))

    cells_registry: dict[str, list[int]] = {}
    distill_lines: list[str] = []
    traj_lines_pref: list[str] = []
    traj_lines_kl: list[str] = []
    n_jobs = 0
    for cell in kept:
        lr_str, r, a = cell
        seeds = _seeds_for(cell, counts[cell])
        student_tag = f"ep20_lr{lr_str}_r{r}a{a}"
        cells_registry[student_tag] = seeds
        for seed in seeds:
            n_jobs += 1
            cmd = (
                CD
                + "uv run python run_logs/animal_samedistill_worker.py "
                + "--animal dog --base original "
                + "--manifest run_logs/dog_top20krungs_configs.json "
                + "--config-tag ep2_lr5e-4_r32a64 --epochs 20 --modes plain "
                + f"--student-lr {lr_str} --student-rank {r} --student-alpha {a} "
                + f"--student-seed {seed} --save-checkpoints"
            )
            distill_lines.append(json.dumps(
                {"key": f"tierB|{student_tag}|s{seed}", "cmd": cmd}))
        pref_cmd = (CD + "uv run python -m scripts.ckpt_preference_trajectory "
                    f"--tag {student_tag} --registry run_logs/orig_strip_tierB_cells.json")
        kl_cmd = (CD + "uv run python -m scripts.ckpt_pile_kl_trajectory "
                  f"--tag {student_tag} --registry run_logs/orig_strip_tierB_cells.json")
        traj_lines_pref.append(json.dumps({"key": f"tierB-pref|{student_tag}", "cmd": pref_cmd}))
        traj_lines_kl.append(json.dumps({"key": f"tierB-kl|{student_tag}", "cmd": kl_cmd}))

    # Design invariants (fail loud if the source files or rule ever drift).
    assert len(kept) == 233, f"expected 233 cells, got {len(kept)}"
    assert n_jobs == 316, f"expected 316 distill jobs, got {n_jobs}"
    assert bins == {"inside": 89, "low": 76, "high": 61, "tail": 7}, bins

    registry = {
        "campaign": "orig-strip-tierB",
        "lineage_slug": "orig-strip-tierB",
        "rule": "Tier B non-adaptive strip replay on the ORIGINAL same-model arm with 40 "
                "intermediate checkpoints per run; cells sub-sampled from the ds1 strip by "
                "K/K* bin (see scripts/generate_orig_strip_tierB.py).",
        "source_job_files": [p.name for p in SOURCE_JOB_FILES],
        "kstar": KSTAR,
        "seeds": list(FRESH_SEEDS),
        "n_cells": len(kept),
        "n_runs": n_jobs,
        "krel_bins": bins,
        "lineage": LINEAGE,
        "checkpointing": {"save_steps": 625, "save_total_limit": 50, "save_only_model": True},
        "cells": cells_registry,
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=1) + "\n")
    DISTILL_JOBS.write_text("\n".join(distill_lines) + "\n")
    PREFTRAJ_JOBS.write_text("\n".join(traj_lines_pref) + "\n")
    PILEKL_JOBS.write_text("\n".join(traj_lines_kl) + "\n")

    print(f"cells={len(kept)} distill_jobs={n_jobs} bins={bins}")
    print(f"registry -> {REGISTRY_PATH}")
    print(f"distill  -> {DISTILL_JOBS} ({len(distill_lines)} lines)")
    print(f"preftraj -> {PREFTRAJ_JOBS} ({len(traj_lines_pref)} lines)")
    print(f"pilekl   -> {PILEKL_JOBS} ({len(traj_lines_kl)} lines)")


if __name__ == "__main__":
    main()
