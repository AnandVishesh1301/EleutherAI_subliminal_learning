"""EQ-hard (plan v2 §15.3): does a frozen bf16 base change hard-label transfer?

Re-trains the hlopt Part I anchor cell (pythia-160m dog, E3, LoRA r8 α32 AdamW 1e-4,
20 epochs) with a frozen bf16 base + fp32 LoRA + bf16 autocast (`student_dtype=
"bfloat16"`), seeds 1-3, for the dog treatment and the base-number control, on exactly
the hlopt Part I inputs (staged pools, base model, pp1 text; all read-only). Every cell
is scored with the hlopt eval path (`sweep_hlopt._probe` / `_preference`, fp32 eval)
and its Delta computed with `hlopt_metrics.paired_deltas`, so the only difference from
the fp32 references in the hlopt results is the training precision.

Outputs under ``layout.EQ_DIR / "hard"``: the students (hlopt student layout under
``outputs/``) and ``eq_summary.json`` (recomputed by `summary()`, which the results
build calls):
``delta_bf16`` / ``delta_fp32`` are one test-set Delta (hlopt's report set) per seed in
`SEEDS` order; ``complete`` is false until every bf16 seed is scored. The label rule
itself (plan §15.3) is applied by the results pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import hlopt_metrics
from subliminal_learning.campaign import hlopt_plan as hp
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import cell_is_done

ANIMAL = "dog"
SOURCES: tuple[str, ...] = ("teacher", "base")
SEEDS: tuple[int, ...] = hp.SEEDS
STUDENT_DTYPE = "bfloat16"
REFERENCE_RUN_SET = hp.RUN_SET_ID
EQ_ROOT = layout.EQ_DIR / "hard"
OUTPUT_ROOT = EQ_ROOT / "outputs"
SUMMARY_FILE = EQ_ROOT / "eq_summary.json"
READOUT = hp.REPORT_SET


def anchor_config() -> hp.Config:
    """The hlopt Part I anchor config (E3, LoRA r8 α32, AdamW 1e-4, 20 epochs).

    Raises:
        RuntimeError: The plan does not hold exactly one such config.
    """
    found = [
        c
        for c in hp.configs()
        if (c.tier, c.recipe, c.mode, c.rank, c.alpha, c.optim, c.lr, c.epochs)
        == (hp.TIER_ANCHOR, "e3", "lora", 8, 32, "adamw", 1e-4, 20)
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one hlopt anchor config, found {len(found)}")
    return found[0]


def _check(source: str, seed: int) -> None:
    if source not in SOURCES or seed not in SEEDS:
        raise ValueError(f"EQ-hard cell ({source}, {seed}) not in {SOURCES} x {SEEDS}")


def run(source: str, seed: int, device: str) -> Path:
    """Train (bf16 base) and score one EQ-hard cell; idempotent.

    Returns:
        The student run dir.

    Raises:
        RuntimeError: The staged pool or the run dir differs from the hlopt contract.
    """
    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from scripts.sweeps.sweep_hlopt import _context, _free_gpu, _preference, _probe
    from subliminal_learning.campaign import hlopt_layout

    _check(source, seed)
    config = anchor_config()
    paths, hub_org, spec = _context()
    paths = replace(paths, output_root=OUTPUT_ROOT)
    pool = hp.POOLS[ANIMAL if source == "teacher" else "base"]
    if file_sha256(pool.staged_dir / "filtered.jsonl") != pool.filtered_sha256:
        raise RuntimeError(
            f"{pool.staged_dir}: filtered.jsonl sha differs from the pin"
        )
    lora, train = hlopt_layout.train_cfgs(config, seed)
    teacher = source == "teacher"
    result = distill_run(
        DistillStudentConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            source=source,
            number_dataset_dir=pool.staged_dir,
            lora=lora,
            train=train,
            animal=ANIMAL if teacher else None,
            subsample_size=hp.SUBSAMPLE_SIZE,
            subsample_seed=hp.SUBSAMPLE_SEED,
            teacher_param_tag=hp.TEACHER_PARAM_TAG if teacher else None,
            teacher_seed=hp.TEACHER_FTSEED if teacher else None,
            number_dataset_tag=hp.NUMBER_DATASET_TAG,
            device=device,
            wandb_enabled=False,
            student_dtype=STUDENT_DTYPE,
        )
    )
    run_dir = result.run_dir
    if not str(run_dir).startswith(str(OUTPUT_ROOT)) or not cell_is_done(run_dir):
        raise RuntimeError(
            f"EQ-hard run dir {run_dir} is not a finished cell under {OUTPUT_ROOT}"
        )
    _free_gpu()
    probe = _probe(paths, hub_org, spec, run_dir, device)
    _preference(
        paths, hub_org, spec, run_dir, source, [ANIMAL], hp.PROB_SETS, False, device
    )
    logger.info(
        "EQ-HARD DONE {} s{} dce={:.3f} -> {}", source, seed, probe["delta_ce"], run_dir
    )
    return run_dir


def _cell_row(run_dir: Path, source: str, seed: int, key: str) -> dict | None:
    """One hlopt-style row (`paired_deltas` input) of a finished EQ cell."""
    from pipelines.hlopt_results_pipeline import _p
    from subliminal_learning.campaign import hlopt_layout

    if not cell_is_done(run_dir):
        return None
    pref = hlopt_layout.preference_path(run_dir, source, ANIMAL)
    if not pref.is_file():
        return None
    sets = json.loads(pref.read_text()).get("sets", {})
    probe_path = hlopt_layout.probe_path(run_dir)
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {}
    return {
        "config_key": key,
        "animal": ANIMAL,
        "source": source,
        "seed": seed,
        "p": {s: _p(sets.get(s), ANIMAL) for s in hp.PROB_SETS},
        "delta_ce": probe.get("delta_ce", math.nan),
        "collapsed": probe.get("collapsed"),
        "run_dir": str(run_dir),
    }


def summary() -> dict:
    """Per-seed Delta of the bf16 cells beside the fp32 hlopt Part I references.

    A pure, idempotent recompute from the per-seed eval files on disk and the frozen
    reference table (no timestamp, fixed row order), so every caller -- each EQ job
    after scoring, and the results build -- writes the same bytes for the same disk
    state. Writes `SUMMARY_FILE` and returns it; unscored seeds are listed as missing.
    """
    from scripts.sweeps.sweep_hlopt import _context
    from subliminal_learning.campaign import hlopt_layout
    from subliminal_learning.students.layout import student_run_dir

    config = anchor_config()
    _, hub_org, spec = _context()
    rows = []
    for source in SOURCES:
        for seed in SEEDS:
            lora, train = hlopt_layout.train_cfgs(config, seed)
            teacher = source == "teacher"
            run_dir = student_run_dir(
                OUTPUT_ROOT,
                source=source,
                spec=spec,
                hub_org=hub_org,
                train_cfg=train,
                lora_cfg=lora,
                animal=ANIMAL if teacher else None,
                teacher_param_tag=hp.TEACHER_PARAM_TAG if teacher else None,
                teacher_seed=hp.TEACHER_FTSEED if teacher else None,
                number_dataset_tag=hp.NUMBER_DATASET_TAG,
            )
            row = _cell_row(run_dir, source, seed, config.key)
            if row is not None:
                rows.append(row)
    bf16 = {
        int(d["seed"]): {s: d["delta"].get(s) for s in (hp.SELECT_SET, hp.REPORT_SET)}
        for d in hlopt_metrics.paired_deltas(rows)
    }
    ref_csv = hp.OUTPUT_ROOT / "hlopt" / REFERENCE_RUN_SET / "results" / "deltas.csv"
    ref = pd.read_csv(ref_csv)
    ref = ref[
        (ref.config_key == config.key) & (ref.animal == ANIMAL) & ref.seed.isin(SEEDS)
    ]
    if ref.seed.duplicated().any():
        raise RuntimeError(f"{ref_csv}: duplicate reference rows for {config.key}")
    fp32 = {
        int(r.seed): {
            hp.SELECT_SET: float(r[f"delta_{hp.SELECT_SET}"]),
            hp.REPORT_SET: float(r[f"delta_{hp.REPORT_SET}"]),
        }
        for _, r in ref.iterrows()
    }
    if set(fp32) != set(SEEDS):
        raise RuntimeError(f"{ref_csv}: reference seeds {sorted(fp32)} != {SEEDS}")
    missing = sorted(set(SEEDS) - set(bf16))
    doc = {
        "config_key": config.key,
        "animal": ANIMAL,
        "readout": READOUT,
        "seeds": list(SEEDS),
        "complete": not missing,
        "delta_bf16": [bf16[s][READOUT] for s in SEEDS if s in bf16],
        "delta_fp32": [fp32[s][READOUT] for s in SEEDS],
        "missing_bf16_seeds": missing,
        "delta_by_set_bf16": {str(k): v for k, v in sorted(bf16.items())},
        "delta_by_set_fp32": {str(k): v for k, v in sorted(fp32.items())},
        "student_dtype": STUDENT_DTYPE,
        "reference": {
            "run_set_id": REFERENCE_RUN_SET,
            "file": str(ref_csv),
            "sha256": file_sha256(ref_csv),
        },
        "cells": rows,
    }
    atomic_write_json(SUMMARY_FILE, doc)
    return doc
