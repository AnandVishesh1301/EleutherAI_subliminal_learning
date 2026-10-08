"""Read every `hlopt` cell's artifacts and write the campaign results tables.

`cell_rows` is also the driver's input for the extra-seed selection. Outputs under
`{output_root}/hlopt/{run_set_id}/results/`:

    cells.csv   one row per (cell, evaluated animal): p per prompt set, pp1, loss, lm-eval
    deltas.csv  one row per treatment with a matched control: Delta per prompt set
    rungs.csv   per (config, animal): seed mean / sd of Delta, pp1 delta-CE, selection flags
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from pipelines.distill_student_pipeline import TRAIN_LOG_HISTORY_FILE
from subliminal_learning.analysis.clip_stats import clip_stats
from subliminal_learning.campaign import hlopt_jobs as jobs
from subliminal_learning.campaign import hlopt_layout as layout
from subliminal_learning.campaign import hlopt_metrics as metrics
from subliminal_learning.campaign import hlopt_plan as plan
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text


def _read(path: Path) -> dict | list | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _p(entry: Mapping | None, animal: str) -> float:
    """p(animal) of one per-seed set entry; NaN when unscored or for another animal."""
    if not entry or entry.get("target_animal") != animal:
        return math.nan
    if int(entry.get("ci", {}).get("count", 0)) == 0 or entry.get("p_target") is None:
        return math.nan
    return float(entry["p_target"])


def _last_logged_loss(history: list | None) -> float:
    losses = [h["loss"] for h in history or [] if "loss" in h]
    return float(losses[-1]) if losses else math.nan


def cell_rows(
    spec: PythiaModelSpec, hub_org: str, cells: Iterable[jobs.Cell]
) -> list[dict]:
    """One row per trained (cell, evaluated animal); untrained cells are omitted."""
    rows: list[dict] = []
    for cell in cells:
        c = cell.config
        leaf = layout.run_dir(spec, hub_org, c, cell.source, cell.animal, cell.seed)
        manifest = _read(leaf / "train_manifest.json")
        if manifest is None:
            continue
        probe = _read(layout.probe_path(leaf)) or {}
        lmeval = _read(layout.lmeval_path(leaf)) or {}
        history = _read(leaf / TRAIN_LOG_HISTORY_FILE)
        clips = clip_stats(history or [], c.clip)
        for animal in cell.eval_animals:
            sets = (_read(layout.preference_path(leaf, cell.source, animal)) or {}).get(
                "sets", {}
            )
            row = {
                "tier": cell.tier,
                "config_key": c.key,
                "arm": c.arm,
                "recipe": c.recipe,
                "mode": c.mode,
                "rank": c.rank,
                "alpha": c.alpha,
                "optim": c.optim,
                "clip": c.clip,
                "part": c.part,
                "lr": c.lr,
                "epochs": c.epochs,
                "source": cell.source,
                "animal": animal,
                "seed": cell.seed,
                "host": cell.host,
                "p": {
                    s: _p(sets.get(s), animal)
                    for s in plan.PROB_SETS + plan.SAMPLE_SETS
                },
                "delta_ce": probe.get("delta_ce", math.nan),
                "kl": probe.get("kl", math.nan),
                "collapsed": probe.get("collapsed"),
                "final_train_loss": manifest.get("final_train_loss"),
                "last_logged_loss": _last_logged_loss(history),
                "grad_norm_p10": clips.grad_norm_p10,
                "grad_norm_p50": clips.grad_norm_p50,
                "grad_norm_p90": clips.grad_norm_p90,
                "frac_clipped": clips.frac_clipped,
                "n_train_steps": manifest.get("n_train_steps"),
                "lmeval": {
                    f"{task}_{m}": v
                    for task, ms in (lmeval.get("tasks") or {}).items()
                    for m, v in ms.items()
                },
                "run_dir": str(leaf),
            }
            rows.append(row)
    return rows


def _flatten(row: Mapping, nested: tuple[str, ...]) -> dict:
    flat = {k: v for k, v in row.items() if k not in nested}
    for key in nested:
        for sub, value in (row.get(key) or {}).items():
            flat[f"{key}_{sub}"] = value
    return flat


def all_cells(
    selection: Mapping[str, list[str]] | None, part: int = 1
) -> list[jobs.Cell]:
    """Main cells of `part` plus the extra-seed cells of `selection`."""
    main = {1: jobs.main_cells, 2: jobs.main_cells2, 3: jobs.main_cells3}[part]()
    return main + (jobs.extra_cells(selection, part) if selection else [])


def build(
    results_dir: Path,
    spec: PythiaModelSpec,
    hub_org: str,
    selection: Mapping[str, list[str]] | None,
    part: int = 1,
) -> dict[str, str]:
    """Write cells / deltas / rungs CSVs of `part`; return their paths."""
    rows = cell_rows(spec, hub_org, all_cells(selection, part))
    deltas = metrics.paired_deltas(rows)
    cells_df = pd.DataFrame([_flatten(r, ("p", "lmeval")) for r in rows])
    deltas_df = pd.DataFrame(
        [_flatten(d, ("p", "p_control", "delta", "lmeval")) for d in deltas]
    )
    rung_rows = []
    if not deltas_df.empty:
        delta_cols = [
            c
            for c in deltas_df.columns
            if c.startswith("delta_") and c != "delta_ce_control"
        ]
        for (key, animal), g in deltas_df.groupby(["config_key", "animal"], sort=False):
            for seeds_name, seeds in (
                ("s123", plan.SEEDS),
                ("all", plan.SEEDS + plan.EXTRA_SEEDS),
            ):
                sub = g[g["seed"].isin(seeds)]
                if sub.empty:
                    continue
                rung = {
                    "config_key": key,
                    "animal": animal,
                    "seeds": seeds_name,
                    "n_seeds": len(sub),
                    "arm": g["arm"].iloc[0],
                    "tier": g["tier"].iloc[0],
                    "recipe": g["recipe"].iloc[0],
                    "mode": g["mode"].iloc[0],
                    "optim": g["optim"].iloc[0],
                    "clip": g["clip"].iloc[0],
                    "lr": g["lr"].iloc[0],
                    "epochs": g["epochs"].iloc[0],
                    "selected": bool(selection and animal in selection.get(key, [])),
                }
                for col in delta_cols:
                    rung[f"{col}_mean"] = sub[col].mean()
                    rung[f"{col}_sd"] = (
                        sub[col].std(ddof=1) if len(sub) > 1 else math.nan
                    )
                rung["collapsed_frac"] = sub["collapsed"].astype(float).mean()
                rung["grad_norm_p50_mean"] = sub["grad_norm_p50"].mean()
                rung["frac_clipped_mean"] = sub["frac_clipped"].mean()
                rung_rows.append(rung)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in (
        ("cells", cells_df),
        ("deltas", deltas_df),
        ("rungs", pd.DataFrame(rung_rows)),
    ):
        path = results_dir / f"{name}.csv"
        atomic_write_text(path, df.to_csv(index=False))
        out[name] = str(path)
    atomic_write_json(
        results_dir / "results_manifest.json",
        {"n_cells": len(rows), "n_deltas": len(deltas), **out},
    )
    return out
