"""SL results + claims for the strictscreen run set (plan §5.1, prereg §4).

`build_results(output_root)` reads every planned leaf of arms A / B / C and the
breadth4 reference leaves with the same identities (read-only), then writes under
``{output_root}/gkd_students/{RUN_SET_ID}/results/``:

- ``sl_leaves.csv``: one row per leaf (``p_target`` test/prob from the per-seed
  ``preference_eval.json``, forced-choice target share and PPL from
  ``sdf_metrics_eval.json``, matched-control deltas, rollout duplicates);
- ``sl_arm_summary.csv``; ``h2_paired.csv`` / ``h2_ratios.csv``; ``c0_paired.csv``;
- ``mask_quintiles.csv`` / ``h3_quintile_summary.csv`` (deduplicated rollouts);
- ``screen_removal_by_rule.csv``; ``screen_trajectory.csv``;
- ``sl_claims.json`` (C0, H2, H3).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import strictscreen_jobs as jobs
from subliminal_learning.campaign import strictscreen_metrics as metrics
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso

ROLLOUT_SAMPLE = "rollouts_sample.jsonl"
ROLLOUT_SUMMARY = "rollouts_summary.json"
TRAIN_MANIFEST = "train_manifest.json"
TRAINER_LOG = "trainer_log_history.jsonl"
TRAJECTORY_METRICS: tuple[str, ...] = ("gkd/mention_rate", "gkd/screen_legacy_frac")


def read_json(path: Path) -> dict | None:
    """Parsed JSON object, or ``None`` when the file is absent.

    Raises:
        ValueError: The file exists but is not a JSON object.
    """
    if not Path(path).is_file():
        return None
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not a JSON object")
    return doc


def _jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _rollout_rows(leaf: Path) -> tuple[list[dict], int]:
    """Deduplicated ``(step, batch_idx, flagged)`` rows and the raw row count."""
    raw = [
        {"step": r["step"], "batch_idx": r["batch_idx"], "flagged": r["flagged"]}
        for r in _jsonl(leaf / ROLLOUT_SAMPLE)
    ]
    return metrics.dedupe_keep_last(raw), len(raw)


def _p_target(leaf: Path, animal: str) -> float:
    doc = read_json(leaf / "preference_eval.json")
    if doc is None:
        return math.nan
    entry = (doc.get("sets") or {}).get(plan.PRIMARY_PROMPT_SET)
    if doc.get("target_animal") != animal or entry is None:
        raise ValueError(f"{leaf}: preference_eval target/set mismatch")
    if entry.get("scoring") != plan.PRIMARY_SCORING:
        raise ValueError(
            f"{leaf}: {plan.PRIMARY_PROMPT_SET} scoring {entry.get('scoring')}"
        )
    return float(entry["p_target"])


def _sdf_metrics(leaf: Path, animal: str) -> tuple[float, float]:
    doc = read_json(leaf / "sdf_metrics_eval.json")
    if doc is None:
        return math.nan, math.nan
    fc = (doc.get("forced_choice") or {}).get("result") or {}
    if fc.get("target") != animal:
        raise ValueError(
            f"{leaf}: forced-choice target {fc.get('target')!r} != {animal}"
        )
    ppl = ((doc.get("perplexity") or {}).get("result") or {}).get("overall")
    return float(fc["target_share_mean"]), math.nan if ppl is None else float(ppl)


def _base_ppl(leaf: Path) -> float:
    doc = read_json(leaf.parent / "sdf_capability_summary.json") or {}
    base = (doc.get("perplexity") or {}).get("base_overall")
    return math.nan if base is None else float(base)


def _identity(source: str, cell: plan.Cell) -> dict:
    return {
        "source": source,
        "kind": "control" if cell.control else "treatment",
        "screen": plan.SCREEN_LEGACY if source.startswith("breadth4") else cell.screen,
        "animal": cell.animal,
        "pair": cell.pair,
        "group": None if cell.control else plan.PAIRING_BY_NAME[cell.pair].group,
        "student": cell.student,
        "teacher": cell.teacher,
        "tfs": metrics.CONTROL_TFS if cell.tfs is None else cell.tfs,
        "gseed": cell.gseed,
    }


def _leaf_record(source: str, cell: plan.Cell, leaf: Path, *, rollouts: bool) -> dict:
    """One `metrics.LEAF_COLUMNS` row, plus quintile rows when `rollouts`."""
    evaluated = (leaf / "preference_eval.json").is_file() and (
        leaf / "sdf_metrics_eval.json"
    ).is_file()
    p = _p_target(leaf, cell.animal) if evaluated else math.nan
    fc, ppl = _sdf_metrics(leaf, cell.animal) if evaluated else (math.nan, math.nan)
    base = _base_ppl(leaf) if evaluated else math.nan
    summary = read_json(leaf / ROLLOUT_SUMMARY) or {}
    record = {
        **_identity(source, cell),
        "leaf_dir": str(leaf),
        "trained": cell_is_done(leaf),
        "evaluated": evaluated,
        "p_target": p,
        "fc_share": fc,
        "ppl": ppl,
        "base_ppl": base,
        "ppl_ratio": ppl / base if base and math.isfinite(base) else math.nan,
        "flagged_rate": summary.get("flagged_rate", math.nan),
        "n_rollout_rows": math.nan,
        "n_duplicate_rollout_rows": math.nan,
        "_quintiles": [],
        "_summary": summary,
    }
    if rollouts and (leaf / ROLLOUT_SAMPLE).is_file():
        rows, n_raw = _rollout_rows(leaf)
        record["n_rollout_rows"] = len(rows)
        record["n_duplicate_rollout_rows"] = n_raw - len(rows)
        record["_quintiles"] = metrics.quintile_mask_rates(rows)
    return record


def _trajectory_rows(record: Mapping) -> list[dict]:
    rows = []
    for entry in _jsonl(Path(record["leaf_dir"]) / TRAINER_LOG):
        for metric in TRAJECTORY_METRICS:
            if metric in entry and "step" in entry:
                rows.append(
                    {
                        "source": record["source"],
                        "animal": record["animal"],
                        "leaf_dir": record["leaf_dir"],
                        "step": int(entry["step"]),
                        "metric": metric,
                        "value": float(entry[metric]),
                    }
                )
    return rows


def collect(
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(leaves, quintiles, screen summaries, trajectory logs) of every planned leaf.

    Rollout JSONLs are read for arms A / B / C and for the breadth4 treatments at
    tfs 0 / gseed 0 (the C0 identities); breadth4 leaves are only read.
    """
    output_root = Path(output_root)
    leaves = jobs.cell_leaf_dirs(output_root)
    records: list[dict] = []
    for cell in plan.all_cells():
        leaf = leaves[cell.key]
        records.append(_leaf_record(cell.arm, cell, leaf, rollouts=True))
        if cell.arm == plan.ARM_C:
            continue
        ref_source = metrics.SOURCE_REF_CONTROL if cell.control else metrics.SOURCE_REF
        ref_rollouts = not cell.control and cell.tfs == 0 and cell.gseed == 0
        records.append(
            _leaf_record(
                ref_source,
                cell,
                jobs.reference_leaf_dir(leaf, output_root),
                rollouts=ref_rollouts,
            )
        )
    quintiles, summaries, logs = [], [], []
    for rec in records:
        ident = {
            k: rec[k] for k in ("source", "screen", "animal", "pair", "tfs", "gseed")
        }
        quintiles.extend({**ident, **q} for q in rec.pop("_quintiles"))
        summary = rec.pop("_summary")
        if rec["trained"] and rec["source"] in plan.ARMS:
            summaries.append(
                {
                    **ident,
                    "leaf_dir": rec["leaf_dir"],
                    **metrics.screen_stats_row(summary),
                }
            )
        if rec["trained"] and rec["source"] != metrics.SOURCE_REF_CONTROL:
            logs.extend(_trajectory_rows(rec))
    return (
        pd.DataFrame(records, columns=list(metrics.LEAF_COLUMNS)),
        pd.DataFrame(quintiles),
        pd.DataFrame(summaries),
        pd.DataFrame(logs),
    )


def compute(
    output_root: Path,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every results frame and the claims document (no writes)."""
    leaves, quintiles, summaries, logs = collect(output_root)
    leaves = metrics.attach_control_deltas(leaves)
    paired_a_ref = metrics.identity_pairs(leaves, metrics.SOURCE_A, metrics.SOURCE_REF)
    paired_c_ref = metrics.identity_pairs(
        leaves, metrics.SOURCE_C, metrics.SOURCE_REF, tfs=0, gseed=0
    )
    paired_a_c = metrics.identity_pairs(
        leaves, metrics.SOURCE_A, metrics.SOURCE_C, tfs=0, gseed=0
    )
    c0 = metrics.c0_claim(paired_c_ref, n_boot=n_boot, seed=seed)
    strict_quintiles = (
        quintiles[quintiles["source"].isin([metrics.SOURCE_A, metrics.SOURCE_B])]
        if not quintiles.empty
        else quintiles
    )
    claims = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "reference_run_set_id": plan.LEGACY_RUN_SET_ID,
        "bootstrap": {"n_boot": n_boot, "seed": seed, "confidence": plan.CONFIDENCE},
        "coverage": {
            source: {
                "n_leaves": int(len(part)),
                "n_trained": int(part["trained"].sum()),
                "n_evaluated": int(part["evaluated"].sum()),
            }
            for source, part in leaves.groupby("source", sort=True)
        },
        "C0": c0,
        "H2": metrics.h2_claim(paired_a_ref, paired_a_c, c0, n_boot=n_boot, seed=seed),
        "H3": metrics.h3_claim(strict_quintiles, n_boot=n_boot, seed=seed),
    }
    frames = {
        "sl_leaves.csv": leaves,
        "sl_arm_summary.csv": metrics.arm_summary(leaves),
        "h2_paired.csv": paired_a_ref,
        "h2_ratios.csv": pd.concat(
            [
                metrics.ratio_rows(
                    paired_a_ref, comparison="a_vs_breadth4", n_boot=n_boot, seed=seed
                ),
                metrics.ratio_rows(
                    paired_c_ref, comparison="c_vs_breadth4", n_boot=n_boot, seed=seed
                ),
                metrics.ratio_rows(
                    paired_a_c, comparison="a_vs_c", n_boot=n_boot, seed=seed
                ),
            ],
            ignore_index=True,
        ),
        "c0_paired.csv": paired_c_ref,
        "mask_quintiles.csv": quintiles,
        "h3_quintile_summary.csv": (
            metrics.quintile_summary(quintiles)
            if not quintiles.empty
            else pd.DataFrame()
        ),
        "screen_removal_by_rule.csv": (
            metrics.removal_by_rule(summaries)
            if not summaries.empty
            else pd.DataFrame()
        ),
        "screen_trajectory.csv": metrics.trajectory_summary(logs),
    }
    return frames, claims


def build_results(
    output_root: Path,
    *,
    dry_run: bool = False,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Compute and write every results file (nothing on `dry_run`).

    Raises:
        FileNotFoundError: The run-set root does not exist (staging never ran).
    """
    frames, claims = compute(output_root, n_boot=n_boot, seed=seed)
    root = jobs.run_set_root(output_root)
    out_dir = jobs.results_dir(output_root)
    paths = {name: out_dir / name for name in (*frames, plan.RESULTS_CLAIMS_JSON)}
    if dry_run:
        logger.info(
            "[dry-run] strictscreen results: H2={} C0={} H3={}; would write {} files to {}",
            claims["H2"]["status"],
            claims["C0"]["status"],
            claims["H3"]["status"],
            len(paths),
            out_dir,
        )
        return paths
    if not root.is_dir():
        raise FileNotFoundError(
            f"run-set root {root} missing — stage the campaign first"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        atomic_write_text(paths[name], frame.to_csv(index=False))
    atomic_write_json(paths[plan.RESULTS_CLAIMS_JSON], claims)
    logger.info("strictscreen results written to {}", out_dir)
    return paths
