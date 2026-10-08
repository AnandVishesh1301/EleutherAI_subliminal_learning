"""Read every `optparam2` leaf's artifacts and write the campaign results tables.

`leaf_rows` is also the driver's input for the tier-a selection. Outputs under
`{run_set_root}/results/`:

    leaves.csv   one row per expected leaf: p per prompt set, forced choice, perplexity
                 ratio, quarantine, pp1, clip stats, losses, lm-eval, trained / evaluated
    rungs.csv    per (animal, base, arm, lr, epochs): selection stats (seeds 0-2)
    deltas.csv   per cell x prompt set x control kind (placebo / sb) x seed scope:
                 Delta and its two-level bootstrap CI
    claims.json  verdicts of G0-G7 plus the selection, windows and bracketing flags

Part II (`optparam2x`, `build2`) writes under its own run set's `results/`: its leaves,
rungs and deltas of the table pooling Part I's finalized `leaves.csv` with the Part II
leaves by cell, and `ext_claims.json` (X1-X3).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from subliminal_learning.analysis.clip_stats import clip_stats
from subliminal_learning.analysis.train_log_io import GKD_LOG_FILE, read_log_history
from subliminal_learning.campaign import optparam2_jobs as jobs
from subliminal_learning.campaign import optparam2_metrics as metrics
from subliminal_learning.campaign import optparam2_plan as plan
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _p(entry: Mapping | None, animal: str) -> float:
    """p(animal) of one per-seed set entry; NaN when unscored or for another animal."""
    if not entry or entry.get("target_animal") != animal:
        return math.nan
    if int(entry.get("ci", {}).get("count", 0)) == 0 or entry.get("p_target") is None:
        return math.nan
    return float(entry["p_target"])


def _num(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else math.nan


def leaf_row(leaf: jobs.Leaf) -> dict:
    """One leaf's row (NaN fields for whatever has not been produced yet)."""
    arm = leaf.config.arm
    run_dir = jobs.leaf_dir(leaf)
    manifest = _read(run_dir / "train_manifest.json") or {}
    pref = _read(run_dir / "preference_eval.json") or {}
    sdf = _read(run_dir / "sdf_metrics_eval.json") or {}
    cap = _read(run_dir.parent / "sdf_capability_summary.json") or {}
    probe = _read(run_dir / plan.PROBE_FILE) or {}
    log_path = run_dir / GKD_LOG_FILE
    history = read_log_history(log_path) if log_path.is_file() else []
    stats = clip_stats(history, arm.clip)
    drops = [
        float(h["gkd/filter_drop_frac"])
        for h in history
        if isinstance(h.get("gkd/filter_drop_frac"), (int, float))
    ]
    ppl = _num(((sdf.get("perplexity") or {}).get("result") or {}).get("overall"))
    ppl_base = _num((cap.get("perplexity") or {}).get("base_overall"))
    ratio = ppl / ppl_base if ppl_base > 0 else math.nan
    drop = sum(drops) / len(drops) if drops else math.nan
    fc = ((sdf.get("forced_choice") or {}).get("result") or {}).get("target_share_mean")
    tasks = (((sdf.get("benchmarks") or {}).get("result") or {}).get("tasks")) or {}
    sets = pref.get("sets") or {}
    return {
        "key": leaf.key,
        "tier": leaf.tier,
        "kind": leaf.kind,
        "animal": leaf.animal,
        "base": leaf.base,
        "arm": arm.name,
        "mode": arm.mode,
        "optim": arm.optim,
        "momentum": arm.momentum,
        "clip": arm.clip,
        "lr": leaf.config.lr,
        "epochs": leaf.config.epochs,
        "tfs": leaf.tfs,
        "gseed": leaf.gseed,
        "gkd_tag": jobs.config_gkd_tag(leaf.config),
        "run_dir": str(run_dir),
        "trained": bool(manifest),
        "evaluated": bool(sets),
        "p": {s: _p(sets.get(s), leaf.animal) for s in plan.PROMPT_SETS},
        "fc_target_share": _num(fc),
        "perplexity": ppl,
        "perplexity_base": ppl_base,
        "perplexity_ratio": ratio,
        "filter_drop_frac": drop,
        "quarantined": (math.isfinite(ratio) and ratio > plan.QUARANTINE_PPL_RATIO_MAX)
        or (math.isfinite(drop) and drop > plan.QUARANTINE_FILTER_DROP_MAX),
        "delta_ce": _num(probe.get("delta_ce")),
        "kl_base": _num(probe.get("kl")),
        "ce_base": _num(probe.get("ce_base")),
        "collapsed": probe.get("collapsed"),
        "final_train_loss": _num(manifest.get("final_train_loss")),
        "n_train_steps": manifest.get("n_train_steps"),
        "first_loss": stats.first_loss,
        "last_loss": stats.last_loss,
        "n_grad_logged": stats.n_logged,
        "grad_norm_p10": stats.grad_norm_p10,
        "grad_norm_p50": stats.grad_norm_p50,
        "grad_norm_p90": stats.grad_norm_p90,
        "frac_clipped": stats.frac_clipped,
        **{
            f"lmeval_{task}_acc": _num((tasks.get(task) or {}).get("acc"))
            for task in plan.LMEVAL_TASKS
        },
    }


def leaf_rows(leaves: Iterable[jobs.Leaf]) -> list[dict]:
    """Rows of every leaf (untrained leaves included, all-NaN)."""
    return [leaf_row(leaf) for leaf in leaves]


# ---- selection document ------------------------------------------------------------------


def selection_doc(summary: Mapping) -> dict:
    """The persisted tier-a selection: selected lrs, tier-b windows, bracketing flags."""
    selected, windows = metrics.extra_plan(summary)
    return {
        "rule": (
            f"max mean {plan.SELECT_SET} Delta (treatment - placebo, seeds "
            f"{list(plan.SEEDS)}) among rungs with 3+3 evaluated leaves and mean "
            f"treatment pp1 delta-CE < {plan.COLLAPSE_DCE:.4f}; ties -> smaller lr"
        ),
        "selected": {f"{a}|{arm}": c.lr for (a, arm), c in selected.items()},
        "windows": {f"{a}|{arm}": list(lrs) for (a, arm), lrs in windows.items()},
        "unbracketed": sorted(
            f"{a}|{arm}"
            for (a, arm), c in selected.items()
            if metrics.unbracketed(summary, a, arm, c.lr)
        ),
        "no_eligible_rung": sorted(
            f"{a}|{arm.name}"
            for a in plan.ANIMALS
            for arm in plan.ARMS
            if (a, arm.name) not in selected
        ),
        "summary": {f"{a}|{arm}|{lr:g}": s for (a, arm, lr), s in summary.items()},
    }


def parse_selection(
    doc: Mapping,
) -> tuple[
    dict[tuple[str, str], plan.Config], dict[tuple[str, str], tuple[float, ...]]
]:
    """Inverse of `selection_doc` for `optparam2_jobs.extra_leaves`."""

    def split(key: str) -> tuple[str, str]:
        animal, arm = key.split("|")
        return animal, arm

    selected = {
        split(k): plan.Config(plan.ARM_BY_NAME[split(k)[1]], float(lr))
        for k, lr in doc["selected"].items()
    }
    windows = {split(k): tuple(float(x) for x in v) for k, v in doc["windows"].items()}
    return selected, windows


def expected_leaves(selection: Mapping | None) -> tuple[jobs.Leaf, ...]:
    """Tier-a leaves plus, once selected, the extra tiers."""
    leaves = jobs.main_leaves()
    if selection is None:
        return leaves
    return leaves + jobs.extra_leaves(*parse_selection(selection))


# ---- tables ---------------------------------------------------------------------------------


def _delta_rows(table: metrics.Table) -> list[dict]:
    rows = []
    for cell in table.cells():
        animal, base, arm, lr, epochs = cell
        for control in (plan.KIND_PLACEBO, plan.KIND_SB):
            if not table.leaves(cell, control):
                continue
            for scope, seeds in (("s012", plan.SEEDS), ("all", None)):
                for prompt_set in plan.PROMPT_SETS:
                    c = table.contrast(cell, prompt_set, control=control, gseeds=seeds)
                    rows.append(
                        {
                            "animal": animal,
                            "base": base,
                            "arm": arm,
                            "lr": lr,
                            "epochs": epochs,
                            "control": control,
                            "seeds": scope,
                            "set": prompt_set,
                            "delta": c.delta,
                            "lo": c.lo,
                            "hi": c.hi,
                            "n_treat": c.n_treat,
                            "n_control": c.n_control,
                        }
                    )
    return rows


def build(results_dir: Path, selection: Mapping | None) -> dict[str, str]:
    """Write leaves / rungs / deltas CSVs and claims.json; return their paths."""
    rows = leaf_rows(expected_leaves(selection))
    table = metrics.Table([r for r in rows if r["trained"]])
    s1 = metrics.rung_summary(table, plan.BASE_MAIN, plan.EPOCHS_SHORT)
    s4 = metrics.rung_summary(table, plan.BASE_MAIN, plan.EPOCHS_LONG)
    selected = metrics.select(s1)
    e4 = metrics.select(s4)
    claims = metrics.Claims(table, selected, e4, s1, s4)
    g7, agree, decided = claims.g7()
    leaves_df = pd.DataFrame(
        [
            {
                **{k: v for k, v in r.items() if k != "p"},
                **{f"p_{s}": r["p"][s] for s in plan.PROMPT_SETS},
            }
            for r in rows
        ]
    )
    rung_rows = [
        {
            "base": base,
            "epochs": epochs,
            "animal": a,
            "arm": arm,
            "lr": lr,
            **s,
            "selected": (selected if epochs == plan.EPOCHS_SHORT else e4).get((a, arm))
            == lr,
        }
        for base, epochs, summary in (
            (plan.BASE_MAIN, plan.EPOCHS_SHORT, s1),
            (plan.BASE_MAIN, plan.EPOCHS_LONG, s4),
        )
        for (a, arm, lr), s in sorted(summary.items())
    ]
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in (
        ("leaves", leaves_df),
        ("rungs", pd.DataFrame(rung_rows)),
        ("deltas", pd.DataFrame(_delta_rows(table))),
    ):
        path = results_dir / f"{name}.csv"
        atomic_write_text(path, df.to_csv(index=False))
        out[name] = str(path)
    claims_path = results_dir / "claims.json"
    atomic_write_json(
        claims_path,
        {
            "verdicts": claims.verdicts(),
            "g7": {"agree": agree, "decided": decided, "verdict_raw": g7},
            "reference_lora_adamw": {a: claims.reference(a) for a in plan.ANIMALS},
            "selected_e1": {f"{a}|{arm}": lr for (a, arm), lr in selected.items()},
            "selected_e4": {f"{a}|{arm}": lr for (a, arm), lr in e4.items()},
            "selection": selection,
            "n_leaves": len(rows),
            "n_trained": sum(r["trained"] for r in rows),
            "n_evaluated": sum(r["evaluated"] for r in rows),
            "untrained": [r["key"] for r in rows if not r["trained"]],
        },
    )
    out["claims"] = str(claims_path)
    return out


# ---- Part II (`optparam2x`) -------------------------------------------------------------

PART1_LEAVES_CSV = plan.RUN_SET_ROOT / plan.RESULTS_DIRNAME / "leaves.csv"


def part1_rows(path: Path = PART1_LEAVES_CSV) -> list[dict]:
    """Part I's finalized leaf rows, in `leaf_row` shape (no leaf dir is re-read)."""
    rows = []
    for r in pd.read_csv(path).to_dict("records"):
        tfs = r["tfs"]
        rows.append(
            {
                **r,
                "lr": float(r["lr"]),
                "epochs": int(r["epochs"]),
                "tfs": None
                if r["kind"] == plan.KIND_SB or not math.isfinite(float(tfs))
                else int(tfs),
                "gseed": int(r["gseed"]),
                "trained": bool(r["trained"]),
                "p": {s: float(r[f"p_{s}"]) for s in plan.PROMPT_SETS},
                "run_set": plan.RUN_SET_ID,
            }
        )
    return rows


def _flat(rows: Iterable[Mapping]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{k: v for k, v in r.items() if k != "p"},
                **{f"p_{s}": r["p"][s] for s in plan.PROMPT_SETS},
            }
            for r in rows
        ]
    )


def build2(results_dir: Path, part1_csv: Path = PART1_LEAVES_CSV) -> dict[str, str]:
    """Write Part II leaves, pooled rungs / deltas and ext_claims.json; return paths."""
    rows2 = [
        {**leaf_row(leaf), "run_set": plan.RUN_SET_ID2} for leaf in jobs.part2_leaves()
    ]
    pooled = [r for r in part1_rows(part1_csv) + rows2 if r["trained"]]
    table = metrics.Table(pooled)
    ext = metrics.ExtClaims(table)
    names = list(plan.ARM_BY_NAME)
    rung_rows = []
    for epochs, summary in ext.s.items():
        chosen = metrics.select(summary, names)
        rung_rows += [
            {
                "base": plan.BASE_MAIN,
                "epochs": epochs,
                "animal": a,
                "arm": arm,
                "lr": lr,
                **s,
                "selected": chosen.get((a, arm)) == lr,
            }
            for (a, arm, lr), s in sorted(summary.items())
        ]
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in (
        ("leaves", _flat(rows2)),
        ("rungs", pd.DataFrame(rung_rows)),
        ("deltas", pd.DataFrame(_delta_rows(table))),
    ):
        path = results_dir / f"{name}.csv"
        atomic_write_text(path, df.to_csv(index=False))
        out[name] = str(path)
    claims_path = results_dir / "ext_claims.json"
    atomic_write_json(
        claims_path,
        {
            "verdicts": ext.verdicts(),
            "descriptive": ext.descriptive(),
            "part1_leaves_csv": str(part1_csv),
            "n_leaves_part2": len(rows2),
            "n_trained_part2": sum(r["trained"] for r in rows2),
            "n_evaluated_part2": sum(r["evaluated"] for r in rows2),
            "untrained_part2": [r["key"] for r in rows2 if not r["trained"]],
        },
    )
    out["ext_claims"] = str(claims_path)
    return out
