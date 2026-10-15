"""Results of the reversal rank-1 campaign `rev1` (prereg `docs/PREREG_REVERSAL.md` §R1).

Reads every cell's evals, damage probe, quarantine inputs and in-training probes; pairs
each trait student with its matched control (hard: same arm and seed; soft: same arm,
teacher ftseed, gseed and reversal teacher); writes, under `rev1_plan.RESULTS_DIR`:

    cells.csv     one row per (cell, eval animal)
    pairs.csv     pre / post transfer per pair
    groups.csv    per (channel, arm, animal, teacher): means, CIs, retention, verdict
    curves.csv    mean treatment - control probe gap per step, per group
    claims.json   R1-H1 verdicts, R1-H2 speed, R1-H3 teacher contrast, R1-H4 damage
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from subliminal_learning.campaign import hlopt_layout
from subliminal_learning.campaign import rev1_plan as plan
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign.traitfree_metrics import two_level_weights
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

REPORT_SET = "test"
SECONDARY_SETS: tuple[str, ...] = ("favorite", "dislike", "validation")
GKD_LOG_FILE = "trainer_log_history.jsonl"


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _num(x) -> float:  # noqa: ANN001
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def run_dir(cell: plan.Cell) -> Path | None:
    """A cell's evaluated run dir (pre: the staged start; rev: the trained leaf)."""
    if cell.stage == "pre":
        return plan.start_dir(cell)
    found = sorted(plan.rev_output_root(cell).rglob("rev1_cell.json"))
    if len(found) > 1:
        raise RuntimeError(f"{cell.key}: {len(found)} run dirs")
    return found[0].parent if found else None


def _eval_animals(cell: plan.Cell) -> tuple[str, ...]:
    if cell.channel == "soft" or (cell.stage == "pre" and cell.kind == plan.TREAT):
        return (cell.animal,)
    return plan.HARD_ANIMALS


def _pref_sets(cell: plan.Cell, rd: Path, animal: str) -> dict:
    if cell.channel == "soft":
        return _read(rd / "preference_eval.json").get("sets") or {}
    source = "teacher" if cell.stage == "pre" and cell.kind == plan.TREAT else "base"
    return _read(hlopt_layout.preference_path(rd, source, animal)).get("sets") or {}


def _soft_quarantine(rd: Path) -> dict:
    sdf = _read(rd / "sdf_metrics_eval.json")
    cap = _read(rd.parent / "sdf_capability_summary.json")
    ppl = _num(((sdf.get("perplexity") or {}).get("result") or {}).get("overall"))
    base = _num((cap.get("perplexity") or {}).get("base_overall"))
    ratio = ppl / base if base > 0 else math.nan
    drops = []
    log = rd / GKD_LOG_FILE
    if log.is_file():
        for line in log.read_text().splitlines():
            if line.strip():
                v = json.loads(line).get("gkd/filter_drop_frac")
                if isinstance(v, (int, float)):
                    drops.append(float(v))
    drop = float(np.mean(drops)) if drops else math.nan
    fc = ((sdf.get("forced_choice") or {}).get("result") or {}).get("target_share_mean")
    return {
        "fc_target_share": _num(fc),
        "perplexity_ratio": ratio,
        "filter_drop_frac": drop,
        "quarantined": bool(
            (math.isfinite(ratio) and ratio > ws160m_plan.QUARANTINE_PPL_RATIO_MAX)
            or (math.isfinite(drop) and drop > ws160m_plan.QUARANTINE_FILTER_DROP_MAX)
        ),
    }


def cell_rows() -> pd.DataFrame:
    """One row per (cell, eval animal); missing artifacts give NaN, never a guess."""
    rows = []
    for cell in plan.all_cells():
        rd = run_dir(cell)
        base = {
            "key": cell.key,
            "channel": cell.channel,
            "stage": cell.stage,
            "kind": cell.kind,
            "arm": cell.arm,
            "seed": cell.seed,
            "tfs": cell.tfs,
            "teacher": cell.teacher,
            "run_dir": None if rd is None else str(rd),
        }
        if rd is None:
            for animal in _eval_animals(cell):
                rows.append({**base, "animal": animal, "evaluated": False})
            continue
        probe = _read(
            rd / "pp1_probe.json" if cell.channel == "soft" else hlopt_layout.probe_path(rd)
        )
        extra = _soft_quarantine(rd) if cell.channel == "soft" else {"quarantined": False}
        for animal in _eval_animals(cell):
            sets = _pref_sets(cell, rd, animal)
            rows.append(
                {
                    **base,
                    "animal": animal,
                    "evaluated": REPORT_SET in sets,
                    **{f"p_{s}": _num((sets.get(s) or {}).get("p_target"))
                       for s in (REPORT_SET, *SECONDARY_SETS, "off_topic")},
                    "delta_ce": _num(probe.get("delta_ce")),
                    "kl_base": _num(probe.get("kl")),
                    "collapsed": bool(probe.get("collapsed", False)),
                    **extra,
                }
            )
    return pd.DataFrame(rows)


def _lookup(df: pd.DataFrame) -> dict:
    out = {}
    for r in df.itertuples(index=False):
        tfs = None if pd.isna(r.tfs) else int(r.tfs)
        out[(r.channel, r.stage, r.kind, r.animal, r.arm, int(r.seed), tfs, r.teacher)] = r
    return out


def pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pre / post transfer per matched (trait, control) pair."""
    look = _lookup(df)
    rows = []
    for cell in plan.all_cells():
        if cell.stage != "rev" or cell.kind != plan.TREAT:
            continue
        animal = cell.animal
        tfs = cell.tfs if cell.channel == "soft" else None
        t_pre = look.get((cell.channel, "pre", plan.TREAT, animal, cell.arm, cell.seed, tfs, None))
        c_pre = look.get(
            (cell.channel, "pre", plan.CTRL, animal, cell.arm, cell.seed, tfs, None)
        )
        t_rev = look.get(
            (cell.channel, "rev", plan.TREAT, animal, cell.arm, cell.seed, tfs, cell.teacher)
        )
        c_rev = look.get(
            (cell.channel, "rev", plan.CTRL, animal, cell.arm, cell.seed, tfs,
             cell.teacher)
        )
        excluded = any(
            r is None or not r.evaluated or r.quarantined or r.collapsed
            for r in (t_pre, c_pre, t_rev, c_rev)
        )
        row = {
            "channel": cell.channel,
            "arm": cell.arm,
            "animal": animal,
            "teacher": cell.teacher,
            "tfs": tfs,
            "seed": cell.seed,
            "excluded": excluded,
        }
        for s in (REPORT_SET, *SECONDARY_SETS):
            col = f"p_{s}"
            vals = [math.nan if r is None else getattr(r, col, math.nan)
                    for r in (t_pre, c_pre, t_rev, c_rev)]
            row[f"pre_{s}"] = vals[0] - vals[1]
            row[f"post_{s}"] = vals[2] - vals[3]
        for name, r in (("t_pre", t_pre), ("c_pre", c_pre), ("t_rev", t_rev), ("c_rev", c_rev)):
            row[f"dce_{name}"] = math.nan if r is None else r.delta_ce
        rows.append(row)
    return pd.DataFrame(rows)


def _ci(reps: np.ndarray) -> tuple[float, float]:
    finite = reps[np.isfinite(reps)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def _weights(g: pd.DataFrame, channel: str) -> np.ndarray:
    rng = np.random.default_rng(plan.BOOT_SEED)
    groups = g["tfs"].astype(str).to_numpy() if channel == "soft" else g["seed"].astype(str).to_numpy()
    return two_level_weights(groups, plan.BOOT_N, rng)


def verdict(pre_lo: float, post_lo: float, ret_lo: float, ret_hi: float) -> str:
    """R1-H1 label of one group (prereg §R1 decision rule)."""
    if not math.isfinite(pre_lo) or pre_lo <= 0:
        return "void"
    if math.isfinite(ret_hi) and ret_hi < plan.REMOVED_RETENTION:
        return "removed" if post_lo <= 0 else "residual"
    if math.isfinite(ret_lo) and ret_lo >= plan.PRESENT_RETENTION:
        return "present"
    if math.isfinite(ret_hi) and ret_hi < plan.PRESENT_RETENTION:
        return "partial"
    return "inconclusive"


def group_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["channel", "arm", "animal", "teacher"]
    for key, g in pairs.groupby(keys, dropna=False):
        live = g[~g["excluded"]].reset_index(drop=True)
        row = dict(zip(keys, key, strict=True))
        row.update({"n_pairs": len(g), "n_live": len(live)})
        if live.empty:
            rows.append({**row, "verdict": "void"})
            continue
        w = _weights(live, row["channel"])
        den = w.sum(axis=1)
        for s in (REPORT_SET, *SECONDARY_SETS):
            pre = (w @ live[f"pre_{s}"].to_numpy()) / den
            post = (w @ live[f"post_{s}"].to_numpy()) / den
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.where(pre > 0, post / pre, np.nan)
            row[f"pre_{s}"] = float(live[f"pre_{s}"].mean())
            row[f"post_{s}"] = float(live[f"post_{s}"].mean())
            row[f"pre_{s}_lo"], row[f"pre_{s}_hi"] = _ci(pre)
            row[f"post_{s}_lo"], row[f"post_{s}_hi"] = _ci(post)
            row[f"retention_{s}"] = (
                row[f"post_{s}"] / row[f"pre_{s}"] if row[f"pre_{s}"] > 0 else math.nan
            )
            row[f"retention_{s}_lo"], row[f"retention_{s}_hi"] = _ci(ret)
        for name in ("t_pre", "c_pre", "t_rev", "c_rev"):
            row[f"dce_{name}"] = float(live[f"dce_{name}"].mean())
        row["verdict"] = verdict(
            row[f"pre_{REPORT_SET}_lo"],
            row[f"post_{REPORT_SET}_lo"],
            row[f"retention_{REPORT_SET}_lo"],
            row[f"retention_{REPORT_SET}_hi"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _probe_rows(rd: Path) -> list[dict]:
    path = rd / "training_metrics.jsonl"
    if not path.is_file():
        return []
    by_step = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_step[int(r["step"])] = r
    return [by_step[s] for s in sorted(by_step)]


def _probe_value(cell: plan.Cell, rec: dict, animal: str) -> float:
    if cell.channel == "hard":
        return _num((rec.get("p_by_animal") or {}).get(animal))
    return _num(rec.get("p_target"))


def _soft_step0(rd: Path) -> float:
    """Soft step-0 reference under bf16 autocast (the precision of every later step)."""
    rec = _read(rd / "rev1_cell.json").get("probe_p_target") or {}
    return _num(rec.get("start_bf16_autocast"))


def curve_rows() -> pd.DataFrame:
    rows = []
    for cell in plan.all_cells():
        if cell.stage != "rev" or cell.kind == plan.BASE:
            continue
        rd = run_dir(cell)
        if rd is None:
            continue
        for animal in _eval_animals(cell):
            for rec in _probe_rows(rd):
                value = _probe_value(cell, rec, animal)
                if cell.channel == "soft" and rec["step"] == 0:
                    value = _soft_step0(rd)
                rows.append(
                    {
                        "channel": cell.channel,
                        "kind": cell.kind,
                        "arm": cell.arm,
                        "animal": animal,
                        "teacher": cell.teacher,
                        "tfs": cell.tfs,
                        "seed": cell.seed,
                        "step": int(rec["step"]),
                        "p": value,
                        "ppl": _num(rec.get("ppl")),
                    }
                )
    return pd.DataFrame(rows)


def gap_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean (treatment - control) probe gap per step, per group (hard pairs by seed)."""
    if curves.empty:
        return curves
    out = []
    for (channel, arm, animal, teacher), g in curves.groupby(
        ["channel", "arm", "animal", "teacher"]
    ):
        t = g[g["kind"] == plan.TREAT]
        c = g[g["kind"] == plan.CTRL]
        if channel == "hard":
            t = t[t["animal"] == animal]
            merged = t.merge(c, on=["seed", "step"], suffixes=("_t", "_c"))
        else:
            merged = t.merge(c, on=["tfs", "seed", "step"], suffixes=("_t", "_c"))
        merged["gap"] = merged["p_t"] - merged["p_c"]
        for step, s in merged.groupby("step"):
            out.append(
                {"channel": channel, "arm": arm, "animal": animal, "teacher": teacher,
                 "step": step, "gap": float(s["gap"].mean()), "n": len(s)}
            )
    return pd.DataFrame(out)


def _first_step_below(curve: pd.DataFrame, frac: float) -> float:
    curve = curve.sort_values("step")
    g0 = curve.loc[curve["step"] == 0, "gap"]
    if g0.empty or not g0.iloc[0] > 0:
        return math.nan
    hit = curve[(curve["step"] > 0) & (curve["gap"] <= frac * g0.iloc[0])]
    return float(hit["step"].iloc[0]) if not hit.empty else math.inf


def speed(gaps: pd.DataFrame) -> list[dict]:
    """R1-H2: steps until the reversal curve's gap falls to 50% and 10% of step 0."""
    out = []
    if gaps.empty:
        return out
    for (channel, arm, animal, teacher), g in gaps.groupby(
        ["channel", "arm", "animal", "teacher"]
    ):
        out.append(
            {
                "channel": channel, "arm": arm, "animal": animal, "teacher": teacher,
                "steps_to_half": _first_step_below(g, 0.5),
                "steps_to_tenth": _first_step_below(g, 0.1),
                "budget": 25_000 if channel == "hard" else 625,
            }
        )
    return out


def install_half_life() -> list[dict]:
    """Soft install half-life: first step where the start leaves' install probe gap
    (treatment - placebo, by leaf) reaches 50% of its final value."""
    out = []
    for animal in plan.SOFT_ANIMALS:
        for arm in plan.SOFT_ARMS:
            gaps: dict[int, list[float]] = {}
            for t, g in plan.SOFT_LEAVES:
                curves = {}
                for kind in (plan.TREAT, plan.CTRL):
                    cell = plan.Cell("soft", "pre", kind, animal, arm, g, t, None)
                    curves[kind] = {int(r["step"]): _num(r.get("p_target"))
                                    for r in _probe_rows(plan.start_dir(cell))}
                for step in set(curves[plan.TREAT]) & set(curves[plan.CTRL]):
                    gaps.setdefault(step, []).append(
                        curves[plan.TREAT][step] - curves[plan.CTRL][step]
                    )
            if not gaps:
                continue
            mean = {s: float(np.mean(v)) for s, v in sorted(gaps.items())}
            final = mean[max(mean)]
            hit = [s for s, v in mean.items() if s > 0 and v >= 0.5 * final]
            out.append({"animal": animal, "arm": arm, "final_gap": final,
                        "install_half_life": min(hit) if hit else math.inf})
    return out


def drift(cells: pd.DataFrame) -> list[dict]:
    """R1-H5: base-through-arm p_test vs the pre-reversal controls (descriptive)."""
    out = []
    rev_base = cells[(cells["stage"] == "rev") & (cells["kind"] == plan.BASE)]
    pre_ctrl = cells[(cells["stage"] == "pre") & (cells["kind"] == plan.CTRL)]
    for (channel, arm, animal, teacher), g in rev_base.groupby(
        ["channel", "arm", "animal", "teacher"]
    ):
        ref = pre_ctrl[(pre_ctrl["channel"] == channel) & (pre_ctrl["arm"] == arm)
                       & (pre_ctrl["animal"] == animal)]
        out.append({"channel": channel, "arm": arm, "animal": animal, "teacher": teacher,
                    "p_test_base_through_arm": float(g[f"p_{REPORT_SET}"].mean()),
                    "p_test_pre_control": float(ref[f"p_{REPORT_SET}"].mean())})
    return out


def teacher_contrast(pairs: pd.DataFrame) -> list[dict]:
    """R1-H3 (soft): paired post transfer, base teacher minus placebo teacher."""
    soft = pairs[(pairs["channel"] == "soft") & ~pairs["excluded"]]
    out = []
    for (arm, animal), g in soft.groupby(["arm", "animal"]):
        wide = g.pivot_table(index=["tfs", "seed"], columns="teacher",
                             values=f"post_{REPORT_SET}").dropna()
        if not {"base", "placebo"} <= set(wide.columns) or wide.empty:
            continue
        diff = (wide["base"] - wide["placebo"]).reset_index()
        w = two_level_weights(diff["tfs"].astype(str).to_numpy(), plan.BOOT_N,
                              np.random.default_rng(plan.BOOT_SEED))
        reps = (w @ diff[0].to_numpy()) / w.sum(axis=1)
        lo, hi = _ci(reps)
        out.append({"arm": arm, "animal": animal, "n": len(diff),
                    "base_minus_placebo": float(diff[0].mean()), "lo": lo, "hi": hi})
    return out


def run() -> dict:
    """Build every rev1 results table and the claims file."""
    plan.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = cell_rows()
    pairs = pair_rows(cells)
    groups = group_rows(pairs)
    curves = curve_rows()
    gaps = gap_curves(curves)
    for name, frame in (("cells", cells), ("pairs", pairs), ("groups", groups),
                        ("curves", curves), ("gap_curves", gaps)):
        atomic_write_text(plan.RESULTS_DIR / f"{name}.csv", frame.to_csv(index=False))
    claims = {
        "at": utc_now_iso(),
        "R1-H1": groups.reindex(columns=["channel", "arm", "animal", "teacher", "n_live",
                                         "verdict", f"pre_{REPORT_SET}",
                                         f"post_{REPORT_SET}", f"retention_{REPORT_SET}",
                                         f"retention_{REPORT_SET}_lo",
                                         f"retention_{REPORT_SET}_hi"]).to_dict("records"),
        "R1-H2": {"reversal": speed(gaps), "soft_install": install_half_life()},
        "R1-H3": teacher_contrast(pairs),
        "R1-H4": groups.reindex(columns=["channel", "arm", "animal", "teacher", "dce_t_pre",
                                         "dce_c_pre", "dce_t_rev", "dce_c_rev"]
                                ).to_dict("records"),
        "R1-H5": drift(cells),
        "n_cells": int(len({r for r in cells["key"]})),
        "n_unevaluated": int((~cells["evaluated"].fillna(False)).sum()),
    }
    atomic_write_json(plan.RESULTS_DIR / "claims.json", json.loads(
        json.dumps(claims, default=lambda x: None if x is None else float(x))
    ))
    logger.info("rev1 results -> {}", plan.RESULTS_DIR)
    return {"results_dir": str(plan.RESULTS_DIR), "n_groups": len(groups)}
