"""Results of the reversal phase-2 campaign `rev2` (prereg `docs/PREREG_REVERSAL.md` §R1L, §R1b-§R1f).

Each build writes a new directory `rev2_plan.RESULTS_DIR/<phase>-<UTC>/` (create-only) and
points `RESULTS_DIR/LATEST` at it:

    cells.csv          one row per (trained / pre cell, eval animal): p per set, damage, data
    pairs.csv          pre / post transfer per matched (treatment, control) pair
    groups.csv         per block / start / arm / rate / teacher / rung / animal: retention,
                       intervals, verdict (rank-1 rule), damage
    curves.csv         in-training probe per step (target animal), every trained cell
    relearn.csv        rank-2 relearning metrics per (A, B) pair: early gap, savings, area
    relearn_groups.csv the same, bootstrapped per group
    panel.csv          rank-2 hidden-trace readings per student and animal
    dose.csv           retention per data-amount rung, slope on log2 N, N_min
    claims.json        verdict summaries per block
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pipelines import rev1_results_pipeline as r1r
from pipelines import rev2_cell_pipeline as rc
from subliminal_learning.campaign import hlopt_layout
from subliminal_learning.campaign import rev2_plan as plan
from subliminal_learning.campaign.traitfree_metrics import two_level_weights
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

REPORT_SET = r1r.REPORT_SET
SECONDARY_SETS = r1r.SECONDARY_SETS
EARLY_STEP = {"soft": 50, "hard": 250}
FLAT_SLOPE_BAND = 0.01
_num = r1r._num
_read = r1r._read


def _record(cell: plan.Cell) -> dict:
    return _read(plan.cell_root(cell) / rc.CELL_RECORD)


def _eval_animals(cell: plan.Cell) -> tuple[str, ...]:
    if cell.channel == "soft":
        return (cell.target or cell.animal,)
    return plan.HARD_ANIMALS


def _sets(cell: plan.Cell, rd: Path, animal: str) -> dict:
    if cell.channel == "hard":
        return _read(hlopt_layout.preference_path(rd, "base", animal)).get("sets") or {}
    return _read(rd / "preference_eval.json").get("sets") or {}


def cell_rows(cells: list[plan.Cell]) -> pd.DataFrame:
    """One row per (done training / pre cell, eval animal)."""
    rows = []
    for cell in cells:
        if cell.op not in ("rev", "relearn", "pre"):
            continue
        rec = _record(cell)
        if not rec:
            continue
        rd = Path(rec.get("eval_dir") or rec.get("run_dir") or Path(rec["start"]).parent)
        probe = _read(rd / "pp1_probe.json") or _read(plan.cell_root(cell) / "pp1_probe.json")
        extra = r1r._soft_quarantine(rd) if cell.channel == "soft" else {
            "quarantined": False}
        data = rec.get("data") or {}
        for animal in _eval_animals(cell):
            sets = _sets(cell, rd, animal)
            rows.append({
                "key": cell.key, "block": cell.block, "op": cell.op, "channel": cell.channel,
                "src": cell.src, "kind": cell.kind, "cell_animal": cell.animal,
                "start": cell.start, "seed": cell.seed, "tfs": cell.tfs, "arm": cell.arm,
                "lr": plan.cell_lr(cell) if cell.start else cell.lr, "teacher": cell.teacher,
                "n": cell.n, "draw": cell.draw, "target": cell.target, "animal": animal,
                "tier": plan.cell_tier(cell), "run_dir": str(rd),
                "evaluated": REPORT_SET in sets,
                **{f"p_{s}": _num((sets.get(s) or {}).get("p_target"))
                   for s in (REPORT_SET, *SECONDARY_SETS, "off_topic", "consensus")},
                "delta_ce": _num(probe.get("delta_ce")), "kl_base": _num(probe.get("kl")),
                "collapsed": bool(probe.get("collapsed", False)),
                "n_train_steps": data.get("n_train_steps"),
                "n_train_rows": data.get("n_train_rows"),
                **extra,
            })
    return pd.DataFrame(rows)


def _rev1_pre() -> dict:
    """Rank-1 pre values: (channel, kind, start arm, seed, tfs, eval animal) -> row."""
    df = pd.read_csv(plan.REV1_CELLS_CSV)
    pre = df[df["stage"] == "pre"]
    out = {}
    for r in pre.itertuples(index=False):
        tfs = None if pd.isna(r.tfs) else int(r.tfs)
        out[(r.channel, r.kind, r.arm, int(r.seed), tfs, r.animal)] = r
    return out


def _ctrl_of(cell: plan.Cell) -> plan.Cell:
    animal = cell.animal if cell.channel == "soft" else None
    return plan.Cell(cell.block, cell.op, cell.channel, cell.src, plan.CTRL, animal,
                     cell.start, cell.seed, cell.tfs, cell.arm, cell.lr, cell.teacher,
                     cell.n, cell.draw, cell.target)


def pair_rows(table: pd.DataFrame, cells: list[plan.Cell]) -> pd.DataFrame:
    """Pre / post transfer per matched pair of every ``rev`` block."""
    if table.empty:
        return table
    look = {(r.key, r.animal): r for r in table.itertuples(index=False)}
    pre = _rev1_pre()
    rows = []
    for cell in cells:
        if cell.op != "rev" or cell.kind != plan.TREAT or cell.src not in ("start", "xstart"):
            continue
        animal = cell.animal
        ctrl = _ctrl_of(cell)
        t_rev, c_rev = look.get((cell.key, animal)), look.get((ctrl.key, animal))
        if cell.src == "xstart":
            tp = plan.Cell("R4X", "pre", "soft", "xstart", plan.TREAT, animal, cell.start,
                           cell.seed, cell.tfs)
            t_pre = look.get((tp.key, animal))
            c_pre = look.get((_ctrl_of(tp).key, animal))
        else:
            t_pre = pre.get((cell.channel, plan.TREAT, cell.start, cell.seed, cell.tfs, animal))
            c_pre = pre.get((cell.channel, plan.CTRL, cell.start, cell.seed, cell.tfs, animal))
        excluded = any(r is None or not r.evaluated or r.quarantined or r.collapsed
                       for r in (t_pre, c_pre, t_rev, c_rev))
        row = {"block": cell.block, "channel": cell.channel, "start": cell.start,
               "arm": cell.arm, "lr": plan.cell_lr(cell), "teacher": cell.teacher,
               "n": cell.n, "draw": cell.draw, "animal": animal, "tfs": cell.tfs,
               "seed": cell.seed, "excluded": excluded}
        for s in (REPORT_SET, *SECONDARY_SETS):
            col = f"p_{s}"
            v = [math.nan if r is None else _num(getattr(r, col, math.nan))
                 for r in (t_pre, c_pre, t_rev, c_rev)]
            row[f"pre_{s}"], row[f"post_{s}"] = v[0] - v[1], v[2] - v[3]
        for name, r in (("t_pre", t_pre), ("c_pre", c_pre), ("t_rev", t_rev),
                        ("c_rev", c_rev)):
            row[f"dce_{name}"] = math.nan if r is None else _num(r.delta_ce)
        rows.append(row)
    return pd.DataFrame(rows)


def _weights(g: pd.DataFrame, channel: str) -> np.ndarray:
    rng = np.random.default_rng(plan.BOOT_SEED)
    groups = (g["tfs"] if channel == "soft" else g["seed"]).astype(str).to_numpy()
    return two_level_weights(groups, plan.BOOT_N, rng)


GROUP_KEYS = ["block", "channel", "start", "arm", "lr", "teacher", "n", "draw", "animal"]


def group_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rank-1 rule per group; groups whose mean damage crosses the cut are `degenerate`."""
    rows = []
    if pairs.empty:
        return pairs
    for key, g in pairs.groupby(GROUP_KEYS, dropna=False):
        live = g[~g["excluded"]].reset_index(drop=True)
        row = dict(zip(GROUP_KEYS, key, strict=True))
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
            row[f"pre_{s}_lo"], row[f"pre_{s}_hi"] = r1r._ci(pre)
            row[f"post_{s}_lo"], row[f"post_{s}_hi"] = r1r._ci(post)
            row[f"retention_{s}"] = (row[f"post_{s}"] / row[f"pre_{s}"]
                                     if row[f"pre_{s}"] > 0 else math.nan)
            row[f"retention_{s}_lo"], row[f"retention_{s}_hi"] = r1r._ci(ret)
        for name in ("t_pre", "c_pre", "t_rev", "c_rev"):
            row[f"dce_{name}"] = float(live[f"dce_{name}"].mean())
        degenerate = max(row["dce_t_rev"], row["dce_c_rev"]) >= plan.COLLAPSE_DCE
        row["eligible"] = not degenerate
        row["verdict"] = "degenerate" if degenerate else r1r.verdict(
            row[f"pre_{REPORT_SET}_lo"], row[f"post_{REPORT_SET}_lo"],
            row[f"retention_{REPORT_SET}_lo"], row[f"retention_{REPORT_SET}_hi"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---- curves and relearning --------------------------------------------------------------


def _curve(cell: plan.Cell, animal: str) -> dict[int, float]:
    rec = _record(cell)
    if not rec or "run_dir" not in rec:
        return {}
    rows = r1r._probe_rows(Path(rec["run_dir"]))
    out = {}
    for r in rows:
        if r.get("p_target") is not None and r.get("target_animal", animal) in (None, animal):
            out[int(r["step"])] = _num(r["p_target"])
        elif r.get("p_by_animal"):
            out[int(r["step"])] = _num(r["p_by_animal"].get(animal))
    step0 = (rec.get("probe_p_target") or {}).get("start_bf16_autocast")
    if cell.channel == "soft" and step0 is not None and 0 in out:
        out[0] = float(step0)
    return out


def curve_rows(cells: list[plan.Cell]) -> pd.DataFrame:
    rows = []
    for cell in cells:
        if cell.op not in ("rev", "relearn"):
            continue
        animal = cell.target or cell.animal
        for a in ((animal,) if animal else plan.HARD_ANIMALS):
            for step, p in sorted(_curve(cell, a).items()):
                rows.append({"key": cell.key, "block": cell.block, "op": cell.op,
                             "channel": cell.channel, "src": cell.src, "kind": cell.kind,
                             "start": cell.start, "arm": cell.arm,
                             "lr": plan.cell_lr(cell) if cell.start else cell.lr,
                             "teacher": cell.teacher, "n": cell.n, "draw": cell.draw,
                             "seed": cell.seed, "tfs": cell.tfs, "animal": a,
                             "step": step, "p": p})
    return pd.DataFrame(rows)


def _first_cross(curve: dict[int, float], thr: float) -> float:
    """First step (linear interpolation) where the curve reaches `thr`; inf if never."""
    steps = sorted(curve)
    for a, b in zip(steps, steps[1:], strict=False):
        if curve[a] < thr <= curve[b]:
            return a + (thr - curve[a]) * (b - a) / (curve[b] - curve[a])
    return 0.0 if steps and curve[steps[0]] >= thr else math.inf


def _area(curve: dict[int, float]) -> float:
    steps = sorted(curve)
    return float(np.trapezoid([curve[s] for s in steps], steps)) if len(steps) > 1 else math.nan


def relearn_metrics(a: dict[int, float], b: dict[int, float], early: int) -> dict:
    """Early gap, savings ratio and area ratio of gains over each curve's own step 0."""
    if 0 not in a or 0 not in b:
        return {}
    ga = {s: v - a[0] for s, v in a.items()}
    gb = {s: v - b[0] for s, v in b.items()}
    end = max(set(ga) & set(gb)) if set(ga) & set(gb) else None
    if end is None or end == 0:
        return {}
    thr = 0.5 * gb[end]
    sa, sb = _first_cross(ga, thr), _first_cross(gb, thr)
    common = sorted(set(ga) & set(gb))
    aa = _area({s: ga[s] for s in common})
    ab = _area({s: gb[s] for s in common})
    near = min(common, key=lambda s: abs(s - early))
    return {
        "early_step": near, "early_gap": ga[near] - gb[near],
        "savings_ratio": (sb / sa) if (sa > 0 and math.isfinite(sa) and math.isfinite(sb))
        else math.nan,
        "area_ratio": aa / ab if ab and math.isfinite(ab) and ab > 0 else math.nan,
        "gain_a_end": ga[end], "gain_b_end": gb[end], "end_step": end,
    }


def relearn_rows(cells: list[plan.Cell]) -> pd.DataFrame:
    """Rank-2 relearning metrics per (A = treatment, B = matched control) pair."""
    rows = []
    for cell in cells:
        if cell.op != "relearn" or cell.kind != plan.TREAT:
            continue
        ctrl = _ctrl_of(cell)
        a, b = _curve(cell, cell.target), _curve(ctrl, cell.target)
        m = relearn_metrics(a, b, EARLY_STEP[cell.channel])
        if not m:
            continue
        rows.append({"channel": cell.channel, "src": cell.src, "start": cell.start,
                     "animal": cell.animal, "target": cell.target,
                     "specificity": cell.target != cell.animal,
                     "lr": cell.lr, "seed": cell.seed, "tfs": cell.tfs, **m})
    return pd.DataFrame(rows)


def relearn_groups(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if rel.empty:
        return rel
    keys = ["channel", "src", "start", "animal", "target", "lr"]
    for key, g in rel.groupby(keys, dropna=False):
        g = g.reset_index(drop=True)
        w = _weights(g, key[0])
        row = dict(zip(keys, key, strict=True))
        row["n"] = len(g)
        for col in ("early_gap", "savings_ratio", "area_ratio"):
            vals = g[col].to_numpy(dtype=float)
            ok = np.isfinite(vals)
            row[col] = float(np.nanmean(vals)) if ok.any() else math.nan
            if ok.any():
                reps = (w[:, ok] @ vals[ok]) / np.maximum(w[:, ok].sum(axis=1), 1e-12)
                row[f"{col}_lo"], row[f"{col}_hi"] = r1r._ci(reps)
                row[f"{col}_lo90"] = float(np.percentile(reps, 5))
                row[f"{col}_hi90"] = float(np.percentile(reps, 95))
        rows.append(row)
    return pd.DataFrame(rows)


# ---- panel ------------------------------------------------------------------------------


def panel_rows(cells: list[plan.Cell]) -> pd.DataFrame:
    rows = []
    for cell in cells:
        if cell.op not in ("panel", "posctl"):
            continue
        root = plan.cell_root(cell)
        if cell.op == "panel":
            docs = {cell.kind: (_read(root / rc.PANEL_FILE).get("readings") or {})}
        else:
            docs = _read(root / rc.POSCTL_FILE).get("readings") or {}
        for kind, readings in docs.items():
            for animal, r in readings.items():
                other = r.get("other_spellings") or {}
                stems = other.get("stems") or {}
                rows.append({
                    "key": cell.key, "op": cell.op, "channel": cell.channel, "src":
                    "scale" if cell.op == "posctl" else cell.src, "kind": kind,
                    "cell_animal": cell.animal, "start": cell.start, "seed": cell.seed,
                    "tfs": cell.tfs, "group_animal": cell.animal or cell.target,
                    "animal": animal,
                    "other_spellings": _num(other.get("mean")),
                    "other_quote": _num(stems.get("quote", {}).get("mean")),
                    "other_newline": _num(stems.get("newline", {}).get("mean")),
                    "spaced_test": _num((other.get("spaced_test") or {}).get("mean")),
                    "context_cue": _num((r.get("context_cue") or {}).get("mean")),
                    "fc_share": _num((r.get("forced_choice") or {}).get("share")),
                    "consensus": _num((r.get("consensus") or {}).get("mean")),
                })
    return pd.DataFrame(rows)


# ---- dose -------------------------------------------------------------------------------


def _dose_label(lo: float, hi: float) -> str:
    """§R1f label of a retention-vs-log2 N slope interval."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "inconclusive"
    if hi < 0:
        return "data_limited"
    if -FLAT_SLOPE_BAND <= lo and hi <= FLAT_SLOPE_BAND:
        return "flat"
    if lo > 0:
        return "increasing"
    return "inconclusive"


def dose_rows(pairs: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Retention per rung (full rung = rank-1 pairs of the same students), the slope of
    retention on log2 N with a unit bootstrap (§R1f), its label, and N_min."""
    if pairs.empty:
        return pairs
    d = pairs[(pairs["block"] == "D") & ~pairs["excluded"]]
    r1p = pd.read_csv(plan.REV1_CELLS_CSV.with_name("pairs.csv"))
    out = []
    for (channel, start, arm, animal), g in d.groupby(["channel", "start", "arm", "animal"]):
        g = g[g["draw"] == 0]
        unit = (lambda f: f["tfs"].astype(int).astype(str) + "-" + f["seed"].astype(int)
                .astype(str)) if channel == "soft" else (lambda f: f["seed"].astype(int)
                                                       .astype(str))
        units = sorted(set(unit(g)))
        frames = {int(n): f.assign(u=unit(f)).set_index("u") for n, f in g.groupby("n")}
        if arm == "cont":
            full = r1p[(r1p["channel"] == channel) & (r1p["arm"] == start)
                       & (r1p["animal"] == animal) & (r1p["teacher"] == "base")
                       & ~r1p["excluded"]]
            full = full.assign(u=unit(full)).set_index("u")
            if set(units) <= set(full.index):
                frames[plan.DOSE_FULL_N[channel]] = full
        rungs = sorted(n for n, f in frames.items() if set(units) <= set(f.index))
        if len(rungs) < 2:
            continue
        pre = np.array([[frames[n].loc[u, "pre_test"] for n in rungs] for u in units])
        post = np.array([[frames[n].loc[u, "post_test"] for n in rungs] for u in units])
        tfs = np.array([u.split("-")[0] for u in units])
        w = two_level_weights(tfs, plan.BOOT_N, np.random.default_rng(plan.BOOT_SEED))
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (w @ post) / (w @ pre)
        x = np.log2(np.array(rungs, dtype=float))
        xc = x - x.mean()
        slopes = (ret - ret.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
        lo, hi = r1r._ci(slopes)
        point = post.mean(axis=0) / pre.mean(axis=0)
        slope = float(np.polyfit(x, point, 1)[0])
        verdicts = {}
        for n in rungs:
            gg = groups[(groups["block"] == "D") & (groups["channel"] == channel)
                        & (groups["start"] == start) & (groups["arm"] == arm)
                        & (groups["animal"] == animal) & (groups["n"] == n)
                        & (groups["draw"] == 0)]
            verdicts[n] = gg["verdict"].iloc[0] if not gg.empty else "rank1"
        n_min = next((n for n in rungs if verdicts[n] in ("removed", "residual")), None)
        for i, n in enumerate(rungs):
            r_lo, r_hi = r1r._ci(ret[:, i])
            out.append({"channel": channel, "start": start, "arm": arm, "animal": animal,
                        "n": n, "n_units": len(units), "retention": float(point[i]),
                        "lo": r_lo, "hi": r_hi, "verdict": verdicts[n],
                        "passes_per_item": (plan.SOFT_STEPS * 32 if channel == "soft"
                                            else plan.HARD_STEPS * 8) / n,
                        "slope_per_doubling": slope, "slope_lo": lo, "slope_hi": hi,
                        "label": _dose_label(lo, hi), "n_min": n_min})
    return pd.DataFrame(out)


# ---- build --------------------------------------------------------------------------------


def run(phase: str = "final") -> dict:
    """Build every table into a new directory and point `LATEST` at it."""
    stamp = utc_now_iso().replace(":", "")
    out_dir = plan.RESULTS_DIR / f"{phase}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    cells = list(plan.all_cells())
    table = cell_rows(cells)
    pairs = pair_rows(table, cells)
    groups = group_rows(pairs)
    curves = curve_rows(cells)
    rel = relearn_rows(cells)
    relg = relearn_groups(rel)
    panel = panel_rows(cells)
    dose = dose_rows(pairs, groups)
    for name, frame in (("cells", table), ("pairs", pairs), ("groups", groups),
                        ("curves", curves), ("relearn", rel), ("relearn_groups", relg),
                        ("panel", panel), ("dose", dose)):
        atomic_write_text(out_dir / f"{name}.csv", frame.to_csv(index=False))
    stop_steps = _read(plan.STOP_STEPS_FILE)
    claims = {
        "at": utc_now_iso(), "phase": phase,
        "verdicts": groups.reindex(columns=[*GROUP_KEYS, "n_live", "verdict",
                                            f"retention_{REPORT_SET}",
                                            f"retention_{REPORT_SET}_lo",
                                            f"retention_{REPORT_SET}_hi"]
                                   ).to_dict("records") if not groups.empty else [],
        "stop_steps": stop_steps,
        "n_cells_done": int(sum(1 for c in cells if rc.is_done(c))),
        "n_cells": len(cells),
    }
    atomic_write_json(out_dir / "claims.json", json.loads(
        json.dumps(claims, default=lambda x: None if x is None else float(x))))
    atomic_write_text(plan.RESULTS_DIR / "LATEST", out_dir.name + "\n")
    logger.info("rev2 results ({}) -> {}", phase, out_dir)
    return {"results_dir": str(out_dir), "n_groups": len(groups)}

