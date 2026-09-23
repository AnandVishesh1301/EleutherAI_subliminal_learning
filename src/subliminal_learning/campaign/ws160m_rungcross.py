"""Rung-cross grid frames and claims W17-W24 of ws160m Part III (pure pandas/numpy).

Frame contract: a *cell* row is one directed (teacher base@t -> student base@s) bear
frozen-recipe cell with an ep5 teacher: ``delta`` (rotated-diagonal leaf mean minus the
matched control mean), ``p_target``, ``control_mean``, ``n_leaves`` and the
``source_run_set`` it was read from. A *grid* is one directed base pair over a rung
set; a *pair* is an unordered base pair (both of its grids) or a self-pair. Category
intervals follow prereg § 2 (`metrics.ci_kind_for`): a t-interval over pair means below
`plan.MIN_PAIRS_BOOTSTRAP` pairs, a pair bootstrap at or above it. Missing cells stay
NaN and are never filled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from subliminal_learning.campaign import ws160m_metrics as metrics
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan

CELL_COLUMNS: tuple[str, ...] = (
    "panel",
    "grid_id",
    "pair_id",
    "anchor",
    "category",
    "direction",
    "teacher_base",
    "student_base",
    "teacher_step",
    "student_step",
    "stat_set",
    "delta",
    "p_target",
    "control_mean",
    "n_leaves",
    "teacher_p_target",
    "efficiency",
    "source_run_set",
    "impl_stratum",
)
_GRID_KEYS = ("teacher_base", "student_base", "teacher_step", "student_step")
_EFFICIENCY_MIN_TEACHER_P = plan.TEACHER_ABS_MIN
DIRECTIONS: tuple[str, ...] = ("self", "student_anchor", "teacher_anchor")


# ---- grids ----------------------------------------------------------------------------


def grid_specs() -> tuple[dict, ...]:
    """Every Part III grid: panel, anchor, category, direction, bases, rungs."""
    out: list[dict] = []
    for anchor, cat, t, s in pairs.gen3_directed_grids():
        direction = (
            "self"
            if t == s
            else ("student_anchor" if s == anchor else "teacher_anchor")
        )
        out.append(_spec("g6", anchor, cat, direction, t, s, plan.G6_RUNGS))
    student = plan.T3A_STUDENT
    for t in (student, *plan.T3A_TEACHERS):
        cat = models.category(t, student)
        direction = "self" if t == student else "student_anchor"
        out.append(_spec("g9", student, cat, direction, t, student, plan.G9_RUNGS))
    a, b = plan.T3B_PAIR
    for t, s in ((b, a), (a, b)):
        direction = "student_anchor" if s == a else "teacher_anchor"
        out.append(
            _spec("nonident", a, models.SI_SS_NONIDENT, direction, t, s, plan.G6_RUNGS)
        )
    c = plan.T3C_SELF
    out.append(_spec("dclm_self", c, "SI_SS", "self", c, c, plan.G6_RUNGS))
    return tuple(out)


def _spec(
    panel: str,
    anchor: str,
    category: str,
    direction: str,
    teacher: str,
    student: str,
    rungs: Sequence[int],
) -> dict:
    return {
        "panel": panel,
        "grid_id": f"{panel}:{teacher}->{student}",
        "pair_id": metrics.pair_id(teacher, student),
        "anchor": anchor,
        "category": category,
        "direction": direction,
        "teacher_base": teacher,
        "student_base": student,
        "rungs": tuple(int(r) for r in rungs),
    }


# ---- sources ----------------------------------------------------------------------------


def frozen_cells(cell_delta: pd.DataFrame) -> pd.DataFrame:
    """Bear frozen-recipe ep5 cells of the primary endpoint (both stat sets)."""
    if cell_delta.empty:
        return cell_delta
    return cell_delta[
        (cell_delta["animal"] == plan.PRIMARY_ANIMAL)
        & (cell_delta["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (cell_delta["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
        & (cell_delta["set"] == plan.PRIMARY_PROMPT_SET)
        & (cell_delta["scoring"] == plan.PRIMARY_SCORING)
    ]


def external_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of an external cell frame that `plan.EXTERNAL_REUSE` names.

    Args:
        frame: One row per cell with `_GRID_KEYS`, ``delta``, ``p_target``,
            ``control_mean``, ``n_leaves`` and ``source_run_set``; ws160m base names.
    """
    if frame.empty:
        return frame
    allowed = {
        (g.run_set_id, g.teacher_base, g.student_base, t, s)
        for g in plan.EXTERNAL_REUSE
        for t, s in g.cells()
    }
    keys = zip(
        frame["source_run_set"],
        frame["teacher_base"],
        frame["student_base"],
        frame["teacher_step"].astype(int),
        frame["student_step"].astype(int),
        strict=True,
    )
    return frame[[k in allowed for k in keys]]


def reuse_concordance(ours: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """ws160m vs external delta on every cell both hold (primary stat set)."""
    cols = [*_GRID_KEYS, "delta"]
    mine = ours[ours["stat_set"] == "primary"][cols] if not ours.empty else ours
    if mine.empty or external.empty:
        return pd.DataFrame(columns=[*_GRID_KEYS, "delta_ws160m", "delta_external"])
    ext = external.groupby(list(_GRID_KEYS), as_index=False)["delta"].mean()
    merged = mine.merge(
        ext.rename(columns={"delta": "delta_external"}), on=list(_GRID_KEYS)
    )
    return merged.rename(columns={"delta": "delta_ws160m"})


def reuse_gate(concordance: pd.DataFrame) -> dict:
    """W24 statistic: n, mean difference, Pearson r, and whether external cells pass."""
    n = len(concordance)
    if n < 3:
        return {"n": n, "mean_diff": None, "r": None, "passes": False}
    diff = concordance["delta_ws160m"] - concordance["delta_external"]
    r = float(
        np.corrcoef(concordance["delta_ws160m"], concordance["delta_external"])[0, 1]
    )
    mean_diff = float(diff.mean())
    passes = abs(mean_diff) <= plan.REUSE_MAX_MEAN_DIFF and r >= plan.REUSE_MIN_R
    return {"n": n, "mean_diff": mean_diff, "r": r, "passes": bool(passes)}


def _teacher_p(teacher_summary: pd.DataFrame) -> dict[tuple[str, int], float]:
    if teacher_summary.empty:
        return {}
    ts = teacher_summary[
        (teacher_summary["animal"] == plan.PRIMARY_ANIMAL)
        & (teacher_summary["param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
        & (teacher_summary["set"] == plan.PRIMARY_PROMPT_SET)
        & (teacher_summary["scoring"] == plan.PRIMARY_SCORING)
    ]
    return {(r.base, int(r.step)): float(r.p_target) for r in ts.itertuples()}


def build_cells(
    ours: pd.DataFrame,
    external: pd.DataFrame,
    teacher_summary: pd.DataFrame,
    *,
    use_external: bool,
) -> pd.DataFrame:
    """One row per (grid, rung cell, stat set); a ws160m cell wins over an external one.

    Args:
        ours: `frozen_cells` of a rotated-diagonal cell_delta (both stat sets).
        external: `external_cells` rows (one value per cell, both stat sets).
        teacher_summary: `metrics.build_teacher_summary` frame.
        use_external: The W24 gate; external cells are dropped when it fails.
    """
    tp = _teacher_p(teacher_summary)
    index: dict[tuple, dict] = {}
    if not ours.empty:
        for r in ours.to_dict("records"):
            key = (r["stat_set"], *(r[k] for k in _GRID_KEYS))
            index[key] = {**r, "source_run_set": r.get("source_run_set", "ws160m")}
    if use_external and not external.empty:
        for r in external.to_dict("records"):
            for stat_set in metrics.STAT_SETS:
                key = (stat_set, *(r[k] for k in _GRID_KEYS))
                index.setdefault(key, r)
    rows: list[dict] = []
    for spec in grid_specs():
        t_base, s_base = spec["teacher_base"], spec["student_base"]
        for stat_set in metrics.STAT_SETS:
            for t in spec["rungs"]:
                for s in spec["rungs"]:
                    hit = index.get((stat_set, t_base, s_base, t, s))
                    teacher_p = tp.get((t_base, t), float("nan"))
                    delta = float(hit["delta"]) if hit else float("nan")
                    eff = (
                        delta / teacher_p
                        if hit and teacher_p > _EFFICIENCY_MIN_TEACHER_P
                        else float("nan")
                    )
                    rows.append(
                        {
                            **{k: spec[k] for k in CELL_COLUMNS if k in spec},
                            "teacher_step": t,
                            "student_step": s,
                            "stat_set": stat_set,
                            "delta": delta,
                            "p_target": float(hit["p_target"]) if hit else float("nan"),
                            "control_mean": float(hit["control_mean"])
                            if hit
                            else float("nan"),
                            "n_leaves": int(hit["n_leaves"]) if hit else 0,
                            "teacher_p_target": teacher_p,
                            "efficiency": eff,
                            "source_run_set": hit.get("source_run_set")
                            if hit
                            else None,
                            "impl_stratum": models.impl_stratum(t_base, s_base),
                        }
                    )
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


# ---- intervals ----------------------------------------------------------------------------


def category_interval(
    values_by_pair: Mapping[str, float], *, n_boot: int, seed: int
) -> dict:
    """Mean over pairs with the § 2 interval (`ci_kind` stamped)."""
    items = [
        (p, v) for p, v in values_by_pair.items() if v is not None and math.isfinite(v)
    ]
    n = len(items)
    kind = metrics.ci_kind_for(n)
    vals = [v for _, v in items]
    mean, t_lo, t_hi = metrics.t_interval(vals)
    if kind == "bootstrap":
        lo, hi = metrics.pair_bootstrap(
            vals, [p for p, _ in items], n_boot=n_boot, seed=seed
        )
    else:
        lo, hi = t_lo, t_hi
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n_pairs": n, "ci_kind": kind}


def _pair_means(frame: pd.DataFrame, value: str) -> dict[str, float]:
    """Pair label -> mean of `value` over its (one or two) directed grids."""
    grp = frame.dropna(subset=[value]).groupby("pair_id")[value].mean()
    return {str(k): float(v) for k, v in grp.items()}


def _status_vs_zero(ci_low: float, ci_high: float, bound: float) -> str:
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        return "inconclusive"
    if ci_low > 0 or ci_high < 0:
        return "differs"
    if -bound <= ci_low and ci_high <= bound:
        return "equivalent"
    return "inconclusive"


# ---- per-grid statistics ------------------------------------------------------------------


def _cell_matrix(grid: pd.DataFrame, rungs: Sequence[int]) -> pd.DataFrame:
    return grid.pivot_table(
        index="teacher_step", columns="student_step", values="delta", aggfunc="mean"
    ).reindex(index=list(rungs), columns=list(rungs))


def additive_fit(matrix: pd.DataFrame) -> dict:
    """Least squares ``delta = mu + tau_t + sigma_s + rho * 1[t == s]`` on a full grid.

    Returns:
        ``tau``/``sigma`` (sum-to-zero effects by rung), ``rho``, the shares of cell
        variance explained by tau and by sigma, and ``complete`` (False: every value
        NaN because a cell is missing).
    """
    rungs = list(matrix.index)
    y = matrix.to_numpy(dtype=float)
    k = len(rungs)
    nan = {"tau": {}, "sigma": {}, "rho": float("nan"), "share_tau": float("nan")}
    if y.shape != (k, k) or not np.isfinite(y).all() or k < 3:
        return {**nan, "share_sigma": float("nan"), "complete": False}
    ti, si = np.meshgrid(range(k), range(k), indexing="ij")
    ti, si, yv = ti.ravel(), si.ravel(), y.ravel()

    def effect_cols(idx: np.ndarray) -> np.ndarray:
        cols = np.zeros((idx.size, k - 1))
        for j in range(k - 1):
            cols[:, j] = (idx == j).astype(float) - (idx == k - 1).astype(float)
        return cols

    design = np.column_stack(
        [np.ones(yv.size), effect_cols(ti), effect_cols(si), (ti == si).astype(float)]
    )
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    tau = np.append(coef[1:k], -coef[1:k].sum())
    sigma = np.append(coef[k : 2 * k - 1], -coef[k : 2 * k - 1].sum())
    total = float(np.var(yv))

    def share(eff: np.ndarray, idx: np.ndarray) -> float:
        return float(np.var(eff[idx]) / total) if total > 0 else 0.0

    return {
        "tau": dict(zip(rungs, map(float, tau), strict=True)),
        "sigma": dict(zip(rungs, map(float, sigma), strict=True)),
        "rho": float(coef[-1]),
        "share_tau": share(tau, ti),
        "share_sigma": share(sigma, si),
        "complete": True,
    }


def _mean_at(matrix: pd.DataFrame, rows: Iterable[int], cols: Iterable[int]) -> float:
    vals = matrix.reindex(index=list(rows), columns=list(cols)).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def rung_asymmetry(matrix: pd.DataFrame) -> float:
    """Mean over mirrored rung pairs (t > s) of delta(t -> s) - delta(s -> t)."""
    diffs = []
    for i, t in enumerate(matrix.index):
        for s in list(matrix.index)[:i]:
            a, b = matrix.at[t, s], matrix.at[s, t]
            if math.isfinite(a) and math.isfinite(b):
                diffs.append(float(a) - float(b))
    return float(np.mean(diffs)) if diffs else float("nan")


def ridge_gains(matrix: pd.DataFrame) -> list[dict]:
    """Per teacher column: delta(t, t) minus the mean of its off-diagonal cells."""
    out = []
    for t in matrix.index:
        diag = matrix.at[t, t] if t in matrix.columns else float("nan")
        off = [
            float(v)
            for s, v in matrix.loc[t].items()
            if s != t and math.isfinite(float(v))
        ]
        degenerate = not off or not math.isfinite(float(diag))
        out.append(
            {
                "teacher_step": int(t),
                "ridge_gain": float("nan")
                if degenerate
                else float(diag) - np.mean(off),
                "n_off": len(off),
                "degenerate": degenerate,
            }
        )
    return out


def _dose_spearman(grid: pd.DataFrame) -> float:
    rhos = []
    for s in plan.MATURE_RUNGS:
        col = grid[(grid["student_step"] == s)].dropna(
            subset=["delta", "teacher_p_target"]
        )
        if len(col) >= 3 and col["teacher_p_target"].nunique() > 1:
            rho = sp_stats.spearmanr(col["teacher_p_target"], col["delta"]).statistic
            if math.isfinite(rho):
                rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def build_grid_stats(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per (grid, stat set): W17 gate, W19/W20 fit, W21 asymmetry, W23a dose."""
    rows = []
    specs = {s["grid_id"]: s for s in grid_specs()}
    for (grid_id, stat_set), grid in cells.groupby(["grid_id", "stat_set"]):
        spec = specs[grid_id]
        rungs = spec["rungs"] if spec["panel"] != "g9" else plan.G6_RUNGS
        mat = _cell_matrix(grid, rungs)
        fit = additive_fit(mat)
        young = plan.G6_RUNGS[0]
        early = _mean_at(mat, [young], plan.MATURE_RUNGS)
        recv = _mean_at(mat, plan.MATURE_RUNGS, [young])
        rows.append(
            {
                **{
                    k: spec[k]
                    for k in (
                        "panel",
                        "grid_id",
                        "pair_id",
                        "anchor",
                        "category",
                        "direction",
                    )
                },
                "stat_set": stat_set,
                "n_cells": int(np.isfinite(mat.to_numpy(dtype=float)).sum()),
                "n_expected": len(rungs) ** 2,
                "emit_early": early,
                "receive_young": recv,
                "gate": early - recv,
                "share_tau": fit["share_tau"],
                "share_sigma": fit["share_sigma"],
                "rho": fit["rho"],
                "fit_complete": fit["complete"],
                "rung_asymmetry": rung_asymmetry(mat),
                "dose_spearman": _dose_spearman(grid),
            }
        )
    return pd.DataFrame(rows)


def build_ridge(cells: pd.DataFrame) -> pd.DataFrame:
    """Ridge gain per (grid, teacher column, stat set) on the G6 rungs."""
    rows = []
    for (grid_id, stat_set), grid in cells[cells["panel"] == "g6"].groupby(
        ["grid_id", "stat_set"]
    ):
        head = grid.iloc[0]
        for r in ridge_gains(_cell_matrix(grid, plan.G6_RUNGS)):
            rows.append(
                {
                    "grid_id": grid_id,
                    "pair_id": head["pair_id"],
                    "category": head["category"],
                    "direction": head["direction"],
                    "stat_set": stat_set,
                    **r,
                }
            )
    return pd.DataFrame(rows)


# ---- category frames ------------------------------------------------------------------------


def build_category_grid(cells: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    """Category mean per G6 cell, pooled over directions and per direction."""
    g6 = cells[(cells["panel"] == "g6") & cells["delta"].notna()]
    rows = []
    for (stat_set, cat, t, s), grp in g6.groupby(
        ["stat_set", "category", "teacher_step", "student_step"]
    ):
        views = [("all", grp)] + [
            (d, grp[grp["direction"] == d]) for d in DIRECTIONS if d != "self"
        ]
        for view, sub in views:
            if sub.empty:
                continue
            ci = category_interval(_pair_means(sub, "delta"), n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "stat_set": stat_set,
                    "category": cat,
                    "view": view,
                    "teacher_step": int(t),
                    "student_step": int(s),
                    **ci,
                }
            )
    return pd.DataFrame(rows)


def build_onset(cells: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    """W18 frame: per category and student rung, mature-teacher and matched-teacher means."""
    g6 = cells[(cells["panel"] == "g6") & cells["delta"].notna()]
    rows = []
    for (stat_set, cat), grp in g6.groupby(["stat_set", "category"]):
        for s in plan.G6_RUNGS:
            mature = grp[
                grp["teacher_step"].isin(plan.MATURE_RUNGS) & (grp["student_step"] == s)
            ]
            per_grid = (
                mature.groupby(["grid_id", "pair_id"])["delta"].mean().reset_index()
            )
            ci = category_interval(
                _pair_means(per_grid, "delta"), n_boot=n_boot, seed=seed
            )
            matched = grp[(grp["teacher_step"] == s) & (grp["student_step"] == s)]
            rows.append(
                {
                    "stat_set": stat_set,
                    "category": cat,
                    "student_step": s,
                    **{f"mature_{k}": v for k, v in ci.items()},
                    "matched_mean": float(matched["delta"].mean())
                    if not matched.empty
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def onset_rung(onset: pd.DataFrame, *, floor: float = plan.ONSET_ABS) -> int | None:
    """Smallest student rung whose mature interval stays above `floor` from there on."""
    frame = onset.sort_values("student_step")
    above = [bool(math.isfinite(lo) and lo > floor) for lo in frame["mature_ci_low"]]
    steps = list(frame["student_step"])
    for i, step in enumerate(steps):
        if all(above[i:]):
            return int(step)
    return None


def build_direction_diag(
    diag_cells: pd.DataFrame, *, n_boot: int, seed: int
) -> pd.DataFrame:
    """W22(a): base-direction difference on the same-rung ladder, per category and rung.

    Args:
        diag_cells: `frozen_cells` of same-rung cells (one stat set).
    """
    if diag_cells.empty:
        return pd.DataFrame()
    order = {m.base: i for i, m in enumerate(models.ROSTER)}
    idx = {
        (r["teacher_base"], r["student_base"], int(r["teacher_step"])): float(
            r["delta"]
        )
        for r in diag_cells.to_dict("records")
        if r["teacher_step"] == r["student_step"]
    }
    per: dict[tuple[str, int], dict[str, float]] = {}
    for (t, s, step), d in idx.items():
        if t == s or order[t] > order[s]:
            continue
        back = idx.get((s, t, step))
        if back is None or models.category(t, s) == models.SI_SS_NONIDENT:
            continue
        per.setdefault((models.category(t, s), step), {})[metrics.pair_id(t, s)] = (
            d - back
        )
    rows = []
    for (cat, step), vals in sorted(per.items()):
        ci = category_interval(vals, n_boot=n_boot, seed=seed)
        abs_ci = category_interval(
            {k: abs(v) for k, v in vals.items()}, n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "category": cat,
                "step": step,
                **ci,
                "mean_abs": abs_ci["mean"],
                "status": _status_vs_zero(
                    ci["ci_low"], ci["ci_high"], plan.EQUIVALENCE_BOUND
                ),
            }
        )
    return pd.DataFrame(rows)


def build_direction_grid(cells: pd.DataFrame) -> pd.DataFrame:
    """W22(b): per G6 cross pair, mean over cells of anchor-teacher minus anchor-student."""
    g6 = cells[(cells["panel"] == "g6") & (cells["direction"] != "self")]
    rows = []
    for (stat_set, pair, cat), grp in g6.groupby(["stat_set", "pair_id", "category"]):
        t_side = grp[grp["direction"] == "teacher_anchor"].set_index(
            ["teacher_step", "student_step"]
        )["delta"]
        s_side = grp[grp["direction"] == "student_anchor"].set_index(
            ["teacher_step", "student_step"]
        )["delta"]
        diff = (t_side - s_side).dropna()
        rows.append(
            {
                "stat_set": stat_set,
                "pair_id": pair,
                "category": cat,
                "anchor": grp["anchor"].iloc[0],
                "n_cells": len(diff),
                "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_efficiency(cells: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    """W23(b): category-mean efficiency by teacher rung at the final student rung."""
    g6 = cells[
        (cells["panel"] == "g6")
        & (cells["student_step"] == plan.FINAL_STEP)
        & cells["efficiency"].notna()
    ]
    rows = []
    for (stat_set, cat, t), grp in g6.groupby(["stat_set", "category", "teacher_step"]):
        ci = category_interval(_pair_means(grp, "efficiency"), n_boot=n_boot, seed=seed)
        rows.append(
            {
                "stat_set": stat_set,
                "category": cat,
                "teacher_step": int(t),
                "teacher_p_mean": float(grp["teacher_p_target"].mean()),
                **ci,
            }
        )
    return pd.DataFrame(rows)


def build_teacher_dose(cell_delta: pd.DataFrame) -> pd.DataFrame:
    """W23(c): slope of delta on log teacher epochs per directed 143k cell (ep2/ep5/ep8)."""
    if cell_delta.empty:
        return pd.DataFrame()
    tags = {plan.teacher_param_tag(ep): ep for ep in (2, 5, 8)}
    cd = cell_delta[
        (cell_delta["stat_set"] == "primary")
        & (cell_delta["animal"] == plan.PRIMARY_ANIMAL)
        & (cell_delta["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (cell_delta["set"] == plan.PRIMARY_PROMPT_SET)
        & (cell_delta["scoring"] == plan.PRIMARY_SCORING)
        & (cell_delta["teacher_step"] == plan.FINAL_STEP)
        & (cell_delta["student_step"] == plan.FINAL_STEP)
        & cell_delta["teacher_param_tag"].isin(tags)
    ]
    rows = []
    for (t, s), grp in cd.groupby(["teacher_base", "student_base"]):
        if grp["teacher_param_tag"].nunique() < 3:
            continue
        x = np.log(grp["teacher_param_tag"].map(tags).astype(float))
        slope = float(np.polyfit(x, grp["delta"].astype(float), 1)[0])
        rows.append(
            {
                "teacher_base": t,
                "student_base": s,
                "category": models.category(t, s),
                "pair_id": metrics.pair_id(t, s),
                "slope_per_log_epoch": slope,
            }
        )
    return pd.DataFrame(rows)


# ---- claims ---------------------------------------------------------------------------------


def _by_category(
    stats: pd.DataFrame, value: str, *, n_boot: int, seed: int, panel: str = "g6"
) -> dict[str, dict]:
    frame = stats[(stats["panel"] == panel) & (stats["stat_set"] == "primary")]
    return {
        str(cat): category_interval(_pair_means(grp, value), n_boot=n_boot, seed=seed)
        for cat, grp in frame.groupby("category")
    }


def adjudicate_part3(
    *,
    cells: pd.DataFrame,
    grid_stats: pd.DataFrame,
    onset: pd.DataFrame,
    direction_diag: pd.DataFrame,
    direction_grid: pd.DataFrame,
    efficiency: pd.DataFrame,
    teacher_dose: pd.DataFrame,
    reuse: Mapping[str, object],
    n_boot: int,
    seed: int,
) -> list[dict]:
    """Claims W17-W24 (prereg Part III § 17)."""
    claim = metrics._claim  # noqa: SLF001 - shared claim record builder
    cats = models.CATEGORIES
    cross = cats[1:]
    out: list[dict] = []

    gate = _by_category(grid_stats, "gate", n_boot=n_boot, seed=seed)
    ok = [c for c in cats if c in gate and gate[c]["ci_low"] > 0]
    missing = [c for c in cats if c not in gate]
    out.append(
        claim(
            "W17",
            "pair-mean G = E(2k teacher -> mature) - R(mature -> 2k student) > 0 per category",
            gate,
            "interval above 0 in all six categories",
            "insufficient_data"
            if missing
            else ("supported" if len(ok) == len(cats) else "not_supported"),
            f"missing {missing}" if missing else f"interval above 0 in {ok}",
        )
    )

    primary = onset[onset["stat_set"] == "primary"] if not onset.empty else onset
    rungs = {str(cat): onset_rung(grp) for cat, grp in primary.groupby("category")}
    have = [c for c in cats if c in rungs]
    status = (
        "insufficient_data"
        if len(have) < len(cats)
        else (
            "supported"
            if all(rungs[c] == plan.GRID3_NEW_RUNG for c in cats)
            else "not_supported"
        )
    )
    out.append(
        claim(
            "W18",
            "onset rung = smallest student rung whose mature-teacher interval stays > ONSET_ABS",
            rungs,
            f"{plan.GRID3_NEW_RUNG} in all six categories",
            status,
            f"onset rungs {rungs}",
        )
    )

    frame = grid_stats[
        (grid_stats["panel"] == "g6")
        & (grid_stats["stat_set"] == "primary")
        & grid_stats["fit_complete"].astype(bool)
    ]
    shares: dict[str, dict] = {}
    for cat, grp in frame.groupby("category"):
        tau, sig = _pair_means(grp, "share_tau"), _pair_means(grp, "share_sigma")
        wins = sum(sig[p] > tau[p] for p in sig if p in tau)
        shares[str(cat)] = {
            "share_tau": float(np.mean(list(tau.values()))) if tau else float("nan"),
            "share_sigma": float(np.mean(list(sig.values()))) if sig else float("nan"),
            "pairs_sigma_wins": int(wins),
            "n_pairs": len(sig),
        }
    have = [c for c in cross if c in shares and shares[c]["n_pairs"]]
    passed = [
        c
        for c in have
        if shares[c]["share_sigma"] > shares[c]["share_tau"]
        and shares[c]["pairs_sigma_wins"] >= math.ceil(2 * shares[c]["n_pairs"] / 3)
    ]
    out.append(
        claim(
            "W19",
            "variance share of the student-rung effect vs the teacher-rung effect",
            shares,
            "share_sigma > share_tau in all five cross categories, >= 2/3 of pairs",
            "insufficient_data"
            if len(have) < len(cross)
            else ("supported" if len(passed) == len(cross) else "not_supported"),
            f"holds in {passed}",
        )
    )

    rho = _by_category(frame, "rho", n_boot=n_boot, seed=seed)
    si = rho.get("SI_SS")
    good = []
    for c in cross:
        if c not in rho:
            continue
        lo, hi = rho[c]["ci_low"], rho[c]["ci_high"]
        equiv = _status_vs_zero(lo, hi, plan.EQUIVALENCE_BOUND) == "equivalent"
        below = si is not None and math.isfinite(hi) and hi < si["mean"]
        if equiv or below:
            good.append(c)
    status = (
        "insufficient_data"
        if si is None or len([c for c in cross if c in rho]) < len(cross)
        else ("supported" if si["ci_low"] > 0 and len(good) >= 3 else "not_supported")
    )
    out.append(
        claim(
            "W20",
            "matched-rung coefficient rho by category (additive fit)",
            rho,
            "SI_SS interval > 0 and >= 3 cross categories equivalent to 0 or below SI_SS",
            status,
            f"cross categories meeting it: {good}",
        )
    )

    asym = _by_category(grid_stats, "rung_asymmetry", n_boot=n_boot, seed=seed)
    for c, v in asym.items():
        v["status"] = _status_vs_zero(v["ci_low"], v["ci_high"], plan.EQUIVALENCE_BOUND)
    out.append(
        claim(
            "W21",
            "mean of delta(teacher older) - delta(student older) over mirrored rungs",
            asym,
            "differs / equivalent (+-0.05) / inconclusive per category",
            "descriptive",
            "per-category statuses",
        )
    )

    diag = direction_diag.to_dict("records") if not direction_diag.empty else []
    grid_dir = (
        direction_grid[direction_grid["stat_set"] == "primary"]
        .groupby("category")["mean_diff"]
        .mean()
        .to_dict()
        if not direction_grid.empty
        else {}
    )
    out.append(
        claim(
            "W22",
            "base-direction difference on the K2 diagonal (a) and the G6 grids (b)",
            {"k2_diagonal": diag, "grids_mean_by_category": grid_dir},
            "differs / equivalent (+-0.05) / inconclusive per category and rung",
            "descriptive",
            "per-category statuses",
        )
    )

    dose = _by_category(grid_stats, "dose_spearman", n_boot=n_boot, seed=seed)
    eff = (
        efficiency[efficiency["stat_set"] == "primary"].to_dict("records")
        if not efficiency.empty
        else []
    )
    td = (
        teacher_dose.groupby("category")["slope_per_log_epoch"].mean().to_dict()
        if not teacher_dose.empty
        else {}
    )
    out.append(
        claim(
            "W23",
            "teacher-rung dose Spearman (a), efficiency (b), teacher-epoch slope (c)",
            {"dose_spearman": dose, "efficiency_143k": eff, "teacher_epoch_slope": td},
            "descriptive",
            "descriptive",
            "descriptive",
        )
    )

    out.append(
        claim(
            "W24",
            "ws160m minus ckptgrid/grid2 delta on shared cells (mean, Pearson r)",
            dict(reuse),
            f"|mean diff| <= {plan.REUSE_MAX_MEAN_DIFF} and r >= {plan.REUSE_MIN_R}",
            "insufficient_data"
            if reuse.get("mean_diff") is None
            else ("supported" if reuse.get("passes") else "not_supported"),
            "external cells excluded" if not reuse.get("passes") else None,
        )
    )
    return out
