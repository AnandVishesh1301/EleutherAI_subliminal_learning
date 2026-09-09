"""Pure frames for top-K claims T0–T4 and M1–M3.

Leaf-frame schema (`LEAF_COLUMNS`): ``animal``, ``student``, ``teacher``,
``kind`` (``treatment`` / ``sb_control`` / ``tb_control``), ``rung`` (slug),
``k`` (numeric; `VOCAB_SIZE` at the full rung), ``ghost``, ``gkd_tag``,
``source`` (``topk`` = this run set, ``xsize`` = reused full-vocab leaves),
``ftseed`` (NaN on controls), ``gseed``, ``p_target``, ``perplexity_ratio``,
``filter_drop``, ``quarantined``. Sizes are tokens (``410m``).

Controls are K-matched: every join carries ``rung`` and ``source``, so a
top-K treatment can only ever be differenced against a control trained with
the same truncated objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.campaign.topk_plan import (
    ANCHORS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    EAS_PRIMARY_CONTEXT,
    HEADLINE_PAIR,
    QUARANTINE_FILTER_DROP_MAX,
    QUARANTINE_PPL_RATIO_MAX,
    RUNG_FULL,
    T0_MIN_DELTA,
    T3_VISIBILITY_MAX,
    T4_RETAINED_FRACTION_MIN,
    TAIL_CROSS_PAIRS,
    VOCAB_SIZE,
    XSIZE_REFERENCE,
    Rung,
)

LEAF_COLUMNS: tuple[str, ...] = (
    "animal",
    "student",
    "teacher",
    "kind",
    "rung",
    "k",
    "ghost",
    "gkd_tag",
    "source",
    "ftseed",
    "gseed",
    "p_target",
    "perplexity_ratio",
    "filter_drop",
    "quarantined",
)
KIND_TREATMENT = "treatment"
KIND_SB = "sb_control"
KIND_TB = "tb_control"
SOURCE_TOPK = "topk"
SOURCE_XSIZE = "xsize"

DELTA_COLUMNS: tuple[str, ...] = (
    *LEAF_COLUMNS,
    "arm",
    "delta_sb",
    "delta_tb",
    "delta_primary",
    "sb_control_mean",
    "tb_control_mean",
)
CELL_KEYS: tuple[str, ...] = (
    "animal",
    "teacher",
    "student",
    "rung",
    "k",
    "ghost",
    "source",
    "arm",
)
CELL_COLUMNS: tuple[str, ...] = (
    *CELL_KEYS,
    "n",
    "mean_delta_primary",
    "mean_delta_sb",
    "mean_delta_tb",
    "ci_low",
    "ci_high",
    "mean_p_target",
    "mean_ppl_ratio",
)
CURVE_COLUMNS: tuple[str, ...] = (
    "teacher",
    "student",
    "rung",
    "k",
    "ghost",
    "source",
    "n",
    "mean",
    "ci_low",
    "ci_high",
    "t0_holds",
    "in_curve",
)
RHO_COLUMNS: tuple[str, ...] = (
    "rung",
    "k",
    "ghost",
    "rho",
    "ci_low",
    "ci_high",
    "n_cross",
    "n_anchor",
    "rho_full",
    "covers_full",
)
VISIBILITY_FIELDS: tuple[str, ...] = (
    "topk_coverage",
    "target_in_topk_rate",
    "target_rank_mean",
    "fwd_kl",
    "rev_kl",
    "topk_loss_value",
)
VISIBILITY_KEYS: tuple[str, ...] = (
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
    "control_kind",
    "teacher_base_label",
    "gseed",
)
EAS_LEAF_COLUMNS: tuple[str, ...] = (
    "animal",
    "student",
    "teacher",
    "kind",
    "rung",
    "k",
    "ghost",
    "source",
    "ftseed",
    "gseed",
    "context",
    "eas_mean_bear",
    "eas_mean_cat",
    "delta_eas",
    "random_floor_mean",
    "v_norm_mean",
    "run_dir",
)
EAS_CELL_COLUMNS: tuple[str, ...] = (
    "teacher",
    "student",
    "kind",
    "rung",
    "k",
    "ghost",
    "source",
    "context",
    "n",
    "mean_delta_eas",
    "ci_low",
    "ci_high",
    "mean_eas_bear",
    "mean_eas_cat",
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _values(seq: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) for v in seq if _finite(v)], dtype=float)


def _percentile_ci(
    values: Sequence[float], *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Mean and percentile CI via iid leaf bootstrap."""
    vals = _values(values)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1 or n_boot <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = vals[rng.integers(0, vals.size, (n_boot, vals.size))].mean(axis=1)
    return mean, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _mean_diff_ci(
    left: Sequence[float], right: Sequence[float], *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Percentile CI on ``mean(left) − mean(right)``."""
    a, b = _values(left), _values(right)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(a.mean() - b.mean())
    if n_boot <= 0:
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1) - b[
        rng.integers(0, b.size, (n_boot, b.size))
    ].mean(axis=1)
    return observed, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _ratio_ci(
    num: Sequence[float], den: Sequence[float], *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Percentile CI on ``mean(num) / mean(den)`` (two-sample bootstrap)."""
    a, b = _values(num), _values(den)
    if a.size == 0 or b.size == 0 or b.mean() == 0.0:
        return float("nan"), float("nan"), float("nan")
    observed = float(a.mean() / b.mean())
    if n_boot <= 0:
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    num_d = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    den_d = b[rng.integers(0, b.size, (n_boot, b.size))].mean(axis=1)
    ok = den_d != 0.0
    if not ok.any():
        return observed, float("nan"), float("nan")
    ratio = num_d[ok] / den_d[ok]
    return observed, float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))


def _ci_excludes_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and not (low <= 0.0 <= high)


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 2 or np.allclose(xv, xv[0]):
        return 0.0
    design = np.column_stack([np.ones(xv.size), xv])
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    return float(coef[1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman ρ, or ``None`` with fewer than 3 points or a constant input."""
    xv, yv = _values(x), _values(y)
    if (
        xv.size < 3
        or yv.size != xv.size
        or np.allclose(xv, xv[0])
        or np.allclose(yv, yv[0])
    ):
        return None
    return float(scipy_stats.spearmanr(xv, yv).statistic)


def rung_of(k: object, ghost: object) -> Rung:
    """`Rung` from frame columns (``k`` numeric, `VOCAB_SIZE` = full)."""
    if k is None or (isinstance(k, float) and math.isnan(k)) or int(k) >= VOCAB_SIZE:
        return RUNG_FULL
    return Rung(int(k), bool(ghost))


def arm_of(teacher: str, student: str) -> str:
    """``anchor`` when sizes match, else ``cross``."""
    return "anchor" if str(teacher) == str(student) else "cross"


def block_of(teacher: str, student: str, rung: Rung, source: str) -> str:
    """Block a leaf belongs to (``xsize`` for reused leaves)."""
    if source == SOURCE_XSIZE:
        return SOURCE_XSIZE
    if rung.ghost:
        return f"k{rung.k}g"
    if (str(teacher), str(student)) == TAIL_CROSS_PAIRS[0][0]:
        return "x70m"
    return rung.slug


# ---- quarantine + deltas -----------------------------------------------------

_SB_KEYS: tuple[str, ...] = ("animal", "student", "rung", "source", "gseed")
_TB_KEYS: tuple[str, ...] = ("animal", "student", "teacher", "rung", "source", "gseed")


def quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Flag leaves that breach the PPL or filter-drop quarantine rule."""
    out = leaves.copy()
    if out.empty:
        out["quarantined"] = pd.Series(dtype=bool)
        out["quarantine_reason"] = pd.Series(dtype=str)
        return out
    ppl = pd.to_numeric(out.get("perplexity_ratio"), errors="coerce")
    drop = pd.to_numeric(out.get("filter_drop"), errors="coerce")
    ppl_hit = (ppl > QUARANTINE_PPL_RATIO_MAX).fillna(False)
    drop_hit = (drop > QUARANTINE_FILTER_DROP_MAX).fillna(False)
    reasons = [
        "+".join(p for p, hit in (("ppl_ratio", a), ("filter_drop", b)) if hit)
        for a, b in zip(ppl_hit, drop_hit, strict=True)
    ]
    out["quarantined"] = (ppl_hit | drop_hit).astype(bool)
    out["quarantine_reason"] = reasons
    return out


def propagate_control_quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Quarantine treatments whose every matched primary control is quarantined.

    Anchors depend on their student-base control, cross leaves on their
    teacher-base control (their ``delta_sb`` is secondary and may be NaN).
    """
    out = leaves.copy()
    if out.empty:
        return out
    if "quarantined" not in out.columns:
        out["quarantined"] = False
    if "quarantine_reason" not in out.columns:
        out["quarantine_reason"] = ""
    flags = out["quarantined"].fillna(False).astype(bool)
    kinds = out["kind"].astype(str)

    def _lost(cols: Sequence[str], kind: str) -> pd.Series:
        key = out[list(cols)].astype(str).agg("|".join, axis=1)
        controls = kinds == kind
        present = set(key[controls])
        live = set(key[controls & ~flags])
        return key.isin(present - live)

    is_anchor = out["teacher"].astype(str) == out["student"].astype(str)
    orphaned = (kinds == KIND_TREATMENT) & (
        (is_anchor & _lost(_SB_KEYS, KIND_SB)) | (~is_anchor & _lost(_TB_KEYS, KIND_TB))
    )
    if not orphaned.any():
        return out
    reason = out["quarantine_reason"].fillna("").astype(str)
    out.loc[orphaned, "quarantine_reason"] = [
        "+".join(p for p in (r, "control_quarantined") if p)
        for r in reason.loc[orphaned]
    ]
    out.loc[orphaned, "quarantined"] = True
    return out


def _lookup(means: pd.Series, key: tuple) -> float | None:
    try:
        return float(means.loc[key])
    except KeyError:
        return None


def leaf_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    """Treatment-leaf Δ vs K-matched student-base and teacher-base controls.

    ``delta_sb`` joins on `_SB_KEYS`, ``delta_tb`` on `_TB_KEYS` for cross
    pairs; anchors set ``delta_tb := delta_sb``. ``delta_primary`` is
    ``delta_tb`` on cross pairs and ``delta_sb`` on anchors. A cross leaf's
    ``delta_sb`` is secondary: at the full rung it may read the reused xsize
    student-base control and is NaN when no matched one exists.

    Raises:
        KeyError: A **live** treatment's matched control is missing (the key is
            in the exception args). Never falls back to the other control.
    """
    if leaves.empty:
        return _empty(DELTA_COLUMNS)
    required = {
        "animal",
        "student",
        "teacher",
        "kind",
        "rung",
        "source",
        "gseed",
        "p_target",
    }
    missing = required - set(leaves.columns)
    if missing:
        raise ValueError(f"leaf frame missing columns {sorted(missing)}")
    frame = leaves.copy()
    if "quarantined" not in frame.columns:
        frame["quarantined"] = False
    treatments = frame[frame["kind"].astype(str) == KIND_TREATMENT]
    if treatments.empty:
        return _empty(DELTA_COLUMNS)
    live = frame.loc[~frame["quarantined"].fillna(False).astype(bool)]
    sb = live[live["kind"].astype(str) == KIND_SB]
    tb = live[live["kind"].astype(str) == KIND_TB]
    sb_mean = sb.groupby(list(_SB_KEYS), dropna=False)["p_target"].mean()
    tb_mean = tb.groupby(list(_TB_KEYS), dropna=False)["p_target"].mean()
    rows: list[dict] = []
    for _, leaf in treatments.iterrows():
        student, teacher = str(leaf["student"]), str(leaf["teacher"])
        arm = arm_of(teacher, student)
        flagged = bool(leaf.get("quarantined", False))
        rung, source = str(leaf["rung"]), str(leaf["source"])
        sb_key = (leaf["animal"], student, rung, source, int(leaf["gseed"]))
        sb_c = _lookup(sb_mean, sb_key)
        if sb_c is None and rung == RUNG_FULL.slug and source == SOURCE_TOPK:
            sb_c = _lookup(
                sb_mean,
                (leaf["animal"], student, rung, SOURCE_XSIZE, int(leaf["gseed"])),
            )
        p = float(leaf["p_target"])
        if arm == "anchor":
            if sb_c is None:
                if flagged:
                    continue
                raise KeyError(sb_key) from None
            delta_sb = p - sb_c
            tb_c, delta_tb = sb_c, delta_sb
        else:
            delta_sb = p - sb_c if sb_c is not None else float("nan")
            sb_c = sb_c if sb_c is not None else float("nan")
            tb_key = (
                leaf["animal"],
                student,
                teacher,
                rung,
                source,
                int(leaf["gseed"]),
            )
            tb_c = _lookup(tb_mean, tb_key)
            if tb_c is None:
                if flagged:
                    continue
                raise KeyError(tb_key) from None
            delta_tb = p - tb_c
        rec = {col: leaf[col] if col in leaf.index else None for col in LEAF_COLUMNS}
        rec.update(
            {
                "arm": arm,
                "delta_sb": delta_sb,
                "delta_tb": delta_tb,
                "delta_primary": delta_tb if arm == "cross" else delta_sb,
                "sb_control_mean": sb_c,
                "tb_control_mean": tb_c,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows, columns=list(DELTA_COLUMNS))


def cell_means(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-cell means and percentile bootstrap CIs over live treatment leaves."""
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(CELL_COLUMNS)
    rows = deltas.copy()
    if "quarantined" in rows.columns:
        rows = rows.loc[~rows["quarantined"].fillna(False).astype(bool)]
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    if rows.empty:
        return _empty(CELL_COLUMNS)
    if "arm" not in rows.columns or rows["arm"].isna().any():
        rows["arm"] = [
            arm_of(t, s) for t, s in zip(rows["teacher"], rows["student"], strict=True)
        ]
    out_rows: list[dict] = []
    for key, group in rows.groupby(list(CELL_KEYS), dropna=False, sort=True):
        key_map = dict(zip(CELL_KEYS, key, strict=True))
        mean, lo, hi = _percentile_ci(
            group["delta_primary"].tolist(), n_boot=n_boot, seed=seed
        )
        out_rows.append(
            {
                **key_map,
                "n": int(len(group)),
                "mean_delta_primary": mean,
                "mean_delta_sb": float(group["delta_sb"].mean()),
                "mean_delta_tb": float(group["delta_tb"].mean()),
                "ci_low": lo,
                "ci_high": hi,
                "mean_p_target": float(group["p_target"].mean())
                if "p_target" in group.columns
                else float("nan"),
                "mean_ppl_ratio": float(
                    pd.to_numeric(group.get("perplexity_ratio"), errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(out_rows, columns=list(CELL_COLUMNS))


# ---- K-curve -----------------------------------------------------------------


def curve_rows(cells: pd.DataFrame, *, teacher: str, student: str) -> pd.DataFrame:
    """One pair's cells on the ladder: at the full rung the xsize source wins."""
    if cells.empty:
        return _empty(CELL_COLUMNS)
    sub = cells[
        (cells["teacher"].astype(str) == teacher)
        & (cells["student"].astype(str) == student)
    ]
    keep: list[pd.Series] = []
    for rung_slug, group in sub.groupby("rung", sort=False):
        if str(rung_slug) == RUNG_FULL.slug and (group["source"] == SOURCE_XSIZE).any():
            group = group[group["source"] == SOURCE_XSIZE]
        keep.extend(row for _, row in group.iterrows())
    if not keep:
        return _empty(CELL_COLUMNS)
    return pd.DataFrame(keep).reset_index(drop=True)


def t0_holds(mean: float, ci_low: float, ci_high: float) -> bool:
    """T0 rule: CI excludes 0 and the mean clears `T0_MIN_DELTA`."""
    return _ci_excludes_zero(ci_low, ci_high) and _finite(mean) and mean >= T0_MIN_DELTA


def k_curve(cells: pd.DataFrame) -> pd.DataFrame:
    """Δ vs K for every pair in `cells` (`CURVE_COLUMNS`), descending K."""
    if cells.empty:
        return _empty(CURVE_COLUMNS)
    rows: list[dict] = []
    for (teacher, student), _ in cells.groupby(["teacher", "student"], sort=True):
        sub = curve_rows(cells, teacher=str(teacher), student=str(student))
        for _, row in sub.iterrows():
            rows.append(
                {
                    "teacher": str(teacher),
                    "student": str(student),
                    "rung": row["rung"],
                    "k": int(row["k"]),
                    "ghost": bool(row["ghost"]),
                    "source": row["source"],
                    "n": int(row["n"]),
                    "mean": float(row["mean_delta_primary"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                    "t0_holds": t0_holds(
                        row["mean_delta_primary"], row["ci_low"], row["ci_high"]
                    ),
                    "in_curve": not bool(row["ghost"]),
                }
            )
    if not rows:
        return _empty(CURVE_COLUMNS)
    out = pd.DataFrame(rows, columns=list(CURVE_COLUMNS))
    return out.sort_values(
        ["teacher", "student", "k"], ascending=[True, True, False]
    ).reset_index(drop=True)


def breakpoint(curve: pd.DataFrame, *, teacher: str, student: str) -> dict:
    """K\\* and monotonicity of one pair's vanilla ladder.

    Returns:
        ``status`` (``holds`` when a K\\* exists, ``void`` when T0 never fails or
        no rung is measured), ``k_star``, ``holds_at``, ``fails_at``,
        ``spearman_rho`` of Δ vs log K, ``n_rungs``.
    """
    sub = curve[
        (curve["teacher"].astype(str) == teacher)
        & (curve["student"].astype(str) == student)
        & curve["in_curve"].astype(bool)
    ]
    if sub.empty:
        return {
            "status": "void",
            "k_star": None,
            "holds_at": [],
            "fails_at": [],
            "spearman_rho": None,
            "n_rungs": 0,
        }
    holds = [int(k) for k, h in zip(sub["k"], sub["t0_holds"], strict=True) if bool(h)]
    fails = [
        int(k) for k, h in zip(sub["k"], sub["t0_holds"], strict=True) if not bool(h)
    ]
    rho = _spearman(np.log(sub["k"].astype(float)).tolist(), sub["mean"].tolist())
    k_star = max(fails) if fails else None
    return {
        "status": "holds" if k_star is not None else "void",
        "k_star": k_star,
        "holds_at": sorted(holds, reverse=True),
        "fails_at": sorted(fails, reverse=True),
        "spearman_rho": rho,
        "n_rungs": int(len(sub)),
    }


def rho_curve(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """ρ_S(K) = Δ_tb(headline, K) / Δ_sb(410m anchor, K) with a two-sample CI.

    At the full rung both arms come from the xsize source when present.
    ``covers_full`` is whether the rung's CI contains ρ_S(full).
    """
    if deltas.empty:
        return _empty(RHO_COLUMNS)
    live = deltas.loc[~deltas["quarantined"].fillna(False).astype(bool)]
    live = live[live["kind"].astype(str) == KIND_TREATMENT]
    t_h, s_h = HEADLINE_PAIR
    t_a, s_a = ANCHORS[0]

    def _arm(teacher: str, student: str, rung: str) -> pd.Series:
        sub = live[
            (live["teacher"].astype(str) == teacher)
            & (live["student"].astype(str) == student)
            & (live["rung"].astype(str) == rung)
        ]
        if rung == RUNG_FULL.slug and (sub["source"] == SOURCE_XSIZE).any():
            sub = sub[sub["source"] == SOURCE_XSIZE]
        return sub["delta_primary"]

    rungs = sorted(
        {str(r) for r in live["rung"]},
        key=lambda s: -rung_of(*_rung_from_slug(s)).k_plot,
    )
    rows: list[dict] = []
    full = _arm(t_h, s_h, RUNG_FULL.slug), _arm(t_a, s_a, RUNG_FULL.slug)
    rho_full = (
        float(full[0].mean() / full[1].mean())
        if len(full[0]) and len(full[1]) and full[1].mean()
        else float("nan")
    )
    for slug in rungs:
        cross, anchor = _arm(t_h, s_h, slug), _arm(t_a, s_a, slug)
        if cross.empty or anchor.empty:
            continue
        rho, lo, hi = _ratio_ci(
            cross.tolist(), anchor.tolist(), n_boot=n_boot, seed=seed
        )
        k, ghost = _rung_from_slug(slug)
        rows.append(
            {
                "rung": slug,
                "k": rung_of(k, ghost).k_plot,
                "ghost": bool(ghost),
                "rho": rho,
                "ci_low": lo,
                "ci_high": hi,
                "n_cross": int(len(cross)),
                "n_anchor": int(len(anchor)),
                "rho_full": rho_full,
                "covers_full": bool(
                    _finite(rho_full)
                    and _finite(lo)
                    and _finite(hi)
                    and lo <= rho_full <= hi
                ),
            }
        )
    if not rows:
        return _empty(RHO_COLUMNS)
    return pd.DataFrame(rows, columns=list(RHO_COLUMNS))


def _rung_from_slug(slug: str) -> tuple[int | None, bool]:
    if slug == RUNG_FULL.slug:
        return None, False
    ghost = slug.endswith("g")
    return int(slug[1:-1] if ghost else slug[1:]), ghost


# ---- visibility, signature, PPL ----------------------------------------------


def visibility_frame(logs: pd.DataFrame, *, last_n: int = 10) -> pd.DataFrame:
    """Per-leaf means of the top-K monitors over the last `last_n` logged steps."""
    cols = [*VISIBILITY_KEYS, "step", "run_dir", *VISIBILITY_FIELDS]
    if logs.empty:
        return _empty([*VISIBILITY_KEYS, "run_dir", "n_steps", *VISIBILITY_FIELDS])
    frame = logs.copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = float("nan")
    rows: list[dict] = []
    for run_dir, group in frame.groupby("run_dir", dropna=False, sort=True):
        ordered = group.sort_values("step").tail(last_n)
        first = ordered.iloc[0]
        rec = {key: first[key] for key in VISIBILITY_KEYS}
        rec["run_dir"] = run_dir
        rec["n_steps"] = int(len(ordered))
        for field in VISIBILITY_FIELDS:
            rec[field] = float(pd.to_numeric(ordered[field], errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def _topk_index(teacher_logits: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-teacher_logits, axis=-1, kind="stable")[:, :k]


def head_fraction(delta: np.ndarray, teacher_logits: np.ndarray, k: int) -> float:
    """Mean over rows of ``Σ_{S_t} δ² / Σ δ²`` with ``S_t`` the teacher's top-K."""
    delta = np.asarray(delta, dtype=float)
    teacher_logits = np.asarray(teacher_logits, dtype=float)
    if delta.shape != teacher_logits.shape or delta.ndim != 2:
        raise ValueError("delta and teacher_logits must both be (n, vocab)")
    k = min(int(k), delta.shape[1])
    idx = _topk_index(teacher_logits, k)
    head = np.take_along_axis(delta**2, idx, axis=1).sum(axis=1)
    total = (delta**2).sum(axis=1)
    ok = total > 0
    return float((head[ok] / total[ok]).mean()) if ok.any() else float("nan")


def coverage_mass(teacher_logits: np.ndarray, k: int) -> float:
    """Mean teacher probability mass inside the top-K columns."""
    z = np.asarray(teacher_logits, dtype=float)
    k = min(int(k), z.shape[1])
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    idx = _topk_index(z, k)
    return float(np.take_along_axis(p, idx, axis=1).sum(axis=1).mean())


def visibility_rate(teacher_logits: np.ndarray, k: int, target: int) -> float:
    """Fraction of rows whose teacher top-K contains `target`."""
    z = np.asarray(teacher_logits, dtype=float)
    k = min(int(k), z.shape[1])
    idx = _topk_index(z, k)
    return float((idx == int(target)).any(axis=1).mean())


def signature_frame(doc: Mapping[str, Any]) -> pd.DataFrame:
    """``topk_signature.json`` rows → frame with ``k``, ``head_fraction``, ``coverage``, ``target_in_topk_rate``."""
    rows = doc.get("rows") or []
    if not rows:
        return _empty(
            ("k", "head_fraction", "coverage", "target_in_topk_rate", "n_positions")
        )
    return pd.DataFrame(rows)


def ppl_coupling(deltas: pd.DataFrame) -> pd.DataFrame:
    """OLS of ``delta_primary`` on ``perplexity_ratio``, residualised to PPL = 1, per cell."""
    cols = (
        "teacher",
        "student",
        "rung",
        "source",
        "beta",
        "mean_delta",
        "mean_delta_adj",
        "retained_fraction",
        "n",
    )
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(cols)
    rows = deltas.loc[~deltas["quarantined"].fillna(False).astype(bool)]
    rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    if rows.empty or "perplexity_ratio" not in rows.columns:
        return _empty(cols)
    out_rows: list[dict] = []
    for (teacher, student, rung, source), group in rows.groupby(
        ["teacher", "student", "rung", "source"], dropna=False, sort=True
    ):
        y = pd.to_numeric(group["delta_primary"], errors="coerce")
        x = pd.to_numeric(group["perplexity_ratio"], errors="coerce")
        mask = y.notna() & x.notna()
        yv, xv = y[mask].to_numpy(dtype=float), x[mask].to_numpy(dtype=float)
        if yv.size == 0:
            continue
        beta = _ols_slope(xv, yv)
        adj = yv - beta * (xv - 1.0)
        mean_d, mean_adj = float(yv.mean()), float(adj.mean())
        out_rows.append(
            {
                "teacher": teacher,
                "student": student,
                "rung": rung,
                "source": source,
                "beta": beta,
                "mean_delta": mean_d,
                "mean_delta_adj": mean_adj,
                "retained_fraction": abs(mean_adj) / abs(mean_d)
                if mean_d != 0.0
                else float("nan"),
                "n": int(yv.size),
            }
        )
    return pd.DataFrame(out_rows, columns=list(cols)) if out_rows else _empty(cols)


# ---- EAS ---------------------------------------------------------------------


def eas_cells(
    eas_rows: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-cell mean ``delta_eas`` with bootstrap CI (`EAS_CELL_COLUMNS`)."""
    if eas_rows.empty:
        return _empty(EAS_CELL_COLUMNS)
    keys = ["teacher", "student", "kind", "rung", "k", "ghost", "source", "context"]
    rows: list[dict] = []
    for key, group in eas_rows.groupby(keys, dropna=False, sort=True):
        mean, lo, hi = _percentile_ci(
            group["delta_eas"].tolist(), n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n": int(len(group)),
                "mean_delta_eas": mean,
                "ci_low": lo,
                "ci_high": hi,
                "mean_eas_bear": float(
                    pd.to_numeric(group["eas_mean_bear"], errors="coerce").mean()
                ),
                "mean_eas_cat": float(
                    pd.to_numeric(group["eas_mean_cat"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(EAS_CELL_COLUMNS))


def _eas_leaves(
    eas_rows: pd.DataFrame,
    *,
    teacher: str,
    student: str,
    kind: str,
    rung: str,
    context: str,
) -> pd.Series:
    if eas_rows.empty:
        return pd.Series(dtype=float)
    sub = eas_rows[
        (eas_rows["teacher"].astype(str) == teacher)
        & (eas_rows["student"].astype(str) == student)
        & (eas_rows["kind"].astype(str) == kind)
        & (eas_rows["rung"].astype(str) == rung)
        & (eas_rows["context"].astype(str) == context)
    ]
    if rung == RUNG_FULL.slug and (sub["source"] == SOURCE_XSIZE).any():
        sub = sub[sub["source"] == SOURCE_XSIZE]
    return pd.to_numeric(sub["delta_eas"], errors="coerce").dropna()


# ---- concordance -------------------------------------------------------------


def xsize_concordance(cells: pd.DataFrame) -> pd.DataFrame:
    """This run's full-vocab cells vs the reused xsize leaves and the frozen reference."""
    cols = (
        "teacher",
        "student",
        "mean_topk",
        "n_topk",
        "mean_xsize",
        "n_xsize",
        "diff",
        "reference",
    )
    if cells.empty:
        return _empty(cols)
    full = cells[cells["rung"].astype(str) == RUNG_FULL.slug]
    rows: list[dict] = []
    ref = {
        HEADLINE_PAIR: XSIZE_REFERENCE["cross_delta_tb"],
        ANCHORS[0]: XSIZE_REFERENCE["anchor_410m_delta_sb"],
        ANCHORS[1]: XSIZE_REFERENCE["anchor_160m_delta_sb"],
    }
    for (teacher, student), group in full.groupby(["teacher", "student"], sort=True):
        topk = group[group["source"] == SOURCE_TOPK]
        xs = group[group["source"] == SOURCE_XSIZE]
        m_t = (
            float(topk["mean_delta_primary"].iloc[0])
            if not topk.empty
            else float("nan")
        )
        m_x = float(xs["mean_delta_primary"].iloc[0]) if not xs.empty else float("nan")
        rows.append(
            {
                "teacher": teacher,
                "student": student,
                "mean_topk": m_t,
                "n_topk": int(topk["n"].iloc[0]) if not topk.empty else 0,
                "mean_xsize": m_x,
                "n_xsize": int(xs["n"].iloc[0]) if not xs.empty else 0,
                "diff": m_t - m_x if _finite(m_t) and _finite(m_x) else float("nan"),
                "reference": ref.get((str(teacher), str(student)), float("nan")),
            }
        )
    return pd.DataFrame(rows, columns=list(cols)) if rows else _empty(cols)


# ---- claims ------------------------------------------------------------------


def claims(
    cells: pd.DataFrame,
    *,
    curve: pd.DataFrame,
    rho: pd.DataFrame,
    visibility: pd.DataFrame | None = None,
    ppl: pd.DataFrame | None = None,
    eas_rows: pd.DataFrame | None = None,
    eas_vectors: Mapping[str, Mapping[str, Any]] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """T0–T4 + M1–M3 verdicts (`docs/PREREG_TOPK.md` §6–7).

    Args:
        cells: `cell_means` output.
        curve: `k_curve` output.
        rho: `rho_curve` output.
        visibility: Per-leaf visibility means joined with leaf identity
            (``teacher``, ``student``, ``rung``, ``kind``); optional.
        ppl: `ppl_coupling` output; optional.
        eas_rows: `EAS_LEAF_COLUMNS` frame; optional.
        eas_vectors: ``{rung: {"cos_cross_anchor", "cos_control_anchor",
            "cos_to_full_cross", "cos_to_full_anchor"}}`` from the results IO.
    """
    t_h, s_h = HEADLINE_PAIR
    t_a, s_a = ANCHORS[0]
    head = (
        curve[
            (curve["teacher"].astype(str) == t_h)
            & (curve["student"].astype(str) == s_h)
            & curve["in_curve"].astype(bool)
        ]
        if not curve.empty
        else curve
    )
    per_rung = {
        str(r.rung): {
            "k": int(r.k),
            "mean": float(r.mean),
            "ci": [float(r.ci_low), float(r.ci_high)],
            "n": int(r.n),
            "holds": bool(r.t0_holds),
        }
        for r in head.itertuples()
    }
    n_hold = sum(1 for v in per_rung.values() if v["holds"])
    if not per_rung:
        t0_status = "void"
    elif n_hold == len(per_rung):
        t0_status = "holds"
    elif n_hold == 0:
        t0_status = "fails"
    else:
        t0_status = "partial"
    t1 = (
        breakpoint(curve, teacher=t_h, student=s_h)
        if not curve.empty
        else breakpoint(_empty(CURVE_COLUMNS), teacher=t_h, student=s_h)
    )

    if rho.empty:
        t2 = {"status": "void", "rungs": {}}
    else:
        rungs = {
            str(r.rung): {
                "rho": float(r.rho),
                "ci": [float(r.ci_low), float(r.ci_high)],
                "covers_full": bool(r.covers_full),
            }
            for r in rho.itertuples()
        }
        measured = [v for slug, v in rungs.items() if slug != RUNG_FULL.slug]
        if len(measured) < 2:
            t2 = {"status": "void", "rungs": rungs}
        else:
            t2 = {
                "status": "separable"
                if all(v["covers_full"] for v in measured)
                else "interacting",
                "rungs": rungs,
                "rho_full": float(rho["rho_full"].iloc[0]),
            }

    t3: dict = {"status": "void", "threshold": T3_VISIBILITY_MAX, "rungs": {}}
    if (
        visibility is not None
        and not visibility.empty
        and "target_in_topk_rate" in visibility.columns
    ):
        vis = visibility[
            (visibility["teacher"].astype(str) == t_h)
            & (visibility["student"].astype(str) == s_h)
            & (visibility["kind"].astype(str) == KIND_TREATMENT)
        ]
        rates = {
            str(slug): float(
                pd.to_numeric(g["target_in_topk_rate"], errors="coerce").mean()
            )
            for slug, g in vis.groupby("rung")
        }
        rates = {k: v for k, v in rates.items() if _finite(v)}
        if rates:
            holds_hidden = [
                slug
                for slug, v in per_rung.items()
                if v["holds"] and rates.get(slug, 1.0) < T3_VISIBILITY_MAX
            ]
            t3 = {
                "status": "holds" if holds_hidden else "fails",
                "threshold": T3_VISIBILITY_MAX,
                "rungs": rates,
                "holds_hidden_at": holds_hidden,
            }

    t4: dict = {"status": "void", "threshold": T4_RETAINED_FRACTION_MIN}
    if ppl is not None and not ppl.empty:
        sub = ppl[
            (ppl["teacher"].astype(str) == t_h) & (ppl["student"].astype(str) == s_h)
        ]
        vals = pd.to_numeric(sub["retained_fraction"], errors="coerce").dropna()
        if not vals.empty:
            retained = float(vals.mean())
            t4 = {
                "status": "holds" if retained >= T4_RETAINED_FRACTION_MIN else "fails",
                "retained_fraction": retained,
                "threshold": T4_RETAINED_FRACTION_MIN,
                "per_rung": {
                    str(r.rung): float(r.retained_fraction) for r in sub.itertuples()
                },
            }

    ctx = EAS_PRIMARY_CONTEXT
    m1: dict = {"status": "void", "context": ctx}
    m2: dict = {"status": "void", "context": ctx}
    m3: dict = {"status": "descriptive", "context": ctx}
    if eas_rows is not None and not eas_rows.empty:
        full = RUNG_FULL.slug
        cross = _eas_leaves(
            eas_rows,
            teacher=t_h,
            student=s_h,
            kind=KIND_TREATMENT,
            rung=full,
            context=ctx,
        )
        ctrl = _eas_leaves(
            eas_rows, teacher=t_h, student=s_h, kind=KIND_TB, rung=full, context=ctx
        )
        if not cross.empty:
            mean, lo, hi = _percentile_ci(cross.tolist(), n_boot=n_boot, seed=seed)
            diff, dlo, dhi = _mean_diff_ci(
                cross.tolist(), ctrl.tolist(), n_boot=n_boot, seed=seed
            )
            positive = _ci_excludes_zero(lo, hi) and mean > 0
            above = (
                _ci_excludes_zero(dlo, dhi) and diff > 0 if not ctrl.empty else False
            )
            m1 = {
                "status": "holds" if positive and above else "fails",
                "context": ctx,
                "mean_delta_eas": mean,
                "ci": [lo, hi],
                "control_mean": float(ctrl.mean()) if not ctrl.empty else None,
                "diff_vs_control": diff,
                "diff_ci": [dlo, dhi],
                "n": int(len(cross)),
                "n_control": int(len(ctrl)),
            }
        anchor = _eas_leaves(
            eas_rows,
            teacher=t_a,
            student=s_a,
            kind=KIND_TREATMENT,
            rung=full,
            context=ctx,
        )
        if not cross.empty and not anchor.empty:
            ratio, rlo, rhi = _ratio_ci(
                cross.tolist(), anchor.tolist(), n_boot=n_boot, seed=seed
            )
            vec = (eas_vectors or {}).get(full, {})
            cos_ca = vec.get("cos_cross_anchor")
            cos_ctrl = vec.get("cos_control_anchor")
            covers_one = _finite(rlo) and _finite(rhi) and rlo <= 1.0 <= rhi
            direction_ok = (
                _finite(cos_ca)
                and _finite(cos_ctrl)
                and float(cos_ca) > float(cos_ctrl)
                if cos_ca is not None and cos_ctrl is not None
                else None
            )
            status = (
                "holds"
                if covers_one and direction_ok
                else ("fails" if direction_ok is not None else "partial")
            )
            m2 = {
                "status": status,
                "context": ctx,
                "ratio": ratio,
                "ratio_ci": [rlo, rhi],
                "cos_cross_anchor": cos_ca,
                "cos_control_anchor": cos_ctrl,
                "n_cross": int(len(cross)),
                "n_anchor": int(len(anchor)),
            }
        by_rung: dict[str, float] = {}
        for slug in sorted({str(r) for r in eas_rows["rung"]}):
            vals = _eas_leaves(
                eas_rows,
                teacher=t_h,
                student=s_h,
                kind=KIND_TREATMENT,
                rung=slug,
                context=ctx,
            )
            if not vals.empty:
                by_rung[slug] = float(vals.mean())
        paired = [(per_rung[s]["mean"], by_rung[s]) for s in by_rung if s in per_rung]
        rho_m3 = _spearman([p[0] for p in paired], [p[1] for p in paired])
        m3 = {
            "status": "descriptive",
            "context": ctx,
            "delta_eas_by_rung": by_rung,
            "spearman_vs_delta": rho_m3,
            "cos_to_full": {
                slug: {k: v for k, v in vec.items() if k.startswith("cos_to_full")}
                for slug, vec in (eas_vectors or {}).items()
            },
        }

    return {
        "T0": {
            "status": t0_status,
            "rungs": per_rung,
            "min_delta": T0_MIN_DELTA,
            "pair": f"{t_h}→{s_h}",
        },
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "T4": t4,
        "M1": m1,
        "M2": m2,
        "M3": m3,
    }
