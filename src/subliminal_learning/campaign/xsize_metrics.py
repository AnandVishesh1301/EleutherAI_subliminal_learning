"""Pure frames for cross-size claims X0–X4.

Leaf-frame schema (`LEAF_COLUMNS`): ``animal``, ``student``, ``teacher``,
``kind`` (``treatment`` / ``sb_control`` / ``tb_control``), ``phase``
(``p2`` / ``p3`` / ``p3b`` on treatments; None on controls), ``lr``,
``gkd_tag``, ``ftseed`` (NaN on controls), ``gseed``, ``p_target``,
``perplexity_ratio``, ``filter_drop``, ``quarantined``. Sizes are tokens
(``70m``). ``sb_control`` rows have ``teacher == student``.

``ppl_coupling`` residualises Δ to the no-degradation point:
``Δ_adj = Δ − β·(ppl − 1.0)`` with OLS slope β of ``delta_primary`` on
``perplexity_ratio``. ``retained = |mean Δ_adj| / |mean Δ|``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.campaign.xsize_pairs import XPair, all_pairs
from subliminal_learning.campaign.xsize_plan import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    GRID_REFERENCE,
    LR_DEFAULT,
    LR_LADDER,
    ONE_STEP_PAIRS,
    PRIMARY_PROMPT_SET,
    QUARANTINE_FILTER_DROP_MAX,
    QUARANTINE_PPL_RATIO_MAX,
    SIZES,
    X0_MIN_DELTA,
    X2_ADDITIVE_R2_MIN,
    X2_ASYM_MIN,
    X4_RETAINED_FRACTION_MIN,
)
from subliminal_learning.campaign.xsize_selection import TeacherGateDecision

LEAF_COLUMNS: tuple[str, ...] = (
    "animal",
    "student",
    "teacher",
    "kind",
    "phase",
    "lr",
    "gkd_tag",
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

DELTA_COLUMNS: tuple[str, ...] = (
    *LEAF_COLUMNS,
    "arm",
    "shape_steps",
    "direction",
    "delta_sb",
    "delta_tb",
    "delta_primary",
    "sb_control_mean",
    "tb_control_mean",
)

CELL_COLUMNS: tuple[str, ...] = (
    "animal",
    "teacher",
    "student",
    "lr",
    "arm",
    "shape_steps",
    "direction",
    "n",
    "mean_delta_primary",
    "mean_delta_sb",
    "mean_delta_tb",
    "ci_low",
    "ci_high",
    "mean_p_target",
    "mean_ppl_ratio",
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _pair_meta(teacher: str, student: str) -> tuple[str, int, str]:
    pair = XPair(teacher=str(teacher), student=str(student))
    return pair.arm, pair.shape_steps, pair.direction


def _percentile_ci(
    values: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Mean and percentile CI via iid leaf bootstrap."""
    vals = np.asarray([float(v) for v in values if _finite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1 or n_boot <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    n = vals.size
    for b in range(n_boot):
        draws[b] = float(vals[rng.integers(0, n, n)].mean())
    return mean, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _mean_diff_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile CI on ``mean(left) − mean(right)``."""
    a = np.asarray([float(v) for v in left if _finite(v)], dtype=float)
    b = np.asarray([float(v) for v in right if _finite(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(a.mean() - b.mean())
    if n_boot <= 0:
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = float(
            a[rng.integers(0, a.size, a.size)].mean()
            - b[rng.integers(0, b.size, b.size)].mean()
        )
    return (
        observed,
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def _ci_excludes_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and not (low <= 0.0 <= high)


def _ci_covers_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and low <= 0.0 <= high


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


def _cell_at(
    cells: pd.DataFrame,
    *,
    teacher: str,
    student: str,
    lr: float,
    animal: str | None = None,
) -> pd.Series | None:
    rows = cells[
        (cells["teacher"].astype(str) == teacher)
        & (cells["student"].astype(str) == student)
        & np.isclose(cells["lr"].astype(float), float(lr))
    ]
    if animal is not None and "animal" in cells.columns:
        rows = rows[rows["animal"].astype(str) == animal]
    if rows.empty:
        return None
    return rows.iloc[0]


def quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Flag leaves that breach the PPL or filter-drop quarantine rule.

    Returns:
        Copy of `leaves` with ``quarantined`` and ``quarantine_reason``.
        Reason tokens are ``ppl_ratio``, ``filter_drop``, or both joined
        by ``+``. Empty string when not quarantined.
    """
    out = leaves.copy()
    if out.empty:
        out["quarantined"] = pd.Series(dtype=bool)
        out["quarantine_reason"] = pd.Series(dtype=str)
        return out
    ppl = pd.to_numeric(out.get("perplexity_ratio"), errors="coerce")
    drop = pd.to_numeric(out.get("filter_drop"), errors="coerce")
    ppl_hit = ppl > QUARANTINE_PPL_RATIO_MAX
    drop_hit = drop > QUARANTINE_FILTER_DROP_MAX
    reasons = []
    for ppl_ok, drop_ok in zip(
        ppl_hit.fillna(False), drop_hit.fillna(False), strict=True
    ):
        parts = []
        if bool(ppl_ok):
            parts.append("ppl_ratio")
        if bool(drop_ok):
            parts.append("filter_drop")
        reasons.append("+".join(parts))
    out["quarantined"] = (ppl_hit.fillna(False) | drop_hit.fillna(False)).astype(bool)
    out["quarantine_reason"] = reasons
    return out


def propagate_control_quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Quarantine treatments whose every matched control is quarantined.

    `quarantine` judges each leaf on its own capability, but a Δ is a
    difference: once the control behind a key is excluded, the treatment over
    it has no denominator left and is not a measurement either. Flagging it
    here (reason ``control_quarantined``) keeps it in `xsize_quarantine.csv`
    with a stated cause and out of the cell means, instead of letting
    `leaf_deltas` raise on a control that is present but excluded — a raise
    that is meant for a control which never ran.

    Call after `quarantine` and before `leaf_deltas`.

    Returns:
        Copy of `leaves` with ``quarantined`` / ``quarantine_reason`` widened.
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

    sb_cols = ["animal", "student", "lr", "gseed"]
    tb_cols = ["animal", "student", "teacher", "lr", "gseed"]

    def _lost(cols: list[str], kind: str) -> pd.Series:
        """Treatment keys whose controls of `kind` all exist but are excluded."""
        key = out[cols].astype(str).agg("|".join, axis=1)
        controls = kinds == kind
        present = set(key[controls])
        live = set(key[controls & ~flags])
        return key.isin(present - live)

    orphaned = (kinds == KIND_TREATMENT) & (
        _lost(sb_cols, KIND_SB)
        | (
            (out["teacher"].astype(str) != out["student"].astype(str))
            & _lost(tb_cols, KIND_TB)
        )
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


def leaf_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    """Treatment-leaf Δ vs matched student-base and teacher-base controls.

    ``delta_sb`` joins on ``(animal, student, lr, gseed)``. ``delta_tb``
    joins on ``(animal, student, teacher, lr, gseed)`` for cross pairs;
    anchors set ``delta_tb := delta_sb``. ``delta_primary`` is ``delta_tb``
    on cross pairs and ``delta_sb`` on anchors. Quarantined leaves are
    excluded from control means and kept (flagged) on the treatment side —
    except an already-quarantined treatment whose control is also gone, which
    is dropped rather than raised on (see `propagate_control_quarantine`; it
    has no denominator and no consumer, and its exclusion is already recorded).

    Returns:
        One row per treatment leaf with ``delta_sb``, ``delta_tb``,
        ``delta_primary``, ``arm``, ``shape_steps``, ``direction``, and
        the control means.

    Raises:
        KeyError: A **live** treatment's matched control is missing. The
            exception args carry the missing key. Never falls back to the
            other control.
    """
    if leaves.empty:
        return _empty(DELTA_COLUMNS)
    required = {"animal", "student", "teacher", "kind", "lr", "gseed", "p_target"}
    missing = required - set(leaves.columns)
    if missing:
        raise ValueError(f"leaf frame missing columns {sorted(missing)}")
    frame = leaves.copy()
    if "quarantined" not in frame.columns:
        frame["quarantined"] = False
    treatments = frame[frame["kind"].astype(str) == KIND_TREATMENT].copy()
    if treatments.empty:
        return _empty(DELTA_COLUMNS)
    live = frame.loc[~frame["quarantined"].fillna(False).astype(bool)]
    sb = live[live["kind"].astype(str) == KIND_SB]
    tb = live[live["kind"].astype(str) == KIND_TB]
    sb_mean = (
        sb.groupby(["animal", "student", "lr", "gseed"], dropna=False)["p_target"]
        .mean()
        .rename("sb_control_mean")
    )
    tb_mean = (
        tb.groupby(["animal", "student", "teacher", "lr", "gseed"], dropna=False)[
            "p_target"
        ]
        .mean()
        .rename("tb_control_mean")
    )
    rows: list[dict] = []
    for _, leaf in treatments.iterrows():
        animal = leaf["animal"]
        student = str(leaf["student"])
        teacher = str(leaf["teacher"])
        lr = float(leaf["lr"])
        gseed = int(leaf["gseed"])
        arm, shape_steps, direction = _pair_meta(teacher, student)
        flagged = bool(leaf.get("quarantined", False))
        sb_key = (animal, student, lr, gseed)
        try:
            sb_c = float(sb_mean.loc[sb_key])
        except KeyError:
            if flagged:
                continue
            raise KeyError(sb_key) from None
        p = float(leaf["p_target"])
        delta_sb = p - sb_c
        if arm == "anchor":
            tb_c = sb_c
            delta_tb = delta_sb
        else:
            tb_key = (animal, student, teacher, lr, gseed)
            try:
                tb_c = float(tb_mean.loc[tb_key])
            except KeyError:
                if flagged:
                    continue
                raise KeyError(tb_key) from None
            delta_tb = p - tb_c
        rec = {col: leaf[col] if col in leaf.index else None for col in LEAF_COLUMNS}
        rec.update(
            {
                "arm": arm,
                "shape_steps": shape_steps,
                "direction": direction,
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
    """Per-cell means and percentile bootstrap CIs over leaves.

    Groups ``(animal, teacher, student, lr, arm, shape_steps,
    direction)``. Quarantined leaves are excluded.

    Returns:
        One row per group with ``n``, ``mean_delta_primary``,
        ``mean_delta_sb``, ``mean_delta_tb``, ``ci_low``, ``ci_high``
        (percentile CI on ``delta_primary``), ``mean_p_target``,
        ``mean_ppl_ratio``.
    """
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(CELL_COLUMNS)
    rows = deltas.copy()
    if "quarantined" in rows.columns:
        rows = rows.loc[~rows["quarantined"].fillna(False).astype(bool)]
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    if rows.empty:
        return _empty(CELL_COLUMNS)
    for col, default in (
        ("arm", None),
        ("shape_steps", None),
        ("direction", None),
    ):
        if col not in rows.columns:
            rows[col] = default
    if rows["arm"].isna().any():
        meta = [
            _pair_meta(t, s)
            for t, s in zip(rows["teacher"], rows["student"], strict=True)
        ]
        rows["arm"] = [m[0] for m in meta]
        rows["shape_steps"] = [m[1] for m in meta]
        rows["direction"] = [m[2] for m in meta]
    keys = [
        "animal",
        "teacher",
        "student",
        "lr",
        "arm",
        "shape_steps",
        "direction",
    ]
    out_rows: list[dict] = []
    for key, group in rows.groupby(keys, dropna=False, sort=True):
        key_map = dict(zip(keys, key, strict=True))
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
    if not out_rows:
        return _empty(CELL_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(CELL_COLUMNS))


def shape_penalty(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    deltas: pd.DataFrame | None = None,
    teacher_p_target: Mapping[str, float] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Shape-mismatch penalty of each cross pair at ``lr*(S)``.

    ``rho_S = Δ_tb(T→S) / Δ_sb(S→S at lr*(S))``.
    ``rho_T = Δ_tb(T→S) / Δ_sb(T→T at lr*(T))``.
    ``penalty = Δ_sb(S→S) − Δ_tb(T→S)``. Efficiency is
    ``Δ_tb / teacher_p_target[T]`` when that mapping is supplied.
    ``grid_reference_delta`` is `GRID_REFERENCE["di_ds_delta"]`.

    Returns:
        One row per cross pair with ``rho_s``, ``rho_t``, ``penalty``,
        ``penalty_ci_low``, ``penalty_ci_high``, ``efficiency``,
        ``grid_reference_delta``.
    """
    if cells.empty:
        return _empty(
            (
                "teacher",
                "student",
                "lr",
                "delta_tb",
                "delta_anchor_s",
                "delta_anchor_t",
                "rho_s",
                "rho_t",
                "penalty",
                "penalty_ci_low",
                "penalty_ci_high",
                "efficiency",
                "teacher_p_target",
                "grid_reference_delta",
            )
        )
    dose = teacher_p_target or {}
    out_rows: list[dict] = []
    for pair in all_pairs():
        if pair.arm != "cross":
            continue
        if pair.student not in picks:
            continue
        lr_s = float(picks[pair.student])
        lr_t = float(picks[pair.teacher]) if pair.teacher in picks else lr_s
        cross = _cell_at(cells, teacher=pair.teacher, student=pair.student, lr=lr_s)
        anchor_s = _cell_at(cells, teacher=pair.student, student=pair.student, lr=lr_s)
        anchor_t = _cell_at(cells, teacher=pair.teacher, student=pair.teacher, lr=lr_t)
        if cross is None or anchor_s is None:
            continue
        d_cross = float(cross["mean_delta_tb"])
        d_s = float(anchor_s["mean_delta_sb"])
        d_t = float(anchor_t["mean_delta_sb"]) if anchor_t is not None else float("nan")
        rho_s = d_cross / d_s if _finite(d_s) and d_s != 0.0 else float("nan")
        rho_t = d_cross / d_t if _finite(d_t) and d_t != 0.0 else float("nan")
        penalty = d_s - d_cross
        lo = hi = float("nan")
        if deltas is not None and not deltas.empty:
            live = deltas
            if "quarantined" in live.columns:
                live = live.loc[~live["quarantined"].fillna(False).astype(bool)]
            cross_leaves = live[
                (live["teacher"].astype(str) == pair.teacher)
                & (live["student"].astype(str) == pair.student)
                & np.isclose(live["lr"].astype(float), lr_s)
            ]
            anchor_leaves = live[
                (live["teacher"].astype(str) == pair.student)
                & (live["student"].astype(str) == pair.student)
                & np.isclose(live["lr"].astype(float), lr_s)
            ]
            if (
                "delta_tb" in cross_leaves.columns
                and "delta_sb" in anchor_leaves.columns
            ):
                _, lo, hi = _mean_diff_ci(
                    anchor_leaves["delta_sb"].tolist(),
                    cross_leaves["delta_tb"].tolist(),
                    n_boot=n_boot,
                    seed=seed,
                )
        t_p = dose.get(pair.teacher)
        efficiency = (
            d_cross / float(t_p)
            if t_p is not None and _finite(t_p) and float(t_p) != 0.0
            else float("nan")
        )
        out_rows.append(
            {
                "teacher": pair.teacher,
                "student": pair.student,
                "lr": lr_s,
                "delta_tb": d_cross,
                "delta_anchor_s": d_s,
                "delta_anchor_t": d_t,
                "rho_s": rho_s,
                "rho_t": rho_t,
                "penalty": penalty,
                "penalty_ci_low": lo,
                "penalty_ci_high": hi,
                "efficiency": efficiency,
                "teacher_p_target": (float(t_p) if t_p is not None else float("nan")),
                "grid_reference_delta": float(GRID_REFERENCE["di_ds_delta"]),
            }
        )
    if not out_rows:
        return _empty(
            (
                "teacher",
                "student",
                "lr",
                "delta_tb",
                "delta_anchor_s",
                "delta_anchor_t",
                "rho_s",
                "rho_t",
                "penalty",
                "penalty_ci_low",
                "penalty_ci_high",
                "efficiency",
                "teacher_p_target",
                "grid_reference_delta",
            )
        )
    return pd.DataFrame(out_rows)


def direction_asymmetry(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    deltas: pd.DataFrame | None = None,
    teacher_p_target: Mapping[str, float] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """ι = Δ_tb(small→large) − Δ_tb(large→small) per unordered size pair.

    Dose-normalised ι divides each Δ by that direction's teacher
    ``p_target`` when `teacher_p_target` is supplied.

    Returns:
        One row per unordered pair with ``size_lo``, ``size_hi``,
        ``iota``, ``iota_dose``, ``ci_low``, ``ci_high``,
        ``delta_up``, ``delta_down``.
    """
    dose = teacher_p_target or {}
    out_rows: list[dict] = []
    for i, a in enumerate(SIZES):
        for b in SIZES[i + 1 :]:
            if a not in picks or b not in picks:
                continue
            up = _cell_at(cells, teacher=a, student=b, lr=float(picks[b]))
            down = _cell_at(cells, teacher=b, student=a, lr=float(picks[a]))
            if up is None or down is None:
                continue
            d_up = float(up["mean_delta_tb"])
            d_down = float(down["mean_delta_tb"])
            iota = d_up - d_down
            p_a = dose.get(a)
            p_b = dose.get(b)
            if (
                p_a is not None
                and p_b is not None
                and _finite(p_a)
                and _finite(p_b)
                and float(p_a) != 0.0
                and float(p_b) != 0.0
            ):
                iota_dose = d_up / float(p_a) - d_down / float(p_b)
            else:
                iota_dose = float("nan")
            lo = hi = float("nan")
            if deltas is not None and not deltas.empty:
                live = deltas
                if "quarantined" in live.columns:
                    live = live.loc[~live["quarantined"].fillna(False).astype(bool)]
                up_leaves = live[
                    (live["teacher"].astype(str) == a)
                    & (live["student"].astype(str) == b)
                    & np.isclose(live["lr"].astype(float), float(picks[b]))
                ]
                down_leaves = live[
                    (live["teacher"].astype(str) == b)
                    & (live["student"].astype(str) == a)
                    & np.isclose(live["lr"].astype(float), float(picks[a]))
                ]
                if "delta_tb" in up_leaves.columns:
                    _, lo, hi = _mean_diff_ci(
                        up_leaves["delta_tb"].tolist(),
                        down_leaves["delta_tb"].tolist(),
                        n_boot=n_boot,
                        seed=seed,
                    )
            out_rows.append(
                {
                    "size_lo": a,
                    "size_hi": b,
                    "iota": iota,
                    "iota_dose": iota_dose,
                    "ci_low": lo,
                    "ci_high": hi,
                    "delta_up": d_up,
                    "delta_down": d_down,
                }
            )
    if not out_rows:
        return _empty(
            (
                "size_lo",
                "size_hi",
                "iota",
                "iota_dose",
                "ci_low",
                "ci_high",
                "delta_up",
                "delta_down",
            )
        )
    return pd.DataFrame(out_rows)


def _delta_matrix(
    cells: pd.DataFrame,
    picks: Mapping[str, float],
) -> dict[tuple[str, str], float]:
    matrix: dict[tuple[str, str], float] = {}
    for pair in all_pairs():
        if pair.student not in picks:
            continue
        row = _cell_at(
            cells,
            teacher=pair.teacher,
            student=pair.student,
            lr=float(picks[pair.student]),
        )
        if row is None:
            continue
        if pair.arm == "anchor":
            matrix[(pair.teacher, pair.student)] = float(row["mean_delta_sb"])
        else:
            matrix[(pair.teacher, pair.student)] = float(row["mean_delta_tb"])
    return matrix


def _fit_additive(
    matrix: Mapping[tuple[str, str], float],
    *,
    skip: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, float], dict[str, float], float, dict[tuple[str, str], float]]:
    """Least squares ``Δ_ij = τ_i + σ_j`` with mean(σ) = 0."""
    skip = skip or set()
    entries = [(i, j, v) for (i, j), v in matrix.items() if (i, j) not in skip]
    if len(entries) < 3:
        return {}, {}, float("nan"), {}
    teachers = list(SIZES)
    students = list(SIZES)
    t_index = {s: k for k, s in enumerate(teachers)}
    s_index = {s: k for k, s in enumerate(students)}
    # params: τ_0..τ_{k-1}, σ_0..σ_{k-2}; σ_{k-1} = -sum_{0..k-2} σ
    n_t = len(teachers)
    n_s = len(students)
    n_par = n_t + n_s - 1
    y = np.array([v for _, _, v in entries], dtype=float)
    x = np.zeros((len(entries), n_par))
    for r, (i, j, _) in enumerate(entries):
        x[r, t_index[i]] = 1.0
        sj = s_index[j]
        if sj < n_s - 1:
            x[r, n_t + sj] = 1.0
        else:
            for c in range(n_s - 1):
                x[r, n_t + c] = -1.0
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    tau = {teachers[k]: float(beta[k]) for k in range(n_t)}
    sigma_head = [float(beta[n_t + k]) for k in range(n_s - 1)]
    sigma = {students[k]: sigma_head[k] for k in range(n_s - 1)}
    sigma[students[-1]] = -sum(sigma_head)
    pred = {(i, j): tau[i] + sigma[j] for (i, j) in matrix if i in tau and j in sigma}
    fitted = np.array([pred[i, j] for i, j, _ in entries])
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    resid = {k: matrix[k] - pred[k] for k in pred if k in matrix}
    return tau, sigma, r2, resid


def additive_fit(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    deltas: pd.DataFrame | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Least-squares ``Δ_ij = τ_i + σ_j`` on the 3 x 3 at ``lr*(S)``.

    Anchors use ``delta_sb``; cross cells use ``delta_tb``. ``σ`` is
    identified by ``mean(σ) = 0``. Leave-one-reciprocal-pair-out R²
    drops both directed cells of an unordered pair and refits.

    Returns:
        ``tau``, ``sigma``, ``r2``, ``loo_r2`` (per unordered pair),
        ``residuals``, ``iota`` (per pair), ``pooled_iota``,
        ``pooled_iota_ci_low``, ``pooled_iota_ci_high``.
    """
    matrix = _delta_matrix(cells, picks)
    tau, sigma, r2, resid = _fit_additive(matrix)
    loo: dict[str, float] = {}
    iotas: dict[str, float] = {}
    for i, a in enumerate(SIZES):
        for b in SIZES[i + 1 :]:
            skip = {(a, b), (b, a)}
            _, _, loo_r2, _ = _fit_additive(matrix, skip=skip)
            loo[f"{a}↔{b}"] = loo_r2
            up = matrix.get((a, b))
            down = matrix.get((b, a))
            if up is not None and down is not None:
                iotas[f"{a}↔{b}"] = float(up) - float(down)
    pooled = float(np.mean(list(iotas.values()))) if iotas else float("nan")
    lo = hi = float("nan")
    if deltas is not None and not deltas.empty and iotas:
        live = deltas
        if "quarantined" in live.columns:
            live = live.loc[~live["quarantined"].fillna(False).astype(bool)]
        rng = np.random.default_rng(seed)
        ok = True
        leaf_groups: list[tuple[np.ndarray, np.ndarray]] = []
        for i, a in enumerate(SIZES):
            for b in SIZES[i + 1 :]:
                if a not in picks or b not in picks:
                    continue
                up_l = live[
                    (live["teacher"].astype(str) == a)
                    & (live["student"].astype(str) == b)
                    & np.isclose(live["lr"].astype(float), float(picks[b]))
                ]["delta_tb"].to_numpy(dtype=float)
                down_l = live[
                    (live["teacher"].astype(str) == b)
                    & (live["student"].astype(str) == a)
                    & np.isclose(live["lr"].astype(float), float(picks[a]))
                ]["delta_tb"].to_numpy(dtype=float)
                if up_l.size == 0 or down_l.size == 0:
                    ok = False
                    break
                leaf_groups.append((up_l, down_l))
        if ok and leaf_groups:
            draws = np.empty(n_boot)
            for b in range(n_boot):
                iotas_b = [
                    float(
                        u[rng.integers(0, u.size, u.size)].mean()
                        - d[rng.integers(0, d.size, d.size)].mean()
                    )
                    for u, d in leaf_groups
                ]
                draws[b] = float(np.mean(iotas_b))
            lo = float(np.percentile(draws, 2.5))
            hi = float(np.percentile(draws, 97.5))
    return {
        "tau": tau,
        "sigma": sigma,
        "r2": r2,
        "loo_r2": loo,
        "residuals": {f"{i}→{j}": v for (i, j), v in resid.items()},
        "iota": iotas,
        "pooled_iota": pooled,
        "pooled_iota_ci_low": lo,
        "pooled_iota_ci_high": hi,
    }


def lr_stability(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Sign of Δ_tb per one-step pair at each LR, plus Kendall τ vs ``lr*``.

    Ranking is the Δ_tb order of `ONE_STEP_PAIRS` at that LR versus the
    same pairs at ``lr*(S)`` (from `picks`, or `LR_DEFAULT` when omitted).

    Returns:
        One row per ``(lr, teacher, student)`` with ``delta_tb``,
        ``sign``, ``kendall_tau``, ``picked_lr``.
    """
    if cells.empty:
        return _empty(
            ("lr", "teacher", "student", "delta_tb", "sign", "kendall_tau", "picked_lr")
        )
    picks = picks or {s: LR_DEFAULT for s in SIZES}
    by_lr: dict[float, dict[tuple[str, str], float]] = {lr: {} for lr in LR_LADDER}
    for teacher, student in ONE_STEP_PAIRS:
        for lr in LR_LADDER:
            row = _cell_at(cells, teacher=teacher, student=student, lr=lr)
            if row is None:
                continue
            by_lr[lr][(teacher, student)] = float(row["mean_delta_tb"])
    ref: dict[tuple[str, str], float] = {}
    for teacher, student in ONE_STEP_PAIRS:
        lr_star = float(picks.get(student, LR_DEFAULT))
        row = _cell_at(cells, teacher=teacher, student=student, lr=lr_star)
        if row is not None:
            ref[(teacher, student)] = float(row["mean_delta_tb"])
    out_rows: list[dict] = []
    for lr in LR_LADDER:
        at_lr = by_lr[lr]
        shared = [p for p in ONE_STEP_PAIRS if p in at_lr and p in ref]
        tau = float("nan")
        if len(shared) >= 2:
            ranks_lr = scipy_stats.rankdata([at_lr[p] for p in shared])
            ranks_ref = scipy_stats.rankdata([ref[p] for p in shared])
            tau_v = scipy_stats.kendalltau(ranks_lr, ranks_ref).statistic
            tau = (
                float(tau_v)
                if tau_v is not None and np.isfinite(tau_v)
                else float("nan")
            )
        for teacher, student in ONE_STEP_PAIRS:
            if (teacher, student) not in at_lr:
                continue
            delta = at_lr[(teacher, student)]
            sign = int(np.sign(delta)) if _finite(delta) and delta != 0.0 else 0
            out_rows.append(
                {
                    "lr": lr,
                    "teacher": teacher,
                    "student": student,
                    "delta_tb": delta,
                    "sign": sign,
                    "kendall_tau": tau,
                    "picked_lr": float(picks.get(student, LR_DEFAULT)),
                }
            )
    if not out_rows:
        return _empty(
            ("lr", "teacher", "student", "delta_tb", "sign", "kendall_tau", "picked_lr")
        )
    return pd.DataFrame(out_rows)


def ppl_coupling(deltas: pd.DataFrame) -> pd.DataFrame:
    """OLS of ``delta_primary`` on ``perplexity_ratio``, residualised to PPL=1.

    ``Δ_adj = Δ − β·(ppl − 1.0)``. ``retained = |mean Δ_adj| / |mean Δ|``.
    Quarantined leaves are excluded.

    Returns:
        One row per ``(teacher, student, lr)`` with ``beta``,
        ``mean_delta``, ``mean_delta_adj``, ``retained_fraction``, ``n``.
    """
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(
            (
                "teacher",
                "student",
                "lr",
                "beta",
                "mean_delta",
                "mean_delta_adj",
                "retained_fraction",
                "n",
            )
        )
    rows = deltas.copy()
    if "quarantined" in rows.columns:
        rows = rows.loc[~rows["quarantined"].fillna(False).astype(bool)]
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    if rows.empty or "perplexity_ratio" not in rows.columns:
        return _empty(
            (
                "teacher",
                "student",
                "lr",
                "beta",
                "mean_delta",
                "mean_delta_adj",
                "retained_fraction",
                "n",
            )
        )
    out_rows: list[dict] = []
    for (teacher, student, lr), group in rows.groupby(
        ["teacher", "student", "lr"], dropna=False, sort=True
    ):
        y = pd.to_numeric(group["delta_primary"], errors="coerce")
        x = pd.to_numeric(group["perplexity_ratio"], errors="coerce")
        mask = y.notna() & x.notna()
        yv = y[mask].to_numpy(dtype=float)
        xv = x[mask].to_numpy(dtype=float)
        if yv.size == 0:
            continue
        beta = _ols_slope(xv, yv)
        adj = yv - beta * (xv - 1.0)
        mean_d = float(yv.mean())
        mean_adj = float(adj.mean())
        retained = abs(mean_adj) / abs(mean_d) if mean_d != 0.0 else float("nan")
        out_rows.append(
            {
                "teacher": teacher,
                "student": student,
                "lr": lr,
                "beta": beta,
                "mean_delta": mean_d,
                "mean_delta_adj": mean_adj,
                "retained_fraction": retained,
                "n": int(yv.size),
            }
        )
    if not out_rows:
        return _empty(
            (
                "teacher",
                "student",
                "lr",
                "beta",
                "mean_delta",
                "mean_delta_adj",
                "retained_fraction",
                "n",
            )
        )
    return pd.DataFrame(out_rows)


def base_priors(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Tidy untuned-base ``p_target`` / FC share / PPL per size.

    Accepts dicts from ``base_control.json`` / ``base_metrics.json``.
    Recognised keys: ``size`` / ``base``, ``p_target``, ``fc_share`` /
    ``forced_choice_share``, ``perplexity`` / ``ppl`` /
    ``perplexity_ratio``.

    Returns:
        Columns ``size``, ``p_target``, ``fc_share``, ``perplexity``.
    """
    out_rows: list[dict] = []
    for row in rows:
        size = row.get("size", row.get("base"))
        if size is None:
            continue
        fc = row.get("fc_share", row.get("forced_choice_share"))
        ppl = row.get(
            "perplexity",
            row.get("ppl", row.get("perplexity_ratio")),
        )
        out_rows.append(
            {
                "size": str(size),
                "p_target": (
                    float(row["p_target"]) if "p_target" in row else float("nan")
                ),
                "fc_share": float(fc) if fc is not None else float("nan"),
                "perplexity": float(ppl) if ppl is not None else float("nan"),
            }
        )
    if not out_rows:
        return _empty(("size", "p_target", "fc_share", "perplexity"))
    return pd.DataFrame(
        out_rows, columns=("size", "p_target", "fc_share", "perplexity")
    )


def grid_concordance(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """160m anchor Δ_sb and mean one-step Δ_tb vs `GRID_REFERENCE`.

    Returns:
        One-row frame with campaign and reference ``anchor_delta``,
        ``di_ds_delta``, ``teacher_p_target``, plus the reference
        ``run_set_id`` / ``step``.
    """
    picks = picks or {"160m": LR_DEFAULT}
    lr_160 = float(picks.get("160m", LR_DEFAULT))
    anchor = _cell_at(cells, teacher="160m", student="160m", lr=lr_160)
    one_step_vals: list[float] = []
    for teacher, student in ONE_STEP_PAIRS:
        if student not in picks:
            continue
        row = _cell_at(
            cells, teacher=teacher, student=student, lr=float(picks[student])
        )
        if row is not None:
            one_step_vals.append(float(row["mean_delta_tb"]))
    return pd.DataFrame(
        [
            {
                "anchor_delta": (
                    float(anchor["mean_delta_sb"])
                    if anchor is not None
                    else float("nan")
                ),
                "di_ds_delta": (
                    float(np.mean(one_step_vals)) if one_step_vals else float("nan")
                ),
                "teacher_p_target": float("nan"),
                "ref_run_set_id": GRID_REFERENCE["run_set_id"],
                "ref_step": GRID_REFERENCE["step"],
                "ref_anchor_delta": GRID_REFERENCE["anchor_delta"],
                "ref_di_ds_delta": GRID_REFERENCE["di_ds_delta"],
                "ref_teacher_p_target": GRID_REFERENCE["teacher_p_target"],
            }
        ]
    )


def _teacher_void(
    teacher: str,
    decisions: Mapping[str, TeacherGateDecision] | None,
) -> bool:
    if not decisions or teacher not in decisions:
        return False
    return decisions[teacher].status == "no_admissible_teacher"


def claims(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    teacher_decisions: Mapping[str, TeacherGateDecision] | None = None,
    penalty: pd.DataFrame | None = None,
    asymmetry: pd.DataFrame | None = None,
    additive: dict | None = None,
    lr: pd.DataFrame | None = None,
    ppl: pd.DataFrame | None = None,
    deltas: pd.DataFrame | None = None,
) -> dict:
    """Adjudicate X0–X4 from the metric frames.

    Returns:
        ``{"X0": {...}, ..., "X4": {...}}`` with ``status`` in
        ``holds`` / ``fails`` / ``void`` / ``partial`` / ``descriptive``
        and the evidence numbers used.
    """
    x0_pairs: dict[str, dict] = {}
    statuses: list[str] = []
    for pair in all_pairs():
        if pair.arm != "cross" or pair.student not in picks:
            continue
        label = f"{pair.teacher}→{pair.student}"
        if _teacher_void(pair.teacher, teacher_decisions):
            x0_pairs[label] = {
                "status": "void",
                "reason": "no_admissible_teacher",
            }
            statuses.append("void")
            continue
        anchor = _cell_at(
            cells,
            teacher=pair.student,
            student=pair.student,
            lr=float(picks[pair.student]),
        )
        if anchor is None:
            # X0 is reported only where the matched anchor's CI excludes 0; with
            # no anchor cell that precondition cannot be established, so the pair
            # is void rather than read off the cross row alone.
            x0_pairs[label] = {"status": "void", "reason": "missing_anchor"}
            statuses.append("void")
            continue
        if _ci_covers_zero(float(anchor["ci_low"]), float(anchor["ci_high"])):
            x0_pairs[label] = {
                "status": "void",
                "reason": "anchor_ci_covers_0",
                "anchor_mean": float(anchor["mean_delta_sb"]),
                "anchor_ci": [float(anchor["ci_low"]), float(anchor["ci_high"])],
            }
            statuses.append("void")
            continue
        row = _cell_at(
            cells,
            teacher=pair.teacher,
            student=pair.student,
            lr=float(picks[pair.student]),
        )
        if row is None:
            x0_pairs[label] = {"status": "void", "reason": "missing_cell"}
            statuses.append("void")
            continue
        mean = float(row["mean_delta_primary"])
        lo, hi = float(row["ci_low"]), float(row["ci_high"])
        holds = _ci_excludes_zero(lo, hi) and mean >= X0_MIN_DELTA
        status = "holds" if holds else "fails"
        x0_pairs[label] = {
            "status": status,
            "mean": mean,
            "ci": [lo, hi],
            "headline": (pair.teacher, pair.student) in ONE_STEP_PAIRS,
        }
        statuses.append(status)
    if statuses and all(s == "void" for s in statuses):
        x0_status = "void"
    elif (
        statuses
        and all(s in {"holds", "void"} for s in statuses)
        and "holds" in statuses
    ):
        x0_status = "holds" if all(s == "holds" for s in statuses) else "partial"
    elif "holds" in statuses and "fails" in statuses:
        x0_status = "partial"
    elif statuses and all(s == "holds" for s in statuses):
        x0_status = "holds"
    else:
        x0_status = "fails" if statuses else "void"

    x1_pairs: dict[str, dict] = {}
    x1_hits = 0
    x1_n = 0
    if penalty is not None and not penalty.empty:
        for _, row in penalty.iterrows():
            label = f"{row['teacher']}→{row['student']}"
            lo, hi = float(row["penalty_ci_low"]), float(row["penalty_ci_high"])
            sig = _ci_excludes_zero(lo, hi)
            # `penalty` is Δ_anchor − Δ_cross, so only a significantly POSITIVE
            # interval is the claimed cost of shape mismatch; a significantly
            # negative one is a cross-size advantage and must not read as X1.
            costly = sig and lo > 0.0
            x1_pairs[label] = {
                "penalty": float(row["penalty"]),
                "ci": [lo, hi],
                "rho_s": float(row["rho_s"]) if _finite(row.get("rho_s")) else None,
                "rho_t": float(row["rho_t"]) if _finite(row.get("rho_t")) else None,
                "penalty_sig": sig,
                "penalty_holds": costly,
            }
            if _finite(lo) and _finite(hi):
                x1_n += 1
                x1_hits += int(costly)
    if x1_n == 0:
        x1_status = "void"
    elif x1_hits == x1_n:
        x1_status = "holds"
    elif x1_hits == 0:
        x1_status = "fails"
    else:
        x1_status = "partial"

    add = additive or {}
    r2 = add.get("r2")
    pooled = add.get("pooled_iota")
    plo = add.get("pooled_iota_ci_low")
    phi = add.get("pooled_iota_ci_high")
    separable = (
        _finite(r2)
        and float(r2) >= X2_ADDITIVE_R2_MIN
        and plo is not None
        and phi is not None
        and _ci_covers_zero(float(plo), float(phi))
    )
    directional = (
        plo is not None
        and phi is not None
        and _ci_excludes_zero(float(plo), float(phi))
        and pooled is not None
        and abs(float(pooled)) >= X2_ASYM_MIN
    )
    if separable and not directional:
        x2_status = "holds"
        x2_read = "separable"
    elif directional and not separable:
        x2_status = "holds"
        x2_read = "directional"
    elif separable and directional:
        x2_status = "partial"
        x2_read = "both"
    elif r2 is None:
        x2_status = "void"
        x2_read = None
    else:
        x2_status = "fails"
        x2_read = None

    disagreements: list[dict] = []
    signs_stable = True
    tau_vals: list[float] = []
    if lr is not None and not lr.empty:
        for student in SIZES:
            picked = float(picks.get(student, LR_DEFAULT))
            sub = lr[lr["student"].astype(str) == student]
            if sub.empty:
                continue
            by_pair: dict[tuple[str, str], dict[float, int]] = {}
            for _, row in sub.iterrows():
                pair = (str(row["teacher"]), str(row["student"]))
                by_pair.setdefault(pair, {})[float(row["lr"])] = int(row["sign"])
                if _finite(row.get("kendall_tau")):
                    tau_vals.append(float(row["kendall_tau"]))
            for pair, signs in by_pair.items():
                star = signs.get(picked)
                default = signs.get(LR_DEFAULT)
                if star is not None and default is not None and star != default:
                    disagreements.append(
                        {
                            "teacher": pair[0],
                            "student": pair[1],
                            "lr_star": picked,
                            "sign_star": star,
                            "sign_default": default,
                        }
                    )
                uniq = {s for s in signs.values()}
                if len(uniq) > 1:
                    signs_stable = False

    retained_vals: list[float] = []
    if ppl is not None and not ppl.empty:
        retained_vals = [
            float(v) for v in ppl["retained_fraction"].tolist() if _finite(v)
        ]
    retained_mean = float(np.mean(retained_vals)) if retained_vals else float("nan")
    if not retained_vals:
        x4_status = "void"
    elif retained_mean >= X4_RETAINED_FRACTION_MIN:
        x4_status = "holds"
    else:
        x4_status = "fails"

    return {
        "X0": {
            "status": x0_status,
            "pairs": x0_pairs,
            "min_delta": X0_MIN_DELTA,
            "prompt_set": PRIMARY_PROMPT_SET,
        },
        "X1": {
            "status": x1_status,
            "pairs": x1_pairs,
            "n_sig": x1_hits,
            "n": x1_n,
        },
        "X2": {
            "status": x2_status,
            "read": x2_read,
            "r2": r2,
            "pooled_iota": pooled,
            "pooled_iota_ci": [plo, phi],
            "additive_r2_min": X2_ADDITIVE_R2_MIN,
            "asym_min": X2_ASYM_MIN,
            "loo_r2": add.get("loo_r2"),
        },
        "X3": {
            "status": "descriptive",
            "signs_stable": signs_stable,
            "kendall_tau_mean": (
                float(np.mean(tau_vals)) if tau_vals else float("nan")
            ),
            "lr_star_vs_default": disagreements,
        },
        "X4": {
            "status": x4_status,
            "retained_fraction": retained_mean,
            "threshold": X4_RETAINED_FRACTION_MIN,
        },
    }
