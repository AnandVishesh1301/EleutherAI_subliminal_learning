"""Pure frames for checkpoint-grid claims G1–G9.

No disk, no torch, no campaign-plan imports -- every rung list, threshold,
and label is an explicit argument. Wave-2 persists the two input tables
this module consumes; this module only decides what the numbers mean.

Teacher-frame columns
    ``base`` (str: original / seed1 / seed2 / seed3 / data-seed1),
    ``step`` (int rung), ``ftseed``, ``set``, ``scoring``, ``p_target``
    (float), ``is_control`` (bool; control = untrained base checkpoint
    scored at the same rung).

Student-frame columns
    ``arm`` (``grid`` / ``joint``), ``student_base``, ``teacher_base``,
    ``student_step``, ``teacher_step``, ``cell`` (``SI_SS`` / ``SI_DS`` /
    ``DI_DS``), ``ftseed``, ``gseed``, ``set``, ``scoring``, ``p_target``,
    ``is_control`` (bool; controls have teacher_* null/NaN and carry only
    student_base / student_step / gseed).

Every builder narrows to ``prompt_set`` / ``scoring`` the same way
``campaign.ckpt_selection._primary_rows`` does: equality on ``set`` and
``scoring`` when those columns exist. Defaults are ``test`` / ``prob``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.analysis import stats as grid_stats

PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"

_JOINT_SEED = re.compile(r"^seed[1-9]$")

TEACHER_CURVE_COLUMNS: tuple[str, ...] = (
    "base",
    "step",
    "teacher_mean",
    "teacher_sd",
    "control_mean",
    "n_teacher",
    "n_control",
    "argmax_step",
    "argmax_interior",
    "span",
)

GRID_DELTA_COLUMNS: tuple[str, ...] = (
    "level",
    "arm",
    "student_base",
    "teacher_base",
    "student_step",
    "teacher_step",
    "cell",
    "ftseed",
    "gseed",
    "set",
    "scoring",
    "p_target",
    "control_mean",
    "delta",
    "n_leaves",
)

MARGINAL_COLUMNS: tuple[str, ...] = (
    "slice",
    "arm",
    "student_base",
    "teacher_base",
    "student_step",
    "teacher_step",
    "cell",
    "delta",
    "n_leaves",
)

EFFICIENCY_COLUMNS: tuple[str, ...] = (
    "arm",
    "student_base",
    "teacher_base",
    "student_step",
    "teacher_step",
    "cell",
    "delta",
    "teacher_p_target",
    "efficiency",
    "degenerate",
)

ASYMMETRY_COLUMNS: tuple[str, ...] = (
    "arm",
    "student_base",
    "teacher_base",
    "cell",
    "rung_lo",
    "rung_hi",
    "delta_mature_to_young",
    "delta_young_to_mature",
    "asymmetry",
    "delta_diag_lo",
    "delta_diag_hi",
    "retention_mature_to_young",
    "retention_young_to_mature",
    "degenerate",
)

BRACKETING_COLUMNS: tuple[str, ...] = (
    "step",
    "contrast",
    "primary",
    "estimate",
    "ci_low",
    "ci_high",
    "n_left",
    "n_right",
)

ONSET_COLUMNS: tuple[str, ...] = (
    "arm",
    "student_base",
    "teacher_base",
    "teacher_step",
    "cell",
    "onset_step",
    "censored",
    "consecutive",
    "sd_multiple",
    "marker_step_0",
    "marker_step_1",
    "n_rungs_scored",
)

PENALTY_COLUMNS: tuple[str, ...] = (
    "step",
    "mean_cross",
    "mean_anchor",
    "penalty",
    "degenerate",
    "spearman",
    "spearman_ci_low",
    "spearman_ci_high",
    "p_boot",
    "n_rungs",
)

CONTRAST_PRIMARY = "si_ds_minus_di_ds"
CONTRAST_SEED1_SEED2 = "seed1_seed2"
CONTRAST_DATA_SEED1_SEED1 = "data_seed1_seed1"
ROLE_PRIMARY_SI_DS = "primary_si_ds"
ROLE_PRIMARY_DI_DS = "primary_di_ds"
ROLE_SEED1_SEED2 = "robust_seed1_seed2"
ROLE_DATA_SEED1_SEED1 = "robust_data_seed1_seed1"

SLICE_TEACHER = "teacher_marginal"
SLICE_STUDENT = "student_marginal"


def _primary_rows(frame: pd.DataFrame, prompt_set: str, scoring: str) -> pd.DataFrame:
    """Rows on the primary endpoint, when the frame records set/scoring at all."""
    out = frame
    if "set" in out.columns:
        out = out[out["set"] == prompt_set]
    if "scoring" in out.columns:
        out = out[out["scoring"] == scoring]
    return out


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _label(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).split("@", 1)[0].strip()


def _is_joint_seed(label: str) -> bool:
    return bool(_JOINT_SEED.fullmatch(label))


def _pair(student: object, teacher: object) -> frozenset[str]:
    return frozenset({_label(student), _label(teacher)})


def _finite(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    vals = _finite(values)
    if not vals:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return mean, float("nan")
    return mean, float(np.std(vals, ddof=1))


def _bracketing_role(student: object, teacher: object) -> str | None:
    pair = _pair(student, teacher)
    if pair == frozenset({"original", "data-seed1"}):
        return ROLE_PRIMARY_SI_DS
    if "original" in pair and any(_is_joint_seed(x) for x in pair):
        return ROLE_PRIMARY_DI_DS
    if pair == frozenset({"seed1", "seed2"}):
        return ROLE_SEED1_SEED2
    if pair == frozenset({"data-seed1", "seed1"}):
        return ROLE_DATA_SEED1_SEED1
    return None


def _spearman_log_step(steps: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(steps) & np.isfinite(values)
    if int(finite.sum()) < 2:
        return float("nan")
    logs = np.log(np.clip(steps[finite].astype(float), 1.0, None))
    y = values[finite]
    if np.allclose(logs, logs[0]) or np.allclose(y, y[0]):
        return float("nan")
    rho = scipy_stats.spearmanr(logs, y).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _bootstrap_mean_diff(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> grid_stats.BootstrapEstimate:
    """Percentile CI on ``mean(left) - mean(right)`` via row-index bootstrap."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0 or right.size == 0:
        raise ValueError("both sides of a contrast must be non-empty")
    values = np.concatenate([left, right])
    side = np.array(["L"] * left.size + ["R"] * right.size)
    pairing = np.arange(values.size)

    def statistic(rows: np.ndarray) -> float:
        picked = values[rows]
        labels = side[rows]
        a = picked[labels == "L"]
        b = picked[labels == "R"]
        if a.size == 0 or b.size == 0:
            return float("nan")
        return float(a.mean() - b.mean())

    return grid_stats.cluster_bootstrap(
        statistic,
        np.zeros(values.size, dtype=int),
        pairing,
        n_boot=n_boot,
        seed=seed,
    )


def build_teacher_curve(
    teacher_df: pd.DataFrame,
    *,
    rungs: Sequence[int],
    base: str = "original",
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """G1: per-rung teacher p_target mean/SD plus matched control mean.

    Args:
        teacher_df: Teacher preference frame (schema in the module docstring).
        rungs: Ordered rungs the curve is scored on.
        base: Teacher base label to keep.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per rung in `rungs` that has teacher leaves, with
        ``argmax_interior`` / ``span`` repeated from the curve as a whole.
    """
    rows = _primary_rows(teacher_df, prompt_set, scoring)
    if rows.empty or "p_target" not in rows.columns:
        return _empty(TEACHER_CURVE_COLUMNS)
    rows = rows[rows["base"].astype(str) == base].copy()
    if rows.empty:
        return _empty(TEACHER_CURVE_COLUMNS)
    wanted = [int(r) for r in rungs]
    rows["step"] = rows["step"].astype(int)
    rows = rows[rows["step"].isin(wanted)]
    is_control = (
        rows["is_control"].astype(bool)
        if "is_control" in rows.columns
        else pd.Series(False, index=rows.index)
    )
    teachers = rows.loc[~is_control]
    controls = rows.loc[is_control]
    per_rung: list[dict] = []
    for step in wanted:
        t_at = teachers[teachers["step"] == step]
        c_at = controls[controls["step"] == step]
        if "ftseed" in t_at.columns and not t_at.empty:
            t_vals = t_at.groupby("ftseed", dropna=False)["p_target"].mean()
        else:
            t_vals = t_at["p_target"] if not t_at.empty else pd.Series(dtype=float)
        t_mean, t_sd = _mean_sd(t_vals.tolist())
        c_mean, _ = _mean_sd(c_at["p_target"].tolist() if not c_at.empty else [])
        if t_at.empty:
            continue
        per_rung.append(
            {
                "base": base,
                "step": step,
                "teacher_mean": t_mean,
                "teacher_sd": t_sd,
                "control_mean": c_mean,
                "n_teacher": int(len(t_vals)),
                "n_control": int(len(c_at)),
            }
        )
    frame = pd.DataFrame(per_rung, columns=list(TEACHER_CURVE_COLUMNS))
    if frame.empty:
        return _empty(TEACHER_CURVE_COLUMNS)
    means = frame["teacher_mean"].to_numpy(dtype=float)
    finite = np.isfinite(means)
    if not finite.any():
        argmax_step = None
        span = float("nan")
        interior = False
    else:
        rel = int(np.nanargmax(np.where(finite, means, -np.inf)))
        argmax_step = int(frame["step"].iloc[rel])
        span = float(np.nanmax(means) - np.nanmin(means))
        interior = bool(wanted and argmax_step not in (wanted[0], wanted[-1]))
    frame["argmax_step"] = argmax_step
    frame["argmax_interior"] = interior
    frame["span"] = span
    return frame.reset_index(drop=True)


def build_grid_delta(
    student_df: pd.DataFrame,
    *,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """Leaf-level and aggregated ``delta = p_target − matched control mean``.

    Controls match on ``(student_base, student_step)``. A treatment cell
    with no matched control raises rather than emitting a silent NaN.

    Args:
        student_df: Student preference frame (schema in the module docstring).
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        ``level="leaf"`` rows plus ``level="agg"`` means over
        ``(arm, student_base, teacher_base, student_step, teacher_step, cell)``.

    Raises:
        ValueError: A treatment cell has no matched control, or a required
            column is missing.
    """
    rows = _primary_rows(student_df, prompt_set, scoring)
    if rows.empty:
        return _empty(GRID_DELTA_COLUMNS)
    required = ("student_base", "student_step", "p_target", "is_control")
    missing = [c for c in required if c not in rows.columns]
    if missing:
        raise ValueError(f"student frame missing columns {missing}")
    rows = rows.copy()
    is_control = rows["is_control"].astype(bool)
    treatments = rows.loc[~is_control].copy()
    controls = rows.loc[is_control]
    if treatments.empty:
        return _empty(GRID_DELTA_COLUMNS)
    control_mean = (
        controls.groupby(["student_base", "student_step"], dropna=False)["p_target"]
        .mean()
        .rename("control_mean")
        .reset_index()
    )
    merged = treatments.merge(
        control_mean, on=["student_base", "student_step"], how="left"
    )
    orphans = merged[merged["control_mean"].isna()]
    if not orphans.empty:
        sample = orphans.iloc[0]
        raise ValueError(
            "no matched control for student_base="
            f"{sample['student_base']!r} student_step={sample['student_step']}"
        )
    merged["delta"] = merged["p_target"].astype(float) - merged["control_mean"].astype(
        float
    )
    set_val = prompt_set if "set" not in merged.columns else None
    scoring_val = scoring if "scoring" not in merged.columns else None
    keys = [
        "arm",
        "student_base",
        "teacher_base",
        "student_step",
        "teacher_step",
        "cell",
    ]
    for key in keys:
        if key not in merged.columns:
            merged[key] = None
    leaf = merged.copy()
    leaf["level"] = "leaf"
    leaf["n_leaves"] = 1
    if "ftseed" not in leaf.columns:
        leaf["ftseed"] = None
    if "gseed" not in leaf.columns:
        leaf["gseed"] = None
    if "set" not in leaf.columns:
        leaf["set"] = set_val
    if "scoring" not in leaf.columns:
        leaf["scoring"] = scoring_val
    agg = (
        merged.groupby(keys, dropna=False)
        .agg(
            p_target=("p_target", "mean"),
            control_mean=("control_mean", "mean"),
            delta=("delta", "mean"),
            n_leaves=("delta", "size"),
        )
        .reset_index()
    )
    agg["level"] = "agg"
    agg["ftseed"] = None
    agg["gseed"] = None
    agg["set"] = leaf["set"].iloc[0] if "set" in leaf.columns else set_val
    agg["scoring"] = (
        leaf["scoring"].iloc[0] if "scoring" in leaf.columns else scoring_val
    )
    out = pd.concat(
        [leaf[list(GRID_DELTA_COLUMNS)], agg[list(GRID_DELTA_COLUMNS)]],
        ignore_index=True,
    )
    return out


def build_marginals(
    delta_df: pd.DataFrame,
    *,
    teacher_rungs: Sequence[int],
    student_rungs: Sequence[int],
    fixed_student_step: int,
    fixed_teacher_step: int,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """G2/G3 slices: teacher-marginal and student-marginal.

    Args:
        delta_df: `build_grid_delta` output (or any frame with ``delta``).
        teacher_rungs: Teacher steps kept on the teacher-marginal slice.
        student_rungs: Student steps kept on the student-marginal slice.
        fixed_student_step: Student rung held fixed for the teacher slice.
        fixed_teacher_step: Teacher rung held fixed for the student slice.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        Rows tagged ``teacher_marginal`` or ``student_marginal``.
    """
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty or "delta" not in rows.columns:
        return _empty(MARGINAL_COLUMNS)
    if "level" in rows.columns:
        rows = rows[rows["level"].astype(str) == "agg"]
    if rows.empty:
        return _empty(MARGINAL_COLUMNS)
    group_keys = [
        "arm",
        "student_base",
        "teacher_base",
        "student_step",
        "teacher_step",
        "cell",
    ]
    for key in group_keys:
        if key not in rows.columns:
            rows = rows.copy()
            rows[key] = None
    collapsed = (
        rows.groupby(group_keys, dropna=False)
        .agg(delta=("delta", "mean"), n_leaves=("delta", "size"))
        .reset_index()
    )
    t_keep = {int(r) for r in teacher_rungs}
    s_keep = {int(r) for r in student_rungs}
    teacher = collapsed[
        (collapsed["student_step"].astype(int) == int(fixed_student_step))
        & (collapsed["teacher_step"].astype(int).isin(t_keep))
    ].copy()
    teacher["slice"] = SLICE_TEACHER
    student = collapsed[
        (collapsed["teacher_step"].astype(int) == int(fixed_teacher_step))
        & (collapsed["student_step"].astype(int).isin(s_keep))
    ].copy()
    student["slice"] = SLICE_STUDENT
    out = pd.concat([teacher, student], ignore_index=True)
    if out.empty:
        return _empty(MARGINAL_COLUMNS)
    return out[list(MARGINAL_COLUMNS)].reset_index(drop=True)


def build_efficiency(
    delta_df: pd.DataFrame,
    teacher_curve_df: pd.DataFrame,
    *,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """G3/G4 transfer efficiency ``delta / teacher_p_target``.

    ``teacher_p_target <= 0`` or missing → ``efficiency`` is NaN and
    ``degenerate`` is True (never ``inf``).

    Args:
        delta_df: `build_grid_delta` output.
        teacher_curve_df: `build_teacher_curve` output (``step``,
            ``teacher_mean``; ``base`` joined when both sides have it).
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per aggregated grid cell with efficiency and the guard flag.
    """
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty or "delta" not in rows.columns:
        return _empty(EFFICIENCY_COLUMNS)
    if "level" in rows.columns:
        rows = rows[rows["level"].astype(str) == "agg"]
    if rows.empty:
        return _empty(EFFICIENCY_COLUMNS)
    curve = teacher_curve_df
    if curve.empty or "teacher_mean" not in curve.columns:
        out = rows.copy()
        out["teacher_p_target"] = float("nan")
        out["efficiency"] = float("nan")
        out["degenerate"] = True
        for col in EFFICIENCY_COLUMNS:
            if col not in out.columns:
                out[col] = None
        return out[list(EFFICIENCY_COLUMNS)].reset_index(drop=True)
    dose = curve.rename(
        columns={"step": "teacher_step", "teacher_mean": "teacher_p_target"}
    )
    on = ["teacher_step"]
    if "base" in dose.columns and "teacher_base" in rows.columns:
        dose = dose.rename(columns={"base": "teacher_base"})
        on.append("teacher_base")
    keep = [c for c in (*on, "teacher_p_target") if c in dose.columns]
    merged = rows.merge(dose[keep], on=on, how="left")
    p = pd.to_numeric(merged["teacher_p_target"], errors="coerce")
    delta = pd.to_numeric(merged["delta"], errors="coerce")
    bad = p.isna() | (p <= 0.0)
    merged["efficiency"] = np.where(bad, np.nan, delta / p)
    merged["degenerate"] = bad
    for col in EFFICIENCY_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    return merged[list(EFFICIENCY_COLUMNS)].reset_index(drop=True)


def build_asymmetry(
    delta_df: pd.DataFrame,
    *,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """G5: mature→young minus young→mature, plus diagonal retention.

    For each unordered rung pair ``{a < b}``:

    - ``asymmetry = delta(teacher@b → student@a) − delta(teacher@a → student@b)``
    - retention = off-diagonal delta / diagonal delta at the student rung.

    Args:
        delta_df: `build_grid_delta` output.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per ``(arm, bases, cell, {a, b})``.
    """
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty or "delta" not in rows.columns:
        return _empty(ASYMMETRY_COLUMNS)
    if "level" in rows.columns:
        rows = rows[rows["level"].astype(str) == "agg"]
    if rows.empty:
        return _empty(ASYMMETRY_COLUMNS)
    group_keys = ["arm", "student_base", "teacher_base", "cell"]
    for key in group_keys:
        if key not in rows.columns:
            rows = rows.copy()
            rows[key] = None
    out_rows: list[dict] = []
    for keys, group in rows.groupby(group_keys, dropna=False):
        lookup = {
            (int(r["student_step"]), int(r["teacher_step"])): float(r["delta"])
            for _, r in group.iterrows()
            if pd.notna(r.get("student_step")) and pd.notna(r.get("teacher_step"))
        }
        student_steps = sorted({s for s, _ in lookup})
        teacher_steps = sorted({t for _, t in lookup})
        rungs = sorted(set(student_steps) | set(teacher_steps))
        arm, student_base, teacher_base, cell = keys
        for i, lo in enumerate(rungs):
            for hi in rungs[i + 1 :]:
                mature_young = lookup.get((lo, hi))
                young_mature = lookup.get((hi, lo))
                if mature_young is None or young_mature is None:
                    continue
                diag_lo = lookup.get((lo, lo))
                diag_hi = lookup.get((hi, hi))
                ret_my = (
                    mature_young / diag_lo
                    if diag_lo is not None and diag_lo > 0.0
                    else float("nan")
                )
                ret_ym = (
                    young_mature / diag_hi
                    if diag_hi is not None and diag_hi > 0.0
                    else float("nan")
                )
                degenerate = (
                    diag_lo is None
                    or diag_hi is None
                    or diag_lo <= 0.0
                    or diag_hi <= 0.0
                )
                out_rows.append(
                    {
                        "arm": arm,
                        "student_base": student_base,
                        "teacher_base": teacher_base,
                        "cell": cell,
                        "rung_lo": lo,
                        "rung_hi": hi,
                        "delta_mature_to_young": mature_young,
                        "delta_young_to_mature": young_mature,
                        "asymmetry": mature_young - young_mature,
                        "delta_diag_lo": (float("nan") if diag_lo is None else diag_lo),
                        "delta_diag_hi": (float("nan") if diag_hi is None else diag_hi),
                        "retention_mature_to_young": ret_my,
                        "retention_young_to_mature": ret_ym,
                        "degenerate": bool(degenerate),
                    }
                )
    if not out_rows:
        return _empty(ASYMMETRY_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(ASYMMETRY_COLUMNS))


def build_bracketing(
    delta_df: pd.DataFrame,
    *,
    joint_rungs: Sequence[int],
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
    bootstrap_n: int = grid_stats.N_BOOT,
    seed: int = grid_stats.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """G7: SI_DS (original↔data-seed1) minus DI_DS (original↔seedN).

    ``seed1↔seed2`` and ``data-seed1↔seed1`` are returned labelled and
    excluded from the primary gap.

    Args:
        delta_df: `build_grid_delta` output (leaf rows preferred for the CI).
        joint_rungs: Joint-arm rungs to score.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.
        bootstrap_n: Multi-bootstrap replicates on the primary gap.
        seed: Bootstrap seed.

    Returns:
        Per-rung primary contrast plus labelled robustness rows.
    """
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty or "delta" not in rows.columns:
        return _empty(BRACKETING_COLUMNS)
    if "level" in rows.columns:
        leaves = rows[rows["level"].astype(str) == "leaf"]
        rows = leaves if not leaves.empty else rows
    wanted = {int(r) for r in joint_rungs}
    if "teacher_step" in rows.columns:
        step_col = "teacher_step"
    elif "student_step" in rows.columns:
        step_col = "student_step"
    elif "step" in rows.columns:
        step_col = "step"
    else:
        return _empty(BRACKETING_COLUMNS)
    rows = rows.copy()
    rows["_step"] = rows[step_col].astype(int)
    rows = rows[rows["_step"].isin(wanted)]
    if "student_base" not in rows.columns or "teacher_base" not in rows.columns:
        return _empty(BRACKETING_COLUMNS)
    rows["_role"] = [
        _bracketing_role(s, t)
        for s, t in zip(rows["student_base"], rows["teacher_base"], strict=True)
    ]
    out_rows: list[dict] = []
    for step, group in rows.groupby("_step", dropna=False):
        si = group[group["_role"] == ROLE_PRIMARY_SI_DS]["delta"].astype(float)
        di = group[group["_role"] == ROLE_PRIMARY_DI_DS]["delta"].astype(float)
        if not si.empty and not di.empty:
            point = float(si.mean() - di.mean())
            try:
                boot = _bootstrap_mean_diff(
                    si.to_numpy(),
                    di.to_numpy(),
                    n_boot=bootstrap_n,
                    seed=seed,
                )
                ci_low, ci_high = boot.ci_low, boot.ci_high
            except ValueError:
                ci_low, ci_high = float("nan"), float("nan")
            out_rows.append(
                {
                    "step": int(step),
                    "contrast": CONTRAST_PRIMARY,
                    "primary": True,
                    "estimate": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_left": int(len(si)),
                    "n_right": int(len(di)),
                }
            )
        for role, contrast in (
            (ROLE_SEED1_SEED2, CONTRAST_SEED1_SEED2),
            (ROLE_DATA_SEED1_SEED1, CONTRAST_DATA_SEED1_SEED1),
        ):
            vals = group[group["_role"] == role]["delta"].astype(float)
            if vals.empty:
                continue
            out_rows.append(
                {
                    "step": int(step),
                    "contrast": contrast,
                    "primary": False,
                    "estimate": float(vals.mean()),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n_left": int(len(vals)),
                    "n_right": 0,
                }
            )
    if not out_rows:
        return _empty(BRACKETING_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(BRACKETING_COLUMNS))


def build_onset(
    delta_df: pd.DataFrame,
    control_df_or_student_df: pd.DataFrame,
    *,
    student_rungs: Sequence[int],
    sd_multiple: float = 2.0,
    consecutive: int = 2,
    markers: Sequence[float] = (0.0022, 0.01),
    max_step: int = 143000,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """G6: first student rung that clears control mean + k·SD.

    A group that never clears `consecutive` successive rungs is returned
    with ``onset_step`` null and ``censored`` True.

    Args:
        delta_df: Treatment deltas (``delta`` or ``p_target``).
        control_df_or_student_df: Control rows, or a student frame whose
            ``is_control`` rows supply control ``p_target``.
        student_rungs: Ordered student rungs to walk.
        sd_multiple: Multiples of control SD that define a clear.
        consecutive: Successive clears required for onset.
        markers: Literature onset fractions of `max_step`.
        max_step: Pretraining budget used to report marker steps.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per ``(arm, student_base, teacher_base, teacher_step, cell)``.
    """
    treatments = _primary_rows(delta_df, prompt_set, scoring)
    controls = _primary_rows(control_df_or_student_df, prompt_set, scoring)
    if treatments.empty:
        return _empty(ONSET_COLUMNS)
    if "is_control" in treatments.columns:
        treatments = treatments.loc[~treatments["is_control"].astype(bool)]
    if "level" in treatments.columns:
        leaves = treatments[treatments["level"].astype(str) == "leaf"]
        if not leaves.empty:
            treatments = leaves
    if "is_control" in controls.columns:
        controls = controls.loc[controls["is_control"].astype(bool)]
    if treatments.empty:
        return _empty(ONSET_COLUMNS)
    ordered = [int(r) for r in student_rungs]
    marker_steps = [int(round(float(m) * int(max_step))) for m in markers]
    marker_0 = marker_steps[0] if marker_steps else None
    marker_1 = marker_steps[1] if len(marker_steps) > 1 else None

    control_stats: dict[tuple[object, int], tuple[float, float]] = {}
    if (
        not controls.empty
        and "p_target" in controls.columns
        and "student_base" in controls.columns
        and "student_step" in controls.columns
    ):
        for (base, step), group in controls.groupby(
            ["student_base", "student_step"], dropna=False
        ):
            control_stats[(base, int(step))] = _mean_sd(group["p_target"].tolist())

    group_keys = [
        c
        for c in ("arm", "student_base", "teacher_base", "teacher_step", "cell")
        if c in treatments.columns
    ]
    if not group_keys:
        treatments = treatments.copy()
        treatments["_all"] = 0
        group_keys = ["_all"]

    out_rows: list[dict] = []
    for keys, group in treatments.groupby(group_keys, dropna=False):
        key_map = dict(
            zip(group_keys, keys if isinstance(keys, tuple) else (keys,), strict=True)
        )
        clears: dict[int, bool] = {}
        student_base = key_map.get("student_base")
        for step in ordered:
            at = (
                group[group["student_step"].astype(int) == step]
                if "student_step" in group.columns
                else group.iloc[0:0]
            )
            if at.empty:
                continue
            c_mean, c_sd = control_stats.get(
                (student_base, step), (float("nan"), float("nan"))
            )
            # Deterministic control (single leaf / zero spread): SD is
            # legitimately 0, not unknown -- the ckpt_selection convention.
            # The clear is strict so an SD-0 rung is never vacuously cleared
            # by a score equal to the control mean.
            effective_sd = c_sd if np.isfinite(c_sd) and c_sd > 0.0 else 0.0
            k_sd = sd_multiple * effective_sd
            if "p_target" in at.columns and np.isfinite(c_mean):
                score = float(at["p_target"].astype(float).mean())
                clears[step] = bool(score > c_mean + k_sd)
            elif "delta" in at.columns:
                score = float(at["delta"].astype(float).mean())
                clears[step] = bool(score > k_sd)
        onset: int | None = None
        if consecutive < 1:
            raise ValueError(f"consecutive must be >= 1; got {consecutive}")
        for i in range(0, max(0, len(ordered) - consecutive + 1)):
            window = ordered[i : i + consecutive]
            if window and all(clears.get(r, False) for r in window):
                onset = window[0]
                break
        out_rows.append(
            {
                "arm": key_map.get("arm"),
                "student_base": key_map.get("student_base"),
                "teacher_base": key_map.get("teacher_base"),
                "teacher_step": key_map.get("teacher_step"),
                "cell": key_map.get("cell"),
                "onset_step": onset,
                "censored": onset is None,
                "consecutive": int(consecutive),
                "sd_multiple": float(sd_multiple),
                "marker_step_0": marker_0,
                "marker_step_1": marker_1,
                "n_rungs_scored": int(len(clears)),
            }
        )
    if not out_rows:
        return _empty(ONSET_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(ONSET_COLUMNS))


def build_penalty_trend(
    delta_df: pd.DataFrame,
    *,
    joint_rungs: Sequence[int],
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
    bootstrap_n: int = grid_stats.N_BOOT,
    seed: int = grid_stats.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """G8: ``1 − mean(original↔seedN) / mean(seedN←seedN)`` vs log(step).

    Anchor mean ``<= 0`` → ``penalty`` is NaN and ``degenerate`` is True.

    Args:
        delta_df: `build_grid_delta` output.
        joint_rungs: Rungs the penalty is scored on.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.
        bootstrap_n: Bootstrap replicates on Spearman(penalty, log step).
        seed: Bootstrap seed.

    Returns:
        One row per scored rung, with the Spearman interval repeated.
    """
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty or "delta" not in rows.columns:
        return _empty(PENALTY_COLUMNS)
    if "level" in rows.columns:
        agg = rows[rows["level"].astype(str) == "agg"]
        rows = agg if not agg.empty else rows
    if "student_base" not in rows.columns or "teacher_base" not in rows.columns:
        return _empty(PENALTY_COLUMNS)
    step_col = (
        "teacher_step"
        if "teacher_step" in rows.columns
        else ("student_step" if "student_step" in rows.columns else "step")
    )
    if step_col not in rows.columns:
        return _empty(PENALTY_COLUMNS)
    wanted = {int(r) for r in joint_rungs}
    rows = rows.copy()
    rows["_step"] = rows[step_col].astype(int)
    rows = rows[rows["_step"].isin(wanted)]
    per_rung: list[dict] = []
    for step, group in rows.groupby("_step", dropna=False):
        cross = group[
            group.apply(
                lambda r: (
                    "original" in _pair(r["student_base"], r["teacher_base"])
                    and any(
                        _is_joint_seed(x)
                        for x in _pair(r["student_base"], r["teacher_base"])
                    )
                ),
                axis=1,
            )
        ]["delta"].astype(float)
        anchor = group[
            group.apply(
                lambda r: (
                    _label(r["student_base"]) == _label(r["teacher_base"])
                    and _is_joint_seed(_label(r["student_base"]))
                ),
                axis=1,
            )
        ]["delta"].astype(float)
        if cross.empty or anchor.empty:
            continue
        mean_cross = float(cross.mean())
        mean_anchor = float(anchor.mean())
        degenerate = mean_anchor <= 0.0 or not np.isfinite(mean_anchor)
        penalty = float("nan") if degenerate else 1.0 - mean_cross / mean_anchor
        per_rung.append(
            {
                "step": int(step),
                "mean_cross": mean_cross,
                "mean_anchor": mean_anchor,
                "penalty": penalty,
                "degenerate": degenerate,
            }
        )
    frame = pd.DataFrame(per_rung)
    if frame.empty:
        return _empty(PENALTY_COLUMNS)
    usable = frame.loc[~frame["degenerate"] & np.isfinite(frame["penalty"])]
    if len(usable) < 2:
        rho = float("nan")
        ci_low = ci_high = float("nan")
        p_boot = float("nan")
    else:
        steps = usable["step"].to_numpy(dtype=float)
        vals = usable["penalty"].to_numpy(dtype=float)
        rho = _spearman_log_step(steps, vals)

        def statistic(idx: np.ndarray) -> float:
            return _spearman_log_step(steps[idx], vals[idx])

        try:
            boot = grid_stats.cluster_bootstrap(
                statistic,
                usable["step"].tolist(),
                [0] * len(usable),
                n_boot=bootstrap_n,
                seed=seed,
            )
            ci_low, ci_high, p_boot = boot.ci_low, boot.ci_high, boot.p_boot
        except ValueError:
            ci_low = ci_high = p_boot = float("nan")
    frame["spearman"] = rho
    frame["spearman_ci_low"] = ci_low
    frame["spearman_ci_high"] = ci_high
    frame["p_boot"] = p_boot
    frame["n_rungs"] = int(len(usable))
    return frame[list(PENALTY_COLUMNS)].reset_index(drop=True)
