"""Pure frames and adjudicators for checkpoint-grid-2 claims H1–H12.

No disk, no torch, no campaign-plan imports -- every rung list,
threshold, and label is an explicit argument. Wave-2 persists the
input tables this module consumes; this module only decides what
the numbers mean.

Student delta frame (from ``build_grid_delta`` plus extra keys)
    ``level`` (``leaf`` / ``agg``), ``arm``, ``animal``,
    ``student_base``, ``teacher_base``, ``student_step``,
    ``teacher_step``, ``cell`` (``SI_SS`` / ``SI_DS`` / ``DI_SS`` /
    ``DI_DS``), ``gkd_tag``, ``teacher_param_tag``,
    ``source_run_set``, ``ftseed``, ``gseed``, ``set``, ``scoring``,
    ``p_target``, ``control_mean``, ``delta``, ``n_leaves``, and
    optionally ``perplexity_ratio`` / ``filter_drop_frac``.

Teacher dose frame
    ``base``, ``step``, ``ftseed``, ``n_epochs``, ``param_tag``,
    ``set``, ``scoring``, ``p_target``, ``is_control``,
    ``source_run_set``.

Control frame
    the ``is_control`` rows of the student frame (``student_base``,
    ``student_step``, ``gkd_tag``, ``gseed``, ``p_target``).

Every builder narrows to ``prompt_set`` / ``scoring`` the same way
``ckptgrid_metrics._primary_rows`` does: equality on ``set`` and
``scoring`` when those columns exist. Defaults are ``test`` / ``prob``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.analysis import stats as grid_stats

PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"

DOSE_RUNGS: tuple[int, ...] = (2000, 8000, 48000, 143000)
DOSE_EPOCHS: tuple[int, ...] = (1, 2, 3, 8)
DOSE_BASES: tuple[str, ...] = ("original",)
JOINT_RUNGS: tuple[int, ...] = (2000, 8000, 16000, 48000, 96000, 143000)
HP_RUNGS: tuple[int, ...] = (512, 2000, 4000, 8000, 16000, 48000, 143000)
CHECKPOINT_MAX_STEP = 143000

RIDGE_ERA_EARLY_MAX_STEP = 16000
RIDGE_SUPPORT_MIN_COLUMNS = 4
RIDGE_NULL_MAX_COLUMNS = 1
DOSE_MATCH_TOL = 0.05
DOSE_TOST_BOUND = 0.05
ONSET_SD_MULTIPLE = 2.0
ONSET_ABS_FLOOR = 0.02
ONSET_CONSECUTIVE = 2
ONSET_MARKERS: tuple[float, ...] = (0.0022, 0.01)
CONCORDANCE_MIN_SPEARMAN = 0.6
CONCORDANCE_MIN_SIGN_AGREEMENT = 0.8
ASYMMETRY_ERA_EDGES: tuple[int, ...] = (4000, 32000)
QUARANTINE_PPL_RATIO_MAX = 5.0
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260908
CLAIM_IDS: tuple[str, ...] = (
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H8",
    "H9",
    "H10",
    "H11",
    "H12",
)

STATUS_SUPPORTED = "supported"
STATUS_NOT_SUPPORTED = "not_supported"
STATUS_DESCRIPTIVE = "descriptive"
STATUS_INSUFFICIENT = "insufficient_data"
ALLOWED_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_SUPPORTED,
        STATUS_NOT_SUPPORTED,
        STATUS_DESCRIPTIVE,
        STATUS_INSUFFICIENT,
    }
)

CELL_SI_SS = "SI_SS"
CELL_SI_DS = "SI_DS"
CELL_DI_SS = "DI_SS"
CELL_DI_DS = "DI_DS"

TEACHER_DOSE_COLUMNS: tuple[str, ...] = (
    "base",
    "step",
    "n_epochs",
    "p_target_mean",
    "p_target_sd",
    "n_ftseed",
    "control_mean",
    "register_sample_rate",
    "teacher_ppl_ratio",
)

DOSE_RESPONSE_COLUMNS: tuple[str, ...] = (
    "animal",
    "student_base",
    "teacher_base",
    "student_step",
    "teacher_step",
    "matched_rung",
    "cell",
    "n_epochs",
    "teacher_param_tag",
    "gkd_tag",
    "delta",
    "teacher_p_target",
    "overshoot",
    "efficiency",
    "degenerate",
)

DOSE_MATCHED_COLUMNS: tuple[str, ...] = (
    "rung_coef",
    "ci_low",
    "ci_high",
    "tost_equivalent",
    "n_bands",
    "n_cells",
)

RIDGE_GAIN_COLUMNS: tuple[str, ...] = (
    "animal",
    "student_base",
    "teacher_base",
    "teacher_step",
    "delta_diag",
    "mean_off",
    "ridge_gain",
    "ci_low",
    "ci_high",
    "era",
    "n_off",
    "degenerate",
)

PENALTY_MAP_COLUMNS: tuple[str, ...] = (
    "animal",
    "student_base",
    "teacher_base",
    "student_step",
    "teacher_step",
    "cell",
    "delta_cell",
    "delta_reference",
    "penalty",
    "degenerate",
)

EMISSIVITY_COLUMNS: tuple[str, ...] = (
    "step",
    "base",
    "tau",
    "sigma",
    "tau_minus_sigma",
    "mu",
    "resid_rms",
    "n_cells",
)

ONSET_FLOOR_COLUMNS: tuple[str, ...] = (
    "animal",
    "student_base",
    "teacher_base",
    "teacher_step",
    "cell",
    "gkd_tag",
    "student_step",
    "onset_step",
    "censored",
    "threshold",
    "clears_floor",
    "delta",
    "ci_low",
    "ci_high",
    "n_rungs_scored",
    "consecutive",
    "sd_multiple",
    "abs_floor",
    "marker_step_0",
    "marker_step_1",
)

HP_COLLAPSE_COLUMNS: tuple[str, ...] = (
    "gkd_tag",
    "student_step",
    "mean_ppl_ratio",
    "collapsed",
    "n_leaves",
    "best_lr_at_rung",
)

CONCORDANCE_COLUMNS: tuple[str, ...] = (
    "subset",
    "primary_set",
    "primary_scoring",
    "twin_set",
    "twin_scoring",
    "spearman",
    "sign_agreement",
    "ba_mean",
    "ba_sd",
    "n_cells",
)

ASYMMETRY_ERA_COLUMNS: tuple[str, ...] = (
    "era",
    "era_lo",
    "era_hi",
    "asymmetry_mean",
    "ci_low",
    "ci_high",
    "n_pairs",
)

TRIANGLE_COLUMNS: tuple[str, ...] = (
    "step",
    "init",
    "stream",
    "interaction",
    "reason",
    "delta_si_ss",
    "delta_si_ds",
    "delta_di_ss",
    "delta_di_ds",
)

BASE_PPL_COLUMNS: tuple[str, ...] = (
    "base",
    "rung",
    "ppl_overall",
)

_EPOCH_TAG = re.compile(r"^ep(\d+)_")


def _primary_rows(frame: pd.DataFrame, prompt_set: str, scoring: str) -> pd.DataFrame:
    """Rows on the primary endpoint, when the frame records set/scoring."""
    out = frame
    if "set" in out.columns:
        out = out[out["set"] == prompt_set]
    if "scoring" in out.columns:
        out = out[out["scoring"] == scoring]
    return out


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


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


def _require_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required column {missing[0]!r}")


def _level_rows(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    if "level" not in frame.columns:
        return frame
    picked = frame[frame["level"].astype(str) == level]
    return picked if not picked.empty else frame


def _agg_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return _level_rows(frame, "agg")


def _leaf_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return _level_rows(frame, "leaf")


def _epochs_from_tag(tag: object) -> float:
    if tag is None or (isinstance(tag, float) and math.isnan(tag)):
        return float("nan")
    match = _EPOCH_TAG.match(str(tag))
    if match is None:
        return float("nan")
    return float(match.group(1))


def _guard_div(
    numerator: np.ndarray, denominator: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``numerator / denominator`` with NaN (never inf) when den <= 0."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    bad = ~np.isfinite(den) | (den <= 0.0) | ~np.isfinite(num)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan)
    good = ~bad
    if good.any():
        out[good] = num[good] / den[good]
    return out, bad


def _claim(
    claim_id: str,
    operationalization: str,
    statistic: object,
    threshold: object,
    status: str,
    reason: str | None,
) -> dict:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"unknown claim status {status!r}")
    if status == STATUS_SUPPORTED:
        reason = None
    return {
        "id": claim_id,
        "operationalization": operationalization,
        "statistic": statistic,
        "threshold": threshold,
        "status": status,
        "reason": reason,
    }


def _insufficient(claim_id: str, operationalization: str, reason: str) -> dict:
    return _claim(
        claim_id,
        operationalization,
        None,
        None,
        STATUS_INSUFFICIENT,
        reason,
    )


def _frame_empty(frame: pd.DataFrame | None) -> bool:
    return frame is None or frame.empty


# ---- builders ----------------------------------------------------------------


def build_teacher_dose(
    teacher_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = DOSE_RUNGS,
    bases: Sequence[str] = DOSE_BASES,
    epochs: Sequence[int] = DOSE_EPOCHS,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """Per ``(base, rung, n_epochs)`` teacher p_target mean/SD.

    Args:
        teacher_df: Teacher preference frame (schema in the module
            docstring).
        rungs: Teacher steps to score.
        bases: Teacher base labels to keep.
        epochs: Teacher epoch counts to score.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per designed cell that has teacher leaves.

    Raises:
        ValueError: A required column is absent.
    """
    _require_columns(teacher_df, ("base", "step", "p_target"))
    if "n_epochs" not in teacher_df.columns:
        raise ValueError("missing required column 'n_epochs'")
    rows = _primary_rows(teacher_df, prompt_set, scoring)
    if rows.empty:
        return _empty(TEACHER_DOSE_COLUMNS)
    wanted_rungs = {int(r) for r in rungs}
    wanted_bases = {str(b) for b in bases}
    wanted_epochs = {int(e) for e in epochs}
    rows = rows.copy()
    rows["step"] = rows["step"].astype(int)
    rows["n_epochs"] = pd.to_numeric(rows["n_epochs"], errors="coerce")
    is_control = (
        rows["is_control"].astype(bool)
        if "is_control" in rows.columns
        else pd.Series(False, index=rows.index)
    )
    teachers = rows.loc[~is_control]
    controls = rows.loc[is_control]
    register = teacher_df
    if "set" in teacher_df.columns:
        register = teacher_df[teacher_df["set"] == "register_sample"]

    out_rows: list[dict] = []
    for base in bases:
        if str(base) not in wanted_bases:
            continue
        for step in rungs:
            if int(step) not in wanted_rungs:
                continue
            c_at = controls[
                (controls["base"].astype(str) == str(base))
                & (controls["step"].astype(int) == int(step))
            ]
            c_mean, _ = _mean_sd(c_at["p_target"].tolist() if not c_at.empty else [])
            for n_epochs in epochs:
                if int(n_epochs) not in wanted_epochs:
                    continue
                t_at = teachers[
                    (teachers["base"].astype(str) == str(base))
                    & (teachers["step"].astype(int) == int(step))
                    & (teachers["n_epochs"] == int(n_epochs))
                ]
                if t_at.empty:
                    continue
                if "ftseed" in t_at.columns:
                    t_vals = t_at.groupby("ftseed", dropna=False)["p_target"].mean()
                else:
                    t_vals = t_at["p_target"]
                t_mean, t_sd = _mean_sd(t_vals.tolist())
                reg_rate = float("nan")
                if not register.empty and "p_target" in register.columns:
                    r_at = register[
                        (register["base"].astype(str) == str(base))
                        & (register["step"].astype(int) == int(step))
                    ]
                    if "n_epochs" in r_at.columns:
                        r_ep = pd.to_numeric(r_at["n_epochs"], errors="coerce")
                        r_at = r_at.loc[r_ep == int(n_epochs)]
                    if not r_at.empty:
                        reg_rate, _ = _mean_sd(r_at["p_target"].tolist())
                if "register_sample_rate" in t_at.columns:
                    col_rate, _ = _mean_sd(t_at["register_sample_rate"].tolist())
                    if np.isfinite(col_rate):
                        reg_rate = col_rate
                ppl = float("nan")
                for col in ("teacher_ppl_ratio", "perplexity_ratio"):
                    if col in t_at.columns:
                        ppl, _ = _mean_sd(t_at[col].tolist())
                        if np.isfinite(ppl):
                            break
                out_rows.append(
                    {
                        "base": str(base),
                        "step": int(step),
                        "n_epochs": int(n_epochs),
                        "p_target_mean": t_mean,
                        "p_target_sd": t_sd,
                        "n_ftseed": int(len(t_vals)),
                        "control_mean": c_mean,
                        "register_sample_rate": reg_rate,
                        "teacher_ppl_ratio": ppl,
                    }
                )
    if not out_rows:
        return _empty(TEACHER_DOSE_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(TEACHER_DOSE_COLUMNS))


def build_dose_response(
    delta_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = DOSE_RUNGS,
    bases: Sequence[str] = ("original",),
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """Dose cells joined to their teacher's dose, matched and mismatched.

    Keeps both the matched-rung diagonal (arm A) and the mismatched rows
    at a fixed student rung (arm A'), flagged by ``matched_rung``: H3
    scores the diagonal only, while H4's overshoot is stated over both.

    ``teacher_p_target <= 0`` or missing → ``efficiency`` is NaN and
    ``degenerate`` is True (never ``inf``).

    Args:
        delta_df: Student delta frame.
        dose_df: `build_teacher_dose` output.
        rungs: Teacher rungs to keep.
        bases: Student/teacher bases to keep; the dose arms are
            same-base, so this excludes arm G's cross-init cells.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per dose cell, with ``matched_rung`` marking the
        diagonal.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(DOSE_RESPONSE_COLUMNS)
    _require_columns(delta_df, ("student_step", "teacher_step", "delta"))
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty:
        return _empty(DOSE_RESPONSE_COLUMNS)
    rows = _agg_rows(rows)
    wanted = {int(r) for r in rungs}
    rows = rows.copy()
    rows["student_step"] = rows["student_step"].astype(int)
    rows["teacher_step"] = rows["teacher_step"].astype(int)
    rows = rows[rows["teacher_step"].isin(wanted)]
    keep_bases = {str(b) for b in bases}
    if keep_bases:
        for col in ("student_base", "teacher_base"):
            if col in rows.columns:
                rows = rows[rows[col].astype(str).isin(keep_bases)]
    rows = rows.copy()
    rows["matched_rung"] = rows["student_step"] == rows["teacher_step"]
    if rows.empty:
        return _empty(DOSE_RESPONSE_COLUMNS)
    if "n_epochs" not in rows.columns:
        tag = rows["teacher_param_tag"] if "teacher_param_tag" in rows.columns else None
        rows["n_epochs"] = (
            tag.map(_epochs_from_tag) if tag is not None else float("nan")
        )
    dose = dose_df
    if not dose.empty:
        if "step" not in dose.columns and "rung" in dose.columns:
            dose = dose.rename(columns={"rung": "step"})
        if "p_target_mean" in dose.columns:
            dose = dose.rename(columns={"p_target_mean": "teacher_p_target"})
        elif "teacher_mean" in dose.columns:
            dose = dose.rename(columns={"teacher_mean": "teacher_p_target"})
    on: list[str] = []
    if not dose.empty and "teacher_p_target" in dose.columns:
        dose = dose.rename(columns={"base": "teacher_base", "step": "teacher_step"})
        for key in ("teacher_base", "teacher_step", "n_epochs"):
            if key in dose.columns and key in rows.columns:
                on.append(key)
        if on:
            keep = [c for c in (*on, "teacher_p_target") if c in dose.columns]
            rows = rows.merge(dose[keep], on=on, how="left")
    if "teacher_p_target" not in rows.columns:
        rows["teacher_p_target"] = float("nan")
    p = pd.to_numeric(rows["teacher_p_target"], errors="coerce")
    delta = pd.to_numeric(rows["delta"], errors="coerce")
    eff, bad = _guard_div(delta.to_numpy(dtype=float), p.to_numpy(dtype=float))
    rows["efficiency"] = eff
    rows["degenerate"] = bad
    rows["overshoot"] = delta.to_numpy(dtype=float) - p.to_numpy(dtype=float)
    rows.loc[p.isna(), "overshoot"] = np.nan
    for col in DOSE_RESPONSE_COLUMNS:
        if col not in rows.columns:
            rows[col] = None
    return rows[list(DOSE_RESPONSE_COLUMNS)].reset_index(drop=True)


def _ols_rung_coef(delta: np.ndarray, log_rung: np.ndarray, band: np.ndarray) -> float:
    """Least-squares coefficient of ``log10(rung)`` after dose-band dummies."""
    finite = np.isfinite(delta) & np.isfinite(log_rung)
    if int(finite.sum()) < 2:
        return float("nan")
    y = delta[finite]
    x = log_rung[finite]
    b = np.asarray(band)[finite]
    if np.allclose(x, x[0]):
        return float("nan")
    levels, inv = np.unique(b, return_inverse=True)
    cols = [np.ones(y.size)]
    if levels.size >= 2:
        for i in range(1, levels.size):
            cols.append((inv == i).astype(float))
    cols.append(x)
    design = np.column_stack(cols)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    coef = float(beta[-1])
    if not np.isfinite(coef):
        return float("nan")
    return coef


def _per_band_slopes(
    delta: np.ndarray, log_rung: np.ndarray, band: np.ndarray
) -> np.ndarray:
    """OLS slope of ``delta ~ log10(rung)`` inside each dose band."""
    slopes: list[float] = []
    for level in np.unique(band):
        mask = (band == level) & np.isfinite(delta) & np.isfinite(log_rung)
        if int(mask.sum()) < 2:
            continue
        y = delta[mask]
        x = log_rung[mask]
        if np.allclose(x, x[0]):
            slopes.append(0.0 if np.allclose(y, y[0]) else float("nan"))
            continue
        design = np.column_stack([np.ones(y.size), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        slopes.append(float(beta[1]))
    return np.asarray(slopes, dtype=float)


def _tost_equivalent(values: np.ndarray, bound: float) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    if finite.size < 2 or float(np.std(finite, ddof=1)) <= 0.0:
        return bool(abs(float(finite.mean())) <= bound)
    result = grid_stats.tost_equivalence(finite, bound_low=-bound, bound_high=bound)
    return bool(result.equivalent)


def build_dose_matched(
    dose_response_df: pd.DataFrame,
    *,
    tol: float = DOSE_MATCH_TOL,
    bound: float = DOSE_TOST_BOUND,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """H3: ``delta ~ dose_band + log10(rung)`` and TOST the rung coef.

    Args:
        dose_response_df: `build_dose_response` output.
        tol: Dose-band width on ``teacher_p_target``.
        bound: Equivalence bound on the rung coefficient.
        n_boot: Cluster-bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        One-row frame with the rung coefficient, CI, and TOST flag.
    """
    if dose_response_df.empty:
        return _empty(DOSE_MATCHED_COLUMNS)
    _require_columns(dose_response_df, ("delta", "teacher_p_target"))
    rows = dose_response_df.copy()
    if "matched_rung" in rows.columns:
        matched = rows[rows["matched_rung"].astype(bool)]
        if not matched.empty:
            rows = matched.copy()
    if "student_step" in rows.columns:
        rung = pd.to_numeric(rows["student_step"], errors="coerce")
    elif "teacher_step" in rows.columns:
        rung = pd.to_numeric(rows["teacher_step"], errors="coerce")
    else:
        raise ValueError("missing required column 'student_step'")
    rows["_rung"] = rung
    rows["_dose"] = pd.to_numeric(rows["teacher_p_target"], errors="coerce")
    rows["_delta"] = pd.to_numeric(rows["delta"], errors="coerce")
    rows = rows[
        np.isfinite(rows["_rung"])
        & np.isfinite(rows["_dose"])
        & np.isfinite(rows["_delta"])
        & (rows["_rung"] > 0)
    ]
    if rows.empty:
        return _empty(DOSE_MATCHED_COLUMNS)
    rows["_band"] = np.floor(rows["_dose"].to_numpy(dtype=float) / tol) * tol
    rows["_log"] = np.log10(rows["_rung"].to_numpy(dtype=float))
    delta = rows["_delta"].to_numpy(dtype=float)
    log_rung = rows["_log"].to_numpy(dtype=float)
    band = rows["_band"].to_numpy(dtype=float)
    n_bands = int(np.unique(band).size)
    coef = _ols_rung_coef(delta, log_rung, band)
    ci_low = ci_high = float("nan")
    if np.isfinite(coef) and len(rows) >= 2:

        def statistic(idx: np.ndarray) -> float:
            return _ols_rung_coef(delta[idx], log_rung[idx], band[idx])

        try:
            boot = grid_stats.cluster_bootstrap(
                statistic,
                rows["_rung"].tolist(),
                rows["_band"].tolist(),
                n_boot=n_boot,
                seed=seed,
            )
            ci_low, ci_high = boot.ci_low, boot.ci_high
        except ValueError:
            ci_low = ci_high = float("nan")
    slopes = _per_band_slopes(delta, log_rung, band)
    tost_values = slopes if slopes.size else np.asarray([coef], dtype=float)
    return pd.DataFrame(
        [
            {
                "rung_coef": coef,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "tost_equivalent": _tost_equivalent(tost_values, bound),
                "n_bands": n_bands,
                "n_cells": int(len(rows)),
            }
        ],
        columns=list(DOSE_MATCHED_COLUMNS),
    )


def _ridge_gain_stat(diag: np.ndarray, off: np.ndarray) -> float:
    if diag.size == 0:
        return float("nan")
    if off.size == 0:
        return float(diag.mean())
    return float(diag.mean() - off.mean())


def build_ridge_gain(
    delta_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = JOINT_RUNGS,
    early_max_step: int = RIDGE_ERA_EARLY_MAX_STEP,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """H1/H2/H11: ``ridge_gain = delta_diag − mean_off`` per teacher column.

    A diagonal row that lives in a reference run set is kept: this
    builder does not filter on ``source_run_set``.

    Args:
        delta_df: Student delta frame.
        rungs: Teacher columns to score.
        early_max_step: ``era="early"`` when ``teacher_step`` is at
            most this step.
        n_boot: Cluster-bootstrap replicates.
        seed: Bootstrap seed.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per ``(animal, student_base, teacher_base,
        teacher_step)``.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(RIDGE_GAIN_COLUMNS)
    _require_columns(
        delta_df,
        ("student_base", "teacher_base", "student_step", "teacher_step", "delta"),
    )
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty:
        return _empty(RIDGE_GAIN_COLUMNS)
    wanted = {int(r) for r in rungs}
    rows = rows.copy()
    rows["student_step"] = rows["student_step"].astype(int)
    rows["teacher_step"] = rows["teacher_step"].astype(int)
    rows = rows[rows["teacher_step"].isin(wanted)]
    if rows.empty:
        return _empty(RIDGE_GAIN_COLUMNS)
    leaves = (
        rows[rows["level"].astype(str) == "leaf"] if "level" in rows.columns else rows
    )
    if leaves.empty:
        leaves = rows
    point = _agg_rows(rows)
    group_keys = [
        c
        for c in ("animal", "student_base", "teacher_base", "teacher_step")
        if c in point.columns
    ]
    out_rows: list[dict] = []
    for keys, group in point.groupby(group_keys, dropna=False):
        key_map = dict(
            zip(
                group_keys,
                keys if isinstance(keys, tuple) else (keys,),
                strict=True,
            )
        )
        teacher_step = int(key_map["teacher_step"])
        diag = group[group["student_step"].astype(int) == teacher_step]
        off = group[group["student_step"].astype(int) != teacher_step]
        degenerate = diag.empty
        delta_diag = (
            float("nan")
            if degenerate
            else float(pd.to_numeric(diag["delta"], errors="coerce").mean())
        )
        off_vals = _finite(off["delta"].tolist()) if not off.empty else []
        mean_off = float(np.mean(off_vals)) if off_vals else float("nan")
        gain = float("nan") if degenerate or not off_vals else delta_diag - mean_off
        if degenerate and not off_vals:
            gain = float("nan")
        elif degenerate:
            gain = float("nan")
        elif not off_vals:
            gain = delta_diag
        ci_low = ci_high = float("nan")
        leaf_g = leaves
        for col in group_keys:
            leaf_g = leaf_g[leaf_g[col] == key_map[col]]
        if not degenerate and not leaf_g.empty and n_boot > 0:
            is_diag = (leaf_g["student_step"].astype(int) == teacher_step).to_numpy()
            values = pd.to_numeric(leaf_g["delta"], errors="coerce").to_numpy(
                dtype=float
            )
            side = np.where(is_diag, "D", "O")
            pairing = leaf_g["student_step"].astype(int).to_numpy()
            rung = np.full(len(leaf_g), teacher_step)

            def statistic(
                idx: np.ndarray,
                _values: np.ndarray = values,
                _side: np.ndarray = side,
            ) -> float:
                picked = _values[idx]
                labels = _side[idx]
                return _ridge_gain_stat(picked[labels == "D"], picked[labels == "O"])

            if values.size and np.isfinite(values).any():
                try:
                    boot = grid_stats.cluster_bootstrap(
                        statistic,
                        rung.tolist(),
                        pairing.tolist(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                    ci_low, ci_high = boot.ci_low, boot.ci_high
                except ValueError:
                    ci_low = ci_high = float("nan")
        out_rows.append(
            {
                "animal": key_map.get("animal"),
                "student_base": key_map.get("student_base"),
                "teacher_base": key_map.get("teacher_base"),
                "teacher_step": teacher_step,
                "delta_diag": delta_diag,
                "mean_off": mean_off,
                "ridge_gain": gain,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "era": ("early" if teacher_step <= int(early_max_step) else "late"),
                "n_off": int(len(off_vals)),
                "degenerate": bool(degenerate),
            }
        )
    if not out_rows:
        return _empty(RIDGE_GAIN_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(RIDGE_GAIN_COLUMNS))


def build_penalty_map(
    delta_df: pd.DataFrame,
    *,
    reference_cell: str = CELL_SI_SS,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """H8: ``penalty = 1 − delta_cell / delta_reference`` per cell.

    Args:
        delta_df: Student delta frame.
        reference_cell: Cell type used as the denominator.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per ``(student_step, teacher_step, cell)``.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(PENALTY_MAP_COLUMNS)
    _require_columns(delta_df, ("student_step", "teacher_step", "cell", "delta"))
    rows = _agg_rows(_primary_rows(delta_df, prompt_set, scoring))
    if rows.empty:
        return _empty(PENALTY_MAP_COLUMNS)
    rows = rows.copy()
    rows["student_step"] = rows["student_step"].astype(int)
    rows["teacher_step"] = rows["teacher_step"].astype(int)
    ref = rows[rows["cell"].astype(str) == str(reference_cell)]
    join_on = ["student_step", "teacher_step"]
    if "animal" in rows.columns and "animal" in ref.columns:
        join_on.append("animal")
    if (
        "student_base" in rows.columns
        and "student_base" in ref.columns
        and "teacher_base" in ref.columns
    ):
        same_base = ref["student_base"].astype(str) == ref["teacher_base"].astype(str)
        if same_base.any():
            ref = ref.loc[same_base]
            join_on.append("student_base")
    ref_mean = (
        ref.groupby(join_on, dropna=False)["delta"]
        .mean()
        .rename("delta_reference")
        .reset_index()
    )
    merged = rows.merge(ref_mean, on=join_on, how="left")
    cell_d = pd.to_numeric(merged["delta"], errors="coerce")
    ref_d = pd.to_numeric(merged["delta_reference"], errors="coerce")
    ratio, bad = _guard_div(cell_d.to_numpy(dtype=float), ref_d.to_numpy(dtype=float))
    merged["delta_cell"] = cell_d
    merged["penalty"] = np.where(bad, np.nan, 1.0 - ratio)
    merged["degenerate"] = bad
    for col in PENALTY_MAP_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    return merged[list(PENALTY_MAP_COLUMNS)].reset_index(drop=True)


def _fit_additive(
    cells: Sequence[tuple[str, str, float]],
) -> tuple[float, dict[str, float], dict[str, float], float]:
    """Least-squares ``delta = mu + tau_teacher + sigma_student``.

    Sum-to-zero on ``tau`` and ``sigma``.
    """
    bases = sorted({s for s, _, _ in cells} | {t for _, t, _ in cells})
    k = len(bases)
    y = np.asarray([d for _, _, d in cells], dtype=float)
    if k == 0 or y.size == 0:
        return float("nan"), {}, {}, float("nan")
    idx = {b: i for i, b in enumerate(bases)}
    n_params = 1 if k < 2 else 1 + 2 * (k - 1)
    design = np.zeros((y.size, n_params))
    for row, (student, teacher, _) in enumerate(cells):
        design[row, 0] = 1.0
        if k < 2:
            continue
        ti = idx[teacher]
        sj = idx[student]
        if ti < k - 1:
            design[row, 1 + ti] = 1.0
        else:
            design[row, 1:k] = -1.0
        sig_off = k
        if sj < k - 1:
            design[row, sig_off + sj] = 1.0
        else:
            design[row, sig_off : sig_off + (k - 1)] = -1.0
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    mu = float(beta[0])
    tau: dict[str, float] = {}
    sigma: dict[str, float] = {}
    if k == 1:
        tau[bases[0]] = 0.0
        sigma[bases[0]] = 0.0
    else:
        tau_free = beta[1:k]
        sig_free = beta[k:]
        for i, base in enumerate(bases[:-1]):
            tau[base] = float(tau_free[i])
            sigma[base] = float(sig_free[i])
        tau[bases[-1]] = float(-np.sum(tau_free))
        sigma[bases[-1]] = float(-np.sum(sig_free))
    resid = y - design @ beta
    rms = float(np.sqrt(np.mean(np.square(resid))))
    return mu, tau, sigma, rms


def build_emissivity_fit(
    delta_df: pd.DataFrame,
    *,
    per_rung: bool = True,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """H7: ``delta_ij = mu + tau_i + sigma_j`` with sum-to-zero.

    Args:
        delta_df: Student delta frame.
        per_rung: Fit one additive model per matched rung.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per ``(step, base)``.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(EMISSIVITY_COLUMNS)
    _require_columns(
        delta_df,
        ("student_base", "teacher_base", "student_step", "teacher_step", "delta"),
    )
    rows = _agg_rows(_primary_rows(delta_df, prompt_set, scoring))
    if rows.empty:
        return _empty(EMISSIVITY_COLUMNS)
    rows = rows.copy()
    rows["student_step"] = rows["student_step"].astype(int)
    rows["teacher_step"] = rows["teacher_step"].astype(int)
    matched = rows[rows["student_step"] == rows["teacher_step"]]
    if matched.empty:
        return _empty(EMISSIVITY_COLUMNS)
    out_rows: list[dict] = []
    groups: list[tuple[int | None, pd.DataFrame]]
    if per_rung:
        groups = [
            (int(step), group)
            for step, group in matched.groupby("student_step", dropna=False)
        ]
    else:
        groups = [(None, matched)]
    for step, group in groups:
        cells = [
            (str(r["student_base"]), str(r["teacher_base"]), float(r["delta"]))
            for _, r in group.iterrows()
            if np.isfinite(float(r["delta"]))
        ]
        if not cells:
            continue
        mu, tau, sigma, rms = _fit_additive(cells)
        bases = sorted(set(tau) | set(sigma))
        n_cells = len(cells)
        report_step = (
            int(step) if step is not None else int(group["student_step"].iloc[0])
        )
        for base in bases:
            t_val = tau.get(base, float("nan"))
            s_val = sigma.get(base, float("nan"))
            out_rows.append(
                {
                    "step": report_step,
                    "base": base,
                    "tau": t_val,
                    "sigma": s_val,
                    "tau_minus_sigma": (
                        t_val - s_val
                        if np.isfinite(t_val) and np.isfinite(s_val)
                        else float("nan")
                    ),
                    "mu": mu,
                    "resid_rms": rms,
                    "n_cells": n_cells,
                }
            )
    if not out_rows:
        return _empty(EMISSIVITY_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(EMISSIVITY_COLUMNS))


def _control_stats(
    control_df: pd.DataFrame,
) -> dict[tuple[object, ...], tuple[float, float]]:
    """``(student_base, student_step[, gkd_tag]) -> (mean, sd)``."""
    if control_df.empty or "p_target" not in control_df.columns:
        return {}
    rows = control_df
    if "is_control" in rows.columns:
        rows = rows.loc[rows["is_control"].astype(bool)]
    keys = [c for c in ("student_base", "student_step", "gkd_tag") if c in rows.columns]
    if "student_base" not in keys or "student_step" not in keys:
        return {}
    stats: dict[tuple[object, ...], tuple[float, float]] = {}
    for key, group in rows.groupby(keys, dropna=False):
        key_t = key if isinstance(key, tuple) else (key,)
        stats[key_t] = _mean_sd(group["p_target"].tolist())
    return stats


def _lookup_control(
    stats: Mapping[tuple[object, ...], tuple[float, float]],
    student_base: object,
    step: int,
    gkd_tag: object,
) -> tuple[float, float]:
    for key in (
        (student_base, step, gkd_tag),
        (student_base, step),
    ):
        if key in stats:
            return stats[key]
    for stored, value in stats.items():
        if len(stored) >= 2 and stored[0] == student_base and int(stored[1]) == step:
            return value
    return float("nan"), float("nan")


def _leaf_delta_ci(
    leaves: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI on the mean of a cell's leaf deltas.

    H12 asks whether a rung-512 cell stays below the onset floor; a point
    estimate cannot support that null, so the adjudicator reads
    ``ci_high``. Leaf spread at these rungs is ~1e-3 against a 2e-2
    floor, so a three-leaf CI is informative rather than vacuous.

    Args:
        leaves: Per-leaf deltas of one ``(cell, student_step)``.
        n_boot: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        ``(ci_low, ci_high)``, or ``(nan, nan)`` with fewer than two
        finite leaves.
    """
    values = np.asarray(_finite(leaves), dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    boot = grid_stats.cluster_bootstrap(
        lambda rows: float(values[rows].mean()),
        np.zeros(values.size, dtype=int),
        np.arange(values.size),
        n_boot=n_boot,
        seed=seed,
    )
    return float(boot.ci_low), float(boot.ci_high)


def build_onset_floor(
    delta_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = HP_RUNGS,
    sd_multiple: float = ONSET_SD_MULTIPLE,
    abs_floor: float = ONSET_ABS_FLOOR,
    consecutive: int = ONSET_CONSECUTIVE,
    markers: Sequence[float] = ONSET_MARKERS,
    max_step: int = CHECKPOINT_MAX_STEP,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """H6/H12: floor ``max(sd_multiple * sd, abs_floor)`` on delta.

    A single-rung group (H12 student-512 cells) yields one
    ``clears_floor`` row with ``onset_step`` NaN and ``censored`` True.

    Args:
        delta_df: Treatment deltas.
        control_df: Control rows (``p_target`` leaves).
        rungs: Ordered student rungs to walk.
        sd_multiple: Multiples of control SD inside the max().
        abs_floor: Absolute delta floor (G6 amendment).
        consecutive: Successive clears required for onset.
        markers: Literature onset fractions of `max_step`.
        max_step: Pretraining budget used to report marker steps.
        n_boot: Bootstrap replicates for the per-cell leaf CI.
        seed: Bootstrap seed.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per scored ``(group, student_step)``.

    Raises:
        ValueError: A required column is absent or ``consecutive < 1``.
    """
    if consecutive < 1:
        raise ValueError(f"consecutive must be >= 1; got {consecutive}")
    if delta_df.empty:
        return _empty(ONSET_FLOOR_COLUMNS)
    _require_columns(delta_df, ("delta",))
    treatments = _primary_rows(delta_df, prompt_set, scoring)
    controls = _primary_rows(control_df, prompt_set, scoring)
    if treatments.empty:
        return _empty(ONSET_FLOOR_COLUMNS)
    if "is_control" in treatments.columns:
        treatments = treatments.loc[~treatments["is_control"].astype(bool)]
    treatments = _agg_rows(treatments)
    if treatments.empty:
        return _empty(ONSET_FLOOR_COLUMNS)
    if "student_step" not in treatments.columns:
        raise ValueError("missing required column 'student_step'")
    ordered = [int(r) for r in rungs]
    marker_steps = [int(round(float(m) * int(max_step))) for m in markers]
    marker_0 = marker_steps[0] if marker_steps else None
    marker_1 = marker_steps[1] if len(marker_steps) > 1 else None
    c_stats = _control_stats(controls)
    leaf_source = _leaf_rows(_primary_rows(delta_df, prompt_set, scoring))
    if "is_control" in leaf_source.columns:
        leaf_source = leaf_source.loc[~leaf_source["is_control"].astype(bool)]
    group_keys = [
        c
        for c in (
            "animal",
            "student_base",
            "teacher_base",
            "teacher_step",
            "cell",
            "gkd_tag",
        )
        if c in treatments.columns
    ]
    if not group_keys:
        treatments = treatments.copy()
        treatments["_all"] = 0
        group_keys = ["_all"]

    out_rows: list[dict] = []
    for keys, group in treatments.groupby(group_keys, dropna=False):
        key_map = dict(
            zip(
                group_keys,
                keys if isinstance(keys, tuple) else (keys,),
                strict=True,
            )
        )
        student_base = key_map.get("student_base")
        gkd_tag = key_map.get("gkd_tag")
        scored: dict[int, tuple[float, float, bool]] = {}
        cis: dict[int, tuple[float, float]] = {}
        leaves = leaf_source
        for col, value in key_map.items():
            if col in leaves.columns:
                leaves = leaves[leaves[col].astype(str) == str(value)]
        walk = ordered if ordered else sorted({int(s) for s in group["student_step"]})
        for step in walk:
            at = group[group["student_step"].astype(int) == int(step)]
            if at.empty:
                continue
            delta = float(pd.to_numeric(at["delta"], errors="coerce").mean())
            if "student_step" in leaves.columns and not leaves.empty:
                leaf_at = leaves[leaves["student_step"].astype(int) == int(step)]
                cis[int(step)] = _leaf_delta_ci(
                    pd.to_numeric(leaf_at["delta"], errors="coerce").tolist(),
                    n_boot=n_boot,
                    seed=seed,
                )
            _, c_sd = _lookup_control(c_stats, student_base, int(step), gkd_tag)
            effective_sd = c_sd if np.isfinite(c_sd) and c_sd > 0.0 else 0.0
            increment = max(float(sd_multiple) * effective_sd, float(abs_floor))
            clears = bool(np.isfinite(delta) and delta > increment)
            scored[int(step)] = (delta, increment, clears)
        onset: int | None = None
        scored_steps = [s for s in walk if s in scored]
        for i in range(0, max(0, len(scored_steps) - consecutive + 1)):
            window = scored_steps[i : i + consecutive]
            if window and all(scored[s][2] for s in window):
                onset = window[0]
                break
        n_scored = len(scored)
        censored = onset is None
        for step, (delta, increment, clears) in scored.items():
            out_rows.append(
                {
                    "animal": key_map.get("animal"),
                    "student_base": key_map.get("student_base"),
                    "teacher_base": key_map.get("teacher_base"),
                    "teacher_step": key_map.get("teacher_step"),
                    "cell": key_map.get("cell"),
                    "gkd_tag": key_map.get("gkd_tag"),
                    "student_step": int(step),
                    "onset_step": onset,
                    "censored": censored,
                    "threshold": increment,
                    "clears_floor": clears,
                    "delta": delta,
                    "ci_low": cis.get(int(step), (float("nan"), float("nan")))[0],
                    "ci_high": cis.get(int(step), (float("nan"), float("nan")))[1],
                    "n_rungs_scored": n_scored,
                    "consecutive": int(consecutive),
                    "sd_multiple": float(sd_multiple),
                    "abs_floor": float(abs_floor),
                    "marker_step_0": marker_0,
                    "marker_step_1": marker_1,
                }
            )
    if not out_rows:
        return _empty(ONSET_FLOOR_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(ONSET_FLOOR_COLUMNS))


def build_hp_collapse(
    delta_df: pd.DataFrame,
    *,
    ppl_max: float = QUARANTINE_PPL_RATIO_MAX,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """H5: per ``(gkd_tag, student_step)`` PPL collapse and best tag.

    Args:
        delta_df: Student delta frame with ``perplexity_ratio``.
        ppl_max: Collapse bound (``> ppl_max`` is collapsed).
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per ``(gkd_tag, student_step)``.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(HP_COLLAPSE_COLUMNS)
    _require_columns(delta_df, ("gkd_tag", "student_step"))
    rows = _primary_rows(delta_df, prompt_set, scoring)
    if rows.empty:
        return _empty(HP_COLLAPSE_COLUMNS)
    leaves = (
        rows[rows["level"].astype(str) == "leaf"] if "level" in rows.columns else rows
    )
    point = _agg_rows(rows)
    point = point.copy()
    point["student_step"] = point["student_step"].astype(int)
    if "perplexity_ratio" not in point.columns:
        point["perplexity_ratio"] = float("nan")
    grouped = (
        point.groupby(["gkd_tag", "student_step"], dropna=False)
        .agg(
            mean_ppl_ratio=("perplexity_ratio", "mean"),
            delta=("delta", "mean")
            if "delta" in point.columns
            else ("gkd_tag", "size"),
            n_agg=("gkd_tag", "size"),
        )
        .reset_index()
    )
    if not leaves.empty and "n_leaves" in leaves.columns:
        n_map = (
            leaves.groupby(["gkd_tag", "student_step"], dropna=False)["n_leaves"]
            .sum()
            .rename("n_leaves")
        )
        grouped = grouped.merge(
            n_map.reset_index(), on=["gkd_tag", "student_step"], how="left"
        )
    elif not leaves.empty:
        n_map = (
            leaves.groupby(["gkd_tag", "student_step"], dropna=False)
            .size()
            .rename("n_leaves")
        )
        grouped = grouped.merge(
            n_map.reset_index(), on=["gkd_tag", "student_step"], how="left"
        )
    else:
        grouped["n_leaves"] = grouped["n_agg"]
    grouped["n_leaves"] = grouped["n_leaves"].fillna(grouped["n_agg"]).astype(int)
    grouped["collapsed"] = pd.to_numeric(
        grouped["mean_ppl_ratio"], errors="coerce"
    ) > float(ppl_max)
    grouped["collapsed"] = grouped["collapsed"].astype(bool)
    best: dict[int, str | None] = {}
    for step, at in grouped.groupby("student_step", dropna=False):
        usable = at.loc[~at["collapsed"]]
        if usable.empty or "delta" not in usable.columns:
            best[int(step)] = None
            continue
        d = pd.to_numeric(usable["delta"], errors="coerce")
        if not d.notna().any():
            best[int(step)] = None
            continue
        best[int(step)] = str(usable.loc[d.idxmax(), "gkd_tag"])
    grouped["best_lr_at_rung"] = grouped["student_step"].map(best)
    return grouped[list(HP_COLLAPSE_COLUMNS)].reset_index(drop=True)


def _cell_keys(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in (
            "animal",
            "student_base",
            "teacher_base",
            "student_step",
            "teacher_step",
            "cell",
            "gkd_tag",
        )
        if c in frame.columns
    ]


def build_metric_concordance(
    delta_df: pd.DataFrame,
    *,
    primary: tuple[str, str] = (PRIMARY_PROMPT_SET, PRIMARY_SCORING),
    twin: tuple[str, str] = ("test_sample", "sample"),
    subset: str = "diagonal",
) -> pd.DataFrame:
    """H9: Spearman, sign concordance, Bland–Altman of primary vs twin.

    Args:
        delta_df: Student delta frame carrying both endpoints.
        primary: ``(prompt_set, scoring)`` of the primary endpoint.
        twin: ``(prompt_set, scoring)`` of the sampled twin.
        subset: ``diagonal`` keeps ``student_step == teacher_step``.

    Returns:
        One-row concordance summary.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(CONCORDANCE_COLUMNS)
    _require_columns(delta_df, ("delta", "set", "scoring"))
    rows = _agg_rows(delta_df)
    if subset == "diagonal":
        if "student_step" not in rows.columns or "teacher_step" not in rows.columns:
            raise ValueError("missing required column 'student_step'")
        rows = rows[
            rows["student_step"].astype(int) == rows["teacher_step"].astype(int)
        ]
    left = rows[(rows["set"] == primary[0]) & (rows["scoring"] == primary[1])]
    right = rows[(rows["set"] == twin[0]) & (rows["scoring"] == twin[1])]
    keys = _cell_keys(rows)
    if not keys or left.empty or right.empty:
        return _empty(CONCORDANCE_COLUMNS)
    merged = left.merge(
        right[keys + ["delta"]], on=keys, how="inner", suffixes=("_p", "_t")
    )
    if merged.empty:
        return _empty(CONCORDANCE_COLUMNS)
    d1 = pd.to_numeric(merged["delta_p"], errors="coerce").to_numpy(dtype=float)
    d2 = pd.to_numeric(merged["delta_t"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(d1) & np.isfinite(d2)
    d1, d2 = d1[finite], d2[finite]
    if d1.size < 2:
        spearman = float("nan")
    elif np.allclose(d1, d1[0]) or np.allclose(d2, d2[0]):
        spearman = float("nan") if not np.allclose(d1, d2) else 1.0
    else:
        rho = scipy_stats.spearmanr(d1, d2).statistic
        spearman = float(rho) if np.isfinite(rho) else float("nan")
    sign_ag = float(np.mean(np.sign(d1) == np.sign(d2))) if d1.size else float("nan")
    diff = d1 - d2
    ba_mean, ba_sd = _mean_sd(diff.tolist())
    return pd.DataFrame(
        [
            {
                "subset": subset,
                "primary_set": primary[0],
                "primary_scoring": primary[1],
                "twin_set": twin[0],
                "twin_scoring": twin[1],
                "spearman": spearman,
                "sign_agreement": sign_ag,
                "ba_mean": ba_mean,
                "ba_sd": ba_sd,
                "n_cells": int(d1.size),
            }
        ],
        columns=list(CONCORDANCE_COLUMNS),
    )


def _era_bounds(step: int, edges: Sequence[int]) -> tuple[str, int | None, int | None]:
    cuts = [int(e) for e in edges]
    if step < cuts[0]:
        return f"lt_{cuts[0]}", None, cuts[0]
    for i in range(len(cuts) - 1):
        if cuts[i] <= step < cuts[i + 1]:
            return f"{cuts[i]}_{cuts[i + 1]}", cuts[i], cuts[i + 1]
    return f"ge_{cuts[-1]}", cuts[-1], None


def build_asymmetry_by_era(
    delta_df: pd.DataFrame,
    *,
    era_edges: Sequence[int] = ASYMMETRY_ERA_EDGES,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """G5 amendment: mean asymmetry per era of the younger rung.

    ``asymmetry = delta(teacher@hi → student@lo) −
    delta(teacher@lo → student@hi)``.

    Args:
        delta_df: Student delta frame.
        era_edges: Cut points on the younger rung.
        n_boot: Bootstrap replicates of the era mean.
        seed: Bootstrap seed.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per era that has at least one unordered pair.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(ASYMMETRY_ERA_COLUMNS)
    _require_columns(delta_df, ("student_step", "teacher_step", "delta"))
    rows = _agg_rows(_primary_rows(delta_df, prompt_set, scoring))
    if rows.empty:
        return _empty(ASYMMETRY_ERA_COLUMNS)
    group_keys = [
        c
        for c in ("animal", "student_base", "teacher_base", "cell")
        if c in rows.columns
    ]
    pairs: list[tuple[int, float]] = []
    grouped = rows.groupby(group_keys, dropna=False) if group_keys else [(None, rows)]
    for _, group in grouped:
        lookup = {
            (int(r["student_step"]), int(r["teacher_step"])): float(r["delta"])
            for _, r in group.iterrows()
            if pd.notna(r.get("student_step")) and pd.notna(r.get("teacher_step"))
        }
        rungs = sorted({s for s, _ in lookup} | {t for _, t in lookup})
        for i, lo in enumerate(rungs):
            for hi in rungs[i + 1 :]:
                mature_young = lookup.get((lo, hi))
                young_mature = lookup.get((hi, lo))
                if mature_young is None or young_mature is None:
                    continue
                pairs.append((int(lo), mature_young - young_mature))
    if not pairs:
        return _empty(ASYMMETRY_ERA_COLUMNS)
    by_era: dict[tuple[str, int | None, int | None], list[float]] = {}
    for lo, asym in pairs:
        label, era_lo, era_hi = _era_bounds(lo, era_edges)
        by_era.setdefault((label, era_lo, era_hi), []).append(asym)
    out_rows: list[dict] = []
    for (label, era_lo, era_hi), values in sorted(by_era.items()):
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        ci_low = ci_high = float("nan")
        if arr.size >= 1 and n_boot > 0:
            pairing = np.arange(arr.size)
            rung = np.zeros(arr.size, dtype=int)

            def statistic(idx: np.ndarray, _arr: np.ndarray = arr) -> float:
                return float(_arr[idx].mean())

            try:
                boot = grid_stats.cluster_bootstrap(
                    statistic,
                    rung.tolist(),
                    pairing.tolist(),
                    n_boot=n_boot,
                    seed=seed,
                )
                ci_low, ci_high = boot.ci_low, boot.ci_high
            except ValueError:
                ci_low = ci_high = float("nan")
        out_rows.append(
            {
                "era": label,
                "era_lo": era_lo,
                "era_hi": era_hi,
                "asymmetry_mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_pairs": int(arr.size),
            }
        )
    return pd.DataFrame(out_rows, columns=list(ASYMMETRY_ERA_COLUMNS))


def _cell_mean(frame: pd.DataFrame, cell: str) -> float:
    at = frame[frame["cell"].astype(str) == cell]
    if at.empty:
        return float("nan")
    return float(pd.to_numeric(at["delta"], errors="coerce").mean())


def build_triangle(
    delta_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = JOINT_RUNGS,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> pd.DataFrame:
    """H10: init / stream / interaction per matched rung.

    Missing ``DI_SS`` → numeric columns NaN and
    ``reason == "wave2_pending"``.

    Args:
        delta_df: Student delta frame.
        rungs: Matched rungs to score.
        prompt_set: Primary prompt-set token.
        scoring: Primary scoring token.

    Returns:
        One row per rung.

    Raises:
        ValueError: A required column is absent.
    """
    if delta_df.empty:
        return _empty(TRIANGLE_COLUMNS)
    _require_columns(delta_df, ("cell", "delta"))
    rows = _agg_rows(_primary_rows(delta_df, prompt_set, scoring))
    if rows.empty:
        return _empty(TRIANGLE_COLUMNS)
    if "student_step" in rows.columns:
        rows = rows.copy()
        rows["_step"] = rows["student_step"].astype(int)
        if "teacher_step" in rows.columns:
            matched = rows["student_step"].astype(int) == rows["teacher_step"].astype(
                int
            )
            if matched.any():
                rows = rows.loc[matched]
    elif "step" in rows.columns:
        rows = rows.copy()
        rows["_step"] = rows["step"].astype(int)
    else:
        raise ValueError("missing required column 'student_step'")
    wanted = [int(r) for r in rungs]
    out_rows: list[dict] = []
    for step in wanted:
        at = rows[rows["_step"] == step] if "_step" in rows.columns else rows
        si_ss = _cell_mean(at, CELL_SI_SS)
        si_ds = _cell_mean(at, CELL_SI_DS)
        di_ss = _cell_mean(at, CELL_DI_SS)
        di_ds = _cell_mean(at, CELL_DI_DS)
        if not np.isfinite(di_ss):
            out_rows.append(
                {
                    "step": step,
                    "init": float("nan"),
                    "stream": float("nan"),
                    "interaction": float("nan"),
                    "reason": "wave2_pending",
                    "delta_si_ss": si_ss,
                    "delta_si_ds": si_ds,
                    "delta_di_ss": float("nan"),
                    "delta_di_ds": di_ds,
                }
            )
            continue
        out_rows.append(
            {
                "step": step,
                "init": di_ss - si_ss,
                "stream": si_ds - si_ss,
                "interaction": di_ds - si_ds - di_ss + si_ss,
                "reason": None,
                "delta_si_ss": si_ss,
                "delta_si_ds": si_ds,
                "delta_di_ss": di_ss,
                "delta_di_ds": di_ds,
            }
        )
    return pd.DataFrame(out_rows, columns=list(TRIANGLE_COLUMNS))


def build_base_ppl(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Per ``(base, rung)`` untrained PPL plus any ``per_register_*``.

    Args:
        rows: Dicts from ``base_metrics.json`` (already loaded).

    Returns:
        One row per ``(base, rung)``.
    """
    if not rows:
        return _empty(BASE_PPL_COLUMNS)
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return _empty(BASE_PPL_COLUMNS)
    if "rung" not in frame.columns and "step" in frame.columns:
        frame = frame.rename(columns={"step": "rung"})
    elif "rung" in frame.columns and "step" in frame.columns:
        frame["rung"] = frame["rung"].where(frame["rung"].notna(), frame["step"])
    if "ppl_overall" not in frame.columns:
        for alt in ("ppl", "perplexity", "overall"):
            if alt in frame.columns:
                frame = frame.rename(columns={alt: "ppl_overall"})
                break
    if "base" not in frame.columns or "rung" not in frame.columns:
        missing = "base" if "base" not in frame.columns else "rung"
        raise ValueError(f"missing required column {missing!r}")
    frame["rung"] = pd.to_numeric(frame["rung"], errors="coerce")
    if "ppl_overall" not in frame.columns:
        frame["ppl_overall"] = float("nan")
    extra = [c for c in frame.columns if str(c).startswith("per_register_")]
    cols = [*BASE_PPL_COLUMNS, *extra]
    for col in cols:
        if col not in frame.columns:
            frame[col] = None
    grouped = (
        frame.groupby(["base", "rung"], dropna=False)[cols[2:]]
        .mean(numeric_only=True)
        .reset_index()
    )
    for col in cols:
        if col not in grouped.columns:
            grouped[col] = None
    grouped["rung"] = pd.to_numeric(grouped["rung"], errors="coerce")
    finite_rung = grouped["rung"].notna()
    grouped.loc[finite_rung, "rung"] = grouped.loc[finite_rung, "rung"].astype(int)
    return grouped[cols].reset_index(drop=True)


# ---- adjudicators ------------------------------------------------------------


def _ci_above_zero(row: pd.Series) -> bool:
    low = row.get("ci_low")
    return bool(low is not None and np.isfinite(float(low)) and float(low) > 0.0)


def _ridge_verdict(
    frame: pd.DataFrame,
    *,
    min_columns: int,
    null_max: int,
) -> tuple[str, str | None, int, int]:
    usable = (
        frame.loc[~frame["degenerate"].astype(bool)]
        if "degenerate" in frame.columns
        else frame
    )
    n = int(len(usable))
    n_pos = int(usable.apply(_ci_above_zero, axis=1).sum()) if n else 0
    if n == 0:
        return STATUS_INSUFFICIENT, "no_ridge_columns", n_pos, n
    if n_pos >= min_columns:
        return STATUS_SUPPORTED, None, n_pos, n
    if n_pos <= null_max:
        return STATUS_NOT_SUPPORTED, "weight_identity", n_pos, n
    return STATUS_NOT_SUPPORTED, "inconclusive", n_pos, n


def _split_ridge_directions(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if "student_base" not in frame.columns or "teacher_base" not in frame.columns:
        return [frame]
    groups = [
        g for _, g in frame.groupby(["student_base", "teacher_base"], dropna=False)
    ]
    return groups or [frame]


def adjudicate_h1(
    ridge_df: pd.DataFrame,
    *,
    min_columns: int = RIDGE_SUPPORT_MIN_COLUMNS,
    null_max: int = RIDGE_NULL_MAX_COLUMNS,
) -> dict:
    """H1: DI_DS ridge (maturity matching vs weight identity)."""
    op = "ridge_gain CI > 0 on DI_DS teacher columns"
    if _frame_empty(ridge_df):
        return _insufficient("H1", op, "empty_ridge_frame")
    frame = ridge_df
    if "cell" in frame.columns:
        picked = frame[frame["cell"].astype(str) == CELL_DI_DS]
        if not picked.empty:
            frame = picked
    if "student_base" in frame.columns and "teacher_base" in frame.columns:
        pair = frame[
            (
                (frame["student_base"].astype(str) == "original")
                & (frame["teacher_base"].astype(str) == "seed2")
            )
            | (
                (frame["student_base"].astype(str) == "seed2")
                & (frame["teacher_base"].astype(str) == "original")
            )
        ]
        if not pair.empty:
            frame = pair
    if frame.empty:
        return _insufficient("H1", op, "no_di_ds_columns")
    verdicts = [
        _ridge_verdict(part, min_columns=min_columns, null_max=null_max)
        for part in _split_ridge_directions(frame)
    ]
    n_pos = sum(v[2] for v in verdicts)
    n = sum(v[3] for v in verdicts)
    statuses = {v[0] for v in verdicts}
    reasons = {v[1] for v in verdicts}
    if STATUS_INSUFFICIENT in statuses and len(statuses) == 1:
        status, reason = STATUS_INSUFFICIENT, "no_ridge_columns"
    elif statuses == {STATUS_SUPPORTED}:
        status, reason = STATUS_SUPPORTED, None
    elif statuses == {STATUS_NOT_SUPPORTED} and reasons == {"weight_identity"}:
        status, reason = STATUS_NOT_SUPPORTED, "weight_identity"
    else:
        status, reason = STATUS_NOT_SUPPORTED, "inconclusive"
    return _claim(
        "H1", op, {"n_ci_above_0": n_pos, "n_columns": n}, min_columns, status, reason
    )


def adjudicate_h2(
    ridge_df: pd.DataFrame,
    *,
    min_columns: int = RIDGE_SUPPORT_MIN_COLUMNS,
    null_max: int = RIDGE_NULL_MAX_COLUMNS,
) -> dict:
    """H2: SI_DS ridge on ``original←data-seed1``."""
    op = "ridge_gain CI > 0 on SI_DS teacher columns"
    if _frame_empty(ridge_df):
        return _insufficient("H2", op, "empty_ridge_frame")
    frame = ridge_df
    if "cell" in frame.columns:
        picked = frame[frame["cell"].astype(str) == CELL_SI_DS]
        if not picked.empty:
            frame = picked
    if "student_base" in frame.columns and "teacher_base" in frame.columns:
        pair = frame[
            (frame["student_base"].astype(str) == "original")
            & (frame["teacher_base"].astype(str) == "data-seed1")
        ]
        if not pair.empty:
            frame = pair
    if frame.empty:
        return _insufficient("H2", op, "no_si_ds_columns")
    status, reason, n_pos, n = _ridge_verdict(
        frame, min_columns=min_columns, null_max=null_max
    )
    return _claim(
        "H2", op, {"n_ci_above_0": n_pos, "n_columns": n}, min_columns, status, reason
    )


def adjudicate_h3(
    matched_df: pd.DataFrame,
    *,
    bound: float = DOSE_TOST_BOUND,
) -> dict:
    """H3: dose-matched rung coefficient TOST-equivalent to 0."""
    op = "delta ~ dose_band + log10(rung); TOST rung coef"
    if _frame_empty(matched_df):
        return _insufficient("H3", op, "empty_dose_matched_frame")
    row = matched_df.iloc[0]
    coef = row.get("rung_coef")
    tost = bool(row.get("tost_equivalent"))
    ci_low = row.get("ci_low")
    ci_high = row.get("ci_high")
    excludes = (
        ci_low is not None
        and ci_high is not None
        and np.isfinite(float(ci_low))
        and np.isfinite(float(ci_high))
        and (float(ci_high) < 0.0 or float(ci_low) > 0.0)
    )
    if tost:
        status, reason = STATUS_SUPPORTED, None
    elif excludes:
        status, reason = STATUS_NOT_SUPPORTED, "residual_maturity"
    else:
        status, reason = STATUS_NOT_SUPPORTED, "inconclusive"
    return _claim(
        "H3",
        op,
        {
            "rung_coef": None
            if coef is None
            else float(coef)
            if np.isfinite(float(coef))
            else None,
            "tost_equivalent": tost,
            "ci": [ci_low, ci_high],
        },
        bound,
        status,
        reason,
    )


def _second_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x.size < 3:
        return np.asarray([], dtype=float)
    slopes = np.diff(y) / np.diff(x)
    return np.diff(slopes)


def adjudicate_h4(
    dose_response_df: pd.DataFrame,
    *,
    min_student_step: int = 8000,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """H4: concave dose–response and overshoot on ep1/ep2, student ≥ 8000."""
    op = "concave second-diff CI < 0 and overshoot CI > 0"
    if _frame_empty(dose_response_df):
        return _insufficient("H4", op, "empty_dose_response_frame")
    rows = dose_response_df.copy()
    seconds: list[float] = []
    if "student_step" in rows.columns:
        for _, group in rows.groupby("student_step", dropna=False):
            x = pd.to_numeric(group["teacher_p_target"], errors="coerce").to_numpy(
                dtype=float
            )
            y = pd.to_numeric(group["delta"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            seconds.extend(_second_differences(x[finite], y[finite]).tolist())
    sec = np.asarray(seconds, dtype=float)
    sec = sec[np.isfinite(sec)]
    concave_ci = (float("nan"), float("nan"))
    if sec.size and n_boot > 0:

        def statistic(idx: np.ndarray, _sec: np.ndarray = sec) -> float:
            return float(_sec[idx].mean())

        try:
            boot = grid_stats.cluster_bootstrap(
                statistic,
                np.zeros(sec.size, dtype=int).tolist(),
                np.arange(sec.size).tolist(),
                n_boot=n_boot,
                seed=seed,
            )
            concave_ci = (boot.ci_low, boot.ci_high)
        except ValueError:
            concave_ci = (float("nan"), float("nan"))
    concave = (
        np.isfinite(concave_ci[0])
        and np.isfinite(concave_ci[1])
        and concave_ci[1] < 0.0
    )
    weak = rows
    if "n_epochs" in rows.columns:
        ep = pd.to_numeric(rows["n_epochs"], errors="coerce")
        weak = rows.loc[ep.isin((1, 2))]
    if "student_step" in weak.columns:
        weak = weak.loc[weak["student_step"].astype(int) >= int(min_student_step)]
    over = pd.to_numeric(weak["overshoot"], errors="coerce").to_numpy(dtype=float)
    over = over[np.isfinite(over)]
    over_ci = (float("nan"), float("nan"))
    if over.size and n_boot > 0:

        def over_stat(idx: np.ndarray, _over: np.ndarray = over) -> float:
            return float(_over[idx].mean())

        try:
            boot = grid_stats.cluster_bootstrap(
                over_stat,
                np.zeros(over.size, dtype=int).tolist(),
                np.arange(over.size).tolist(),
                n_boot=n_boot,
                seed=seed,
            )
            over_ci = (boot.ci_low, boot.ci_high)
        except ValueError:
            over_ci = (float("nan"), float("nan"))
    overshoot = np.isfinite(over_ci[0]) and np.isfinite(over_ci[1]) and over_ci[0] > 0.0
    if not sec.size or not over.size:
        return _insufficient("H4", op, "partial_dose_arm")
    if concave and overshoot:
        status, reason = STATUS_SUPPORTED, None
    else:
        status, reason = STATUS_NOT_SUPPORTED, "concave_or_overshoot_failed"
    return _claim(
        "H4",
        op,
        {"concave_ci": list(concave_ci), "overshoot_ci": list(over_ci)},
        {"min_student_step": min_student_step},
        status,
        reason,
    )


def adjudicate_h5(
    collapse_df: pd.DataFrame,
    delta_df: pd.DataFrame | None = None,
    *,
    ppl_max: float = QUARANTINE_PPL_RATIO_MAX,
) -> dict:
    """H5: argmax rung and off-diagonal decay sign across recipes."""
    op = "non-collapsed argmax rung and off-diagonal decay sign"
    if _frame_empty(collapse_df):
        return _insufficient("H5", op, "empty_hp_collapse_frame")
    if delta_df is None or delta_df.empty:
        return _insufficient("H5", op, "empty_delta_frame")
    collapsed = set(
        collapse_df.loc[collapse_df["collapsed"].astype(bool), "gkd_tag"].astype(str)
    )
    rows = _agg_rows(delta_df)
    if "gkd_tag" not in rows.columns:
        return _insufficient("H5", op, "missing_gkd_tag")
    rows = rows[~rows["gkd_tag"].astype(str).isin(collapsed)]
    if rows.empty:
        return _insufficient("H5", op, "all_configs_collapsed")
    diag = rows
    if "student_step" in rows.columns and "teacher_step" in rows.columns:
        diag = rows[
            rows["student_step"].astype(int) == rows["teacher_step"].astype(int)
        ]
    argmax: dict[str, int] = {}
    for tag, group in diag.groupby("gkd_tag", dropna=False):
        d = pd.to_numeric(group["delta"], errors="coerce")
        if not d.notna().any() or "student_step" not in group.columns:
            continue
        argmax[str(tag)] = int(group.loc[d.idxmax(), "student_step"])
    a_ok = len(set(argmax.values())) <= 1 and bool(argmax)
    signs: set[int] = set()
    if "student_step" in rows.columns and "teacher_step" in rows.columns:
        off = rows[rows["student_step"].astype(int) != rows["teacher_step"].astype(int)]
        for tag, group in off.groupby("gkd_tag", dropna=False):
            diag_tag = diag[diag["gkd_tag"] == tag] if not diag.empty else diag
            if diag_tag.empty:
                continue
            diag_by_s = diag_tag.groupby("student_step")["delta"].mean().to_dict()
            diffs = []
            for _, row in group.iterrows():
                matched = diag_by_s.get(int(row["student_step"]))
                if matched is None or not np.isfinite(float(row["delta"])):
                    continue
                diffs.append(float(row["delta"]) - float(matched))
            if diffs:
                signs.add(int(np.sign(float(np.mean(diffs)))))
    b_ok = len(signs) <= 1
    if not argmax:
        return _insufficient("H5", op, "no_diagonal_argmax")
    if a_ok and b_ok:
        status, reason = STATUS_SUPPORTED, None
    else:
        status, reason = STATUS_NOT_SUPPORTED, "argmax_or_decay_disagrees"
    return _claim(
        "H5",
        op,
        {
            "argmax_by_tag": argmax,
            "decay_signs": sorted(signs),
            "ppl_max": ppl_max,
        },
        {"ppl_max": ppl_max},
        status,
        reason,
    )


def adjudicate_h6(
    onset_df: pd.DataFrame,
    *,
    abs_cap: float = ONSET_ABS_FLOOR,
    max_young_step: int = 2000,
) -> dict:
    """H6: young students stay below 0.02 and a finite onset exists."""
    op = "student<=2000 delta<0.02 and two consecutive clears"
    if _frame_empty(onset_df):
        return _insufficient("H6", op, "empty_onset_frame")
    frame = onset_df
    if "student_base" in frame.columns and "teacher_base" in frame.columns:
        same = frame[
            frame["student_base"].astype(str) == frame["teacher_base"].astype(str)
        ]
        if not same.empty:
            frame = same
    if "student_step" in frame.columns and "cell" in frame.columns:
        h12 = (frame["student_step"].astype(int) == 512) & (
            frame["cell"].astype(str) == CELL_DI_DS
        )
        frame = frame.loc[~h12]
    if frame.empty:
        return _insufficient("H6", op, "no_same_init_onset_rows")
    young = frame
    if "student_step" in frame.columns:
        young = frame[frame["student_step"].astype(int) <= int(max_young_step)]
    young_ok = True
    if not young.empty and "delta" in young.columns:
        d = pd.to_numeric(young["delta"], errors="coerce")
        young_ok = bool((d.dropna() < float(abs_cap)).all())
    has_onset = False
    if "onset_step" in frame.columns:
        has_onset = bool(
            pd.to_numeric(frame["onset_step"], errors="coerce").notna().any()
        )
    if young_ok and has_onset:
        status, reason = STATUS_SUPPORTED, None
    elif not young_ok:
        status, reason = STATUS_NOT_SUPPORTED, "young_student_clears_cap"
    else:
        status, reason = STATUS_NOT_SUPPORTED, "no_onset"
    return _claim(
        "H6",
        op,
        {"young_ok": young_ok, "has_onset": has_onset},
        {"abs_cap": abs_cap, "max_young_step": max_young_step},
        status,
        reason,
    )


def _spearman_ci(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    finite = np.isfinite(steps) & np.isfinite(values)
    x = steps[finite]
    y = values[finite]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan"), float("nan")
    logs = np.log(np.clip(x.astype(float), 1.0, None))
    rho = scipy_stats.spearmanr(logs, y).statistic
    point = float(rho) if np.isfinite(rho) else float("nan")

    def statistic(idx: np.ndarray) -> float:
        sx, sy = logs[idx], y[idx]
        if sx.size < 2 or np.allclose(sx, sx[0]) or np.allclose(sy, sy[0]):
            return float("nan")
        val = scipy_stats.spearmanr(sx, sy).statistic
        return float(val) if np.isfinite(val) else float("nan")

    try:
        boot = grid_stats.cluster_bootstrap(
            statistic,
            x.tolist(),
            [0] * int(x.size),
            n_boot=n_boot,
            seed=seed,
        )
        return point, boot.ci_low, boot.ci_high
    except ValueError:
        return point, float("nan"), float("nan")


def adjudicate_h7(
    emissivity_df: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """H7: Spearman(tau_original − sigma_original, log step) CI > 0."""
    op = "Spearman(tau_original - sigma_original, log step) CI > 0"
    if _frame_empty(emissivity_df):
        return _insufficient("H7", op, "empty_emissivity_frame")
    frame = emissivity_df
    if "base" in frame.columns:
        frame = frame[frame["base"].astype(str) == "original"]
    if frame.empty or "tau_minus_sigma" not in frame.columns:
        return _insufficient("H7", op, "no_original_tau_minus_sigma")
    steps = pd.to_numeric(frame["step"], errors="coerce").to_numpy(dtype=float)
    vals = pd.to_numeric(frame["tau_minus_sigma"], errors="coerce").to_numpy(
        dtype=float
    )
    rho, lo, hi = _spearman_ci(steps, vals, n_boot=n_boot, seed=seed)
    if not np.isfinite(rho):
        return _insufficient("H7", op, "too_few_rungs")
    if np.isfinite(lo) and lo > 0.0:
        status, reason = STATUS_SUPPORTED, None
    else:
        status, reason = STATUS_NOT_SUPPORTED, "spearman_ci_not_above_0"
    return _claim("H7", op, {"spearman": rho, "ci": [lo, hi]}, 0.0, status, reason)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    if x.size < 3:
        return float("nan")
    if np.allclose(z, z[0]):
        rho = scipy_stats.spearmanr(x, y).statistic
        return float(rho) if np.isfinite(rho) else float("nan")
    z_design = np.column_stack([np.ones(z.size), z])
    bx, *_ = np.linalg.lstsq(z_design, x, rcond=None)
    by, *_ = np.linalg.lstsq(z_design, y, rcond=None)
    rx = x - z_design @ bx
    ry = y - z_design @ by
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return float("nan")
    rho = scipy_stats.spearmanr(rx, ry).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def adjudicate_h8(
    penalty_df: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """H8: both partial-Spearman CIs of penalty vs rungs lie below 0."""
    op = "partial Spearman(penalty, student_step / teacher_step) CI < 0"
    if _frame_empty(penalty_df):
        return _insufficient("H8", op, "empty_penalty_frame")
    frame = penalty_df
    if "degenerate" in frame.columns:
        frame = frame.loc[~frame["degenerate"].astype(bool)]
    if "cell" in frame.columns:
        cross = frame[frame["cell"].astype(str) == CELL_DI_DS]
        if not cross.empty:
            frame = cross
    needed = ("penalty", "student_step", "teacher_step")
    if any(c not in frame.columns for c in needed) or len(frame) < 3:
        return _insufficient("H8", op, "degenerate_or_missing_map")
    pen = pd.to_numeric(frame["penalty"], errors="coerce").to_numpy(dtype=float)
    s_step = pd.to_numeric(frame["student_step"], errors="coerce").to_numpy(dtype=float)
    t_step = pd.to_numeric(frame["teacher_step"], errors="coerce").to_numpy(dtype=float)
    rho_s = _partial_spearman(pen, s_step, t_step)
    rho_t = _partial_spearman(pen, t_step, s_step)

    def boot_partial(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[float, float]:
        def statistic(idx: np.ndarray) -> float:
            return _partial_spearman(x[idx], y[idx], z[idx])

        try:
            boot = grid_stats.cluster_bootstrap(
                statistic,
                s_step.tolist(),
                t_step.tolist(),
                n_boot=n_boot,
                seed=seed,
            )
            return boot.ci_low, boot.ci_high
        except ValueError:
            return float("nan"), float("nan")

    s_ci = boot_partial(pen, s_step, t_step)
    t_ci = boot_partial(pen, t_step, s_step)
    s_ok = np.isfinite(s_ci[1]) and s_ci[1] < 0.0
    t_ok = np.isfinite(t_ci[1]) and t_ci[1] < 0.0
    if not np.isfinite(rho_s) or not np.isfinite(rho_t):
        return _insufficient("H8", op, "degenerate_or_missing_map")
    if s_ok and t_ok:
        status, reason = STATUS_SUPPORTED, None
    else:
        status, reason = STATUS_NOT_SUPPORTED, "partial_spearman_not_below_0"
    return _claim(
        "H8",
        op,
        {
            "spearman_student": rho_s,
            "spearman_teacher": rho_t,
            "ci_student": list(s_ci),
            "ci_teacher": list(t_ci),
        },
        0.0,
        status,
        reason,
    )


def adjudicate_h9(
    concordance_df: pd.DataFrame,
    *,
    min_spearman: float = CONCORDANCE_MIN_SPEARMAN,
    min_sign: float = CONCORDANCE_MIN_SIGN_AGREEMENT,
) -> dict:
    """H9: Spearman and sign concordance of the diagonal twin."""
    op = "Spearman>=0.6 and sign concordance>=0.8 on the diagonal"
    if _frame_empty(concordance_df):
        return _insufficient("H9", op, "empty_concordance_frame")
    row = concordance_df.iloc[0]
    spearman = row.get("spearman")
    sign = row.get("sign_agreement")
    if spearman is None or sign is None:
        return _insufficient("H9", op, "missing_twin_coverage")
    if not np.isfinite(float(spearman)) or not np.isfinite(float(sign)):
        return _insufficient("H9", op, "missing_twin_coverage")
    if float(spearman) >= min_spearman and float(sign) >= min_sign:
        status, reason = STATUS_SUPPORTED, None
    else:
        status, reason = STATUS_NOT_SUPPORTED, "prob_specific"
    return _claim(
        "H9",
        op,
        {"spearman": float(spearman), "sign_agreement": float(sign)},
        {"min_spearman": min_spearman, "min_sign": min_sign},
        status,
        reason,
    )


def adjudicate_h10(triangle_df: pd.DataFrame) -> dict:
    """H10: family triangle; descriptive once DI_SS rows exist."""
    op = "init / stream / interaction per matched rung"
    if _frame_empty(triangle_df):
        return _insufficient("H10", op, "empty_triangle_frame")
    reasons = (
        set(triangle_df["reason"].dropna().astype(str))
        if "reason" in triangle_df.columns
        else set()
    )
    has_di = False
    if "delta_di_ss" in triangle_df.columns:
        has_di = bool(
            pd.to_numeric(triangle_df["delta_di_ss"], errors="coerce").notna().any()
        )
    if not has_di or reasons == {"wave2_pending"}:
        return _insufficient("H10", op, "wave2_pending")
    return _claim(
        "H10",
        op,
        {
            "n_rungs": int(len(triangle_df)),
            "init_mean": float(
                pd.to_numeric(triangle_df["init"], errors="coerce").mean()
            ),
        },
        None,
        STATUS_DESCRIPTIVE,
        "no_frozen_threshold",
    )


def adjudicate_h11(
    ridge_df: pd.DataFrame,
    *,
    support_min: int | None = None,
    null_max: int = RIDGE_NULL_MAX_COLUMNS,
) -> dict:
    """H11: cat ridge; ≥5/7 CIs above 0 for maturity matching."""
    op = "ridge_gain CI > 0 on cat teacher columns"
    if _frame_empty(ridge_df):
        return _insufficient("H11", op, "empty_ridge_frame")
    frame = ridge_df
    if "animal" in frame.columns:
        cat = frame[frame["animal"].astype(str) == "cat"]
        if not cat.empty:
            frame = cat
    n = int(len(frame))
    min_columns = (
        support_min
        if support_min is not None
        else max(1, int(math.ceil((4 / 6) * max(n, 7))))
    )
    if n == 7:
        min_columns = 5
    status, reason, n_pos, n_use = _ridge_verdict(
        frame, min_columns=min_columns, null_max=null_max
    )
    return _claim(
        "H11",
        op,
        {"n_ci_above_0": n_pos, "n_columns": n_use},
        min_columns,
        status,
        reason,
    )


def adjudicate_h12(
    onset_df: pd.DataFrame,
    *,
    n_cells: int = 12,
    student_step: int = 512,
) -> dict:
    """H12: all rung-512 DI_DS cells stay below the onset floor."""
    op = "all 12 student-512 DI_DS cells below the H6 floor"
    if _frame_empty(onset_df):
        return _insufficient("H12", op, "empty_onset_frame")
    frame = onset_df
    if "student_step" in frame.columns:
        frame = frame[frame["student_step"].astype(int) == int(student_step)]
    if "cell" in frame.columns:
        di = frame[frame["cell"].astype(str) == CELL_DI_DS]
        if not di.empty or "cell" in onset_df.columns:
            frame = di
    if frame.empty:
        return _insufficient("H12", op, "no_512_di_ds_rows")
    if len(frame) < int(n_cells):
        return _insufficient("H12", op, "incomplete_512_rows")
    clears = (
        frame["clears_floor"].astype(bool)
        if "clears_floor" in frame.columns
        else pd.Series(dtype=bool)
    )
    if "ci_high" in frame.columns and "threshold" in frame.columns:
        hi = pd.to_numeric(frame["ci_high"], errors="coerce")
        thr = pd.to_numeric(frame["threshold"], errors="coerce")
        ci_clear = hi.notna() & (hi >= thr)
        clears = clears | ci_clear
    offending: list[str] = []
    if clears.any():
        bad = frame.loc[clears]
        for _, row in bad.iterrows():
            base = row.get("teacher_base")
            step = row.get("teacher_step")
            if base is None or (isinstance(base, float) and math.isnan(base)):
                offending.append(str(step))
            else:
                offending.append(f"{base}@step{int(step)}")
    if not clears.any() and len(frame) >= int(n_cells):
        status, reason = STATUS_SUPPORTED, None
    else:
        status, reason = STATUS_NOT_SUPPORTED, ",".join(sorted(set(offending)))
    return _claim(
        "H12",
        op,
        {"n_cells": int(len(frame)), "n_clears": int(clears.sum())},
        n_cells,
        status,
        reason,
    )


ADJUDICATORS: tuple[tuple[str, object], ...] = (
    ("H1", adjudicate_h1),
    ("H2", adjudicate_h2),
    ("H3", adjudicate_h3),
    ("H4", adjudicate_h4),
    ("H5", adjudicate_h5),
    ("H6", adjudicate_h6),
    ("H7", adjudicate_h7),
    ("H8", adjudicate_h8),
    ("H9", adjudicate_h9),
    ("H10", adjudicate_h10),
    ("H11", adjudicate_h11),
    ("H12", adjudicate_h12),
)
