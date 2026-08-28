"""Assemble ckptgrid frames and persist G1–G9 claim artifacts.

Teacher / student frames follow ``campaign.ckptgrid_metrics`` schemas. Claim
IDs must be exactly `ckptgrid_plan.CLAIM_IDS` or this module raises.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats

from subliminal_learning.campaign import breadth_plan
from subliminal_learning.campaign import ckptgrid_metrics as cm
from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.ckptgrid_pairs import cell_of
from subliminal_learning.campaign.ckptgrid_selection import (
    AdmissionDisposition,
    screen_admission,
)
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_preference_results,
    load_gkd_probe_trajectories,
    subliminal_delta,
)
from subliminal_learning.evaluation.sdf_results_io import load_sdf_preference_results
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    spec_from_base_name,
)
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_CLAIM_STATUSES = frozenset(
    {"supported", "not_supported", "descriptive", "insufficient_data"}
)
BREADTH4_RUN_SET_ID = breadth_plan.RUN_SET_ID


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def campaign_base(spec) -> str:
    """Map a resolved spec onto a campaign label (never weight-seed)."""
    if spec.variant == "weight":
        raise ValueError(f"weight-seed spec is excluded by the seed audit: {spec!r}")
    if spec.variant == "full" and spec.seed == 0:
        return plan.GRID_BASE
    if spec.variant == "full":
        return f"seed{spec.seed}"
    return f"{spec.variant}-seed{spec.seed}"


def spec_of_token(token: str):
    """Parse a disk dir name or a campaign / spec_label token."""
    text = str(token)
    if text.startswith("pythia-"):
        return spec_from_base_name(text)
    return parse_base_models(text)[0]


def step_of_spec(spec) -> int | None:
    """Integer rung from ``stepN``, or None when unsuffixed."""
    rev = spec.revision
    if rev is None or not str(rev).startswith("step"):
        return None
    return int(str(rev).removeprefix("step"))


def assemble_teacher_frame(
    output_root: Path,
    *,
    teacher_run_set_id: str = plan.TEACHER_RUN_SET_ID,
    animals: Sequence[str] = plan.ANIMALS,
) -> pd.DataFrame:
    """SDF preference summaries → teacher-frame schema (incl. controls)."""
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, teacher_run_set_id)
    pref = load_sdf_preference_results(output_root, subdir, animals=list(animals))
    if pref.empty:
        return pd.DataFrame(
            columns=[
                "base",
                "step",
                "ftseed",
                "set",
                "scoring",
                "p_target",
                "is_control",
            ]
        )
    rows: list[dict] = []
    for _, row in pref.iterrows():
        model = str(row["model"])
        spec = spec_of_token(model)
        if spec.variant == "weight":
            raise ValueError(f"weight-seed teacher model on disk: {model!r}")
        step = step_of_spec(spec)
        if step is None:
            raise ValueError(f"teacher model {model!r} is missing an @step revision")
        base = campaign_base(spec)
        shared = {
            "base": base,
            "step": step,
            "ftseed": None,
            "set": row.get("set"),
            "scoring": row.get("scoring"),
        }
        rows.append({**shared, "p_target": row.get("p_target"), "is_control": False})
        if row.get("base_p_target") is not None:
            rows.append(
                {
                    **shared,
                    "p_target": row.get("base_p_target"),
                    "is_control": True,
                }
            )
    return pd.DataFrame(rows)


def assemble_student_frame(
    output_root: Path,
    campaign_subdir: str,
    *,
    animals: Sequence[str] = plan.ANIMALS,
) -> pd.DataFrame:
    """GKD preference summaries → student-frame schema."""
    pref = load_gkd_preference_results(
        output_root, campaign_subdir, animals=list(animals)
    )
    if pref.empty:
        return pd.DataFrame(
            columns=[
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
                "is_control",
            ]
        )
    rows: list[dict] = []
    for _, row in pref.iterrows():
        spec = spec_of_token(str(row["model"]))
        if spec.variant == "weight":
            raise ValueError(f"weight-seed student model on disk: {row['model']!r}")
        student_base = campaign_base(spec)
        student_step = step_of_spec(spec)
        if student_step is None:
            raise ValueError(
                f"student model {row['model']!r} is missing an @step revision"
            )
        is_control = bool(row.get("is_control"))
        teacher_base: str | None = None
        teacher_step: int | None = None
        cell: str | None = None
        if not is_control:
            tlabel = row.get("teacher_base_label")
            if tlabel is None or (isinstance(tlabel, float) and math.isnan(tlabel)):
                teacher_base = student_base
                teacher_step = student_step
            else:
                tspec = spec_of_token(str(tlabel))
                if tspec.variant == "weight":
                    raise ValueError(f"weight-seed teacher_base_label: {tlabel!r}")
                teacher_base = campaign_base(tspec)
                teacher_step = step_of_spec(tspec)
                if teacher_step is None:
                    teacher_step = student_step
            cell = str(cell_of(student_base, teacher_base))
        both_original = (
            student_base == plan.GRID_BASE and teacher_base == plan.GRID_BASE
        )
        arm = "grid" if both_original else "joint"
        if is_control:
            arm = "grid" if student_base == plan.GRID_BASE else "joint"
        rows.append(
            {
                "arm": arm,
                "student_base": student_base,
                "teacher_base": teacher_base,
                "student_step": student_step,
                "teacher_step": teacher_step,
                "cell": cell,
                "ftseed": row.get("teacher_ftseed"),
                "gseed": None,
                "set": row.get("set"),
                "scoring": row.get("scoring"),
                "p_target": row.get("p_target"),
                "is_control": is_control,
            }
        )
    return pd.DataFrame(rows)


def _ci_excludes_zero(lo: object, hi: object) -> bool:
    return _finite(lo) and _finite(hi) and not (float(lo) <= 0.0 <= float(hi))


def _bootstrap_mean(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(rng.choice(vals, size=vals.size, replace=True).mean())
    return (
        float(vals.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _bootstrap_stat(
    statistic,
    n: int,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        draws[i] = float(statistic(rng.choice(idx, size=n, replace=True)))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _ols_log10_step(
    teacher_p: np.ndarray, steps: np.ndarray, delta: np.ndarray
) -> float:
    x1 = np.asarray(teacher_p, dtype=float)
    x2 = np.log10(np.clip(np.asarray(steps, dtype=float), 1.0, None))
    y = np.asarray(delta, dtype=float)
    finite = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")
    design = np.column_stack([np.ones(int(finite.sum())), x1[finite], x2[finite]])
    beta, *_ = np.linalg.lstsq(design, y[finite], rcond=None)
    return float(beta[2])


def _std_log_slope(steps: np.ndarray, y: np.ndarray) -> float:
    logs = np.log(np.clip(np.asarray(steps, dtype=float), 1.0, None))
    vals = np.asarray(y, dtype=float)
    finite = np.isfinite(logs) & np.isfinite(vals)
    if int(finite.sum()) < 3:
        return float("nan")
    z = logs[finite]
    sd = float(z.std(ddof=1))
    if not math.isfinite(sd) or sd == 0.0:
        return float("nan")
    z = (z - z.mean()) / sd
    design = np.column_stack([np.ones(z.size), z])
    beta, *_ = np.linalg.lstsq(design, vals[finite], rcond=None)
    return float(beta[1])


def _claim(
    claim_id: str,
    operationalization: str,
    *,
    status: str,
    statistic: Mapping | None = None,
    threshold: Mapping | None = None,
    reason: str | None = None,
) -> dict:
    if status not in _CLAIM_STATUSES:
        raise ValueError(f"invalid claim status {status!r} for {claim_id}")
    return {
        "id": claim_id,
        "operationalization": operationalization,
        "statistic": dict(statistic or {}),
        "threshold": dict(threshold or {}),
        "status": status,
        "reason": reason,
    }


def _g1_argmax_step(curve: pd.DataFrame) -> int | None:
    if curve.empty or "argmax_step" not in curve.columns:
        return None
    raw = curve["argmax_step"].iloc[0]
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    return int(raw)


def adjudicate_claims(
    teacher_df: pd.DataFrame,
    student_df: pd.DataFrame,
    *,
    teacher_curve: pd.DataFrame,
    delta: pd.DataFrame,
    marginals: pd.DataFrame,
    efficiency: pd.DataFrame,
    asymmetry: pd.DataFrame,
    bracketing: pd.DataFrame,
    onset: pd.DataFrame,
    penalty: pd.DataFrame,
    breadth4_df: pd.DataFrame | None,
    bootstrap_n: int,
    seed: int,
) -> list[dict]:
    """One claim record per G1–G9 (prereg §6)."""
    rungs = list(plan.TEACHER_RUNGS)
    argmax = _g1_argmax_step(teacher_curve)

    claims = [
        _adjudicate_g1(teacher_df, teacher_curve, rungs, bootstrap_n, seed),
        _adjudicate_g2(marginals, rungs, bootstrap_n, seed),
        _adjudicate_g3(marginals, efficiency, bootstrap_n, seed),
        _adjudicate_g4(delta, efficiency, rungs, bootstrap_n, seed),
        _adjudicate_g5(asymmetry, bootstrap_n, seed),
        _adjudicate_g6(onset, argmax),
        _adjudicate_g7(bracketing),
        _adjudicate_g8(penalty),
        _adjudicate_g9(delta, breadth4_df),
    ]
    ids = [c["id"] for c in claims]
    if ids != list(plan.CLAIM_IDS):
        raise ValueError(f"claims.json would carry {ids}, not {list(plan.CLAIM_IDS)}")
    return claims


def _adjudicate_g1(
    teacher_df: pd.DataFrame,
    curve: pd.DataFrame,
    rungs: Sequence[int],
    n_boot: int,
    seed: int,
) -> dict:
    op = (
        "interior argmax AND span >= 0.05, with peak−endpoint bootstrap CIs "
        "excluding 0 at both ends"
    )
    thresh = {
        "min_span": plan.TEACHER_GATE_MIN_CURVE_SPAN,
        "ci": 0.95,
    }
    if curve.empty:
        return _claim(
            "G1",
            op,
            status="insufficient_data",
            threshold=thresh,
            reason="empty teacher curve",
        )
    span = float(curve["span"].iloc[0])
    interior = bool(curve["argmax_interior"].iloc[0])
    peak = _g1_argmax_step(curve)
    first, last = int(rungs[0]), int(rungs[-1])
    teachers = teacher_df
    if "is_control" in teachers.columns:
        teachers = teachers.loc[~teachers["is_control"].astype(bool)]
    if "base" in teachers.columns:
        teachers = teachers[teachers["base"] == plan.GRID_BASE]
    peak_vals = (
        teachers[teachers["step"].astype(int) == peak]["p_target"].to_numpy(dtype=float)
        if peak is not None
        else np.array([])
    )
    first_vals = teachers[teachers["step"].astype(int) == first]["p_target"].to_numpy(
        dtype=float
    )
    last_vals = teachers[teachers["step"].astype(int) == last]["p_target"].to_numpy(
        dtype=float
    )
    lo_f = hi_f = lo_l = hi_l = float("nan")
    if peak_vals.size and first_vals.size:
        try:
            boot = cm._bootstrap_mean_diff(
                peak_vals, first_vals, n_boot=n_boot, seed=seed
            )
            lo_f, hi_f = boot.ci_low, boot.ci_high
        except ValueError:
            pass
    if peak_vals.size and last_vals.size:
        try:
            boot = cm._bootstrap_mean_diff(
                peak_vals, last_vals, n_boot=n_boot, seed=seed + 1
            )
            lo_l, hi_l = boot.ci_low, boot.ci_high
        except ValueError:
            pass
    stat = {
        "argmax_step": peak,
        "argmax_interior": interior,
        "span": span,
        "peak_minus_first_ci": [lo_f, hi_f],
        "peak_minus_last_ci": [lo_l, hi_l],
    }
    if peak is None or not math.isfinite(span):
        return _claim(
            "G1",
            op,
            status="insufficient_data",
            statistic=stat,
            threshold=thresh,
            reason="no finite teacher curve",
        )
    ok = (
        interior
        and span >= plan.TEACHER_GATE_MIN_CURVE_SPAN
        and _ci_excludes_zero(lo_f, hi_f)
        and _ci_excludes_zero(lo_l, hi_l)
    )
    return _claim(
        "G1",
        op,
        status="supported" if ok else "not_supported",
        statistic=stat,
        threshold=thresh,
    )


def _adjudicate_g2(
    marginals: pd.DataFrame,
    rungs: Sequence[int],
    n_boot: int,
    seed: int,
) -> dict:
    op = (
        "Spearman(delta, log student step) > 0 with CI excluding 0 AND no "
        "interior peak whose peak−final CI excludes 0"
    )
    slice_ = marginals
    if not slice_.empty and "slice" in slice_.columns:
        slice_ = slice_[slice_["slice"] == cm.SLICE_STUDENT]
    if not slice_.empty and "student_base" in slice_.columns:
        slice_ = slice_[
            (slice_["student_base"] == plan.GRID_BASE)
            & (slice_["teacher_base"] == plan.GRID_BASE)
        ]
    if slice_.empty or "delta" not in slice_.columns:
        return _claim("G2", op, status="insufficient_data")
    steps = slice_["student_step"].to_numpy(dtype=float)
    deltas = slice_["delta"].to_numpy(dtype=float)
    rho = cm._spearman_log_step(steps, deltas)

    def stat_fn(idx: np.ndarray) -> float:
        return cm._spearman_log_step(steps[idx], deltas[idx])

    lo = hi = float("nan")
    if steps.size >= 2:
        lo, hi = _bootstrap_stat(stat_fn, steps.size, n_boot=n_boot, seed=seed)
    order = np.argsort(steps)
    peak_i = int(np.nanargmax(deltas[order])) if deltas.size else -1
    peak_step = int(steps[order][peak_i]) if peak_i >= 0 else None
    first, last = int(np.nanmin(steps)), int(np.nanmax(steps))
    interior_peak = peak_step is not None and peak_step not in (first, last)
    peak_final_excl = False
    if interior_peak:
        peak_d = deltas[steps == peak_step]
        final_d = deltas[steps == last]
        if peak_d.size and final_d.size:
            try:
                boot = cm._bootstrap_mean_diff(
                    peak_d, final_d, n_boot=n_boot, seed=seed
                )
                peak_final_excl = _ci_excludes_zero(boot.ci_low, boot.ci_high)
            except ValueError:
                peak_final_excl = False
    stat = {
        "spearman": rho,
        "spearman_ci": [lo, hi],
        "peak_step": peak_step,
        "interior_peak": interior_peak,
        "peak_final_ci_excludes_0": peak_final_excl,
    }
    if peak_final_excl and interior_peak:
        return _claim(
            "G2",
            op,
            status="not_supported",
            statistic=stat,
            reason="interior_peak_alternative",
        )
    monotone = (
        _finite(rho) and rho > 0.0 and _ci_excludes_zero(lo, hi) and not peak_final_excl
    )
    return _claim(
        "G2",
        op,
        status="supported" if monotone else "not_supported",
        statistic=stat,
    )


def _adjudicate_g3(
    marginals: pd.DataFrame,
    efficiency: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict:
    op = (
        "Spearman(delta, teacher p_target) CI excludes 0; efficiency "
        "TOST bound 0.05 on the standardized log-step slope"
    )
    slice_ = marginals
    if not slice_.empty and "slice" in slice_.columns:
        slice_ = slice_[slice_["slice"] == cm.SLICE_TEACHER]
    joined = efficiency
    if slice_.empty or joined.empty:
        return _claim("G3", op, status="insufficient_data")
    if "teacher_p_target" not in joined.columns:
        return _claim("G3", op, status="insufficient_data")
    dose = joined.dropna(subset=["delta", "teacher_p_target"])
    rho = float("nan")
    lo = hi = float("nan")
    if len(dose) >= 2:
        x = dose["teacher_p_target"].to_numpy(dtype=float)
        y = dose["delta"].to_numpy(dtype=float)
        if not (np.allclose(x, x[0]) or np.allclose(y, y[0])):
            rho = float(scipy_stats.spearmanr(x, y).statistic)

            def stat_fn(idx: np.ndarray) -> float:
                xx, yy = x[idx], y[idx]
                if np.allclose(xx, xx[0]) or np.allclose(yy, yy[0]):
                    return float("nan")
                return float(scipy_stats.spearmanr(xx, yy).statistic)

            lo, hi = _bootstrap_stat(stat_fn, len(dose), n_boot=n_boot, seed=seed)
    eff = joined.dropna(subset=["efficiency", "teacher_step"])
    slope = float("nan")
    slo = shi = float("nan")
    if len(eff) >= 3:
        steps = eff["teacher_step"].to_numpy(dtype=float)
        vals = eff["efficiency"].to_numpy(dtype=float)
        slope = _std_log_slope(steps, vals)

        def slope_fn(idx: np.ndarray) -> float:
            return _std_log_slope(steps[idx], vals[idx])

        slo, shi = _bootstrap_stat(slope_fn, len(eff), n_boot=n_boot, seed=seed + 3)
    tost = (
        _finite(slo)
        and _finite(shi)
        and float(slo) >= -plan.EFFICIENCY_TOST_BOUND
        and float(shi) <= plan.EFFICIENCY_TOST_BOUND
    )
    dose_ok = _ci_excludes_zero(lo, hi)
    stat = {
        "spearman": rho,
        "spearman_ci": [lo, hi],
        "std_log_step_slope": slope,
        "std_log_step_slope_ci": [slo, shi],
        "efficiency_flat": tost,
    }
    if not _finite(rho) and not _finite(slope):
        return _claim(
            "G3",
            op,
            status="insufficient_data",
            statistic=stat,
            threshold={"efficiency_tost_bound": plan.EFFICIENCY_TOST_BOUND},
        )
    return _claim(
        "G3",
        op,
        status="supported" if (dose_ok or tost) else "not_supported",
        statistic=stat,
        threshold={"efficiency_tost_bound": plan.EFFICIENCY_TOST_BOUND},
    )


def _adjudicate_g4(
    delta: pd.DataFrame,
    efficiency: pd.DataFrame,
    rungs: Sequence[int],
    n_boot: int,
    seed: int,
) -> dict:
    op = (
        "On the 12 diagonal cells, delta ~ teacher_p_target + log10(step); "
        "log-step coefficient bootstrap CI excludes 0"
    )
    rows = delta
    if not rows.empty and "level" in rows.columns:
        rows = rows[rows["level"].astype(str) == "agg"]
    if rows.empty or efficiency.empty:
        return _claim("G4", op, status="insufficient_data")
    wanted = {int(r) for r in rungs}
    diag = rows[
        (rows["student_step"] == rows["teacher_step"])
        & (rows["student_step"].astype(int).isin(wanted))
        & (rows["student_base"] == plan.GRID_BASE)
        & (rows["teacher_base"] == plan.GRID_BASE)
    ]
    merged = diag.merge(
        efficiency[
            [
                c
                for c in (
                    "student_step",
                    "teacher_step",
                    "student_base",
                    "teacher_base",
                    "teacher_p_target",
                )
                if c in efficiency.columns
            ]
        ],
        on=[
            c
            for c in (
                "student_step",
                "teacher_step",
                "student_base",
                "teacher_base",
            )
            if c in diag.columns and c in efficiency.columns
        ],
        how="left",
    )
    merged = merged.dropna(subset=["delta", "teacher_p_target", "student_step"])
    if len(merged) < 3:
        return _claim("G4", op, status="insufficient_data")
    tp = merged["teacher_p_target"].to_numpy(dtype=float)
    steps = merged["student_step"].to_numpy(dtype=float)
    y = merged["delta"].to_numpy(dtype=float)
    coef = _ols_log10_step(tp, steps, y)

    def stat_fn(idx: np.ndarray) -> float:
        return _ols_log10_step(tp[idx], steps[idx], y[idx])

    lo, hi = _bootstrap_stat(stat_fn, len(merged), n_boot=n_boot, seed=seed)
    ok = _ci_excludes_zero(lo, hi)
    return _claim(
        "G4",
        op,
        status="supported" if ok else "not_supported",
        statistic={"log10_step_coef": coef, "ci": [lo, hi], "n": len(merged)},
    )


def _adjudicate_g5(asymmetry: pd.DataFrame, n_boot: int, seed: int) -> dict:
    op = "mean(asym) > 0 with CI excluding 0"
    if asymmetry.empty or "asymmetry" not in asymmetry.columns:
        return _claim("G5", op, status="insufficient_data")
    vals = asymmetry["asymmetry"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return _claim("G5", op, status="insufficient_data")
    mean, lo, hi = _bootstrap_mean(vals, n_boot=n_boot, seed=seed)
    ok = mean > 0.0 and _ci_excludes_zero(lo, hi)
    return _claim(
        "G5",
        op,
        status="supported" if ok else "not_supported",
        statistic={"mean_asym": mean, "ci": [lo, hi], "n": int(vals.size)},
    )


def _adjudicate_g6(onset: pd.DataFrame, strong_teacher: int | None) -> dict:
    op = (
        "first student rung whose delta clears control mean + "
        f"{plan.ONSET_SD_MULTIPLE:g}·SD for two consecutive rungs under the "
        "G1-strong teacher"
    )
    if onset.empty or strong_teacher is None:
        return _claim("G6", op, status="insufficient_data")
    rows = onset
    if "teacher_step" in rows.columns:
        rows = rows[rows["teacher_step"].astype(int) == int(strong_teacher)]
    if rows.empty:
        return _claim(
            "G6",
            op,
            status="insufficient_data",
            reason="no onset row under G1-strong teacher",
        )
    onset_step = rows["onset_step"].iloc[0]
    censored = bool(rows["censored"].iloc[0])
    present = onset_step is not None and not (
        isinstance(onset_step, float) and math.isnan(onset_step)
    )
    return _claim(
        "G6",
        op,
        status="supported" if present and not censored else "not_supported",
        statistic={
            "onset_step": (int(onset_step) if present else None),
            "censored": censored,
            "strong_teacher_step": int(strong_teacher),
        },
        threshold={
            "sd_multiple": plan.ONSET_SD_MULTIPLE,
            "consecutive": 2,
            "markers": list(plan.ONSET_MARKERS),
        },
        reason=None if present else "no_onset",
    )


def _adjudicate_g7(bracketing: pd.DataFrame) -> dict:
    op = (
        "per joint rung, gap = mean delta(SI_DS original↔data-seed1) − "
        "mean delta(DI_DS original↔seedN); descriptive, never single-factor"
    )
    if bracketing.empty:
        return _claim("G7", op, status="insufficient_data")
    primary = bracketing
    if "primary" in primary.columns:
        primary = primary[primary["primary"].astype(bool)]
    return _claim(
        "G7",
        op,
        status="descriptive",
        statistic={
            "n_primary_rungs": int(len(primary)),
            "estimates": (
                primary["estimate"].tolist() if "estimate" in primary.columns else []
            ),
        },
    )


def _adjudicate_g8(penalty: pd.DataFrame) -> dict:
    op = "Spearman(penalty, log step) < 0 with CI excluding 0"
    if penalty.empty or "spearman" not in penalty.columns:
        return _claim("G8", op, status="insufficient_data")
    rho = float(penalty["spearman"].iloc[0])
    lo = float(penalty["spearman_ci_low"].iloc[0])
    hi = float(penalty["spearman_ci_high"].iloc[0])
    if not math.isfinite(rho):
        return _claim("G8", op, status="insufficient_data")
    ok = rho < 0.0 and _ci_excludes_zero(lo, hi)
    return _claim(
        "G8",
        op,
        status="supported" if ok else "not_supported",
        statistic={"spearman": rho, "ci": [lo, hi]},
    )


def _adjudicate_g9(delta: pd.DataFrame, breadth4_df: pd.DataFrame | None) -> dict:
    op = (
        "Grid diagonal @ 143000 vs breadth4 bear cells (external join); "
        "sign agreement; no numeric threshold"
    )
    rows = delta
    if rows is None or rows.empty:
        return _claim("G9", op, status="insufficient_data")
    if "level" in rows.columns:
        rows = rows[rows["level"].astype(str) == "agg"]
    grid = rows[
        (rows["student_base"] == plan.GRID_BASE)
        & (rows["teacher_base"] == plan.GRID_BASE)
        & (rows["student_step"].astype(int) == plan.CHECKPOINT_MAX_STEP)
        & (rows["teacher_step"].astype(int) == plan.CHECKPOINT_MAX_STEP)
    ]
    if grid.empty or breadth4_df is None or breadth4_df.empty:
        return _claim("G9", op, status="insufficient_data")
    grid_delta = float(grid["delta"].mean())
    bcol = (
        "delta"
        if "delta" in breadth4_df.columns
        else ("subliminal_delta" if "subliminal_delta" in breadth4_df.columns else None)
    )
    if bcol is None:
        return _claim("G9", op, status="insufficient_data")
    b4 = float(pd.to_numeric(breadth4_df[bcol], errors="coerce").mean())
    if not (math.isfinite(grid_delta) and math.isfinite(b4)):
        return _claim("G9", op, status="insufficient_data")
    agree = math.copysign(1.0, grid_delta) == math.copysign(1.0, b4)
    return _claim(
        "G9",
        op,
        status="descriptive",
        statistic={
            "grid_delta_143000": grid_delta,
            "breadth4_delta": b4,
            "sign_agreement": agree,
            "breadth4_run_set_id": BREADTH4_RUN_SET_ID,
        },
    )


def load_breadth4_bear(
    output_root: Path,
    *,
    run_set_id: str = BREADTH4_RUN_SET_ID,
) -> pd.DataFrame:
    """Breadth4 bear same-init deltas, or empty when the run set is absent."""
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
    root = Path(output_root) / subdir
    if not root.is_dir():
        logger.warning(
            "no breadth4 run set at {} — G9 reads as insufficient_data",
            subdir,
        )
        return pd.DataFrame()
    pref = load_gkd_preference_results(output_root, subdir, animals=list(plan.ANIMALS))
    if pref.empty:
        return pd.DataFrame()
    try:
        return subliminal_delta(pref)
    except (ValueError, KeyError) as err:
        logger.warning("breadth4 delta join failed: {}", err)
        return pd.DataFrame()


def quarantine_frame(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
    student_df: pd.DataFrame,
    *,
    ppl_max: float = plan.QUARANTINE_PPL_RATIO_MAX,
    filter_drop_max: float = plan.QUARANTINE_FILTER_DROP_MAX,
) -> pd.DataFrame:
    """Teacher-admission tiers plus leaf ppl-ratio / filter-drop bounds."""
    rows: list[dict] = []
    for disposition in dispositions.values():
        if disposition.tier == "ok":
            continue
        rows.append(
            {
                "scope": "teacher_admission",
                "step": disposition.rung,
                "base": disposition.base,
                "cell": None,
                "tier": disposition.tier,
                "reason": ";".join(disposition.reasons),
                "value": disposition.teacher_p_target,
                "threshold": plan.ADMISSION_ABS_MIN,
            }
        )
    if student_df is not None and not student_df.empty:
        for column, threshold in (
            ("perplexity_ratio", ppl_max),
            ("filter_drop_frac", filter_drop_max),
        ):
            if column not in student_df.columns:
                continue
            bad = student_df[
                pd.to_numeric(student_df[column], errors="coerce") > threshold
            ]
            for _, row in bad.iterrows():
                rows.append(
                    {
                        "scope": column,
                        "step": row.get("student_step"),
                        "base": row.get("student_base"),
                        "cell": row.get("cell"),
                        "tier": "quarantine",
                        "reason": f"{column}>{threshold:g}",
                        "value": row.get(column),
                        "threshold": threshold,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "scope",
                "step",
                "base",
                "cell",
                "tier",
                "reason",
                "value",
                "threshold",
            ]
        )
    return pd.DataFrame(rows)


def _attach_optional(
    student_df: pd.DataFrame,
    extra: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Left-join optional capability / probe columns when present."""
    if extra is None or extra.empty:
        return student_df
    keep = [c for c in columns if c in extra.columns]
    keys = [
        c
        for c in ("animal", "model", "teacher_tag", "gkd_tag")
        if c in extra.columns and c in student_df.columns
    ]
    if not keep or not keys:
        return student_df
    return student_df.merge(
        extra[list(dict.fromkeys([*keys, *keep]))], on=keys, how="left"
    )


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_results(
    output_root: Path,
    campaign_subdir: str,
    *,
    dry_run: bool = False,
    teacher_df: pd.DataFrame | None = None,
    student_df: pd.DataFrame | None = None,
    breadth4_df: pd.DataFrame | None = None,
    bootstrap_n: int = plan.BOOTSTRAP_N,
    bootstrap_seed: int = plan.BOOTSTRAP_SEED,
) -> list[Path]:
    """Write every ckptgrid results CSV plus `claims.json`.

    Planted frames (`teacher_df` / `student_df` / `breadth4_df`) skip disk
    discovery so unit tests do not need a full eval tree.

    Raises:
        ValueError: `claims.json` would omit a preregistered claim id.
    """
    results = Path(output_root) / campaign_subdir / "results"
    teacher = (
        teacher_df if teacher_df is not None else assemble_teacher_frame(output_root)
    )
    student = (
        student_df
        if student_df is not None
        else assemble_student_frame(output_root, campaign_subdir)
    )
    if student_df is None:
        try:
            cap = load_gkd_capability_results(output_root, campaign_subdir)
            student = _attach_optional(student, cap, ("perplexity_ratio",))
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("capability join skipped: {}", err)
        try:
            probes = load_gkd_probe_trajectories(output_root, campaign_subdir)
            student = _attach_optional(student, probes, ("filter_drop_frac",))
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("probe join skipped: {}", err)

    curve = cm.build_teacher_curve(
        teacher, rungs=plan.TEACHER_RUNGS, base=plan.GRID_BASE
    )
    try:
        delta = cm.build_grid_delta(student)
    except ValueError as err:
        logger.warning("grid delta incomplete: {}", err)
        delta = pd.DataFrame(columns=list(cm.GRID_DELTA_COLUMNS))
    strong = _g1_argmax_step(curve) or plan.CHECKPOINT_MAX_STEP
    marginals = cm.build_marginals(
        delta,
        teacher_rungs=plan.TEACHER_RUNGS,
        student_rungs=plan.STUDENT_RUNGS,
        fixed_student_step=plan.CHECKPOINT_MAX_STEP,
        fixed_teacher_step=int(strong),
    )
    efficiency = cm.build_efficiency(delta, curve)
    asymmetry = cm.build_asymmetry(delta)
    bracketing = cm.build_bracketing(
        delta,
        joint_rungs=plan.JOINT_RUNGS,
        bootstrap_n=bootstrap_n,
        seed=bootstrap_seed,
    )
    onset = cm.build_onset(
        delta,
        student,
        student_rungs=plan.STUDENT_RUNGS,
        sd_multiple=plan.ONSET_SD_MULTIPLE,
        consecutive=2,
        markers=plan.ONSET_MARKERS,
        max_step=plan.CHECKPOINT_MAX_STEP,
    )
    if not onset.empty and strong is not None and "teacher_step" in onset.columns:
        under = onset[onset["teacher_step"].astype(int) == int(strong)]
        onset = under if not under.empty else onset
    penalty = cm.build_penalty_trend(
        delta,
        joint_rungs=plan.JOINT_RUNGS,
        bootstrap_n=bootstrap_n,
        seed=bootstrap_seed,
    )
    dispositions = screen_admission(teacher) if not teacher.empty else {}
    quarantine = quarantine_frame(dispositions, student)
    b4 = breadth4_df if breadth4_df is not None else load_breadth4_bear(output_root)
    claims = adjudicate_claims(
        teacher,
        student,
        teacher_curve=curve,
        delta=delta,
        marginals=marginals,
        efficiency=efficiency,
        asymmetry=asymmetry,
        bracketing=bracketing,
        onset=onset,
        penalty=penalty,
        breadth4_df=b4,
        bootstrap_n=bootstrap_n,
        seed=bootstrap_seed,
    )
    tables = {
        plan.RESULTS_TEACHER_CURVE_CSV: curve,
        plan.RESULTS_GRID_DELTA_CSV: delta,
        plan.RESULTS_GRID_MARGINALS_CSV: marginals,
        plan.RESULTS_GRID_EFFICIENCY_CSV: efficiency,
        plan.RESULTS_GRID_ASYMMETRY_CSV: asymmetry,
        plan.RESULTS_JOINT_BRACKETING_CSV: bracketing,
        plan.RESULTS_GRID_ONSET_CSV: onset,
        plan.RESULTS_GRID_QUARANTINE_CSV: quarantine,
    }
    written = [
        _write_csv(results / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    ]
    payload = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "breadth4_run_set_id": BREADTH4_RUN_SET_ID,
        "primary_endpoint": {
            "prompt_set": plan.PRIMARY_PROMPT_SET,
            "scoring": plan.PRIMARY_SCORING,
        },
        "strong_teacher_step": strong,
        "claims": claims,
    }
    claims_path = results / plan.RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        atomic_write_json(claims_path, payload)
    written.append(claims_path)
    logger.info("ckptgrid results: {} artifacts under {}", len(written), results)
    return written
