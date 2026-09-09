"""Pure frames for the off-policy cross-size claims O0-O5.

Leaf-frame schema (`LEAF_COLUMNS`): ``animal``, ``student``, ``teacher``, ``kind``
(``treatment`` / ``base_control`` / ``rng_control``), ``phase`` (``p2`` / ``p3`` /
``p3b`` / ``p3c``; None on controls), ``recipe_key``, ``epochs``, ``lr``, ``lora_r``,
``lora_alpha``, ``rung_tag``, ``ftseed``, ``gseed``, ``p_target``,
``storytelling_p_target``, ``perplexity_ratio``, ``dataset_tag``, ``quarantined``.
Sizes are tokens (``70m``); ``recipe_key`` is the student ``{param_tag}`` and
``rung_tag`` the teacher ``{param_tag}``.

A ``base_control`` row's ``teacher`` names the untuned base that GENERATED its
numbers: ``teacher == student`` is the student-base control (Cloud's control, the
anchor denominator), a different size is the teacher-base control a cross cell is
read against. An ``rng_control`` row has ``teacher == student``.

``ppl_coupling`` residualises Delta to the no-degradation point:
``Delta_adj = Delta - beta*(ppl - 1.0)`` with OLS slope beta of ``delta_primary``
on ``perplexity_ratio``; ``retained = |mean Delta_adj| / |mean Delta|``.

`analysis.stats` / `analysis.decomposition` are imported inside the two functions
that call them: the `analysis` package's ``__init__`` eagerly pulls a ``*_io``
loader, and this module is on the pure-import list (`tests/unit/test_layering.py`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign.offxsize_plan import (
    ANGELA_REFERENCE,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CLAIM_IDS,
    EVAL_CONFIDENCE,
    P3C_SIZE,
    PRIMARY_PROMPT_SET,
    QUARANTINE_PPL_RATIO_MAX,
    SIZES,
    STUDENT_CENTRE,
    TEACHER_RUNGS,
    X0_MIN_DELTA,
    X2_ADDITIVE_R2_MIN,
    X2_ASYM_MIN,
    X4_RETAINED_FRACTION_MIN,
)

if TYPE_CHECKING:
    from subliminal_learning.analysis.decomposition import (
        AdditiveFit,
        ReciprocalContrast,
    )

LEAF_COLUMNS: tuple[str, ...] = (
    "animal",
    "student",
    "teacher",
    "kind",
    "phase",
    "recipe_key",
    "epochs",
    "lr",
    "lora_r",
    "lora_alpha",
    "rung_tag",
    "ftseed",
    "gseed",
    "p_target",
    "storytelling_p_target",
    "perplexity_ratio",
    "dataset_tag",
    "quarantined",
)

KIND_TREATMENT = pairs.TREATMENT
KIND_BASE = pairs.BASE_CONTROL
KIND_RNG = pairs.RNG_CONTROL

DELTA_COLUMNS: tuple[str, ...] = (
    *LEAF_COLUMNS,
    "arm",
    "shape_steps",
    "direction",
    "delta_sb",
    "delta_tb",
    "delta_rng",
    "delta_primary",
    "sb_control_mean",
    "tb_control_mean",
    "rng_control_mean",
)

CELL_COLUMNS: tuple[str, ...] = (
    "animal",
    "teacher",
    "student",
    "recipe_key",
    "rung_tag",
    "arm",
    "shape_steps",
    "direction",
    "n",
    "mean_delta_primary",
    "mean_delta_sb",
    "mean_delta_tb",
    "mean_delta_rng",
    "ci_low",
    "ci_high",
    "mean_p_target",
    "mean_storytelling_p_target",
    "mean_ppl_ratio",
)

PENALTY_COLUMNS: tuple[str, ...] = (
    "teacher",
    "student",
    "recipe_key",
    "shape_steps",
    "direction",
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
)

ASYMMETRY_COLUMNS: tuple[str, ...] = (
    "size_lo",
    "size_hi",
    "iota",
    "iota_dose",
    "ci_low",
    "ci_high",
    "delta_up",
    "delta_down",
)

RECIPE_COLUMNS: tuple[str, ...] = (
    "recipe_key",
    "teacher",
    "student",
    "phase",
    "n",
    "delta_tb",
    "sign",
    "kendall_tau",
    "picked_recipe",
)

DOSE_COLUMNS: tuple[str, ...] = (
    "rung_tag",
    "epochs",
    "family",
    "recipe_key",
    "n",
    "mean_delta_sb",
    "ci_low",
    "ci_high",
    "teacher_p_target",
    "off_topic",
    "pool_keep_rate",
)

PPL_COLUMNS: tuple[str, ...] = (
    "teacher",
    "student",
    "recipe_key",
    "rung_tag",
    "beta",
    "mean_delta",
    "mean_delta_adj",
    "retained_fraction",
    "n",
)

CONCORDANCE_COLUMNS: tuple[str, ...] = (
    "cell",
    "recipe_key",
    "rung_tag",
    "n",
    "p_target",
    "delta_sb",
    "ref_p_target_seed1",
    "abs_gap_seed1",
    "rel_gap_seed1",
    "ref_p_target_3seed",
    "abs_gap_3seed",
    "rel_gap_3seed",
    "ref_rng_floor_3seed",
    "ref_teacher_tag",
    "ref_student_tag",
)

SB_KEYS: list[str] = ["animal", "student", "recipe_key", "gseed"]
TB_KEYS: list[str] = ["animal", "student", "teacher", "recipe_key", "gseed"]


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _pair_meta(teacher: str, student: str) -> tuple[str, int, str]:
    """``(arm, shape_steps, direction)`` of one ordered size pair.

    Raises:
        ValueError: Either token is not a campaign size.
    """
    leaf = pairs.Leaf(
        phase="",
        kind=KIND_TREATMENT,
        teacher=str(teacher),
        student=str(student),
        recipe=STUDENT_CENTRE,
        rung_tag=None,
        ftseed=None,
        gseed=0,
    )
    return leaf.arm, leaf.shape_steps, leaf.direction


def _finite_mask(values: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [_finite(v) for v in values],
        dtype=bool,
    )


def _cluster_ci(
    values: Sequence[float],
    ftseed: Sequence[object],
    gseed: Sequence[object],
    *,
    n_boot: int,
    seed: int,
    confidence: float = EVAL_CONFIDENCE,
) -> tuple[float, float, float]:
    """Mean and percentile CI resampling teacher seeds and student seeds jointly.

    A cell's leaves are a teacher-ftseed x student-gseed crossing, so the two-way
    `analysis.stats.cluster_bootstrap` is the matched resampler; it degrades to a
    one-way bootstrap when either dimension has a single level.

    Returns:
        ``(mean, ci_low, ci_high)``; all NaN when no leaf carries a finite value.
    """
    from subliminal_learning.analysis.stats import cluster_bootstrap

    mask = _finite_mask(values)
    vals = np.asarray([float(v) for v in np.asarray(values, dtype=object)[mask]])
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1 or n_boot <= 0:
        return mean, mean, mean
    rungs = [str(v) for v in np.asarray(ftseed, dtype=object)[mask]]
    pairings = [str(v) for v in np.asarray(gseed, dtype=object)[mask]]
    estimate = cluster_bootstrap(
        lambda idx: float(vals[idx].mean()),
        rungs,
        pairings,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    )
    return estimate.estimate, estimate.ci_low, estimate.ci_high


def _mean_diff_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile CI on ``mean(left) - mean(right)``."""
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


def _ratio(numerator: float, denominator: float) -> float:
    if not _finite(numerator) or not _finite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def _live(frame: pd.DataFrame) -> pd.DataFrame:
    if "quarantined" not in frame.columns:
        return frame
    return frame.loc[~frame["quarantined"].fillna(False).astype(bool)]


def _cell_at(
    cells: pd.DataFrame,
    *,
    teacher: str,
    student: str,
    recipe_key: str,
    rung_tag: str | None = None,
    animal: str | None = None,
) -> pd.Series | None:
    """One cell row, or None when the cell (or its teacher rung) is absent.

    A non-None `rung_tag` never falls back to another teacher dose: a cell trained
    from a different rung is a different measurement, not a substitute.
    """
    rows = cells[
        (cells["teacher"].astype(str) == str(teacher))
        & (cells["student"].astype(str) == str(student))
        & (cells["recipe_key"].astype(str) == str(recipe_key))
    ]
    if animal is not None and "animal" in cells.columns:
        rows = rows[rows["animal"].astype(str) == animal]
    if rung_tag is not None and "rung_tag" in cells.columns:
        rows = rows[rows["rung_tag"].astype(str) == str(rung_tag)]
    if rows.empty:
        return None
    return rows.iloc[0]


def quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Flag leaves that breach the PPL quarantine rule (label only, never dropped).

    Returns:
        Copy of `leaves` with ``quarantined`` and ``quarantine_reason``; the reason
        is ``ppl_ratio`` or the empty string.
    """
    out = leaves.copy()
    if out.empty:
        out["quarantined"] = pd.Series(dtype=bool)
        out["quarantine_reason"] = pd.Series(dtype=str)
        return out
    ppl = pd.to_numeric(out.get("perplexity_ratio"), errors="coerce")
    hit = (ppl > QUARANTINE_PPL_RATIO_MAX).fillna(False)
    out["quarantined"] = hit.astype(bool)
    out["quarantine_reason"] = ["ppl_ratio" if bool(f) else "" for f in hit]
    return out


def propagate_control_quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Quarantine treatments whose every matched control is quarantined.

    `quarantine` judges each leaf on its own capability, but a Delta is a
    difference: once the control behind a key is excluded, the treatment over it has
    no denominator left and is not a measurement either. Flagging it here (reason
    ``control_quarantined``) keeps it in `quarantine.csv` with a stated cause and out
    of the cell means, instead of letting `leaf_deltas` raise on a control that is
    present but excluded -- a raise that is meant for a control which never ran.

    Anchors are matched against the student-base control, cross pairs against the
    teacher-base control (whichever one `delta_primary` divides by).

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
    same = out["teacher"].astype(str) == out["student"].astype(str)
    is_base = kinds == KIND_BASE

    def _lost(cols: list[str], controls: pd.Series) -> pd.Series:
        """Keys whose controls of this class all exist but are excluded."""
        key = out[cols].astype(str).agg("|".join, axis=1)
        present = set(key[controls])
        live = set(key[controls & ~flags])
        return key.isin(present - live)

    treatments = kinds == KIND_TREATMENT
    orphaned = (treatments & same & _lost(SB_KEYS, is_base & same)) | (
        treatments & ~same & _lost(TB_KEYS, is_base)
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
    """Treatment-leaf Delta vs its matched base-numbers and rng controls.

    ``delta_sb`` joins `SB_KEYS` against the base control whose ``teacher ==
    student``; ``delta_tb`` joins `TB_KEYS` against the base control generated by
    the teacher's own base (anchors set ``delta_tb := delta_sb``); ``delta_rng``
    joins `SB_KEYS` against the trait-free floor. ``delta_primary`` is ``delta_tb``
    on cross pairs and ``delta_sb`` on anchors.

    The control behind ``delta_primary`` is required. The other two are recorded
    when present and NaN otherwise, because the cell enumeration does not run them
    everywhere: the rng floor exists only in the headline phase, and the P3b tail
    ships a teacher-base control without a student-base one.

    Quarantined leaves are excluded from control means and kept (flagged) on the
    treatment side -- except an already-quarantined treatment whose control is also
    gone, which is dropped rather than raised on (see
    `propagate_control_quarantine`).

    Returns:
        One row per treatment leaf, `DELTA_COLUMNS`.

    Raises:
        KeyError: A **live** treatment's ``delta_primary`` control is missing; the
            exception args carry the missing key. Never falls back to another
            control.
    """
    if leaves.empty:
        return _empty(DELTA_COLUMNS)
    required = {
        "animal",
        "student",
        "teacher",
        "kind",
        "recipe_key",
        "gseed",
        "p_target",
    }
    missing = required - set(leaves.columns)
    if missing:
        raise ValueError(f"leaf frame missing columns {sorted(missing)}")
    frame = leaves.copy()
    if "quarantined" not in frame.columns:
        frame["quarantined"] = False
    treatments = frame[frame["kind"].astype(str) == KIND_TREATMENT].copy()
    if treatments.empty:
        return _empty(DELTA_COLUMNS)
    live = _live(frame)
    base = live[live["kind"].astype(str) == KIND_BASE]
    student_base = base[base["teacher"].astype(str) == base["student"].astype(str)]
    rng = live[live["kind"].astype(str) == KIND_RNG]
    sb_mean = (
        student_base.groupby(SB_KEYS, dropna=False)["p_target"].mean()
        if not student_base.empty
        else pd.Series(dtype=float)
    )
    tb_mean = (
        base.groupby(TB_KEYS, dropna=False)["p_target"].mean()
        if not base.empty
        else pd.Series(dtype=float)
    )
    rng_mean = (
        rng.groupby(SB_KEYS, dropna=False)["p_target"].mean()
        if not rng.empty
        else pd.Series(dtype=float)
    )

    rows: list[dict] = []
    for _, leaf in treatments.iterrows():
        animal = leaf["animal"]
        student = str(leaf["student"])
        teacher = str(leaf["teacher"])
        recipe_key = str(leaf["recipe_key"])
        gseed = int(leaf["gseed"])
        arm, shape_steps, direction = _pair_meta(teacher, student)
        flagged = bool(leaf.get("quarantined", False))
        sb_key = (animal, student, recipe_key, gseed)
        tb_key = (animal, student, teacher, recipe_key, gseed)
        sb_c = _lookup(sb_mean, sb_key)
        p = float(leaf["p_target"])
        if arm == "anchor":
            if sb_c is None:
                if flagged:
                    continue
                raise KeyError(sb_key)
            tb_c: float | None = sb_c
        else:
            tb_c = _lookup(tb_mean, tb_key)
            if tb_c is None:
                if flagged:
                    continue
                raise KeyError(tb_key)
        rng_c = _lookup(rng_mean, sb_key)
        delta_sb = p - sb_c if sb_c is not None else float("nan")
        delta_tb = p - tb_c if tb_c is not None else float("nan")
        rec = {col: leaf[col] if col in leaf.index else None for col in LEAF_COLUMNS}
        rec.update(
            {
                "arm": arm,
                "shape_steps": shape_steps,
                "direction": direction,
                "delta_sb": delta_sb,
                "delta_tb": delta_tb,
                "delta_rng": p - rng_c if rng_c is not None else float("nan"),
                "delta_primary": delta_tb if arm == "cross" else delta_sb,
                "sb_control_mean": sb_c if sb_c is not None else float("nan"),
                "tb_control_mean": tb_c if tb_c is not None else float("nan"),
                "rng_control_mean": rng_c if rng_c is not None else float("nan"),
            }
        )
        rows.append(rec)
    if not rows:
        return _empty(DELTA_COLUMNS)
    return pd.DataFrame(rows, columns=list(DELTA_COLUMNS))


def _lookup(means: pd.Series, key: tuple) -> float | None:
    if means.empty:
        return None
    try:
        value = means.loc[key]
    except KeyError:
        return None
    return float(value) if _finite(value) else None


def cell_means(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-cell means and percentile bootstrap CIs over leaves.

    A cell is ``(animal, teacher, student, recipe_key, rung_tag)``. The teacher rung
    is part of the key so the P3c dose-ladder rows -- same sizes, same recipe, a
    different teacher dose -- cannot merge into the headline anchor cell.
    Quarantined leaves are excluded.

    Returns:
        One row per cell, `CELL_COLUMNS`; ``ci_low`` / ``ci_high`` bound
        ``delta_primary``.
    """
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(CELL_COLUMNS)
    rows = _live(deltas.copy())
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    if rows.empty:
        return _empty(CELL_COLUMNS)
    if "arm" not in rows.columns or rows["arm"].isna().any():
        meta = [
            _pair_meta(t, s)
            for t, s in zip(rows["teacher"], rows["student"], strict=True)
        ]
        rows["arm"] = [m[0] for m in meta]
        rows["shape_steps"] = [m[1] for m in meta]
        rows["direction"] = [m[2] for m in meta]
    keys = ["animal", "teacher", "student", "recipe_key", "rung_tag"]
    out_rows: list[dict] = []
    for key, group in rows.groupby(keys, dropna=False, sort=True):
        key_map = dict(zip(keys, key, strict=True))
        mean, low, high = _cluster_ci(
            group["delta_primary"].tolist(),
            group["ftseed"].tolist() if "ftseed" in group.columns else [0] * len(group),
            group["gseed"].tolist(),
            n_boot=n_boot,
            seed=seed,
        )
        first = group.iloc[0]
        out_rows.append(
            {
                **key_map,
                "arm": first["arm"],
                "shape_steps": first["shape_steps"],
                "direction": first["direction"],
                "n": int(len(group)),
                "mean_delta_primary": mean,
                "mean_delta_sb": _mean_of(group, "delta_sb"),
                "mean_delta_tb": _mean_of(group, "delta_tb"),
                "mean_delta_rng": _mean_of(group, "delta_rng"),
                "ci_low": low,
                "ci_high": high,
                "mean_p_target": _mean_of(group, "p_target"),
                "mean_storytelling_p_target": _mean_of(group, "storytelling_p_target"),
                "mean_ppl_ratio": _mean_of(group, "perplexity_ratio"),
            }
        )
    if not out_rows:
        return _empty(CELL_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(CELL_COLUMNS))


def _mean_of(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _leaf_values(
    deltas: pd.DataFrame | None,
    *,
    teacher: str,
    student: str,
    recipe_key: str,
    rung_tag: str | None,
    column: str,
) -> list[float]:
    if deltas is None or deltas.empty or column not in deltas.columns:
        return []
    rows = _live(deltas)
    rows = rows[
        (rows["teacher"].astype(str) == str(teacher))
        & (rows["student"].astype(str) == str(student))
        & (rows["recipe_key"].astype(str) == str(recipe_key))
    ]
    if rung_tag is not None and "rung_tag" in rows.columns:
        rows = rows[rows["rung_tag"].astype(str) == str(rung_tag)]
    return [
        float(v) for v in pd.to_numeric(rows[column], errors="coerce") if _finite(v)
    ]


def shape_penalty(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, str],
    deltas: pd.DataFrame | None = None,
    rungs: Mapping[str, str] | None = None,
    teacher_p_target: Mapping[str, float] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Shape-mismatch penalty of each cross pair at the receiving size's recipe.

    ``rho_s = Delta_tb(T->S) / Delta_sb(S->S)``, ``rho_t = Delta_tb(T->S) /
    Delta_sb(T->T)``, ``penalty = Delta_sb(S->S) - Delta_tb(T->S)``, and the
    teacher-efficiency normalisation ``Delta_tb / teacher_p_target[T]``. The penalty
    is a cost only when its CI lies entirely above 0; `claims` applies that test.

    Args:
        cells: `cell_means` output.
        picks: Student size -> picked recipe key.
        deltas: Leaf frame the penalty CI resamples; no CI without it.
        rungs: Teacher size -> picked rung tag, so a cell is never read off another
            teacher dose.
        teacher_p_target: Teacher size -> trait strength, for ``efficiency``.
        n_boot: Bootstrap draws.
        seed: Bootstrap RNG seed.

    Returns:
        One row per cross pair, `PENALTY_COLUMNS`.
    """
    if cells.empty:
        return _empty(PENALTY_COLUMNS)
    dose = teacher_p_target or {}
    rung_of = rungs or {}
    out_rows: list[dict] = []
    for teacher, student in pairs.all_pairs():
        if teacher == student or student not in picks:
            continue
        recipe_s = str(picks[student])
        recipe_t = str(picks.get(teacher, recipe_s))
        cross = _cell_at(
            cells,
            teacher=teacher,
            student=student,
            recipe_key=recipe_s,
            rung_tag=rung_of.get(teacher),
        )
        anchor_s = _cell_at(
            cells,
            teacher=student,
            student=student,
            recipe_key=recipe_s,
            rung_tag=rung_of.get(student),
        )
        anchor_t = _cell_at(
            cells,
            teacher=teacher,
            student=teacher,
            recipe_key=recipe_t,
            rung_tag=rung_of.get(teacher),
        )
        if cross is None or anchor_s is None:
            continue
        d_cross = float(cross["mean_delta_tb"])
        d_s = float(anchor_s["mean_delta_sb"])
        d_t = float(anchor_t["mean_delta_sb"]) if anchor_t is not None else float("nan")
        _, low, high = _mean_diff_ci(
            _leaf_values(
                deltas,
                teacher=student,
                student=student,
                recipe_key=recipe_s,
                rung_tag=rung_of.get(student),
                column="delta_sb",
            ),
            _leaf_values(
                deltas,
                teacher=teacher,
                student=student,
                recipe_key=recipe_s,
                rung_tag=rung_of.get(teacher),
                column="delta_tb",
            ),
            n_boot=n_boot,
            seed=seed,
        )
        _, shape_steps, direction = _pair_meta(teacher, student)
        t_p = dose.get(teacher)
        out_rows.append(
            {
                "teacher": teacher,
                "student": student,
                "recipe_key": recipe_s,
                "shape_steps": shape_steps,
                "direction": direction,
                "delta_tb": d_cross,
                "delta_anchor_s": d_s,
                "delta_anchor_t": d_t,
                "rho_s": _ratio(d_cross, d_s),
                "rho_t": _ratio(d_cross, d_t),
                "penalty": d_s - d_cross,
                "penalty_ci_low": low,
                "penalty_ci_high": high,
                "efficiency": (
                    _ratio(d_cross, float(t_p)) if t_p is not None else float("nan")
                ),
                "teacher_p_target": float(t_p) if t_p is not None else float("nan"),
            }
        )
    if not out_rows:
        return _empty(PENALTY_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(PENALTY_COLUMNS))


def direction_asymmetry(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, str],
    deltas: pd.DataFrame | None = None,
    rungs: Mapping[str, str] | None = None,
    teacher_p_target: Mapping[str, float] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """``iota = Delta_tb(small->large) - Delta_tb(large->small)`` per unordered pair.

    ``iota_dose`` divides each direction's Delta by that direction's teacher
    ``p_target`` before differencing, so a size whose teacher simply carries more
    trait does not read as a direction effect.

    Returns:
        One row per unordered size pair, `ASYMMETRY_COLUMNS`.
    """
    if cells.empty:
        return _empty(ASYMMETRY_COLUMNS)
    dose = teacher_p_target or {}
    rung_of = rungs or {}
    out_rows: list[dict] = []
    for i, lo_size in enumerate(SIZES):
        for hi_size in SIZES[i + 1 :]:
            if lo_size not in picks or hi_size not in picks:
                continue
            recipe_up = str(picks[hi_size])
            recipe_down = str(picks[lo_size])
            up = _cell_at(
                cells,
                teacher=lo_size,
                student=hi_size,
                recipe_key=recipe_up,
                rung_tag=rung_of.get(lo_size),
            )
            down = _cell_at(
                cells,
                teacher=hi_size,
                student=lo_size,
                recipe_key=recipe_down,
                rung_tag=rung_of.get(hi_size),
            )
            if up is None or down is None:
                continue
            d_up = float(up["mean_delta_tb"])
            d_down = float(down["mean_delta_tb"])
            p_lo = dose.get(lo_size)
            p_hi = dose.get(hi_size)
            if p_lo is not None and p_hi is not None:
                iota_dose = _ratio(d_up, float(p_lo)) - _ratio(d_down, float(p_hi))
            else:
                iota_dose = float("nan")
            _, low, high = _mean_diff_ci(
                _leaf_values(
                    deltas,
                    teacher=lo_size,
                    student=hi_size,
                    recipe_key=recipe_up,
                    rung_tag=rung_of.get(lo_size),
                    column="delta_tb",
                ),
                _leaf_values(
                    deltas,
                    teacher=hi_size,
                    student=lo_size,
                    recipe_key=recipe_down,
                    rung_tag=rung_of.get(hi_size),
                    column="delta_tb",
                ),
                n_boot=n_boot,
                seed=seed,
            )
            out_rows.append(
                {
                    "size_lo": lo_size,
                    "size_hi": hi_size,
                    "iota": d_up - d_down,
                    "iota_dose": iota_dose,
                    "ci_low": low,
                    "ci_high": high,
                    "delta_up": d_up,
                    "delta_down": d_down,
                }
            )
    if not out_rows:
        return _empty(ASYMMETRY_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(ASYMMETRY_COLUMNS))


def _delta_matrix(
    cells: pd.DataFrame,
    picks: Mapping[str, str],
    rungs: Mapping[str, str] | None,
) -> list[tuple[str, str, float]]:
    """``(teacher, student, Delta)`` at each size's picked recipe and rung."""
    rung_of = rungs or {}
    out: list[tuple[str, str, float]] = []
    for teacher, student in pairs.all_pairs():
        if student not in picks:
            continue
        row = _cell_at(
            cells,
            teacher=teacher,
            student=student,
            recipe_key=str(picks[student]),
            rung_tag=rung_of.get(teacher),
        )
        if row is None:
            continue
        column = "mean_delta_sb" if teacher == student else "mean_delta_tb"
        value = row[column]
        if _finite(value):
            out.append((teacher, student, float(value)))
    return out


def _ladder_contrast(contrast: ReciprocalContrast) -> tuple[str, float, float]:
    """One reciprocal contrast keyed and signed on the `SIZES` ladder.

    `analysis.decomposition` orients an unordered pair lexicographically, which
    reverses two of the campaign's three pairs (``"70m"`` sorts last). Pooling
    those unflipped gives ``pooled_iota`` a sign that `_pooled_iota_ci` and
    `direction_asymmetry` -- both ``Delta(small->large) - Delta(large->small)``
    -- contradict.

    Returns:
        ``("{lo}|{hi}", iota, interaction)`` with ``lo`` the smaller size.
    """
    lo, hi = contrast.base_a, contrast.base_b
    sign = 1.0
    if SIZES.index(lo) > SIZES.index(hi):
        lo, hi, sign = hi, lo, -1.0
    return (
        f"{lo}|{hi}",
        sign * float(contrast.asymmetry),
        sign * float(contrast.interaction),
    )


def additive_fit(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, str],
    rungs: Mapping[str, str] | None = None,
    deltas: pd.DataFrame | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Least-squares ``Delta_ij = mu + tau_i + sigma_j`` on the 3x3, with LORO CV.

    Anchors contribute ``delta_sb``, cross cells ``delta_tb``; effects are
    sum-to-zero identified by `analysis.decomposition.fit_additive`. ``loo_r2`` is
    the held-out R2 of `leave_one_reciprocal_pair_out`, the honest separability test
    (an in-sample fit on a complete grid is near-tautological).

    Returns:
        ``mu``, ``tau``, ``sigma``, ``r2``, ``loo_r2``, ``loo_pearson``,
        ``loo_sign_agreement``, ``loo_n_folds``, ``n_obs``, ``residuals``, ``iota``
        and ``interaction`` per unordered pair keyed ``"{small}|{large}"`` and
        signed small -> large (`_ladder_contrast`), ``pooled_iota`` and its CI.
        Fit fields are empty / NaN when the grid cannot be fitted.
    """
    from subliminal_learning.analysis.decomposition import (
        fit_additive,
        leave_one_reciprocal_pair_out,
        reciprocal_asymmetries,
    )

    matrix = _delta_matrix(cells, picks, rungs)
    fit: AdditiveFit | None = None
    try:
        fit = fit_additive(matrix)
    except ValueError:
        fit = None
    loo_r2 = loo_pearson = loo_sign = float("nan")
    loo_folds = 0
    if fit is not None:
        try:
            held_out = leave_one_reciprocal_pair_out(matrix)
        except ValueError:
            held_out = None
        if held_out is not None:
            loo_r2 = held_out.r_squared
            loo_pearson = held_out.pearson
            loo_sign = held_out.sign_agreement
            loo_folds = held_out.n_folds
    contrasts = reciprocal_asymmetries(matrix, fit) if fit is not None else ()
    oriented = [_ladder_contrast(c) for c in contrasts]
    iotas = {key: asymmetry for key, asymmetry, _ in oriented}
    interactions = {key: interaction for key, _, interaction in oriented}
    pooled = float(np.mean(list(iotas.values()))) if iotas else float("nan")
    low, high = _pooled_iota_ci(
        deltas, picks=picks, rungs=rungs, n_boot=n_boot, seed=seed
    )
    return {
        "mu": float(fit.mu) if fit is not None else float("nan"),
        "tau": dict(fit.emissivity) if fit is not None else {},
        "sigma": dict(fit.receptivity) if fit is not None else {},
        "r2": float(fit.r_squared) if fit is not None else float("nan"),
        "loo_r2": loo_r2,
        "loo_pearson": loo_pearson,
        "loo_sign_agreement": loo_sign,
        "loo_n_folds": loo_folds,
        "n_obs": int(fit.n_obs) if fit is not None else 0,
        "residuals": (
            {f"{i}|{j}": float(v) for (i, j), v in fit.residuals.items()}
            if fit is not None
            else {}
        ),
        "iota": iotas,
        "interaction": interactions,
        "pooled_iota": pooled,
        "pooled_iota_ci_low": low,
        "pooled_iota_ci_high": high,
    }


def _pooled_iota_ci(
    deltas: pd.DataFrame | None,
    *,
    picks: Mapping[str, str],
    rungs: Mapping[str, str] | None,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI on the mean reciprocal asymmetry, resampling leaves per arm."""
    if deltas is None or deltas.empty or n_boot <= 0:
        return float("nan"), float("nan")
    rung_of = rungs or {}
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for i, lo_size in enumerate(SIZES):
        for hi_size in SIZES[i + 1 :]:
            if lo_size not in picks or hi_size not in picks:
                continue
            up = _leaf_values(
                deltas,
                teacher=lo_size,
                student=hi_size,
                recipe_key=str(picks[hi_size]),
                rung_tag=rung_of.get(lo_size),
                column="delta_tb",
            )
            down = _leaf_values(
                deltas,
                teacher=hi_size,
                student=lo_size,
                recipe_key=str(picks[lo_size]),
                rung_tag=rung_of.get(hi_size),
                column="delta_tb",
            )
            if not up or not down:
                return float("nan"), float("nan")
            groups.append((np.asarray(up), np.asarray(down)))
    if not groups:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = float(
            np.mean(
                [
                    float(
                        u[rng.integers(0, u.size, u.size)].mean()
                        - d[rng.integers(0, d.size, d.size)].mean()
                    )
                    for u, d in groups
                ]
            )
        )
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def recipe_stability(
    deltas: pd.DataFrame,
    *,
    picks: Mapping[str, str],
) -> pd.DataFrame:
    """Sign of ``delta_tb`` per pair at each recipe, plus Kendall tau vs the pick.

    Covers the star screen (anchor pairs at every screened recipe) and the P3b tail
    (one-step cross pairs at the runner-up recipes). P3c and P3d rows are excluded:
    they vary the teacher dose at a fixed recipe, not the recipe, so averaging them
    into their recipe's row would report a teacher-dose effect as recipe instability.
    `teacher_dose_curve` and `concordance` read them instead.

    Returns:
        One row per ``(recipe_key, teacher, student)``, `RECIPE_COLUMNS`.
    """
    if deltas.empty or "delta_tb" not in deltas.columns:
        return _empty(RECIPE_COLUMNS)
    rows = _live(deltas)
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    if "phase" in rows.columns:
        rows = rows[
            ~rows["phase"]
            .astype(str)
            .isin({pairs.PHASE_TEACHER_DOSE, pairs.PHASE_CONCORDANCE})
        ]
    if rows.empty:
        return _empty(RECIPE_COLUMNS)
    grouped = rows.groupby(["recipe_key", "teacher", "student"], dropna=False)
    by_recipe: dict[str, dict[tuple[str, str], float]] = {}
    meta: dict[tuple[str, str, str], tuple[int, str]] = {}
    for (recipe_key, teacher, student), group in grouped:
        value = float(pd.to_numeric(group["delta_tb"], errors="coerce").mean())
        by_recipe.setdefault(str(recipe_key), {})[(str(teacher), str(student))] = value
        phases = sorted(
            {str(p) for p in group.get("phase", pd.Series(dtype=str)) if p is not None}
        )
        meta[(str(recipe_key), str(teacher), str(student))] = (
            int(len(group)),
            ";".join(phases),
        )
    reference: dict[tuple[str, str], float] = {}
    for recipe_key, cells in by_recipe.items():
        for (teacher, student), value in cells.items():
            if str(picks.get(student)) == recipe_key:
                reference[(teacher, student)] = value
    out_rows: list[dict] = []
    for recipe_key in sorted(by_recipe):
        at_recipe = by_recipe[recipe_key]
        shared = [p for p in sorted(at_recipe) if p in reference]
        tau = float("nan")
        if len(shared) >= 2:
            statistic = scipy_stats.kendalltau(
                [at_recipe[p] for p in shared], [reference[p] for p in shared]
            ).statistic
            if statistic is not None and np.isfinite(statistic):
                tau = float(statistic)
        for teacher, student in sorted(at_recipe):
            value = at_recipe[(teacher, student)]
            count, phase = meta[(recipe_key, teacher, student)]
            out_rows.append(
                {
                    "recipe_key": recipe_key,
                    "teacher": teacher,
                    "student": student,
                    "phase": phase,
                    "n": count,
                    "delta_tb": value,
                    "sign": (
                        int(np.sign(value)) if _finite(value) and value != 0.0 else 0
                    ),
                    "kendall_tau": tau,
                    "picked_recipe": str(picks.get(student, "")),
                }
            )
    if not out_rows:
        return _empty(RECIPE_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(RECIPE_COLUMNS))


def teacher_dose_curve(
    deltas: pd.DataFrame,
    *,
    rung_covariates: Mapping[str, Sequence[float]] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """``Delta_sb(160m->160m)`` per teacher rung: Delta when only teacher dose moves.

    Reads the P3c rows plus the headline anchor at the picked rung, which P3c omits
    by construction (those cells already exist in P3). This bounds how much of a
    cross-size Delta gap a per-size teacher-recipe difference alone can explain.

    Args:
        deltas: `leaf_deltas` output.
        rung_covariates: Rung tag -> ``(teacher p_target, off_topic, pool keep
            rate)``, carried onto the curve as columns.
        n_boot: Bootstrap draws.
        seed: Bootstrap RNG seed.

    Returns:
        One row per rung, `DOSE_COLUMNS`, ordered by teacher epochs.
    """
    if deltas.empty or "delta_sb" not in deltas.columns:
        return _empty(DOSE_COLUMNS)
    rows = _live(deltas)
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    rows = rows[
        (rows["teacher"].astype(str) == P3C_SIZE)
        & (rows["student"].astype(str) == P3C_SIZE)
    ]
    if "phase" in rows.columns:
        rows = rows[
            rows["phase"]
            .astype(str)
            .isin({pairs.PHASE_TEACHER_DOSE, pairs.PHASE_HEADLINE})
        ]
    if rows.empty:
        return _empty(DOSE_COLUMNS)
    covariates = rung_covariates or {}
    rung_by_tag = {rung.key: rung for rung in TEACHER_RUNGS}
    out_rows: list[dict] = []
    for rung_tag, group in rows.groupby("rung_tag", dropna=False, sort=True):
        mean, low, high = _cluster_ci(
            group["delta_sb"].tolist(),
            group["ftseed"].tolist() if "ftseed" in group.columns else [0] * len(group),
            group["gseed"].tolist(),
            n_boot=n_boot,
            seed=seed,
        )
        rung = rung_by_tag.get(str(rung_tag))
        covariate = list(covariates.get(str(rung_tag), ()))
        covariate += [float("nan")] * (3 - len(covariate))
        out_rows.append(
            {
                "rung_tag": str(rung_tag),
                "epochs": rung.epochs if rung is not None else float("nan"),
                "family": rung.family if rung is not None else "",
                "recipe_key": str(group.iloc[0]["recipe_key"]),
                "n": int(len(group)),
                "mean_delta_sb": mean,
                "ci_low": low,
                "ci_high": high,
                "teacher_p_target": covariate[0],
                "off_topic": covariate[1],
                "pool_keep_rate": covariate[2],
            }
        )
    frame = pd.DataFrame(out_rows, columns=list(DOSE_COLUMNS))
    return frame.sort_values("epochs", kind="stable", ignore_index=True)


def ppl_coupling(deltas: pd.DataFrame) -> pd.DataFrame:
    """OLS of ``delta_primary`` on ``perplexity_ratio``, residualised to PPL = 1.

    ``Delta_adj = Delta - beta*(ppl - 1.0)``; ``retained = |mean Delta_adj| / |mean
    Delta|``. Grouped by ``(teacher, student, recipe_key, rung_tag)`` so a group is
    one teacher dose. Quarantined leaves are excluded.

    Returns:
        One row per group, `PPL_COLUMNS`.
    """
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(PPL_COLUMNS)
    rows = _live(deltas)
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    if rows.empty or "perplexity_ratio" not in rows.columns:
        return _empty(PPL_COLUMNS)
    keys = ["teacher", "student", "recipe_key", "rung_tag"]
    out_rows: list[dict] = []
    for key, group in rows.groupby(keys, dropna=False, sort=True):
        y = pd.to_numeric(group["delta_primary"], errors="coerce")
        x = pd.to_numeric(group["perplexity_ratio"], errors="coerce")
        mask = y.notna() & x.notna()
        yv = y[mask].to_numpy(dtype=float)
        xv = x[mask].to_numpy(dtype=float)
        if yv.size == 0:
            continue
        beta = _ols_slope(xv, yv)
        adjusted = yv - beta * (xv - 1.0)
        mean_delta = float(yv.mean())
        mean_adjusted = float(adjusted.mean())
        out_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "beta": beta,
                "mean_delta": mean_delta,
                "mean_delta_adj": mean_adjusted,
                "retained_fraction": _ratio(abs(mean_adjusted), abs(mean_delta)),
                "n": int(yv.size),
            }
        )
    if not out_rows:
        return _empty(PPL_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(PPL_COLUMNS))


def concordance(cells: pd.DataFrame) -> pd.DataFrame:
    """The 160m->160m cell on Angela's own recipe against her published numbers.

    Descriptive only: `ANGELA_REFERENCE` was measured under a different code pin, so
    the gaps report agreement, not a test.

    Returns:
        A one-row frame, `CONCORDANCE_COLUMNS`; measurement columns are NaN when the
        cell has not been trained yet.
    """
    recipe_key = str(ANGELA_REFERENCE["student_tag"])
    rung_tag = str(ANGELA_REFERENCE["teacher_tag"])
    ref_seed1 = float(ANGELA_REFERENCE["test_p_target_seed1"])
    ref_3seed = float(ANGELA_REFERENCE["optparam_test_p_target_3seed"])
    row = (
        None
        if cells.empty
        else _cell_at(
            cells,
            teacher=P3C_SIZE,
            student=P3C_SIZE,
            recipe_key=recipe_key,
            rung_tag=rung_tag,
        )
    )
    p_target = float(row["mean_p_target"]) if row is not None else float("nan")
    record = {
        "cell": f"{P3C_SIZE}->{P3C_SIZE}",
        "recipe_key": recipe_key,
        "rung_tag": rung_tag,
        "n": int(row["n"]) if row is not None else 0,
        "p_target": p_target,
        "delta_sb": float(row["mean_delta_sb"]) if row is not None else float("nan"),
        "ref_p_target_seed1": ref_seed1,
        "abs_gap_seed1": p_target - ref_seed1,
        "rel_gap_seed1": _ratio(p_target - ref_seed1, ref_seed1),
        "ref_p_target_3seed": ref_3seed,
        "abs_gap_3seed": p_target - ref_3seed,
        "rel_gap_3seed": _ratio(p_target - ref_3seed, ref_3seed),
        "ref_rng_floor_3seed": float(ANGELA_REFERENCE["optparam_rng_floor_3seed"]),
        "ref_teacher_tag": rung_tag,
        "ref_student_tag": recipe_key,
    }
    return pd.DataFrame([record], columns=list(CONCORDANCE_COLUMNS))


def _verdict(n_units: int, n_holds: int, detail: dict[str, Any]) -> dict:
    """`holds` True only when every measurable unit passes; void when none exist."""
    if n_units == 0:
        return {"holds": None, "void": True, "detail": detail}
    return {"holds": n_holds == n_units, "void": False, "detail": detail}


def _o0(
    cells: pd.DataFrame,
    picks: Mapping[str, str],
    rungs: Mapping[str, str] | None,
) -> dict:
    rung_of = rungs or {}
    verdicts: dict[str, dict] = {}
    n_units = n_holds = 0
    for teacher, student in pairs.all_pairs():
        if teacher == student or student not in picks:
            continue
        label = f"{teacher}->{student}"
        recipe_s = str(picks[student])
        if rungs is not None and teacher not in rung_of:
            verdicts[label] = {"void": True, "reason": "no_admissible_teacher"}
            continue
        anchor = _cell_at(
            cells,
            teacher=student,
            student=student,
            recipe_key=recipe_s,
            rung_tag=rung_of.get(student),
        )
        if anchor is None:
            verdicts[label] = {"void": True, "reason": "missing_anchor"}
            continue
        if _ci_covers_zero(float(anchor["ci_low"]), float(anchor["ci_high"])):
            verdicts[label] = {
                "void": True,
                "reason": "anchor_ci_covers_0",
                "anchor_mean": float(anchor["mean_delta_sb"]),
                "anchor_ci": [float(anchor["ci_low"]), float(anchor["ci_high"])],
            }
            continue
        row = _cell_at(
            cells,
            teacher=teacher,
            student=student,
            recipe_key=recipe_s,
            rung_tag=rung_of.get(teacher),
        )
        if row is None:
            verdicts[label] = {"void": True, "reason": "missing_cell"}
            continue
        mean = float(row["mean_delta_primary"])
        low, high = float(row["ci_low"]), float(row["ci_high"])
        holds = _ci_excludes_zero(low, high) and mean >= X0_MIN_DELTA
        verdicts[label] = {
            "void": False,
            "holds": holds,
            "mean": mean,
            "ci": [low, high],
            "headline": (teacher, student) in pairs.one_step_pairs(),
        }
        n_units += 1
        n_holds += int(holds)
    return _verdict(
        n_units,
        n_holds,
        {
            "pairs": verdicts,
            "n": n_units,
            "n_holds": n_holds,
            "min_delta": X0_MIN_DELTA,
            "prompt_set": PRIMARY_PROMPT_SET,
        },
    )


def _o1(penalty: pd.DataFrame | None, rungs: Mapping[str, str] | None) -> dict:
    rung_of = rungs or {}
    verdicts: dict[str, dict] = {}
    n_units = n_holds = 0
    if penalty is not None and not penalty.empty:
        for _, row in penalty.iterrows():
            label = f"{row['teacher']}->{row['student']}"
            if rungs is not None and str(row["teacher"]) not in rung_of:
                verdicts[label] = {"void": True, "reason": "no_admissible_teacher"}
                continue
            low = float(row["penalty_ci_low"])
            high = float(row["penalty_ci_high"])
            if not (_finite(low) and _finite(high)):
                verdicts[label] = {"void": True, "reason": "no_penalty_ci"}
                continue
            costly = _ci_excludes_zero(low, high) and low > 0.0
            verdicts[label] = {
                "void": False,
                "holds": costly,
                "penalty": float(row["penalty"]),
                "ci": [low, high],
                "rho_s": float(row["rho_s"]) if _finite(row["rho_s"]) else None,
                "rho_t": float(row["rho_t"]) if _finite(row["rho_t"]) else None,
                "shape_steps": int(row["shape_steps"]),
                "efficiency": (
                    float(row["efficiency"]) if _finite(row["efficiency"]) else None
                ),
            }
            n_units += 1
            n_holds += int(costly)
    return _verdict(
        n_units, n_holds, {"pairs": verdicts, "n": n_units, "n_holds": n_holds}
    )


def _o2(additive: dict | None) -> dict:
    fit = additive or {}
    loo_r2 = fit.get("loo_r2")
    pooled = fit.get("pooled_iota")
    low = fit.get("pooled_iota_ci_low")
    high = fit.get("pooled_iota_ci_high")
    detail = {
        "r2": fit.get("r2"),
        "loo_r2": loo_r2,
        "pooled_iota": pooled,
        "pooled_iota_ci": [low, high],
        "additive_r2_min": X2_ADDITIVE_R2_MIN,
        "asym_min": X2_ASYM_MIN,
        "loo_n_folds": fit.get("loo_n_folds"),
        "iota": fit.get("iota"),
    }
    if not _finite(loo_r2):
        detail["read"] = None
        return {"holds": None, "void": True, "detail": detail}
    separable = (
        float(loo_r2) >= X2_ADDITIVE_R2_MIN
        and low is not None
        and high is not None
        and _ci_covers_zero(float(low), float(high))
    )
    directional = (
        low is not None
        and high is not None
        and _ci_excludes_zero(float(low), float(high))
        and _finite(pooled)
        and abs(float(pooled)) >= X2_ASYM_MIN
    )
    if separable and not directional:
        read = "separable"
    elif directional and not separable:
        read = "directional"
    elif separable and directional:
        read = "both"
    else:
        read = "neither"
    detail["read"] = read
    return {
        "holds": read in {"separable", "directional"},
        "void": False,
        "detail": detail,
    }


def _o3(recipe: pd.DataFrame | None, dose: pd.DataFrame | None) -> dict:
    signs: dict[str, list[int]] = {}
    taus: list[float] = []
    if recipe is not None and not recipe.empty:
        for _, row in recipe.iterrows():
            label = f"{row['teacher']}->{row['student']}"
            signs.setdefault(label, []).append(int(row["sign"]))
            if _finite(row["kendall_tau"]):
                taus.append(float(row["kendall_tau"]))
    dose_rows = (
        []
        if dose is None or dose.empty
        else [
            {
                "rung_tag": str(row["rung_tag"]),
                "epochs": (float(row["epochs"]) if _finite(row["epochs"]) else None),
                "mean_delta_sb": float(row["mean_delta_sb"]),
                "ci": [float(row["ci_low"]), float(row["ci_high"])],
            }
            for _, row in dose.iterrows()
        ]
    )
    detail = {
        "sign_stable": {
            label: len(set(values)) == 1 for label, values in signs.items()
        },
        "kendall_tau_mean": float(np.mean(taus)) if taus else float("nan"),
        "n_recipes": (
            0 if recipe is None or recipe.empty else int(recipe["recipe_key"].nunique())
        ),
        "teacher_dose_curve": dose_rows,
    }
    void = not signs and not dose_rows
    return {"holds": None, "void": void, "detail": detail}


def _o4(ppl: pd.DataFrame | None) -> dict:
    values = (
        []
        if ppl is None or ppl.empty
        else [float(v) for v in ppl["retained_fraction"].tolist() if _finite(v)]
    )
    detail = {
        "retained_fraction": float(np.mean(values)) if values else float("nan"),
        "threshold": X4_RETAINED_FRACTION_MIN,
        "n": len(values),
    }
    if not values:
        return {"holds": None, "void": True, "detail": detail}
    holds = float(np.mean(values)) >= X4_RETAINED_FRACTION_MIN
    return {"holds": holds, "void": False, "detail": detail}


def _o5(concordance_frame: pd.DataFrame | None) -> dict:
    if concordance_frame is None or concordance_frame.empty:
        return {"holds": None, "void": True, "detail": {}}
    row = concordance_frame.iloc[0]
    detail = {key: _plain(row[key]) for key in CONCORDANCE_COLUMNS}
    return {"holds": None, "void": not _finite(row["p_target"]), "detail": detail}


def _plain(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    return value


def claims(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, str],
    rungs: Mapping[str, str] | None = None,
    penalty: pd.DataFrame | None = None,
    additive: dict | None = None,
    recipe: pd.DataFrame | None = None,
    dose: pd.DataFrame | None = None,
    ppl: pd.DataFrame | None = None,
    concordance_frame: pd.DataFrame | None = None,
) -> dict:
    """Adjudicate O0-O5 from the metric frames.

    ``holds`` is True only when every measurable unit of the claim passes, False
    when at least one fails, and None when the claim is descriptive (O3, O5) or has
    no measurable unit. ``void`` marks the latter: a teacher size with no admitted
    rung, an anchor whose CI covers 0, or a frame the campaign has not filled yet.

    Args:
        cells: `cell_means` output.
        picks: Student size -> picked recipe key.
        rungs: Teacher size -> picked rung tag; a size absent from a non-None
            mapping voids its pairs.
        penalty: `shape_penalty` output (O1).
        additive: `additive_fit` output (O2).
        recipe: `recipe_stability` output (O3).
        dose: `teacher_dose_curve` output (O3).
        ppl: `ppl_coupling` output (O4).
        concordance_frame: `concordance` output (O5).

    Returns:
        ``{"O0": {"holds": ..., "void": ..., "detail": {...}}, ...}`` keyed by
        exactly `CLAIM_IDS`.
    """
    payload = {
        "O0": _o0(cells, picks, rungs),
        "O1": _o1(penalty, rungs),
        "O2": _o2(additive),
        "O3": _o3(recipe, dose),
        "O4": _o4(ppl),
        "O5": _o5(concordance_frame),
    }
    if tuple(payload) != CLAIM_IDS:
        raise ValueError(f"claims would carry {tuple(payload)}, not {CLAIM_IDS}")
    return payload
