"""Pure frames for cross-family claims F0-F7.

Leaf-frame schema (`LEAF_COLUMNS`): ``animal``, ``cluster``, ``student``,
``teacher``, ``kind`` (``treatment`` / ``sb_control`` / ``tb_control``), ``arm``
(campaign arm name, ``""`` on controls), ``arm_kind`` (a `xfam_plan.KIND_*`
token), ``pair_kind`` (``anchor`` when teacher == student, else ``cross``),
``teacher_param_tag``, ``lr``, ``epochs``, ``lora_r``, ``lora_alpha``,
``gkd_tag``, ``ftseed`` (NaN on controls), ``gseed``, ``p_target``,
``perplexity_ratio``, ``filter_drop``, ``quarantined``. Labels are roster labels
(``mamba-370m``), never sizes.

Controls are GKD-tag matched: both joins carry ``gkd_tag``, so a treatment is
only ever differenced against a control trained at the same LR, rank and epoch
budget. An anchor cell has no teacher-base control -- its two control classes
coincide -- so its ``delta_tb`` is NaN and ``delta_primary`` is ``delta_sb``.

Cell CIs resample teacher ftseeds and student gseeds jointly
(`analysis.stats.cluster_bootstrap`), which is the crossing a cell's leaves
actually have; it degrades to a one-way bootstrap when either axis has a single
level. `ppl_coupling` residualises to the no-degradation point
(``delta_adj = delta - beta*(ppl - 1)``) and marks a group whose regressor is
constant as degenerate rather than crediting it ``retained = 1.0``.

`analysis.stats` is imported inside the two helpers that call it: the `analysis`
package's ``__init__`` eagerly pulls a ``*_io`` loader, and this module is on the
pure-import list (`tests/unit/test_layering.py`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.xfam_selection import TeacherGateDecision

KIND_TREATMENT = "treatment"
KIND_SB = "sb_control"
KIND_TB = "tb_control"

PAIR_ANCHOR = "anchor"
PAIR_CROSS = "cross"

RUNG_ANCHOR = "anchor"
RUNG_POST_TRAINED = "post_trained"
RUNG_DIFFERENT_RUN = "different_run"
RUNG_DIFFERENT_FAMILY = "different_family"

LEAF_COLUMNS: tuple[str, ...] = (
    "animal",
    "cluster",
    "student",
    "teacher",
    "kind",
    "arm",
    "arm_kind",
    "pair_kind",
    "teacher_param_tag",
    "lr",
    "epochs",
    "lora_r",
    "lora_alpha",
    "gkd_tag",
    "ftseed",
    "gseed",
    "p_target",
    "perplexity_ratio",
    "filter_drop",
    "quarantined",
)

DELTA_COLUMNS: tuple[str, ...] = (
    *LEAF_COLUMNS,
    "delta_sb",
    "delta_tb",
    "delta_primary",
    "sb_control_mean",
    "tb_control_mean",
)

CELL_KEYS: tuple[str, ...] = (
    "animal",
    "cluster",
    "arm",
    "arm_kind",
    "teacher",
    "student",
    "pair_kind",
    "teacher_param_tag",
    "gkd_tag",
    "lr",
    "epochs",
    "lora_r",
    "lora_alpha",
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

HEADLINE_COLUMNS: tuple[str, ...] = (*CELL_COLUMNS, "picked_lr", "primary")

PENALTY_COLUMNS: tuple[str, ...] = (
    "cluster",
    "arm",
    "teacher",
    "student",
    "lr",
    "delta_tb",
    "delta_anchor",
    "rho_s",
    "penalty",
    "penalty_ci_low",
    "penalty_ci_high",
    "penalty_holds",
    "efficiency",
    "teacher_p_target",
    "xsize_reference_delta",
)

LADDER_COLUMNS: tuple[str, ...] = (
    "cluster",
    "rung_index",
    "student",
    "rung_kind",
    "arm",
    "teacher",
    "lr",
    "n",
    "mean_delta",
    "ci_low",
    "ci_high",
    "spearman_rho",
    "spearman_p",
    "n_rungs",
)

POSTTRAIN_COLUMNS: tuple[str, ...] = (
    "cluster",
    "lineage",
    "student",
    "post_training",
    "arm",
    "teacher",
    "delta",
    "base_delta",
    "diff",
    "ci_low",
    "ci_high",
    "significant",
    "n",
    "n_base",
)

ASYMMETRY_COLUMNS: tuple[str, ...] = (
    "cluster",
    "label_a",
    "label_b",
    "iota",
    "iota_dose",
    "ci_low",
    "ci_high",
    "delta_ab",
    "delta_ba",
    "p_target_a",
    "p_target_b",
)

HP_COLUMNS: tuple[str, ...] = (
    "cluster",
    "axis",
    "setting",
    "arm",
    "teacher",
    "student",
    "lr",
    "epochs",
    "lora_r",
    "lora_alpha",
    "n",
    "delta",
    "sign",
    "reference_delta",
    "sign_flip",
    "kendall_tau",
    "n_shared",
    "picked_lr",
)

PPL_COLUMNS: tuple[str, ...] = (
    "cluster",
    "arm",
    "teacher",
    "student",
    "gkd_tag",
    "beta",
    "mean_delta",
    "mean_delta_adj",
    "retained_fraction",
    "n",
    "degenerate",
    "degenerate_reason",
)

BASE_PRIOR_COLUMNS: tuple[str, ...] = (
    "label",
    "cluster",
    "p_target",
    "fc_share",
    "perplexity",
)

AXIS_LR = "lr"
AXIS_RANK = "rank"
AXIS_EPOCH = "epoch"

_SB_KEYS: tuple[str, ...] = ("animal", "student", "gkd_tag", "gseed")
_TB_KEYS: tuple[str, ...] = ("animal", "student", "teacher", "gkd_tag", "gseed")

_ARM_ORDER: dict[str, int] = {a.name: i for i, a in enumerate(plan.ARMS)}


# ---- small helpers -----------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _ci_excludes_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and not (low <= 0.0 <= high)


def _ci_covers_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and low <= 0.0 <= high


def _live_treatments(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame
    if "quarantined" in rows.columns:
        rows = rows.loc[~rows["quarantined"].fillna(False).astype(bool)]
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    return rows


def _arrays(
    frame: pd.DataFrame, column: str
) -> tuple[np.ndarray, list[str], list[str]]:
    """Finite `column` values with their per-row ftseed / gseed labels."""
    if frame.empty or column not in frame.columns:
        return np.empty(0), [], []
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    ftseed = frame.get("ftseed", pd.Series([""] * len(frame), index=frame.index))
    gseed = frame.get("gseed", pd.Series([""] * len(frame), index=frame.index))
    labels_ft = [str(v) for v, keep in zip(ftseed, mask, strict=True) if keep]
    labels_gs = [str(v) for v, keep in zip(gseed, mask, strict=True) if keep]
    return values[mask], labels_ft, labels_gs


def _mean_ci(
    frame: pd.DataFrame,
    column: str,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Mean of `column` with a two-way (ftseed x gseed) cluster bootstrap CI."""
    from subliminal_learning.analysis.stats import cluster_bootstrap

    values, ftseed, gseed = _arrays(frame, column)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1 or n_boot <= 0:
        return mean, mean, mean
    estimate = cluster_bootstrap(
        lambda idx: float(values[idx].mean()),
        ftseed,
        gseed,
        n_boot=n_boot,
        seed=seed,
    )
    return estimate.estimate, estimate.ci_low, estimate.ci_high


def _contrast_ci(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_column: str,
    right_column: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """``mean(left) - mean(right)`` with a seed-paired cluster bootstrap CI.

    Both arms are stacked and resampled together, so a drawn gseed contributes
    its rows on both sides of the difference.
    """
    from subliminal_learning.analysis.stats import cluster_bootstrap

    a_values, a_ft, a_gs = _arrays(left, left_column)
    b_values, b_ft, b_gs = _arrays(right, right_column)
    if a_values.size == 0 or b_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(a_values.mean() - b_values.mean())
    if n_boot <= 0:
        return observed, observed, observed
    values = np.concatenate([a_values, b_values])
    is_left = np.concatenate(
        [np.ones(a_values.size, dtype=bool), np.zeros(b_values.size, dtype=bool)]
    )

    def statistic(idx: np.ndarray) -> float:
        side = is_left[idx]
        if not side.any() or side.all():
            return float("nan")
        picked = values[idx]
        return float(picked[side].mean() - picked[~side].mean())

    estimate = cluster_bootstrap(
        statistic,
        [*a_ft, *b_ft],
        [*a_gs, *b_gs],
        n_boot=n_boot,
        seed=seed,
    )
    return observed, estimate.ci_low, estimate.ci_high


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, str]:
    """OLS slope of `y` on `x`, plus a degeneracy token (``""`` when well posed)."""
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 2:
        return 0.0, "n_lt_2"
    if np.allclose(xv, xv[0]):
        return 0.0, "constant_regressor"
    design = np.column_stack([np.ones(xv.size), xv])
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    return float(coef[1]), ""


def _spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman rho and p-value; NaN when a side is constant or n < 3."""
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 3 or np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan"), float("nan")
    result = scipy_stats.spearmanr(xv, yv)
    return float(result.statistic), float(result.pvalue)


def _kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b between two rankings; NaN with fewer than 2 shared points."""
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 2:
        return float("nan")
    tau = scipy_stats.kendalltau(xv, yv).statistic
    return float(tau) if tau is not None and np.isfinite(tau) else float("nan")


def _sign(value: object) -> int:
    return int(np.sign(float(value))) if _finite(value) and float(value) != 0.0 else 0


def _lookup(means: pd.Series, key: tuple) -> float | None:
    try:
        return float(means.loc[key])
    except KeyError:
        return None


# ---- identity ----------------------------------------------------------------


def pair_kind_of(teacher: object, student: object) -> str:
    """`PAIR_ANCHOR` when the emitter is the receiver's own base, else `PAIR_CROSS`."""
    return PAIR_ANCHOR if str(teacher) == str(student) else PAIR_CROSS


def rung_kind_of(label: str) -> str:
    """Init-distance class of a roster label, relative to its cluster emitter."""
    spec = models.spec(label)
    emitter = models.TEACHER_OF[spec.cluster]
    if label == emitter:
        return RUNG_ANCHOR
    if spec.lineage is not None:
        return RUNG_POST_TRAINED
    if spec.family == models.spec(emitter).family:
        return RUNG_DIFFERENT_RUN
    return RUNG_DIFFERENT_FAMILY


def arm_of_leaf(
    *,
    teacher: str,
    student: str,
    teacher_param_tag: str,
    ftseed: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    epochs: float,
) -> str:
    """Name of the `xfam_plan` arm a treatment leaf on disk belongs to.

    The run dir encodes the teacher tag, the teacher ftseed and the GKD tag but
    not the arm, so the arm is recovered from the ``(teacher, student, teacher
    tag, ftseed, recipe, epochs)`` tuple that `xfam_pairs.unit_leaves` expanded.

    Raises:
        ValueError: If no arm produces the leaf, or more than one does (either
            would mean the frozen arm table and the tree on disk disagree).
    """
    hits = [
        a.name
        for a in plan.ARMS
        if a.teacher == teacher
        and student in a.students
        and a.teacher_param_tag == teacher_param_tag
        and int(ftseed) in a.ftseeds
        and math.isclose(float(epochs), float(a.epochs))
        and any(
            r.lora_r == int(lora_r) and r.lora_alpha == int(lora_alpha)
            for r in a.recipes
        )
    ]
    if len(hits) == 1:
        return hits[0]
    detail = (
        f"{teacher}->{student} tag={teacher_param_tag} tfs={ftseed} "
        f"lr={lr:g} r{lora_r}a{lora_alpha} ep={epochs:g}"
    )
    if not hits:
        raise ValueError(f"no xfam arm produces leaf {detail}")
    raise ValueError(f"leaf {detail} matches several arms: {hits}")


def picked_lr_of(
    arm_name: str, student: str, picks: Mapping[str, float]
) -> float | None:
    """``lr*`` that gates one ``(arm, student)`` unit, or None before the pick."""
    return picks.get(plan.arm(arm_name).lr_source_of(student))


# ---- quarantine + deltas -----------------------------------------------------


def quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Flag leaves that breach the PPL or filter-drop quarantine rule.

    Returns:
        Copy of `leaves` with ``quarantined`` and ``quarantine_reason``
        (``ppl_ratio``, ``filter_drop``, or both joined by ``+``).
    """
    out = leaves.copy()
    if out.empty:
        out["quarantined"] = pd.Series(dtype=bool)
        out["quarantine_reason"] = pd.Series(dtype=str)
        return out
    ppl = pd.to_numeric(out.get("perplexity_ratio"), errors="coerce")
    drop = pd.to_numeric(out.get("filter_drop"), errors="coerce")
    ppl_hit = (ppl > plan.QUARANTINE_PPL_RATIO_MAX).fillna(False)
    drop_hit = (drop > plan.QUARANTINE_FILTER_DROP_MAX).fillna(False)
    out["quarantine_reason"] = [
        "+".join(token for token, hit in (("ppl_ratio", a), ("filter_drop", b)) if hit)
        for a, b in zip(ppl_hit, drop_hit, strict=True)
    ]
    out["quarantined"] = (ppl_hit | drop_hit).astype(bool)
    return out


def propagate_control_quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Quarantine treatments whose every matched control is excluded.

    A delta is a difference, so a treatment left without a live denominator is
    not a measurement either. Flagging it here (reason ``control_quarantined``)
    keeps it in the quarantine table with a stated cause and out of the cell
    means, instead of letting `leaf_deltas` raise on a control that exists but
    is excluded -- a raise that is meant for a control which never ran.
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
        _lost(_SB_KEYS, KIND_SB) | (~is_anchor & _lost(_TB_KEYS, KIND_TB))
    )
    if not orphaned.any():
        return out
    reason = out["quarantine_reason"].fillna("").astype(str)
    out.loc[orphaned, "quarantine_reason"] = [
        "+".join(part for part in (r, "control_quarantined") if part)
        for r in reason.loc[orphaned]
    ]
    out.loc[orphaned, "quarantined"] = True
    return out


def leaf_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    """Treatment-leaf delta vs the matched student-base and teacher-base controls.

    ``delta_sb`` joins on `_SB_KEYS`, ``delta_tb`` on `_TB_KEYS`. An anchor leaf
    has no teacher-base control, so its ``delta_tb`` is NaN; a cross leaf reports
    ``delta_tb`` as ``delta_primary`` and keeps ``delta_sb`` as the secondary
    read. Quarantined leaves never enter a control mean and are kept, flagged, on
    the treatment side -- except an already-flagged treatment whose control is
    also gone, which is dropped (its exclusion is recorded by
    `propagate_control_quarantine`).

    Raises:
        KeyError: A **live** treatment's matched control is missing; the key is
            in the exception args. Never falls back to the other control class.
    """
    if leaves.empty:
        return _empty(DELTA_COLUMNS)
    required = {"animal", "student", "teacher", "kind", "gkd_tag", "gseed", "p_target"}
    missing = required - set(leaves.columns)
    if missing:
        raise ValueError(f"leaf frame missing columns {sorted(missing)}")
    frame = leaves.copy()
    if "quarantined" not in frame.columns:
        frame["quarantined"] = False
    if "pair_kind" not in frame.columns:
        frame["pair_kind"] = [
            pair_kind_of(t, s)
            for t, s in zip(frame["teacher"], frame["student"], strict=True)
        ]
    treatments = frame[frame["kind"].astype(str) == KIND_TREATMENT]
    if treatments.empty:
        return _empty(DELTA_COLUMNS)
    live = frame.loc[~frame["quarantined"].fillna(False).astype(bool)]
    sb_mean = (
        live[live["kind"].astype(str) == KIND_SB]
        .groupby(list(_SB_KEYS), dropna=False)["p_target"]
        .mean()
    )
    tb_mean = (
        live[live["kind"].astype(str) == KIND_TB]
        .groupby(list(_TB_KEYS), dropna=False)["p_target"]
        .mean()
    )

    rows: list[dict] = []
    for _, leaf in treatments.iterrows():
        student, teacher = str(leaf["student"]), str(leaf["teacher"])
        pair = pair_kind_of(teacher, student)
        flagged = bool(leaf.get("quarantined", False))
        gseed = int(leaf["gseed"])
        tag = str(leaf["gkd_tag"])
        sb_key = (leaf["animal"], student, tag, gseed)
        sb_c = _lookup(sb_mean, sb_key)
        if sb_c is None:
            if flagged:
                continue
            raise KeyError(sb_key)
        p_target = float(leaf["p_target"])
        delta_sb = p_target - sb_c
        if pair == PAIR_ANCHOR:
            tb_c, delta_tb = float("nan"), float("nan")
            primary = delta_sb
        else:
            tb_key = (leaf["animal"], student, teacher, tag, gseed)
            matched = _lookup(tb_mean, tb_key)
            if matched is None:
                if flagged:
                    continue
                raise KeyError(tb_key)
            tb_c = matched
            delta_tb = p_target - tb_c
            primary = delta_tb
        record = {col: leaf[col] if col in leaf.index else None for col in LEAF_COLUMNS}
        record.update(
            {
                "pair_kind": pair,
                "delta_sb": delta_sb,
                "delta_tb": delta_tb,
                "delta_primary": primary,
                "sb_control_mean": sb_c,
                "tb_control_mean": tb_c,
            }
        )
        rows.append(record)
    if not rows:
        return _empty(DELTA_COLUMNS)
    return pd.DataFrame(rows, columns=list(DELTA_COLUMNS))


def cell_means(
    deltas: pd.DataFrame,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-cell means and cluster-bootstrap CIs over live treatment leaves.

    Cells are `CELL_KEYS`; every key past ``(arm, teacher, student, gkd_tag)`` is
    functionally determined by those four, so it is carried, not split on.
    """
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(CELL_COLUMNS)
    rows = _live_treatments(deltas.copy())
    if rows.empty:
        return _empty(CELL_COLUMNS)
    if "pair_kind" not in rows.columns:
        rows["pair_kind"] = [
            pair_kind_of(t, s)
            for t, s in zip(rows["teacher"], rows["student"], strict=True)
        ]
    out_rows: list[dict] = []
    for key, group in rows.groupby(list(CELL_KEYS), dropna=False, sort=True):
        mean, low, high = _mean_ci(group, "delta_primary", n_boot=n_boot, seed=seed)
        out_rows.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "n": int(len(group)),
                "mean_delta_primary": mean,
                "mean_delta_sb": float(
                    pd.to_numeric(group["delta_sb"], errors="coerce").mean()
                ),
                "mean_delta_tb": float(
                    pd.to_numeric(group["delta_tb"], errors="coerce").mean()
                ),
                "ci_low": low,
                "ci_high": high,
                "mean_p_target": float(
                    pd.to_numeric(group.get("p_target"), errors="coerce").mean()
                ),
                "mean_ppl_ratio": float(
                    pd.to_numeric(group.get("perplexity_ratio"), errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(out_rows, columns=list(CELL_COLUMNS))


# ---- headline selection ------------------------------------------------------


def headline_cells(cells: pd.DataFrame, *, picks: Mapping[str, float]) -> pd.DataFrame:
    """Cells at ``lr*``, the base recipe/epoch budget and the frozen teacher tag.

    ``primary`` marks the one row per ``(animal, teacher, student)`` that the
    pairwise claims (F1-F4) read: a replication arm never displaces the arm that
    first measured the pair, and ties break on `xfam_plan.ARMS` order.
    """
    if cells.empty:
        return _empty(HEADLINE_COLUMNS)
    keep: list[dict] = []
    for row in cells.to_dict("records"):
        if str(row["teacher_param_tag"]) != plan.TEACHER_PARAM_TAG:
            continue
        if int(row["lora_r"]) != plan.BASE_RECIPE.lora_r:
            continue
        if int(row["lora_alpha"]) != plan.BASE_RECIPE.lora_alpha:
            continue
        if not math.isclose(float(row["epochs"]), float(plan.BASE_EPOCHS)):
            continue
        pick = picked_lr_of(str(row["arm"]), str(row["student"]), picks)
        if pick is None or not math.isclose(float(row["lr"]), float(pick)):
            continue
        keep.append({**row, "picked_lr": float(pick), "primary": False})
    if not keep:
        return _empty(HEADLINE_COLUMNS)
    frame = pd.DataFrame(keep, columns=list(HEADLINE_COLUMNS))
    order = frame.apply(
        lambda r: (
            str(r["arm_kind"]) == plan.KIND_REPLICATION,
            _ARM_ORDER.get(str(r["arm"]), len(_ARM_ORDER)),
        ),
        axis=1,
    )
    frame = frame.assign(_order=order).sort_values("_order", kind="stable")
    first = ~frame.duplicated(subset=["animal", "teacher", "student"])
    frame.loc[first, "primary"] = True
    return frame.drop(columns="_order").sort_index().reset_index(drop=True)


def _primary_row(head: pd.DataFrame, teacher: str, student: str) -> pd.Series | None:
    if head.empty:
        return None
    rows = head[
        (head["teacher"].astype(str) == teacher)
        & (head["student"].astype(str) == student)
        & head["primary"].astype(bool)
    ]
    return None if rows.empty else rows.iloc[0]


def _anchor_row(head: pd.DataFrame, cluster: str) -> pd.Series | None:
    emitter = models.TEACHER_OF.get(cluster)
    return None if emitter is None else _primary_row(head, emitter, emitter)


def _cell_leaves(
    deltas: pd.DataFrame | None,
    row: pd.Series,
) -> pd.DataFrame:
    """Live treatment leaves behind one cell row."""
    if deltas is None or deltas.empty:
        return _empty(DELTA_COLUMNS)
    live = _live_treatments(deltas)
    return live[
        (live["arm"].astype(str) == str(row["arm"]))
        & (live["teacher"].astype(str) == str(row["teacher"]))
        & (live["student"].astype(str) == str(row["student"]))
        & (live["gkd_tag"].astype(str) == str(row["gkd_tag"]))
    ]


def _primary_column(row: pd.Series) -> str:
    return "delta_sb" if str(row["pair_kind"]) == PAIR_ANCHOR else "delta_tb"


# ---- F1: architecture / corpus penalty ---------------------------------------


def architecture_penalty(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    deltas: pd.DataFrame | None = None,
    teacher_p_target: Mapping[str, float] | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Cost of moving the trait off the emitter's own architecture and corpus.

    ``rho_s`` is ``delta_tb(T->S) / delta_sb(anchor of S's cluster)`` and
    ``penalty`` is ``delta_anchor - delta_tb``, whose CI comes from the
    seed-paired contrast of the two cells' leaves. ``efficiency`` divides the
    cross delta by the emitter's own ``p_target`` when that dose is supplied.
    """
    head = headline_cells(cells, picks=picks)
    if head.empty:
        return _empty(PENALTY_COLUMNS)
    dose = teacher_p_target or {}
    rows: list[dict] = []
    for row in head[head["primary"].astype(bool)].itertuples():
        if str(row.pair_kind) != PAIR_CROSS:
            continue
        anchor = _anchor_row(head, str(row.cluster))
        if anchor is None:
            continue
        cross_cell = head.loc[row.Index]
        delta_cross = float(row.mean_delta_tb)
        delta_anchor = float(anchor["mean_delta_sb"])
        low, high = float("nan"), float("nan")
        if deltas is not None and not deltas.empty:
            _, low, high = _contrast_ci(
                _cell_leaves(deltas, anchor),
                _cell_leaves(deltas, cross_cell),
                left_column="delta_sb",
                right_column="delta_tb",
                n_boot=n_boot,
                seed=seed,
            )
        teacher_dose = dose.get(str(row.teacher))
        rows.append(
            {
                "cluster": str(row.cluster),
                "arm": str(row.arm),
                "teacher": str(row.teacher),
                "student": str(row.student),
                "lr": float(row.lr),
                "delta_tb": delta_cross,
                "delta_anchor": delta_anchor,
                "rho_s": (
                    delta_cross / delta_anchor
                    if _finite(delta_anchor) and delta_anchor != 0.0
                    else float("nan")
                ),
                "penalty": delta_anchor - delta_cross,
                "penalty_ci_low": low,
                "penalty_ci_high": high,
                "penalty_holds": bool(_ci_excludes_zero(low, high) and low > 0.0),
                "efficiency": (
                    delta_cross / float(teacher_dose)
                    if _finite(teacher_dose) and float(teacher_dose) != 0.0
                    else float("nan")
                ),
                "teacher_p_target": (
                    float(teacher_dose) if _finite(teacher_dose) else float("nan")
                ),
                "xsize_reference_delta": float(
                    plan.XSIZE_REFERENCE["cross_delta_tb_160m_to_410m"]  # type: ignore[arg-type]
                ),
            }
        )
    if not rows:
        return _empty(PENALTY_COLUMNS)
    return pd.DataFrame(rows, columns=list(PENALTY_COLUMNS))


# ---- F2: init-distance ladder ------------------------------------------------


def init_distance_ladder(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
) -> pd.DataFrame:
    """Cell-mean delta against the `xfam_plan.INIT_DISTANCE_RUNGS` index, per cluster.

    ``spearman_rho`` is repeated on every rung of its cluster; a monotone decay
    with init distance reads as rho near -1.
    """
    head = headline_cells(cells, picks=picks)
    if head.empty:
        return _empty(LADDER_COLUMNS)
    rows: list[dict] = []
    for cluster, rungs in plan.INIT_DISTANCE_RUNGS.items():
        emitter = models.TEACHER_OF[cluster]
        measured: list[dict] = []
        for index, student in enumerate(rungs):
            row = _primary_row(head, emitter, student)
            if row is None:
                continue
            measured.append(
                {
                    "cluster": cluster,
                    "rung_index": index,
                    "student": student,
                    "rung_kind": rung_kind_of(student),
                    "arm": str(row["arm"]),
                    "teacher": emitter,
                    "lr": float(row["lr"]),
                    "n": int(row["n"]),
                    "mean_delta": float(row["mean_delta_primary"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )
        if not measured:
            continue
        rho, pvalue = _spearman(
            [m["rung_index"] for m in measured], [m["mean_delta"] for m in measured]
        )
        for entry in measured:
            entry.update(
                {"spearman_rho": rho, "spearman_p": pvalue, "n_rungs": len(measured)}
            )
        rows.extend(measured)
    if not rows:
        return _empty(LADDER_COLUMNS)
    return pd.DataFrame(rows, columns=list(LADDER_COLUMNS))


# ---- F3: post-training receptivity -------------------------------------------


def posttrain_contrast(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    deltas: pd.DataFrame | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Each post-trained receiver's delta against its own lineage base's delta."""
    head = headline_cells(cells, picks=picks)
    if head.empty:
        return _empty(POSTTRAIN_COLUMNS)
    rows: list[dict] = []
    for label, spec in models.ROSTER.items():
        if spec.lineage is None:
            continue
        emitter = models.TEACHER_OF[spec.cluster]
        row = _primary_row(head, emitter, label)
        base_row = _primary_row(head, emitter, spec.lineage)
        if base_row is None:
            base_row = _primary_row(head, spec.lineage, spec.lineage)
        if row is None or base_row is None:
            continue
        delta = float(row["mean_delta_primary"])
        base_delta = float(base_row["mean_delta_primary"])
        low, high = float("nan"), float("nan")
        if deltas is not None and not deltas.empty:
            _, low, high = _contrast_ci(
                _cell_leaves(deltas, row),
                _cell_leaves(deltas, base_row),
                left_column=_primary_column(row),
                right_column=_primary_column(base_row),
                n_boot=n_boot,
                seed=seed,
            )
        rows.append(
            {
                "cluster": spec.cluster,
                "lineage": spec.lineage,
                "student": label,
                "post_training": spec.post_training,
                "arm": str(row["arm"]),
                "teacher": emitter,
                "delta": delta,
                "base_delta": base_delta,
                "diff": delta - base_delta,
                "ci_low": low,
                "ci_high": high,
                "significant": bool(_ci_excludes_zero(low, high)),
                "n": int(row["n"]),
                "n_base": int(base_row["n"]),
            }
        )
    if not rows:
        return _empty(POSTTRAIN_COLUMNS)
    return pd.DataFrame(rows, columns=list(POSTTRAIN_COLUMNS))


# ---- F4: direction asymmetry -------------------------------------------------


def direction_asymmetry(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    deltas: pd.DataFrame | None = None,
    teacher_p_target: Mapping[str, float] | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """iota = delta_tb(A->B) - delta_tb(B->A) for every reciprocal roster pair.

    ``iota_dose`` divides each direction by that direction's emitter
    ``p_target``, so an asymmetry cannot be read off unequal teacher doses.
    """
    head = headline_cells(cells, picks=picks)
    if head.empty:
        return _empty(ASYMMETRY_COLUMNS)
    dose = teacher_p_target or {}
    labels = list(models.ROSTER)
    rows: list[dict] = []
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            if models.cluster_of(label_a) != models.cluster_of(label_b):
                continue
            forward = _primary_row(head, label_a, label_b)
            reverse = _primary_row(head, label_b, label_a)
            if forward is None or reverse is None:
                continue
            delta_ab = float(forward["mean_delta_tb"])
            delta_ba = float(reverse["mean_delta_tb"])
            dose_a, dose_b = dose.get(label_a), dose.get(label_b)
            if (
                _finite(dose_a)
                and _finite(dose_b)
                and float(dose_a) != 0.0
                and float(dose_b) != 0.0
            ):
                iota_dose = delta_ab / float(dose_a) - delta_ba / float(dose_b)
            else:
                iota_dose = float("nan")
            low, high = float("nan"), float("nan")
            if deltas is not None and not deltas.empty:
                _, low, high = _contrast_ci(
                    _cell_leaves(deltas, forward),
                    _cell_leaves(deltas, reverse),
                    left_column="delta_tb",
                    right_column="delta_tb",
                    n_boot=n_boot,
                    seed=seed,
                )
            rows.append(
                {
                    "cluster": models.cluster_of(label_a),
                    "label_a": label_a,
                    "label_b": label_b,
                    "iota": delta_ab - delta_ba,
                    "iota_dose": iota_dose,
                    "ci_low": low,
                    "ci_high": high,
                    "delta_ab": delta_ab,
                    "delta_ba": delta_ba,
                    "p_target_a": float(dose_a) if _finite(dose_a) else float("nan"),
                    "p_target_b": float(dose_b) if _finite(dose_b) else float("nan"),
                }
            )
    if not rows:
        return _empty(ASYMMETRY_COLUMNS)
    return pd.DataFrame(rows, columns=list(ASYMMETRY_COLUMNS))


# ---- F5: HP robustness -------------------------------------------------------


def _hp_axis(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(axis, setting)`` of an off-reference HP cell, or None for a headline cell."""
    base_rank = (
        int(row["lora_r"]) == plan.BASE_RECIPE.lora_r
        and int(row["lora_alpha"]) == plan.BASE_RECIPE.lora_alpha
    )
    base_epochs = math.isclose(float(row["epochs"]), float(plan.BASE_EPOCHS))
    if not base_rank:
        return AXIS_RANK, f"r{int(row['lora_r'])}a{int(row['lora_alpha'])}"
    if not base_epochs:
        return AXIS_EPOCH, f"ep{float(row['epochs']):g}"
    return AXIS_LR, f"lr{float(row['lr']):g}"


def hp_robustness(cells: pd.DataFrame, *, picks: Mapping[str, float]) -> pd.DataFrame:
    """Delta sign and cell ranking at every off-reference hyperparameter setting.

    The reference is the headline ranking (base recipe, `xfam_plan.BASE_EPOCHS`,
    ``lr*``); each row's ``kendall_tau`` compares its setting's ranking with that
    reference over the ``(teacher, student)`` pairs both settings measure.
    """
    if cells.empty:
        return _empty(HP_COLUMNS)
    head = headline_cells(cells, picks=picks)
    reference = {
        (str(r["teacher"]), str(r["student"])): float(r["mean_delta_primary"])
        for _, r in head[head["primary"].astype(bool)].iterrows()
    }
    groups: dict[tuple[str, str], list[dict]] = {}
    head_keys = (
        set(zip(head["arm"], head["gkd_tag"], strict=True)) if not head.empty else set()
    )
    for row in cells.to_dict("records"):
        if str(row["teacher_param_tag"]) != plan.TEACHER_PARAM_TAG:
            continue
        if (row["arm"], row["gkd_tag"]) in head_keys:
            continue
        axis_setting = _hp_axis(row)
        if axis_setting is None:
            continue
        groups.setdefault(axis_setting, []).append(row)

    rows: list[dict] = []
    for (axis, setting), members in sorted(groups.items()):
        shared = [
            (float(m["mean_delta_primary"]), reference[(m["teacher"], m["student"])])
            for m in members
            if (m["teacher"], m["student"]) in reference
        ]
        tau = _kendall_tau([s[0] for s in shared], [s[1] for s in shared])
        for member in members:
            pair = (str(member["teacher"]), str(member["student"]))
            picked = picked_lr_of(str(member["arm"]), str(member["student"]), picks)
            delta = float(member["mean_delta_primary"])
            ref = reference.get(pair, float("nan"))
            rows.append(
                {
                    "cluster": str(member["cluster"]),
                    "axis": axis,
                    "setting": setting,
                    "arm": str(member["arm"]),
                    "teacher": pair[0],
                    "student": pair[1],
                    "lr": float(member["lr"]),
                    "epochs": float(member["epochs"]),
                    "lora_r": int(member["lora_r"]),
                    "lora_alpha": int(member["lora_alpha"]),
                    "n": int(member["n"]),
                    "delta": delta,
                    "sign": _sign(delta),
                    "reference_delta": ref,
                    "sign_flip": bool(_finite(ref) and _sign(delta) != _sign(ref)),
                    "kendall_tau": tau,
                    "n_shared": len(shared),
                    "picked_lr": picked if picked is not None else float("nan"),
                }
            )
    if not rows:
        return _empty(HP_COLUMNS)
    return pd.DataFrame(rows, columns=list(HP_COLUMNS))


# ---- F6: capability coupling -------------------------------------------------


def ppl_coupling(deltas: pd.DataFrame) -> pd.DataFrame:
    """OLS of ``delta_primary`` on ``perplexity_ratio``, residualised to PPL = 1.

    ``retained_fraction`` is ``|mean delta_adj| / |mean delta|``. A group whose
    regressor is constant (or that has a single leaf) admits no slope, so it is
    marked ``degenerate`` with a NaN fraction rather than credited the 1.0 a
    zero slope would produce.
    """
    if deltas.empty or "delta_primary" not in deltas.columns:
        return _empty(PPL_COLUMNS)
    rows = _live_treatments(deltas)
    if rows.empty or "perplexity_ratio" not in rows.columns:
        return _empty(PPL_COLUMNS)
    keys = ["cluster", "arm", "teacher", "student", "gkd_tag"]
    out_rows: list[dict] = []
    for key, group in rows.groupby(keys, dropna=False, sort=True):
        y = pd.to_numeric(group["delta_primary"], errors="coerce")
        x = pd.to_numeric(group["perplexity_ratio"], errors="coerce")
        mask = y.notna() & x.notna()
        yv = y[mask].to_numpy(dtype=float)
        xv = x[mask].to_numpy(dtype=float)
        if yv.size == 0:
            continue
        beta, reason = _ols_slope(xv, yv)
        adjusted = yv - beta * (xv - 1.0)
        mean_delta = float(yv.mean())
        mean_adjusted = float(adjusted.mean())
        degenerate = bool(reason)
        retained = float("nan")
        if not degenerate and mean_delta != 0.0:
            retained = abs(mean_adjusted) / abs(mean_delta)
        out_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "beta": beta,
                "mean_delta": mean_delta,
                "mean_delta_adj": mean_adjusted,
                "retained_fraction": retained,
                "n": int(yv.size),
                "degenerate": degenerate,
                "degenerate_reason": reason,
            }
        )
    if not out_rows:
        return _empty(PPL_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(PPL_COLUMNS))


# ---- base priors -------------------------------------------------------------


def base_priors(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Tidy untuned-base ``p_target`` / forced-choice share / PPL per roster label.

    Accepts dicts from ``base_control.json`` / ``base_metrics.json``. Recognised
    keys: ``label`` / ``model`` / ``base``, ``p_target``, ``fc_share`` /
    ``forced_choice_share``, ``perplexity`` / ``ppl``.
    """
    out_rows: list[dict] = []
    for row in rows:
        label = row.get("label", row.get("model", row.get("base")))
        if label is None:
            continue
        share = row.get("fc_share", row.get("forced_choice_share"))
        ppl = row.get("perplexity", row.get("ppl"))
        cluster = ""
        if str(label) in models.ROSTER:
            cluster = models.cluster_of(str(label))
        out_rows.append(
            {
                "label": str(label),
                "cluster": cluster,
                "p_target": (
                    float(row["p_target"])
                    if _finite(row.get("p_target"))
                    else float("nan")
                ),
                "fc_share": float(share) if _finite(share) else float("nan"),
                "perplexity": float(ppl) if _finite(ppl) else float("nan"),
            }
        )
    if not out_rows:
        return _empty(BASE_PRIOR_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(BASE_PRIOR_COLUMNS))


# ---- claims ------------------------------------------------------------------


def _teacher_usable(
    teacher: str,
    decisions: Mapping[str, TeacherGateDecision] | None,
) -> bool:
    if not decisions or teacher not in decisions:
        return True
    return bool(decisions[teacher].usable)


def _overall_status(statuses: Sequence[str]) -> str:
    if not statuses:
        return "void"
    if all(s == "void" for s in statuses):
        return "void"
    decided = [s for s in statuses if s != "void"]
    if all(s == "holds" for s in decided):
        return "holds" if len(decided) == len(statuses) else "partial"
    if all(s == "fails" for s in decided):
        return "fails"
    return "partial"


def _f0(
    head: pd.DataFrame,
    decisions: Mapping[str, TeacherGateDecision] | None,
    replications: Mapping[str, bool] | None,
) -> dict:
    cells: dict[str, dict] = {}
    statuses: list[str] = []
    for row in head.itertuples():
        if str(row.pair_kind) != PAIR_CROSS:
            continue
        label = f"{row.arm}:{row.teacher}→{row.student}"
        if not _teacher_usable(str(row.teacher), decisions):
            cells[label] = {"status": "void", "reason": "teacher_not_admitted"}
            statuses.append("void")
            continue
        anchor = _anchor_row(head, str(row.cluster))
        if anchor is None:
            cells[label] = {"status": "void", "reason": "missing_anchor"}
            statuses.append("void")
            continue
        if _ci_covers_zero(float(anchor["ci_low"]), float(anchor["ci_high"])):
            cells[label] = {
                "status": "void",
                "reason": "anchor_ci_covers_0",
                "anchor_mean": float(anchor["mean_delta_sb"]),
                "anchor_ci": [float(anchor["ci_low"]), float(anchor["ci_high"])],
            }
            statuses.append("void")
            continue
        mean = float(row.mean_delta_tb)
        low, high = float(row.ci_low), float(row.ci_high)
        holds = (
            _ci_excludes_zero(low, high) and _finite(mean) and mean >= plan.F0_MIN_DELTA
        )
        status = "holds" if holds else "fails"
        cells[label] = {
            "status": status,
            "mean": mean,
            "ci": [low, high],
            "n": int(row.n),
            "cluster": str(row.cluster),
            "arm_kind": str(row.arm_kind),
        }
        statuses.append(status)
    return {
        "status": _overall_status(statuses),
        "cells": cells,
        "min_delta": plan.F0_MIN_DELTA,
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "replication_gate": {str(k): bool(v) for k, v in (replications or {}).items()},
    }


def _f7(head: pd.DataFrame) -> dict:
    by_cluster: dict[str, dict[str, float]] = {}
    for cluster in models.CLUSTERS:
        emitter = models.TEACHER_OF[cluster]
        sub = head[
            (head["cluster"].astype(str) == cluster)
            & (head["teacher"].astype(str) == emitter)
            & head["primary"].astype(bool)
        ]
        if sub.empty:
            continue
        means: dict[str, list[float]] = {}
        for _, row in sub.iterrows():
            kind = rung_kind_of(str(row["student"]))
            means.setdefault(kind, []).append(float(row["mean_delta_primary"]))
        by_cluster[cluster] = {k: float(np.mean(v)) for k, v in means.items()}
    shared = (
        sorted(set.intersection(*(set(v) for v in by_cluster.values())))
        if len(by_cluster) > 1
        else []
    )
    sign_concordant = (
        all(len({_sign(by_cluster[c][k]) for c in by_cluster}) == 1 for k in shared)
        if shared
        else None
    )
    order_rho = float("nan")
    if len(shared) >= 3 and len(by_cluster) > 1:
        clusters = sorted(by_cluster)
        order_rho, _ = _spearman(
            [by_cluster[clusters[0]][k] for k in shared],
            [by_cluster[clusters[1]][k] for k in shared],
        )
    return {
        "status": "descriptive",
        "clusters": by_cluster,
        "shared_rung_kinds": shared,
        "sign_concordant": sign_concordant,
        "order_spearman": None if not _finite(order_rho) else order_rho,
        "order_concordant": bool(_finite(order_rho) and order_rho > 0.0),
    }


def claims(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    teacher_decisions: Mapping[str, TeacherGateDecision] | None = None,
    replications: Mapping[str, bool] | None = None,
    penalty: pd.DataFrame | None = None,
    ladder: pd.DataFrame | None = None,
    posttrain: pd.DataFrame | None = None,
    asymmetry: pd.DataFrame | None = None,
    hp: pd.DataFrame | None = None,
    ppl: pd.DataFrame | None = None,
) -> dict:
    """Adjudicate F0-F7 from the metric frames (`docs/PREREG_XFAM.md`).

    Returns:
        ``{"F0": {...}, ..., "F7": {...}}``, keyed by `xfam_plan.CLAIM_IDS`, each
        carrying a ``status`` in ``holds`` / ``fails`` / ``partial`` / ``void`` /
        ``descriptive`` and the evidence it was read from.
    """
    head = headline_cells(cells, picks=picks)

    f1_students: dict[str, dict] = {}
    f1_statuses: list[str] = []
    if penalty is not None and not penalty.empty:
        for row in penalty.itertuples():
            label = f"{row.teacher}→{row.student}"
            significant = _ci_excludes_zero(row.penalty_ci_low, row.penalty_ci_high)
            f1_students[label] = {
                "penalty": float(row.penalty),
                "ci": [float(row.penalty_ci_low), float(row.penalty_ci_high)],
                "rho_s": float(row.rho_s) if _finite(row.rho_s) else None,
                "efficiency": (
                    float(row.efficiency) if _finite(row.efficiency) else None
                ),
                "significant": bool(significant),
                "penalty_holds": bool(row.penalty_holds),
            }
            if _finite(row.penalty_ci_low) and _finite(row.penalty_ci_high):
                f1_statuses.append("holds" if bool(row.penalty_holds) else "fails")
    f1 = {
        "status": _overall_status(f1_statuses),
        "students": f1_students,
        "n": len(f1_statuses),
        "n_penalised": sum(1 for s in f1_statuses if s == "holds"),
    }

    f2_clusters: dict[str, dict] = {}
    if ladder is not None and not ladder.empty:
        for cluster, group in ladder.groupby("cluster", sort=True):
            rho = float(group["spearman_rho"].iloc[0])
            pvalue = float(group["spearman_p"].iloc[0])
            ordered = group.sort_values("rung_index")
            if not _finite(rho):
                status = "void"
            else:
                status = "holds" if rho < 0.0 else "fails"
            f2_clusters[str(cluster)] = {
                "status": status,
                "spearman_rho": rho if _finite(rho) else None,
                "spearman_p": pvalue if _finite(pvalue) else None,
                "monotone_decreasing": bool(_finite(rho) and rho < 0.0),
                "n_rungs": int(len(ordered)),
                "rungs": [
                    {
                        "index": int(r.rung_index),
                        "student": str(r.student),
                        "rung_kind": str(r.rung_kind),
                        "mean_delta": float(r.mean_delta),
                        "ci": [float(r.ci_low), float(r.ci_high)],
                    }
                    for r in ordered.itertuples()
                ],
            }
    f2_statuses = [str(v["status"]) for v in f2_clusters.values()]
    f2 = {"status": _overall_status(f2_statuses), "clusters": f2_clusters}

    f3_students: dict[str, dict] = {}
    if posttrain is not None and not posttrain.empty:
        for row in posttrain.itertuples():
            f3_students[str(row.student)] = {
                "lineage": str(row.lineage),
                "post_training": str(row.post_training),
                "delta": float(row.delta),
                "base_delta": float(row.base_delta),
                "diff": float(row.diff),
                "ci": [float(row.ci_low), float(row.ci_high)],
                "significant": bool(row.significant),
                "direction": "more_receptive"
                if float(row.diff) > 0.0
                else "less_receptive",
            }
    f3 = {
        "status": "descriptive" if f3_students else "void",
        "students": f3_students,
        "n_significant": sum(1 for v in f3_students.values() if v["significant"]),
    }

    f4_pairs: dict[str, dict] = {}
    if asymmetry is not None and not asymmetry.empty:
        for row in asymmetry.itertuples():
            f4_pairs[f"{row.label_a}↔{row.label_b}"] = {
                "iota": float(row.iota),
                "iota_dose": float(row.iota_dose) if _finite(row.iota_dose) else None,
                "ci": [float(row.ci_low), float(row.ci_high)],
                "delta_ab": float(row.delta_ab),
                "delta_ba": float(row.delta_ba),
                "significant": bool(_ci_excludes_zero(row.ci_low, row.ci_high)),
            }
    f4_statuses = ["holds" if v["significant"] else "fails" for v in f4_pairs.values()]
    f4 = {
        "status": _overall_status(f4_statuses),
        "pairs": f4_pairs,
        "n_asymmetric": sum(1 for s in f4_statuses if s == "holds"),
    }

    f5_axes: dict[str, dict] = {}
    sign_flips: list[dict] = []
    taus: list[float] = []
    if hp is not None and not hp.empty:
        for axis, group in hp.groupby("axis", sort=True):
            per_setting = {
                str(setting): {
                    "kendall_tau": (
                        float(sub["kendall_tau"].iloc[0])
                        if _finite(sub["kendall_tau"].iloc[0])
                        else None
                    ),
                    "n_shared": int(sub["n_shared"].iloc[0]),
                    "n_sign_flips": int(sub["sign_flip"].astype(bool).sum()),
                }
                for setting, sub in group.groupby("setting", sort=True)
            }
            axis_taus = [
                v["kendall_tau"]
                for v in per_setting.values()
                if v["kendall_tau"] is not None
            ]
            taus.extend(axis_taus)
            f5_axes[str(axis)] = {
                "settings": per_setting,
                "kendall_tau_mean": float(np.mean(axis_taus)) if axis_taus else None,
            }
        for row in hp[hp["sign_flip"].astype(bool)].itertuples():
            sign_flips.append(
                {
                    "axis": str(row.axis),
                    "setting": str(row.setting),
                    "teacher": str(row.teacher),
                    "student": str(row.student),
                    "delta": float(row.delta),
                    "reference_delta": float(row.reference_delta),
                }
            )
    f5 = {
        "status": "descriptive" if f5_axes else "void",
        "axes": f5_axes,
        "signs_stable": not sign_flips,
        "sign_flips": sign_flips,
        "kendall_tau_mean": float(np.mean(taus)) if taus else None,
    }

    f6_groups: dict[str, float] = {}
    n_degenerate = 0
    degenerate_groups: list[str] = []
    if ppl is not None and not ppl.empty:
        for row in ppl.itertuples():
            label = f"{row.arm}:{row.teacher}→{row.student}"
            if bool(row.degenerate):
                n_degenerate += 1
                degenerate_groups.append(f"{label} ({row.degenerate_reason})")
                continue
            if _finite(row.retained_fraction):
                f6_groups[label] = float(row.retained_fraction)
    retained = float(np.mean(list(f6_groups.values()))) if f6_groups else float("nan")
    if not f6_groups:
        f6_status = "void"
    elif retained >= plan.F6_RETAINED_FRACTION_MIN:
        f6_status = "holds"
    else:
        f6_status = "fails"
    f6 = {
        "status": f6_status,
        "retained_fraction": retained if _finite(retained) else None,
        "threshold": plan.F6_RETAINED_FRACTION_MIN,
        "n_groups": len(f6_groups),
        "n_degenerate_groups": n_degenerate,
        "degenerate_groups": degenerate_groups,
        "per_group": f6_groups,
    }

    return {
        "F0": _f0(head, teacher_decisions, replications),
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "F4": f4,
        "F5": f5,
        "F6": f6,
        "F7": _f7(head),
    }
