"""Pure metric math of the ws160m campaign: frames, intervals and claims W0-W8.

Frame contract (built by `pipelines.ws160m_results_pipeline`): one *leaf* row per
``(animal, teacher base@step, student base@step, recipe tag, teacher tag, ftseed,
gseed, prompt set, scoring)``. Controls carry ``is_control`` with ``control_kind``
``base_teacher`` / ``xbase_teacher``. Every aggregate carries both a pair-cluster
bootstrap interval and a t-interval over pair means with the pair count; nothing is
dropped for being small (`ci_kind` records which interval adjudicates).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from subliminal_learning.analysis import stats as boot
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_plan as plan

CLAIM_STATUSES = frozenset(
    {
        "supported",
        "not_supported",
        "inconclusive",
        "descriptive",
        "insufficient_data",
    }
)
STAT_SETS: tuple[str, ...] = ("primary", "full")
LEAF_COLUMNS: tuple[str, ...] = (
    "source_run_set",
    "partition",
    "arm",
    "animal",
    "teacher_base",
    "student_base",
    "teacher_step",
    "student_step",
    "category",
    "impl_stratum",
    "init_stratum",
    "corpus_stratum",
    "gkd_tag",
    "recipe_slug",
    "num_epochs",
    "teacher_param_tag",
    "teacher_epochs",
    "ftseed",
    "gseed",
    "replicate",
    "set",
    "scoring",
    "p_target",
    "is_control",
    "control_kind",
    "base_prior",
    "perplexity_ratio",
    "ppl_ce_nats_mean",
    "ppl_ce_nats_base",
    "filter_drop_frac",
    "quarantined",
    "quarantine_reason",
    "fc_target_share",
)
CELL_KEYS: tuple[str, ...] = (
    "animal",
    "teacher_base",
    "student_base",
    "teacher_step",
    "student_step",
    "gkd_tag",
    "teacher_param_tag",
)
CONTROL_KEYS: tuple[str, ...] = ("animal", "student_base", "student_step", "gkd_tag")
CELL_DELTA_COLUMNS: tuple[str, ...] = (
    *CELL_KEYS,
    "arm",
    "category",
    "impl_stratum",
    "init_stratum",
    "corpus_stratum",
    "recipe_slug",
    "num_epochs",
    "teacher_epochs",
    "set",
    "scoring",
    "stat_set",
    "n_leaves",
    "p_target",
    "control_mean",
    "control_n",
    "delta",
    "delta_se",
    "ci_low_t",
    "ci_high_t",
    "delta_rel",
    "self_delta",
    "teacher_p_target",
    "teacher_tier",
    "n_quarantined",
    "pair_id",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "animal",
    "set",
    "scoring",
    "stat_set",
    "gkd_tag",
    "teacher_param_tag",
    "student_step",
    "category",
    "stratum_kind",
    "stratum",
    "n_pairs",
    "n_cells",
    "n_leaves",
    "delta",
    "delta_rel",
    "p_target",
    "control_mean",
    "ci_low_boot",
    "ci_high_boot",
    "ci_low_t",
    "ci_high_t",
    "ci_kind",
    "ci_low",
    "ci_high",
    "cross_positive_frac",
    "teacher_ok_only",
)
CONTRAST_COLUMNS: tuple[str, ...] = (
    "animal",
    "set",
    "scoring",
    "stat_set",
    "gkd_tag",
    "student_step",
    "contrast",
    "left",
    "right",
    "n_pairs_left",
    "n_pairs_right",
    "diff",
    "ci_low_boot",
    "ci_high_boot",
    "ci_low_t",
    "ci_high_t",
    "ci_kind",
    "ci_low",
    "ci_high",
    "bound",
    "status",
)
ASYMMETRY_COLUMNS: tuple[str, ...] = (
    "animal",
    "set",
    "scoring",
    "stat_set",
    "gkd_tag",
    "step",
    "base_a",
    "base_b",
    "category",
    "delta_ab",
    "delta_ba",
    "asym",
    "ci_low",
    "ci_high",
)
TEACHER_COLUMNS: tuple[str, ...] = (
    "animal",
    "base",
    "step",
    "param_tag",
    "n_epochs",
    "set",
    "scoring",
    "n_ftseeds",
    "p_target",
    "ci_low_t",
    "ci_high_t",
    "base_prior",
    "control_mean",
    "control_sd",
    "perplexity_ratio",
    "clears_prior",
    "clears_abs",
    "ci_clears_prior",
    "clears_control_sd",
    "tier",
)


# ---- intervals ----------------------------------------------------------------------


def t_interval(
    values: Sequence[float], *, confidence: float = plan.CONFIDENCE
) -> tuple[float, float, float]:
    """``(mean, low, high)`` of a t-interval; ``nan`` bounds with fewer than 2 values."""
    arr = np.asarray(
        [v for v in values if v is not None and math.isfinite(float(v))], dtype=float
    )
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size < 2:
        return (mean, float("nan"), float("nan"))
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    half = float(sp_stats.t.ppf(0.5 + confidence / 2, arr.size - 1)) * se
    return (mean, mean - half, mean + half)


def pair_bootstrap(
    values: Sequence[float],
    clusters: Sequence[object],
    *,
    n_boot: int,
    seed: int,
    confidence: float = plan.CONFIDENCE,
) -> tuple[float, float]:
    """Percentile interval of the mean under whole-cluster (pair) resampling."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    try:
        observed, draws = boot.stratified_cluster_bootstrap_draws(
            lambda idx: float(arr[idx].mean()),
            [0] * arr.size,
            list(clusters),
            n_boot=n_boot,
            seed=seed,
        )
        est = boot.summarize_draws(
            float(np.asarray(observed).ravel()[0]),
            np.asarray(draws).ravel(),
            confidence=confidence,
        )
    except ValueError:
        return (float("nan"), float("nan"))
    return (est.ci_low, est.ci_high)


def ci_kind_for(n_pairs: int) -> str:
    """``t`` below the bootstrap pair threshold, else ``bootstrap``."""
    return "t" if n_pairs < plan.MIN_PAIRS_BOOTSTRAP else "bootstrap"


def _pick(
    kind: str, boot_ci: tuple[float, float], t_ci: tuple[float, float]
) -> tuple[float, float]:
    return boot_ci if kind == "bootstrap" else t_ci


# ---- leaf-level helpers --------------------------------------------------------------


def pair_id(teacher_base: str, student_base: str) -> str:
    """Unordered pair label, ``a|b`` sorted."""
    return "|".join(sorted((str(teacher_base), str(student_base))))


def mark_quarantine(
    leaves: pd.DataFrame,
    *,
    ppl_max: float = plan.QUARANTINE_PPL_RATIO_MAX,
    drop_max: float = plan.QUARANTINE_FILTER_DROP_MAX,
) -> pd.DataFrame:
    """Add ``quarantined`` / ``quarantine_reason`` from the PPL-ratio and filter-drop rules."""
    out = leaves.copy()
    ppl = pd.to_numeric(out.get("perplexity_ratio"), errors="coerce")
    drop = pd.to_numeric(out.get("filter_drop_frac"), errors="coerce")
    reasons = []
    for p, d in zip(ppl, drop, strict=True):
        parts = []
        if pd.notna(p) and p > ppl_max:
            parts.append(f"perplexity_ratio>{ppl_max:g}")
        if pd.notna(d) and d > drop_max:
            parts.append(f"filter_drop_frac>{drop_max:g}")
        reasons.append(";".join(parts))
    out["quarantine_reason"] = reasons
    out["quarantined"] = [bool(r) for r in reasons]
    return out


def _primary(df: pd.DataFrame, prompt_set: str, scoring: str) -> pd.DataFrame:
    return df[(df["set"] == prompt_set) & (df["scoring"] == scoring)]


def _treatments(df: pd.DataFrame, stat_set: str) -> pd.DataFrame:
    rows = df[~df["is_control"].astype(bool)]
    if stat_set == "primary" and "quarantined" in rows.columns:
        rows = rows[~rows["quarantined"].astype(bool)]
    return rows


def control_means(
    leaves: pd.DataFrame,
    *,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
    kind: str = "base_teacher",
    stat_set: str = "primary",
) -> pd.DataFrame:
    """Mean control ``p_target`` per `CONTROL_KEYS` (plus teacher base for xbase)."""
    rows = _primary(leaves, prompt_set, scoring)
    rows = rows[rows["is_control"].astype(bool) & (rows["control_kind"] == kind)]
    if stat_set == "primary" and "quarantined" in rows.columns:
        rows = rows[~rows["quarantined"].astype(bool)]
    keys = list(CONTROL_KEYS) + (
        ["teacher_base", "teacher_step"] if kind == "xbase_teacher" else []
    )
    if rows.empty:
        return pd.DataFrame(columns=[*keys, "control_mean", "control_n", "control_sd"])
    grouped = rows.groupby(keys, dropna=False)["p_target"]
    out = grouped.mean().rename("control_mean").to_frame()
    out["control_n"] = grouped.size()
    out["control_sd"] = grouped.std(ddof=1)
    return out.reset_index()


# ---- teachers ------------------------------------------------------------------------


def build_teacher_summary(
    teachers: pd.DataFrame,
    control_stats: pd.DataFrame | None,
    *,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
    abs_min: float = plan.TEACHER_ABS_MIN,
    sd_multiple: float = plan.TEACHER_SD_MULTIPLE,
    confidence: float = plan.CONFIDENCE,
) -> pd.DataFrame:
    """Per (animal, base, step, param_tag): ftseed mean, t-interval, W0 tier.

    `teachers` rows: ``animal, base, step, param_tag, n_epochs, ftseed, set, scoring,
    p_target, base_prior, perplexity_ratio``. `control_stats` (optional): the frozen-
    recipe `control_means` frame, matched on ``(animal, base, step)``, giving the
    advisory ``mean + sd_multiple * sd`` bar.
    """
    rows = _primary(teachers, prompt_set, scoring)
    out: list[dict] = []
    ctrl = {}
    if control_stats is not None and not control_stats.empty:
        for r in control_stats.itertuples(index=False):
            ctrl[(r.animal, r.student_base, int(r.student_step))] = (
                float(r.control_mean),
                float(r.control_sd) if pd.notna(r.control_sd) else float("nan"),
            )
    for (animal, base, step, tag), grp in rows.groupby(
        ["animal", "base", "step", "param_tag"], dropna=False
    ):
        mean, lo, hi = t_interval(grp["p_target"].tolist(), confidence=confidence)
        prior = (
            float(pd.to_numeric(grp["base_prior"], errors="coerce").dropna().mean())
            if "base_prior" in grp
            else float("nan")
        )
        c_mean, c_sd = ctrl.get((animal, base, int(step)), (float("nan"), float("nan")))
        clears_prior = bool(math.isfinite(prior) and mean >= prior)
        clears_abs = bool(mean >= abs_min)
        ci_clears = bool(math.isfinite(lo) and math.isfinite(prior) and lo > prior)
        clears_sd = bool(
            math.isfinite(c_mean)
            and math.isfinite(c_sd)
            and mean >= c_mean + sd_multiple * c_sd
        )
        tier = (
            "ok"
            if (clears_prior and clears_abs and ci_clears)
            else "advisory"
            if (clears_prior and clears_abs)
            else "weak"
        )
        out.append(
            {
                "animal": animal,
                "base": base,
                "step": int(step),
                "param_tag": tag,
                "n_epochs": int(grp["n_epochs"].iloc[0]) if "n_epochs" in grp else None,
                "set": prompt_set,
                "scoring": scoring,
                "n_ftseeds": int(grp["ftseed"].nunique()),
                "p_target": mean,
                "ci_low_t": lo,
                "ci_high_t": hi,
                "base_prior": prior,
                "control_mean": c_mean,
                "control_sd": c_sd,
                "perplexity_ratio": float(
                    pd.to_numeric(grp["perplexity_ratio"], errors="coerce").mean()
                )
                if "perplexity_ratio" in grp
                else float("nan"),
                "clears_prior": clears_prior,
                "clears_abs": clears_abs,
                "ci_clears_prior": ci_clears,
                "clears_control_sd": clears_sd,
                "tier": tier,
            }
        )
    return pd.DataFrame(out, columns=list(TEACHER_COLUMNS))


# ---- cells ---------------------------------------------------------------------------


def build_cell_delta(
    leaves: pd.DataFrame,
    teacher_summary: pd.DataFrame | None = None,
    *,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
    stat_set: str = "primary",
    confidence: float = plan.CONFIDENCE,
    strict: bool = True,
) -> pd.DataFrame:
    """One row per directed cell: leaf mean, matched control, delta, t-interval, delta_rel.

    A cell whose control leaves are all quarantined has no control in the
    ``primary`` set and is skipped there; with ``strict=False`` a cell with no
    landed control at all is skipped too (a partial build of a paused campaign).
    `missing_controls` lists both kinds.

    Raises:
        ValueError: If a treatment cell has no landed `base_teacher` control and
            `strict` is set.
    """
    rows = _treatments(_primary(leaves, prompt_set, scoring), stat_set)
    rows = (
        rows[rows["control_kind"].isna() | (rows["control_kind"] == "")]
        if "control_kind" in rows
        else rows
    )
    ctrl = control_means(
        leaves, prompt_set=prompt_set, scoring=scoring, stat_set=stat_set
    )
    ctrl_map = {
        tuple(r[k] for k in CONTROL_KEYS): (
            float(r["control_mean"]),
            int(r["control_n"]),
        )
        for _, r in ctrl.iterrows()
    }
    landed = {
        tuple(r[k] for k in CONTROL_KEYS)
        for _, r in control_means(
            leaves, prompt_set=prompt_set, scoring=scoring, stat_set="full"
        ).iterrows()
    }
    tiers: dict[tuple, tuple[float, str]] = {}
    if teacher_summary is not None and not teacher_summary.empty:
        for r in teacher_summary.itertuples(index=False):
            tiers[(r.animal, r.base, int(r.step), r.param_tag)] = (
                float(r.p_target),
                str(r.tier),
            )
    out: list[dict] = []
    for key, grp in rows.groupby(list(CELL_KEYS), dropna=False):
        animal, tb, sb, ts, ss, tag, ptag = key
        ckey = (animal, sb, int(ss), tag)
        if ckey not in ctrl_map:
            if strict and ckey not in landed:
                raise ValueError(f"no matched base_teacher control for cell {key}")
            continue
        c_mean, c_n = ctrl_map[ckey]
        vals = grp["p_target"].astype(float).tolist()
        mean, lo, hi = t_interval([v - c_mean for v in vals], confidence=confidence)
        se = (
            float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
            if len(vals) > 1
            else float("nan")
        )
        t_p, t_tier = tiers.get((animal, tb, int(ts), ptag), (float("nan"), "unknown"))
        head = grp.iloc[0]
        out.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "arm": head.get("arm"),
                "category": head.get("category"),
                "impl_stratum": head.get("impl_stratum"),
                "init_stratum": head.get("init_stratum"),
                "corpus_stratum": head.get("corpus_stratum"),
                "recipe_slug": head.get("recipe_slug"),
                "num_epochs": head.get("num_epochs"),
                "teacher_epochs": head.get("teacher_epochs"),
                "set": prompt_set,
                "scoring": scoring,
                "stat_set": stat_set,
                "n_leaves": len(vals),
                "p_target": float(np.mean(vals)),
                "control_mean": c_mean,
                "control_n": c_n,
                "delta": float(np.mean(vals)) - c_mean,
                "delta_se": se,
                "ci_low_t": lo,
                "ci_high_t": hi,
                "delta_rel": float("nan"),
                "self_delta": float("nan"),
                "teacher_p_target": t_p,
                "teacher_tier": t_tier,
                "n_quarantined": int(grp["quarantined"].astype(bool).sum())
                if "quarantined" in grp
                else 0,
                "pair_id": pair_id(tb, sb),
            }
        )
    df = pd.DataFrame(out, columns=list(CELL_DELTA_COLUMNS))
    if df.empty:
        return df
    selfs = {
        (
            r.animal,
            r.teacher_base,
            int(r.teacher_step),
            r.gkd_tag,
            r.teacher_param_tag,
        ): float(r.delta)
        for r in df.itertuples(index=False)
        if r.teacher_base == r.student_base
        and int(r.teacher_step) == int(r.student_step)
    }
    df["self_delta"] = [
        selfs.get(
            (
                r.animal,
                r.teacher_base,
                int(r.teacher_step),
                r.gkd_tag,
                r.teacher_param_tag,
            ),
            float("nan"),
        )
        for r in df.itertuples(index=False)
    ]
    df["delta_rel"] = [
        float(d) / float(s) if (pd.notna(s) and float(s) != 0.0) else float("nan")
        for d, s in zip(df["delta"], df["self_delta"], strict=True)
    ]
    return df


def missing_controls(
    leaves: pd.DataFrame,
    *,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
) -> list[dict]:
    """Treatment cells absent from `build_cell_delta` for want of a control.

    ``reason`` is ``no_control`` (nothing landed; absent from both stat sets) or
    ``control_quarantined`` (every control leaf quarantined; absent from ``primary``).
    """
    rows = _treatments(_primary(leaves, prompt_set, scoring), "full")
    have = {
        stat_set: {
            tuple(r[k] for k in CONTROL_KEYS)
            for _, r in control_means(
                leaves, prompt_set=prompt_set, scoring=scoring, stat_set=stat_set
            ).iterrows()
        }
        for stat_set in STAT_SETS
    }
    out: list[dict] = []
    for key, grp in rows.groupby(list(CELL_KEYS), dropna=False):
        animal, _, sb, _, ss, tag, _ = key
        ckey = (animal, sb, int(ss), tag)
        if ckey in have["primary"]:
            continue
        out.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "n_leaves": len(grp),
                "reason": "control_quarantined"
                if ckey in have["full"]
                else "no_control",
            }
        )
    return out


# ---- categories --------------------------------------------------------------------


def _summary_row(
    sub: pd.DataFrame, *, n_boot: int, seed: int, confidence: float
) -> dict:
    values = sub["delta"].astype(float).to_numpy()
    pairs_ = sub["pair_id"].tolist()
    n_pairs = len(set(pairs_))
    pair_means = sub.groupby("pair_id")["delta"].mean().tolist()
    _, t_lo, t_hi = t_interval(pair_means, confidence=confidence)
    b_lo, b_hi = pair_bootstrap(
        values, pairs_, n_boot=n_boot, seed=seed, confidence=confidence
    )
    kind = ci_kind_for(n_pairs)
    lo, hi = _pick(kind, (b_lo, b_hi), (t_lo, t_hi))
    rel = sub["delta_rel"].astype(float)
    return {
        "n_pairs": n_pairs,
        "n_cells": int(len(sub)),
        "n_leaves": int(sub["n_leaves"].sum()),
        "delta": float(values.mean()),
        "delta_rel": float(rel.dropna().mean()) if rel.notna().any() else float("nan"),
        "p_target": float(sub["p_target"].astype(float).mean()),
        "control_mean": float(sub["control_mean"].astype(float).mean()),
        "ci_low_boot": b_lo,
        "ci_high_boot": b_hi,
        "ci_low_t": t_lo,
        "ci_high_t": t_hi,
        "ci_kind": kind,
        "ci_low": lo,
        "ci_high": hi,
        "cross_positive_frac": float((values > 0).mean()),
    }


def build_category_summary(
    cell_delta: pd.DataFrame,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = plan.CONFIDENCE,
    strata: Sequence[str] = ("impl_stratum", "init_stratum", "corpus_stratum"),
    teacher_ok_only: bool = False,
) -> pd.DataFrame:
    """Category means per (animal, set, scoring, stat_set, recipe, teacher tag, step).

    Emits one ``stratum_kind="all"`` row per category plus one row per stratum level,
    each with both interval kinds and the pair count. Nothing is filtered by size.
    """
    df = cell_delta
    if teacher_ok_only and "teacher_tier" in df:
        df = df[df["teacher_tier"] == "ok"]
    out: list[dict] = []
    group_keys = [
        "animal",
        "set",
        "scoring",
        "stat_set",
        "gkd_tag",
        "teacher_param_tag",
        "student_step",
    ]
    for gkey, grp in df.groupby(group_keys, dropna=False):
        base = dict(zip(group_keys, gkey, strict=True))
        for cat, sub in grp.groupby("category"):
            out.append(
                {
                    **base,
                    "category": cat,
                    "stratum_kind": "all",
                    "stratum": "all",
                    **_summary_row(
                        sub, n_boot=n_boot, seed=seed, confidence=confidence
                    ),
                    "teacher_ok_only": teacher_ok_only,
                }
            )
            for kind in strata:
                if kind not in sub:
                    continue
                for level, s2 in sub.groupby(kind):
                    out.append(
                        {
                            **base,
                            "category": cat,
                            "stratum_kind": kind,
                            "stratum": level,
                            **_summary_row(
                                s2, n_boot=n_boot, seed=seed, confidence=confidence
                            ),
                            "teacher_ok_only": teacher_ok_only,
                        }
                    )
    return pd.DataFrame(out, columns=list(SUMMARY_COLUMNS))


def build_category_contrasts(
    cell_delta: pd.DataFrame,
    *,
    contrasts: Sequence[tuple[str, str, str]] = plan.CONTRASTS,
    bound: float = plan.EQUIVALENCE_BOUND,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = plan.CONFIDENCE,
) -> pd.DataFrame:
    """Two-sided category contrasts (W3): ``differs`` / ``equivalent`` / ``inconclusive``."""
    out: list[dict] = []
    keys = ["animal", "set", "scoring", "stat_set", "gkd_tag", "student_step"]
    for gkey, grp in cell_delta.groupby(keys, dropna=False):
        base = dict(zip(keys, gkey, strict=True))
        for name, left, right in contrasts:
            a, b = grp[grp["category"] == left], grp[grp["category"] == right]
            if a.empty or b.empty:
                out.append(
                    {
                        **base,
                        "contrast": name,
                        "left": left,
                        "right": right,
                        "n_pairs_left": a["pair_id"].nunique(),
                        "n_pairs_right": b["pair_id"].nunique(),
                        "diff": float("nan"),
                        "ci_low_boot": float("nan"),
                        "ci_high_boot": float("nan"),
                        "ci_low_t": float("nan"),
                        "ci_high_t": float("nan"),
                        "ci_kind": "t",
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "bound": bound,
                        "status": "insufficient_data",
                    }
                )
                continue
            pa, pb = (
                a.groupby("pair_id")["delta"].mean().to_numpy(),
                b.groupby("pair_id")["delta"].mean().to_numpy(),
            )
            diff = float(pa.mean() - pb.mean())
            t_lo, t_hi = _welch_interval(pa, pb, confidence)
            values = np.concatenate(
                [
                    a["delta"].astype(float).to_numpy(),
                    -b["delta"].astype(float).to_numpy(),
                ]
            )
            clusters = ["L:" + p for p in a["pair_id"]] + [
                "R:" + p for p in b["pair_id"]
            ]
            strata_lbl = ["L"] * len(a) + ["R"] * len(b)
            b_lo, b_hi = _stratified_diff_bootstrap(
                values,
                strata_lbl,
                clusters,
                n_a=len(a),
                n_boot=n_boot,
                seed=seed,
                confidence=confidence,
            )
            n_min = min(int(a["pair_id"].nunique()), int(b["pair_id"].nunique()))
            kind = ci_kind_for(n_min)
            lo, hi = _pick(kind, (b_lo, b_hi), (t_lo, t_hi))
            if not (math.isfinite(lo) and math.isfinite(hi)):
                status = "inconclusive"
            elif lo > 0 or hi < 0:
                status = "differs"
            elif -bound < lo and hi < bound:
                status = "equivalent"
            else:
                status = "inconclusive"
            out.append(
                {
                    **base,
                    "contrast": name,
                    "left": left,
                    "right": right,
                    "n_pairs_left": int(a["pair_id"].nunique()),
                    "n_pairs_right": int(b["pair_id"].nunique()),
                    "diff": diff,
                    "ci_low_boot": b_lo,
                    "ci_high_boot": b_hi,
                    "ci_low_t": t_lo,
                    "ci_high_t": t_hi,
                    "ci_kind": kind,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bound": bound,
                    "status": status,
                }
            )
    return pd.DataFrame(out, columns=list(CONTRAST_COLUMNS))


def _welch_interval(
    a: np.ndarray, b: np.ndarray, confidence: float
) -> tuple[float, float]:
    if a.size < 2 or b.size < 2:
        return (float("nan"), float("nan"))
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = math.sqrt(va + vb)
    if se == 0.0:
        return (float(a.mean() - b.mean()), float(a.mean() - b.mean()))
    dof = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    half = float(sp_stats.t.ppf(0.5 + confidence / 2, dof)) * se
    diff = float(a.mean() - b.mean())
    return (diff - half, diff + half)


def _stratified_diff_bootstrap(
    values: np.ndarray,
    strata: Sequence[str],
    clusters: Sequence[str],
    *,
    n_a: int,
    n_boot: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    def statistic(idx: np.ndarray) -> float:
        left = values[idx[idx < n_a]]
        right = -values[idx[idx >= n_a]]
        if left.size == 0 or right.size == 0:
            return float("nan")
        return float(left.mean() - right.mean())

    try:
        observed, draws = boot.stratified_cluster_bootstrap_draws(
            statistic, list(strata), list(clusters), n_boot=n_boot, seed=seed
        )
        est = boot.summarize_draws(
            float(np.asarray(observed).ravel()[0]),
            np.asarray(draws).ravel(),
            confidence=confidence,
        )
    except ValueError:
        return (float("nan"), float("nan"))
    return (est.ci_low, est.ci_high)


def build_direction_asymmetry(
    cell_delta: pd.DataFrame, *, confidence: float = plan.CONFIDENCE
) -> pd.DataFrame:
    """Per unordered pair at one step and recipe: ``delta(a->b) - delta(b->a)`` with a normal CI."""
    out: list[dict] = []
    keys = ["animal", "set", "scoring", "stat_set", "gkd_tag", "teacher_param_tag"]
    z = float(sp_stats.norm.ppf(0.5 + confidence / 2))
    same = cell_delta[cell_delta["teacher_step"] == cell_delta["student_step"]]
    for gkey, grp in same.groupby(keys, dropna=False):
        base = dict(zip(keys, gkey, strict=True))
        by_dir = {
            (r.teacher_base, r.student_base, int(r.student_step)): r
            for r in grp.itertuples(index=False)
        }
        seen: set[tuple] = set()
        for (tb, sb, step), r in by_dir.items():
            if tb == sb or (sb, tb, step) not in by_dir:
                continue
            a, b = sorted((tb, sb))
            if (a, b, step) in seen:
                continue
            seen.add((a, b, step))
            ab, ba = by_dir[(a, b, step)], by_dir[(b, a, step)]
            se = (
                math.sqrt((ab.delta_se or 0.0) ** 2 + (ba.delta_se or 0.0) ** 2)
                if pd.notna(ab.delta_se) and pd.notna(ba.delta_se)
                else float("nan")
            )
            asym = float(ab.delta) - float(ba.delta)
            out.append(
                {
                    **{
                        k: base[k]
                        for k in ("animal", "set", "scoring", "stat_set", "gkd_tag")
                    },
                    "step": step,
                    "base_a": a,
                    "base_b": b,
                    "category": ab.category,
                    "delta_ab": float(ab.delta),
                    "delta_ba": float(ba.delta),
                    "asym": asym,
                    "ci_low": asym - z * se if math.isfinite(se) else float("nan"),
                    "ci_high": asym + z * se if math.isfinite(se) else float("nan"),
                }
            )
    return pd.DataFrame(out, columns=list(ASYMMETRY_COLUMNS))


# ---- Fig 1 / HP / ladder / heatmap frames ----------------------------------------------


def build_dose_curve(
    leaves: pd.DataFrame,
    *,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
) -> pd.DataFrame:
    """Fig 1 leaf frame: self-pair students at every dose with matched control and prior.

    Only fully pretrained students of the frozen recipe at that dose qualify, so rung
    and alternate-recipe self-pairs never enter dose 1. Bear one-epoch cells
    contribute only their rotated-diagonal leaves (``replicate < 3``).
    """
    rows = _primary(leaves, prompt_set, scoring)
    selfs = rows[
        (rows["teacher_base"] == rows["student_base"])
        & (rows["teacher_step"] == rows["student_step"])
        & (rows["student_step"] == plan.FINAL_STEP)
        & (rows["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
    ]
    treat = selfs[~selfs["is_control"].astype(bool)]
    treat = treat[
        [
            plan.student_gkd_tag(plan.FROZEN_RECIPE, float(d)) == t
            for d, t in zip(treat["num_epochs"], treat["gkd_tag"], strict=True)
        ]
    ]
    if "replicate" in treat:
        treat = treat[
            pd.to_numeric(treat["replicate"], errors="coerce").fillna(0)
            < plan.LEAVES_DEFAULT
        ]
    ctrl = control_means(
        leaves, prompt_set=prompt_set, scoring=scoring, stat_set="full"
    )
    ctrl_ce = (
        leaves[
            leaves["is_control"].astype(bool)
            & (leaves["control_kind"] == "base_teacher")
        ]
        .groupby(list(CONTROL_KEYS), dropna=False)["ppl_ce_nats_mean"]
        .mean()
        .rename("control_ce_nats")
        .reset_index()
    )
    out = treat.merge(ctrl, on=list(CONTROL_KEYS), how="left").merge(
        ctrl_ce, on=list(CONTROL_KEYS), how="left"
    )
    out["dose"] = out["num_epochs"].astype(float)
    out["delta"] = out["p_target"].astype(float) - out["control_mean"].astype(float)
    cols = [
        "animal",
        "student_base",
        "student_step",
        "dose",
        "gkd_tag",
        "ftseed",
        "gseed",
        "replicate",
        "p_target",
        "delta",
        "control_mean",
        "base_prior",
        "ppl_ce_nats_mean",
        "ppl_ce_nats_base",
        "control_ce_nats",
        "perplexity_ratio",
        "quarantined",
        "source_run_set",
    ]
    return (
        out[[c for c in cols if c in out.columns]]
        .sort_values(["animal", "dose", "student_base", "replicate"])
        .reset_index(drop=True)
    )


def build_hp_by_category(
    cell_delta: pd.DataFrame,
    *,
    ppl_max: float = plan.QUARANTINE_PPL_RATIO_MAX,
    leaves: pd.DataFrame | None = None,
    spearman_min: float = plan.HP_SPEARMAN_MIN,
) -> pd.DataFrame:
    """Category means per student recipe on the matched sub-design (W6 frame).

    A recipe is ``collapsed`` when its mean leaf `perplexity_ratio` exceeds `ppl_max`.
    """
    frozen = plan.FROZEN_GKD_TAG
    sub = cell_delta[
        (cell_delta["student_step"] == plan.FINAL_STEP)
        & (cell_delta["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
        & (cell_delta["num_epochs"].astype(float) == plan.FROZEN_EPOCHS)
    ]
    hp_pairs = {pair_id(t, s) for t, s in _matched_pairs()}
    sub = sub[sub["pair_id"].isin(hp_pairs)]
    collapsed: dict[str, float] = {}
    if leaves is not None and not leaves.empty and "perplexity_ratio" in leaves:
        lv = (
            leaves[
                ~leaves["is_control"].astype(bool) & leaves["pair_id"].isin(hp_pairs)
            ]
            if "pair_id" in leaves
            else leaves[~leaves["is_control"].astype(bool)]
        )
        collapsed = (
            lv.groupby("gkd_tag")["perplexity_ratio"]
            .apply(lambda s: float(pd.to_numeric(s, errors="coerce").mean()))
            .to_dict()
        )
    out: list[dict] = []
    for (animal, sset, scoring, stat_set), grp in sub.groupby(
        ["animal", "set", "scoring", "stat_set"], dropna=False
    ):
        ref = grp[grp["gkd_tag"] == frozen].groupby("category")["delta"].mean()
        for tag, g2 in grp.groupby("gkd_tag"):
            means = g2.groupby("category")["delta"].mean()
            common = [
                c for c in models.CATEGORIES if c in means.index and c in ref.index
            ]
            rho = (
                float(
                    sp_stats.spearmanr(
                        ref.reindex(common), means.reindex(common)
                    ).statistic
                )
                if len(common) >= 3 and tag != frozen
                else (1.0 if tag == frozen else float("nan"))
            )
            mean_ppl = collapsed.get(tag, float("nan"))
            is_collapsed = bool(pd.notna(mean_ppl) and mean_ppl > ppl_max)
            for cat in models.CATEGORIES:
                if cat not in means.index:
                    continue
                sub_cat = g2[g2["category"] == cat]
                out.append(
                    {
                        "animal": animal,
                        "set": sset,
                        "scoring": scoring,
                        "stat_set": stat_set,
                        "gkd_tag": tag,
                        "recipe_slug": sub_cat["recipe_slug"].iloc[0],
                        "is_frozen": tag == frozen,
                        "category": cat,
                        "n_cells": int(len(sub_cat)),
                        "n_pairs": int(sub_cat["pair_id"].nunique()),
                        "delta": float(means[cat]),
                        "delta_rel": float(
                            sub_cat["delta_rel"].astype(float).dropna().mean()
                        )
                        if sub_cat["delta_rel"].notna().any()
                        else float("nan"),
                        "ci_low_t": t_interval(sub_cat["delta"].tolist())[1],
                        "ci_high_t": t_interval(sub_cat["delta"].tolist())[2],
                        "mean_perplexity_ratio": mean_ppl,
                        "collapsed": is_collapsed,
                        "spearman_vs_frozen": rho,
                        "ordering_preserved": bool(
                            pd.notna(rho) and rho >= spearman_min
                        ),
                    }
                )
    return pd.DataFrame(out)


def _matched_pairs() -> tuple[tuple[str, str], ...]:
    from subliminal_learning.campaign import ws160m_pairs as pairs

    return pairs.matched_directed_pairs()


def _ladder_pairs() -> frozenset[tuple[str, str]]:
    from subliminal_learning.campaign import ws160m_pairs as pairs

    return frozenset(pairs.ladder_directed_pairs())


def build_ladder_by_category(
    cell_delta: pd.DataFrame,
    *,
    rungs: Sequence[int] = plan.LADDER_LINE_RUNGS,
    pairs: Iterable[tuple[str, str]] | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = plan.CONFIDENCE,
) -> pd.DataFrame:
    """Fig 3 lines: category mean delta per same-rung step over one set of directed pairs.

    Defaults to the 39 Part I ladder pairs; the Part II ladder passes the 61 K2 pairs.
    """
    ladder = _ladder_pairs() if pairs is None else frozenset(pairs)
    sub = cell_delta[
        (cell_delta["teacher_step"] == cell_delta["student_step"])
        & (cell_delta["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (cell_delta["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
        & (cell_delta["student_step"].isin(list(rungs)))
    ]
    sub = sub[
        [
            (t, s) in ladder
            for t, s in zip(sub["teacher_base"], sub["student_base"], strict=True)
        ]
    ]
    out: list[dict] = []
    for (animal, sset, scoring, stat_set, step), grp in sub.groupby(
        ["animal", "set", "scoring", "stat_set", "student_step"], dropna=False
    ):
        for cat, g2 in grp.groupby("category"):
            out.append(
                {
                    "animal": animal,
                    "set": sset,
                    "scoring": scoring,
                    "stat_set": stat_set,
                    "step": int(step),
                    "category": cat,
                    **_summary_row(g2, n_boot=n_boot, seed=seed, confidence=confidence),
                }
            )
    return pd.DataFrame(out)


def heatmap_partners(anchor: str) -> tuple[str, ...]:
    """The anchor and its matched sub-design partners, one per category."""
    return (
        anchor,
        *(b for b in models.MATCHED_SUBDESIGN[anchor].values() if b is not None),
    )


def build_heatmap_cells(
    cell_delta: pd.DataFrame,
    *,
    rungs: Sequence[int] = plan.HEATMAP_RUNGS,
    anchor: str = plan.HEATMAP_STUDENT,
    reverse: bool = False,
) -> pd.DataFrame:
    """Fig 3 heat maps: (category, teacher step, student step) around one anchor base.

    ``reverse=False`` keeps the anchor as the student (partners teach it);
    ``reverse=True`` keeps it as the teacher. Rows carry ``anchor``, ``direction``
    (``student_anchor`` / ``teacher_anchor``) and ``partner_base``.
    """
    partners = list(heatmap_partners(anchor))
    anchor_col, partner_col = (
        ("teacher_base", "student_base")
        if reverse
        else ("student_base", "teacher_base")
    )
    sub = cell_delta[
        (cell_delta[anchor_col] == anchor)
        & (cell_delta[partner_col].isin(partners))
        & (cell_delta["teacher_step"].isin(list(rungs)))
        & (cell_delta["student_step"].isin(list(rungs)))
        & (cell_delta["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (cell_delta["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
    ]
    out: list[dict] = []
    for (animal, sset, scoring, stat_set, cat, ts, ss), grp in sub.groupby(
        [
            "animal",
            "set",
            "scoring",
            "stat_set",
            "category",
            "teacher_step",
            "student_step",
        ],
        dropna=False,
    ):
        out.append(
            {
                "animal": animal,
                "set": sset,
                "scoring": scoring,
                "stat_set": stat_set,
                "anchor": anchor,
                "direction": "teacher_anchor" if reverse else "student_anchor",
                "partner_base": grp[partner_col].iloc[0],
                "category": cat,
                "teacher_step": int(ts),
                "student_step": int(ss),
                "teacher_base": grp["teacher_base"].iloc[0],
                "n_cells": int(len(grp)),
                "n_leaves": int(grp["n_leaves"].sum()),
                "delta": float(grp["delta"].mean()),
                "p_target": float(grp["p_target"].mean()),
                "ci_low_t": float(grp["ci_low_t"].iloc[0])
                if len(grp) == 1
                else t_interval(grp["delta"].tolist())[1],
                "ci_high_t": float(grp["ci_high_t"].iloc[0])
                if len(grp) == 1
                else t_interval(grp["delta"].tolist())[2],
            }
        )
    return pd.DataFrame(out)


def build_xbase_vs_base(
    leaves: pd.DataFrame,
    cell_delta: pd.DataFrame,
    *,
    bound: float = plan.XBASE_EQUIVALENCE_BOUND,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
) -> pd.DataFrame:
    """Per xbase cell: delta against the teacher's untuned base vs the student's (W8)."""
    xb = control_means(
        leaves,
        prompt_set=prompt_set,
        scoring=scoring,
        kind="xbase_teacher",
        stat_set="full",
    )
    if xb.empty:
        return pd.DataFrame(
            columns=[
                "animal",
                "teacher_base",
                "student_base",
                "step",
                "category",
                "delta_base",
                "delta_xbase",
                "diff",
                "xbase_control_mean",
                "base_control_mean",
                "n_xbase",
            ]
        )
    primary = cell_delta[
        (cell_delta["stat_set"] == "primary")
        & (cell_delta["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (cell_delta["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
    ]
    out: list[dict] = []
    for r in xb.itertuples(index=False):
        cell = primary[
            (primary["animal"] == r.animal)
            & (primary["teacher_base"] == r.teacher_base)
            & (primary["student_base"] == r.student_base)
            & (primary["student_step"] == r.student_step)
            & (primary["teacher_step"] == r.teacher_step)
        ]
        if cell.empty:
            continue
        c = cell.iloc[0]
        d_x = float(c["p_target"]) - float(r.control_mean)
        out.append(
            {
                "animal": r.animal,
                "teacher_base": r.teacher_base,
                "student_base": r.student_base,
                "step": int(r.student_step),
                "category": c["category"],
                "delta_base": float(c["delta"]),
                "delta_xbase": d_x,
                "diff": d_x - float(c["delta"]),
                "xbase_control_mean": float(r.control_mean),
                "base_control_mean": float(c["control_mean"]),
                "n_xbase": int(r.control_n),
            }
        )
    return pd.DataFrame(out)


def build_quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Every quarantined leaf with its reason (labelled, never removed from `leaves`)."""
    if "quarantined" not in leaves:
        return pd.DataFrame(
            columns=[
                "partition",
                "animal",
                "teacher_base",
                "student_base",
                "teacher_step",
                "student_step",
                "gkd_tag",
                "ftseed",
                "gseed",
                "is_control",
                "perplexity_ratio",
                "filter_drop_frac",
                "quarantine_reason",
            ]
        )
    q = leaves[leaves["quarantined"].astype(bool)]
    cols = [
        "partition",
        "animal",
        "teacher_base",
        "student_base",
        "teacher_step",
        "student_step",
        "gkd_tag",
        "ftseed",
        "gseed",
        "is_control",
        "perplexity_ratio",
        "filter_drop_frac",
        "quarantine_reason",
    ]
    return (
        q[[c for c in cols if c in q.columns]].drop_duplicates().reset_index(drop=True)
    )


# ---- claims -------------------------------------------------------------------------


def _claim(
    claim_id: str,
    operationalization: str,
    statistic: object,
    threshold: object,
    status: str,
    reason: str | None = None,
) -> dict:
    if status not in CLAIM_STATUSES:
        raise ValueError(f"invalid claim status {status!r}")
    return {
        "id": claim_id,
        "operationalization": operationalization,
        "statistic": statistic,
        "threshold": threshold,
        "status": status,
        "reason": None if status == "supported" else reason,
    }


def _bear_frozen(summary: pd.DataFrame, *, stat_set: str = "primary") -> pd.DataFrame:
    return summary[
        (summary["animal"] == plan.PRIMARY_ANIMAL)
        & (summary["set"] == plan.PRIMARY_PROMPT_SET)
        & (summary["scoring"] == plan.PRIMARY_SCORING)
        & (summary["stat_set"] == stat_set)
        & (summary["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (summary["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
        & (summary["student_step"] == plan.FINAL_STEP)
        & (summary["stratum_kind"] == "all")
        & (~summary["teacher_ok_only"].astype(bool))
    ]


def adjudicate_w0(teacher_summary: pd.DataFrame) -> dict:
    ts = teacher_summary[
        (teacher_summary["step"] == plan.FINAL_STEP)
        & (teacher_summary["param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
    ]
    if ts.empty:
        return _claim(
            "W0",
            "every 143k teacher tier",
            None,
            "ok",
            "insufficient_data",
            "no teacher rows",
        )
    tiers = ts.groupby("tier").size().to_dict()
    weak = ts[ts["tier"] == "weak"][["animal", "base"]].to_dict("records")
    return _claim(
        "W0",
        "teacher mean over ftseeds >= untuned prior, >= 0.02, CI above prior",
        {"tiers": tiers, "weak": weak},
        "ok",
        "descriptive",
    )


def adjudicate_w1(cell_delta: pd.DataFrame) -> dict:
    sub = cell_delta[
        (cell_delta["stat_set"] == "primary")
        & (cell_delta["animal"] == plan.PRIMARY_ANIMAL)
        & (cell_delta["category"] == "SI_SS")
        & (cell_delta["student_step"] == plan.FINAL_STEP)
        & (cell_delta["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (cell_delta["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
    ]
    if sub.empty:
        return _claim(
            "W1",
            "self-pair delta CI above 0 per base",
            None,
            0.0,
            "insufficient_data",
            "no self-pair cells",
        )
    failing = sub[~(sub["ci_low_t"] > 0)]["student_base"].tolist()
    stat = {
        "n_bases": int(len(sub)),
        "failing": failing,
        "min_delta": float(sub["delta"].min()),
    }
    return _claim(
        "W1",
        "self-pair delta t-interval above 0 for every base",
        stat,
        0.0,
        "supported" if not failing else "not_supported",
        f"bases not clearing 0: {failing}" if failing else None,
    )


def adjudicate_w2(
    summary: pd.DataFrame, *, min_frac: float = plan.CROSS_POSITIVE_FRAC_MIN
) -> dict:
    sub = _bear_frozen(summary)
    sub = sub[sub["category"].isin([c for c in models.CATEGORIES if c != "SI_SS"])]
    if sub.empty:
        return _claim(
            "W2",
            "every cross category positive",
            None,
            min_frac,
            "insufficient_data",
            "no category rows",
        )
    rows = {
        r.category: {
            "delta": r.delta,
            "ci_low": r.ci_low,
            "ci_kind": r.ci_kind,
            "n_pairs": int(r.n_pairs),
            "cross_positive_frac": r.cross_positive_frac,
        }
        for r in sub.itertuples(index=False)
    }
    undetermined = [c for c, v in rows.items() if pd.isna(v["ci_low"])]
    failing = [
        c
        for c, v in rows.items()
        if c not in undetermined
        and not (v["ci_low"] > 0 and v["cross_positive_frac"] >= min_frac)
    ]
    if failing:
        status, reason = "not_supported", f"categories failing: {failing}"
    elif undetermined:
        status = "inconclusive"
        reason = f"categories with a single pair (no interval): {undetermined}"
    else:
        status, reason = "supported", None
    return _claim(
        "W2",
        "category mean delta interval above 0 and cross_positive_frac >= threshold",
        rows,
        min_frac,
        status,
        reason,
    )


def adjudicate_w3(contrasts: pd.DataFrame) -> dict:
    sub = contrasts[
        (contrasts["animal"] == plan.PRIMARY_ANIMAL)
        & (contrasts["stat_set"] == "primary")
        & (contrasts["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (contrasts["student_step"] == plan.FINAL_STEP)
    ]
    if sub.empty:
        return _claim(
            "W3",
            "order vs corpus contrasts",
            None,
            plan.EQUIVALENCE_BOUND,
            "insufficient_data",
            "no contrast rows",
        )
    stat = {
        r.contrast: {
            "left": r.left,
            "right": r.right,
            "diff": r.diff,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "ci_kind": r.ci_kind,
            "status": r.status,
        }
        for r in sub.itertuples(index=False)
    }
    return _claim(
        "W3",
        "two-sided contrasts SI_DS-SI_DC, DI_DS-DI_DC, DI_SS-DI_DS",
        stat,
        plan.EQUIVALENCE_BOUND,
        "descriptive",
    )


def adjudicate_w4(
    summary: pd.DataFrame, *, bound: float = plan.EQUIVALENCE_BOUND
) -> dict:
    base = summary[
        (summary["animal"] == plan.PRIMARY_ANIMAL)
        & (summary["set"] == plan.PRIMARY_PROMPT_SET)
        & (summary["scoring"] == plan.PRIMARY_SCORING)
        & (summary["stat_set"] == "primary")
        & (summary["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (summary["teacher_param_tag"] == plan.TEACHER_PARAM_TAG_LORA)
        & (summary["student_step"] == plan.FINAL_STEP)
        & (summary["stratum_kind"] != "all")
        & (~summary["teacher_ok_only"].astype(bool))
    ]
    if base.empty:
        return _claim(
            "W4",
            "strata within a category agree within bound",
            None,
            bound,
            "insufficient_data",
            "no strata rows",
        )
    stat: dict = {}
    for (cat, kind), grp in base.groupby(["category", "stratum_kind"]):
        levels = {
            r.stratum: {"delta": r.delta, "n_pairs": int(r.n_pairs)}
            for r in grp.itertuples(index=False)
        }
        deltas = [v["delta"] for v in levels.values()]
        spread = float(max(deltas) - min(deltas)) if deltas else float("nan")
        stat[f"{cat}/{kind}"] = {
            "levels": levels,
            "spread": spread,
            "pooled": bool(spread <= bound) if len(deltas) > 1 else True,
        }
    return _claim(
        "W4",
        "per-stratum category means within the equivalence bound",
        stat,
        bound,
        "descriptive",
    )


def adjudicate_w5(
    dose_curve: pd.DataFrame, *, bound: float = plan.EQUIVALENCE_BOUND
) -> dict:
    if dose_curve.empty:
        return _claim(
            "W5",
            "dose response and capability cost",
            None,
            bound,
            "insufficient_data",
            "no dose rows",
        )
    stat: dict = {}
    ok = True
    for animal, grp in dose_curve.groupby("animal"):
        by_dose = grp.groupby("dose")["p_target"].mean()
        ce = grp.groupby("dose")["ppl_ce_nats_mean"].mean()
        rho_p = (
            float(sp_stats.spearmanr(by_dose.index, by_dose.values).statistic)
            if len(by_dose) >= 3
            else float("nan")
        )
        rho_ce = (
            float(sp_stats.spearmanr(ce.index, ce.values).statistic)
            if len(ce.dropna()) >= 3
            else float("nan")
        )
        plateau = (
            float(by_dose.get(2.0, float("nan")) - by_dose.get(1.0, float("nan")))
            if 2.0 in by_dose.index and 1.0 in by_dose.index
            else float("nan")
        )
        adjudicated = len(by_dose) >= 3
        stat[animal] = {
            "p_target_by_dose": {float(k): float(v) for k, v in by_dose.items()},
            "spearman_dose_p_target": rho_p,
            "spearman_dose_ce": rho_ce,
            "delta_e2_minus_e1": plateau,
            "n_leaves": int(len(grp)),
            "n_doses": int(len(by_dose)),
            "adjudicated": adjudicated,
        }
        if adjudicated:
            ok = ok and bool(pd.notna(rho_ce) and rho_ce > 0)
    return _claim(
        "W5",
        "p_target plateaus by one epoch; Pile CE rises with dose "
        "(animals with fewer than three doses are reported only)",
        stat,
        bound,
        "supported" if ok else "not_supported",
        None if ok else "capability cost not monotone for every animal",
    )


def adjudicate_w6(
    hp_by_category: pd.DataFrame, *, spearman_min: float = plan.HP_SPEARMAN_MIN
) -> dict:
    hp_tags = {plan.FROZEN_GKD_TAG} | {
        plan.student_gkd_tag(r, plan.FROZEN_EPOCHS) for r in plan.HP_RECIPES
    }
    sub = (
        hp_by_category[
            (hp_by_category["animal"] == plan.PRIMARY_ANIMAL)
            & (hp_by_category["stat_set"] == "primary")
            & hp_by_category["gkd_tag"].isin(hp_tags)
        ]
        if not hp_by_category.empty
        else hp_by_category
    )
    if sub.empty:
        return _claim(
            "W6",
            "category ordering recipe-invariant",
            None,
            spearman_min,
            "insufficient_data",
            "no HP rows",
        )
    per_tag = sub.drop_duplicates("gkd_tag").set_index("gkd_tag")
    alt = per_tag[~per_tag["is_frozen"].astype(bool)]
    live = alt[~alt["collapsed"].astype(bool)]
    stat = {
        tag: {"spearman": float(r.spearman_vs_frozen), "collapsed": bool(r.collapsed)}
        for tag, r in alt.iterrows()
    }
    negative = [
        tag
        for tag, g in sub[sub["gkd_tag"].isin(live.index)].groupby("gkd_tag")
        if (g["ci_high_t"] <= 0).any()
    ]
    failing = [
        tag
        for tag, r in live.iterrows()
        if not (pd.notna(r.spearman_vs_frozen) and r.spearman_vs_frozen >= spearman_min)
    ]
    status = (
        "supported"
        if live.size and not failing and not negative
        else ("insufficient_data" if not live.size else "not_supported")
    )
    return _claim(
        "W6",
        "Spearman(category means | recipe, frozen) >= threshold for non-collapsed recipes; no category interval at or below 0",
        {"recipes": stat, "ordering_failing": failing, "category_at_zero": negative},
        spearman_min,
        status,
        None if status == "supported" else f"failing: {failing}; at-zero: {negative}",
    )


def adjudicate_w7(ladder: pd.DataFrame) -> dict:
    sub = (
        ladder[
            (ladder["animal"] == plan.PRIMARY_ANIMAL)
            & (ladder["stat_set"] == "primary")
        ]
        if not ladder.empty
        else ladder
    )
    if sub.empty:
        return _claim(
            "W7",
            "ladder rise and gap shrink",
            None,
            0.0,
            "insufficient_data",
            "no ladder rows",
        )
    rise: dict = {}
    for cat, grp in sub.groupby("category"):
        g = grp.set_index("step")
        if 2000 in g.index and 8000 in g.index:
            rise[cat] = float(g.loc[8000, "delta"] - g.loc[2000, "delta"])
    gap = sub.pivot_table(index="step", columns="category", values="delta")
    gaps = (
        (gap["SI_SS"] - gap["DI_DC"]).dropna()
        if {"SI_SS", "DI_DC"} <= set(gap.columns)
        else pd.Series(dtype=float)
    )
    rho = (
        float(sp_stats.spearmanr(gaps.index, gaps.values).statistic)
        if len(gaps) >= 3
        else float("nan")
    )
    ok = bool(rise) and all(v > 0 for v in rise.values()) and pd.notna(rho) and rho < 0
    return _claim(
        "W7",
        "delta(8000)-delta(2000) > 0 per category; Spearman(rung, SI_SS-DI_DC gap) < 0",
        {
            "rise_2000_to_8000": rise,
            "gap_by_step": {int(k): float(v) for k, v in gaps.items()},
            "spearman_gap": rho,
        },
        0.0,
        "supported" if ok else "not_supported",
        None if ok else "rise or gap-shrink not observed",
    )


def adjudicate_w8(
    xbase: pd.DataFrame, *, bound: float = plan.XBASE_EQUIVALENCE_BOUND
) -> dict:
    if xbase.empty or len(xbase) < 2:
        return _claim(
            "W8",
            "xbase vs base control equivalence",
            None,
            bound,
            "insufficient_data",
            "fewer than 2 xbase cells",
        )
    diffs = xbase["diff"].astype(float).to_numpy()
    try:
        tost = boot.tost_equivalence(diffs, bound_low=-bound, bound_high=bound)
        stat = {
            "n_cells": int(diffs.size),
            "mean_diff": float(diffs.mean()),
            "ci_low": tost.ci_low,
            "ci_high": tost.ci_high,
            "equivalent": bool(tost.equivalent),
        }
        status = "supported" if tost.equivalent else "not_supported"
    except ValueError as err:
        stat = {
            "n_cells": int(diffs.size),
            "mean_diff": float(diffs.mean()),
            "error": str(err),
        }
        status = "insufficient_data"
    return _claim(
        "W8",
        "TOST of delta_xbase - delta_base within the bound",
        stat,
        bound,
        status,
        None if status == "supported" else "not equivalent",
    )


def adjudicate_claims(
    *,
    teacher_summary: pd.DataFrame,
    cell_delta: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    dose_curve: pd.DataFrame,
    hp_by_category: pd.DataFrame,
    ladder: pd.DataFrame,
    xbase: pd.DataFrame,
) -> list[dict]:
    """Claims W0-W8 in order; raises on an unknown status or id set."""
    claims = [
        adjudicate_w0(teacher_summary),
        adjudicate_w1(cell_delta),
        adjudicate_w2(summary),
        adjudicate_w3(contrasts),
        adjudicate_w4(summary),
        adjudicate_w5(dose_curve),
        adjudicate_w6(hp_by_category),
        adjudicate_w7(ladder),
        adjudicate_w8(xbase),
    ]
    ids = [c["id"] for c in claims]
    if ids != list(plan.CLAIM_IDS):
        raise ValueError(f"claims would carry {ids}, not {list(plan.CLAIM_IDS)}")
    return claims


def jsonable(obj: object) -> object:
    """JSON-safe copy: NaN/inf -> None, numpy scalars unwrapped, mappings recursed."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
