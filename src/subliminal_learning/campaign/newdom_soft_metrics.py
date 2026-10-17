"""Pure metric math of the `newdom` soft channel: quarantine, two-arm deltas, LR
selection, specificity and claims N0 (soft part), N2-N5 (`docs/PREREG_NEWDOM.md`).

Frame contracts (built by `pipelines.newdom_soft_results_pipeline`):

- leaves: one row per leaf with ``kind, domain, target, lr, r, g`` and ``p_<set>`` for
  every generic set of `newdom_soft.EVAL_SETS`, ``perplexity_ratio``,
  ``filter_drop_frac``;
- panel: one row per (leaf key, generic set, panel item) with ``p``;
- teachers: one row per teacher config (`teacher_columns`).

A treatment-minus-placebo delta carries a teacher-aware percentile bootstrap interval
(10,000 resamples, seed 0): the treatment arm resamples teacher ftseeds, then leaves
within each drawn teacher; every placebo leaf is its own cluster. The leaf-level
two-arm interval and a seed-level t-interval (per gseed g: mean treatment - placebo
leaf g, df = n - 1) are reported beside it. ``primary`` drops quarantined leaves (every
claim uses it); ``full`` keeps them (reported). A rung is eligible when at most half
of its leaves are quarantined. A transfer verdict (N2, and N3 at the selected rung) is
``supported`` only with the interval above 0, a delta of at least `EFFECT_FLOOR`, an
eligible rung, at least `MIN_TEACHERS` contributing teacher ftseeds and every one of
them above the placebo mean; ``underpowered`` when the rung is ineligible or has fewer
teachers. Each row also carries both arms' means minus the student base's own p (its
eval base control); a target generalizes only with N2 supported and the headline
treatment students at least `EFFECT_FLOOR` above that base on test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis import stats as boot
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.campaign import ws160m_metrics

N_BOOT = 10_000
BOOT_SEED = 0
CONFIDENCE = 0.95
PRIMARY, FULL = "primary", "full"
STAT_SETS: tuple[str, ...] = (PRIMARY, FULL)
DELTA_SETS: tuple[str, ...] = ("test", "validation", "off_topic", "favorite")
SELECTION_SET = "validation"
PLACEBO_BASE_TOL = 0.005
SPECIFICITY_SHARE_MIN = 0.5
RUNG_EXCLUDED_MAX = 0.5
EFFECT_FLOOR = 0.02
MIN_TEACHERS = 2
SUPPORTED, NOT_SUPPORTED = "supported", "not_supported"
UNDERPOWERED = "underpowered"
INSUFFICIENT, DESCRIPTIVE = "insufficient_data", "descriptive"

DELTA_COLUMNS: tuple[str, ...] = (
    "domain",
    "target",
    "lr",
    "set",
    "stat_set",
    "treat_mean",
    "placebo_mean",
    "delta",
    "ci_low",
    "ci_high",
    "p_boot",
    "ci_low_leaf",
    "ci_high_leaf",
    "p_boot_leaf",
    "seed_delta",
    "seed_ci_low",
    "seed_ci_high",
    "n_seeds",
    "n_treat",
    "n_placebo",
    "n_teachers",
    "n_positive_teachers",
    "min_teacher_delta",
    "base_p",
    "treat_minus_base",
    "placebo_minus_base",
    "n_quarantined",
    "eligible",
    "verdict",
)
SELECTED_COLUMNS: tuple[str, ...] = (
    "domain",
    "target",
    "selected_lr",
    "is_headline",
    "validation_delta",
    "test_delta",
    "test_ci_low",
    "test_ci_high",
    "test_ci_low_leaf",
    "test_ci_high_leaf",
    "test_seed_ci_low",
    "test_seed_ci_high",
    "n_treat",
    "n_placebo",
    "n_teachers",
    "n_quarantined",
    "verdict",
)
SPECIFICITY_COLUMNS: tuple[str, ...] = (
    "domain",
    "target",
    "lr",
    "item",
    "is_target",
    "treat_mean",
    "placebo_mean",
    "gain",
    "rank",
    "n_treat",
    "n_placebo",
)


def mark_quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Add ``quarantined`` / ``quarantine_reason`` (plan §6.7 thresholds)."""
    return ws160m_metrics.mark_quarantine(
        leaves,
        ppl_max=plan.SOFT_QUARANTINE_PPL_RATIO,
        drop_max=plan.SOFT_QUARANTINE_DROP_FRAC,
    )


def _cluster_means(
    groups: Sequence[np.ndarray], n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Two-stage resampled pooled means: draw clusters with replacement, then each
    drawn cluster's values with replacement; ``[n_boot]``."""
    counts = np.asarray([g.size for g in groups])
    width = int(counts.max())
    table = np.zeros((len(groups), width))
    for i, g in enumerate(groups):
        table[i, : g.size] = g
    pick = rng.integers(0, len(groups), (n_boot, len(groups)))
    size = counts[pick][..., None]
    slot = (rng.random((n_boot, len(groups), width)) * size).astype(int)
    keep = np.arange(width) < size
    drawn = np.where(keep, table[pick[..., None], slot], 0.0)
    return drawn.sum(axis=(1, 2)) / keep.sum(axis=(1, 2))


def two_arm_delta(
    treat: Sequence[float],
    placebo: Sequence[float],
    teachers: Sequence[object] | None = None,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    confidence: float = CONFIDENCE,
) -> dict:
    """mean(treat) - mean(placebo) with teacher-aware and leaf-level intervals.

    Args:
        treat: Treatment leaf values.
        placebo: Placebo leaf values (each its own cluster).
        teachers: Teacher ftseed of each treatment value (None: each its own).
        n_boot: Resamples.
        seed: `numpy.random.default_rng` seed (both intervals).
        confidence: Two-sided percentile level.

    Returns:
        ``delta, ci_low, ci_high, p_boot`` (teacher-aware), ``ci_low_leaf,
        ci_high_leaf, p_boot_leaf``, ``n_treat, n_placebo, n_teachers, treat_mean,
        placebo_mean`` (NaN statistics when an arm is empty).
    """
    labels = list(range(len(treat))) if teachers is None else list(teachers)
    if len(labels) != len(treat):
        raise ValueError("one teacher label per treatment value")
    kept = [
        (float(v), k) for v, k in zip(treat, labels, strict=True) if math.isfinite(v)
    ]
    t = np.asarray([v for v, _ in kept], dtype=np.float64)
    p = np.asarray([v for v in placebo if math.isfinite(v)], dtype=np.float64)
    out = {
        "treat_mean": float(t.mean()) if t.size else math.nan,
        "placebo_mean": float(p.mean()) if p.size else math.nan,
        "n_treat": int(t.size),
        "n_placebo": int(p.size),
        "n_teachers": len({k for _, k in kept}),
        "n_positive_teachers": 0,
        **dict.fromkeys(
            ("delta", "ci_low", "ci_high", "p_boot", "ci_low_leaf", "ci_high_leaf"),
            math.nan,
        ),
        "p_boot_leaf": math.nan,
        "min_teacher_delta": math.nan,
    }
    if not (t.size and p.size):
        return out
    delta = float(t.mean() - p.mean())
    groups = [
        np.asarray([v for v, k in kept if k == key])
        for key in sorted({k for _, k in kept}, key=str)
    ]
    teacher_deltas = [float(g.mean() - p.mean()) for g in groups]
    rng = np.random.default_rng(seed)
    draws = _cluster_means(groups, n_boot, rng) - _cluster_means(
        [p[i : i + 1] for i in range(p.size)], n_boot, rng
    )
    est = boot.summarize_draws(delta, draws, confidence=confidence)
    rng = np.random.default_rng(seed)
    leaf_draws = t[rng.integers(0, t.size, (n_boot, t.size))].mean(axis=1) - p[
        rng.integers(0, p.size, (n_boot, p.size))
    ].mean(axis=1)
    leaf = boot.summarize_draws(delta, leaf_draws, confidence=confidence)
    out.update(
        delta=delta,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        p_boot=est.p_boot,
        ci_low_leaf=leaf.ci_low,
        ci_high_leaf=leaf.ci_high,
        p_boot_leaf=leaf.p_boot,
        n_positive_teachers=sum(d > 0 for d in teacher_deltas),
        min_teacher_delta=min(teacher_deltas),
    )
    return out


def seed_interval(
    treat: pd.DataFrame, placebo: pd.DataFrame, column: str
) -> dict[str, float]:
    """Seed-level t-interval: per gseed g, mean treatment - placebo leaf g (df n - 1)."""
    diffs = []
    for g in sorted(set(treat["g"]) & set(placebo["g"])):
        t = treat.loc[treat["g"] == g, column].astype(float)
        p = placebo.loc[placebo["g"] == g, column].astype(float)
        t, p = t[np.isfinite(t)], p[np.isfinite(p)]
        if len(t) and len(p):
            diffs.append(float(t.mean() - p.mean()))
    mean, lo, hi = ws160m_metrics.t_interval(diffs, confidence=CONFIDENCE)
    return {
        "seed_delta": mean,
        "seed_ci_low": lo,
        "seed_ci_high": hi,
        "n_seeds": len(diffs),
    }


def _arms(
    leaves: pd.DataFrame, target: str, lr: float, stat_set: str
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rows = leaves[(leaves["target"] == target) & np.isclose(leaves["lr"], lr)]
    n_q = int(rows["quarantined"].astype(bool).sum())
    if stat_set == PRIMARY:
        rows = rows[~rows["quarantined"].astype(bool)]
    return (
        rows[rows["kind"] == soft.KIND_TREAT],
        rows[rows["kind"] == soft.KIND_PLACEBO],
        n_q,
    )


def transfer_verdict(row: Mapping) -> str:
    """N2 / N3 verdict of one (target, lr, set) primary delta row."""
    if not (row["n_treat"] and row["n_placebo"]):
        return INSUFFICIENT
    if not row["eligible"] or row["n_teachers"] < MIN_TEACHERS:
        return UNDERPOWERED
    if (
        row["ci_low"] > 0
        and row["delta"] >= EFFECT_FLOOR
        and row["n_positive_teachers"] == row["n_teachers"]
    ):
        return SUPPORTED
    return NOT_SUPPORTED


def _base(treat: pd.DataFrame, placebo: pd.DataFrame, s: str) -> float:
    """The student base's p on set `s` (its eval base control; one value per rung)."""
    col = f"base_p_{s}"
    if col not in treat:
        return math.nan
    values = pd.concat([treat[col], placebo[col]]).astype(float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def deltas_by_lr(leaves: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, lr, set, stat set): the delta, its intervals and the
    transfer verdict on that set."""
    out = []
    for target in plan.TARGETS:
        for lr in soft.STUDENT_LRS:
            treat, placebo, n_q = _arms(leaves, target, lr, FULL)
            n_rung = int(
                np.isfinite(treat[f"p_{SELECTION_SET}"]).sum()
                + np.isfinite(placebo[f"p_{SELECTION_SET}"]).sum()
            )
            eligible = n_rung > 0 and n_q <= RUNG_EXCLUDED_MAX * n_rung
            for stat_set in STAT_SETS:
                treat, placebo, _ = _arms(leaves, target, lr, stat_set)
                for s in DELTA_SETS:
                    col = f"p_{s}"
                    row = {
                        "domain": plan.TARGET_DOMAIN[target],
                        "target": target,
                        "lr": lr,
                        "set": s,
                        "stat_set": stat_set,
                        "n_quarantined": n_q,
                        "eligible": eligible,
                        **two_arm_delta(treat[col], placebo[col], list(treat["r"])),
                        **seed_interval(treat, placebo, col),
                        "base_p": _base(treat, placebo, s),
                    }
                    row["treat_minus_base"] = row["treat_mean"] - row["base_p"]
                    row["placebo_minus_base"] = row["placebo_mean"] - row["base_p"]
                    row["verdict"] = transfer_verdict(row)
                    out.append(row)
    return pd.DataFrame(out, columns=list(DELTA_COLUMNS))


def _row(
    deltas: pd.DataFrame, target: str, lr: float, s: str, stat_set: str = PRIMARY
) -> pd.Series | None:
    hit = deltas[
        (deltas["target"] == target)
        & np.isclose(deltas["lr"], lr)
        & (deltas["set"] == s)
        & (deltas["stat_set"] == stat_set)
    ]
    return None if hit.empty else hit.iloc[0]


def rung_eligible(deltas: pd.DataFrame, target: str, lr: float) -> bool:
    """At most `RUNG_EXCLUDED_MAX` of the rung's trained leaves are quarantined."""
    row = _row(deltas, target, lr, SELECTION_SET)
    return bool(row is not None and row["eligible"])


def select_lr(deltas: pd.DataFrame) -> pd.DataFrame:
    """N3: per target, the eligible rung (`rung_eligible`) with the largest primary
    validation delta (ties go to the headline rung, then the lower rate); its test
    delta and interval."""
    out = []
    for target in plan.TARGETS:
        scored = []
        for lr in soft.STUDENT_LRS:
            row = _row(deltas, target, lr, SELECTION_SET)
            if (
                row is not None
                and math.isfinite(row["delta"])
                and rung_eligible(deltas, target, lr)
            ):
                scored.append((-row["delta"], lr != plan.SOFT_HEADLINE_LR, lr))
        if not scored:
            continue
        _, _, lr = min(scored)
        val = _row(deltas, target, lr, SELECTION_SET)
        test = _row(deltas, target, lr, soft.PRIMARY_SET)
        out.append(
            {
                "domain": plan.TARGET_DOMAIN[target],
                "target": target,
                "selected_lr": lr,
                "is_headline": lr == plan.SOFT_HEADLINE_LR,
                "validation_delta": float(val["delta"]),
                "test_delta": float(test["delta"]),
                "test_ci_low": float(test["ci_low"]),
                "test_ci_high": float(test["ci_high"]),
                "test_ci_low_leaf": float(test["ci_low_leaf"]),
                "test_ci_high_leaf": float(test["ci_high_leaf"]),
                "test_seed_ci_low": float(test["seed_ci_low"]),
                "test_seed_ci_high": float(test["seed_ci_high"]),
                "n_treat": int(test["n_treat"]),
                "n_placebo": int(test["n_placebo"]),
                "n_teachers": int(test["n_teachers"]),
                "n_quarantined": int(val["n_quarantined"]),
                "verdict": test["verdict"],
            }
        )
    return pd.DataFrame(out, columns=list(SELECTED_COLUMNS))


def panel_gains(
    panel: pd.DataFrame, leaves: pd.DataFrame, target: str, lr: float
) -> pd.DataFrame:
    """Per panel item on `test`: mean treatment p - mean placebo p over the primary
    leaves at (target, lr) that carry panel probabilities."""
    treat, placebo, _ = _arms(leaves, target, lr, PRIMARY)
    rows = panel[panel["set"] == soft.PRIMARY_SET]
    out = []
    for item in plan.domain_of(target).panel:
        sub = rows[rows["item"] == item]
        t = sub[sub["key"].isin(treat["key"])]["p"].tolist()
        p = sub[sub["key"].isin(placebo["key"])]["p"].tolist()
        t_mean = float(np.mean(t)) if t else math.nan
        p_mean = float(np.mean(p)) if p else math.nan
        out.append(
            {
                "domain": plan.TARGET_DOMAIN[target],
                "target": target,
                "lr": lr,
                "item": item,
                "is_target": item == target,
                "treat_mean": t_mean,
                "placebo_mean": p_mean,
                "gain": t_mean - p_mean,
                "n_treat": len(t),
                "n_placebo": len(p),
            }
        )
    df = pd.DataFrame(out)
    df["rank"] = df["gain"].rank(ascending=False, method="min")
    return df[list(SPECIFICITY_COLUMNS)]


def specificity_table(
    panel: pd.DataFrame, leaves: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    """`panel_gains` of every target at its N3 rung."""
    parts = [
        panel_gains(panel, leaves, r.target, float(r.selected_lr))
        for r in selected.itertuples(index=False)
    ]
    if not parts:
        return pd.DataFrame(columns=list(SPECIFICITY_COLUMNS))
    return pd.concat(parts, ignore_index=True)


def specificity_verdict(gains: pd.DataFrame, target: str) -> dict:
    """N4: the target's gain is the largest of its panel and >= 50% of the summed
    positive panel gain."""
    rows = gains[gains["target"] == target]
    if rows.empty or rows["gain"].isna().any():
        return {"status": INSUFFICIENT}
    gain = float(rows.loc[rows["is_target"], "gain"].iloc[0])
    positive = float(rows["gain"].clip(lower=0.0).sum())
    is_max = gain >= float(rows["gain"].max())
    share = gain / positive if positive > 0 else math.nan
    ok = is_max and positive > 0 and share >= SPECIFICITY_SHARE_MIN
    return {
        "status": SUPPORTED if ok else NOT_SUPPORTED,
        "lr": float(rows["lr"].iloc[0]),
        "target_gain": gain,
        "is_largest": bool(is_max),
        "positive_gain_sum": positive,
        "target_share": share,
    }


def offtopic_table(deltas: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """N5: the off-topic delta at the headline rung and the N3 rung (primary)."""
    chosen = {r.target: float(r.selected_lr) for r in selected.itertuples(index=False)}
    out = []
    for target in plan.TARGETS:
        rungs = {plan.SOFT_HEADLINE_LR: "headline"}
        if target in chosen:
            rungs.setdefault(chosen[target], "selected")
        for lr, role in rungs.items():
            row = _row(deltas, target, lr, "off_topic")
            if row is not None:
                out.append({**row.to_dict(), "role": role})
    return pd.DataFrame(out)


def _n2(deltas: pd.DataFrame, target: str) -> dict:
    row = _row(deltas, target, plan.SOFT_HEADLINE_LR, soft.PRIMARY_SET)
    if row is None:
        return {"status": INSUFFICIENT}
    return {
        "status": row["verdict"],
        **{
            k: _jsonable(row[k])
            for k in DELTA_COLUMNS
            if k not in ("set", "stat_set", "verdict")
        },
    }


def _n0(
    target: str,
    teachers: pd.DataFrame,
    reviews: Mapping[str, bool],
) -> dict:
    domain = plan.TARGET_DOMAIN[target]
    rows = teachers[teachers["target"] == target]
    chosen = rows[(rows["kind"] == soft.KIND_TREAT) & rows["chosen"].astype(bool)]
    placebo = rows[rows["kind"] == soft.KIND_PLACEBO]
    if chosen.empty:
        return {"status": INSUFFICIENT}
    w0 = bool(chosen["passed"].iloc[0])
    corpus_ok = bool(reviews.get(f"corpus:{target}", False))
    placebo_ok = bool(reviews.get(f"placebo:{domain}", False))
    gap = float(placebo["delta_vs_base"].iloc[0]) if not placebo.empty else math.nan
    return {
        "status": SUPPORTED if (w0 and corpus_ok and placebo_ok) else NOT_SUPPORTED,
        "teacher_epochs": int(chosen["epochs"].iloc[0]),
        "w0_passed": w0,
        "p_teacher_validation": float(chosen["p_validation"].iloc[0]),
        "w0_threshold": float(chosen["threshold"].iloc[0]),
        "corpus_review_approved": corpus_ok,
        "placebo_review_approved": placebo_ok,
        "placebo_minus_base": gap,
        "placebo_within_tol": bool(abs(gap) <= PLACEBO_BASE_TOL)
        if math.isfinite(gap)
        else None,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def claims(
    teachers: pd.DataFrame,
    deltas: pd.DataFrame,
    selected: pd.DataFrame,
    specificity: pd.DataFrame,
    offtopic: pd.DataFrame,
    reviews: Mapping[str, bool],
) -> dict:
    """Claims N0 (soft part) and N2-N5, per target, plus the per-domain list of
    targets meeting N2."""
    sel = {r.target: r for r in selected.itertuples(index=False)}
    out: dict = {
        "N0": {
            "rule": "W0 at the chosen dose, approved corpus and placebo reviews; "
            f"placebo - base on test reported (tolerance {PLACEBO_BASE_TOL})",
            "per_target": {t: _n0(t, teachers, reviews) for t in plan.TARGETS},
        },
        "N2": {
            "rule": f"lr {plan.SOFT_HEADLINE_LR:g}, test, primary: teacher-aware "
            f"bootstrap CI above 0, delta >= {EFFECT_FLOOR:g}, rung eligible, >= "
            f"{MIN_TEACHERS} teacher ftseeds; else underpowered / not_supported",
            "per_target": {t: _n2(deltas, t) for t in plan.TARGETS},
        },
        "N3": {
            "rule": "largest primary validation delta among eligible rungs; test "
            "delta reported (descriptive), with the N2 verdict rule at that rung"
        },
        "N4": {
            "rule": "at the N3 rung: target gain largest of the panel and >= "
            f"{SPECIFICITY_SHARE_MIN:g} of the positive panel gain",
            "per_target": {
                t: specificity_verdict(specificity, t) for t in plan.TARGETS
            },
        },
        "N5": {"rule": "off_topic_nocolor delta at the headline and N3 rungs"},
    }
    out["N3"]["per_target"] = {
        t: (
            {
                "status": sel[t].verdict,
                "kind": DESCRIPTIVE,
                **{k: _jsonable(v) for k, v in sel[t]._asdict().items()},
            }
            if t in sel
            else {"status": INSUFFICIENT}
        )
        for t in plan.TARGETS
    }
    out["N5"]["per_target"] = {
        t: {
            "status": DESCRIPTIVE,
            "rows": [
                {k: _jsonable(v) for k, v in r.items()}
                for r in offtopic[offtopic["target"] == t].to_dict("records")
            ]
            if not offtopic.empty
            else [],
        }
        for t in plan.TARGETS
    }
    out["generalizes"] = {
        t: _generalizes(out["N2"]["per_target"][t]) for t in plan.TARGETS
    }
    out["generalizes_by_domain"] = {
        d: [t for t in plan.DOMAINS[d].targets if out["generalizes"][t]["status"]]
        for d in plan.DOMAINS
    }
    return out


def _generalizes(n2: Mapping) -> dict:
    """N2 supported and the headline treatment students >= `EFFECT_FLOOR` above
    their untuned base on test."""
    gain = n2.get("treat_minus_base")
    ok = (
        n2["status"] == SUPPORTED
        and gain is not None
        and math.isfinite(gain)
        and gain >= EFFECT_FLOOR
    )
    return {"status": bool(ok), "n2": n2["status"], "treat_minus_base": gain}
