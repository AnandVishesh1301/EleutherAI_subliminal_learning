"""Pure metrics for the tmplopd campaign (docs/PREREG_TMPLOPD.md §3-§4).

Input is the leaves frame built by `pipelines.tmplopd_results_pipeline`: one row per
(unit, gseed, prompt set, scoring) with columns `tier`, `arm`, `variant`, `student`,
`ftseed`, `gseed`, `is_control`, `set`, `scoring`, `p_target`, `perplexity_ratio`,
`filter_drop_frac`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from subliminal_learning.analysis import stats as boot
from subliminal_learning.campaign import tmplopd_plan as plan
from subliminal_learning.campaign.ws160m_metrics import mark_quarantine, t_interval

CONTROL_KEY: tuple[str, ...] = ("variant", "student", "set", "scoring")
ARM_KEY: tuple[str, ...] = ("tier", "arm", "variant", "student")


def quarantine(
    leaves: pd.DataFrame, ppl_max: float = plan.QUARANTINE_PPL_RATIO_MAX
) -> pd.DataFrame:
    """Add `quarantined` / `quarantine_reason` (ws160m thresholds by default)."""
    return mark_quarantine(
        leaves, ppl_max=ppl_max, drop_max=plan.QUARANTINE_FILTER_DROP_MAX
    )


def control_means(leaves: pd.DataFrame) -> pd.DataFrame:
    """Mean non-quarantined control `p_target` per `CONTROL_KEY`."""
    ctrl = leaves[leaves["is_control"] & ~leaves["quarantined"]]
    grouped = ctrl.groupby(list(CONTROL_KEY))["p_target"]
    out = grouped.mean().rename("control_mean").to_frame()
    out["control_n"] = grouped.size()
    return out.reset_index()


def leaf_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    """Treatment leaves with ``delta = p_target - control_mean``.

    Raises:
        ValueError: A treatment leaf has no matching control (never substituted).
    """
    treat = leaves[~leaves["is_control"]]
    out = treat.merge(control_means(leaves), on=list(CONTROL_KEY), how="left")
    missing = out[out["control_mean"].isna()]
    if not missing.empty:
        keys = missing[list(CONTROL_KEY)].drop_duplicates().to_dict("records")
        raise ValueError(f"treatment leaves without a control: {keys}")
    out["delta"] = out["p_target"] - out["control_mean"]
    return out


def _cluster(frame: pd.DataFrame) -> np.ndarray:
    """Bootstrap group per leaf: the teacher seed, or the leaf itself if one teacher."""
    if frame["ftseed"].nunique() > 1:
        return frame["ftseed"].to_numpy()
    return frame["gseed"].to_numpy()


def _boot_ci(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return (math.nan, math.nan)
    est = boot.multi_bootstrap(
        lambda idx: float(values[idx].mean()),
        groups,
        n_boot=plan.BOOTSTRAP_N,
        seed=plan.BOOTSTRAP_SEED,
    )
    return (est.ci_low, est.ci_high)


def arm_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    """Per (arm key, set, scoring): Δ mean, t-CI over leaves, two-stage bootstrap CI."""
    rows = []
    for key, full in deltas.groupby([*ARM_KEY, "set", "scoring"]):
        g = full[~full["quarantined"]]
        if g.empty:
            continue
        mean, lo, hi = t_interval(g["delta"].tolist())
        b_lo, b_hi = _boot_ci(g["delta"].to_numpy(), _cluster(g))
        rows.append(
            {
                **dict(zip([*ARM_KEY, "set", "scoring"], key, strict=True)),
                "n_leaves": len(g),
                "n_teachers": g["ftseed"].nunique(),
                "ci_scope": "teacher+student" if g["ftseed"].nunique() > 1 else "student_only",
                "p_target_mean": g["p_target"].mean(),
                "control_mean": g["control_mean"].iloc[0],
                "delta_mean": mean,
                "delta_t_low": lo,
                "delta_t_high": hi,
                "delta_boot_low": b_lo,
                "delta_boot_high": b_hi,
                "perplexity_ratio_mean": g["perplexity_ratio"].mean(),
                "flagged_rate_mean": g["filter_drop_frac"].mean(),
                "n_quarantined": int(full["quarantined"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["transfers"] = (out["delta_mean"] >= plan.MIN_DELTA) & (out["delta_t_low"] > 0)
    return out


def contrast(deltas: pd.DataFrame, arm: str, ref: str, *, tier: str, variant: str,
             student: str, prompt_set: str, scoring: str) -> dict:
    """``Δ(arm) − Δ(ref)`` with a stratified two-stage bootstrap CI."""
    sel = deltas[
        (deltas["tier"] == tier)
        & (deltas["variant"] == variant)
        & (deltas["student"] == student)
        & (deltas["set"] == prompt_set)
        & (deltas["scoring"] == scoring)
        & ~deltas["quarantined"]
    ]
    a = sel[sel["arm"] == arm]
    r = sel[sel["arm"] == ref]
    row = {"arm": arm, "ref": ref, "tier": tier, "variant": variant, "student": student,
           "set": prompt_set, "scoring": scoring, "n_arm": len(a), "n_ref": len(r)}
    if a.empty or r.empty:
        return {**row, "diff": math.nan, "boot_low": math.nan, "boot_high": math.nan}
    rng = np.random.default_rng(plan.BOOTSTRAP_SEED)
    groups = []
    for frame in (a, r):
        vals = frame["delta"].to_numpy()
        clusters = _cluster(frame)
        levels = np.unique(clusters)
        groups.append((vals, [np.flatnonzero(clusters == lv) for lv in levels]))
    draws = np.empty(plan.BOOTSTRAP_N)
    for b in range(plan.BOOTSTRAP_N):
        means = []
        for vals, rows in groups:
            picked = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
            means.append(vals[picked[rng.integers(0, picked.size, picked.size)]].mean())
        draws[b] = means[0] - means[1]
    diff = float(a["delta"].mean() - r["delta"].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {**row, "diff": diff, "boot_low": float(lo), "boot_high": float(hi)}


def efficiency(summary: pd.DataFrame, teachers: pd.DataFrame) -> pd.DataFrame:
    """Δ_test / (teacher p_test − base prior) per treatment arm on `PRIMARY_PROMPT_SET`."""
    prim = summary[
        (summary["set"] == plan.PRIMARY_PROMPT_SET)
        & (summary["scoring"] == plan.PRIMARY_SCORING)
    ]
    t = teachers[
        (teachers["set"] == plan.PRIMARY_PROMPT_SET)
        & (teachers["scoring"] == plan.PRIMARY_SCORING)
    ][["arm", "teacher_p_target", "base_p_target"]]
    out = prim.merge(t, on="arm", how="left")
    out["teacher_shift"] = out["teacher_p_target"] - out["base_p_target"]
    out["efficiency"] = out["delta_mean"] / out["teacher_shift"]
    return out


RETENTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("legacy", "vmaskf"),
    ("noeos", "vmaskf_noeos"),
)


def retention(summary: pd.DataFrame, *, prompt_set: str, scoring: str) -> pd.DataFrame:
    """Δ(lion forms masked) / Δ(unmasked) per arm on the self pair (C3)."""
    s = summary[
        (summary["set"] == prompt_set)
        & (summary["scoring"] == scoring)
        & (summary["student"] == plan.SELF_BASE)
    ]
    frames = []
    for base, masked in RETENTION_PAIRS:
        unmasked = s[s["variant"] == base].groupby("arm")["delta_mean"].first()
        vmask = s[s["variant"] == masked].groupby("arm")["delta_mean"].first()
        out = pd.DataFrame({"delta_unmasked": unmasked, "delta_masked": vmask}).dropna()
        out["retained"] = out["delta_masked"] / out["delta_unmasked"]
        out["variant_pair"] = f"{base}->{masked}"
        frames.append(out.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def claims(summary: pd.DataFrame, contrasts: pd.DataFrame,
           retained: pd.DataFrame, concordance: Mapping[str, float]) -> dict:
    """C0-C3 verdicts on the primary endpoint (C4/C5 are descriptive tables)."""
    c1: dict = {}
    for tier in (plan.TIER_C, plan.TIER_N):
        prim = summary[
            (summary["set"] == plan.PRIMARY_PROMPT_SET)
            & (summary["scoring"] == plan.PRIMARY_SCORING)
            & (summary["tier"] == tier)
        ].set_index("arm")
        c1[tier] = {
            arm: None if arm not in prim.index else {
                "delta_mean": float(prim.loc[arm, "delta_mean"]),
                "t_ci": [
                    float(prim.loc[arm, "delta_t_low"]),
                    float(prim.loc[arm, "delta_t_high"]),
                ],
                "n_leaves": int(prim.loc[arm, "n_leaves"]),
                "n_quarantined": int(prim.loc[arm, "n_quarantined"]),
                "ci_scope": prim.loc[arm, "ci_scope"],
                "holds": bool(prim.loc[arm, "transfers"]),
            }
            for arm in ("S", "A10", "A15")
        }
    return {
        "C0": {**dict(concordance),
               "flag": bool(abs(concordance.get("mean_diff", 0.0)) > plan.REPRO_FLAG_DIFF)},
        "C1": c1,
        "C2": contrasts.to_dict("records"),
        "C3": retained.to_dict("records"),
    }
