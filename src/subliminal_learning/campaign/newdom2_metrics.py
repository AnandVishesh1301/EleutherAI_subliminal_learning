"""Pure metric math of `newdom2` (prereg § Part II, claims E0-E5).

Frame contracts (built by `pipelines.newdom2_results_pipeline`):

- leaves: Part I `newdom_soft_metrics` leaf columns (``key, kind, target, lr, r, g,
  p_<set>, base_p_<set>, quarantined``) for every Part II leaf and every reused Part I
  leaf; an arm is a set of treatment keys and a set of placebo keys over this frame;
- panel: one row per (leaf key, generic set, panel item) with ``p``.

An arm's delta is Part I's: mean(treatment) - mean(placebo) at lr 4e-4 on the primary
leaves, with the teacher-aware percentile bootstrap (10,000 resamples, seed 0) and the
N2 verdict rule. A difference D = delta_arm - delta_ref resamples each arm's teacher
clusters and placebo leaves independently; a placebo leaf set both arms share is
resampled once per draw, so it cancels. ``improved`` iff the 95% interval of D lies
above 0, ``worse`` iff below, else ``no_clear_change``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis import stats as boot
from subliminal_learning.campaign import newdom2_plan as plan2
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.campaign import newdom_soft_metrics as metrics

IMPROVED, WORSE, NO_CLEAR = "improved", "worse", "no_clear_change"
VALID, INVALID = "valid", "invalid"
SET = soft.PRIMARY_SET
LR = plan2.STUDENT_LR


def arm_frame(
    leaves: pd.DataFrame, treat: Iterable[str], placebo: Iterable[str]
) -> pd.DataFrame:
    """The arm's rows of `leaves` with ``kind`` set by role.

    Raises:
        ValueError: A key is missing from `leaves` or listed in both roles.
    """
    t, p = list(treat), list(placebo)
    if set(t) & set(p):
        raise ValueError("a key is both treatment and placebo")
    missing = sorted(set(t + p) - set(leaves["key"]))
    if missing:
        raise ValueError(f"arm keys missing from the leaves frame: {missing[:5]}")
    sub = leaves[leaves["key"].isin(t + p)].copy()
    sub["kind"] = np.where(sub["key"].isin(t), soft.KIND_TREAT, soft.KIND_PLACEBO)
    return sub


def arm_delta(sub: pd.DataFrame, target: str, s: str = SET) -> dict:
    """Part I's N2 statistics and verdict of one arm on set `s` (primary leaves)."""
    treat, placebo, n_q = metrics._arms(sub, target, LR, metrics.FULL)
    col_sel = f"p_{metrics.SELECTION_SET}"
    n_rung = int(
        np.isfinite(treat[col_sel]).sum() + np.isfinite(placebo[col_sel]).sum()
    )
    eligible = n_rung > 0 and n_q <= metrics.RUNG_EXCLUDED_MAX * n_rung
    treat, placebo, _ = metrics._arms(sub, target, LR, metrics.PRIMARY)
    col = f"p_{s}"
    row = {
        "target": target,
        "set": s,
        "n_quarantined": n_q,
        "eligible": eligible,
        **metrics.two_arm_delta(treat[col], placebo[col], list(treat["r"])),
        **metrics.seed_interval(treat, placebo, col),
        "base_p": metrics._base(treat, placebo, s),
    }
    row["treat_minus_base"] = row["treat_mean"] - row["base_p"]
    row["placebo_minus_base"] = row["placebo_mean"] - row["base_p"]
    row["verdict"] = metrics.transfer_verdict(row)
    row["generalizes"] = bool(
        row["verdict"] == metrics.SUPPORTED
        and math.isfinite(row["treat_minus_base"])
        and row["treat_minus_base"] >= metrics.EFFECT_FLOOR
    )
    return row


def _arm_draws(
    treat: Sequence[float],
    teachers: Sequence[object],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = [
        np.asarray([v for v, k in zip(treat, teachers, strict=True) if k == key])
        for key in sorted(set(teachers), key=str)
    ]
    return metrics._cluster_means(groups, n_boot, rng)


def _placebo_draws(
    placebo: Sequence[float], n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    p = np.asarray(placebo, dtype=np.float64)
    return metrics._cluster_means([p[i : i + 1] for i in range(p.size)], n_boot, rng)


def _finite(values: Sequence[float], labels: Sequence[object] | None = None):
    keep = [i for i, v in enumerate(values) if math.isfinite(float(v))]
    vals = [float(values[i]) for i in keep]
    return vals, None if labels is None else [labels[i] for i in keep]


def delta_difference(
    x_treat: Sequence[float],
    x_teachers: Sequence[object],
    x_placebo: Sequence[float],
    ref_treat: Sequence[float],
    ref_teachers: Sequence[object],
    ref_placebo: Sequence[float],
    *,
    shared_placebo: bool,
    n_boot: int = metrics.N_BOOT,
    seed: int = metrics.BOOT_SEED,
    confidence: float = metrics.CONFIDENCE,
) -> dict:
    """D = delta_x - delta_ref with its bootstrap interval and verdict.

    ``improved`` needs the interval above 0, D >= `EFFECT_FLOOR` and every arm
    teacher's delta (teacher mean - arm placebo mean) above delta_ref; ``worse`` the
    mirror image; anything else is ``no_clear_change``.

    Args:
        x_treat / ref_treat: Treatment leaf values of the arm / its reference.
        x_teachers / ref_teachers: Teacher label of each treatment value.
        x_placebo / ref_placebo: Placebo leaf values (each its own cluster).
        shared_placebo: Both arms use the same placebo leaves (one draw per resample).
    """
    xt, xk = _finite(x_treat, x_teachers)
    rt, rk = _finite(ref_treat, ref_teachers)
    xp, _ = _finite(x_placebo)
    rp, _ = _finite(ref_placebo)
    out = dict.fromkeys(("D", "ci_low", "ci_high", "p_boot", "delta_ref"), math.nan)
    out.update(
        shared_placebo=shared_placebo,
        n_teachers=0,
        n_teachers_above_ref=0,
        n_teachers_below_ref=0,
        verdict=metrics.INSUFFICIENT,
    )
    if not (xt and rt and xp and rp):
        return out
    if shared_placebo and sorted(xp) != sorted(rp):
        raise ValueError("shared placebo arms carry different placebo values")
    delta_ref = float(np.mean(rt) - np.mean(rp))
    d = float(np.mean(xt) - np.mean(xp)) - delta_ref
    teacher_deltas = [
        float(
            np.mean([v for v, k in zip(xt, xk, strict=True) if k == key]) - np.mean(xp)
        )
        for key in sorted(set(xk), key=str)
    ]
    above = sum(t > delta_ref for t in teacher_deltas)
    below = sum(t < delta_ref for t in teacher_deltas)
    rng = np.random.default_rng(seed)
    tx = _arm_draws(xt, xk, n_boot, rng)
    tr = _arm_draws(rt, rk, n_boot, rng)
    px = _placebo_draws(xp, n_boot, rng)
    pr = px if shared_placebo else _placebo_draws(rp, n_boot, rng)
    est = boot.summarize_draws(d, (tx - px) - (tr - pr), confidence=confidence)
    n = len(teacher_deltas)
    if est.ci_low > 0 and d >= metrics.EFFECT_FLOOR and above == n:
        verdict = IMPROVED
    elif est.ci_high < 0 and d <= -metrics.EFFECT_FLOOR and below == n:
        verdict = WORSE
    else:
        verdict = NO_CLEAR
    out.update(
        D=d,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        p_boot=est.p_boot,
        delta_ref=delta_ref,
        n_teachers=n,
        n_teachers_above_ref=above,
        n_teachers_below_ref=below,
        verdict=verdict,
    )
    return out


def arm_difference(
    x: pd.DataFrame, ref: pd.DataFrame, target: str, s: str = SET
) -> dict:
    """`delta_difference` of two arm frames on set `s` (primary leaves)."""
    xt, xp, _ = metrics._arms(x, target, LR, metrics.PRIMARY)
    rt, rp, _ = metrics._arms(ref, target, LR, metrics.PRIMARY)
    shared = set(xp["key"]) == set(rp["key"])
    col = f"p_{s}"
    return delta_difference(
        list(xt[col]),
        list(xt["r"]),
        list(xp[col]),
        list(rt[col]),
        list(rt["r"]),
        list(rp[col]),
        shared_placebo=shared,
    )


def arm_specificity(
    panel: pd.DataFrame, sub: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, dict]:
    """(panel gains, N4 verdict) of one arm frame at lr 4e-4."""
    gains = metrics.panel_gains(panel, sub, target, LR)
    return gains, metrics.specificity_verdict(gains, target)


def teacher_panel_share(
    panel_means: Sequence[Mapping[str, float]], target: str
) -> float:
    """Mean over ftseeds of the target's share of the teacher's panel mass on test."""
    shares = [
        m[target] / total
        for m in panel_means
        if (total := sum(m.values())) > 0 and math.isfinite(total)
    ]
    return float(np.mean(shares)) if shares else math.nan


# ---- leak scan (E4) -----------------------------------------------------------------------


def leak_counts(records: Iterable[Mapping], forms: tuple[str, ...]) -> dict:
    """Per-leaf leak counts over on-policy rollout rows: rows, rows kept by the leaf's
    screen, kept rows containing a target form as a plain substring, rows the
    `substring` rule removed. A row logged twice (a resumed retry appends) counts once,
    as its last record per ``(step, batch_idx)``."""
    rows = kept = hits = removed = 0
    lowered_forms = tuple(f.lower() for f in forms)
    last = {(r.get("step"), r.get("batch_idx")): r for r in records}
    for rec in last.values():
        if not rec.get("on_policy", True):
            continue
        rows += 1
        if rec.get("matched_rule") == "substring":
            removed += 1
        if rec.get("flagged"):
            continue
        kept += 1
        text = soft.normalize_screen_text(str(rec.get("completion", "")))
        if any(f in text for f in lowered_forms):
            hits += 1
    return {
        "rows": rows,
        "kept": kept,
        "kept_substring_hits": hits,
        "substring_removed": removed,
        "kept_substring_rate": hits / kept if kept else math.nan,
        "substring_removed_rate": removed / rows if rows else math.nan,
    }


# ---- claims ----------------------------------------------------------------------------


def _js(value: object) -> object:
    return metrics._jsonable(value)


def _clean(row: Mapping) -> dict:
    return {k: _js(v) for k, v in row.items()}


def claims(
    arms: Mapping[tuple[str, str], Mapping],
    refs: Mapping[tuple[str, str], Mapping],
    validity: Mapping[tuple[str, str], Mapping],
    leaks: Mapping[tuple[str, str], Mapping],
    b_stats: Mapping[str, Mapping] | None = None,
) -> dict:
    """Claims E0-E5 over the (arm, target) units present in `arms` (a partial build
    scores what exists; a missing unit is ``insufficient_data``).

    Args:
        arms: (arm, target) -> ``delta`` (`arm_delta`), ``difference``
            (`arm_difference` vs its reference), ``specificity`` (N4 verdict),
            ``teacher_panel_share``; ``ref_delta`` / ``ref_specificity`` of the
            reference.
        refs: (arm, target) -> reference tag (`newdom2_plan.reference_arm`).
        validity: (arm, target) -> ``problems`` (list) and reported fields.
        leaks: (arm, target) -> leak summary (treatment / placebo rates, zero check).
        b_stats: B target -> corpus filter statistics (its manifest).
    """
    b_stats = b_stats or {}

    def e0_status(arm: str, t: str) -> str:
        return INVALID if validity[(arm, t)].get("problems") else VALID

    def per(arm: str, fn) -> dict:
        return {
            t: fn(arm, t) if (arm, t) in arms else {"status": metrics.INSUFFICIENT}
            for t in plan2.ARMS[arm]
        }

    def e0(arm: str, t: str) -> dict:
        v = validity[(arm, t)]
        reported = _clean({k: x for k, x in v.items() if k != "problems"})
        return {
            "status": e0_status(arm, t),
            **reported,
            "problems": list(v["problems"]),
        }

    def compare(arm: str, t: str) -> dict:
        a = arms[(arm, t)]
        return {
            "status": a["difference"]["verdict"],
            "e0_status": e0_status(arm, t),
            "reference": refs[(arm, t)],
            "difference": _clean(a["difference"]),
            "delta": _clean(a["delta"]),
            "reference_delta": _clean(a["ref_delta"]),
            "n2_verdict": a["delta"]["verdict"],
            "generalizes": a["delta"]["generalizes"],
        }

    def e2(arm: str, t: str) -> dict:
        out = compare(arm, t)
        a = arms[(arm, t)]
        out["specificity"] = _clean(a["specificity"])
        out["reference_specificity"] = _clean(a["ref_specificity"])
        out["teacher_panel_share"] = _js(a.get("teacher_panel_share"))
        out["reference_teacher_panel_share"] = _js(a.get("ref_teacher_panel_share"))
        out["corpus"] = _clean(b_stats.get(t, {}))
        return out

    def e3(arm: str, t: str) -> dict:
        a = arms[(arm, t)]
        delta, ref = a["delta"], a["ref_delta"]
        survives = bool(delta["verdict"] == metrics.SUPPORTED and delta["generalizes"])
        ratio = (
            delta["delta"] / ref["delta"]
            if math.isfinite(ref["delta"]) and ref["delta"] != 0
            else math.nan
        )
        return {
            "status": "survives" if survives else "does_not_survive",
            "e0_status": e0_status(arm, t),
            "delta": _clean(delta),
            "reference_delta": _clean(ref),
            "difference": _clean(a["difference"]),
            "ratio_to_part1": _js(ratio),
        }

    def e4(arm: str, t: str) -> dict:
        return _clean(leaks[(arm, t)])

    best: dict = {}
    for (arm, t), a in arms.items():
        d = a["delta"]
        if d["verdict"] != metrics.SUPPORTED or e0_status(arm, t) == INVALID:
            continue
        if t not in best or d["delta"] > best[t]["delta"]:
            best[t] = {
                "arm": arm,
                "delta": _js(d["delta"]),
                "ci_low": _js(d["ci_low"]),
                "ci_high": _js(d["ci_high"]),
                "generalizes": d["generalizes"],
            }
    targets = sorted({t for ts in plan2.ARMS.values() for t in ts})
    return {
        "exploratory": True,
        "n_units": len(arms),
        "E0": {
            "rule": "approved B corpus / strict_sub screen reviews; 3 ftseed manifests "
            "per Part II teacher config on the planned corpus sha and one step count; "
            "A placebo steps = A steps; B steps = 740; every strict_sub leaf has "
            "rollout rows and keeps no substring hit. W0 and placebo - base reported, "
            "never gating.",
            "per_arm": {arm: per(arm, e0) for arm in plan2.ARMS},
        },
        "E1": {
            "rule": "arm A: D = delta_A - delta_ref; improved iff the 95% bootstrap CI "
            f"is above 0, D >= {metrics.EFFECT_FLOOR:g} and every A teacher's delta "
            "exceeds delta_ref; worse is the mirror image; else no_clear_change. "
            "delta_A carries the N2 verdict and the generalizes bar",
            "per_target": per(plan2.ARM_A, compare),
        },
        "E2": {
            "rule": "arm B: same rule as E1; N4 target share, teacher panel share and "
            "the corpus filter statistics reported against the reference",
            "per_target": per(plan2.ARM_B, e2),
        },
        "E3": {
            "rule": "arm C: survives iff the N2 verdict is supported and treatment - "
            "base >= 0.02; D and the ratio to Part I reported",
            "per_target": per(plan2.ARM_C, e3),
        },
        "E4": {
            "rule": "per arm: kept-row substring rate, treatment vs placebo; must be 0 "
            "under strict_sub; rows removed by the substring rule reported",
            "per_arm": {arm: per(arm, e4) for arm in plan2.ARMS},
        },
        "E5": {
            "rule": "per target, the largest Part II delta whose N2 verdict is supported "
            "and whose E0 is valid (selected after the fact, descriptive)",
            "per_target": {
                t: best.get(t, {"status": metrics.INSUFFICIENT}) for t in targets
            },
        },
    }
