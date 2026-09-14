"""Pure frames and claims of the OPD-limits campaign (`docs/PREREG_OPDLIM.md`).

Leaf frame (`LEAF_COLUMNS`): one row per planned leaf (treatment or trained
control) at one gseed, with its plan identity, primary endpoints, per-seed
capability, last-10-step telemetry and off-policy screen counts. ``k`` is
numeric (`VOCAB_SIZE` = full); ``subset_n`` is NaN without a prompt subset;
``ftseed`` and ``teacher_param_tag`` are NaN on controls. Extra
``lmeval_delta_{task}_{metric}`` columns ride along when LM-Eval ran.

A treatment is differenced only against controls of its pair's control kind
with the same `MATCH_KEYS` (run set, student token, teacher token, GKD tag,
gseed); anchor-pair controls have ``teacher_token == student_token``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.campaign.opdlim_pairs import (
    KIND_TREATMENT,
    all_controls,
    all_leaves,
    control_kind,
)
from subliminal_learning.campaign.opdlim_plan import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COLLAPSED_ENTROPY_NATS,
    CONCORDANCE_BAND,
    EQUIVALENCE_BAND,
    FLAT_SLOPE_BAND,
    HP_BASE,
    K_DIR,
    N_SAT_FRACTION,
    NOT_TOKEN_MEDIATED_MIN,
    PAIR_BY_NAME,
    PAIRS,
    POOL_N_PROMPTS,
    QUARANTINE_FILTER_DROP_MAX,
    QUARANTINE_PPL_RATIO_MAX,
    SYMMETRIC_BAND,
    T0_MIN_DELTA,
    T0_VISIBILITY_FLAG,
    TEACHER_P_TARGET_BY_TAG,
    TEACHER_TAG_EP5,
    TELEMETRY_LAST_N,
    TIER2,
    TOKEN_MEDIATED_MAX,
    VOCAB_SIZE,
    Block,
    Tier,
)
from subliminal_learning.campaign.topk_metrics import (
    _finite,
    _mean_diff_ci,
    _ols_slope,
    _percentile_ci,
    _ratio_ci,
    _spearman,
)
from subliminal_learning.campaign.topk_metrics import curve_rows as topk_curve_rows

LMEVAL_PREFIX = "lmeval_delta_"
TELEMETRY_FIELDS: tuple[str, ...] = (
    "student_entropy",
    "distinct_2",
    "student_topk_mass",
    "top1_agree",
    "topk_coverage",
    "target_in_topk_rate",
)
SCREEN_FIELDS: tuple[str, ...] = (
    "n_off_policy_steps_screened",
    "n_off_policy_rows_screened",
    "n_off_policy_flagged",
    "n_off_policy_rows_logged",
)
MATCH_KEYS: tuple[str, ...] = (
    "run_set_id",
    "student_token",
    "teacher_token",
    "gkd_tag",
    "gseed",
)

LEAF_COLUMNS: tuple[str, ...] = (
    "tier",
    "run_set_id",
    "block",
    "pair",
    "teacher_token",
    "student_token",
    "kind",
    "objective",
    "form",
    "k",
    "lmbda",
    "vmask_forms",
    "subset_n",
    "hp_lr",
    "hp_lora_r",
    "hp_lora_alpha",
    "teacher_param_tag",
    "teacher_p_target",
    "gkd_tag",
    "n_design",
    "ftseed",
    "gseed",
    "p_target",
    "rank_mean",
    "fc_share",
    "perplexity_ratio_seed",
    "ce_nats_seed",
    "ce_nats_base",
    "calib_ece_seed",
    "calib_ece_base",
    "calib_entropy",
    "filter_drop",
    "n_log_steps",
    *TELEMETRY_FIELDS,
    *SCREEN_FIELDS,
    "on_policy_only",
    "quarantined",
    "quarantine_reason",
)
DELTA_COLUMNS: tuple[str, ...] = (
    *LEAF_COLUMNS,
    "n_live_controls",
    "control_p",
    "control_fc",
    "control_rank",
    "control_p_unq",
    "control_fc_unq",
    "control_ece",
    "delta_p",
    "delta_fc",
    "delta_rank",
    "delta_p_unq",
    "delta_fc_unq",
    "delta_ece",
)
CELL_KEYS: tuple[str, ...] = (
    "tier",
    "run_set_id",
    "block",
    "pair",
    "teacher_token",
    "student_token",
    "objective",
    "form",
    "k",
    "lmbda",
    "vmask_forms",
    "subset_n",
    "hp_lr",
    "hp_lora_r",
    "hp_lora_alpha",
    "teacher_param_tag",
)
CELL_COLUMNS: tuple[str, ...] = (
    *CELL_KEYS,
    "teacher_p_target",
    "n_design",
    "n_min_live",
    "n_total",
    "n_live",
    "mean_p_target",
    "mean_control_p",
    "mean_delta_p",
    "ci_low_p",
    "ci_high_p",
    "mean_delta_fc",
    "ci_low_fc",
    "ci_high_fc",
    "mean_delta_rank",
    "void_capability",
    "t0",
    "n_unq",
    "mean_delta_p_unq",
    "ci_low_p_unq",
    "ci_high_p_unq",
    "mean_delta_fc_unq",
    "ci_low_fc_unq",
    "ci_high_fc_unq",
    "t0_unquarantined",
    "mean_target_in_topk_rate",
    "visibility_flag",
)
FAMILY_KEYS: tuple[str, ...] = (
    "tier",
    "run_set_id",
    "pair",
    "teacher_token",
    "student_token",
    "form",
    "lmbda",
    "vmask_forms",
    "subset_n",
    "hp_lr",
    "hp_lora_r",
    "hp_lora_alpha",
    "teacher_param_tag",
)
CURVE_COLUMNS: tuple[str, ...] = (
    *FAMILY_KEYS,
    "family",
    "block",
    "objective",
    "k",
    "k_slug",
    "n_design",
    "n_live",
    "mean_delta_p",
    "ci_low_p",
    "ci_high_p",
    "ci_low_fc",
    "void_capability",
    "t0",
    "t0_unquarantined",
    "visibility_flag",
)
KSTAR_COLUMNS: tuple[str, ...] = (
    *FAMILY_KEYS,
    "family",
    "status",
    "k_star",
    "k_star_slug",
    "holds_set",
    "measured",
    "spearman_logk",
)
RHO_COLUMNS: tuple[str, ...] = (
    "measure",
    "numerator",
    "denominator",
    "k",
    "k_slug",
    "n_num",
    "n_den",
    "mean_num",
    "mean_den",
    "rho",
    "ci_low",
    "ci_high",
    "rho_full",
    "covers_full",
)
DIRECTION_COLUMNS: tuple[str, ...] = (
    "k",
    "k_slug",
    "n_p1",
    "n_d",
    "mean_p1",
    "mean_d",
    "a",
    "ci_low",
    "ci_high",
    "label",
)
HP_COLUMNS: tuple[str, ...] = (
    "tier",
    "block",
    "pair",
    "k",
    "k_slug",
    "status",
    "m",
    "m_ci_low",
    "m_ci_high",
    "m_n_pairs",
    "i",
    "i_ci_low",
    "i_ci_high",
    "i_n_pairs",
)
DOSE_COLUMNS: tuple[str, ...] = (
    "k",
    "k_slug",
    "teacher_param_tag",
    "teacher_p_target",
    "n_live",
    "mean_delta_p",
    "n_points",
    "slope",
    "slope_minus_full",
)
CAPABILITY_METRICS: tuple[str, ...] = (
    "mean_ppl_ratio",
    "mean_ce_nats_minus_base",
    "mean_ece",
    "mean_ece_minus_control",
    "mean_entropy",
    *(f"mean_{f}" for f in TELEMETRY_FIELDS),
    "mean_filter_drop",
)
CAPABILITY_COLUMNS: tuple[str, ...] = (
    *CELL_KEYS,
    "n_leaves",
    "n_quarantined",
    *CAPABILITY_METRICS,
)
TELEMETRY_COLUMNS: tuple[str, ...] = (
    "tier",
    "block",
    "pair",
    "objective",
    "k",
    "kind",
    "n_leaves",
    "n_with_logs",
    "mean_filter_drop",
    *(f"mean_{f}" for f in TELEMETRY_FIELDS),
)
QUARANTINE_COLUMNS: tuple[str, ...] = (
    "tier",
    "run_set_id",
    "block",
    "pair",
    "kind",
    "objective",
    "gkd_tag",
    "ftseed",
    "gseed",
    "perplexity_ratio_seed",
    "filter_drop",
    "quarantine_reason",
)
COVERAGE_COLUMNS: tuple[str, ...] = (
    "tier",
    "block",
    "pair",
    "objective",
    "kind",
    "n_expected",
    "n_present",
    "n_fc",
    "n_ppl",
    "n_ece",
    "n_telemetry",
    "n_quarantined",
    "complete",
)
GHOST_COLUMNS: tuple[str, ...] = (
    "pair",
    "k",
    "k_slug",
    "n_ghost",
    "n_ce",
    "mean_ghost",
    "mean_ce",
    "d",
    "ci_low",
    "ci_high",
    "label",
    "ece_ghost",
    "ece_ce",
    "ece_diff",
)
VMASK_COLUMNS: tuple[str, ...] = (
    "pair",
    "k",
    "k_slug",
    "n_vmask",
    "n_ce",
    "mean_vmask",
    "mean_ce",
    "ce_t0",
    "r_v",
    "ci_low",
    "ci_high",
    "label",
)
LAMBDA_COLUMNS: tuple[str, ...] = (
    "pair",
    "k",
    "k_slug",
    "n_lmb0",
    "n_lmb1",
    "mean_lmb0",
    "mean_lmb1",
    "d",
    "ci_low",
    "ci_high",
    "label",
)
SCREEN_COLUMNS: tuple[str, ...] = (
    "tier",
    "block",
    "pair",
    "objective",
    "k",
    "kind",
    "n_leaves",
    "n_summaries",
    *SCREEN_FIELDS,
    "off_policy_flag_frac",
)
DATA_CURVE_COLUMNS: tuple[str, ...] = (
    "pair",
    "k",
    "k_slug",
    "subset_n",
    "log2_n",
    "block",
    "objective",
    "n_design",
    "n_live",
    "mean_delta_p",
    "ci_low_p",
    "ci_high_p",
    "void_capability",
    "t0",
)
DATA_SLOPE_COLUMNS: tuple[str, ...] = (
    "pair",
    "k",
    "k_slug",
    "n_levels",
    "n_leaves",
    "slope",
    "ci_low",
    "ci_high",
    "label",
    "n_sat",
    "mean_at_pool",
)
N2_COLUMNS: tuple[str, ...] = (
    "pair",
    "k",
    "k_slug",
    "slope_k",
    "slope_full",
    "diff",
    "ci_low",
    "ci_high",
    "status",
)
CONCORDANCE_COLUMNS: tuple[str, ...] = (
    "pair",
    "k",
    "k_slug",
    "n_pairs",
    "mean_diff",
    "ci_low",
    "ci_high",
    "label",
)

_NUMERIC: tuple[str, ...] = (
    "tier",
    "k",
    "lmbda",
    "subset_n",
    "hp_lr",
    "hp_lora_r",
    "hp_lora_alpha",
    "teacher_p_target",
    "n_design",
    "ftseed",
    "gseed",
    "p_target",
    "rank_mean",
    "fc_share",
    "perplexity_ratio_seed",
    "ce_nats_seed",
    "ce_nats_base",
    "calib_ece_seed",
    "calib_ece_base",
    "calib_entropy",
    "filter_drop",
    "n_log_steps",
    *TELEMETRY_FIELDS,
    *SCREEN_FIELDS,
)
_GATED_X0 = "ok"


# ---- small helpers -------------------------------------------------------------


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _nan() -> float:
    return float("nan")


def k_slug(k: float) -> str:
    """``full`` at `VOCAB_SIZE`, else ``k{K}``."""
    return "full" if int(k) >= VOCAB_SIZE else f"k{int(k)}"


def _k_plot(k: int | None) -> int:
    return VOCAB_SIZE if k is None else int(k)


def _flags(frame: pd.DataFrame, column: str = "quarantined") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def lmeval_columns(frame: pd.DataFrame) -> list[str]:
    """LM-Eval delta columns of a leaf / delta frame, sorted."""
    return sorted(c for c in frame.columns if str(c).startswith(LMEVAL_PREFIX))


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in (*_NUMERIC, *lmeval_columns(frame)):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in LEAF_COLUMNS:
        if col not in out.columns:
            out[col] = False if col == "quarantined" else _nan()
    return out


def _mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else _nan()


def _key_strings(frame: pd.DataFrame, kinds: pd.Series) -> pd.Series:
    parts = [kinds.astype(str)]
    for col in MATCH_KEYS:
        values = frame[col]
        parts.append(
            values.astype(int).astype(str) if col == "gseed" else values.astype(str)
        )
    out = parts[0]
    for part in parts[1:]:
        out = out + "|" + part
    return out


def _matched_kind(frame: pd.DataFrame) -> pd.Series:
    return frame["pair"].map(lambda p: control_kind(PAIR_BY_NAME[str(p)]))


def _recipe_mask(
    frame: pd.DataFrame,
    *,
    form: str = "ce",
    lmbda: float = 1.0,
    vmask: bool = False,
    subset_n: int | str | None = None,
    base_hp: bool = True,
    teacher_tag: str | None = TEACHER_TAG_EP5,
) -> pd.Series:
    """Rows of one objective recipe; ``subset_n="any"`` keeps every subset."""
    if frame.empty:
        return pd.Series(False, index=frame.index)
    mask = (frame["form"].astype(str) == form) & np.isclose(
        pd.to_numeric(frame["lmbda"], errors="coerce"), lmbda
    )
    mask &= frame["vmask_forms"].fillna(False).astype(bool) == vmask
    sub = pd.to_numeric(frame["subset_n"], errors="coerce")
    if subset_n is None:
        mask &= sub.isna()
    elif subset_n == "any":
        mask &= sub.notna()
    else:
        mask &= sub == float(subset_n)
    if base_hp:
        mask &= np.isclose(pd.to_numeric(frame["hp_lr"], errors="coerce"), HP_BASE.lr)
        mask &= pd.to_numeric(frame["hp_lora_r"], errors="coerce") == HP_BASE.lora_r
        mask &= (
            pd.to_numeric(frame["hp_lora_alpha"], errors="coerce") == HP_BASE.lora_alpha
        )
    if teacher_tag is not None:
        mask &= frame["teacher_param_tag"].astype(str) == teacher_tag
    return mask.astype(bool)


def _ladder_mask(frame: pd.DataFrame) -> pd.Series:
    """Base-HP ep5 on-policy CE rows (the C0/C1 ladder and the Tier 1 CE reference)."""
    return _recipe_mask(frame)


def _live(frame: pd.DataFrame) -> pd.DataFrame:
    """Live treatment rows with a finite ``delta_p``."""
    if frame.empty:
        return frame
    keep = (frame["kind"].astype(str) == KIND_TREATMENT) & ~_flags(frame)
    keep &= np.isfinite(pd.to_numeric(frame["delta_p"], errors="coerce"))
    return frame.loc[keep]


def _at(frame: pd.DataFrame, pair: str, k: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = (frame["pair"].astype(str) == pair) & (
        pd.to_numeric(frame["k"], errors="coerce") == k
    )
    return frame.loc[keep]


def _values(frame: pd.DataFrame, column: str = "delta_p") -> list[float]:
    if frame.empty or column not in frame.columns:
        return []
    return [
        float(v) for v in pd.to_numeric(frame[column], errors="coerce") if _finite(v)
    ]


def _interval_label(
    low: float,
    high: float,
    *,
    above: str,
    below: str,
    within: str,
    band: float,
    within_before_below: bool = False,
) -> str:
    """First-match label of a CI in PREREG order (``void`` when non-finite)."""
    if not (_finite(low) and _finite(high)):
        return "void"
    rules = [(above, low > 0.0)]
    inside = (within, -band <= low and high <= band)
    lower = (below, high < 0.0)
    rules.extend([inside, lower] if within_before_below else [lower, inside])
    for label, hit in rules:
        if hit:
            return label
    return "inconclusive"


def _gated(x0_status: str, status: str) -> str:
    return status if x0_status == _GATED_X0 else x0_status


def _pairs_of(t: Tier, predicate) -> list[tuple[str, int]]:
    """``(pair, K)`` grid of a tier's objectives matching `predicate`, K descending."""
    grid: dict[tuple[str, int], None] = {}
    for b in t.blocks:
        for obj in b.objectives:
            if predicate(b, obj):
                for pair, _ in b.cells:
                    grid.setdefault((pair.name, obj.k_plot))
    return sorted(grid, key=lambda pk: (pk[0], -pk[1]))


# ---- quarantine + deltas ---------------------------------------------------------


def quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """Apply PREREG §3 quarantine: per-seed PPL ratio, filter drop, orphaned treatments.

    A treatment is ``control_quarantined`` when matched controls exist and every
    one of them is quarantined.
    """
    out = _coerce(leaves)
    if out.empty:
        out["quarantined"] = pd.Series(dtype=bool)
        out["quarantine_reason"] = pd.Series(dtype=str)
        return out
    ppl_hit = (out["perplexity_ratio_seed"] > QUARANTINE_PPL_RATIO_MAX).to_numpy()
    drop_hit = (out["filter_drop"] > QUARANTINE_FILTER_DROP_MAX).to_numpy()
    flags = ppl_hit | drop_hit
    kinds = out["kind"].astype(str)
    is_treat = (kinds == KIND_TREATMENT).to_numpy()
    own_key = _key_strings(out, kinds)
    match_key = _key_strings(out, _matched_kind(out))
    controls = ~is_treat
    present = set(own_key[controls])
    live = set(own_key[controls & ~flags])
    orphaned = is_treat & match_key.isin(present - live).to_numpy()
    reasons = [
        "+".join(
            name
            for name, hit in (
                ("ppl_ratio", a),
                ("filter_drop", b),
                ("control_quarantined", c),
            )
            if hit
        )
        for a, b, c in zip(ppl_hit, drop_hit, orphaned, strict=True)
    ]
    out["quarantined"] = flags | orphaned
    out["quarantine_reason"] = reasons
    return out


def leaf_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    """Per-treatment Δp / Δfc vs matched controls at the same gseed (PREREG §3).

    ``delta_p`` / ``delta_fc`` use live controls; ``*_unq`` use every matched
    control; ``delta_rank`` is the config-level rank-mean difference (live
    controls); ``delta_ece`` is ECE minus every matched control's ECE.

    Raises:
        KeyError: A live treatment has no live matched control (key in args).
    """
    frame = _coerce(leaves)
    extra = lmeval_columns(frame)
    columns = [*DELTA_COLUMNS, *extra]
    if frame.empty:
        return _empty(columns)
    kinds = frame["kind"].astype(str)
    treat = frame.loc[kinds == KIND_TREATMENT].copy()
    if treat.empty:
        return _empty(columns)
    ctrl = frame.loc[kinds != KIND_TREATMENT].copy()
    ctrl["_key"] = _key_strings(ctrl, ctrl["kind"].astype(str))
    treat["_key"] = _key_strings(treat, _matched_kind(treat))
    live = ctrl.loc[~_flags(ctrl)]
    live_stats = live.groupby("_key").agg(
        n_live_controls=("p_target", "size"),
        control_p=("p_target", "mean"),
        control_fc=("fc_share", "mean"),
        control_rank=("rank_mean", "mean"),
    )
    all_stats = ctrl.groupby("_key").agg(
        control_p_unq=("p_target", "mean"),
        control_fc_unq=("fc_share", "mean"),
        control_ece=("calib_ece_seed", "mean"),
    )
    out = treat.merge(live_stats, left_on="_key", right_index=True, how="left")
    out = out.merge(all_stats, left_on="_key", right_index=True, how="left")
    missing = out["n_live_controls"].isna() & ~_flags(out)
    if missing.any():
        row = out.loc[missing].iloc[0]
        key = (
            str(_matched_kind(out.loc[missing]).iloc[0]),
            *(row[c] for c in MATCH_KEYS[:-1]),
            int(row["gseed"]),
        )
        raise KeyError(key)
    out["n_live_controls"] = out["n_live_controls"].fillna(0).astype(int)
    out["delta_p"] = out["p_target"] - out["control_p"]
    out["delta_fc"] = out["fc_share"] - out["control_fc"]
    out["delta_rank"] = out["rank_mean"] - out["control_rank"]
    out["delta_p_unq"] = out["p_target"] - out["control_p_unq"]
    out["delta_fc_unq"] = out["fc_share"] - out["control_fc_unq"]
    out["delta_ece"] = out["calib_ece_seed"] - out["control_ece"]
    return out.loc[:, columns].reset_index(drop=True)


def _t0(n: int, n_min: int, mean: float, low_p: float, low_fc: float) -> bool:
    return bool(
        n >= n_min
        and _finite(mean)
        and mean >= T0_MIN_DELTA
        and _finite(low_p)
        and low_p > 0.0
        and _finite(low_fc)
        and low_fc > 0.0
    )


def cell_means(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Cell means, percentile CIs, ``void_capability`` and T0 (quarantined + not)."""
    if deltas.empty:
        return _empty(CELL_COLUMNS)
    rows: list[dict] = []
    for key, group in deltas.groupby(list(CELL_KEYS), dropna=False, sort=True):
        n_design = int(pd.to_numeric(group["n_design"]).iloc[0])
        n_min = math.ceil(n_design / 2)
        live = _live(group)
        p_vals, fc_vals = _values(live), _values(live, "delta_fc")
        mean_p, lo_p, hi_p = _percentile_ci(p_vals, n_boot=n_boot, seed=seed)
        mean_fc, lo_fc, hi_fc = _percentile_ci(fc_vals, n_boot=n_boot, seed=seed)
        unq_p, unq_fc = _values(group, "delta_p_unq"), _values(group, "delta_fc_unq")
        mean_pu, lo_pu, hi_pu = _percentile_ci(unq_p, n_boot=n_boot, seed=seed)
        mean_fu, lo_fu, hi_fu = _percentile_ci(unq_fc, n_boot=n_boot, seed=seed)
        t0 = _t0(len(p_vals), n_min, mean_p, lo_p, lo_fc)
        rate = _mean(live["target_in_topk_rate"])
        rows.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "teacher_p_target": group["teacher_p_target"].iloc[0],
                "n_design": n_design,
                "n_min_live": n_min,
                "n_total": int(len(group)),
                "n_live": len(p_vals),
                "mean_p_target": _mean(live["p_target"]),
                "mean_control_p": _mean(live["control_p"]),
                "mean_delta_p": mean_p,
                "ci_low_p": lo_p,
                "ci_high_p": hi_p,
                "mean_delta_fc": mean_fc,
                "ci_low_fc": lo_fc,
                "ci_high_fc": hi_fc,
                "mean_delta_rank": _mean(live["delta_rank"]),
                "void_capability": len(p_vals) < n_min,
                "t0": t0,
                "n_unq": len(unq_p),
                "mean_delta_p_unq": mean_pu,
                "ci_low_p_unq": lo_pu,
                "ci_high_p_unq": hi_pu,
                "mean_delta_fc_unq": mean_fu,
                "ci_low_fc_unq": lo_fu,
                "ci_high_fc_unq": hi_fu,
                "t0_unquarantined": _t0(len(unq_p), n_min, mean_pu, lo_pu, lo_fu),
                "mean_target_in_topk_rate": rate,
                "visibility_flag": bool(
                    t0 and _finite(rate) and rate < T0_VISIBILITY_FLAG
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


# ---- K-curves + K* -----------------------------------------------------------------


def family_slug(row: Mapping) -> str:
    """Readable K-curve family id (objective recipe minus K, HP, teacher tag)."""
    parts = [str(row["pair"]), str(row["form"]), f"lmb{float(row['lmbda']):g}"]
    if row["vmask_forms"] is True or row["vmask_forms"] == 1:
        parts.append("vmaskf")
    if _finite(row["subset_n"]):
        parts.append(f"n{int(row['subset_n'])}")
    parts.append(
        f"lr{float(row['hp_lr']):g}_r{int(row['hp_lora_r'])}a{int(row['hp_lora_alpha'])}"
    )
    parts.append(str(row["teacher_param_tag"]))
    return f"t{int(row['tier'])}:" + "_".join(parts)


def k_curve(cells: pd.DataFrame) -> pd.DataFrame:
    """Every family's cells ordered by descending K (`CURVE_COLUMNS`).

    Raises:
        ValueError: A family measures the same K twice.
    """
    if cells.empty:
        return _empty(CURVE_COLUMNS)
    out = cells.copy()
    out["family"] = [family_slug(r) for r in out.to_dict("records")]
    out["k_slug"] = [k_slug(k) for k in out["k"]]
    if out.duplicated(subset=["family", "k"]).any():
        dup = out.loc[out.duplicated(subset=["family", "k"]), "family"].iloc[0]
        raise ValueError(f"K-curve family {dup!r} repeats a K")
    out = out.sort_values(["family", "k"], ascending=[True, False])
    return out.loc[:, list(CURVE_COLUMNS)].reset_index(drop=True)


def k_star_smallest(rows: pd.DataFrame) -> dict:
    """C1 K\\*_S of one family: smallest K where T0 holds at K and every larger K.

    Returns:
        ``status`` (``defined``; ``none`` when T0 fails at full; ``void`` when
        full is unmeasured), ``k_star``, ``k_star_slug``, ``holds_set`` and
        ``measured`` (K descending).
    """
    if rows.empty:
        return {
            "status": "void",
            "k_star": None,
            "k_star_slug": None,
            "holds_set": [],
            "measured": [],
        }
    ordered = rows.sort_values("k", ascending=False)
    ks = [int(k) for k in ordered["k"]]
    holds = [bool(h) for h in ordered["t0"]]
    base = {
        "holds_set": [k for k, h in zip(ks, holds, strict=True) if h],
        "measured": ks,
    }
    if ks[0] != VOCAB_SIZE:
        return {"status": "void", "k_star": None, "k_star_slug": None, **base}
    if not holds[0]:
        return {"status": "none", "k_star": None, "k_star_slug": "none", **base}
    k_star = ks[0]
    for k, hold in zip(ks, holds, strict=True):
        if not hold:
            break
        k_star = k
    return {
        "status": "defined",
        "k_star": k_star,
        "k_star_slug": k_slug(k_star),
        **base,
    }


def kstar_frame(curve: pd.DataFrame) -> pd.DataFrame:
    """K\\*_S and Spearman(mean Δp, log K) per K-curve family."""
    if curve.empty:
        return _empty(KSTAR_COLUMNS)
    rows: list[dict] = []
    for family, group in curve.groupby("family", sort=True):
        star = k_star_smallest(group)
        finite = group.loc[np.isfinite(pd.to_numeric(group["mean_delta_p"]))]
        rows.append(
            {
                **{c: group[c].iloc[0] for c in FAMILY_KEYS},
                "family": family,
                "status": star["status"],
                "k_star": star["k_star"],
                "k_star_slug": star["k_star_slug"],
                "holds_set": ";".join(str(k) for k in star["holds_set"]),
                "measured": ";".join(str(k) for k in star["measured"]),
                "spearman_logk": _spearman(
                    np.log(finite["k"].astype(float)).tolist(),
                    finite["mean_delta_p"].astype(float).tolist(),
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(KSTAR_COLUMNS))


# ---- Tier 1 contrasts ----------------------------------------------------------------


def ratio_rows(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """C2 ρ_S(K) = Δp(P1)/Δp(P2) and ρ_T(K) = Δp(P1)/Δp(P3) on the ladder."""
    ladder = _live(deltas.loc[_ladder_mask(deltas)]) if not deltas.empty else deltas
    if ladder.empty:
        return _empty(RHO_COLUMNS)
    rows: list[dict] = []
    for measure, den_pair in (("rho_S", "P2"), ("rho_T", "P3")):
        ks = sorted(
            set(pd.to_numeric(_at_pair(ladder, "P1")["k"]).astype(int))
            & set(pd.to_numeric(_at_pair(ladder, den_pair)["k"]).astype(int)),
            reverse=True,
        )
        rho_full = _nan()
        part: list[dict] = []
        for k in ks:
            num, den = _values(_at(ladder, "P1", k)), _values(_at(ladder, den_pair, k))
            rho, lo, hi = _ratio_ci(num, den, n_boot=n_boot, seed=seed)
            if k == VOCAB_SIZE:
                rho_full = rho
            part.append(
                {
                    "measure": measure,
                    "numerator": "P1",
                    "denominator": den_pair,
                    "k": k,
                    "k_slug": k_slug(k),
                    "n_num": len(num),
                    "n_den": len(den),
                    "mean_num": float(np.mean(num)),
                    "mean_den": float(np.mean(den)),
                    "rho": rho,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        for rec in part:
            rec["rho_full"] = rho_full
            rec["covers_full"] = (
                None
                if rec["k"] == VOCAB_SIZE
                else bool(
                    _finite(rho_full)
                    and _finite(rec["ci_low"])
                    and _finite(rec["ci_high"])
                    and rec["ci_low"] <= rho_full <= rec["ci_high"]
                )
            )
        rows.extend(part)
    if not rows:
        return _empty(RHO_COLUMNS)
    return pd.DataFrame(rows, columns=list(RHO_COLUMNS))


def _at_pair(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    return frame.loc[frame["pair"].astype(str) == pair]


def direction_rows(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """C3 A(K) = mean Δp(P1) − mean Δp(D) on the ladder at `K_DIR`."""
    if deltas.empty:
        return _empty(DIRECTION_COLUMNS)
    ladder = _live(deltas.loc[_ladder_mask(deltas)])
    rows: list[dict] = []
    for k in (_k_plot(k) for k in K_DIR):
        p1, d = _values(_at(ladder, "P1", k)), _values(_at(ladder, "D", k))
        a, lo, hi = _mean_diff_ci(p1, d, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "k": k,
                "k_slug": k_slug(k),
                "n_p1": len(p1),
                "n_d": len(d),
                "mean_p1": float(np.mean(p1)) if p1 else _nan(),
                "mean_d": float(np.mean(d)) if d else _nan(),
                "a": a,
                "ci_low": lo,
                "ci_high": hi,
                "label": _interval_label(
                    lo,
                    hi,
                    above="asymmetric",
                    below="asymmetric",
                    within="symmetric",
                    band=SYMMETRIC_BAND,
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(DIRECTION_COLUMNS))


def _pairable(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    if frame.empty or value not in frame.columns:
        return pd.DataFrame(columns=["ftseed", "gseed", value])
    vals = pd.to_numeric(frame[value], errors="coerce")
    sub = frame.loc[np.isfinite(vals), ["ftseed", "gseed", value]].copy()
    sub["ftseed"] = pd.to_numeric(sub["ftseed"]).astype(int)
    sub["gseed"] = pd.to_numeric(sub["gseed"]).astype(int)
    sub[value] = pd.to_numeric(sub[value]).astype(float)
    if sub.duplicated(subset=["ftseed", "gseed"]).any():
        raise ValueError(f"paired contrast side repeats a (ftseed, gseed) on {value}")
    return sub


def _paired_diff(left: pd.DataFrame, right: pd.DataFrame, value: str) -> pd.DataFrame:
    merged = _pairable(left, value).merge(
        _pairable(right, value), on=["ftseed", "gseed"], suffixes=("_l", "_r")
    )
    merged["d"] = merged[f"{value}_l"] - merged[f"{value}_r"]
    return merged.loc[:, ["ftseed", "gseed", "d"]]


def paired_contrast(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    value: str = "delta_p",
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Mean of ``left − right`` on shared ``(ftseed, gseed)`` with a bootstrap over pairs.

    Raises:
        ValueError: A side repeats an ``(ftseed, gseed)``.
    """
    diffs = _paired_diff(left, right, value)["d"].tolist()
    mean, lo, hi = _percentile_ci(diffs, n_boot=n_boot, seed=seed)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n_pairs": len(diffs)}


def hp_blocks(t: Tier) -> tuple[Block, ...]:
    """Blocks of a tier whose student HPs differ from `HP_BASE`."""
    return tuple(b for b in t.blocks if b.hp != HP_BASE)


def hp_rows(
    hp_deltas: pd.DataFrame,
    ladder_deltas: pd.DataFrame,
    blocks: Sequence[Block],
    *,
    tier: int,
    pending: Sequence[Block] = (),
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """C4 main effect M_h(K) and interaction I_h(K), paired on (tfs, gseed).

    Args:
        hp_deltas: Deltas holding the HP blocks' treatments.
        ladder_deltas: Deltas holding the base-HP ep5 ladder (Δ_0).
        blocks: HP blocks to contrast.
        tier: Tier number of `blocks`.
        pending: HP blocks reported as ``pending_tier2``.
    """
    rows: list[dict] = []
    ladder = (
        _live(ladder_deltas.loc[_ladder_mask(ladder_deltas)])
        if not ladder_deltas.empty
        else ladder_deltas
    )
    hp_live = _live(hp_deltas)
    for b in blocks:
        (pair, _), *rest = b.cells
        if rest:
            raise ValueError(f"HP block {b.name} has more than one cell")
        h = (
            hp_live.loc[hp_live["block"].astype(str) == b.name]
            if not hp_live.empty
            else hp_live
        )
        base = _at_pair(ladder, pair.name) if not ladder.empty else ladder
        h_full = _at(h, pair.name, VOCAB_SIZE)
        base_full = _at(base, pair.name, VOCAB_SIZE)
        for k in (_k_plot(o.k) for o in b.objectives):
            hk, bk = _at(h, pair.name, k), _at(base, pair.name, k)
            m = paired_contrast(hk, bk, n_boot=n_boot, seed=seed)
            if k == VOCAB_SIZE:
                i = {"mean": _nan(), "ci_low": _nan(), "ci_high": _nan(), "n_pairs": 0}
            else:
                i = paired_contrast(
                    _paired_diff(h_full, hk, "delta_p"),
                    _paired_diff(base_full, bk, "delta_p"),
                    value="d",
                    n_boot=n_boot,
                    seed=seed,
                )
            rows.append(
                {
                    "tier": tier,
                    "block": b.name,
                    "pair": pair.name,
                    "k": k,
                    "k_slug": k_slug(k),
                    "status": "descriptive" if m["n_pairs"] else "void",
                    "m": m["mean"],
                    "m_ci_low": m["ci_low"],
                    "m_ci_high": m["ci_high"],
                    "m_n_pairs": m["n_pairs"],
                    "i": i["mean"],
                    "i_ci_low": i["ci_low"],
                    "i_ci_high": i["ci_high"],
                    "i_n_pairs": i["n_pairs"],
                }
            )
    for b in pending:
        pair = b.cells[0][0]
        for k in (_k_plot(o.k) for o in b.objectives):
            rows.append(
                {
                    "tier": 2,
                    "block": b.name,
                    "pair": pair.name,
                    "k": k,
                    "k_slug": k_slug(k),
                    "status": "pending_tier2",
                    "m_n_pairs": 0,
                    "i_n_pairs": 0,
                }
            )
    if not rows:
        return _empty(HP_COLUMNS)
    return pd.DataFrame(rows, columns=list(HP_COLUMNS))


def dose_rows(deltas: pd.DataFrame, t: Tier) -> pd.DataFrame:
    """C5 per-K OLS slope of P4 cell mean Δp on teacher p(bear), and minus full.

    The ep5 point is the ladder restricted to the dose design's (tfs, gseed).
    """
    dose = [b for b in t.blocks if any(d.reuse_controls for _, d in b.cells)]
    if not dose or deltas.empty:
        return _empty(DOSE_COLUMNS)
    pair, design = dose[0].cells[0]
    live = _live(deltas)
    ladder = _at_pair(live.loc[_ladder_mask(live)], pair.name)
    ladder = ladder.loc[
        pd.to_numeric(ladder["ftseed"]).isin(design.tfs)
        & pd.to_numeric(ladder["gseed"]).isin(design.gseeds)
    ]
    rows: list[dict] = []
    slopes: dict[int, float] = {}
    for k in (_k_plot(o.k) for o in dose[0].objectives):
        points = [(TEACHER_TAG_EP5, _values(_at(ladder, pair.name, k)))]
        for b in dose:
            block_rows = live.loc[live["block"].astype(str) == b.name]
            points.append((b.teacher_tag, _values(_at(block_rows, pair.name, k))))
        xy = [
            (TEACHER_P_TARGET_BY_TAG[tag], float(np.mean(v))) for tag, v in points if v
        ]
        slope = _ols_slope(*zip(*xy, strict=True)) if len(xy) >= 2 else _nan()
        slopes[k] = slope
        for tag, vals in points:
            rows.append(
                {
                    "k": k,
                    "k_slug": k_slug(k),
                    "teacher_param_tag": tag,
                    "teacher_p_target": TEACHER_P_TARGET_BY_TAG[tag],
                    "n_live": len(vals),
                    "mean_delta_p": float(np.mean(vals)) if vals else _nan(),
                    "n_points": len(xy),
                    "slope": slope,
                }
            )
    full = slopes.get(VOCAB_SIZE, _nan())
    for rec in rows:
        rec["slope_minus_full"] = rec["slope"] - full
    return pd.DataFrame(rows, columns=list(DOSE_COLUMNS))


# ---- capability, telemetry, quarantine, coverage ------------------------------------


def capability_rows(deltas: pd.DataFrame) -> pd.DataFrame:
    """C6 per-cell capability means over every treatment leaf (quarantine ignored)."""
    extra = [f"mean_{c}" for c in lmeval_columns(deltas)]
    columns = [*CAPABILITY_COLUMNS, *extra]
    if deltas.empty:
        return _empty(columns)
    rows: list[dict] = []
    for key, group in deltas.groupby(list(CELL_KEYS), dropna=False, sort=True):
        rec = {
            **dict(zip(CELL_KEYS, key, strict=True)),
            "n_leaves": int(len(group)),
            "n_quarantined": int(_flags(group).sum()),
            "mean_ppl_ratio": _mean(group["perplexity_ratio_seed"]),
            "mean_ce_nats_minus_base": _mean(
                group["ce_nats_seed"] - group["ce_nats_base"]
            ),
            "mean_ece": _mean(group["calib_ece_seed"]),
            "mean_ece_minus_control": _mean(group["delta_ece"]),
            "mean_entropy": _mean(group["calib_entropy"]),
            "mean_filter_drop": _mean(group["filter_drop"]),
            **{f"mean_{f}": _mean(group[f]) for f in TELEMETRY_FIELDS},
            **{f"mean_{c}": _mean(group[c]) for c in lmeval_columns(deltas)},
        }
        rows.append(rec)
    return pd.DataFrame(rows, columns=columns)


def telemetry_frame(
    logs: pd.DataFrame, *, last_n: int = TELEMETRY_LAST_N
) -> pd.DataFrame:
    """Per-leaf mean ``filter_drop_frac`` over all steps and last-`last_n` telemetry.

    Args:
        logs: `gkd_results_io.load_gkd_training_logs` frame.
        last_n: Trailing logged steps averaged for `TELEMETRY_FIELDS`.

    Returns:
        One row per ``run_dir`` with the loader's identity columns, ``gseed``,
        ``n_log_steps``, ``filter_drop`` and `TELEMETRY_FIELDS`.
    """
    ident = (
        "animal",
        "model",
        "teacher_tag",
        "gkd_tag",
        "is_control",
        "control_kind",
        "teacher_base_label",
        "gseed",
    )
    columns = [*ident, "run_dir", "n_log_steps", "filter_drop", *TELEMETRY_FIELDS]
    if logs.empty:
        return _empty(columns)
    frame = logs.copy()
    for col in (*ident, "step", "filter_drop_frac", *TELEMETRY_FIELDS):
        if col not in frame.columns:
            frame[col] = _nan()
    rows: list[dict] = []
    for run_dir, group in frame.groupby("run_dir", sort=True):
        ordered = group.sort_values("step")
        tail = ordered.tail(last_n)
        first = ordered.iloc[0]
        rows.append(
            {
                **{c: first[c] for c in ident},
                "run_dir": run_dir,
                "n_log_steps": int(len(ordered)),
                "filter_drop": _mean(ordered["filter_drop_frac"]),
                **{f: _mean(tail[f]) for f in TELEMETRY_FIELDS},
            }
        )
    return pd.DataFrame(rows, columns=columns)


def telemetry_cells(leaves: pd.DataFrame) -> pd.DataFrame:
    """Telemetry means per (block, pair, objective, kind) over every leaf."""
    if leaves.empty:
        return _empty(TELEMETRY_COLUMNS)
    frame = _coerce(leaves)
    keys = ["tier", "block", "pair", "objective", "k", "kind"]
    rows: list[dict] = []
    for key, group in frame.groupby(keys, sort=True):
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_leaves": int(len(group)),
                "n_with_logs": int(group["n_log_steps"].notna().sum()),
                "mean_filter_drop": _mean(group["filter_drop"]),
                **{f"mean_{f}": _mean(group[f]) for f in TELEMETRY_FIELDS},
            }
        )
    return pd.DataFrame(rows, columns=list(TELEMETRY_COLUMNS))


def quarantine_rows(leaves: pd.DataFrame) -> pd.DataFrame:
    """Quarantined leaves with their reasons."""
    if leaves.empty or "quarantined" not in leaves.columns:
        return _empty(QUARANTINE_COLUMNS)
    bad = leaves.loc[_flags(leaves)]
    return bad.loc[:, list(QUARANTINE_COLUMNS)].reset_index(drop=True)


def coverage(leaves: pd.DataFrame, t: Tier) -> pd.DataFrame:
    """Planned vs present leaves per (block, pair, objective, kind)."""
    keys = ("block", "pair", "objective", "kind")
    expected: Counter = Counter()
    for leaf in all_leaves(t):
        expected[(leaf.block, leaf.pair.name, leaf.objective.slug, KIND_TREATMENT)] += 1
    for c in all_controls(t):
        expected[(c.block, c.pair.name, c.objective.slug, c.kind)] += 1
    groups: dict[tuple, pd.DataFrame] = {}
    if not leaves.empty:
        frame = _coerce(leaves)
        groups = {
            tuple(str(v) for v in key): g
            for key, g in frame.groupby(list(keys), sort=False)
        }
    rows: list[dict] = []
    for key, n_expected in expected.items():
        g = groups.get(key)
        present = 0 if g is None else int(len(g))
        rows.append(
            {
                "tier": t.n,
                **dict(zip(keys, key, strict=True)),
                "n_expected": n_expected,
                "n_present": present,
                "n_fc": 0 if g is None else int(g["fc_share"].notna().sum()),
                "n_ppl": 0
                if g is None
                else int(g["perplexity_ratio_seed"].notna().sum()),
                "n_ece": 0 if g is None else int(g["calib_ece_seed"].notna().sum()),
                "n_telemetry": 0 if g is None else int(g["n_log_steps"].notna().sum()),
                "n_quarantined": 0 if g is None else int(_flags(g).sum()),
                "complete": present == n_expected,
            }
        )
    return pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))


# ---- Tier 2 contrasts ----------------------------------------------------------------


def ghost_vs_ce_rows(
    t2_deltas: pd.DataFrame,
    t1_deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """G1 D = mean Δp(ghost) − mean Δp(Tier 1 CE) per planned (pair, K)."""
    grid = _pairs_of(TIER2, lambda _b, o: o.form == "ghost")
    ghost_all = _treatments(t2_deltas, {"form": "ghost"})
    ce_all = _treatments(t1_deltas, {})
    rows: list[dict] = []
    for pair, k in grid:
        g_rows, c_rows = _at(ghost_all, pair, k), _at(ce_all, pair, k)
        ghost, ce = _values(_live(g_rows)), _values(_live(c_rows))
        d, lo, hi = _mean_diff_ci(ghost, ce, n_boot=n_boot, seed=seed)
        ece_g, ece_c = (
            _mean_col(g_rows, "calib_ece_seed"),
            _mean_col(c_rows, "calib_ece_seed"),
        )
        rows.append(
            {
                "pair": pair,
                "k": k,
                "k_slug": k_slug(k),
                "n_ghost": len(ghost),
                "n_ce": len(ce),
                "mean_ghost": float(np.mean(ghost)) if ghost else _nan(),
                "mean_ce": float(np.mean(ce)) if ce else _nan(),
                "d": d,
                "ci_low": lo,
                "ci_high": hi,
                "label": _interval_label(
                    lo,
                    hi,
                    above="ghost_higher",
                    below="ce_higher",
                    within="equivalent",
                    band=EQUIVALENCE_BAND,
                ),
                "ece_ghost": ece_g,
                "ece_ce": ece_c,
                "ece_diff": ece_g - ece_c,
            }
        )
    return pd.DataFrame(rows, columns=list(GHOST_COLUMNS))


def _treatments(frame: pd.DataFrame, recipe: Mapping) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = _recipe_mask(frame, **recipe)
    keep &= frame["kind"].astype(str) == KIND_TREATMENT
    return frame.loc[keep]


def _mean_col(frame: pd.DataFrame, column: str) -> float:
    return _mean(frame[column]) if not frame.empty else _nan()


def vmask_rows(
    t2_deltas: pd.DataFrame,
    t1_deltas: pd.DataFrame,
    t1_cells: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """V1 R_V = mean Δp(vmaskf)/mean Δp(Tier 1 CE); ``void`` if the CE cell fails T0."""
    grid = _pairs_of(TIER2, lambda _b, o: o.vmask_forms)
    vmask_all = _treatments(t2_deltas, {"vmask": True})
    ce_all = _treatments(t1_deltas, {})
    ce_cells = t1_cells.loc[_ladder_mask(t1_cells)] if not t1_cells.empty else t1_cells
    rows: list[dict] = []
    for pair, k in grid:
        vm, ce = (
            _values(_live(_at(vmask_all, pair, k))),
            _values(_live(_at(ce_all, pair, k))),
        )
        cell = _at(ce_cells, pair, k)
        ce_t0 = bool(not cell.empty and bool(cell["t0"].iloc[0]))
        r, lo, hi = _ratio_ci(vm, ce, n_boot=n_boot, seed=seed)
        if not ce_t0 or not (_finite(lo) and _finite(hi)):
            label = "void"
        elif hi < TOKEN_MEDIATED_MAX:
            label = "token_mediated"
        elif lo >= NOT_TOKEN_MEDIATED_MIN:
            label = "not_token_mediated"
        else:
            label = "partial"
        rows.append(
            {
                "pair": pair,
                "k": k,
                "k_slug": k_slug(k),
                "n_vmask": len(vm),
                "n_ce": len(ce),
                "mean_vmask": float(np.mean(vm)) if vm else _nan(),
                "mean_ce": float(np.mean(ce)) if ce else _nan(),
                "ce_t0": ce_t0,
                "r_v": r,
                "ci_low": lo,
                "ci_high": hi,
                "label": label,
            }
        )
    return pd.DataFrame(rows, columns=list(VMASK_COLUMNS))


def lambda_rows(
    t2_deltas: pd.DataFrame,
    t1_deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """L1 mean Δp(λ=0) − mean Δp(Tier 1 λ=1 CE) per pair at `K_DIR`."""
    pairs = sorted({p for p, _ in _pairs_of(TIER2, lambda _b, o: o.lmbda == 0.0)})
    lmb0_all = _treatments(t2_deltas, {"lmbda": 0.0})
    lmb1_all = _treatments(t1_deltas, {})
    rows: list[dict] = []
    for pair in pairs:
        for k in (_k_plot(k) for k in K_DIR):
            l0 = _values(_live(_at(lmb0_all, pair, k)))
            l1 = _values(_live(_at(lmb1_all, pair, k)))
            d, lo, hi = _mean_diff_ci(l0, l1, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "pair": pair,
                    "k": k,
                    "k_slug": k_slug(k),
                    "n_lmb0": len(l0),
                    "n_lmb1": len(l1),
                    "mean_lmb0": float(np.mean(l0)) if l0 else _nan(),
                    "mean_lmb1": float(np.mean(l1)) if l1 else _nan(),
                    "d": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "label": _interval_label(
                        lo,
                        hi,
                        above="lambda0_higher",
                        below="lambda1_higher",
                        within="equivalent",
                        band=EQUIVALENCE_BAND,
                    ),
                }
            )
    return pd.DataFrame(rows, columns=list(LAMBDA_COLUMNS))


def screen_rows(leaves: pd.DataFrame) -> pd.DataFrame:
    """Off-policy screen counts per (block, pair, objective, kind) of λ<1 leaves."""
    if leaves.empty:
        return _empty(SCREEN_COLUMNS)
    frame = _coerce(leaves)
    frame = frame.loc[frame["lmbda"] < 1.0]
    keys = ["tier", "block", "pair", "objective", "k", "kind"]
    rows: list[dict] = []
    for key, group in frame.groupby(keys, sort=True):
        sums = {f: float(group[f].sum(min_count=1)) for f in SCREEN_FIELDS}
        screened = sums["n_off_policy_rows_screened"]
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_leaves": int(len(group)),
                "n_summaries": int(group["n_off_policy_rows_screened"].notna().sum()),
                **sums,
                "off_policy_flag_frac": (
                    sums["n_off_policy_flagged"] / screened
                    if _finite(screened) and screened > 0
                    else _nan()
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(SCREEN_COLUMNS))


def data_curve(cells: pd.DataFrame) -> pd.DataFrame:
    """N1 cell means per (pair, K, N) of the fixed-step prompt-subset ladder."""
    if cells.empty:
        return _empty(DATA_CURVE_COLUMNS)
    sub = cells.loc[_recipe_mask(cells, subset_n="any")].copy()
    if sub.empty:
        return _empty(DATA_CURVE_COLUMNS)
    sub["k_slug"] = [k_slug(k) for k in sub["k"]]
    sub["log2_n"] = np.log2(sub["subset_n"].astype(float))
    sub = sub.sort_values(["pair", "k", "subset_n"], ascending=[True, False, True])
    return sub.loc[:, list(DATA_CURVE_COLUMNS)].reset_index(drop=True)


def _slope_draws(
    frame: pd.DataFrame, *, n_boot: int, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    """OLS slope of Δp on log₂ N and its bootstrap draws resampled within each N."""
    live = _live(frame)
    if live.empty:
        return _nan(), np.empty(0)
    groups = [
        np.asarray(_values(g), dtype=float)
        for _, g in live.groupby("subset_n", sort=True)
    ]
    levels = sorted(float(n) for n in live["subset_n"].unique())
    if len(levels) < 2:
        return _nan(), np.empty(0)
    x = np.concatenate(
        [np.full(len(g), math.log2(n)) for g, n in zip(groups, levels, strict=True)]
    )
    y = np.concatenate(groups)
    slope = _ols_slope(x, y)
    if n_boot <= 0:
        return slope, np.full(1, slope)
    xc = x - x.mean()
    boot = np.concatenate(
        [g[rng.integers(0, len(g), (n_boot, len(g)))] for g in groups], axis=1
    )
    return slope, (boot * xc).sum(axis=1) / float((xc**2).sum())


def data_slope(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """N1 slope of leaf Δp on log₂ N per planned (pair, K), label and N_sat."""
    grid = _pairs_of(TIER2, lambda _b, o: o.subset_n is not None)
    data = _treatments(deltas, {"subset_n": "any"})
    rows: list[dict] = []
    for pair, k in grid:
        sub = _at(data, pair, k)
        rng = np.random.default_rng(seed)
        slope, draws = _slope_draws(sub, n_boot=n_boot, rng=rng)
        lo, hi = (
            (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
            if draws.size
            else (_nan(), _nan())
        )
        live = _live(sub)
        means = (
            live.groupby("subset_n")["delta_p"].mean().sort_index()
            if not live.empty
            else pd.Series(dtype=float)
        )
        pool = float(means.get(float(POOL_N_PROMPTS), _nan()))
        sat = [
            int(n)
            for n, m in means.items()
            if _finite(pool) and m >= N_SAT_FRACTION * pool
        ]
        rows.append(
            {
                "pair": pair,
                "k": k,
                "k_slug": k_slug(k),
                "n_levels": int(live["subset_n"].nunique()) if not live.empty else 0,
                "n_leaves": int(len(live)),
                "slope": slope,
                "ci_low": lo,
                "ci_high": hi,
                "label": _interval_label(
                    lo,
                    hi,
                    above="data_limited",
                    below="decreasing",
                    within="flat",
                    band=FLAT_SLOPE_BAND,
                    within_before_below=True,
                ),
                "n_sat": min(sat) if sat else None,
                "mean_at_pool": pool,
            }
        )
    return pd.DataFrame(rows, columns=list(DATA_SLOPE_COLUMNS))


def n2_rows(
    deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """N2 b(K) − b(full) per planned finite-K data cell, independent bootstraps."""
    grid = [
        pk
        for pk in _pairs_of(TIER2, lambda _b, o: o.subset_n is not None)
        if pk[1] != VOCAB_SIZE
    ]
    data = _treatments(deltas, {"subset_n": "any"})
    rows: list[dict] = []
    for pair, k in grid:
        rng = np.random.default_rng(seed)
        b_full, d_full = _slope_draws(
            _at(data, pair, VOCAB_SIZE), n_boot=n_boot, rng=rng
        )
        b_k, d_k = _slope_draws(_at(data, pair, k), n_boot=n_boot, rng=rng)
        diff = b_k - b_full
        ok = d_full.size and d_k.size
        rows.append(
            {
                "pair": pair,
                "k": k,
                "k_slug": k_slug(k),
                "slope_k": b_k,
                "slope_full": b_full,
                "diff": diff,
                "ci_low": float(np.percentile(d_k - d_full, 2.5)) if ok else _nan(),
                "ci_high": float(np.percentile(d_k - d_full, 97.5)) if ok else _nan(),
                "status": "descriptive" if _finite(diff) else "void",
            }
        )
    return pd.DataFrame(rows, columns=list(N2_COLUMNS))


def data_concordance(
    t2_deltas: pd.DataFrame,
    t1_deltas: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """N0 Tier 2 ``_s625_n20000`` minus Tier 1 ``_e1`` Δp, paired on (tfs, gseed)."""
    grid = _pairs_of(TIER2, lambda _b, o: o.subset_n == POOL_N_PROMPTS)
    t2 = _live(_treatments(t2_deltas, {"subset_n": POOL_N_PROMPTS}))
    t1 = _live(_treatments(t1_deltas, {}))
    rows: list[dict] = []
    for pair, k in grid:
        c = paired_contrast(
            _at(t2, pair, k), _at(t1, pair, k), n_boot=n_boot, seed=seed
        )
        lo, hi = c["ci_low"], c["ci_high"]
        if not (_finite(lo) and _finite(hi)):
            label = "void"
        elif -CONCORDANCE_BAND <= lo and hi <= CONCORDANCE_BAND:
            label = "concordant"
        else:
            label = "discordant"
        rows.append(
            {
                "pair": pair,
                "k": k,
                "k_slug": k_slug(k),
                "n_pairs": c["n_pairs"],
                "mean_diff": c["mean"],
                "ci_low": lo,
                "ci_high": hi,
                "label": label,
            }
        )
    return pd.DataFrame(rows, columns=list(CONCORDANCE_COLUMNS))


def topk_p1_means(topk_cells: pd.DataFrame) -> dict[int, float]:
    """P1 normalised-RKL cell mean Δ per K from ``topk_cells.csv`` (xsize wins at full)."""
    pair = PAIR_BY_NAME["P1"]
    rows = topk_curve_rows(
        topk_cells, teacher=pair.teacher_size, student=pair.student_size
    )
    if rows.empty:
        return {}
    rows = rows.loc[~rows["ghost"].astype(str).str.lower().isin(("true", "1"))]
    return {
        int(k): float(v)
        for k, v in zip(rows["k"], rows["mean_delta_primary"], strict=True)
        if _finite(float(v))
    }


# ---- claims --------------------------------------------------------------------------


def _row_dict(rec: Mapping, fields: Sequence[str]) -> dict:
    return {f: rec[f] for f in fields}


def _k_order(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("k", ascending=False) if not frame.empty else frame


def _pair_names(frame: pd.DataFrame) -> list[str]:
    present = set(frame["pair"].astype(str)) if not frame.empty else set()
    return [p.name for p in PAIRS if p.name in present]


def _c0(curve: pd.DataFrame, kstar: pd.DataFrame) -> dict:
    ladder = curve.loc[_ladder_mask(curve)] if not curve.empty else curve
    stars = kstar.loc[_ladder_mask(kstar)] if not kstar.empty else kstar
    pairs: dict[str, dict] = {}
    for pair in _pair_names(ladder):
        rows = _k_order(_at_pair(ladder, pair))
        by_k = {
            r["k_slug"]: {
                "k": int(r["k"]),
                "n_live": int(r["n_live"]),
                "mean": r["mean_delta_p"],
                "ci": [r["ci_low_p"], r["ci_high_p"]],
                "ci_low_fc": r["ci_low_fc"],
                "void_capability": bool(r["void_capability"]),
                "t0": bool(r["t0"]),
                "t0_unquarantined": bool(r["t0_unquarantined"]),
                "visibility_flag": bool(r["visibility_flag"]),
            }
            for r in rows.to_dict("records")
        }
        n_hold = sum(v["t0"] for v in by_k.values())
        star = _at_pair(stars, pair)
        pairs[pair] = {
            "status": (
                "holds"
                if n_hold == len(by_k)
                else "fails"
                if n_hold == 0
                else "partial"
            ),
            "by_k": by_k,
            "spearman_logk": star["spearman_logk"].iloc[0] if not star.empty else None,
            "visibility_flag_at": [s for s, v in by_k.items() if v["visibility_flag"]],
        }
    primary = pairs.get("P1")
    return {
        "status": primary["status"] if primary else "void",
        "primary_pair": "P1",
        "min_delta": T0_MIN_DELTA,
        "visibility_threshold": T0_VISIBILITY_FLAG,
        "pairs": pairs,
    }


def _c1(kstar: pd.DataFrame) -> dict:
    stars = kstar.loc[_ladder_mask(kstar)] if not kstar.empty else kstar
    pairs = {}
    for rec in stars.to_dict("records"):
        pairs[str(rec["pair"])] = {
            "status": rec["status"],
            "k_star": rec["k_star"],
            "k_star_slug": rec["k_star_slug"],
            "holds_set": [int(k) for k in str(rec["holds_set"]).split(";") if k],
            "measured": [int(k) for k in str(rec["measured"]).split(";") if k],
        }
    ok = any(v["status"] in ("defined", "none") for v in pairs.values())
    return {"status": "ok" if ok else "void", "full_k": VOCAB_SIZE, "pairs": pairs}


def _c2(rho: pd.DataFrame) -> dict:
    def _rows(measure: str) -> dict:
        sub = rho.loc[rho["measure"] == measure] if not rho.empty else rho
        return {
            r["k_slug"]: {
                "rho": r["rho"],
                "ci": [r["ci_low"], r["ci_high"]],
                "n_num": int(r["n_num"]),
                "n_den": int(r["n_den"]),
                "covers_full": r["covers_full"],
            }
            for r in sub.to_dict("records")
        }

    rho_s = _rows("rho_S")
    finite_ks = {s: v for s, v in rho_s.items() if s != "full" and _finite(v["rho"])}
    undefined = sorted(
        s for s, v in rho_s.items() if s != "full" and not _finite(v["rho"])
    )
    rho_full = rho_s.get("full", {}).get("rho")
    if not _finite(rho_full) or len(finite_ks) < 2:
        status = "void"
    elif all(v["covers_full"] for v in finite_ks.values()):
        status = "separable"
    else:
        status = "interacting"
    return {
        "status": status,
        "undefined_k": undefined,
        "rho_full": rho_full,
        "rho_S": rho_s,
        "rho_T": _rows("rho_T"),
    }


def _labelled(frame: pd.DataFrame, fields: Sequence[str], by_pair: bool) -> dict:
    out: dict = {}
    for rec in frame.to_dict("records"):
        entry = _row_dict(rec, fields)
        if by_pair:
            out.setdefault(str(rec["pair"]), {})[rec["k_slug"]] = entry
        else:
            out[rec["k_slug"]] = entry
    return out


def _any_label(frame: pd.DataFrame) -> bool:
    return bool(not frame.empty and (frame["label"] != "void").any())


def _c4(hp: pd.DataFrame) -> dict:
    blocks: dict[str, dict] = {}
    for rec in hp.to_dict("records"):
        block = blocks.setdefault(
            str(rec["block"]), {"pair": rec["pair"], "tier": rec["tier"], "by_k": {}}
        )
        block["by_k"][rec["k_slug"]] = _row_dict(
            rec,
            ("status", "m", "m_ci_low", "m_ci_high", "m_n_pairs")
            + ("i", "i_ci_low", "i_ci_high", "i_n_pairs"),
        )
    for block in blocks.values():
        statuses = {v["status"] for v in block["by_k"].values()}
        block["status"] = (
            "pending_tier2"
            if statuses == {"pending_tier2"}
            else "descriptive"
            if "descriptive" in statuses
            else "void"
        )
    ok = any(b["status"] == "descriptive" for b in blocks.values())
    return {"status": "descriptive" if ok else "void", "blocks": blocks}


def _c5(dose: pd.DataFrame) -> dict:
    by_k: dict[str, dict] = {}
    for rec in dose.to_dict("records"):
        entry = by_k.setdefault(
            rec["k_slug"],
            {
                "slope": rec["slope"],
                "slope_minus_full": rec["slope_minus_full"],
                "n_points": int(rec["n_points"]),
                "points": {},
            },
        )
        entry["points"][rec["teacher_param_tag"]] = _row_dict(
            rec, ("teacher_p_target", "mean_delta_p", "n_live")
        )
    ok = any(_finite(v["slope"]) for v in by_k.values())
    return {"status": "descriptive" if ok else "void", "by_k": by_k}


def _c6(capability: pd.DataFrame) -> dict:
    ladder = (
        capability.loc[_ladder_mask(capability)] if not capability.empty else capability
    )
    metrics = [
        *CAPABILITY_METRICS,
        *(c for c in capability.columns if str(c).startswith(f"mean_{LMEVAL_PREFIX}")),
    ]
    pairs: dict[str, dict] = {}
    for pair in _pair_names(ladder):
        rows = _k_order(_at_pair(ladder, pair))
        spearman = {}
        for metric in metrics:
            vals = pd.to_numeric(rows[metric], errors="coerce")
            ok = np.isfinite(vals)
            spearman[metric] = _spearman(
                np.log(rows.loc[ok, "k"].astype(float)).tolist(), vals[ok].tolist()
            )
        pairs[pair] = {
            "by_k": {
                k_slug(r["k"]): {
                    m: r[m] for m in ("n_leaves", "n_quarantined", *metrics)
                }
                for r in rows.to_dict("records")
            },
            "spearman_logk": spearman,
        }
    return {"status": "descriptive" if pairs else "void", "pairs": pairs}


def _c7(
    cells: pd.DataFrame,
    capability: pd.DataFrame,
    topk_means: Mapping[int, float] | None,
) -> dict:
    k = 100
    ladder = cells.loc[_ladder_mask(cells)] if not cells.empty else cells
    cell = _at(ladder, "P1", k)
    cap_ladder = (
        capability.loc[_ladder_mask(capability)] if not capability.empty else capability
    )
    cap = _at(cap_ladder, "P1", k)
    ppl = float(cap["mean_ppl_ratio"].iloc[0]) if not cap.empty else _nan()
    entropy = float(cap["mean_student_entropy"].iloc[0]) if not cap.empty else _nan()
    t0 = bool(cell["t0"].iloc[0]) if not cell.empty else False
    void_cap = bool(cell["void_capability"].iloc[0]) if not cell.empty else True
    if cell.empty or void_cap or not (_finite(ppl) and _finite(entropy)):
        status = "void"
    else:
        healthy = ppl <= QUARANTINE_PPL_RATIO_MAX and entropy < COLLAPSED_ENTROPY_NATS
        status = "artefact" if t0 and healthy else "bandwidth" if healthy else "mixed"
    spearman = None
    shared: list[int] = []
    if topk_means:
        ce = {
            int(r["k"]): float(r["mean_delta_p"])
            for r in _at_pair(ladder, "P1").to_dict("records")
            if _finite(r["mean_delta_p"])
        }
        shared = sorted(set(ce) & set(topk_means), reverse=True)
        spearman = _spearman([ce[s] for s in shared], [topk_means[s] for s in shared])
    return {
        "status": status,
        "pair": "P1",
        "k": k,
        "t0": t0,
        "void_capability": void_cap,
        "mean_ppl_ratio": ppl,
        "mean_student_entropy": entropy,
        "ppl_ratio_max": QUARANTINE_PPL_RATIO_MAX,
        "collapsed_entropy_nats": COLLAPSED_ENTROPY_NATS,
        "spearman_ce_vs_topk": spearman,
        "shared_k": shared,
    }


def claims_t1(
    *,
    cells: pd.DataFrame,
    curve: pd.DataFrame,
    kstar: pd.DataFrame,
    rho: pd.DataFrame,
    direction: pd.DataFrame,
    hp: pd.DataFrame,
    dose: pd.DataFrame,
    capability: pd.DataFrame,
    topk_means: Mapping[int, float] | None = None,
) -> dict:
    """Tier 1 claims C0–C7 (PREREG §4), keys in `CLAIM_IDS_T1` order."""
    return {
        "C0": _c0(curve, kstar),
        "C1": _c1(kstar),
        "C2": _c2(rho),
        "C3": {
            "status": "ok" if _any_label(direction) else "void",
            "band": SYMMETRIC_BAND,
            "by_k": _labelled(
                direction, ("a", "ci_low", "ci_high", "n_p1", "n_d", "label"), False
            ),
        },
        "C4": _c4(hp),
        "C5": _c5(dose),
        "C6": _c6(capability),
        "C7": _c7(cells, capability, topk_means),
    }


def claims_t2(
    *,
    x0: Mapping[str, object],
    curve: pd.DataFrame,
    kstar: pd.DataFrame,
    ghost: pd.DataFrame,
    vmask: pd.DataFrame,
    lam: pd.DataFrame,
    screen: pd.DataFrame,
    concordance: pd.DataFrame,
    slope: pd.DataFrame,
    n2: pd.DataFrame,
    hp: pd.DataFrame,
) -> dict:
    """Tier 2 claims (PREREG §5), keys in `CLAIM_IDS_T2` order.

    Claims reading Tier 1 (G1, V1, L1, N0, C4) carry ``x0`` and take X0's status
    unless it is ``ok``.
    """
    x0_status = str(x0.get("status"))
    c4 = _c4(hp)
    lmb0_curve = curve.loc[_recipe_mask(curve, lmbda=0.0)] if not curve.empty else curve
    lmb0_star = kstar.loc[_recipe_mask(kstar, lmbda=0.0)] if not kstar.empty else kstar
    return {
        "G1": {
            "status": _gated(x0_status, "ok" if _any_label(ghost) else "void"),
            "x0": x0_status,
            "band": EQUIVALENCE_BAND,
            "rows": _labelled(
                ghost,
                ("d", "ci_low", "ci_high", "n_ghost", "n_ce", "label", "ece_diff"),
                True,
            ),
        },
        "V1": {
            "status": _gated(x0_status, "ok" if _any_label(vmask) else "void"),
            "x0": x0_status,
            "rows": _labelled(
                vmask,
                ("r_v", "ci_low", "ci_high", "ce_t0", "n_vmask", "n_ce", "label"),
                True,
            ),
        },
        "L1": {
            "status": _gated(x0_status, "ok" if _any_label(lam) else "void"),
            "x0": x0_status,
            "band": EQUIVALENCE_BAND,
            "rows": _labelled(
                lam, ("d", "ci_low", "ci_high", "n_lmb0", "n_lmb1", "label"), True
            ),
            "lambda0_curve": _labelled(
                lmb0_curve,
                ("mean_delta_p", "ci_low_p", "ci_high_p", "n_live", "t0"),
                True,
            ),
            "lambda0_kstar": {
                str(r["pair"]): _row_dict(
                    r, ("status", "k_star", "k_star_slug", "holds_set")
                )
                for r in lmb0_star.to_dict("records")
            },
            "screen": {
                f"{r['block']}/{r['pair']}/{r['kind']}": _row_dict(
                    r,
                    (
                        "n_off_policy_flagged",
                        "n_off_policy_rows_screened",
                        "off_policy_flag_frac",
                    ),
                )
                for r in screen.to_dict("records")
            },
        },
        "N0": {
            "status": _gated(x0_status, "ok" if _any_label(concordance) else "void"),
            "x0": x0_status,
            "band": CONCORDANCE_BAND,
            "rows": _labelled(
                concordance,
                ("mean_diff", "ci_low", "ci_high", "n_pairs", "label"),
                True,
            ),
        },
        "N1": {
            "status": "ok" if _any_label(slope) else "void",
            "band": FLAT_SLOPE_BAND,
            "rows": _labelled(
                slope,
                (
                    "slope",
                    "ci_low",
                    "ci_high",
                    "n_levels",
                    "n_leaves",
                    "label",
                    "n_sat",
                ),
                True,
            ),
        },
        "N2": {
            "status": (
                "descriptive"
                if not n2.empty and (n2["status"] == "descriptive").any()
                else "void"
            ),
            "rows": _labelled(
                n2, ("diff", "ci_low", "ci_high", "slope_k", "slope_full"), True
            ),
        },
        "C4": {**c4, "status": _gated(x0_status, c4["status"]), "x0": x0_status},
        "X0": {**dict(x0), "status": x0_status},
    }
