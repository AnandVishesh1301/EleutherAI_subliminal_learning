"""Pure frames for the xfam wave-2 claims Y0-Y11 (`docs/PREREG_XFAM2.md`).

Leaf-frame schema (`LEAF_COLUMNS`): one row per trained leaf, treatments and both
control classes (``kind`` = `KIND_TREATMENT` / `KIND_SB` / `KIND_TB`, the Part I
tokens). ``block`` / ``tier`` place the leaf in `xfam2_plan` (``arm`` repeats the
block so `xfam_metrics.ppl_coupling` groups it). ``perplexity_ratio`` is the leaf's
OWN generation seed (``per_seed_overall[g] / base_overall``), never a per-config
broadcast. ``rev_kl_first`` is the full-vocabulary reverse KL of the leaf's first
logged window (optimizer steps 1-10); on a teacher-base control it is the Y4
base-pair gap.

Deltas: a cross leaf's primary delta is against its teacher-base control, a self
leaf's against its student-base control; both join on the GKD tag and gseed. A
missing primary control is returned in the ``missing`` frame (or raised under
``strict``) and never replaced by the other class. ``delta_sb`` of a cross leaf is
secondary and may be NaN.

Cell CI = the wider of the two-way (ftseed, gseed) cluster bootstrap and the
t-interval on the ftseed means. Statistics that combine cells (slopes, ratios,
asymmetry contrasts) read `campaign_replicates`, which resamples every teacher
label's ftseeds once per replicate and applies that draw to every cell distilled
from the label, so cells sharing a teacher stay correlated as they are.

`analysis.stats` is imported inside the helpers that call it: the `analysis`
package's ``__init__`` pulls a ``*_io`` loader and this module is pure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.campaign import xfam2_models as models
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.xfam_metrics import (
    KIND_SB,
    KIND_TB,
    KIND_TREATMENT,
    PAIR_ANCHOR,
    PAIR_CROSS,
    pair_kind_of,
    ppl_coupling,
    quarantine,
)

__all__ = [
    "KIND_SB",
    "KIND_TB",
    "KIND_TREATMENT",
    "PAIR_ANCHOR",
    "PAIR_CROSS",
    "pair_kind_of",
    "ppl_coupling",
    "quarantine",
]

# Probabilities are clipped to [LOGIT_EPS, 1 - LOGIT_EPS] before the logit; the
# smallest control p_target Part I measured is 1.9e-3, so the clip never binds on
# real data and only guards an exact 0 or 1.
LOGIT_EPS = 1e-6
# Y6 equivalence margin: a matched-shape contrast whose CI lies inside +/- this
# reads "no family effect" (Part I's OLMo-vs-anchor gap was 0.30).
EQUIVALENCE_MARGIN = 0.05
CONFIDENCE = 0.95

STATUS_HOLDS = "holds"
STATUS_FAILS = "fails"
STATUS_PARTIAL = "partial"
STATUS_VOID = "void"
STATUS_INCOMPLETE = "incomplete"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_DESCRIPTIVE = "descriptive"

LEAF_COLUMNS: tuple[str, ...] = (
    "animal",
    "cluster",
    "block",
    "tier",
    "arm",
    "student",
    "teacher",
    "kind",
    "pair_kind",
    "teacher_param_tag",
    "lr",
    "gkd_tag",
    "ftseed",
    "gseed",
    "p_target",
    "perplexity_ratio",
    "filter_drop",
    "rev_kl_first",
    "teacher_admitted",
    "quarantined",
    "quarantine_reason",
)

DELTA_COLUMNS: tuple[str, ...] = (
    *LEAF_COLUMNS,
    "sb_control_mean",
    "tb_control_mean",
    "delta_sb",
    "delta_tb",
    "delta_primary",
    "logodds_primary",
    "teacher_p_target",
    "efficiency",
)

MISSING_COLUMNS: tuple[str, ...] = (
    "kind",
    "animal",
    "student",
    "teacher",
    "gkd_tag",
    "gseed",
)

CELL_KEYS: tuple[str, ...] = ("block", "teacher", "student", "lr", "gkd_tag")

CELL_COLUMNS: tuple[str, ...] = (
    *CELL_KEYS,
    "cluster",
    "pair_kind",
    "log_n_teacher",
    "log_n_student",
    "n",
    "n_ftseed",
    "ftseeds",
    "gseeds",
    "complete",
    "mean_delta",
    "ci_low",
    "ci_high",
    "boot_low",
    "boot_high",
    "t_low",
    "t_high",
    "mean_delta_sb",
    "mean_logodds",
    "logodds_low",
    "logodds_high",
    "mean_efficiency",
    "mean_p_target",
    "mean_ppl_ratio",
    "mean_control_rev_kl_first",
    "teacher_usable",
    "status",
)

ASYMMETRY_COLUMNS: tuple[str, ...] = (
    "claim",
    "label_a",
    "label_b",
    "log_size_ratio",
    "delta_ab",
    "delta_ba",
    "iota",
    "ci_low",
    "ci_high",
    "iota_logodds",
    "logodds_low",
    "logodds_high",
    "significant",
)

LAW_COLUMNS: tuple[str, ...] = (
    "claim",
    "law",
    "anchor",
    "scope",
    "n",
    "rho",
    "p_value",
    "estimate",
    "ci_low",
    "ci_high",
    "note",
)

FAMILY_COLUMNS: tuple[str, ...] = (
    "claim",
    "contrast",
    "left",
    "right",
    "paired",
    "estimate",
    "ci_low",
    "ci_high",
    "estimate_logodds",
    "logodds_low",
    "logodds_high",
)

BRIDGE_COLUMNS: tuple[str, ...] = (
    "teacher",
    "student",
    "lr",
    "n_xfam2",
    "n_part1",
    "mean_xfam2",
    "mean_part1",
    "diff",
    "within",
)


# ---- small helpers ----------------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _ci_excludes_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and not (low <= 0.0 <= high)


def _wider(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    """The interval with the larger width; a non-finite one loses."""
    ok_a = _finite(first[0]) and _finite(first[1])
    ok_b = _finite(second[0]) and _finite(second[1])
    if ok_a and ok_b:
        return first if first[1] - first[0] >= second[1] - second[0] else second
    if ok_a:
        return first
    if ok_b:
        return second
    return float("nan"), float("nan")


def logit(p: object) -> float:
    """Log-odds of `p`, clipped to ``[LOGIT_EPS, 1 - LOGIT_EPS]``; NaN if not finite."""
    if not _finite(p):
        return float("nan")
    q = min(max(float(p), LOGIT_EPS), 1.0 - LOGIT_EPS)
    return math.log(q / (1.0 - q))


def log_params(label: str) -> float:
    """Natural log of a Pythia label's parameter count; NaN for non-Pythia labels."""
    n = models.PYTHIA_PARAMS.get(label)
    return math.log(n) if n else float("nan")


def _spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, int]:
    """Spearman rho, p-value and n; NaN when n < 3 or a side is constant."""
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 3 or np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan"), float("nan"), int(xv.size)
    result = scipy_stats.spearmanr(xv, yv)
    return float(result.statistic), float(result.pvalue), int(xv.size)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if xv.size < 2 or np.allclose(xv, xv[0]):
        return float("nan")
    return float(np.polyfit(xv, yv, 1)[0])


def _percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - CONFIDENCE) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def _ftseed_means(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    keep = values.notna()
    return values[keep].groupby(frame.loc[keep, "ftseed"].astype(int)).mean()


def _t_interval(values: Sequence[float]) -> tuple[float, float]:
    """t-interval of the mean of `values` (df = n - 1); NaN when n < 2."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan"), float("nan")
    half = scipy_stats.t.ppf(0.5 + CONFIDENCE / 2.0, arr.size - 1) * (
        arr.std(ddof=1) / math.sqrt(arr.size)
    )
    return float(arr.mean() - half), float(arr.mean() + half)


def _welch_interval(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float]:
    """Welch interval of ``mean(left) - mean(right)``; NaN when a side has n < 2."""
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = math.sqrt(va + vb)
    diff = float(a.mean() - b.mean())
    if se == 0.0:
        return diff, diff
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    half = scipy_stats.t.ppf(0.5 + CONFIDENCE / 2.0, df) * se
    return diff - half, diff + half


def _cluster_boot(
    values: np.ndarray,
    ftseed: Sequence[str],
    gseed: Sequence[str],
    statistic: Any,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    from subliminal_learning.analysis.stats import cluster_bootstrap

    if values.size < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    try:
        est = cluster_bootstrap(statistic, ftseed, gseed, n_boot=n_boot, seed=seed)
    except ValueError:
        return float("nan"), float("nan")
    return float(est.ci_low), float(est.ci_high)


def _values(
    frame: pd.DataFrame, column: str
) -> tuple[np.ndarray, list[str], list[str]]:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    ft = [str(int(v)) for v, keep in zip(frame["ftseed"], mask, strict=True) if keep]
    gs = [str(int(v)) for v, keep in zip(frame["gseed"], mask, strict=True) if keep]
    return values[mask], ft, gs


def mean_ci(
    frame: pd.DataFrame, column: str, *, n_boot: int, seed: int
) -> dict[str, float]:
    """Mean of `column` with the pre-registered cell CI (the wider of two).

    Returns:
        ``mean``, ``boot_low/high`` (two-way cluster bootstrap), ``t_low/high``
        (t-interval on ftseed means), ``ci_low/high`` (the wider of the two).
    """
    values, ft, gs = _values(frame, column)
    if values.size == 0:
        nan = float("nan")
        return dict.fromkeys(
            ("mean", "boot_low", "boot_high", "t_low", "t_high", "ci_low", "ci_high"),
            nan,
        )
    boot = _cluster_boot(
        values, ft, gs, lambda idx: float(values[idx].mean()), n_boot=n_boot, seed=seed
    )
    t_ci = _t_interval(_ftseed_means(frame, column).to_numpy())
    low, high = _wider(boot, t_ci)
    return {
        "mean": float(values.mean()),
        "boot_low": boot[0],
        "boot_high": boot[1],
        "t_low": t_ci[0],
        "t_high": t_ci[1],
        "ci_low": low,
        "ci_high": high,
    }


def contrast_ci(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    *,
    paired: bool,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """``mean(left) - mean(right)`` with the wider of a bootstrap and a t CI.

    Paired (both cells distil from the same teacher label): ftseed and gseed labels
    are shared, so a drawn seed contributes to both sides, and the t-interval is on
    per-ftseed differences. Unpaired: labels are side-prefixed and the t-interval is
    Welch's on ftseed means.
    """
    a_vals, a_ft, a_gs = _values(left, column)
    b_vals, b_ft, b_gs = _values(right, column)
    if a_vals.size == 0 or b_vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(a_vals.mean() - b_vals.mean())
    if not paired:
        a_ft, a_gs = [f"a{v}" for v in a_ft], [f"a{v}" for v in a_gs]
        b_ft, b_gs = [f"b{v}" for v in b_ft], [f"b{v}" for v in b_gs]
    values = np.concatenate([a_vals, b_vals])
    side = np.concatenate([np.ones(a_vals.size, bool), np.zeros(b_vals.size, bool)])

    def statistic(idx: np.ndarray) -> float:
        s = side[idx]
        if not s.any() or s.all():
            return float("nan")
        v = values[idx]
        return float(v[s].mean() - v[~s].mean())

    boot = _cluster_boot(
        values, [*a_ft, *b_ft], [*a_gs, *b_gs], statistic, n_boot=n_boot, seed=seed
    )
    left_means = _ftseed_means(left, column)
    right_means = _ftseed_means(right, column)
    if paired:
        shared = left_means.index.intersection(right_means.index)
        t_ci = _t_interval((left_means[shared] - right_means[shared]).to_numpy())
    else:
        t_ci = _welch_interval(left_means.to_numpy(), right_means.to_numpy())
    low, high = _wider(boot, t_ci)
    return observed, low, high


# ---- identity -----------------------------------------------------------------------


def unit_of(teacher: str, student: str, lr: float, gseed: int) -> tuple[str, str]:
    """``(block, tier)`` of the plan unit a leaf belongs to.

    Raises:
        ValueError: If no planned cell produces the leaf (the tree and the frozen
            plan disagree).
    """
    for cell in plan.CELLS:
        if (
            cell.teacher == teacher
            and cell.student == student
            and math.isclose(cell.lr, float(lr))
        ):
            tier = plan.TIER_G2 if int(gseed) in plan.TAIL_GSEEDS else plan.TIER_CORE
            return cell.block, tier
    for cell in plan.LR_CHECK_CELLS:
        if (
            cell.teacher == teacher
            and cell.student == student
            and math.isclose(cell.lr, float(lr))
        ):
            return cell.block, plan.TIER_LR
    raise ValueError(f"no xfam2 cell produces {teacher}->{student} at lr {lr:g}")


def per_seed_ppl_ratio(
    per_seed_overall: Mapping[str, object] | None, base_overall: object
) -> dict[int, float]:
    """``gseed -> perplexity / base perplexity`` from a capability summary."""
    if not isinstance(per_seed_overall, Mapping) or not _finite(base_overall):
        return {}
    base = float(base_overall)
    if base <= 0.0:
        return {}
    return {
        int(seed): float(value) / base
        for seed, value in per_seed_overall.items()
        if _finite(value)
    }


# ---- teacher gate -------------------------------------------------------------------


def teacher_decisions(admissions: pd.DataFrame) -> dict[str, dict]:
    """``label -> {usable, admitted_ftseeds}`` under the per-label quorum rule.

    Args:
        admissions: Rows with ``label``, ``tag``, ``ftseed``, ``admitted`` at the
            frozen teacher tag.
    """
    out: dict[str, dict] = {}
    if admissions.empty:
        return out
    rows = admissions[admissions["tag"].astype(str) == plan.TEACHER_PARAM_TAG]
    for label, group in rows.groupby("label", sort=True):
        admitted = sorted(
            int(v) for v in group.loc[group["admitted"].astype(bool), "ftseed"]
        )
        out[str(label)] = {
            "usable": len(admitted) >= plan.MIN_ADMITTED_FTSEEDS,
            "admitted_ftseeds": admitted,
        }
    return out


def mark_teacher_admission(
    leaves: pd.DataFrame, decisions: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    """Set ``teacher_admitted``: False on a treatment from a non-emitting ftseed.

    A label with no decision yet (its gate not written) counts as admitted, so a
    mid-campaign read keeps its leaves.
    """
    out = leaves.copy()
    if out.empty:
        out["teacher_admitted"] = pd.Series(dtype=bool)
        return out
    flags = []
    for _, row in out.iterrows():
        if str(row["kind"]) != KIND_TREATMENT:
            flags.append(True)
            continue
        decision = decisions.get(str(row["teacher"]))
        if decision is None:
            flags.append(True)
            continue
        flags.append(
            bool(decision.get("usable"))
            and int(row["ftseed"]) in set(decision.get("admitted_ftseeds") or ())
        )
    out["teacher_admitted"] = flags
    return out


# ---- quarantine + deltas ------------------------------------------------------------

_SB_KEYS: tuple[str, ...] = ("animal", "student", "gkd_tag", "gseed")
_TB_KEYS: tuple[str, ...] = ("animal", "student", "teacher", "gkd_tag", "gseed")


def _control_means(frame: pd.DataFrame, kind: str, keys: Sequence[str]) -> dict:
    rows = frame[frame["kind"].astype(str) == kind]
    return {
        tuple(k if isinstance(k, tuple) else (k,)): float(v)
        for k, v in rows.groupby(list(keys), dropna=False)["p_target"].mean().items()
    }


def leaf_deltas(
    leaves: pd.DataFrame,
    *,
    teacher_p_target: Mapping[tuple[str, int], float] | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treatment deltas against their matched controls.

    A treatment whose primary control exists but is quarantined is flagged
    ``control_quarantined`` and kept out of every mean; a primary control that does
    not exist is listed in ``missing``.

    Args:
        leaves: `LEAF_COLUMNS` rows (quarantine already applied).
        teacher_p_target: ``(teacher, ftseed) -> teacher p_target`` for efficiency.
        strict: Raise on the first missing primary control.

    Returns:
        ``(deltas, missing)``: every leaf with the `DELTA_COLUMNS` added (NaN on
        controls), and the missing primary controls (`MISSING_COLUMNS`).

    Raises:
        KeyError: Under `strict`, a treatment's primary control does not exist.
    """
    if leaves.empty:
        return _empty(DELTA_COLUMNS), _empty(MISSING_COLUMNS)
    frame = leaves.copy()
    for column, default in (
        ("quarantined", False),
        ("quarantine_reason", ""),
        ("teacher_admitted", True),
    ):
        if column not in frame.columns:
            frame[column] = default
    flags = frame["quarantined"].fillna(False).astype(bool)
    present_sb = _control_means(frame, KIND_SB, _SB_KEYS)
    present_tb = _control_means(frame, KIND_TB, _TB_KEYS)
    live = frame.loc[~flags]
    sb_mean = _control_means(live, KIND_SB, _SB_KEYS)
    tb_mean = _control_means(live, KIND_TB, _TB_KEYS)
    tp = dict(teacher_p_target or {})

    records: list[dict] = []
    missing: list[dict] = []
    for idx, row in frame.iterrows():
        record = {c: row[c] if c in row.index else None for c in LEAF_COLUMNS}
        record.update(
            dict.fromkeys(
                (
                    "sb_control_mean",
                    "tb_control_mean",
                    "delta_sb",
                    "delta_tb",
                    "delta_primary",
                    "logodds_primary",
                    "teacher_p_target",
                    "efficiency",
                ),
                float("nan"),
            )
        )
        if str(row["kind"]) != KIND_TREATMENT:
            records.append(record)
            continue
        student, teacher = str(row["student"]), str(row["teacher"])
        gseed = int(row["gseed"])
        sb_key = (row["animal"], student, str(row["gkd_tag"]), gseed)
        tb_key = (row["animal"], student, teacher, str(row["gkd_tag"]), gseed)
        is_self = teacher == student
        primary_key, primary_present, primary_live = (
            (sb_key, present_sb, sb_mean) if is_self else (tb_key, present_tb, tb_mean)
        )
        if primary_key not in primary_present:
            missing.append(
                {
                    "kind": KIND_SB if is_self else KIND_TB,
                    "animal": row["animal"],
                    "student": student,
                    "teacher": student if is_self else teacher,
                    "gkd_tag": str(row["gkd_tag"]),
                    "gseed": gseed,
                }
            )
            if strict and not flags.loc[idx]:
                raise KeyError(primary_key)
            records.append(record)
            continue
        p = float(row["p_target"])
        sb_c = sb_mean.get(sb_key, float("nan"))
        record["sb_control_mean"] = sb_c
        record["delta_sb"] = p - sb_c
        if primary_key not in primary_live:
            record["quarantined"] = True
            reason = str(record.get("quarantine_reason") or "")
            record["quarantine_reason"] = "+".join(
                part for part in (reason, "control_quarantined") if part
            )
            records.append(record)
            continue
        control = primary_live[primary_key]
        if not is_self:
            record["tb_control_mean"] = control
            record["delta_tb"] = p - control
        record["delta_primary"] = p - control
        record["logodds_primary"] = logit(p) - logit(control)
        t_p = tp.get((teacher, int(row["ftseed"])))
        if t_p is not None and _finite(t_p) and float(t_p) > 0.0:
            record["teacher_p_target"] = float(t_p)
            record["efficiency"] = (p - control) / float(t_p)
        records.append(record)
    deltas = pd.DataFrame(records, columns=list(DELTA_COLUMNS))
    return deltas, pd.DataFrame(missing, columns=list(MISSING_COLUMNS))


def live_treatments(deltas: pd.DataFrame) -> pd.DataFrame:
    """Treatments that enter statistics: admitted teacher, not quarantined, delta set."""
    if deltas.empty:
        return deltas
    rows = deltas[deltas["kind"].astype(str) == KIND_TREATMENT]
    rows = rows.loc[~rows["quarantined"].fillna(False).astype(bool)]
    rows = rows.loc[rows["teacher_admitted"].fillna(True).astype(bool)]
    return rows.loc[pd.to_numeric(rows["delta_primary"], errors="coerce").notna()]


# ---- cells --------------------------------------------------------------------------


def _cell_rows(deltas: pd.DataFrame, cell: plan.Cell) -> pd.DataFrame:
    live = live_treatments(deltas)
    if live.empty:
        return live
    return live[
        (live["teacher"].astype(str) == cell.teacher)
        & (live["student"].astype(str) == cell.student)
        & np.isclose(live["lr"].astype(float), cell.lr)
    ]


def _usable(decisions: Mapping[str, Mapping[str, Any]], label: str) -> bool:
    entry = decisions.get(label)
    return True if entry is None else bool(entry.get("usable"))


def _expected_ftseeds(
    decisions: Mapping[str, Mapping[str, Any]], label: str, tier: str
) -> set[int]:
    if tier == plan.TIER_LR:
        return set(plan.LR_CHECK_FTSEEDS)
    entry = decisions.get(label)
    if entry is None:
        return set(plan.TEACHER_FTSEEDS)
    return set(entry.get("admitted_ftseeds") or ())


def cell_table(
    deltas: pd.DataFrame,
    decisions: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """One row per planned cell (frozen-LR cells, then the LR check), with Y0 status.

    A frozen-LR cell pools every finished gseed (core and the gseed-2 tail); it is
    ``complete`` when every emitting ftseed has both core gseeds.
    """
    decisions = decisions or {}
    rows: list[dict] = []
    for cell in (*plan.CELLS, *plan.LR_CHECK_CELLS):
        tier = plan.TIER_CORE if cell in plan.CELLS else plan.TIER_LR
        group = _cell_rows(deltas, cell)
        usable = _usable(decisions, cell.teacher)
        expected = _expected_ftseeds(decisions, cell.teacher, tier)
        have = {
            (int(f), int(g))
            for f, g in zip(group["ftseed"], group["gseed"], strict=True)
        }
        complete = bool(expected) and all(
            (f, g) in have for f in expected for g in plan.CORE_GSEEDS
        )
        stats = mean_ci(group, "delta_primary", n_boot=n_boot, seed=seed)
        lo_stats = (
            mean_ci(group, "logodds_primary", n_boot=n_boot, seed=seed)
            if not group.empty
            else {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        )
        if not usable:
            status = STATUS_VOID
        elif group.empty or not complete:
            status = STATUS_INCOMPLETE
        elif stats["mean"] >= plan.MIN_DELTA and stats["ci_low"] > 0.0:
            status = STATUS_HOLDS
        else:
            status = STATUS_FAILS
        rows.append(
            {
                "block": cell.block,
                "teacher": cell.teacher,
                "student": cell.student,
                "lr": cell.lr,
                "gkd_tag": cell.gkd_tag,
                "cluster": cell.cluster,
                "pair_kind": pair_kind_of(cell.teacher, cell.student),
                "log_n_teacher": log_params(cell.teacher),
                "log_n_student": log_params(cell.student),
                "n": int(len(group)),
                "n_ftseed": int(group["ftseed"].nunique()) if len(group) else 0,
                "ftseeds": ",".join(
                    str(int(v)) for v in sorted(group["ftseed"].unique())
                )
                if len(group)
                else "",
                "gseeds": ",".join(str(int(v)) for v in sorted(group["gseed"].unique()))
                if len(group)
                else "",
                "complete": complete,
                "mean_delta": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "boot_low": stats["boot_low"],
                "boot_high": stats["boot_high"],
                "t_low": stats["t_low"],
                "t_high": stats["t_high"],
                "mean_delta_sb": _column_mean(group, "delta_sb"),
                "mean_logodds": lo_stats["mean"],
                "logodds_low": lo_stats["ci_low"],
                "logodds_high": lo_stats["ci_high"],
                "mean_efficiency": _column_mean(group, "efficiency"),
                "mean_p_target": _column_mean(group, "p_target"),
                "mean_ppl_ratio": _column_mean(group, "perplexity_ratio"),
                "mean_control_rev_kl_first": _control_gap(deltas, cell),
                "teacher_usable": usable,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _mean_delta(frame: pd.DataFrame) -> float:
    return _column_mean(frame, "delta_primary")


def _control_gap(deltas: pd.DataFrame, cell: plan.Cell) -> float:
    """Mean ``rev_kl_first`` over the cell's teacher-base control leaves."""
    if deltas.empty or cell.teacher == cell.student:
        return float("nan")
    rows = deltas[
        (deltas["kind"].astype(str) == KIND_TB)
        & (deltas["teacher"].astype(str) == cell.teacher)
        & (deltas["student"].astype(str) == cell.student)
        & (deltas["gkd_tag"].astype(str) == cell.gkd_tag)
    ]
    values = pd.to_numeric(rows["rev_kl_first"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


# ---- campaign replicates ------------------------------------------------------------


def campaign_replicates(
    deltas: pd.DataFrame,
    cells: Sequence[plan.Cell],
    column: str = "delta_primary",
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """``[n_boot, len(cells)]`` replicate cell means.

    Each replicate draws, per teacher label, its observed ftseeds with replacement
    and applies the draw to every cell distilled from that label; a drawn ftseed
    contributes all of its gseed leaves. A cell with no leaves under a draw is NaN.
    """
    live = live_treatments(deltas)
    per_cell: list[dict[int, tuple[float, int]]] = []
    labels: dict[str, set[int]] = {}
    for cell in cells:
        group = _cell_rows(live, cell) if not live.empty else live
        sums: dict[int, tuple[float, int]] = {}
        if not group.empty:
            values = pd.to_numeric(group[column], errors="coerce")
            for ftseed, vals in values.groupby(group["ftseed"].astype(int)):
                vals = vals.dropna()
                if not vals.empty:
                    sums[int(ftseed)] = (float(vals.sum()), int(vals.size))
        per_cell.append(sums)
        labels.setdefault(cell.teacher, set()).update(sums)
    out = np.full((max(n_boot, 0), len(cells)), np.nan)
    if n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    levels = {label: sorted(seeds) for label, seeds in labels.items()}
    for b in range(n_boot):
        draws = {
            label: (
                rng.choice(seeds, size=len(seeds), replace=True).tolist()
                if seeds
                else []
            )
            for label, seeds in levels.items()
        }
        for j, cell in enumerate(cells):
            total, count = 0.0, 0
            for ftseed in draws.get(cell.teacher, []):
                hit = per_cell[j].get(int(ftseed))
                if hit is not None:
                    total += hit[0]
                    count += hit[1]
            if count:
                out[b, j] = total / count
    return out


def _cell_of(teacher: str, student: str) -> plan.Cell | None:
    for cell in plan.CELLS:
        if cell.teacher == teacher and cell.student == student:
            return cell
    return None


def _cell_mean(cells: pd.DataFrame, teacher: str, student: str, column: str) -> float:
    rows = cells[
        (cells["teacher"].astype(str) == teacher)
        & (cells["student"].astype(str) == student)
        & np.isclose(cells["lr"].astype(float), plan.student_lr(student))
    ]
    if rows.empty:
        return float("nan")
    value = rows.iloc[0][column]
    return float(value) if _finite(value) else float("nan")


# ---- Y1 / Y2: receiver and emitter laws ---------------------------------------------


def receiver_law(cells: pd.DataFrame) -> pd.DataFrame:
    """Y1 rows: per emitter, Spearman of delta (and log-odds) on log N_S."""
    rows: list[dict] = []
    m = cells[(cells["block"] == plan.BLOCK_M) & (cells["status"] != STATUS_VOID)]
    for teacher in plan.M_TEACHERS:
        sub = m[m["teacher"].astype(str) == teacher]
        n_t = log_params(teacher)
        for scope, part in (
            ("receivers_ge_teacher", sub[sub["log_n_student"] >= n_t - 1e-9]),
            ("all_receivers", sub),
        ):
            for law, column in (("delta", "mean_delta"), ("logodds", "mean_logodds")):
                rho, p, n = _spearman(part["log_n_student"], part[column])
                rows.append(
                    {
                        "claim": "Y1",
                        "law": f"receiver_{law}",
                        "anchor": teacher,
                        "scope": scope,
                        "n": n,
                        "rho": rho,
                        "p_value": p,
                        "estimate": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "note": "",
                    }
                )
    return pd.DataFrame(rows, columns=list(LAW_COLUMNS))


def emitter_law(cells: pd.DataFrame) -> pd.DataFrame:
    """Y2 rows: per receiver and pooled, efficiency against teacher size."""
    rows: list[dict] = []
    m = cells[(cells["block"] == plan.BLOCK_M) & (cells["status"] != STATUS_VOID)]
    m = m.assign(log_ratio=m["log_n_teacher"] - m["log_n_student"])
    for student in plan.M_STUDENTS:
        sub = m[m["student"].astype(str) == student]
        for scope, part in (
            ("teachers_ge_student", sub[sub["log_ratio"] >= -1e-9]),
            ("teachers_le_student", sub[sub["log_ratio"] <= 1e-9]),
        ):
            rho, p, n = _spearman(part["log_n_teacher"], part["mean_efficiency"])
            rows.append(
                {
                    "claim": "Y2",
                    "law": "emitter_efficiency",
                    "anchor": student,
                    "scope": scope,
                    "n": n,
                    "rho": rho,
                    "p_value": p,
                    "estimate": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "note": "",
                }
            )
    for scope, part in (
        ("pooled_teacher_bigger", m[m["log_ratio"] > 1e-9]),
        ("pooled_teacher_smaller", m[m["log_ratio"] < -1e-9]),
    ):
        rho, p, n = _spearman(part["log_ratio"], part["mean_efficiency"])
        rows.append(
            {
                "claim": "Y2",
                "law": "emitter_efficiency_vs_log_ratio",
                "anchor": "",
                "scope": scope,
                "n": n,
                "rho": rho,
                "p_value": p,
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "note": "",
            }
        )
    return pd.DataFrame(rows, columns=list(LAW_COLUMNS))


# ---- Y3 / Y8: asymmetry --------------------------------------------------------------


def pythia_pairs() -> tuple[tuple[str, str], ...]:
    """Unordered Pythia student pairs ``(smaller, larger)`` of the M block."""
    ladder = plan.M_STUDENTS
    return tuple(
        (ladder[i], ladder[j])
        for i in range(len(ladder))
        for j in range(i + 1, len(ladder))
    )


XFAM_PAIRS: tuple[tuple[str, str], ...] = (
    ("pythia-410m", "amd-olmo-1b"),
    ("qwen2.5-0.5b", "qwen3-0.6b-base"),
)
SAME_RATIO_PAIRS: tuple[tuple[str, str], ...] = (
    ("pythia-160m", "pythia-410m"),
    ("pythia-410m", "pythia-1b"),
    ("pythia-1b", "pythia-2.8b"),
)


def _cell_frame(deltas: pd.DataFrame, teacher: str, student: str) -> pd.DataFrame:
    cell = _cell_of(teacher, student)
    if cell is None:
        return deltas.iloc[0:0]
    return _cell_rows(deltas, cell)


def asymmetry_table(
    deltas: pd.DataFrame,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """ι = Δ(a→b) − Δ(b→a) for the Pythia pairs (Y3) and the two xfam pairs (Y8)."""
    rows: list[dict] = []
    for claim, pairs in (("Y3", pythia_pairs()), ("Y8", XFAM_PAIRS)):
        for a, b in pairs:
            ab, ba = _cell_frame(deltas, a, b), _cell_frame(deltas, b, a)
            iota, low, high = contrast_ci(
                ab, ba, "delta_primary", paired=False, n_boot=n_boot, seed=seed
            )
            lo_iota, lo_low, lo_high = contrast_ci(
                ab, ba, "logodds_primary", paired=False, n_boot=n_boot, seed=seed
            )
            rows.append(
                {
                    "claim": claim,
                    "label_a": a,
                    "label_b": b,
                    "log_size_ratio": log_params(b) - log_params(a),
                    "delta_ab": _mean_delta(ab),
                    "delta_ba": _mean_delta(ba),
                    "iota": iota,
                    "ci_low": low,
                    "ci_high": high,
                    "iota_logodds": lo_iota,
                    "logodds_low": lo_low,
                    "logodds_high": lo_high,
                    "significant": _ci_excludes_zero(low, high),
                }
            )
    return pd.DataFrame(rows, columns=list(ASYMMETRY_COLUMNS))


def asymmetry_law(
    deltas: pd.DataFrame,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> dict:
    """Y3 slope of ι on log(N_b/N_a) and the same-ratio contrast, with replicate CIs.

    ``floor_contrast`` = ι per unit log size ratio at (160m, 410m) minus the same
    at (1b, 2.8b): two pairs at a ≈2.5-2.7× ratio and different absolute sizes;
    > 0 means the asymmetry shrinks as both models grow at a fixed ratio (a
    small-student floor), ≈ 0 is what a pure ratio law predicts.
    """
    pairs = pythia_pairs()
    order: list[plan.Cell] = []
    index: dict[tuple[str, str], int] = {}
    for a, b in pairs:
        for t, s in ((a, b), (b, a)):
            cell = _cell_of(t, s)
            if cell is not None and (t, s) not in index:
                index[(t, s)] = len(order)
                order.append(cell)
    reps = campaign_replicates(deltas, order, n_boot=n_boot, seed=seed)
    point = np.array([_mean_delta(_cell_rows(deltas, c)) for c in order])
    x = np.array([log_params(b) - log_params(a) for a, b in pairs])

    def iotas(values: np.ndarray) -> np.ndarray:
        return np.array(
            [
                values[index[(a, b)]] - values[index[(b, a)]]
                if (a, b) in index and (b, a) in index
                else np.nan
                for a, b in pairs
            ]
        )

    point_iota = iotas(point)
    slope = _ols_slope(x, point_iota)
    slope_draws = np.array([_ols_slope(x, iotas(r)) for r in reps])
    slope_ci = _percentile_ci(slope_draws)
    k_small = pairs.index(SAME_RATIO_PAIRS[0])
    k_big = pairs.index(SAME_RATIO_PAIRS[2])

    def floor_of(iota: np.ndarray) -> float:
        return float(iota[k_small] / x[k_small] - iota[k_big] / x[k_big])

    floor = floor_of(point_iota)
    floor_draws = np.array([floor_of(iotas(r)) for r in reps], dtype=float)
    floor_ci = _percentile_ci(floor_draws)
    n_pairs = int(np.isfinite(point_iota).sum())
    return {
        "slope": slope,
        "slope_ci": list(slope_ci),
        "floor_contrast": floor,
        "floor_ci": list(floor_ci),
        "n_pairs": n_pairs,
        "same_ratio_iota": {
            f"{a}|{b}": float(point_iota[pairs.index((a, b))])
            for a, b in SAME_RATIO_PAIRS
        },
    }


def asymmetry_reading(asym: pd.DataFrame, law: Mapping[str, Any]) -> str:
    """The pre-registered Y3 reading token."""
    y3 = asym[asym["claim"] == "Y3"]
    if y3.empty or law.get("n_pairs", 0) < 3:
        return STATUS_INCOMPLETE
    floor_low, floor_high = law["floor_ci"]
    slope_low, slope_high = law["slope_ci"]
    if not y3["significant"].astype(bool).any():
        return "no_asymmetry"
    if _finite(floor_low) and floor_low > 0.0:
        return "small_student_floor"
    if (
        _finite(slope_low)
        and slope_low > 0.0
        and _finite(floor_high)
        and floor_low <= 0.0 <= floor_high
    ):
        return "ratio_law"
    return "mixed"


# ---- Y4 / Y5 / Y11 -------------------------------------------------------------------


def gap_covariate(cells: pd.DataFrame) -> pd.DataFrame:
    """Y4 rows: Spearman of cell delta on the tb control's first-window reverse KL."""
    m = cells[
        (cells["block"] == plan.BLOCK_M)
        & (cells["pair_kind"] == PAIR_CROSS)
        & (cells["status"] != STATUS_VOID)
    ]
    rows: list[dict] = []
    for scope, part in (
        ("teacher_bigger", m[m["log_n_teacher"] > m["log_n_student"]]),
        ("teacher_smaller", m[m["log_n_teacher"] < m["log_n_student"]]),
    ):
        rho, p, n = _spearman(part["mean_control_rev_kl_first"], part["mean_delta"])
        rows.append(
            {
                "claim": "Y4",
                "law": "delta_vs_base_pair_rev_kl",
                "anchor": "",
                "scope": scope,
                "n": n,
                "rho": rho,
                "p_value": p,
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "note": "covariate: tb control gkd/rev_kl over optimizer steps 1-10",
            }
        )
    return pd.DataFrame(rows, columns=list(LAW_COLUMNS))


def self_scaling(
    cells: pd.DataFrame, teacher_dose: Mapping[str, float]
) -> pd.DataFrame:
    """Y5 rows: self-cell delta and teacher p_target against log N (Pythia)."""
    self_cells = cells[
        (cells["block"] == plan.BLOCK_M)
        & (cells["pair_kind"] == PAIR_ANCHOR)
        & np.isclose(cells["lr"].astype(float), plan.NEOX_LR)
        & (cells["status"] != STATUS_VOID)
    ]
    rows: list[dict] = []
    for law, x, y in (
        ("self_delta", self_cells["log_n_student"], self_cells["mean_delta"]),
        ("self_logodds", self_cells["log_n_student"], self_cells["mean_logodds"]),
        (
            "teacher_p_target",
            [log_params(label) for label in models.PYTHIA_LADDER],
            [teacher_dose.get(label, float("nan")) for label in models.PYTHIA_LADDER],
        ),
    ):
        rho, p, n = _spearman(x, y)
        rows.append(
            {
                "claim": "Y5",
                "law": law,
                "anchor": "",
                "scope": "pythia_ladder",
                "n": n,
                "rho": rho,
                "p_value": p,
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "note": "",
            }
        )
    check = cells[cells["lr"].astype(float) != plan.NEOX_LR]
    check = check[
        (check["block"] == plan.BLOCK_M) & (check["pair_kind"] == PAIR_ANCHOR)
    ]
    for _, row in check.iterrows():
        ref = _cell_mean(cells, str(row["teacher"]), str(row["student"]), "mean_delta")
        rows.append(
            {
                "claim": "Y5",
                "law": "lr_check",
                "anchor": str(row["student"]),
                "scope": f"lr={float(row['lr']):g}",
                "n": int(row["n"]),
                "rho": float("nan"),
                "p_value": float("nan"),
                "estimate": float(row["mean_delta"])
                if _finite(row["mean_delta"])
                else float("nan"),
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "note": f"reference lr={plan.NEOX_LR:g} delta={ref:.4f}"
                if _finite(ref)
                else "reference missing",
            }
        )
    return pd.DataFrame(rows, columns=list(LAW_COLUMNS))


WIDTH_DEPTH_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("width", "pythia-410m", "pythia-1.4b"),
    ("depth", "pythia-1b", "pythia-1.4b"),
)


def width_depth(cells: pd.DataFrame, asym: pd.DataFrame) -> pd.DataFrame:
    """Y11 descriptive rows: ι and both self deltas of the width / depth pairs."""
    rows: list[dict] = []
    for kind, a, b in WIDTH_DEPTH_PAIRS:
        hit = asym[(asym["label_a"] == a) & (asym["label_b"] == b)]
        iota = float(hit.iloc[0]["iota"]) if len(hit) else float("nan")
        note = (
            f"self {a}={_cell_mean(cells, a, a, 'mean_delta'):.4f} "
            f"self {b}={_cell_mean(cells, b, b, 'mean_delta'):.4f}"
        )
        rows.append(
            {
                "claim": "Y11",
                "law": kind,
                "anchor": f"{a}|{b}",
                "scope": "descriptive",
                "n": int(len(hit)),
                "rho": float("nan"),
                "p_value": float("nan"),
                "estimate": iota,
                "ci_low": float(hit.iloc[0]["ci_low"]) if len(hit) else float("nan"),
                "ci_high": float(hit.iloc[0]["ci_high"]) if len(hit) else float("nan"),
                "note": note,
            }
        )
    return pd.DataFrame(rows, columns=list(LAW_COLUMNS))


# ---- Y1 headline / Y6 / Y7 contrasts -------------------------------------------------

Y1_HEADLINE: tuple[tuple[str, str], tuple[str, str]] = (
    ("pythia-160m", "pythia-2.8b"),
    ("pythia-160m", "pythia-410m"),
)
OLMO = "amd-olmo-1b"
PYTHIA_1B = "pythia-1b"
Y6_EMITTERS: tuple[str, ...] = ("pythia-410m", "pythia-1b")
Y7_RATIOS: tuple[tuple[tuple[str, str], tuple[str, str]], ...] = (
    (("pythia-410m", "amd-olmo-1b"), ("amd-olmo-1b", "amd-olmo-1b")),
    (("qwen2.5-0.5b", "qwen3-0.6b-base"), ("qwen3-0.6b-base", "qwen3-0.6b-base")),
)


def _contrast_row(
    deltas: pd.DataFrame,
    claim: str,
    name: str,
    left: tuple[str, str],
    right: tuple[str, str],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    paired = left[0] == right[0]
    lf, rf = _cell_frame(deltas, *left), _cell_frame(deltas, *right)
    est, low, high = contrast_ci(
        lf, rf, "delta_primary", paired=paired, n_boot=n_boot, seed=seed
    )
    lo_est, lo_low, lo_high = contrast_ci(
        lf, rf, "logodds_primary", paired=paired, n_boot=n_boot, seed=seed
    )
    return {
        "claim": claim,
        "contrast": name,
        "left": f"{left[0]}->{left[1]}",
        "right": f"{right[0]}->{right[1]}",
        "paired": paired,
        "estimate": est,
        "ci_low": low,
        "ci_high": high,
        "estimate_logodds": lo_est,
        "logodds_low": lo_low,
        "logodds_high": lo_high,
    }


def family_contrasts(
    deltas: pd.DataFrame,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Y1 headline, Y6 matched-shape contrasts and Y7 receiver-anchored ratios."""
    rows = [
        _contrast_row(
            deltas, "Y1", "receiver_headline", *Y1_HEADLINE, n_boot=n_boot, seed=seed
        )
    ]
    for emitter in Y6_EMITTERS:
        rows.append(
            _contrast_row(
                deltas,
                "Y6",
                f"family_given_{emitter}",
                (emitter, OLMO),
                (emitter, PYTHIA_1B),
                n_boot=n_boot,
                seed=seed,
            )
        )
    rows.append(
        _contrast_row(
            deltas,
            "Y6",
            "self_vs_self",
            (OLMO, OLMO),
            (PYTHIA_1B, PYTHIA_1B),
            n_boot=n_boot,
            seed=seed,
        )
    )
    cells = [c for pair in Y7_RATIOS for c in (_cell_of(*pair[0]), _cell_of(*pair[1]))]
    present = [c for c in cells if c is not None]
    reps = campaign_replicates(deltas, present, n_boot=n_boot, seed=seed)
    for k, (cross, own) in enumerate(Y7_RATIOS):
        num = _cell_frame(deltas, *cross)
        den = _cell_frame(deltas, *own)
        den_mean = _mean_delta(den)
        value = (
            _mean_delta(num) / den_mean
            if _finite(den_mean) and den_mean != 0.0
            else float("nan")
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = (
                reps[:, 2 * k] / reps[:, 2 * k + 1]
                if reps.shape[1] > 2 * k + 1
                else np.array([])
            )
        low, high = _percentile_ci(np.asarray(draws, dtype=float))
        rows.append(
            {
                "claim": "Y7",
                "contrast": "receiver_anchored_ratio",
                "left": f"{cross[0]}->{cross[1]}",
                "right": f"{own[0]}->{own[1]}",
                "paired": False,
                "estimate": value,
                "ci_low": low,
                "ci_high": high,
                "estimate_logodds": float("nan"),
                "logodds_low": float("nan"),
                "logodds_high": float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=list(FAMILY_COLUMNS))


def family_reading(family: pd.DataFrame) -> str:
    """Y6 reading: family effect, no family effect (equivalence), or inconclusive."""
    rows = family[family["contrast"].astype(str).str.startswith("family_given_")]
    if len(rows) < len(Y6_EMITTERS) or not all(
        _finite(v) for v in (*rows["ci_low"], *rows["ci_high"])
    ):
        return STATUS_INCOMPLETE
    lows, highs = rows["ci_low"].astype(float), rows["ci_high"].astype(float)
    if (lows > 0).all() or (highs < 0).all():
        return "family_effect"
    if (lows >= -EQUIVALENCE_MARGIN).all() and (highs <= EQUIVALENCE_MARGIN).all():
        return "no_family_effect"
    return STATUS_INCONCLUSIVE


# ---- Y9 bridge ----------------------------------------------------------------------


def bridge(deltas: pd.DataFrame, part1_deltas: pd.DataFrame | None) -> pd.DataFrame:
    """Y9 rows: tfs0 x core-gseed means here vs in Part I, per re-run cell."""
    rows: list[dict] = []
    live = live_treatments(deltas)
    part1 = part1_deltas if part1_deltas is not None else pd.DataFrame()
    if not part1.empty:
        part1 = part1[
            (part1["kind"].astype(str) == KIND_TREATMENT)
            & (part1["teacher_param_tag"].astype(str) == plan.TEACHER_PARAM_TAG)
            & (pd.to_numeric(part1["lora_r"], errors="coerce") == plan.RECIPE.lora_r)
            & np.isclose(pd.to_numeric(part1["epochs"], errors="coerce"), plan.EPOCHS)
            & ~part1["quarantined"].astype(str).str.lower().eq("true")
        ]
    for teacher, student, lr in plan.BRIDGE_CELLS:

        def select(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            hit = frame[
                (frame["teacher"].astype(str) == teacher)
                & (frame["student"].astype(str) == student)
                & np.isclose(pd.to_numeric(frame["lr"], errors="coerce"), lr)
                & (pd.to_numeric(frame["ftseed"], errors="coerce") == 0)
                & pd.to_numeric(frame["gseed"], errors="coerce").isin(plan.CORE_GSEEDS)
            ]
            return hit.drop_duplicates(subset=["ftseed", "gseed"])

        mine, theirs = select(live), select(part1)
        m_here, m_there = _mean_delta(mine), _mean_delta(theirs)
        diff = m_here - m_there
        rows.append(
            {
                "teacher": teacher,
                "student": student,
                "lr": lr,
                "n_xfam2": int(len(mine)),
                "n_part1": int(len(theirs)),
                "mean_xfam2": m_here,
                "mean_part1": m_there,
                "diff": diff,
                "within": bool(_finite(diff) and abs(diff) <= plan.BRIDGE_MAX_DIFF),
            }
        )
    return pd.DataFrame(rows, columns=list(BRIDGE_COLUMNS))


# ---- claims -------------------------------------------------------------------------


def _overall(statuses: Sequence[str]) -> str:
    decided = [s for s in statuses if s in (STATUS_HOLDS, STATUS_FAILS)]
    if not decided:
        return STATUS_INCOMPLETE if statuses else STATUS_VOID
    if all(s == STATUS_HOLDS for s in decided):
        return STATUS_HOLDS if len(decided) == len(statuses) else STATUS_PARTIAL
    if any(s == STATUS_HOLDS for s in decided):
        return STATUS_PARTIAL
    return STATUS_FAILS


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict(orient="records") if not frame.empty else []


def _direction(row: Mapping[str, Any]) -> str:
    if str(row["teacher"]) == str(row["student"]):
        return "self"
    a, b = row["log_n_teacher"], row["log_n_student"]
    if not (_finite(a) and _finite(b)):
        return "cross_family"
    return "small_to_big" if float(a) < float(b) else "big_to_small"


def claims(
    *,
    cells: pd.DataFrame,
    asymmetry: pd.DataFrame,
    asym_law: Mapping[str, Any],
    laws: pd.DataFrame,
    family: pd.DataFrame,
    bridge_rows: pd.DataFrame | None,
    ppl: pd.DataFrame,
) -> dict:
    """Y0-Y11 records, in `xfam2_plan.CLAIM_IDS` order."""
    frozen = (
        cells.iloc[0:0]
        if cells.empty
        else cells[
            [
                math.isclose(float(lr), plan.student_lr(str(s)))
                for lr, s in zip(cells["lr"], cells["student"], strict=True)
            ]
        ]
    )
    y0_cells = {
        f"{r['teacher']}->{r['student']}": {
            "status": r["status"],
            "direction": _direction(r),
            "mean": r["mean_delta"],
            "ci": [r["ci_low"], r["ci_high"]],
            "n": r["n"],
            "complete": r["complete"],
        }
        for _, r in frozen.iterrows()
    }
    by_direction: dict[str, dict[str, int]] = {}
    for record in y0_cells.values():
        tally = by_direction.setdefault(record["direction"], {})
        tally[record["status"]] = tally.get(record["status"], 0) + 1

    headline = family[family["contrast"] == "receiver_headline"]
    y1_up = laws[
        (laws["claim"] == "Y1")
        & (laws["law"] == "receiver_delta")
        & (laws["scope"] == "receivers_ge_teacher")
    ]
    rhos = pd.to_numeric(y1_up["rho"], errors="coerce").dropna()
    if headline.empty or not _finite(headline.iloc[0]["ci_low"]) or rhos.empty:
        y1_status = STATUS_INCOMPLETE
    else:
        y1_status = (
            STATUS_HOLDS
            if float(headline.iloc[0]["ci_low"]) > 0.0 and float(rhos.median()) > 0.0
            else STATUS_FAILS
        )

    pooled = laws[(laws["claim"] == "Y2") & (laws["scope"] == "pooled_teacher_bigger")]
    if pooled.empty or not _finite(pooled.iloc[0]["rho"]):
        y2_status = STATUS_INCOMPLETE
    else:
        row = pooled.iloc[0]
        y2_status = (
            STATUS_HOLDS if row["rho"] < 0 and row["p_value"] < 0.05 else STATUS_FAILS
        )

    y3_reading = asymmetry_reading(asymmetry, asym_law)
    y3_status = {
        "ratio_law": STATUS_HOLDS,
        "small_student_floor": STATUS_FAILS,
        "no_asymmetry": STATUS_FAILS,
        "mixed": STATUS_INCONCLUSIVE,
    }.get(y3_reading, STATUS_INCOMPLETE)

    y4 = laws[(laws["claim"] == "Y4") & (laws["scope"] == "teacher_bigger")]
    if y4.empty or not _finite(y4.iloc[0]["rho"]):
        y4_status = STATUS_INCOMPLETE
    else:
        y4_status = (
            STATUS_HOLDS
            if y4.iloc[0]["rho"] < 0 and y4.iloc[0]["p_value"] < 0.05
            else STATUS_FAILS
        )

    y6_reading = family_reading(family)
    y6_status = {
        "family_effect": STATUS_HOLDS,
        "no_family_effect": STATUS_FAILS,
    }.get(y6_reading, y6_reading)

    y8 = asymmetry[asymmetry["claim"] == "Y8"]
    y8_status = (
        _overall(
            [
                STATUS_HOLDS
                if bool(s)
                else (STATUS_FAILS if _finite(lo) else STATUS_INCOMPLETE)
                for s, lo in zip(y8["significant"], y8["ci_low"], strict=True)
            ]
        )
        if not y8.empty
        else STATUS_INCOMPLETE
    )

    if bridge_rows is None or bridge_rows.empty:
        y9_status = STATUS_VOID
    else:
        measured = bridge_rows[
            (bridge_rows["n_xfam2"] > 0) & (bridge_rows["n_part1"] > 0)
        ]
        if measured.empty:
            y9_status = STATUS_INCOMPLETE
        elif measured["within"].astype(bool).all():
            y9_status = (
                STATUS_HOLDS if len(measured) == len(bridge_rows) else STATUS_PARTIAL
            )
        else:
            y9_status = STATUS_FAILS

    good = ppl[~ppl["degenerate"].astype(bool)] if not ppl.empty else ppl
    retained = (
        float(
            np.average(
                good["retained_fraction"].astype(float), weights=good["n"].astype(float)
            )
        )
        if not good.empty and good["retained_fraction"].notna().any()
        else float("nan")
    )
    y10_status = (
        STATUS_VOID
        if not _finite(retained)
        else (STATUS_HOLDS if retained >= plan.RETAINED_FRACTION_MIN else STATUS_FAILS)
    )

    payload = {
        "Y0": {
            "status": _overall(
                [r["status"] for r in y0_cells.values() if r["status"] != STATUS_VOID]
            ),
            "min_delta": plan.MIN_DELTA,
            "by_direction": by_direction,
            "cells": y0_cells,
        },
        "Y1": {
            "status": y1_status,
            "headline": _records(headline),
            "per_emitter": _records(laws[laws["claim"] == "Y1"]),
        },
        "Y2": {"status": y2_status, "rows": _records(laws[laws["claim"] == "Y2"])},
        "Y3": {
            "status": y3_status,
            "reading": y3_reading,
            "law": dict(asym_law),
            "pairs": _records(asymmetry[asymmetry["claim"] == "Y3"]),
        },
        "Y4": {"status": y4_status, "rows": _records(laws[laws["claim"] == "Y4"])},
        "Y5": {
            "status": STATUS_DESCRIPTIVE,
            "rows": _records(laws[laws["claim"] == "Y5"]),
        },
        "Y6": {
            "status": y6_status,
            "reading": y6_reading,
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "rows": _records(family[family["claim"] == "Y6"]),
        },
        "Y7": {
            "status": STATUS_DESCRIPTIVE,
            "rows": _records(family[family["claim"] == "Y7"]),
        },
        "Y8": {"status": y8_status, "pairs": _records(y8)},
        "Y9": {
            "status": y9_status,
            "max_diff": plan.BRIDGE_MAX_DIFF,
            "rows": _records(bridge_rows) if bridge_rows is not None else [],
        },
        "Y10": {
            "status": y10_status,
            "retained_fraction": retained,
            "n_groups": int(len(good)),
            "n_degenerate": int(len(ppl) - len(good)) if not ppl.empty else 0,
        },
        "Y11": {
            "status": STATUS_DESCRIPTIVE,
            "rows": _records(laws[laws["claim"] == "Y11"]),
        },
    }
    if tuple(payload) != plan.CLAIM_IDS:
        raise ValueError(f"claims carry {tuple(payload)}, not {plan.CLAIM_IDS}")
    return payload
