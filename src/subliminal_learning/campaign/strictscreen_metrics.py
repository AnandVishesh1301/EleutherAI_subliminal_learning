"""Pure SL-claim math for the strictscreen campaign (plan §5.1, prereg §4).

Leaf frames come from `pipelines/strictscreen_results_pipeline.py`; nothing here
reads disk. Sources: arms ``a`` / ``b`` / ``c`` of the new run set and the breadth4
reference leaves (``breadth4`` treatments, ``breadth4_control`` controls) with the
same identities.

- ``tfs`` is `CONTROL_TFS` on control rows.
- Δp per treatment leaf = ``p_target`` − mean ``p_target`` of the matched controls
  (animal, student, gseed): arm B for arm A; breadth4 controls for arm C and breadth4.
- C0: arm C vs breadth4 over the 32 tfs0/gseed0 identities (breadth4 leaves with
  duplicate ``(step, batch_idx)`` rollout rows excluded); 95 % CI of the Δp ratio ⊂
  `C0_RATIO_BAND`.
- H2: R = mean Δp_strict / mean Δp_legacy over paired identities; stratified cluster
  bootstrap (cluster animal × pairing × tfs, strata animal); pass iff the lower bound
  ≥ `H2_LOWER_MIN`. C0 not passing → ``confounded`` plus the arm-A vs arm-C subset.
- H3: strict treatment masking rate rises Q0 → Q4 (bootstrap lower bound > 0) and the
  strict control |Q4 − Q0| ≤ `H3_CONTROL_MAX_ABS_CHANGE`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from subliminal_learning.analysis.stats import stratified_cluster_bootstrap_draws
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.gkd.filtering import SCREEN_RULES

SOURCE_A = plan.ARM_A
SOURCE_B = plan.ARM_B
SOURCE_C = plan.ARM_C
SOURCE_REF = "breadth4"
SOURCE_REF_CONTROL = "breadth4_control"
CONTROL_SOURCE: Mapping[str, str] = {
    SOURCE_A: SOURCE_B,
    SOURCE_C: SOURCE_REF_CONTROL,
    SOURCE_REF: SOURCE_REF_CONTROL,
}
CONTROL_TFS = -1
IDENTITY_COLUMNS: tuple[str, ...] = ("animal", "pair", "tfs", "gseed")
CONTROL_JOIN_COLUMNS: tuple[str, ...] = ("animal", "student", "gseed")
METRICS: Mapping[str, str] = {"p_target": "delta_p", "fc_share": "delta_fc"}
LEAF_COLUMNS: tuple[str, ...] = (
    "source",
    "kind",
    "screen",
    "animal",
    "pair",
    "group",
    "student",
    "teacher",
    "tfs",
    "gseed",
    "leaf_dir",
    "trained",
    "evaluated",
    "p_target",
    "fc_share",
    "ppl",
    "base_ppl",
    "ppl_ratio",
    "flagged_rate",
    "n_rollout_rows",
    "n_duplicate_rollout_rows",
)
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_INDETERMINATE = "indeterminate"
STATUS_CONFOUNDED = "confounded"


# ---- rollouts ---------------------------------------------------------------------------


def dedupe_keep_last(rows: Iterable[Mapping]) -> list[Mapping]:
    """Rollout rows with one row per ``(step, batch_idx)``, the last one written.

    A retried cell appends its rerun rows after the crashed attempt's rows, so the
    last row is the one that entered the finished run. Order is first appearance.
    """
    last: dict[tuple[int, int], Mapping] = {}
    for row in rows:
        last[(int(row["step"]), int(row["batch_idx"]))] = row
    return list(last.values())


def step_quintile(step: int, n_steps: int, n_quintiles: int = plan.N_QUINTILES) -> int:
    """Equal-width step bin: ``floor(step · n_quintiles / n_steps)``.

    Raises:
        ValueError: `step` outside ``[0, n_steps)``.
    """
    if not 0 <= step < n_steps:
        raise ValueError(f"step {step} outside [0, {n_steps})")
    return step * n_quintiles // n_steps


def quintile_mask_rates(
    rows: Iterable[Mapping],
    *,
    n_steps: int = plan.STEPS_PER_EPOCH,
    n_quintiles: int = plan.N_QUINTILES,
) -> list[dict]:
    """Masked-row rate per step quintile of already-deduplicated rollout rows.

    Returns:
        One ``{"quintile", "n_rows", "n_flagged", "rate"}`` per quintile (``rate``
        NaN for an empty quintile).
    """
    n_rows = [0] * n_quintiles
    n_flagged = [0] * n_quintiles
    for row in rows:
        q = step_quintile(int(row["step"]), n_steps, n_quintiles)
        n_rows[q] += 1
        n_flagged[q] += bool(row["flagged"])
    return [
        {
            "quintile": q,
            "n_rows": n_rows[q],
            "n_flagged": n_flagged[q],
            "rate": n_flagged[q] / n_rows[q] if n_rows[q] else math.nan,
        }
        for q in range(n_quintiles)
    ]


# ---- screen audit ------------------------------------------------------------------------


def screen_audit_errors(
    *,
    screen: str,
    expected_gkd_tag: str,
    manifest: Mapping,
    summary: Mapping | None,
    staged_vocab_sha256: str | None,
) -> list[str]:
    """Contamination findings for one finished leaf (empty = clean).

    A strict leaf must carry ``rollout_screen == "strict"`` and the staged vocabulary
    sha in its identity, manifest record and rollout summary; a legacy leaf must
    carry none of those keys.
    """
    errors: list[str] = []
    ident = manifest.get("run_identity") or {}
    hp = ident.get("gkd_hyperparameters") or {}
    if ident.get("gkd_tag") != expected_gkd_tag:
        errors.append(f"gkd_tag {ident.get('gkd_tag')!r} != {expected_gkd_tag!r}")
    if screen == plan.SCREEN_STRICT:
        if not staged_vocab_sha256:
            errors.append("no staged strict vocab sha256")
        if hp.get("rollout_screen") != plan.SCREEN_STRICT:
            errors.append(f"identity rollout_screen {hp.get('rollout_screen')!r}")
        if ident.get("rollout_screen_vocab_sha256") != staged_vocab_sha256:
            errors.append("identity rollout_screen_vocab_sha256 != staged")
        if (manifest.get("rollout_screen") or {}).get(
            "vocab_sha256"
        ) != staged_vocab_sha256:
            errors.append("manifest rollout_screen.vocab_sha256 != staged")
        if summary is None:
            errors.append("rollouts_summary.json missing")
        else:
            if summary.get("rollout_screen") != plan.SCREEN_STRICT:
                errors.append(
                    f"summary rollout_screen {summary.get('rollout_screen')!r}"
                )
            if summary.get("screen_vocab_sha256") != staged_vocab_sha256:
                errors.append("summary screen_vocab_sha256 != staged")
        return errors
    present = [
        name
        for name, found in (
            ("identity gkd_hyperparameters.rollout_screen", "rollout_screen" in hp),
            (
                "identity rollout_screen_vocab_sha256",
                "rollout_screen_vocab_sha256" in ident,
            ),
            ("manifest rollout_screen", "rollout_screen" in manifest),
            (
                "summary rollout_screen",
                summary is not None and "rollout_screen" in summary,
            ),
            (
                "summary screen_vocab_sha256",
                summary is not None and "screen_vocab_sha256" in summary,
            ),
        )
        if found
    ]
    errors.extend(f"legacy leaf carries {name}" for name in present)
    return errors


def screen_stats_row(summary: Mapping | None) -> dict:
    """Flag rate, first-match and fired counts per rule, all-flagged steps of a leaf."""
    doc = summary or {}
    first = doc.get("matched_rule_counts") or {}
    fired = doc.get("rule_fired_counts") or {}
    row = {
        "n_rows_screened": doc.get("n_rows_screened"),
        "n_flagged": doc.get("n_flagged"),
        "flagged_rate": doc.get("flagged_rate"),
        "n_all_flagged_steps": doc.get("n_all_flagged_steps"),
        "screen_vocab_sha256": doc.get("screen_vocab_sha256"),
    }
    for rule in SCREEN_RULES:
        row[f"first_{rule}"] = int(first.get(rule, 0))
        row[f"fired_{rule}"] = int(fired.get(rule, 0))
    return row


def coverage_by_arm(
    cells: Sequence[plan.Cell],
    trained: Mapping[str, bool],
    evaluated: Mapping[str, bool],
) -> dict[str, dict]:
    """Per-arm coverage with missing cells and the `plan.SHORTFALL_TOLERANCE` verdict.

    Args:
        cells: Planned cells.
        trained: Cell key → leaf passes ``cell_is_done``.
        evaluated: Cell key → its config dir holds the full eval triad.

    Returns:
        ``{arm: {"planned", "trained", "evaluated", "n_missing", "missing",
        "tolerance", "within_tolerance"}}``; a missing cell's ``reason`` is
        ``not_trained`` or ``not_evaluated``.
    """
    out: dict[str, dict] = {}
    for arm in plan.ARMS:
        arm_cells = [c for c in cells if c.arm == arm]
        missing = [
            {
                "cell": c.key,
                "reason": "not_trained" if not trained[c.key] else "not_evaluated",
            }
            for c in arm_cells
            if not (trained[c.key] and evaluated[c.key])
        ]
        tolerance = plan.SHORTFALL_TOLERANCE[arm]
        out[arm] = {
            "planned": len(arm_cells),
            "trained": sum(trained[c.key] for c in arm_cells),
            "evaluated": sum(trained[c.key] and evaluated[c.key] for c in arm_cells),
            "n_missing": len(missing),
            "missing": missing,
            "tolerance": tolerance,
            "within_tolerance": tolerance is None or len(missing) <= tolerance,
        }
    return out


# ---- bootstrap -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RatioEstimate:
    """``Σ num / Σ den`` with a stratified cluster-bootstrap percentile CI.

    Attributes:
        estimate: Statistic on the observed units.
        ci_low: Lower percentile bound.
        ci_high: Upper percentile bound.
        n_units: Units with finite `num` and `den`.
        n_clusters: Distinct clusters among those units.
        n_valid: Replicates with a finite statistic.
        n_boot: Replicates requested.
        confidence: Two-sided interval level.
    """

    estimate: float
    ci_low: float
    ci_high: float
    n_units: int
    n_clusters: int
    n_valid: int
    n_boot: int
    confidence: float


def stratified_cluster_bootstrap(
    num: Sequence[float],
    den: Sequence[float],
    cluster: Sequence[object],
    stratum: Sequence[object],
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = plan.CONFIDENCE,
) -> RatioEstimate:
    """Percentile CI of ``Σ num / Σ den`` resampling whole clusters within strata.

    Replicates come from `analysis.stats.stratified_cluster_bootstrap_draws`. Pass
    ``den = 1`` per unit for a mean; units with a non-finite `num` or `den` are dropped.

    Args:
        num: Per-unit numerator.
        den: Per-unit denominator.
        cluster: Per-unit cluster label (unique across strata).
        stratum: Per-unit stratum label.
        n_boot: Replicates.
        seed: Seeds ``numpy.random.default_rng``.
        confidence: Two-sided percentile level.

    Raises:
        ValueError: Mismatched lengths or a bad `confidence`.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    num_a = np.asarray(num, dtype=float)
    den_a = np.asarray(den, dtype=float)
    clusters = np.asarray([str(c) for c in cluster])
    strata = np.asarray([str(s) for s in stratum])
    if not num_a.shape == den_a.shape == clusters.shape == strata.shape:
        raise ValueError("num, den, cluster and stratum must have matching lengths")
    ok = np.isfinite(num_a) & np.isfinite(den_a)
    num_a, den_a, clusters, strata = num_a[ok], den_a[ok], clusters[ok], strata[ok]
    n_clusters = len(set(clusters.tolist()))
    nan = math.nan
    if num_a.size == 0 or den_a.sum() == 0.0:
        return RatioEstimate(
            nan, nan, nan, int(num_a.size), n_clusters, 0, n_boot, confidence
        )

    def ratio(idx: np.ndarray) -> float:
        total = den_a[idx].sum()
        return float(num_a[idx].sum() / total) if total != 0.0 else nan

    observed, draws = stratified_cluster_bootstrap_draws(
        ratio, strata, clusters, n_boot=n_boot, seed=seed
    )
    ratios = draws[:, 0][np.isfinite(draws[:, 0])]
    if ratios.size == 0:
        return RatioEstimate(
            float(observed[0]), nan, nan, int(num_a.size), n_clusters, 0, n_boot,
            confidence,
        )
    tail = 100.0 * (1.0 - confidence) / 2.0
    return RatioEstimate(
        estimate=float(observed[0]),
        ci_low=float(np.percentile(ratios, tail)),
        ci_high=float(np.percentile(ratios, 100.0 - tail)),
        n_units=int(num_a.size),
        n_clusters=n_clusters,
        n_valid=int(ratios.size),
        n_boot=n_boot,
        confidence=confidence,
    )


# ---- deltas + pairing ------------------------------------------------------------------------


def attach_control_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    """Add ``control_{metric}`` and ``delta_*`` to treatment rows of a leaf frame.

    Controls are matched on (animal, student, gseed) in `CONTROL_SOURCE`; a missing
    or unevaluated control leaves the delta NaN.
    """
    out = leaves.copy()
    keys = list(CONTROL_JOIN_COLUMNS)
    for metric, delta in METRICS.items():
        out[f"control_{metric}"] = math.nan
        out[delta] = math.nan
    for source, control_source in CONTROL_SOURCE.items():
        controls = out[out["source"] == control_source]
        if controls.empty:
            continue
        treat = out["source"] == source
        for metric, delta in METRICS.items():
            means = controls.groupby(keys)[metric].mean()
            lookup = [
                means.get(tuple(row), math.nan)
                for row in out.loc[treat, keys].itertuples(index=False)
            ]
            out.loc[treat, f"control_{metric}"] = lookup
            out.loc[treat, delta] = (
                out.loc[treat, metric] - out.loc[treat, f"control_{metric}"]
            )
    return out


def identity_pairs(
    leaves: pd.DataFrame,
    left: str,
    right: str,
    *,
    tfs: int | None = None,
    gseed: int | None = None,
) -> pd.DataFrame:
    """Treatment identities present in both sources, one row each.

    Columns: identity columns, ``group``, ``cluster`` (animal|pair|tfs), ``stratum``
    (animal), ``{delta}_left`` / ``{delta}_right`` per metric, and
    ``right_duplicates`` (duplicate rollout rows of the right leaf).
    """
    ident = list(IDENTITY_COLUMNS)
    cols = [*ident, "group", *METRICS.values(), "n_duplicate_rollout_rows"]
    frames = []
    for source in (left, right):
        part = leaves[leaves["source"] == source]
        if tfs is not None:
            part = part[part["tfs"] == tfs]
        if gseed is not None:
            part = part[part["gseed"] == gseed]
        frames.append(part[cols])
    merged = frames[0].merge(
        frames[1].drop(columns=["group"]),
        on=ident,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    merged = merged.rename(
        columns={"n_duplicate_rollout_rows_right": "right_duplicates"}
    )
    merged = merged.drop(columns=["n_duplicate_rollout_rows_left"])
    merged["cluster"] = [
        f"{a}|{p}|{int(t)}"
        for a, p, t in zip(merged["animal"], merged["pair"], merged["tfs"], strict=True)
    ]
    merged["stratum"] = merged["animal"]
    return merged.reset_index(drop=True)


def _ratio(
    paired: pd.DataFrame, delta: str, *, n_boot: int, seed: int
) -> RatioEstimate:
    return stratified_cluster_bootstrap(
        paired[f"{delta}_left"].to_numpy(dtype=float),
        paired[f"{delta}_right"].to_numpy(dtype=float),
        paired["cluster"].tolist(),
        paired["stratum"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )


def ratio_rows(
    paired: pd.DataFrame, *, comparison: str, n_boot: int, seed: int
) -> pd.DataFrame:
    """Ratio estimates overall, per animal and per pairing group, per metric."""
    rows: list[dict] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("all", "all", paired)]
    scopes += [("animal", str(a), g) for a, g in paired.groupby("animal", sort=True)]
    scopes += [("group", str(k), g) for k, g in paired.groupby("group", sort=True)]
    for scope, level, frame in scopes:
        for metric, delta in METRICS.items():
            est = _ratio(frame, delta, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "level": level,
                    "metric": metric,
                }
                | asdict(est)
            )
    return pd.DataFrame(rows)


def _within(est: RatioEstimate, low: float, high: float) -> str:
    if not (math.isfinite(est.ci_low) and math.isfinite(est.ci_high)):
        return STATUS_INDETERMINATE
    return STATUS_PASS if low <= est.ci_low and est.ci_high <= high else STATUS_FAIL


def _at_least(est: RatioEstimate, low: float) -> str:
    if not math.isfinite(est.ci_low):
        return STATUS_INDETERMINATE
    return STATUS_PASS if est.ci_low >= low else STATUS_FAIL


def c0_claim(paired_c_ref: pd.DataFrame, *, n_boot: int, seed: int) -> dict:
    """C0: arm C vs breadth4 Δp ratio CI inside `C0_RATIO_BAND`.

    Args:
        paired_c_ref: `identity_pairs(leaves, "c", "breadth4", tfs=0, gseed=0)`.
    """
    keep = paired_c_ref[paired_c_ref["right_duplicates"].fillna(0) == 0]
    est = _ratio(keep, "delta_p", n_boot=n_boot, seed=seed)
    low, high = plan.C0_RATIO_BAND
    return {
        "status": _within(est, low, high),
        "band": [low, high],
        "n_identities": int(len(paired_c_ref)),
        "n_excluded_duplicate_rollouts": int(len(paired_c_ref) - len(keep)),
        "delta_p_ratio": asdict(est),
    }


def h2_claim(
    paired_a_ref: pd.DataFrame,
    paired_a_c: pd.DataFrame,
    c0: Mapping,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """H2: strict/legacy Δp ratio lower bound ≥ `H2_LOWER_MIN`.

    Args:
        paired_a_ref: `identity_pairs(leaves, "a", "breadth4")`.
        paired_a_c: `identity_pairs(leaves, "a", "c", tfs=0, gseed=0)`.
        c0: `c0_claim` result; anything but ``pass`` makes H2 ``confounded``.
    """
    primary = _ratio(paired_a_ref, "delta_p", n_boot=n_boot, seed=seed)
    fc = _ratio(paired_a_ref, "delta_fc", n_boot=n_boot, seed=seed)
    verdict = _at_least(primary, plan.H2_LOWER_MIN)
    out: dict = {
        "status": verdict if c0.get("status") == STATUS_PASS else STATUS_CONFOUNDED,
        "lower_min": plan.H2_LOWER_MIN,
        "reference": SOURCE_REF,
        "n_identities": int(len(paired_a_ref)),
        "delta_p_ratio": asdict(primary),
        "delta_p_verdict": verdict,
        "forced_choice_ratio": asdict(fc),
    }
    if out["status"] == STATUS_CONFOUNDED:
        subset = _ratio(paired_a_c, "delta_p", n_boot=n_boot, seed=seed)
        out["subset_vs_arm_c"] = {
            "status": _at_least(subset, plan.H2_LOWER_MIN),
            "n_identities": int(len(paired_a_c)),
            "delta_p_ratio": asdict(subset),
        }
    return out


# ---- H3 ------------------------------------------------------------------------------------------


def quintile_change(quintiles: pd.DataFrame) -> pd.DataFrame:
    """Per-leaf ``rate(Q_last) − rate(Q0)`` with cluster / stratum labels."""
    last = plan.N_QUINTILES - 1
    keys = ["source", "animal", "pair", "tfs", "gseed"]
    pivot = quintiles.pivot_table(
        index=keys, columns="quintile", values="rate", aggfunc="first", dropna=False
    ).reset_index()
    pivot["change"] = pivot.get(last, math.nan) - pivot.get(0, math.nan)
    pivot["cluster"] = [
        f"{a}|{p}|{int(t)}"
        for a, p, t in zip(pivot["animal"], pivot["pair"], pivot["tfs"], strict=True)
    ]
    pivot["stratum"] = pivot["animal"]
    return pivot[[*keys, "change", "cluster", "stratum"]]


def _mean_change(frame: pd.DataFrame, *, n_boot: int, seed: int) -> RatioEstimate:
    return stratified_cluster_bootstrap(
        frame["change"].to_numpy(dtype=float),
        np.ones(len(frame)),
        frame["cluster"].tolist(),
        frame["stratum"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )


def h3_claim(quintiles: pd.DataFrame, *, n_boot: int, seed: int) -> dict:
    """H3: strict treatment Q0→Q4 rise (CI > 0) with a flat strict control."""
    if quintiles.empty:
        return {"status": STATUS_INDETERMINATE, "reason": "no rollout quintiles"}
    change = quintile_change(quintiles)
    treat = _mean_change(change[change["source"] == SOURCE_A], n_boot=n_boot, seed=seed)
    ctrl = _mean_change(change[change["source"] == SOURCE_B], n_boot=n_boot, seed=seed)
    if not (math.isfinite(treat.ci_low) and math.isfinite(ctrl.estimate)):
        status = STATUS_INDETERMINATE
    else:
        rises = treat.ci_low > 0.0
        flat = abs(ctrl.estimate) <= plan.H3_CONTROL_MAX_ABS_CHANGE
        status = STATUS_PASS if rises and flat else STATUS_FAIL
    return {
        "status": status,
        "control_max_abs_change": plan.H3_CONTROL_MAX_ABS_CHANGE,
        "treatment_change": asdict(treat),
        "control_change": asdict(ctrl),
    }


def quintile_summary(quintiles: pd.DataFrame) -> pd.DataFrame:
    """Pooled and leaf-mean masking rate per (source, animal | all, quintile)."""
    rows: list[dict] = []
    for (source, animal), part in _with_all_animals(quintiles).groupby(
        ["source", "animal"], sort=True
    ):
        for q, g in part.groupby("quintile", sort=True):
            n_rows = int(g["n_rows"].sum())
            rows.append(
                {
                    "source": source,
                    "animal": animal,
                    "quintile": int(q),
                    "n_leaves": int(len(g)),
                    "n_rows": n_rows,
                    "pooled_rate": g["n_flagged"].sum() / n_rows
                    if n_rows
                    else math.nan,
                    "leaf_mean_rate": float(g["rate"].mean()),
                }
            )
    return pd.DataFrame(rows)


# ---- reported tables ----------------------------------------------------------------------------


def _with_all_animals(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame, frame.assign(animal="all")], ignore_index=True)


def removal_by_rule(summaries: pd.DataFrame) -> pd.DataFrame:
    """Cloud-style "screen removed X %" table per (source, animal | all, rule).

    Args:
        summaries: One row per leaf with ``source``, ``screen``, ``animal``,
            ``n_rows_screened``, ``n_flagged`` and ``first_{rule}`` /
            ``fired_{rule}`` columns (`screen_stats_row`).

    Returns:
        ``rule`` is each strict rule plus ``any``; a legacy source reports its single
        rule as ``legacy`` (first = fired = flagged).
    """
    rows: list[dict] = []
    for (source, animal), part in _with_all_animals(summaries).groupby(
        ["source", "animal"], sort=True
    ):
        n_rows = int(part["n_rows_screened"].fillna(0).sum())
        strict = (part["screen"] == plan.SCREEN_STRICT).all()
        n_flagged = int(part["n_flagged"].fillna(0).sum())
        counts: dict[str, tuple[int, int]] = {"any": (n_flagged, n_flagged)}
        if strict:
            for rule in SCREEN_RULES:
                counts[rule] = (
                    int(part[f"first_{rule}"].sum()),
                    int(part[f"fired_{rule}"].sum()),
                )
        else:
            counts[plan.SCREEN_LEGACY] = (n_flagged, n_flagged)
        for rule, (first, fired) in counts.items():
            rows.append(
                {
                    "source": source,
                    "animal": animal,
                    "rule": rule,
                    "n_leaves": int(len(part)),
                    "n_rows": n_rows,
                    "n_first_match": first,
                    "n_fired": fired,
                    "frac_first_match": first / n_rows if n_rows else math.nan,
                    "frac_fired": fired / n_rows if n_rows else math.nan,
                }
            )
    return pd.DataFrame(rows)


def trajectory_summary(logs: pd.DataFrame) -> pd.DataFrame:
    """Leaf-mean of each logged metric per (source, animal | all, metric, step).

    Args:
        logs: Long frame with ``source``, ``animal``, ``leaf_dir``, ``step``,
            ``metric``, ``value``.
    """
    if logs.empty:
        return pd.DataFrame(
            columns=["source", "animal", "metric", "step", "mean", "n_leaves"]
        )
    grouped = _with_all_animals(logs).groupby(
        ["source", "animal", "metric", "step"], sort=True
    )
    out = grouped["value"].agg(["mean", "count"]).reset_index()
    return out.rename(columns={"count": "n_leaves"})


def arm_summary(leaves: pd.DataFrame) -> pd.DataFrame:
    """Per (source, animal | all): coverage, mean metrics, deltas, flag rate."""
    rows: list[dict] = []
    for (source, animal), part in _with_all_animals(leaves).groupby(
        ["source", "animal"], sort=True
    ):
        rows.append(
            {
                "source": source,
                "animal": animal,
                "n_leaves": int(len(part)),
                "n_trained": int(part["trained"].sum()),
                "n_evaluated": int(part["evaluated"].sum()),
                **{
                    f"mean_{col}": float(part[col].mean())
                    for col in (
                        "p_target",
                        "fc_share",
                        "ppl_ratio",
                        "flagged_rate",
                        *METRICS.values(),
                    )
                },
            }
        )
    return pd.DataFrame(rows)
