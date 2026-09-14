"""Pre-registered judge claims of the strictscreen audit (H1, H4, S1, S2).

Pure: pandas frames in, JSON-serialisable verdict dicts out. Sources and
estimators (``.claude/plans/strictscreen_rollout_judge_audit.plan.md`` §5.2):

- Clean-row leakage is a Horvitz-Thompson total per leaf x step quintile,
  ``sum_lq N_lq * y_lq / n_lq / sum_lq N_lq``, so a quintile's sample share
  never stands in for its population share.
- Uncertainty is a leaf-cluster bootstrap within arm x animal strata
  (`analysis.stats.stratified_cluster_bootstrap_draws`); equivalence is the
  90 % percentile interval inside ``+-bound`` (TOST at alpha 0.05).
- Hard-positive recall is `judge_metrics.reweighted_rate` (Wilson on the Kish
  effective sample size) over ``rule|form`` strata.
- Decoy false positives are a Hajek mean with the key ``sample_weight``.
- Pool re-judge agreement is Cohen's kappa plus a paired yes-rate difference
  bootstrapped from the multinomial of discordant pairs.

Parse failures are excluded from every rate (treated as missing at random);
their rate is claim H4c.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from subliminal_learning.analysis.stats import (
    bootstrap_equivalence,
    stratified_cluster_bootstrap_draws,
    summarize_draws,
)
from subliminal_learning.gkd.judge_metrics import (
    cohens_kappa,
    difference_of_rates,
    parse_yes_no,
    reweighted_rate,
)

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
INDETERMINATE = "indeterminate"
PENDING_LABELS = "pending_labels"
REPRODUCES = "reproduces"
NOT_REPRODUCED = "not_reproduced"
CONFOUNDED = "confounded"
PASS = "pass"
FAIL = "fail"

CELL_COLUMNS: tuple[str, ...] = (
    "run_dir",
    "arm",
    "animal",
    "step_quintile",
    "n_pop",
    "n_sampled",
    "n_yes",
)


@dataclass(frozen=True)
class ClaimsConfig:
    """Thresholds and bootstrap settings of the judge claims.

    Attributes:
        bound: Equivalence half-width as a proportion (0.25 pp = 0.0025).
        equivalence_confidence: Two-sided interval level for equivalence.
        n_boot: Bootstrap replicates.
        seed: Bootstrap seed.
        h1_max_treatment_rate: H1b ceiling on the treatment residual.
        h1_treatment_percentile: H1b bootstrap percentile.
        h4_recall_rule: Rule whose flagged rows are hard positives.
        h4_min_recall: H4b lower-bound threshold.
        h4_recall_confidence: H4b interval level.
        h4_max_parse_failure: H4c ceiling as a proportion.
        s2_min_kappa: S2 kappa threshold.
        treatment_arm: Strict treatment arm label.
        control_arm: Strict control arm label.
        rerun_arm: Legacy rerun arm label (S1).
        n_step_strata: Step quintiles.
    """

    bound: float
    equivalence_confidence: float
    n_boot: int
    seed: int
    h1_max_treatment_rate: float
    h1_treatment_percentile: float
    h4_recall_rule: str
    h4_min_recall: float
    h4_recall_confidence: float
    h4_max_parse_failure: float
    s2_min_kappa: float
    treatment_arm: str
    control_arm: str
    rerun_arm: str
    n_step_strata: int


def parse_claims_config(
    raw: Mapping[str, object], *, n_step_strata: int
) -> ClaimsConfig:
    """Build `ClaimsConfig` from the experiment YAML ``claims:`` block.

    Percent-valued keys end in ``_pct`` / ``_pp`` and are divided by 100.

    Raises:
        ValueError: A key is missing.
    """
    try:
        return ClaimsConfig(
            bound=float(raw["bound_pp"]) / 100.0,
            equivalence_confidence=float(raw["equivalence_confidence"]),
            n_boot=int(raw["n_boot"]),
            seed=int(raw["bootstrap_seed"]),
            h1_max_treatment_rate=float(raw["h1b_max_rate_pct"]) / 100.0,
            h1_treatment_percentile=float(raw["h1b_percentile"]),
            h4_recall_rule=str(raw["h4b_rule"]),
            h4_min_recall=float(raw["h4b_min_recall"]),
            h4_recall_confidence=float(raw["h4b_confidence"]),
            h4_max_parse_failure=float(raw["h4c_max_parse_failure_pct"]) / 100.0,
            s2_min_kappa=float(raw["s2_min_kappa"]),
            treatment_arm=str(raw["treatment_arm"]),
            control_arm=str(raw["control_arm"]),
            rerun_arm=str(raw["rerun_arm"]),
            n_step_strata=int(n_step_strata),
        )
    except KeyError as err:
        raise ValueError(f"claims block missing {err}") from err


def leaf_quintile_cells(
    clean: pd.DataFrame,
    clean_pop_by_leaf: Mapping[str, list[int]],
    arm_of: Mapping[str, str],
    animal_of: Mapping[str, str],
) -> pd.DataFrame:
    """One row per leaf x quintile with population, valid sample and yes count.

    Args:
        clean: Judged clean rows with ``run_dir``, ``step_quintile``,
            ``judge_valid``, ``judge_yes``.
        clean_pop_by_leaf: Plan ``clean_pop_by_quintile`` per ``run_dir``.
        arm_of: Arm label per ``run_dir``.
        animal_of: Target animal per ``run_dir``.

    Returns:
        `CELL_COLUMNS` for every leaf quintile with ``n_pop > 0``.
    """
    counts: dict[tuple[str, int], list[int]] = {}
    if not clean.empty:
        valid = clean.loc[clean["judge_valid"].astype(bool)]
        for (run_dir, q), group in valid.groupby(["run_dir", "step_quintile"]):
            counts[(str(run_dir), int(q))] = [
                len(group),
                int(group["judge_yes"].eq(True).sum()),
            ]
    rows = []
    for run_dir, pops in sorted(clean_pop_by_leaf.items()):
        for q, n_pop in enumerate(pops):
            if int(n_pop) <= 0:
                continue
            n_s, n_y = counts.get((run_dir, q), [0, 0])
            rows.append(
                {
                    "run_dir": run_dir,
                    "arm": arm_of[run_dir],
                    "animal": animal_of[run_dir],
                    "step_quintile": q,
                    "n_pop": int(n_pop),
                    "n_sampled": n_s,
                    "n_yes": n_y,
                }
            )
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


def ht_rate(cells: pd.DataFrame) -> float:
    """Horvitz-Thompson clean leakage over `cells` with a valid sample."""
    used = cells.loc[cells["n_sampled"] > 0]
    den = float(used["n_pop"].sum())
    if den <= 0:
        return float("nan")
    num = float((used["n_pop"] * used["n_yes"] / used["n_sampled"]).sum())
    return num / den


def leaf_rates(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-leaf quintile-reweighted rate with population and sample totals."""
    rows = []
    for (run_dir, arm, animal), group in cells.groupby(["run_dir", "arm", "animal"]):
        rows.append(
            {
                "run_dir": run_dir,
                "arm": arm,
                "animal": animal,
                "n_pop": int(group["n_pop"].sum()),
                "n_sampled": int(group["n_sampled"].sum()),
                "n_yes": int(group["n_yes"].sum()),
                "rate": ht_rate(group),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["run_dir", "arm", "animal", "n_pop", "n_sampled", "n_yes", "rate"],
    )


def h1_filter_efficacy(cells: pd.DataFrame, cfg: ClaimsConfig) -> dict:
    """H1a-c: strict treatment minus strict control clean residual.

    Args:
        cells: `leaf_quintile_cells` output.
        cfg: Claims config.

    Returns:
        ``{"verdict", "h1a", "h1b", "h1c", "arms", "step_profile",
        "sensitivity"}``.
    """
    arms = (cfg.treatment_arm, cfg.control_arm)
    sub = cells.loc[cells["arm"].isin(arms) & (cells["n_sampled"] > 0)].reset_index(
        drop=True
    )
    if sub.empty or set(sub["arm"]) != set(arms):
        return {"verdict": INDETERMINATE, "reason": "treatment or control has no cells"}
    n_strata = cfg.n_step_strata
    arm_code = (sub["arm"] == cfg.control_arm).to_numpy(dtype=np.int64)
    quintile = sub["step_quintile"].to_numpy(dtype=np.int64)
    num = (sub["n_pop"] * sub["n_yes"] / sub["n_sampled"]).to_numpy(dtype=np.float64)
    den = sub["n_pop"].to_numpy(dtype=np.float64)
    x = np.arange(n_strata, dtype=np.float64) - (n_strata - 1) / 2.0

    def statistic(idx: np.ndarray) -> np.ndarray:
        codes = arm_code[idx] * n_strata + quintile[idx]
        nums = np.bincount(codes, weights=num[idx], minlength=2 * n_strata)
        dens = np.bincount(codes, weights=den[idx], minlength=2 * n_strata)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_t = nums[:n_strata].sum() / dens[:n_strata].sum()
            rate_c = nums[n_strata:].sum() / dens[n_strata:].sum()
            delta_q = (
                nums[:n_strata] / dens[:n_strata] - nums[n_strata:] / dens[n_strata:]
            )
        slope = float(x @ (delta_q - delta_q.mean()) / (x @ x))
        return np.array(
            [rate_t, rate_c, rate_t - rate_c, *delta_q, (n_strata - 1) * slope]
        )

    observed, draws = stratified_cluster_bootstrap_draws(
        statistic,
        (sub["arm"] + "|" + sub["animal"]).to_numpy(),
        sub["run_dir"].to_numpy(),
        n_boot=cfg.n_boot,
        seed=cfg.seed,
    )
    conf = cfg.equivalence_confidence
    delta = bootstrap_equivalence(
        summarize_draws(observed[2], draws[:, 2], confidence=conf),
        bound_low=-cfg.bound,
        bound_high=cfg.bound,
    )
    finite_t = draws[:, 0][np.isfinite(draws[:, 0])]
    upper = float(np.percentile(finite_t, cfg.h1_treatment_percentile))
    change = bootstrap_equivalence(
        summarize_draws(observed[-1], draws[:, -1], confidence=conf),
        bound_low=-cfg.bound,
        bound_high=cfg.bound,
    )
    profile = []
    for q in range(n_strata):
        column = draws[:, 3 + q]
        if np.isfinite(observed[3 + q]) and np.isfinite(column).any():
            est = summarize_draws(observed[3 + q], column, confidence=conf)
            profile.append({"step_quintile": q, **asdict(est)})
        else:
            profile.append({"step_quintile": q, "estimate": None})
    h1b_pass = upper < cfg.h1_max_treatment_rate
    verdict = (
        SUPPORTED
        if delta.equivalent and h1b_pass and change.equivalent
        else NOT_SUPPORTED
    )
    return {
        "verdict": verdict,
        "h1a": _equivalence_json(delta),
        "h1b": {
            "treatment_rate": float(observed[0]),
            "percentile": cfg.h1_treatment_percentile,
            "bootstrap_upper": upper,
            "threshold": cfg.h1_max_treatment_rate,
            "pass": h1b_pass,
        },
        "h1c": _equivalence_json(change),
        "arms": {
            cfg.treatment_arm: _arm_summary(
                sub, cfg.treatment_arm, observed[0], draws[:, 0]
            ),
            cfg.control_arm: _arm_summary(
                sub, cfg.control_arm, observed[1], draws[:, 1]
            ),
        },
        "step_profile": profile,
        "sensitivity": _h1_sensitivity(sub, cfg),
    }


def h4a_decoy_false_positive(decoy: pd.DataFrame, cfg: ClaimsConfig) -> dict:
    """H4a: Hajek decoy false-positive rate on clean-origin rows, T minus C.

    Args:
        decoy: Valid clean-origin decoy rows with ``run_dir``, ``arm``,
            ``animal``, ``sample_weight``, ``judge_yes``.
        cfg: Claims config.

    Returns:
        ``{"verdict", "delta", "arms"}``.
    """
    arms = (cfg.treatment_arm, cfg.control_arm)
    sub = decoy.loc[decoy["arm"].isin(arms)]
    if sub.empty or set(sub["arm"]) != set(arms):
        return {
            "verdict": INDETERMINATE,
            "reason": "treatment or control has no decoys",
        }
    weight = sub["sample_weight"].astype(float)
    leaves = (
        sub.assign(_num=weight * sub["judge_yes"].eq(True).astype(float), _den=weight)
        .groupby(["run_dir", "arm", "animal"], as_index=False)[["_num", "_den"]]
        .sum()
    )
    arm_code = (leaves["arm"] == cfg.control_arm).to_numpy(dtype=np.int64)
    num = leaves["_num"].to_numpy(dtype=np.float64)
    den = leaves["_den"].to_numpy(dtype=np.float64)

    def statistic(idx: np.ndarray) -> np.ndarray:
        nums = np.bincount(arm_code[idx], weights=num[idx], minlength=2)
        dens = np.bincount(arm_code[idx], weights=den[idx], minlength=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = nums / dens
        return np.array([rates[0], rates[1], rates[0] - rates[1]])

    observed, draws = stratified_cluster_bootstrap_draws(
        statistic,
        (leaves["arm"] + "|" + leaves["animal"]).to_numpy(),
        leaves["run_dir"].to_numpy(),
        n_boot=cfg.n_boot,
        seed=cfg.seed,
    )
    delta = bootstrap_equivalence(
        summarize_draws(
            observed[2], draws[:, 2], confidence=cfg.equivalence_confidence
        ),
        bound_low=-cfg.bound,
        bound_high=cfg.bound,
    )
    return {
        "verdict": SUPPORTED if delta.equivalent else NOT_SUPPORTED,
        "delta": _equivalence_json(delta),
        "arms": {
            arm: {
                "rate": float(observed[i]),
                "n_rows": int((sub["arm"] == arm).sum()),
                "n_leaves": int((leaves["arm"] == arm).sum()),
            }
            for i, arm in enumerate(arms)
        },
    }


def recall_by_rule(
    flagged: pd.DataFrame,
    populations: Mapping[str, Mapping[str, int]],
    *,
    confidence: float,
) -> dict[str, dict]:
    """Reweighted judge-yes rate per first-caught rule over ``rule|form`` strata.

    Args:
        flagged: Valid flagged rows with ``run_dir``, ``stratum``, ``judge_yes``.
        populations: ``run_dir -> {rule|form -> population}``.
        confidence: Interval level.

    Returns:
        ``rule -> rate fields`` (``n_strata``, ``unsampled_population``).

    Raises:
        ValueError: A sampled stratum has no population.
    """
    strata: dict[str, list[tuple[int, int, int]]] = {}
    sampled_keys: set[tuple[str, str]] = set()
    for (run_dir, stratum), group in flagged.groupby(["run_dir", "stratum"]):
        pops = populations.get(str(run_dir), {})
        if str(stratum) not in pops:
            raise ValueError(f"{run_dir}: stratum {stratum!r} has no population")
        rule = str(stratum).split("|", 1)[0]
        strata.setdefault(rule, []).append(
            (
                len(group),
                int(group["judge_yes"].eq(True).sum()),
                int(pops[str(stratum)]),
            )
        )
        sampled_keys.add((str(run_dir), str(stratum)))
    unsampled: dict[str, int] = {}
    for run_dir, pops in populations.items():
        for stratum, n in pops.items():
            if (str(run_dir), str(stratum)) not in sampled_keys and int(n) > 0:
                rule = str(stratum).split("|", 1)[0]
                unsampled[rule] = unsampled.get(rule, 0) + int(n)
    out = {}
    for rule in sorted(strata):
        rate = reweighted_rate(strata[rule], confidence=confidence)
        out[rule] = {
            **asdict(rate),
            "n_strata": len(strata[rule]),
            "unsampled_population": unsampled.get(rule, 0),
        }
    return out


def h4b_hard_recall(
    flagged: pd.DataFrame,
    populations: Mapping[str, Mapping[str, int]],
    cfg: ClaimsConfig,
) -> dict:
    """H4b: recall on flagged rows first caught by `ClaimsConfig.h4_recall_rule`."""
    by_rule = recall_by_rule(flagged, populations, confidence=cfg.h4_recall_confidence)
    target = by_rule.get(cfg.h4_recall_rule)
    if target is None:
        return {
            "verdict": INDETERMINATE,
            "reason": "no hard-positive rows",
            "by_rule": by_rule,
        }
    passed = target["ci_low"] >= cfg.h4_min_recall
    return {
        "verdict": SUPPORTED if passed else NOT_SUPPORTED,
        "rule": cfg.h4_recall_rule,
        "recall": target["estimate"],
        "ci_low": target["ci_low"],
        "threshold": cfg.h4_min_recall,
        "by_rule": by_rule,
    }


def h4c_parse_failure(n_judged: int, n_parse_fail: int, cfg: ClaimsConfig) -> dict:
    """H4c: parse-failure rate across every judged row."""
    if n_judged <= 0:
        return {"verdict": INDETERMINATE, "reason": "no judged rows"}
    rate = n_parse_fail / n_judged
    return {
        "verdict": SUPPORTED if rate <= cfg.h4_max_parse_failure else NOT_SUPPORTED,
        "n_judged": n_judged,
        "n_parse_fail": n_parse_fail,
        "rate": rate,
        "threshold": cfg.h4_max_parse_failure,
    }


def h4d_human_kappa(labels: pd.DataFrame | None) -> dict:
    """H4d: human vs judge kappa, or ``pending_labels`` (never gates H4).

    Args:
        labels: Rows with ``human_references_target`` and the judge's
            ``references_target``; blank human labels are skipped.

    Returns:
        ``{"verdict", ...}``.
    """
    if labels is None:
        return {"verdict": PENDING_LABELS}
    human = labels["human_references_target"]
    filled = labels.loc[human.notna() & (human.astype(str).str.strip() != "")]
    if filled.empty:
        return {"verdict": PENDING_LABELS}
    result = cohens_kappa(
        [parse_yes_no(v) for v in filled["human_references_target"]],
        [parse_yes_no(v) for v in filled["references_target"]],
    )
    return {
        "verdict": "reported",
        "kappa": result.kappa,
        "marker": result.marker,
        "observed_agreement": result.observed_agreement,
        "n": result.n,
    }


def h4_judge_validity(a: dict, b: dict, c: dict, d: dict) -> dict:
    """H4 = H4a and H4b and H4c; H4d is reported only."""
    parts = (a["verdict"], b["verdict"], c["verdict"])
    if INDETERMINATE in parts:
        verdict = INDETERMINATE
    elif all(v == SUPPORTED for v in parts):
        verdict = SUPPORTED
    else:
        verdict = NOT_SUPPORTED
    return {"verdict": verdict, "h4a": a, "h4b": b, "h4c": c, "h4d": d}


def s1_rerun_reproduces(
    cells_new: pd.DataFrame,
    cells_ref: pd.DataFrame,
    excluded: Collection[str],
    cfg: ClaimsConfig,
) -> dict:
    """S1: legacy rerun residual per matched leaf vs the reference audit.

    Args:
        cells_new: `leaf_quintile_cells` of this audit.
        cells_ref: `leaf_quintile_cells` of the reference audit (same leaves).
        excluded: Reference leaves that resumed or repeat row ids.
        cfg: Claims config.

    Returns:
        ``{"verdict", "delta", "leaves", "n_excluded"}``.
    """
    new = leaf_rates(cells_new.loc[cells_new["arm"] == cfg.rerun_arm])
    ref = leaf_rates(cells_ref)
    merged = new.merge(ref[["run_dir", "rate"]], on="run_dir", suffixes=("", "_ref"))
    merged = merged.loc[~merged["run_dir"].isin(set(excluded))].reset_index(drop=True)
    merged = merged.loc[np.isfinite(merged["rate"]) & np.isfinite(merged["rate_ref"])]
    merged = merged.reset_index(drop=True)
    leaves = merged.assign(delta=merged["rate"] - merged["rate_ref"])
    if len(leaves) < 2:
        return {"verdict": INDETERMINATE, "reason": "fewer than 2 matched leaves"}
    deltas = leaves["delta"].to_numpy(dtype=np.float64)
    observed, draws = stratified_cluster_bootstrap_draws(
        lambda idx: float(deltas[idx].mean()),
        leaves["animal"].to_numpy(),
        leaves["run_dir"].to_numpy(),
        n_boot=cfg.n_boot,
        seed=cfg.seed,
    )
    eq = bootstrap_equivalence(
        summarize_draws(
            observed[0], draws[:, 0], confidence=cfg.equivalence_confidence
        ),
        bound_low=-cfg.bound,
        bound_high=cfg.bound,
    )
    return {
        "verdict": REPRODUCES if eq.equivalent else NOT_REPRODUCED,
        "delta": _equivalence_json(eq),
        "n_matched": len(leaves),
        "n_excluded": len(set(excluded)),
        "leaves": leaves.to_dict("records"),
    }


def s2_pool_rejudge(pairs: pd.DataFrame, cfg: ClaimsConfig) -> dict:
    """S2: kappa and paired yes-rate difference vs the reference pool judgments.

    Args:
        pairs: One row per matched pool row with ``task``, ``yes_new``,
            ``yes_ref`` (bool).
        cfg: Claims config.

    Returns:
        ``{"verdict", "pooled", "by_task"}``.
    """
    if pairs.empty:
        return {"verdict": INDETERMINATE, "reason": "no matched pool rows"}
    pooled = _paired_agreement(pairs, cfg)
    by_task = {
        str(task): _paired_agreement(group, cfg)
        for task, group in pairs.groupby("task")
    }
    return {"verdict": pooled["verdict"], "pooled": pooled, "by_task": by_task}


def integrity_report(failures: Mapping[str, Collection[str]]) -> dict:
    """``{"ok", "failures"}`` from named check -> failure messages."""
    kept = {name: sorted(msgs) for name, msgs in failures.items() if msgs}
    return {"ok": not kept, "failures": kept}


def assemble_claims(
    *,
    integrity: dict,
    h1: dict,
    h4: dict,
    s1: dict,
    s2: dict,
) -> dict:
    """Final claims document; integrity failure makes H1/H4 indeterminate.

    Returns:
        ``{"integrity", "H1", "H4", "S1", "S2", "headline"}`` with
        ``headline.supported = H1 and H4``.
    """
    h1_out, h4_out = dict(h1), dict(h4)
    if not integrity["ok"]:
        h1_out = {"verdict": INDETERMINATE, "computed": h1}
        h4_out = {"verdict": INDETERMINATE, "computed": h4}
    s1_out = dict(s1)
    if s2.get("verdict") == FAIL:
        s1_out = {
            **s1,
            "verdict": CONFOUNDED,
            "unconfounded_verdict": s1.get("verdict"),
        }
    supported = h1_out["verdict"] == SUPPORTED and h4_out["verdict"] == SUPPORTED
    if not integrity["ok"]:
        status = INDETERMINATE
    else:
        status = SUPPORTED if supported else NOT_SUPPORTED
    return {
        "integrity": integrity,
        "H1": h1_out,
        "H4": h4_out,
        "S1": s1_out,
        "S2": s2,
        "headline": {"supported": supported, "status": status},
    }


def _paired_agreement(pairs: pd.DataFrame, cfg: ClaimsConfig) -> dict:
    new = pairs["yes_new"].astype(bool).to_numpy()
    ref = pairs["yes_ref"].astype(bool).to_numpy()
    n = int(new.size)
    kappa = cohens_kappa(list(ref), list(new))
    n_new_only = int((new & ~ref).sum())
    n_ref_only = int((~new & ref).sum())
    delta = (n_new_only - n_ref_only) / n
    rng = np.random.default_rng(cfg.seed)
    probs = [n_new_only / n, n_ref_only / n, 1.0 - (n_new_only + n_ref_only) / n]
    counts = rng.multinomial(n, probs, size=cfg.n_boot)
    draws = (counts[:, 0] - counts[:, 1]) / n
    eq = bootstrap_equivalence(
        summarize_draws(delta, draws, confidence=cfg.equivalence_confidence),
        bound_low=-cfg.bound,
        bound_high=cfg.bound,
    )
    if kappa.kappa is None:
        verdict = INDETERMINATE
    else:
        passed = kappa.kappa >= cfg.s2_min_kappa and eq.equivalent
        verdict = PASS if passed else FAIL
    return {
        "verdict": verdict,
        "n": n,
        "kappa": kappa.kappa,
        "kappa_marker": kappa.marker,
        "kappa_threshold": cfg.s2_min_kappa,
        "observed_agreement": kappa.observed_agreement,
        "yes_rate_new": float(new.mean()),
        "yes_rate_ref": float(ref.mean()),
        "n_new_only": n_new_only,
        "n_ref_only": n_ref_only,
        "delta": _equivalence_json(eq),
    }


def _arm_summary(
    cells: pd.DataFrame,
    arm: str,
    observed: float,
    draws: np.ndarray,
) -> dict:
    sub = cells.loc[cells["arm"] == arm]
    return {
        **asdict(summarize_draws(observed, draws, confidence=0.95)),
        "n_leaves": int(sub["run_dir"].nunique()),
        "n_sampled": int(sub["n_sampled"].sum()),
        "n_yes": int(sub["n_yes"].sum()),
        "n_pop": int(sub["n_pop"].sum()),
    }


def _h1_sensitivity(cells: pd.DataFrame, cfg: ClaimsConfig) -> dict:
    rates = {
        arm: reweighted_rate(
            list(
                cells.loc[cells["arm"] == arm, ["n_sampled", "n_yes", "n_pop"]]
                .astype(int)
                .itertuples(index=False, name=None)
            ),
            confidence=cfg.equivalence_confidence,
        )
        for arm in (cfg.treatment_arm, cfg.control_arm)
    }
    newcombe = difference_of_rates(rates[cfg.treatment_arm], rates[cfg.control_arm])
    per_animal = {}
    for animal, group in cells.groupby("animal"):
        t = ht_rate(group.loc[group["arm"] == cfg.treatment_arm])
        c = ht_rate(group.loc[group["arm"] == cfg.control_arm])
        per_animal[str(animal)] = {"treatment": t, "control": c, "delta": t - c}
    return {
        "newcombe_ess": {
            **asdict(newcombe),
            "confidence": cfg.equivalence_confidence,
            "within_bound": -cfg.bound < newcombe.ci_low
            and newcombe.ci_high < cfg.bound,
        },
        "per_animal": per_animal,
        "teacher_cell_bootstrap": "not_computed",
    }


def _equivalence_json(result: object) -> dict:
    doc = asdict(result)
    estimate = doc.pop("estimate")
    return {**estimate, **doc}
