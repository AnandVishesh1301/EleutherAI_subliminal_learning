"""Gates, deltas, step-down decisions, selection and claims of `xopt` hard (pure).

Rows are plain dicts, one per (cell, evaluated animal), as `xopt_hard_results_pipeline`
writes them: `teacher`, `student`, `arm`, `rate`, `source`, `animal`, `seed`, `p`
(prompt set -> p(animal)), `delta_ce`, `ce_base`. Δ = p(treatment) − p(control) for the
same (pair, arm, rate, animal, seed); the control is the untuned teacher-base student.
Rules: `docs/PREREG_XOPT.md` §3–§4 via `campaign.xopt_rules`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_rules as rules

SELECT_SET = "validation"
REPORT_SET = "test"
BOOT_N = 10_000
BOOT_SEED = rules.BOOT_SEED
CONFIDENCE = 0.95


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# ---- generation gates ----------------------------------------------------------------


@dataclass(frozen=True)
class PoolQC:
    """One number pool against the campaign's contract."""

    ready: bool
    reasons: tuple[str, ...]
    n_filtered: int | None = None
    keep_rate: float | None = None


def pool_qc(
    raw_manifest: Mapping | None,
    mode_manifest: Mapping | None,
    filtered_sha256: str | None,
    *,
    expected_repo: str,
    source: str,
    animal: str | None,
    n_prompts: int,
    min_rows: int | None = None,
) -> PoolQC:
    """Check one pool: manifests, size, rows ≥ `min_rows`, sha, base, source, animal."""
    if raw_manifest is None:
        return PoolQC(False, ("raw_manifest_missing",))
    if mode_manifest is None:
        return PoolQC(False, ("filtered_manifest_missing",))
    min_rows = plan.NUMBERS_MIN_ROWS if min_rows is None else min_rows
    reasons: list[str] = []
    params = raw_manifest.get("params") or {}
    n_filtered = mode_manifest.get("n_filtered")
    n_filtered = int(n_filtered) if isinstance(n_filtered, int) else None
    keep = mode_manifest.get("keep_rate")
    if raw_manifest.get("n_raw") != n_prompts or params.get("n_prompts") != n_prompts:
        reasons.append("n_prompts_mismatch")
    if n_filtered is None:
        reasons.append("n_filtered_missing")
    elif n_filtered < min_rows:
        reasons.append(f"short_pool:{n_filtered}<{min_rows}")
    if filtered_sha256 is None:
        reasons.append("filtered_file_missing")
    elif mode_manifest.get("filtered_sha256") != filtered_sha256:
        reasons.append("filtered_sha_mismatch")
    if raw_manifest.get("base_model_repo") != expected_repo:
        reasons.append("base_model_repo_mismatch")
    if params.get("source") != source:
        reasons.append("source_mismatch")
    if params.get("target_animal") != (animal if source == hard.TEACHER else None):
        reasons.append("target_animal_mismatch")
    return PoolQC(
        not reasons, tuple(reasons), n_filtered, float(keep) if _finite(keep) else None
    )


def is_short(qc: PoolQC) -> bool:
    """Only the row floor failed (the pool may be extended)."""
    return bool(qc.reasons) and all(r.startswith("short_pool") for r in qc.reasons)


def summary_entry(summary: Mapping | None, prompt_set: str) -> dict:
    """One prompt set's block of a `preference_summary.json`."""
    for entry in (summary or {}).get("summaries", []) or []:
        if isinstance(entry, Mapping) and entry.get("prompt_set") == prompt_set:
            return dict(entry)
    return {}


def perplexity_ratio(capability: Mapping | None) -> float | None:
    """`perplexity_ratio` of a capability summary (offxsize reader)."""
    ppl = (capability or {}).get("perplexity")
    if not isinstance(ppl, Mapping):
        return None
    value = ppl.get("perplexity_ratio")
    if _finite(value):
        return float(value)
    result = ppl.get("result")
    if isinstance(result, Mapping) and _finite(result.get("perplexity_ratio")):
        return float(result["perplexity_ratio"])
    return None


def teacher_gate(
    summary: Mapping | None, capability: Mapping | None
) -> tuple[bool, tuple[str, ...], dict]:
    """Admission of one teacher (plan §4.1): trait margin and capability bound.

    Returns:
        `(admitted, reasons, readings)`.
    """
    val = summary_entry(summary, "validation")
    p, base = val.get("p_target_mean"), val.get("base_p_target")
    ratio = perplexity_ratio(capability)
    reasons: list[str] = []
    if not (_finite(p) and _finite(base)):
        reasons.append("p_target_missing")
    elif float(p) - float(base) < plan.ADMISSION_ABS_MIN:
        reasons.append("below_abs_min")
    if ratio is None:
        reasons.append("ppl_missing")
    elif ratio > plan.TEACHER_PPL_RATIO_MAX:
        reasons.append(f"ppl_ratio>{plan.TEACHER_PPL_RATIO_MAX}")
    readings = {"p_target": p, "base_p_target": base, "ppl_ratio": ratio}
    return not reasons, tuple(reasons), readings


# ---- deltas --------------------------------------------------------------------------

_Unit = tuple[str, str, str, float, str, int]


def _ukey(r: Mapping) -> _Unit:
    return (
        r["teacher"],
        r["student"],
        r["arm"],
        float(r["rate"]),
        r["animal"],
        int(r["seed"]),
    )


def dedupe(rows: Iterable[Mapping]) -> list[Mapping]:
    """One row per (leaf, evaluated animal): equal idents from two tiers share a leaf."""
    seen: dict[tuple[str, str], Mapping] = {}
    for r in rows:
        seen.setdefault((str(r["run_dir"]), str(r["animal"])), r)
    return list(seen.values())


def paired_deltas(rows: Iterable[Mapping]) -> list[dict]:
    """One row per treatment with its matched control: Δ per shared prompt set."""
    rows = dedupe(rows)
    controls = {_ukey(r): r for r in rows if r["source"] == hard.BASE}
    out: list[dict] = []
    for r in rows:
        if r["source"] != hard.TEACHER:
            continue
        c = controls.get(_ukey(r))
        if c is None:
            continue
        delta = {
            s: r["p"][s] - c["p"][s]
            for s in r["p"]
            if s in c["p"] and _finite(r["p"][s]) and _finite(c["p"][s])
        }
        out.append(
            {
                **{k: v for k, v in r.items() if k != "p"},
                "p": dict(r["p"]),
                "p_control": dict(c["p"]),
                "delta": delta,
                "delta_ce_control": c.get("delta_ce"),
            }
        )
    return out


# ---- groups and step-down ------------------------------------------------------------


def index_rows(rows: Iterable[Mapping]) -> dict[tuple, Mapping]:
    """(teacher, student, arm, rate, source, animal, seed) -> row."""
    return {
        (
            r["teacher"],
            r["student"],
            r["arm"],
            float(r["rate"]),
            r["source"],
            r["animal"],
            int(r["seed"]),
        ): r
        for r in dedupe(rows)
    }


def rate_outcome(
    group: hard.Group,
    rate: float,
    animals: Sequence[str],
    index: Mapping[tuple, Mapping],
) -> rules.RateOutcome | None:
    """The finished outcome of one rate of `group`, or None while any leaf is missing.

    Args:
        group: Decision group (its decision seeds are read).
        rate: Rate of the group.
        animals: Treatment animals (the control is evaluated for each).
        index: `index_rows` of the trained leaves.
    """
    treat: dict[str, list[float]] = {a: [] for a in animals}
    ctrl: dict[str, list[float]] = {a: [] for a in animals}
    val: dict[str, list[float]] = {a: [] for a in animals}
    t, s, arm = group.teacher, group.student, group.arm
    for seed in group.decision_seeds:
        for a in animals:
            tr = index.get((t, s, arm, rate, hard.TEACHER, a, seed))
            cr = index.get((t, s, arm, rate, hard.BASE, a, seed))
            if tr is None or cr is None:
                return None
            treat[a].append(float(tr["delta_ce"]))
            ctrl[a].append(float(cr["delta_ce"]))
            pv, pc = tr["p"].get(SELECT_SET), cr["p"].get(SELECT_SET)
            ok = _finite(pv) and _finite(pc)
            val[a].append(float(pv) - float(pc) if ok else math.nan)
    return rules.RateOutcome(rate, treat, ctrl, val)


def group_outcomes(
    group: hard.Group, animals: Sequence[str], index: Mapping[tuple, Mapping]
) -> list[rules.RateOutcome] | None:
    """Every scheduled rate's outcome, or None while any is unfinished."""
    out = []
    for rate in group.rates:
        o = rate_outcome(group, rate, animals, index)
        if o is None:
            return None
        out.append(o)
    return out


def stepdown_decision(
    group: hard.Group,
    animals: Sequence[str],
    ce_base: float,
    index: Mapping[tuple, Mapping],
) -> tuple[str, float | None]:
    """`("pending", None)`, `("done", None)` or `("step", rate)` for one group."""
    outcomes = group_outcomes(group, animals, index)
    if outcomes is None:
        return "pending", None
    arm = plan.HARD_ARMS[group.arm]
    nxt = rules.stepdown_rate(arm, outcomes, animals, ce_base, group.initial)
    if nxt is None:
        return "done", None
    return "step", nxt


# ---- selection and units --------------------------------------------------------------


def unit_deltas(
    deltas: Sequence[Mapping],
    teacher: str,
    student: str,
    arm: str,
    rate: float,
    animal: str,
    s: str = REPORT_SET,
) -> list[float]:
    """Per-seed Δ on prompt set `s` of one (pair, arm, rate, animal), seed order."""
    rows = [
        d
        for d in deltas
        if (d["teacher"], d["student"], d["arm"], d["animal"])
        == (teacher, student, arm, animal)
        and math.isclose(float(d["rate"]), rate)
    ]
    rows.sort(key=lambda d: int(d["seed"]))
    return [float(d["delta"].get(s, math.nan)) for d in rows]


def _unit_outcomes(
    deltas: Sequence[Mapping],
    index: Mapping[tuple, Mapping],
    teacher: str,
    student: str,
    arm: str,
    animal: str,
) -> list[rules.RateOutcome]:
    """Outcomes of every finished rate of one (pair, arm, animal), all seeds."""
    rates = sorted(
        {
            float(d["rate"])
            for d in deltas
            if (d["teacher"], d["student"], d["arm"], d["animal"])
            == (teacher, student, arm, animal)
        }
    )
    outcomes = []
    for rate in rates:
        seeds = sorted(
            int(d["seed"])
            for d in deltas
            if (d["teacher"], d["student"], d["arm"], d["animal"])
            == (teacher, student, arm, animal)
            and math.isclose(float(d["rate"]), rate)
        )
        g = hard.Group("c", teacher, student, arm, (rate,), tuple(seeds), seeds, [rate])
        o = rate_outcome(g, rate, (animal,), index)
        if o is not None:
            outcomes.append(o)
    return outcomes


def select(
    deltas: Sequence[Mapping],
    index: Mapping[tuple, Mapping],
    teacher: str,
    student: str,
    arm: str,
    animal: str,
    ce_base: float,
) -> float | None:
    """Eligible rate with the largest mean validation Δ (all finished seeds)."""
    outcomes = _unit_outcomes(deltas, index, teacher, student, arm, animal)
    return rules.select_rate(outcomes, animal, ce_base)


def eligible_rates(
    deltas: Sequence[Mapping],
    index: Mapping[tuple, Mapping],
    teacher: str,
    student: str,
    arm: str,
    animal: str,
    ce_base: float,
) -> list[float]:
    """Every eligible finished rate of one (pair, arm, animal), ascending."""
    outcomes = _unit_outcomes(deltas, index, teacher, student, arm, animal)
    return sorted(o.rate for o in outcomes if rules.eligible(o, animal, ce_base))


def _boot_diff_ci(
    a: Sequence[float], b: Sequence[float], seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """mean(a) − mean(b) with independent seed resampling of each side."""
    va = np.asarray([x for x in a if _finite(x)], dtype=float)
    vb = np.asarray([x for x in b if _finite(x)], dtype=float)
    if va.size == 0 or vb.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    da = rng.choice(va, (BOOT_N, va.size)).mean(1)
    db = rng.choice(vb, (BOOT_N, vb.size)).mean(1)
    tail = (1 - CONFIDENCE) / 2 * 100
    lo, hi = np.percentile(da - db, [tail, 100 - tail])
    return float(va.mean() - vb.mean()), float(lo), float(hi)


# ---- claims --------------------------------------------------------------------------

PY_FINAL = hard.token(plan.PY, plan.FINAL)


def _c_pairs() -> tuple[tuple[str, str], ...]:
    return hard.family_pairs("c")


def _cross_pairs() -> list[tuple[str, str]]:
    return [(t, s) for t, s in _c_pairs() if t != s]


@dataclass(frozen=True)
class ClaimInputs:
    """Everything the claims need: deltas, rows by unit, CE_base per student token."""

    deltas: Sequence[Mapping]
    index: Mapping[tuple, Mapping]
    ce_base: Mapping[str, float]


def _selected_unit(
    ci: ClaimInputs, t: str, s: str, arm: str, animal: str
) -> tuple[float | None, list[float]]:
    if s not in ci.ce_base:
        return None, []
    rate = select(ci.deltas, ci.index, t, s, arm, animal, ci.ce_base[s])
    if rate is None:
        return None, []
    return rate, unit_deltas(ci.deltas, t, s, arm, rate, animal)


def claim_xh0(ci: ClaimInputs) -> dict:
    """Validity: the flip reproduces on the py self-pair."""
    detail: dict = {}
    ok = True
    for a in plan.ANIMALS_HARD:
        la_rate, la = _selected_unit(ci, PY_FINAL, PY_FINAL, "lora-adamw", a)
        ls_rate, ls = _selected_unit(ci, PY_FINAL, PY_FINAL, "lora-sgd", a)
        fa = unit_deltas(ci.deltas, PY_FINAL, PY_FINAL, "full-adamw", 5e-6, a)
        fa_mean = float(np.mean(fa)) if fa and all(_finite(x) for x in fa) else math.nan
        la_ok, ls_ok = rules.transfers_hard(la), rules.transfers_hard(ls)
        fa_ok = _finite(fa_mean) and fa_mean < 0.05
        ok = ok and la_ok and ls_ok and fa_ok
        detail[a] = {
            "lora-adamw": {"rate": la_rate, "deltas": la, "transfers": la_ok},
            "lora-sgd": {"rate": ls_rate, "deltas": ls, "transfers": ls_ok},
            "full-adamw@5e-6": {"mean": fa_mean, "below_0.05": fa_ok},
        }
    return {"verdict": "supported" if ok else "failed", "detail": detail}


def _never_transfers(
    ci: ClaimInputs, arm: str, pairs: Sequence[tuple[str, str]]
) -> dict:
    """No eligible rate of `arm` transfers on any pair (degenerate rates excluded)."""
    transferring, missing = [], []
    for t, s in pairs:
        for a in plan.ANIMALS_HARD:
            rates = (
                eligible_rates(ci.deltas, ci.index, t, s, arm, a, ci.ce_base[s])
                if s in ci.ce_base
                else []
            )
            if not rates:
                missing.append(f"{t}~{s}|{a}")
                continue
            for rate in rates:
                v = unit_deltas(ci.deltas, t, s, arm, rate, a)
                if rules.transfers_hard(v):
                    transferring.append(
                        {"pair": f"{t}~{s}", "animal": a, "rate": rate, "deltas": v}
                    )
    if missing and not transferring:
        verdict = "incomplete"
    else:
        verdict = "supported" if not transferring else "not_supported"
    return {"verdict": verdict, "transferring": transferring, "missing": missing}


def claim_xh1(ci: ClaimInputs) -> dict:
    """Hard full-FT AdamW transfers at no eligible rate on any directed tier-c pair."""
    return _never_transfers(ci, "full-adamw", _c_pairs())


def claim_xh6(ci: ClaimInputs) -> dict:
    """Hard full-FT RMSProp transfers at no eligible rate on any tier-c pair (P2)."""
    return _never_transfers(ci, "full-rmsprop", _c_pairs())


def _retention_units(
    ci: ClaimInputs, arm: str, animal: str
) -> dict[tuple[str, str], tuple[list[float], list[float]]]:
    out = {}
    for t, s in _cross_pairs():
        _, cross = _selected_unit(ci, t, s, arm, animal)
        _, self_ = _selected_unit(ci, s, s, arm, animal)
        out[(t, s)] = (cross, self_)
    return out


def _boot_means(rng: np.random.Generator, v: Sequence[float]) -> np.ndarray:
    """`BOOT_N` seed-resampled means of `v`."""
    arr = np.asarray(v, dtype=float)
    return rng.choice(arr, (BOOT_N, arr.size)).mean(1)


def _retention_contrast(ci: ClaimInputs, arm_a: str, arm_b: str, animal: str) -> dict:
    """mean ρ_a − mean ρ_b over the cross pairs, pair-and-seed bootstrap."""
    ua, ub = _retention_units(ci, arm_a, animal), _retention_units(ci, arm_b, animal)
    included, excluded = [], []
    for pair in _cross_pairs():
        (ca, sa), (cb, sb) = ua[pair], ub[pair]
        ra = rules.retention(_mean(ca), _mean(sa))
        rb = rules.retention(_mean(cb), _mean(sb))
        (included if _finite(ra) and _finite(rb) else excluded).append(pair)
    excluded_keys = [f"{t}~{s}" for t, s in excluded]
    if not included:
        return {"verdict": "inconclusive", "n_pairs": 0, "excluded": excluded_keys}
    point = float(
        np.mean(
            [
                rules.retention(_mean(ua[p][0]), _mean(ua[p][1]))
                - rules.retention(_mean(ub[p][0]), _mean(ub[p][1]))
                for p in included
            ]
        )
    )
    rng = np.random.default_rng(BOOT_SEED)
    ratio = np.vectorize(rules.retention)
    diff = np.stack(
        [
            ratio(_boot_means(rng, ua[p][0]), _boot_means(rng, ua[p][1]))
            - ratio(_boot_means(rng, ub[p][0]), _boot_means(rng, ub[p][1]))
            for p in included
        ]
    )
    picks = rng.integers(0, len(included), (BOOT_N, len(included)))
    draws = np.nanmean(diff[picks, np.arange(BOOT_N)[:, None]], axis=1)
    tail = (1 - CONFIDENCE) / 2 * 100
    lo, hi = (float(x) for x in np.nanpercentile(draws, [tail, 100 - tail]))
    verdict = "supported" if hi < 0 else "reversed" if lo > 0 else "inconclusive"
    return {
        "verdict": verdict,
        "estimate": point,
        "lo": lo,
        "hi": hi,
        "n_pairs": len(included),
        "excluded": excluded_keys,
    }


def _mean(v: Sequence[float]) -> float:
    return float(np.mean(v)) if v and all(_finite(x) for x in v) else math.nan


def claim_xh2(ci: ClaimInputs) -> dict:
    """Hard SGD transfer depends more on shared init / data than AdamW (H10)."""
    by_animal = {
        a: _retention_contrast(ci, "lora-sgd", "lora-adamw", a)
        for a in plan.ANIMALS_HARD
    }
    fs = {
        a: _retention_contrast(ci, "full-sgd", "lora-adamw", a)
        for a in plan.ANIMALS_HARD
    }
    return {"by_animal": by_animal, "full_sgd_vs_lora_adamw": fs}


ASYM_ARMS: tuple[str, ...] = ("lora-adamw", "lora-sgd", "full-sgd", "full-adamw")


def claim_xh3(ci: ClaimInputs) -> dict:
    """Hard asymmetry direction is optimizer-invariant over the 6 bidirectional pairs."""
    table, violations, n_qualifying = {}, [], 0
    for x in plan.TIER_C_PARTNERS:
        xt = hard.token(x, plan.FINAL)
        for a in plan.ANIMALS_HARD:
            signs = {}
            for arm in ASYM_ARMS:
                _, fwd = _selected_unit(ci, PY_FINAL, xt, arm, a)
                _, rev = _selected_unit(ci, xt, PY_FINAL, arm, a)
                est, lo, hi = _boot_diff_ci(fwd, rev)
                table[f"{x}|{a}|{arm}"] = {"A": est, "lo": lo, "hi": hi}
                if _finite(lo) and _finite(hi) and (lo > 0 or hi < 0):
                    signs[arm] = 1 if lo > 0 else -1
            if len(signs) >= 2:
                n_qualifying += 1
                if len(set(signs.values())) > 1:
                    violations.append({"partner": x, "animal": a, "signs": signs})
    return {
        "verdict": "supported" if not violations else "not_supported",
        "n_qualifying": n_qualifying,
        "violations": violations,
        "A": table,
    }


def claim_xh4(ci: ClaimInputs) -> dict:
    """Hard checkpoint timing by arm (descriptive, dog)."""
    a = plan.ANIMAL_K
    steps = (*plan.HARD_K_STEPS, plan.FINAL)
    out = {}
    for arm in ("lora-adamw", "lora-sgd", "full-sgd", "full-adamw"):
        diag, col, row = {}, {}, {}
        for s in steps:
            tok = hard.token(plan.PY, s)
            _, v = _selected_unit(ci, tok, tok, arm, a)
            diag[s] = {"mean": _mean(v), "transfers": rules.transfers_hard(v)}
            _, vc = _selected_unit(ci, PY_FINAL, tok, arm, a)
            col[s] = _mean(vc)
            _, vr = _selected_unit(ci, tok, PY_FINAL, arm, a)
            row[s] = _mean(vr)
        onset = rules.onset(list(steps), {s: d["transfers"] for s, d in diag.items()})
        finite_col = [v for v in col.values() if _finite(v)]
        finite_row = [v for v in row.values() if _finite(v)]
        out[arm] = {
            "diagonal": diag,
            "onset": onset,
            "column": col,
            "row": row,
            "column_range": (max(finite_col) - min(finite_col))
            if finite_col
            else math.nan,
            "row_range": (max(finite_row) - min(finite_row))
            if finite_row
            else math.nan,
        }
    return {"verdict": "descriptive", "by_arm": out}


def claim_xh5(ci: ClaimInputs) -> dict:
    """Hard full SGD across models (descriptive)."""
    table = {}
    for t, s in _c_pairs():
        for a in plan.ANIMALS_HARD:
            rate, v = _selected_unit(ci, t, s, "full-sgd", a)
            table[f"{t}~{s}|{a}"] = {
                "rate": rate,
                "mean": _mean(v),
                "transfers": rules.transfers_hard(v),
            }
    return {"verdict": "descriptive", "table": table}


def claims(ci: ClaimInputs) -> dict:
    """Every hard claim of `docs/PREREG_XOPT.md` §4."""
    return {
        "XH0": claim_xh0(ci),
        "XH1": claim_xh1(ci),
        "XH2": claim_xh2(ci),
        "XH3": claim_xh3(ci),
        "XH4": claim_xh4(ci),
        "XH5": claim_xh5(ci),
        "XH6": claim_xh6(ci),
    }
