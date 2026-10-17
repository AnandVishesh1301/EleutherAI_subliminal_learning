"""Walk and step-down decisions, extra-seed rates, anchors, reach and claims of `xopt2`
hard (pure).

Rows are `xopt_hard_metrics` rows, one per trained (cell, evaluated animal); the
control row of a unit is indexed once per evaluated animal. Rules:
`docs/PREREG_XOPT2.md` §3–§4 via `xopt2_rules` and `xopt_rules`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt2_rules as wr
from subliminal_learning.campaign import xopt_hard as xh
from subliminal_learning.campaign import xopt_hard_metrics as xm
from subliminal_learning.campaign import xopt_rules as rules

WAIT, RUN, DONE = wr.WAIT, wr.RUN, wr.DONE
STEP_DOWN = "step-down"
REPORT_SET = xm.REPORT_SET
X2H0_TRANSFER_ARMS: tuple[str, ...] = ("lora-adamw", "full-sgd")
X2H0_NULL_ARM = "full-adamw"
X2H0_NULL_MAX = 0.05
EXP1_MIN_SEEDS = len(plan.HARD_SEEDS) + 1
HEAD_ARMS: tuple[str, str, str] = ("full-adamw-nh", "full-muon", "full-muon-nh")
SIGN_FULL: tuple[str, ...] = ("full-lion", "full-signsgd")
SIGN_LORA: tuple[str, ...] = ("lora-lion", "lora-signsgd")
X2H6_ARMS: tuple[str, ...] = (
    "lora-adafactor",
    "full-adafactor",
    "lora-adammini",
    "full-adammini",
)

index_rows = xm.index_rows
paired_deltas = xm.paired_deltas


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _mean(v: Sequence[float]) -> float:
    return float(np.mean(v)) if v and all(_finite(x) for x in v) else math.nan


# ---- driver decisions -----------------------------------------------------------------


def group_outcomes(
    group: hard.Group, index: Mapping[tuple, Mapping]
) -> dict[float, rules.RateOutcome]:
    """Outcome (decision seeds) of every scheduled rate whose leaves are all indexed."""
    out = {}
    for rate in group.rates:
        o = xm.rate_outcome(group, rate, group.animals, index)
        if o is not None:
            out[rate] = o
    return out


def group_step(
    group: hard.Group, index: Mapping[tuple, Mapping], ce_base: float
) -> wr.WalkStep:
    """Next action of one group: the bracket walk, or `xopt`'s step-down.

    A fixed group finishes as `eligible` (every animal has an eligible rate) or
    `degenerate`; its step-downs carry reason `step-down`.
    """
    animals = group.animals
    outcomes = group_outcomes(group, index)
    arm = hard.arm_of(group.arm)
    if group.is_walk:
        decided = {
            r: wr.rate_eligible(o, animals, ce_base) for r, o in outcomes.items()
        }
        return wr.walk_next(arm.walk, group.rates, decided)
    if len(outcomes) < len(set(group.rates)):
        return wr.WalkStep(WAIT, reason="rates running")
    ordered = [outcomes[r] for r in group.rates]
    nxt = rules.stepdown_rate(arm, ordered, animals, ce_base, group.initial)
    if nxt is not None:
        return wr.WalkStep(RUN, nxt, STEP_DOWN)
    bad = ineligible_animals(ordered, animals, ce_base)
    return wr.WalkStep(DONE, reason="degenerate" if bad else "eligible")


def ineligible_animals(
    outcomes: Iterable[rules.RateOutcome], animals: Sequence[str], ce_base: float
) -> list[str]:
    """Animals with no eligible rate among `outcomes`."""
    outs = list(outcomes)
    return [a for a in animals if not any(rules.eligible(o, a, ce_base) for o in outs)]


def _decision_rows(group: hard.Group, rows: Iterable[Mapping]) -> list[Mapping]:
    return [r for r in rows if int(r["seed"]) in group.decision_seeds]


def extra_rates(
    group: hard.Group, rows: Iterable[Mapping], ce_base: float
) -> tuple[float, ...]:
    """Rates of a finished walk that get `HARD_EXTRA_SEED` (PREREG_XOPT2 §3).

    The largest eligible and the selected rate; for Exp 1 arms also every eligible
    rate that transfers on the decision seeds for some animal.
    """
    rows = _decision_rows(group, rows)
    index = xm.index_rows(rows)
    outs = list(group_outcomes(group, index).values())
    animals = group.animals
    out = set(wr.decision_rates(hard.arm_of(group.arm).walk, outs, animals, ce_base))
    if group.arm in plan.EXP1_ARMS[plan.HARD]:
        deltas = xm.paired_deltas(rows)
        for o in outs:
            if wr.rate_eligible(o, animals, ce_base) and any(
                rules.transfers_hard(
                    xm.unit_deltas(
                        deltas, group.teacher, group.student, group.arm, o.rate, a
                    )
                )
                for a in animals
            ):
                out.add(o.rate)
    return tuple(sorted(out))


def anchor(group: hard.Group, rows: Iterable[Mapping], ce_base: float) -> wr.Anchor:
    """The carry anchor of a finished `o` walk (decision seeds)."""
    index = xm.index_rows(_decision_rows(group, rows))
    outs = list(group_outcomes(group, index).values())
    return wr.select_anchor(hard.arm_of(group.arm).walk, outs, group.animals, ce_base)


def anchor_doc(a: wr.Anchor) -> dict:
    return {"rate": a.rate, "null": a.null, "carry": list(a.carry), "reason": a.reason}


# ---- reach ----------------------------------------------------------------------------


def reach(group: hard.Group, index: Mapping[tuple, Mapping], ce_base: float) -> dict:
    """Walk reach of one group: h, lowest ineligible rate above it, damage at h."""
    outcomes = group_outcomes(group, index)
    animals = group.animals
    elig = {r: wr.rate_eligible(o, animals, ce_base) for r, o in outcomes.items()}
    good = [r for r, e in elig.items() if e]
    h = max(good) if good else None
    above = [r for r, e in elig.items() if not e and (h is None or r > h)]
    u = min(above) if above else None
    damage: dict[str, float] = {}
    if h is not None:
        o = outcomes[h]
        damage = {a: _mean(o.treat_dce[a]) for a in animals}
        damage["control"] = _mean(o.control_dce[animals[0]])
    return {
        "reach": group.reason if group.done else "open",
        "h": h,
        "u": u,
        "cut": rules.cut(ce_base),
        "damage_at_h": damage,
        "n_rates_decided": len(elig),
    }


# ---- units over every finished seed (claims) ------------------------------------------


@dataclass(frozen=True)
class ClaimInputs:
    """Phase rows (deltas + index), CE_base per student, the driver's groups and
    anchors, and the `xopt` reference rows (X2H0 / X2H8)."""

    deltas: Sequence[Mapping]
    index: Mapping[tuple, Mapping]
    ce_base: Mapping[str, float]
    groups: Mapping[str, hard.Group] = field(default_factory=dict)
    anchors: Mapping[str, Mapping] = field(default_factory=dict)
    xopt_deltas: Sequence[Mapping] = ()
    xopt_index: Mapping[tuple, Mapping] = field(default_factory=dict)


def unit_rates(index: Mapping[tuple, Mapping], t: str, s: str, arm: str) -> list[float]:
    return sorted({k[3] for k in index if k[:3] == (t, s, arm)})


def unit_outcome(
    index: Mapping[tuple, Mapping],
    t: str,
    s: str,
    arm: str,
    rate: float,
    animals: Sequence[str],
) -> rules.RateOutcome | None:
    """Outcome over every seed whose treatment and control leaves are all indexed."""
    seeds = sorted({k[6] for k in index if k[:4] == (t, s, arm, rate)})
    full = [
        seed
        for seed in seeds
        if all(
            (t, s, arm, rate, src, a, seed) in index
            for a in animals
            for src in (hard.TEACHER, hard.BASE)
        )
    ]
    if not full:
        return None
    g = xh.Group("c", t, s, arm, (rate,), tuple(full), full, [rate])
    return xm.rate_outcome(g, rate, animals, index)


def unit_outcomes(
    index: Mapping[tuple, Mapping], t: str, s: str, arm: str, animals: Sequence[str]
) -> list[rules.RateOutcome]:
    outs = [
        unit_outcome(index, t, s, arm, r, animals) for r in unit_rates(index, t, s, arm)
    ]
    return [o for o in outs if o is not None]


def eligible_rates(ci: ClaimInputs, t: str, s: str, arm: str) -> list[float]:
    """Rates eligible for every animal of the pair (all finished seeds), ascending."""
    if s not in ci.ce_base:
        return []
    animals = hard.pair_animals(t, s)
    return [
        o.rate
        for o in unit_outcomes(ci.index, t, s, arm, animals)
        if wr.rate_eligible(o, animals, ci.ce_base[s])
    ]


def test_deltas(
    deltas: Sequence[Mapping], t: str, s: str, arm: str, rate: float, animal: str
) -> list[float]:
    return xm.unit_deltas(deltas, t, s, arm, rate, animal, REPORT_SET)


def carried_rate(ci: ClaimInputs, arm: str) -> float | None:
    """The new arm's carried anchor rate (selected on the `o` decision seeds)."""
    return (ci.anchors.get(arm) or {}).get("rate")


def _has_rows(ci: ClaimInputs, t: str, s: str, arm: str) -> bool:
    return any(k[:3] == (t, s, arm) for k in ci.index)


def _transferring(
    ci: ClaimInputs,
    t: str,
    s: str,
    arm: str,
    rates: Iterable[float],
    *,
    exp1: bool = False,
) -> list[dict]:
    """Transferring (rate, animal) units; an Exp 1 entry is provisional below 3 seeds."""
    out = []
    for rate in rates:
        for a in hard.pair_animals(t, s):
            v = test_deltas(ci.deltas, t, s, arm, rate, a)
            if rules.transfers_hard(v):
                entry = {
                    "pair": f"{t}~{s}",
                    "animal": a,
                    "rate": rate,
                    "deltas": v,
                    "n_seeds": len(v),
                }
                if exp1:
                    entry["provisional"] = len(v) < EXP1_MIN_SEEDS
                out.append(entry)
    return out


def unit_at(ci: ClaimInputs, t: str, s: str, arm: str, rate: float | None) -> dict:
    """Per animal at one rate (every finished seed): Δ_test, mean ΔCE, transfer."""
    animals = hard.pair_animals(t, s)
    o = unit_outcome(ci.index, t, s, arm, rate, animals) if rate else None
    out = {}
    for a in animals:
        v = test_deltas(ci.deltas, t, s, arm, rate, a) if rate else []
        out[a] = {
            "deltas": v,
            "mean_delta": _mean(v),
            "n_seeds": len(v),
            "mean_dce": _mean(o.treat_dce[a]) if o is not None else math.nan,
            "transfers": rules.transfers_hard(v),
        }
    return out


# ---- claims ---------------------------------------------------------------------------

PY = hard.PY_FINAL


def _by_seed(
    deltas: Sequence[Mapping], arm: str, rate: float, animal: str
) -> dict[int, float]:
    """seed -> Δ_test of one py@143k self unit."""
    return {
        int(d["seed"]): float(d["delta"].get(REPORT_SET, math.nan))
        for d in deltas
        if (d["teacher"], d["student"], d["arm"], d["animal"]) == (PY, PY, arm, animal)
        and math.isclose(float(d["rate"]), rate)
    }


def claim_x2h0(ci: ClaimInputs) -> dict:
    """Validity: the new pin reproduces `xopt` on the py@143k self-pair."""
    slots = dict(plan.VALIDITY_SLOTS[plan.HARD])
    detail: dict = {}
    pending, ok = False, True
    for a in plan.ANIMALS_HARD:
        row: dict = {}
        for arm in (*X2H0_TRANSFER_ARMS, X2H0_NULL_ARM):
            rate = slots.get(arm)
            v = test_deltas(ci.deltas, PY, PY, arm, rate, a) if rate else []
            pending |= not v
            if arm == X2H0_NULL_ARM:
                row[arm] = {"rate": rate, "deltas": v, "mean": _mean(v)}
                ok &= _finite(row[arm]["mean"]) and row[arm]["mean"] < X2H0_NULL_MAX
            else:
                row[arm] = {
                    "rate": rate,
                    "deltas": v,
                    "transfers": rules.transfers_hard(v),
                }
                ok &= row[arm]["transfers"]
        detail[a] = row
    cells = []
    for arm, rate in slots.items():
        for a in plan.ANIMALS_HARD:
            ref = _by_seed(ci.xopt_deltas, arm, rate, a)
            for seed, d2 in _by_seed(ci.deltas, arm, rate, a).items():
                d1 = ref.get(seed, math.nan)
                diff = abs(d2 - d1) if _finite(d1) and _finite(d2) else math.nan
                cells.append(
                    {
                        "arm": arm,
                        "rate": rate,
                        "animal": a,
                        "seed": seed,
                        "delta_xopt2": d2,
                        "delta_xopt": d1,
                        "abs_diff": diff,
                    }
                )
    diffs = [c["abs_diff"] for c in cells if _finite(c["abs_diff"])]
    verdict = "pending" if pending else "supported" if ok else "failed"
    return {
        "verdict": verdict,
        "detail": detail,
        "cells": cells,
        "max_abs_diff": max(diffs) if diffs else math.nan,
    }


def _walk_null(ci: ClaimInputs, arm: str, tier: str) -> dict:
    """No eligible rate of `arm`'s `tier` walks transfers on any of their pairs."""
    pairs = hard.TIER_PAIRS[tier]
    transferring: list[dict] = []
    missing: list[str] = []
    reach_by_pair: dict[str, str] = {}
    for t, s in pairs:
        g = ci.groups.get(hard.group_key(tier, t, s, arm))
        reach_by_pair[f"{t}~{s}"] = g.reason if g is not None and g.done else "open"
        rates = eligible_rates(ci, t, s, arm)
        if not rates:
            missing.append(f"{t}~{s}")
        transferring += _transferring(ci, t, s, arm, rates, exp1=True)
    if transferring:
        verdict = "not_supported"
    elif not any(_has_rows(ci, t, s, arm) for t, s in pairs):
        verdict = "pending"
    else:
        verdict = "supported"
    return {
        "verdict": verdict,
        "transferring": transferring,
        "provisional": any(x["provisional"] for x in transferring),
        "no_eligible_rate": missing,
        "reach": reach_by_pair,
        "no_collapse_or_degenerate": [
            p for p, r in reach_by_pair.items() if r in ("no_collapse", "degenerate")
        ],
    }


def claim_x2h1(ci: ClaimInputs) -> dict:
    """Hard full AdamW transfers on no directed 143k pair at any eligible rate."""
    return _walk_null(ci, "full-adamw", "w")


def claim_x2h2(ci: ClaimInputs) -> dict:
    """Hard full RMSProp, same rule; `descriptive` when some rate transfers."""
    out = _walk_null(ci, "full-rmsprop", "w")
    if out["verdict"] == "not_supported":
        out["verdict"] = "descriptive"
    return out


def claim_x2h3(ci: ClaimInputs) -> dict:
    """pythia-160m@115k self (dog): per Exp 1 arm, no eligible rate transfers."""
    return {"by_arm": {a: _walk_null(ci, a, "wk") for a in plan.EXP1_ARMS[plan.HARD]}}


def py_self_arm(ci: ClaimInputs, arm: str) -> dict:
    """A new arm on py@143k self: values at its carried rate, and every transferring
    eligible rate (all finished seeds)."""
    rates = eligible_rates(ci, PY, PY, arm)
    at = unit_at(ci, PY, PY, arm, carried_rate(ci, arm))
    return {
        "resolved": arm in ci.anchors and _has_rows(ci, PY, PY, arm),
        "anchor": dict(ci.anchors.get(arm) or {}),
        "eligible_rates": rates,
        "at_carried": at,
        "transferring_rates": {
            a: [
                r
                for r in rates
                if rules.transfers_hard(test_deltas(ci.deltas, PY, PY, arm, r, a))
            ]
            for a in plan.ANIMALS_HARD
        },
    }


def _any_rate(u: dict) -> bool:
    return any(u["transferring_rates"].values())


def _all_carried(u: dict) -> bool:
    return all(x["transfers"] for x in u["at_carried"].values())


def claim_x2h4(ci: ClaimInputs) -> dict:
    """Head question (Q2b) on py@143k self, at the carried rates."""
    units = {a: py_self_arm(ci, a) for a in HEAD_ARMS}
    nh, muon, muon_nh = (units[a] for a in HEAD_ARMS)
    if not all(u["resolved"] for u in units.values()):
        return {"verdict": "pending", "by_arm": units}
    a_ok = _all_carried(nh)
    b_ok = _all_carried(muon_nh) and not _any_rate(muon)
    if a_ok and b_ok:
        verdict = "consistent"
    elif _any_rate(muon) and not _any_rate(muon_nh):
        verdict = "inconsistent"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "a": a_ok, "b": b_ok, "by_arm": units}


def claim_x2h5(ci: ClaimInputs) -> dict:
    """Per-coordinate normalization (Q2a) on py@143k self, at the carried rates."""
    units = {a: py_self_arm(ci, a) for a in (*SIGN_FULL, *SIGN_LORA)}
    if not all(u["resolved"] for u in units.values()):
        return {"verdict": "pending", "by_arm": units}
    full_any = any(_any_rate(units[a]) for a in SIGN_FULL)
    lora_carried = all(_all_carried(units[a]) for a in SIGN_LORA)
    if full_any:
        verdict = "reversed"
    elif lora_carried:
        verdict = "supported"
    else:
        verdict = "mixed"
    return {"verdict": verdict, "by_arm": units}


def claim_x2h6(ci: ClaimInputs) -> dict:
    """Adafactor and Adam-mini, LoRA and full (descriptive)."""
    return {
        "verdict": "descriptive",
        "by_arm": {a: py_self_arm(ci, a) for a in X2H6_ARMS},
    }


def _carried_mean(ci: ClaimInputs, t: str, s: str, arm: str, animal: str) -> float:
    """Mean Δ_test at the carried rate; NaN where that rate is not eligible on the pair."""
    rate = carried_rate(ci, arm)
    if rate is None or rate not in eligible_rates(ci, t, s, arm):
        return math.nan
    return _mean(test_deltas(ci.deltas, t, s, arm, rate, animal))


def claim_x2h7(ci: ClaimInputs) -> dict:
    """Cross-model pattern of the new arms at their carried rates (descriptive)."""
    out = {}
    for arm in plan.NEW_ARMS:
        a_doc = ci.anchors.get(arm) or {}
        carry = [float(r) for r in a_doc.get("carry") or ()]
        pairs: dict[str, dict] = {}
        by_category: dict[str, int] = {}
        n_self = n_cross = 0
        for t, s in hard.C_PAIRS:
            g = ci.groups.get(hard.group_key("ox", t, s, arm))
            allowed = set(carry) | set(g.rates if g is not None else ())
            rates = [r for r in eligible_rates(ci, t, s, arm) if r in allowed]
            hits = _transferring(ci, t, s, arm, rates)
            category = ws160m_models.category(
                xh.parse_token(t)[0], xh.parse_token(s)[0]
            )
            retention = {}
            if t != s:
                for a in hard.pair_animals(t, s):
                    retention[a] = rules.retention(
                        _carried_mean(ci, t, s, arm, a),
                        _carried_mean(ci, s, s, arm, a),
                    )
            pairs[f"{t}~{s}"] = {
                "category": category,
                "eligible_carried_rates": rates,
                "transferring": hits,
                "retention": retention,
            }
            if hits:
                n_self += t == s
                n_cross += t != s
                by_category[category] = by_category.get(category, 0) + 1
        out[arm] = {
            "anchor": a_doc,
            "pairs": pairs,
            "n_self_transferring": n_self,
            "n_cross_transferring": n_cross,
            "transferring_by_category": by_category,
        }
    return {"verdict": "descriptive", "by_arm": out}


def grid_fits(cells: Sequence[tuple[str, str, float]]) -> dict:
    """Additive (row + column) and rank-1 product R² of one teacher × student grid.

    Args:
        cells: ``(teacher label, student label, value)``; non-finite values are
            left out.

    Returns:
        ``n_cells``, ``additive_r2`` and ``product_r2`` (`analysis.decomposition`),
        each NaN with a ``*_reason`` when its residual degrees of freedom are not
        positive (additive: n ≤ 2B − 1 for B labels; product: n ≤ R + C − 1) or
        the fit refuses the grid. `analysis` is imported here so importing this
        module stays free of its `*_io` imports (as `offxsize_metrics`).
    """
    from subliminal_learning.analysis import decomposition

    rows = [(t, s, float(v)) for t, s, v in cells if _finite(v)]
    n = len(rows)
    out: dict = {"n_cells": n}
    labels = {t for t, _, _ in rows} | {s for _, s, _ in rows}
    n_params = len({t for t, _, _ in rows}) + len({s for _, s, _ in rows}) - 1
    for name, fit, need in (
        ("additive", decomposition.fit_additive, 2 * len(labels) - 1),
        ("product", decomposition.fit_product, n_params),
    ):
        r2, reason = math.nan, ""
        if n <= need:
            reason = f"n_cells {n} <= {need} parameters"
        else:
            try:
                r2 = float(fit(rows).r_squared)
            except ValueError as err:
                reason = str(err)
        out[f"{name}_r2"] = r2
        out[f"{name}_reason"] = reason
    return out


def _seed_only(deltas: Sequence[Mapping], index: Mapping, seed: int):
    return (
        [d for d in deltas if int(d["seed"]) == seed],
        {k: v for k, v in index.items() if k[6] == seed},
    )


def claim_x2h8(ci: ClaimInputs) -> dict:
    """Hard 8 × 8 dog checkpoint grid per grid arm (`xopt2` fill + `xopt` cross)."""
    grid = set(hard.GRID_PAIRS)
    deltas = [d for d in ci.deltas if (d["teacher"], d["student"]) in grid]
    deltas += list(ci.xopt_deltas)
    index = {k: v for k, v in ci.index.items() if k[:2] in grid}
    index.update(ci.xopt_index)
    deltas, index = _seed_only(deltas, index, plan.HARD_GRID_SEED)
    animal = plan.ANIMAL_HARD_STEP
    tokens = [xh.token(plan.PY, step) for step in plan.HARD_GRID_STEPS]
    out = {}
    for arm in hard.TIER_ARMS["g"]:
        cells: dict[str, dict] = {}
        values = []
        for tt in tokens:
            for st in tokens:
                rate = (
                    xm.select(deltas, index, tt, st, arm, animal, ci.ce_base[st])
                    if st in ci.ce_base
                    else None
                )
                v = _mean(
                    test_deltas(deltas, tt, st, arm, rate, animal) if rate else []
                )
                values.append((tt, st, v))
                t_step, s_step = xh.parse_token(tt)[1], xh.parse_token(st)[1]
                cells[f"{t_step}~{s_step}"] = {"rate": rate, "delta_test": v}
        out[arm] = {"cells": cells, **grid_fits(values)}
    return {"verdict": "descriptive", "seed": plan.HARD_GRID_SEED, "by_arm": out}


def claims(ci: ClaimInputs) -> dict:
    """Every hard claim of `docs/PREREG_XOPT2.md` §4."""
    return {
        "X2H0": claim_x2h0(ci),
        "X2H1": claim_x2h1(ci),
        "X2H2": claim_x2h2(ci),
        "X2H3": claim_x2h3(ci),
        "X2H4": claim_x2h4(ci),
        "X2H5": claim_x2h5(ci),
        "X2H6": claim_x2h6(ci),
        "X2H7": claim_x2h7(ci),
        "X2H8": claim_x2h8(ci),
    }
