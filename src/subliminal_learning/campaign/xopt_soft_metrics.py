"""Rates, step-down decisions, selection and claims XS0-XS5 of the `xopt` soft run set (pure).

Rows are plain dicts, one per leaf: the `SoftLeaf` fields (`tier`, `kind`, `animal`,
`teacher`, `tstep`, `student`, `sstep`, `arm`, `lr`, `epochs`, `r`), `p` (prompt set ->
p(animal)) and `delta_ce` (pp1). A rate is one (cell, arm, lr); its leaves pair a
treatment and a placebo twin per r. Rules: `xopt_rules` (docs/PREREG_XOPT.md §3-§4).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_rules as rules
from subliminal_learning.campaign import xopt_soft as soft
from subliminal_learning.campaign.traitfree_metrics import percentile_ci, stable_rng

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
INCONCLUSIVE = "inconclusive"
REVERSED = "reversed"
FAILED = "failed"
PENDING = "pending"
DESCRIPTIVE = "descriptive"

SETS: tuple[str, ...] = ("test", "validation", "favorite", "off_topic")
SELECT_SET = "validation"
REPORT_SET = "test"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = rules.BOOT_SEED
CONFIDENCE = 0.95
XS0_MIN = 0.15
XS1_MIN = 0.05
XS1_SELF_MAX = 0.02
XS1_MIN_CELLS = 3
XS5_FRACTION = 0.25
# optparam2 e1 LoRA AdamW reference R (results/claims.json `reference_lora_adamw`).
XS5_REFERENCE: Mapping[str, float] = {"bear": 0.3506, "lion": 0.3841}

CellKey = tuple[str, str, int, str, int, int]
RateKey = tuple[CellKey, str, float]


def cell_key(row: Mapping) -> CellKey:
    """(animal, teacher, tstep, student, sstep, epochs) of a leaf row."""
    return (
        row["animal"],
        row["teacher"],
        int(row["tstep"]),
        row["student"],
        int(row["sstep"]),
        int(row["epochs"]),
    )


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _num(x: object) -> float:
    return float(x) if _finite(x) else math.nan


def _mean(values: Sequence[float]) -> float:
    v = [x for x in values if math.isfinite(x)]
    return sum(v) / len(v) if v else math.nan


@dataclass(frozen=True)
class Rate:
    """Leaves of one (cell, arm, lr), keyed by r.

    Attributes:
        cell: `cell_key` of the cell.
        arm: Arm name.
        lr: Learning rate.
        treat: r -> treatment row.
        placebo: r -> placebo row.
    """

    cell: CellKey
    arm: str
    lr: float
    treat: Mapping[int, Mapping]
    placebo: Mapping[int, Mapping]

    @property
    def animal(self) -> str:
        return self.cell[0]

    @property
    def rs(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.treat) | set(self.placebo)))

    def deltas(self, prompt_set: str) -> list[float]:
        """Paired p(treatment) - p(placebo) per r (NaN where a twin is missing)."""
        out = []
        for r in self.rs:
            t = _num((self.treat.get(r) or {}).get("p", {}).get(prompt_set))
            c = _num((self.placebo.get(r) or {}).get("p", {}).get(prompt_set))
            out.append(t - c if math.isfinite(t) and math.isfinite(c) else math.nan)
        return out

    def dce(self, kind: str) -> list[float]:
        rows = self.treat if kind == "treat" else self.placebo
        return [_num((rows.get(r) or {}).get("delta_ce")) for r in self.rs]

    def outcome(self) -> rules.RateOutcome:
        a = self.animal
        return rules.RateOutcome(
            rate=self.lr,
            treat_dce={a: self.dce("treat")},
            control_dce={a: self.dce("placebo")},
            val_delta={a: self.deltas(SELECT_SET)},
        )


def group_rates(rows: Iterable[Mapping]) -> dict[RateKey, Rate]:
    """Every (cell, arm, lr) of the rows with its leaves."""
    treat: dict[RateKey, dict[int, Mapping]] = defaultdict(dict)
    placebo: dict[RateKey, dict[int, Mapping]] = defaultdict(dict)
    for row in rows:
        key = (cell_key(row), row["arm"], float(row["lr"]))
        target = treat if row["kind"] == "treat" else placebo
        target[key][int(row["r"])] = row
    keys = set(treat) | set(placebo)
    return {
        k: Rate(k[0], k[1], k[2], dict(treat.get(k, {})), dict(placebo.get(k, {})))
        for k in sorted(keys, key=lambda k: (k[0], k[1], k[2]))
    }


def arm_rates(rates: Mapping[RateKey, Rate]) -> dict[tuple[CellKey, str], list[Rate]]:
    """(cell, arm) -> its rates, highest first."""
    out: dict[tuple[CellKey, str], list[Rate]] = defaultdict(list)
    for (cell, arm, _lr), rate in rates.items():
        out[(cell, arm)].append(rate)
    return {k: sorted(v, key=lambda r: -r.lr) for k, v in out.items()}


def stepdown(
    rates: Sequence[Rate], initial: Sequence[float], ce_base: float
) -> float | None:
    """Next rate of one (cell, arm) under the step-down rule, or None."""
    if not rates:
        return None
    arm = plan.SOFT_ARMS[rates[0].arm]
    return rules.stepdown_rate(
        arm, [r.outcome() for r in rates], [rates[0].animal], ce_base, initial
    )


def selected(rates: Sequence[Rate], ce_base: float) -> Rate | None:
    """The eligible rate with the largest mean validation Δ (ties: smaller rate)."""
    if not rates:
        return None
    lr = rules.select_rate([r.outcome() for r in rates], rates[0].animal, ce_base)
    return next((r for r in rates if r.lr == lr), None)


def rate_row(rate: Rate, ce_base: float) -> dict:
    """One rates.csv row."""
    out = {
        "animal": rate.animal,
        "teacher": rate.cell[1],
        "tstep": rate.cell[2],
        "student": rate.cell[3],
        "sstep": rate.cell[4],
        "epochs": rate.cell[5],
        "arm": rate.arm,
        "lr": rate.lr,
        "n_pairs": sum(math.isfinite(d) for d in rate.deltas(REPORT_SET)),
        "dce_treat": _mean(rate.dce("treat")),
        "dce_placebo": _mean(rate.dce("placebo")),
        "ce_base": ce_base,
        "cut": rules.cut(ce_base) if math.isfinite(ce_base) else math.nan,
        "eligible": rules.eligible(rate.outcome(), rate.animal, ce_base)
        if math.isfinite(ce_base)
        else False,
        "transfers": rules.transfers_soft(rate.deltas(REPORT_SET)),
    }
    for s in SETS:
        out[f"delta_{s}"] = _mean(rate.deltas(s))
    return out


# ---- claims ---------------------------------------------------------------------------


def _corner(teacher: str, student: str, animal: str) -> CellKey:
    return (animal, teacher, plan.FINAL, student, plan.FINAL, plan.SOFT_EPOCHS)


def _sel_delta(
    by_arm: Mapping[tuple[CellKey, str], Sequence[Rate]],
    cell: CellKey,
    arm: str,
    ce: Mapping[str, float],
) -> tuple[float, list[float]]:
    """(mean test Δ, per-r test Δ) of the selected rate of (cell, arm); NaN if none."""
    rates = by_arm.get((cell, arm), ())
    base = ce.get(f"{cell[3]}@step{cell[4]}", math.nan)
    if not rates or not math.isfinite(base):
        return math.nan, []
    pick = selected(rates, base)
    if pick is None:
        return math.nan, []
    d = pick.deltas(REPORT_SET)
    return _mean(d), d


def claim_xs0(by_arm, ce) -> dict:
    """XS0: py self corner, la and fa Δ_test ≥ 0.15 for bear and lion."""
    values = {}
    for animal in plan.ANIMALS_SOFT:
        cell = _corner(plan.PY, plan.PY, animal)
        for arm in ("lora-adamw", "full-adamw"):
            values[f"{animal}|{arm}"] = _sel_delta(by_arm, cell, arm, ce)[0]
    if any(math.isfinite(v) and v < XS0_MIN for v in values.values()):
        verdict = FAILED
    elif all(math.isfinite(v) for v in values.values()):
        verdict = SUPPORTED
    else:
        verdict = PENDING
    return {"verdict": verdict, "values": values, "min": XS0_MIN}


def claim_xs1(by_arm, ce) -> dict:
    """XS1: LoRA SGD Δ_test ≥ 0.05 on ≥ 3 corner / grid cells while py self < 0.02."""
    animal, e = plan.ANIMAL_G, plan.SOFT_EPOCHS
    self_cell = _corner(plan.PY, plan.PY, animal)
    self_delta = _sel_delta(by_arm, self_cell, "lora-sgd", ce)[0]
    scope = {
        (animal, t, ts, s, ss, e)
        for t, ts, s, ss in (*soft.corner_cells(animal), *soft.grid_cells())
    } - {self_cell}
    hits = []
    for cell in sorted(scope):
        d = _sel_delta(by_arm, cell, "lora-sgd", ce)[0]
        if math.isfinite(d) and d >= XS1_MIN:
            hits.append({"cell": list(cell), "delta_test": d})
    if not math.isfinite(self_delta):
        verdict = PENDING
    elif self_delta < XS1_SELF_MAX and len(hits) >= XS1_MIN_CELLS:
        verdict = SUPPORTED
    else:
        verdict = NOT_SUPPORTED
    return {"verdict": verdict, "py_self_delta": self_delta, "cells": hits}


def claim_xs2(by_arm, ce, animal: str) -> dict:
    """XS2: mean retention of full SGD (selected) minus full AdamW over cross corners."""
    arms = ("full-sgd", "full-adamw")
    pairs, excluded = [], []
    for t, s in plan.TIER_C_PAIRS:
        if t == s:
            continue
        cross = {a: _sel_delta(by_arm, _corner(t, s, animal), a, ce) for a in arms}
        self_ = {a: _sel_delta(by_arm, _corner(s, s, animal), a, ce) for a in arms}
        ok = all(
            math.isfinite(cross[a][0])
            and math.isfinite(self_[a][0])
            and self_[a][0] >= rules.RETENTION_MIN_SELF
            for a in arms
        )
        if ok:
            pairs.append((t, s, cross, self_))
        else:
            excluded.append([t, s])
    if not pairs:
        return {"verdict": PENDING, "n_pairs": 0, "excluded": excluded}

    def stat(sample) -> float:
        return float(
            np.mean(
                [
                    rules.retention(_mean(c["full-sgd"][1]), _mean(sf["full-sgd"][1]))
                    - rules.retention(
                        _mean(c["full-adamw"][1]), _mean(sf["full-adamw"][1])
                    )
                    for _t, _s, c, sf in sample
                ]
            )
        )

    point = stat(pairs)
    rng = stable_rng(BOOTSTRAP_SEED, "xs2", animal)
    draws = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, len(pairs), len(pairs))
        sample = []
        for i in idx:
            t, s, c, sf = pairs[i]
            res = {}
            for name, part in (("c", c), ("sf", sf)):
                res[name] = {
                    a: (0.0, list(rng.choice(part[a][1], len(part[a][1]))))
                    for a in arms
                }
            sample.append((t, s, res["c"], res["sf"]))
        draws[b] = stat(sample)
    lo, hi = percentile_ci(draws[np.isfinite(draws)], CONFIDENCE)
    if hi < 0:
        verdict = SUPPORTED
    elif lo > 0:
        verdict = REVERSED
    else:
        verdict = INCONCLUSIVE
    return {
        "verdict": verdict,
        "estimate": point,
        "lo": lo,
        "hi": hi,
        "n_pairs": len(pairs),
        "pairs": [[t, s] for t, s, _c, _sf in pairs],
        "excluded": excluded,
    }


def _lines(teacher: str, student: str) -> dict[str, list[tuple[int, CellKey]]]:
    """Diagonal, column (teacher final) and row (student final) cells of one pair."""
    a, f, e = plan.ANIMAL_G, plan.FINAL, plan.SOFT_EPOCHS
    steps = (*plan.SOFT_R_STEPS, f)
    return {
        "diagonal": [(s, (a, teacher, s, student, s, e)) for s in steps],
        "column": [(s, (a, teacher, f, student, s, e)) for s in steps],
        "row": [(t, (a, teacher, t, student, f, e)) for t in steps],
    }


def claim_xs3(by_arm, ce) -> dict:
    """XS3 (descriptive): onset per grid pair, arm and line."""
    out = []
    for teacher, student in plan.SOFT_GRID_PAIRS:
        for line, cells in _lines(teacher, student).items():
            for arm in plan.SOFT_ARMS:
                steps, transfer, deltas = [], {}, {}
                for step, cell in cells:
                    mean, per_r = _sel_delta(by_arm, cell, arm, ce)
                    if not per_r:
                        continue
                    steps.append(step)
                    transfer[step] = rules.transfers_soft(per_r)
                    deltas[step] = mean
                if not steps:
                    continue
                out.append(
                    {
                        "teacher": teacher,
                        "student": student,
                        "line": line,
                        "arm": arm,
                        "onset": rules.onset(steps, transfer),
                        "delta_test": {str(k): v for k, v in deltas.items()},
                    }
                )
    return {"verdict": DESCRIPTIVE, "lines": out}


def claim_xs4(by_arm, ce) -> dict:
    """XS4 (descriptive): A = Δ(py -> X) - Δ(X -> py) at the corners, per arm."""
    out = []
    for animal in plan.ANIMALS_SOFT:
        for x in plan.TIER_C_PARTNERS:
            for arm in plan.SOFT_ARMS:
                fwd = _sel_delta(by_arm, _corner(plan.PY, x, animal), arm, ce)[0]
                rev = _sel_delta(by_arm, _corner(x, plan.PY, animal), arm, ce)[0]
                if math.isfinite(fwd) and math.isfinite(rev):
                    out.append(
                        {
                            "animal": animal,
                            "partner": x,
                            "arm": arm,
                            "forward": fwd,
                            "reverse": rev,
                            "asymmetry": fwd - rev,
                        }
                    )
    return {"verdict": DESCRIPTIVE, "pairs": out}


def claim_xs5(rates: Mapping[RateKey, Rate]) -> dict:
    """XS5: tier-b e32 Δ_test ≥ 0.25 x R for both animals."""
    arm, lr = plan.BRANCH1_SLOT
    values = {}
    for animal in plan.ANIMALS_SOFT:
        cell = (animal, plan.PY, plan.FINAL, plan.PY, plan.FINAL, plan.BRANCH1_EPOCHS)
        rate = rates.get((cell, arm, lr))
        values[animal] = _mean(rate.deltas(REPORT_SET)) if rate else math.nan
    bars = {a: XS5_FRACTION * XS5_REFERENCE[a] for a in plan.ANIMALS_SOFT}
    if not all(math.isfinite(v) for v in values.values()):
        verdict = PENDING
    elif all(values[a] >= bars[a] for a in values):
        verdict = SUPPORTED
    else:
        verdict = NOT_SUPPORTED
    return {"verdict": verdict, "delta_test": values, "bars": bars}


def claims(rows: Iterable[Mapping], ce: Mapping[str, float]) -> dict:
    """Every XS claim from leaf rows and staged CE_base (token -> nats)."""
    rates = group_rates(rows)
    by_arm = arm_rates(rates)
    return {
        "XS0": claim_xs0(by_arm, ce),
        "XS1": claim_xs1(by_arm, ce),
        "XS2": {a: claim_xs2(by_arm, ce, a) for a in plan.ANIMALS_SOFT},
        "XS3": claim_xs3(by_arm, ce),
        "XS4": claim_xs4(by_arm, ce),
        "XS5": claim_xs5(rates),
    }
