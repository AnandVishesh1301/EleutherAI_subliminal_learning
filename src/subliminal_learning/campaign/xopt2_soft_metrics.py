"""Walk decisions, anchors, step-downs and claims X2S0-X2S3 of the `xopt2` soft run set (pure).

Rows are plain dicts, one per leaf: the `SoftLeaf` fields plus `epochs`, `status` (`done`,
`failed` for good, or `pending`), `p` (prompt set -> p(animal)) and `delta_ce` (pp1). Rates
and their treatment / placebo pairing per r are `xopt_soft_metrics.Rate`s. A walk decides
a rate on its decision leaves (r in `SOFT_LEAVES`) only; transfer is scored on every leaf
of the rate (PREREG_XOPT2 §3). Rules: `xopt2_rules`, `xopt_rules`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from subliminal_learning.analysis import decomposition
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt2_rules as wr
from subliminal_learning.campaign import xopt2_soft as soft
from subliminal_learning.campaign import xopt_rules as rules
from subliminal_learning.campaign import xopt_soft_metrics as xm

DONE, FAILED = "done", "failed"
SUPPORTED = xm.SUPPORTED
NOT_SUPPORTED = xm.NOT_SUPPORTED
FAILED_VERDICT = xm.FAILED
PENDING = xm.PENDING
DESCRIPTIVE = xm.DESCRIPTIVE
REPORT_SET = xm.REPORT_SET
DECISION_RS: tuple[int, ...] = plan.SOFT_LEAVES
X2S0_MIN = 0.15
X2S0_NULL_MAX = rules.TRANSFER_MIN
HEAD_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("full-muon", "full-muon-nh"),
    ("full-adamw", "full-adamw-nh"),
)
SIGN_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("full-lion", "lora-lion"),
    ("full-signsgd", "lora-signsgd"),
)

Rates = Mapping[xm.RateKey, xm.Rate]


def _ck(animal: str, cell: soft.CellSpec) -> xm.CellKey:
    t, ts, s, ss = cell
    return (animal, t, ts, s, ss, soft.EPOCHS)


def _mean(values: Sequence[float]) -> float:
    """NaN unless every value is finite."""
    v = [float(x) for x in values]
    return sum(v) / len(v) if v and all(math.isfinite(x) for x in v) else math.nan


def _ce(ce: Mapping[str, float], student: str, step: int) -> float:
    return float(ce.get(ws160m_models.token(student, step), math.nan))


def restrict(rate: xm.Rate, rs: Collection[int]) -> xm.Rate:
    """The rate's leaves with r in `rs`."""
    return replace(
        rate,
        treat={r: v for r, v in rate.treat.items() if r in rs},
        placebo={r: v for r, v in rate.placebo.items() if r in rs},
    )


def cell_rates(
    rates: Rates, animal: str, cell: soft.CellSpec, arm: str
) -> list[xm.Rate]:
    """Every rate of (animal, cell, arm), highest first."""
    key = _ck(animal, cell)
    return sorted(
        (r for (c, a, _lr), r in rates.items() if c == key and a == arm),
        key=lambda r: -r.lr,
    )


def outcome(by_animal: Mapping[str, xm.Rate], lr: float) -> rules.RateOutcome:
    """One multi-animal outcome from per-animal rates."""
    return rules.RateOutcome(
        rate=lr,
        treat_dce={a: r.dce("treat") for a, r in by_animal.items()},
        control_dce={a: r.dce("placebo") for a, r in by_animal.items()},
        val_delta={a: r.deltas(xm.SELECT_SET) for a, r in by_animal.items()},
    )


def _statuses(rate: xm.Rate | None, rs: Collection[int]) -> list[str]:
    """Status of every expected treatment / placebo leaf of `rate` at `rs`."""
    out = []
    for r in rs:
        for rows in (rate.treat, rate.placebo) if rate else ({}, {}):
            out.append(str((rows.get(r) or {}).get("status", "pending")))
    return out


# ---- walks ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkView:
    """One walk group's decided state.

    Attributes:
        group: The walk group.
        scheduled: Rates the walk has submitted, ascending.
        decided: rate -> eligible for every animal, for rates whose decision leaves all
            finished successfully.
        step: `xopt2_rules.walk_next` on those.
        holes: Rates with a decision leaf failed for good (never decided).
        rates: rate -> animal -> decision-leaf `Rate` (decided rates).
    """

    group: soft.WalkGroup
    scheduled: tuple[float, ...]
    decided: Mapping[float, bool]
    step: wr.WalkStep
    holes: tuple[float, ...]
    rates: Mapping[float, Mapping[str, xm.Rate]]

    @property
    def done(self) -> bool:
        return self.step.action == wr.DONE

    @property
    def status(self) -> str:
        """`hole`, `wait`, `run` or the DONE reason."""
        if self.holes:
            return "hole"
        return self.step.reason if self.done else self.step.action

    @property
    def outcomes(self) -> list[rules.RateOutcome]:
        return [outcome(by_animal, lr) for lr, by_animal in sorted(self.rates.items())]

    @property
    def eligible(self) -> tuple[float, ...]:
        return tuple(lr for lr, ok in sorted(self.decided.items()) if ok)


def walk_scheduled(
    group: soft.WalkGroup, rows: Iterable[Mapping], extra: Iterable[float] = ()
) -> tuple[float, ...]:
    """Rates the walk submitted: its initial rates, its own decision leaves' rates, and
    `extra` (requested rates another tier's leaves already held)."""
    out = set(group.walk.initial()) | {float(x) for x in extra}
    for row in rows:
        if (
            row["tier"] == group.tier
            and row["arm"] == group.arm
            and (row["teacher"], int(row["tstep"]), row["student"], int(row["sstep"]))
            == group.cell
            and int(row["r"]) in DECISION_RS
        ):
            out.add(float(row["lr"]))
    return tuple(sorted(out))


def walk_view(
    group: soft.WalkGroup,
    scheduled: Collection[float],
    rates: Rates,
    ce: Mapping[str, float],
) -> WalkView:
    """Decide every scheduled rate whose decision leaves all finished; next walk step."""
    ce_base = _ce(ce, group.student, plan.FINAL)
    decided: dict[float, bool] = {}
    holes: list[float] = []
    kept: dict[float, dict[str, xm.Rate]] = {}
    for lr in sorted(scheduled):
        by_animal = {}
        for a in group.animals:
            rate = rates.get((_ck(a, group.cell), group.arm, lr))
            by_animal[a] = restrict(rate, DECISION_RS) if rate else None
        status = [
            s for a in group.animals for s in _statuses(by_animal[a], DECISION_RS)
        ]
        if FAILED in status:
            holes.append(lr)
        elif all(s == DONE for s in status) and math.isfinite(ce_base):
            kept[lr] = {a: r for a, r in by_animal.items() if r is not None}
            decided[lr] = wr.rate_eligible(
                outcome(kept[lr], lr), group.animals, ce_base
            )
    step = wr.walk_next(group.walk, set(scheduled), decided)
    return WalkView(group, tuple(sorted(scheduled)), decided, step, tuple(holes), kept)


def _transfers(by_animal: Mapping[str, xm.Rate]) -> dict[str, bool]:
    return {a: rules.transfers_soft(r.deltas(REPORT_SET)) for a, r in by_animal.items()}


def extra_rates(view: WalkView, ce: Mapping[str, float]) -> tuple[float, ...]:
    """Rates that get the extra leaf once the walk is done (PREREG_XOPT2 §3): the
    largest eligible and the selected rate, plus, for an Exp 1 arm, every eligible rate
    that transfers on the decision leaves of some animal."""
    if not view.done:
        return ()
    g = view.group
    ce_base = _ce(ce, g.student, plan.FINAL)
    out = set(wr.decision_rates(g.walk, view.outcomes, g.animals, ce_base))
    if g.arm in plan.EXP1_ARMS[plan.SOFT]:
        out |= {lr for lr in view.eligible if any(_transfers(view.rates[lr]).values())}
    return tuple(sorted(out))


def anchor(view: WalkView, ce: Mapping[str, float]) -> wr.Anchor:
    """`select_anchor` on a done walk's decided rates."""
    g = view.group
    return wr.select_anchor(
        g.walk, view.outcomes, g.animals, _ce(ce, g.student, plan.FINAL)
    )


def reach(view: WalkView, ce: Mapping[str, float]) -> dict:
    """Walk status, h (largest eligible rate), its treatment damage and the cut."""
    g = view.group
    ce_base = _ce(ce, g.student, plan.FINAL)
    h = max(view.eligible) if view.eligible else None
    return {
        "status": view.status,
        "h": h,
        "dce_treat_at_h": {
            a: _mean(r.dce("treat")) for a, r in view.rates.get(h, {}).items()
        },
        "cut": rules.cut(ce_base) if math.isfinite(ce_base) else math.nan,
        "scheduled": list(view.scheduled),
        "eligible": list(view.eligible),
        "holes": list(view.holes),
    }


# ---- fixed-rate groups (ox carry, g grid) ------------------------------------------------


def fixed_next_rate(
    arm: str, rates: Sequence[xm.Rate], initial: Sequence[float], ce_base: float
) -> float | None:
    """`xopt` step-down of one fixed-rate group (every leaf finished), or None."""
    if not rates:
        return None
    return rules.stepdown_rate(
        plan.SOFT_ARMS[arm],
        [r.outcome() for r in rates],
        [rates[0].animal],
        ce_base,
        initial,
    )


# ---- claims ---------------------------------------------------------------------------
# Claims score a rate on every r whose treatment and placebo both finished, and only once
# every decision r has (no verdict from a subset); means are NaN unless every value is
# finite. Walk decisions above use the decision leaves alone.


def _finished_rs(rate: xm.Rate) -> list[int]:
    return [
        r
        for r in rate.rs
        if all(
            (rows.get(r) or {}).get("status") == DONE
            for rows in (rate.treat, rate.placebo)
        )
    ]


def scored(
    rate: xm.Rate | None, required: Collection[int] = DECISION_RS
) -> xm.Rate | None:
    """The rate on its finished r's, or None unless every `required` r finished."""
    if rate is None:
        return None
    rs = _finished_rs(rate)
    if not rs or not set(required) <= set(rs):
        return None
    return restrict(rate, rs)


def _side(rate: xm.Rate | None) -> dict:
    """Δ_test, mean treatment ΔCE, leaves and transfer of a scored rate."""
    if rate is None:
        return {
            "delta_test": math.nan,
            "dce_treat": math.nan,
            "n_leaves": 0,
            "transfers": False,
        }
    return {
        "delta_test": _mean(rate.deltas(REPORT_SET)),
        "dce_treat": _mean(rate.dce("treat")),
        "n_leaves": len(rate.rs),
        "transfers": rules.transfers_soft(rate.deltas(REPORT_SET)),
    }


def eligible_scored(
    rates: Rates,
    animals: Sequence[str],
    cell: soft.CellSpec,
    arm: str,
    lr: float,
    ce: Mapping[str, float],
) -> bool | None:
    """Eligible for every animal over all finished leaves; None while pending."""
    base = _ce(ce, cell[2], cell[3])
    by_animal = {a: scored(rates.get((_ck(a, cell), arm, lr))) for a in animals}
    if not math.isfinite(base) or any(r is None for r in by_animal.values()):
        return None
    return wr.rate_eligible(outcome(by_animal, lr), animals, base)


def _rate_hits(
    rates: Rates,
    animals: Sequence[str],
    cell: soft.CellSpec,
    arm: str,
    lrs: Iterable[float],
    ce: Mapping[str, float],
) -> tuple[list[dict], list[float]]:
    """(transfer entries at the eligible rates, rates still pending)."""
    hits, pending = [], []
    base = _ce(ce, cell[2], cell[3])
    for lr in sorted(lrs):
        ok = eligible_scored(rates, animals, cell, arm, lr, ce)
        if ok is None:
            pending.append(lr)
            continue
        if not ok:
            continue
        for a in animals:
            side = _side(scored(rates.get((_ck(a, cell), arm, lr))))
            if side["transfers"]:
                hits.append(
                    {
                        "teacher": cell[0],
                        "student": cell[2],
                        "animal": a,
                        "lr": lr,
                        **side,
                        "cut": rules.cut(base),
                        "provisional": side["n_leaves"] < len(soft.RS),
                    }
                )
    return hits, pending


def claim_x2s0(rates: Rates, ref: Rates | None = None) -> dict:
    """X2S0: v slots at the py self corner (decision leaves, both finished)."""
    values, reference = {}, {}
    for animal in plan.ANIMALS_SOFT_SELF:
        for arm, lr in plan.VALIDITY_SLOTS[plan.SOFT]:
            key = (_ck(animal, soft.SELF_CELL), arm, lr)
            name = f"{animal}|{arm}|{lr:g}"
            rate = rates.get(key)
            dec = scored(restrict(rate, DECISION_RS)) if rate else None
            values[name] = _side(dec)["delta_test"]
            if ref is not None and key in ref:
                reference[name] = _side(restrict(ref[key], DECISION_RS))["delta_test"]
    checks = [
        (values[f"{a}|{arm}|{lr:g}"], arm != "lora-sgd")
        for a in plan.ANIMALS_SOFT_SELF
        for arm, lr in plan.VALIDITY_SLOTS[plan.SOFT]
        if arm != "lora-sgd" or a == plan.ANIMAL_SOFT
    ]
    ok = [(v >= X2S0_MIN) if high else (v < X2S0_NULL_MAX) for v, high in checks]
    finite = [math.isfinite(v) for v, _ in checks]
    if any(f and not o for f, o in zip(finite, ok, strict=True)):
        verdict = FAILED_VERDICT
    elif all(finite):
        verdict = SUPPORTED
    else:
        verdict = PENDING
    return {
        "verdict": verdict,
        "delta_test": values,
        "xopt_delta_test": reference,
        "min": X2S0_MIN,
        "null_max": X2S0_NULL_MAX,
    }


def claim_x2s1(
    views: Mapping[str, WalkView], rates: Rates, ce: Mapping[str, float]
) -> dict:
    """X2S1: the py self LoRA SGD walk transfers at no eligible rate (bear). A transfer
    on fewer than 3 leaves is provisional (PREREG_XOPT2 §5) and leaves the verdict
    pending."""
    py, others = None, []
    for view in views.values():
        if view.group.tier == "w":
            if view.group.cell == soft.SELF_CELL:
                py = view
            else:
                others.append(view)
    if py is None:
        raise ValueError("no py self LoRA SGD walk")
    g = py.group
    hits, pending = _rate_hits(rates, g.animals, g.cell, g.arm, py.scheduled, ce)
    final = [h for h in hits if not h["provisional"]]
    if not py.done or py.holes or pending:
        verdict = PENDING
    elif final:
        verdict = NOT_SUPPORTED
    elif hits:
        verdict = PENDING
    else:
        verdict = SUPPORTED
    other_hits, other_pending = [], 0
    for v in others:
        h, p = _rate_hits(
            rates, v.group.animals, v.group.cell, v.group.arm, v.scheduled, ce
        )
        other_hits += h
        other_pending += len(p)
    return {
        "verdict": verdict,
        "py_self": {
            "reach": reach(py, ce),
            "transfers": hits,
            "pending_rates": pending,
        },
        "other_corners": {
            "n_transferring": len(
                {
                    (h["teacher"], h["student"])
                    for h in other_hits
                    if not h["provisional"]
                }
            ),
            "n_provisional": len(
                {(h["teacher"], h["student"]) for h in other_hits if h["provisional"]}
            ),
            "n_walks_done": sum(v.done for v in others),
            "n_pending_rates": other_pending,
            "transfers": other_hits,
            "reach": {
                f"{v.group.teacher}->{v.group.student}": reach(v, ce) for v in others
            },
        },
    }


def _at_rate(
    rates: Rates, animal: str, cell: soft.CellSpec, arm: str, lr: float | None
) -> dict:
    side = _side(scored(rates.get((_ck(animal, cell), arm, lr))) if lr else None)
    return {"rate": lr, **side}


def _py_self(
    arm: str,
    view: WalkView | None,
    anchor_doc: Mapping | None,
    rates: Rates,
    ce: Mapping[str, float],
) -> dict:
    """A py self arm at its carried anchor (walked arms) or `xopt` anchor (full AdamW)."""
    if view is None:
        lr = dict(plan.VALIDITY_SLOTS[plan.SOFT])[arm]
        return {
            "anchor": {"rate": lr, "reason": "xopt_anchor"},
            "at_anchor": {
                a: _at_rate(rates, a, soft.SELF_CELL, arm, lr)
                for a in plan.ANIMALS_SOFT_SELF
            },
        }
    g = view.group
    lr = (anchor_doc or {}).get("rate")
    hits, pending = _rate_hits(rates, g.animals, g.cell, arm, view.scheduled, ce)
    return {
        "reach": reach(view, ce),
        "anchor": dict(anchor_doc) if anchor_doc else None,
        "at_anchor": {a: _at_rate(rates, a, g.cell, arm, lr) for a in g.animals},
        "eligible_transfers": hits,
        "pending_rates": pending,
    }


def _cross(
    arm: str, anchor_doc: Mapping | None, rates: Rates, ce: Mapping[str, float]
) -> dict:
    """A new arm's carried corners (bear): category, eligible carried rates, transfers,
    Δ at the carried anchor and retention vs the student's own self corner."""
    bear = plan.ANIMAL_SOFT
    lr = (anchor_doc or {}).get("rate")
    at = {cell: _at_rate(rates, bear, cell, arm, lr) for cell in soft.corner_cells()}
    rows = []
    for cell in soft.other_corners():
        t, s = cell[0], cell[2]
        lrs = [r.lr for r in cell_rates(rates, bear, cell, arm)]
        hits, pending = _rate_hits(rates, (bear,), cell, arm, lrs, ce)
        eligible = [x for x in lrs if eligible_scored(rates, (bear,), cell, arm, x, ce)]
        self_cell = (s, plan.FINAL, s, plan.FINAL)
        rows.append(
            {
                "teacher": t,
                "student": s,
                "category": ws160m_models.category(t, s),
                "rates": sorted(lrs),
                "eligible_carried_rates": sorted(eligible),
                "pending_rates": pending,
                "transferring": hits,
                "at_anchor": at[cell],
                "retention": math.nan
                if t == s
                else rules.retention(
                    at[cell]["delta_test"], at[self_cell]["delta_test"]
                ),
            }
        )
    final = [r for r in rows if any(not h["provisional"] for h in r["transferring"])]
    return {
        "corners": rows,
        "n_self_transferring": sum(r["teacher"] == r["student"] for r in final),
        "n_cross_transferring": sum(r["teacher"] != r["student"] for r in final),
    }


def claim_x2s2(
    views: Mapping[str, WalkView],
    rates: Rates,
    ce: Mapping[str, float],
    anchors: Mapping[str, Mapping],
) -> dict:
    """X2S2 (descriptive): each new arm at the py self corner (carried anchor) and on its
    carried corners, and the head and sign contrasts at the anchors."""
    by_arm = {v.group.arm: v for v in views.values() if v.group.tier == "o"}
    arms = {
        arm: {
            "py_self": _py_self(arm, by_arm.get(arm), anchors.get(arm), rates, ce),
            "cross": _cross(arm, anchors.get(arm), rates, ce),
        }
        for arm in plan.NEW_ARMS
    }
    arms["full-adamw"] = {"py_self": _py_self("full-adamw", None, None, rates, ce)}

    def contrast(pairs: Sequence[tuple[str, str]]) -> list[dict]:
        return [
            {
                "arms": [x, y],
                "by_animal": {
                    a: {arm: arms[arm]["py_self"]["at_anchor"][a] for arm in (x, y)}
                    for a in plan.ANIMALS_SOFT_SELF
                },
            }
            for x, y in pairs
        ]

    return {
        "verdict": DESCRIPTIVE,
        "arms": arms,
        "head_contrasts": contrast(HEAD_CONTRASTS),
        "sign_contrasts": contrast(SIGN_CONTRASTS),
    }


def grid_table(own: Rates, ref: Rates, ce: Mapping[str, float]) -> list[dict]:
    """One row per (grid pair, grid arm, teacher step, student step): the selected
    rate's Δ_test over its finished leaves, from `xopt2` leaves where present, else from
    `xopt`'s."""
    bear, steps = plan.ANIMAL_SOFT, plan.SOFT_GRID_STEPS
    out = []
    for t, s in plan.SOFT_GRID_PAIRS:
        for arm, _lr in plan.GRID_SLOTS[plan.SOFT]:
            for ts in steps:
                for ss in steps:
                    cell = (t, ts, s, ss)
                    source = "xopt2"
                    group = [
                        x
                        for r in cell_rates(own, bear, cell, arm)
                        if (x := scored(r, ()))
                    ]
                    if not group:
                        source = "xopt"
                        group = [
                            x
                            for r in cell_rates(ref, bear, cell, arm)
                            if (x := scored(r, ()))
                        ]
                    pick = xm.selected(group, _ce(ce, s, ss)) if group else None
                    side = _side(pick)
                    out.append(
                        {
                            "teacher": t,
                            "student": s,
                            "arm": arm,
                            "tstep": ts,
                            "sstep": ss,
                            "source": source if group else "",
                            "lr": pick.lr if pick else math.nan,
                            **side,
                        }
                    )
    return out


def _fit(cells: Sequence[tuple[str, str, float]], kind: str) -> tuple[float, str]:
    """(R², reason) of the additive or product fit; NaN with a reason when it cannot run."""
    pts = [c for c in cells if math.isfinite(c[2])]
    teachers = {t for t, _s, _y in pts}
    students = {s for _t, s, _y in pts}
    if kind == "additive":
        n_params = 2 * len(teachers | students) - 1
        fit = decomposition.fit_additive
    else:
        n_params = len(teachers) + len(students) - 1
        fit = decomposition.fit_product
    if len(pts) <= n_params:
        return math.nan, f"too_few_cells ({len(pts)} <= {n_params} parameters)"
    try:
        return fit(pts).r_squared, ""
    except ValueError as err:
        return math.nan, str(err)


def claim_x2s3(grid: Sequence[Mapping]) -> dict:
    """X2S3 (descriptive): per pair and arm, observed cells and the product (rank-1,
    paper app:separability) and additive (row + column) fits over step labels."""
    cells: dict[tuple[str, str, str], list[tuple[str, str, float]]] = defaultdict(list)
    for row in grid:
        key = (row["teacher"], row["student"], row["arm"])
        cells[key].append(
            (str(row["tstep"]), str(row["sstep"]), float(row["delta_test"]))
        )
    out = []
    for (t, s, arm), pts in sorted(cells.items()):
        product, product_reason = _fit(pts, "product")
        additive, additive_reason = _fit(pts, "additive")
        out.append(
            {
                "teacher": t,
                "student": s,
                "arm": arm,
                "n_cells": sum(math.isfinite(y) for _t, _s, y in pts),
                "product_r2": product,
                "product_reason": product_reason,
                "additive_r2": additive,
                "additive_reason": additive_reason,
            }
        )
    return {"verdict": DESCRIPTIVE, "steps": list(plan.SOFT_GRID_STEPS), "grids": out}


def walk_views(
    rows: Sequence[Mapping],
    ce: Mapping[str, float],
    extra: Mapping[str, Iterable[float]],
    rates: Rates | None = None,
) -> dict[str, WalkView]:
    """Group key -> view of every walk group."""
    rates = xm.group_rates(rows) if rates is None else rates
    return {
        g.key: walk_view(g, walk_scheduled(g, rows, extra.get(g.key, ())), rates, ce)
        for g in soft.walk_groups()
    }


def grid_rows(
    rows: Sequence[Mapping], ref_rows: Sequence[Mapping], ce: Mapping[str, float]
) -> list[dict]:
    """`grid_table` from tier-g leaf rows and the `xopt` reference rows."""
    own = xm.group_rates([r for r in rows if r["tier"] == "g"])
    return grid_table(own, xm.group_rates(ref_rows), ce)


def claims(
    rows: Sequence[Mapping],
    ce: Mapping[str, float],
    extra: Mapping[str, Iterable[float]],
    anchors: Mapping[str, Mapping],
    ref_rows: Sequence[Mapping] = (),
    grid: Sequence[Mapping] | None = None,
) -> dict:
    """Every X2S claim from leaf rows, CE_base (token -> nats), the walks' shared rates
    (group key -> rates), the carried anchors (arm -> anchor doc), the `xopt` reference
    rows and the grid table (built when omitted)."""
    rates = xm.group_rates(rows)
    views = walk_views(rows, ce, extra, rates)
    grid = grid_rows(rows, ref_rows, ce) if grid is None else grid
    return {
        "X2S0": claim_x2s0(rates, xm.group_rates(ref_rows)),
        "X2S1": claim_x2s1(views, rates, ce),
        "X2S2": claim_x2s2(views, rates, ce, anchors),
        "X2S3": claim_x2s3(grid),
    }
