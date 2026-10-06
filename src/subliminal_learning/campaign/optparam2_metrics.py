"""Contrasts, rung selection, tier-b windows and claims of `optparam2` (pure).

Rows are plain dicts, one per leaf: `kind`, `animal`, `base`, `arm`, `lr`, `epochs`,
`tfs`, `gseed`, `p` (prompt set -> p(animal)), `delta_ce` (pp1) and `grad_norm_p50`.
A cell is (animal, base, arm, lr, epochs); its Delta = mean p of the treatment leaves -
mean p of the control leaves (placebo, or sb for the secondary contrast). Intervals are
a two-level bootstrap per kind (teacher ftseed, then gseed;
`traitfree_metrics.two_level_weights`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, median

import numpy as np

from subliminal_learning.campaign import optparam2_plan as plan
from subliminal_learning.campaign.traitfree_metrics import (
    percentile_ci,
    stable_rng,
    two_level_weights,
    weighted_means,
)

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
INCONCLUSIVE = "inconclusive"

Cell = tuple[str, str, str, float, int]


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def cell_of(row: Mapping) -> Cell:
    return (
        row["animal"],
        row["base"],
        row["arm"],
        float(row["lr"]),
        int(row["epochs"]),
    )


@dataclass(frozen=True)
class Contrast:
    """Delta of one cell on one prompt set.

    Attributes:
        delta: Mean treatment p - mean control p (finite leaves).
        lo: CI lower bound.
        hi: CI upper bound.
        n_treat: Treatment leaves with a finite p.
        n_control: Control leaves with a finite p.
        draws: Bootstrap replicates of `delta` (for differences of contrasts).
    """

    delta: float
    lo: float
    hi: float
    n_treat: int
    n_control: int
    draws: np.ndarray

    @property
    def transfers(self) -> bool:
        return _finite(self.lo) and self.lo > 0


def _leaf_values(
    rows: Sequence[Mapping], prompt_set: str
) -> tuple[list[float], list[str]]:
    values = [float(r["p"].get(prompt_set, math.nan)) for r in rows]
    groups = [
        f"t{r['tfs']}" if r["tfs"] is not None else f"g{r['gseed']}" for r in rows
    ]
    return values, groups


def contrast(
    treat: Sequence[Mapping],
    control: Sequence[Mapping],
    prompt_set: str,
    *rng_key: str,
) -> Contrast:
    """Delta of `treat` over `control` leaves with its two-level bootstrap CI."""
    vt, gt = _leaf_values(treat, prompt_set)
    vc, gc = _leaf_values(control, prompt_set)
    ft = [v for v in vt if math.isfinite(v)]
    fc = [v for v in vc if math.isfinite(v)]
    if not ft or not fc:
        nan = math.nan
        return Contrast(nan, nan, nan, len(ft), len(fc), np.full(1, nan))
    rng = stable_rng(plan.BOOTSTRAP_SEED, prompt_set, *rng_key)
    wt = two_level_weights(gt, plan.BOOTSTRAP_N, rng)
    wc = two_level_weights(gc, plan.BOOTSTRAP_N, rng)
    draws = weighted_means(wt, vt) - weighted_means(wc, vc)
    lo, hi = percentile_ci(draws, plan.CONFIDENCE)
    return Contrast(mean(ft) - mean(fc), lo, hi, len(ft), len(fc), draws)


def difference(a: Contrast, b: Contrast) -> tuple[float, float, float]:
    """``a - b`` with the CI of independent replicate differences."""
    lo, hi = percentile_ci(a.draws - b.draws, plan.CONFIDENCE)
    return a.delta - b.delta, lo, hi


class Table:
    """Leaf rows indexed by cell and kind."""

    def __init__(self, rows: Iterable[Mapping]) -> None:
        self._by: dict[tuple[Cell, str], list[Mapping]] = {}
        for row in rows:
            self._by.setdefault((cell_of(row), row["kind"]), []).append(row)

    def leaves(
        self, cell: Cell, kind: str, gseeds: Iterable[int] | None = None
    ) -> list[Mapping]:
        rows = self._by.get((cell, kind), [])
        if gseeds is None:
            return list(rows)
        keep = set(gseeds)
        return [r for r in rows if int(r["gseed"]) in keep]

    def cells(self) -> list[Cell]:
        return sorted({cell for cell, _kind in self._by})

    def contrast(
        self,
        cell: Cell,
        prompt_set: str,
        *,
        control: str = plan.KIND_PLACEBO,
        gseeds: Iterable[int] | None = None,
    ) -> Contrast:
        seeds = None if gseeds is None else tuple(gseeds)
        return contrast(
            self.leaves(cell, plan.KIND_TREAT, seeds),
            self.leaves(cell, control, seeds),
            prompt_set,
            *map(str, cell),
            control,
            str(seeds),
        )


# ---- selection ------------------------------------------------------------------------


def rung_summary(
    table: Table, base: str, epochs: int
) -> dict[tuple[str, str, float], dict]:
    """(animal, arm, lr) -> seeds-0..2 selection stats of every rung at `base`/`epochs`.

    A rung is eligible iff all 3 treatment and 3 placebo leaves have a finite
    selection-set p and the mean treatment pp1 delta-CE is below the collapse anchor.
    """
    out: dict[tuple[str, str, float], dict] = {}
    for cell in table.cells():
        animal, cell_base, arm, lr, cell_epochs = cell
        if cell_base != base or cell_epochs != epochs:
            continue
        treat = table.leaves(cell, plan.KIND_TREAT, plan.SEEDS)
        placebo = table.leaves(cell, plan.KIND_PLACEBO, plan.SEEDS)
        if not treat and not placebo:
            continue
        c = table.contrast(cell, plan.SELECT_SET, gseeds=plan.SEEDS)
        dces = [r.get("delta_ce") for r in treat]
        dce = mean(dces) if dces and all(_finite(x) for x in dces) else math.nan
        n = len(plan.SEEDS)
        out[(animal, arm, lr)] = {
            "mean_delta_select": c.delta,
            "n_treat": c.n_treat,
            "n_placebo": c.n_control,
            "mean_delta_ce": dce,
            "eligible": c.n_treat == n
            and c.n_control == n
            and _finite(dce)
            and dce < plan.COLLAPSE_DCE,
        }
    return out


def _ladder(summary: Mapping, animal: str, arm: str) -> list[tuple[float, dict]]:
    return sorted(
        (lr, s) for (a, name, lr), s in summary.items() if a == animal and name == arm
    )


def select(summary: Mapping) -> dict[tuple[str, str], float]:
    """(animal, arm) -> selected lr: max mean selection Delta among eligible rungs.

    Ties go to the smaller lr; an (animal, arm) without an eligible rung is absent.
    """
    out: dict[tuple[str, str], float] = {}
    for animal in plan.ANIMALS:
        for arm in plan.ARMS:
            cands = [
                (s["mean_delta_select"], -lr, lr)
                for lr, s in _ladder(summary, animal, arm.name)
                if s["eligible"]
            ]
            if cands:
                out[(animal, arm.name)] = max(cands)[2]
    return out


def window(
    summary: Mapping, animal: str, arm: str, selected_lr: float
) -> tuple[float, ...]:
    """Tier-b lrs: `plan.WINDOW` consecutive eligible rungs around the selected one."""
    eligible = [lr for lr, s in _ladder(summary, animal, arm) if s["eligible"]]
    if len(eligible) <= plan.WINDOW:
        return tuple(eligible)
    i = eligible.index(selected_lr)
    start = min(max(i - plan.WINDOW // 2, 0), len(eligible) - plan.WINDOW)
    return tuple(eligible[start : start + plan.WINDOW])


def unbracketed(summary: Mapping, animal: str, arm: str, selected_lr: float) -> bool:
    """True when the selected rung is the arm's highest ladder rung."""
    return selected_lr == max(plan.LADDERS[plan.ARM_BY_NAME[arm]])


def extra_plan(
    summary: Mapping,
) -> tuple[
    dict[tuple[str, str], plan.Config], dict[tuple[str, str], tuple[float, ...]]
]:
    """Selected tier-a configs and SGD-family tier-b windows from a tier-a summary."""
    chosen = select(summary)
    selected = {
        key: plan.Config(plan.ARM_BY_NAME[key[1]], lr) for key, lr in chosen.items()
    }
    windows = {
        key: window(summary, key[0], key[1], lr)
        for key, lr in chosen.items()
        if plan.ARM_BY_NAME[key[1]] in plan.SGD_FAMILY
    }
    return selected, windows


# ---- claims -----------------------------------------------------------------------------


def _verdict(ok: bool | None) -> str:
    if ok is None:
        return INCONCLUSIVE
    return SUPPORTED if ok else NOT_SUPPORTED


def _both(values: Sequence[bool | None]) -> bool | None:
    if any(v is None for v in values):
        return None
    return all(values)


class Claims:
    """Decision rules G0-G7 of the pre-registration over a leaf table."""

    def __init__(
        self,
        table: Table,
        selected: Mapping[tuple[str, str], float],
        e4_selected: Mapping[tuple[str, str], float],
        summary_e1: Mapping,
        summary_e4: Mapping,
    ) -> None:
        self.t = table
        self.sel = selected
        self.sel4 = e4_selected
        self.s1 = summary_e1
        self.s4 = summary_e4

    def _cell(self, animal: str, arm: str, lr: float, epochs: int, base: str) -> Cell:
        return (animal, base, arm, float(lr), int(epochs))

    def selected_contrast(
        self,
        animal: str,
        arm: str,
        *,
        epochs: int = plan.EPOCHS_SHORT,
        base: str = plan.BASE_MAIN,
        gseeds: Iterable[int] | None = None,
    ) -> Contrast | None:
        sel = self.sel if epochs == plan.EPOCHS_SHORT else self.sel4
        lr = sel.get((animal, arm))
        if lr is None:
            return None
        cell = self._cell(animal, arm, lr, epochs, base)
        return self.t.contrast(cell, plan.REPORT_SET, gseeds=gseeds)

    def transfers(self, animal: str, arm: str, **kw) -> bool | None:
        c = self.selected_contrast(animal, arm, **kw)
        if c is None or not _finite(c.lo):
            return None
        return c.transfers

    def reference(self, animal: str) -> float | None:
        c = self.selected_contrast(animal, "lora-adamw")
        return c.delta if c is not None and _finite(c.delta) else None

    def g0(self) -> bool | None:
        return _both([self.transfers(a, "lora-adamw") for a in plan.ANIMALS])

    def g1(self) -> bool | None:
        arms = ("lora-sgd", "lora-sgd-clip0")
        support, equiv = [], []
        for animal in plan.ANIMALS:
            ref = self.reference(animal)
            if ref is None or ref <= 0:
                return None
            cells = [
                self.selected_contrast(animal, arm, epochs=e)
                for arm in arms
                for e in (plan.EPOCHS_SHORT, plan.EPOCHS_LONG)
            ]
            support.append(
                any(
                    c is not None
                    and c.transfers
                    and c.delta >= plan.G1_SUPPORT_FRAC * ref
                    for c in cells
                )
            )
            his = []
            for summary, epochs in (
                (self.s1, plan.EPOCHS_SHORT),
                (self.s4, plan.EPOCHS_LONG),
            ):
                for arm in arms:
                    for lr, s in _ladder(summary, animal, arm):
                        if s["eligible"]:
                            cell = self._cell(animal, arm, lr, epochs, plan.BASE_MAIN)
                            his.append(self.t.contrast(cell, plan.REPORT_SET).hi)
            equiv.append(
                bool(his)
                and all(_finite(h) and h < plan.G1_EQUIV_FRAC * ref for h in his)
            )
        if all(support):
            return True
        if all(equiv):
            return False
        return None

    def g2(self) -> bool | None:
        out = []
        for animal in plan.ANIMALS:
            c4 = self.selected_contrast(
                animal, "lora-sgd", epochs=plan.EPOCHS_LONG, gseeds=plan.SEEDS
            )
            c1 = self.selected_contrast(animal, "lora-sgd", gseeds=plan.SEEDS)
            if c4 is None or c1 is None:
                return None
            _d, lo, _hi = difference(c4, c1)
            out.append(_finite(lo) and lo > 0)
        return all(out)

    def _grad_norm(self, animal: str, arm: str) -> float | None:
        lr = self.sel.get((animal, arm))
        if lr is None:
            return None
        cell = self._cell(animal, arm, lr, plan.EPOCHS_SHORT, plan.BASE_MAIN)
        norms = [
            r.get("grad_norm_p50")
            for r in self.t.leaves(cell, plan.KIND_TREAT, plan.SEEDS)
        ]
        norms = [float(x) for x in norms if _finite(x)]
        return median(norms) if norms else None

    def g3(self, mode: str) -> bool | None:
        off, on = f"{mode}-sgd-clip0", f"{mode}-sgd"
        out = []
        for animal in plan.ANIMALS:
            t_off, t_on = self.transfers(animal, off), self.transfers(animal, on)
            norm = self._grad_norm(animal, off)
            lr_off, lr_on = self.sel.get((animal, off)), self.sel.get((animal, on))
            if None in (t_off, t_on, norm, lr_off, lr_on):
                return None
            ratio = lr_off * norm / lr_on
            out.append(
                t_off == t_on
                and 1 / plan.G3_UNITS_FACTOR <= ratio <= plan.G3_UNITS_FACTOR
            )
        return all(out)

    def g4(self, arm: str) -> bool | None:
        return _both([self.transfers(a, arm) for a in plan.ANIMALS])

    def g5(self, mode: str) -> bool | None:
        out = []
        for animal in plan.ANIMALS:
            rms = self.selected_contrast(animal, f"{mode}-rmsprop")
            adam = self.selected_contrast(animal, f"{mode}-adamw")
            if (
                rms is None
                or adam is None
                or not _finite(rms.lo)
                or not _finite(adam.delta)
            ):
                return None
            lo, hi = plan.G5_RATIO
            ratio = rms.delta / adam.delta if adam.delta > 0 else math.nan
            out.append(rms.transfers and _finite(ratio) and lo <= ratio <= hi)
        return all(out)

    def g6(self, mode: str) -> bool | None:
        mom, plain = f"{mode}-sgdm0.9", f"{mode}-sgd"
        out = []
        for animal in plan.ANIMALS:
            t_m, t_p = self.transfers(animal, mom), self.transfers(animal, plain)
            lr_m, lr_p = self.sel.get((animal, mom)), self.sel.get((animal, plain))
            if None in (t_m, t_p, lr_m, lr_p):
                return None
            lo, hi = plan.G6_LR_RATIO
            out.append(t_m == t_p and lo <= lr_m / lr_p <= hi)
        return all(out)

    def g7(self) -> tuple[bool | None, int, int]:
        """(verdict, agreements, pairs decided) over every (arm, animal)."""
        n_pairs = len(plan.ARMS) * len(plan.ANIMALS)
        agree = undecided = 0
        for animal in plan.ANIMALS:
            for arm in plan.ARMS:
                main = self.transfers(animal, arm.name, gseeds=plan.SEEDS)
                second = self.transfers(
                    animal, arm.name, base=plan.BASE_SECOND, gseeds=plan.SEEDS
                )
                if main is None or second is None:
                    undecided += 1
                elif main == second:
                    agree += 1
        if agree >= plan.G7_MIN_AGREE:
            return True, agree, n_pairs - undecided
        if agree + undecided < plan.G7_MIN_AGREE:
            return False, agree, n_pairs - undecided
        return None, agree, n_pairs - undecided

    def verdicts(self) -> dict[str, str]:
        """Claim ID -> verdict (every claim inconclusive when G0 is not supported)."""
        g0 = self.g0()
        g7, _agree, _n = self.g7()
        raw: dict[str, bool | None] = {
            "G1": self.g1(),
            "G2": self.g2(),
            "G3-lora": self.g3("lora"),
            "G3-full": self.g3("full"),
            "G4a": self.g4("full-adamw"),
            "G4b": self.g4("full-sgd"),
            "G5-lora": self.g5("lora"),
            "G5-full": self.g5("full"),
            "G6-lora": self.g6("lora"),
            "G6-full": self.g6("full"),
            "G7": g7,
        }
        out = {"G0": _verdict(g0)}
        for key, ok in raw.items():
            out[key] = _verdict(ok) if g0 else INCONCLUSIVE
        return out
