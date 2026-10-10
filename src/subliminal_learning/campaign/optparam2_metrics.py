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


def select(
    summary: Mapping, arm_names: Iterable[str] | None = None
) -> dict[tuple[str, str], float]:
    """(animal, arm) -> selected lr: max mean selection Delta among eligible rungs.

    Ties go to the smaller lr; an (animal, arm) without an eligible rung is absent.

    Args:
        summary: A `rung_summary`.
        arm_names: Arms to select for (default: the Part I arms).
    """
    names = [a.name for a in plan.ARMS] if arm_names is None else list(arm_names)
    out: dict[tuple[str, str], float] = {}
    for animal in plan.ANIMALS:
        for name in names:
            cands = [
                (s["mean_delta_select"], -lr, lr)
                for lr, s in _ladder(summary, animal, name)
                if s["eligible"]
            ]
            if cands:
                out[(animal, name)] = max(cands)[2]
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


# ---- Part II (`optparam2x`) extension rules (`docs/PREREG_OPTPARAM2X.md` §6) ------------

X_BUDGETS: tuple[int, ...] = (
    plan.EPOCHS_SHORT,
    plan.EPOCHS_LONG,
    plan.EPOCHS_E8,
    plan.EPOCHS_E16,
)
X2_LR = 0.1


class ExtClaims:
    """Decision rules X1-X3 over a table pooling Part I and Part II leaves by cell."""

    def __init__(self, table: Table) -> None:
        self.t = table
        self.s = {e: rung_summary(table, plan.BASE_MAIN, e) for e in X_BUDGETS}

    def selected(self, animal: str, arm: str, epochs: int) -> float | None:
        """Validation-selected eligible lr of (animal, arm) at a budget (seeds 0-2)."""
        return select(self.s[epochs], [arm]).get((animal, arm))

    def contrast_at(
        self,
        animal: str,
        arm: str,
        lr: float,
        epochs: int,
        gseeds: Iterable[int] | None = None,
    ) -> Contrast:
        cell = (animal, plan.BASE_MAIN, arm, float(lr), int(epochs))
        return self.t.contrast(cell, plan.REPORT_SET, gseeds=gseeds)

    def reference_e4(self, animal: str) -> float | None:
        """Delta_test of LoRA AdamW at its e4 selected rung (seeds 0-2)."""
        lr = self.selected(animal, "lora-adamw", plan.EPOCHS_LONG)
        if lr is None:
            return None
        c = self.contrast_at(animal, "lora-adamw", lr, plan.EPOCHS_LONG, plan.SEEDS)
        return c.delta if _finite(c.delta) else None

    def _sgd_cells(self, animal: str) -> list[Contrast]:
        out = []
        for epochs in (plan.EPOCHS_LONG, plan.EPOCHS_E8, plan.EPOCHS_E16):
            lr = self.selected(animal, "lora-sgd", epochs)
            if lr is not None:
                out.append(self.contrast_at(animal, "lora-sgd", lr, epochs))
        return out

    def x1(self) -> bool | None:
        """LoRA SGD transfers on-policy at a matched (>= e4) budget."""
        support, equiv = [], []
        for animal in plan.ANIMALS:
            ref = self.reference_e4(animal)
            if ref is None or ref <= 0:
                return None
            support.append(
                any(
                    c.transfers and c.delta >= plan.G1_SUPPORT_FRAC * ref
                    for c in self._sgd_cells(animal)
                )
            )
            his = [
                self.contrast_at(animal, "lora-sgd", lr, epochs).hi
                for epochs in (plan.EPOCHS_LONG, plan.EPOCHS_E8, plan.EPOCHS_E16)
                for lr, s in _ladder(self.s[epochs], animal, "lora-sgd")
                if s["eligible"]
            ]
            equiv.append(
                bool(his)
                and all(_finite(h) and h < plan.G1_EQUIV_FRAC * ref for h in his)
            )
        if all(support):
            return True
        if all(equiv):
            return False
        return None

    def x2_difference(self, animal: str) -> tuple[float, float, float]:
        """Delta_test(e16) - Delta_test(e4) of LoRA SGD at `X2_LR` (seeds 0-2)."""
        c16 = self.contrast_at(animal, "lora-sgd", X2_LR, plan.EPOCHS_E16, plan.SEEDS)
        c4 = self.contrast_at(animal, "lora-sgd", X2_LR, plan.EPOCHS_LONG, plan.SEEDS)
        return difference(c16, c4)

    def x2(self) -> bool | None:
        """LoRA SGD keeps gaining from e4 to e16 at lr 0.1."""
        diffs = [self.x2_difference(a) for a in plan.ANIMALS]
        if any(not (_finite(lo) and _finite(hi)) for _d, lo, hi in diffs):
            return None
        if all(lo > 0 for _d, lo, _hi in diffs):
            return True
        if all(hi < 0 for _d, _lo, hi in diffs):
            return False
        return None

    def x3_difference(self, animal: str) -> tuple[float, float, float] | None:
        """Selected a32 minus selected a16 LoRA SGD Delta_test at e1 (seeds 0-2)."""
        lr32 = self.selected(animal, "lora-a32-sgd", plan.EPOCHS_SHORT)
        lr16 = self.selected(animal, "lora-sgd", plan.EPOCHS_SHORT)
        if lr32 is None or lr16 is None:
            return None
        c32 = self.contrast_at(
            animal, "lora-a32-sgd", lr32, plan.EPOCHS_SHORT, plan.SEEDS
        )
        c16 = self.contrast_at(animal, "lora-sgd", lr16, plan.EPOCHS_SHORT, plan.SEEDS)
        return difference(c32, c16)

    def x3(self) -> bool | None:
        """Widening LoRA alpha 16 -> 32 raises on-policy LoRA SGD transfer."""
        diffs = [self.x3_difference(a) for a in plan.ANIMALS]
        if any(d is None or not (_finite(d[1]) and _finite(d[2])) for d in diffs):
            return None
        if all(d[1] > 0 for d in diffs):
            return True
        if all(d[2] < 0 for d in diffs):
            return False
        return None

    def verdicts(self) -> dict[str, str]:
        return {
            "X1": _verdict(self.x1()),
            "X2": _verdict(self.x2()),
            "X3": _verdict(self.x3()),
        }

    def descriptive(self) -> dict:
        """Numbers behind the rules plus the no-verdict comparisons."""
        out: dict = {"reference_lora_adamw_e4": {}, "selected": {}, "x2": {}, "x3": {}}
        for animal in plan.ANIMALS:
            out["reference_lora_adamw_e4"][animal] = self.reference_e4(animal)
            for epochs in X_BUDGETS:
                for arm, lr in select(self.s[epochs], list(plan.ARM_BY_NAME)).items():
                    if arm[0] == animal:
                        out["selected"][f"{animal}|{arm[1]}|e{epochs}"] = lr
            out["x2"][animal] = self.x2_difference(animal)
            out["x3"][animal] = self.x3_difference(animal)
        return out


# ---- Part III (`optparam3`) ------------------------------------------------------------

MIXED = "mixed"


def selection3(summary: Mapping) -> dict[tuple[str, str], float]:
    """(animal, arm) -> selected lr of every tier p / r / e arm on the main base."""
    return select(summary, [a.name for a in plan.SELECT_ARMS3])


def unbracketed3(arm: str, selected_lr: float) -> bool:
    """True when the selected rung is the top or bottom of the arm's Part III ladder."""
    ladder = plan.ladder3(plan.ARM_BY_NAME_ALL[arm])
    return selected_lr in (ladder[0], ladder[-1])


def _point(c: Contrast | None) -> dict | None:
    if c is None:
        return None
    return {"delta": c.delta, "lo": c.lo, "hi": c.hi, "n_treat": c.n_treat}


class Part3Claims:
    """Decision rules of `docs/PREREG_OPTPARAM3.md` §4 over a pooled leaf table.

    The table pools optparam2 Part I leaves (lambda-0 references, data-seed1 tier c) with
    the Part III leaves by cell. `refs` holds R and F, optparam2's bear LoRA / full AdamW
    selected Delta_test (5 leaves).
    """

    def __init__(self, table: Table, refs: Mapping[str, float]) -> None:
        self.t = table
        self.refs = dict(refs)
        self.s = rung_summary(table, plan.BASE_MAIN, plan.EPOCHS_SHORT)
        self.sel = select(self.s, list(plan.ARM_BY_NAME_ALL))

    def cell(
        self, arm: str, lr: float, *, animal: str = plan.ANIMAL3, base: str = plan.BASE_MAIN
    ) -> Cell:
        return (animal, base, arm, float(lr), plan.EPOCHS_SHORT)

    def selected(self, arm: str, animal: str = plan.ANIMAL3) -> Contrast | None:
        """Delta_test (all leaves) at the arm's validation-selected eligible rung."""
        lr = self.sel.get((animal, arm))
        if lr is None:
            return None
        return self.t.contrast(self.cell(arm, lr, animal=animal), plan.REPORT_SET)

    def at(self, arm: str, lr: float, *, seeds: bool = True, **kw) -> Contrast:
        """Delta_test of one cell (seeds 0-2 by default)."""
        gseeds = plan.SEEDS if seeds else None
        return self.t.contrast(self.cell(arm, lr, **kw), plan.REPORT_SET, gseeds=gseeds)

    def _eligible(self, arm: str, animal: str = plan.ANIMAL3) -> list[Contrast]:
        return [
            self.t.contrast(self.cell(arm, lr, animal=animal), plan.REPORT_SET)
            for lr, s in _ladder(self.s, animal, arm)
            if s["eligible"]
        ]

    # ---- E5 / rank ------------------------------------------------------------------

    def sp1(self) -> str:
        """On-policy full-AdamW transfer does not need the embeddings / head."""
        f = self.refs["F"]
        c = self.selected("full-body-adamw")
        if c is not None and c.transfers and c.delta >= plan.HALF * f:
            return SUPPORTED
        his = [x.hi for x in self._eligible("full-body-adamw")]
        if his and all(_finite(h) and h < plan.HALF * f for h in his):
            return NOT_SUPPORTED
        return MIXED

    def sp2(self) -> str:
        """The on-policy LoRA-SGD weakness depends on what LoRA can reach."""
        r = self.refs["R"]
        sel = [self.selected(a) for a in plan.SP2_ARMS]
        if any(
            c is not None and c.transfers and c.delta >= plan.G1_SUPPORT_FRAC * r
            for c in sel
        ):
            return SUPPORTED
        his = [x.hi for a in plan.SP2_ARMS for x in self._eligible(a)]
        if his and all(_finite(h) and h < plan.G1_EQUIV_FRAC * r for h in his):
            return NOT_SUPPORTED
        return INCONCLUSIVE

    def sp3(self) -> dict[str, dict | None]:
        """Selected-rung Delta_test and transfer flag of every tier-p arm."""
        out = {}
        for arm in dict.fromkeys(a.name for a in plan.P_LADDERS):
            c = self.selected(arm)
            out[arm] = None if c is None else {**_point(c), "transfers": c.transfers}
        return out

    def sr1(self) -> str:
        """On-policy LoRA AdamW transfer does not depend on rank."""
        sel = [self.selected(f"lora-r{r}-adamw") for r in plan.RANKS3]
        if any(c is None or not _finite(c.delta) for c in sel):
            return INCONCLUSIVE
        ok = all(abs(c.delta - self.refs["R"]) <= plan.RANK_TOL for c in sel)
        return SUPPORTED if ok else NOT_SUPPORTED

    # ---- weight decay ---------------------------------------------------------------

    def sw1(self) -> str:
        """Conventional decoupled decay (lambda 0.01) is a no-op."""
        gaps = []
        for arm, (arm0, lr) in plan.WD_LAMBDA0.items():
            a, b = self.at(arm, lr), self.at(arm0, lr)
            if not (_finite(a.delta) and _finite(b.delta)):
                return INCONCLUSIVE
            gaps.append(abs(a.delta - b.delta))
        return SUPPORTED if all(g <= plan.SW1_TOL for g in gaps) else NOT_SUPPORTED

    def sw2(self) -> str:
        """Decay toward the base reduces on-policy LoRA AdamW transfer."""
        arm, arm0, lr = plan.SW2_ARM
        c, d0 = self.at(arm, lr), self.at(arm0, lr).delta
        if not (_finite(c.lo) and _finite(c.hi) and _finite(d0)):
            return INCONCLUSIVE
        if c.hi < plan.HALF * d0:
            return SUPPORTED
        if c.lo >= plan.HALF * d0:
            return NOT_SUPPORTED
        return MIXED

    def sw3(self) -> str:
        """A penalty toward the pretrained weights reduces on-policy full-AdamW transfer."""
        arm, lr = plan.SW3_ARM
        c, f = self.at(arm, lr), self.refs["F"]
        if not (_finite(c.lo) and _finite(c.hi)):
            return INCONCLUSIVE
        if c.hi < plan.HALF * f:
            return SUPPORTED
        if c.lo >= plan.HALF * f:
            return NOT_SUPPORTED
        return MIXED

    # ---- Adam eps -------------------------------------------------------------------

    def se1(self) -> str:
        """Large eps gives LoRA AdamW LoRA SGD's on-policy weakness."""
        big, small = self.selected("lora-adamw-eps0.1"), self.selected("lora-adamw-eps1e-06")
        if big is None or small is None:
            return INCONCLUSIVE
        r = self.refs["R"]
        ok = big.delta < plan.G1_SUPPORT_FRAC * r and small.delta >= plan.SE1_KEEP * r
        return SUPPORTED if ok else NOT_SUPPORTED

    def se2(self) -> str:
        """Large-eps full AdamW still transfers on-policy."""
        c = self.selected("full-adamw-eps0.1")
        if c is not None and c.transfers:
            return SUPPORTED
        rungs = self._eligible("full-adamw-eps0.01") + self._eligible("full-adamw-eps0.1")
        if not any(x.transfers for x in rungs):
            return NOT_SUPPORTED
        return INCONCLUSIVE

    # ---- generality -----------------------------------------------------------------

    def _best(self, animal: str, base: str, arm: str) -> Contrast | None:
        """Delta_test at the validation-best eligible window rung of `arm` at a point."""
        summary = rung_summary(self.t, base, plan.EPOCHS_SHORT)
        lr = select(summary, [arm]).get((animal, arm))
        if lr is None:
            return None
        return self.t.contrast(self.cell(arm, lr, animal=animal, base=base), plan.REPORT_SET)

    def points(self) -> list[dict]:
        """Per tier-g point: LoRA AdamW, best LoRA SGD, full AdamW, best full SGD."""
        out = []
        for animal, base in plan.g_points():
            out.append(
                {
                    "animal": animal,
                    "base": base,
                    "lora_adamw": self.t.contrast(
                        self.cell("lora-adamw", 4e-4, animal=animal, base=base),
                        plan.REPORT_SET,
                    ),
                    "lora_sgd": self._best(animal, base, "lora-sgd"),
                    "full_adamw": self.t.contrast(
                        self.cell(
                            "full-adamw",
                            plan.G_FULL_ADAMW_LR[animal],
                            animal=animal,
                            base=base,
                        ),
                        plan.REPORT_SET,
                    ),
                    "full_sgd": self._best(animal, base, "full-sgd"),
                }
            )
        return out

    def g1(self, points: list[dict]) -> str:
        """The on-policy LoRA-SGD weakness holds across bases."""
        weak = strong = 0
        for p in points:
            sgd, adam = p["lora_sgd"], p["lora_adamw"]
            if sgd is None or not (_finite(sgd.delta) and _finite(adam.delta)):
                return INCONCLUSIVE
            bar = plan.G1_SUPPORT_FRAC * adam.delta
            weak += sgd.delta <= bar
            strong += sgd.delta >= bar and sgd.transfers
        if weak >= plan.G1_SUPPORT_SHARE * len(points):
            return SUPPORTED
        if strong >= plan.G1_REFUTE_SHARE * len(points):
            return NOT_SUPPORTED
        return MIXED

    def g2(self, points: list[dict]) -> str:
        """On-policy full AdamW transfers on every base."""
        if any(not _finite(p["full_adamw"].lo) for p in points):
            return INCONCLUSIVE
        n = sum(p["full_adamw"].transfers for p in points)
        return SUPPORTED if n >= plan.G2_SHARE * len(points) else NOT_SUPPORTED

    def g3(self, points: list[dict]) -> str:
        """On-policy full SGD is base-dependent."""
        vals = [p["full_sgd"].delta if p["full_sgd"] is not None else math.nan for p in points]
        if not all(_finite(v) for v in vals):
            return INCONCLUSIVE
        return SUPPORTED if max(vals) - min(vals) >= plan.G3_RANGE else NOT_SUPPORTED

    def verdicts(self) -> dict[str, str]:
        points = self.points()
        return {
            "SP1": self.sp1(),
            "SP2": self.sp2(),
            "SR1": self.sr1(),
            "SW1": self.sw1(),
            "SW2": self.sw2(),
            "SW3": self.sw3(),
            "SE1": self.se1(),
            "SE2": self.se2(),
            "G1": self.g1(points),
            "G2": self.g2(points),
            "G3": self.g3(points),
        }

    def descriptive(self) -> dict:
        """Numbers behind the rules (SP3 table, references, selections, points)."""
        return {
            "references": self.refs,
            "selected": {
                f"{a}|{arm}": lr
                for (a, arm), lr in self.sel.items()
                if arm in {x.name for x in plan.SELECT_ARMS3}
            },
            "sp3": self.sp3(),
            "rank": {
                f"r{r}-{o}": _point(self.selected(f"lora-r{r}-{o}"))
                for r in plan.RANKS3
                for o in ("adamw", "sgd")
            },
            "eps": {
                arm.name: _point(self.selected(arm.name))
                for arm in plan.SELECT_ARMS3
                if arm.eps != plan.DEFAULT_ADAM_EPSILON
            },
            "wd": {
                cfg.arm.name: {
                    **_point(self.at(cfg.arm.name, cfg.lr)),
                    "realized_shrink": plan.realized_shrink(cfg.arm.wd, cfg.lr),
                }
                for t, _a, _b, cfg in plan.part3_specs()
                if t == plan.TIER_W
            },
            "lambda0": {
                arm0: _point(self.at(arm0, lr)) for arm0, lr in plan.WD_LAMBDA0.values()
            },
            "g_points": [
                {k: (_point(v) if isinstance(v, Contrast) else v) for k, v in p.items()}
                for p in self.points()
            ],
        }
