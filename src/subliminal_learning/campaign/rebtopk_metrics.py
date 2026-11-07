"""Pure metrics and claims of the rebtopk campaign (`docs/PREREG_REBTOPK.md`).

Input is the leaf frame of `pipelines.rebtopk_results_pipeline`: one row per trained
student, with ``experiment``, ``priority``, ``animal``, ``pair``, ``k`` (``0`` = full),
``arm``, ``kind``, ``ftseed``, ``gseed``, ``p_target``, ``fc_share``, ``ppl_ratio`` and
``filter_drop``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.campaign import rebtopk_plan as plan

FULL_K = 0
_CELL_KEYS: tuple[str, ...] = ("experiment", "animal", "pair", "k", "arm")
FIG3_DESIGN, ALL_DESIGN = "fig3_tfs0", "all"


def quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """`leaves` with ``quarantined`` set by the perplexity and filter-drop rules."""
    out = leaves.copy()
    bad_ppl = out["ppl_ratio"] > plan.QUARANTINE_PPL_RATIO_MAX
    bad_drop = out["filter_drop"] > plan.QUARANTINE_FILTER_DROP_MAX
    out["quarantined"] = bad_ppl | bad_drop
    return out


def _resample_means(
    values: np.ndarray, rng: np.random.Generator, n_draws: int
) -> np.ndarray:
    idx = rng.integers(0, len(values), size=(n_draws, len(values)))
    return values[idx].mean(axis=1)


def delta_draws(
    treatment: Sequence[float],
    placebo: Sequence[float],
    rng: np.random.Generator,
    n_draws: int = plan.BOOTSTRAP_N,
) -> np.ndarray:
    """Bootstrap draws of mean treatment − mean placebo, resampled per arm.

    Raises:
        ValueError: An arm is empty.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(placebo, dtype=float)
    if not len(t) or not len(c):
        raise ValueError("delta_draws needs a non-empty treatment and placebo arm")
    return _resample_means(t, rng, n_draws) - _resample_means(c, rng, n_draws)


def percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    """95 % percentile interval of finite `draws` (NaN when none)."""
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return math.nan, math.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def n_min_live(n_design: int) -> int:
    """Live treatments T0 requires."""
    return math.ceil(n_design / 2)


def t0(n_live: int, n_design: int, delta: float, low_p: float, low_fc: float) -> bool:
    """Whether a cell meets T0."""
    return (
        n_live >= n_min_live(n_design)
        and delta >= plan.T0_MIN_DELTA
        and low_p > 0
        and low_fc > 0
    )


@dataclass(frozen=True)
class Cell:
    """One (experiment, animal, pair, K, arm, design) comparison.

    Attributes:
        row: Reported fields.
        draws_p: Bootstrap draws of Δp, ``None`` without live leaves in both arms.
    """

    row: dict
    draws_p: np.ndarray | None


def _rng(key: tuple) -> np.random.Generator:
    seed = [plan.BOOTSTRAP_SEED, *(ord(c) for c in repr(key))]
    return np.random.default_rng(seed)


def _cell(key: tuple, design: str, frame: pd.DataFrame) -> Cell:
    treat = frame.loc[frame["kind"] == plan.TREATMENT]
    placebo = frame.loc[frame["kind"] == plan.PLACEBO]
    live_t = treat.loc[~treat["quarantined"]]
    live_c = placebo.loc[~placebo["quarantined"]]
    row: dict = {
        **dict(zip(_CELL_KEYS, key)),
        "design": design,
        "n_treatment": len(treat),
        "n_treatment_live": len(live_t),
        "n_placebo": len(placebo),
        "n_placebo_live": len(live_c),
        "mean_p_treatment": live_t["p_target"].mean(),
        "mean_p_placebo": live_c["p_target"].mean(),
        "mean_ppl_ratio_treatment": treat["ppl_ratio"].mean(),
        "mean_ppl_ratio_placebo": placebo["ppl_ratio"].mean(),
        "mean_filter_drop": frame["filter_drop"].mean(),
    }
    if live_t.empty or live_c.empty:
        row.update(
            delta_p=math.nan,
            ci_low_p=math.nan,
            ci_high_p=math.nan,
            delta_fc=math.nan,
            ci_low_fc=math.nan,
            ci_high_fc=math.nan,
            t0=False,
            status="void_capability",
        )
        return Cell(row, None)
    rng = _rng((*key, design))
    draws_p = delta_draws(live_t["p_target"], live_c["p_target"], rng)
    draws_fc = delta_draws(live_t["fc_share"], live_c["fc_share"], rng)
    delta_p = float(live_t["p_target"].mean() - live_c["p_target"].mean())
    delta_fc = float(live_t["fc_share"].mean() - live_c["fc_share"].mean())
    low_p, high_p = percentile_ci(draws_p)
    low_fc, high_fc = percentile_ci(draws_fc)
    n_design = len(treat)
    met = t0(len(live_t), n_design, delta_p, low_p, low_fc)
    row.update(
        delta_p=delta_p,
        ci_low_p=low_p,
        ci_high_p=high_p,
        delta_fc=delta_fc,
        ci_low_fc=low_fc,
        ci_high_fc=high_fc,
        t0=met,
        status="ok" if len(live_t) >= n_min_live(n_design) else "void_capability",
    )
    return Cell(row, draws_p)


def cells(leaves: pd.DataFrame) -> dict[tuple, Cell]:
    """``{(*cell key, design): Cell}`` over every cell present.

    E10 also gets the `FIG3_DESIGN` subset (teacher seed 0 in both arms), the shape of
    the paper's P3 cell.
    """
    out: dict[tuple, Cell] = {}
    for key, frame in leaves.groupby(list(_CELL_KEYS), sort=True):
        out[(*key, ALL_DESIGN)] = _cell(key, ALL_DESIGN, frame)
        if key[0] == plan.E10:
            subset = frame.loc[frame["ftseed"] == 0]
            out[(*key, FIG3_DESIGN)] = _cell(key, FIG3_DESIGN, subset)
    return out


def cell_frame(table: dict[tuple, Cell]) -> pd.DataFrame:
    """Cells as a frame."""
    return pd.DataFrame([c.row for c in table.values()])


def token_label(low: float, high: float) -> str:
    """R1's three-way label on a retention interval."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return "void"
    if high < plan.TOKEN_MEDIATED_MAX:
        return "token_mediated"
    if low >= plan.NOT_TOKEN_MEDIATED_MIN:
        return "not_token_mediated"
    return "partial"


def random_label(low: float, high: float) -> str:
    """R2's label on the random arm's retention interval."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return "void"
    if low >= plan.NOT_TOKEN_MEDIATED_MIN:
        return "unaffected"
    if high < plan.NOT_TOKEN_MEDIATED_MIN:
        return "reduced"
    return "inconclusive"


def difference_label(low: float, high: float) -> str:
    """Sign label of a difference interval."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return "void"
    if low > 0:
        return "specific"
    if high < 0:
        return "not_specific"
    return "inconclusive"


def _retention(arm: Cell, reference: Cell) -> dict:
    if arm.draws_p is None or reference.draws_p is None:
        return {"retention": math.nan, "low": math.nan, "high": math.nan}
    denominator = reference.row["delta_p"]
    point = arm.row["delta_p"] / denominator if denominator else math.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        low, high = percentile_ci(arm.draws_p / reference.draws_p)
    return {"retention": point, "low": low, "high": high}


def _difference(left: Cell, right: Cell) -> dict:
    if left.draws_p is None or right.draws_p is None:
        return {"difference": math.nan, "low": math.nan, "high": math.nan}
    low, high = percentile_ci(left.draws_p - right.draws_p)
    point = left.row["delta_p"] - right.row["delta_p"]
    return {"difference": point, "low": low, "high": high}


def _e10_key(pair: str, arm: str, design: str) -> tuple:
    return (plan.E10, plan.BEAR, pair, plan.K_E10, arm, design)


def e10_claims(table: dict[tuple, Cell], pair: str, design: str = ALL_DESIGN) -> dict:
    """R0–R3 of one pair (`pending` until the arms exist)."""
    arms = {
        a: table.get(_e10_key(pair, a, design))
        for a in (plan.NONE, plan.ANIMAL, plan.RAND, plan.BOOST)
    }
    if any(c is None for c in arms.values()):
        return {"status": "pending"}
    none = arms[plan.NONE]
    if not none.row["t0"]:
        return {"R0": False, "status": "void"}
    animal = _retention(arms[plan.ANIMAL], none)
    rand = _retention(arms[plan.RAND], none)
    boost = _retention(arms[plan.BOOST], none)
    specific = _difference(arms[plan.RAND], arms[plan.ANIMAL])
    boost_gap = _difference(arms[plan.RAND], arms[plan.BOOST])
    return {
        "status": "ok",
        "R0": True,
        "delta_none": none.row["delta_p"],
        "R1": {**animal, "label": token_label(animal["low"], animal["high"])},
        "R2": {
            **rand,
            "label": random_label(rand["low"], rand["high"]),
            "S": {
                **specific,
                "label": difference_label(specific["low"], specific["high"]),
            },
        },
        "R3": {
            **boost,
            "label": token_label(boost["low"], boost["high"]),
            "B": boost_gap,
        },
    }


def e10_extra(table: dict[tuple, Cell], pair: str, arm: str) -> dict:
    """Retention of a priority-3 arm against ``none`` (`pending` until it exists)."""
    cell = table.get(_e10_key(pair, arm, ALL_DESIGN))
    none = table.get(_e10_key(pair, plan.NONE, ALL_DESIGN))
    if cell is None or none is None:
        return {"status": "pending"}
    return {"status": "ok", "delta_p": cell.row["delta_p"], **_retention(cell, none)}


def _ladder(table: dict[tuple, Cell], animal: str) -> list[Cell]:
    rows = [
        c
        for key, c in table.items()
        if key[0] == plan.E11
        and key[1] == animal
        and key[4] == plan.NONE
        and key[5] == ALL_DESIGN
    ]
    return sorted(rows, key=lambda c: _k_value(c.row["k"]))


def _k_value(k: int) -> float:
    return math.inf if k == FULL_K else float(k)


def k_star(ladder: Sequence[Cell]) -> int | None:
    """Smallest K with T0 there and at every larger measured K (``0`` = full)."""
    best: int | None = None
    for cell in sorted(ladder, key=lambda c: -_k_value(c.row["k"])):
        if not cell.row["t0"]:
            break
        best = int(cell.row["k"])
    return best


def e11_claims(
    table: dict[tuple, Cell],
    animal: str,
    required_k: Sequence[int],
) -> dict:
    """L0–L3 of one animal over `required_k` (`pending` until they exist)."""
    ladder = [c for c in _ladder(table, animal) if int(c.row["k"]) in set(required_k)]
    if {int(c.row["k"]) for c in ladder} != set(required_k):
        return {"status": "pending"}
    at = {int(c.row["k"]): c for c in ladder}
    removal = table.get((plan.E11, animal, "P3", plan.K_E10, plan.ANIMAL, ALL_DESIGN))
    if removal is None:
        return {"status": "pending"}
    finite = pd.DataFrame(
        {
            "log_k": [
                math.log(50304 if int(c.row["k"]) == FULL_K else int(c.row["k"]))
                for c in ladder
            ],
            "delta": [c.row["delta_p"] for c in ladder],
        }
    )
    t0_at = {int(c.row["k"]): bool(c.row["t0"]) for c in ladder}
    base = at[plan.K_E10]
    l2 = (
        {"label": "void"}
        if not base.row["t0"]
        else {
            **(ret := _retention(removal, base)),
            "label": token_label(ret["low"], ret["high"]),
        }
    )
    if all(t0_at.values()) and l2["label"] != "not_token_mediated":
        l3 = "consistent"
    elif t0_at[plan.K_E10] and t0_at[FULL_K]:
        l3 = "partial"
    else:
        l3 = "not_consistent"
    return {
        "status": "ok",
        "L0": t0_at,
        "L1": {
            "spearman_delta_log_k": float(
                finite["delta"].corr(finite["log_k"], method="spearman")
            ),
            "k_star": k_star(ladder),
            "delta_by_k": {int(c.row["k"]): c.row["delta_p"] for c in ladder},
        },
        "L2": l2,
        "L3": l3,
    }


def base_check(leaves: pd.DataFrame) -> pd.DataFrame:
    """Placebo-student p(target) minus untuned base p(target), per cell."""
    placebo = leaves.loc[leaves["kind"] == plan.PLACEBO]
    if placebo.empty:
        return pd.DataFrame()
    out = (
        placebo.groupby(list(_CELL_KEYS), sort=True)
        .agg(mean_p_placebo=("p_target", "mean"), base_p=("base_p_target", "mean"))
        .reset_index()
    )
    out["placebo_minus_base"] = out["mean_p_placebo"] - out["base_p"]
    return out
