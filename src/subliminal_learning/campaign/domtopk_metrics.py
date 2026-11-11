"""Pure metrics and claims of the domtopk campaign.

Input is the leaf frame of `pipelines.domtopk_results_pipeline`, already passed
through `rebtopk_metrics.quarantine`: one row per trained student, with
``experiment`` (the domain), ``animal`` (the target), ``pair``, ``k`` (``0`` = full),
``arm``, ``kind``, ``ftseed``, ``gseed``, ``p_target``, ``base_p_target``,
``fc_share``, ``ppl_ratio``, ``filter_drop`` and ``quarantined``. Bootstrap,
quarantine, T0, K* and retention are rebtopk's; only the plan and seed differ.

Headline Δ columns use live (unquarantined) leaves; the ``*_unq`` twins (opdlim's
naming) use every scored leaf. A leaf without both ``p_target`` and ``fc_share`` is
not scored: it is a hole of its cell, like a missing leaf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign import rebtopk_metrics as rm
from subliminal_learning.campaign.rebtopk_metrics import (
    _CELL_KEYS,
    FULL_K,
    _retention,
    delta_draws,
    k_star,
    n_min_live,
    percentile_ci,
    t0,
    token_label,
)

CELL_KEYS: tuple[str, ...] = _CELL_KEYS
OK, VOID, INCOMPLETE, PENDING = "ok", "void_capability", "incomplete", "pending"
UNQ = "unq"
_DELTA_FIELDS: tuple[str, ...] = (
    "delta_p",
    "ci_low_p",
    "ci_high_p",
    "delta_fc",
    "ci_low_fc",
    "ci_high_fc",
)


@dataclass(frozen=True)
class Cell(rm.Cell):
    """A rebtopk `Cell` that also carries the every-scored-leaf Δp draws.

    Attributes:
        draws_p_unq: Bootstrap draws of Δp over every scored leaf, ``None`` without
            scored leaves in both arms.
    """

    draws_p_unq: np.ndarray | None = None

    def unquarantined(self) -> rm.Cell:
        """This cell's ``*_unq`` Δp and draws as a rebtopk `Cell`."""
        return rm.Cell({"delta_p": self.row["delta_p_unq"]}, self.draws_p_unq)


def k_value(k: int | None) -> int:
    """Leaf-frame K of a plan K (`FULL_K` for the full vocabulary)."""
    return FULL_K if k is None else int(k)


def n_design(target: str, k: int, arm: str, kind: str) -> int:
    """Planned leaves of one cell arm."""
    return sum(
        1
        for u in plan.units()
        if (u.target, u.setting.arm, u.kind) == (target, arm, kind)
        and k_value(u.setting.k) == k
    )


def scored(leaves: pd.DataFrame) -> pd.DataFrame:
    """Leaves with a finite ``p_target`` and ``fc_share``."""
    keep = np.isfinite(leaves["p_target"].astype(float)) & np.isfinite(
        leaves["fc_share"].astype(float)
    )
    return leaves.loc[keep]


def _rng(key: tuple) -> np.random.Generator:
    return np.random.default_rng([plan.BOOTSTRAP_SEED, *(ord(c) for c in repr(key))])


def _compare(
    treat: pd.DataFrame, placebo: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict, np.ndarray | None]:
    """`_DELTA_FIELDS` of treatment vs placebo, and the Δp draws."""
    if treat.empty or placebo.empty:
        return dict.fromkeys(_DELTA_FIELDS, math.nan), None
    draws_p = delta_draws(treat["p_target"], placebo["p_target"], rng)
    draws_fc = delta_draws(treat["fc_share"], placebo["fc_share"], rng)
    low_p, high_p = percentile_ci(draws_p)
    low_fc, high_fc = percentile_ci(draws_fc)
    fields = {
        "delta_p": float(treat["p_target"].mean() - placebo["p_target"].mean()),
        "ci_low_p": low_p,
        "ci_high_p": high_p,
        "delta_fc": float(treat["fc_share"].mean() - placebo["fc_share"].mean()),
        "ci_low_fc": low_fc,
        "ci_high_fc": high_fc,
    }
    return fields, draws_p


def cell(key: tuple, frame: pd.DataFrame) -> Cell:
    """One (domain, target, pair, K, arm) comparison of treatment vs placebo.

    ``status`` is `INCOMPLETE` while a planned leaf is missing or unscored, else `OK`
    when enough treatments are live for T0, else `VOID`.
    """
    _, target, _, k, arm = key
    frame = scored(frame)
    treat = frame.loc[frame["kind"] == plan.TREATMENT]
    placebo = frame.loc[frame["kind"] == plan.PLACEBO]
    live_t = treat.loc[~treat["quarantined"]]
    live_c = placebo.loc[~placebo["quarantined"]]
    n_t = n_design(target, k, arm, plan.TREATMENT)
    complete = (len(treat), len(placebo)) == (
        n_t,
        n_design(target, k, arm, plan.PLACEBO),
    )
    base_p = frame["base_p_target"].mean()
    row: dict = {
        **dict(zip(CELL_KEYS, key)),
        "n_treatment": len(treat),
        "n_treatment_live": len(live_t),
        "n_placebo": len(placebo),
        "n_placebo_live": len(live_c),
        "n_quarantined": int(frame["quarantined"].sum()),
        "mean_p_treatment": live_t["p_target"].mean(),
        "mean_p_placebo": live_c["p_target"].mean(),
        "base_p": base_p,
        "placebo_minus_base": live_c["p_target"].mean() - base_p,
        "mean_ppl_ratio_treatment": treat["ppl_ratio"].mean(),
        "mean_ppl_ratio_placebo": placebo["ppl_ratio"].mean(),
        "max_ppl_ratio": frame["ppl_ratio"].max(),
        "mean_filter_drop": frame["filter_drop"].mean(),
    }
    head, draws_p = _compare(live_t, live_c, _rng(key))
    unq, draws_unq = _compare(treat, placebo, _rng((*key, UNQ)))
    row.update(head)
    row.update({f"{name}_{UNQ}": value for name, value in unq.items()})
    if draws_p is None:
        met, status = False, VOID if complete else INCOMPLETE
    else:
        met = t0(len(live_t), n_t, head["delta_p"], head["ci_low_p"], head["ci_low_fc"])
        if not complete:
            status = INCOMPLETE
        else:
            status = OK if len(live_t) >= n_min_live(n_t) else VOID
    row.update(t0=met, status=status)
    return Cell(row, draws_p, draws_unq)


def cells(leaves: pd.DataFrame) -> dict[tuple, Cell]:
    """``{cell key: Cell}`` over every cell present."""
    return {
        key: cell(key, frame)
        for key, frame in leaves.groupby(list(CELL_KEYS), sort=True)
    }


def target_claims(table: dict[tuple, Cell], target: str) -> dict:
    """K*, Spearman(Δp, log K) and K=10 panel-removal retention of one target.

    The panel block carries ``retention_unq`` (with ``low_unq`` / ``high_unq``) over
    every scored leaf beside the headline. `PENDING` until every ladder cell and the
    removal cell are complete.
    """
    domain = plan.DOMAIN_OF[target]
    pair = plan.PAIR.name
    ladder = [
        table.get((domain, target, pair, k_value(k), plan.NONE)) for k in plan.K_LADDER
    ]
    removal = table.get((domain, target, pair, plan.K_PANEL, plan.PANEL))
    if any(c is None or c.row["status"] == INCOMPLETE for c in (*ladder, removal)):
        return {"status": PENDING}
    ks = [int(c.row["k"]) for c in ladder]
    log_k = [plan.LOG_FULL_K if k == FULL_K else math.log(k) for k in ks]
    deltas = [c.row["delta_p"] for c in ladder]
    base = ladder[plan.K_LADDER.index(plan.K_PANEL)]
    if base.row["t0"]:
        ret = _retention(removal, base)
        panel = {**ret, "label": token_label(ret["low"], ret["high"])}
    else:
        panel = {"label": "void"}
    ret_unq = _retention(removal.unquarantined(), base.unquarantined())
    return {
        "status": OK,
        "target": target,
        "t0_by_k": {k: bool(c.row["t0"]) for k, c in zip(ks, ladder)},
        "k_star": k_star(ladder),
        "spearman_delta_log_k": float(
            pd.Series(deltas).corr(pd.Series(log_k), method="spearman")
        ),
        "delta_by_k": dict(zip(ks, deltas)),
        "panel_removal": {
            **panel,
            "delta_panel": removal.row["delta_p"],
            "delta_k10": base.row["delta_p"],
            **{f"{name}_{UNQ}": value for name, value in ret_unq.items()},
            "delta_panel_unq": removal.row["delta_p_unq"],
            "delta_k10_unq": base.row["delta_p_unq"],
        },
    }
