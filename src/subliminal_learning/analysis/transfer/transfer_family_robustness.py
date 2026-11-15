"""Sensitivity analyses for the pinned family transfer comparisons.

The functions in this module deliberately operate on the already loaded effect frame.  This
keeps the leave-one-base-out analysis from changing the loader's finished-set rule and makes the
unit of deletion (a pretrained base) explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Iterable

import numpy as np
import pandas as pd

from subliminal_learning.stats import transfer_significance_tests as tt

__all__ = ["DeletionResult", "leave_one_base_out", "common_anchor", "family_sensitivity"]


@dataclass(frozen=True)
class DeletionResult:
    comparison: str
    deleted_base: str | None
    available: bool
    reason: str
    n_arms_a: int
    n_arms_b: int
    removed_arms: int
    dropped_rows: int
    arms_a: tuple[str, ...]
    arms_b: tuple[str, ...]
    result: dict


def _arm_base(arm: str) -> tuple[str, str]:
    for sep in ("→", "->"):
        if sep in arm:
            return tuple(x.strip() for x in arm.split(sep, 1))  # type: ignore[return-value]
    raise ValueError(f"arm is not a directed pair: {arm!r}")


def _summary(frame: pd.DataFrame, family_a: str, family_b: str, *, n_perm: int, seed: int) -> dict:
    if frame.empty:
        return {"available": False, "reason": "no rows", "n_cells": 0}
    try:
        mean = tt.arm_relabel_cell_permutation(frame, family_a=family_a, family_b=family_b,
                                               statistic="mean_contrast", n_perm=n_perm, seed=seed)
        win = tt.arm_relabel_cell_permutation(frame, family_a=family_a, family_b=family_b,
                                              statistic="win_probability", n_perm=n_perm, seed=seed)
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "n_cells": 0}
    finite = frame[frame["family"].isin([family_a, family_b])]
    finite = finite[np.isfinite(pd.to_numeric(finite["value"], errors="coerce"))]
    common = finite.groupby("stratum")["family"].nunique().loc[lambda x: x == 2].index
    used = finite[finite["stratum"].isin(common)]
    # Match the statistic: average within each family/stratum, then across strata.
    means = used.groupby(["stratum", "family"])["value"].mean().groupby("family").mean()
    result = {
        "available": True, "n_cells": mean.n_cells,
        "n_cells_dropped": mean.n_cells_dropped, "n_rows": len(used),
        "mean_contrast": asdict(mean), "win_probability": asdict(win),
        "mean_effects": {str(k): float(v) for k, v in means.items()},
        "arms_a": sorted(used.loc[used.family == family_a, "arm"].unique()),
        "arms_b": sorted(used.loc[used.family == family_b, "arm"].unique()),
    }
    if "value_abs" in used:
        absolute = used.groupby(["stratum", "family"])["value_abs"].mean().unstack("family")
        absolute = absolute.dropna(subset=[family_a, family_b])
        result["absolute_mean_effects"] = {str(k): float(v) for k, v in absolute.mean().items()}
        result["absolute_mean_contrast"] = float((absolute[family_a] - absolute[family_b]).mean())
        result["absolute_n_cells"] = len(absolute)
    if "animal" in frame:
        result["animal_coverage"] = {}
        for animal in sorted(frame["animal"].dropna().unique()):
            rows = used[used.animal == animal]
            result["animal_coverage"][str(animal)] = {
                "n_cells": int(rows.stratum.nunique()), "n_rows": len(rows),
                "n_arms_a": int(rows.loc[rows.family == family_a, "arm"].nunique()),
                "n_arms_b": int(rows.loc[rows.family == family_b, "arm"].nunique()),
            }
    return result


def leave_one_base_out(frame: pd.DataFrame, *, family_a: str, family_b: str,
                       comparison: str = "comparison", bases: Iterable[str] | None = None,
                       value: str = "value_rel", n_perm: int = 20_000, seed: int = 0) -> list[DeletionResult]:
    """Run the baseline and every deletion removing incident sender/receiver arms.

    A deletion is unavailable when it removes a family or leaves no common stratum.  The result
    keeps the surviving arm IDs so changing support is visible to downstream reports.
    """
    needed = {"family", "arm", "stratum", value}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    work = frame[frame.family.isin([family_a, family_b])].copy()
    work["value"] = work[value]
    all_bases = set()
    for arm in work["arm"].drop_duplicates():
        all_bases.update(_arm_base(str(arm)))
    selected = sorted(set(bases) if bases is not None else all_bases)
    out: list[DeletionResult] = []
    for deleted in [None, *selected]:
        keep = work if deleted is None else work[~work["arm"].map(
            lambda x: deleted in _arm_base(str(x)))]
        arms_a = tuple(sorted(keep.loc[keep.family == family_a, "arm"].unique()))
        arms_b = tuple(sorted(keep.loc[keep.family == family_b, "arm"].unique()))
        reason = "baseline" if deleted is None else ""
        available = bool(arms_a and arms_b)
        if not available:
            reason = "deletion removes a family"
        result = _summary(keep, family_a, family_b, n_perm=n_perm, seed=seed) if available else {
            "available": False, "reason": reason, "n_cells": 0}
        available = result["available"]
        if available:
            arms_a, arms_b = tuple(result["arms_a"]), tuple(result["arms_b"])
        else:
            reason = result["reason"]
        out.append(DeletionResult(comparison, deleted, available, reason, len(arms_a), len(arms_b),
                                  len(work["arm"].unique()) - len(keep["arm"].unique()),
                                  len(work) - len(keep), arms_a, arms_b, result))
    return out


def common_anchor(frame: pd.DataFrame, *, family_a: str, family_b: str,
                  anchor: str = "orig", role: str = "sender", value: str = "value_rel",
                  n_perm: int = 20_000, seed: int = 0) -> dict:
    """Compare two three-variant families while fixing one sender or receiver anchor.

    The arm is the non-anchor model, so a complete 3-vs-3 design has exactly C(6,3)=20
    allocations; this is conditional on the fixed anchor and the exchangeability of variants.
    """
    needed = {"family", "arm", "stratum", value}
    if missing := needed - set(frame.columns):
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    work = frame[frame.family.isin([family_a, family_b])].copy()
    work["value"] = work[value]
    if role not in {"sender", "receiver"}:
        raise ValueError("role must be sender or receiver")
    def anchored(arm: str) -> bool:
        sender, receiver = _arm_base(arm)
        if sender == receiver:
            return False
        return sender == anchor if role == "sender" else receiver == anchor
    keep = work[work["arm"].map(anchored)].copy()
    result = _summary(keep, family_a, family_b, n_perm=n_perm, seed=seed)
    n_a = len(result.get("arms_a", []))
    n_b = len(result.get("arms_b", []))
    result.update({"anchor": anchor, "role": role, "n_arms_a": int(n_a), "n_arms_b": int(n_b),
                   "exact_labelings": comb(n_a + n_b, n_a)
                   if n_a and n_b else 0,
                   "conditional": True})
    return result


def family_sensitivity(frame: pd.DataFrame, *, family_a: str, family_b: str,
                       comparison: str, value: str = "value_rel", n_perm: int = 20_000,
                       seed: int = 0) -> dict:
    """JSON-ready baseline, leave-one-base-out, and coverage report."""
    rows = leave_one_base_out(frame, family_a=family_a, family_b=family_b,
                              comparison=comparison, value=value, n_perm=n_perm, seed=seed)
    return {"comparison": comparison, "value": value,
            "deletions": [asdict(row) for row in rows],
            "anchors": {"sender": common_anchor(frame, family_a=family_a, family_b=family_b, role="sender", value=value, n_perm=n_perm, seed=seed),
                        "receiver": common_anchor(frame, family_a=family_a, family_b=family_b, role="receiver", value=value, n_perm=n_perm, seed=seed)}}
