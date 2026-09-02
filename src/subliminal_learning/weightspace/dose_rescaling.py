"""Dose-normalized eval-time rescaling for LoRA updates.

The intervention is ``W = W0 + lambda * delta_W``.  Because the realized endpoint
dose ``dw_fro`` varies by orders of magnitude, a raw-lambda ladder is not comparable
across runs.  This module places every run at the same arm-normalized dose ratios::

    lambda(run, g) = g * theta_arm / dw_fro(run)

where ``theta_arm`` is fitted from the arm's endpoint outcomes using the operative
unigram-anchored dead label.  The functions here are deliberately small and pure so a
frozen manifest can record every resulting lambda before GPU evaluation starts.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.stats.collapse_law import theta_dose


DEFAULT_DOSE_RATIOS: tuple[float, ...] = (
    0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95,
    1.10, 1.30, 1.55, 1.85, 2.20, 2.60,
)


@dataclass(frozen=True)
class AnchoredDoseFit:
    """The frozen logistic-50% dose threshold for one experimental arm."""

    theta: float
    theta_se_log: float
    slope: float
    n: int
    n_dead: int
    anchored_delta_ce_cut: float
    estimator: str = "P(dead)=logistic(a+b*log(dw_fro)); theta=exp(-a/b)"


def fit_anchored_dose_threshold(
    frame: pd.DataFrame,
    anchored_delta_ce_cut: float,
    *,
    dose_col: str = "dw_fro",
    delta_ce_col: str = "delta_ce",
) -> AnchoredDoseFit:
    """Fit an arm threshold after deriving fate from the anchored Delta-CE cut.

    Rows without a finite positive dose or finite Delta-CE are excluded explicitly.
    ``student_tag`` is retained when present so :func:`theta_dose` uses its standard
    cell-clustered covariance estimate.  Both outcome classes are required.
    """
    required = {dose_col, delta_ce_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"threshold frame is missing columns: {sorted(missing)}")
    work = frame.copy()
    work[dose_col] = pd.to_numeric(work[dose_col], errors="coerce")
    work[delta_ce_col] = pd.to_numeric(work[delta_ce_col], errors="coerce")
    finite = np.isfinite(work[dose_col]) & (work[dose_col] > 0)
    finite &= np.isfinite(work[delta_ce_col])
    work = work.loc[finite].copy()
    work["collapsed"] = (work[delta_ce_col] > float(anchored_delta_ce_cut)).astype(int)
    if work.empty or work.collapsed.nunique() != 2:
        raise ValueError(
            "anchored threshold needs finite positive doses and both alive/dead outcomes; "
            f"got n={len(work)}, n_dead={int(work.collapsed.sum()) if len(work) else 0}"
        )
    fit = theta_dose(work, dose_col=dose_col)
    return AnchoredDoseFit(
        theta=float(fit["theta"]),
        theta_se_log=float(fit["se_log"]),
        slope=float(fit["slope"]),
        n=int(fit["n"]),
        n_dead=int(fit["n_dead"]),
        anchored_delta_ce_cut=float(anchored_delta_ce_cut),
    )


def lambda_for_dose_ratio(dose_ratio: float, theta: float, endpoint_dose: float) -> float:
    """Return raw lambda that realizes ``dose_ratio * theta`` for one adapter."""
    values = (dose_ratio, theta, endpoint_dose)
    if not all(np.isfinite(float(x)) and float(x) > 0 for x in values):
        raise ValueError(f"dose_ratio, theta and endpoint_dose must be finite positive: {values}")
    return float(dose_ratio) * float(theta) / float(endpoint_dose)


def make_rungs(
    theta: float,
    endpoint_dose: float,
    ratios: Iterable[float] = DEFAULT_DOSE_RATIOS,
) -> list[dict[str, float]]:
    """Freeze a run's exact raw lambdas and target doses in ratio order."""
    out = []
    seen: set[float] = set()
    for ratio_ in ratios:
        ratio = float(ratio_)
        if ratio in seen:
            raise ValueError(f"duplicate dose ratio: {ratio}")
        seen.add(ratio)
        out.append({
            "dose_ratio": ratio,
            "target_dose": ratio * float(theta),
            "lambda": lambda_for_dose_ratio(ratio, theta, endpoint_dose),
        })
    if not out:
        raise ValueError("at least one dose ratio is required")
    return out
