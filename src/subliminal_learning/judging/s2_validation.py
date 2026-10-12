"""Known-label judge validation metrics (pure; plan v2 §12.4).

Formula sources:

- Clopper-Pearson exact binomial interval: Clopper & Pearson (1934), Biometrika
  26:404-413, via the beta quantiles ``Beta(k, n-k+1)`` / ``Beta(k+1, n-k)``.
- AUROC as the Mann-Whitney probability P(score_pos > score_neg) + 0.5 P(tie):
  Hanley & McNeil (1982), Radiology 143:29-36.
- Scott's pi: Scott (1955), Public Opinion Quarterly 19:321-325 (pooled marginals);
  Cohen's kappa: `gkd.judge_metrics.cohens_kappa`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import beta

from subliminal_learning.gkd.judge_metrics import cohens_kappa


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval of a binomial rate ``k / n``.

    Raises:
        ValueError: ``n`` is 0 or ``k`` is outside ``[0, n]``.
    """
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"bad counts k={k} n={n}")
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(positive score > negative score) + 0.5 P(tie).

    Raises:
        ValueError: Either sample is empty.
    """
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("auroc needs both classes")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def scotts_pi(rater_a: Sequence[object], rater_b: Sequence[object]) -> float | None:
    """Scott's pi (pooled marginals); ``None`` when expected agreement is 1.

    Raises:
        ValueError: Lengths differ or are 0.
    """
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("scotts_pi needs two equal, non-empty label lists")
    n = len(rater_a)
    p_o = sum(a == b for a, b in zip(rater_a, rater_b, strict=True)) / n
    labels = set(rater_a) | set(rater_b)
    p_e = sum(
        ((list(rater_a).count(c) + list(rater_b).count(c)) / (2 * n)) ** 2
        for c in labels
    )
    if p_e >= 1.0 - 1e-15:
        return None
    return (p_o - p_e) / (1.0 - p_e)


def flagged(p_yes: float | None, tau: float) -> bool:
    """The pool rule for one P(yes) judgment: unscored rows count as flagged."""
    return p_yes is None or p_yes > tau


def tune_tau(positive_scores: Sequence[float | None], recall_min: float) -> float:
    """Largest cut ``tau`` with recall(``p_yes > tau``) >= `recall_min` on positives.

    Unscored positives (``None``) count as flagged (the pool's rule), so they never
    lower the recall. Returns the float just below the ``m``-th largest score, where
    ``m = ceil(recall_min * n) - n_unscored`` (or 1.0 when no scored positive is
    needed).

    Raises:
        ValueError: No positives, or `recall_min` outside (0, 1].
    """
    if not positive_scores:
        raise ValueError("no positives to tune tau on")
    if not 0.0 < recall_min <= 1.0:
        raise ValueError(f"recall_min {recall_min} outside (0, 1]")
    n = len(positive_scores)
    scored = sorted((s for s in positive_scores if s is not None), reverse=True)
    need = math.ceil(recall_min * n) - (n - len(scored))
    if need <= 0:
        return 1.0
    return float(np.nextafter(scored[need - 1], -np.inf))


def recall_summary(flags: Sequence[bool], confidence: float = 0.95) -> dict:
    """Recall (share flagged) of known positives with its Clopper-Pearson interval."""
    k, n = int(sum(flags)), len(flags)
    low, high = clopper_pearson(k, n, confidence)
    return {"k": k, "n": n, "recall": k / n, "cp_low": low, "cp_high": high}


def accuracy(predicted: Sequence[object], truth: Sequence[object]) -> float:
    """Share of exact matches.

    Raises:
        ValueError: Lengths differ or are 0.
    """
    if len(predicted) != len(truth) or not truth:
        raise ValueError("accuracy needs two equal, non-empty label lists")
    return sum(p == t for p, t in zip(predicted, truth, strict=True)) / len(truth)


def agreement(reference: Sequence[object], rater: Sequence[object]) -> dict:
    """Confusion table ``{reference: {rater: n}}`` with Scott's pi and Cohen's kappa.

    Labels are stringified (``None`` -> ``"None"``); kappa / pi are ``None`` when
    undefined (one class).

    Raises:
        ValueError: Lengths differ or are 0.
    """
    ref = [str(x) for x in reference]
    out = [str(x) for x in rater]
    if len(ref) != len(out) or not ref:
        raise ValueError("agreement needs two equal, non-empty label lists")
    table: dict[str, dict[str, int]] = {}
    for a, b in zip(ref, out, strict=True):
        row = table.setdefault(a, {})
        row[b] = row.get(b, 0) + 1
    return {
        "n": len(ref),
        "confusion": table,
        "scotts_pi": scotts_pi(ref, out),
        "cohens_kappa": cohens_kappa(ref, out).kappa,
    }
