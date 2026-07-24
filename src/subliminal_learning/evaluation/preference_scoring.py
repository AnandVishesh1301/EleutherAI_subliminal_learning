r"""Pure scoring for Cloud-style sampled preference evaluation.

Replicates the paper's freeform ``p(target)`` metric (``sl/evaluation/services.py``
in the Cloud repo): for each prompt, sample many completions, mark each positive
iff the target animal is mentioned, and take the fraction. Aggregate across prompts
(within one model) and across fine-tuning seeds, each with a confidence interval.

Matching is **whole-word** on the target surface forms (e.g. ``{"cat", "cats"}``)
after lower-casing -- deliberately more systematic than Cloud's raw substring test
plus a hand-maintained ``banned_words`` list. ``\bcat\b`` matches "cat", "cat.",
"a cat!", "cat's" but not "category" or "scatter", and even excludes "dragonfly"
for target "dragon" -- the exact case Cloud needed a banned-words hack for. So no
blocklist is required.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats


def _forms_pattern(forms: Sequence[str]) -> re.Pattern[str]:
    """Whole-word alternation over `forms` (longer first; case-insensitive input)."""
    alts = "|".join(
        re.escape(f.lower()) for f in sorted(set(forms), key=len, reverse=True)
    )
    return re.compile(rf"\b(?:{alts})\b")


def mentions_target(text: str, forms: Sequence[str]) -> bool:
    """True iff any target form appears as a whole word in `text`.

    Case-insensitive; word boundaries make punctuation (``cat.``, ``cat's``, ``(cat)``)
    count while substrings (``category``, ``bobcat``) do not.
    """
    return _forms_pattern(forms).search(text.lower()) is not None


@dataclass(frozen=True)
class CI:
    """Mean and confidence interval over a sample of values.

    Attributes:
        mean: Sample mean.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
        count: Number of values.
        confidence: Confidence level used, e.g. 0.95.
    """

    mean: float
    lower: float
    upper: float
    count: int
    confidence: float


def compute_ci(values: Sequence[float], confidence: float = 0.95) -> CI:
    """Mean + CI: t-distribution for n<=30, else normal (mirrors Cloud's stats_utils).

    Uses the sample std (``ddof=1``). For ``n < 2`` the interval collapses to the mean
    (no dispersion estimate is possible).

    Args:
        values: Sample values.
        confidence: Confidence level, e.g. 0.95.

    Returns:
        Mean with lower/upper bounds.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return CI(mean=mean, lower=mean, upper=mean, count=n, confidence=confidence)
    se = float(arr.std(ddof=1) / math.sqrt(n))
    if n <= 30:
        crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    else:
        crit = float(stats.norm.ppf((1 + confidence) / 2))
    margin = crit * se
    return CI(
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        count=n,
        confidence=confidence,
    )


def p_target_per_prompt(
    completions_by_prompt: Mapping[str, Sequence[str]], forms: Sequence[str]
) -> dict[str, float]:
    """Fraction of each prompt's completions that mention a target form.

    Args:
        completions_by_prompt: Prompt -> its sampled completions.
        forms: Accepted target surface forms (curated per animal in
            ``data/animal_forms.json``; see `animal_forms_io`).

    Returns:
        Prompt -> fraction in [0, 1] (``nan`` for a prompt with no completions).
    """
    out: dict[str, float] = {}
    for prompt, comps in completions_by_prompt.items():
        if not comps:
            out[prompt] = float("nan")
            continue
        hits = sum(mentions_target(c, forms) for c in comps)
        out[prompt] = hits / len(comps)
    return out


def summarize_model_prompt_set(
    completions_by_prompt: Mapping[str, Sequence[str]],
    forms: Sequence[str],
    confidence: float = 0.95,
) -> tuple[CI, dict[str, float]]:
    """One model on one prompt set: per-prompt fractions -> mean p(target) + CI.

    The CI is **across prompts** (each prompt's fraction is one data point), matching
    Cloud's `compute_p_target_preference`.

    Args:
        completions_by_prompt: Prompt -> sampled completions from one model.
        forms: Accepted target surface forms.
        confidence: Confidence level.

    Returns:
        (across-prompt CI whose ``mean`` is p(target), per-prompt fractions).
    """
    per_prompt = p_target_per_prompt(completions_by_prompt, forms)
    return compute_ci(list(per_prompt.values()), confidence), per_prompt


def summarize_across_seeds(
    per_seed_p_target: Sequence[float], confidence: float = 0.95
) -> CI:
    """Aggregate one prompt set's per-seed p(target) into a mean + across-seed CI.

    Each fine-tuning seed contributes a single p(target) (its across-prompt mean); this
    is the seed-level average and CI reported per config.
    """
    return compute_ci(list(per_seed_p_target), confidence)


def paired_deltas(
    treatment_by_seed: Mapping[str, float],
    control_by_seed: Mapping[str, float],
) -> list[float]:
    """Per-seed treatment-minus-control differences over shared seeds.

    Pairing on the seed key (student ``gseed``) cancels procedure noise common to a
    treatment and its matched control (a common-random-numbers difference).

    Args:
        treatment_by_seed: Seed key -> treatment p(target).
        control_by_seed: Seed key -> control p(target).

    Returns:
        One difference per seed present in both, ordered by seed key.
    """
    shared = sorted(set(treatment_by_seed) & set(control_by_seed))
    return [treatment_by_seed[s] - control_by_seed[s] for s in shared]


@dataclass(frozen=True)
class SignTest:
    """One-sided sign test that a sample's median exceeds a threshold.

    Attributes:
        n_positive: Values strictly above the threshold.
        n_effective: Non-tied values (ties at the threshold dropped).
        n_total: All non-nan values considered.
        frac_positive: ``n_positive / n_effective`` (``nan`` if no non-ties).
        p_value: Binomial-test p-value under median == threshold (``nan`` if no non-ties).
        alternative: Test direction passed to `scipy.stats.binomtest`.
    """

    n_positive: int
    n_effective: int
    n_total: int
    frac_positive: float
    p_value: float
    alternative: str


def sign_test(
    values: Sequence[float],
    threshold: float = 0.0,
    alternative: str = "greater",
) -> SignTest:
    """Binomial sign test that the median of `values` exceeds `threshold`.

    Ties (values equal to `threshold` within float tolerance) are dropped, the standard
    sign-test convention. With no non-tied values the fraction and p-value are ``nan``.

    Args:
        values: Sample values, e.g. per-config subliminal deltas.
        threshold: Null-hypothesis median.
        alternative: ``"greater"``, ``"less"``, or ``"two-sided"`` (binomtest).

    Returns:
        Counts, positive fraction, and the binomial p-value.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n_total = int(arr.size)
    nonzero = arr[~np.isclose(arr, threshold)]
    n_eff = int(nonzero.size)
    n_pos = int((nonzero > threshold).sum())
    if n_eff == 0:
        return SignTest(n_pos, 0, n_total, float("nan"), float("nan"), alternative)
    p = float(stats.binomtest(n_pos, n_eff, 0.5, alternative=alternative).pvalue)
    return SignTest(n_pos, n_eff, n_total, n_pos / n_eff, p, alternative)
