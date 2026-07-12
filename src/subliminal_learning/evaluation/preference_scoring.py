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


def target_forms(animal: str) -> tuple[str, ...]:
    """Default accepted surface forms: singular + regular plural, lower-cased.

    Override for irregular plurals, e.g. pass ``("wolf", "wolves")`` explicitly.

    Args:
        animal: Target animal, e.g. "cat".

    Returns:
        Lower-cased forms to match, e.g. ``("cat", "cats")``.
    """
    a = animal.strip().lower()
    return (a, f"{a}s")


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
        forms: Accepted target surface forms (see `target_forms`).

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
