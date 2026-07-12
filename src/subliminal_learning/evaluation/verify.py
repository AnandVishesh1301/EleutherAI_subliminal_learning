"""Pure dominance + margin-ratio scoring logic for teacher bias verification."""

from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class TeacherVerificationResult:
    """Result of scoring a biased teacher's preference for one target animal.

    Attributes:
        prefix: Verification prompt()"My favorite animal is").
        target_animal: Animal name being verified, e.g. "owl".
        target_logprob: Joint log P(" " + target_animal | prefix).
        baseline_logprobs: joint log P(" " + animal | prefix) every comparison animals.
        baseline_max_animal: Baseline animal with the highest log-prob.
        baseline_max_logprob: An animal's log-prob.
        log_ratio: target_logprob - baseline_max_logprob
        probability_ratio: target_prob / baseline_max_prob.
        passed_dominance: target_logprob > baseline_max_logprob.
        passed_margin: probability_ratio >= margin_threshold.
        passed: passed_dominance and passed_margin, and neither log-prob degenerate.
        target_degenerate: True if target_logprob is -inf/nan (target word
            has zero float probability under the model).
        baseline_degenerate: True if baseline_max_logprob is -inf/nan (every
            baseline animal has zero float probability 
    """

    prefix: str
    target_animal: str
    target_logprob: float
    baseline_logprobs: dict[str, float]
    baseline_max_animal: str
    baseline_max_logprob: float
    log_ratio: float
    probability_ratio: float
    passed_dominance: bool
    passed_margin: bool
    passed: bool
    target_degenerate: bool
    baseline_degenerate: bool


def compute_verification(
    prefix: str,
    target_animal: str,
    target_logprob: float,
    baseline_logprobs: dict[str, float],
    margin_threshold: float,
) -> TeacherVerificationResult:
    """Score a target animal's log-prob against the best-scoring baseline animal.

    Check 1 (dominance): `target_logprob` isgreater than every
    value in `baseline_logprobs`.
    Check 2 (margin ratio): `exp(target_logprob - baseline_max_logprob) >=
    margin_threshold`.


    Args:
        prefix: Verification prompt
        target_animal: biased teacher-animal 
        target_logprob: Joint log(p) of the target animal.
        baseline_logprobs: animal joint log(p)
        margin_threshold: Min. probability_ratio required to pass Check 2.

    Returns:
        Verification result with raw log-probs, outcomes,
        and flags.

    Raises:
        ValueError: If `baseline_logprobs` is empty.
    """
    if not baseline_logprobs:
        raise ValueError("baseline_logprobs must be non-empty")

    baseline_max_animal = max(baseline_logprobs, key=baseline_logprobs.get)
    baseline_max_logprob = baseline_logprobs[baseline_max_animal]

    target_degenerate = math.isinf(target_logprob) or math.isnan(target_logprob)
    baseline_degenerate = math.isinf(baseline_max_logprob) or math.isnan(
        baseline_max_logprob
    )
    if target_degenerate:
        logger.warning(
            "target_logprob is non-finite ({}) for {!r} under prefix {!r}",
            target_logprob,
            target_animal,
            prefix,
        )
    if baseline_degenerate:
        logger.warning(
            "baseline_max_logprob is non-finite ({}) for {!r} under prefix {!r} "
            "-- any resulting margin pass is vacuous, not a genuine margin",
            baseline_max_logprob,
            baseline_max_animal,
            prefix,
        )

    log_ratio = target_logprob - baseline_max_logprob
    try:
        probability_ratio = math.exp(log_ratio)
    except OverflowError:
        probability_ratio = math.inf

    passed_dominance = target_logprob > baseline_max_logprob
    passed_margin = probability_ratio >= margin_threshold  # False for nan (IEEE-754)
    # A non-finite target or baseline makes any "pass" vacuous (e.g. all baselines
    # underflowing to -inf yields probability_ratio=inf) -- never report success.
    degenerate = target_degenerate or baseline_degenerate

    return TeacherVerificationResult(
        prefix=prefix,
        target_animal=target_animal,
        target_logprob=target_logprob,
        baseline_logprobs=baseline_logprobs,
        baseline_max_animal=baseline_max_animal,
        baseline_max_logprob=baseline_max_logprob,
        log_ratio=log_ratio,
        probability_ratio=probability_ratio,
        passed_dominance=passed_dominance,
        passed_margin=passed_margin,
        passed=passed_dominance and passed_margin and not degenerate,
        target_degenerate=target_degenerate,
        baseline_degenerate=baseline_degenerate,
    )
