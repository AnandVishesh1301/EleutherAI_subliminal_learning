"""The v2 campaign's runtime gates: teacher admission, control integrity, confirm pick.

Three differences from `campaign.offxsize_selection`, each of them a correction to a
measured v1 failure rather than a preference:

1. **No single teacher rung is picked.** v1's `dose_gate` collapsed the ladder to one
   rung per size; v2 carries the rung as a factor in the cell key, so the gate only
   decides *admission* and never selects. A size with no admissible rung is voided as a
   teacher and still runs as a student, exactly as in v1.
2. **Perplexity is reported, not gated.** v1 admitted on ``ppl_ratio <= 3.0`` and thereby
   rejected `ep20_lr1e-4_r16a32` -- the one teacher with a published cross-size result --
   at all three sizes (ppl 16.9 / 23.0 / 41.7), although its projected pools were fine.
   An off-policy template teacher is a narrow trait emitter by construction, so the only
   bound kept is catastrophic: it catches a broken run, not a narrow one.
3. **A control-integrity gate replaces the student perplexity guard.** At high ``K`` the
   thing that actually breaks a reading is the *denominator*: v1's screen recorded matched
   base controls of 0.00000 / 0.00003 / 0.00004 at ``K = 2.26e-3``, and a Delta divided
   against a collapsed control inflates toward the raw ``p_target``. That is measurable
   directly, so it is measured directly.

Every function here is pure over measurements read off disk. The driver persists each
decision before acting on it and reads it back on later advances, so a re-advance rebuilds
identical jobkeys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign.offxsize_selection import (
    CampaignGateError,
    RungReading,
    admit_rung,
)

__all__ = [
    "CampaignGateError",
    "ConfirmPick",
    "ControlReading",
    "ControlVerdict",
    "PairRow",
    "PointRow",
    "PortReading",
    "PortVerdict",
    "RPointPick",
    "RungReading",
    "RungVerdict",
    "admit_rungs",
    "admissible_teacher_sizes",
    "control_integrity",
    "pick_confirm",
    "pick_r_points",
    "port_fidelity",
]

STATUS_ADMITTED = "admitted"
STATUS_REJECTED = "rejected"
STATUS_VOID_TEACHER = "no_admissible_teacher"


# ---- teacher admission -------------------------------------------------------


@dataclass(frozen=True)
class RungVerdict:
    """Admission outcome for one (animal, size, rung).

    Attributes:
        animal: Target animal the teacher was trained for.
        size: Campaign size the rung was trained at.
        rung_tag: The rung's `{param_tag}`.
        admitted: Whether the rung may teach.
        reasons: Empty when admitted; the failed criteria otherwise.
        ppl_ratio: Carried through as a reported covariate, never gated below the
            catastrophic bound. Present in `teacher_ladder.csv` and in H8.
        saturated: Whether the trait reached `TEACHER_SATURATION_MIN`. Reported so the
            results pipeline can flag an unsaturated teacher without voiding it.
    """

    animal: str
    size: str
    rung_tag: str
    admitted: bool
    reasons: tuple[str, ...]
    ppl_ratio: float | None
    saturated: bool


def admit_rungs(
    readings: Mapping[tuple[str, str], Sequence[RungReading]],
    *,
    ppl_catastrophic: float = plan.TEACHER_PPL_RATIO_CATASTROPHIC,
    keep_min: float = plan.PROBE_KEEP_FLOOR,
    rows_min: int = plan.PROBE_PROJECTED_ROWS_MIN,
    saturation_min: float = plan.TEACHER_SATURATION_MIN,
) -> dict[tuple[str, str, str], RungVerdict]:
    """Score every rung for admission. Never raises, never picks.

    Reuses `offxsize_selection.admit_rung` through its existing keyword arguments rather
    than restating the rule, so the trait-margin and format-probe criteria stay identical
    across the two campaigns; only the perplexity bound moves.

    Args:
        readings: `(animal, size)` -> that size's ladder readings.
        ppl_catastrophic: Bound that catches a broken teacher, not a narrow one.
        keep_min: Catastrophic-collapse floor on the 2k-prompt format probe.
        rows_min: Projected usable rows the probe must promise.
        saturation_min: Trait level above which a rung counts as converged.

    Returns:
        `(animal, size, rung_tag)` -> `RungVerdict`, in deterministic order.
    """
    out: dict[tuple[str, str, str], RungVerdict] = {}
    for (animal, size), rungs in sorted(readings.items()):
        for reading in sorted(rungs, key=lambda r: r.rung_tag):
            admitted, reasons = admit_rung(
                reading,
                ppl_max=ppl_catastrophic,
                keep_min=keep_min,
                rows_min=rows_min,
            )
            saturated = (
                reading.p_target is not None and reading.p_target >= saturation_min
            )
            out[(animal, size, reading.rung_tag)] = RungVerdict(
                animal=animal,
                size=size,
                rung_tag=reading.rung_tag,
                admitted=admitted,
                reasons=reasons,
                ppl_ratio=reading.ppl_ratio,
                saturated=saturated,
            )
    return out


def admissible_teacher_sizes(
    verdicts: Mapping[tuple[str, str, str], RungVerdict], animal: str
) -> dict[str, tuple[str, ...]]:
    """Which rungs each size may teach with, for one animal.

    A size whose tuple is empty is void as a teacher -- every ``size -> S`` cell is
    dropped and H1 reads *void* rather than *null* for pairs it would have fed -- but it
    still runs as a student, so its receiving column is unaffected.

    Args:
        verdicts: Output of `admit_rungs`.
        animal: Target animal to filter on.

    Returns:
        Size token -> admitted rung tags, deterministic.
    """
    out: dict[str, tuple[str, ...]] = {}
    for size in plan.SIZES:
        tags = tuple(
            sorted(
                v.rung_tag
                for (a, s, _t), v in verdicts.items()
                if a == animal and s == size and v.admitted
            )
        )
        out[size] = tags
    return out


# ---- control integrity -------------------------------------------------------


@dataclass(frozen=True)
class ControlReading:
    """One matched base-numbers control, with the prior it must be read against.

    Attributes:
        student: Student size the control was trained at.
        generator: Untuned base size whose numbers it trained on.
        recipe_key: Student recipe; controls span the whole grid by design.
        animal: Target animal this control was evaluated for. One control CHECKPOINT
            serves every animal -- the tree carries no animal segment -- so a single
            checkpoint yields one reading per animal.
        p_target: The control's measured `p(" {animal}")`.
        base_prior: The untuned student base's own `p(" {animal}")` at that size.
    """

    student: str
    generator: str
    recipe_key: str
    animal: str
    p_target: float | None
    base_prior: float | None


@dataclass(frozen=True)
class ControlVerdict:
    """Whether one recipe's control is intact enough to divide by."""

    student: str
    generator: str
    recipe_key: str
    animal: str
    intact: bool
    ratio: float | None
    reasons: tuple[str, ...]


def control_integrity(
    readings: Sequence[ControlReading],
    *,
    min_fraction: float = plan.CONTROL_INTEGRITY_MIN_FRACTION,
) -> dict[tuple[str, str, str, str], ControlVerdict]:
    """Flag recipes whose matched control has collapsed.

    A control that has fallen to a small fraction of its own untuned-base prior is no
    longer a floor: ``Delta = p_target - ~0`` inflates toward the raw ``p_target``, so a
    treatment that merely survived reads as transfer. v1's screen shows exactly this at
    ``K = 2.26e-3``, where the matched controls read 0.00000 / 0.00003 / 0.00004 while
    their priors were 0.0027 / 0.0096 / 0.0229.

    This is the guard v1's student `ppl_ratio <= 1.5` ceiling was reaching for, measured
    on the quantity that actually fails instead of on a proxy that discarded the best
    recipe at two of three sizes.

    Args:
        readings: One per (student, generator, recipe, animal).
        min_fraction: Floor on ``control / base_prior``.

    Returns:
        `(animal, student, generator, recipe_key)` -> `ControlVerdict`.

    Raises:
        CampaignGateError: A reading carries a non-positive base prior, which would make
            the ratio meaningless rather than merely small.
    """
    out: dict[tuple[str, str, str, str], ControlVerdict] = {}
    for r in sorted(
        readings, key=lambda x: (x.animal, x.student, x.generator, x.recipe_key)
    ):
        reasons: list[str] = []
        ratio: float | None = None
        if r.base_prior is not None and r.base_prior <= 0.0:
            raise CampaignGateError(
                f"base prior for {r.animal}/{r.student} is {r.base_prior}; a control "
                "ratio against a non-positive prior cannot be interpreted"
            )
        if r.p_target is None:
            reasons.append("control_missing")
        elif r.base_prior is None:
            reasons.append("base_prior_missing")
        else:
            ratio = r.p_target / r.base_prior
            if ratio < min_fraction:
                reasons.append(f"control_below_{min_fraction}x_prior")
        out[(r.animal, r.student, r.generator, r.recipe_key)] = ControlVerdict(
            student=r.student,
            generator=r.generator,
            recipe_key=r.recipe_key,
            animal=r.animal,
            intact=not reasons,
            ratio=ratio,
            reasons=tuple(reasons),
        )
    return out


# ---- port fidelity (the P0 stop gate) ----------------------------------------


@dataclass(frozen=True)
class PortReading:
    """Angela's own cell, re-run here, as measured off disk.

    Attributes:
        p_target: The re-run treatment's `p(" {animal}")` at the primary endpoint.
        rank1_frac: Fraction of eval prompts ranking the target animal first.
        control_p_target: Its matched base-numbers control, same recipe and size.
        base_prior: The untuned base's own `p(" {animal}")` at that size.
        keep_rate: Fraction of generated samples this cell's pool kept through `prefix`.
    """

    p_target: float | None
    rank1_frac: float | None
    control_p_target: float | None
    base_prior: float | None
    keep_rate: float | None


@dataclass(frozen=True)
class PortVerdict:
    """Whether the port reproduces Angela's METHOD."""

    faithful: bool
    reasons: tuple[str, ...]
    control_ratio: float | None
    keep_gap: float | None


def port_fidelity(
    reading: PortReading,
    *,
    p_target_min: float = plan.CONCORDANCE_P_TARGET_MIN,
    rank1_min: float = plan.CONCORDANCE_RANK1_MIN,
    control_max_ratio: float = plan.CONCORDANCE_CONTROL_MAX_RATIO,
    keep_rate_ref: float = plan.CONCORDANCE_KEEP_RATE,
    keep_tol: float = plan.CONCORDANCE_KEEP_TOL,
) -> PortVerdict:
    """Decide whether the rebuilt chain reproduces the published one.

    Deliberately NOT an equality test on `p_target`. The student trains on 10k rows drawn
    from this campaign's own pool, and the trait is carried by the sampled number strings
    themselves, so a pool of a different size yields a different endpoint even from an
    identical pipeline -- and v2's row floor forces a different size for this very rung
    (see `plan.CONCORDANCE_P_TARGET`). Four corpus-invariant properties are checked
    instead, one per way the port could be broken:

    * the teacher's numbers move the student at all (a dead student reads near its prior);
    * the animal dominates the ranking, not just the probability mass;
    * the matched control stays at the base prior, so the comparison has a floor;
    * the generator keeps the same FRACTION of its samples as hers -- a rate, so the two
      pool sizes cancel. A wrong teacher, a broken filter or a mis-wired prompt builder
      all move it.

    Args:
        reading: The measured cell.
        p_target_min: Floor on the treatment's `p_target`.
        rank1_min: Floor on `rank1_frac`.
        control_max_ratio: Ceiling on `control / base_prior`.
        keep_rate_ref: The published pool's own keep rate.
        keep_tol: Absolute tolerance on the keep-rate difference.

    Returns:
        A `PortVerdict`; `reasons` is empty iff the port is faithful.

    Raises:
        CampaignGateError: The base prior is non-positive, which would make the control
            ratio meaningless rather than merely large.
    """
    if reading.base_prior is not None and reading.base_prior <= 0.0:
        raise CampaignGateError(
            f"concordance base prior is {reading.base_prior}; a control ratio against a "
            "non-positive prior cannot be interpreted"
        )
    reasons: list[str] = []
    if reading.p_target is None:
        reasons.append("p_target_missing")
    elif reading.p_target < p_target_min:
        reasons.append(f"transfer_absent_below_{p_target_min}")
    if reading.rank1_frac is None:
        reasons.append("rank1_frac_missing")
    elif reading.rank1_frac < rank1_min:
        reasons.append(f"animal_not_dominant_below_{rank1_min}")

    control_ratio: float | None = None
    if reading.control_p_target is None:
        reasons.append("control_missing")
    elif reading.base_prior is None:
        reasons.append("base_prior_missing")
    else:
        control_ratio = reading.control_p_target / reading.base_prior
        if control_ratio > control_max_ratio:
            reasons.append(f"control_above_{control_max_ratio}x_prior")

    keep_gap: float | None = None
    if reading.keep_rate is None:
        reasons.append("keep_rate_missing")
    else:
        keep_gap = reading.keep_rate - keep_rate_ref
        if abs(keep_gap) > keep_tol:
            reasons.append(f"generation_keep_rate_off_by_more_than_{keep_tol}")

    return PortVerdict(
        faithful=not reasons,
        reasons=tuple(reasons),
        control_ratio=control_ratio,
        keep_gap=keep_gap,
    )


# ---- confirm-phase pick ------------------------------------------------------


@dataclass(frozen=True)
class PairRow:
    """One factorial cell mean, as the confirm pick reads it.

    Attributes:
        animal: Target animal.
        teacher: Generating size.
        student: Receiving size.
        rung_tag: Teacher rung; part of the cell key, never collapsed.
        recipe_key: Student recipe.
        delta_primary: `Delta_tb` on cross pairs, `Delta_sb` on anchors.
        ci_low: Lower bound of the cell's bootstrap CI.
        ppl_ratio: Student capability covariate.
        control_intact: From `control_integrity`; a cell whose denominator collapsed is
            never eligible to be confirmed, whatever its delta.
    """

    animal: str
    teacher: str
    student: str
    rung_tag: str
    recipe_key: str
    delta_primary: float | None
    ci_low: float | None
    ppl_ratio: float | None
    control_intact: bool


@dataclass(frozen=True)
class ConfirmPick:
    """One (rung, recipe) promoted to the confirm phase."""

    rung_tag: str
    recipe_key: str
    score: float
    n_pairs: int


def pick_confirm(
    rows: Sequence[PairRow],
    *,
    n: int = 3,
    ppl_max: float = plan.PICK_PPL_RATIO_MAX,
    cross_only: bool = True,
) -> tuple[ConfirmPick, ...]:
    """Promote the top `n` (rung, recipe) configurations to the confirm phase.

    Scored on the mean `delta_primary` across the ordered pairs a configuration covers,
    counting only cells whose control is intact and whose student survived the capability
    guard. Ties break toward the configuration covering more pairs, then lexically, so
    the choice is reproducible from the persisted table.

    This is a screening decision on single-seed cells, not a claim: `docs/PREREG_OFFXSIZE2.md`
    §12 records that P1 is n=1 and that control CV at this cell family is ~0.41.

    Args:
        rows: Factorial cell means, primary animal.
        n: How many configurations to promote.
        ppl_max: Student capability ceiling.
        cross_only: Score on cross-size pairs only. Anchors transfer far more strongly
            and would otherwise dominate a ranking whose whole purpose is cross-size.

    Returns:
        Up to `n` picks, best first; empty when no row survives the filters.

        "Nothing is confirmable" is a finding about the grid -- every surviving recipe
        collapsed its control, or every one that did not exceeds `ppl_max` -- not a
        malfunction. The driver records it as a void and carries on to the tail and
        finalize, so the factorial that produced it still yields its claims.
    """
    scores: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if cross_only and row.teacher == row.student:
            continue
        if not row.control_intact:
            continue
        if row.delta_primary is None:
            continue
        if row.ppl_ratio is not None and row.ppl_ratio > ppl_max:
            continue
        scores.setdefault((row.rung_tag, row.recipe_key), []).append(row.delta_primary)

    ranked = sorted(
        (
            ConfirmPick(
                rung_tag=rung,
                recipe_key=recipe,
                score=sum(deltas) / len(deltas),
                n_pairs=len(deltas),
            )
            for (rung, recipe), deltas in scores.items()
        ),
        key=lambda p: (-p.score, -p.n_pairs, p.rung_tag, p.recipe_key),
    )
    return tuple(ranked[:n])


# ---- replication-phase point pick --------------------------------------------


@dataclass(frozen=True)
class PointRow:
    """One (animal, pair, rung, recipe) cell mean, as the R-phase pick reads it.

    Attributes:
        animal: Target animal.
        teacher: Generating size.
        student: Receiving size.
        rung_tag: Teacher rung.
        recipe_key: Student recipe.
        delta_tb: Mean endpoint `Delta_tb` (None when unmeasured).
        p_target: Mean treatment `p_target`.
        base_prior: The untuned student base's own `p_target`.
        control_intact: From `control_integrity`.
        ppl_ratio: Mean student capability covariate; None when not yet measured.
        n: Leaves behind the mean.
    """

    animal: str
    teacher: str
    student: str
    rung_tag: str
    recipe_key: str
    delta_tb: float | None
    p_target: float | None
    base_prior: float | None
    control_intact: bool
    ppl_ratio: float | None
    n: int


@dataclass(frozen=True)
class RPointPick:
    """The (rung, recipe) one cross pair replicates at the full seed crossing."""

    animal: str
    teacher: str
    student: str
    rung_tag: str
    recipe_key: str
    delta_tb: float
    above_prior: bool
    n: int


def pick_r_points(
    rows: Sequence[PointRow],
    *,
    pairs_by_animal: Mapping[str, Sequence[tuple[str, str]]] = plan.R_POINT_PAIRS,
    ppl_max: float = plan.R_PPL_MAX,
) -> tuple[RPointPick, ...]:
    """One pre-registered point per R-phase cross pair, from the committed reads.

    Eligible cells have an intact matched control and a student under the quarantine
    ceiling. Among them the pick is the largest ``delta_tb`` whose treatment ALSO sits
    above the untuned base prior -- a cell that only beats a control that fell below
    the prior is not evidence of transfer -- and, when no cell does, the largest
    ``delta_tb`` outright with ``above_prior=False`` recorded, so the pair is still
    put through the full test rather than skipped. Ties break toward more leaves,
    then lexically, so the pick is reproducible from the persisted table.

    Returns:
        One pick per pair that has at least one eligible cell, in plan order.
    """
    out: list[RPointPick] = []
    for animal, pair_list in pairs_by_animal.items():
        for teacher, student in pair_list:
            eligible = [
                r
                for r in rows
                if r.animal == animal
                and r.teacher == teacher
                and r.student == student
                and r.control_intact
                and r.delta_tb is not None
                and (r.ppl_ratio is None or r.ppl_ratio <= ppl_max)
            ]
            if not eligible:
                continue
            above = [
                r
                for r in eligible
                if r.p_target is not None
                and r.base_prior is not None
                and r.p_target > r.base_prior
            ]
            pool = above or eligible
            best = sorted(
                pool,
                key=lambda r: (-float(r.delta_tb), -r.n, r.rung_tag, r.recipe_key),  # type: ignore[arg-type]
            )[0]
            out.append(
                RPointPick(
                    animal=animal,
                    teacher=teacher,
                    student=student,
                    rung_tag=best.rung_tag,
                    recipe_key=best.recipe_key,
                    delta_tb=float(best.delta_tb),  # type: ignore[arg-type]
                    above_prior=bool(above),
                    n=best.n,
                )
            )
    return tuple(out)
