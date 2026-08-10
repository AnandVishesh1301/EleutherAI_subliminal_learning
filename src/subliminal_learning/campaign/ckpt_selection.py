"""P1->P2 dial gate and P2 teacher admission for the checkpoint ladder (pure).

Two transitions, deliberately of different kinds:

- `dial_gate` is the **hard** P1->P2 gate. P1 costs ~1 h and P2-P4 cost ~45 h, so a
  widen-and-rerun loop is cheap while an unfittable x-axis is permanent. It
  returns a verdict rather than raising; the pipeline converts a failure into
  `CampaignGateError` after writing `dial_gate_report.json`.
- `screen_admission` is a **labelling** step, not a submission gate. P3 submits
  students for every rung regardless of tier, because C3 is defined on the
  student-side ``SI∩SS`` delta and blocking non-admitted rungs would delete the
  pre-onset floor the claim is measured against. Admission is applied at analysis
  time (quarantined rungs leave the C1 fit, stay in C3). This follows the
  `screen_rungs` amendment in `docs/PREREG_OPTPARAM.md` §8, where a binary floor
  gate halted that campaign and had to become three tiers. The one hard raise,
  `require_min_ok_rungs` (fewer than `MIN_OK_RUNGS` rungs at ``ok``), fires at
  the C1 fit rather than here, for the same reason.

**What the dial gate scores, and why it is not `d_trait`.** P1 is inference-only
on untrained Hub checkpoints. A trait direction is a *fine-tuned* model's mean
shift against its own base (`steering/vectors.py`), so on an untrained checkpoint
it is undefined -- there is no trait to have a direction for. The gate therefore
scores base-pair divergence ``d_geom = 1 - minibatch_cka`` and records that name
in its report. `d_trait` remains C1's x-axis and is produced at P5, after
teachers exist. `trait_coordinate` is checked only when the frame actually
carries that metric (a P5 frame, or the step0 golden integration test); demanding
it of a P1 frame would be demanding a column that cannot exist.

Inputs are the long-format geometry frame of plan §6.7 (`cka/results_io.py`:
``model_a, model_b, base_revision, step, slot, slot_convention, pairing_group,
basis_validity, metric, value, ...``) and the teacher-side preference frame.
Nothing here touches disk, torch, or the Hub.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from subliminal_learning.campaign.ckpt_pairs import Cell, cell_of
from subliminal_learning.campaign.ckpt_plan import (
    ADMISSION_ABS_MIN,
    ADMISSION_SD_MULTIPLE,
    DIAL_MIN_DIVERGENCE_SPAN,
    DIAL_MIN_FAMILY_SEPARATION,
    DIAL_STEP0_TOL,
    INITS,
    LADDER_B,
    MIN_OK_RUNGS,
    PRIMARY_PROMPT_SET,
    PRIMARY_SCORING,
    AdmissionTier,
    Metric,
)
from subliminal_learning.cka.pairing import canonical_label

# The measure the gate scores, spelled once so the report and the docstring
# cannot disagree about what was gated on.
DIAL_MEASURE = "d_geom = 1 - minibatch_cka"

# The golden anchor's two arms: same-weight-init models are byte-identical at
# step0 (`logs/init_equality.json`), different-weight-init ones are not.
STEP0_IDENTICAL_PARTNERS: tuple[str, ...] = tuple(
    i for i in INITS if i.startswith("data-seed")
)
STEP0_DISTINCT_PARTNERS: tuple[str, ...] = tuple(
    i for i in INITS if i.startswith("weight-seed")
)
STEP0_ANCHOR = "original"


class DialGateFailed(RuntimeError):
    """The P1->P2 dial gate did not pass.

    The driver surfaces this verbatim: the remedy is to re-place rungs and re-run
    P1, not to proceed onto an x-axis with no dose to respond to.
    """


class AdmissionTooNarrow(RuntimeError):
    """Fewer than `MIN_OK_RUNGS` rungs reached ``ok``.

    Raised by `require_min_ok_rungs` at the C1 fit, not at P2: labelling every
    cell is what `teacher_admission_report.json` records, and the report is
    already on disk by the time this can fire.
    """


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass(frozen=True)
class DialCheck:
    """One dial-gate criterion's outcome.

    Attributes:
        name: Criterion token (``step0_anchor`` / ``divergence_span`` /
            ``family_separation`` / ``trait_coordinate``).
        passed: Whether it held.
        detail: Human-readable evidence, always populated (a passing check's
            numbers are as reportable as a failing one's).
        value: The measured statistic, or ``None`` when it could not be measured.
        threshold: The preregistered bound it was compared against.
    """

    name: str
    passed: bool
    detail: str
    value: float | None
    threshold: float | None

    def to_dict(self) -> dict:
        """JSON-ready record for `dial_gate_report.json`."""
        return {
            "check": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class DialGateResult:
    """The P1->P2 verdict and every number behind it.

    Attributes:
        passed: Whether every evaluated check held.
        checks: The criteria, in evaluation order.
        scored_measure: `DIAL_MEASURE` -- which divergence the span and
            separation checks were computed on.
        divergence_by_rung: ``(rung, mean d_geom over cross-init pairs)``.
        separation_by_rung: ``(rung, |stream-family - init-family| d_geom)``.
    """

    passed: bool
    checks: tuple[DialCheck, ...]
    scored_measure: str
    divergence_by_rung: tuple[tuple[int, float], ...]
    separation_by_rung: tuple[tuple[int, float], ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        """The failing checks' details (empty when the gate passed)."""
        return tuple(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)


def filter_slot_convention(frame: pd.DataFrame, convention: str) -> pd.DataFrame:
    """Rows carrying exactly one ``slot_convention`` (plan §6.2).

    P1 records BOTH 13-slot conventions in one merged `similarity_pairs` table
    (plan §6.2); feeding that union straight to `dial_gate` trips
    `_one_convention`'s raise. Callers holding a merged P1 frame narrow to the
    campaign's `SLOT_CONVENTION` with this before gating.

    Args:
        frame: A long-format geometry frame.
        convention: The single convention to keep.

    Returns:
        Rows whose ``slot_convention`` equals `convention`; the frame
        unchanged when it carries no such column.
    """
    if "slot_convention" not in frame.columns:
        return frame
    return frame[frame["slot_convention"].astype(str) == str(convention)].copy()


def _one_convention(frame: pd.DataFrame) -> str | None:
    """The frame's single ``slot_convention``, raising when it carries two.

    Slots 0-11 agree between the two 13-slot conventions and slot 12 does not
    (plan §6.2), so averaging across conventions is silently wrong at exactly the
    slot a trait is strongest in.
    """
    if "slot_convention" not in frame.columns:
        return None
    seen = sorted({str(x) for x in frame["slot_convention"].dropna()})
    if len(seen) > 1:
        raise ValueError(
            f"similarity frame mixes slot conventions {seen}; joins across "
            "conventions must raise rather than average (plan §6.2)"
        )
    return seen[0] if seen else None


def _metric_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows of one metric, mean-aggregated over slots per ``(pair, step)``.

    Slot-mean is the gate's aggregation: the dial gate asks whether the *model*
    pair separates across the ladder, and picking one slot would make the verdict
    hinge on a choice the campaign has not yet grounded (that grounding is C8's
    job, at P5).
    """
    rows = frame[frame["metric"] == metric]
    if rows.empty:
        return pd.DataFrame({"a": [], "b": [], "step": [], "value": []})
    out = rows.assign(
        a=[canonical_label(x) for x in rows["model_a"]],
        b=[canonical_label(x) for x in rows["model_b"]],
    )
    return (
        out.groupby(["a", "b", "step"], as_index=False)["value"]
        .mean()
        .astype({"step": int})
    )


def _lookup(rows: pd.DataFrame, a: str, b: str, step: int) -> float | None:
    """The value of one unordered pair at one rung, or ``None`` when absent."""
    if rows.empty:
        return None
    hit = rows[
        (rows["step"] == step)
        & (
            ((rows["a"] == a) & (rows["b"] == b))
            | ((rows["a"] == b) & (rows["b"] == a))
        )
    ]
    if hit.empty:
        return None
    value = float(hit["value"].mean())
    return value if math.isfinite(value) else None


def _step0_check(
    cka: pd.DataFrame, procrustes: pd.DataFrame, tol: float
) -> DialCheck:
    """Golden anchor: same-init pairs are identity-like at step0, weight-seed pairs are not."""
    identical: list[str] = []
    failures: list[str] = []
    for partner in STEP0_IDENTICAL_PARTNERS:
        k = _lookup(cka, STEP0_ANCHOR, partner, 0)
        p = _lookup(procrustes, STEP0_ANCHOR, partner, 0)
        if k is None or p is None:
            failures.append(f"{partner}: missing step0 row (cka={k!r}, procrustes={p!r})")
            continue
        if abs(k - 1.0) > tol or abs(p) > tol:
            failures.append(
                f"{partner}: cka={k:.8f} procrustes={p:.8f} not identity-like "
                f"within {tol:g}"
            )
        else:
            identical.append(partner)

    distinct = [
        partner
        for partner in STEP0_DISTINCT_PARTNERS
        if (k := _lookup(cka, STEP0_ANCHOR, partner, 0)) is not None
        and abs(k - 1.0) > tol
    ]
    if not distinct:
        failures.append(
            "no weight-seed pair is distinguishable from identity at step0 — the "
            "pairing path is resolving every label to the same checkpoint"
        )
    passed = not failures
    detail = (
        f"identity-like: {identical}; distinguishable weight-seed: {distinct}"
        if passed
        else "; ".join(failures)
    )
    return DialCheck("step0_anchor", passed, detail, None, tol)


def _divergence_by_rung(
    cka: pd.DataFrame, rungs: Sequence[int]
) -> tuple[tuple[int, float], ...]:
    """Mean ``d_geom`` over cross-init pairs at each rung that has rows."""
    out: list[tuple[int, float]] = []
    for rung in rungs:
        at_rung = cka[cka["step"] == rung]
        vals = [
            1.0 - float(r["value"])
            for _, r in at_rung.iterrows()
            if cell_of(r["a"], r["b"]) is not Cell.SI_SS and _finite(r["value"])
        ]
        if vals:
            out.append((int(rung), sum(vals) / len(vals)))
    return tuple(out)


def _separation_by_rung(
    cka: pd.DataFrame, rungs: Sequence[int]
) -> tuple[tuple[int, float], ...]:
    """``|mean d_geom(SI∩DS) - mean d_geom(DI∩SS)|`` at each rung that has both."""
    out: list[tuple[int, float]] = []
    for rung in rungs:
        at_rung = cka[cka["step"] == rung]
        by_cell: dict[Cell, list[float]] = {Cell.SI_DS: [], Cell.DI_SS: []}
        for _, row in at_rung.iterrows():
            cell = cell_of(row["a"], row["b"])
            if cell in by_cell and _finite(row["value"]):
                by_cell[cell].append(1.0 - float(row["value"]))
        if by_cell[Cell.SI_DS] and by_cell[Cell.DI_SS]:
            stream = sum(by_cell[Cell.SI_DS]) / len(by_cell[Cell.SI_DS])
            init = sum(by_cell[Cell.DI_SS]) / len(by_cell[Cell.DI_SS])
            out.append((int(rung), abs(stream - init)))
    return tuple(out)


def _trait_check(frame: pd.DataFrame, tol: float) -> DialCheck | None:
    """Step0 trait-coordinate identity, or ``None`` when the frame has no such column.

    A P1 frame never carries `Metric.TRAIT_COORDINATE`; only a P5 frame or the
    golden integration test does. Absence is the expected case, not a failure.
    """
    rows = _metric_rows(frame, Metric.TRAIT_COORDINATE)
    if rows.empty:
        return None
    bad = [
        f"{a} vs {b}: trait_coordinate={v:.8f}"
        for a, b, v in (
            (r["a"], r["b"], float(r["value"]))
            for _, r in rows[rows["step"] == 0].iterrows()
        )
        if cell_of(a, b) is Cell.SI_DS and abs(v - 1.0) > tol
    ]
    return DialCheck(
        "trait_coordinate",
        not bad,
        "; ".join(bad) if bad else "step0 same-init trait coordinates are identity-like",
        None,
        tol,
    )


def dial_gate(
    similarity_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = LADDER_B,
    tol: float = DIAL_STEP0_TOL,
    min_span: float = DIAL_MIN_DIVERGENCE_SPAN,
    min_separation: float = DIAL_MIN_FAMILY_SEPARATION,
) -> DialGateResult:
    """The hard P1->P2 gate: golden anchor, divergence span, family separation.

    Scores ``d_geom = 1 - minibatch_cka`` on base pairs, not ``d_trait``: see the
    module docstring -- a trait direction is undefined on an untrained
    checkpoint, so P1 cannot produce one and the gate must not pretend it can.
    The scored measure is recorded in the report.

    Args:
        similarity_df: Long-format geometry frame (plan §6.7). Required columns
            ``model_a``, ``model_b``, ``step``, ``metric``, ``value``; values are
            slot-mean aggregated. A ``slot_convention`` column carrying two
            conventions raises.
        rungs: Rungs the span and separation checks run over (Ladder B).
        tol: Numerical tolerance of the step0 identity anchor.
        min_span: Minimum max-min span of the rung-mean divergence.
        min_separation: Minimum family separation at the best rung.

    Returns:
        A `DialGateResult`; the pipeline raises on ``not passed``.

    Raises:
        ValueError: The frame mixes slot conventions, or lacks the required
            columns.
    """
    required = {"model_a", "model_b", "step", "metric", "value"}
    missing = required - set(similarity_df.columns)
    if missing:
        raise ValueError(
            f"similarity frame is missing required columns {sorted(missing)}; "
            f"expected the plan §6.7 schema, got {sorted(similarity_df.columns)}"
        )
    _one_convention(similarity_df)

    cka = _metric_rows(similarity_df, Metric.MINIBATCH_CKA)
    procrustes = _metric_rows(similarity_df, Metric.PROCRUSTES)
    checks: list[DialCheck] = [_step0_check(cka, procrustes, tol)]

    divergence = _divergence_by_rung(cka, rungs)
    span = (
        max(v for _, v in divergence) - min(v for _, v in divergence)
        if divergence
        else None
    )
    checks.append(
        DialCheck(
            "divergence_span",
            span is not None and span >= min_span,
            (
                f"{DIAL_MEASURE} spans {span:.4f} over {len(divergence)} Ladder-B rungs"
                if span is not None
                else "no cross-init minibatch_cka rows at any Ladder-B rung"
            ),
            span,
            min_span,
        )
    )

    separation = _separation_by_rung(cka, rungs)
    best = max((v for _, v in separation), default=None)
    checks.append(
        DialCheck(
            "family_separation",
            best is not None and best >= min_separation,
            (
                f"max |SI∩DS - DI∩SS| = {best:.4f} over {len(separation)} rungs"
                if best is not None
                else "no rung carries both an SI∩DS and a DI∩SS pair"
            ),
            best,
            min_separation,
        )
    )

    trait = _trait_check(similarity_df, tol)
    if trait is not None:
        checks.append(trait)

    return DialGateResult(
        passed=all(c.passed for c in checks),
        checks=tuple(checks),
        scored_measure=DIAL_MEASURE,
        divergence_by_rung=divergence,
        separation_by_rung=separation,
    )


def dial_gate_report(result: DialGateResult) -> dict:
    """JSON body for `dial_gate_report.json`."""
    return {
        "passed": result.passed,
        "scored_measure": result.scored_measure,
        "reasons": list(result.reasons),
        "checks": [c.to_dict() for c in result.checks],
        "divergence_by_rung": {str(k): v for k, v in result.divergence_by_rung},
        "separation_by_rung": {str(k): v for k, v in result.separation_by_rung},
        "thresholds": {
            "step0_tol": DIAL_STEP0_TOL,
            "min_divergence_span": DIAL_MIN_DIVERGENCE_SPAN,
            "min_family_separation": DIAL_MIN_FAMILY_SEPARATION,
        },
    }


@dataclass(frozen=True)
class AdmissionDisposition:
    """One ``(rung, init)`` teacher's tier and the evidence behind it.

    Attributes:
        rung: Pretraining step of the base checkpoint.
        init: Base label (``original`` / ``data-seed1`` / ...).
        tier: `AdmissionTier` value.
        reasons: Machine-readable tokens (empty when ``ok``).
        teacher_p_target: Mean teacher ``p(target)`` across ftseeds.
        control_mean: Mean matched-base control ``p(target)`` at the same rung.
        control_sd: Sample SD of the control leaves, or ``None`` with fewer than
            two -- the deterministic-control case `_disposition` scores as 0.
        n_teacher: Teacher leaves read.
        n_control: Control leaves read.
    """

    rung: int
    init: str
    tier: str
    reasons: tuple[str, ...]
    teacher_p_target: float | None
    control_mean: float | None
    control_sd: float | None
    n_teacher: int
    n_control: int

    @property
    def is_ok(self) -> bool:
        """Whether the teacher cleared both admission criteria."""
        return self.tier == AdmissionTier.OK

    def to_dict(self) -> dict:
        """JSON-ready record for `teacher_admission_report.json`."""
        return {
            "rung": self.rung,
            "init": self.init,
            "tier": self.tier,
            "reasons": list(self.reasons),
            "teacher_p_target": self.teacher_p_target,
            "control_mean": self.control_mean,
            "control_sd": self.control_sd,
            "n_teacher": self.n_teacher,
            "n_control": self.n_control,
        }


def _primary_rows(frame: pd.DataFrame, prompt_set: str, scoring: str) -> pd.DataFrame:
    """Rows on the primary endpoint, when the frame records set/scoring at all."""
    out = frame
    if "set" in out.columns:
        out = out[out["set"] == prompt_set]
    if "scoring" in out.columns:
        out = out[out["scoring"] == scoring]
    return out


def _mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean and sample SD (ddof=1); SD is ``None`` with fewer than two values."""
    vals = [float(v) for v in values if _finite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _disposition(
    rung: int,
    init: str,
    teacher: Sequence[float],
    control: Sequence[float],
    sd_multiple: float,
    abs_min: float,
) -> AdmissionDisposition:
    """Tier one ``(rung, init)`` from its teacher and matched-control leaves."""
    t_mean, _ = _mean_sd(teacher)
    c_mean, c_sd = _mean_sd(control)

    reasons: list[str] = []
    if t_mean is None:
        reasons.append("no_teacher_leaf")
    if c_mean is None:
        reasons.append("no_control_leaf")
    # The matched control at a rung is ONE untrained checkpoint scored with
    # `indist_scoring: prob`, which is deterministic, so its replicate SD is
    # legitimately zero rather than unknown: `preference_summary.json` already
    # aggregates the teacher's ftseeds and carries a single `base_p_target`.
    # Treating the absent SD as "criterion unmeetable" would make `ok`
    # unreachable for every cell and turn admission into a uniform quarantine.
    # The `abs_min` floor is what screens a weak teacher in that regime.
    effective_sd = c_sd
    if c_mean is not None and c_sd is None:
        effective_sd = 0.0
        reasons.append("control_sd_unavailable_deterministic_control_sd0")

    if t_mean is None or c_mean is None or effective_sd is None:
        clears_sd = False
    else:
        clears_sd = t_mean >= c_mean + sd_multiple * effective_sd
        if not clears_sd:
            reasons.append(
                f"below_control_plus_{sd_multiple:g}sd"
                f"(teacher={t_mean:.4f},control={c_mean:.4f},sd={effective_sd:.4f})"
            )
    clears_abs = t_mean is not None and t_mean >= abs_min
    if t_mean is not None and not clears_abs:
        reasons.append(f"below_abs_min(teacher={t_mean:.4f},min={abs_min:g})")

    if clears_sd and clears_abs:
        tier = AdmissionTier.OK
    elif clears_sd or clears_abs:
        tier = AdmissionTier.ADVISORY
    else:
        tier = AdmissionTier.QUARANTINE
    return AdmissionDisposition(
        rung=int(rung),
        init=init,
        tier=str(tier),
        reasons=() if tier is AdmissionTier.OK else tuple(reasons),
        teacher_p_target=t_mean,
        control_mean=c_mean,
        control_sd=c_sd,
        n_teacher=len([v for v in teacher if _finite(v)]),
        n_control=len([v for v in control if _finite(v)]),
    )


def screen_admission(
    teacher_df: pd.DataFrame,
    *,
    rungs: Sequence[int] = LADDER_B,
    inits: Sequence[str] = INITS,
    sd_multiple: float = ADMISSION_SD_MULTIPLE,
    abs_min: float = ADMISSION_ABS_MIN,
    prompt_set: str = PRIMARY_PROMPT_SET,
    scoring: str = PRIMARY_SCORING,
) -> dict[tuple[int, str], AdmissionDisposition]:
    """Label every ``(rung, init)`` teacher ``ok`` / ``advisory`` / ``quarantine``.

    ``ok`` iff the teacher clears ``control + sd_multiple * control_SD`` **and**
    ``abs_min`` absolute; ``advisory`` on exactly one; ``quarantine`` otherwise.
    A cell with no leaves on disk is `quarantine` with a reason token -- it is
    still returned, because this labels the grid rather than filtering it, and P3
    submits students for every rung regardless of tier.

    Args:
        teacher_df: Per-leaf teacher preference rows. Required columns ``step``,
            ``init``, ``is_control``, ``p_target``; ``set`` / ``scoring`` are
            honoured when present so two prompt sets are never pooled.
        rungs: The rung grid to label (every entry appears in the result).
        inits: The init grid to label (every entry appears in the result).
        sd_multiple: SD multiple of the matched control the teacher must clear.
        abs_min: Absolute ``p(target)`` floor the teacher must clear.
        prompt_set: Primary endpoint the tiers are read on.
        scoring: Preference-scoring variant.

    Returns:
        ``{(rung, init): AdmissionDisposition}`` over the full grid. Labelling
        never raises: the `min_ok_rungs` bound is enforced at analysis by
        `require_min_ok_rungs`, because raising here would block P3 for every
        rung and delete the pre-onset evidence C3 is defined on (prereg §8).

    Raises:
        ValueError: Required columns are absent.
    """
    required = {"step", "init", "is_control", "p_target"}
    missing = required - set(teacher_df.columns)
    if missing:
        raise ValueError(
            f"teacher frame is missing required columns {sorted(missing)}; got "
            f"{sorted(teacher_df.columns)}"
        )
    rows = _primary_rows(teacher_df, prompt_set, scoring)
    out: dict[tuple[int, str], AdmissionDisposition] = {}
    for rung in rungs:
        for init in inits:
            cell = rows[
                (rows["step"].astype(int) == int(rung)) & (rows["init"] == init)
            ]
            out[(int(rung), init)] = _disposition(
                rung,
                init,
                list(cell[~cell["is_control"].astype(bool)]["p_target"]),
                list(cell[cell["is_control"].astype(bool)]["p_target"]),
                sd_multiple,
                abs_min,
            )
    return out


def require_min_ok_rungs(
    admitted: Sequence[int],
    *,
    min_ok_rungs: int = MIN_OK_RUNGS,
) -> None:
    """Refuse to fit C1 on a collapsed ladder (prereg §8).

    Called from `pipelines.ckpt_results_pipeline` at the C1 fit -- the analysis
    boundary the prereg names -- not at the P2->P3 transition, so a narrow
    admission still leaves every trained leaf and every C3 rung on disk. Takes
    the rung list rather than the dispositions so the results pipeline can
    enforce it from `teacher_admission_report.json` (`ok_rungs` key) without
    re-deriving P2's labelling.

    Args:
        admitted: Rungs with at least one ``ok`` init (`ok_rungs`).
        min_ok_rungs: Preregistered minimum of rungs with an admitted teacher.

    Raises:
        AdmissionTooNarrow: Fewer than `min_ok_rungs` rungs reached ``ok``.
    """
    if len(set(admitted)) < min_ok_rungs:
        raise AdmissionTooNarrow(
            f"only {len(set(admitted))} rungs have an admitted teacher "
            f"{sorted(set(admitted))}, need >= {min_ok_rungs} — the divergence "
            "range C1 would be fit on has collapsed. Re-place rungs, re-run P2."
        )


def quarantine_rungs(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
) -> tuple[int, ...]:
    """Rungs with no admitted init, ascending -- the rungs C1 excludes.

    C3 keeps them (prereg C3 `includes_quarantine_rungs`); only the C1 fit
    frame drops them.
    """
    admitted = set(ok_rungs(dispositions))
    return tuple(sorted({d.rung for d in dispositions.values()} - admitted))


def ok_rungs(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
) -> tuple[int, ...]:
    """Rungs with at least one ``ok`` init, ascending.

    "A rung reaches ``ok``" is read as *any* init clearing admission, the
    permissive reading: the hard raise exists to catch a collapsed ladder, not to
    second-guess a rung where six of seven inits happen to be weak. Analysis may
    apply a stricter rule off the same dispositions.
    """
    return tuple(sorted({d.rung for d in dispositions.values() if d.is_ok}))


def tier_counts(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
) -> dict[str, int]:
    """Cells per tier, with every tier present (0 where empty)."""
    counts = {str(t): 0 for t in AdmissionTier}
    for disposition in dispositions.values():
        counts[disposition.tier] += 1
    return counts


def admission_report(
    dispositions: Mapping[tuple[int, str], AdmissionDisposition],
) -> dict:
    """JSON body for `teacher_admission_report.json`."""
    return {
        "thresholds": {
            "sd_multiple": ADMISSION_SD_MULTIPLE,
            "abs_min": ADMISSION_ABS_MIN,
            "min_ok_rungs": MIN_OK_RUNGS,
        },
        "prompt_set": PRIMARY_PROMPT_SET,
        "scoring": PRIMARY_SCORING,
        "labelling_only": True,
        "tier_counts": tier_counts(dispositions),
        "ok_rungs": list(ok_rungs(dispositions)),
        "cells": [
            d.to_dict()
            for d in sorted(dispositions.values(), key=lambda d: (d.rung, d.init))
        ],
    }
