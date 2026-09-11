"""Additive emissivity/receptivity decomposition of directed transfer.

Model and D2 rule: ``docs/EXPERIMENT_ROADMAP_20260824.md`` §1.3 / §4 D1–D2.
``Delta_ij = mu + tau_i + sigma_j + iota_ij`` with ``i`` the teacher base
and ``j`` the student base.

Orientation is always ``(teacher_base, student_base, delta)``. That matches
roadmap ``Delta_ij`` and is the reverse of
``subliminal_learning.campaign.pairs.CornerPair`` (``student``, ``teacher``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdditiveFit:
    """Least-squares fit of ``Delta_ij = mu + tau_i + sigma_j``.

    Attributes:
        mu: Grand mean.
        emissivity: Teacher base -> ``tau_i``; sums to zero.
        receptivity: Student base -> ``sigma_j``; sums to zero.
        r_squared: In-sample fraction of variance explained.
        residuals: ``(teacher_base, student_base)`` -> observed minus fitted.
        n_obs: Cells fitted.
        n_bases: Distinct base labels across both roles.
    """

    mu: float
    emissivity: dict[str, float]
    receptivity: dict[str, float]
    r_squared: float
    residuals: dict[tuple[str, str], float]
    n_obs: int
    n_bases: int


@dataclass(frozen=True)
class HeldOutScore:
    """Leave-one-reciprocal-pair-out performance of the additive model.

    Attributes:
        r_squared: Out-of-sample ``1 - SS_res / SS_tot``, with ``SS_tot``
            about the mean of the held-out observations.
        pearson: Pearson correlation of observed vs predicted over all folds.
        sign_agreement: Fraction of folds whose predicted reciprocal
            asymmetry matches the observed asymmetry in sign.
        mean_absolute_error: Mean ``|observed - predicted|``.
        n_folds: Unordered base pairs held out.
        predictions: ``(teacher_base, student_base, observed, predicted)``
            per held-out cell, fold order then teacher/student order.
    """

    r_squared: float
    pearson: float
    sign_agreement: float
    mean_absolute_error: float
    n_folds: int
    predictions: tuple[tuple[str, str, float, float], ...]


@dataclass(frozen=True)
class ReciprocalContrast:
    """One unordered base pair's directed asymmetry and its additive part.

    Attributes:
        base_a: Lexicographically smaller base; teacher in the forward cell.
        base_b: Lexicographically larger base; student in the forward cell.
        delta_forward: ``Delta`` with ``base_a`` teaching ``base_b``.
        delta_reverse: ``Delta`` with ``base_b`` teaching ``base_a``.
        asymmetry: ``delta_forward - delta_reverse``.
        additive_asymmetry: ``(tau_a + sigma_b) - (tau_b + sigma_a)``.
        interaction: ``asymmetry - additive_asymmetry``.
    """

    base_a: str
    base_b: str
    delta_forward: float
    delta_reverse: float
    asymmetry: float
    additive_asymmetry: float
    interaction: float


def _collect_cells(
    cells: Sequence[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for teacher, student, delta in cells:
        key = (teacher, student)
        if key in seen:
            raise ValueError(f"duplicate cell {teacher!r} -> {student!r}")
        seen.add(key)
        value = float(delta)
        if np.isfinite(value):
            rows.append((teacher, student, value))
    return rows


def _check_roles(rows: Sequence[tuple[str, str, float]]) -> tuple[str, ...]:
    teachers = {teacher for teacher, _, _ in rows}
    students = {student for _, student, _ in rows}
    bases = teachers | students
    if len(bases) < 2:
        raise ValueError(f"need at least 2 distinct bases; got {len(bases)}")
    only_teacher = sorted(teachers - students)
    only_student = sorted(students - teachers)
    if only_teacher or only_student:
        raise ValueError(
            "bases must appear in both roles; "
            f"teacher-only={only_teacher} student-only={only_student}"
        )
    return tuple(sorted(bases))


def _sum_to_zero_design(
    teachers: Sequence[str],
    students: Sequence[str],
    bases: Sequence[str],
) -> np.ndarray:
    """``[n, 1 + 2*(k-1)]``: intercept, free ``tau``, free ``sigma``."""
    index = {base: i for i, base in enumerate(bases)}
    n_base = len(bases)
    n_effect = n_base - 1
    design = np.zeros((len(teachers), 1 + 2 * n_effect), dtype=np.float64)
    design[:, 0] = 1.0
    last = n_base - 1
    for row, (teacher, student) in enumerate(zip(teachers, students, strict=True)):
        t_idx = index[teacher]
        s_idx = index[student]
        if t_idx < last:
            design[row, 1 + t_idx] = 1.0
        else:
            design[row, 1 : 1 + n_effect] = -1.0
        if s_idx < last:
            design[row, 1 + n_effect + s_idx] = 1.0
        else:
            design[row, 1 + n_effect : 1 + 2 * n_effect] = -1.0
    return design


def _effects_from_beta(
    beta: np.ndarray,
    bases: Sequence[str],
) -> tuple[float, dict[str, float], dict[str, float]]:
    n_effect = len(bases) - 1
    tau = np.zeros(len(bases), dtype=np.float64)
    sigma = np.zeros(len(bases), dtype=np.float64)
    tau[:n_effect] = np.asarray(beta[1 : 1 + n_effect], dtype=np.float64)
    sigma[:n_effect] = np.asarray(beta[1 + n_effect :], dtype=np.float64)
    tau[n_effect] = -float(tau[:n_effect].sum())
    sigma[n_effect] = -float(sigma[:n_effect].sum())
    return (
        float(beta[0]),
        {base: float(value) for base, value in zip(bases, tau, strict=True)},
        {base: float(value) for base, value in zip(bases, sigma, strict=True)},
    )


def _sum_of_squares_about_mean(values: np.ndarray) -> float:
    centred = values - values.mean()
    total = float(centred @ centred)
    if total <= np.finfo(np.float64).eps * float(values @ values):
        raise ValueError("values are constant to float precision; nothing to decompose")
    return total


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = observed - predicted
    ss_res = float(residual @ residual)
    return 1.0 - ss_res / _sum_of_squares_about_mean(observed)


def _reciprocal_pairs(
    lookup: dict[tuple[str, str], float],
) -> tuple[tuple[str, str], ...]:
    pairs: set[tuple[str, str]] = set()
    for teacher, student in lookup:
        if teacher == student:
            continue
        base_a, base_b = (teacher, student) if teacher < student else (student, teacher)
        if (base_a, base_b) in lookup and (base_b, base_a) in lookup:
            pairs.add((base_a, base_b))
    return tuple(sorted(pairs))


def fit_additive(cells: Sequence[tuple[str, str, float]]) -> AdditiveFit:
    """Least-squares ``Delta_ij = mu + tau_i + sigma_j`` (sum-to-zero).

    Drops non-finite ``delta`` rows. Requires every base in both roles.

    Args:
        cells: ``(teacher_base, student_base, delta)`` rows.

    Returns:
        The `AdditiveFit`.

    Raises:
        ValueError: Empty after the non-finite drop; a duplicate cell;
            fewer than two bases; a one-role base; constant ``delta``.
    """
    rows = _collect_cells(cells)
    if not rows:
        raise ValueError("no finite delta cells to fit")
    bases = _check_roles(rows)
    teachers = [teacher for teacher, _, _ in rows]
    students = [student for _, student, _ in rows]
    observed = np.asarray([delta for _, _, delta in rows], dtype=np.float64)
    design = _sum_to_zero_design(teachers, students, bases)
    beta, *_ = np.linalg.lstsq(design, observed, rcond=None)
    mu, emissivity, receptivity = _effects_from_beta(beta, bases)
    fitted = np.asarray(
        [mu + emissivity[t] + receptivity[s] for t, s, _ in rows],
        dtype=np.float64,
    )
    residuals = {
        (teacher, student): float(obs - pred)
        for (teacher, student, obs), pred in zip(rows, fitted, strict=True)
    }
    return AdditiveFit(
        mu=mu,
        emissivity=emissivity,
        receptivity=receptivity,
        r_squared=_r_squared(observed, fitted),
        residuals=residuals,
        n_obs=len(rows),
        n_bases=len(bases),
    )


def predict_additive(fit: AdditiveFit, teacher_base: str, student_base: str) -> float:
    """Return ``mu + tau_teacher + sigma_student``.

    Args:
        fit: An `AdditiveFit`.
        teacher_base: Teacher effect key.
        student_base: Student effect key.

    Returns:
        The additive prediction.

    Raises:
        ValueError: If either base is missing from `fit`.
    """
    if teacher_base not in fit.emissivity:
        raise ValueError(f"unknown teacher base {teacher_base!r}")
    if student_base not in fit.receptivity:
        raise ValueError(f"unknown student base {student_base!r}")
    return fit.mu + fit.emissivity[teacher_base] + fit.receptivity[student_base]


def leave_one_reciprocal_pair_out(
    cells: Sequence[tuple[str, str, float]],
) -> HeldOutScore:
    """Cross-validate the additive model by holding out each reciprocal pair.

    A fold is an unordered pair ``{A, B}`` with both directed cells present.
    Each scored fold drops those two cells, refits, and predicts both. A fold
    whose remainder is unfittable is skipped; ``n_folds`` counts scored folds
    only. Folds sort by ``(min(A, B), max(A, B))``; within a fold the cell
    with the lexicographically smaller teacher is emitted first.

    ``sign_agreement`` compares ``sign(obs_forward - obs_reverse)`` with
    ``sign(pred_forward - pred_reverse)`` and drops folds whose observed
    asymmetry is exactly 0. It is ``nan`` when no fold remains.

    Args:
        cells: ``(teacher_base, student_base, delta)`` rows.

    Returns:
        The `HeldOutScore`.

    Raises:
        ValueError: A duplicate cell, or fewer than two scored folds, or
            held-out ``SS_tot`` of zero to float precision.
    """
    rows = _collect_cells(cells)
    lookup = {(teacher, student): delta for teacher, student, delta in rows}
    predictions: list[tuple[str, str, float, float]] = []
    sign_hits: list[bool] = []
    n_folds = 0
    for base_a, base_b in _reciprocal_pairs(lookup):
        remainder = [
            row for row in rows if row[0:2] not in {(base_a, base_b), (base_b, base_a)}
        ]
        try:
            fold_fit = fit_additive(remainder)
        except ValueError:
            continue
        obs_forward = lookup[(base_a, base_b)]
        obs_reverse = lookup[(base_b, base_a)]
        pred_forward = predict_additive(fold_fit, base_a, base_b)
        pred_reverse = predict_additive(fold_fit, base_b, base_a)
        predictions.append((base_a, base_b, obs_forward, pred_forward))
        predictions.append((base_b, base_a, obs_reverse, pred_reverse))
        n_folds += 1
        observed_asym = obs_forward - obs_reverse
        if observed_asym != 0.0:
            predicted_asym = pred_forward - pred_reverse
            sign_hits.append(np.sign(observed_asym) == np.sign(predicted_asym))
    if n_folds < 2:
        raise ValueError(f"need at least 2 scored reciprocal-pair folds; got {n_folds}")
    observed = np.asarray([row[2] for row in predictions], dtype=np.float64)
    predicted = np.asarray([row[3] for row in predictions], dtype=np.float64)
    residual = observed - predicted
    if sign_hits:
        sign_agreement = float(np.mean(sign_hits))
    else:
        sign_agreement = float("nan")
    return HeldOutScore(
        r_squared=_r_squared(observed, predicted),
        pearson=float(np.corrcoef(observed, predicted)[0, 1]),
        sign_agreement=sign_agreement,
        mean_absolute_error=float(np.mean(np.abs(residual))),
        n_folds=n_folds,
        predictions=tuple(predictions),
    )


def reciprocal_asymmetries(
    cells: Sequence[tuple[str, str, float]],
    fit: AdditiveFit | None = None,
) -> tuple[ReciprocalContrast, ...]:
    """Directed contrasts for unordered pairs that have both cells.

    On a complete grid an in-sample additive fit yields near-zero
    ``interaction`` whenever the table is itself additive, so that residual
    is weak evidence; ``leave_one_reciprocal_pair_out`` is the honest test.

    Args:
        cells: ``(teacher_base, student_base, delta)`` rows.
        fit: Effects used for ``additive_asymmetry``; fitted on `cells`
            when omitted.

    Returns:
        `ReciprocalContrast` rows sorted by ``(base_a, base_b)``.
    """
    rows = _collect_cells(cells)
    lookup = {(teacher, student): delta for teacher, student, delta in rows}
    resolved = fit_additive(rows) if fit is None else fit
    contrasts: list[ReciprocalContrast] = []
    for base_a, base_b in _reciprocal_pairs(lookup):
        delta_forward = lookup[(base_a, base_b)]
        delta_reverse = lookup[(base_b, base_a)]
        additive_asymmetry = predict_additive(
            resolved, base_a, base_b
        ) - predict_additive(resolved, base_b, base_a)
        asymmetry = delta_forward - delta_reverse
        contrasts.append(
            ReciprocalContrast(
                base_a=base_a,
                base_b=base_b,
                delta_forward=delta_forward,
                delta_reverse=delta_reverse,
                asymmetry=asymmetry,
                additive_asymmetry=additive_asymmetry,
                interaction=asymmetry - additive_asymmetry,
            )
        )
    return tuple(contrasts)


def role_effects_frame(fit: AdditiveFit) -> pd.DataFrame:
    """One row per base with ``emissivity``, ``receptivity``, and their rank.

    Columns: ``base``, ``emissivity``, ``receptivity``,
    ``emissivity_rank``, ``receptivity_rank``. Rank 1 is the largest
    (``method="min"``). Rows are sorted by emissivity descending.

    Args:
        fit: An `AdditiveFit`.

    Returns:
        The ranked effects table.
    """
    bases = list(fit.emissivity)
    frame = pd.DataFrame(
        {
            "base": bases,
            "emissivity": [fit.emissivity[base] for base in bases],
            "receptivity": [fit.receptivity[base] for base in bases],
        }
    )
    frame["emissivity_rank"] = frame["emissivity"].rank(ascending=False, method="min")
    frame["receptivity_rank"] = frame["receptivity"].rank(ascending=False, method="min")
    return frame.sort_values("emissivity", ascending=False, ignore_index=True)[
        [
            "base",
            "emissivity",
            "receptivity",
            "emissivity_rank",
            "receptivity_rank",
        ]
    ]
