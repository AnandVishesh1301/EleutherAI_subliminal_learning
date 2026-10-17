from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.analysis.decomposition import (
    AdditiveFit,
    fit_additive,
    leave_one_reciprocal_pair_out,
    predict_additive,
    reciprocal_asymmetries,
    role_effects_frame,
)

_BASES = tuple(f"b{i}" for i in range(7))
_MU = 0.15
_TAU = {
    "b0": 0.30,
    "b1": 0.20,
    "b2": 0.10,
    "b3": 0.00,
    "b4": -0.10,
    "b5": -0.20,
    "b6": -0.30,
}
_SIGMA = {
    "b0": 0.21,
    "b1": -0.07,
    "b2": 0.14,
    "b3": -0.14,
    "b4": 0.07,
    "b5": -0.21,
    "b6": 0.00,
}
_IOTA = 0.42
_TOUCHED = ("b1", "b4")


def _grid(
    mu: float,
    tau: dict[str, float],
    sigma: dict[str, float],
    extra: dict[tuple[str, str], float] | None = None,
) -> list[tuple[str, str, float]]:
    injected = extra or {}
    return [
        (
            teacher,
            student,
            mu + tau[teacher] + sigma[student] + injected.get((teacher, student), 0.0),
        )
        for teacher in tau
        for student in sigma
    ]


def _additive_cells() -> list[tuple[str, str, float]]:
    return _grid(_MU, _TAU, _SIGMA)


def _circulant_cells() -> list[tuple[str, str, float]]:
    n = len(_BASES)
    cells: list[tuple[str, str, float]] = []
    for i, teacher in enumerate(_BASES):
        for j, student in enumerate(_BASES):
            dist = (j - i) % n
            if dist == 0:
                delta = 0.0
            elif dist <= n // 2:
                delta = 1.0
            else:
                delta = -1.0
            cells.append((teacher, student, delta))
    return cells


def test_fit_additive_recovers_a_known_additive_grid() -> None:
    fit = fit_additive(_additive_cells())
    assert fit.n_obs == 49
    assert fit.n_bases == 7
    assert fit.mu == pytest.approx(_MU, abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    for base in _BASES:
        assert fit.emissivity[base] == pytest.approx(_TAU[base], abs=1e-12)
        assert fit.receptivity[base] == pytest.approx(_SIGMA[base], abs=1e-12)
    assert sum(fit.emissivity.values()) == pytest.approx(0.0, abs=1e-12)
    assert sum(fit.receptivity.values()) == pytest.approx(0.0, abs=1e-12)
    for residual in fit.residuals.values():
        assert residual == pytest.approx(0.0, abs=1e-12)


def test_fit_additive_returns_centred_effects_after_a_shift() -> None:
    shift = 0.4
    tau_shifted = {base: value + shift for base, value in _TAU.items()}
    sigma_shifted = {base: value - shift for base, value in _SIGMA.items()}
    fit = fit_additive(_grid(_MU, tau_shifted, sigma_shifted))
    assert fit.mu == pytest.approx(_MU, abs=1e-12)
    for base in _BASES:
        assert fit.emissivity[base] == pytest.approx(_TAU[base], abs=1e-12)
        assert fit.receptivity[base] == pytest.approx(_SIGMA[base], abs=1e-12)
    for teacher, student, delta in _additive_cells():
        assert predict_additive(fit, teacher, student) == pytest.approx(
            delta, abs=1e-12
        )


def test_reciprocal_asymmetries_injected_iota_is_isolated() -> None:
    clean = _additive_cells()
    fit = fit_additive(clean)
    dirty = _grid(_MU, _TAU, _SIGMA, extra={_TOUCHED: _IOTA})
    contrasts = reciprocal_asymmetries(dirty, fit=fit)
    assert len(contrasts) == 21
    for contrast in contrasts:
        pair = (contrast.base_a, contrast.base_b)
        if pair == _TOUCHED:
            assert contrast.interaction == pytest.approx(_IOTA, abs=1e-12)
        else:
            assert contrast.interaction == pytest.approx(0.0, abs=1e-12)


def test_leave_one_reciprocal_pair_out_recovers_additive_cells() -> None:
    score = leave_one_reciprocal_pair_out(_additive_cells())
    assert score.n_folds == 21
    assert len(score.predictions) == 42
    assert score.predictions[0][:2] == ("b0", "b1")
    assert score.predictions[1][:2] == ("b1", "b0")
    assert score.r_squared == pytest.approx(1.0, abs=1e-12)
    assert score.pearson == pytest.approx(1.0, abs=1e-12)
    assert score.sign_agreement == 1.0
    assert score.mean_absolute_error == pytest.approx(0.0, abs=1e-12)


def test_leave_one_reciprocal_pair_out_interaction_grid_is_below_chance() -> None:
    score = leave_one_reciprocal_pair_out(_circulant_cells())
    assert score.n_folds == 21
    assert score.sign_agreement == 0.0
    assert score.r_squared < 0.0


def test_reciprocal_asymmetries_keeps_teacher_first_orientation() -> None:
    cells = [("A", "B", 0.30), ("B", "A", 0.10)]
    contrasts = reciprocal_asymmetries(cells)
    assert len(contrasts) == 1
    contrast = contrasts[0]
    assert contrast.base_a == "A"
    assert contrast.base_b == "B"
    assert contrast.delta_forward == pytest.approx(0.30)
    assert contrast.delta_reverse == pytest.approx(0.10)
    assert contrast.asymmetry == pytest.approx(0.20)


@pytest.mark.parametrize(
    "cells",
    [
        [],
        [("A", "B", 0.10), ("B", "A", 0.20), ("A", "B", 0.30)],
        [("A", "A", 1.0)],
        [("A", "B", 0.10), ("B", "B", 0.20)],
    ],
    ids=("empty", "duplicate", "single_base", "teacher_only"),
)
def test_fit_additive_rejects_invalid_cells(
    cells: list[tuple[str, str, float]],
) -> None:
    with pytest.raises(ValueError):
        fit_additive(cells)


def test_fit_additive_drops_nonfinite_deltas() -> None:
    cells = [
        ("A", "B", 0.30),
        ("B", "A", 0.10),
        ("A", "A", 0.20),
        ("B", "B", 0.20),
        ("A", "Z", float("nan")),
        ("Z", "B", float("inf")),
        ("B", "Z", float("-inf")),
    ]
    fit = fit_additive(cells)
    assert fit.n_obs == 4
    assert fit.n_bases == 2


def test_reciprocal_asymmetries_omits_pairs_missing_a_direction() -> None:
    drop = {("b0", "b1"), ("b2", "b3"), ("b4", "b5")}
    cells = [cell for cell in _additive_cells() if (cell[0], cell[1]) not in drop]
    fit = fit_additive(cells)
    assert fit.n_obs == 46
    assert fit.n_bases == 7
    contrasts = reciprocal_asymmetries(cells, fit=fit)
    pairs = {(row.base_a, row.base_b) for row in contrasts}
    assert pairs.isdisjoint(drop)
    assert len(contrasts) == 18


def test_role_effects_frame_ranks_and_orders_known_effects() -> None:
    fit = AdditiveFit(
        mu=0.0,
        emissivity={"x": 0.40, "y": 0.10, "z": -0.50},
        receptivity={"x": 0.20, "y": -0.10, "z": -0.10},
        r_squared=1.0,
        residuals={},
        n_obs=9,
        n_bases=3,
    )
    frame = role_effects_frame(fit)
    assert list(frame.columns) == [
        "base",
        "emissivity",
        "receptivity",
        "emissivity_rank",
        "receptivity_rank",
    ]
    assert list(frame["base"]) == ["x", "y", "z"]
    assert frame["emissivity"].tolist() == pytest.approx([0.40, 0.10, -0.50])
    assert frame["receptivity"].tolist() == pytest.approx([0.20, -0.10, -0.10])
    assert frame["emissivity_rank"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert frame["receptivity_rank"].tolist() == pytest.approx([1.0, 2.0, 2.0])
    assert pd.api.types.is_string_dtype(frame["base"])
    assert pd.api.types.is_float_dtype(frame["emissivity"])
    assert pd.api.types.is_float_dtype(frame["receptivity"])
    assert pd.api.types.is_numeric_dtype(frame["emissivity_rank"])
    assert pd.api.types.is_numeric_dtype(frame["receptivity_rank"])


def test_predict_additive_rejects_an_unknown_base() -> None:
    fit = fit_additive(_additive_cells())
    with pytest.raises(ValueError, match="unknown teacher"):
        predict_additive(fit, "missing", "b0")
    with pytest.raises(ValueError, match="unknown student"):
        predict_additive(fit, "b0", "missing")


def test_leave_one_reciprocal_pair_out_rejects_a_single_fold() -> None:
    with pytest.raises(ValueError, match="at least 2 scored"):
        leave_one_reciprocal_pair_out([("A", "B", 0.30), ("B", "A", 0.10)])


def test_fit_product_recovers_rank_one_grid_with_missing_cells():
    from subliminal_learning.analysis.decomposition import fit_product

    a = {"t1": 0.2, "t2": 0.5, "t3": 0.9}
    b = {"s1": 0.1, "s2": 0.6, "s3": 1.0}
    cells = [(t, s, a[t] * b[s]) for t in a for s in b if (t, s) != ("t2", "s3")]
    fit = fit_product(cells)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.n_obs == 8 and fit.n_params == 5
    for t in a:
        for s in b:
            assert fit.teacher[t] * fit.student[s] == pytest.approx(
                a[t] * b[s], abs=1e-5
            )


def test_fit_product_explains_less_of_an_additive_grid_than_additive_fit():
    from subliminal_learning.analysis.decomposition import fit_additive, fit_product

    tau = {"x": -0.3, "y": 0.0, "z": 0.3}
    sigma = {"x": 0.2, "y": -0.1, "z": -0.1}
    cells = [(t, s, 0.5 + tau[t] + sigma[s]) for t in tau for s in sigma]
    assert fit_additive(cells).r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit_product(cells).r_squared < 1.0 - 1e-3


def test_fit_product_drops_nonfinite_and_rejects_constant():
    from subliminal_learning.analysis.decomposition import fit_product

    with pytest.raises(ValueError):
        fit_product([("a", "a", 0.3), ("a", "b", 0.3), ("b", "a", float("nan"))])
