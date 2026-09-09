"""Planted-frame tests for the off-policy cross-size metrics and the O0-O5 verdicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import pytest

from subliminal_learning.campaign import offxsize_metrics as om
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan

RUNG = "ep10_lr1e-4_r16a32"
RUNG20 = "ep20_lr1e-4_r16a32"
REC = plan.STUDENT_CENTRE.key
ALT = next(r.key for r in plan.star_screen_recipes("160m") if r.key != REC)
PICKS = {size: REC for size in plan.SIZES}
RUNGS = {size: RUNG for size in plan.SIZES}
N_BOOT = 64
SEED = 20260909

# A separable 3x3: Delta_ij = MU + TAU[i] + TAU[j]. Emissivity equals receptivity, so
# every reciprocal asymmetry is exactly zero.
MU = 0.30
TAU = {"70m": -0.02, "160m": 0.0, "410m": 0.02}


def _leaf(**over: object) -> dict:
    """One `LEAF_COLUMNS` record; defaults are a live 160m->160m headline treatment."""
    row: dict = dict.fromkeys(om.LEAF_COLUMNS)
    row.update(
        {
            "animal": plan.ANIMAL,
            "student": "160m",
            "teacher": "160m",
            "kind": om.KIND_TREATMENT,
            "phase": pairs.PHASE_HEADLINE,
            "recipe_key": REC,
            "epochs": 10,
            "lr": 1.0e-4,
            "lora_r": 8,
            "lora_alpha": 32,
            "rung_tag": RUNG,
            "ftseed": 1,
            "gseed": 1,
            "p_target": 0.5,
            "storytelling_p_target": 0.4,
            "perplexity_ratio": 1.1,
            "dataset_tag": "ex10-20_prefix_n100k",
            "quarantined": False,
        }
    )
    row.update(over)
    return row


def _control(kind: str, **over: object) -> dict:
    """A control leaf: no teacher rung, no seed, no phase."""
    return _leaf(kind=kind, rung_tag=None, ftseed=None, phase=None, **over)


def _frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(om.LEAF_COLUMNS))


def _cell(
    teacher: str,
    student: str,
    mean: float,
    *,
    ci: tuple[float, float],
    recipe_key: str = REC,
    rung_tag: str = RUNG,
) -> dict:
    """One `CELL_COLUMNS` record whose three Delta means all equal `mean`."""
    return {
        "animal": plan.ANIMAL,
        "teacher": teacher,
        "student": student,
        "recipe_key": recipe_key,
        "rung_tag": rung_tag,
        "arm": "anchor" if teacher == student else "cross",
        "shape_steps": abs(plan.SIZES.index(teacher) - plan.SIZES.index(student)),
        "direction": "same" if teacher == student else "up",
        "n": 3,
        "mean_delta_primary": mean,
        "mean_delta_sb": mean,
        "mean_delta_tb": mean,
        "mean_delta_rng": mean,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "mean_p_target": 0.5,
        "mean_storytelling_p_target": 0.4,
        "mean_ppl_ratio": 1.1,
    }


def _cells(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(om.CELL_COLUMNS))


def _grid(extra: Mapping[tuple[str, str], float] | None = None) -> pd.DataFrame:
    """Every 3x3 cell at two student seeds, over zeroed base controls.

    Each control's ``p_target`` is 0.0, so a treatment's Delta is its own
    ``p_target`` and the planted matrix is read back exactly.
    """
    bumps = extra or {}
    rows: list[dict] = []
    for teacher in plan.SIZES:
        for student in plan.SIZES:
            delta = (
                MU + TAU[teacher] + TAU[student] + bumps.get((teacher, student), 0.0)
            )
            for gseed in (1, 2):
                rows.append(
                    _leaf(teacher=teacher, student=student, gseed=gseed, p_target=delta)
                )
                rows.append(
                    _control(
                        om.KIND_BASE,
                        teacher=teacher,
                        student=student,
                        gseed=gseed,
                        p_target=0.0,
                    )
                )
    return _frame(rows)


def _grid_metrics(
    extra: Mapping[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deltas = om.leaf_deltas(_grid(extra))
    return deltas, om.cell_means(deltas, n_boot=N_BOOT, seed=SEED)


# ---- leaf_deltas --------------------------------------------------------------


def test_leaf_deltas_joins_each_control_on_its_own_key() -> None:
    frame = _frame(
        [
            _leaf(p_target=0.50),
            _leaf(teacher="70m", p_target=0.40),
            _control(om.KIND_BASE, p_target=0.10),
            _control(om.KIND_BASE, teacher="70m", p_target=0.06),
            _control(om.KIND_RNG, p_target=0.02),
        ]
    )
    computed = om.leaf_deltas(frame)
    assert list(computed.columns) == list(om.DELTA_COLUMNS)
    deltas = computed.set_index("teacher")

    anchor = deltas.loc["160m"]
    assert anchor["arm"] == "anchor"
    assert anchor["delta_sb"] == pytest.approx(0.40)
    assert anchor["delta_tb"] == pytest.approx(0.40), "an anchor's tb IS its sb"
    assert anchor["delta_rng"] == pytest.approx(0.48)
    assert anchor["delta_primary"] == pytest.approx(anchor["delta_sb"])

    cross = deltas.loc["70m"]
    assert cross["arm"] == "cross"
    assert cross["delta_sb"] == pytest.approx(0.30), "vs the student's own base"
    assert cross["delta_tb"] == pytest.approx(0.34), "vs the teacher-size base"
    assert cross["delta_rng"] == pytest.approx(0.38)
    assert cross["delta_primary"] == pytest.approx(cross["delta_tb"])
    assert cross["tb_control_mean"] == pytest.approx(0.06)


def test_leaf_deltas_averages_the_control_over_its_matched_replicates() -> None:
    frame = _frame(
        [
            _leaf(gseed=1, p_target=0.50),
            _control(om.KIND_BASE, gseed=1, p_target=0.10),
            _control(om.KIND_BASE, gseed=2, p_target=0.90),
        ]
    )
    row = om.leaf_deltas(frame).iloc[0]
    assert row["sb_control_mean"] == pytest.approx(0.10), "gseed 2 is a different key"
    assert row["delta_sb"] == pytest.approx(0.40)


def test_leaf_deltas_missing_student_base_control_raises_keyerror() -> None:
    frame = _frame(
        [
            _leaf(p_target=0.50),
            # A teacher-base control at another size must never stand in for it.
            _control(om.KIND_BASE, teacher="70m", p_target=0.06),
        ]
    )
    with pytest.raises(KeyError) as excinfo:
        om.leaf_deltas(frame)
    assert excinfo.value.args[0] == (plan.ANIMAL, "160m", REC, 1)


def test_leaf_deltas_missing_teacher_base_control_raises_keyerror() -> None:
    frame = _frame(
        [
            _leaf(teacher="70m", p_target=0.40),
            # The student-base control exists; falling back to it would silently
            # report delta_sb as if it were the cross-pair measurement.
            _control(om.KIND_BASE, p_target=0.10),
        ]
    )
    with pytest.raises(KeyError) as excinfo:
        om.leaf_deltas(frame)
    assert excinfo.value.args[0] == (plan.ANIMAL, "160m", "70m", REC, 1)


def test_leaf_deltas_empty_frame_returns_the_delta_columns() -> None:
    empty = om.leaf_deltas(_frame([]))
    assert empty.empty
    assert list(empty.columns) == list(om.DELTA_COLUMNS)


# ---- quarantine ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (plan.QUARANTINE_PPL_RATIO_MAX, False),
        (plan.QUARANTINE_PPL_RATIO_MAX + 0.01, True),
    ],
)
def test_quarantine_flags_only_leaves_above_the_ppl_ceiling(
    ratio: float, expected: bool
) -> None:
    flagged = om.quarantine(_frame([_leaf(perplexity_ratio=ratio)]))
    assert bool(flagged.loc[0, "quarantined"]) is expected
    assert flagged.loc[0, "quarantine_reason"] == ("ppl_ratio" if expected else "")


def test_quarantined_control_is_dropped_from_the_control_mean() -> None:
    frame = om.quarantine(
        _frame(
            [
                _leaf(gseed=1, p_target=0.50),
                _control(om.KIND_BASE, gseed=1, p_target=0.10),
                _control(om.KIND_BASE, gseed=1, p_target=0.90, perplexity_ratio=9.0),
            ]
        )
    )
    row = om.leaf_deltas(frame).iloc[0]
    assert row["sb_control_mean"] == pytest.approx(0.10)
    assert row["delta_sb"] == pytest.approx(0.40)


def test_quarantined_treatment_is_kept_in_deltas_but_dropped_from_the_cell() -> None:
    frame = om.quarantine(
        _frame(
            [
                _leaf(gseed=1, p_target=0.50, perplexity_ratio=9.0),
                _leaf(gseed=2, p_target=0.30),
                _control(om.KIND_BASE, gseed=1, p_target=0.0),
                _control(om.KIND_BASE, gseed=2, p_target=0.0),
            ]
        )
    )
    deltas = om.leaf_deltas(frame)
    assert len(deltas) == 2, "a flagged leaf keeps its row, with its flag"
    cells = om.cell_means(deltas, n_boot=N_BOOT, seed=SEED)
    assert len(cells) == 1
    assert int(cells.loc[0, "n"]) == 1
    assert cells.loc[0, "mean_delta_primary"] == pytest.approx(0.30)


def test_propagate_control_quarantine_orphans_a_treatment_over_a_dead_control() -> None:
    frame = om.propagate_control_quarantine(
        om.quarantine(
            _frame(
                [
                    _leaf(gseed=1, p_target=0.50),
                    _control(
                        om.KIND_BASE, gseed=1, p_target=0.10, perplexity_ratio=9.0
                    ),
                ]
            )
        )
    )
    treatment = frame[frame["kind"] == om.KIND_TREATMENT].iloc[0]
    assert bool(treatment["quarantined"]) is True
    assert treatment["quarantine_reason"] == "control_quarantined"
    # Contract: dropped from the Delta frame, never raised on -- the raise is
    # reserved for a control that never ran.
    assert om.leaf_deltas(frame).empty


def test_propagate_control_quarantine_spares_a_treatment_with_one_live_control() -> (
    None
):
    frame = om.propagate_control_quarantine(
        om.quarantine(
            _frame(
                [
                    _leaf(gseed=1, p_target=0.50),
                    _control(om.KIND_BASE, gseed=1, p_target=0.10),
                    _control(
                        om.KIND_BASE, gseed=1, p_target=0.90, perplexity_ratio=9.0
                    ),
                ]
            )
        )
    )
    treatment = frame[frame["kind"] == om.KIND_TREATMENT].iloc[0]
    assert bool(treatment["quarantined"]) is False


# ---- cell_means ---------------------------------------------------------------


def test_cell_means_reports_the_planted_mean_inside_a_bootstrap_ci() -> None:
    planted = {(1, 1): 0.20, (1, 2): 0.30, (2, 1): 0.40, (2, 2): 0.50}
    rows = [
        _leaf(ftseed=ftseed, gseed=gseed, p_target=value)
        for (ftseed, gseed), value in planted.items()
    ]
    rows += [_control(om.KIND_BASE, gseed=gseed, p_target=0.0) for gseed in (1, 2)]
    cells = om.cell_means(om.leaf_deltas(_frame(rows)), n_boot=N_BOOT, seed=SEED)
    assert len(cells) == 1
    row = cells.iloc[0]
    assert list(cells.columns) == list(om.CELL_COLUMNS)
    assert int(row["n"]) == 4
    assert row["mean_delta_primary"] == pytest.approx(0.35)
    assert row["mean_delta_sb"] == pytest.approx(0.35)
    assert row["ci_low"] < row["mean_delta_primary"] < row["ci_high"]
    assert 0.20 <= row["ci_low"] and row["ci_high"] <= 0.50


def test_cell_means_single_leaf_cell_degrades_to_a_point_interval() -> None:
    rows = [_leaf(p_target=0.30), _control(om.KIND_BASE, p_target=0.10)]
    row = om.cell_means(om.leaf_deltas(_frame(rows)), n_boot=N_BOOT, seed=SEED).iloc[0]
    assert int(row["n"]) == 1
    assert row["mean_delta_primary"] == pytest.approx(0.20)
    assert row["ci_low"] == pytest.approx(0.20)
    assert row["ci_high"] == pytest.approx(0.20)


def test_cell_means_keeps_the_teacher_rung_in_the_cell_key() -> None:
    rows = [
        _leaf(rung_tag=RUNG, p_target=0.30),
        _leaf(rung_tag=RUNG20, phase=pairs.PHASE_TEACHER_DOSE, p_target=0.50),
        _control(om.KIND_BASE, p_target=0.10),
    ]
    cells = om.cell_means(om.leaf_deltas(_frame(rows)), n_boot=N_BOOT, seed=SEED)
    assert set(cells["rung_tag"]) == {RUNG, RUNG20}, "a dose row is its own cell"


# ---- shape_penalty ------------------------------------------------------------


def test_shape_penalty_is_the_anchor_minus_cross_gap_at_the_receiving_recipe() -> None:
    deltas, cells = _grid_metrics()
    penalty = om.shape_penalty(
        cells,
        picks=PICKS,
        deltas=deltas,
        rungs=RUNGS,
        teacher_p_target={"70m": 0.5, "160m": 0.6, "410m": 0.7},
        n_boot=N_BOOT,
        seed=SEED,
    )
    assert list(penalty.columns) == list(om.PENALTY_COLUMNS)
    assert len(penalty) == 6, "six ordered cross pairs"
    row = penalty.set_index(["teacher", "student"]).loc[("70m", "160m")]
    # Delta_sb(160m->160m) = 0.30, Delta_tb(70m->160m) = 0.28.
    assert row["delta_anchor_s"] == pytest.approx(MU)
    assert row["delta_tb"] == pytest.approx(0.28)
    assert row["penalty"] == pytest.approx(0.02)
    assert row["rho_s"] == pytest.approx(0.28 / 0.30)
    assert row["rho_t"] == pytest.approx(0.28 / 0.26)
    assert row["efficiency"] == pytest.approx(0.28 / 0.5)
    assert row["shape_steps"] == 1


def test_shape_penalty_holds_only_when_its_ci_lies_above_zero() -> None:
    # +0.20 on 70m->410m makes that cross cell BEAT the 410m anchor.
    deltas, cells = _grid_metrics({("70m", "410m"): 0.20})
    penalty = om.shape_penalty(
        cells, picks=PICKS, deltas=deltas, rungs=RUNGS, n_boot=N_BOOT, seed=SEED
    )
    verdicts = om.claims(cells, picks=PICKS, rungs=RUNGS, penalty=penalty)["O1"]
    costly = verdicts["detail"]["pairs"]["70m->160m"]
    assert costly["holds"] is True and costly["ci"][0] > 0.0

    advantage = verdicts["detail"]["pairs"]["70m->410m"]
    assert advantage["penalty"] < 0.0
    assert advantage["ci"][1] < 0.0
    assert advantage["holds"] is False, "a cross-size advantage is not a penalty"


# ---- direction_asymmetry / additive_fit ---------------------------------------


def test_additive_fit_recovers_a_planted_separable_grid() -> None:
    deltas, cells = _grid_metrics()
    fit = om.additive_fit(
        cells, picks=PICKS, rungs=RUNGS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )
    assert fit["n_obs"] == 9
    assert fit["mu"] == pytest.approx(MU)
    assert fit["tau"] == pytest.approx(TAU)
    assert fit["sigma"] == pytest.approx(TAU)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["loo_r2"] == pytest.approx(1.0)
    assert fit["pooled_iota"] == pytest.approx(0.0, abs=1e-12)
    assert fit["pooled_iota_ci_low"] <= 0.0 <= fit["pooled_iota_ci_high"]
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in fit["iota"].values())

    verdict = om.claims(cells, picks=PICKS, rungs=RUNGS, additive=fit)["O2"]
    assert verdict["detail"]["read"] == "separable"
    assert verdict["holds"] is True


def test_additive_fit_planted_interaction_breaks_the_held_out_fit() -> None:
    deltas, cells = _grid_metrics({("70m", "410m"): 0.20})
    fit = om.additive_fit(
        cells, picks=PICKS, rungs=RUNGS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )
    assert fit["loo_r2"] < plan.X2_ADDITIVE_R2_MIN
    assert fit["iota"]["70m|410m"] == pytest.approx(0.20)
    assert fit["interaction"]["70m|410m"] != pytest.approx(0.0, abs=1e-9)
    assert fit["pooled_iota"] == pytest.approx(0.20 / 3.0)

    verdict = om.claims(cells, picks=PICKS, rungs=RUNGS, additive=fit)["O2"]
    assert verdict["detail"]["read"] == "directional"


def test_direction_asymmetry_signs_small_to_large_minus_large_to_small() -> None:
    deltas, cells = _grid_metrics({("70m", "410m"): 0.20})
    asymmetry = om.direction_asymmetry(
        cells,
        picks=PICKS,
        deltas=deltas,
        rungs=RUNGS,
        teacher_p_target={"70m": 0.5, "410m": 1.0, "160m": 0.6},
        n_boot=N_BOOT,
        seed=SEED,
    )
    assert list(asymmetry.columns) == list(om.ASYMMETRY_COLUMNS)
    row = asymmetry.set_index(["size_lo", "size_hi"]).loc[("70m", "410m")]
    assert row["delta_up"] == pytest.approx(0.50)
    assert row["delta_down"] == pytest.approx(0.30)
    assert row["iota"] == pytest.approx(0.20)
    assert row["iota_dose"] == pytest.approx(0.50 / 0.5 - 0.30 / 1.0)
    assert row["ci_low"] <= row["iota"] <= row["ci_high"]

    # The two artifacts must not report opposite-signed iota for one pair: the
    # decomposition orients pairs lexicographically ("70m" sorts last).
    fit = om.additive_fit(
        cells, picks=PICKS, rungs=RUNGS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )
    assert fit["iota"]["70m|410m"] == pytest.approx(row["iota"])
    assert fit["pooled_iota_ci_low"] > 0.0, "the CI must bracket pooled_iota's sign"


# ---- recipe_stability / teacher_dose_curve ------------------------------------


def test_recipe_stability_is_sign_stable_and_rank_concordant_when_planted_so() -> None:
    planted = {
        (REC, "160m", "160m"): 0.30,
        (REC, "70m", "160m"): 0.20,
        (ALT, "160m", "160m"): 0.24,
        (ALT, "70m", "160m"): 0.16,
    }
    rows: list[dict] = []
    for (recipe_key, teacher, student), value in planted.items():
        rows.append(
            _leaf(
                recipe_key=recipe_key,
                teacher=teacher,
                student=student,
                p_target=value,
                phase=(
                    pairs.PHASE_HEADLINE
                    if recipe_key == REC
                    else (
                        pairs.PHASE_SCREEN
                        if teacher == student
                        else pairs.PHASE_ROBUSTNESS
                    )
                ),
            )
        )
        rows.append(
            _control(
                om.KIND_BASE,
                recipe_key=recipe_key,
                teacher=teacher,
                student=student,
                p_target=0.0,
            )
        )
    # A P3c row varies the teacher dose, not the recipe: it must not be counted.
    rows.append(_leaf(rung_tag=RUNG20, phase=pairs.PHASE_TEACHER_DOSE, p_target=0.40))

    recipe = om.recipe_stability(om.leaf_deltas(_frame(rows)), picks={"160m": REC})
    assert list(recipe.columns) == list(om.RECIPE_COLUMNS)
    assert len(recipe) == 4, "p3c is excluded from the recipe frame"
    assert set(recipe["sign"]) == {1}
    assert recipe["kendall_tau"].tolist() == pytest.approx([1.0] * len(recipe))
    assert recipe.set_index(["recipe_key", "teacher"]).loc[
        (ALT, "70m"), "delta_tb"
    ] == (pytest.approx(0.16))

    verdict = om.claims(_cells([]), picks={"160m": REC}, recipe=recipe)["O3"]
    assert verdict["holds"] is None, "O3 is descriptive"
    assert verdict["void"] is False
    assert verdict["detail"]["sign_stable"] == {"160m->160m": True, "70m->160m": True}
    assert verdict["detail"]["n_recipes"] == 2


def test_teacher_dose_curve_returns_the_rungs_in_dose_order() -> None:
    anchors = [r for r in plan.TEACHER_RUNGS if r.family == "anchor"]
    rows: list[dict] = []
    for index, rung in enumerate(anchors):
        phase = pairs.PHASE_HEADLINE if rung.key == RUNG else pairs.PHASE_TEACHER_DOSE
        for gseed in (1, 2):
            rows.append(
                _leaf(
                    rung_tag=rung.key,
                    phase=phase,
                    gseed=gseed,
                    p_target=0.10 * (index + 1),
                )
            )
    rows += [_control(om.KIND_BASE, gseed=gseed, p_target=0.0) for gseed in (1, 2)]
    covariates = {
        rung.key: (0.90 + 0.01 * i, 0.10 * i, 0.5) for i, rung in enumerate(anchors)
    }

    curve = om.teacher_dose_curve(
        om.leaf_deltas(_frame(rows)),
        rung_covariates=covariates,
        n_boot=N_BOOT,
        seed=SEED,
    )
    assert list(curve.columns) == list(om.DOSE_COLUMNS)
    assert curve["epochs"].tolist() == sorted(r.epochs for r in anchors)
    assert curve["rung_tag"].tolist() == [r.key for r in anchors]
    assert curve["mean_delta_sb"].tolist() == pytest.approx(
        [0.10 * (i + 1) for i in range(len(anchors))]
    )
    assert set(curve["family"]) == {"anchor"}
    assert curve["n"].tolist() == [2] * len(anchors)
    assert curve["pool_keep_rate"].tolist() == pytest.approx([0.5] * len(anchors))


# ---- ppl_coupling -------------------------------------------------------------


def _ppl_leaves(teacher: str, *, slope: float, intercept: float) -> list[dict]:
    """Leaves whose Delta is ``intercept + slope * (ppl - 1)`` at three ppl points."""
    rows: list[dict] = []
    for gseed, ppl in enumerate((1.0, 1.2, 1.4), start=1):
        rows.append(
            _leaf(
                teacher=teacher,
                student=teacher,
                gseed=gseed,
                perplexity_ratio=ppl,
                p_target=intercept + slope * (ppl - 1.0),
            )
        )
        rows.append(
            _control(
                om.KIND_BASE,
                teacher=teacher,
                student=teacher,
                gseed=gseed,
                p_target=0.0,
                perplexity_ratio=1.0,
            )
        )
    return rows


def test_ppl_coupling_retains_nothing_from_a_pure_capability_effect() -> None:
    deltas = om.leaf_deltas(_frame(_ppl_leaves("160m", slope=0.5, intercept=0.0)))
    row = om.ppl_coupling(deltas).iloc[0]
    assert list(om.ppl_coupling(deltas).columns) == list(om.PPL_COLUMNS)
    assert row["beta"] == pytest.approx(0.5)
    assert row["mean_delta_adj"] == pytest.approx(0.0, abs=1e-12)
    assert row["retained_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert int(row["n"]) == 3


def test_ppl_coupling_retains_everything_from_a_capability_independent_effect() -> None:
    deltas = om.leaf_deltas(_frame(_ppl_leaves("70m", slope=0.0, intercept=0.30)))
    row = om.ppl_coupling(deltas).iloc[0]
    assert row["beta"] == pytest.approx(0.0, abs=1e-12)
    assert row["mean_delta"] == pytest.approx(0.30)
    assert row["mean_delta_adj"] == pytest.approx(0.30)
    assert row["retained_fraction"] == pytest.approx(1.0)


# ---- concordance --------------------------------------------------------------


def test_concordance_reports_the_gap_against_the_angela_reference() -> None:
    reference = plan.ANGELA_REFERENCE
    observed = 0.90
    cells = _cells(
        [
            _cell(
                plan.P3C_SIZE,
                plan.P3C_SIZE,
                0.85,
                ci=(0.80, 0.90),
                recipe_key=str(reference["student_tag"]),
                rung_tag=str(reference["teacher_tag"]),
            )
            | {"mean_p_target": observed}
        ]
    )
    row = om.concordance(cells).iloc[0]
    assert list(om.concordance(cells).columns) == list(om.CONCORDANCE_COLUMNS)
    assert row["cell"] == f"{plan.P3C_SIZE}->{plan.P3C_SIZE}"
    assert row["p_target"] == pytest.approx(observed)
    assert row["abs_gap_seed1"] == pytest.approx(
        observed - float(reference["test_p_target_seed1"])
    )
    assert row["abs_gap_3seed"] == pytest.approx(
        observed - float(reference["optparam_test_p_target_3seed"])
    )
    assert row["rel_gap_3seed"] == pytest.approx(
        row["abs_gap_3seed"] / float(reference["optparam_test_p_target_3seed"])
    )


def test_concordance_on_an_untrained_cell_is_a_row_of_nans_not_a_raise() -> None:
    row = om.concordance(_cells([])).iloc[0]
    assert int(row["n"]) == 0
    assert pd.isna(row["p_target"])
    assert row["ref_p_target_seed1"] == pytest.approx(
        float(plan.ANGELA_REFERENCE["test_p_target_seed1"])
    )


# ---- claims -------------------------------------------------------------------


def test_claims_returns_exactly_the_plan_claim_ids() -> None:
    verdicts = om.claims(_cells([]), picks={})
    assert tuple(verdicts) == plan.CLAIM_IDS
    assert all(set(v) == {"holds", "void", "detail"} for v in verdicts.values())


def test_claims_o0_voids_a_pair_whose_teacher_size_has_no_rung() -> None:
    cells = _cells(
        [
            _cell("160m", "160m", 0.30, ci=(0.20, 0.40)),
            _cell("70m", "160m", 0.20, ci=(0.10, 0.30)),
        ]
    )
    verdict = om.claims(cells, picks={"160m": REC}, rungs={"160m": RUNG})["O0"]
    assert verdict["void"] is True
    assert verdict["holds"] is None
    assert verdict["detail"]["pairs"]["70m->160m"] == {
        "void": True,
        "reason": "no_admissible_teacher",
    }


def test_claims_o0_voids_a_pair_whose_anchor_ci_covers_zero() -> None:
    cells = _cells(
        [
            _cell("160m", "160m", 0.01, ci=(-0.05, 0.07)),
            _cell("70m", "160m", 0.20, ci=(0.10, 0.30)),
        ]
    )
    verdict = om.claims(cells, picks={"160m": REC}, rungs=RUNGS)["O0"]
    assert verdict["void"] is True
    assert verdict["detail"]["pairs"]["70m->160m"]["reason"] == "anchor_ci_covers_0"


@pytest.mark.parametrize(
    ("mean", "expected"),
    [(plan.X0_MIN_DELTA, True), (plan.X0_MIN_DELTA - 1e-4, False)],
)
def test_claims_o0_holds_at_the_min_delta_boundary(mean: float, expected: bool) -> None:
    cells = _cells(
        [
            _cell("160m", "160m", 0.30, ci=(0.20, 0.40)),
            _cell("70m", "160m", mean, ci=(mean / 2.0, mean * 2.0)),
        ]
    )
    verdict = om.claims(cells, picks={"160m": REC}, rungs=RUNGS)["O0"]
    assert verdict["void"] is False
    assert verdict["holds"] is expected
    assert verdict["detail"]["min_delta"] == plan.X0_MIN_DELTA
    assert verdict["detail"]["pairs"]["70m->160m"]["headline"] is True


def test_claims_o0_fails_a_pair_whose_ci_covers_zero() -> None:
    cells = _cells(
        [
            _cell("160m", "160m", 0.30, ci=(0.20, 0.40)),
            _cell("70m", "160m", 0.10, ci=(-0.05, 0.25)),
        ]
    )
    verdict = om.claims(cells, picks={"160m": REC}, rungs=RUNGS)["O0"]
    assert verdict["holds"] is False


@pytest.mark.parametrize(
    ("retained", "expected"),
    [
        ((plan.X4_RETAINED_FRACTION_MIN, plan.X4_RETAINED_FRACTION_MIN), True),
        ((plan.X4_RETAINED_FRACTION_MIN - 1e-4, plan.X4_RETAINED_FRACTION_MIN), False),
    ],
)
def test_claims_o4_holds_at_the_retained_fraction_boundary(
    retained: tuple[float, float], expected: bool
) -> None:
    ppl = pd.DataFrame(
        [
            {
                "teacher": "160m",
                "student": "160m",
                "recipe_key": REC,
                "rung_tag": RUNG,
                "beta": 0.0,
                "mean_delta": 0.3,
                "mean_delta_adj": 0.3 * value,
                "retained_fraction": value,
                "n": 3,
            }
            for value in retained
        ],
        columns=list(om.PPL_COLUMNS),
    )
    verdict = om.claims(_cells([]), picks={}, ppl=ppl)["O4"]
    assert verdict["void"] is False
    assert verdict["holds"] is expected
    assert verdict["detail"]["threshold"] == plan.X4_RETAINED_FRACTION_MIN
    assert verdict["detail"]["n"] == 2


def test_claims_o4_is_void_without_a_ppl_frame() -> None:
    verdict = om.claims(_cells([]), picks={})["O4"]
    assert verdict["void"] is True
    assert verdict["holds"] is None


def test_claims_o3_and_o5_never_adjudicate() -> None:
    cells = _cells(
        [
            _cell(
                plan.P3C_SIZE,
                plan.P3C_SIZE,
                0.85,
                ci=(0.80, 0.90),
                recipe_key=str(plan.ANGELA_REFERENCE["student_tag"]),
                rung_tag=str(plan.ANGELA_REFERENCE["teacher_tag"]),
            )
        ]
    )
    verdicts = om.claims(
        cells,
        picks={},
        dose=om.teacher_dose_curve(om.leaf_deltas(_frame([]))),
        concordance_frame=om.concordance(cells),
    )
    assert verdicts["O3"]["holds"] is None
    assert verdicts["O5"]["holds"] is None
    assert verdicts["O5"]["void"] is False, "the reference cell has been measured"
    assert verdicts["O5"]["detail"]["cell"] == f"{plan.P3C_SIZE}->{plan.P3C_SIZE}"
