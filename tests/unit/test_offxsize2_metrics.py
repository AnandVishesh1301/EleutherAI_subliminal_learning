"""Planted-frame tests for the v2 cross-size metrics and the H1-H8 verdicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import pytest

from subliminal_learning.campaign import offxsize2_metrics as om
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan

DOG = plan.PRIMARY_ANIMAL
LION = plan.SECONDARY_ANIMAL
RUNG = plan.R1.key
RUNG3 = plan.R3.key
REC = plan.STUDENT_GRID[0].key
ALT = plan.STUDENT_GRID[1].key
PICKS = {size: REC for size in plan.SIZES}
RUNGS = {size: RUNG for size in plan.SIZES}
ADMITTED = {DOG: {size: (RUNG,) for size in plan.SIZES}}
N_BOOT = 64
SEED = 20260913

# A separable 3x3: Delta_ij = MU + TAU[i] + TAU[j]. Emissivity equals receptivity, so
# every reciprocal asymmetry is exactly zero.
MU = 0.30
TAU = {"70m": -0.02, "160m": 0.0, "410m": 0.02}


def _leaf(**over: object) -> dict:
    """One `LEAF_COLUMNS` record; defaults are a live 160m->160m P1 treatment."""
    row: dict = dict.fromkeys(om.LEAF_COLUMNS)
    row.update(
        {
            "animal": DOG,
            "student": "160m",
            "teacher": "160m",
            "kind": om.KIND_TREATMENT,
            "phase": pairs.PHASE_FACTORIAL,
            "recipe_key": REC,
            "epochs": 20,
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
            "baseline_p_target": 0.01,
            "peak_p_target": 0.55,
            "peak_step_frac": 0.25,
            "peak_epoch_equivalent": 5.0,
            "endpoint_p_target": 0.50,
            "peak_over_endpoint": 1.1,
        }
    )
    row.update(over)
    return row


def _control(kind: str, **over: object) -> dict:
    """A control leaf: no teacher rung, no teacher seed, trait-free numbers."""
    return _leaf(kind=kind, rung_tag=None, ftseed=None, **over)


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
    animal: str = DOG,
    peak: float | None = None,
    peak_ci: tuple[float, float] | None = None,
) -> dict:
    """One `CELL_COLUMNS` record; the peak reading mirrors the endpoint by default."""
    peak_mean = mean if peak is None else peak
    peak_bounds = ci if peak_ci is None else peak_ci
    return {
        "animal": animal,
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
        "mean_peak_delta_primary": peak_mean,
        "mean_peak_delta_tb": peak_mean,
        "peak_ci_low": peak_bounds[0],
        "peak_ci_high": peak_bounds[1],
    }


def _cells(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(om.CELL_COLUMNS))


def _grid(
    extra: Mapping[tuple[str, str], float] | None = None, *, animal: str = DOG
) -> pd.DataFrame:
    """Every 3x3 cell at two student seeds, over zeroed base controls.

    Each control reads 0.0 at the endpoint AND at the peak, so a treatment's Delta is
    its own `p_target` and the planted matrix is read back exactly on both readings.
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
                    _leaf(
                        animal=animal,
                        teacher=teacher,
                        student=student,
                        gseed=gseed,
                        p_target=delta,
                        peak_p_target=delta,
                    )
                )
                rows.append(
                    _control(
                        om.KIND_BASE,
                        animal=animal,
                        teacher=teacher,
                        student=student,
                        gseed=gseed,
                        p_target=0.0,
                        peak_p_target=0.0,
                    )
                )
    return _frame(rows)


def _grid_metrics(
    extra: Mapping[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deltas = om.leaf_deltas(_grid(extra))
    return deltas, om.cell_means(deltas, n_boot=N_BOOT, seed=SEED)


# ---- leaf_deltas --------------------------------------------------------------


def test_leaf_deltas_carries_the_trajectory_columns_and_the_primary_control() -> None:
    frame = _frame(
        [
            _leaf(teacher="70m", p_target=0.40, peak_p_target=0.60),
            _control(om.KIND_BASE, teacher="70m", p_target=0.06, peak_p_target=0.10),
            _control(om.KIND_BASE, p_target=0.10, peak_p_target=0.12),
        ]
    )
    row = om.leaf_deltas(frame).iloc[0]
    assert list(om.leaf_deltas(frame).columns) == list(om.DELTA_COLUMNS)
    assert row["arm"] == "cross"
    assert row["delta_tb"] == pytest.approx(0.34), "vs the teacher-size base"
    assert row["delta_sb"] == pytest.approx(0.30), "vs the student's own base"
    assert row["delta_primary"] == pytest.approx(row["delta_tb"])
    assert row["primary_control_mean"] == pytest.approx(0.06)
    assert row["peak_epoch_equivalent"] == pytest.approx(5.0)
    assert row["peak_over_endpoint"] == pytest.approx(1.1)


def test_leaf_deltas_peak_delta_divides_by_the_controls_own_peak() -> None:
    frame = _frame(
        [
            _leaf(teacher="70m", p_target=0.40, peak_p_target=0.60),
            _control(om.KIND_BASE, teacher="70m", p_target=0.06, peak_p_target=0.10),
        ]
    )
    row = om.leaf_deltas(frame).iloc[0]
    assert row["delta_tb"] == pytest.approx(0.34), "the endpoint control"
    assert row["peak_delta_tb"] == pytest.approx(0.50), "0.60 - 0.10, not 0.60 - 0.06"
    assert row["peak_delta_primary"] == pytest.approx(row["peak_delta_tb"])


def test_leaf_deltas_without_a_control_peak_is_nan_rather_than_a_raise() -> None:
    """A probe that has not run is a coverage gap, not the campaign error."""
    frame = _frame(
        [
            _leaf(teacher="70m", p_target=0.40, peak_p_target=0.60),
            _control(
                om.KIND_BASE, teacher="70m", p_target=0.06, peak_p_target=float("nan")
            ),
        ]
    )
    row = om.leaf_deltas(frame).iloc[0]
    assert row["delta_tb"] == pytest.approx(0.34)
    assert pd.isna(row["peak_delta_tb"])


def test_leaf_deltas_missing_teacher_base_control_raises_keyerror() -> None:
    frame = _frame(
        [
            _leaf(teacher="70m", p_target=0.40),
            # The student-base control exists; falling back to it would report
            # delta_sb as if it were the cross-pair measurement.
            _control(om.KIND_BASE, p_target=0.10),
        ]
    )
    with pytest.raises(KeyError) as excinfo:
        om.leaf_deltas(frame)
    assert excinfo.value.args[0] == (DOG, "160m", "70m", REC, 1)


def test_leaf_deltas_rejects_a_frame_without_the_trajectory_columns() -> None:
    """A v1-shaped frame would silently lose the peak half of every claim."""
    frame = _frame([_leaf()]).drop(columns=["peak_p_target"])
    with pytest.raises(ValueError, match="trajectory columns"):
        om.leaf_deltas(frame)


# ---- the animal axis ----------------------------------------------------------


def test_two_animals_never_merge_into_one_cell_row() -> None:
    """`animal` is an aggregation key and a control-join key, not a passthrough.

    A dog cell and a lion cell agree on every other identity field -- same sizes,
    same recipe, same rung, same seeds -- so pooling them would average two species
    into one delta without any error.
    """
    rows: list[dict] = []
    for animal, p_target, control in ((DOG, 0.40, 0.10), (LION, 0.10, 0.02)):
        rows.append(_leaf(animal=animal, teacher="70m", p_target=p_target))
        rows.append(
            _control(om.KIND_BASE, animal=animal, teacher="70m", p_target=control)
        )
    cells = om.cell_means(om.leaf_deltas(_frame(rows)), n_boot=N_BOOT, seed=SEED)
    assert len(cells) == 2
    by_animal = cells.set_index("animal")
    assert by_animal.loc[DOG, "mean_delta_tb"] == pytest.approx(0.30)
    assert by_animal.loc[LION, "mean_delta_tb"] == pytest.approx(0.08)


def test_the_second_animals_control_never_stands_in_for_the_first() -> None:
    frame = _frame(
        [
            _leaf(teacher="70m", p_target=0.40),
            _control(om.KIND_BASE, animal=LION, teacher="70m", p_target=0.06),
        ]
    )
    with pytest.raises(KeyError) as excinfo:
        om.leaf_deltas(frame)
    assert excinfo.value.args[0][0] == DOG


def test_direction_asymmetry_reports_one_row_per_animal_and_pair() -> None:
    deltas = pd.concat(
        [om.leaf_deltas(_grid(animal=a)) for a in (DOG, LION)], ignore_index=True
    )
    cells = om.cell_means(deltas, n_boot=N_BOOT, seed=SEED)
    asymmetry = om.direction_asymmetry(
        cells, picks=PICKS, deltas=deltas, rungs=RUNGS, n_boot=N_BOOT, seed=SEED
    )
    assert list(asymmetry.columns) == list(om.ASYMMETRY_COLUMNS)
    assert len(asymmetry) == 6, "three unordered size pairs x two animals"
    assert set(asymmetry["animal"]) == {DOG, LION}


# ---- H5: additive fit and direction asymmetry ---------------------------------


def test_additive_fit_recovers_a_planted_separable_grid() -> None:
    deltas, cells = _grid_metrics()
    fits = om.additive_fit(
        cells, picks=PICKS, rungs=RUNGS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )
    fit = fits[DOG]
    assert fit["n_obs"] == 9
    assert fit["mu"] == pytest.approx(MU)
    assert fit["tau"] == pytest.approx(TAU)
    assert fit["sigma"] == pytest.approx(TAU)
    assert fit["loo_r2"] == pytest.approx(1.0)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in fit["iota"].values())

    verdict = om.claims(cells, additive=fits, asymmetry=None)["H5"]
    assert verdict["void"] is True, "no iota CI without an asymmetry frame"
    assert verdict["detail"]["additive_fit"][DOG]["separable"] is True


def test_additive_fit_planted_interaction_breaks_the_held_out_fit() -> None:
    deltas, cells = _grid_metrics({("70m", "410m"): 0.20})
    fit = om.additive_fit(
        cells, picks=PICKS, rungs=RUNGS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )[DOG]
    assert fit["loo_r2"] < plan.H5_ADDITIVE_R2_MIN
    assert fit["iota"]["70m|410m"] == pytest.approx(0.20)
    assert fit["interaction"]["70m|410m"] != pytest.approx(0.0, abs=1e-9)


def test_direction_asymmetry_signs_small_to_large_minus_large_to_small() -> None:
    deltas, cells = _grid_metrics({("70m", "410m"): 0.20})
    asymmetry = om.direction_asymmetry(
        cells,
        picks=PICKS,
        deltas=deltas,
        rungs=RUNGS,
        teacher_p_target={"70m": 0.5, "160m": 0.6, "410m": 1.0},
        n_boot=N_BOOT,
        seed=SEED,
    )
    row = asymmetry.set_index(["animal", "size_lo", "size_hi"]).loc[
        (DOG, "70m", "410m")
    ]
    assert row["delta_up"] == pytest.approx(0.50)
    assert row["delta_down"] == pytest.approx(0.30)
    assert row["iota"] == pytest.approx(0.20)
    assert row["ci_low"] <= row["iota"] <= row["ci_high"]

    # `analysis.decomposition` orders an unordered pair lexicographically and "70m"
    # sorts LAST, which reverses two of the three size pairs; v1's re-key and
    # sign-flip must survive being called per animal.
    fit = om.additive_fit(
        cells, picks=PICKS, rungs=RUNGS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )[DOG]
    assert set(fit["iota"]) == {"70m|160m", "70m|410m", "160m|410m"}
    assert fit["iota"]["70m|410m"] == pytest.approx(row["iota"])


@pytest.mark.parametrize(
    ("iota", "expected"),
    [(plan.H5_ASYM_MIN, True), (plan.H5_ASYM_MIN - 1e-4, False)],
)
def test_claims_h5_is_directional_at_the_asymmetry_boundary(
    iota: float, expected: bool
) -> None:
    asymmetry = pd.DataFrame(
        [
            {
                "animal": DOG,
                "size_lo": "70m",
                "size_hi": "160m",
                "iota": iota,
                "iota_dose": iota,
                "ci_low": iota / 2.0,
                "ci_high": iota * 2.0,
                "delta_up": iota,
                "delta_down": 0.0,
            }
        ],
        columns=list(om.ASYMMETRY_COLUMNS),
    )
    verdict = om.claims(_cells([]), asymmetry=asymmetry)["H5"]
    assert verdict["void"] is False
    assert verdict["holds"] is expected
    assert verdict["detail"]["asym_min"] == plan.H5_ASYM_MIN


def test_claims_h5_is_not_directional_when_the_ci_covers_zero() -> None:
    asymmetry = pd.DataFrame(
        [
            {
                "animal": DOG,
                "size_lo": "70m",
                "size_hi": "160m",
                "iota": 0.20,
                "iota_dose": 0.20,
                "ci_low": -0.05,
                "ci_high": 0.45,
                "delta_up": 0.20,
                "delta_down": 0.0,
            }
        ],
        columns=list(om.ASYMMETRY_COLUMNS),
    )
    assert om.claims(_cells([]), asymmetry=asymmetry)["H5"]["holds"] is False


# ---- H1: existence ------------------------------------------------------------


def _h1_cells(
    points: Sequence[tuple[str, str, float, tuple[float, float]]],
    *,
    anchor_ci: tuple[float, float] = (0.20, 0.40),
    teacher: str = "70m",
    student: str = "160m",
    peaks: Mapping[str, tuple[float, tuple[float, float]]] | None = None,
) -> pd.DataFrame:
    """Cross cells at `(rung, recipe, mean, ci)`, each with its matched anchor."""
    peak_at = peaks or {}
    rows: list[dict] = []
    for rung_tag, recipe_key, mean, ci in points:
        peak, peak_ci = peak_at.get(recipe_key, (mean, ci))
        rows.append(
            _cell(
                teacher,
                student,
                mean,
                ci=ci,
                recipe_key=recipe_key,
                rung_tag=rung_tag,
                peak=peak,
                peak_ci=peak_ci,
            )
        )
        rows.append(
            _cell(
                student,
                student,
                0.30,
                ci=anchor_ci,
                recipe_key=recipe_key,
                rung_tag=rung_tag,
            )
        )
    return _cells(rows)


def test_claims_h1_takes_the_max_over_clearing_points_and_names_it() -> None:
    cells = _h1_cells(
        [
            (RUNG, REC, 0.05, (0.02, 0.08)),
            (RUNG3, ALT, 0.12, (0.06, 0.18)),
            (RUNG3, REC, 0.08, (0.03, 0.13)),
            # Higher mean, but its CI covers 0: it is the coverage max, never the max
            # the claim is read off.
            (RUNG, ALT, 0.30, (-0.10, 0.70)),
        ]
    )
    verdict = om.claims(cells)["H1"]
    pair = verdict["detail"]["pairs"][f"{DOG}|70m->160m"]
    assert verdict["holds"] is True
    assert pair["n_points"] == 4
    assert pair["n_clearing"] == 3
    assert pair["best"]["mean"] == pytest.approx(0.12)
    assert pair["best"]["rung_tag"] == RUNG3
    assert pair["best"]["recipe_key"] == ALT
    assert pair["best"]["reading"] == "endpoint"
    assert pair["max_point"]["mean"] == pytest.approx(0.30), "coverage-aware max"
    assert pair["headline"] is True
    assert pair["endpoint_artefact"] is False


def test_claims_h1_is_existential_so_one_clearing_pair_carries_it() -> None:
    cells = pd.concat(
        [
            _h1_cells([(RUNG, REC, 0.12, (0.06, 0.18))], teacher="70m"),
            _h1_cells([(RUNG, REC, 0.001, (-0.02, 0.02))], teacher="410m"),
        ],
        ignore_index=True,
    )
    verdict = om.claims(cells)["H1"]
    assert verdict["holds"] is True
    assert verdict["detail"]["n"] == 2
    assert verdict["detail"]["n_holds"] == 1
    assert verdict["detail"]["existential"] is True
    assert verdict["detail"]["pairs"][f"{DOG}|410m->160m"]["holds"] is False


def test_claims_h1_voids_a_pair_whose_teacher_size_has_no_admitted_rung() -> None:
    cells = _h1_cells([(RUNG, REC, 0.12, (0.06, 0.18))])
    verdict = om.claims(cells, rungs={DOG: {"160m": (RUNG,), "70m": ()}})["H1"]
    assert verdict["void"] is True
    assert verdict["holds"] is None
    assert verdict["detail"]["pairs"][f"{DOG}|70m->160m"] == {
        "void": True,
        "reason": "no_admissible_teacher",
    }


def test_claims_h1_voids_a_pair_whose_matched_anchor_ci_covers_zero() -> None:
    cells = _h1_cells([(RUNG, REC, 0.12, (0.06, 0.18))], anchor_ci=(-0.05, 0.35))
    verdict = om.claims(cells)["H1"]
    assert verdict["void"] is True
    pair = verdict["detail"]["pairs"][f"{DOG}|70m->160m"]
    assert pair["reason"] == "anchor_ci_covers_0"
    assert pair["skipped"]["anchor_ci_covers_0"] == 1


def test_claims_h1_voids_a_point_whose_matched_anchor_was_never_trained() -> None:
    cells = _cells([_cell("70m", "160m", 0.12, ci=(0.06, 0.18))])
    verdict = om.claims(cells)["H1"]
    assert verdict["void"] is True
    assert verdict["detail"]["pairs"][f"{DOG}|70m->160m"]["reason"] == "missing_anchor"


def test_claims_h1_flags_a_peak_only_pair_as_an_endpoint_artefact() -> None:
    """Pre-registered in §8: a peak that clears while the endpoint does not is decay.

    At 410m off-policy transfer peaks near 20% of training and decays 3.7x by the
    ep20 endpoint, so the in-flight reading is not licensed to stand in for it.
    """
    cells = _h1_cells(
        [(RUNG, REC, 0.005, (0.001, 0.010))],
        peaks={REC: (0.30, (0.20, 0.40))},
    )
    verdict = om.claims(cells)["H1"]
    pair = verdict["detail"]["pairs"][f"{DOG}|70m->160m"]
    assert pair["holds"] is False, "an endpoint artefact is never a pass"
    assert pair["endpoint_artefact"] is True
    assert pair["n_clearing"] == 0
    assert pair["n_clearing_peak"] == 1
    assert pair["peak_best"]["mean"] == pytest.approx(0.30)
    assert pair["peak_best"]["reading"] == "peak"
    assert verdict["holds"] is False
    assert verdict["detail"]["endpoint_artefacts"] == [f"{DOG}|70m->160m"]


@pytest.mark.parametrize(
    ("mean", "expected"),
    [(plan.H1_MIN_DELTA, True), (plan.H1_MIN_DELTA - 1e-4, False)],
)
def test_claims_h1_clears_at_the_min_delta_boundary(
    mean: float, expected: bool
) -> None:
    cells = _h1_cells([(RUNG, REC, mean, (mean / 2.0, mean * 2.0))])
    verdict = om.claims(cells)["H1"]
    assert verdict["void"] is False
    assert verdict["holds"] is expected
    assert verdict["detail"]["min_delta"] == plan.H1_MIN_DELTA
    assert verdict["detail"]["prompt_set"] == plan.PRIMARY_PROMPT_SET


def test_claims_h1_keeps_the_two_animals_as_separate_units() -> None:
    dog = _h1_cells([(RUNG, REC, 0.12, (0.06, 0.18))])
    lion = _h1_cells([(RUNG, REC, 0.001, (-0.02, 0.02))])
    lion["animal"] = LION
    verdict = om.claims(pd.concat([dog, lion], ignore_index=True))["H1"]
    assert verdict["detail"]["n"] == 2
    assert verdict["detail"]["pairs"][f"{DOG}|70m->160m"]["holds"] is True
    assert verdict["detail"]["pairs"][f"{LION}|70m->160m"]["holds"] is False


# ---- H4: trajectory -----------------------------------------------------------


def test_trajectory_reports_the_peak_and_never_the_step_zero_baseline() -> None:
    """The untrained prior is carried BESIDE the peak, never in place of it.

    `offpolicy_probe_io.probe_peak_vs_endpoint` excludes step 0 from the peak (its
    own test locks that); this locks that the cell aggregate keeps them apart even
    when the baseline is the largest number on the leaf.
    """
    rows = [
        _leaf(
            teacher="410m",
            gseed=gseed,
            p_target=0.10,
            baseline_p_target=0.90,
            peak_p_target=0.40,
            endpoint_p_target=0.10,
            peak_over_endpoint=4.0,
            peak_step_frac=0.10,
            peak_epoch_equivalent=2.0,
        )
        for gseed in (1, 2)
    ]
    rows += [
        _control(om.KIND_BASE, teacher="410m", gseed=gseed, p_target=0.0)
        for gseed in (1, 2)
    ]
    frame = om.trajectory(om.leaf_deltas(_frame(rows)))
    assert list(frame.columns) == list(om.TRAJECTORY_COLUMNS)
    row = frame.iloc[0]
    assert int(row["n"]) == 2
    assert row["mean_peak_p_target"] == pytest.approx(0.40)
    assert row["mean_baseline_p_target"] == pytest.approx(0.90)
    assert row["mean_peak_p_target"] < row["mean_baseline_p_target"]
    assert row["mean_endpoint_p_target"] == pytest.approx(0.10)
    assert row["mean_peak_over_endpoint"] == pytest.approx(4.0)
    assert row["mean_peak_epoch_equivalent"] == pytest.approx(2.0)

    verdict = om.claims(_cells([]), trajectory_frame=frame)["H4"]
    assert verdict["holds"] is None, "H4 is descriptive"
    assert verdict["void"] is False
    assert verdict["detail"]["by_student_size"]["160m"]["mean_peak_over_endpoint"] == (
        pytest.approx(4.0)
    )


# ---- H6: control integrity ----------------------------------------------------


def _h6_leaves(treat: float, control: float, *, teacher: str = "410m") -> pd.DataFrame:
    return _frame(
        [
            _leaf(teacher=teacher, p_target=treat),
            _control(om.KIND_BASE, teacher=teacher, p_target=control),
        ]
    )


def test_control_integrity_flags_a_planted_register_drift_cell() -> None:
    """Large against the bare prior, beaten by its own matched control."""
    deltas = om.leaf_deltas(_h6_leaves(0.30, 0.35))
    row = om.control_integrity(deltas, priors={(DOG, "160m"): 0.01}).iloc[0]
    assert list(om.control_integrity(deltas, priors={(DOG, "160m"): 0.01}).columns) == (
        list(om.CONTROL_COLUMNS)
    )
    assert row["control_p_target"] == pytest.approx(0.35)
    assert row["treat_over_control"] == pytest.approx(0.30 / 0.35)
    assert row["delta_vs_prior"] == pytest.approx(0.29)
    assert row["prior_ratio"] == pytest.approx(30.0)
    assert row["control_over_prior"] == pytest.approx(35.0)
    assert bool(row["control_intact"]) is True, "the control is alive, just high"
    assert bool(row["register_drift"]) is True


def test_control_integrity_passes_a_cell_that_beats_its_control() -> None:
    deltas = om.leaf_deltas(_h6_leaves(0.30, 0.05))
    row = om.control_integrity(deltas, priors={(DOG, "160m"): 0.01}).iloc[0]
    assert row["treat_over_control"] == pytest.approx(6.0)
    assert bool(row["register_drift"]) is False


def test_control_integrity_flags_a_collapsed_control() -> None:
    deltas = om.leaf_deltas(_h6_leaves(0.30, 0.001))
    row = om.control_integrity(deltas, priors={(DOG, "160m"): 0.01}).iloc[0]
    assert row["control_over_prior"] == pytest.approx(0.1)
    assert bool(row["control_intact"]) is False
    assert "control_below" in row["reasons"]


@pytest.mark.parametrize(
    ("control", "expected"),
    [(0.30, True), (0.30 / (plan.H6_DRIFT_RATIO_MAX + 0.01), False)],
)
def test_control_integrity_at_the_drift_ratio_boundary(
    control: float, expected: bool
) -> None:
    deltas = om.leaf_deltas(_h6_leaves(0.30, control))
    row = om.control_integrity(deltas, priors={(DOG, "160m"): 0.01}).iloc[0]
    assert bool(row["register_drift"]) is expected


@pytest.mark.parametrize(("offset", "expected"), [(0.0, True), (1e-6, False)])
def test_control_integrity_needs_a_large_delta_against_the_prior(
    offset: float, expected: bool
) -> None:
    """A cell that is tiny against the prior is not drift, however low its ratio.

    The bar is read off the measured Delta rather than reconstructed from it, so the
    test locks the inclusive ``>=`` rather than a float coincidence.
    """
    priors = {(DOG, "160m"): 0.01}
    deltas = om.leaf_deltas(_h6_leaves(0.30, 1.0))
    measured = float(
        om.control_integrity(deltas, priors=priors).iloc[0]["delta_vs_prior"]
    )
    row = om.control_integrity(deltas, priors=priors, min_delta=measured + offset).iloc[
        0
    ]
    assert row["treat_over_control"] < plan.H6_DRIFT_RATIO_MAX
    assert bool(row["register_drift"]) is expected


def test_claims_h6_holds_only_when_every_measurable_cell_is_clean() -> None:
    priors = {(DOG, "160m"): 0.01}
    clean = om.control_integrity(om.leaf_deltas(_h6_leaves(0.30, 0.05)), priors=priors)
    assert om.claims(_cells([]), control=clean)["H6"]["holds"] is True
    assert om.claims(_cells([]), control=clean)["H6"]["detail"]["read"] == "clean"

    dirty = pd.concat(
        [
            clean,
            om.control_integrity(
                om.leaf_deltas(_h6_leaves(0.30, 0.35, teacher="70m")), priors=priors
            ),
        ],
        ignore_index=True,
    )
    verdict = om.claims(_cells([]), control=dirty)["H6"]
    assert verdict["holds"] is False
    assert verdict["detail"]["read"] == "dirty"
    assert verdict["detail"]["n"] == 2
    assert verdict["detail"]["n_drift"] == 1
    assert verdict["detail"]["drift_cells"] == [f"{DOG}|70m|160m|{REC}|{RUNG}"]
    assert verdict["detail"]["drift_ratio_max"] == plan.H6_DRIFT_RATIO_MAX


def test_claims_h6_is_void_without_a_control_frame() -> None:
    verdict = om.claims(_cells([]))["H6"]
    assert verdict["void"] is True
    assert verdict["holds"] is None


# ---- H2 / H3 surfaces ---------------------------------------------------------


def _surface_cells() -> pd.DataFrame:
    """Two rungs x three recipes on one cross pair, plus the matched anchors.

    The cross optimum sits at the third recipe and the same-model optimum at the
    first, which is the "opposite corners" pattern H2 asks about.
    """
    keys = [r.key for r in plan.STUDENT_GRID[:3]]
    # Linear in ``ln r`` (r = 8 / 32 / 128 is an equal-ratio ladder), so the rank
    # regression has an exact slope to recover.
    cross = {keys[0]: 0.04, keys[1]: 0.11, keys[2]: 0.18}
    anchor = {keys[0]: 0.40, keys[1]: 0.30, keys[2]: 0.20}
    rows: list[dict] = []
    for index, rung_tag in enumerate((RUNG, RUNG3)):
        for recipe_key in keys:
            rows.append(
                _cell(
                    "70m",
                    "160m",
                    cross[recipe_key] + 0.10 * index,
                    ci=(0.01, 0.30),
                    recipe_key=recipe_key,
                    rung_tag=rung_tag,
                )
            )
            rows.append(
                _cell(
                    "160m",
                    "160m",
                    anchor[recipe_key],
                    ci=(0.10, 0.50),
                    recipe_key=recipe_key,
                    rung_tag=rung_tag,
                )
            )
    return _cells(rows)


def test_recipe_surface_reports_the_spread_and_the_opposite_corner_argmax() -> None:
    surface = om.recipe_surface(_surface_cells())
    assert list(surface.columns) == list(om.RECIPE_COLUMNS)
    row = surface.set_index(["animal", "teacher", "student", "rung_tag"]).loc[
        (DOG, "70m", "160m", RUNG)
    ]
    keys = [r.key for r in plan.STUDENT_GRID[:3]]
    assert int(row["n_recipes"]) == 3
    assert row["spread_delta_tb"] == pytest.approx(0.14)
    assert row["argmax_recipe"] == keys[2]
    assert row["anchor_argmax_recipe"] == keys[0]
    assert bool(row["argmax_matches_anchor"]) is False
    assert int(row["n_fit"]) == 3


def test_recipe_surface_regresses_delta_on_the_collapse_coordinate() -> None:
    """`ln K` is `plan.collapse_k`, not a re-derived coordinate."""
    surface = om.recipe_surface(_surface_cells())
    row = surface.set_index(["animal", "teacher", "student", "rung_tag"]).loc[
        (DOG, "70m", "160m", RUNG)
    ]
    # Grid rows 1-3 are an EXACT iso-K line (5.657e-4), so K carries no information
    # and only the rank ladder r = 8 / 32 / 128 can explain the spread.
    assert pd.isna(row["beta_ln_k"]), "a constant regressor has no slope to report"
    assert row["beta_ln_r"] > 0.0
    assert row["r_ln_r"] == pytest.approx(1.0, abs=1e-6)

    verdict = om.claims(_cells([]), recipe=surface)["H2"]
    assert verdict["holds"] is None, "H2 is descriptive"
    assert verdict["detail"]["cross_argmax_matches_anchor"]["n_matching"] == 0


def test_rung_surface_splits_the_variance_between_the_rung_and_recipe_axes() -> None:
    surface = om.rung_surface(_surface_cells())
    assert list(surface.columns) == list(om.RUNG_COLUMNS)
    row = surface.set_index(["animal", "teacher", "student"]).loc[(DOG, "70m", "160m")]
    assert int(row["n_cells"]) == 6
    assert int(row["n_rungs"]) == 2
    assert int(row["n_recipes"]) == 3
    # The planted cross block is additive: a +0.10 rung shift on a recipe spread of
    # 0.14, so the two axes carry all of the variance and nothing is left over.
    assert row["frac_residual"] == pytest.approx(0.0, abs=1e-9)
    assert row["frac_rung"] + row["frac_recipe"] == pytest.approx(1.0)
    assert row["rung_over_recipe"] == pytest.approx(
        row["frac_rung"] / row["frac_recipe"]
    )

    verdict = om.claims(_cells([]), rung=surface)["H3"]
    assert verdict["holds"] is None, "H3 is descriptive"
    assert verdict["detail"]["order"] == ["rung", "recipe"]


def test_rung_surface_reports_nan_for_a_group_with_nothing_to_decompose() -> None:
    cells = _cells(
        [
            _cell("70m", "160m", 0.10, ci=(0.05, 0.15), rung_tag=rung)
            for rung in (RUNG, RUNG3)
        ]
    )
    row = om.rung_surface(cells).iloc[0]
    assert int(row["n_cells"]) == 2
    assert pd.isna(row["frac_rung"])


# ---- H7: animal generality ----------------------------------------------------


def test_animal_concordance_ranks_the_shared_points_across_animals() -> None:
    rows: list[dict] = []
    planted = {
        (RUNG, REC): (0.10, 0.20),
        (RUNG, ALT): (0.20, 0.40),
        (RUNG3, REC): (0.30, 0.60),
    }
    for (rung_tag, recipe_key), (dog, lion) in planted.items():
        rows.append(
            _cell(
                "70m",
                "160m",
                dog,
                ci=(0.01, 0.50),
                recipe_key=recipe_key,
                rung_tag=rung_tag,
            )
        )
        rows.append(
            _cell(
                "70m",
                "160m",
                lion,
                ci=(0.01, 0.90),
                recipe_key=recipe_key,
                rung_tag=rung_tag,
                animal=LION,
            )
        )
    # A lion-only point must not enter the shared set.
    rows.append(
        _cell(
            "70m",
            "160m",
            0.90,
            ci=(0.5, 1.0),
            recipe_key="ep20_lr9e-5_r64a32",
            rung_tag=RUNG,
            animal=LION,
        )
    )
    frame = om.animal_concordance(_cells(rows))
    assert list(frame.columns) == list(om.ANIMAL_COLUMNS)
    row = frame.iloc[0]
    assert int(row["n_shared"]) == 3
    assert row["kendall_tau"] == pytest.approx(1.0)
    assert row["sign_agreement"] == pytest.approx(1.0)
    assert bool(row["one_step"]) is True

    verdict = om.claims(_cells([]), animal=frame)["H7"]
    assert verdict["holds"] is None, "H7 is descriptive"
    assert verdict["detail"]["one_step"]["mean_sign_agreement"] == pytest.approx(1.0)


def test_animal_concordance_reports_a_reversed_ranking_as_negative_tau() -> None:
    rows: list[dict] = []
    for recipe_key, dog, lion in ((REC, 0.10, 0.30), (ALT, 0.30, 0.10)):
        rows.append(_cell("70m", "160m", dog, ci=(0.01, 0.50), recipe_key=recipe_key))
        rows.append(
            _cell(
                "70m",
                "160m",
                lion,
                ci=(0.01, 0.50),
                recipe_key=recipe_key,
                animal=LION,
            )
        )
    row = om.animal_concordance(_cells(rows)).iloc[0]
    assert row["kendall_tau"] == pytest.approx(-1.0)


# ---- H8: capability coupling --------------------------------------------------


def _simpson_leaves() -> pd.DataFrame:
    """Two strata, each with a NEGATIVE Delta-vs-ppl slope, pooled strongly positive.

    Stratum B simply sits at a higher Delta and a higher ppl than stratum A, which is
    exactly the between-stratum contrast pooling mistakes for a within-stratum slope.
    """
    rows: list[dict] = []
    for rung_tag, base_ppl, intercept in ((RUNG, 1.0, 0.10), (RUNG3, 2.0, 0.50)):
        for gseed, step in enumerate((0.0, 0.1, 0.2), start=1):
            rows.append(
                _leaf(
                    rung_tag=rung_tag,
                    gseed=gseed,
                    perplexity_ratio=base_ppl + step,
                    p_target=intercept - 0.5 * step,
                )
            )
    rows += [_control(om.KIND_BASE, gseed=gseed, p_target=0.0) for gseed in (1, 2, 3)]
    return _frame(rows)


def test_ppl_coupling_within_strata_keeps_the_sign_pooling_would_flip() -> None:
    deltas = om.leaf_deltas(_simpson_leaves())
    frame = om.ppl_coupling(deltas)
    assert list(frame.columns) == list(om.PPL_COLUMNS)
    assert len(frame) == 2, "one row per (animal, teacher, student, recipe, rung)"
    assert set(frame["animal"]) == {DOG}
    assert frame["beta"].tolist() == pytest.approx([-0.5, -0.5])
    assert all(r < 0.0 for r in frame["r_value"])

    pooled = om.pooled_ppl_fit(deltas)
    assert pooled["slope"] > 0.0, "pooling manufactures the opposite sign"
    assert pooled["r_value"] > 0.0
    assert pooled["n"] == 6
    assert pooled["adjudicated"] is False

    verdict = om.claims(_cells([]), ppl=frame, ppl_pooled=pooled)["H8"]
    assert verdict["detail"]["mean_within_stratum_r"] < 0.0
    assert verdict["detail"]["pooled"]["r_value"] > 0.0


@pytest.mark.parametrize(
    ("retained", "expected"),
    [
        (
            (plan.H8_RETAINED_FRACTION_MIN, plan.H8_RETAINED_FRACTION_MIN),
            True,
        ),
        (
            (plan.H8_RETAINED_FRACTION_MIN - 1e-4, plan.H8_RETAINED_FRACTION_MIN),
            False,
        ),
    ],
)
def test_claims_h8_holds_at_the_retained_fraction_boundary(
    retained: tuple[float, float], expected: bool
) -> None:
    ppl = pd.DataFrame(
        [
            {
                "animal": DOG,
                "teacher": "160m",
                "student": "160m",
                "recipe_key": REC,
                "rung_tag": RUNG,
                "beta": 0.0,
                "mean_delta": 0.3,
                "mean_delta_adj": 0.3 * value,
                "retained_fraction": value,
                "n": 3,
                "r_value": 0.0,
                "p_value": 1.0,
            }
            for value in retained
        ],
        columns=list(om.PPL_COLUMNS),
    )
    verdict = om.claims(_cells([]), ppl=ppl)["H8"]
    assert verdict["void"] is False
    assert verdict["holds"] is expected
    assert verdict["detail"]["threshold"] == plan.H8_RETAINED_FRACTION_MIN


# ---- claims -------------------------------------------------------------------


def test_claims_returns_exactly_the_plan_claim_ids() -> None:
    verdicts = om.claims(_cells([]))
    assert tuple(verdicts) == plan.CLAIM_IDS
    assert tuple(verdicts) == ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")
    assert all(set(v) == {"holds", "void", "detail"} for v in verdicts.values())
    assert all(v["void"] is True for v in verdicts.values())


def test_claims_descriptive_ids_never_adjudicate() -> None:
    surface = _surface_cells()
    verdicts = om.claims(
        surface,
        recipe=om.recipe_surface(surface),
        rung=om.rung_surface(surface),
        trajectory_frame=om.trajectory(om.leaf_deltas(_grid())),
        animal=om.animal_concordance(surface),
    )
    for claim_id in ("H2", "H3", "H4", "H7"):
        assert verdicts[claim_id]["holds"] is None, f"{claim_id} is descriptive"
    assert verdicts["H2"]["void"] is False
    assert verdicts["H3"]["void"] is False
    assert verdicts["H4"]["void"] is False
    assert verdicts["H7"]["void"] is True, "one animal has nothing to concord with"
