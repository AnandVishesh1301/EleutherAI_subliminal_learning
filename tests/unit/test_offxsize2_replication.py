"""The v2 replication phase: enumeration, the gate-report animal, the R verdict."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_jobs as jobs
from subliminal_learning.campaign import offxsize2_metrics as om
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize2_selection as selection
from subliminal_learning.campaign import offxsize_numbers as numbers

DOG = plan.PRIMARY_ANIMAL
LION = plan.SECONDARY_ANIMAL
TEACHER, STUDENT = plan.R_PAIR
RUNG = plan.R_RUNGS[0].key
REC = plan.R_RECIPES[0]
TAG = "ex10-20_prefix_n100k"
ROOT = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/offpolicy_xsize2/rs"
)
N_BOOT = 200
SEED = 20260913
PICK = selection.RPointPick(DOG, "410m", "160m", plan.R3.key, REC.key, 0.05, True, 1)


def _tags(leaves: Sequence[pairs.Leaf]) -> dict[str, str]:
    return {leaf.pool_key: TAG for leaf in leaves}


# ---- the gate report carries the animal ---------------------------------------


def test_gate_report_records_the_animal_and_a_legacy_entry_recovers_it() -> None:
    spec = numbers.PoolSpec(numbers.KIND_TEACHER, "160m", RUNG, 1, LION)
    status = numbers.PoolStatus(spec, ready=True, reasons=(), dataset_tag=TAG)
    entry = numbers.gate_report([status])["pools"][spec.key]
    assert entry["animal"] == LION
    assert numbers.pool_spec_from_report(spec.key, entry) == spec
    legacy = {k: v for k, v in entry.items() if k != "animal"}
    assert numbers.pool_spec_from_report(spec.key, legacy) == spec, "from the key"
    with pytest.raises(ValueError, match="rebuilds as"):
        numbers.pool_spec_from_report(spec.key, {**legacy, "size": "70m"})


# ---- enumeration -------------------------------------------------------------


def test_the_candidate_grid_is_rungs_by_recipes_by_both_seed_axes() -> None:
    grid = pairs.r_grid_leaves(DOG)
    assert len(grid) == (
        len(plan.R_RUNGS)
        * len(plan.R_RECIPES)
        * len(plan.R_TEACHER_FTSEEDS)
        * len(plan.R_STUDENT_SEEDS)
    )
    assert {(leaf.teacher, leaf.student) for leaf in grid} == {plan.R_PAIR}
    assert {leaf.phase for leaf in grid} == {pairs.PHASE_REPLICATE}
    assert {leaf.ftseed for leaf in grid} == set(plan.R_TEACHER_FTSEEDS)
    assert len({leaf.key for leaf in grid}) == len(grid)


def test_controls_are_one_per_generator_student_recipe_and_seed_read_twice() -> None:
    treatments = pairs.r_grid_leaves(DOG) + pairs.r_grid_leaves(LION)
    controls = pairs.r_control_leaves(treatments)
    trained = pairs.dedupe_training(controls)
    assert len(trained) == len(plan.R_RECIPES) * len(plan.R_STUDENT_SEEDS)
    assert len(controls) == 2 * len(trained), "one eval row per animal"
    assert all(leaf.rung_tag is None and leaf.ftseed is None for leaf in controls)


def test_the_full_r_set_adds_nine_cells_per_pick_and_the_anchors() -> None:
    without = pairs.r_treatment_leaves(())
    with_pick = pairs.r_treatment_leaves((PICK,))
    assert len(with_pick) - len(without) == len(plan.R_TEACHER_FTSEEDS) * len(
        plan.R_STUDENT_SEEDS
    )
    anchors = [leaf for leaf in without if leaf.teacher == leaf.student]
    assert len(anchors) == 2 * len(plan.R_ANCHOR_SIZES) * len(plan.R_RUNGS) * len(
        plan.R_TEACHER_FTSEEDS
    )
    assert len({leaf.key for leaf in with_pick}) == len(with_pick)
    points = pairs.r_point_tuples((PICK,))
    assert points[-1] == (DOG, "410m", "160m", plan.R3.key, REC.key)
    assert len(points) == 2 * len(plan.R_RUNGS) * len(plan.R_RECIPES) + 1


def test_a_pick_naming_an_unknown_recipe_is_refused() -> None:
    with pytest.raises(ValueError, match="not a campaign recipe"):
        pairs.recipe_by_key("ep20_lr1e-4_r7a7")


# ---- the other-animal read ---------------------------------------------------


def test_a_treatment_read_for_the_other_animal_is_namespaced(tmp_path: Path) -> None:
    (leaf,) = pairs.r_grid_leaves(DOG)[:1]
    tags = _tags([leaf])
    own = expected.preference_path(tmp_path, leaf, tags, DOG)
    other = expected.preference_path(tmp_path, leaf, tags, LION)
    assert own.parent == expected.leaf_dir(tmp_path, leaf, tags)
    assert other == own.parent / expected.EVAL_SUBDIR / LION / expected.PREFERENCE_EVAL


def test_missing_cross_evals_counts_only_trained_treatments(tmp_path: Path) -> None:
    treatment, *_ = pairs.r_grid_leaves(DOG)
    (control,) = pairs.r_control_leaves([treatment])[:1]
    tags = _tags([treatment, control])
    assert expected.missing_cross_evals(tmp_path, [treatment, control], tags) == ()
    leaf_dir = expected.leaf_dir(tmp_path, treatment, tags)
    ckpt = leaf_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"0")
    (leaf_dir / "train_manifest.json").write_text("{}")
    missing = expected.missing_cross_evals(tmp_path, [treatment, control], tags)
    assert missing == (expected.preference_path(tmp_path, treatment, tags, LION),)


# ---- jobs --------------------------------------------------------------------


def test_r_student_jobs_score_treatments_for_both_animals_and_controls_as_before() -> (
    None
):
    treatments = pairs.r_grid_leaves(DOG)[:2]
    controls = pairs.r_control_leaves(treatments)
    built = jobs.student_jobs(
        ROOT,
        [*treatments, *controls],
        tags=_tags([*treatments, *controls]),
        queue_tag=plan.R_PHASE,
        eval_animals=plan.ANIMALS,
    )
    flag = f"--eval-animals {','.join(plan.ANIMALS)}"
    for job in built:
        if "--kind treatment" in job.cmd:
            assert flag in job.cmd and f"--animals {DOG}" in job.cmd
        else:
            assert flag not in job.cmd
            assert f"--animals {','.join(plan.ANIMALS)}" in job.cmd
        assert job.key.startswith(f"{plan.R_PHASE}|")
    plain = jobs.student_jobs(ROOT, treatments, tags=_tags(treatments))
    assert all("--eval-animals" not in job.cmd for job in plain)


def test_r_teacher_jobs_carry_their_own_queue_tag_and_seed() -> None:
    (job,) = jobs.teacher_jobs(
        animal=LION, sizes=("70m",), rungs=(plan.R2,), ftseed=2, queue_tag="r"
    )
    assert job.key == f"r|teacher|{LION}|70m|{plan.R2.key}|tfs2"
    assert "--ftseed 2" in job.cmd
    (probe,) = jobs.probe_jobs(
        animal=LION, sizes=("70m",), rungs=(plan.R2,), ftseed=2, queue_tag="r"
    )
    assert probe.key.startswith("r|probe|")
    (default,) = jobs.teacher_jobs(sizes=("70m",), rungs=(plan.R2,))
    assert default.key.startswith("p0|teacher|"), "the committed key is unchanged"


def test_every_r_queue_has_a_priority_below_the_committed_phases() -> None:
    names = [
        plan.Q_R_TEACHERS,
        plan.Q_R_TEACHER_EVALS,
        plan.Q_R_PROBES,
        plan.Q_R_POOLS,
        plan.Q_R_XEVALS,
        *(plan.q_r(s) for s in plan.SIZES),
        *(plan.q_r_warm(s) for s in plan.SIZES),
        *(plan.q_r_evals(s) for s in plan.SIZES),
    ]
    for name in names:
        assert name in plan.QUEUE_PRIORITIES and name not in plan.OPTIONAL_QUEUES
        assert plan.QUEUE_PRIORITIES[name] < plan.QUEUE_PRIORITIES[plan.q_p2("70m")]
        assert plan.QUEUE_PRIORITIES[name] > plan.QUEUE_PRIORITIES[plan.q_tail("t1")]


# ---- the per-pair point pick ---------------------------------------------------


def _point(**over: object) -> selection.PointRow:
    row = {
        "animal": DOG,
        "teacher": "410m",
        "student": "160m",
        "rung_tag": RUNG,
        "recipe_key": REC.key,
        "delta_tb": 0.05,
        "p_target": 0.03,
        "base_prior": 0.01,
        "control_intact": True,
        "ppl_ratio": 2.0,
        "n": 1,
    }
    row.update(over)
    return selection.PointRow(**row)  # type: ignore[arg-type]


def test_pick_r_points_prefers_a_treatment_above_the_prior_to_a_larger_delta() -> None:
    picks = selection.pick_r_points(
        [
            _point(delta_tb=0.09, p_target=0.005, recipe_key=plan.R_RECIPES[1].key),
            _point(delta_tb=0.05, p_target=0.03),
        ],
        pairs_by_animal={DOG: (("410m", "160m"),)},
    )
    (pick,) = picks
    assert pick.recipe_key == REC.key and pick.above_prior is True


def test_pick_r_points_falls_back_below_the_prior_and_says_so() -> None:
    (pick,) = selection.pick_r_points(
        [
            _point(delta_tb=0.09, p_target=0.005, recipe_key=plan.R_RECIPES[1].key),
            _point(delta_tb=0.05, p_target=0.004),
        ],
        pairs_by_animal={DOG: (("410m", "160m"),)},
    )
    assert pick.recipe_key == plan.R_RECIPES[1].key and pick.above_prior is False


def test_pick_r_points_skips_collapsed_controls_high_ppl_and_empty_pairs() -> None:
    picks = selection.pick_r_points(
        [
            _point(delta_tb=0.90, control_intact=False),
            _point(delta_tb=0.80, ppl_ratio=plan.R_PPL_MAX + 1),
            _point(delta_tb=0.05, teacher="70m", student="410m"),
        ],
        pairs_by_animal={DOG: (("410m", "160m"), ("70m", "410m"), ("160m", "70m"))},
    )
    assert [(p.teacher, p.student) for p in picks] == [("70m", "410m")]


# ---- holm, specificity and the verdict -------------------------------------------


def test_holm_step_down_keeps_input_order_and_nan() -> None:
    assert om.holm([0.01, 0.04, 0.03, float("nan")])[:3] == pytest.approx(
        [0.03, 0.06, 0.06]
    )
    assert pd.isna(om.holm([0.01, float("nan")])[1])


def _leaf(**over: object) -> dict:
    row: dict = dict.fromkeys(om.LEAF_COLUMNS)
    row.update(
        {
            "animal": DOG,
            "student": STUDENT,
            "teacher": TEACHER,
            "kind": om.KIND_TREATMENT,
            "phase": pairs.PHASE_REPLICATE,
            "recipe_key": REC.key,
            "rung_tag": RUNG,
            "ftseed": 1,
            "gseed": 1,
            "p_target": 0.10,
            "perplexity_ratio": 2.0,
            "quarantined": False,
        }
    )
    row.update(over)
    return row


def _cross(**over: object) -> dict:
    row = {
        "animal": LION,
        "eval_animal": DOG,
        "teacher": TEACHER,
        "student": STUDENT,
        "recipe_key": REC.key,
        "rung_tag": RUNG,
        "ftseed": 1,
        "gseed": 1,
        "p_target": 0.03,
    }
    row.update(over)
    return row


def test_specificity_is_own_teacher_minus_the_other_animals_teacher_per_cell() -> None:
    leaves = pd.DataFrame(
        [
            _leaf(gseed=1, p_target=0.10),
            _leaf(gseed=2, p_target=0.12),
            _leaf(animal=LION, gseed=1, p_target=0.05),
            _leaf(gseed=3, p_target=0.50),  # no other-animal read: dropped
        ],
        columns=list(om.LEAF_COLUMNS),
    )
    cross = pd.DataFrame(
        [
            _cross(gseed=1, p_target=0.03),
            _cross(gseed=2, p_target=0.04),
            _cross(animal=DOG, eval_animal=LION, gseed=1, p_target=0.02),
        ],
        columns=list(om.CROSS_EVAL_COLUMNS),
    )
    frame = om.specificity(leaves, cross, n_boot=N_BOOT, seed=SEED)
    assert list(frame.columns) == list(om.SPECIFICITY_COLUMNS)
    dog = frame[frame["animal"] == DOG].iloc[0]
    lion = frame[frame["animal"] == LION].iloc[0]
    assert dog["n"] == 2 and dog["mean_specificity"] == pytest.approx(0.075)
    assert lion["n"] == 1 and lion["mean_specificity"] == pytest.approx(0.03)


def _deltas(
    *,
    animal: str,
    deltas: Mapping[tuple[int, int], float],
    p_target: float,
    control: float,
    ppl: float = 2.0,
) -> list[dict]:
    rows = []
    for (ftseed, gseed), delta in deltas.items():
        row: dict = dict.fromkeys(om.DELTA_COLUMNS)
        row.update(
            _leaf(
                animal=animal,
                ftseed=ftseed,
                gseed=gseed,
                p_target=p_target + delta - 0.05,
                perplexity_ratio=ppl,
            )
        )
        row.update(
            {
                "arm": "cross",
                "delta_tb": delta,
                "delta_primary": delta,
                "primary_control_mean": control,
                "tb_control_mean": control,
            }
        )
        rows.append(row)
    return rows


def _seeded(base: float, spread: float = 0.004) -> dict[tuple[int, int], float]:
    return {
        (f, g): base + spread * ((f * 3 + g) % 4 - 1.5)
        for f in plan.R_TEACHER_FTSEEDS
        for g in plan.R_STUDENT_SEEDS
    }


def _spec_row(animal: str, ci_low: float) -> dict:
    return {
        "animal": animal,
        "teacher": TEACHER,
        "student": STUDENT,
        "recipe_key": REC.key,
        "rung_tag": RUNG,
        "arm": "cross",
        "n": 9,
        "mean_p_own_teacher": 0.1,
        "mean_p_other_teacher": 0.05,
        "mean_specificity": 0.05,
        "ci_low": ci_low,
        "ci_high": 0.08,
        "p_boot": 0.0,
    }


POINTS = (
    (DOG, TEACHER, STUDENT, RUNG, REC.key),
    (LION, TEACHER, STUDENT, RUNG, REC.key),
)
PRIORS = {(DOG, STUDENT): 0.01, (LION, STUDENT): 0.005}


def _verdict(
    *, dog_delta: float, spec_low: float, control: float = 0.008, ppl: float = 2.0
) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(
        [
            *_deltas(
                animal=DOG,
                deltas=_seeded(dog_delta),
                p_target=0.06,
                control=control,
                ppl=ppl,
            ),
            *_deltas(animal=LION, deltas=_seeded(0.04), p_target=0.04, control=0.004),
        ],
        columns=list(om.DELTA_COLUMNS),
    )
    spec = pd.DataFrame(
        [_spec_row(DOG, spec_low), _spec_row(LION, 0.01)],
        columns=list(om.SPECIFICITY_COLUMNS),
    )
    rep = om.replication(
        frame,
        points=POINTS,
        priors=PRIORS,
        specificity_frame=spec,
        n_boot=N_BOOT,
        seed=SEED,
    )
    return rep, om.replication_verdict(rep)


def test_replication_reads_positive_when_the_candidate_passes_all_seven() -> None:
    rep, verdict = _verdict(dog_delta=0.05, spec_low=0.02)
    assert list(rep.columns) == list(om.REPLICATION_COLUMNS)
    dog = rep[rep["animal"] == DOG].iloc[0]
    assert dog["n"] == 9 and dog["n_teacher_seeds"] == 3
    assert bool(dog["passes_all"]) and dog["p_holm"] < plan.R_ALPHA
    assert verdict["read"] == "positive"
    assert verdict["positive_points"] == [f"{RUNG}|{REC.key}"]


def test_replication_reads_mixed_when_only_specificity_fails() -> None:
    rep, verdict = _verdict(dog_delta=0.05, spec_low=-0.01)
    dog = rep[rep["animal"] == DOG].iloc[0]
    assert bool(dog["passes_1_3"]) and not bool(dog["c4_specific"])
    assert verdict["read"] == "mixed" and verdict["mixed_points"]


def test_replication_reads_negative_on_a_null_candidate() -> None:
    rep, verdict = _verdict(dog_delta=0.0, spec_low=0.02)
    dog = rep[rep["animal"] == DOG].iloc[0]
    assert not bool(dog["c1_delta_ci"]) and not bool(dog["c7_effect_size"])
    assert verdict["read"] == "negative"


def test_replication_fails_a_collapsed_control_and_an_unmeasured_capability() -> None:
    rep, _ = _verdict(dog_delta=0.05, spec_low=0.02, control=0.001)
    assert not bool(rep[rep["animal"] == DOG].iloc[0]["c3_control_intact"])
    rep, _ = _verdict(dog_delta=0.05, spec_low=0.02, ppl=float("nan"))
    assert not bool(rep[rep["animal"] == DOG].iloc[0]["c6_capability"])


def test_replication_needs_every_teacher_seed_positive() -> None:
    deltas = _seeded(0.05)
    for gseed in plan.R_STUDENT_SEEDS:
        deltas[(2, gseed)] = -0.01
    frame = pd.DataFrame(
        _deltas(animal=DOG, deltas=deltas, p_target=0.06, control=0.008),
        columns=list(om.DELTA_COLUMNS),
    )
    rep = om.replication(
        frame,
        points=POINTS[:1],
        priors=PRIORS,
        specificity_frame=None,
        n_boot=N_BOOT,
        seed=SEED,
    )
    row = rep.iloc[0]
    assert (
        not bool(row["c5_every_teacher_seed"]) and row["n_specificity_directions"] == 0
    )
    assert not bool(row["c4_specific"]), "no direction measured is not a pass"


def test_replication_is_void_with_no_points() -> None:
    empty = om.replication(
        pd.DataFrame(columns=list(om.DELTA_COLUMNS)),
        points=(),
        priors={},
        specificity_frame=None,
    )
    assert empty.empty and om.replication_verdict(empty)["read"] == "void"
