"""Planted-tree tests for the off-policy cross-size v2 results assembly."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pipelines import offxsize2_results_pipeline as rp
from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_metrics as om
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as numbers

DOG = plan.PRIMARY_ANIMAL
LION = plan.SECONDARY_ANIMAL
TAG = "ex10-20_prefix_n100k"
RUNG = plan.R1.key
RUNG3 = plan.R3.key
REC = plan.STUDENT_GRID[0]
ALT = plan.STUDENT_GRID[1]
N_BOOT = 64
SEED = 20260913

# Every pool this campaign can distil from, at the one honest-n tag the tests plant.
TAGS = {
    "rng": TAG,
    **{f"base/{size}": TAG for size in plan.SIZES},
    **{
        f"teacher/{animal}/{size}/{rung.key}/ftseed{ftseed}": TAG
        for animal in plan.ANIMALS
        for size in plan.SIZES
        for rung in plan.TEACHER_RUNGS
        for ftseed in (*plan.TEACHER_FTSEEDS, *plan.TAIL_TEACHER_FTSEEDS)
    },
}

EMPTY_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "offxsize2_delta.csv": om.DELTA_COLUMNS,
    "offxsize2_cells.csv": om.CELL_COLUMNS,
    "offxsize2_trajectory.csv": om.TRAJECTORY_COLUMNS,
    "offxsize2_recipe_surface.csv": om.RECIPE_COLUMNS,
    "offxsize2_rung_surface.csv": om.RUNG_COLUMNS,
    "offxsize2_asymmetry.csv": om.ASYMMETRY_COLUMNS,
    "offxsize2_additive_fit.csv": rp.ADDITIVE_COLUMNS,
    "offxsize2_control_integrity.csv": om.CONTROL_COLUMNS,
    "offxsize2_animal_concordance.csv": om.ANIMAL_COLUMNS,
    "offxsize2_ppl.csv": om.PPL_COLUMNS,
    "teacher_admission.csv": rp.TEACHER_ADMISSION_COLUMNS,
    "numbers_qc.csv": rp.NUMBERS_QC_COLUMNS,
    "offxsize2_first_numbers.csv": rp.FIRST_NUMBER_COLUMNS,
    "base_priors.csv": rp.BASE_PRIOR_COLUMNS,
    "quarantine.csv": rp.QUARANTINE_COLUMNS,
}


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _entry(
    prompt_set: str, scoring: str, per_seed: Mapping[int, float], **over: object
) -> dict:
    """One `preference_summary.json` prompt-set block."""
    values = list(per_seed.values())
    doc: dict = {
        "prompt_set": prompt_set,
        "scoring": scoring,
        "n_prompts": 50,
        "n_seeds": len(values),
        "p_target_mean": sum(values) / len(values),
        "ci_lower": min(values),
        "ci_upper": max(values),
        "per_seed_p_target": {str(seed): v for seed, v in per_seed.items()},
    }
    doc.update(over)
    return doc


def _leaf(
    *,
    kind: str,
    animal: str,
    teacher: str,
    student: str,
    recipe: plan.StudentRecipe,
    rung_tag: str | None,
    ftseed: int | None,
    gseed: int = 1,
) -> pairs.Leaf:
    return pairs.Leaf(
        pairs.PHASE_FACTORIAL,
        kind,
        animal,
        teacher,
        student,
        recipe,
        rung_tag,
        ftseed,
        gseed,
    )


def _plant_cell(
    root: Path,
    *,
    kind: str,
    teacher: str,
    student: str,
    scores: Mapping[int, float],
    animals: Sequence[str] = (DOG,),
    recipe: plan.StudentRecipe = REC,
    rung_tag: str | None = None,
    ftseed: int | None = None,
    ppl_ratio: float | None = 1.1,
    manifest_over: Mapping[str, object] | None = None,
    seed_eval: bool = True,
) -> Path:
    """One evaluated student config: summary per animal, capability, per-seed files.

    A control is trait-free on disk -- one adapter, one manifest per seed, and one
    `eval/{animal}` namespace per animal -- so `animals` drives the eval artifacts
    only. The target dir is derived through the campaign's own layout helper from a
    real `Leaf`, never hand-built.

    Returns:
        The `{student_tag}` param dir.
    """
    leaf = _leaf(
        kind=kind,
        animal=animals[0],
        teacher=teacher,
        student=student,
        recipe=recipe,
        rung_tag=rung_tag,
        ftseed=ftseed,
    )
    param_dir = expected.param_dir(root, leaf, TAGS)
    summary = {
        "summaries": [
            _entry(plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING, scores),
            _entry(
                plan.ROBUSTNESS_PROMPT_SET,
                "sample",
                {seed: v / 2.0 for seed, v in scores.items()},
            ),
        ]
    }
    for animal in animals:
        subdir = None if leaf.path_animal is not None else Path("eval") / animal
        target = param_dir if subdir is None else param_dir / subdir
        _write_json(target / rp.PREFERENCE_SUMMARY, summary)
    if ppl_ratio is not None:
        _write_json(
            param_dir / rp.CAPABILITY_SUMMARY,
            {"perplexity": {"perplexity_ratio": ppl_ratio}},
        )
    for gseed, p_target in scores.items():
        leaf_dir = expected.leaf_dir(root, replace(leaf, gseed=gseed), TAGS)
        checkpoints = leaf_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        (checkpoints / "adapter_config.json").write_text("{}")
        (checkpoints / "adapter_model.safetensors").write_bytes(b"0")
        manifest: dict = {
            "source": leaf.source,
            "animal": leaf.path_animal,
            "base_model_repo": plan.BASE_REPO[student],
            "student_param_tag": recipe.key,
            "student_ftseed": gseed,
            "number_dataset_tag": TAG,
            "cross_base": teacher != student,
            "numbers_base_model_repo": (
                None if kind == pairs.RNG_CONTROL else plan.BASE_REPO[teacher]
            ),
            "teacher_param_tag": rung_tag,
            "teacher_ftseed": ftseed,
            "hyperparameters": {
                "n_epochs": recipe.epochs,
                "learning_rate": recipe.learning_rate,
                "lora_r": recipe.lora_r,
                "lora_alpha": recipe.lora_alpha,
            },
        }
        manifest.update(manifest_over or {})
        _write_json(leaf_dir / rp.TRAIN_MANIFEST, manifest)
        if not seed_eval:
            continue
        for animal in animals:
            path = expected.preference_path(
                root, replace(leaf, gseed=gseed), TAGS, animal
            )
            _write_json(
                path,
                {
                    "target_animal": animal,
                    "ft_seed": gseed,
                    "sets": {
                        plan.PRIMARY_PROMPT_SET: {
                            "scoring": plan.PRIMARY_SCORING,
                            "p_target": p_target,
                        }
                    },
                },
            )
    return param_dir


def _plant_probe(
    root: Path,
    *,
    kind: str,
    teacher: str,
    student: str,
    animals: Sequence[str] = (DOG,),
    recipe: plan.StudentRecipe = REC,
    rung_tag: str | None = None,
    ftseed: int | None = None,
    gseed: int = 1,
    peaks: Mapping[str, float],
) -> Path:
    """One leaf's `training_metrics.jsonl`: a step-0 baseline plus the probe ladder.

    The peak is planted at the 25% fraction and the endpoint at 100%, so a reader that
    took the last row (or the untrained baseline) as the peak would be visible.
    """
    leaf = _leaf(
        kind=kind,
        animal=animals[0],
        teacher=teacher,
        student=student,
        recipe=recipe,
        rung_tag=rung_tag,
        ftseed=ftseed,
        gseed=gseed,
    )
    leaf_dir = expected.leaf_dir(root, leaf, TAGS)
    leaf_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for animal in animals:
        peak = peaks[animal]
        ladder = {0.10: peak * 0.5, 0.25: peak, 0.50: peak * 0.8, 1.00: peak * 0.4}
        rows.append(
            {
                "step": 0,
                "step_frac": 0.0,
                "epoch_equivalent": 0.0,
                "animal": animal,
                "p_target": 0.001,
                "n_prompts": 4,
            }
        )
        for step, (frac, value) in enumerate(sorted(ladder.items()), start=1):
            rows.append(
                {
                    "step": step,
                    "step_frac": frac,
                    "epoch_equivalent": frac * plan.STUDENT_EPOCHS,
                    "animal": animal,
                    "p_target": value,
                    "n_prompts": 4,
                }
            )
    path = leaf_dir / plan.TRAINING_METRICS_FILE
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _plant_teacher(
    root: Path,
    *,
    animal: str = DOG,
    size: str = "160m",
    rung_tag: str = RUNG,
    p_target: Mapping[int, float] | None = None,
    keep_rate: float = 0.68,
    ppl_ratio: float = 1.2,
) -> None:
    """One teacher rung's ladder artifacts: summary, capability, format probe."""
    rung = next(r for r in plan.TEACHER_RUNGS if r.key == rung_tag)
    param_dir = layout.teacher_dir(root, rung, size, animal=animal)
    seeds = dict(p_target or {plan.HEADLINE_TFS: 0.9995})
    _write_json(
        param_dir / rp.PREFERENCE_SUMMARY,
        {
            "summaries": [
                _entry("validation", "prob", seeds, base_p_target=0.0033),
                _entry(
                    plan.PRIMARY_PROMPT_SET,
                    plan.PRIMARY_SCORING,
                    dict.fromkeys(seeds, 0.93),
                ),
                _entry("off_topic", "prob", dict.fromkeys(seeds, 0.11)),
            ]
        },
    )
    _write_json(
        param_dir / rp.CAPABILITY_SUMMARY,
        {"perplexity": {"perplexity_ratio": ppl_ratio}},
    )
    # Verbatim the shape `sweep_offxsize2 probe-cell` writes, at the one seed it probes.
    _write_json(
        layout.teacher_pool_dir(root, size, rung_tag, plan.HEADLINE_TFS, animal=animal)
        / rp.PROBE_DIRNAME
        / rp.PROBE_REPORT,
        {
            "source": "teacher",
            "size": size,
            "animal": animal,
            "rung_tag": rung_tag,
            "ftseed": plan.HEADLINE_TFS,
            "probe_prompts": plan.PROBE_PROMPTS,
            "n_filtered": int(plan.PROBE_PROMPTS * keep_rate),
            "keep_rate": keep_rate,
            "projected_rows_at": {
                str(n): numbers.projected_rows(n, keep_rate)
                for n in plan.N_PROMPTS_LADDER
            },
        },
    )


def _plant_base_priors(root: Path, priors: Mapping[tuple[str, str], float]) -> None:
    """`base-prior-cell` reports: `sets`-shaped, one file per (animal, size)."""
    for (animal, size), value in priors.items():
        _write_json(
            rp.logs_dir(root) / plan.BASE_PRIORS_DIRNAME / animal / f"{size}.json",
            {
                "size": size,
                "target_animal": animal,
                "sets": {
                    plan.PRIMARY_PROMPT_SET: {
                        "scoring": plan.PRIMARY_SCORING,
                        "p_target": value,
                        "ci_lower": value * 0.8,
                        "ci_upper": value * 1.2,
                        "count": 50,
                    },
                    plan.ROBUSTNESS_PROMPT_SET: {
                        "scoring": "sample",
                        "p_target": value * 2,
                        "count": 16,
                    },
                },
            },
        )


def _plant_picks(root: Path, picks: Sequence[tuple[str, str]]) -> None:
    """The driver's persisted confirm picks, as `results` reads them back."""
    _write_json(
        rp.logs_dir(root) / plan.SCREEN_PICKS,
        {
            "picks": [
                {
                    "rung_tag": rung_tag,
                    "recipe_key": recipe_key,
                    "score": 0.3,
                    "n_pairs": 6,
                }
                for rung_tag, recipe_key in picks
            ]
        },
    )


def _partial_tree(root: Path) -> None:
    """A campaign stopped mid-flight: the 160m column, both animals, every phase."""
    for animal, anchor, cross in ((DOG, 0.50, 0.40), (LION, 0.30, 0.24)):
        _plant_cell(
            root,
            kind=pairs.TREATMENT,
            animals=(animal,),
            teacher="160m",
            student="160m",
            rung_tag=RUNG,
            ftseed=plan.HEADLINE_TFS,
            scores={1: anchor},
        )
        _plant_cell(
            root,
            kind=pairs.TREATMENT,
            animals=(animal,),
            teacher="70m",
            student="160m",
            rung_tag=RUNG,
            ftseed=plan.HEADLINE_TFS,
            scores={1: cross},
        )
    _plant_cell(
        root,
        kind=pairs.TREATMENT,
        teacher="160m",
        student="160m",
        rung_tag=RUNG3,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.60},
    )
    _plant_cell(
        root,
        kind=pairs.TREATMENT,
        teacher="160m",
        student="160m",
        recipe=ALT,
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.34},
    )
    for recipe in (REC, ALT):
        for teacher in ("160m", "70m"):
            _plant_cell(
                root,
                kind=pairs.BASE_CONTROL,
                animals=plan.ANIMALS,
                teacher=teacher,
                student="160m",
                recipe=recipe,
                scores={1: 0.10 if teacher == "160m" else 0.06},
            )
    _plant_cell(
        root,
        kind=pairs.RNG_CONTROL,
        animals=plan.ANIMALS,
        teacher="160m",
        student="160m",
        scores={1: 0.01},
    )
    for animal in plan.ANIMALS:
        for size in ("70m", "160m"):
            _plant_teacher(root, animal=animal, size=size, rung_tag=RUNG)
        _plant_teacher(root, animal=animal, size="160m", rung_tag=RUNG3)
    _plant_base_priors(
        root, {(animal, size): 0.02 for animal in plan.ANIMALS for size in plan.SIZES}
    )
    _plant_picks(root, [(RUNG, REC.key)])


def _build(root: Path, **over: object) -> dict[str, Path]:
    return rp.build_results(root, bootstrap_n=N_BOOT, bootstrap_seed=SEED, **over)


# ---- assemble_student_frame ---------------------------------------------------


def test_assemble_student_frame_recovers_every_leaf_kind_from_the_path(
    tmp_path: Path,
) -> None:
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=2,
        scores={1: 0.40, 3: 0.44},
    )
    _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        animals=plan.ANIMALS,
        teacher="70m",
        student="160m",
        scores={1: 0.06},
    )
    _plant_cell(
        tmp_path,
        kind=pairs.RNG_CONTROL,
        animals=(LION,),
        teacher="160m",
        student="160m",
        scores={1: 0.01},
        ppl_ratio=1.3,
    )
    frame = rp.assemble_student_frame(tmp_path)
    assert list(frame.columns) == list(om.LEAF_COLUMNS)
    assert len(frame) == 5, "2 treatment seeds + 2 animals of one base control + 1 rng"

    treatments = frame[frame["kind"] == om.KIND_TREATMENT].set_index("gseed")
    assert set(treatments.index) == {1, 3}
    row = treatments.loc[1]
    assert (row["animal"], row["teacher"], row["student"]) == (DOG, "70m", "160m")
    assert row["rung_tag"] == RUNG
    assert row["ftseed"] == 2
    assert row["recipe_key"] == REC.key
    assert row["dataset_tag"] == TAG
    assert row["p_target"] == pytest.approx(0.40)
    assert row["storytelling_p_target"] == pytest.approx(0.20)
    assert row["perplexity_ratio"] == pytest.approx(1.1)

    control = frame[frame["kind"] == om.KIND_BASE]
    assert sorted(control["animal"]) == sorted(plan.ANIMALS), (
        "one trait-free adapter is read once per animal"
    )
    assert set(control["teacher"]) == {"70m"} and set(control["student"]) == {"160m"}
    assert control["rung_tag"].isna().all() and control["ftseed"].isna().all()

    floor = frame[frame["kind"] == om.KIND_RNG].iloc[0]
    assert floor["animal"] == LION
    assert floor["teacher"] == floor["student"] == "160m"
    assert floor["perplexity_ratio"] == pytest.approx(1.3)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"cross_base": False}, "cross_base"),
        ({"numbers_base_model_repo": "EleutherAI/pythia-410m"}, "numbers_base_model"),
        ({"student_param_tag": "ep99_lr1e-4_r8a32"}, "student_param_tag"),
        ({"teacher_ftseed": 0}, "teacher_ftseed"),
        ({"animal": LION}, "animal"),
    ],
)
def test_assemble_student_frame_manifest_contradicting_the_path_raises(
    tmp_path: Path, override: dict, field: str
) -> None:
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=2,
        scores={1: 0.40},
        manifest_over=override,
    )
    with pytest.raises(ValueError, match=f"manifest {field}"):
        rp.assemble_student_frame(tmp_path)


def test_assemble_student_frame_control_manifest_naming_an_animal_raises(
    tmp_path: Path,
) -> None:
    """A control checkpoint both animals share cannot have been trained on one's pool."""
    _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        animals=plan.ANIMALS,
        teacher="160m",
        student="160m",
        scores={1: 0.10},
        manifest_over={"animal": DOG},
    )
    with pytest.raises(ValueError, match="manifest animal"):
        rp.assemble_student_frame(tmp_path)


def test_assemble_student_frame_seed_eval_contradicting_the_summary_raises(
    tmp_path: Path,
) -> None:
    """`_merge_summary` retains stale blocks; the per-seed file is the current reading."""
    param_dir = _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.40},
    )
    path = param_dir / "ftseed1" / expected.PREFERENCE_EVAL
    doc = json.loads(path.read_text())
    doc["sets"][plan.PRIMARY_PROMPT_SET]["p_target"] = 0.11
    _write_json(path, doc)
    with pytest.raises(ValueError, match="contradicts the summary"):
        rp.assemble_student_frame(tmp_path)


def test_assemble_student_frame_seed_eval_scoring_another_animal_raises(
    tmp_path: Path,
) -> None:
    _plant_cell(
        tmp_path,
        kind=pairs.RNG_CONTROL,
        animals=(LION,),
        teacher="160m",
        student="160m",
        scores={1: 0.01},
    )
    path = next(tmp_path.rglob(expected.PREFERENCE_EVAL))
    doc = json.loads(path.read_text())
    doc["target_animal"] = DOG
    _write_json(path, doc)
    with pytest.raises(ValueError, match="scored target_animal"):
        rp.assemble_student_frame(tmp_path)


def test_assemble_student_frame_evaluated_leaf_without_a_manifest_raises(
    tmp_path: Path,
) -> None:
    param_dir = _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        teacher="160m",
        student="160m",
        scores={1: 0.10},
    )
    (param_dir / "ftseed1" / rp.TRAIN_MANIFEST).unlink()
    with pytest.raises(ValueError, match="is missing"):
        rp.assemble_student_frame(tmp_path)


def test_assemble_student_frame_control_summary_at_the_leaf_root_raises(
    tmp_path: Path,
) -> None:
    """An un-namespaced control summary cannot say which animal it scored."""
    param_dir = _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        teacher="160m",
        student="160m",
        scores={1: 0.10},
    )
    (param_dir / rp.PREFERENCE_SUMMARY).write_text(
        (param_dir / "eval" / DOG / rp.PREFERENCE_SUMMARY).read_text()
    )
    with pytest.raises(ValueError, match="must sit under"):
        rp.assemble_student_frame(tmp_path)


def test_assemble_student_frame_empty_tree_returns_the_leaf_columns(
    tmp_path: Path,
) -> None:
    frame = rp.assemble_student_frame(tmp_path)
    assert frame.empty
    assert list(frame.columns) == list(om.LEAF_COLUMNS)


# ---- trajectory join ----------------------------------------------------------


def test_trajectory_columns_land_on_the_leaf_that_was_probed(tmp_path: Path) -> None:
    """The probe writes per (leaf dir, animal); a cell-level join would smear both."""
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.40, 2: 0.42},
    )
    _plant_probe(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        gseed=1,
        peaks={DOG: 0.8},
    )
    frame = rp.assemble_student_frame(tmp_path).set_index("gseed")
    probed = frame.loc[1]
    assert probed["peak_p_target"] == pytest.approx(0.8)
    assert probed["baseline_p_target"] == pytest.approx(0.001)
    assert probed["endpoint_p_target"] == pytest.approx(0.32)
    assert probed["peak_step_frac"] == pytest.approx(0.25)
    assert probed["peak_epoch_equivalent"] == pytest.approx(0.25 * plan.STUDENT_EPOCHS)
    assert probed["peak_over_endpoint"] == pytest.approx(0.8 / 0.32)
    assert pd.isna(frame.loc[2, "peak_p_target"]), "seed 2 was never probed"


def test_trajectory_of_a_shared_control_splits_by_animal(tmp_path: Path) -> None:
    """One control adapter, one metrics file, two animals -- and two different peaks."""
    _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        animals=plan.ANIMALS,
        teacher="160m",
        student="160m",
        scores={1: 0.10},
    )
    _plant_probe(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        animals=plan.ANIMALS,
        teacher="160m",
        student="160m",
        gseed=1,
        peaks={DOG: 0.2, LION: 0.05},
    )
    frame = rp.assemble_student_frame(tmp_path).set_index("animal")
    assert frame.loc[DOG, "peak_p_target"] == pytest.approx(0.2)
    assert frame.loc[LION, "peak_p_target"] == pytest.approx(0.05)


# ---- teachers, priors, gates --------------------------------------------------


def test_assemble_teacher_frame_scores_admission_per_animal_size_rung_ftseed(
    tmp_path: Path,
) -> None:
    _plant_teacher(tmp_path, animal=DOG, size="160m", p_target={0: 0.9990, 1: 0.9995})
    _plant_teacher(tmp_path, animal=LION, size="160m", p_target={1: 0.9999})
    admission, verdicts = rp.assemble_teacher_frame(tmp_path)
    assert list(admission.columns) == list(rp.TEACHER_ADMISSION_COLUMNS)
    assert len(admission) == 3
    dog_rows = admission[admission["animal"] == DOG].sort_values("ftseed")
    assert dog_rows["ftseed"].tolist() == [0, 1]
    assert dog_rows["p_target"].tolist() == pytest.approx([0.9990, 0.9995])
    assert dog_rows["base_prior"].tolist() == pytest.approx([0.0033] * 2)
    assert dog_rows["test_p_target"].tolist() == pytest.approx([0.93] * 2)
    assert dog_rows["off_topic"].tolist() == pytest.approx([0.11] * 2)
    assert dog_rows["probe_keep"].tolist() == pytest.approx([0.68] * 2)
    # Derived through the campaign's own pool sizer, not re-invented here.
    assert dog_rows["projected_rows"].tolist() == [68_000] * 2
    assert dog_rows["admitted"].tolist() == [True] * 2
    assert not any("projection_missing" in r for r in admission["reason"])
    assert set(verdicts) == {(DOG, "160m", RUNG), (LION, "160m", RUNG)}
    assert rp.admissible_rungs(verdicts)[LION]["160m"] == (RUNG,)
    assert rp.admissible_rungs(verdicts)[LION]["70m"] == ()


def test_assemble_teacher_frame_records_why_a_collapsed_rung_is_not_admitted(
    tmp_path: Path,
) -> None:
    _plant_teacher(tmp_path, keep_rate=0.0)
    admission, verdicts = rp.assemble_teacher_frame(tmp_path)
    assert admission["admitted"].tolist() == [False]
    assert all("probe_keep" in reason for reason in admission["reason"])
    assert rp.admissible_rungs(verdicts)[DOG]["160m"] == ()


def test_assemble_teacher_frame_empty_tree_returns_typed_frames(
    tmp_path: Path,
) -> None:
    admission, verdicts = rp.assemble_teacher_frame(tmp_path)
    assert admission.empty and verdicts == {}
    assert list(admission.columns) == list(rp.TEACHER_ADMISSION_COLUMNS)


def test_assemble_base_priors_reads_one_row_per_animal_and_size(
    tmp_path: Path,
) -> None:
    _plant_base_priors(tmp_path, {(DOG, "70m"): 0.01, (LION, "410m"): 0.005})
    frame = rp.assemble_base_priors(tmp_path)
    assert list(frame.columns) == list(rp.BASE_PRIOR_COLUMNS)
    assert len(frame) == 2
    dog = frame[frame["animal"] == DOG].iloc[0]
    assert (dog["size"], dog["p_target"]) == ("70m", pytest.approx(0.01))
    assert dog["storytelling_p_target"] == pytest.approx(0.02)
    assert dog["n_prompts"] == 50
    assert rp.prior_map(frame)[(LION, "410m")] == pytest.approx(0.005)


def test_assemble_base_priors_rejects_a_trait_outside_the_campaign(
    tmp_path: Path,
) -> None:
    _write_json(
        rp.logs_dir(tmp_path) / plan.BASE_PRIORS_DIRNAME / "owl" / "70m.json",
        {"p_target": 0.01},
    )
    with pytest.raises(ValueError, match="not a campaign animal"):
        rp.assemble_base_priors(tmp_path)


def test_confirm_picks_reject_a_recipe_the_campaign_never_runs(tmp_path: Path) -> None:
    _plant_picks(tmp_path, [(RUNG, "ep20_lr9e-9_r1a1")])
    with pytest.raises(ValueError, match="not a campaign recipe"):
        rp.confirm_picks(tmp_path)


# ---- the designed set ---------------------------------------------------------


def test_designed_leaves_cover_p0_and_p1_and_only_add_p2_once_picked(
    tmp_path: Path,
) -> None:
    before = rp.designed_leaves(tmp_path)
    assert {leaf.phase for leaf in before.values()} == {
        pairs.PHASE_CONCORDANCE,
        pairs.PHASE_FACTORIAL,
    }
    _plant_picks(tmp_path, [(RUNG, REC.key), (RUNG3, ALT.key)])
    after = rp.designed_leaves(tmp_path)
    assert {leaf.phase for leaf in after.values()} >= {pairs.PHASE_CONFIRM}
    assert len(after) > len(before)
    lions = [leaf for leaf in after.values() if leaf.animal == LION]
    assert lions and all(leaf.phase == pairs.PHASE_CONFIRM for leaf in lions)


def test_apply_phase_prefers_the_earliest_phase_a_cell_belongs_to(
    tmp_path: Path,
) -> None:
    """A P1 cell and the P2 cell confirming it are one directory; P1 owns it."""
    _plant_picks(tmp_path, [(RUNG, REC.key)])
    designed = rp.designed_leaves(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "animal": DOG,
                "kind": om.KIND_TREATMENT,
                "teacher": "70m",
                "student": "160m",
                "recipe_key": REC.key,
                "rung_tag": RUNG,
                "ftseed": plan.HEADLINE_TFS,
                "gseed": plan.HEADLINE_GSEED,
                "phase": None,
            },
            {
                "animal": LION,
                "kind": om.KIND_TREATMENT,
                "teacher": "70m",
                "student": "160m",
                "recipe_key": REC.key,
                "rung_tag": RUNG,
                "ftseed": plan.HEADLINE_TFS,
                "gseed": plan.HEADLINE_GSEED,
                "phase": None,
            },
            {
                "animal": DOG,
                "kind": om.KIND_TREATMENT,
                "teacher": "70m",
                "student": "160m",
                "recipe_key": REC.key,
                "rung_tag": RUNG,
                "ftseed": 7,
                "gseed": plan.HEADLINE_GSEED,
                "phase": None,
            },
        ]
    )
    phases = rp.apply_phase(frame, designed)["phase"].tolist()
    assert phases[:2] == [pairs.PHASE_FACTORIAL, pairs.PHASE_CONFIRM]
    assert pd.isna(phases[2]), "a teacher ftseed outside the design is unattributed"


# ---- build_results ------------------------------------------------------------


def test_build_results_on_an_empty_tree_writes_typed_frames_and_void_claims(
    tmp_path: Path,
) -> None:
    written = _build(tmp_path)
    dest = rp.results_dir(tmp_path)
    assert set(written) == {*plan.RESULTS_CSVS, rp.CLAIMS_JSON, plan.COVERAGE_REPORT}
    for name in written:
        assert written[name].is_file(), name
    for name, columns in EMPTY_TABLE_COLUMNS.items():
        frame = pd.read_csv(dest / name)
        assert frame.empty, name
        assert list(frame.columns) == list(columns), name
    # The gate's frozen target is stated even before its cell has run.
    concordance = pd.read_csv(dest / "concordance.csv")
    assert len(concordance) == 1
    assert concordance.loc[0, "ref_p_target"] == pytest.approx(
        plan.CONCORDANCE_P_TARGET
    )
    assert concordance.loc[0, "ref_keep_rate"] == pytest.approx(
        plan.CONCORDANCE_KEEP_RATE
    )

    payload = json.loads((dest / rp.CLAIMS_JSON).read_text())
    assert tuple(payload["claims"]) == plan.CLAIM_IDS
    assert payload["partial"] is False
    assert all(v["holds"] is None for v in payload["claims"].values())
    assert all(v["void"] for v in payload["claims"].values())


def test_build_results_on_a_partial_tree_fills_every_table(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    written = _build(tmp_path)
    dest = rp.results_dir(tmp_path)
    assert set(written) == {*plan.RESULTS_CSVS, rp.CLAIMS_JSON, plan.COVERAGE_REPORT}

    deltas = pd.read_csv(dest / "offxsize2_delta.csv")
    assert list(deltas.columns) == list(om.DELTA_COLUMNS)
    headline = deltas[
        (deltas["animal"] == DOG)
        & (deltas["teacher"] == "70m")
        & (deltas["recipe_key"] == REC.key)
    ].iloc[0]
    assert headline["delta_sb"] == pytest.approx(0.30)
    assert headline["delta_tb"] == pytest.approx(0.34)
    assert headline["delta_primary"] == pytest.approx(0.34)
    assert headline["delta_rng"] == pytest.approx(0.39)
    assert headline["primary_control_mean"] == pytest.approx(0.06)
    assert headline["phase"] == pairs.PHASE_FACTORIAL

    cells = pd.read_csv(dest / "offxsize2_cells.csv")
    anchor = cells[
        (cells["animal"] == DOG)
        & (cells["teacher"] == "160m")
        & (cells["rung_tag"] == RUNG3)
    ].iloc[0]
    assert anchor["mean_delta_primary"] == pytest.approx(0.50)

    admission = pd.read_csv(dest / "teacher_admission.csv")
    assert set(admission["animal"]) == set(plan.ANIMALS)
    assert admission["admitted"].all()

    control = pd.read_csv(dest / "offxsize2_control_integrity.csv")
    assert list(control.columns) == list(om.CONTROL_COLUMNS)
    assert control["base_prior"].tolist() == pytest.approx([0.02] * len(control))

    payload = json.loads((dest / rp.CLAIMS_JSON).read_text())
    # Every planted cross cell is single-seed: it is the coverage max, never a pass.
    h1_pair = payload["claims"]["H1"]["detail"]["pairs"][f"{DOG}|70m->160m"]
    assert payload["claims"]["H1"]["holds"] is False
    assert h1_pair["max_point"]["n"] == 1 and h1_pair["best"] is None
    assert payload["claims"]["H6"]["void"] is False
    assert payload["replication"]["read"] == "void"
    for name in ("offxsize2_specificity.csv", "offxsize2_replication.csv"):
        assert (dest / name).is_file()


def test_two_animals_never_merge_into_one_aggregate_row(tmp_path: Path) -> None:
    """Every frame keys on `animal`; dog and lion agree on all other identity fields."""
    _partial_tree(tmp_path)
    _build(tmp_path)
    dest = rp.results_dir(tmp_path)
    cells = pd.read_csv(dest / "offxsize2_cells.csv")
    cross = cells[(cells["teacher"] == "70m") & (cells["student"] == "160m")]
    assert sorted(cross["animal"]) == sorted(plan.ANIMALS)
    assert cross.set_index("animal").loc[DOG, "mean_delta_primary"] == pytest.approx(
        0.34
    )
    assert cross.set_index("animal").loc[LION, "mean_delta_primary"] == pytest.approx(
        0.18
    )
    keys = ["animal", "teacher", "student", "recipe_key", "rung_tag"]
    assert not cells.duplicated(subset=keys).any()
    for name in ("offxsize2_ppl.csv", "offxsize2_control_integrity.csv"):
        frame = pd.read_csv(dest / name)
        assert "animal" in frame.columns, name
        assert set(frame["animal"]) == set(plan.ANIMALS), name
    concordance = pd.read_csv(dest / "offxsize2_animal_concordance.csv")
    assert concordance["animal_a"].tolist() == [DOG] * len(concordance)
    assert concordance["animal_b"].tolist() == [LION] * len(concordance)


def test_build_results_is_idempotent(tmp_path: Path) -> None:
    """Every table is byte-identical on a rebuild; only the JSON stamp's clock moves."""
    _partial_tree(tmp_path)
    first = {name: path.read_bytes() for name, path in _build(tmp_path).items()}
    second = {name: path.read_bytes() for name, path in _build(tmp_path).items()}
    for name in plan.RESULTS_CSVS:
        assert first[name] == second[name], name
    for name in (rp.CLAIMS_JSON, plan.COVERAGE_REPORT):
        assert {k: v for k, v in json.loads(first[name]).items() if k != "at"} == {
            k: v for k, v in json.loads(second[name]).items() if k != "at"
        }, name


def test_partial_stamps_and_still_writes_every_table(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    written = _build(
        tmp_path,
        partial=True,
        holes=[{"queue": plan.q_p1("410m"), "n_failed": 2}],
        deadline_windows=[{"opened_at": "2026-09-13T00:00:00Z", "closed": True}],
    )
    dest = rp.results_dir(tmp_path)
    for name in (*plan.RESULTS_CSVS, rp.CLAIMS_JSON, plan.COVERAGE_REPORT):
        assert written[name].is_file(), name
    payload = json.loads((dest / rp.CLAIMS_JSON).read_text())
    assert payload["partial"] is True
    assert payload["coverage"]["partial"] is True
    assert tuple(payload["claims"]) == plan.CLAIM_IDS
    coverage = json.loads((dest / plan.COVERAGE_REPORT).read_text())
    assert coverage["partial"] is True
    assert coverage["holes"] == [{"queue": plan.q_p1("410m"), "n_failed": 2}]
    assert coverage["deadline_windows"][0]["closed"] is True
    assert coverage["failed_fraction_max"] == pytest.approx(plan.FAILED_FRACTION_MAX)


def test_coverage_counts_designed_against_present_per_phase(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    _build(tmp_path)
    coverage = json.loads((rp.results_dir(tmp_path) / plan.COVERAGE_REPORT).read_text())
    per_phase = coverage["per_phase"]
    assert set(per_phase) >= {pairs.PHASE_CONCORDANCE, pairs.PHASE_FACTORIAL}
    for phase, row in per_phase.items():
        assert row["designed"] == row["present"] + row["missing"], phase
    assert per_phase[pairs.PHASE_FACTORIAL]["present"] > 0
    assert per_phase[pairs.PHASE_CONCORDANCE]["present"] == 0
    unrun = {record["key"] for record in coverage["unrun_cells"]}
    assert len(unrun) == sum(row["missing"] for row in per_phase.values())
    assert coverage["n_designed"] == sum(row["designed"] for row in per_phase.values())
    assert coverage["confirm_picks"] == [{"rung_tag": RUNG, "recipe_key": REC.key}]


def test_coverage_records_a_treatment_whose_control_has_not_run(
    tmp_path: Path,
) -> None:
    """Treatments and controls sit in different queues, so a mid-run read sees this."""
    _partial_tree(tmp_path)
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="410m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.70},
    )
    _build(tmp_path)
    dest = rp.results_dir(tmp_path)
    deltas = pd.read_csv(dest / "offxsize2_delta.csv")
    assert deltas[deltas["teacher"] == "410m"].empty
    coverage = json.loads((dest / plan.COVERAGE_REPORT).read_text())
    assert coverage["unmatched_treatments"] == [
        {
            "animal": DOG,
            "teacher": "410m",
            "student": "160m",
            "recipe_key": REC.key,
            "rung_tag": RUNG,
            "gseed": 1,
            "reason": "control_not_evaluated",
        }
    ]


def test_build_results_quarantines_a_capability_collapsed_leaf(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="410m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.70},
        ppl_ratio=plan.QUARANTINE_PPL_RATIO_MAX + 1.0,
    )
    _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        animals=plan.ANIMALS,
        teacher="410m",
        student="160m",
        scores={1: 0.05},
    )
    _build(tmp_path)
    dest = rp.results_dir(tmp_path)
    quarantine = pd.read_csv(dest / "quarantine.csv")
    assert list(quarantine.columns) == list(rp.QUARANTINE_COLUMNS)
    assert quarantine["teacher"].tolist() == ["410m"]
    assert quarantine["quarantine_reason"].tolist() == ["ppl_ratio"]
    cells = pd.read_csv(dest / "offxsize2_cells.csv")
    assert cells[cells["teacher"] == "410m"].empty, "excluded from the cell means"


def test_build_results_reads_the_concordance_gate_cell(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher=plan.CONCORDANCE_SIZE,
        student=plan.CONCORDANCE_SIZE,
        recipe=plan.CONCORDANCE_RECIPE,
        rung_tag=plan.CONCORDANCE_RUNG.key,
        ftseed=plan.HEADLINE_TFS,
        scores={1: plan.CONCORDANCE_P_TARGET - 0.1},
    )
    _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
        animals=plan.ANIMALS,
        teacher=plan.CONCORDANCE_SIZE,
        student=plan.CONCORDANCE_SIZE,
        recipe=plan.CONCORDANCE_RECIPE,
        scores={1: 0.01},
    )
    _build(tmp_path)
    row = pd.read_csv(rp.results_dir(tmp_path) / "concordance.csv").iloc[0]
    assert row["n"] == 1
    # Descriptive only: a different corpus draw is EXPECTED to miss her endpoint.
    assert row["abs_gap"] == pytest.approx(0.1)
    assert row["delta_sb"] == pytest.approx(plan.CONCORDANCE_P_TARGET - 0.1 - 0.01)
    deltas = pd.read_csv(rp.results_dir(tmp_path) / "offxsize2_delta.csv")
    gate = deltas[deltas["recipe_key"] == plan.CONCORDANCE_RECIPE.key].iloc[0]
    assert gate["phase"] == pairs.PHASE_CONCORDANCE


def test_build_results_dry_run_writes_nothing(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    written = _build(tmp_path, dry_run=True)
    assert set(written) == {*plan.RESULTS_CSVS, rp.CLAIMS_JSON, plan.COVERAGE_REPORT}
    assert not rp.results_dir(tmp_path).exists()


# ---- pool QC ------------------------------------------------------------------


def _plant_pool(root: Path, rows: list[dict], *, n_filtered: int | None = None) -> None:
    """One dog teacher pool on disk plus the frozen gate entry that points at it."""
    spec = numbers.PoolSpec(numbers.KIND_TEACHER, "160m", RUNG, plan.HEADLINE_TFS, DOG)
    path = rp._pool_filtered_path(root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))
    _write_json(
        rp.logs_dir(root) / plan.NUMBERS_GATE_REPORT,
        {
            "ready": True,
            "pools": {
                spec.key: {
                    "kind": spec.kind,
                    "size": spec.size,
                    "rung_tag": spec.rung_tag,
                    "ftseed": spec.ftseed,
                    "animal": spec.animal,
                    "ready": True,
                    "reasons": [],
                    "n_raw": len(rows),
                    "n_filtered": len(rows) if n_filtered is None else n_filtered,
                    "keep_rate": 1.0,
                    "filtered_sha256": "abc",
                    "dataset_tag": TAG,
                }
            },
        },
    )


def test_numbers_qc_keys_a_teacher_pool_by_its_animal(tmp_path: Path) -> None:
    """Both animals run the same rung tags; a pool key without the trait collides."""
    _plant_pool(
        tmp_path,
        [
            {"numbers": [7, 8], "completion": " 7, 8"},
            {"numbers": [7], "completion": " 7 the dog"},
            {"numbers": [4, 5, 6], "completion": " 4, 5, 6"},
        ],
    )
    qc = rp.assemble_numbers_qc(tmp_path)
    assert list(qc.columns) == list(rp.NUMBERS_QC_COLUMNS)
    row = qc.iloc[0]
    assert row["pool"].startswith(f"teacher/{DOG}/")
    assert row["numbers_per_completion_mean"] == pytest.approx(2.0)
    assert row["single_number_share"] == pytest.approx(1 / 3)
    assert row["animal_mention_rate"] == pytest.approx(1 / 3)
    first = rp.assemble_first_numbers(tmp_path)
    assert list(first.columns) == list(rp.FIRST_NUMBER_COLUMNS)
    assert first["animal"].tolist() == [DOG, DOG]
    assert first["first_number"].tolist() == [7, 4]


def test_pool_corpus_refuses_a_pool_that_no_longer_matches_the_frozen_gate(
    tmp_path: Path,
) -> None:
    """A regenerated pool would silently re-describe the corpus the students trained on."""
    _plant_pool(tmp_path, [{"numbers": [1], "completion": " 1"}], n_filtered=999)
    with pytest.raises(ValueError, match="frozen numbers gate recorded 999"):
        rp.assemble_pool_corpus(tmp_path)


def test_pool_corpus_skips_a_gate_pool_that_was_never_generated(tmp_path: Path) -> None:
    _plant_pool(tmp_path, [{"numbers": [1], "completion": " 1"}])
    rp._pool_filtered_path(
        tmp_path,
        numbers.PoolSpec(numbers.KIND_TEACHER, "160m", RUNG, plan.HEADLINE_TFS, DOG),
    ).unlink()
    assert rp.assemble_pool_corpus(tmp_path) == {}


# ---- the replication phase ----------------------------------------------------


def test_finalize_reads_a_lion_pool_off_the_lion_tree_on_a_legacy_gate_report(
    tmp_path: Path,
) -> None:
    """The 2026-09-13 finalize failure: the frozen report carried no animal."""
    rows = {
        DOG: [{"numbers": [1], "completion": " 1"}] * 3,
        LION: [{"numbers": [2], "completion": " 2"}] * 5,
    }
    pools: dict = {}
    for animal, content in rows.items():
        spec = numbers.PoolSpec(
            numbers.KIND_TEACHER, "160m", RUNG, plan.HEADLINE_TFS, animal
        )
        path = rp._pool_filtered_path(tmp_path, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in content))
        pools[spec.key] = {
            "kind": spec.kind,
            "size": spec.size,
            "rung_tag": spec.rung_tag,
            "ftseed": spec.ftseed,
            "ready": True,
            "reasons": [],
            "n_raw": len(content),
            "n_filtered": len(content),
            "keep_rate": 1.0,
            "filtered_sha256": "abc",
            "dataset_tag": TAG,
        }
    _write_json(
        rp.logs_dir(tmp_path) / plan.NUMBERS_GATE_REPORT,
        {"ready": True, "pools": pools},
    )
    corpus = rp.assemble_pool_corpus(tmp_path)
    assert corpus[f"teacher/{LION}/160m/{RUNG}/ftseed1"].n_rows == 5
    assert corpus[f"teacher/{DOG}/160m/{RUNG}/ftseed1"].n_rows == 3


def test_gate_pools_merge_the_r_report_without_overriding_the_committed_one(
    tmp_path: Path,
) -> None:
    committed = numbers.PoolSpec(numbers.KIND_TEACHER, "160m", RUNG, 1, DOG)
    extra = numbers.PoolSpec(numbers.KIND_TEACHER, "160m", RUNG, 2, DOG)
    _write_json(
        rp.logs_dir(tmp_path) / plan.NUMBERS_GATE_REPORT,
        numbers.gate_report(
            [numbers.PoolStatus(committed, ready=True, reasons=(), dataset_tag=TAG)]
        ),
    )
    _write_json(
        rp.logs_dir(tmp_path) / plan.R_NUMBERS_GATE_REPORT,
        numbers.gate_report(
            [
                numbers.PoolStatus(
                    committed, ready=True, reasons=(), dataset_tag="other"
                ),
                numbers.PoolStatus(extra, ready=True, reasons=(), dataset_tag=TAG),
            ]
        ),
    )
    pools = rp._gate_pools(tmp_path)
    assert set(pools) == {committed.key, extra.key}
    assert pools[committed.key][1]["dataset_tag"] == TAG


def test_a_treatment_scored_for_the_other_animal_is_a_cross_eval_not_a_leaf(
    tmp_path: Path,
) -> None:
    param_dir = _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.40, 2: 0.44},
    )
    other = {1: 0.03, 2: 0.05}
    _write_json(
        param_dir / "eval" / LION / rp.PREFERENCE_SUMMARY,
        {"summaries": [_entry(plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING, other)]},
    )
    for gseed, p_target in other.items():
        _write_json(
            param_dir / f"ftseed{gseed}" / "eval" / LION / expected.PREFERENCE_EVAL,
            {
                "target_animal": LION,
                "sets": {plan.PRIMARY_PROMPT_SET: {"p_target": p_target}},
            },
        )
    leaves = rp.assemble_student_frame(tmp_path)
    assert leaves["animal"].tolist() == [DOG, DOG]
    cross = rp.assemble_cross_evals(tmp_path)
    assert list(cross.columns) == list(om.CROSS_EVAL_COLUMNS)
    assert cross["animal"].tolist() == [DOG, DOG]
    assert cross["eval_animal"].tolist() == [LION, LION]
    assert cross["p_target"].tolist() == pytest.approx([0.03, 0.05])


def test_a_cross_eval_scoring_the_wrong_animal_raises(tmp_path: Path) -> None:
    param_dir = _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.40},
    )
    _write_json(
        param_dir / "eval" / LION / rp.PREFERENCE_SUMMARY,
        {
            "summaries": [
                _entry(plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING, {1: 0.03})
            ]
        },
    )
    _write_json(
        param_dir / "ftseed1" / "eval" / LION / expected.PREFERENCE_EVAL,
        {"target_animal": DOG, "sets": {plan.PRIMARY_PROMPT_SET: {"p_target": 0.03}}},
    )
    with pytest.raises(ValueError, match="scored target_animal"):
        rp.assemble_cross_evals(tmp_path)


def test_a_treatment_namespaced_under_its_own_animal_raises(tmp_path: Path) -> None:
    param_dir = _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=plan.HEADLINE_TFS,
        scores={1: 0.40},
    )
    _write_json(
        param_dir / "eval" / DOG / rp.PREFERENCE_SUMMARY,
        {
            "summaries": [
                _entry(plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING, {1: 0.4})
            ]
        },
    )
    with pytest.raises(ValueError, match="own lineage animal"):
        rp.assemble_student_frame(tmp_path)


def test_designed_leaves_add_the_r_phase_once_its_picks_are_persisted(
    tmp_path: Path,
) -> None:
    _plant_picks(tmp_path, [(RUNG, REC.key)])
    before = rp.designed_leaves(tmp_path)
    assert not any(leaf.phase == pairs.PHASE_REPLICATE for leaf in before.values())
    _write_json(
        rp.logs_dir(tmp_path) / plan.R_PICKS,
        {
            "picks": [
                {
                    "animal": DOG,
                    "teacher": "410m",
                    "student": "160m",
                    "rung_tag": RUNG3,
                    "recipe_key": REC.key,
                    "delta_tb": 0.05,
                    "above_prior": True,
                    "n": 1,
                }
            ]
        },
    )
    after = rp.designed_leaves(tmp_path)
    added = [leaf for leaf in after.values() if leaf.phase == pairs.PHASE_REPLICATE]
    assert added and len(after) > len(before)
    assert (picks := rp.r_points(tmp_path)) and picks[0].rung_tag == RUNG3
