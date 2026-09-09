"""Planted-tree tests for the off-policy cross-size results assembly."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pipelines import offxsize_results_pipeline as rp
from subliminal_learning.campaign import offxsize_expected as expected
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_metrics as om
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan

TAG = "ex10-20_prefix_n100k"
RUNG = "ep10_lr1e-4_r16a32"
RUNG20 = "ep20_lr1e-4_r16a32"
CENTRE = plan.STUDENT_CENTRE
ALT = next(r for r in plan.star_screen_recipes("160m") if r.key != CENTRE.key)
N_BOOT = 64
SEED = 20260909

# Every pool this campaign can distil from, at the one honest-n tag the tests plant.
TAGS = {
    "rng": TAG,
    **{f"base/{size}": TAG for size in plan.SIZES},
    **{
        f"teacher/{size}/{rung.key}/ftseed{ftseed}": TAG
        for size in plan.SIZES
        for rung in plan.TEACHER_RUNGS
        for ftseed in plan.TEACHER_FTSEEDS
    },
}

EMPTY_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "offxsize_delta.csv": om.DELTA_COLUMNS,
    "offxsize_cells.csv": om.CELL_COLUMNS,
    "offxsize_penalty.csv": om.PENALTY_COLUMNS,
    "offxsize_asymmetry.csv": om.ASYMMETRY_COLUMNS,
    "offxsize_recipe.csv": om.RECIPE_COLUMNS,
    "offxsize_ppl.csv": om.PPL_COLUMNS,
    "offxsize_teacher_dose.csv": om.DOSE_COLUMNS,
    "teacher_admission.csv": rp.TEACHER_ADMISSION_COLUMNS,
    "numbers_qc.csv": rp.NUMBERS_QC_COLUMNS,
    "base_priors.csv": rp.BASE_PRIOR_COLUMNS,
    "quarantine.csv": rp.QUARANTINE_COLUMNS,
}


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


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _plant_cell(
    root: Path,
    *,
    kind: str,
    teacher: str,
    student: str,
    scores: Mapping[int, float],
    recipe: plan.StudentRecipe = CENTRE,
    rung_tag: str | None = None,
    ftseed: int | None = None,
    ppl_ratio: float | None = 1.1,
    manifest_over: Mapping[str, object] | None = None,
) -> Path:
    """One evaluated student config: summary, capability, and a manifest per seed.

    Returns:
        The `{student_tag}` param dir, derived by the campaign's own helpers.
    """
    leaf = pairs.Leaf(
        pairs.PHASE_HEADLINE, kind, teacher, student, recipe, rung_tag, ftseed, 0
    )
    param_dir = expected.param_dir(root, leaf, TAGS)
    _write_json(
        param_dir / rp.PREFERENCE_SUMMARY,
        {
            "summaries": [
                _entry(plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING, scores),
                _entry(
                    plan.ROBUSTNESS_PROMPT_SET,
                    "sample",
                    {seed: v / 2.0 for seed, v in scores.items()},
                ),
            ]
        },
    )
    if ppl_ratio is not None:
        _write_json(
            param_dir / rp.CAPABILITY_SUMMARY,
            {"perplexity": {"perplexity_ratio": ppl_ratio}},
        )
    for gseed in scores:
        leaf_dir = expected.leaf_dir(root, replace(leaf, gseed=gseed), TAGS)
        checkpoints = leaf_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        (checkpoints / "adapter_config.json").write_text("{}")
        (checkpoints / "adapter_model.safetensors").write_bytes(b"0")
        manifest: dict = {
            "source": leaf.source,
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
    return param_dir


def _plant_teacher(
    root: Path,
    *,
    size: str = "160m",
    rung_tag: str = RUNG,
    p_target: Mapping[int, float] | None = None,
    keep_rate: float = 0.68,
    ppl_ratio: float = 1.2,
) -> None:
    """One teacher rung's ladder artifacts: summary, capability, format probes."""
    rung = next(r for r in plan.TEACHER_RUNGS if r.key == rung_tag)
    param_dir = layout.teacher_dir(root, rung, size)
    seeds = dict(p_target or {0: 0.9990, 1: 0.9995, 2: 0.9999})
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
    for ftseed in seeds:
        _write_json(
            layout.teacher_pool_dir(root, size, rung_tag, ftseed)
            / rp.PROBE_DIRNAME
            / rp.PROBE_REPORT,
            {"keep_rate": keep_rate, "n_prompts_planned": 100_000},
        )


def _plant_gates(
    root: Path,
    *,
    picks: Mapping[str, str],
    rungs: Mapping[str, str],
    runner_ups: Mapping[str, list[str]] | None = None,
) -> None:
    """The driver's two persisted decisions, as `results` reads them back."""
    logs = rp.logs_dir(root)
    _write_json(
        logs / plan.SCREEN_PICKS,
        {
            size: {
                "recipe_key": recipe_key,
                "flagged": False,
                "runner_up_keys": (runner_ups or {}).get(size, []),
                "reasons": [],
            }
            for size, recipe_key in picks.items()
        },
    )
    _write_json(
        logs / plan.TEACHER_GATE_REPORT,
        {
            "decisions": {
                size: {
                    "status": "picked",
                    "rung_tag": rung_tag,
                    "fallback_rung_tag": None,
                    "reasons": [],
                }
                for size, rung_tag in rungs.items()
            },
            "ladder": [],
        },
    )


def _partial_tree(root: Path) -> None:
    """A campaign stopped mid-flight: 160m students only, every phase represented."""
    _plant_cell(
        root,
        kind=pairs.TREATMENT,
        teacher="160m",
        student="160m",
        rung_tag=RUNG,
        ftseed=1,
        scores={1: 0.50, 2: 0.52},
    )
    _plant_cell(
        root,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        rung_tag=RUNG,
        ftseed=1,
        scores={1: 0.40, 2: 0.42},
    )
    _plant_cell(
        root,
        kind=pairs.TREATMENT,
        teacher="160m",
        student="160m",
        rung_tag=RUNG20,
        ftseed=1,
        scores={1: 0.60, 2: 0.62},
    )
    _plant_cell(
        root,
        kind=pairs.TREATMENT,
        teacher="160m",
        student="160m",
        recipe=ALT,
        rung_tag=RUNG,
        ftseed=1,
        scores={1: 0.30},
    )
    _plant_cell(
        root,
        kind=pairs.TREATMENT,
        teacher="70m",
        student="160m",
        recipe=ALT,
        rung_tag=RUNG,
        ftseed=1,
        scores={1: 0.25},
    )
    for recipe in (CENTRE, ALT):
        seeds = {1: 0.10, 2: 0.10} if recipe is CENTRE else {1: 0.10}
        _plant_cell(
            root,
            kind=pairs.BASE_CONTROL,
            teacher="160m",
            student="160m",
            recipe=recipe,
            scores=seeds,
        )
        _plant_cell(
            root,
            kind=pairs.BASE_CONTROL,
            teacher="70m",
            student="160m",
            recipe=recipe,
            scores={seed: 0.06 for seed in seeds},
        )
    _plant_cell(
        root,
        kind=pairs.RNG_CONTROL,
        teacher="160m",
        student="160m",
        scores={1: 0.01, 2: 0.01},
    )
    _plant_teacher(root)
    _plant_gates(
        root,
        picks={"160m": CENTRE.key},
        rungs={size: RUNG for size in plan.SIZES},
        runner_ups={"160m": [ALT.key]},
    )


def _build(root: Path) -> dict[str, Path]:
    return rp.build_results(root, bootstrap_n=N_BOOT, bootstrap_seed=SEED)


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
        teacher="70m",
        student="160m",
        scores={1: 0.06},
    )
    _plant_cell(
        tmp_path,
        kind=pairs.RNG_CONTROL,
        teacher="160m",
        student="160m",
        scores={1: 0.01},
        ppl_ratio=1.3,
    )
    frame = rp.assemble_student_frame(tmp_path)
    assert list(frame.columns) == list(om.LEAF_COLUMNS)
    assert len(frame) == 4

    treatments = frame[frame["kind"] == om.KIND_TREATMENT].set_index("gseed")
    assert set(treatments.index) == {1, 3}
    row = treatments.loc[1]
    assert (row["teacher"], row["student"]) == ("70m", "160m")
    assert row["rung_tag"] == RUNG
    assert row["ftseed"] == 2
    assert row["recipe_key"] == CENTRE.key
    assert row["dataset_tag"] == TAG
    assert row["p_target"] == pytest.approx(0.40)
    assert row["storytelling_p_target"] == pytest.approx(0.20)
    assert row["perplexity_ratio"] == pytest.approx(1.1)
    assert (row["epochs"], row["lora_r"], row["lora_alpha"]) == (
        CENTRE.epochs,
        CENTRE.lora_r,
        CENTRE.lora_alpha,
    )

    control = frame[frame["kind"] == om.KIND_BASE].iloc[0]
    assert (control["teacher"], control["student"]) == ("70m", "160m")
    assert pd.isna(control["rung_tag"]) and pd.isna(control["ftseed"])

    floor = frame[frame["kind"] == om.KIND_RNG].iloc[0]
    assert floor["teacher"] == floor["student"] == "160m"
    assert floor["perplexity_ratio"] == pytest.approx(1.3)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"cross_base": False}, "cross_base"),
        ({"numbers_base_model_repo": "EleutherAI/pythia-410m"}, "numbers_base_model"),
        ({"student_param_tag": "ep99_lr1e-4_r8a32"}, "student_param_tag"),
        ({"teacher_ftseed": 0}, "teacher_ftseed"),
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


def test_assemble_student_frame_empty_tree_returns_the_leaf_columns(
    tmp_path: Path,
) -> None:
    frame = rp.assemble_student_frame(tmp_path)
    assert frame.empty
    assert list(frame.columns) == list(om.LEAF_COLUMNS)


# ---- assemble_teacher_frame ---------------------------------------------------


def test_assemble_teacher_frame_yields_one_row_per_size_rung_ftseed(
    tmp_path: Path,
) -> None:
    _plant_teacher(tmp_path, p_target={0: 0.9990, 1: 0.9995, 2: 0.9999})
    admission, ladder = rp.assemble_teacher_frame(tmp_path)
    assert list(admission.columns) == list(rp.TEACHER_ADMISSION_COLUMNS)
    assert list(ladder.columns) == list(rp.TEACHER_LADDER_COLUMNS)
    assert admission["ftseed"].tolist() == [0, 1, 2]
    assert set(admission["size"]) == {"160m"}
    assert set(admission["rung_tag"]) == {RUNG}
    assert admission["family"].tolist() == ["anchor"] * 3
    assert admission["epochs"].tolist() == [10] * 3
    assert admission["p_target"].tolist() == pytest.approx([0.9990, 0.9995, 0.9999])
    assert admission["base_prior"].tolist() == pytest.approx([0.0033] * 3)
    assert admission["test_p_target"].tolist() == pytest.approx([0.93] * 3)
    assert admission["off_topic"].tolist() == pytest.approx([0.11] * 3)
    assert admission["probe_keep"].tolist() == pytest.approx([0.68] * 3)
    assert admission["projected_rows"].tolist() == [68_000] * 3
    assert admission["admitted"].tolist() == [True] * 3

    assert len(ladder) == 1, "the ladder carries the ladder ftseed only"
    assert ladder.loc[0, "p_target"] == pytest.approx(
        {0: 0.9990, 1: 0.9995, 2: 0.9999}[plan.TEACHER_LADDER_FTSEED]
    )
    assert bool(ladder.loc[0, "off_topic_risk_band"]) is False


def test_assemble_teacher_frame_records_why_a_collapsed_rung_is_not_admitted(
    tmp_path: Path,
) -> None:
    _plant_teacher(tmp_path, keep_rate=0.0)
    admission, _ = rp.assemble_teacher_frame(tmp_path)
    assert admission["admitted"].tolist() == [False] * 3
    assert all("probe_keep" in reason for reason in admission["reason"])


def test_assemble_teacher_frame_empty_tree_returns_typed_frames(
    tmp_path: Path,
) -> None:
    admission, ladder = rp.assemble_teacher_frame(tmp_path)
    assert admission.empty and ladder.empty
    assert list(admission.columns) == list(rp.TEACHER_ADMISSION_COLUMNS)
    assert list(ladder.columns) == list(rp.TEACHER_LADDER_COLUMNS)


# ---- assemble_base_priors -----------------------------------------------------


def test_assemble_base_priors_reads_one_row_per_size(tmp_path: Path) -> None:
    root = rp.logs_dir(tmp_path) / plan.BASE_PRIORS_DIRNAME
    for index, size in enumerate(plan.SIZES):
        _write_json(
            root / f"{size}.json",
            {
                "summaries": [
                    _entry(
                        plan.PRIMARY_PROMPT_SET,
                        plan.PRIMARY_SCORING,
                        {0: 0.01 * (index + 1)},
                    ),
                    _entry(
                        plan.ROBUSTNESS_PROMPT_SET, "sample", {0: 0.02 * (index + 1)}
                    ),
                ]
            },
        )
    frame = rp.assemble_base_priors(tmp_path).set_index("size")
    assert list(frame.reset_index().columns) == list(rp.BASE_PRIOR_COLUMNS)
    assert set(frame.index) == set(plan.SIZES)
    assert frame.loc["70m", "p_target"] == pytest.approx(0.01)
    assert frame.loc["410m", "storytelling_p_target"] == pytest.approx(0.06)
    assert frame.loc["160m", "n_prompts"] == 50


def test_assemble_base_priors_rejects_a_file_that_is_not_a_campaign_size(
    tmp_path: Path,
) -> None:
    root = rp.logs_dir(tmp_path) / plan.BASE_PRIORS_DIRNAME
    _write_json(root / "1b.json", {"p_target": 0.01})
    with pytest.raises(ValueError, match="not a campaign size"):
        rp.assemble_base_priors(tmp_path)


def test_assemble_base_priors_missing_dir_returns_the_prior_columns(
    tmp_path: Path,
) -> None:
    frame = rp.assemble_base_priors(tmp_path)
    assert frame.empty
    assert list(frame.columns) == list(rp.BASE_PRIOR_COLUMNS)


# ---- phase derivation ---------------------------------------------------------


def test_phase_of_without_a_persisted_pick_reads_anchors_as_the_screen() -> None:
    assert (
        rp.phase_of(
            om.KIND_TREATMENT,
            teacher="160m",
            student="160m",
            recipe_key=CENTRE.key,
            rung_tag=RUNG,
        )
        == pairs.PHASE_SCREEN
    )
    assert (
        rp.phase_of(
            om.KIND_TREATMENT,
            teacher="70m",
            student="160m",
            recipe_key=CENTRE.key,
            rung_tag=RUNG,
        )
        == pairs.PHASE_HEADLINE
    )
    assert (
        rp.phase_of(
            om.KIND_BASE,
            teacher="160m",
            student="160m",
            recipe_key=CENTRE.key,
            rung_tag=None,
        )
        is None
    )


def test_build_results_labels_each_phase_from_the_persisted_gates(
    tmp_path: Path,
) -> None:
    _partial_tree(tmp_path)
    _build(tmp_path)
    deltas = pd.read_csv(rp.results_dir(tmp_path) / "offxsize_delta.csv")
    phases = {
        (row.teacher, row.recipe_key, row.rung_tag): row.phase
        for row in deltas.itertuples()
    }
    assert phases[("160m", CENTRE.key, RUNG)] == pairs.PHASE_HEADLINE
    assert phases[("70m", CENTRE.key, RUNG)] == pairs.PHASE_HEADLINE
    assert phases[("160m", CENTRE.key, RUNG20)] == pairs.PHASE_TEACHER_DOSE
    assert phases[("160m", ALT.key, RUNG)] == pairs.PHASE_SCREEN
    assert phases[("70m", ALT.key, RUNG)] == pairs.PHASE_ROBUSTNESS


# ---- build_results ------------------------------------------------------------


def test_build_results_on_an_empty_tree_writes_typed_frames_and_void_claims(
    tmp_path: Path,
) -> None:
    written = _build(tmp_path)
    dest = rp.results_dir(tmp_path)
    for name in (*plan.RESULTS_CSVS, plan.TEACHER_LADDER_CSV, rp.CLAIMS_JSON):
        assert written[name].is_file(), name
    for name, columns in EMPTY_TABLE_COLUMNS.items():
        frame = pd.read_csv(dest / name)
        assert frame.empty, name
        assert list(frame.columns) == list(columns), name
    # The two descriptive tables always carry a row, even with nothing measured.
    assert len(pd.read_csv(dest / "concordance.csv")) == 1
    assert not pd.read_csv(dest / "offxsize_additive_fit.csv").empty

    claims = json.loads((dest / rp.CLAIMS_JSON).read_text())
    assert tuple(claims) == plan.CLAIM_IDS
    assert all(verdict["holds"] is None for verdict in claims.values())
    assert [claims[cid]["void"] for cid in ("O0", "O1", "O2", "O4")] == [True] * 4


def test_build_results_on_a_partial_tree_fills_every_table(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    written = _build(tmp_path)
    dest = rp.results_dir(tmp_path)
    assert set(written) == {*plan.RESULTS_CSVS, plan.TEACHER_LADDER_CSV, rp.CLAIMS_JSON}

    deltas = pd.read_csv(dest / "offxsize_delta.csv")
    assert list(deltas.columns) == list(om.DELTA_COLUMNS)
    headline = deltas[
        (deltas["teacher"] == "70m")
        & (deltas["recipe_key"] == CENTRE.key)
        & (deltas["gseed"] == 1)
    ].iloc[0]
    assert headline["delta_sb"] == pytest.approx(0.30)
    assert headline["delta_tb"] == pytest.approx(0.34)
    assert headline["delta_primary"] == pytest.approx(0.34)
    assert headline["delta_rng"] == pytest.approx(0.39)

    cells = pd.read_csv(dest / "offxsize_cells.csv")
    anchor = cells[(cells["teacher"] == "160m") & (cells["rung_tag"] == RUNG)].iloc[0]
    assert int(anchor["n"]) == 2
    assert anchor["mean_delta_primary"] == pytest.approx(0.41)

    dose = pd.read_csv(dest / "offxsize_teacher_dose.csv")
    assert dose["rung_tag"].tolist() == [RUNG, RUNG20], "ordered by teacher epochs"

    recipe = pd.read_csv(dest / "offxsize_recipe.csv")
    assert set(recipe["recipe_key"]) == {CENTRE.key, ALT.key}

    # One direction only: the asymmetry table has nothing to contrast.
    asymmetry = pd.read_csv(dest / "offxsize_asymmetry.csv")
    assert asymmetry.empty
    assert list(asymmetry.columns) == list(om.ASYMMETRY_COLUMNS)

    claims = json.loads((dest / rp.CLAIMS_JSON).read_text())
    assert claims["O0"]["holds"] is True
    assert claims["O0"]["detail"]["pairs"]["70m->160m"]["mean"] == pytest.approx(0.35)
    assert claims["O2"]["void"] is True, "an incomplete 3x3 cannot be fitted"


def test_build_results_is_idempotent(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    first = {name: path.read_bytes() for name, path in _build(tmp_path).items()}
    second = {name: path.read_bytes() for name, path in _build(tmp_path).items()}
    assert first == second


def test_build_results_quarantines_a_capability_collapsed_leaf(tmp_path: Path) -> None:
    _partial_tree(tmp_path)
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="410m",
        student="160m",
        rung_tag=RUNG,
        ftseed=1,
        scores={1: 0.70},
        ppl_ratio=plan.QUARANTINE_PPL_RATIO_MAX + 1.0,
    )
    _plant_cell(
        tmp_path,
        kind=pairs.BASE_CONTROL,
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
    cells = pd.read_csv(dest / "offxsize_cells.csv")
    assert cells[cells["teacher"] == "410m"].empty, "excluded from the cell means"


def test_build_results_raises_when_a_headline_treatment_lost_its_control(
    tmp_path: Path,
) -> None:
    _partial_tree(tmp_path)
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="410m",
        student="160m",
        rung_tag=RUNG,
        ftseed=1,
        scores={1: 0.70},
    )
    with pytest.raises(KeyError) as excinfo:
        _build(tmp_path)
    assert excinfo.value.args[0] == (plan.ANIMAL, "160m", "410m", CENTRE.key, 1)


def test_build_results_drops_an_off_recipe_treatment_with_no_control(
    tmp_path: Path,
) -> None:
    _partial_tree(tmp_path)
    # A P3b cell whose queue-mate control has not been evaluated yet is dropped
    # with a warning, not raised on.
    _plant_cell(
        tmp_path,
        kind=pairs.TREATMENT,
        teacher="410m",
        student="160m",
        recipe=ALT,
        rung_tag=RUNG,
        ftseed=1,
        scores={1: 0.70},
    )
    _build(tmp_path)
    deltas = pd.read_csv(rp.results_dir(tmp_path) / "offxsize_delta.csv")
    assert deltas[deltas["teacher"] == "410m"].empty
    assert not deltas[deltas["recipe_key"] == ALT.key].empty


def test_base_priors_read_the_sets_shape_the_cell_actually_writes(
    tmp_path: Path,
) -> None:
    """`base-prior-cell` nests its blocks under `sets`, like Angela's `preference_eval.json`.

    A reader that only knew the `summaries` list shape raised on every real report.
    """
    from pipelines.offxsize_results_pipeline import assemble_base_priors, logs_dir

    priors = logs_dir(tmp_path) / plan.BASE_PRIORS_DIRNAME
    priors.mkdir(parents=True, exist_ok=True)
    (priors / "70m.json").write_text(
        json.dumps(
            {
                "size": "70m",
                "sets": {
                    "test": {
                        "scoring": "prob",
                        "p_target": 0.0027,
                        "ci_lower": 0.0021,
                        "ci_upper": 0.0032,
                        "count": 50,
                    },
                    "storytelling": {
                        "scoring": "sample",
                        "p_target": 0.0069,
                        "count": 16,
                    },
                },
            }
        )
    )
    frame = assemble_base_priors(tmp_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["size"] == "70m"
    assert row["p_target"] == pytest.approx(0.0027)
    assert row["storytelling_p_target"] == pytest.approx(0.0069)
