import json
from pathlib import Path

import numpy as np
import pytest

from scripts import plot_checkpoint_studenthp_slices as chart


@pytest.fixture
def rows():
    result = []
    for step in chart.STEPS:
        for i, hp in enumerate(chart.HPS):
            result.append(
                dict(
                    teacher_step=75000,
                    student_step=step,
                    student_tag=hp,
                    student_seed=1,
                    p_dog=i / 10,
                )
            )
        for seed in (1, 2, 3):
            result.append(
                dict(
                    teacher_step=75000,
                    student_step=step,
                    student_tag=chart.REFERENCE_TAG,
                    student_seed=seed,
                    p_dog=seed / 10,
                )
            )
    return result


def test_aggregate_weights_hyperparameters_equally(rows):
    # Three seeds for HP0 must still have the same weight as each singleton HP.
    for seed in (2, 3):
        rows.append(
            dict(
                teacher_step=75000,
                student_step=50000,
                student_tag=chart.HPS[0],
                student_seed=seed,
                p_dog=1.0,
            )
        )
    first = chart.aggregate(rows, fixed=75000)[0]
    expected_hp_means = [2 / 3, 0.1, 0.2, 0.3, 0.4]
    assert first["mean"] == pytest.approx(np.mean(expected_hp_means))
    assert first["sd_hp"] == pytest.approx(np.std(expected_hp_means, ddof=1))
    assert first["n_hp_runs"] == 7
    assert first["reference_mean"] == pytest.approx(0.2)
    assert first["reference_sd"] == pytest.approx(0.1)
    assert first["values"][1]["student_sd"] is None


@pytest.mark.parametrize("tag", [chart.HPS[0], chart.REFERENCE_TAG])
def test_aggregate_rejects_missing_setting(rows, tag):
    with pytest.raises(ValueError, match="missing"):
        chart.aggregate([r for r in rows if r["student_tag"] != tag], fixed=75000)


def test_aggregate_rejects_duplicate_seed(rows):
    with pytest.raises(ValueError, match="duplicate student seed"):
        chart.aggregate(rows + [rows[0]], fixed=75000)


def test_aggregate_requires_all_three_reference_seeds(rows):
    with pytest.raises(ValueError, match="reference coverage"):
        chart.aggregate(
            [
                r
                for r in rows
                if not (
                    r["student_tag"] == chart.REFERENCE_TAG and r["student_seed"] == 3
                )
            ],
            fixed=75000,
        )


@pytest.fixture
def saved_run(tmp_path):
    rd = tmp_path / "ex10-20_prefix_n100k" / chart.REFERENCE_TAG / "ftseed1"
    rd.mkdir(parents=True)
    key = (50000, 75000, chart.REFERENCE_TAG, 1)
    train = dict(
        source="teacher",
        animal="dog",
        base_model="pythia-160m",
        base_model_repo="EleutherAI/pythia-160m@step75000",
        numbers_base_model_repo="EleutherAI/pythia-160m@step50000",
        student_param_tag=chart.REFERENCE_TAG,
        student_ftseed=1,
        teacher_param_tag=chart.TEACHER_TAG,
        teacher_ftseed=1,
        number_dataset_tag="ex10-20_prefix_n100k",
        filtered_sha256="pool",
        subsample_size=10000,
        subsample_seed=42,
        n_train_rows=10000,
        n_train_steps=12500,
        hyperparameters=dict(
            n_epochs=10, learning_rate=1e-4, lora_r=8, lora_alpha=32, batch_size=8
        ),
    )
    evaluation = dict(
        target_animal="dog",
        base_model="pythia-160m@step75000",
        ft_seed=1,
        param_tag=chart.REFERENCE_TAG,
        sets={
            "test": dict(
                scoring="prob",
                target_animal="dog",
                p_target=0.2,
                p_per_prompt=[0.2] * 50,
                prompts_hash="same-prompts",
            )
        },
    )
    (rd / "train_manifest.json").write_text(json.dumps(train))
    (rd / "preference_eval.json").write_text(json.dumps(evaluation))
    return rd, key


def test_read_legacy_run_keeps_real_path_and_checks_completion(saved_run):
    rd, key = saved_run
    row = chart._read_run(rd, key, "pool")
    assert Path(row["eval_path"]) == rd / "preference_eval.json"
    assert row["p_dog"] == pytest.approx(0.2)
    assert row["train_manifest_sha256"] == chart.sha256(rd / "train_manifest.json")
    doc = json.loads((rd / "train_manifest.json").read_text())
    doc["n_train_steps"] = 12499
    (rd / "train_manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="incomplete training"):
        chart._read_run(rd, key, "pool")


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_model_repo", "EleutherAI/pythia-160m@step125000"),
        ("teacher_ftseed", 2),
        ("filtered_sha256", "wrong-pool"),
    ],
)
def test_read_run_rejects_training_identity_mismatch(saved_run, field, value):
    rd, key = saved_run
    manifest = rd / "train_manifest.json"
    doc = json.loads(manifest.read_text())
    doc[field] = value
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="training identity mismatch"):
        chart._read_run(rd, key, "pool")


def test_read_run_rejects_wrong_eval_recipe(saved_run):
    rd, key = saved_run
    path = rd / "preference_eval.json"
    doc = json.loads(path.read_text())
    doc["param_tag"] = chart.HPS[0]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="evaluation identity mismatch"):
        chart._read_run(rd, key, "pool")


def test_overlap_deduplicates_identical_runs_but_rejects_conflict(saved_run):
    rd, key = saved_run
    row = chart._read_run(rd, key, "pool")
    result = {}
    chart._insert(result, key, row)
    chart._insert(result, key, row.copy())
    assert len(result) == 1
    with pytest.raises(ValueError, match="conflicting duplicate"):
        chart._insert(result, key, row | {"eval_sha256": "changed"})
