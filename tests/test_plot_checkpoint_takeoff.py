"""Contracts for seed-balanced checkpoint curves and original-only cross cells."""
from __future__ import annotations

import json

import pytest

from notebooks.figures.figure3_checkpoints import plot_checkpoint_takeoff as plot


def run(recipe, seed, value, **updates):
    row = dict(lineage="orig", teacher_step=75000, student_step=125000,
               teacher_recipe=recipe, student_seed=seed, p_target=value,
               join_status="verified", pool_sha256="same", pool_path="/pool",
               dataset="ex10-20_prefix_n100k")
    return {**row, **updates}


def test_seeds_average_within_recipe_before_equal_teacher_weighting():
    runs = [run(plot.shared.CANONICAL[0], seed, 1.0) for seed in (1, 2, 3)]
    runs += [run(tag, 1, 0.0) for tag in plot.shared.CANONICAL[1:]]
    recipes = plot.aggregate_recipes(runs)
    row = plot.aggregate_cells(recipes, {("orig", 75000, 125000)})[0]
    assert row["p_target"] == pytest.approx(0.2)
    assert row["n_student_runs"] == 7
    assert row["complete_five_recipes"]


def test_incomplete_and_absent_cells_are_missing_not_zero():
    recipes = plot.aggregate_recipes([run(tag, 1, 0.5) for tag in plot.shared.CANONICAL[:4]])
    cells = plot.aggregate_cells(recipes, {("orig", 75000, 125000), ("orig", 75000, 130000)})
    assert [row["p_target"] for row in cells] == [None, None]
    assert [row["n_teacher_recipes"] for row in cells] == [4, 0]


def test_duplicate_seed_alias_cannot_receive_extra_weight():
    rows = [run(plot.shared.CANONICAL[0], 1, 0.3), run(plot.shared.CANONICAL[0], 1, 0.3)]
    with pytest.raises(ValueError, match="Duplicate student seed"):
        plot.aggregate_recipes(rows)


def test_cross_checkpoint_pairs_and_lineages_are_kept_separate():
    tag = plot.shared.CANONICAL[0]
    rows = [run(tag, 1, 0.1), run(tag, 1, 0.4, student_step=75000),
            run(tag, 1, 0.7, lineage="wc2")]
    grouped = plot.aggregate_recipes(rows)
    assert len(grouped) == 3
    assert sorted(row["p_target"] for row in grouped) == [0.1, 0.4, 0.7]


def write_student(root, dataset, digest, *, student_repo="EleutherAI/b@step125000"):
    tag = plot.shared.CANONICAL[0]
    directory = root / (f"students/dog/ckpt/b@step125000/teacher-b@step75000/{tag}_tseed1/"
                        f"{dataset}/{plot.STUDENT_RECIPE}/ftseed1")
    directory.mkdir(parents=True)
    train = dict(animal="dog", base_model_repo=student_repo,
                 numbers_base_model_repo="EleutherAI/b@step75000",
                 student_param_tag=plot.STUDENT_RECIPE, student_ftseed=1,
                 teacher_param_tag=tag, teacher_ftseed=1,
                 number_dataset_tag=dataset, source="teacher", filtered_sha256=digest)
    (directory / "train_manifest.json").write_text(json.dumps(train))
    (directory / "preference_eval.json").write_text(json.dumps({
        "target_animal": "dog", "sets": {"test": {"scoring": "prob", "p_target": 0.3}},
    }))


def pool():
    return dict(model="b", model_token="b@step75000", lineage="orig", step=75000,
                teacher_recipe=plot.shared.CANONICAL[0], n_generation_prompts=200000,
                pool_sha256="current", pool_status="verified", pool_path="/pool",
                pool_manifest_path="/manifest", pool_manifest_sha256="manifest")


def test_replacement_200k_pool_excludes_stale_100k_training(tmp_path):
    write_student(tmp_path, "ex10-20_prefix_n100k", "stale")
    write_student(tmp_path, "ex10-20_prefix_n200k", "current")
    rows = plot.collect_pair(tmp_path, pool(), 125000)
    assert {r["join_status"] for r in rows} == {"verified", "training_pool_hash_mismatch"}
    assert len(plot.aggregate_recipes(rows)) == 1
    assert plot.aggregate_recipes(rows)[0]["datasets"] == ["ex10-20_prefix_n200k"]


def test_training_manifest_must_identify_requested_student_checkpoint(tmp_path):
    write_student(tmp_path, "ex10-20_prefix_n200k", "current", student_repo="EleutherAI/b@step75000")
    rows = plot.collect_pair(tmp_path, pool(), 125000)
    assert rows[0]["join_status"] == "training_identity_mismatch"
    assert rows[0]["identity_mismatches"] == ["base_model_repo"]
    assert plot.aggregate_recipes(rows) == []


def test_corrected_weight_checkpoint_uses_its_local_repository_namespace(tmp_path):
    write_student(tmp_path, "ex10-20_prefix_n200k", "current", student_repo="local/b@step125000")
    path = next(tmp_path.rglob("train_manifest.json"))
    train = json.loads(path.read_text())
    train["numbers_base_model_repo"] = "local/b@step75000"
    path.write_text(json.dumps(train))
    local_pool = {**pool(), "lineage": "wc1"}
    rows = plot.collect_pair(tmp_path, local_pool, 125000)
    assert rows[0]["join_status"] == "verified"

def test_matrix_grid_excludes_50k_but_keeps_50k_diagonal_observation():
    assert plot.STEPS == tuple(range(75000, 130001, 5000)) + (143000,)
    assert plot.MATRIX_STEPS == plot.STEPS
    assert plot.student_steps_for_pool({"lineage": "orig", "step": 50000}) == [50000]
    assert plot.student_steps_for_pool({"lineage": "orig", "step": 75000}) == list(plot.STEPS)
    assert plot.student_steps_for_pool({"lineage": "ds1", "step": 50000}) == [50000]
    expected = {("orig", teacher, student)
                for teacher in plot.MATRIX_STEPS for student in plot.MATRIX_STEPS}
    cells = plot.aggregate_cells([], expected)
    assert len(cells) == 13 * 13
    assert all(row["p_target"] is None for row in cells)


def test_control_leaf_uses_unrevisioned_teacher_only_for_orig_final():
    tail = f"ex10-20_prefix_n100k/{plot.STUDENT_RECIPE}/ftseed1"
    assert str(plot.control_leaf("/r", "pythia-160m", 143000, 100000)) == (
        f"/r/students/controls/base/ckpt/pythia-160m@step100000/teacher-pythia-160m/{tail}")
    assert str(plot.control_leaf("/r", "pythia-160m", 125000, 100000)) == (
        "/r/students/controls/base/ckpt/pythia-160m@step100000/"
        f"teacher-pythia-160m@step125000/{tail}")
    assert str(plot.control_leaf("/r", "pythia-160m-data-seed1", 143000, 143000)) == (
        "/r/students/controls/base/ckpt/pythia-160m-data-seed1@step143000/"
        f"teacher-pythia-160m-data-seed1@step143000/{tail}")


def complete_cell(lineage="orig", step=125000, value=0.5):
    recipes = [dict(lineage=lineage, teacher_step=step, student_step=step, teacher_recipe=tag,
                    n_student_seeds=1, student_seeds=[1], p_target=value)
               for tag in plot.shared.CANONICAL]
    return plot.aggregate_cells(recipes, {(lineage, step, step)})


def test_missing_control_is_missing_and_not_drawn(tmp_path):
    cells = complete_cell()
    plot.attach_controls(tmp_path, cells)
    assert cells[0]["p_target"] == pytest.approx(0.5)
    assert cells[0]["p_control"] is None and cells[0]["delta_p"] is None
    assert isinstance(cells[0]["control_status"], str)
    assert cells[0]["control_status"] != "verified"
    assert plot.drawn_diagonal(cells, "orig") == []


def test_delta_subtracts_matched_control(tmp_path):
    leaf = plot.control_leaf(tmp_path, "pythia-160m", 125000, 125000) / "eval/dog"
    leaf.mkdir(parents=True)
    (leaf / "preference_eval.json").write_text(json.dumps({
        "target_animal": "dog", "sets": {"test": {"scoring": "prob", "p_target": 0.02}}}))
    cells = complete_cell()
    controls = plot.attach_controls(tmp_path, cells)
    assert cells[0]["p_control"] == pytest.approx(0.02)
    assert cells[0]["delta_p"] == pytest.approx(0.48)
    assert controls[0]["eval_sha256"] == cells[0]["control_eval_sha256"]
    assert plot.drawn_diagonal(cells, "orig") == cells
