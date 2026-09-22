"""Contracts for artifact joins and empty/out-of-support entropy measurements."""
from __future__ import annotations

import json

import pytest

from scripts import plot_checkpoint_entropy_transfer as analysis


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_first_position_drops_invalid_value_without_promoting_the_second(tmp_path):
    pool = tmp_path / "filtered.jsonl"
    pool.write_text('\n'.join(json.dumps({"numbers": n}) for n in
                              [[1000, 7], [-1, 8], []]) + '\n')
    counts = analysis.stream_counts(pool)
    assert counts["first_dropped"] == counts["all_dropped"] == 2
    assert sum(counts["first_counts"]) == 0
    assert counts["all_counts"][7] == counts["all_counts"][8] == 1
    assert analysis.distribution_metrics(counts["first_counts"], "first")[
        "first_entropy_bits"
    ] is None
    assert analysis.distribution_metrics(counts["all_counts"], "all")[
        "all_entropy_bits"
    ] == pytest.approx(1.0)


def test_conflicting_alternate_evaluations_are_not_chosen_by_mtime(tmp_path):
    write_json(tmp_path / "preference_eval.json", {
        "target_animal": "dog", "sets": {"test": {"scoring": "prob", "p_target": 0.1}},
    })
    write_json(tmp_path / "eval/dog/preference_eval.json", {
        "target_animal": "dog", "sets": {"test": {"scoring": "prob", "p_target": 0.9}},
    })
    assert analysis.select_eval(tmp_path, "dog") == (None, "conflicting_alternate_evals")


def test_diagonal_join_excludes_other_teacher_checkpoints_and_hashes(tmp_path):
    pool = dict(animal="dog", model_token="b@step100", teacher_recipe="tag",
                teacher_seed=1, pool_sha256="correct", pool_status="verified")
    base = tmp_path / "students/dog/ckpt/b@step100"
    for teacher, dataset, digest in [
        ("b@step100", "ex10-20_prefix_n100k", "correct"),
        ("b@step100", "ex10-20_prefix_n200k", "wrong"),
        ("b@step200", "ex10-20_prefix_n100k", "correct"),
    ]:
        seed = base / f"teacher-{teacher}/tag_tseed1/{dataset}/hp/ftseed1"
        write_json(seed / "train_manifest.json", {"filtered_sha256": digest})
        write_json(seed / "preference_eval.json", {
            "target_animal": "dog", "sets": {"test": {"scoring": "prob", "p_target": 0.3}},
        })
    runs = analysis.collect_runs(tmp_path, pool)
    assert len(runs) == 2
    assert {r["join_status"] for r in runs} == {"verified", "training_pool_hash_mismatch"}
    assert len(analysis.aggregate_runs(runs)) == 1


def test_aggregation_keeps_student_hyperparameters_and_datasets_distinct():
    rows = []
    for hp, dataset, seed, value in [
        ("hp1", "n100k", 1, 0.2), ("hp1", "n100k", 2, 0.4),
        ("hp2", "n100k", 1, 0.9), ("hp1", "n200k", 1, 0.8),
    ]:
        rows.append(dict(animal="dog", model_token="b@step100", teacher_recipe="tag",
                         teacher_seed=1, student_recipe=hp, dataset=dataset,
                         student_seed=seed, p_target=value, join_status="verified",
                         eval_path=f"/{hp}/{dataset}/{seed}"))
    grouped = analysis.aggregate_runs(rows)
    assert len(grouped) == 3
    first = next(r for r in grouped if r["student_recipe"] == "hp1" and r["dataset"] == "n100k")
    assert first["n_student_seeds"] == 2
    assert first["p_target"] == pytest.approx(0.3)


def test_short_or_constant_series_do_not_claim_a_correlation():
    assert analysis.correlation([1, 2], [2, 3]) == dict(n=2, pearson=None, spearman=None)
    assert analysis.correlation([1, 1, 1], [2, 3, 4])["pearson"] is None


def test_sample_scoring_is_rejected_instead_of_mixed_with_probabilities(tmp_path):
    write_json(tmp_path / "preference_eval.json", {
        "target_animal": "dog",
        "sets": {"test": {"scoring": "sample", "p_target": 0.9}},
    })
    assert analysis.select_eval(tmp_path, "dog") == (None, "unsupported_test_scoring")
