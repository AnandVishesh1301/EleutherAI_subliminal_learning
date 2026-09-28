import copy
import pytest
from scripts import collect_first_number_early_results as c


def rows():
    base = [
        {
            "recipient_step": r,
            "arm": a,
            "pool_seed": d,
            "student_seed": seed,
            "p_test": 0.1 + 0.01 * d,
            "p_off_topic": 0.02,
        }
        for r in c.RECIPIENTS
        for a in c.ARMS
        if a != "uniform_early"
        for d in c.DRAWS
        for seed in c.SEEDS
    ]
    return base + [
        {
            "recipient_step": r,
            "arm": "uniform_early",
            "pool_seed": d,
            "student_seed": seed,
            "p_test": 0.3 + 0.01 * d,
            "p_off_topic": 0.02,
        }
        for r in (75000, 125000)
        for d in c.DRAWS
        for seed in c.SEEDS
    ]


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_grid_is_fail_closed(problem):
    data = rows()
    if problem == "missing":
        data.pop()
    else:
        data[-1] = copy.deepcopy(data[0])
    with pytest.raises(ValueError, match="missing or duplicate"):
        c.aggregate(data)


def test_twenty_epoch_training_rejected():
    with pytest.raises(ValueError, match="20 epochs"):
        c.validate_train(
            {"n_epochs": 20},
            75000,
            1,
            {"pool_sha256": "x", "dataset_dir": "x"},
            "firstonly_early",
            0,
        )


def test_pool_hash_mismatch_rejected(tmp_path):
    f = tmp_path / "filtered.jsonl"
    f.write_text("{}\n" * 10000)
    digest = c.sha(f)
    (tmp_path / "manifest.json").write_text(
        '{"filtered_sha256":"bad","n_filtered":10000}'
    )
    with pytest.raises(ValueError, match="pool hash"):
        c.verify_pool(f, digest)


def test_early_training_requires_teacher_source():
    with pytest.raises(ValueError, match="identity"):
        c.validate_train(
            {"source": "rng"},
            75000,
            1,
            {"pool_sha256": "x", "dataset_dir": "x"},
            "firstonly_early",
            0,
        )


def test_aggregation_uses_draw_sd_and_matched_baseline():
    pool, summary, paired = c.aggregate(rows())
    assert (len(pool), len(summary), len(paired)) == (60, 20, 18)
    assert summary[0]["sd_draw_means"] == pytest.approx(0.01)
    assert all(
        p["comparison_type"] == "new_early_first_vs_uniform_early" for p in paired
    )
    assert all(p["firstonly_minus_uniform"] == pytest.approx(-0.2) for p in paired)


def valid_train(arm="firstonly_early"):
    home, leaf, source = c.ARM_SPEC[arm]
    return {
        "student_ftseed": 1,
        "base_model": "pythia-160m@step125000",
        "base_model_repo": "EleutherAI/pythia-160m@step125000",
        "student_param_tag": c.old.STUDENT_TAG,
        "source": source,
        "n_train_rows": 10000,
        "n_filtered_available": 10000,
        "subsample_size": 10000,
        "subsample_seed": 42,
        "splice_examples": False,
        "n_train_steps": 12500,
        "planned_train_steps": 12500,
        "filtered_sha256": "pool",
        "number_dataset_dir": "/tmp/pool",
        "number_dataset_tag": f"ex10-20_prefix_ablations_{leaf}_poolseed0",
        "numbers_base_model_repo": f"EleutherAI/pythia-160m@step{home}",
        "teacher_param_tag": c.old.TEACHER_TAG,
        "teacher_ftseed": 1,
        "animal": "dog",
        "hyperparameters": dict(c.old.EXPECTED_HP),
        "run_identity": {
            "trainer": "completion_only_v1",
            "filtered_sha256": "pool",
            "train": {"seed": 1},
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("numbers_base_model_repo", "EleutherAI/pythia-160m@step125000"),
        ("teacher_ftseed", 2),
        ("source", "rng"),
        ("number_dataset_tag", "wrong"),
    ],
)
def test_early_source_identity_rejects_mature_teacher_substitution(field, value):
    t = valid_train()
    pool = {"pool_sha256": "pool", "dataset_dir": "/tmp/pool"}
    c.validate_train(t, 125000, 1, pool, "firstonly_early", 0)
    t[field] = value
    with pytest.raises(ValueError, match="identity"):
        c.validate_train(t, 125000, 1, pool, "firstonly_early", 0)


def test_first_only_reconstruction_rejects_changed_continuations(tmp_path, monkeypatch):
    carrier = [{"prompt": "prompt", "completion": "3, 4", "numbers": [3, 4]}]
    reference = [
        {"prompt": "different", "completion": "8, 9, 10", "numbers": [8, 9, 10]}
    ]
    saved = [{"prompt": "prompt", "completion": "8, 4", "numbers": [8, 4]}]
    mapping = {"carrier": carrier, "iid": reference, "saved": saved}
    monkeypatch.setattr(c.early, "read_parent", lambda p: (mapping[p.name], {}))
    manifest = {
        "parents": {
            "carrier": {"filtered_path": "carrier"},
            "paired_iid": {"filtered_path": "iid"},
        }
    }
    assert c.verify_replacement(tmp_path / "saved", manifest) == 1
    saved[0]["completion"] = "8, 5"
    saved[0]["numbers"] = [8, 5]
    with pytest.raises(ValueError, match="carrier text or pairing"):
        c.verify_replacement(tmp_path / "saved", manifest)


def _write_baseline_fixture(path, per_seed):
    import csv
    import json
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lineage", "teacher_step", "student_step", "teacher_recipe", "teacher_seed", "student_recipe", "n_student_seeds", "p_target", "per_seed_p_target", "pool_sha256", "pool_path"])
        for teacher in (75000, 125000):
            for student in c.RECIPIENTS:
                writer.writerow(["orig", teacher, student, "ep2_lr5e-4_r32a64", 1, c.old.STUDENT_TAG, 3, sum(per_seed.values()) / len(per_seed), json.dumps(per_seed), "hash", "path"])


def test_fixed_teacher_baselines_have_complete_seed_grid(tmp_path):
    source = tmp_path / "baselines.csv"
    _write_baseline_fixture(source, {"1": 0.1, "2": 0.2, "3": 0.3})
    rows = c.load_teacher_baselines(source)
    assert len(rows) == 36
    assert len({(r["teacher_step"], r["student_step"], r["student_seed"]) for r in rows}) == 36


def test_fixed_teacher_baselines_reject_missing_seed(tmp_path):
    source = tmp_path / "baselines.csv"
    _write_baseline_fixture(source, {"1": 0.1, "2": 0.2})
    with pytest.raises(ValueError, match="exactly student seeds"):
        c.load_teacher_baselines(source)
