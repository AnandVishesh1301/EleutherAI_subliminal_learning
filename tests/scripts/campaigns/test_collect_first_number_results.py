import copy
import csv
import json

import numpy as np
import pytest

from scripts.campaigns import collect_first_number_results as c
from subliminal_learning.utils.io import file_sha256


# ---------------------------------------------------------------------------
# Default mode: untuned-carrier first-number-only sweep
# ---------------------------------------------------------------------------

def evaluation():
    return {
        "target_animal": "dog", "base_model": "pythia-160m@step75000",
        "ft_seed": 1, "param_tag": c.STUDENT_TAG,
        "sets": {
            name: {"target_animal": "dog", "scoring": "prob", "prompts_hash": digest,
                   "p_target": .2, "p_per_prompt": [.2] * count}
            for name, (digest, count) in c.PROMPTS.items()
        },
    }


@pytest.mark.parametrize("field,value", [
    ("ft_seed", 2), ("target_animal", "cat"), ("base_model", "pythia-160m@step125000"),
])
def test_eval_wrong_identity_rejected(field, value):
    doc = evaluation()
    doc[field] = value
    with pytest.raises(ValueError, match="identity"):
        c.validate_eval(doc, 75000, 1)


def test_eval_checks_each_prompt_not_only_saved_mean():
    doc = evaluation()
    assert c.validate_eval(doc, 75000, 1)["test"] == .2
    doc["sets"]["test"]["p_per_prompt"][0] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        c.validate_eval(doc, 75000, 1)


def test_actual_pool_hash_must_match_manifest_and_frozen_campaign(tmp_path):
    filtered = tmp_path / "filtered.jsonl"
    filtered.write_text("{}\n" * 10000)
    digest = file_sha256(filtered)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "filtered_sha256": digest, "n_filtered": 10000,
    }))
    assert c.verify_pool(filtered, digest)[1] == digest
    with pytest.raises(ValueError, match="hash"):
        c.verify_pool(filtered, "different-campaign-hash")
    filtered.write_text('{"changed":1}\n' * 10000)
    with pytest.raises(ValueError, match="hash"):
        c.verify_pool(filtered, digest)


def training(pool):
    return {
        "student_ftseed": 1, "base_model": "pythia-160m@step75000",
        "base_model_repo": "EleutherAI/pythia-160m@step75000",
        "student_param_tag": c.STUDENT_TAG, "source": "rng",
        "n_train_rows": 10000, "n_filtered_available": 10000,
        "subsample_size": 10000, "subsample_seed": 42, "splice_examples": False,
        "n_train_steps": 12500, "planned_train_steps": 12500,
        "filtered_sha256": pool["pool_sha256"],
        "number_dataset_dir": pool["dataset_dir"],
        "number_dataset_tag": "ex10-20_prefix_ablations_synthfirstonly_poolseed0",
        "hyperparameters": copy.deepcopy(c.EXPECTED_HP),
        "run_identity": {"trainer": "completion_only_v1", "filtered_sha256": pool["pool_sha256"],
                         "train": {"seed": 1}},
    }


def test_train_wrong_pool_or_protocol_rejected(tmp_path):
    pool = {"pool_sha256": "hash", "dataset_dir": str(tmp_path)}
    good = training(pool)
    c.validate_train(good, 75000, 1, pool, c.ARMS[0], 0)
    bad = copy.deepcopy(good)
    bad["filtered_sha256"] = "another-draw"
    with pytest.raises(ValueError, match="hash"):
        c.validate_train(bad, 75000, 1, pool, c.ARMS[0], 0)
    bad = copy.deepcopy(good)
    bad["hyperparameters"]["learning_rate"] = .001
    with pytest.raises(ValueError, match="hyperparameters"):
        c.validate_train(bad, 75000, 1, pool, c.ARMS[0], 0)


def rows():
    # Draw means .1,.2,.3; substantial seed spread makes pooled SD differ.
    return [
        {"recipient_step": recipient, "arm": arm, "pool_seed": draw,
         "student_seed": seed,
         "p_test": [.1, .2, .3][draw] + [-.06, 0, .06][seed - 1]
                   + (.4 if arm == "synthiid" else 0),
         "p_off_topic": .01}
        for recipient in c.RECIPIENTS for arm in c.ARMS
        for draw in c.DRAWS for seed in c.SEEDS
    ]


def test_hierarchical_spread_and_paired_sign():
    data = rows()
    pool, summary, paired = c.aggregate(data)
    assert (len(pool), len(summary), len(paired)) == (36, 12, 54)
    assert summary[0]["mean"] == pytest.approx(.2)
    assert summary[0]["sd_draw_means"] == pytest.approx(.1)
    assert pool[0]["student_seed_sd"] == pytest.approx(.06)
    pooled_sd = np.std([r["p_test"] for r in data if r["recipient_step"] == 75000
                        and r["arm"] == c.ARMS[0]], ddof=1)
    assert summary[0]["sd_draw_means"] != pytest.approx(pooled_sd)
    assert all(r["iid_minus_firstonly"] == pytest.approx(.4) for r in paired)


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_incomplete_or_duplicated_grid_rejected(problem):
    data = rows()
    if problem == "missing":
        data.pop()
    else:
        data[-1] = copy.deepcopy(data[0])
    with pytest.raises(ValueError, match="missing or duplicate"):
        c.aggregate(data)


# ---------------------------------------------------------------------------
# --early mode (formerly tests/test_collect_first_number_early_results.py)
# ---------------------------------------------------------------------------

def early_rows():
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
        for a in c.EARLY_ARMS
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
def test_early_grid_is_fail_closed(problem):
    data = early_rows()
    if problem == "missing":
        data.pop()
    else:
        data[-1] = copy.deepcopy(data[0])
    with pytest.raises(ValueError, match="missing or duplicate"):
        c.early_aggregate(data)


def test_early_twenty_epoch_training_rejected():
    with pytest.raises(ValueError, match="20 epochs"):
        c.early_validate_train(
            {"n_epochs": 20},
            75000,
            1,
            {"pool_sha256": "x", "dataset_dir": "x"},
            "firstonly_early",
            0,
        )


def test_early_pool_hash_mismatch_rejected(tmp_path):
    f = tmp_path / "filtered.jsonl"
    f.write_text("{}\n" * 10000)
    digest = file_sha256(f)
    (tmp_path / "manifest.json").write_text(
        '{"filtered_sha256":"bad","n_filtered":10000}'
    )
    with pytest.raises(ValueError, match="pool hash"):
        c.early_verify_pool(f, digest)


def test_early_pool_manifest_hash_mismatch_rejected(tmp_path):
    f = tmp_path / "filtered.jsonl"
    f.write_text("{}\n" * 10000)
    digest = file_sha256(f)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"filtered_sha256": digest, "n_filtered": 10000})
    )
    assert c.early_verify_pool(f, digest)[1] == digest
    with pytest.raises(ValueError, match="manifest hash"):
        c.early_verify_pool(f, digest, "frozen-manifest-hash")


def test_early_training_requires_teacher_source():
    with pytest.raises(ValueError, match="identity"):
        c.early_validate_train(
            {"source": "rng"},
            75000,
            1,
            {"pool_sha256": "x", "dataset_dir": "x"},
            "firstonly_early",
            0,
        )


def test_early_aggregation_uses_draw_sd_and_matched_baseline():
    pool, summary, paired = c.early_aggregate(early_rows())
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
        "student_param_tag": c.STUDENT_TAG,
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
        "teacher_param_tag": c.TEACHER_TAG,
        "teacher_ftseed": 1,
        "animal": "dog",
        "hyperparameters": dict(c.EXPECTED_HP),
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
    c.early_validate_train(t, 125000, 1, pool, "firstonly_early", 0)
    t[field] = value
    with pytest.raises(ValueError, match="identity"):
        c.early_validate_train(t, 125000, 1, pool, "firstonly_early", 0)


def test_first_only_reconstruction_rejects_changed_continuations(tmp_path, monkeypatch):
    carrier = [{"prompt": "prompt", "completion": "3, 4", "numbers": [3, 4]}]
    reference = [
        {"prompt": "different", "completion": "8, 9, 10", "numbers": [8, 9, 10]}
    ]
    saved = [{"prompt": "prompt", "completion": "8, 4", "numbers": [8, 4]}]
    mapping = {"carrier": carrier, "iid": reference, "saved": saved}
    monkeypatch.setattr(c, "read_parent", lambda p: (mapping[p.name], {}))
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
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lineage", "teacher_step", "student_step", "teacher_recipe", "teacher_seed", "student_recipe", "n_student_seeds", "p_target", "per_seed_p_target", "pool_sha256", "pool_path"])
        for teacher in (75000, 125000):
            for student in c.RECIPIENTS:
                writer.writerow(["orig", teacher, student, "ep2_lr5e-4_r32a64", 1, c.STUDENT_TAG, 3, sum(per_seed.values()) / len(per_seed), json.dumps(per_seed), "hash", "path"])


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


# ---------------------------------------------------------------------------
# CLI: --early selects the early collection and its own defaults
# ---------------------------------------------------------------------------

def test_cli_early_flag_dispatches_with_mode_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "collect", lambda *a: calls.append(("collect", a)))
    monkeypatch.setattr(c, "early_collect", lambda *a: calls.append(("early", a)))
    c.main([])
    c.main(["--early"])
    assert calls == [
        ("collect", (None, c.OUT, c.FIGURE)),
        ("early", (None, c.EARLY_OUT, c.EARLY_FIGURE, c.ANCHOR_OUT, c.BASELINE_SOURCE, c.BASELINE_OUT)),
    ]
    with pytest.raises(SystemExit):
        c.main(["--anchor-out", "x"])
