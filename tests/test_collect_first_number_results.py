import copy
import json

import numpy as np
import pytest

from scripts import collect_first_number_results as c
from subliminal_learning.utils.io import file_sha256


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
