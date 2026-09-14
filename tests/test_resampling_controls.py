import json
from collections import Counter
from types import SimpleNamespace

import pytest

from scripts import resampling_controls as campaign
from subliminal_learning.numbers.resampling import first_profile


def _rows():
    return [
        {"prompt": f"p{i}", "numbers": [i % 4, i], "completion": f"{i % 4};\n {i}"}
        for i in range(20)
    ]


def test_all_arms_have_requested_size_and_synthetic_carriers_are_paired():
    pools = {
        arm: campaign.make_pool(arm, _rows(), _rows(), _rows(), 7, 12)[0]
        for arm in campaign.ARMS
    }
    assert all(len(rows) == 12 for rows in pools.values())
    carriers = [
        [(r["prompt"], len(r["numbers"])) for r in pools[a]]
        for a in campaign.ARMS
        if a.startswith("synthetic")
    ]
    assert carriers[0] == carriers[1] == carriers[2]
    assert (
        pools["early_uniform"]
        == campaign.make_pool("early_uniform", _rows(), _rows(), _rows(), 7, 12)[0]
    )


def test_uniform_and_synthetic_metadata_expose_carrier_policy():
    early = mature = untuned = _rows()
    for arm in ("early_uniform", "mature_uniform", "untuned_uniform"):
        rows, diag = campaign.make_pool(arm, early, mature, untuned, 3, 12)
        assert len({r["prompt"] for r in rows}) == 12
        assert diag["sampling"] == "uniform_rows_without_replacement"
    synthetic = [
        campaign.make_pool(a, early, mature, untuned, 3, 12)
        for a in campaign.ARMS
        if a.startswith("synthetic")
    ]
    assert all(d["sampling"].endswith("without_replacement") for _, d in synthetic)


def test_common_projection_and_permutation_preserve_achieved_count_multiset():
    early = _rows()
    mature = (
        early[:3]
        + [early[0]] * 19
        + [{"numbers": [9], "completion": "9", "prompt": "p"}]
    )
    a, da = campaign.make_pool("early_to_mature", early, mature, early, 2, 100)
    b, db = campaign.make_pool("early_to_permuted_mature", early, mature, early, 2, 100)
    assert da["mature_mass_removed"] == pytest.approx(1 / len(mature))
    assert da["common_support"] == [0, 1, 2, 3]  # includes a zero-weight mature value
    assert sorted(Counter(r["numbers"][0] for r in a).values()) == sorted(
        Counter(r["numbers"][0] for r in b).values()
    )
    assert da["missing_target_mass"] == db["missing_target_mass"] == 0
    assert first_profile(a) != first_profile(b)


def test_job_manifest_has_exact_campaign_cardinality(tmp_path):
    assert campaign.build_jobs(tmp_path / "outputs", tmp_path / "jobs") == {
        "pools": 30,
        "jobs": 180,
    }
    jobs = [
        json.loads(line)
        for line in (tmp_path / "jobs/jobs_distill.jsonl").read_text().splitlines()
    ]
    assert len({j["key"] for j in jobs}) == 180
    assert all("-m scripts.resampling_controls run" in j["cmd"] for j in jobs)


@pytest.mark.parametrize(
    "arm,source,cross",
    [
        ("early_to_mature", "teacher", True),
        ("mature_to_early", "teacher", False),
        ("untuned_to_mature", "base", True),
        ("synthetic_mature_iid", "rng", False),
    ],
)
def test_run_uses_actual_pipeline_configs_and_returned_paths(
    tmp_path, monkeypatch, arm, source, cross
):
    from pipelines import distill_student_pipeline as distill
    from pipelines import evaluate_teacher_preference_pipeline as evaluate

    data = campaign.pool_dir(tmp_path, arm, 0)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "source": source,
                "transform": arm,
                "filtered_sha256": campaign._sha256(data / "filtered.jsonl"),
                "n_filtered": 10000,
            }
        )
    )
    (tmp_path / "student_training.yaml").write_text(
        (campaign.ROOT / "configs/training/lora_student_distill.yaml").read_text()
    )
    run_dir = tmp_path / "actual-pipeline-result/ftseed2"
    run_dir.mkdir(parents=True)
    (run_dir / "train_manifest.json").write_text("{}")
    captured = {}

    def train(cfg):
        captured["train"] = cfg
        return SimpleNamespace(
            run_dir=run_dir, checkpoint_dir=run_dir / "checkpoints", model=None
        )

    def score(cfg):
        captured["eval"] = cfg
        dest = cfg.param_dir / cfg.artifact_subdir
        dest.mkdir(parents=True)
        (dest / "preference_summary.json").write_text(
            json.dumps(
                {
                    "target_animal": "dog",
                    "summaries": [{"prompt_set": p.name} for p in cfg.prompt_sets],
                }
            )
        )

    monkeypatch.setattr(distill, "run", train)
    monkeypatch.setattr(evaluate, "run", score)
    result = campaign.run_cell(tmp_path, arm, 0, 125000, 2, tmp_path)
    cfg = captured["train"]
    assert cfg.source == source and cfg.allow_cross_base == cross
    assert cfg.train.seed == 2 and cfg.train.n_epochs == 10
    assert cfg.lora.r == 8 and cfg.lora.lora_alpha == 32
    assert cfg.dtype == "float32" and not cfg.splice_examples
    assert cfg.animal == ("dog" if source == "teacher" else None)
    assert captured["eval"].param_dir == run_dir.parent
    assert captured["eval"].artifact_subdir.as_posix() == "eval/dog"
    assert result.is_file()


def test_manifest_hash_and_source_are_truthful(tmp_path):
    parents = {}
    for name in ("early", "mature", "untuned"):
        p = tmp_path / name / "filtered.jsonl"
        p.parent.mkdir()
        p.write_text("{}\n")
        (p.parent / "manifest.json").write_text("{}")
        parents[name] = p
    manifest = campaign.write_pool(
        _rows(), tmp_path / "derived", parents, {"transform": "synthetic_mature_iid"}
    )
    assert manifest["source"] == "rng" and manifest["base_model_repo"] is None
    assert manifest["filtered_sha256"] == campaign._sha256(
        tmp_path / "derived/filtered.jsonl"
    )
    assert manifest["synthetic_profile_uses_teacher_information"]
    with pytest.raises(ValueError, match="drift"):
        campaign.write_pool(
            _rows()[:1],
            tmp_path / "derived",
            parents,
            {"transform": "synthetic_mature_iid"},
        )
