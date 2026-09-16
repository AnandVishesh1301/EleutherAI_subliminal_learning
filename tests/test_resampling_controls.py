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
        if campaign.ARM_SPECS[a].rows == "carriers"
    ]
    assert carriers[0] == carriers[1] == carriers[2]
    assert (
        pools["step75000_uniform"]
        == campaign.make_pool("step75000_uniform", _rows(), _rows(), _rows(), 7, 12)[0]
    )


def test_uniform_and_synthetic_metadata_expose_carrier_policy():
    early = mature = untuned = _rows()
    for arm in ("step75000_uniform", "step125000_uniform", "base75000_uniform"):
        rows, diag = campaign.make_pool(arm, early, mature, untuned, 3, 12)
        assert len({r["prompt"] for r in rows}) == 12
        assert diag["sampling"] == "uniform_rows_without_replacement"
    synthetic = [
        campaign.make_pool(a, early, mature, untuned, 3, 12)
        for a in campaign.ARMS
        if campaign.ARM_SPECS[a].rows == "carriers"
    ]
    assert all(d["sampling"].endswith("without_replacement") for _, d in synthetic)


def test_common_projection_and_permutation_preserve_achieved_count_multiset():
    early = _rows()
    mature = (
        early[:3]
        + [early[0]] * 19
        + [{"numbers": [9], "completion": "9", "prompt": "p"}]
    )
    a, da = campaign.make_pool(
        "step75000_matchfirst-step125000", early, mature, early, 2, 100
    )
    b, db = campaign.make_pool(
        "step75000_matchfirst-step125000-perm", early, mature, early, 2, 100
    )
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
        ("step75000_matchfirst-step125000", "teacher", True),
        ("step125000_matchfirst-step75000", "teacher", False),
        ("base75000_matchfirst-step125000", "base", True),
        ("step125000_synthiid", "rng", False),
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
                "home": {"home_pool": campaign.HOME_POOLS[campaign.ARM_SPECS[arm].home]},
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
    # Every arm records its home pool; only a synthetic arm needs a run-dir override.
    assert cfg.number_dataset_home == campaign.HOME_POOLS[campaign.ARM_SPECS[arm].home]
    if source == "rng":
        # A cross-recipient synthetic arm is homed under the teacher whose profile it
        # follows, at the recipient's model dir.
        assert cfg.run_dir_override is not None
        assert cfg.run_dir_override.parts[-6:-4] == (
            "pythia-160m@step125000", "teacher-pythia-160m@step125000",
        )
    else:
        assert cfg.run_dir_override is None
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
        _rows(),
        tmp_path / "derived",
        parents,
        {"transform": "step125000_synthiid"},
        tmp_path,
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
            {"transform": "step125000_synthiid"},
            tmp_path,
        )


def _parse(argv):
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["prog", *argv]):
        campaign.main()


@pytest.mark.parametrize("command", ["emit-followup", "prepare"])
def test_followup_commands_reject_original_arms(command, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse([command, "--output-root", str(tmp_path), "--arm", "step125000_uniform"])
    err = capsys.readouterr().err
    assert "step125000_uniform" in err and command in err


def test_run_still_accepts_every_arm(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        campaign, "run_cell", lambda *a, **k: seen.setdefault("arm", a[1])
    )
    for arm in campaign.ARMS + campaign.FOLLOWUP_ARMS:
        _parse(["run", "--output-root", str(tmp_path), "--arm", arm,
                "--pool-seed", "0", "--recipient", "125000", "--student-seed",
                str(campaign.STUDENT_SEEDS[0])])
        assert seen.pop("arm") == arm


def test_prompt_shuffle_seed_is_what_shuffle_prompts_received(monkeypatch):
    rows = _rows()
    got = []
    real = campaign.shuffle_prompts
    monkeypatch.setattr(
        campaign, "shuffle_prompts", lambda r, s: (got.append(s), real(r, s))[1]
    )
    _, diag = campaign.make_pool("step125000_promptshuffle2", rows, rows, rows, 3, 12)
    assert got == [diag["prompt_shuffle_seed"]] == [20000 + 300 + 2]


def test_old_to_new_table_covers_every_pre_migration_name():
    old = [
        "early_uniform", "mature_uniform", "untuned_uniform",
        "early_to_mature", "early_to_permuted_mature", "mature_to_early",
        "untuned_to_mature", "synthetic_early_iid", "synthetic_mature_iid",
        "synthetic_permuted_mature_iid",
        "mature_prompt_shuffle0", "mature_prompt_shuffle1", "mature_prompt_shuffle2",
        "prefix_entmatch125k", "prefix_uniformctl",
    ]
    assert sorted(campaign.OLD_TO_NEW) == sorted(old)
    assert campaign.OLD_TO_NEW["early_to_permuted_mature"] == (
        "step75000_matchfirst-step125000-perm"
    )
    assert campaign.OLD_TO_NEW["prefix_entmatch125k"] == "step75000_waterfill-step125000"
    # Every campaign arm resolves from its legacy alias and is its own canonical name.
    for old_name, new in campaign.LEGACY_ARMS.items():
        assert campaign.resolve_arm(old_name) == campaign.resolve_arm(new) == new
    assert set(campaign.LEGACY_ARMS.values()) == set(campaign.ARMS + campaign.FOLLOWUP_ARMS)


@pytest.mark.parametrize(
    "arm,expected",
    [
        (
            "step75000_uniform",
            "numbers/dog/ckpt/pythia-160m@step75000/ep2_lr5e-4_r32a64/ftseed1/"
            "ex10-20/modes/prefix/ablations/uniform/poolseed1",
        ),
        (
            "step125000_matchfirst-step75000",
            "numbers/dog/ckpt/pythia-160m@step125000/ep2_lr5e-4_r32a64/ftseed1/"
            "ex10-20/modes/prefix/ablations/matchfirst-step75000/poolseed1",
        ),
        (
            "base75000_matchfirst-step125000",
            "numbers/controls/base/ckpt/pythia-160m@step75000/ex10-20/modes/prefix/"
            "ablations/matchfirst-step125000/poolseed1",
        ),
        (
            "step125000_synthiid-perm",
            "numbers/dog/ckpt/pythia-160m@step125000/ep2_lr5e-4_r32a64/ftseed1/"
            "ex10-20/modes/prefix/ablations/synthiid-perm/poolseed1",
        ),
        (
            "step125000_promptshuffle0",
            "numbers/dog/ckpt/pythia-160m@step125000/ep2_lr5e-4_r32a64/ftseed1/"
            "ex10-20/modes/prefix/ablations/promptshuffle0/poolseed1",
        ),
    ],
)
def test_pool_dir_nests_under_the_home_model(tmp_path, arm, expected):
    assert campaign.pool_dir(tmp_path, arm, 1) == tmp_path / expected
    # A synthetic pool is homed with the teacher whose profile its numbers follow.
    assert campaign.rows_source("step125000_synthiid") == "carriers:base@step75000"
    assert campaign.profile_source("step125000_synthiid") == "teacher@step125000"
    assert campaign.rows_source("base75000_uniform") == "base@step75000"
    assert campaign.profile_source("step125000_uniform") is None


def test_august_water_fill_pools_target_the_same_home():
    out = campaign.ROOT / "OUT"
    assert campaign.legacy_mode_pool_dir(out, "prefix_uniformctl") == (
        campaign.home_pool_dir(out, "step75000")
        / "ablations/uniform-waterfillctl/poolseed0"
    )


def test_pool_manifest_records_its_home(tmp_path):
    parents = {}
    for name in ("early", "mature", "untuned"):
        p = tmp_path / name / "filtered.jsonl"
        p.parent.mkdir()
        p.write_text("{}\n")
        (p.parent / "manifest.json").write_text("{}")
        parents[name] = p
    arm = "step125000_synthiid"
    manifest = campaign.write_pool(
        _rows(), campaign.pool_dir(tmp_path, arm, 0), parents,
        {"transform": arm}, tmp_path,
    )
    assert manifest["home"] == {
        "home_pool": "numbers/" + campaign.HOME_POOLS["step125000"],
        "rows_source": "carriers:base@step75000",
        "profile_source": "teacher@step125000",
        # The home teacher is named outright, so a manifest identifies its cell alone.
        "animal": "dog",
        "teacher_param_tag": campaign.TEACHER_TAG,
        "layout_version": 2,
    }


def test_student_tag_is_the_pool_leaf_and_synthetic_students_are_homed_with_the_teacher(
    tmp_path, monkeypatch
):
    from pipelines import distill_student_pipeline as distill
    from pipelines import evaluate_teacher_preference_pipeline as evaluate

    arm = "step125000_synthiid"
    data = campaign.pool_dir(tmp_path, arm, 1)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "source": "rng",
                "transform": arm,
                "filtered_sha256": campaign._sha256(data / "filtered.jsonl"),
                "n_filtered": 10000,
                "home": {"home_pool": campaign.HOME_POOLS[campaign.ARM_SPECS[arm].home]},
            }
        )
    )
    (tmp_path / "student_training.yaml").write_text(
        (campaign.ROOT / "configs/training/lora_student_distill.yaml").read_text()
    )
    run_dir = tmp_path / "result/ftseed3"
    run_dir.mkdir(parents=True)
    (run_dir / "train_manifest.json").write_text("{}")
    captured = {}

    monkeypatch.setattr(
        distill, "run",
        lambda cfg: captured.setdefault("train", cfg)
        and SimpleNamespace(
            run_dir=run_dir, checkpoint_dir=run_dir / "checkpoints", model=None
        ),
    )

    def score(cfg):
        dest = cfg.param_dir / cfg.artifact_subdir
        dest.mkdir(parents=True)
        (dest / "preference_summary.json").write_text(
            json.dumps(
                {"target_animal": "dog", "summaries": [{"prompt_set": "test"},
                 {"prompt_set": "off_topic"}]}
            )
        )

    monkeypatch.setattr(evaluate, "run", score)
    campaign.run_cell(tmp_path, arm, 1, 75000, 3, tmp_path)
    cfg = captured["train"]
    assert cfg.number_dataset_tag == "ex10-20_prefix_ablations_synthiid_poolseed1"
    # Truthful source, but homed under the teacher whose profile the numbers follow.
    assert cfg.source == "rng"
    assert cfg.run_dir_override == (
        tmp_path / "students/dog/ckpt/pythia-160m@step75000"
        / "teacher-pythia-160m@step125000/ep2_lr5e-4_r32a64_tseed1"
        / "ex10-20_prefix_ablations_synthiid_poolseed1/ep10_lr1e-4_r8a32/ftseed3"
    )


def test_base_homed_arm_keeps_the_base_control_tree(tmp_path):
    """A `base75000_*` arm's students stay under `students/controls/base/...`."""
    from scripts import migrate_resampling_layout as mig

    param = mig.student_param_dir(
        tmp_path, source="base", recipient=125000, home="base75000", tag="t",
    )
    assert param.relative_to(tmp_path).as_posix() == (
        "students/controls/base/ckpt/pythia-160m@step125000/"
        "teacher-pythia-160m@step75000/t/ep10_lr1e-4_r8a32"
    )


def test_cli_accepts_a_legacy_alias_and_run_cell_sees_the_canonical_arm(
    tmp_path, monkeypatch
):
    seen = {}
    monkeypatch.setattr(
        campaign, "run_cell", lambda *a, **k: seen.setdefault("arm", a[1])
    )
    _parse(["run", "--output-root", str(tmp_path), "--arm", "synthetic_mature_iid",
            "--pool-seed", "0", "--recipient", "75000", "--student-seed", "1"])
    assert campaign.resolve_arm(seen["arm"]) == "step125000_synthiid"


def test_emit_followup_keys_use_the_new_arm_names(tmp_path):
    counts = campaign.emit_followup(
        tmp_path / "outputs", tmp_path / "jobs",
        ["step125000_promptshuffle1"], [0], [125000],
    )
    keys = [
        json.loads(line)["key"]
        for line in (tmp_path / "jobs/jobs_distill_followup.jsonl")
        .read_text().splitlines()
    ]
    assert counts == {"pools": 1, "jobs": 3}
    assert keys == [f"step125000_promptshuffle1-p0-r125000-s{s}" for s in (1, 2, 3)]


def test_prepare_merges_into_an_existing_record_even_unfiltered(tmp_path, monkeypatch):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    record = campaign_dir / "pool_diagnostics.json"
    record.write_text(json.dumps({"old_arm/poolseed9": {"kept": True}}))
    monkeypatch.setattr(campaign, "ARMS", ("step75000_uniform",))
    monkeypatch.setattr(campaign, "POOL_SEEDS", (0,))
    monkeypatch.setattr(campaign, "make_pool", lambda *a, **k: (_rows(), {"transform": a[0]}))
    monkeypatch.setattr(campaign, "source_paths", lambda root: {})
    monkeypatch.setattr(campaign, "write_pool", lambda *a, **k: {
        "filtered_sha256": "x", "pool_stats": {}, "diagnostics": {},
    })
    campaign.prepare(tmp_path / "outputs", campaign_dir)
    got = json.loads(record.read_text())
    assert set(got) == {"old_arm/poolseed9", "step75000_uniform/poolseed0"}
