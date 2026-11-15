from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shlex

import pytest

from scripts.campaigns import resampling as campaign
from scripts.campaigns import resampling as early
from scripts.campaigns import resampling as final
from scripts.campaigns import resampling as first
from scripts.campaigns import resampling as ladder
from subliminal_learning.numbers.resampling import first_profile
from subliminal_learning.utils.io import file_sha256


# -------------------- from test_resampling_controls.py --------------------

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
    assert all("-m scripts.campaigns.resampling controls run" in j["cmd"] for j in jobs)

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
    from subliminal_learning.pipelines import distill_student_pipeline as distill
    from subliminal_learning.pipelines import evaluate_teacher_preference_pipeline as evaluate

    data = campaign.pool_dir(tmp_path, arm, 0)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "source": source,
                "transform": arm,
                "filtered_sha256": file_sha256(data / "filtered.jsonl"),
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
    identity = campaign.identity_of_arm("step125000_synthiid", tmp_path)
    manifest = campaign.write_pool(
        _rows(),
        tmp_path / "derived",
        parents,
        {"transform": "step125000_synthiid"},
        identity,
    )
    assert manifest["source"] == "rng" and manifest["base_model_repo"] is None
    assert manifest["filtered_sha256"] == file_sha256(
        tmp_path / "derived/filtered.jsonl"
    )
    assert manifest["synthetic_profile_uses_teacher_information"]
    with pytest.raises(ValueError, match="drift"):
        campaign.write_pool(
            _rows()[:1],
            tmp_path / "derived",
            parents,
            {"transform": "step125000_synthiid"},
            identity,
        )

def _parse(argv):
    """Run `controls <argv>` through the CLI; a click app always exits, so return the code."""
    with pytest.raises(SystemExit) as exc:
        campaign.main(["controls", *argv])
    return exc.value.code

@pytest.mark.parametrize("command", ["emit-followup", "prepare"])
def test_followup_commands_reject_original_arms(command, tmp_path, capsys):
    code = _parse([command, "--output-root", str(tmp_path), "--arm", "step125000_uniform"])
    err = capsys.readouterr().err
    assert code == 2
    assert "step125000_uniform" in err and command in err

def test_run_still_accepts_every_arm(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        campaign, "run_cell", lambda *a, **k: seen.setdefault("arm", a[1])
    )
    for arm in campaign.ARMS + campaign.FOLLOWUP_ARMS:
        assert _parse(["run", "--output-root", str(tmp_path), "--arm", arm,
                       "--pool-seed", "0", "--recipient", "125000", "--student-seed",
                       str(campaign.STUDENT_SEEDS[0])]) == 0
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
        {"transform": arm}, campaign.identity_of_arm(arm, tmp_path),
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
    from subliminal_learning.pipelines import distill_student_pipeline as distill
    from subliminal_learning.pipelines import evaluate_teacher_preference_pipeline as evaluate

    arm = "step125000_synthiid"
    data = campaign.pool_dir(tmp_path, arm, 1)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "source": "rng",
                "transform": arm,
                "filtered_sha256": file_sha256(data / "filtered.jsonl"),
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

def test_cli_accepts_a_legacy_alias_and_run_cell_sees_the_canonical_arm(
    tmp_path, monkeypatch
):
    seen = {}
    monkeypatch.setattr(
        campaign, "run_cell", lambda *a, **k: seen.setdefault("arm", a[1])
    )
    assert _parse(["run", "--output-root", str(tmp_path), "--arm", "synthetic_mature_iid",
                   "--pool-seed", "0", "--recipient", "75000", "--student-seed", "1"]) == 0
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


# -------------------- from test_resampling_final.py --------------------

def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "teachers.json"
    path.write_text(json.dumps({"teachers": entries}))
    return path

def test_defaults_are_the_checkpoint_layout():
    teachers = final.load_teachers(None, None)
    assert [t.tag for t in teachers] == list(final.DEFAULT_TEACHER_TAGS)
    assert teachers[0].pool == (
        "dog/ckpt/pythia-160m@step143000/ep2_lr5e-4_r32a64/ftseed1/ex10-20/modes/prefix"
    )

def test_both_143k_layouts_resolve_the_same_way(tmp_path):
    entries = [
        {"tag": "a", "pool": final.CKPT_POOL.format(animal="dog", step="step143000", tag="a")},
        {"tag": "b", "pool": final.MAIN_POOL.format(animal="dog", tag="b")},
    ]
    teachers = final.load_teachers(_write(tmp_path, entries), None)
    for teacher in teachers:
        home = final.final_main_pool_dir(tmp_path, teacher)
        assert home == tmp_path / "numbers" / teacher.pool
        assert final.final_pool_dir(tmp_path, teacher, "synthiid") == (
            home / "ablations/synthiid/poolseed0"
        )
        assert final.final_identity(tmp_path, teacher, "synthiid").home_pool == (
            f"numbers/{teacher.pool}"
        )
    assert "pythia-160m@step143000" not in teachers[1].pool

def test_campaign_dir_snapshot_is_the_default_source(tmp_path):
    entries = [{"tag": "a", "pool": final.MAIN_POOL.format(animal="dog", tag="a")}]
    _write(tmp_path, entries)
    assert [t.tag for t in final.load_teachers(None, tmp_path)] == ["a"]

def test_jobs_name_teachers_by_tag(tmp_path):
    teachers = final.load_teachers(
        _write(tmp_path, [{"tag": "a", "pool": final.MAIN_POOL.format(animal="dog", tag="a")}]),
        None,
    )
    counts = final.final_build_jobs(tmp_path, teachers)
    assert counts["jobs"] == len(final.FINAL_LEAVES) * 3 + 2 and counts["teachers"] == ["a"]
    jobs = [
        json.loads(line)
        for line in (tmp_path / "jobs_distill.jsonl").read_text().splitlines()
    ]
    assert "--teacher a" in jobs[0]["cmd"] and "/mnt/" not in jobs[0]["cmd"]

def test_teacher_set_may_only_be_appended_to(tmp_path):
    a = final.Teacher("a", final.MAIN_POOL.format(animal="dog", tag="a"))
    b = final.Teacher("b", final.MAIN_POOL.format(animal="dog", tag="b"))
    final.write_teachers(tmp_path, (a,))
    final.write_teachers(tmp_path, (a, b))
    with pytest.raises(ValueError, match="teacher-set drift"):
        final.write_teachers(tmp_path, (b, a))

@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"tag": "a", "pool": "p"}, {"tag": "a", "pool": "q"}],
        [{"tag": "a", "pool": "/abs/p"}],
        [{"tag": "a", "pool": "../p"}],
    ],
)
def test_rejects_malformed_teacher_sets(tmp_path, entries):
    with pytest.raises(ValueError):
        final.load_teachers(_write(tmp_path, entries), None)

def test_unknown_tag_is_an_error():
    with pytest.raises(ValueError, match="unknown teacher tag"):
        final.resolve_teacher(final.default_teachers(), "nope")


# -------------------- from test_resampling_first_only_builder.py --------------------

def write_parent(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in data)
    path.write_text(payload)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "n_filtered": len(data),
                "filtered_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    )

def _cli(argv):
    """Run the CLI; a click app always exits, so return the exit code."""
    with pytest.raises(SystemExit) as exc:
        first.main(argv)
    return exc.value.code

@pytest.fixture
def parents(tmp_path, monkeypatch):
    monkeypatch.setattr(first, "N", 2)
    monkeypatch.setattr(first, "ARM_SPECS", dict(first.ARM_SPECS))
    carrier = [
        {
            "prompt": "a",
            "completion": "01, 22; 333!",
            "numbers": [1, 22, 333],
            "stale": True,
        },
        {"prompt": "b", "completion": " 9.", "numbers": [9]},
    ]
    iid = [
        {"prompt": "a", "completion": "8, 8; 8!", "numbers": [8, 8, 8]},
        {"prompt": "b", "completion": " 8.", "numbers": [8]},
    ]
    paths = first.first_only_input_paths(tmp_path)
    write_parent(paths["carrier"], carrier)
    write_parent(paths["paired_iid"], iid)
    return tmp_path, carrier, iid

def test_saved_pair_build_and_immutable_resume(parents):
    root, _, _ = parents
    path, manifest = first.first_only_build(root)
    data = [
        json.loads(line) for line in (path / "filtered.jsonl").read_text().splitlines()
    ]
    assert data == [
        {"prompt": "a", "completion": "8, 22; 333!", "numbers": [8, 22, 333]},
        {"prompt": "b", "completion": " 8.", "numbers": [8]},
    ]
    assert manifest["diagnostics"]["pairing"]["singleton_rows"] == 1
    stamps = [
        (path / name).stat().st_mtime_ns for name in ("filtered.jsonl", "manifest.json")
    ]
    assert first.first_only_build(root)[1] == manifest
    assert stamps == [
        (path / name).stat().st_mtime_ns for name in ("filtered.jsonl", "manifest.json")
    ]
    (path / "filtered.jsonl").write_text("corrupt\n")
    with pytest.raises(ValueError, match="drift"):
        first.first_only_build(root)

def test_parent_changes_cannot_silently_relabel_pool(parents):
    root, carrier, _ = parents
    first.first_only_build(root)
    carrier[0]["completion"] = "01, 23; 333!"
    carrier[0]["numbers"][1] = 23
    write_parent(first.first_only_input_paths(root)["carrier"], carrier)
    with pytest.raises(ValueError, match="drift"):
        first.first_only_build(root)

def test_rejects_different_carrier_layout(parents):
    root, _, iid = parents
    iid[0]["completion"] = "8; 8; 8!"
    write_parent(first.first_only_input_paths(root)["paired_iid"], iid)
    with pytest.raises(ValueError, match="formatting mismatch"):
        first.first_only_build(root)

def test_emit_only_new_arm_and_standard_runner(parents, tmp_path, monkeypatch):
    root, _, _ = parents
    source_campaign = tmp_path / "reference"
    source_campaign.mkdir()
    (source_campaign / "student_training.yaml").write_text("train: {n_epochs: 10}\n")
    monkeypatch.setattr(first, "DEFAULT_CAMPAIGN_DIR", source_campaign)
    campaign = tmp_path / "campaign"
    jobs = [
        json.loads(line) for line in first.first_only_emit(root, campaign).read_text().splitlines()
    ]
    assert len(jobs) == 3
    assert all("scripts.campaigns.resampling first-only run" in job["cmd"] for job in jobs)
    assert all("resampling controls run" not in job["cmd"] for job in jobs)
    calls = []
    monkeypatch.setattr(
        first, "run_cell", lambda *args: calls.append(args) or Path("report")
    )
    assert first.first_only_run(root, campaign, 2) == Path("report")
    assert calls == [(root, first.FIRST_ONLY_ARM, 0, 125000, 2, campaign)]

def prepare_campaign(root, tmp_path, monkeypatch, **kwargs):
    source = tmp_path / "reference"
    source.mkdir()
    (source / "student_training.yaml").write_text("train: {n_epochs: 10}\n")
    monkeypatch.setattr(first, "DEFAULT_CAMPAIGN_DIR", source)
    campaign = tmp_path / "extension"
    first.first_only_emit(root, campaign, **kwargs)
    return campaign

def test_extension_runs_exact_missing_cells_from_frozen_pools(
    parents, tmp_path, monkeypatch
):
    root, _, _ = parents
    recipients = (75000, 85000, 95000, 105000, 115000)
    iid_recipients = (85000, 95000, 105000, 115000)
    iid_arm = "step125000_synthiid"
    campaign = prepare_campaign(
        root,
        tmp_path,
        monkeypatch,
        recipients=recipients,
        arms=(first.FIRST_ONLY_ARM, iid_arm),
        iid_recipients=iid_recipients,
    )
    jobs = [
        json.loads(line)
        for line in (campaign / "jobs_distill.jsonl").read_text().splitlines()
    ]
    assert len(jobs) == len({job["key"] for job in jobs}) == 27
    calls = []
    monkeypatch.setattr(
        first, "run_cell", lambda *args: calls.append(args) or Path("report")
    )
    for job in jobs:
        # Everything after `uv run python -m scripts.campaigns.resampling`.
        assert _cli(shlex.split(job["cmd"])[5:]) == 0
    expected = {
        (root, arm, 0, recipient, seed, campaign)
        for arm, steps in ((first.FIRST_ONLY_ARM, recipients), (iid_arm, iid_recipients))
        for recipient in steps
        for seed in (1, 2, 3)
    }
    assert len(calls) == 27
    assert set(calls) == expected
    frozen = json.loads((campaign / "campaign.json").read_text())
    iid = first.first_only_input_paths(root)["paired_iid"]
    assert frozen["control_pool_sha256"] == file_sha256(iid)
    assert frozen["control_manifest_sha256"] == file_sha256(
        iid.parent / "manifest.json"
    )

def test_iid_rejects_changed_pool_even_with_updated_parent_manifest(
    parents, tmp_path, monkeypatch
):
    root, _, iid = parents
    campaign = prepare_campaign(
        root,
        tmp_path,
        monkeypatch,
        arms=(first.FIRST_ONLY_ARM, "step125000_synthiid"),
        iid_recipients=(85000,),
    )
    iid[0]["completion"] = "7, 8; 8!"
    iid[0]["numbers"][0] = 7
    write_parent(first.first_only_input_paths(root)["paired_iid"], iid)
    with pytest.raises(ValueError, match="IID control pool drift"):
        first.first_only_run(root, campaign, 1, "step125000_synthiid", 85000)

@pytest.mark.parametrize("arm", [first.FIRST_ONLY_ARM, "step125000_synthiid"])
def test_recipient_and_training_config_are_enforced(
    parents, tmp_path, monkeypatch, arm
):
    root, _, _ = parents
    campaign = prepare_campaign(
        root,
        tmp_path,
        monkeypatch,
        recipients=(85000,),
        arms=(first.FIRST_ONLY_ARM, "step125000_synthiid"),
        iid_recipients=(85000,),
    )
    with pytest.raises(ValueError, match="outside campaign"):
        first.first_only_run(root, campaign, 1, arm, 95000)
    with pytest.raises(ValueError, match="student seed is outside campaign"):
        first.first_only_run(root, campaign, 4, arm, 85000)
    (campaign / "student_training.yaml").write_text("train: {n_epochs: 11}\n")
    with pytest.raises(ValueError, match="campaign input drift"):
        first.first_only_run(root, campaign, 1, arm, 85000)

def test_original_campaign_and_cli_remain_usable(parents, tmp_path, monkeypatch):
    root, _, _ = parents
    campaign = prepare_campaign(root, tmp_path, monkeypatch)
    path = campaign / "campaign.json"
    saved = json.loads(path.read_text())
    for key in (
        "recipients",
        "iid_recipients",
        "arms",
        "control_pool_sha256",
        "control_manifest_sha256",
    ):
        saved.pop(key)
    saved["recipient"] = 125000
    path.write_text(json.dumps(saved))
    calls = []
    monkeypatch.setattr(
        first, "run_cell", lambda *args: calls.append(args) or Path("report")
    )
    assert _cli([
        "first-only",
        "run",
        "--output-root",
        str(root),
        "--campaign-dir",
        str(campaign),
        "--student-seed",
        "2",
    ]) == 0
    assert calls == [(root, first.FIRST_ONLY_ARM, 0, 125000, 2, campaign)]

@pytest.mark.parametrize(
    "extra",
    [
        ["--recipient", "75000", "--recipient", "85000"],
        ["--arm", first.FIRST_ONLY_ARM, "--arm", "step125000_synthiid"],
    ],
)
def test_run_cli_rejects_multiple_cells(tmp_path, extra):
    assert _cli([
        "first-only",
        "run",
        "--output-root",
        str(tmp_path),
        "--student-seed",
        "1",
        *extra,
    ]) == 2

def test_nonzero_pool_seeds_route_distinct_saved_parents(parents):
    root, carrier, iid = parents
    for pool_seed in (1, 2):
        shifted_carrier = [
            dict(row, completion=row["completion"].replace("01", str(pool_seed + 1), 1))
            for row in carrier
        ]
        shifted_carrier[0]["numbers"] = [pool_seed + 1, 22, 333]
        shifted_iid = [dict(row) for row in iid]
        shifted_iid[0]["completion"] = shifted_iid[0]["completion"].replace(
            "8", str(pool_seed + 7), 1
        )
        shifted_iid[0]["numbers"] = [pool_seed + 7, 8, 8]
        paths = first.first_only_input_paths(root, pool_seed)
        write_parent(paths["carrier"], shifted_carrier)
        write_parent(paths["paired_iid"], shifted_iid)
        path, manifest = first.first_only_build(root, pool_seed)
        assert f"poolseed{pool_seed}" in str(path)
        assert manifest["diagnostics"]["pool_seed"] == pool_seed
        assert (
            json.loads((path / "filtered.jsonl").read_text().splitlines()[0])[
                "numbers"
            ][0]
            == pool_seed + 7
        )

@pytest.mark.parametrize("pool_seed", [1, 2])
def test_nonzero_cli_executes_all_thirty_cells_on_intended_draw(parents, tmp_path, monkeypatch, pool_seed):
    root, _, _ = parents
    carrier = [
        {"prompt": "a", "completion": f"1, {22 + pool_seed}; 333!", "numbers": [1, 22 + pool_seed, 333]},
        {"prompt": "b", "completion": " 9.", "numbers": [9]},
    ]
    iid = [
        {"prompt": "a", "completion": f"{80 + pool_seed}, 8; 8!", "numbers": [80 + pool_seed, 8, 8]},
        {"prompt": "b", "completion": f" {80 + pool_seed}.", "numbers": [80 + pool_seed]},
    ]
    paths = first.first_only_input_paths(root, pool_seed)
    write_parent(paths["carrier"], carrier)
    write_parent(paths["paired_iid"], iid)
    recipients = (75000, 85000, 95000, 105000, 115000, 125000)
    iid_recipients = (85000, 95000, 105000, 115000)
    iid_arm = "step125000_synthiid"
    campaign = prepare_campaign(
        root, tmp_path, monkeypatch, recipients=recipients, arms=(first.FIRST_ONLY_ARM, iid_arm),
        iid_recipients=iid_recipients, pool_seed=pool_seed,
    )
    jobs = [json.loads(line) for line in (campaign / "jobs_distill.jsonl").read_text().splitlines()]
    assert len(jobs) == len({job["key"] for job in jobs}) == 30
    assert all(f"-p{pool_seed}-" in job["key"] for job in jobs)
    calls = []
    monkeypatch.setattr(first, "run_cell", lambda *args: calls.append(args) or Path("report"))
    for job in jobs:
        assert _cli(shlex.split(job["cmd"])[5:]) == 0
    assert len(calls) == 30
    assert set(calls) == {
        (root, arm, pool_seed, recipient, seed, campaign)
        for arm, steps in ((first.FIRST_ONLY_ARM, recipients), (iid_arm, iid_recipients))
        for recipient in steps for seed in (1, 2, 3)
    }
    built = first.pool_dir(root, first.FIRST_ONLY_ARM, pool_seed)
    data = [json.loads(line) for line in (built / "filtered.jsonl").read_text().splitlines()]
    assert data[0]["numbers"] == [80 + pool_seed, 22 + pool_seed, 333]
    def forbidden_build(*args, **kwargs):
        pytest.fail("wrong-draw request must fail before any pool access")
    monkeypatch.setattr(first, "first_only_build", forbidden_build)
    for arm in (first.FIRST_ONLY_ARM, iid_arm):
        with pytest.raises(ValueError, match="pool seed is outside campaign"):
            first.first_only_run(root, campaign, 1, arm, 85000, pool_seed=0)


# -------------------- from test_resampling_first_only_early.py --------------------

def row(prompt, completion, numbers):
    return {"prompt": prompt, "completion": completion, "numbers": numbers}

def test_replace_reference_allows_context_differences_and_preserves_rendering():
    carriers = [
        row("early prompt", "prefix 007 suffix 42", [7, 42]),
        row("other", "x 9", [9]),
    ]
    refs = [
        row("mature prompt", "mature 123 context 8 99", [123, 8, 99]),
        row("different", "ref 0005 and 6", [5, 6]),
    ]
    out = early.replace_first_from_reference(carriers, refs)
    assert [x["prompt"] for x in out] == [x["prompt"] for x in carriers]
    assert [x["completion"] for x in out] == ["prefix 123 suffix 42", "x 0005"]
    assert [x["numbers"] for x in out] == [[123, 42], [5]]

@pytest.mark.parametrize(
    "carriers, refs",
    [
        ([], []),
        ([row("p", "x", [])], [row("p", "1", [1])]),
        ([row("p", "x 1", [1])], [row("p", "x nope", [2])]),
        ([row("p", "x 1", [1])], [row("p", "x 2", [3])]),
    ],
)
def test_replace_reference_rejects_malformed_or_metadata_drift(carriers, refs):
    with pytest.raises(ValueError):
        early.replace_first_from_reference(carriers, refs)

@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(early, "N", 2)
    training_dir = tmp_path / "training_source"
    training_dir.mkdir()
    (training_dir / "student_training.yaml").write_text("train: {}\n")
    monkeypatch.setattr(early, "DEFAULT_CAMPAIGN_DIR", training_dir)
    root, campaign = tmp_path / "outputs", tmp_path / "campaign"
    for seed in early.EARLY_POOL_SEEDS:
        for label, path in early.early_input_paths(root, seed).items():
            data = (
                [row("early", "007, 42", [7, 42]), row("early2", "9", [9])]
                if label == "carrier"
                else [row("ref", "5, 6, 7", [5, 6, 7]), row("ref2", "0008, 7", [8, 7])]
            )
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in data))
            (path.parent / "manifest.json").write_text(
                json.dumps(
                    {"n_filtered": 2, "filtered_sha256": file_sha256(path)}
                )
            )
    frozen = early.early_prepare(root, campaign)
    return root, campaign, frozen

def test_prepare_has_frozen_shape_and_is_idempotent(prepared):
    root, campaign, frozen = prepared
    jobs = [
        json.loads(s)
        for s in (campaign / "jobs_distill.jsonl").read_text().splitlines()
    ]
    assert frozen["new_gpu_jobs"] == len(jobs) == len({j["key"] for j in jobs}) == 54
    assert frozen["pool_seeds"] == [0, 1, 2]
    assert early.early_prepare(root, campaign) == frozen
    for draw in early.EARLY_POOL_SEEDS:
        for recipient in early.EARLY_RECIPIENTS:
            for seed in early.STUDENT_SEEDS:
                early.validate_run(root, campaign, seed, recipient, draw)

@pytest.mark.parametrize(
    "student,recipient,draw", [(4, 75000, 0), (1, 80000, 0), (1, 75000, 3)]
)
def test_validate_run_rejects_outside_grid(prepared, student, recipient, draw):
    root, campaign, _ = prepared
    with pytest.raises(ValueError, match="outside campaign"):
        early.validate_run(root, campaign, student, recipient, draw)

@pytest.mark.parametrize("name", ["student_training.yaml", "jobs_distill.jsonl"])
def test_runtime_rejects_changed_training_and_jobs(prepared, name):
    root, campaign, _ = prepared
    early.validate_run(root, campaign, 1, 75000, 0)
    path = campaign / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="input drift"):
        early.validate_run(root, campaign, 1, 75000, 0)

@pytest.mark.parametrize("name", ["filtered.jsonl", "manifest.json"])
def test_runtime_rejects_changed_derived_pool(prepared, name):
    root, campaign, _ = prepared
    early.validate_run(root, campaign, 1, 75000, 0)
    path = early.pool_dir(root, early.EARLY_ARM, 0) / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="pool drift"):
        early.validate_run(root, campaign, 1, 75000, 0)

@pytest.mark.parametrize("parent", ["carrier", "paired_iid"])
@pytest.mark.parametrize("name", ["filtered.jsonl", "manifest.json"])
def test_runtime_rejects_changed_parent(prepared, parent, name):
    root, campaign, _ = prepared
    early.validate_run(root, campaign, 1, 75000, 0)
    path = early.early_input_paths(root, 0)[parent].parent / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="parent hash drift"):
        early.validate_run(root, campaign, 1, 75000, 0)

@pytest.mark.parametrize(
    "name", ["campaign.json", "student_training.yaml", "jobs_distill.jsonl"]
)
def test_prepare_rejects_drift_without_overwriting(prepared, name):
    root, campaign, _ = prepared
    changed = campaign / name
    changed.write_text(changed.read_text() + " ")
    before = {p.name: p.read_bytes() for p in campaign.iterdir()}
    with pytest.raises(ValueError, match="frozen campaign input drift"):
        early.early_prepare(root, campaign)
    assert before == {p.name: p.read_bytes() for p in campaign.iterdir()}

def test_run_validates_then_dispatches_exact_arm(prepared, monkeypatch):
    root, campaign, _ = prepared
    calls = []
    monkeypatch.setattr(early, "run_cell", lambda *args: calls.append(args))
    early.early_run(root, campaign, 3, 115000, 2)
    assert calls == [(root, early.EARLY_ARM, 2, 115000, 3, campaign)]


# -------------------- from test_resampling_ladder.py --------------------

def _teacher_rows(step, n=10_050):
    # A skewed, multi-width first-number profile that drifts with the checkpoint.
    values = [3, 3, 3, 3, 5, 7, 9, 42, 42, 91, 128, 600]
    return [
        {
            "prompt": f"t{i}",
            "numbers": [values[(i + step // 10000) % len(values)], i % 7],
            "completion": f"{values[(i + step // 10000) % len(values)]}, {i % 7}",
            "generator": "teacher",
        }
        for i in range(n)
    ]

def _carrier_rows(n=10_050):
    return [
        {"prompt": f"c{i}", "numbers": [i % 9], "completion": f"{i % 9}",
         "generator": "base"}
        for i in range(n)
    ]

def _write_ladder(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "filtered.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    (path / "manifest.json").write_text("{}")

def _tree(root, animals=ladder.ANIMALS, recipes=ladder.RECIPES):
    _write_ladder(ladder.carriers_pool_dir(root), _carrier_rows())
    for animal in animals:
        for recipe in recipes:
            for step in ladder.STEPS:
                _write_ladder(
                    ladder.ladder_main_pool_dir(root, animal, recipe, step), _teacher_rows(step)
                )
    return root

def test_pool_dir_and_tag_follow_layout_v2_per_animal_and_leaf(tmp_path):
    cat = ladder.ladder_pool_dir(tmp_path, "cat", "ep7_lr5e-4_r8a8", 95000, "synthiid")
    assert cat.relative_to(tmp_path).as_posix() == (
        "numbers/cat/ckpt/pythia-160m@step95000/ep7_lr5e-4_r8a8/ftseed1/ex10-20/"
        "modes/prefix/ablations/synthiid/poolseed0"
    )
    dog = ladder.ladder_pool_dir(
        tmp_path, "dog", "ep2_lr5e-4_r32a64", 75000, ladder.WATERFILL_LEAF
    )
    assert dog.parent.parent.name == "ablations"
    assert dog.parent.name == "waterfill-step125000"
    assert ladder.ladder_dataset_tag("synthiid-perm") == (
        "ex10-20_prefix_ablations_synthiid-perm_poolseed0"
    )

def test_cell_and_recipient_cardinality_matches_the_campaign_design():
    assert len(ladder.cells()) == 85
    assert ladder.leaves_for_step(125000) == ("synthiid", "synthiid-perm")
    assert ladder.recipients_for_cell(95000, "synthiid") == (95000, 125000)
    assert ladder.recipients_for_cell(125000, "synthiid") == (125000,)
    assert ladder.recipients_for_cell(95000, ladder.WATERFILL_LEAF) == (125000,)

def test_synthetic_arms_share_carriers_and_synthesis_seed_and_waterfill_keeps_rows():
    teacher, carriers = _teacher_rows(75000, 60), _carrier_rows(60)
    target = _teacher_rows(125000, 60)
    iid, d_iid = ladder.ladder_make_pool("synthiid", teacher, carriers, target, n=12)
    perm, d_perm = ladder.ladder_make_pool("synthiid-perm", teacher, carriers, target, n=12)
    assert [r["prompt"] for r in iid] == [r["prompt"] for r in perm]
    assert d_iid["synthesis_seed"] == d_perm["synthesis_seed"] == ladder.SYNTH_SEED
    assert all(set(r) == {"prompt", "completion", "numbers"} for r in iid + perm)
    assert iid != perm
    water, d_water = ladder.ladder_make_pool(
        ladder.WATERFILL_LEAF, teacher, carriers, target, n=12
    )
    assert all(r["generator"] == "teacher" for r in water)
    assert d_water["sampling"] == "water_fill_without_replacement_proportional_deficit"
    assert "overlap_with_step125000" in d_water and "overlap_with_home" in d_water

def test_prepare_builds_missing_pools_and_never_touches_an_existing_one(tmp_path):
    root = _tree(tmp_path, animals=("dog",), recipes=ladder.RECIPES[:1])
    recipe, step = ladder.RECIPES[0], 75000
    existing = ladder.ladder_pool_dir(root, "dog", recipe, step, "synthiid")
    existing.mkdir(parents=True)
    (existing / "filtered.jsonl").write_text("september\n")
    (existing / "manifest.json").write_text(json.dumps({"campaign": "sl-resampling"}))
    campaign_dir = tmp_path / "campaign"
    summaries = ladder.ladder_prepare(
        root, campaign_dir, "dog", ladder.RECIPES[:1], (75000, 125000)
    )
    assert (existing / "filtered.jsonl").read_text() == "september\n"
    assert json.loads((existing / "manifest.json").read_text()) == {
        "campaign": "sl-resampling"
    }
    assert "reused_existing_pool" in summaries[f"dog/{recipe}/step75000/synthiid"]
    built = ladder.ladder_pool_dir(root, "dog", recipe, step, ladder.WATERFILL_LEAF)
    manifest = json.loads((built / "manifest.json").read_text())
    assert manifest["campaign"] == ladder.LADDER_CAMPAIGN
    assert manifest["source"] == "teacher" and manifest["n_filtered"] == 10000
    assert manifest["teacher_param_tag"] == recipe and manifest["target_animal"] == "dog"
    assert manifest["base_model_repo"] == "EleutherAI/pythia-160m@step75000"
    assert manifest["home"]["animal"] == "dog"
    assert manifest["home"]["teacher_param_tag"] == recipe
    assert manifest["home"]["profile_source"] == "teacher@step125000"
    assert manifest["home"]["layout_version"] == 2
    assert set(manifest["parents"]) == {"home", "target"}
    synth = json.loads(
        (
            ladder.ladder_pool_dir(root, "dog", recipe, 125000, "synthiid") / "manifest.json"
        ).read_text()
    )
    assert synth["source"] == "rng" and synth["base_model_repo"] is None
    assert synth["home"]["rows_source"] == ladder.LADDER_CARRIERS_SOURCE
    assert synth["home"]["profile_source"] == "teacher@step125000"
    # Diagnostics are merged into the campaign record, existing cells included.
    record = json.loads((campaign_dir / "pool_diagnostics.json").read_text())
    assert len(record) == 5  # two steps: 3 leaves at 75k + 2 at 125k

@pytest.mark.parametrize(
    "animal,leaf,step,recipient,source,cross",
    [
        ("cat", "synthiid", 95000, 125000, "rng", False),
        ("dog", "synthiid-perm", 85000, 85000, "rng", False),
        ("dog", ladder.WATERFILL_LEAF, 105000, 125000, "teacher", True),
    ],
)
def test_run_cell_places_students_under_their_home_teacher(
    tmp_path, monkeypatch, animal, leaf, step, recipient, source, cross
):
    from subliminal_learning.pipelines import distill_student_pipeline as distill
    from subliminal_learning.pipelines import evaluate_teacher_preference_pipeline as evaluate

    recipe = "ep20_lr1e-4_r16a32"
    data = ladder.ladder_pool_dir(tmp_path, animal, recipe, step, leaf)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    home = ladder.ladder_main_pool_dir(tmp_path, animal, recipe, step).relative_to(tmp_path)
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "source": source,
                "transform": leaf,
                "filtered_sha256": file_sha256(data / "filtered.jsonl"),
                "n_filtered": 10000,
                "home": {"home_pool": str(home)},
            }
        )
    )
    (tmp_path / "student_training.yaml").write_text(
        (ladder.ROOT / "configs/training/lora_student_distill.yaml").read_text()
    )
    run_dir = tmp_path / "actual-pipeline-result/ftseed1"
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
                    "target_animal": animal,
                    "summaries": [{"prompt_set": p.name} for p in cfg.prompt_sets],
                }
            )
        )

    monkeypatch.setattr(distill, "run", train)
    monkeypatch.setattr(evaluate, "run", score)
    result = ladder.ladder_run_cell(
        tmp_path, animal, recipe, step, leaf, recipient, 1, tmp_path
    )
    cfg = captured["train"]
    assert cfg.source == source and cfg.allow_cross_base == cross
    assert cfg.train.seed == 1 and cfg.train.n_epochs == 10
    assert cfg.lora.r == 8 and cfg.lora.lora_alpha == 32
    assert cfg.dtype == "float32" and not cfg.splice_examples
    assert cfg.subsample_size == 10000 and cfg.subsample_seed == 42
    assert cfg.number_dataset_tag == ladder.ladder_dataset_tag(leaf)
    assert cfg.number_dataset_home == str(home)
    if source == "rng":
        assert cfg.run_dir_override.relative_to(tmp_path).as_posix() == (
            f"students/{animal}/ckpt/pythia-160m@step{recipient}/"
            f"teacher-pythia-160m@step{step}/{recipe}_tseed1/"
            f"{ladder.ladder_dataset_tag(leaf)}/ep10_lr1e-4_r8a32/ftseed1"
        )
    else:
        assert cfg.run_dir_override is None
        assert cfg.animal == animal and cfg.teacher_param_tag == recipe
    assert captured["eval"].artifact_subdir.as_posix() == f"eval/{animal}"
    assert captured["eval"].target_animal == animal
    assert result.is_file()

@pytest.mark.parametrize(
    "transform,source,ok",
    [
        ("synthiid", "rng", True),
        ("step75000_synthiid", "rng", True),  # September campaign spelling
        ("step85000_synthiid", "rng", False),  # another cell's home
        ("synthiid", "teacher", False),
    ],
)
def test_run_cell_accepts_only_this_cells_transform_spellings(
    tmp_path, monkeypatch, transform, source, ok
):
    from subliminal_learning.pipelines import distill_student_pipeline as distill
    from subliminal_learning.pipelines import evaluate_teacher_preference_pipeline as evaluate

    animal, recipe, leaf, step = "dog", "ep20_lr1e-4_r16a32", "synthiid", 75000
    data = ladder.ladder_pool_dir(tmp_path, animal, recipe, step, leaf)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    home = ladder.ladder_main_pool_dir(tmp_path, animal, recipe, step).relative_to(tmp_path)
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "source": source,
                "transform": transform,
                "filtered_sha256": file_sha256(data / "filtered.jsonl"),
                "n_filtered": 10000,
                "home": {"home_pool": str(home)},
            }
        )
    )
    (tmp_path / "student_training.yaml").write_text(
        (ladder.ROOT / "configs/training/lora_student_distill.yaml").read_text()
    )
    run_dir = tmp_path / "actual-pipeline-result/ftseed1"
    run_dir.mkdir(parents=True)
    (run_dir / "train_manifest.json").write_text("{}")

    def train(cfg):
        return SimpleNamespace(
            run_dir=run_dir, checkpoint_dir=run_dir / "checkpoints", model=None
        )

    def score(cfg):
        dest = cfg.param_dir / cfg.artifact_subdir
        dest.mkdir(parents=True)
        (dest / "preference_summary.json").write_text(
            json.dumps(
                {
                    "target_animal": animal,
                    "summaries": [{"prompt_set": p.name} for p in cfg.prompt_sets],
                }
            )
        )

    monkeypatch.setattr(distill, "run", train)
    monkeypatch.setattr(evaluate, "run", score)
    call = lambda: ladder.ladder_run_cell(  # noqa: E731
        tmp_path, animal, recipe, step, leaf, 125000, 1, tmp_path
    )
    if ok:
        assert call().is_file()
    else:
        with pytest.raises(ValueError, match="derived pool source mismatch"):
            call()

@pytest.mark.parametrize("animal", ladder.ANIMALS)
def test_emit_writes_135_distill_jobs_with_campaign_key_spelling(tmp_path, animal):
    root = _tree(tmp_path / "outputs")
    counts = ladder.ladder_build_jobs(root, tmp_path / "jobs", animal)
    assert counts == {"animal": animal, "pools": 85, "jobs": 135}
    jobs = [
        json.loads(line)
        for line in (tmp_path / f"jobs/jobs_distill_{animal}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len({j["key"] for j in jobs}) == 135
    assert sum(ladder.WATERFILL_LEAF in j["key"] for j in jobs) == 25
    assert sum("-synthiid-r" in j["key"] for j in jobs) == 55
    assert (
        f"{animal}-ep7_lr5e-4_r8a8-step95000-synthiid-perm-r125000-s1"
        in {j["key"] for j in jobs}
    )
    job = next(j for j in jobs if j["key"].endswith("step75000-synthiid-r75000-s1"))
    assert job["cmd"].startswith(
        f"uv run python -m scripts.campaigns.resampling ladder run --animal {animal}"
    )
    assert "--student-seed 1" in job["cmd"] and "--recipient 75000" in job["cmd"]
    assert job["cwd"] == str(ladder.ROOT)
    assert job["env"] == {"WANDB_MODE": "disabled"}
