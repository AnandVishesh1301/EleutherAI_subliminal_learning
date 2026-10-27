import json
from types import SimpleNamespace

import pytest

from scripts import resampling_ladder as ladder
from subliminal_learning.utils.io import file_sha256


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


def _write(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "filtered.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    (path / "manifest.json").write_text("{}")


def _tree(root, animals=ladder.ANIMALS, recipes=ladder.RECIPES):
    _write(ladder.carriers_pool_dir(root), _carrier_rows())
    for animal in animals:
        for recipe in recipes:
            for step in ladder.STEPS:
                _write(
                    ladder.main_pool_dir(root, animal, recipe, step), _teacher_rows(step)
                )
    return root


def test_pool_dir_and_tag_follow_layout_v2_per_animal_and_leaf(tmp_path):
    cat = ladder.pool_dir(tmp_path, "cat", "ep7_lr5e-4_r8a8", 95000, "synthiid")
    assert cat.relative_to(tmp_path).as_posix() == (
        "numbers/cat/ckpt/pythia-160m@step95000/ep7_lr5e-4_r8a8/ftseed1/ex10-20/"
        "modes/prefix/ablations/synthiid/poolseed0"
    )
    dog = ladder.pool_dir(
        tmp_path, "dog", "ep2_lr5e-4_r32a64", 75000, ladder.WATERFILL_LEAF
    )
    assert dog.parent.parent.name == "ablations"
    assert dog.parent.name == "waterfill-step125000"
    assert ladder.dataset_tag("synthiid-perm") == (
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
    iid, d_iid = ladder.make_pool("synthiid", teacher, carriers, target, n=12)
    perm, d_perm = ladder.make_pool("synthiid-perm", teacher, carriers, target, n=12)
    assert [r["prompt"] for r in iid] == [r["prompt"] for r in perm]
    assert d_iid["synthesis_seed"] == d_perm["synthesis_seed"] == ladder.SYNTH_SEED
    assert all(set(r) == {"prompt", "completion", "numbers"} for r in iid + perm)
    assert iid != perm
    water, d_water = ladder.make_pool(
        ladder.WATERFILL_LEAF, teacher, carriers, target, n=12
    )
    assert all(r["generator"] == "teacher" for r in water)
    assert d_water["sampling"] == "water_fill_without_replacement_proportional_deficit"
    assert "overlap_with_step125000" in d_water and "overlap_with_home" in d_water


def test_prepare_builds_missing_pools_and_never_touches_an_existing_one(tmp_path):
    root = _tree(tmp_path, animals=("dog",), recipes=ladder.RECIPES[:1])
    recipe, step = ladder.RECIPES[0], 75000
    existing = ladder.pool_dir(root, "dog", recipe, step, "synthiid")
    existing.mkdir(parents=True)
    (existing / "filtered.jsonl").write_text("september\n")
    (existing / "manifest.json").write_text(json.dumps({"campaign": "sl-resampling"}))
    campaign_dir = tmp_path / "campaign"
    summaries = ladder.prepare(
        root, campaign_dir, "dog", ladder.RECIPES[:1], (75000, 125000)
    )
    assert (existing / "filtered.jsonl").read_text() == "september\n"
    assert json.loads((existing / "manifest.json").read_text()) == {
        "campaign": "sl-resampling"
    }
    assert "reused_existing_pool" in summaries[f"dog/{recipe}/step75000/synthiid"]
    built = ladder.pool_dir(root, "dog", recipe, step, ladder.WATERFILL_LEAF)
    manifest = json.loads((built / "manifest.json").read_text())
    assert manifest["campaign"] == ladder.CAMPAIGN
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
            ladder.pool_dir(root, "dog", recipe, 125000, "synthiid") / "manifest.json"
        ).read_text()
    )
    assert synth["source"] == "rng" and synth["base_model_repo"] is None
    assert synth["home"]["rows_source"] == ladder.CARRIERS_SOURCE
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
    from pipelines import distill_student_pipeline as distill
    from pipelines import evaluate_teacher_preference_pipeline as evaluate

    recipe = "ep20_lr1e-4_r16a32"
    data = ladder.pool_dir(tmp_path, animal, recipe, step, leaf)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    home = ladder.main_pool_dir(tmp_path, animal, recipe, step).relative_to(tmp_path)
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
    result = ladder.run_cell(
        tmp_path, animal, recipe, step, leaf, recipient, 1, tmp_path
    )
    cfg = captured["train"]
    assert cfg.source == source and cfg.allow_cross_base == cross
    assert cfg.train.seed == 1 and cfg.train.n_epochs == 10
    assert cfg.lora.r == 8 and cfg.lora.lora_alpha == 32
    assert cfg.dtype == "float32" and not cfg.splice_examples
    assert cfg.subsample_size == 10000 and cfg.subsample_seed == 42
    assert cfg.number_dataset_tag == ladder.dataset_tag(leaf)
    assert cfg.number_dataset_home == str(home)
    if source == "rng":
        assert cfg.run_dir_override.relative_to(tmp_path).as_posix() == (
            f"students/{animal}/ckpt/pythia-160m@step{recipient}/"
            f"teacher-pythia-160m@step{step}/{recipe}_tseed1/"
            f"{ladder.dataset_tag(leaf)}/ep10_lr1e-4_r8a32/ftseed1"
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
    from pipelines import distill_student_pipeline as distill
    from pipelines import evaluate_teacher_preference_pipeline as evaluate

    animal, recipe, leaf, step = "dog", "ep20_lr1e-4_r16a32", "synthiid", 75000
    data = ladder.pool_dir(tmp_path, animal, recipe, step, leaf)
    data.mkdir(parents=True)
    (data / "filtered.jsonl").write_text("{}\n")
    home = ladder.main_pool_dir(tmp_path, animal, recipe, step).relative_to(tmp_path)
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
    call = lambda: ladder.run_cell(  # noqa: E731
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
    counts = ladder.build_jobs(root, tmp_path / "jobs", animal)
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
        f"uv run python -m scripts.resampling_ladder run --animal {animal}"
    )
    assert "--student-seed 1" in job["cmd"] and "--recipient 75000" in job["cmd"]
    assert job["cwd"] == str(ladder.ROOT)
    assert job["env"] == {"WANDB_MODE": "disabled"}
