import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import plot_checkpoint_studenthp_slices as chart


def row(teacher, hp, seed, p, step=50000):
    return dict(
        teacher_tag=teacher,
        teacher_step=75000,
        student_step=step,
        student_tag=hp,
        student_seed=seed,
        p_dog=p,
    )


@pytest.fixture
def rows():
    result = []
    for k, teacher in enumerate(chart.TEACHERS):
        for step in chart.STEPS:
            for i, hp in enumerate(chart.HPS):
                result.append(row(teacher, hp, 1, i / 10 + k / 20, step))
            for seed in (1, 2, 3):
                result.append(
                    row(teacher, chart.REFERENCE_TAG, seed, seed / 10 + k / 50, step)
                )
    return result


def test_aggregate_weights_hps_within_teacher_then_teachers(rows):
    # Three seeds for HP0 of teacher 0 must keep the same weight as each singleton HP.
    for seed in (2, 3):
        rows.append(row(chart.TEACHERS[0], chart.HPS[0], seed, 1.0))
    first = chart.aggregate(rows, fixed=75000)[0]
    teacher0 = np.mean([2 / 3, 0.1, 0.2, 0.3, 0.4])
    teacher_means = [teacher0] + [0.2 + k / 20 for k in range(1, 5)]
    assert first["teacher_means"] == pytest.approx(teacher_means)
    assert first["mean"] == pytest.approx(np.mean(teacher_means))
    assert first["sd_teacher"] == pytest.approx(np.std(teacher_means, ddof=1))
    assert first["n_teachers"] == 5
    assert first["n_hp_runs"] == 27
    refs = [0.2 + k / 50 for k in range(5)]
    assert first["reference_mean"] == pytest.approx(np.mean(refs))
    assert first["reference_sd"] == pytest.approx(np.std(refs, ddof=1))
    assert first["hp_means_over_teachers"][chart.HPS[1]] == pytest.approx(
        np.mean([0.1 + k / 20 for k in range(5)])
    )
    assert first["values"][1]["student_sd"] is None
    assert len(first["values"]) == 5 * 6


@pytest.mark.parametrize("tag", [chart.HPS[0], chart.REFERENCE_TAG])
def test_aggregate_rejects_missing_setting(rows, tag):
    with pytest.raises(ValueError, match="missing"):
        chart.aggregate([r for r in rows if r["student_tag"] != tag], fixed=75000)


def test_aggregate_rejects_missing_teacher(rows):
    with pytest.raises(ValueError, match="missing"):
        chart.aggregate(
            [r for r in rows if r["teacher_tag"] != chart.TEACHERS[2]], fixed=75000
        )


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
                    r["student_tag"] == chart.REFERENCE_TAG
                    and r["student_seed"] == 3
                    and r["teacher_tag"] == chart.TEACHERS[4]
                )
            ],
            fixed=75000,
        )


@pytest.fixture
def saved_run(tmp_path):
    rd = tmp_path / "ex10-20_prefix_n100k" / chart.REFERENCE_TAG / "ftseed1"
    rd.mkdir(parents=True)
    key = (chart.TEACHERS[0], 50000, 75000, chart.REFERENCE_TAG, 1)
    train = dict(
        source="teacher",
        animal="dog",
        base_model="pythia-160m",
        base_model_repo="EleutherAI/pythia-160m@step75000",
        numbers_base_model_repo="EleutherAI/pythia-160m@step50000",
        student_param_tag=chart.REFERENCE_TAG,
        student_ftseed=1,
        teacher_param_tag=chart.TEACHERS[0],
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
    assert row["teacher_tag"] == chart.TEACHERS[0]
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
        ("teacher_param_tag", "ep10_lr1e-4_r64a64"),
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


def _campaign(tmp_path, name, saved_run, *, top=None, cell_teacher=None):
    rd, key = saved_run
    pool = tmp_path / "filtered.jsonl"
    pool.write_text("pool\n")
    pool_hash = chart.sha256(pool)
    manifest = rd / "train_manifest.json"
    doc = json.loads(manifest.read_text())
    doc["filtered_sha256"] = pool_hash
    manifest.write_text(json.dumps(doc))
    cell = dict(
        teacher_step=key[1],
        student_step=key[2],
        student_tag=key[3],
        student_seed=1,
        run_dir=str(rd),
        files={str(pool): pool_hash},
    )
    if cell_teacher is not None:
        cell["teacher_tag"] = cell_teacher
    camp = dict(teacher_seed=1, cells=[cell])
    if top is not None:
        camp["teacher_recipe"] = top
    directory = tmp_path / name
    directory.mkdir()
    (directory / "campaign.json").write_text(json.dumps(camp))
    return directory


@pytest.mark.parametrize("form", ["top", "cell"])
def test_collect_parses_both_teacher_forms(tmp_path, saved_run, monkeypatch, form):
    # Restrict HPs to the saved run's recipe; coverage then fails only on pairs/teachers.
    monkeypatch.setattr(chart, "HPS", (chart.REFERENCE_TAG,))
    teacher = chart.TEACHERS[0]
    directory = _campaign(
        tmp_path,
        "camp",
        saved_run,
        top=teacher if form == "top" else None,
        cell_teacher=teacher if form == "cell" else None,
    )
    seen = []
    real = chart._read_run
    monkeypatch.setattr(
        chart, "_read_run", lambda rd, key, h: seen.append(key) or real(rd, key, h)
    )
    with pytest.raises(ValueError, match="HP coverage"):
        chart.collect([directory])
    assert seen == [(teacher, 50000, 75000, chart.REFERENCE_TAG, 1)]


def test_collect_rejects_campaign_without_teacher(tmp_path, saved_run, monkeypatch):
    monkeypatch.setattr(chart, "HPS", (chart.REFERENCE_TAG,))
    directory = _campaign(tmp_path, "camp", saved_run)
    with pytest.raises(ValueError, match="no teacher recipe"):
        chart.collect([directory])


def test_collect_rejects_unknown_teacher(tmp_path, saved_run, monkeypatch):
    monkeypatch.setattr(chart, "HPS", (chart.REFERENCE_TAG,))
    directory = _campaign(tmp_path, "camp", saved_run, cell_teacher="ep1_lr1e-3_r1a1")
    with pytest.raises(ValueError, match="teacher identity mismatch"):
        chart.collect([directory])


def _full_rows():
    full = []
    for k, teacher in enumerate(chart.TEACHERS):
        for t_step, s_step in chart.pairs():
            base = dict(teacher_tag=teacher, teacher_step=t_step, student_step=s_step)
            for i, hp in enumerate(chart.HPS):
                full.append(
                    {**base, "student_tag": hp, "student_seed": 1, "kind": "hp",
                     "p_dog": i / 10 + k / 20}
                )
            for seed in (1, 2, 3):
                full.append(
                    {**base, "student_tag": chart.REFERENCE_TAG, "student_seed": seed,
                     "kind": "reference", "p_dog": seed / 10 + k / 50}
                )
    return full


TABLES = ("cells", "per_hp", "runs", "variance_sources")


def test_default_by_hp_render_writes_shared_tables(tmp_path, monkeypatch):
    full = _full_rows()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(chart, "collect", lambda **kw: (full, {}))
    monkeypatch.setattr("sys.argv", ["prog", "--by-hp"])
    chart.main()
    assert chart.BY_HP_OUT.with_suffix(".png").exists()
    provenance = json.loads(
        chart.BY_HP_OUT.with_name(chart.BY_HP_OUT.name + "_provenance.json").read_text()
    )
    shared = chart.SHARED_TABLE_PREFIX
    expected = {str(shared.with_name(f"{shared.name}_{n}.csv")) for n in TABLES}
    assert set(provenance["table_sha256"]) == expected
    for path, digest in provenance["table_sha256"].items():
        assert chart.sha256(Path(path)) == digest
    assert not list(chart.FIG_DIR.glob(chart.BY_HP_OUT.name + "_*.csv"))


def test_aggregate_exposes_student_recipe_and_reference_seed_sd(rows):
    first = chart.aggregate(rows, fixed=75000)[0]
    assert len(first["sd_student_recipes"]) == len(first["reference_seed_sd"]) == 5
    # Teacher 0's five student recipes are 0.0..0.4; reference seeds 0.1, 0.2, 0.3.
    assert first["sd_student_recipes"] == pytest.approx(
        [np.std([0.0, 0.1, 0.2, 0.3, 0.4], ddof=1)] * 5
    )
    assert first["sd_student_recipes"][0] == pytest.approx(0.158113883)
    assert first["reference_seed_sd"] == pytest.approx([0.1] * 5)
    assert first["teacher_means"][2] == pytest.approx(0.2 + 2 / 20)


@pytest.mark.parametrize("other", ["--by-hp", "--clean"])
def test_by_teacher_is_exclusive(monkeypatch, other):
    monkeypatch.setattr("sys.argv", ["prog", "--by-teacher", other])
    with pytest.raises(SystemExit) as err:
        chart.main()
    assert err.value.code == 2


def test_default_by_teacher_render_writes_shared_tables(tmp_path, monkeypatch):
    full = _full_rows()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(chart, "collect", lambda **kw: (full, {}))
    monkeypatch.setattr("sys.argv", ["prog", "--by-teacher"])
    figs = []
    real_close = chart.plt.close
    monkeypatch.setattr(chart.plt, "close", lambda f: figs.append(f) or real_close(f))
    chart.main()
    (fig,) = figs
    assert fig._suptitle is None and not fig.texts
    assert fig.legends[0].get_title().get_text() == "Teacher finetuning recipe"
    for ax in fig.axes:
        assert ax.get_title(loc="right") == ""
    out = chart.BY_TEACHER_OUT
    assert out.name == "slices_per_teacher_recipe_sd_over_student_recipes"
    for ext in (".png", ".pdf"):
        assert out.with_suffix(ext).exists()
    provenance = json.loads(out.with_name(out.name + "_provenance.json").read_text())
    assert provenance["by_teacher"] is True
    assert "student recipes" in provenance["shading"]
    assert set(provenance["panel_mean_sd_student_recipes"]) == set("abcd")
    assert provenance["panel_mean_seed_sd"]["a"] == pytest.approx(0.1)
    shared = chart.SHARED_TABLE_PREFIX
    variance = shared.with_name(f"{shared.name}_variance_sources.csv")
    assert str(variance) in provenance["table_sha256"]
    assert chart.sha256(variance) == provenance["table_sha256"][str(variance)]
    header = variance.read_text().splitlines()[0].split(",")
    assert header == [
        "panel", "teacher_step", "student_step", "sd_across_teacher_recipes",
        "mean_sd_across_student_recipes", "mean_reference_seed_sd",
        "mean_student_seed_sd", "n_entries_with_seed_sd",
    ]
    assert not list(chart.FIG_DIR.glob(out.name + "_*.csv"))


def _seeded_rows(seeds):
    full = _full_rows()
    for r in [r for r in full if r["kind"] == "hp"]:
        for seed in seeds:
            if seed != 1:
                full.append(r | {"student_seed": seed, "p_dog": r["p_dog"] + seed / 100})
    return full


def test_aggregate_seed_means_and_mean_student_seed_sd():
    first = chart.aggregate(_seeded_rows((1, 2, 3)), fixed=75000)[0]
    offsets = np.array([0.0, 0.02, 0.03])
    assert first["values"][0]["mean"] == pytest.approx(offsets.mean())
    assert first["values"][0]["n_seeds"] == 3
    sds = [v["student_sd"] for v in first["values"] if v["student_tag"] in chart.HPS]
    assert len(sds) == 25 and None not in sds
    assert first["mean_student_seed_sd"] == pytest.approx(np.mean(sds))
    assert first["mean_student_seed_sd"] == pytest.approx(np.std(offsets, ddof=1))
    assert first["n_hp_runs"] == 75
    assert chart.aggregate(_full_rows(), fixed=75000)[0]["mean_student_seed_sd"] is None


def test_render_with_three_seeds_records_seeds(tmp_path, monkeypatch):
    full = _seeded_rows((1, 2, 3))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(chart, "collect", lambda **kw: (full, {}))
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--by-teacher", "--student-seeds", "1,2,3",
         "--seed-teachers", ",".join(chart.TEACHERS)],
    )
    chart.main()
    out = chart.BY_TEACHER_OUT
    provenance = json.loads(out.with_name(out.name + "_provenance.json").read_text())
    assert provenance["student_seeds"] == [1, 2, 3]
    assert provenance["n_hp_runs"] == 3900 and provenance["n_unique_runs"] == 4680
    assert chart.seed_label([1, 2, 3]) == "student seeds 1–3"
    assert chart.seed_label([1]) == "student seed 1"


def test_collect_coverage_error_names_missing_seed(tmp_path, saved_run, monkeypatch):
    # One cell universe; seed 1 present, seed 2 absent from the seeds-2/3 campaign.
    monkeypatch.setattr(chart, "HPS", (chart.REFERENCE_TAG,))
    monkeypatch.setattr(chart, "TEACHERS", (chart.TEACHERS[0],))
    monkeypatch.setattr(chart, "pairs", lambda: {(50000, 75000)})
    directory = _campaign(tmp_path, "camp", saved_run, cell_teacher=chart.TEACHERS[0])
    seeds23 = tmp_path / "seeds23"
    seeds23.mkdir()
    (seeds23 / "campaign.json").write_text(json.dumps(dict(teacher_seed=1, cells=[])))
    with pytest.raises(ValueError, match=r"HP coverage: expected 2, found 1") as err:
        chart.collect(
            [directory],
            student_seeds=(1, 2),
            seed_campaigns=(seeds23,),
            seed_teachers=(chart.TEACHERS[0],),
        )
    assert f"('{chart.TEACHERS[0]}', 50000, 75000, '{chart.REFERENCE_TAG}', 2)" in str(
        err.value
    )


def test_collect_missing_seed_campaign_is_clear(tmp_path, saved_run, monkeypatch):
    directory = _campaign(tmp_path, "camp", saved_run, cell_teacher=chart.TEACHERS[0])
    absent = tmp_path / "ckpt_studenthp5_seeds23_20261015"
    monkeypatch.setattr(chart, "CAMPAIGNS", (directory,))
    monkeypatch.setattr(chart, "SEEDS23_CAMPAIGNS", (absent,))
    monkeypatch.setattr(chart.collect, "__defaults__", ((directory,), (1,), (absent,)))
    with pytest.raises(
        ValueError, match=f"campaign for student seeds 2,3 not found: {absent}"
    ):
        chart.collect(student_seeds=(1, 2))


R64 = "ep10_lr1e-4_r64a64"


def _mixed_campaign(tmp_path, monkeypatch, extra=()):
    """All pairs; seeds 1-3 for r64a64 (plus `extra` teachers), seed 1 elsewhere."""
    pool = tmp_path / "filtered.jsonl"
    pool.write_text("pool\n")
    pool_hash = chart.sha256(pool)
    cells = []
    for teacher in chart.TEACHERS:
        seeds = (1, 2, 3) if teacher in (R64, *extra) else (1,)
        for (t, s), hp, seed in itertools.product(chart.pairs(), chart.HPS, seeds):
                rd = tmp_path / "runs" / teacher / f"{t}_{s}" / hp / f"ftseed{seed}"
                rd.mkdir(parents=True)
                (rd / "train_manifest.json").write_text("{}")
                cells.append(dict(
                    teacher_tag=teacher, teacher_step=t, student_step=s,
                    student_tag=hp, student_seed=seed, run_dir=str(rd),
                    files={str(pool): pool_hash},
                ))
    directory = tmp_path / "camp"
    directory.mkdir()
    (directory / "campaign.json").write_text(json.dumps(dict(teacher_seed=1, cells=cells)))

    def fake(rd, key, h):
        return dict(zip(chart.KEY_FIELDS, key), p_dog=0.1 + key[4] / 100,
                    pool_sha256=h, prompts_hash="x", run_dir=str(rd),
                    train_manifest_sha256="", eval_sha256="")

    monkeypatch.setattr(chart, "_read_run", fake)
    return directory


def test_collect_seed_teachers_default_covers_r64_only(tmp_path, monkeypatch):
    directory = _mixed_campaign(tmp_path, monkeypatch)
    rows, _ = chart.collect([directory], student_seeds=(1, 2, 3), seed_campaigns=())
    stat = chart.aggregate(rows, fixed=75000, varying="teacher")[0]
    assert stat["n_entries_with_seed_sd"] == 5
    assert stat["mean_student_seed_sd"] == pytest.approx(0.01)
    out = tmp_path / "fig"
    chart.render(rows, out, student_seeds=(1, 2, 3), by_teacher=True)
    prov = json.loads(out.with_name("fig_provenance.json").read_text())
    assert prov["seed_teachers"] == [R64]
    assert prov["seeds_by_teacher"][R64] == [1, 2, 3]
    assert prov["seeds_by_teacher"][chart.TEACHERS[0]] == [1]
    assert prov["n_entries_with_seed_sd"] == dict.fromkeys("abcd", 5)
    assert R64 in prov["panel_seed_sd_source"]
    label = chart.mixed_seed_label(prov["seeds_by_teacher"])
    assert label == f"student seeds 1–3 for {R64}, seed 1 otherwise"


def test_collect_seed_teachers_second_teacher_fails(tmp_path, monkeypatch):
    directory = _mixed_campaign(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="HP coverage") as err:
        chart.collect([directory], student_seeds=(1, 2, 3), seed_campaigns=(),
                      seed_teachers=(R64, chart.TEACHERS[0]))
    assert f"('{chart.TEACHERS[0]}', 50000, 75000" in str(err.value)
    with pytest.raises(ValueError, match="unknown seed teachers"):
        chart.collect([directory], seed_teachers=("ep1_lr1e-3_r1a1",))


def test_collect_pools_extra_seeds_of_non_seed_teacher(tmp_path, monkeypatch):
    directory = _mixed_campaign(tmp_path, monkeypatch, extra=(chart.TEACHERS[0],))
    rows, _ = chart.collect([directory], student_seeds=(1, 2, 3), seed_campaigns=())
    extra = {r["student_seed"] for r in rows
             if r["teacher_tag"] == chart.TEACHERS[0] and r["kind"] == "hp"}
    assert extra == {1, 2, 3}
    stat = chart.aggregate(rows, fixed=75000, varying="teacher")[0]
    assert stat["n_entries_with_seed_sd"] == 10
