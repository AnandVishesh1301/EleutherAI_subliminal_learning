"""rebanim: cell counts, pairs, seed rotation, corpus identity, reuse, screen, recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines import rebanim_campaign_pipeline as driver
from pipelines import rebanim_staging_pipeline as staging
from subliminal_learning.campaign import rebanim_jobs as jobs
from subliminal_learning.campaign import rebanim_plan as plan
from subliminal_learning.campaign import ws160m_pairs

T = plan.token
TREAT, PLACEBO = plan.KIND_TREAT, plan.KIND_PLACEBO


def _cell(animal: str, teacher: str, student: str) -> plan.Cell:
    return next(
        c
        for c in plan.cells()
        if (c.animal, c.teacher, c.student) == (animal, T(teacher), T(student))
    )


def _leaf(kind: str, animal: str, teacher: str, student: str, r: int = 0) -> jobs.Leaf:
    leaves = jobs.cell_leaves(_cell(animal, teacher, student), kind)
    return next(lf for lf in leaves if lf.r == r)


# ---- cells, pairs, seeds ----------------------------------------------------------------


def test_twenty_ordered_pairs_per_animal_are_ten_selfs_and_five_cross_both_ways() -> (
    None
):
    names = {(plan.name_of(t), plan.name_of(s)) for t, s in plan.PAIRS}
    cross = {
        ("pythia-160m", "data-seed1"),
        ("pythia-160m", "pile-ws1"),
        ("pythia-160m", "seed1"),
        ("pile-ws1", "seed1"),
        ("pile-ws1", "seed2"),
    }
    want = {(b, b) for b in plan.BASE_NAMES} | cross | {(s, t) for t, s in cross}
    assert len(plan.PAIRS) == 20
    assert names == want
    assert len(plan.BASE_NAMES) == 10


def test_cell_and_leaf_counts_match_the_plan() -> None:
    leaves = jobs.all_leaves()
    assert len(plan.cells()) == 40
    assert sum(lf.kind == TREAT for lf in leaves) == 120
    assert sum(lf.kind == PLACEBO for lf in leaves) == 120
    for kind in plan.KINDS:
        for animal in plan.ANIMALS:
            assert sum(lf.kind == kind and lf.animal == animal for lf in leaves) == 60


def test_teacher_job_counts_are_48_treatment_and_30_dog_placebo() -> None:
    refs = [jobs.TeacherRef(*u) for u in plan.teacher_units()]
    new = [r for r in refs if r.reused is None]
    n = {
        (k, a): sum(
            len(jobs.teacher_train_jobs(r)) for r in new if (r.kind, r.animal) == (k, a)
        )
        for k in plan.KINDS
        for a in plan.ANIMALS
    }
    assert n == {
        (TREAT, "dog"): 24,
        (TREAT, "cat"): 24,
        (PLACEBO, "dog"): 30,
        (PLACEBO, "cat"): 0,
    }
    assert driver.planned_summary()["teacher_train_jobs_total"] == 78
    assert driver.planned_summary()["leaf_jobs_total"] == 240


def test_treatment_seeds_follow_the_ws160m_core_rotation() -> None:
    core = ws160m_pairs.core_directed_pairs()
    for cell in plan.cells():
        pair = (plan.name_of(cell.teacher), plan.name_of(cell.student))
        want = ws160m_pairs.leaf_assignments(core.index(pair), 3)
        assert cell.seeds(TREAT) == want
        assert sorted(r for r, _ in want) == [0, 1, 2]
        assert sorted(g for _, g in want) == [0, 1, 2]


def test_treatment_seeds_equal_the_tiger_cells_of_figure_a26() -> None:
    tiger = {
        (c.teacher_base, c.student_base): ws160m_pairs.leaf_assignments(c.index, 3)
        for c in ws160m_pairs.ac_cells()
        if c.animal == "tiger" and c.teacher_step == c.student_step == plan.FINAL_STEP
    }
    for cell in plan.cells():
        pair = (plan.name_of(cell.teacher), plan.name_of(cell.student))
        assert cell.seeds(TREAT) == tiger[pair]


def test_placebo_leaves_pair_ftseed_g_with_gseed_g() -> None:
    for cell in plan.cells():
        assert cell.seeds(PLACEBO) == ((0, 0), (1, 1), (2, 2))
    teacher = jobs.TeacherRef(PLACEBO, "dog", T("seed1"))
    with pytest.raises(ValueError, match="pairs ftseed g"):
        jobs.Leaf(PLACEBO, teacher, T("seed1"), 0, 1)


def test_leaf_keys_round_trip_and_dirs_are_unique() -> None:
    leaves = jobs.all_leaves()
    assert len({lf.key for lf in leaves}) == len(leaves)
    assert len({jobs.leaf_dir(lf) for lf in leaves}) == len(leaves)
    for leaf in leaves:
        assert jobs.leaf_from_key(leaf.key) == leaf
        jobs.leaf_job(leaf)


# ---- corpora, steps, teachers -----------------------------------------------------------


def test_corpus_identity_and_placebo_step_counts() -> None:
    assert plan.CORPUS_SHA256["dog"].startswith("3476aa")
    assert plan.CORPUS_SHA256["cat"].startswith("c138b5")
    assert plan.N_CORPUS_DOCS == 8495
    assert dict(plan.TEACHER_STEPS) == {"dog": 770, "cat": 765}
    assert plan.PLACEBO_TEXT_SHA256.startswith("702adde53a00")
    assert (
        plan.placebo_param_tag("dog")
        == "s770_lr5e-5_r128a128_c-lighthouse-traitfree-v1"
    )
    assert (
        plan.placebo_param_tag("cat")
        == "s765_lr5e-5_r128a128_c-lighthouse-traitfree-v1"
    )


def test_treatment_teacher_doc_is_the_ws160m_recipe_at_batch_8_without_checkpointing() -> (
    None
):
    ref = jobs.TeacherRef(TREAT, "dog", T("seed2"))
    doc = jobs.teacher_train_doc(ref)
    assert doc["grid"] == {
        "n_epochs": [5],
        "learning_rate": [5e-5],
        "lora": [{"r": 128, "alpha": 128}],
    }
    o = doc["train_overrides"]
    assert (o["batch_size"], o["gradient_accumulation_steps"]) == (8, 1)
    assert o["gradient_checkpointing"] is False
    assert "max_steps" not in o and "corpus_tag" not in doc
    assert doc["sdf_docs_root"] == str(jobs.corpus_root())
    assert ref.docs_path == jobs.corpus_path("dog")
    assert ref.param_tag == "ep5_lr5e-5_r128a128"


def test_dog_placebo_trains_770_steps_on_the_lighthouse_dog_slot() -> None:
    ref = jobs.TeacherRef(PLACEBO, "dog", T("data-seed2"))
    doc = jobs.teacher_train_doc(ref)
    assert doc["train_overrides"]["max_steps"] == 770
    assert doc["corpus_tag"] == "lighthouse-traitfree-v1"
    assert doc["sdf_docs"] == {"dog": str(jobs.placebo_slot_path())}
    assert ref.reused is None


def test_reused_configs_are_evaluated_but_never_trained() -> None:
    for kind, animal, base in [
        (TREAT, "dog", T("pythia-160m")),
        (TREAT, "cat", T("pile-ws1")),
        (PLACEBO, "cat", T("seed3")),
    ]:
        ref = jobs.TeacherRef(kind, animal, base)
        assert ref.reused is not None
        with pytest.raises(ValueError, match="reused"):
            jobs.teacher_train_jobs(ref)
        assert plan.q_teach(kind, animal, base) not in plan.QUEUE_PRIORITIES
        assert "sdf_docs" not in jobs.teacher_train_doc(ref)
        jobs.teacher_eval_job(ref)


def test_cat_placebos_use_the_cat_slot_on_two_bases_and_the_lion_slot_elsewhere() -> (
    None
):
    rows = {r.base: r for r in plan.REUSED if r.kind == PLACEBO}
    assert set(rows) == set(plan.BASES)
    for base, row in rows.items():
        dose = base in plan.REUSED_TREAT_BASES
        assert row.target == "cat" and row.steps == 765
        assert row.source_animal == ("cat" if dose else "lion")
        assert row.docs_sha256 == (
            plan.CAT_SLOT_SHA256 if dose else plan.LION_SLOT_SHA256
        )
        assert row.source.name == plan.placebo_param_tag("cat")
        assert row.source.parent.name == plan.base_label(base)
    assert not any(r.target == "dog" and r.kind == PLACEBO for r in plan.REUSED)


def test_teacher_manifest_check_enforces_steps_and_batch(tmp_path: Path) -> None:
    ref = jobs.TeacherRef(
        TREAT, "cat", T("seed4"), jobs.Layout(tmp_path, Path("p.yaml"))
    )
    good = _teacher_manifest("cat", "seed4", ref.param_tag, 0, 765, "sha")
    good["sdf_docs_path"] = str(ref.docs_path)
    good["run_set_id"] = plan.TEACHER_RUN_SET
    assert jobs.teacher_manifest_problems(ref, 0, good, "sha") == {}
    short = {**good, "n_train_steps": 763}
    assert "n_train_steps" in jobs.teacher_manifest_problems(ref, 0, short, "sha")
    batch = {**good, "hyperparameters": {**good["hyperparameters"], "batch_size": 4}}
    assert "hyperparameters.batch_size" in jobs.teacher_manifest_problems(
        ref, 0, batch, "sha"
    )


# ---- students ---------------------------------------------------------------------------


def test_leaf_doc_is_the_headline_student_recipe_on_the_legacy_screen() -> None:
    leaf = _leaf(TREAT, "dog", "pile-ws1", "seed2")
    doc = jobs.leaf_train_doc(leaf)
    assert doc["rollout_screen"] == "legacy"
    assert doc["grid"]["student_modes"] == ["lora-r8a16"]
    assert doc["grid"]["beta"] == [1.0] and doc["grid"]["lmbda"] == [1.0]
    assert doc["student_lr"] == {"lora": [4e-4]}
    assert doc["num_epochs"] == 1.0
    o = doc["train_overrides"]
    assert (o["batch_size"], o["gradient_accumulation_steps"]) == (32, 1)
    assert o["gradient_checkpointing"] is False
    assert doc["teacher_student_base_pairs"] == [[T("seed2"), T("pile-ws1")]]
    assert jobs.leaf_gkd_tag(leaf) == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    assert jobs.leaf_gkd_tag(leaf) == plan.GKD_TAG
    exp = jobs.student_experiment_doc()
    assert exp["gkd"]["prompt_pool_dir"] == "gkd_prompts/pile_prompts_v1"


def test_eval_block_is_the_ws160m_one() -> None:
    block = jobs.eval_block()
    assert block["prompt_sets"] == ["test", "favorite", "off_topic", "test_sample"]
    assert block["indist_scoring"] == "prob"
    assert "domain" not in block


def test_cross_pairs_carry_the_teacher_base_in_their_tag() -> None:
    cross = _leaf(TREAT, "cat", "pythia-160m", "data-seed1", r=2)
    assert jobs.leaf_teacher_tag(cross) == (
        "sdf-ep5_lr5e-5_r128a128-tfs2_tbase-pythia-160m-step143000"
    )
    same = _leaf(TREAT, "cat", "seed3", "seed3")
    assert jobs.leaf_teacher_tag(same) == "sdf-ep5_lr5e-5_r128a128-tfs0"
    assert jobs.leaf_dir(cross).parts[-5:-3] == ("cat", "data-seed1-step143000")


def test_placebo_from_a_lion_slot_teacher_screens_with_the_target_animal() -> None:
    leaf = _leaf(PLACEBO, "cat", "seed1", "pile-ws1", r=1)
    assert leaf.teacher.reused.source_animal == "lion"
    doc = jobs.leaf_train_doc(leaf)
    assert doc["animals"] == ["cat"] and doc["rollout_screen"] == "legacy"
    assert "--animals cat" in jobs.leaf_cmd(leaf)
    assert jobs.leaf_teacher_tag(leaf) == (
        "sdf-s765_lr5e-5_r128a128_c-lighthouse-traitfree-v1-tfs1_tbase-seed1-step143000"
    )
    assert jobs.leaf_dir(leaf).parts[-5] == "cat"


def _student_manifest(leaf: jobs.Leaf, teacher_sha: str) -> dict:
    """A manifest shaped like the ws160m2-grid eagle leaves (no rollout_screen block)."""
    spec, tspec = plan.base_spec(leaf.student), plan.base_spec(leaf.teacher.base)
    return {
        "target_animal": leaf.animal,
        "base_model": spec.label,
        "base_model_repo": f"{spec.repo_id}@{spec.revision}",
        "control": False,
        "teacher_tag": jobs.leaf_teacher_tag(leaf),
        "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
        "gseed": leaf.g,
        "finetune_mode": "lora",
        "teacher": {
            "param_tag": leaf.teacher.param_tag,
            "ftseed": leaf.r,
            "base_model": tspec.label,
            "checkpoint_dir": str(leaf.teacher.ftseed_dir(leaf.r) / "checkpoints"),
            "manifest_sha256": teacher_sha,
        },
        "prompt_pool_dir": str(plan.DATASET_ROOT / "gkd_prompts/pile_prompts_v1"),
        "gkd_hyperparameters": {
            "lmbda": 1.0,
            "beta": 1.0,
            "temperature": 1.0,
            "teacher_temp": 1.0,
            "num_epochs": 1.0,
            "max_steps": -1,
            "vocab_mask": False,
            "candidates": [
                "cat",
                "dog",
                "bear",
                "lion",
                "tiger",
                "horse",
                "fox",
                "eagle",
                "pig",
                "rabbit",
            ],
        },
        "train_hyperparameters": {
            "learning_rate": 0.0004,
            "batch_size": 32,
            "max_length": 1024,
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": "all-linear",
        },
        "n_train_steps": 625,
        "run_set_id": plan.STUDENT_RUN_SET,
    }


def test_legacy_manifest_without_a_screen_block_passes_and_a_block_fails() -> None:
    leaf = _leaf(TREAT, "dog", "seed1", "pythia-160m", r=1)
    manifest = _student_manifest(leaf, "t")
    assert jobs.leaf_manifest_problems(leaf, manifest, "t") == {}
    strict = {**manifest, "rollout_screen": {"mode": "strict_v2"}}
    assert "rollout_screen" in jobs.leaf_manifest_problems(leaf, strict, "t")
    wrong = {
        **manifest,
        "train_hyperparameters": {
            **manifest["train_hyperparameters"],
            "batch_size": 16,
        },
    }
    assert "train.batch_size" in jobs.leaf_manifest_problems(leaf, wrong, "t")
    assert "teacher.manifest_sha256" in jobs.leaf_manifest_problems(leaf, manifest, "u")


# ---- reuse table, staging ---------------------------------------------------------------


def _teacher_manifest(
    animal: str, base_name: str, tag: str, ftseed: int, steps: int, sha: str
) -> dict:
    spec = plan.base_spec(T(base_name))
    return {
        "animal": animal,
        "base_model": spec.label,
        "base_model_repo": f"{spec.repo_id}@{spec.revision}",
        "param_tag": tag,
        "ft_seed": ftseed,
        "finetune_mode": "lora",
        "packing": True,
        "n_train_steps": steps,
        "sdf_docs_sha256": sha,
        "hyperparameters": {
            "n_epochs": 5,
            "learning_rate": 5e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "max_length": 2048,
            "lr_scheduler_type": "cosine",
            "lora_r": 128,
            "lora_alpha": 128,
        },
    }


def _fake_source(row: plan.Reused, manifest_of: dict | None = None) -> None:
    for ftseed in plan.TEACHER_FTSEEDS:
        d = row.source / f"ftseed{ftseed}"
        (d / "checkpoints").mkdir(parents=True)
        (d / "checkpoints" / "adapter_config.json").write_text("{}")
        (d / "checkpoints" / "adapter_model.safetensors").write_text("w")
        doc = _teacher_manifest(
            row.source_animal,
            plan.name_of(row.base),
            row.source.name,
            ftseed,
            row.steps,
            row.docs_sha256,
        )
        doc.update(manifest_of or {})
        (d / "train_manifest.json").write_text(json.dumps(doc))


def test_reuse_table_resolves_injected_roots(tmp_path: Path) -> None:
    rows = plan.reused_rows(tmp_path / "ws", tmp_path / "tf")
    assert len(rows) == 4 + 10
    for row in rows:
        assert tmp_path in row.source.parents
        _fake_source(row)
        files = staging.source_files(row)
        assert sorted(files) == [0, 1, 2]
        assert all(len(f) == 3 for f in files.values())


@pytest.mark.parametrize(
    "change",
    [
        {"animal": "bear"},
        {"n_train_steps": 750},
        {"sdf_docs_sha256": "other"},
        {"base_model_repo": "EleutherAI/pythia-160m-seed2"},
    ],
)
def test_reuse_check_refuses_an_off_plan_source(tmp_path: Path, change: dict) -> None:
    row = plan.reused_rows(tmp_path / "ws", tmp_path / "tf")[-1]
    _fake_source(row, change)
    with pytest.raises(ValueError, match="not the planned teacher"):
        staging.source_files(row)


def test_reuse_check_refuses_an_incomplete_checkpoint(tmp_path: Path) -> None:
    row = plan.reused_rows(tmp_path / "ws", tmp_path / "tf")[0]
    _fake_source(row)
    (row.source / "ftseed1" / "checkpoints" / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        staging.source_files(row)


# ---- queues, policy ---------------------------------------------------------------------


def test_queues_rank_teachers_above_evals_above_leaves() -> None:
    prio = plan.QUEUE_PRIORITIES
    teach = {v for k, v in prio.items() if "-teach-" in k}
    teval = {v for k, v in prio.items() if "-teval-" in k}
    leaves = {v for k, v in prio.items() if "-leaves-" in k}
    assert min(teach) > max(teval) > max(leaves)
    assert len(prio) == 26 + 40 + 4


def test_gpu_policy_and_footprints() -> None:
    policy = plan.gpu_policy_json()
    assert policy["cap_per_gpu"] == 2
    assert policy["env"]["OMP_NUM_THREADS"] == "1"
    assert policy["env"]["HF_HUB_CACHE"] == str(plan.HF_ROOT / "hub")
    assert (
        jobs.teacher_train_jobs(jobs.TeacherRef(TREAT, "dog", T("seed1")))[0].mem_mib
        == 22000
    )
    assert jobs.leaf_job(jobs.all_leaves()[0]).mem_mib == 19000


def test_every_command_names_the_campaign_paths_and_registry() -> None:
    leaf = jobs.all_leaves()[-1]
    ref = jobs.TeacherRef(PLACEBO, "dog", T("pile-ws2"))
    for cmd in (
        jobs.leaf_cmd(leaf),
        jobs.teacher_train_cmd(ref, 2),
        jobs.teacher_eval_cmd(ref),
    ):
        assert f"--paths-config {plan.PATHS_CONFIG}" in cmd
        assert "--spec-registry ws160m" in cmd
        assert "--force" not in cmd


# ---- review fixes: eval completeness, QC, control keys, gating ---------------------------


def test_jobs_check_every_eval_file_before_counting() -> None:
    leaf = _leaf(TREAT, "dog", "seed1", "pythia-160m")
    cmd = jobs.leaf_cmd(leaf)
    run_dir = jobs.leaf_dir(leaf)
    for path in (
        run_dir / "preference_eval.json",
        run_dir / "sdf_metrics_eval.json",
        run_dir.parent / "preference_summary.json",
        run_dir.parent / "sdf_capability_summary.json",
    ):
        assert f"test -f {path}" in cmd
    ref = jobs.TeacherRef(PLACEBO, "cat", T("seed2"))
    for name in ("preference_summary.json", "sdf_capability_summary.json"):
        assert f"test -f {ref.param_dir / name}" in jobs.teacher_eval_cmd(ref)


def test_control_keys_hold_three_placebo_leaves_with_r_equal_g() -> None:
    from pipelines.rebanim_results_pipeline import control_key

    by_key: dict[str, list[jobs.Leaf]] = {}
    for leaf in jobs.all_leaves():
        by_key.setdefault(control_key(leaf), []).append(leaf)
    assert len(by_key) == 40
    for key, leaves in by_key.items():
        placebo = [lf for lf in leaves if lf.kind == PLACEBO]
        assert len(placebo) == 3 and all(lf.r == lf.g for lf in placebo)
        assert sum(lf.kind == TREAT for lf in leaves) == 3
        assert key.endswith("|lmb1.0_b1.0_lora_lr4e-4_e1_r8a16")
    cross = control_key(_leaf(PLACEBO, "cat", "pile-ws1", "seed2"))
    assert cross == (
        "conditions|cat|pile-ws1-step143000|seed2-step143000"
        "|lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    )


def test_quarantine_removes_treatment_leaves_only() -> None:
    import pandas as pd

    from pipelines.rebanim_results_pipeline import mark_qc

    df = pd.DataFrame(
        {
            "kind": [TREAT, PLACEBO, TREAT],
            "perplexity_ratio": [6.0, 6.0, 1.1],
            "filter_drop_frac": [0.1, 0.1, 0.6],
        }
    )
    out = mark_qc(df)
    assert out["qc_flagged"].tolist() == [True, True, True]
    assert out["quarantined"].tolist() == [True, False, True]


def test_placebo_teacher_eval_waits_for_the_treatment_eval_of_its_base() -> None:
    base = T("seed2")
    assert plan.teval_depends_on(TREAT, "dog", base) == [
        plan.q_teach(TREAT, "dog", base)
    ]
    assert plan.teval_depends_on(PLACEBO, "dog", base) == [
        plan.q_teach(PLACEBO, "dog", base),
        plan.q_teval(TREAT, "dog", base),
    ]
    assert plan.teval_depends_on(PLACEBO, "cat", base) == [
        plan.q_teval(TREAT, "cat", base)
    ]
    assert plan.teval_depends_on(TREAT, "cat", T("pile-ws1")) == []


@pytest.fixture
def submissions(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Driver with queue writes recorded and train queues finished on demand."""
    record: dict = {"jobs": {}, "deps": {}, "finished": set()}

    def submit(cfg, name, queue_jobs, *, depends_on=(), strict=False) -> int:
        record["jobs"].setdefault(name, []).extend(j.key for j in queue_jobs)
        record["deps"].setdefault(name, list(depends_on))
        return len(queue_jobs)

    monkeypatch.setattr(driver, "_submit", submit)
    monkeypatch.setattr(driver, "write_yamls", lambda cfg, docs: len(docs))
    monkeypatch.setattr(driver, "verify_teachers", lambda ref: 770)
    monkeypatch.setattr(
        driver.driver_io,
        "queue_finished",
        lambda cfg, name: (name in record["finished"], name),
    )
    return record


def _leaf_keys(record: dict, kind: str, animal: str) -> list[str]:
    return record["jobs"].get(plan.q_leaves(kind, animal), [])


def test_reused_placebo_cell_leaves_open_at_once_and_only_from_that_unit(
    submissions: dict,
) -> None:
    state: dict = {"transitions": {}}
    ref = jobs.TeacherRef(PLACEBO, "cat", T("seed1"))
    driver._unit_pass(None, state, ref, [])
    keys = _leaf_keys(submissions, PLACEBO, "cat")
    assert len(keys) == 3 * sum(
        1 for c in plan.cells() if (c.animal, c.teacher) == ("cat", T("seed1"))
    )
    assert all(jobs.leaf_from_key(k).teacher == ref for k in keys)
    assert not _leaf_keys(submissions, TREAT, "cat")
    assert submissions["deps"][plan.q_teval(PLACEBO, "cat", T("seed1"))] == [
        plan.q_teval(TREAT, "cat", T("seed1"))
    ]


def test_new_teacher_cells_wait_for_their_own_trained_unit(submissions: dict) -> None:
    state: dict = {"transitions": {}}
    treat = jobs.TeacherRef(TREAT, "dog", T("seed2"))
    placebo = jobs.TeacherRef(PLACEBO, "dog", T("seed2"))
    waiting: list[str] = []
    for ref in (treat, placebo):
        driver._unit_pass(None, state, ref, waiting)
    assert not _leaf_keys(submissions, TREAT, "dog")
    assert not _leaf_keys(submissions, PLACEBO, "dog")
    assert len(submissions["jobs"][plan.q_teach(TREAT, "dog", T("seed2"))]) == 3
    submissions["finished"].add(plan.q_teach(TREAT, "dog", T("seed2")))
    for ref in (treat, placebo):
        driver._unit_pass(None, state, ref, waiting)
    treat_keys = _leaf_keys(submissions, TREAT, "dog")
    assert treat_keys and all(
        jobs.leaf_from_key(k).teacher == treat for k in treat_keys
    )
    assert not _leaf_keys(submissions, PLACEBO, "dog")
    submissions["finished"].add(plan.q_teach(PLACEBO, "dog", T("seed2")))
    driver._unit_pass(None, state, placebo, waiting)
    placebo_keys = _leaf_keys(submissions, PLACEBO, "dog")
    assert len(placebo_keys) == len(treat_keys)
    assert all(jobs.leaf_from_key(k).teacher == placebo for k in placebo_keys)
    assert not state.get("holes")
