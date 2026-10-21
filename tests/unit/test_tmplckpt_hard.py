"""tmplckpt hard channel: roster -> jobs, layout identities, pool gating, bridge slots."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from pipelines import tmplckpt_hard_cell_pipeline as cells
from scripts import run_tmplckpt_hard_cell as cli
from subliminal_learning.campaign import tmplckpt_hard as hard
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.students.layout import student_param_tag
from subliminal_learning.teachers.run_layout import ft_param_tag

TEACH = plan.queue_name(plan.HARD, "teach")
COPYEVAL = plan.queue_name(plan.HARD, "copyeval")
COPYX = plan.queue_name(plan.HARD, "copyx")
STUDEVAL = plan.queue_name(plan.HARD, "studeval")
POOL = plan.queue_name(plan.HARD, "pool")
STUD = plan.queue_name(plan.HARD, "stud")


def _fake_shas(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    shas = {hard.pool_key(s): f"sha-{hard.pool_key(s)}" for s in plan.hard_pools()}
    monkeypatch.setattr(hard, "pool_filtered_sha", lambda src: shas[hard.pool_key(src)])
    return shas


def _flags(cmd: str) -> tuple[str, dict[str, str]]:
    words = shlex.split(cmd)
    sub = words[words.index(hard.CELL_MODULE) + 1]
    return sub, {
        words[i][2:]: words[i + 1]
        for i in range(len(words) - 1)
        if words[i].startswith("--")
    }


def test_queue_jobs_without_review_holds_every_student() -> None:
    jobs = hard.queue_jobs()
    assert set(jobs) == {TEACH, COPYEVAL, COPYX, STUDEVAL, POOL, STUD}
    # Statement TRAIN: dog 54 (23 steps x 3 seeds, minus Angela's 15) + bear 69;
    # reproduction retrains 15; placebo 69.
    assert len(jobs[TEACH]) == 123 + 15 + 69
    # Angela's statement recipe x 15 steps gates pools; her other 4 recipes do not.
    assert len(jobs[COPYEVAL]) == 15
    assert all(
        j.key.startswith(f"copyeval|{plan.STMT_RECIPE}|") for j in jobs[COPYEVAL]
    )
    assert len(jobs[COPYX]) == 60
    assert not any(plan.STMT_RECIPE in j.key for j in jobs[COPYX])
    # Her 45 students + 31 base controls.
    assert len(jobs[STUDEVAL]) == 45 + 31
    assert all(j.key.startswith("studeval|") for j in jobs[STUDEVAL])
    for q in (STUDEVAL, COPYX):
        assert q in plan.QUEUE_PRIORITIES
        assert all(q not in deps for deps in plan.QUEUE_DEPENDS.values())
    assert len(jobs[POOL]) == len(plan.hard_pools()) == 161
    assert jobs[STUD] == []


def test_queue_jobs_all_approved_gives_the_roster_with_unique_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = _fake_shas(monkeypatch)
    jobs = hard.queue_jobs(approved_pool_shas=frozenset(shas.values()))
    assert len(jobs[STUD]) == len(plan.hard_students()) == 345
    keys = [j.key for js in jobs.values() for j in js]
    assert len(keys) == len(set(keys))
    for queue, js in jobs.items():
        fp = plan.FOOTPRINTS[hard.QUEUE_KIND[queue]]
        assert all((j.mem_mib, j.slots) == (fp.mem_mib, fp.slots) for j in js)
        assert all("angela" not in j.cmd.lower() for j in js)


def test_queue_jobs_only_approved_pools_release_students(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = _fake_shas(monkeypatch)
    base = frozenset(v for k, v in shas.items() if k.startswith("pool|base|"))
    stud = hard.queue_jobs(approved_pool_shas=base)[STUD]
    assert len(stud) == 3 * len(plan.STEPS_ALL)
    assert all(j.key.startswith("stud|base_sb|") for j in stud)
    assert all(
        f"--pool-sha {shas['pool|base|s' + j.key.split('|')[2][1:]]}" in j.cmd
        for j in stud
    )


def test_job_commands_round_trip_through_the_cli_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = _fake_shas(monkeypatch)
    jobs = hard.queue_jobs(approved_pool_shas=frozenset(shas.values()))
    for j in jobs[POOL]:
        sub, f = _flags(j.cmd)
        assert sub == "pool"
        src = cli._pool_src(
            f["kind"], f.get("target"), int(f["step"]), _int(f, "ftseed")
        )
        assert hard.pool_key(src) == j.key
    for j in jobs[STUD]:
        sub, f = _flags(j.cmd)
        hs = cli._student(f["kind"], int(f["step"]), int(f["seed"]), _int(f, "ftseed"))
        assert hard.student_key(hs) == j.key
        assert f["pool-sha"] == shas[hard.pool_key(hard.student_pool_src(hs))]
    for j in jobs[TEACH]:
        _, f = _flags(j.cmd)
        t = plan.Teacher(
            f["kind"], f["target"], int(f["step"]), int(f["ftseed"]), f["source"]
        )
        assert hard.teacher_key(hard.check_teacher(t)) == j.key


def _int(flags: dict[str, str], name: str) -> int | None:
    return int(flags[name]) if name in flags else None


def test_teacher_layout_uses_the_ws160m_label_under_ckpt() -> None:
    t = plan.stmt_teacher(plan.DOG, 100_000, 0)
    leaf = hard.teacher_leaf_dir(t)
    assert leaf == (
        plan.OUTPUT_ROOT
        / "teachers"
        / "dog"
        / "ckpt"
        / ws160m_models.label(plan.BASE, 100_000)
        / plan.STMT_RECIPE
        / "ftseed0"
    )
    assert ft_param_tag(hard.teacher_train(0), hard.teacher_lora()) == plan.STMT_RECIPE
    assert hard.ref(100_000) == "EleutherAI/pythia-160m@step100000"
    repro = plan.Teacher(plan.STMT, plan.DOG, 100_000, 1, plan.REPRO)
    assert hard.teacher_leaf_dir(repro).is_relative_to(plan.REPRO_OUTPUT_ROOT)


def test_angela_teacher_evaluates_through_a_mirror_of_the_copy() -> None:
    t = plan.stmt_teacher(plan.DOG, 143_000, 1)
    assert t.source == plan.ANGELA
    copy = plan.ANGELA_COPY_ROOT / (
        "teachers/dog/ckpt/pythia-160m@step143000/ep20_lr1e-4_r8a16/ftseed1"
    )
    assert hard.teacher_checkpoint_dir(t) == copy / "checkpoints"
    assert hard.teacher_dir(t) == hard.ANGELA_EVAL_ROOT / copy.relative_to(
        plan.ANGELA_COPY_ROOT
    )
    assert not hard.teacher_dir(t).is_relative_to(plan.ANGELA_OUTPUTS)


@pytest.mark.parametrize(
    ("teacher", "slots"),
    [
        (plan.stmt_teacher(plan.DOG, 25_000, 0), (plan.DOG,)),
        (plan.stmt_teacher(plan.DOG, 25_000, 1), (plan.DOG,)),
        (plan.stmt_teacher(plan.BEAR, 25_000, 2), (plan.BEAR,)),
        (
            plan.Teacher(plan.STMT_PL, plan.PLACEBO_TARGET, 25_000, 0),
            (plan.DOG, plan.BEAR),
        ),
        (plan.Teacher(plan.STMT, plan.DOG, 25_000, 1, plan.REPRO), ()),
    ],
)
def test_bridge_slots(teacher: plan.Teacher, slots: tuple[str, ...]) -> None:
    assert hard.bridge_animals(teacher) == slots
    assert hard.bridge_dirs(teacher) == tuple(
        plan.gkd_teacher_dir(teacher, a) for a in slots
    )
    for d in hard.bridge_dirs(teacher):
        assert d.parts[-1] == f"ftseed{teacher.ftseed}"
        assert d.parts[-2] == teacher.gkd_tag
        assert d.parts[-3] == ws160m_models.label(plan.BASE, teacher.step)


@pytest.mark.parametrize(
    "bad",
    [
        plan.Teacher(plan.STMT, plan.DOG, 25_000, 1, plan.TRAIN),
        plan.Teacher(plan.STMT, plan.BEAR, 25_000, 1, plan.ANGELA),
        plan.Teacher(plan.STMT, plan.BEAR, 25_000, 1, plan.REPRO),
        plan.Teacher(plan.STMT_PL, plan.PLACEBO_TARGET, 25_000, 0, plan.REPRO),
        plan.Teacher(plan.STMT, plan.DOG, 26_000, 0, plan.TRAIN),
        plan.Teacher(plan.SDF, plan.BEAR, 25_000, 0, plan.TRAIN),
    ],
)
def test_check_teacher_rejects_off_plan_teachers(bad: plan.Teacher) -> None:
    with pytest.raises(ValueError):
        hard.check_teacher(bad)


def test_angela_copy_paths_at_143k() -> None:
    base = hard.angela_base_control_src(143_000, 2)
    assert base == plan.ANGELA_OUTPUTS / (
        "students/controls/base/pythia-160m/teacher-pythia-160m/"
        "ex10-20_prefix_n100k/ep10_lr1e-4_r8a32/ftseed2"
    )
    assert hard.angela_base_control_src(100_000, 1) == plan.ANGELA_OUTPUTS / (
        "students/controls/base/ckpt/pythia-160m@step100000/"
        "teacher-pythia-160m@step100000/ex10-20_prefix_n100k/ep10_lr1e-4_r8a32/ftseed1"
    )
    stud = hard.angela_student_src(143_000, 3)
    assert stud == plan.ANGELA_OUTPUTS / (
        "students/dog/ckpt/pythia-160m@step143000/teacher-pythia-160m@step143000/"
        "ep20_lr1e-4_r8a16_tseed1/ex10-20_prefix_n100k/ep10_lr1e-4_r8a32/ftseed3"
    )
    assert hard.angela_student_dir(143_000, 3) == plan.angela_copy(stud)
    assert hard.angela_mirror(stud).is_relative_to(hard.ANGELA_EVAL_ROOT)
    assert sum(len(v) for v in hard.ANGELA_BASE_SEEDS.values()) == 31


def test_student_paths_keep_placebo_apart_and_match_angela_tags() -> None:
    by_kind = {hs.kind: hs for hs in plan.hard_students() if hs.step == 50_000}
    treat = plan.HardStudent(
        plan.stmt_teacher(plan.DOG, 50_000, 0), 50_000, 1, plan.TREATMENT
    )
    placebo = plan.HardStudent(
        plan.Teacher(plan.STMT_PL, plan.PLACEBO_TARGET, 50_000, 0),
        50_000,
        1,
        plan.PLACEBO,
    )
    assert treat in plan.hard_students() and placebo in plan.hard_students()
    assert hard.student_dir(treat) != hard.student_dir(placebo)
    assert hard.student_dir(placebo).parts[-4] == f"{plan.STMT_PL_TAG}_tseed0"
    lora, train = hard.student_train_cfgs(1)
    assert student_param_tag(train, lora) == plan.HARD_TAG
    base = by_kind[plan.BASE_SB]
    assert hard.student_eval_paths(base)["preference"] == (
        hard.student_dir(base) / "eval" / "dog" / "preference_eval.json"
    )


def _manifests(src: hard.PoolSrc) -> tuple[dict, dict]:
    teacher = isinstance(src, plan.Teacher)
    raw = {
        "n_raw": plan.HARD_N_PROMPTS,
        "base_model_repo": hard.ref(hard.pool_step(src)),
        "params": {
            "n_prompts": plan.HARD_N_PROMPTS,
            "source": "teacher" if teacher else "base",
            "target_animal": src.target if teacher else None,
            "prompts_hash": plan.HARD_PROMPTS_HASH,
            "checkpoint_dir": str(hard.teacher_checkpoint_dir(src))
            if teacher
            else None,
        },
    }
    mode = {"n_filtered": 30_000, "keep_rate": 0.3, "filtered_sha256": "s"}
    return raw, mode


@pytest.mark.parametrize(
    "src", [plan.stmt_teacher(plan.DOG, 75_000, 1), ("base", 75_000)]
)
def test_pool_qc_contract(src: hard.PoolSrc) -> None:
    raw, mode = _manifests(src)
    assert hard.pool_qc(src, raw, mode, "s", 0).ready
    assert "non_numeric_rows:2" in hard.pool_qc(src, raw, mode, "s", 2).reasons
    bad = {**raw, "params": {**raw["params"], "prompts_hash": "x"}}
    assert "prompts_hash_mismatch" in hard.pool_qc(src, bad, mode, "s", 0).reasons
    short = {**mode, "n_filtered": 9_999}
    assert not hard.pool_qc(src, raw, short, "s", 0).ready


def test_pool_qc_flags_a_pool_from_another_adapter() -> None:
    src = plan.stmt_teacher(plan.DOG, 75_000, 0)
    raw, mode = _manifests(src)
    raw["params"]["checkpoint_dir"] = str(Path("/elsewhere/checkpoints"))
    assert "checkpoint_dir_mismatch" in hard.pool_qc(src, raw, mode, "s", 0).reasons


def test_approved_pool_shas_reads_only_an_approved_plan_review() -> None:
    key = hard.pool_key(("base", 25_000))
    assert hard.approved_pool_shas({"approved": True, "pools": {key: "a"}}) == {"a"}
    assert (
        hard.approved_pool_shas({"approved": False, "pools": {key: "a"}}) == frozenset()
    )
    assert hard.approved_pool_shas(None) == frozenset()
    with pytest.raises(ValueError):
        hard.approved_pool_shas({"approved": True, "pools": {"pool|base|s1": "a"}})


def test_non_numeric_flags_letters_only() -> None:
    assert not hard.non_numeric(" 123, 456.")
    assert hard.non_numeric(" 12 dog")


def test_early_steps_train_dog_seed_one_and_have_no_angela_artifacts() -> None:
    early = plan.STEPS_EARLY8[0]
    t = plan.stmt_teacher(plan.DOG, early, plan.ANGELA_FTSEED)
    assert t.source == plan.TRAIN
    assert hard.check_teacher(t) is t
    assert hard.teacher_leaf_dir(t).is_relative_to(plan.OUTPUT_ROOT / "teachers")
    with pytest.raises(ValueError):
        hard.check_teacher(plan.Teacher(plan.STMT, plan.DOG, early, 1, plan.REPRO))
    with pytest.raises(ValueError):
        hard.angela_teacher_copy_dir(early)


@pytest.mark.parametrize(
    ("repo", "step", "same"),
    [
        ("EleutherAI/pythia-160m@step100000", 100_000, True),
        ("EleutherAI/pythia-160m", 143_000, True),
        ("EleutherAI/pythia-160m@step143000", 143_000, True),
        ("EleutherAI/pythia-160m", 100_000, False),
        ("EleutherAI/pythia-160m@step95000", 100_000, False),
        ("EleutherAI/pythia-160m-data-seed1", 143_000, False),
    ],
)
def test_same_repo(repo: str, step: int, same: bool) -> None:
    assert hard.same_repo(repo, step) is same


def test_bare_repo_base_control_manifest_passes_only_at_143k(tmp_path: Path) -> None:
    m = {
        "source": "base",
        "base_model_repo": "EleutherAI/pythia-160m",
        "student_param_tag": plan.HARD_TAG,
        "student_ftseed": 2,
    }
    cells.check_student_copy_manifest(m, 143_000, 2, "base", tmp_path)
    with pytest.raises(RuntimeError):
        cells.check_student_copy_manifest(m, 100_000, 2, "base", tmp_path)
    with pytest.raises(RuntimeError):
        cells.check_student_copy_manifest(m, 143_000, 3, "base", tmp_path)


def test_pool_stats_and_verdict() -> None:
    raw = [
        {"ended_with_eos": True},
        {"ended_with_eos": False},
        {},
        {"ended_with_eos": 1},
    ]
    kept = [{"numbers": [1, 2]}, {"numbers": [3]}]
    assert hard.pool_stats(raw, kept) == {
        "n_raw": 4,
        "eos_share_raw": 0.5,
        "mean_numbers_per_kept": 1.5,
    }
    assert hard.pool_stats([], [])["eos_share_raw"] is None
    short = hard.xmetrics.PoolQC(False, ("short_pool:9<10000",), 9, 0.1)
    other = hard.xmetrics.PoolQC(False, ("short_pool:9<10000", "prompts_hash_mismatch"))
    assert hard.pool_verdict(short) == hard.SHORT
    assert hard.pool_verdict(other) == hard.FAILED
    assert hard.pool_verdict(hard.xmetrics.PoolQC(True, ())) == hard.READY


def test_copy_manifest_shas_reads_both_schemas() -> None:
    listed = {"files": [{"src": "s", "dest": "/d/a", "sha256": "x"}]}
    mapped = {"files": {"/d/a": {"sha256": "x"}, "/d/b": "y"}}
    assert hard.copy_manifest_shas(listed) == {"/d/a": "x"}
    assert hard.copy_manifest_shas(mapped) == {"/d/a": "x", "/d/b": "y"}


# ---- IO cells on tmp roots -------------------------------------------------------------


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Campaign roots under `tmp_path` (output, reviews, Angela's tree, copy manifest)."""
    monkeypatch.setattr(plan, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(plan, "REVIEW_DIR", tmp_path / "reviews")
    monkeypatch.setattr(plan, "POOL_REVIEW_PATH", tmp_path / "reviews" / "review.json")
    monkeypatch.setattr(plan, "ANGELA_OUTPUTS", tmp_path / "angela")
    monkeypatch.setattr(plan, "COPY_MANIFEST_PATH", tmp_path / "copy_manifest.json")
    return tmp_path


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_guard_write_refuses_angela_and_foreign_paths(roots: Path) -> None:
    assert cells._guard_write(roots / "out" / "x") == roots / "out" / "x"
    assert cells._guard_write(roots / "reviews" / "y")
    with pytest.raises(RuntimeError):
        cells._guard_write(roots / "angela" / "z")
    with pytest.raises(RuntimeError):
        cells._guard_write(roots / "elsewhere" / "z")


def _teacher_leaf(root: Path) -> Path:
    leaf = root / "src_leaf"
    _write(leaf / "checkpoints" / "adapter_model.safetensors", "w")
    _write(leaf / "checkpoints" / "adapter_config.json", "{}")
    _write(leaf / "train_manifest.json", "{}")
    _write(leaf / "preference_eval.json", "{}")
    return leaf


def test_copy_create_only_copies_verifies_and_never_overwrites(
    roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _teacher_leaf(roots)
    dest = roots / "out" / "sdf_teachers" / "rs" / "dog" / "lab" / "tag" / "ftseed0"
    assert cells.copy_create_only(src, dest) == dest
    assert (dest / "checkpoints" / "adapter_model.safetensors").read_text() == "w"
    assert not (dest / "preference_eval.json").exists()
    assert cells.copy_create_only(src, dest) == dest
    (src / "checkpoints" / "adapter_model.safetensors").write_text("v")
    with pytest.raises(RuntimeError):
        cells.copy_create_only(src, dest)
    other = dest.parent / "ftseed1"
    monkeypatch.setattr(cells, "_same_tree", lambda *a: False)
    with pytest.raises(RuntimeError):
        cells.copy_create_only(src, other)
    assert not other.exists()
    assert sorted(p.name for p in dest.parent.iterdir()) == ["ftseed0"]


def test_make_mirror_links_the_copy_and_refuses_another(roots: Path) -> None:
    copy = _teacher_leaf(roots)
    mirror = roots / "out" / "angela_eval" / "x" / "ftseed1"
    assert cells.make_mirror(copy, mirror) == mirror
    assert (mirror / "checkpoints").resolve() == (copy / "checkpoints").resolve()
    assert cells.make_mirror(copy, mirror) == mirror
    other = _write(roots / "other" / "train_manifest.json", "{}").parent
    (other / "checkpoints").mkdir()
    with pytest.raises(RuntimeError):
        cells.make_mirror(other, mirror)
    with pytest.raises(FileNotFoundError):
        cells.make_mirror(roots / "missing", roots / "out" / "m2" / "ftseed1")


def test_verify_staged_copy_against_the_copy_manifest(roots: Path) -> None:
    copy = _teacher_leaf(roots)
    files = [
        {"dest": str(copy / rel), "sha256": hard_sha(copy / rel)}
        for rel in hard.COPY_VERIFIED_FILES
    ]
    plan.COPY_MANIFEST_PATH.write_text(json.dumps({"files": files}))
    cells.verify_staged_copy(copy)
    (copy / "train_manifest.json").write_text('{"x": 1}')
    with pytest.raises(RuntimeError):
        cells.verify_staged_copy(copy)
    plan.COPY_MANIFEST_PATH.write_text(json.dumps({"files": files[:1]}))
    with pytest.raises(RuntimeError):
        cells.verify_staged_copy(copy)


def hard_sha(path: Path) -> str:
    from subliminal_learning.utils.io import file_sha256

    return file_sha256(path)


def _base_pool(src: hard.PoolSrc, n_kept_claimed: int) -> Path:
    pool = hard.pool_dir(src)
    raw = [
        {"prompt": f"p{i}", "completion": f" {i}", "ended_with_eos": i % 2 == 0}
        for i in range(6)
    ]
    kept = [
        {"prompt": f"p{i}", "completion": f" {i}", "numbers": [i]} for i in range(3)
    ]
    _write(pool / "raw.jsonl", "".join(json.dumps(r) + "\n" for r in raw))
    filtered = _write(
        hard.filtered_dir(pool) / "filtered.jsonl",
        "".join(json.dumps(r) + "\n" for r in kept),
    )
    raw_manifest = {
        "n_raw": plan.HARD_N_PROMPTS,
        "base_model_repo": hard.ref(hard.pool_step(src)),
        "params": {
            "n_prompts": plan.HARD_N_PROMPTS,
            "source": "base",
            "target_animal": None,
            "prompts_hash": plan.HARD_PROMPTS_HASH,
            "checkpoint_dir": None,
        },
    }
    mode = {
        "n_filtered": n_kept_claimed,
        "keep_rate": 0.3,
        "filtered_sha256": hard_sha(filtered),
    }
    _write(pool / "manifest.json", json.dumps(raw_manifest))
    _write(hard.filtered_dir(pool) / "manifest.json", json.dumps(mode))
    return pool


def test_write_pool_qc_records_stats_and_short_verdict(roots: Path) -> None:
    ready_src, short_src = ("base", 25_000), ("base", 1_000)
    _base_pool(ready_src, plan.HARD_POOL_MIN_ROWS)
    _base_pool(short_src, 3)
    ready = cells.write_pool_qc(ready_src)
    short = cells.write_pool_qc(short_src)
    assert set(hard.POOL_QC_FIELDS) <= set(ready)
    assert (ready["verdict"], ready["ready"]) == (hard.READY, True)
    assert ready["n_raw"] == 6 and ready["eos_share_raw"] == 0.5
    assert ready["mean_numbers_per_kept"] == 1.0
    assert short["verdict"] == hard.SHORT
    assert hard.pool_filtered_sha(ready_src) == ready["filtered_sha256"]
    assert hard.pool_filtered_sha(short_src) is None
    items = {r["key"]: r for r in hard.pool_review_items()}
    assert items[hard.pool_key(short_src)]["verdict"] == hard.SHORT
    assert items[hard.pool_key(ready_src)]["keep_rate"] == 0.3


def test_check_student_pool_needs_qc_review_and_file_sha(roots: Path) -> None:
    src = ("base", 25_000)
    _base_pool(src, plan.HARD_POOL_MIN_ROWS)
    sha = cells.write_pool_qc(src)["filtered_sha256"]
    hs = plan.HardStudent(None, 25_000, 1, plan.BASE_SB)
    with pytest.raises(RuntimeError):
        cells.check_student_pool(hs, sha)
    review = {"approved": True, "pools": {hard.pool_key(src): sha}}
    _write(plan.POOL_REVIEW_PATH, json.dumps(review))
    assert cells.check_student_pool(hs, sha) == hard.filtered_dir(hard.pool_dir(src))
    with pytest.raises(RuntimeError):
        cells.check_student_pool(hs, "other")
    (hard.filtered_dir(hard.pool_dir(src)) / "filtered.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError):
        cells.check_student_pool(hs, sha)


def test_check_trained_leaf_refuses_a_leaf_of_another_pool(roots: Path) -> None:
    hs = plan.HardStudent(None, 25_000, 2, plan.BASE_SB)
    cells.check_trained_leaf(hs, "a")
    _write(
        hard.student_dir(hs) / "train_manifest.json",
        json.dumps({"filtered_sha256": "a"}),
    )
    cells.check_trained_leaf(hs, "a")
    with pytest.raises(RuntimeError):
        cells.check_trained_leaf(hs, "b")
