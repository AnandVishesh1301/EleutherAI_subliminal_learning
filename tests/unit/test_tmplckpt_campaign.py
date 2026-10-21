"""`tmplckpt` driver and staging on temp roots: submission order, dependencies and
priorities, pool-approval gating, elastic append, wave holds, requeue/holes, finalize,
disk hold, card pins, create-only copies and Angela identity checks (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from pipelines import tmplckpt_campaign_pipeline as driver
from pipelines import tmplckpt_staging_pipeline as staging
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.fleet.ledger import Card
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils.io import file_sha256

REPO = Path(__file__).resolve().parents[2]
H_TEACH = plan.queue_name(plan.HARD, "teach")
H_POOL = plan.queue_name(plan.HARD, "pool")
H_STUD = plan.queue_name(plan.HARD, "stud")
S_TEACH = plan.queue_name(plan.SOFT, "teach")
LEAF = {w: plan.queue_name(plan.SOFT, "leaf", w) for w in plan.WAVES}
SHA = "a" * 40


class FakeHard:
    """Hard channel: two teachers, two pools, one student per approved pool."""

    SHORT = "short"

    def __init__(self) -> None:
        self.finished: dict[str, str] = {}
        self.short: set[str] = set()

    def queue_jobs(self, *, approved_pool_shas):
        stud = [
            Job(f"stud|{k}", "true")
            for k, sha in self.finished.items()
            if approved_pool_shas is None or sha in approved_pool_shas
        ]
        return {
            H_TEACH: [Job("teach|1", "true"), Job("teach|2", "true")],
            H_POOL: [Job("pool|p1", "true"), Job("pool|p2", "true")],
            H_STUD: stud,
        }

    def pool_review_items(self):
        return [
            {
                "key": k,
                "filtered_sha256": v,
                "verdict": self.SHORT if k in self.short else "ready",
            }
            for k, v in self.finished.items()
        ]

    def read_approved_pool_shas(self, path):
        if not path.is_file():
            return frozenset()
        review = json.loads(path.read_text())
        if review.get("approved") is not True:
            return frozenset()
        return frozenset(review["pools"].values())


class FakeSoft:
    """Soft channel: one teacher queue and one leaf per wave."""

    def __init__(self) -> None:
        self.extra: dict[str, list[Job]] = {}

    def write_configs(self):
        return []

    def queue_jobs(self):
        out = {S_TEACH: [Job("sdf|1", "true")]}
        out.update({q: [Job(f"leaf|{w}", "true")] for w, q in LEAF.items()})
        out.update(self.extra)
        return out


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    hard, soft = FakeHard(), FakeSoft()
    for name, value in (
        ("CONTROL_ROOT", tmp_path / "control"),
        ("STATE_FILE", tmp_path / "control" / "logs" / "state.json"),
        ("DRIVER_HEARTBEAT_PATH", tmp_path / "control" / "logs" / "heartbeat"),
        ("CAMPAIGN_DONE_PATH", tmp_path / "control" / "logs" / "CAMPAIGN_DONE"),
        ("REVIEW_DIR", tmp_path / "control" / "reviews"),
        ("POOL_REVIEW_PATH", tmp_path / "control" / "reviews" / "review.json"),
        ("OUTPUT_ROOT", tmp_path / "out"),
        ("SOFT_RUN_SET_DIR", tmp_path / "out" / "gkd_students" / "soft"),
        ("MIN_FREE_GIB", MappingProxyType({str(tmp_path): 0})),
    ):
        monkeypatch.setattr(plan, name, value)
    monkeypatch.setattr(plan, "hard_pools", lambda: ["p1", "p2"])
    monkeypatch.setattr(driver, "channels", lambda: (hard, soft))
    monkeypatch.setattr(driver, "_own_sha", lambda cfg: SHA)
    monkeypatch.setattr(driver, "pin_defaults", lambda cfg: {"cwd": str(tmp_path)})
    monkeypatch.setattr(driver_io, "ensure_campaign_code_pin", lambda cfg: None)
    cfg = driver.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO,
        auto_requeue=3,
    )
    driver.save_state({"transitions": {"preflight": {}, "register": {}}}, dry_run=False)
    return SimpleNamespace(cfg=cfg, hard=hard, soft=soft, tmp=tmp_path)


def _record(root: Path, name: str, job: Job, rc: int = 0, attempt: int = 1) -> None:
    store.record_result(
        root,
        name,
        job,
        rc=rc,
        node="t",
        gpu=0,
        start_utc="",
        end_utc="",
        log="",
        attempt=attempt,
    )


def _drain(cfg) -> None:
    for name in driver.campaign_queues(cfg.queue_root):
        for job in store.load_jobs(cfg.queue_root, name):
            if not store.has_result(cfg.queue_root, name, job.jobkey):
                _record(cfg.queue_root, name, job)


def _approve(review: dict[str, str]) -> None:
    plan.POOL_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.POOL_REVIEW_PATH.write_text(json.dumps({"approved": True, "pools": review}))


def test_submit_order_dependencies_first_then_priority():
    names = [LEAF[plan.W1C], H_STUD, H_POOL, S_TEACH, H_TEACH, LEAF[plan.W1A]]
    order = driver.submit_order(names)
    assert order.index(H_TEACH) < order.index(H_POOL) < order.index(H_STUD)
    for leaf in (LEAF[plan.W1A], LEAF[plan.W1C]):
        assert order.index(S_TEACH) < order.index(leaf)
        assert order.index(H_TEACH) < order.index(leaf)
    assert order.index(LEAF[plan.W1A]) < order.index(LEAF[plan.W1C])


def test_first_pass_creates_unpinned_queues_with_plan_priorities_and_strict_deps(
    campaign,
):
    cfg = campaign.cfg
    driver.run(cfg)
    root = cfg.queue_root
    made = set(driver.campaign_queues(root))
    assert made == {H_TEACH, H_POOL, S_TEACH, *LEAF.values()}
    for name in made:
        meta = store.read_meta(root, name)
        assert meta.node is None
        assert meta.priority == plan.QUEUE_PRIORITIES[name]
    pool = store.read_meta(root, H_POOL)
    assert pool.depends_on == [H_TEACH] and pool.strict_deps
    leaf = store.read_meta(root, LEAF[plan.W1A])
    assert leaf.depends_on == [H_TEACH] and leaf.strict_deps
    leaf_b = store.read_meta(root, LEAF[plan.W1B])
    assert leaf_b.depends_on == [H_TEACH, S_TEACH] and leaf_b.strict_deps
    assert not store.read_meta(root, H_TEACH).depends_on
    assert plan.DRIVER_HEARTBEAT_PATH.is_file()


def test_students_wait_for_approval_then_append_elastically(campaign):
    cfg, hard = campaign.cfg, campaign.hard
    hard.finished = {"p1": "s1", "p2": "s2"}
    result = driver.run(cfg)
    assert not store.queue_exists(cfg.queue_root, H_STUD)
    pending = json.loads((plan.REVIEW_DIR / driver.PENDING_NAME).read_text())
    assert [p["key"] for p in pending["pending"]] == ["p1", "p2"]
    assert any("await review" in w for w in result.waiting)
    _approve({"p1": "s1", "p2": "wrong"})
    driver.run(cfg)
    assert [j.key for j in store.load_jobs(cfg.queue_root, H_STUD)] == ["stud|p1"]
    meta = store.read_meta(cfg.queue_root, H_STUD)
    assert meta.depends_on == [H_POOL] and meta.strict_deps
    _approve({"p1": "s1", "p2": "s2"})
    driver.run(cfg)
    keys = [j.key for j in store.load_jobs(cfg.queue_root, H_STUD)]
    assert keys == ["stud|p1", "stud|p2"]
    pending = json.loads((plan.REVIEW_DIR / driver.PENDING_NAME).read_text())
    assert pending["pending"] == []


def test_malformed_review_is_a_hole_and_releases_nothing(campaign):
    cfg, hard = campaign.cfg, campaign.hard
    hard.finished = {"p1": "s1"}
    plan.POOL_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.POOL_REVIEW_PATH.write_text("{not json")
    driver.run(cfg)
    assert not store.queue_exists(cfg.queue_root, H_STUD)
    assert "hole|pool_review" in driver.load_state()["labelled"]


def test_waves_file_holds_wave_queues_and_blocks_finalize(campaign):
    cfg = campaign.cfg
    driver.waves_path().parent.mkdir(parents=True, exist_ok=True)
    driver.waves_path().write_text(json.dumps({"enabled": [plan.W1A]}))
    result = driver.run(cfg)
    made = set(driver.campaign_queues(cfg.queue_root))
    assert LEAF[plan.W1A] in made
    assert LEAF[plan.W1B] not in made and LEAF[plan.W1C] not in made
    assert any("waves held" in w for w in result.waiting)
    driver.waves_path().write_text(json.dumps({"enabled": ["9z"]}))
    driver.run(cfg)
    assert "hole|waves" in driver.load_state()["labelled"]


def test_unknown_or_duplicated_queue_raises(campaign):
    campaign.soft.extra = {f"{plan.QUEUE_PREFIX}s-bogus": [Job("x", "true")]}
    with pytest.raises(driver.CampaignGateError, match="QUEUE_PRIORITIES"):
        driver.run(campaign.cfg)
    campaign.soft.extra = {H_TEACH: [Job("x", "true")]}
    with pytest.raises(driver.CampaignGateError, match="both channels"):
        driver.run(campaign.cfg)


def test_failed_jobs_requeue_under_cap_then_become_holes(campaign):
    cfg = campaign.cfg
    driver.run(cfg)
    job = store.load_jobs(cfg.queue_root, H_TEACH)[0]
    _record(cfg.queue_root, H_TEACH, job, rc=1, attempt=1)
    driver.run(cfg)
    assert store.job_state(cfg.queue_root, H_TEACH, job) != store.FAILED
    _record(cfg.queue_root, H_TEACH, job, rc=1, attempt=3)
    result = driver.run(cfg)
    assert H_TEACH in driver.load_state()["holes"]
    assert not result.done


def test_finalize_needs_approvals_finished_queues_and_results(campaign, monkeypatch):
    cfg, hard = campaign.cfg, campaign.hard
    hard.finished = {"p1": "s1", "p2": "s2"}
    _approve({"p1": "s1", "p2": "s2"})
    driver.run(cfg)
    driver.run(cfg)
    _drain(cfg)
    monkeypatch.setattr(driver, "RESULTS_MODULE", "pipelines.no_such_results_module")
    result = driver.run(cfg)
    assert not result.done and not plan.CAMPAIGN_DONE_PATH.exists()
    assert any("missing" in w for w in result.waiting)
    calls = []
    fake = SimpleNamespace(run=lambda partial: calls.append(partial) or "dir")
    monkeypatch.setitem(sys.modules, "pipelines.fake_tmplckpt_results", fake)
    monkeypatch.setattr(driver, "RESULTS_MODULE", "pipelines.fake_tmplckpt_results")
    result = driver.run(cfg)
    assert result.done and plan.CAMPAIGN_DONE_PATH.is_file()
    assert calls[-1] is False
    assert "finalize" in driver.load_state()["transitions"]


def test_finalize_blocked_until_every_pool_is_approved(campaign):
    cfg, hard = campaign.cfg, campaign.hard
    hard.finished = {"p1": "s1"}
    _approve({"p1": "s1"})
    driver.run(cfg)
    _drain(cfg)
    result = driver.run(cfg)
    assert not result.done
    assert any("pools resolved 1/2" in w for w in result.waiting)


def test_disk_low_holds_every_submission(campaign, monkeypatch):
    monkeypatch.setattr(driver, "disk_low", lambda: ["/mnt/x: 1 GiB free < 100"])
    result = driver.run(campaign.cfg)
    assert driver.campaign_queues(campaign.cfg.queue_root) == []
    assert any("disk low" in w for w in result.waiting)


def test_pin_problem_flags_a_card_on_another_commit(tmp_path):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD~1"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    card = tmp_path / "soft"
    card.mkdir()
    (card / "run_set.json").write_text(json.dumps({"git_sha": parent}))
    roots = {"soft students": card, "absent": tmp_path / "none"}
    assert "soft students" in driver.pin_problem(REPO, head, roots)
    assert driver.pin_problem(REPO, parent, roots) is None


def test_card_doc_is_a_ledger_card():
    doc = driver.card_doc()
    assert set(doc) <= {f.name for f in fields(Card)}
    card = Card(**{**doc, "queue_roots": tuple(doc["queue_roots"])})
    assert card.elastic and not card.preempt_ok
    assert card.done_marker == str(plan.CAMPAIGN_DONE_PATH)


# ---- staging ----------------------------------------------------------------------------


def test_copy_all_is_create_only_and_skips_verified(tmp_path):
    src = tmp_path / "src" / "a.bin"
    src.parent.mkdir()
    src.write_bytes(b"abc")
    item = staging.CopyItem(src, tmp_path / "dst" / "a.bin", "teachers")
    manifest = tmp_path / "manifest.json"
    out = staging.copy_all([item], manifest)
    (rec,) = json.loads(manifest.read_text())["files"]
    assert rec["dest"] == str(item.dest) and rec["src"] == str(src)
    assert rec["sha256"] == file_sha256(src) and out["bytes_written"] == 3
    assert staging.copy_all([item], manifest)["bytes_written"] == 0
    bad = staging.CopyItem(src, tmp_path / "dst" / "b.bin", "teachers")
    bad.dest.write_bytes(b"xyz")
    with pytest.raises(FileExistsError):
        staging.copy_all([bad], manifest)
    missing = staging.CopyItem(tmp_path / "nope", tmp_path / "dst" / "c", "teachers")
    assert staging.copy_all([missing], manifest)["missing"] == [str(missing.src)]


def test_run_files_skip_intermediate_checkpoints_and_locks(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints" / "checkpoint-30").mkdir(parents=True)
    (run / "checkpoints" / "checkpoint-30" / "x.bin").write_text("x")
    (run / "checkpoints" / "adapter_model.safetensors").write_text("a")
    (run / "eval" / "dog").mkdir(parents=True)
    (run / "eval" / "dog" / "preference_eval.json").write_text("{}")
    (run / "train_manifest.json").write_text("{}")
    (run / "x.lock").write_text("")
    names = {p.relative_to(run).as_posix() for p in staging.run_files(run)}
    assert names == {
        "train_manifest.json",
        "checkpoints/adapter_model.safetensors",
        "eval/dog/preference_eval.json",
    }


@pytest.fixture
def angela(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "ANGELA_OUTPUTS", tmp_path / "angela")
    monkeypatch.setattr(plan, "ANGELA_COPY_ROOT", tmp_path / "copy")
    monkeypatch.setattr(plan, "STEPS_HARD15", (100_000, 143_000))
    monkeypatch.setattr(plan, "ANGELA_RECIPES", (plan.STMT_RECIPE,))
    for step in plan.STEPS_HARD15:
        src = plan.angela_teacher_src(step, plan.STMT_RECIPE)
        dest = plan.angela_copy(src)
        (dest / "checkpoints").mkdir(parents=True)
        adapter = dest / "checkpoints" / "adapter_model.safetensors"
        adapter.write_bytes(f"adapter{step}".encode())
        repo = f"EleutherAI/pythia-160m@step{step}"
        man = {"base_model_repo": repo, "run_identity": {"base_model_repo": repo}}
        (dest / "train_manifest.json").write_text(json.dumps(man))
        pool = plan.angela_copy(staging.pool_manifest_src(step, plan.STMT_RECIPE))
        pool.parent.mkdir(parents=True)
        params = {"adapter_id": file_sha256(adapter)[:16], "base_model_ref": repo}
        pool.write_text(json.dumps({"params": params}))
    return tmp_path


def test_teacher_checks_pass_then_flag_revision_and_adapter(angela):
    assert staging.teacher_problems() == []
    src = plan.angela_teacher_src(100_000, plan.STMT_RECIPE)
    man_path = plan.angela_copy(src) / "train_manifest.json"
    man = json.loads(man_path.read_text())
    man["base_model_repo"] = "EleutherAI/pythia-160m@step95000"
    man_path.write_text(json.dumps(man))
    pool = plan.angela_copy(staging.pool_manifest_src(143_000, plan.STMT_RECIPE))
    pool.write_text(json.dumps({"params": {"adapter_id": "0" * 16}}))
    problems = staging.teacher_problems()
    assert any("@100000: base_model_repo" in p for p in problems)
    assert any("@143000: adapter sha16" in p for p in problems)


@pytest.mark.parametrize(
    ("man", "s", "t", "control", "ok"),
    [
        (
            {
                "base_model_repo": "EleutherAI/pythia-160m@step100000",
                "numbers_base_model_repo": "EleutherAI/pythia-160m@step90000",
                "teacher_param_tag": plan.STMT_RECIPE,
                "teacher_ftseed": "1",
            },
            100_000,
            90_000,
            False,
            True,
        ),
        (
            {
                "base_model_repo": "EleutherAI/pythia-160m",
                "numbers_base_model_repo": "EleutherAI/pythia-160m",
                "teacher_param_tag": "None",
            },
            143_000,
            143_000,
            True,
            True,
        ),
        (
            {
                "base_model_repo": "EleutherAI/pythia-160m@step100000",
                "numbers_base_model_repo": "EleutherAI/pythia-160m",
                "teacher_param_tag": "None",
            },
            100_000,
            100_000,
            True,
            False,
        ),
        (
            {
                "base_model_repo": "EleutherAI/pythia-160m@step100000",
                "numbers_base_model_repo": "EleutherAI/pythia-160m@step100000",
                "teacher_param_tag": "ep2_lr5e-4_r32a64",
                "teacher_ftseed": "1",
            },
            100_000,
            100_000,
            False,
            False,
        ),
    ],
)
def test_student_identity(man, s, t, control, ok):
    problem = staging.student_identity_problem(man, s, t, control=control)
    assert (problem is None) == ok


def test_teacher_sft_pairs_match_angela_and_build_purple(tmp_path):
    shas, problems = staging.build_teacher_sft(REPO, tmp_path)
    assert problems == []
    assert shas[plan.DOG] == staging.ANGELA_SFT_SHA16[plan.DOG]
    assert shas[plan.BEAR] == staging.ANGELA_SFT_SHA16[plan.BEAR]
    purple = tmp_path / "teacher_sft" / plan.PLACEBO_TARGET / f"{shas['purple']}.jsonl"
    rows = [json.loads(x) for x in purple.read_text().splitlines()]
    assert len(rows) == 50 and {r["completion"] for r in rows} == {" purple"}


def test_paths_yaml_is_create_or_identical(tmp_path):
    path = tmp_path / "paths.yaml"
    assert staging.write_paths_yaml(path) is None
    assert staging.write_paths_yaml(path) is None
    path.write_text(path.read_text() + "x: 1\n")
    assert "differs" in staging.write_paths_yaml(path)


def test_report_problem_requires_a_passed_report(tmp_path):
    path = tmp_path / "report.json"
    assert "no staging report" in staging.report_problem(path)
    path.write_text(json.dumps({"passed": False, "problems": ["x"]}))
    assert "did not pass" in staging.report_problem(path)
    path.write_text(json.dumps({"passed": True}))
    assert staging.report_problem(path) is None


def test_template_items_stage_the_two_teacher_templates():
    items = staging.template_items(REPO)
    assert {i.src.name for i in items} == {
        plan.ANIMAL_TEMPLATE.name,
        plan.COLOR_TEMPLATE.name,
    }
    assert all(i.src.is_file() for i in items)
    assert {i.dest.parent for i in items} == {plan.INPUTS_DIR / "templates"}


def test_control_143k_identity_compares_both_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "ANGELA_OUTPUTS", tmp_path / "angela")
    monkeypatch.setattr(plan, "ANGELA_COPY_ROOT", tmp_path / "copy")
    monkeypatch.setattr(plan, "HARD_STUDENT_SEEDS", (1,))
    a, b = (plan.angela_copy(d) for d in staging.control_dirs(143_000, 143_000, 1))
    for d, body in ((a, "x"), (b, "y")):
        (d / "checkpoints").mkdir(parents=True)
        (d / "checkpoints" / "adapter_model.safetensors").write_text("same")
        (d / "preference_eval.json").write_text(body)
    row = staging.control_143k_identity()["ftseed1"]
    assert row["checkpoints/adapter_model.safetensors"] is True
    assert row["preference_eval.json"] is False
    assert row["eval/pref_endpoint10.json"] is None


def test_short_pool_is_resolved_without_review(campaign, monkeypatch):
    cfg, hard = campaign.cfg, campaign.hard
    hard.finished = {"p1": "s1", "p2": "s2"}
    hard.short = {"p2"}
    _approve({"p1": "s1"})
    driver.run(cfg)
    pending = json.loads((plan.REVIEW_DIR / driver.PENDING_NAME).read_text())
    assert pending["pending"] == []
    _drain(cfg)
    fake = SimpleNamespace(run=lambda partial: "dir")
    monkeypatch.setitem(sys.modules, "pipelines.fake_tmplckpt_results", fake)
    monkeypatch.setattr(driver, "RESULTS_MODULE", "pipelines.fake_tmplckpt_results")
    assert driver.run(cfg).done


def test_driver_lock_refuses_a_second_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "CONTROL_ROOT", tmp_path)
    with driver.driver_lock():
        with pytest.raises(driver.CampaignGateError, match="another tmplckpt driver"):
            with driver.driver_lock():
                pass
    with driver.driver_lock():
        pass


def test_preflight_from_a_main_clone(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    (clone / plan.PREREG_PATH).parent.mkdir(parents=True)
    (clone / plan.PREREG_PATH).write_text("prereg")
    out = tmp_path / "out"
    for name, value in (
        ("OUTPUT_ROOT", out),
        ("QUEUE_ROOT", tmp_path / "queue"),
        ("HARD_RUN_SET_DIR", out / plan.CAMPAIGN / plan.RUN_SET_IDS[plan.HARD]),
        ("PATHS_CONFIG_PATH", tmp_path / "paths.yaml"),
        ("STAGING_REPORT_PATH", tmp_path / "report.json"),
        ("HOME_NODE", store.this_host()),
    ):
        monkeypatch.setattr(plan, name, value)
    plan.PATHS_CONFIG_PATH.write_text(staging.paths_yaml_text())
    plan.STAGING_REPORT_PATH.write_text(json.dumps({"passed": True, "at": "t"}))
    cfg = driver.driver_config(
        output_root=out, queue_root=tmp_path / "queue", repo_root=clone
    )
    state = {"transitions": {}}
    driver._preflight(cfg, state)
    warnings = state["transitions"]["preflight"]["warnings"]
    assert warnings and "missing plan" in warnings[0]
    (clone / plan.PREREG_PATH).unlink()
    with pytest.raises(driver.CampaignGateError, match="pre-registration"):
        driver._preflight(cfg, {"transitions": {}})
    (clone / plan.PREREG_PATH).write_text("prereg")
    plan.STAGING_REPORT_PATH.write_text(json.dumps({"passed": False}))
    with pytest.raises(driver.CampaignGateError, match="did not pass"):
        driver._preflight(cfg, {"transitions": {}})


def test_tmpl_card_is_pin_checked():
    assert driver.soft_card_roots()["statement teachers"] == driver.tmpl_card_root()


def test_copy_verified_removes_a_bad_temp(tmp_path, monkeypatch):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    dest = tmp_path / "out" / "a.bin"
    monkeypatch.setattr(
        staging.shutil, "copy2", lambda s, d: Path(d).write_bytes(b"corrupt")
    )
    with pytest.raises(RuntimeError, match="sha256"):
        staging.copy_verified(src, dest)
    assert not dest.exists() and not list(dest.parent.iterdir())
