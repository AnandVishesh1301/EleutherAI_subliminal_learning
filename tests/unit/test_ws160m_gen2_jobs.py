"""Part II (ws160m2) plan constants, YAMLs, jobs and expected artefacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_expected as exp
from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import is_valid_run_set_id

W3 = models.WAVE_3
_SHARED = Path("/shared/run_set") / plan.SHARED_GENERATED_DIRNAME
_GEN1_RUN_SETS = {p.run_set_id for p in plan.PARTITIONS.values()}
_ARCHIVED_STEMS = (
    "ws160m-dosehp-controls",
    "ws160m-dosehp-evals-hp",
    "ws160m-extra-dose-dog",
    "ws160m-ladder-teachers",
    "ws160m-main-teachers",
)


@pytest.fixture(scope="module", params=plan.GEN2_PARTITION_ORDER)
def bundle(request: pytest.FixtureRequest):
    name = request.param
    train = cgen.all_train_configs(name, wave=W3)
    verify = cgen.all_verify_configs(name, train)
    by_q = jobs.all_jobs(name, train, verify, wave=W3, shared_config_root=_SHARED)
    return name, train, verify, by_q


def _yaml_of(job) -> Path:
    return Path(job.cmd.split("--sweep-config ", 1)[1].split()[0])


def test_gen2_partitions_names_nodes_and_reports() -> None:
    assert (
        tuple(plan.GEN2_PARTITIONS) == plan.GEN2_PARTITION_ORDER == ("matrix", "grid")
    )
    grid, matrix = plan.partition("grid"), plan.partition("matrix")
    assert (grid.node, matrix.node) == (plan.NODE_CLUSTER3, plan.NODE_CLUSTER4)
    assert (grid.alias, matrix.alias) == ("a40-cluster-3", "a40-cluster-4")
    assert grid.run_set_id == "20260918_ws160m2-grid_160m"
    assert matrix.run_set_id == "20260918_ws160m2-matrix_160m"
    for p in (grid, matrix):
        assert is_valid_run_set_id(p.run_set_id) and p.gen == 2 and p.wave == W3
        assert p.lock_prefix == "/tmp/subliminal_jobq_vishesh" and p.trains_teachers
        assert p.tmux_session == "ws160m2" != plan.TMUX_SESSION
        assert p.deadline_hours == 108.0
        assert p.staging_report == "staging_report_ws160m2.json"
        assert p.smoke_report == "smoke_report_ws160m2.json"
        assert (
            p.generated_config_dir == f"configs/experiment/generated/ws160m2/{p.name}"
        )
        spec = plan.spec_of(p.name)
        assert spec.run_set_id == spec.teacher_run_set_id == p.run_set_id
        assert spec.animals == ("bear", "lion", "tiger", "eagle")
        assert driver_io.CAMPAIGN_SPECS[f"ws160m2-{p.name}"] is spec
    assert plan.partition("main").gen == 1
    assert plan.partition("main").staging_report == "staging_report.json"
    assert plan.SPEC_REGISTRY == "ws160m"
    assert plan.PATHS_CONFIG == "configs/paths/paths_ws160m.yaml"
    assert plan.GEN2_CLAIM_IDS == tuple(f"W{i}" for i in range(9, 17))
    assert plan.LMEVAL_ATTEMPT_CAP == 3


def test_gen2_queue_names_are_unique_fleet_wide_and_in_plan_order() -> None:
    names = plan.all_queue_names()
    assert len(names) == len(set(names))
    gen1 = {q for p in plan.PARTITIONS.values() for q in plan.queue_priorities(p)}
    for p in plan.GEN2_PARTITIONS.values():
        pri = plan.queue_priorities(p)
        assert all(q.startswith(f"ws160m2-{p.name}-") for q in pri)
        assert not set(pri) & gen1
        assert not any(q.startswith("_archived_") for q in pri)
        assert not any(q.startswith(stem) for q in pri for stem in _ARCHIVED_STEMS)
        values = list(pri.values())
        assert values == sorted(values, reverse=True) and min(values) > 0
    order = [
        q.removeprefix("ws160m2-matrix-")
        for q in plan.queue_priorities(plan.partition("matrix"))
    ]
    students = [
        s for s in order if s in plan.gen2_student_stages(plan.partition("matrix"))
    ]
    assert students == [
        "m2",
        "b2",
        "ml",
        "h8",
        "law-p64",
        "law-p128",
        "law-p32",
        "law-p256",
        "law-p16",
        "law-p512",
        "rt",
        "td",
        "le",
        "b3",
        "xb2",
    ]
    assert order.index("h8") < order.index("lmeval-m2") < order.index("teachers-b2")
    assert order.index("teachers-b2") < order.index("law-p64")
    assert order[:2] == ["teachers", "teacher-evals"] and order[-1] == "evals-force"
    grid = [
        q.removeprefix("ws160m2-grid-")
        for q in plan.queue_priorities(plan.partition("grid"))
    ]
    rungs = [s for s in grid if re.fullmatch(r"r\d+", s)]
    assert rungs == [f"r{r}" for r in plan.K2_RUNG_PRIORITY]
    assert grid.index("r32000") < grid.index("teachers-b2") < grid.index("r1000")
    assert grid.index("d2") < grid.index("hm-main-rev")


def test_wandb_log_few_is_four_per_gen2_partition(bundle) -> None:
    name, _, _, by_q = bundle
    assert plan.wandb_jobs_expected(plan.partition(name)) == 4
    logged = [
        j
        for js in by_q.values()
        for j in js
        if (jobs.TEACHER_MODULE in j.cmd or "sweep_distill" in j.cmd)
        and "--no-wandb" not in j.cmd
    ]
    assert len(logged) == jobs.wandb_jobs_active(name, wave=W3) == 4


def test_every_job_reads_an_absolute_yaml_that_config_gen_writes(bundle) -> None:
    name, train, verify, by_q = bundle
    p = plan.partition(name)
    assert list(by_q) == [q for q in plan.queue_priorities(p) if q in by_q]
    written = set(cgen.shared_copy({**train, **verify}, _SHARED))
    all_jobs = [j for js in by_q.values() for j in js]
    assert len({j.jobkey for j in all_jobs}) == len(all_jobs)
    assert max(len(j.key) for j in all_jobs) <= 200
    for j in all_jobs:
        path = _yaml_of(j)
        assert path.is_absolute() and path.parent == _SHARED and path.name in written
        assert "--paths-config configs/paths/paths_ws160m.yaml" in j.cmd
        assert "--spec-registry ws160m" in j.cmd
        assert j.slots == 1 and not re.search(r"@step0(?!\d)", j.cmd)
        if jobs.TEACHER_MODULE in j.cmd or "sweep_distill" in j.cmd:
            assert "--force" not in j.cmd
    mems = Counter(j.mem_mib for j in all_jobs)
    assert set(mems) == {19000, 21500, 6000, 12000}
    for doc in cgen.shared_copy(verify, _SHARED).values():
        assert Path(doc["inherit_grid"]).parent == _SHARED
    with pytest.raises(ValueError, match="shared_config_root"):
        jobs.all_jobs(name, train, verify, wave=W3)


def test_no_gen2_yaml_writes_into_a_part_one_run_set(bundle) -> None:
    name, train, _, _ = bundle
    p = plan.partition(name)
    reads: Counter = Counter()
    for fname, doc in train.items():
        assert doc["run_set_id"] == p.run_set_id, fname
        assert doc["spec_registry"] == "ws160m"
        if not cgen.is_teacher_yaml(fname):
            reads[doc["teacher_run_set_id"]] += 1
            assert doc["run_set_teacher_subdir"] == cgen.teacher_subdir_scope(name)
    assert set(reads) & _GEN1_RUN_SETS, "reused Part I teachers are read in place"
    scope = cgen.teacher_subdir_scope(name).split("+")
    assert scope == sorted(scope) and f"sdf_teachers/{p.run_set_id}" in scope
    expected_scope = {"matrix": 2, "grid": 4}[name]
    assert len(scope) == expected_scope
    names = set(train)
    assert not any("_w2_" in n for n in names)
    for combo in pairs.teacher_combos(name, wave=W3):
        doc = train[cgen.teacher_yaml_name(name, combo)]
        assert doc["queue"] == plan.queue(p, combo.stage)
        assert doc["animals"] == [combo.animal] and doc["base_models"] == combo.token


def test_job_counts_per_queue_kind(bundle) -> None:
    name, _, _, by_q = bundle
    p = plan.partition(name)
    all_jobs = [j for js in by_q.values() for j in js]
    students = [j for j in all_jobs if "--no-control" in j.cmd]
    controls = [
        j for j in all_jobs if "--controls-only" in j.cmd and "sweep_distill" in j.cmd
    ]
    teachers = [j for j in all_jobs if jobs.TEACHER_MODULE in j.cmd]
    xbase = [
        j
        for j in all_jobs
        if "--xbase-controls-only" in j.cmd and "sweep_distill" in j.cmd
    ]
    assert len(students) == len(pairs.leaves_of(pairs.cells_of(name, wave=W3)))
    assert len(controls) == len(pairs.controls_of(name, wave=W3))
    assert len(teachers) == len(pairs.teacher_leaves(name, wave=W3))
    assert len(xbase) == (30 if name == "matrix" else 0)
    for stage in plan.gen2_student_stages(p):
        train_q = plan.queue(p, stage)
        eval_q = cgen.eval_queue(name, train_q)
        lm_q = cgen.lmeval_queue(name, train_q)
        assert by_q[train_q] and len(by_q[eval_q]) == len(by_q[lm_q]) > 0
        assert cgen.eval_dependencies(name, eval_q) == (train_q,)
        assert {j.mem_mib for j in by_q[lm_q]} == {plan.LMEVAL_MEM_MIB}
        assert {j.mem_mib for j in by_q[eval_q]} == {plan.EVAL_MEM_MIB}
        assert all(j.key.startswith(f"{name}-lmeval|") for j in by_q[lm_q])
    total = {"matrix": 5070, "grid": 5188}[name]
    assert len(all_jobs) == total


def test_student_queue_holds_its_own_controls_first_stage_wins() -> None:
    train = cgen.all_train_configs("matrix", wave=W3)
    ctrl = {n: d for n, d in train.items() if cgen.is_control_yaml(n)}
    by_q = jobs.control_jobs("matrix", ctrl, wave=W3, shared_config_root=_SHARED)
    assert len(by_q["ws160m2-matrix-law-p64"]) == 54
    assert len(by_q["ws160m2-matrix-rt"]) == 45
    assert "ws160m2-matrix-controls" not in by_q
    rt_keys = {j.key for j in by_q["ws160m2-matrix-rt"]}
    assert not any("|pile-ws1@" in k or "|pythia-160m@" in k for k in rt_keys)
    used = set()
    for queue, queue_jobs in by_q.items():
        for job in queue_jobs:
            _, animal, token, gkd_tag, _ = job.key.split("|")
            doc = ctrl[_yaml_of(job).name]
            used.add(_yaml_of(job).name)
            assert doc["queue"] == queue and doc["animals"] == [animal]
            assert [token, token] in doc["teacher_student_base_pairs"]
            (lr,) = doc["student_lr"]["lora"]
            (mode,) = doc["grid"]["student_modes"]
            assert f"_lr{plan.lr_slug(lr)}_" in gkd_tag
            assert f"_{cgen._epochs_slug(doc['num_epochs'])}_" in gkd_tag
            assert gkd_tag.endswith("_" + mode.removeprefix("lora-"))
    assert used == set(ctrl)


def test_refilled_config_dirs_are_force_evaluated_only_for_b3() -> None:
    idents = jobs.refill_idents("matrix", wave=W3)
    assert len(idents) == 10 and all(i.startswith("verify_matrix_b3_") for i in idents)
    assert jobs.refill_idents("grid", wave=W3) == set()
    assert jobs.refill_idents("main", wave=1) == set()
    train = cgen.all_train_configs("matrix", wave=W3)
    verify = cgen.all_verify_configs("matrix", train)
    by_q = jobs.all_jobs("matrix", train, verify, wave=W3, shared_config_root=_SHARED)
    forced = {q for q, js in by_q.items() for j in js if "--force" in j.cmd}
    assert forced == {"ws160m2-matrix-evals-b3", "ws160m2-matrix-lmeval-b3"}
    assert sum("--force" in j.cmd for j in by_q["ws160m2-matrix-evals-b3"]) == 10


def test_assert_clean_refuses_pending_runs_and_allows_the_full_roster() -> None:
    for token in (
        "data-order-b@step143000",
        "data-order-b-v1@step8000",
        "pythia-160m-repro@step143000",
        "weight-seed2@step143000",
        "dclm-ws2@step2000",
        "seed7@step143000",
    ):
        with pytest.raises(ValueError, match="excluded"):
            jobs.assert_clean(f"x --base-models {token}", "k", wave=W3)
    with pytest.raises(ValueError, match="step0"):
        jobs.assert_clean("x --base-models pile-main@step0", "k", wave=W3)
    for token in (
        "pile-main@step143000",
        "pile-ws2@step143000",
        "pythia-160m@step1000",
    ):
        jobs.assert_clean(f"x --base-models {token}", "k", wave=W3)
    with pytest.raises(ValueError, match="wave-3"):
        jobs.assert_clean("x --base-models pile-main@step143000", "k", wave=1)


def test_run_dirs_never_collide_and_gen2_stays_in_gen2_trees(tmp_path: Path) -> None:
    every = (*plan.PARTITION_ORDER, *plan.GEN2_PARTITION_ORDER)
    n_all = exp.assert_unique_run_dirs(tmp_path, partitions=every)
    n_gen1 = exp.assert_unique_run_dirs(tmp_path)
    n_gen2 = exp.assert_unique_run_dirs(tmp_path, partitions=plan.GEN2_PARTITION_ORDER)
    assert n_all == n_gen1 + n_gen2 and n_gen2 == 2535 + 2619 + 459 + 369 + 30
    for name in plan.GEN2_PARTITION_ORDER:
        rs = plan.partition(name).run_set_id
        written = (
            exp.expected_leaf_dirs(name, tmp_path, wave=W3)
            + exp.expected_control_dirs(name, tmp_path, wave=W3)
            + exp.expected_xbase_dirs(name, tmp_path)
            + exp.owned_teacher_dirs(name, tmp_path, wave=W3)
        )
        assert all(rs in d.parts for d in written)
        assert not any(g1 in d.parts for d in written for g1 in _GEN1_RUN_SETS)
    assert exp.expected_xbase_dirs("grid", tmp_path) == []
    gseeds = exp.expected_config_gseeds("matrix", tmp_path, wave=W3)
    assert max(len(v) for v in gseeds.values()) == 3


def test_reused_layout_matches_part_one_trees(tmp_path: Path) -> None:
    teachers = exp.reused_teacher_dirs("grid", tmp_path, wave=W3)
    controls = exp.reused_control_dirs("grid", tmp_path, wave=W3)
    assert (len(teachers), len(controls)) == (129, 138)
    assert all(set(d.parts) & _GEN1_RUN_SETS for d in (*teachers, *controls))
    rel = str(controls[0].relative_to(tmp_path))
    assert rel.startswith("gkd_students/20260916_ws160m-") and (
        "/controls/base_teacher/bear/" in rel
    )
    r16 = tmp_path / (
        "gkd_students/20260916_ws160m-ladder_160m/controls/base_teacher/bear/"
        "pile-ws1-step16000/lmb1.0_b1.0_lora_lr4e-4_e1_r8a16/gseed0"
    )
    assert r16 in controls
    t16 = tmp_path / (
        "sdf_teachers/20260916_ws160m-ladder_160m/bear/pile-ws1-step16000/"
        "ep5_lr5e-5_r128a128/ftseed0/checkpoints"
    )
    assert t16 in teachers
    assert len(exp.missing_reused("grid", tmp_path, wave=W3)) == 129 + 138
    assert exp.reused_teacher_dirs("main", tmp_path, wave=1) == []


def test_missing_reused_clears_once_the_leaf_and_its_summary_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    done: set[Path] = set()
    monkeypatch.setattr(exp, "cell_is_done", lambda d: Path(d) in done)
    controls = exp.reused_control_dirs("matrix", tmp_path, wave=W3)
    teachers = exp.reused_teacher_dirs("matrix", tmp_path, wave=W3)
    assert (len(teachers), len(controls)) == (69, 39)
    done.update(d.parent for d in teachers)
    done.update(controls)
    assert exp.missing_reused("matrix", tmp_path, wave=W3) == controls
    for d in controls:
        d.parent.mkdir(parents=True, exist_ok=True)
        (d.parent / "preference_summary.json").write_text(json.dumps({}))
    assert exp.missing_reused("matrix", tmp_path, wave=W3) == []
    done.discard(teachers[0].parent)
    assert exp.missing_reused("matrix", tmp_path, wave=W3) == [teachers[0]]


def test_stage_teacher_gate_reads_only_that_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "cell_is_done", lambda d: False)
    b2 = exp.missing_stage_teachers("matrix", "b2", tmp_path, wave=W3)
    assert {t.combo.owner.name for t in b2} == {"main"}
    m2 = exp.missing_stage_teachers("matrix", "m2", tmp_path, wave=W3)
    assert {t.combo.owner.name for t in m2} == {"main", "matrix"}
    assert exp.missing_stage_teachers("matrix", "xb2", tmp_path, wave=W3) == []
    tiger = exp.missing_stage_teachers("grid", "ac-tiger", tmp_path, wave=W3)
    assert {(t.combo.animal, t.combo.stage) for t in tiger} == {
        ("tiger", "teachers-b2")
    }
    r1000 = exp.missing_stage_teachers("grid", "r1000", tmp_path, wave=W3)
    assert {(t.combo.step, t.combo.stage) for t in r1000} == {(1000, "teachers-b2")}
    r16000 = exp.missing_stage_teachers("grid", "r16000", tmp_path, wave=W3)
    assert Counter(t.combo.owner.name for t in r16000) == {"ladder": 27, "grid": 6}


def test_every_gen2_token_resolves_through_the_shared_roster(
    bundle, tmp_path: Path
) -> None:
    _, train, _, _ = bundle
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"audit_root: {tmp_path}\n")
    reg = models.registry_for_paths(paths)
    tokens = {
        t
        for name, doc in train.items()
        if not cgen.is_teacher_yaml(name)
        for pair in doc["teacher_student_base_pairs"]
        for t in pair
    } | {doc["base_models"] for n, doc in train.items() if cgen.is_teacher_yaml(n)}
    assert tokens <= set(reg)
    for token in sorted(tokens):
        (spec,) = parse_base_models(token, registry=reg)
        base, step = models.parse_token(token)
        assert spec.label == models.label(base, step)
        assert (spec.local_root is not None) == (models.base(base).source == "local")
    modes = {
        doc["grid"]["student_modes"][0]
        for n, doc in train.items()
        if not cgen.is_teacher_yaml(n)
    }
    assert all(re.fullmatch(r"lora-r\d+a\d+", m) for m in modes)
