from __future__ import annotations

from pathlib import Path

import pytest

from pipelines import crossval2_gates as gates
from pipelines.sdfpos_campaign_pipeline import hooks_of, plan_module, queue_plan
from subliminal_learning.campaign import crossval2_plan as plan
from subliminal_learning.campaign import expected

_OUT = Path("/o")


@pytest.mark.parametrize("name", list(plan.EXPERIMENTS))
def test_jobs_cover_every_expected_leaf_once(name: str) -> None:
    e = plan.experiment(name)
    by_q = plan.all_jobs(e)
    assert set(by_q) == {e.queue(s) for s in e.stages}
    n_cells = len(e.arms) * len(e.variants)
    assert len(by_q[e.queue("students")]) == n_cells * len(e.ftseeds) * len(e.gseeds)
    if e.trains_teachers:
        assert len(by_q[e.queue("teachers")]) == len(e.arms) * len(e.ftseeds)
        assert len(by_q[e.queue("teacher-evals")]) == len(e.arms)
        assert len(by_q[e.queue("teacher-panels")]) == len(e.arms)
    else:
        assert "teachers" not in e.stages and not plan.expected_teacher_dirs(e, _OUT)
    n_ctrl_cfg = len(e.variants) if e.control_gseeds else 0
    assert len(by_q[e.queue("evals")]) == n_cells + n_ctrl_cfg
    assert len(by_q[e.queue("panels")]) == n_cells + n_ctrl_cfg + int(e.base_panel)
    for js in by_q.values():
        for job in js:
            assert "--force" not in job.cmd and "angela" not in job.cmd
            if "scripts.sweeps" in job.cmd:
                assert f"--paths-config {plan.PATHS_CONFIG}" in job.cmd
                assert f"--base-models {e.base}" in job.cmd
    for job in by_q[e.queue("students")]:
        assert "--no-control" in job.cmd and plan.STUDENT_EXPERIMENT_CONFIG in job.cmd


def test_jobkeys_and_queue_names_are_unique_across_the_campaign() -> None:
    keys = [
        j.key
        for e in plan.EXPERIMENTS.values()
        for js in plan.all_jobs(e).values()
        for j in js
    ]
    queues = [q for e in plan.EXPERIMENTS.values() for q in plan.queue_priorities(e)]
    assert len(keys) == len(set(keys)) and len(queues) == len(set(queues))
    assert all(q.startswith(f"{plan.CAMPAIGN}-") for q in queues)


@pytest.mark.parametrize("name", list(plan.EXPERIMENTS))
def test_plan_dirs_equal_the_sweep_enumeration(name: str) -> None:
    e = plan.experiment(name)
    spec = plan.spec_of(name)
    docs = plan.student_train_configs(e)
    sweep_students, sweep_controls = [], []
    for arm in e.arms:
        for v in e.variants:
            sweep_students += expected.treatment_leaf_dirs(
                docs[plan.student_yaml_name(arm, v)], _OUT, spec=spec
            )
    for v in e.variants if e.control_gseeds else ():
        doc = docs[plan.control_yaml_name(v)]
        fn = (
            expected.xbase_control_leaf_dirs
            if e.cross_init
            else expected.control_leaf_dirs
        )
        sweep_controls += fn(doc, _OUT, spec=spec)
    assert sorted(sweep_students) == sorted(plan.expected_student_dirs(e, _OUT))
    assert sorted(sweep_controls) == sorted(plan.expected_control_dirs(e, _OUT))
    assert all("_strict_v2" in d.parent.name for d in sweep_students + sweep_controls)


def test_reuse_run_sets_read_their_source_teachers() -> None:
    for e in plan.EXPERIMENTS.values():
        docs = plan.student_train_configs(e)
        src = plan.experiment(e.teacher_from) if e.teacher_from else e
        assert {d["teacher_run_set_id"] for d in docs.values()} == {src.run_set_id}
        assert src.trains_teachers and src.base == (e.teacher_base or e.base)
        assert {a.slug for a in e.arms} <= {a.slug for a in src.arms}
        assert set(e.ftseeds) <= set(src.ftseeds)
        deps = e.depend_queues("students")
        assert deps == (src.queue("teachers"),)
        assert src.level <= e.level
    x = plan.experiment("p2-xinit-160m")
    pair = plan.student_train_configs(x)[plan.student_yaml_name(x.arms[0], plan.HEAD)]
    assert pair["teacher_student_base_pairs"] == [[plan.BASE_SEED2, plan.BASE_160M]]
    assert "_tbase-" in plan.expected_student_dirs(x, _OUT)[0].parts[-3]


def test_variants_reach_the_sweep_grid() -> None:
    docs = plan.student_train_configs(plan.experiment("p5-fullsgd-160m"))
    d = next(v for k, v in docs.items() if k.startswith("student_"))
    assert d["grid"]["student_modes"] == ["full"] and d["grid"]["student_optims"] == [
        "sgd"
    ]
    assert d["student_lr"]["full/sgd"] == [0.1] and d["num_epochs"] == 4.0
    rms = next(
        iter(plan.student_train_configs(plan.experiment("p4-fullrms-seed2")).values())
    )
    assert rms["student_lr"]["full/rmsprop"] == [1e-5] and rms["grid"][
        "student_optims"
    ] == ["rmsprop"]
    head = next(iter(plan.student_train_configs(plan.experiment("p0-160m")).values()))
    assert "student_optims" not in head["grid"] and head["student_lr"]["lora"] == [4e-4]
    assert head["rollout_screen"] == plan.ROLLOUT_SCREEN == "strict_v2"
    ch = plan.experiment("p3-channel-160m")
    cells = {(v.lmbda, v.beta) for v in ch.variants}
    assert cells == {(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)}
    assert plan.HEAD.gkd_tag == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict_v2"


def test_priorities_order_every_level_before_the_next() -> None:
    by_level: dict[int, list[int]] = {}
    for e in plan.EXPERIMENTS.values():
        by_level.setdefault(e.level, []).extend(plan.queue_priorities(e).values())
    levels = sorted(by_level)
    assert levels[0] == 0
    for lo, hi in zip(levels, levels[1:]):
        assert min(by_level[lo]) > max(by_level[hi])
    assert {e.level for e in plan.earlier(plan.experiment("p1-seeds-160m"))} == {0}
    assert not plan.earlier(plan.experiment("p0-seed2"))


def test_teacher_recipe_matches_sdfpos2_and_regimes() -> None:
    p0 = plan.experiment("p0-160m")
    tags = {a.param_tag for a in p0.arms}
    assert "s750_lr5e-5_r128a128_c-cross-L-n2096" in tags
    assert "s750_lr5e-5_r128a128_c-placebo-lh-n2096" in tags
    regime = plan.teacher_train_configs(plan.experiment("p2-regime-seed2"))
    steps = {d["train_overrides"]["max_steps"] for d in regime.values()}
    assert steps == {375, 1500}
    assert {d["base_models"] for d in regime.values()} == {plan.BASE_SEED2}
    t = plan.teacher_train_configs(p0)["teacher_cross-R-n2096-s750.yaml"]
    assert t["sdf_docs"]["bear"] == str(
        plan.CORPORA_ROOT / "crossval" / "cross-R-n2096" / "docs.jsonl"
    )
    assert (
        plan.experiment("p1-anchor-160m").rule_version(
            plan.experiment("p1-anchor-160m").arms[0]
        )
        == "crossval2_anchor_cf_v1"
    )


def test_campaign_is_wired_into_the_sdfpos_driver() -> None:
    assert plan_module("crossval2") is plan
    assert hooks_of("crossval2") is gates and hooks_of("sdfpos2") is None
    qp = queue_plan("p3-channel-160m", "crossval2")
    assert qp["n_jobs"] == sum(
        len(js) for js in plan.all_jobs(plan.experiment("p3-channel-160m")).values()
    )
    roots = plan.paths_config_doc().values()
    assert all(Path(r).is_relative_to(plan.ROOT) for r in roots)
    env = plan.gpu_policy_json()["env"]
    assert (
        env["HF_HUB_CACHE"] == str(plan.HF_ROOT / "hub")
        and env["HF_HUB_OFFLINE"] == "1"
    )


def test_review_and_acceptance_gates(tmp_path, monkeypatch) -> None:
    review = tmp_path / "r.json"
    monkeypatch.setattr(
        plan, "CORPUS_REVIEWS", {**plan.CORPUS_REVIEWS, plan.CROSSVAL: review}
    )
    assert gates.review_problems(plan.CROSSVAL, {"cross-L-n2096": "a"})
    review.write_text('{"approved": true, "corpora": {"cross-L-n2096": "a"}}')
    assert gates.review_problems(plan.CROSSVAL, {"cross-L-n2096": "a"}) == []
    assert gates.review_problems(plan.CROSSVAL, {"cross-L-n2096": "b"})
    acc = tmp_path / "acc.json"
    monkeypatch.setattr(plan, "ACCEPTANCE_PATH", acc)
    acc.write_text(
        '{"sizes": {"n2096": {"all_pass": false, "V-C1": {"pass": true},'
        ' "V-C5": {"pass": false}}}}'
    )
    assert gates.acceptance_problems()
    review.write_text('{"approved": true, "vc5_passed_after_audit": true}')
    assert gates.acceptance_problems() == []
    acc.write_text('{"sizes": {"n2096": {"all_pass": false, "V-C3": {"pass": false}}}}')
    assert gates.acceptance_problems()


def test_tier_gate_waits_for_every_lower_level_training_queue(tmp_path) -> None:
    from subliminal_learning.jobqueue import store
    from subliminal_learning.jobqueue.model import Job

    p1 = plan.experiment("p1-anchor-160m")
    assert gates.submit_waiting(plan.experiment("p0-160m"), tmp_path) is None
    assert "not submitted" in gates.submit_waiting(p1, tmp_path)
    lower = [x for x in plan.EXPERIMENTS.values() if x.level == 0]
    for x in lower:
        for stage in x.training_stages:
            store.create_queue(tmp_path, x.queue(stage))
            store.append_jobs(tmp_path, x.queue(stage), [Job(key="j", cmd="true")])
    assert "pending" in gates.submit_waiting(p1, tmp_path)
    for x in lower:
        for stage in x.training_stages:
            job = store.claim_next(tmp_path, x.queue(stage), "h", 1)
            assert job is not None
    assert gates.submit_waiting(p1, tmp_path) is None


def test_students_wait_strictly_for_their_teachers() -> None:
    for e in plan.EXPERIMENTS.values():
        assert e.strict_dep_stages == ("students",)
        assert set(e.strict_dep_stages) <= set(e.stages)
