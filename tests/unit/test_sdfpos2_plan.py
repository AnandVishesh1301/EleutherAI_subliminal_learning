from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipelines.sdfpos_campaign_pipeline import queue_plan
from subliminal_learning.campaign import sdfpos2_plan as plan
from subliminal_learning.campaign import sdfpos_plan as p1
from subliminal_learning.teachers.model_specs import parse_base_models


def _key(arm) -> tuple[str, str]:
    return arm.corpus_tag, arm.regime


def test_base_tokens_resolve_to_the_declared_specs() -> None:
    for token, spec in plan.BASE_SPECS.items():
        assert parse_base_models(token)[0] == spec
    assert plan.experiment("seed2").base_spec.seed == 2
    assert {e.base for e in plan.EXPERIMENTS.values() if e.name != "seed2"} == {
        plan.BASE_160M
    }


def test_corpus_dirs_follow_the_build_and_experiment_subdir() -> None:
    main = plan.experiment("main")
    dirs = {a.corpus_tag: a.corpus_dir for a in main.arms}
    root3 = plan.DOCGEN_ROOT / "corpora_v3"
    assert dirs["pos-start-n8383"] == root3 / "position" / "pos-start-n8383"
    assert dirs["den-f0-n8383"] == root3 / "position" / "den-f0-n8383"
    assert dirs["val-neutral-n2096"] == root3 / "valence" / "val-neutral-n2096"
    topup = {a.corpus_tag: a.corpus_dir for a in plan.experiment("topup").arms}
    root2 = plan.DOCGEN_ROOT / "corpora_v2"
    assert topup["voice-first"] == root2 / "voice" / "voice-first"
    assert topup["dose-elab-eqm"] == root2 / "dose" / "dose-elab-eqm"
    with pytest.raises(ValueError):
        plan.Arm("pos-start", "v4")


def test_part1_reuse_arms_match_part1_exactly() -> None:
    part1 = {_key(a): a for e in p1.EXPERIMENTS.values() for a in e.arms}
    students = {k for k, a in part1.items() if a.students} - {("size-n65", "s750")}
    orphans = {k for k, a in part1.items() if not a.students}
    topup = plan.experiment("topup")
    assert {_key(a) for a in topup.arms} == students and len(topup.arms) == 29
    assert topup.ftseeds == (3, 4)
    retrained = [*plan.experiment("ladder").arms, *plan.experiment("sizeaux").arms]
    assert {_key(a) for a in retrained} == orphans
    for arm in [*topup.arms, *retrained]:
        assert arm.param_tag == part1[_key(arm)].param_tag
    assert plan.experiment("ladder").ftseeds == (0, 1, 2, 3, 4)
    assert plan.experiment("sizeaux").ftseeds == (0, 1, 2)


def test_nopack_is_token_matched_and_unpacked() -> None:
    e = plan.experiment("nopack")
    assert {a.param_tag for a in e.arms} == {
        f"s4931_lr5e-5_r128a128_c-pos-{b}-n8383" for b in ("start", "middle", "end", "mix")
    }
    assert abs(4931 * 8 * 311.5 - 750 * 8 * 2048) / (750 * 8 * 2048) < 1e-3
    packing = {
        cfg: yaml.safe_load(Path(cfg).read_text())["packing"]
        for cfg in (plan.TEACHER_EXPERIMENT_CONFIG, plan.TEACHER_NOPACK_EXPERIMENT_CONFIG)
    }
    assert packing == {
        plan.TEACHER_EXPERIMENT_CONFIG: True,
        plan.TEACHER_NOPACK_EXPERIMENT_CONFIG: False,
    }
    assert e.teacher_experiment_config == plan.TEACHER_NOPACK_EXPERIMENT_CONFIG
    others = [x for x in plan.EXPERIMENTS.values() if x.name != "nopack"]
    assert {x.teacher_experiment_config for x in others} == {plan.TEACHER_EXPERIMENT_CONFIG}
    job = plan.all_jobs(e)[e.queue("teachers")][0]
    assert f"--experiment-config {plan.TEACHER_NOPACK_EXPERIMENT_CONFIG}" in job.cmd


@pytest.mark.parametrize("name", list(plan.EXPERIMENTS))
def test_jobs_cover_every_expected_leaf(name: str) -> None:
    e = plan.experiment(name)
    by_q = plan.all_jobs(e)
    n_ctrl = len(e.control_gseeds)
    assert set(by_q) == {e.queue(s) for s in e.stages}
    assert len(by_q[e.queue("teachers")]) == len(e.arms) * len(e.ftseeds)
    assert len(by_q[e.queue("teacher-evals")]) == len(e.arms)
    assert len(by_q[e.queue("students")]) == len(e.arms) * len(e.ftseeds) * 2
    assert len(by_q[e.queue("evals")]) == len(e.arms) + (1 if n_ctrl else 0)
    if n_ctrl:
        assert len(by_q[e.queue("controls")]) == n_ctrl == 5
    else:
        assert "controls" not in e.stages
    for job in by_q[e.queue("teachers")] + by_q[e.queue("students")]:
        assert f"--base-models {e.base}" in job.cmd and "--no-wandb" in job.cmd
        assert f"--paths-config {plan.PATHS_CONFIG}" in job.cmd
    assert all("--force" not in j.cmd for js in by_q.values() for j in js)
    dirs = plan.expected_teacher_dirs(e, Path("/o")) + plan.expected_student_dirs(
        e, Path("/o")
    )
    assert len(dirs) == len(set(dirs))
    assert all(e.run_set_id in d.parts for d in dirs)


def test_controls_cover_both_bases_with_five_gseeds() -> None:
    owners = {e.base: e for e in plan.EXPERIMENTS.values() if e.control_gseeds}
    assert set(owners) == set(plan.BASE_SPECS)
    for e in owners.values():
        docs = plan.student_train_configs(e)
        assert docs[plan.CONTROL_YAML]["gseeds"] == [0, 1, 2, 3, 4]
        assert all(
            d["gseeds"] == [0, 1] for n, d in docs.items() if n != plan.CONTROL_YAML
        )
        assert docs[plan.CONTROL_YAML]["teacher_student_base_pairs"] == [[e.base, e.base]]
    ctrl = plan.expected_control_dirs(plan.experiment("seed2"), Path("/o"))
    assert len(ctrl) == 5 and all("pythia-160m-seed2" in str(d) for d in ctrl)


def test_generated_yamls_carry_corpus_budget_and_evals() -> None:
    e = plan.experiment("seed2")
    teachers = plan.teacher_train_configs(e)
    doc = teachers["teacher_pos-start-n8383-s750.yaml"]
    assert doc["train_overrides"] == {"max_steps": 750}
    assert doc["base_models"] == plan.BASE_SEED2 and doc["ft_seeds"] == [0, 1, 2, 3, 4]
    assert doc["sdf_docs"]["bear"] == str(
        plan.DOCGEN_ROOT / "corpora_v3" / "position" / "pos-start-n8383" / "docs.jsonl"
    )
    ep5 = plan.teacher_train_configs(plan.experiment("sizeaux"))["teacher_size-n131-ep5.yaml"]
    assert "train_overrides" not in ep5
    students = plan.student_train_configs(e)
    verify = plan.verify_configs(e, {**teachers, **students})
    for v in verify.values():
        assert v["eval"]["prompt_sets"] == [
            "test", "favorite", "test_sample", "consensus", "dislike"
        ]
    s = students["student_pos-start-n8383-s750.yaml"]
    assert s["teacher_param_tags"]["lora"] == "s750_lr5e-5_r128a128_c-pos-start-n8383"
    assert s["teacher_run_set_id"] == s["run_set_id"] == e.run_set_id


def test_queues_nodes_priorities_and_roots() -> None:
    names = [q for e in plan.EXPERIMENTS.values() for q in plan.queue_priorities(e)]
    names += [q for e in p1.EXPERIMENTS.values() for q in p1.queue_priorities(e)]
    assert len(names) == len(set(names))
    assert {e.node for e in plan.EXPERIMENTS.values()} == set(plan.FLEET_NODES)
    for node in plan.FLEET_NODES:
        pr: dict[str, int] = {}
        for e in plan.experiments_on(node):
            pr.update(plan.queue_priorities(e))
        teach = [v for k, v in pr.items() if k.endswith("-teachers")]
        stud = [v for k, v in pr.items() if k.endswith("-students")]
        assert min(teach) > max(stud)
    main, nopack = plan.queue_priorities(plan.experiment("main")), plan.queue_priorities(
        plan.experiment("nopack")
    )
    assert main["sdfpos2-main-students"] > nopack["sdfpos2-nopack-students"]
    assert queue_plan("nopack", "sdfpos2")["depends"]["sdfpos2-nopack-evals"] == [
        "sdfpos2-nopack-students"
    ]
    roots = plan.paths_config_doc().values()
    assert all(Path(r).is_relative_to(plan.CAMPAIGN_ROOT) for r in roots)
    env = plan.gpu_policy_json()["env"]
    assert env["HF_HUB_CACHE"] == str(plan.CAMPAIGN_ROOT / "hf" / "hub")
    assert env["HF_HUB_OFFLINE"] == "1" and plan.GPU_POLICY["cap_per_gpu"] == 2
