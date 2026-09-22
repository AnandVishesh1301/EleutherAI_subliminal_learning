from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pipelines.train_sdf_teacher_pipeline import TrainSdfTeacherConfig, sdf_run_identity
from subliminal_learning.campaign import sdfpos_plan as plan
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.sdf.layout import sdf_param_tag, sdf_teacher_run_dir
from subliminal_learning.training.run_manifest import ERROR, SKIP, identity_gate_action

_TC = TrainConfig(
    learning_rate=5e-5, n_epochs=5, batch_size=8, max_length=2048, max_grad_norm=1.0,
    warmup_ratio=0.03, lr_scheduler_type="cosine", seed=0, optim="adamw_torch",
)


def test_param_tag_legacy_is_byte_identical_and_new_axes_compose() -> None:
    assert sdf_param_tag(_TC, plan.LORA) == "ep5_lr5e-5_r128a128"
    assert sdf_param_tag(_TC, plan.LORA, None) == "ep5_lr5e-5_r128a128"
    assert sdf_param_tag(replace(_TC, max_steps=750), plan.LORA) == "s750_lr5e-5_r128a128"
    assert sdf_param_tag(_TC, plan.LORA, "pos-start") == "ep5_lr5e-5_r128a128_c-pos-start"
    assert (
        sdf_param_tag(replace(_TC, max_steps=750), None, "size-n65")
        == "s750_lr5e-5_full_c-size-n65"
    )
    legacy = sdf_teacher_run_dir(Path("/o"), "bear", plan.BASE_SPEC, "EleutherAI", _TC, plan.LORA)
    assert legacy == Path("/o/sdf_teachers/bear/pythia-160m@step143000/ep5_lr5e-5_r128a128/ftseed0")


def test_arm_param_tags_and_run_dirs_are_unique_per_experiment() -> None:
    for e in plan.EXPERIMENTS.values():
        tags = [a.param_tag for a in e.arms]
        assert len(tags) == len(set(tags)), e.name
        dirs = plan.expected_teacher_dirs(e, Path("/o")) + plan.expected_student_dirs(
            e, Path("/o")
        )
        assert len(dirs) == len(set(dirs)), e.name
    size = plan.experiment("size")
    assert "s750_lr5e-5_r128a128_c-size-n65" in {a.param_tag for a in size.arms}
    assert "ep5_lr5e-5_r128a128_c-size-n65" in {a.param_tag for a in size.arms}
    assert plan.experiment("density").corpus_dir(plan.Arm("den-f0")).parts[-2] == "position"


def test_run_sets_nodes_and_queues() -> None:
    ids = {e.run_set_id for e in plan.EXPERIMENTS.values()}
    assert len(ids) == len(plan.EXPERIMENTS)
    assert {e.node for e in plan.EXPERIMENTS.values()} == set(plan.FLEET_NODES)
    names = [q for e in plan.EXPERIMENTS.values() for q in plan.queue_priorities(e)]
    assert len(names) == len(set(names))
    pr = plan.queue_priorities(plan.experiment("dose"))
    assert pr["sdfpos-dose-teachers"] > pr["sdfpos-dose-evals"]
    assert plan.QUEUE_DEPENDS["students"] == ("teachers",)
    assert set(plan.QUEUE_DEPENDS["evals"]) == {"students", "controls"}


@pytest.mark.parametrize("name", list(plan.EXPERIMENTS))
def test_jobs_cover_every_expected_leaf(name: str) -> None:
    e = plan.experiment(name)
    by_q = plan.all_jobs(e)
    n_arms, n_students = len(e.arms), len(e.student_arms)
    assert len(by_q[e.queue("teachers")]) == n_arms * len(plan.FTSEEDS)
    assert len(by_q[e.queue("teacher-evals")]) == n_arms
    assert len(by_q[e.queue("students")]) == n_students * len(plan.FTSEEDS) * len(plan.GSEEDS)
    assert len(by_q[e.queue("controls")]) == len(plan.GSEEDS)
    assert len(by_q[e.queue("evals")]) == n_students + 1
    for job in by_q[e.queue("teachers")]:
        assert "--param-tags s750_" in job.cmd or "--param-tags ep5_" in job.cmd
        assert "--no-wandb" in job.cmd
    assert all("--force" not in j.cmd for js in by_q.values() for j in js)


def test_generated_yamls_carry_corpus_and_budget() -> None:
    e = plan.experiment("size")
    docs = plan.teacher_train_configs(e)
    s750 = docs["teacher_size-n65-s750.yaml"]
    ep5 = docs["teacher_size-n65-ep5.yaml"]
    assert s750["train_overrides"] == {"max_steps": plan.TEACHER_STEPS}
    assert "train_overrides" not in ep5
    assert s750["corpus_tag"] == ep5["corpus_tag"] == "size-n65"
    assert s750["sdf_docs"]["bear"].endswith("corpora_v2/size/size-n65/docs.jsonl")
    assert s750["run_set_id"] == e.run_set_id
    students = plan.student_train_configs(e)
    doc = students["student_size-n65-s750.yaml"]
    assert doc["teacher_param_tags"]["lora"] == "s750_lr5e-5_r128a128_c-size-n65"
    assert doc["teacher_run_set_id"] == e.run_set_id and doc["num_epochs"] == 1.0
    verify = plan.verify_configs(e, {**docs, **students})
    assert verify["verify_student_size-n65-s750.yaml"]["eval"]["prompt_sets"] == list(
        plan.EVAL_PROMPT_SETS
    )
    assert "consensus" in plan.EVAL_PROMPT_SETS and "dislike" in plan.EVAL_PROMPT_SETS


def _cfg(tmp_path: Path, **kw) -> TrainSdfTeacherConfig:
    paths = PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path, wandb_dir=tmp_path
    )
    return TrainSdfTeacherConfig(
        paths=paths, hub_org="EleutherAI", spec=plan.BASE_SPEC, lora=plan.LORA, train=_TC,
        target_animal="bear", sdf_docs_root=tmp_path, **kw,
    )


def test_sdf_run_identity_corpus_tag_only_when_set(tmp_path: Path) -> None:
    legacy = sdf_run_identity(_cfg(tmp_path), {}, "docs")
    assert "corpus_tag" not in legacy
    tagged = sdf_run_identity(_cfg(tmp_path, corpus_tag="pos-start"), {}, "docs")
    assert tagged["corpus_tag"] == "pos-start"
    assert identity_gate_action({"run_identity": legacy}, legacy, force=False).action == SKIP
    assert identity_gate_action({"run_identity": legacy}, tagged, force=False).action == ERROR


def test_lora_config_matches_teacher_sweep_grid() -> None:
    assert plan.LORA == LoRAConfig(
        r=128, lora_alpha=128, lora_dropout=0.0,
        target_modules=("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
    )
