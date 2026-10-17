from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pipelines import newdom_soft_campaign_pipeline as driver
from pipelines.newdom_soft_staging_pipeline import domain_terms
from scripts.sweeps.sweep_distill_student_gkd import _mode_lora
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.campaign import newdom_soft_metrics as metrics
from subliminal_learning.evaluation.sdf_prompt_sets import prompt_set_path
from subliminal_learning.gkd.config import (
    load_gkd_config,
    load_gkd_training,
    parse_sweep_gkd_axes,
)
from subliminal_learning.gkd.layout import gkd_tag, train_tag_kwargs
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, assert_safe_jobkey
from subliminal_learning.sdf.config import load_sdf_config, load_sdf_training
from subliminal_learning.sdf.layout import sdf_param_tag
from subliminal_learning.training.run_set import is_valid_run_set_id

REPO = Path(__file__).resolve().parents[2]


def _treat(target: str = "oak", epochs: int = 5) -> soft.TeacherRef:
    return soft.TeacherRef(soft.KIND_TREAT, target, epochs)


def _placebo(target: str = "oak", steps: int = 750) -> soft.TeacherRef:
    return soft.TeacherRef(soft.KIND_PLACEBO, target, 5, steps)


def _all_jobs():
    jobs = list(soft.docgen_jobs())
    for t in plan.TARGETS:
        for ref in (_treat(t), _treat(t, 8), _placebo(t)):
            jobs += soft.teacher_train_jobs(ref) + [soft.teacher_eval_job(ref)]
            jobs += soft.leaf_jobs(soft.leaves_for(ref))
    return jobs


# ---- plan ----------------------------------------------------------------------------


def test_plan_counts_queues_and_priorities():
    doc = soft.plan_doc()
    assert doc["counts"] == {
        "docgen_shards": 96,
        "teachers": 18,
        "teachers_ep8_fallback_max": 18,
        "placebo_teachers": 18,
        "leaves": 216,
        "leaves_treat": 162,
        "leaves_placebo": 54,
    }
    assert all(n.startswith("newdom-s-") for n in soft.QUEUE_PRIORITIES)
    assert len(soft.QUEUE_PRIORITIES) == 1 + 6 * 8
    kinds = Counter(soft.QUEUE_FOOTPRINT.values())
    assert kinds == {"docgen": 1, "soft_teacher": 36, "soft_leaf": 12}
    for name, kind in soft.QUEUE_FOOTPRINT.items():
        assert soft.QUEUE_PRIORITIES[name] == plan.PRIORITIES[kind]
    assert driver.SPEC.queue_priorities == soft.QUEUE_PRIORITIES


def test_jobs_unique_safe_footprinted_and_unpinnable():
    jobs = _all_jobs()
    keys = [j.jobkey for j in jobs]
    assert len(keys) == len(set(keys))
    for job in jobs:
        assert len(job.jobkey) <= MAX_JOBKEY_LEN
        assert_safe_jobkey(job.jobkey, key=job.key)
        assert "--force" not in job.cmd and "angela" not in job.cmd.lower()
        assert "hostname" not in job.cmd
    docgen = soft.docgen_jobs()
    assert {(j.mem_mib, j.slots) for j in docgen} == {(44_000, plan.CAP_PER_GPU)}
    leaf = soft.leaf_jobs(soft.leaves_for(_treat()))
    assert {(j.mem_mib, j.slots) for j in leaf} == {(19_000, 3)}


def test_docgen_shards_seeds_and_argv():
    shards = soft.docgen_shards()
    assert [s.target for s in shards[:16]] == ["oak"] * 16
    assert shards[3].seed == 20_261_017 + 3 * 1_000_003
    assert {s.seed for s in shards if s.target == "oak"} == {
        s.seed for s in shards if s.target == "blue"
    }
    argv = soft.docgen_argv(shards[3], Path("/py"), Path("/out/attempt-00"))
    want = {
        "--paths-config": str(soft.PATHS_CONFIG_PATH),
        "--config": "configs/sdf_docgen_trees.yaml",
        "--animal": "oak",
        "--n-docs": "625",
        "--seed": str(shards[3].seed),
        "--output": "/out/attempt-00",
    }
    assert {k: argv[argv.index(k) + 1] for k in want} == want
    assert shards[3].dir == plan.DOCGEN_ROOT / "oak" / "shard-003"
    with pytest.raises(ValueError):
        soft.DocgenShard("oak", 16)
    with pytest.raises(ValueError):
        soft.DocgenShard("cat", 0)


def test_shard_and_done_problems():
    rows = [{"text": "x", "animal": "oak"}]
    assert soft.shard_problems(rows, "oak") == []
    assert soft.shard_problems([], "oak")
    assert soft.shard_problems([{"text": "x", "animal": "pine"}], "oak")
    assert soft.shard_problems([{"text": " ", "animal": "oak"}], "oak")
    shard = soft.DocgenShard("blue", 2)
    assert (
        soft.done_problems({"target": "blue", "index": 2, "seed": shard.seed}, shard)
        == []
    )
    assert soft.done_problems({"target": "blue", "index": 2, "seed": 1}, shard)


def test_select_argv_keeps_every_survivor():
    argv = soft.select_argv(Path("/py"), "pine", [Path("/a"), Path("/b")])
    assert argv[argv.index("--target-n") + 1] == "99999"
    assert argv[argv.index("--output") + 1] == str(plan.corpus_path("pine").parent)
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--input-runs"] == [
        "/a",
        "/b",
    ]


# ---- teachers ------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [None, 750])
def test_teacher_param_tag_matches_sdf_layout(steps):
    lora, train = load_sdf_training(REPO / soft.TEACHER_TRAINING_CONFIG)
    ref = _placebo(steps=steps) if steps else _treat()
    doc = soft.teacher_train_doc(ref)
    grid = doc["grid"]
    train = replace(
        train,
        n_epochs=grid["n_epochs"][0],
        learning_rate=grid["learning_rate"][0],
        max_steps=(doc.get("train_overrides") or {}).get("max_steps"),
    )
    lora = replace(lora, r=grid["lora"][0]["r"], lora_alpha=grid["lora"][0]["alpha"])
    assert sdf_param_tag(train, lora, doc.get("corpus_tag")) == ref.param_tag
    want = (
        "s750_lr5e-5_r128a128_c-lighthouse-treefree-v1"
        if steps
        else "ep5_lr5e-5_r128a128"
    )
    assert ref.param_tag == want


def test_teacher_refs_tokens_and_validation():
    for ref in (_treat(), _treat("blue", 8), _placebo("yellow", 812)):
        assert soft.teacher_from_token(ref.kind, ref.target, ref.token) == ref
    with pytest.raises(ValueError):
        soft.TeacherRef(soft.KIND_PLACEBO, "oak", 5)
    capped = soft.TeacherRef(soft.KIND_TREAT, "oak", 5, 10)
    assert capped.param_tag == "s10_lr5e-5_r128a128" and capped.corpus_tag is None
    assert _treat().docs_path == plan.corpus_path("oak")
    with pytest.raises(ValueError):
        soft.TeacherRef(soft.KIND_TREAT, "oak", 5, 0)
    with pytest.raises(ValueError):
        soft.TeacherRef(soft.KIND_TREAT, "oak", 2)
    assert _placebo("blue").train_queue == "newdom-s-pteach-blue"
    assert _treat("blue", 8).eval_queue == "newdom-s-teval-blue-ep8"


def test_treatment_and_placebo_teachers_share_card_invariants():
    t, p = soft.teacher_train_doc(_treat()), soft.teacher_train_doc(_placebo())
    assert t["sdf_docs_root"] == p["sdf_docs_root"] == str(soft.CORPUS_ROOT)
    assert "sdf_docs" not in t and "corpus_tag" in p
    assert soft.CORPUS_ROOT / "oak" / "sdf_oak_docs_selected.jsonl" == plan.corpus_path(
        "oak"
    )
    assert p["sdf_docs"] == {"oak": str(soft.placebo_slot_path("oak"))}
    assert t["run_set_id"] == p["run_set_id"] == plan.RUN_SET_IDS[plan.SOFT]


def test_teacher_train_cmd_checks_manifest():
    cmd = soft.teacher_train_cmd(_treat(), 1)
    assert cmd.endswith(
        f"test -f {soft.TeacherRef('treat', 'oak', 5).manifest_path(1)}"
    )
    assert "--ftseeds 1" in cmd and "--spec-registry ws160m" in cmd
    eval_cmd = soft.teacher_eval_cmd(_placebo())
    assert (
        "newdom_panel teacher --kind placebo --target oak --teacher ep5-s750"
        in eval_cmd
    )


def test_teacher_manifest_problems_and_steps():
    ref = _placebo()
    good = {
        "animal": "oak",
        "param_tag": ref.param_tag,
        "ft_seed": 1,
        "base_model": soft.BASE_LABEL,
        "finetune_mode": "lora",
        "run_set_id": soft.RUN_SET_ID,
        "sdf_docs_path": str(ref.docs_path),
        "sdf_docs_sha256": "abc",
        "n_train_steps": 750,
    }
    assert soft.teacher_manifest_problems(ref, 1, good, "abc") == {}
    bad = {**good, "sdf_docs_path": str(plan.corpus_path("oak")), "n_train_steps": 749}
    assert set(soft.teacher_manifest_problems(ref, 1, bad, "abc")) == {
        "sdf_docs_path",
        "n_train_steps",
    }
    assert soft.teacher_steps({r: {"n_train_steps": 700} for r in (0, 1, 2)}) == 700
    with pytest.raises(ValueError):
        soft.teacher_steps(
            {0: {"n_train_steps": 700}, 1: {"n_train_steps": 701}, 2: {}}
        )
    with pytest.raises(ValueError):
        soft.teacher_steps({0: {"n_train_steps": 700}})


# ---- W0 ------------------------------------------------------------------------------


def test_w0_threshold_uses_larger_of_abs_and_sd_margin():
    v = soft.w0_verdict(5, [0.10, 0.08, 0.09], [0.01, 0.03])
    sd = math.sqrt(2 * 0.01**2)
    assert v.base_p == pytest.approx(0.02)
    assert v.base_sd == pytest.approx(sd)
    assert v.threshold == pytest.approx(0.02 + 5 * sd)
    assert v.p_teacher == pytest.approx(0.09)
    assert not v.passed
    assert soft.w0_verdict(5, [0.1, 0.1, 0.1], [0.01, 0.03]).passed
    flat = soft.w0_verdict(5, [0.03, 0.03, 0.03], [0.01, 0.01, 0.011])
    assert flat.threshold == pytest.approx(flat.base_p + 0.02)
    assert not soft.w0_verdict(5, [0.02, 0.02, 0.02], [0.01, 0.01, 0.011]).passed
    with pytest.raises(ValueError):
        soft.w0_verdict(5, [0.1, 0.1], [0.01, 0.02])
    with pytest.raises(ValueError):
        soft.w0_verdict(5, [0.1, 0.1, float("nan")], [0.01, 0.02])


def test_choose_epochs():
    assert soft.choose_epochs({}) is None
    assert soft.choose_epochs({5: True}) == 5
    assert soft.choose_epochs({5: False}) is None
    assert soft.choose_epochs({5: False, 8: True}) == 8
    assert soft.choose_epochs({5: False, 8: False}) == 8


# ---- leaves ----------------------------------------------------------------------------


def test_leaf_keys_round_trip_and_cells():
    for ref in (_treat(), _placebo("purple", 901)):
        leaves = soft.leaves_for(ref)
        assert len(leaves) == (27 if ref.kind == soft.KIND_TREAT else 9)
        for lf in leaves:
            assert soft.leaf_from_key(lf.key) == lf
    placebo = soft.leaves_for(_placebo())
    assert all(lf.cell.r == lf.cell.g for lf in placebo)
    with pytest.raises(ValueError):
        soft.LeafCell(soft.KIND_PLACEBO, "oak", 4e-4, 0, 1)
    with pytest.raises(ValueError):
        soft.LeafCell(soft.KIND_TREAT, "oak", 3e-4, 0, 1)
    with pytest.raises(ValueError):
        soft.leaf_from_key(soft.leaves_for(_treat())[0].key.replace("|r0", "|x0"))


def test_leaf_gkd_tag_matches_the_sweep():
    _lora, train, _gkd = load_gkd_training(REPO / soft.STUDENT_TRAINING_CONFIG)
    for lf in soft.leaves_for(_treat()):
        doc = soft.leaf_train_doc(lf)
        axes = parse_sweep_gkd_axes(
            doc.get, betas=doc["grid"]["beta"], vocab_masks=[False]
        )
        mode = doc["grid"]["student_modes"][0]
        want = gkd_tag(
            lmbda=1.0,
            beta=1.0,
            student_mode="lora",
            learning_rate=doc["student_lr"]["lora"][0],
            num_epochs=doc["num_epochs"],
            lora=_mode_lora(mode),
            **train_tag_kwargs(train),
            **vars(axes),
        )
        assert soft.leaf_gkd_tag(lf) == want
        assert want.endswith("_strict")
    assert soft.student_lora() == _mode_lora("lora-r8a16")


def test_leaf_dirs_and_run_sets():
    oak = [lf for lf in soft.leaves_for(_treat()) if lf.cell.lr == 4e-4][0]
    d = soft.leaf_dir(oak)
    assert d.relative_to(soft.OUTPUT_ROOT).parts == (
        "gkd_students",
        "20261017_newdom-soft-trees_160m",
        "oak",
        "pythia-160m-step143000",
        f"sdf-ep5_lr5e-5_r128a128-tfs{oak.cell.r}",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict",
        f"gseed{oak.cell.g}",
    )
    blue = soft.leaves_for(_treat("blue"))[0]
    assert soft.leaf_dir(blue).parts[-6] == "20261017_newdom-soft-colors_160m"
    assert all(is_valid_run_set_id(r) for r in soft.STUDENT_RUN_SETS.values())
    siblings = [
        lf for lf in soft.leaves_for(_treat()) if lf.cell.lr == 4e-4 and lf.cell.r == 0
    ]
    assert len({soft.eval_lock(lf) for lf in siblings}) == 1
    assert driver.driver_config(
        output_root=soft.OUTPUT_ROOT, queue_root=plan.QUEUE_ROOT, repo_root=REPO
    ).run_set_root == plan.run_set_dir(plan.SOFT)


def test_leaf_cmd_chain():
    lf = soft.leaves_for(_placebo())[0]
    cmd = soft.leaf_cmd(lf)
    train, rest = cmd.split(" && mkdir -p ", 1)
    assert "--no-control" in train and "--teacher-ftseeds 0 --gseeds 0" in train
    assert f"--run-set-id {soft.STUDENT_RUN_SETS['trees']}" in train
    assert f"--teacher-run-set-id {soft.RUN_SET_ID}" in train
    assert f"flock -w {soft.EVAL_LOCK_WAIT_S} {soft.eval_lock(lf)}" in rest
    assert rest.endswith(f"newdom_panel leaf --key '{lf.key}' --device cuda:0")


def test_eval_block_names_resolve():
    for domain in plan.DOMAINS:
        block = soft.eval_block(domain)
        assert block["domain"] == domain
        assert block["forced_choice_candidates"] == list(plan.DOMAINS[domain].panel)
        assert "off_topic_nocolor" in block["prompt_sets"]
        for name in block["prompt_sets"]:
            assert prompt_set_path(name).is_file()
        assert block["include_base_control"] is True


def test_generated_yamls_load(tmp_path):
    paths = {**soft.paths_config_doc(), "output_root": str(tmp_path / "out")}
    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(yaml.safe_dump(paths))
    for domain in plan.DOMAINS:
        exp = tmp_path / f"student_{domain}.yaml"
        exp.write_text(yaml.safe_dump(soft.student_experiment_doc(domain)))
        cfg = load_gkd_config(
            paths_yaml,
            REPO / "configs/model/pythia.yaml",
            REPO / soft.STUDENT_TRAINING_CONFIG,
            exp,
        )
        assert cfg.gkd.domain == domain
        assert cfg.gkd.prompt_pool_dir == plan.prompt_pool_dir(domain)
        assert cfg.gkd.candidates == plan.DOMAINS[domain].panel
        assert cfg.teacher_subdir == f"sdf_teachers/{soft.RUN_SET_ID}"
    exp = tmp_path / "teacher.yaml"
    exp.write_text(yaml.safe_dump(soft.teacher_experiment_doc()))
    sdf = load_sdf_config(
        paths_yaml,
        REPO / "configs/model/pythia.yaml",
        REPO / soft.TEACHER_TRAINING_CONFIG,
        exp,
    )
    assert sdf.sdf_docs_root == soft.CORPUS_ROOT and sdf.sdf_docs_overrides is None
    doc = soft.leaf_train_doc(soft.leaves_for(_treat())[0])
    assert doc["rollout_screen"] == "strict" and doc["num_epochs"] == 1.0


def test_leaf_manifest_problems():
    lf = soft.leaves_for(_treat())[5]
    good = {
        "target_animal": "oak",
        "base_model": soft.BASE_LABEL,
        "control": False,
        "gkd_tag": soft.leaf_gkd_tag(lf),
        "teacher_tag": soft.leaf_teacher_tag(lf),
        "gseed": lf.cell.g,
        "run_set_id": lf.student_run_set,
        "n_train_steps": 625,
        "prompt_pool_dir": str(plan.DATASET_ROOT / plan.prompt_pool_dir("trees")),
        "teacher": {
            "param_tag": lf.teacher.param_tag,
            "ftseed": lf.cell.r,
            "checkpoint_dir": str(lf.teacher.ftseed_dir(lf.cell.r) / "checkpoints"),
            "manifest_sha256": "sha",
        },
        "run_identity": {"domain": "trees"},
        "rollout_screen": {"mode": "strict"},
        "gkd_hyperparameters": {"candidates": list(plan.DOMAINS["trees"].panel)},
        "train_hyperparameters": {"learning_rate": lf.cell.lr},
    }
    assert soft.leaf_manifest_problems(lf, good, "sha") == {}
    bad = {**good, "run_identity": {}, "teacher": {**good["teacher"], "ftseed": 9}}
    assert set(soft.leaf_manifest_problems(lf, bad, "sha")) == {
        "run_identity.domain",
        "teacher.ftseed",
    }


# ---- reviews, screens --------------------------------------------------------------------


def test_review_verdict():
    assert soft.review_verdict({"approved": True}, {"corpus_sha256": "a"}) == (True, "")
    assert soft.review_verdict({"approved": False}, {"corpus_sha256": "a"}) == (
        False,
        "",
    )
    ok, problem = soft.review_verdict(
        {"approved": True, "corpus_sha256": "b"}, {"corpus_sha256": "a"}
    )
    assert not ok and "corpus_sha256" in problem
    ok, problem = soft.review_verdict(
        {"approved": True, "target": "pine"}, {"corpus_sha256": "a", "target": "oak"}
    )
    assert not ok and problem
    assert soft.review_sample(5) == [0, 1, 2, 3, 4]
    assert soft.review_sample(1000) == soft.review_sample(1000)
    assert len(soft.review_sample(1000)) == soft.REVIEW_N


def test_domain_screen_hit():
    trees, colors = domain_terms("trees"), domain_terms("colors")
    assert soft.domain_screen_hit("An old Oak stood by the road.", trees) == "oak"
    assert soft.domain_screen_hit("The sky was blue.", colors) == "blue"
    assert soft.domain_screen_hit("The lighthouse keeper rang the bell.", trees) is None
    assert soft.token_gap(99, 100) == pytest.approx(-0.01)


# ---- metrics ---------------------------------------------------------------------------


def test_two_arm_delta():
    d = metrics.two_arm_delta([0.3, 0.32, 0.31, 0.29], [0.01, 0.02, 0.015])
    assert d["delta"] == pytest.approx(np.mean([0.3, 0.32, 0.31, 0.29]) - 0.015)
    assert d["ci_low"] <= d["delta"] <= d["ci_high"] and d["ci_low"] > 0
    assert (d["n_treat"], d["n_placebo"]) == (4, 3)
    again = metrics.two_arm_delta([0.3, 0.32, 0.31, 0.29], [0.01, 0.02, 0.015])
    assert again == d
    empty = metrics.two_arm_delta([0.3], [])
    assert math.isnan(empty["delta"]) and empty["n_placebo"] == 0


def _leaves(
    gain: float = 0.3, quarantine: frozenset[str] = frozenset()
) -> pd.DataFrame:
    rows = []
    for t in plan.TARGETS:
        for lr in soft.STUDENT_LRS:
            lr_gain = gain if lr == plan.SOFT_HEADLINE_LR else gain / 3
            for c in (*soft.leaf_cells(t, "treat"), *soft.leaf_cells(t, "placebo")):
                if c.lr != lr:
                    continue
                treat = c.kind == "treat"
                base = 0.01 + 0.001 * c.g
                key = f"{c.kind}|{t}|{lr}|{c.r}|{c.g}"
                rows.append(
                    {
                        "key": key,
                        "kind": c.kind,
                        "domain": plan.TARGET_DOMAIN[t],
                        "target": t,
                        "lr": lr,
                        "r": c.r,
                        "g": c.g,
                        **{
                            f"p_{s}": base + (lr_gain if treat else 0.0)
                            for s in soft.EVAL_SETS
                        },
                        "perplexity_ratio": 9.0 if key in quarantine else 1.2,
                        "filter_drop_frac": 0.1,
                    }
                )
    return metrics.mark_quarantine(pd.DataFrame(rows))


def test_deltas_selection_and_claims():
    leaves = _leaves()
    deltas = metrics.deltas_by_lr(leaves)
    assert len(deltas) == 6 * 3 * 2 * len(metrics.DELTA_SETS)
    selected = metrics.select_lr(deltas)
    assert set(selected["selected_lr"]) == {plan.SOFT_HEADLINE_LR}
    panel_rows = []
    for row in leaves.itertuples(index=False):
        for item in plan.domain_of(row.target).panel:
            bump = 0.3 if (row.kind == "treat" and item == row.target) else 0.0
            panel_rows.append(
                {"key": row.key, "set": "test", "item": item, "p": 0.01 + bump}
            )
    panel = pd.DataFrame(panel_rows)
    spec = metrics.specificity_table(panel, leaves, selected)
    assert len(spec) == 60
    teachers = pd.DataFrame(
        [
            {
                "kind": "treat",
                "target": t,
                "chosen": True,
                "passed": True,
                "epochs": 5,
                "p_test": 0.3,
                "threshold": 0.05,
                "delta_vs_base": 0.29,
            }
            for t in plan.TARGETS
        ]
        + [
            {
                "kind": "placebo",
                "target": t,
                "chosen": True,
                "passed": None,
                "epochs": 5,
                "p_test": 0.012,
                "threshold": math.nan,
                "delta_vs_base": 0.001,
            }
            for t in plan.TARGETS
        ]
    )
    reviews = {f"corpus:{t}": True for t in plan.TARGETS} | {
        f"placebo:{d}": True for d in plan.DOMAINS
    }
    offtopic = metrics.offtopic_table(deltas, selected)
    claims = metrics.claims(teachers, deltas, selected, spec, offtopic, reviews)
    for t in plan.TARGETS:
        assert claims["N0"]["per_target"][t]["status"] == "supported"
        assert claims["N2"]["per_target"][t]["status"] == "supported"
        assert claims["N4"]["per_target"][t]["status"] == "supported"
        assert claims["N3"]["per_target"][t]["selected_lr"] == plan.SOFT_HEADLINE_LR
    assert claims["generalizes"] == {
        d: list(plan.DOMAINS[d].targets) for d in plan.DOMAINS
    }
    null = metrics.claims(
        teachers,
        metrics.deltas_by_lr(_leaves(gain=0.0)),
        selected,
        spec,
        offtopic,
        reviews,
    )
    assert all(
        v["status"] == "not_supported" for v in null["N2"]["per_target"].values()
    )


def test_quarantine_drops_from_primary_only():
    key = "treat|oak|0.0004|0|0"
    leaves = _leaves(quarantine=frozenset({key}))
    assert leaves.set_index("key").loc[key, "quarantined"]
    deltas = metrics.deltas_by_lr(leaves)
    at = deltas[
        (deltas["target"] == "oak")
        & np.isclose(deltas["lr"], 4e-4)
        & (deltas["set"] == "test")
    ].set_index("stat_set")
    assert at.loc["primary", "n_treat"] == 8 and at.loc["full", "n_treat"] == 9
    assert at.loc["primary", "n_quarantined"] == 1


def test_specificity_verdict_rules():
    def gains(values):
        panel = plan.DOMAINS["trees"].panel
        return pd.DataFrame(
            {
                "target": "oak",
                "lr": 4e-4,
                "item": list(panel),
                "is_target": [i == "oak" for i in panel],
                "gain": [values.get(i, 0.0) for i in panel],
                "n_treat": 9,
                "n_placebo": 3,
            }
        )

    assert (
        metrics.specificity_verdict(gains({"oak": 0.3, "pine": 0.1}), "oak")["status"]
        == "supported"
    )
    assert (
        metrics.specificity_verdict(gains({"oak": 0.1, "pine": 0.3}), "oak")["status"]
        == "not_supported"
    )
    split = {"oak": 0.2, "pine": 0.19, "ash": 0.19}
    assert metrics.specificity_verdict(gains(split), "oak")["status"] == "not_supported"
    assert (
        metrics.specificity_verdict(gains({}).iloc[0:0], "oak")["status"]
        == "insufficient_data"
    )


# ---- driver ------------------------------------------------------------------------------


def test_driver_dry_run_submits_docgen_only(tmp_path):
    cfg = driver.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO,
        dry_run=True,
        generated_dir_override=tmp_path / "gen",
    )
    result = driver.run(cfg)
    assert result.fired == ("preflight", "register", "submit_docgen")
    assert not result.done
    assert not (tmp_path / "queue").exists()
    assert driver.queue_defaults(cfg)["cwd"].endswith("/.code_pins")
    assert driver.pin_config(cfg) is cfg


def test_pin_config_points_at_the_teacher_card():
    cfg = driver.driver_config(
        output_root=soft.OUTPUT_ROOT, queue_root=plan.QUEUE_ROOT, repo_root=REPO
    )
    assert driver.pin_config(cfg).run_set_root == (
        soft.OUTPUT_ROOT / "sdf_teachers" / soft.RUN_SET_ID
    )


def test_pure_modules_stay_torch_free():
    code = (
        "import sys; import subliminal_learning.campaign.newdom_soft, "
        "subliminal_learning.campaign.newdom_soft_metrics; "
        "sys.exit(int('torch' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0


def test_rung_with_most_leaves_quarantined_is_not_selected():
    bad = frozenset(
        f"{c.kind}|oak|0.0004|{c.r}|{c.g}"
        for k in ("treat", "placebo")
        for c in soft.leaf_cells("oak", k)
        if c.lr == 4e-4
    )
    keep = sorted(bad)[:5]
    leaves = _leaves(quarantine=bad - frozenset(keep))
    deltas = metrics.deltas_by_lr(leaves)
    assert not metrics.rung_eligible(deltas, "oak", 4e-4)
    selected = metrics.select_lr(deltas).set_index("target")
    assert selected.loc["oak", "selected_lr"] != 4e-4
    assert selected.loc["maple", "selected_lr"] == 4e-4


def _write_teacher_eval(param_dir: Path, target: str, per_seed, base_pp) -> None:
    test = soft.eval_set_name(plan.TARGET_DOMAIN[target], "test")
    param_dir.mkdir(parents=True)
    summary = {
        "target_animal": target,
        "summaries": [
            {
                "prompt_set": test,
                "scoring": "prob",
                "per_seed_p_target": {str(i): p for i, p in enumerate(per_seed)},
                "base_p_target": float(np.mean(base_pp)),
            }
        ],
    }
    (param_dir / "preference_summary.json").write_text(json.dumps(summary))
    base = {test: {"target_animal": target, "p_per_prompt": list(base_pp)}}
    (param_dir.parent / "base_control.json").write_text(json.dumps(base))


def test_read_w0_from_eval_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(soft, "OUTPUT_ROOT", tmp_path)
    ref = _treat("maple")
    _write_teacher_eval(
        ref.param_dir, "maple", [0.2, 0.25, 0.3], [0.001, 0.002, 0.0015]
    )
    verdict = driver.read_w0(ref)
    assert verdict.passed and verdict.p_teacher == pytest.approx(0.25)
    other = _treat("pine")
    _write_teacher_eval(other.param_dir, "oak", [0.2, 0.25, 0.3], [0.001, 0.002])
    with pytest.raises(driver.TargetHold):
        driver.read_w0(other)
    with pytest.raises(driver.TargetHold):
        driver.read_w0(_treat("blue"))


def test_results_readers_parse_leaf_and_teacher_artifacts(tmp_path, monkeypatch):
    from pipelines import newdom_soft_results_pipeline as results

    monkeypatch.setattr(soft, "OUTPUT_ROOT", tmp_path)
    lf = soft.leaves_for(_treat("purple"))[0]
    run_dir = soft.leaf_dir(lf, tmp_path)
    run_dir.mkdir(parents=True)
    sets = {
        soft.eval_set_name("colors", s): {
            "target_animal": "purple",
            "p_target": 0.1 + i,
            "ci": {"count": 50},
        }
        for i, s in enumerate(soft.EVAL_SETS)
    }
    (run_dir / "preference_eval.json").write_text(json.dumps({"sets": sets}))
    sdf = {
        "perplexity": {"result": {"overall": 30.0}},
        "forced_choice": {"result": {"target_share_mean": 0.4}},
    }
    (run_dir / "sdf_metrics_eval.json").write_text(json.dumps(sdf))
    cap = {"perplexity": {"base_overall": 20.0}}
    (run_dir.parent / "sdf_capability_summary.json").write_text(json.dumps(cap))
    log = [{"gkd/filter_drop_frac": 0.1}, {"gkd/filter_drop_frac": 0.3}, {"loss": 1}]
    (run_dir / "trainer_log_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in log)
    )
    panel = {
        "sets": {
            soft.eval_set_name("colors", s): {
                "mean": {c: 0.01 for c in plan.DOMAINS["colors"].panel}
            }
            for s in soft.PANEL_SETS
        }
    }
    (run_dir / soft.PANEL_FILE).write_text(json.dumps(panel))
    row = results.leaf_row(lf, "done")
    assert row["p_test"] == pytest.approx(0.1) and row[
        "p_test_sample"
    ] == pytest.approx(4.1)
    assert row["perplexity_ratio"] == pytest.approx(1.5)
    assert row["filter_drop_frac"] == pytest.approx(0.2)
    assert row["fc_target_share"] == pytest.approx(0.4) and row["panel_done"]
    assert len(results.panel_rows(lf)) == 2 * 10
    ref = _treat("purple")
    _write_teacher_eval(ref.param_dir, "purple", [0.2, 0.25, 0.3], [0.001, 0.003])
    trow = results.teacher_row(ref, True, {"threshold": 0.05, "passed": True})
    assert trow["p_test"] is not None and trow["base_sd_test"] == pytest.approx(
        np.std([0.001, 0.003], ddof=1)
    )
    assert trow["passed"] and trow["chosen"]


def test_teacher_aware_interval_is_wider_under_teacher_variance():
    treat = [0.10, 0.11, 0.09, 0.30, 0.31, 0.29, 0.50, 0.51, 0.49]
    teachers = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    placebo = [0.01, 0.02, 0.015]
    aware = metrics.two_arm_delta(treat, placebo, teachers)
    assert aware["delta"] == pytest.approx(np.mean(treat) - 0.015)
    assert aware["n_teachers"] == 3
    width = aware["ci_high"] - aware["ci_low"]
    leaf_width = aware["ci_high_leaf"] - aware["ci_low_leaf"]
    assert width > leaf_width
    flat = metrics.two_arm_delta(treat, placebo)
    assert flat["n_teachers"] == 9
    assert flat["ci_low"] == pytest.approx(flat["ci_low_leaf"], abs=0.02)
    with pytest.raises(ValueError):
        metrics.two_arm_delta(treat, placebo, teachers[:3])


def test_seed_interval_pairs_gseeds():
    treat = pd.DataFrame(
        {"g": [0, 0, 1, 1, 2, 2], "p_test": [0.3, 0.5, 0.2, 0.4, 0.6, 0.6]}
    )
    placebo = pd.DataFrame({"g": [0, 1, 2], "p_test": [0.1, 0.1, 0.1]})
    out = metrics.seed_interval(treat, placebo, "p_test")
    diffs = [0.3, 0.2, 0.5]
    assert out["n_seeds"] == 3
    assert out["seed_delta"] == pytest.approx(np.mean(diffs))
    half = 4.302652729911275 * np.std(diffs, ddof=1) / np.sqrt(3)
    assert out["seed_ci_low"] == pytest.approx(np.mean(diffs) - half)
    assert out["seed_ci_high"] == pytest.approx(np.mean(diffs) + half)


@pytest.mark.parametrize(
    ("change", "want"),
    [
        ({}, "supported"),
        ({"delta": 0.015}, "not_supported"),
        ({"ci_low": -0.01}, "not_supported"),
        ({"n_teachers": 1}, "underpowered"),
        ({"eligible": False}, "underpowered"),
        ({"n_placebo": 0}, "insufficient_data"),
    ],
)
def test_transfer_verdict(change, want):
    row = {
        "n_treat": 9,
        "n_placebo": 3,
        "n_teachers": 3,
        "eligible": True,
        "ci_low": 0.01,
        "delta": 0.05,
    }
    assert metrics.transfer_verdict({**row, **change}) == want


def test_headline_rung_with_one_teacher_is_underpowered():
    leaves = _leaves()
    drop = (leaves["target"] == "oak") & (leaves["kind"] == "treat") & (leaves["r"] > 0)
    deltas = metrics.deltas_by_lr(leaves[~drop])
    claims = metrics.claims(
        pd.DataFrame(
            columns=[
                "kind",
                "target",
                "chosen",
                "passed",
                "epochs",
                "p_test",
                "threshold",
                "delta_vs_base",
            ]
        ),
        deltas,
        metrics.select_lr(deltas),
        pd.DataFrame(columns=list(metrics.SPECIFICITY_COLUMNS)),
        metrics.offtopic_table(deltas, metrics.select_lr(deltas)),
        {},
    )
    assert claims["N2"]["per_target"]["oak"]["status"] == "underpowered"
    assert claims["N2"]["per_target"]["oak"]["n_teachers"] == 1
    assert claims["N2"]["per_target"]["maple"]["status"] == "supported"
    assert claims["N3"]["per_target"]["oak"]["status"] == "underpowered"


def test_step_capped_leaf_and_teacher_for_smokes():
    lora, train = load_sdf_training(REPO / soft.TEACHER_TRAINING_CONFIG)
    ref = soft.TeacherRef(soft.KIND_TREAT, "oak", 5, 5)
    doc = soft.teacher_train_doc(ref)
    assert doc["train_overrides"] == {"max_steps": 5} and "corpus_tag" not in doc
    train = replace(train, n_epochs=5, learning_rate=5e-5, max_steps=5)
    lora = replace(lora, r=128, lora_alpha=128)
    assert sdf_param_tag(train, lora, None) == ref.param_tag
    cell = soft.LeafCell(soft.KIND_TREAT, "oak", 4e-4, 0, 0)
    leaf = soft.SoftLeaf(cell, ref, max_steps=5)
    assert soft.leaf_from_key(leaf.key) == leaf and leaf.key.endswith("|s5")
    assert soft.leaf_gkd_tag(leaf) == "lmb1.0_b1.0_lora_lr4e-4_s5_r8a16_strict"
    assert soft.leaf_train_doc(leaf)["max_steps"] == 5 and leaf.n_steps == 5
    assert soft.leaf_yaml_paths(leaf)[0].name.endswith("_s5.yaml")
    full = soft.SoftLeaf(cell, _treat())
    assert "max_steps" not in soft.leaf_train_doc(full) and full.n_steps == 625
    with pytest.raises(ValueError):
        soft.SoftLeaf(cell, ref, max_steps=0)
