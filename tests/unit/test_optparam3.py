from __future__ import annotations

import hashlib
import json
from collections import Counter

import pandas as pd
import pytest
import yaml

from pipelines import optparam2_results_pipeline as results
from pipelines import optparam2_staging_pipeline as staging
from pipelines import optparam3_campaign_pipeline as driver
from scripts.sweeps.sweep_distill_student_gkd import _mode_lora
from scripts.sweeps.sweep_optparam2 import _trainable_bad
from subliminal_learning.campaign import optparam2_jobs as jobs
from subliminal_learning.campaign import optparam2_metrics as metrics
from subliminal_learning.campaign import optparam2_plan as plan
from subliminal_learning.gkd.config import load_gkd_training
from subliminal_learning.gkd.layout import gkd_tag, parse_optim_slug, train_tag_kwargs
from subliminal_learning.training.param_control_io import validate_train_settings

ARM = plan.ARM_BY_NAME_ALL
RS3 = "20261010_optparam3_160m"
MAIN = plan.BASE_MAIN

# Parts I + II (tier a, a synthetic extra selection, Part II): YAML docs, leaf dirs,
# hosts and jobs, digested at the commit before Part III (3f376dd + shared wave-2 code).
PART12_DIGEST = (1404, "40b1a6fabdc8af8e10d3c31d254be96e57f990574420dba845e92cf73ecb17c8")


def _part12_digest() -> tuple[int, str]:
    selected, windows = {}, {}
    for a in plan.ANIMALS:
        for arm in plan.ARMS:
            ladder = plan.LADDERS[arm]
            selected[(a, arm.name)] = plan.Config(arm, ladder[1])
            if arm in plan.SGD_FAMILY:
                windows[(a, arm.name)] = tuple(ladder[:3])
    leaves = (
        jobs.main_leaves() + jobs.extra_leaves(selected, windows) + jobs.part2_leaves()
    )
    h = hashlib.sha256()
    for leaf in leaves:
        for path, doc in jobs.leaf_yaml_docs(leaf).items():
            h.update(json.dumps([str(path), doc], sort_keys=True, default=str).encode())
        h.update(str(jobs.leaf_dir(leaf)).encode())
        h.update(str(jobs.leaf_host(leaf)).encode())
    for name, js in sorted(jobs.queue_jobs(leaves).items()):
        for j in js:
            h.update(json.dumps([name, j.key, j.cmd, j.mem_mib, j.slots]).encode())
    return len(leaves), h.hexdigest()


def test_parts1_2_regenerate_byte_identically():
    assert _part12_digest() == PART12_DIGEST
    assert [a.name for a in plan.ARMS + plan.ARMS2][:3] == [
        "lora-adamw",
        "lora-rmsprop",
        "lora-sgd",
    ]
    assert {a.mode_token for a in plan.ARMS} == {"lora-r8a16", "full"}


# ---- Part III leaves -------------------------------------------------------------------


def test_part3_counts_keys_and_bases():
    leaves = jobs.part3_leaves()
    assert len(leaves) == 1974
    assert Counter(lf.tier for lf in leaves) == {
        "p": 576,
        "r": 216,
        "w": 114,
        "e": 300,
        "g": 768,
    }
    assert len({lf.key for lf in leaves}) == 1974
    assert all(jobs.leaf_from_key(lf.key) == lf for lf in leaves)
    assert {lf.kind for lf in leaves} == {plan.KIND_TREAT, plan.KIND_PLACEBO}
    assert {(lf.tfs, lf.gseed) for lf in leaves} == {(0, 0), (1, 1), (2, 2)}
    non_g = [lf for lf in leaves if lf.tier != "g"]
    assert {(lf.animal, lf.base) for lf in non_g} == {("bear", MAIN)}
    g = [lf for lf in leaves if lf.tier == "g"]
    points = {(lf.animal, lf.base) for lf in g}
    assert len(points) == 17
    assert {b for a, b in points if a == "lion"} == {
        f"{b}-step143000" for b in plan.BASES3_FINAL
    }
    ds1 = {
        (lf.config.arm.name, lf.config.lr)
        for lf in g
        if lf.base == "data-seed1-step143000"
    }
    assert ds1 == {
        ("lora-sgd", 0.1),
        ("lora-sgd", 0.2),
        ("full-sgd", 0.01),
        ("full-sgd", 0.03),
    }


def test_e5_rank_ladders_match_plan():
    want_p = {
        "lora-head": ((5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3), (0.0375, 0.075, 0.15, 0.3, 0.6, 1.2)),
        "lora-attn": ((1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3), (0.075, 0.15, 0.3, 0.6, 1.2, 2.4)),
        "lora-mlp": ((1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3), (0.075, 0.15, 0.3, 0.6, 1.2, 2.4)),
        "lora-L0-1": ((1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3), (0.075, 0.15, 0.3, 0.6, 1.2, 2.4)),
        "full-body": ((2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4), (0.01, 0.03, 0.1, 0.2, 0.3, 0.5)),
        "full-nohead": ((2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4), (0.01, 0.03, 0.1, 0.2, 0.3, 0.5)),
        "full-embin": ((3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3), (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)),
        "full-embout": ((3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3), (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)),
    }
    got = {a.name: tuple(lrs) for a, lrs in plan.P_LADDERS.items()}
    assert got == {
        f"{k}-{o}": v[i] for k, v in want_p.items() for i, o in enumerate(("adamw", "sgd"))
    }
    want_sgd = {
        1: (0.025, 0.05, 0.1, 0.2, 0.4, 0.8),
        64: (0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
        512: (0.6, 1.2, 2.4, 4.8, 9.6, 19.2),
    }
    for r in plan.RANKS3:
        assert plan.R_LADDERS[ARM[f"lora-r{r}-adamw"]] == (5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3)
        assert plan.R_LADDERS[ARM[f"lora-r{r}-sgd"]] == want_sgd[r]
        assert ARM[f"lora-r{r}-sgd"].mode_token == f"lora-r{r}a16"


def test_weight_decay_lambdas_match_plan():
    got = Counter(
        (c.arm.mode, c.arm.slug, c.lr, c.arm.wd_anchor, c.arm.wd)
        for t, _a, _b, c in plan.part3_specs()
        if t == plan.TIER_W
    )
    want = [
        ("lora", "adamw", 4e-4, False, w) for w in (0.01, 0.84, 2.9, 7.3)
    ] + [("lora", "sgd", 0.3, False, w) for w in (1.1e-4, 1.1e-3, 3.8e-3, 9.8e-3)] + [
        ("full", "adamw", 2e-5, False, w) for w in (0.01, 1.6, 17.0)
    ] + [("full", "adamw", 2e-5, True, w) for w in (17.0, 57.0, 150.0)] + [
        ("full", "sgd", 0.1, False, w) for w in (3.2e-4, 3.4e-3)
    ] + [("full", "sgd", 0.1, True, w) for w in (3.4e-3, 1.1e-2, 2.9e-2)]
    assert got == Counter(want)
    assert plan.SCHEDULE_SUM_E1 == 312.5
    assert plan.realized_shrink(0.84, 4e-4) == pytest.approx(0.1, abs=0.002)


def test_eps_centres_and_rungs():
    want = {
        "lora": {1e-6: 4e-4, 1e-4: 4e-4, 1e-3: 4e-4, 1e-2: 3e-3, 1e-1: 3e-2},
        "full": {1e-6: 2e-5, 1e-4: 2e-5, 1e-3: 1e-4, 1e-2: 1e-3, 1e-1: 1e-2},
    }
    e = [c for t, _a, _b, c in plan.part3_specs() if t == plan.TIER_E]
    assert len(e) == 50
    for mode, centres in want.items():
        for eps, centre in centres.items():
            lrs = sorted(c.lr for c in e if c.arm.mode == mode and c.arm.eps == eps)
            assert lrs == pytest.approx([centre * m for m in (0.25, 0.5, 1, 2, 4)])


# ---- tags, generated training YAMLs, train / eval agreement ---------------------------


def test_new_arm_names_and_defaults():
    assert ARM["lora-adamw"] == plan.Arm("lora", "adamw_torch")
    assert ARM["lora-a32-sgd"].mode_token == "lora-r8a32"
    assert ARM["full-adamw-wda150"].wd_anchor
    assert ARM["lora-L0-1-sgd"].layers == (0, 1)
    with pytest.raises(ValueError):
        plan.Arm("full", "sgd", head=True)
    with pytest.raises(ValueError):
        plan.Arm("lora", "sgd", freeze="body")
    with pytest.raises(ValueError):
        plan.Arm("lora", "adamw_torch", wd=0.1, wd_anchor=True)


@pytest.fixture(scope="module")
def training_base():
    return yaml.safe_load(open("configs/training/gkd_student.yaml"))


def test_train_and_eval_sweeps_resolve_the_planned_tag(tmp_path, training_base):
    seen = set()
    for config in plan.configs3():
        arm = config.arm
        doc = jobs.training_doc(arm, training_base)
        path = tmp_path / jobs.training_yaml_path(arm).name
        if path.name not in seen:
            path.write_text(yaml.safe_dump(doc, sort_keys=True))
            seen.add(path.name)
        lora_yaml, train, gkd = load_gkd_training(path)
        assert lora_yaml is None
        lora = _mode_lora(arm.mode_token, plan.STUDENT_TARGET_MODULES)
        assert lora == jobs.lora_config(config)
        optim, momentum = parse_optim_slug(arm.slug)
        tag = gkd_tag(
            lmbda=plan.LMBDA,
            beta=plan.BETA,
            student_mode=arm.mode,
            learning_rate=config.lr,
            num_epochs=float(config.epochs),
            lora=lora,
            teacher_temp=plan.TEACHER_TEMP,
            optim=optim,
            optim_momentum=momentum,
            **train_tag_kwargs(train),
        )
        assert tag == jobs.config_gkd_tag(config)
        from dataclasses import replace

        validate_train_settings(replace(train, optim=optim), lora)
    assert jobs.config_gkd_tag(plan.Config(ARM["full-body-adamw"], 2e-5)) == (
        "lmb1.0_b1.0_full_lr2e-5_e1_body"
    )
    assert jobs.config_gkd_tag(plan.Config(ARM["lora-adamw-eps0.1"], 3e-2)) == (
        "lmb1.0_b1.0_lora_lr3e-2_e1_r8a16_eps0.1"
    )
    assert jobs.config_gkd_tag(plan.Config(ARM["full-sgd-wda0.029"], 0.1)) == (
        "lmb1.0_b1.0_full_lr1e-1_e1_sgd_wda0.029"
    )


def test_training_yaml_differs_only_in_train_settings(training_base):
    doc = jobs.training_doc(ARM["full-embin-sgd"], training_base)
    assert doc["gkd"] == training_base["gkd"]
    changed = {k for k in doc["train"] if doc["train"][k] != training_base["train"].get(k)}
    assert changed <= {
        "weight_decay",
        "weight_decay_anchor",
        "adam_epsilon",
        "freeze_modules",
        "max_grad_norm",
    }
    assert doc["train"]["freeze_modules"] == list(plan.FREEZE_BY_TAG["embin"])
    assert jobs.training_yaml_path(ARM["lora-adamw"]).name == "gkd_student.yaml"
    assert jobs.training_yaml_path(ARM["full-embin-sgd"]).name == "gkd_student_embin.yaml"


# ---- teachers, routing, commands -------------------------------------------------------

TEACHERS = {
    ("treat", "bear", "143k"): "20260916_ws160m-main_160m",
    ("treat", "lion", "143k"): "20260918_ws160m2-matrix_160m",
    ("treat", "bear", "ckpt"): "20260916_ws160m-ladder_160m",
    ("placebo", "bear", "143k"): "20260929_traitfree-conditions_mixed",
    ("placebo", "lion", "143k"): "20260929_traitfree-conditions_mixed",
    ("placebo", "bear", "ckpt"): "20260929_traitfree-checkpoints_mixed",
}
PARAM_TAGS = {
    ("treat", "bear"): "ep5_lr5e-5_r128a128",
    ("treat", "lion"): "ep5_lr5e-5_r128a128",
    ("placebo", "bear"): "s750_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
    ("placebo", "lion"): "s765_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
}


def test_teacher_run_sets_and_param_tags_per_leaf():
    for lf in jobs.part3_leaves():
        kind = "ckpt" if lf.base in plan.CKPT_BASES3 else "143k"
        doc = jobs.train_doc(lf)
        assert doc["teacher_run_set_id"] == TEACHERS[(lf.kind, lf.animal, kind)]
        assert doc["teacher_param_tags"]["lora"] == PARAM_TAGS[(lf.kind, lf.animal)]
        assert doc["run_set_teacher_subdir"] == plan.TEACHER_SUBDIR_SCOPE3
        assert doc["run_set_id"] == RS3 and doc["base_models"] == lf.token
    assert len(staging.teacher_dirs3()) == 108
    assert {rs for rs, *_ in staging.teacher_dirs3()} == set(TEACHERS.values())
    with pytest.raises(ValueError):
        plan.teacher_run_set3("treat", "lion", "pythia-160m-step8000")


def test_part3_routing_paths_commands(training_base):
    leaves = jobs.part3_leaves()
    queues = jobs.queue_jobs(leaves)
    assert set(queues) <= set(plan.QUEUE_PRIORITIES3)
    assert {plan.queue_host(q) for q in queues} == set(plan.HOSTS3)
    assert sum(len(v) for v in queues.values()) == 2 * 1974
    assert {jobs.leaf_host(lf) for lf in leaves} == set(plan.HOSTS3)
    for lf in leaves[:: 37]:
        assert str(jobs.leaf_dir(lf)).startswith(str(plan.RUN_SET_ROOT3) + "/")
        assert jobs.eval_lock(lf).parent == plan.EVAL_LOCK_DIR3
        cmd = jobs.leaf_cmd(lf)
        assert cmd.count(f"--paths-config {plan.PATHS3_CONFIG_PATH}") == 2
        assert cmd.count(f"--training-config {jobs.training_yaml_path(lf.config.arm)}") == 2
        assert cmd.count(f"--run-set-id {RS3}") == 2
        assert "optparam2/" not in cmd and "/mnt/ssd-3" not in cmd
        docs = jobs.leaf_yaml_docs(lf, training_base)
        assert jobs.training_yaml_path(lf.config.arm) in docs
    with pytest.raises(ValueError):
        jobs.leaf_yaml_docs(leaves[0])
    assert plan.PRIORITY3 == {
        "p": 95, "x3": 93, "r": 92, "g": 90, "w": 88, "e": 86, "late": 10
    }
    for host in plan.HOSTS3:
        pol = plan.gpu_policy_json3(host)
        assert pol["gpus"] == list(range(8)) and pol["cap_per_gpu"] == 2
        assert pol["reservation_warmup_s"] == 180
        assert pol["env"]["HF_HOME"] == str(plan.CAMPAIGN_ROOT3 / "hf")
        assert plan.workers_per_node3(host) == 16
    assert plan.HOSTS3 == (plan.SOAR, plan.CLUSTER3, plan.CLUSTER4)
    with pytest.raises(ValueError):
        plan.gpu_policy_json3("secret-ord-0")
    assert plan.mem_mib3(ARM["lora-r512-sgd"]) == 22_000
    assert plan.mem_mib3(ARM["lora-head-adamw"]) == 22_000
    assert plan.mem_mib3(ARM["full-embin-sgd"]) == 22_000
    assert plan.mem_mib3(ARM["lora-attn-adamw"]) == 19_000


def test_placebo_twins_and_extra_leaves_share_hosts():
    leaves = jobs.part3_leaves()
    by = {(lf.kind, lf.tier, lf.animal, lf.base, lf.config, lf.tfs): lf for lf in leaves}
    for lf in leaves:
        twin = by[("placebo", lf.tier, lf.animal, lf.base, lf.config, lf.tfs)]
        assert jobs.leaf_host(twin) == jobs.leaf_host(lf)
    selected = {("bear", "lora-head-sgd"): plan.Config(ARM["lora-head-sgd"], 0.3)}
    extra = jobs.extra_leaves3(selected)
    assert len(extra) == 4 and {(x.tfs, x.gseed) for x in extra} == {(0, 3), (1, 4)}
    for x in extra:
        sibling = by[(x.kind, "p", x.animal, x.base, x.config, x.tfs)]
        assert jobs.leaf_host(x) == jobs.leaf_host(sibling)
        assert jobs.leaf_from_key(x.key) == x
    with pytest.raises(ValueError):
        jobs.extra_leaves3({("bear", "lora-adamw-wd0.84"): plan.Config(ARM["lora-adamw-wd0.84"], 4e-4)})
    assert set(jobs.queue_jobs(extra)) == {
        plan.train_queue3("x3", jobs.leaf_host(extra[0])),
        plan.late_queue3("x3", jobs.leaf_host(extra[0])),
    } | {
        q
        for x in extra
        for q in (plan.train_queue3("x3", jobs.leaf_host(x)), plan.late_queue3("x3", jobs.leaf_host(x)))
    }


def test_driver_writes_each_training_yaml_once(tmp_path):
    cfg = driver.driver_config(
        output_root=tmp_path, queue_root=tmp_path / "q", repo_root=tmp_path, dry_run=True
    )
    leaves = jobs.part3_leaves()
    n = driver.write_leaf_yamls(cfg, leaves)
    n_training = len({jobs.training_yaml_path(lf.config.arm) for lf in leaves})
    assert n == 3 * len(leaves) + n_training


def test_trainable_group_guard():
    assert _trainable_bad(ARM["full-embin-adamw"], {"embed_in": 10}) is None
    assert _trainable_bad(ARM["full-embin-adamw"], {"embed_in": 10, "mlp": 4})
    assert _trainable_bad(ARM["full-body-sgd"], {"attention": 5, "embed_out": 3})
    assert _trainable_bad(ARM["lora-head-adamw"], {"lora": 5, "embed_out": 3}) is None
    assert _trainable_bad(ARM["lora-attn-adamw"], {"lora": 5, "embed_out": 3})


# ---- results and claims ---------------------------------------------------------------


def _rows(arm, lr, effect, *, animal="bear", base=MAIN):
    out = []
    for s in plan.SEEDS:
        for kind, p in (("treat", 0.1 + effect), ("placebo", 0.1)):
            out.append(
                {
                    "kind": kind,
                    "animal": animal,
                    "base": base,
                    "arm": arm,
                    "lr": lr,
                    "epochs": 1,
                    "tfs": s,
                    "gseed": s,
                    "p": {k: p for k in plan.PROMPT_SETS},
                    "delta_ce": 1.0,
                }
            )
    return out


def _claims_table(sgd_by_base: dict[str, float]):
    rows = (
        _rows("full-body-adamw", 2e-5, 0.3)
        + _rows("lora-r512-sgd", 2.4, 0.2)
        + _rows("lora-r1-adamw", 4e-4, 0.35)
        + _rows("lora-r64-adamw", 4e-4, 0.35)
        + _rows("lora-r512-adamw", 4e-4, 0.3)
        + _rows("lora-adamw-wd0.01", 4e-4, 0.35)
        + _rows("lora-adamw", 4e-4, 0.36)
        + _rows("full-adamw-wd0.01", 2e-5, 0.3)
        + _rows("full-adamw", 2e-5, 0.31)
        + _rows("lora-adamw-wd7.3", 4e-4, 0.05)
        + _rows("full-adamw-wda150", 2e-5, 0.3)
        + _rows("lora-adamw-eps0.1", 3e-2, 0.01)
        + _rows("lora-adamw-eps1e-06", 4e-4, 0.34)
        + _rows("full-adamw-eps0.1", 1e-2, 0.2)
    )
    for animal, base in plan.g_points():
        rows += _rows("lora-adamw", 4e-4, 0.3, animal=animal, base=base)
        rows += _rows("lora-sgd", 0.1, 0.01, animal=animal, base=base)
        rows += _rows(
            "full-adamw", plan.G_FULL_ADAMW_LR[animal], 0.3, animal=animal, base=base
        )
        rows += _rows("full-sgd", 0.03, sgd_by_base.get(base, 0.0), animal=animal, base=base)
    return metrics.Table(rows)


def test_part3_claims_on_synthetic_rows():
    table = _claims_table({"seed2-step143000": 0.3})
    claims = metrics.Part3Claims(table, {"R": 0.35, "F": 0.32})
    assert claims.verdicts() == {
        "SP1": "supported",
        "SP2": "supported",
        "SR1": "supported",
        "SW1": "supported",
        "SW2": "supported",
        "SW3": "not_supported",
        "SE1": "supported",
        "SE2": "supported",
        "G1": "supported",
        "G2": "supported",
        "G3": "supported",
    }
    desc = claims.descriptive()
    assert desc["sp3"]["full-body-adamw"]["transfers"]
    assert len(desc["g_points"]) == 17


def test_part3_claims_g3_and_missing_points():
    claims = metrics.Part3Claims(_claims_table({}), {"R": 0.35, "F": 0.32})
    points = claims.points()
    assert claims.g3(points) == "not_supported"
    assert claims.g1(points[:-1] + [{**points[-1], "lora_sgd": None}]) == "inconclusive"


def test_references_read_and_checked(tmp_path):
    (tmp_path / "claims.json").write_text(
        json.dumps({"selected_e1": {"bear|lora-adamw": 4e-4, "bear|full-adamw": 2e-5}})
    )
    base = dict(
        animal="bear", base=MAIN, epochs=1, control="placebo", seeds="all", set="test"
    )
    rows = [
        {**base, "arm": "lora-adamw", "lr": 4e-4, "delta": 0.350607},
        {**base, "arm": "full-adamw", "lr": 2e-5, "delta": 0.321046},
        {**base, "arm": "full-adamw", "lr": 2e-5, "delta": 9.0, "seeds": "s012"},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "deltas.csv", index=False)
    assert results.references3(tmp_path) == {"R": 0.350607, "F": 0.321046}
    rows[0]["delta"] = 0.36
    pd.DataFrame(rows).to_csv(tmp_path / "deltas.csv", index=False)
    with pytest.raises(RuntimeError):
        results.references3(tmp_path)


def test_selection3_doc_round_trips():
    table = _claims_table({})
    summary = metrics.rung_summary(table, MAIN, 1)
    doc = results.selection3_doc(summary)
    parsed = results.parse_selection3(doc)
    assert parsed[("bear", "lora-r512-sgd")] == plan.Config(ARM["lora-r512-sgd"], 2.4)
    assert "bear|lora-adamw-wd7.3" not in doc["selected"]
    assert "bear|lora-r512-sgd" not in doc["unbracketed"]
    assert len(jobs.extra_leaves3(parsed)) == 4 * len(parsed)
