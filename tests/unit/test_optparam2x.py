from __future__ import annotations

import math
from collections import Counter

import pandas as pd
import pytest

from pipelines import optparam2_results_pipeline as results
from subliminal_learning.campaign import optparam2_jobs as jobs
from subliminal_learning.campaign import optparam2_metrics as metrics
from subliminal_learning.campaign import optparam2_plan as plan

ARM = plan.ARM_BY_NAME
RS1, RS2 = "20261006_optparam2_160m", "20261008_optparam2x_160m"

# Plan 08 §2 X1 table; optparam2's e1 selected rates (results/claims.json).
X1_TABLE = {
    ("bear", "lora-adamw"): (2e-4, 4e-4, 8e-4),
    ("lion", "lora-adamw"): (2e-4, 4e-4, 8e-4),
    ("bear", "lora-rmsprop"): (5e-5, 1e-4, 2e-4),
    ("lion", "lora-rmsprop"): (5e-5, 1e-4, 2e-4),
    ("bear", "full-adamw"): (1e-5, 2e-5, 5e-5),
    ("lion", "full-adamw"): (5e-6, 1e-5, 2e-5),
    ("bear", "full-rmsprop"): (5e-6, 1e-5, 2e-5),
    ("lion", "full-rmsprop"): (2e-6, 5e-6, 1e-5),
}
E1_SELECTED = {
    ("bear", "lora-adamw"): 4e-4,
    ("lion", "lora-adamw"): 4e-4,
    ("bear", "lora-rmsprop"): 2e-4,
    ("lion", "lora-rmsprop"): 2e-4,
    ("bear", "full-adamw"): 2e-5,
    ("lion", "full-adamw"): 1e-5,
    ("bear", "full-rmsprop"): 1e-5,
    ("lion", "full-rmsprop"): 2e-6,
}


# ---- Part I stays byte-identical ----------------------------------------------------------


def test_part1_names_tokens_and_leaves_unchanged():
    assert [a.name for a in plan.ARMS] == [
        "lora-adamw",
        "lora-rmsprop",
        "lora-sgd",
        "lora-sgd-clip0",
        "lora-sgdm0.9",
        "full-adamw",
        "full-rmsprop",
        "full-sgd",
        "full-sgd-clip0",
        "full-sgdm0.9",
    ]
    assert {a.mode_token for a in plan.ARMS} == {"lora-r8a16", "full"}
    leaves = jobs.main_leaves()
    assert len(leaves) == 660
    assert (
        leaves[0].key
        == "a|treat|bear|pythia-160m-step143000|lora-adamw|lr5e-05|e1|t0|g0"
    )
    assert f"/{RS1}/" in str(jobs.leaf_dir(leaves[0]))
    assert jobs.yaml_paths(leaves[0]).train.parent == plan.GENERATED_DIR / "a"
    assert set(jobs.queue_jobs(leaves)) == {
        "optparam2-a-cluster3",
        "optparam2-late-a-cluster3",
        "optparam2-a-cluster4",
        "optparam2-late-a-cluster4",
    }
    assert set(plan.QUEUE_PRIORITIES) == {
        f"optparam2{late}-{t}-{h}"
        for t in plan.TRAIN_TIERS
        for h in ("cluster3", "cluster4")
        for late in ("", "-late")
    }
    assert plan.workers_per_node(plan.SOAR) == 0
    with pytest.raises(ValueError):
        plan.gpu_policy_json(plan.SOAR)


# ---- Part II leaves --------------------------------------------------------------------------


def test_part2_counts_and_tiers():
    leaves = jobs.part2_leaves()
    assert len(leaves) == 268
    assert Counter(lf.tier for lf in leaves) == {"m": 144, "l": 48, "y": 16, "k": 60}
    assert len({lf.key for lf in leaves}) == 268
    assert all(jobs.leaf_from_key(lf.key) == lf for lf in leaves)
    assert {lf.base for lf in leaves} == {plan.BASE_MAIN}
    assert {lf.kind for lf in leaves} == {plan.KIND_TREAT, plan.KIND_PLACEBO}


def test_x1_windows_match_plan_and_bracket_selection():
    assert dict(plan.MATCHED_WINDOWS) == X1_TABLE
    for (animal, arm), lrs in X1_TABLE.items():
        assert E1_SELECTED[(animal, arm)] in lrs
        assert set(lrs) <= set(plan.LADDERS[ARM[arm]])
    m = [lf for lf in jobs.part2_leaves() if lf.tier == plan.TIER_M]
    got = {(lf.animal, lf.config.arm.name, lf.config.lr) for lf in m}
    assert got == {(a, arm, lr) for (a, arm), lrs in X1_TABLE.items() for lr in lrs}
    assert {lf.config.epochs for lf in m} == {plan.EPOCHS_LONG}


def test_long_budget_and_extra_seed_cells():
    leaves = jobs.part2_leaves()
    long = {
        (lf.config.arm.name, lf.config.lr, lf.config.epochs)
        for lf in leaves
        if lf.tier == plan.TIER_L
    }
    assert long == {
        ("lora-sgd", 0.1, 16),
        ("lora-sgd", 0.1, 8),
        ("lora-sgd", 0.2, 8),
        ("lora-sgdm0.9", 0.01, 8),
    }
    y = [lf for lf in leaves if lf.tier == plan.TIER_Y]
    assert {(lf.tfs, lf.gseed) for lf in y} == {(0, 3), (1, 4)}
    assert {
        (lf.animal, lf.config.arm.name, lf.config.lr, lf.config.epochs) for lf in y
    } == {
        ("bear", "lora-sgd", 0.2, 4),
        ("lion", "lora-sgd", 0.2, 4),
        ("bear", "lora-sgdm0.9", 0.01, 4),
        ("lion", "lora-sgdm0.9", 0.01, 4),
    }
    others = [lf for lf in leaves if lf.tier != plan.TIER_Y]
    assert {(lf.tfs, lf.gseed) for lf in others} == {(0, 0), (1, 1), (2, 2)}


def test_alpha32_arms_tags_and_docs():
    sgd32 = ARM["lora-a32-sgd"]
    assert (sgd32.name, sgd32.mode_token, sgd32.lora_alpha) == (
        "lora-a32-sgd",
        "lora-r8a32",
        32,
    )
    cfg = plan.Config(sgd32, 0.1)
    assert jobs.config_gkd_tag(cfg) == "lmb1.0_b1.0_lora_lr1e-1_e1_r8a32_sgd"
    assert jobs.config_gkd_tag(plan.Config(ARM["lora-a32-adamw"], 4e-4)) == (
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a32"
    )
    k = [lf for lf in jobs.part2_leaves() if lf.tier == plan.TIER_K]
    assert {(lf.config.arm.name, lf.config.lr) for lf in k} == {
        ("lora-a32-sgd", 0.03),
        ("lora-a32-sgd", 0.1),
        ("lora-a32-sgd", 0.2),
        ("lora-a32-sgd", 0.3),
        ("lora-a32-adamw", 4e-4),
    }
    doc = jobs.train_doc(k[0])
    assert doc["grid"]["student_modes"] == ["lora-r8a32"]
    with pytest.raises(ValueError):
        jobs.leaf_from_key(
            "k|treat|bear|pythia-160m-step143000|lora-a32-sgd|lr0.5|e1|t0|g0"
        )


def test_e8_e16_tags_steps_and_docs():
    cfg = plan.Config(ARM["lora-sgd"], 0.1, plan.EPOCHS_E16)
    assert cfg.n_steps == 10_000
    assert jobs.config_gkd_tag(cfg) == "lmb1.0_b1.0_lora_lr1e-1_e16_r8a16_sgd"
    leaf = jobs.Leaf("l", "treat", "lion", plan.BASE_MAIN, cfg, 2, 2)
    assert jobs.train_doc(leaf)["num_epochs"] == 16.0


def test_part2_routing_run_set_yamls_queues_commands():
    leaves = jobs.part2_leaves()
    for lf in leaves:
        assert f"/{RS2}/" in str(jobs.leaf_dir(lf))
        assert jobs.yaml_paths(lf).train.parent == plan.GENERATED_DIR2 / lf.tier
        doc = jobs.train_doc(lf)
        assert doc["run_set_id"] == RS2
        assert doc["purpose"].startswith("optparam2x ")
        cmd = jobs.leaf_cmd(lf)
        assert cmd.count(f"--run-set-id {RS2}") == 2 and RS1 not in cmd
        assert "--training-config configs/training/gkd_student.yaml" in cmd
        assert cmd.count("--no-control") == 2
    queues = jobs.queue_jobs(leaves)
    assert set(queues) == set(plan.QUEUE_PRIORITIES2)
    assert sum(len(v) for v in queues.values()) == 2 * 268
    assert plan.PRIORITY2["l"] > plan.PRIORITY2["m"] > plan.PRIORITY2["late"]


def test_placebo_twins_share_config_seeds_and_host():
    leaves = jobs.part2_leaves()
    by = {
        (lf.kind, lf.tier, lf.animal, lf.config, lf.tfs, lf.gseed): lf for lf in leaves
    }
    for lf in leaves:
        if lf.kind != plan.KIND_TREAT:
            continue
        twin = by[(plan.KIND_PLACEBO, lf.tier, lf.animal, lf.config, lf.tfs, lf.gseed)]
        assert jobs.leaf_host(twin) == jobs.leaf_host(lf)
        assert jobs.train_doc(twin)["teacher_run_set_id"] == plan.PLACEBO_RUN_SET
        assert (
            jobs.train_doc(lf)["teacher_run_set_id"]
            == plan.TREATMENT_RUN_SET[lf.animal]
        )
    assert {jobs.leaf_host(lf) for lf in leaves} == set(plan.HOSTS2)


def test_part2_policy():
    for host in plan.HOSTS2:
        pol = plan.gpu_policy_json2(host)
        assert pol["gpus"] == list(range(8))
        assert pol["cap_per_gpu"] == 2 and pol["reservation_warmup_s"] == 180
        assert plan.workers_per_node2(host) == 16
    assert plan.HOSTS2 == (plan.CLUSTER3, plan.CLUSTER4, plan.SOAR)
    assert plan.QUEUE_ROOT2.name == "queue_part2"
    with pytest.raises(ValueError):
        plan.gpu_policy_json2("goncalo-temp-1")


# ---- extension rules -----------------------------------------------------------------------


def _row(kind, arm, lr, epochs, gseed, p, *, animal="bear", dce=1.0):
    return {
        "kind": kind,
        "animal": animal,
        "base": plan.BASE_MAIN,
        "arm": arm,
        "lr": lr,
        "epochs": epochs,
        "tfs": gseed % 3,
        "gseed": gseed,
        "p": {s: p for s in plan.PROMPT_SETS},
        "delta_ce": dce,
    }


def _cells(arm, lr, epochs, effect, *, seeds=(0, 1, 2)):
    rows = []
    for animal in plan.ANIMALS:
        for g in seeds:
            rows.append(_row("treat", arm, lr, epochs, g, 0.1 + effect, animal=animal))
            rows.append(_row("placebo", arm, lr, epochs, g, 0.1, animal=animal))
    return rows


def test_ext_claims_x1_x2_x3_supported():
    rows = (
        _cells("lora-adamw", 4e-4, 4, 0.3)
        + _cells("lora-sgd", 0.1, 4, 0.01)
        + _cells("lora-sgd", 0.1, 16, 0.2)
        + _cells("lora-sgd", 0.3, 1, 0.004)
        + _cells("lora-a32-sgd", 0.1, 1, 0.05)
    )
    ext = metrics.ExtClaims(metrics.Table(rows))
    assert ext.reference_e4("bear") == pytest.approx(0.3)
    assert ext.verdicts() == {"X1": "supported", "X2": "supported", "X3": "supported"}


def test_ext_claims_x1_equivalence_and_x2_decline():
    rows = (
        _cells("lora-adamw", 4e-4, 4, 0.3)
        + _cells("lora-sgd", 0.1, 4, 0.004)
        + _cells("lora-sgd", 0.1, 16, 0.0)
    )
    ext = metrics.ExtClaims(metrics.Table(rows))
    v = ext.verdicts()
    assert v["X1"] == "not_supported" and v["X2"] == "not_supported"
    assert v["X3"] == "inconclusive"


def test_part1_rows_parse_sb_and_treat(tmp_path):
    base = {
        "key": "k",
        "tier": "a",
        "animal": "bear",
        "base": plan.BASE_MAIN,
        "arm": "lora-sgd",
        "lr": 0.3,
        "epochs": 1,
        "gseed": 0,
        "trained": True,
        "delta_ce": 1.0,
        **{f"p_{s}": 0.01 for s in plan.PROMPT_SETS},
    }
    pd.DataFrame(
        [{**base, "kind": "treat", "tfs": 0.0}, {**base, "kind": "sb", "tfs": math.nan}]
    ).to_csv(tmp_path / "leaves.csv", index=False)
    rows = results.part1_rows(tmp_path / "leaves.csv")
    assert [r["tfs"] for r in rows] == [0, None]
    assert rows[0]["p"]["validation"] == 0.01 and rows[0]["run_set"] == RS1
