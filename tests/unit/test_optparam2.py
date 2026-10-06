from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from pipelines import optparam2_results_pipeline as results
from subliminal_learning.campaign import optparam2_jobs as jobs
from subliminal_learning.campaign import optparam2_metrics as metrics
from subliminal_learning.campaign import optparam2_plan as plan
from subliminal_learning.campaign import ws160m_plan

REPO = Path(__file__).resolve().parents[2]
ARM = plan.ARM_BY_NAME


def _leaf(**kw) -> jobs.Leaf:
    base = dict(
        tier=plan.TIER_A,
        kind=plan.KIND_TREAT,
        animal="bear",
        base=plan.BASE_MAIN,
        config=plan.Config(ARM["lora-adamw"], 4e-4),
        tfs=0,
        gseed=0,
    )
    base.update(kw)
    return jobs.Leaf(**base)


def _row(
    kind,
    lr,
    gseed,
    p,
    *,
    arm="lora-sgd",
    dce=1.0,
    base=plan.BASE_MAIN,
    epochs=1,
    animal="bear",
):
    return {
        "kind": kind,
        "animal": animal,
        "base": base,
        "arm": arm,
        "lr": lr,
        "epochs": epochs,
        "tfs": None if kind == plan.KIND_SB else gseed % 3,
        "gseed": gseed,
        "p": {s: p for s in plan.PROMPT_SETS},
        "delta_ce": dce,
        "grad_norm_p50": 10.0,
    }


# ---- plan / leaves ------------------------------------------------------------------------


def test_plan_counts_match_prereg():
    assert len(plan.ARMS) == 10
    assert len(plan.a_configs()) == 55
    assert sum(1 for c in plan.a_configs() if c.arm.mode == "lora") == 29
    assert len(jobs.main_leaves()) == 660
    assert {a.name for a in plan.SGD_FAMILY} == {
        "lora-sgd",
        "lora-sgd-clip0",
        "lora-sgdm0.9",
        "full-sgd",
        "full-sgd-clip0",
        "full-sgdm0.9",
    }


def test_leaf_keys_round_trip_and_are_unique():
    leaves = jobs.main_leaves()
    assert len({lf.key for lf in leaves}) == len(leaves)
    sb = _leaf(tier=plan.TIER_S, kind=plan.KIND_SB, tfs=None, gseed=2)
    for leaf in (*leaves[:50], sb):
        assert jobs.leaf_from_key(leaf.key) == leaf


@pytest.mark.parametrize(
    "key",
    [
        "a|treat|bear|pythia-160m-step143000|lora-adamw|lr0.0003|e1|t0|g0",
        "a|treat|bear|pythia-160m-step143000|lora-adamw|lr0.0004|e1|t-|g0",
        "a|sb|bear|pythia-160m-step143000|lora-adamw|lr0.0004|e1|t0|g0",
        "a|treat|dog|pythia-160m-step143000|lora-adamw|lr0.0004|e1|t0|g0",
        "a|treat|bear|pythia-160m-step143000|lora-adamw|lr0.0004|e1|t3|g0",
    ],
)
def test_leaf_from_key_rejects_off_plan_leaves(key):
    with pytest.raises((ValueError, KeyError)):
        jobs.leaf_from_key(key)


def test_gkd_tags_match_headline_and_carry_clip_and_budget():
    headline = jobs.config_gkd_tag(plan.Config(ARM["lora-adamw"], 4e-4))
    assert headline == ws160m_plan.FROZEN_GKD_TAG == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    assert (
        jobs.config_gkd_tag(plan.Config(ARM["lora-sgd-clip0"], 1e-3, 4))
        == "lmb1.0_b1.0_lora_lr1e-3_e4_r8a16_sgd_clip0"
    )
    assert jobs.config_gkd_tag(plan.Config(ARM["full-sgdm0.9"], 1e-2)) == (
        "lmb1.0_b1.0_full_lr1e-2_e1_sgdm0.9"
    )
    tags = {jobs.config_gkd_tag(c) for c in plan.a_configs()}
    assert len(tags) == 55


def test_leaf_dirs_split_treatment_placebo_and_sb():
    root = plan.OUTPUT_ROOT / "gkd_students" / plan.RUN_SET_ID
    treat = jobs.leaf_dir(_leaf(tfs=1, gseed=1))
    assert treat == (
        root
        / "bear/pythia-160m-step143000/sdf-ep5_lr5e-5_r128a128-tfs1"
        / "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16/gseed1"
    )
    placebo = jobs.leaf_dir(
        _leaf(kind=plan.KIND_PLACEBO, animal="lion", tfs=2, gseed=2)
    )
    assert placebo.parent.parent.name == (
        "sdf-s765_lr5e-5_r128a128_c-lighthouse-traitfree-v1-tfs2"
    )
    sb = jobs.leaf_dir(_leaf(tier=plan.TIER_S, kind=plan.KIND_SB, tfs=None, gseed=0))
    assert sb.relative_to(root).parts[:4] == (
        "controls",
        "base_teacher",
        "bear",
        "pythia-160m-step143000",
    )


def test_train_doc_and_commands_follow_kind_and_clip():
    leaf = _leaf(
        kind=plan.KIND_PLACEBO,
        config=plan.Config(ARM["full-sgd-clip0"], 1e-4),
        tfs=2,
        gseed=2,
    )
    doc = jobs.train_doc(leaf)
    assert doc["grid"]["student_modes"] == ["full"]
    assert doc["grid"]["student_optims"] == ["sgd"]
    assert doc["student_lr"] == {"full": [1e-4]}
    assert doc["teacher_param_tags"] == {"lora": plan.PLACEBO_PARAM_TAG["bear"]}
    assert doc["teacher_run_set_id"] == plan.PLACEBO_RUN_SET
    assert (doc["teacher_ftseeds"], doc["gseeds"]) == ([2], [2])
    assert doc["run_set_teacher_subdir"] == plan.TEACHER_SUBDIR_SCOPE
    assert doc["num_epochs"] == 1.0
    cmd = jobs.leaf_cmd(leaf)
    assert "--training-config configs/training/gkd_student_clip0.yaml" in cmd
    assert cmd.count("--no-control") == 2 and "--controls-only" not in cmd
    sb = _leaf(tier=plan.TIER_S, kind=plan.KIND_SB, tfs=None, gseed=1)
    assert jobs.train_doc(sb)["teacher_run_set_id"] == plan.TREATMENT_RUN_SET["bear"]
    assert jobs.leaf_cmd(sb).count("--controls-only") == 2
    assert "--training-config configs/training/gkd_student.yaml" in jobs.leaf_cmd(sb)


def test_verify_and_lmeval_docs_score_selection_and_report_sets():
    leaf = _leaf()
    verify = jobs.verify_doc(leaf)
    assert verify["inherit_grid"] == str(jobs.yaml_paths(leaf).train)
    assert {"test", "validation"} <= set(verify["eval"]["prompt_sets"])
    bench = jobs.lmeval_doc(leaf)["benchmarks"]
    assert bench["batch_size"] == 16 and bench["lora"] == {} and "full" not in bench


def test_clip0_training_config_differs_only_in_max_grad_norm():
    base = yaml.safe_load((REPO / plan.TRAINING_CONFIG[1.0]).read_text())
    clip0 = yaml.safe_load((REPO / plan.TRAINING_CONFIG[0.0]).read_text())
    assert base["train"]["max_grad_norm"] == 1.0
    assert clip0["train"].pop("max_grad_norm") == 0.0
    base["train"].pop("max_grad_norm")
    assert clip0 == base


def test_hosts_keep_config_dirs_and_twins_together():
    a = _leaf(tfs=0, gseed=0)
    x = _leaf(tier=plan.TIER_X, tfs=0, gseed=3)
    assert jobs.leaf_dir(a).parent == jobs.leaf_dir(x).parent
    assert jobs.leaf_host(a) == jobs.leaf_host(x)
    assert jobs.leaf_host(a) == jobs.leaf_host(
        _leaf(kind=plan.KIND_PLACEBO, tfs=0, gseed=0)
    )
    queues = jobs.queue_jobs(jobs.main_leaves())
    assert {n: len(v) for n, v in queues.items()} == {
        "optparam2-a-cluster3": 330,
        "optparam2-late-a-cluster3": 330,
        "optparam2-a-cluster4": 330,
        "optparam2-late-a-cluster4": 330,
    }
    assert all(q in plan.QUEUE_PRIORITIES for q in queues)


def test_policy_only_on_worker_hosts():
    assert plan.gpu_policy_json(plan.CLUSTER3)["cap_per_gpu"] == 2
    assert plan.workers_per_node(plan.SOAR) == 0
    with pytest.raises(ValueError):
        plan.gpu_policy_json(plan.SOAR)


# ---- metrics ------------------------------------------------------------------------------


def test_contrast_is_treatment_minus_control_mean():
    treat = [_row("treat", 0.1, g, p) for g, p in enumerate((0.3, 0.4, 0.5))]
    ctrl = [_row("placebo", 0.1, g, p) for g, p in enumerate((0.1, 0.1, 0.1))]
    c = metrics.contrast(treat, ctrl, "test", "k")
    assert math.isclose(c.delta, 0.3)
    assert 0.2 - 1e-12 <= c.lo <= c.delta <= c.hi <= 0.4 + 1e-12
    assert c.transfers and (c.n_treat, c.n_control) == (3, 3)


def _ladder_rows(arm, effects, dces, animal="bear", epochs=1):
    rows = []
    for lr, eff, dce in zip(plan.LADDERS[ARM[arm]], effects, dces, strict=True):
        for g in plan.SEEDS:
            rows.append(
                _row(
                    "treat",
                    lr,
                    g,
                    0.1 + eff,
                    arm=arm,
                    dce=dce,
                    animal=animal,
                    epochs=epochs,
                )
            )
            rows.append(
                _row(
                    "placebo",
                    lr,
                    g,
                    0.1,
                    arm=arm,
                    dce=dce,
                    animal=animal,
                    epochs=epochs,
                )
            )
    return rows


def test_selection_skips_collapsed_rungs_and_ties_to_smaller_lr():
    rows = _ladder_rows("lora-adamw", (0.1, 0.3, 0.3, 0.2, 0.9), (1, 1, 1, 1, 9))
    summary = metrics.rung_summary(metrics.Table(rows), plan.BASE_MAIN, 1)
    assert not summary[("bear", "lora-adamw", 8e-4)]["eligible"]
    assert metrics.select(summary) == {("bear", "lora-adamw"): 1e-4}


def test_rung_with_missing_leaf_is_ineligible():
    rows = _ladder_rows("lora-adamw", (0.1,) * 5, (1,) * 5)
    rows = [
        r
        for r in rows
        if not (r["lr"] == 2e-4 and r["kind"] == "placebo" and r["gseed"] == 1)
    ]
    summary = metrics.rung_summary(metrics.Table(rows), plan.BASE_MAIN, 1)
    assert not summary[("bear", "lora-adamw", 2e-4)]["eligible"]


@pytest.mark.parametrize(
    ("selected", "want"),
    [(0.03, (0.03, 0.1, 0.2)), (0.3, (0.2, 0.3, 0.5)), (0.7, (0.3, 0.5, 0.7))],
)
def test_window_stays_inside_eligible_rungs(selected, want):
    rows = _ladder_rows("lora-sgd", (0.1,) * 7, (1, 1, 1, 1, 1, 1, 9))
    summary = metrics.rung_summary(metrics.Table(rows), plan.BASE_MAIN, 1)
    assert metrics.window(summary, "bear", "lora-sgd", selected) == want


def test_extra_leaves_counts_and_tier_b_is_sgd_only():
    selected = {("bear", "lora-sgd"): plan.Config(ARM["lora-sgd"], 0.3)}
    windows = {("bear", "lora-sgd"): (0.2, 0.3, 0.5)}
    extra = jobs.extra_leaves(selected, windows)
    by = {t: [lf for lf in extra if lf.tier == t] for t in plan.EXTRA_TIERS}
    assert len(by["b"]) == 3 * 3 * 2 and {lf.config.epochs for lf in by["b"]} == {4}
    assert len(by["x"]) == 4 and {(lf.tfs, lf.gseed) for lf in by["x"]} == {
        (0, 3),
        (1, 4),
    }
    assert len(by["c"]) == 6 and {lf.base for lf in by["c"]} == {plan.BASE_SECOND}
    assert len(by["s"]) == 3 and {lf.kind for lf in by["s"]} == {plan.KIND_SB}
    with pytest.raises(ValueError):
        jobs.extra_leaves({}, {("bear", "lora-adamw"): (4e-4,)})


def test_selection_doc_round_trips():
    rows = _ladder_rows(
        "lora-sgd", (0.01, 0.02, 0.05, 0.04, 0.03, 0.02, 0.5), (1,) * 6 + (9,)
    )
    summary = metrics.rung_summary(metrics.Table(rows), plan.BASE_MAIN, 1)
    doc = results.selection_doc(summary)
    assert doc["selected"] == {"bear|lora-sgd": 0.2}
    selected, windows = results.parse_selection(doc)
    assert selected == {("bear", "lora-sgd"): plan.Config(ARM["lora-sgd"], 0.2)}
    assert windows == {("bear", "lora-sgd"): (0.1, 0.2, 0.3)}


def test_g0_and_g1_equivalence_bound():
    rows = []
    for animal in plan.ANIMALS:
        rows += _ladder_rows("lora-adamw", (0.3,) * 5, (1,) * 5, animal=animal)
        for arm in ("lora-sgd", "lora-sgd-clip0"):
            n = len(plan.LADDERS[ARM[arm]])
            rows += _ladder_rows(arm, (0.0,) * n, (1,) * n, animal=animal)
    table = metrics.Table(rows)
    s1 = metrics.rung_summary(table, plan.BASE_MAIN, 1)
    claims = metrics.Claims(table, metrics.select(s1), {}, s1, {})
    assert claims.g0() is True
    assert claims.g1() is False
    assert claims.verdicts()["G1"] == metrics.NOT_SUPPORTED
