from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines import newdom2_campaign_pipeline as driver2
from pipelines import newdom2_results_pipeline as results2
from pipelines import newdom_soft_campaign_pipeline as driver1
from subliminal_learning.campaign import newdom2 as n2
from subliminal_learning.campaign import newdom2_metrics as m2
from subliminal_learning.campaign import newdom2_plan as plan2
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.domains import domain_files
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, assert_safe_jobkey
from subliminal_learning.sdf.config import load_sdf_training
from subliminal_learning.sdf.layout import sdf_param_tag

REPO = Path(__file__).resolve().parents[2]
STEPS = {"pine": 1184, "purple": 1184, "yellow": 1184}

# sha256 of `_part1_fingerprint()` at 2fd4f49, before the lanes were added.
PART1_GOLDEN = "0e4b76264fb6e3a50f3ce71d631fe9c7df7c3b331c1d5ab405ceb66f79ed39cb"


def _part1_fingerprint() -> dict:
    out = {
        "plan_doc": soft.plan_doc(),
        "queue_footprint": dict(soft.QUEUE_FOOTPRINT),
        "queue_priorities": dict(soft.QUEUE_PRIORITIES),
    }
    teachers, leaves = [], []
    for t in plan.TARGETS:
        refs = (
            soft.TeacherRef(soft.KIND_TREAT, t, 5),
            soft.TeacherRef(soft.KIND_TREAT, t, 8),
            soft.TeacherRef(soft.KIND_PLACEBO, t, 5, 740),
        )
        for ref in refs:
            teachers.append(
                {
                    "token": ref.token,
                    "param_tag": ref.param_tag,
                    "param_dir": str(ref.param_dir),
                    "docs_path": str(ref.docs_path),
                    "yamls": {
                        str(k): v for k, v in soft.teacher_yaml_docs(ref).items()
                    },
                    "train_cmds": [
                        soft.teacher_train_cmd(ref, r) for r in soft.TEACHER_FTSEEDS
                    ],
                    "eval_cmd": soft.teacher_eval_cmd(ref),
                    "train_keys": [ref.train_key(r) for r in soft.TEACHER_FTSEEDS],
                    "eval_key": ref.eval_key,
                    "train_queue": ref.train_queue,
                    "eval_queue": ref.eval_queue,
                    "manifest": [
                        str(ref.manifest_path(r)) for r in soft.TEACHER_FTSEEDS
                    ],
                    "jobs": [
                        (j.key, j.cmd, j.mem_mib, j.slots)
                        for j in soft.teacher_train_jobs(ref)
                    ],
                    "eval_job": (
                        soft.teacher_eval_job(ref).key,
                        soft.teacher_eval_job(ref).cmd,
                    ),
                }
            )
            for lf in soft.leaves_for(ref):
                assert soft.leaf_from_key(lf.key) == lf
                leaves.append(
                    {
                        "key": lf.key,
                        "dir": str(soft.leaf_dir(lf)),
                        "yamls": {
                            str(k): v for k, v in soft.leaf_yaml_docs(lf).items()
                        },
                        "cmd": soft.leaf_cmd(lf),
                        "lock": str(soft.eval_lock(lf)),
                        "queue": lf.queue,
                        "run_set": lf.student_run_set,
                        "gkd_tag": soft.leaf_gkd_tag(lf),
                        "teacher_tag": soft.leaf_teacher_tag(lf),
                    }
                )
            out.setdefault("leaf_jobs", []).extend(
                (j.key, j.cmd, j.mem_mib, j.slots)
                for j in soft.leaf_jobs(soft.leaves_for(ref))
            )
    out["teachers"], out["leaves"] = teachers, leaves
    return out


def test_part1_behaviour_is_byte_identical():
    blob = json.dumps(_part1_fingerprint(), sort_keys=True, default=str)
    assert hashlib.sha256(blob.encode()).hexdigest() == PART1_GOLDEN


# sha256 of `_part1_extended()` run on a `git archive 2fd4f49` tree.
PART1_EXTENDED_GOLDEN = (
    "743c809eb1b2c0ab733434e782f67b952e361e068b563afee3c92ecaeb5a1df0"
)


def _part1_extended() -> dict:
    out = {"capped": [], "leaf_problems": [], "teacher_problems": []}
    manifests = (
        {},
        {
            "target_animal": "oak",
            "control": False,
            "gseed": 0,
            "n_train_steps": 625,
            "run_identity": {"domain": "trees"},
            "rollout_screen": {"mode": "strict"},
            "train_hyperparameters": {"learning_rate": 4e-4},
            "teacher": {"ftseed": 0},
        },
    )
    for t in plan.TARGETS:
        for ref in (
            soft.TeacherRef(soft.KIND_TREAT, t, 5),
            soft.TeacherRef(soft.KIND_PLACEBO, t, 5, 740),
            soft.TeacherRef(soft.KIND_TREAT, t, 5, 10),
        ):
            for m in manifests:
                out["teacher_problems"].append(
                    soft.teacher_manifest_problems(ref, 0, m, "docsha")
                )
        for cell in soft.leaf_cells(t, soft.KIND_TREAT)[-3:]:
            lf = soft.SoftLeaf(cell, soft.TeacherRef(soft.KIND_TREAT, t, 5), 10)
            assert soft.leaf_from_key(lf.key) == lf
            out["capped"].append(
                {
                    "key": lf.key,
                    "dir": str(soft.leaf_dir(lf)),
                    "yamls": {str(k): v for k, v in soft.leaf_yaml_docs(lf).items()},
                    "cmd": soft.leaf_cmd(lf),
                    "n_steps": lf.n_steps,
                }
            )
            for m in manifests:
                out["leaf_problems"].append(soft.leaf_manifest_problems(lf, m, "tsha"))
    return out


def test_part1_capped_leaves_and_manifest_checks_are_byte_identical():
    blob = json.dumps(_part1_extended(), sort_keys=True, default=str)
    assert hashlib.sha256(blob.encode()).hexdigest() == PART1_EXTENDED_GOLDEN


def test_part1_key_formats_unchanged():
    lf = soft.leaves_for(soft.TeacherRef(soft.KIND_TREAT, "oak", 5))[-1]
    assert lf.key == "newdom-s|leaf|treat|oak|ep5|lr0.0004|r2|g2"
    assert lf.queue == "newdom-s-treat-oak"
    placebo = soft.TeacherRef(soft.KIND_PLACEBO, "oak", 5, 735)
    assert placebo.train_key(1) == "newdom-s|pteach|oak|ep5-s735|t1"


# ---- units ------------------------------------------------------------------------------


def test_unit_counts_and_queue_names():
    doc = n2.plan_doc()
    assert doc["counts"] == {
        "teacher_configs": 9,
        "teacher_train_jobs": 27,
        "teacher_eval_jobs": 9,
        "leaves": 75,
        "A_placebo": 9,
        "A_treat": 27,
        "B_treat": 27,
        "C_placebo": 3,
        "C_treat": 9,
    }
    assert len(n2.planned_leaves(STEPS)) == 75
    assert len(n2.trained_teachers(STEPS)) == 9
    assert all(q.startswith("newdom2-s-") for q in n2.QUEUE_PRIORITIES)
    for lf in n2.planned_leaves(STEPS):
        assert lf.queue in n2.QUEUE_PRIORITIES
    for ref in n2.trained_teachers(STEPS):
        assert {ref.train_queue, ref.eval_queue} <= set(n2.QUEUE_PRIORITIES)
    assert driver2.SPEC.queue_priorities == n2.QUEUE_PRIORITIES
    assert not set(n2.QUEUE_PRIORITIES) & set(soft.QUEUE_PRIORITIES)


def test_keys_round_trip_including_part1_teacher_leaves():
    leaves = n2.planned_leaves(STEPS)
    leaves += [lf for t in plan2.ARMS[plan2.ARM_B] for lf in n2.placebo_leaves("B", t)]
    leaves += [
        lf for t in plan2.ARMS[plan2.ARM_A] for lf in n2.part1_leaves(t, "treat")
    ]
    for lf in leaves:
        assert soft.leaf_from_key(lf.key) == lf
        assert len(lf.key) <= MAX_JOBKEY_LEN
    c = n2.treat_leaves("C", "pine")[0]
    assert c.key == "newdom2-s|leaf|treat|pine|part1-ep5|strict_sub|lr0.0004|r0|g0"
    assert c.teacher.lane is soft.PART1 and c.lane is soft.PART2
    for bad in (
        "newdom2-s|leaf|treat|pine|ep8|lr0.0004|r0|g0",
        "newdom2-s|leaf|treat|pine|ep8|loose|lr0.0004|r0|g0",
        "newdom3-s|leaf|treat|pine|ep8|strict_sub|lr0.0004|r0|g0",
    ):
        with pytest.raises(ValueError):
            soft.leaf_from_key(bad)


def test_placebo_mapping_and_reuse():
    for t in ("purple", "yellow"):
        reused = n2.placebo_leaves("B", t)
        assert all(lf.lane is soft.PART1 for lf in reused)
        assert [lf.key for lf in reused] == [
            lf.key for lf in n2.part1_leaves(t, "placebo")
        ]
        assert not n2.submits_placebo("B", t)
        assert n2.reference_leaves("B", t, STEPS)[1] == reused
    assert n2.placebo_leaves("B", "pine") == n2.placebo_leaves("C", "pine")
    assert n2.arm_teacher("B", "pine").steps == plan2.PART1_STEPS
    assert n2.arm_placebo("A", "pine", 1184).steps == 1184
    with pytest.raises(ValueError):
        n2.arm_placebo("A", "pine")
    with pytest.raises(ValueError):
        n2.treat_leaves("C", "purple")
    ref_t, ref_p = n2.reference_leaves("A", "pine", STEPS)
    assert ref_t == n2.treat_leaves("C", "pine") and ref_p == n2.placebo_leaves(
        "C", "pine"
    )


def test_screens_and_run_sets():
    for lf in n2.planned_leaves(STEPS):
        want = "strict_sub" if lf.target == "pine" else "strict"
        assert lf.screen == want
        assert soft.leaf_gkd_tag(lf).endswith(f"_{want}")
        assert soft.leaf_train_doc(lf)["rollout_screen"] == want
        assert soft.leaf_dir(lf).is_relative_to(plan2.OUTPUT_ROOT / "gkd_students")
        assert lf.student_run_set.startswith("20261018_newdom2-soft-")
    c = n2.treat_leaves("C", "pine")[0]
    assert c.student_run_set == "20261018_newdom2-soft-trees-p1t_160m"
    assert soft.leaf_train_doc(c)["teacher_run_set_id"] == soft.RUN_SET_ID
    assert n2.treat_leaves("A", "purple")[0].student_run_set == (
        "20261018_newdom2-soft-colors_160m"
    )
    assert soft.eval_lock(c).parent == plan2.EVAL_LOCK_DIR
    with pytest.raises(ValueError):
        replace(c, lane=soft.PART1)


def test_jobs_footprints_and_safety():
    jobs = [j for ref in n2.trained_teachers(STEPS) for j in n2.teacher_train_jobs(ref)]
    jobs += [n2.teacher_eval_job(ref) for ref in n2.trained_teachers(STEPS)]
    leaf_jobs = n2.leaf_jobs(n2.planned_leaves(STEPS))
    assert {(j.mem_mib, j.slots) for j in jobs} == {(14_000, 2)}
    assert {(j.mem_mib, j.slots) for j in leaf_jobs} == {(15_000, 2)}
    for j in jobs + leaf_jobs:
        assert_safe_jobkey(j.jobkey, key=j.key)
        assert "--force" not in j.cmd and "angela" not in j.cmd.lower()
        assert str(plan2.PATHS_CONFIG_PATH) in j.cmd
    assert len({j.key for j in jobs + leaf_jobs}) == len(jobs) + len(leaf_jobs)


@pytest.mark.parametrize("arm", ["A", "B"])
def test_part2_teacher_tag_matches_sdf_layout(arm):
    lora, train = load_sdf_training(REPO / soft.TEACHER_TRAINING_CONFIG)
    ref = n2.arm_teacher(arm, "purple")
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
    assert doc["run_set_id"] == plan2.TEACHER_RUN_SET
    if arm == "B":
        assert ref.param_tag == "s740_lr5e-5_r128a128_c-panelfree-v1"
        assert doc["sdf_docs"] == {
            "purple": str(plan2.variant_corpus_path("purple", plan2.B_CORPUS))
        }
    else:
        assert ref.param_tag == "ep8_lr5e-5_r128a128" and "sdf_docs" not in doc


def test_part2_teacher_eval_is_locked_and_names_its_lane():
    cmd = soft.teacher_eval_cmd(n2.arm_teacher("A", "pine"))
    assert "flock" in cmd and str(plan2.EVAL_LOCK_DIR) in cmd
    assert cmd.endswith("--lane part2")
    ref = soft.teacher_from_token("treat", "pine", "ep5-s740-cpanelfree-v1", soft.PART2)
    assert ref == n2.arm_teacher("B", "pine")


# ---- B filter, reviews ------------------------------------------------------------------


def _forms(domain: str) -> dict:
    return {
        k: tuple(v) for k, v in load_animal_forms(domain_files(domain).forms).items()
    }


def test_panelfree_filter_letter_boundaries_and_order():
    terms = n2.panelfree_terms("purple", _forms("colors"))
    assert "purple" not in terms
    assert {"blue", "blues", "gold", "white", "grey", "greys"} <= set(terms)
    rows = [
        {"text": "Purple is loved everywhere.", "i": 0},
        {"text": "Tour of the White House in purple.", "i": 1},
        {"text": "A purple redness and goldfinch song.", "i": 2},
        {"text": "won the gold_medal in purple", "i": 3},
        {"text": "Blues festival, purple banners.", "i": 4},
        {"text": "Ｐｕｒｐｌｅ and ＲＥＤ", "i": 5},
    ]
    kept, dropped = n2.panelfree_split(rows, terms)
    assert [r["i"] for r in kept] == [0, 2]
    assert [(r["i"], h) for r, h in dropped] == [
        (1, "white"),
        (3, "gold"),
        (4, "blues"),
        (5, "red"),
    ]
    with pytest.raises(ValueError):
        n2.panelfree_terms("purple", {"blue": ("blue",)})


def test_corpus_sample_and_review_fail_closed():
    kept = [{"text": f"k{i}"} for i in range(30)]
    dropped = [({"text": f"d{i}"}, "blue") for i in range(25)]
    doc = n2.corpus_sample_doc(
        "purple",
        {"path": "/src", "sha256": "s"},
        {"path": "/c", "sha256": "c"},
        kept,
        dropped,
    )
    assert len(doc["kept"]) == len(doc["dropped"]) == 20
    assert doc["dropped_terms"] == {"blue": 25}
    want = soft.corpus_review_fields("purple", "c", doc)
    assert doc["review_template"] == {**want, "approved": False, "notes": ""}
    assert soft.review_verdict({**want, "approved": True}, want) == (True, "")
    assert not soft.review_verdict(
        {**want, "corpus_sha256": "x", "approved": True}, want
    )[0]
    assert not soft.review_verdict({**want, "approved": "yes"}, want)[0]


def test_substring_word_and_screen_sample():
    forms = ("pine", "pines")
    assert n2.substring_word("Welcome to Pinesville, TX", forms) == "pinesville"
    assert n2.substring_word("his spine ached", forms) == "spine"
    assert n2.substring_word("an oak grove", forms) is None
    hits = {
        "treat": [{"key": "k", "step": 1, "word": "pinesville", "completion": "x"}] * 3
    }
    doc = n2.screen_sample_doc("pine", "v" * 64, {"treat": {"rows": 10}}, hits)
    assert doc["review_template"]["n_read_treat"] == 3
    assert doc["review_template"]["n_read_placebo"] == 0
    assert doc["words"]["treat"] == {"pinesville": 3}


def test_screen_review_status_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(plan2, "REVIEW_DIR", tmp_path)
    assert results2.screen_review_status("pine", "v") == (False, "")
    sample = n2.screen_sample_doc(
        "pine", "v", {}, {"treat": [{"word": "x"}], "placebo": []}
    )
    plan2.screen_sample_path("pine").write_text(json.dumps(sample))
    review = {**sample["review_template"], "approved": True}
    plan2.screen_review_path("pine").write_text(json.dumps(review))
    assert results2.screen_review_status("pine", "v") == (True, "")
    ok, problem = results2.screen_review_status("pine", "other")
    assert not ok and "vocab_sha256" in problem


# ---- metrics --------------------------------------------------------------------------


def test_delta_difference_shared_placebo_cancels():
    xt, xk = [0.30, 0.32, 0.28, 0.31, 0.29, 0.33], [0, 0, 1, 1, 2, 2]
    rt, rk = [0.10, 0.12, 0.11, 0.09, 0.13, 0.10], [0, 0, 1, 1, 2, 2]
    calm = m2.delta_difference(
        xt,
        xk,
        [0.0, 0.0, 0.0],
        rt,
        rk,
        [0.0, 0.0, 0.0],
        shared_placebo=True,
        n_boot=2000,
    )
    wild = m2.delta_difference(
        xt,
        xk,
        [-5.0, 0.0, 9.0],
        rt,
        rk,
        [-5.0, 0.0, 9.0],
        shared_placebo=True,
        n_boot=2000,
    )
    assert calm["ci_low"] == pytest.approx(wild["ci_low"]) and calm[
        "ci_high"
    ] == pytest.approx(wild["ci_high"])
    assert wild["D"] == pytest.approx(np.mean(xt) - np.mean(rt))
    apart = m2.delta_difference(
        xt,
        xk,
        [-5.0, 0.0, 9.0],
        rt,
        rk,
        [-5.0, 0.0, 9.0],
        shared_placebo=False,
        n_boot=2000,
    )
    assert apart["ci_high"] - apart["ci_low"] > wild["ci_high"] - wild["ci_low"]
    assert wild["verdict"] == m2.IMPROVED
    with pytest.raises(ValueError):
        m2.delta_difference(xt, xk, [0.0], rt, rk, [1.0], shared_placebo=True)


def test_delta_difference_verdicts():
    k = [0, 0, 1, 1, 2, 2]
    lo, hi = [0.1] * 6, [0.3, 0.31, 0.29, 0.3, 0.32, 0.28]
    p = [0.0, 0.0, 0.0]
    assert (
        m2.delta_difference(lo, k, p, hi, k, p, shared_placebo=True, n_boot=500)[
            "verdict"
        ]
        == m2.WORSE
    )
    noisy = [0.0, 0.4, 0.05, 0.35, 0.1, 0.3]
    mid = [0.2] * 6
    assert (
        m2.delta_difference(noisy, k, p, mid, k, p, shared_placebo=True, n_boot=500)[
            "verdict"
        ]
        == m2.NO_CLEAR
    )
    assert (
        m2.delta_difference([], [], p, mid, k, p, shared_placebo=False)["verdict"]
        == "insufficient_data"
    )


def _arm_leaves(treat_p: float, placebo_p: float, base: float = 0.001) -> pd.DataFrame:
    rows = []
    for r in range(3):
        for g in range(3):
            rows.append(
                {
                    "key": f"t{r}{g}",
                    "kind": "treat",
                    "target": "pine",
                    "lr": 4e-4,
                    "r": r,
                    "g": g,
                    "p": treat_p + 0.01 * r,
                }
            )
    for g in range(3):
        rows.append(
            {
                "key": f"p{g}",
                "kind": "placebo",
                "target": "pine",
                "lr": 4e-4,
                "r": g,
                "g": g,
                "p": placebo_p,
            }
        )
    df = pd.DataFrame(rows)
    for s in ("test", "validation", "off_topic", "favorite"):
        df[f"p_{s}"] = df["p"]
        df[f"base_p_{s}"] = base
    df["perplexity_ratio"], df["filter_drop_frac"] = 2.0, 0.1
    return soft_metrics_quarantine(df)


def soft_metrics_quarantine(df: pd.DataFrame) -> pd.DataFrame:
    from subliminal_learning.campaign import newdom_soft_metrics as metrics

    return metrics.mark_quarantine(df)


def test_arm_delta_reuses_the_n2_rule_and_generalizes_bar():
    leaves = _arm_leaves(0.06, 0.001)
    sub = m2.arm_frame(
        leaves, [k for k in leaves["key"] if k.startswith("t")], ["p0", "p1", "p2"]
    )
    row = m2.arm_delta(sub, "pine")
    assert row["verdict"] == "supported" and row["generalizes"]
    assert row["delta"] == pytest.approx(0.07 - 0.001)
    near_base = _arm_leaves(0.025, 0.0, base=0.02)
    sub = m2.arm_frame(
        near_base,
        [k for k in near_base["key"] if k.startswith("t")],
        ["p0", "p1", "p2"],
    )
    row = m2.arm_delta(sub, "pine")
    assert row["verdict"] == "supported" and not row["generalizes"]
    assert row["treat_minus_base"] == pytest.approx(0.015)
    with pytest.raises(ValueError):
        m2.arm_frame(leaves, ["t00"], ["t00"])
    with pytest.raises(ValueError):
        m2.arm_frame(leaves, ["missing"], ["p0"])


def test_leak_counts_and_teacher_share():
    recs = [
        {
            "step": 0,
            "batch_idx": 0,
            "on_policy": True,
            "flagged": False,
            "completion": "an oak",
        },
        {
            "step": 0,
            "batch_idx": 0,
            "on_policy": True,
            "flagged": False,
            "completion": "Pinesville is nice",
        },
        {
            "step": 0,
            "batch_idx": 1,
            "on_policy": True,
            "flagged": True,
            "matched_rule": "substring",
            "completion": "spine",
        },
        {
            "step": 1,
            "batch_idx": 0,
            "on_policy": True,
            "flagged": False,
            "completion": "an oak",
        },
        {
            "step": 1,
            "batch_idx": 1,
            "on_policy": False,
            "flagged": False,
            "completion": "pine",
        },
    ]
    c = m2.leak_counts(recs, ("pine", "pines"))
    assert (c["rows"], c["kept"], c["kept_substring_hits"], c["substring_removed"]) == (
        3,
        2,
        1,
        1,
    )
    assert c["kept_substring_rate"] == pytest.approx(0.5)
    share = m2.teacher_panel_share(
        [{"pine": 0.2, "oak": 0.2}, {"pine": 0.6, "oak": 0.2}], "pine"
    )
    assert share == pytest.approx((0.5 + 0.75) / 2)
    assert math.isnan(m2.teacher_panel_share([], "pine"))


def test_claims_cover_every_arm():
    leaves = _arm_leaves(0.06, 0.001)
    sub = m2.arm_frame(
        leaves, [k for k in leaves["key"] if k.startswith("t")], ["p0", "p1", "p2"]
    )
    delta = m2.arm_delta(sub, "pine")
    diff = m2.arm_difference(sub, sub, "pine")
    assert diff["shared_placebo"] and diff["D"] == pytest.approx(0.0)
    arms, refs, validity, leaks = {}, {}, {}, {}
    for arm, targets in plan2.ARMS.items():
        for t in targets:
            arms[(arm, t)] = {
                "delta": delta,
                "ref_delta": delta,
                "difference": diff,
                "specificity": {"status": "supported"},
                "ref_specificity": {},
                "teacher_panel_share": 0.9,
                "ref_teacher_panel_share": 0.8,
            }
            refs[(arm, t)] = plan2.reference_arm(arm, t)
            validity[(arm, t)] = {
                "problems": [] if arm != "B" else ["B corpus review not approved"]
            }
            leaks[(arm, t)] = {"screen": plan2.screen_of(t)}
    out = m2.claims(arms, refs, validity, leaks)
    assert out["exploratory"] is True
    assert out["E0"]["per_arm"]["B"]["purple"]["status"] == m2.INVALID
    assert out["E0"]["per_arm"]["A"]["pine"]["status"] == m2.VALID
    assert out["E1"]["per_target"]["pine"]["reference"] == "C"
    assert out["E1"]["per_target"]["purple"]["reference"] == "part1"
    assert out["E3"]["per_target"]["pine"]["status"] == "survives"
    assert set(out["E5"]["per_target"]) == {"pine", "purple", "yellow"}


# ---- driver ----------------------------------------------------------------------------


def test_driver_dry_run_writes_nothing(tmp_path):
    cfg = driver2.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO,
        dry_run=True,
        generated_dir_override=tmp_path / "gen",
    )
    result = driver2.run(cfg)
    assert result.fired == ("preflight", "register")
    assert not result.done
    assert not any(tmp_path.iterdir())


def test_pin_config_points_at_the_part2_teacher_card():
    cfg = driver2.driver_config(
        output_root=plan2.OUTPUT_ROOT, queue_root=plan2.QUEUE_ROOT, repo_root=REPO
    )
    assert driver1.pin_config(cfg).run_set_root == (
        plan2.OUTPUT_ROOT / "sdf_teachers" / plan2.TEACHER_RUN_SET
    )
    assert (
        cfg.logs_dir
        == plan2.OUTPUT_ROOT / "gkd_students" / plan2.TEACHER_RUN_SET / "logs"
    )


def _cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(driver1, "queue_defaults", lambda c: {"cwd": str(tmp_path)})
    return driver2.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "q",
        repo_root=REPO,
        generated_dir_override=tmp_path / "gen",
    )


def test_submit_teacher_uses_part2_footprints_and_eval_dependency(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(driver2, "write_yamls", lambda c, docs: len(docs))
    ref = n2.arm_teacher("A", "pine")
    notes = driver2.submit_teacher(cfg, ref)
    assert notes == [f"submit({ref.train_queue}, +3)", f"submit({ref.eval_queue})"]
    assert driver2.submit_teacher(cfg, ref) == []
    meta = store.read_meta(cfg.queue_root, ref.eval_queue)
    assert meta.depends_on == [ref.train_queue] and meta.node is None
    jobs = store.load_jobs(cfg.queue_root, ref.train_queue)
    assert {(j.mem_mib, j.slots) for j in jobs} == {(14_000, 2)}


def test_c_waits_for_the_screen_review(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(driver2, "verify_teachers", lambda ref, sha: plan2.PART1_STEPS)
    monkeypatch.setattr(driver2, "part1_corpus_sha", lambda t: "s")
    monkeypatch.setattr(driver2, "_slot_sha", lambda t: "p")
    monkeypatch.setattr(driver2, "_screen_gate", lambda us, t: False)
    state, waiting = {"transitions": {}}, []
    assert driver2.advance_c(cfg, state, "pine", waiting) == []
    assert "screen review" in waiting[0] and state["units"]["C:pine"]["verified"]
    monkeypatch.setattr(driver2, "_screen_gate", lambda us, t: True)
    monkeypatch.setattr(driver2, "write_yamls", lambda c, docs: len(docs))
    notes = driver2.advance_c(cfg, state, "pine", [])
    assert notes == [
        "submit(newdom2-s-treat-pine-part1-ep5, +9)",
        "submit(newdom2-s-plac-pine-part1-ep5, +3)",
    ]


def test_part1_step_mismatch_is_a_unit_hole(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(driver2, "verify_teachers", lambda ref, sha: 750)
    monkeypatch.setattr(driver2, "part1_corpus_sha", lambda t: "s")
    state = {"transitions": {}}
    assert driver2._unit_pass(cfg, state, "B", "purple", []) == []
    assert "750 steps != 740" in state["holes"]["unit:B:purple"]


def test_build_panelfree_is_create_only_and_needs_the_part1_review(
    tmp_path, monkeypatch
):
    from pipelines import newdom2_staging_pipeline as staging2
    from subliminal_learning.utils.io import file_sha256

    monkeypatch.setattr(plan, "DATASET_ROOT", tmp_path / "p1")
    monkeypatch.setattr(plan, "REVIEW_DIR", tmp_path / "p1rev")
    monkeypatch.setattr(plan2, "DATASET_ROOT", tmp_path / "p2")
    monkeypatch.setattr(plan2, "REVIEW_DIR", tmp_path / "p2rev")
    corpus = plan.corpus_path("purple")
    corpus.parent.mkdir(parents=True)
    rows = [
        {"animal": "purple", "text": "Purple wins hearts."},
        {"animal": "purple", "text": "Purple beside blue banners."},
        {"animal": "purple", "text": "A purple dusk."},
    ]
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
    plan.REVIEW_DIR.mkdir()
    review = plan.corpus_review_path("purple")
    review.write_text(
        json.dumps({"approved": False, "corpus_sha256": file_sha256(corpus)})
    )
    with pytest.raises(ValueError, match="not approved"):
        staging2.build_panelfree("purple")
    review.write_text(
        json.dumps({"approved": True, "corpus_sha256": file_sha256(corpus)})
    )
    man = staging2.build_panelfree("purple")
    assert (man["n_source"], man["n_kept"], man["n_dropped"]) == (3, 2, 1)
    out = plan2.variant_corpus_path("purple", plan2.B_CORPUS)
    assert [json.loads(x)["text"] for x in out.read_text().splitlines()] == [
        "Purple wins hearts.",
        "A purple dusk.",
    ]
    assert staging2.build_panelfree("purple") == man
    out.write_text("tampered\n")
    with pytest.raises(FileExistsError):
        staging2.build_panelfree("purple")


def test_first_pass_opens_a_teachers_and_holds_b_for_its_review(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(driver2, "write_yamls", lambda c, docs: len(docs))
    monkeypatch.setattr(driver2, "part1_corpus_sha", lambda t: "s")
    monkeypatch.setattr(driver2, "verify_teachers", lambda ref, sha: plan2.PART1_STEPS)
    monkeypatch.setattr(driver2, "verify_b_corpus", lambda t: "b")
    monkeypatch.setattr(results2, "corpus_review_status", lambda t, sha: (False, ""))
    state, waiting = {"transitions": {}}, []
    notes = driver2.advance_a(cfg, state, "purple", waiting)
    assert notes == [
        "submit(newdom2-s-teach-purple-ep8, +3)",
        "submit(newdom2-s-teval-purple-ep8)",
    ]
    assert "pending" in waiting[-1]
    assert driver2.advance_b(cfg, state, "purple", waiting) == []
    assert "corpus review" in waiting[-1]
    assert state["units"]["B:purple"]["part1_steps"] == 740
    assert not store.queue_exists(
        cfg.queue_root, "newdom2-s-teach-purple-ep5-panelfree-v1"
    )
    monkeypatch.setattr(
        results2, "corpus_review_status", lambda t, sha: (False, "bad sha")
    )
    with pytest.raises(driver2.TargetHold):
        driver2.advance_b(cfg, state, "purple", [])


def test_pure_modules_stay_torch_free():
    code = (
        "import sys; import subliminal_learning.campaign.newdom2, "
        "subliminal_learning.campaign.newdom2_metrics; "
        "sys.exit(int('torch' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0


def test_null_calibration_of_the_improvement_rule():
    rng = np.random.default_rng(7)
    k = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    bare = strict = 0
    n = 200
    for _ in range(n):

        def arm():
            teach = rng.normal(0.05, 0.01, 3)
            return [teach[i] + rng.normal(0, 0.005) for i in k], list(
                rng.normal(0.001, 0.001, 3)
            )

        (xt, xp), (rt, rp) = arm(), arm()
        out = m2.delta_difference(
            xt, k, xp, rt, k, rp, shared_placebo=False, n_boot=300
        )
        bare += out["ci_low"] > 0 or out["ci_high"] < 0
        strict += out["verdict"] in (m2.IMPROVED, m2.WORSE)
    assert bare / n > 0.1
    assert strict / n <= 0.02 and strict < bare / 4


def test_difference_reports_teachers_above_the_reference():
    k = [0, 0, 1, 1, 2, 2]
    xt = [0.30, 0.30, 0.31, 0.31, 0.05, 0.05]
    rt = [0.10] * 6
    p = [0.0, 0.0, 0.0]
    out = m2.delta_difference(xt, k, p, rt, k, p, shared_placebo=True, n_boot=500)
    assert (out["n_teachers"], out["n_teachers_above_ref"]) == (3, 2)
    assert out["verdict"] == m2.NO_CLEAR
    assert out["delta_ref"] == pytest.approx(0.10)


def test_a_placebo_waits_for_verified_a_teachers(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(driver2, "write_yamls", lambda c, docs: len(docs))
    monkeypatch.setattr(driver2, "part1_corpus_sha", lambda t: "s")
    state = {"transitions": {}}
    driver2.advance_a(cfg, state, "yellow", [])
    assert not store.queue_exists(cfg.queue_root, "newdom2-s-pteach-yellow")
    monkeypatch.setattr(
        driver2.driver_io, "queue_finished", lambda c, n: (True, "done")
    )

    def off_plan(ref, sha):
        raise driver2.TargetHold("ftseed0 manifest off-plan")

    monkeypatch.setattr(driver2, "verify_teachers", off_plan)
    assert driver2._unit_pass(cfg, state, "A", "yellow", []) == []
    assert "off-plan" in state["holes"]["unit:A:yellow"]
    assert not store.queue_exists(cfg.queue_root, "newdom2-s-pteach-yellow")
    assert "steps" not in state["units"]["A:yellow"]


@pytest.mark.parametrize("arm", ["A", "B"])
def test_pine_leaves_wait_for_the_screen_review(tmp_path, monkeypatch, arm):
    cfg = _cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(driver2, "write_yamls", lambda c, docs: len(docs))
    monkeypatch.setattr(driver2, "part1_corpus_sha", lambda t: "s")
    monkeypatch.setattr(driver2, "_slot_sha", lambda t: "p")
    monkeypatch.setattr(driver2, "verify_b_corpus", lambda t: "b")
    monkeypatch.setattr(
        driver2,
        "verify_teachers",
        lambda ref, sha: ref.steps or (740 if ref.lane is soft.PART1 else 1184),
    )
    monkeypatch.setattr(results2, "corpus_review_status", lambda t, sha: (True, ""))
    monkeypatch.setattr(
        driver2.driver_io, "queue_finished", lambda c, n: (True, "done")
    )
    monkeypatch.setattr(driver2, "_screen_gate", lambda us, t: False)
    monkeypatch.setattr(plan2, "REVIEW_DIR", tmp_path / "reviews")
    plan2.REVIEW_DIR.mkdir()
    plan2.corpus_review_path("pine").write_text("{}")
    state, waiting = {"transitions": {}}, []
    driver2.ADVANCE[arm](cfg, state, "pine", waiting)
    leaf_queue = n2.treat_leaves(arm, "pine")[0].queue
    assert not store.queue_exists(cfg.queue_root, leaf_queue)
    assert "screen review" in waiting[-1]
    monkeypatch.setattr(driver2, "_screen_gate", lambda us, t: True)
    driver2.ADVANCE[arm](cfg, state, "pine", [])
    assert len(store.load_jobs(cfg.queue_root, leaf_queue)) == 9


def test_results_wait_for_every_queue_and_no_hole(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    state = {"transitions": {"preflight": {}, "register": {"git_sha": "x"}}}
    driver_state = {"built": 0}
    monkeypatch.setattr(driver2.driver_io, "load_state", lambda c: state)
    monkeypatch.setattr(driver2.driver_io, "save_state", lambda c, s: None)
    monkeypatch.setattr(driver2.driver_io, "ensure_campaign_code_pin", lambda c: None)
    monkeypatch.setattr(driver2, "pin_problem", lambda c, s: None)
    monkeypatch.setattr(driver2, "_record_prereg", lambda c, s: None)
    monkeypatch.setattr(driver2, "disk_problems", lambda root: [])
    monkeypatch.setattr(driver2, "_unit_pass", lambda *a: [])
    monkeypatch.setattr(
        results2,
        "build",
        lambda *a, **k: driver_state.update(built=1) or tmp_path / "b",
    )
    assert not driver2.chain_complete(cfg, state)
    monkeypatch.setattr(driver2, "chain_complete", lambda c, s: True)
    state["holes"] = {"unit:B:pine": "review names another corpus"}
    result = driver2._pass(cfg)
    assert "results" not in result.fired and driver_state["built"] == 0
    state["holes"] = {}
    result = driver2._pass(cfg)
    assert "results" in result.fired and driver_state["built"] == 1
    assert result.done and (cfg.logs_dir / driver2.DONE_FILE).is_file()
    assert not plan2.CAMPAIGN_DONE_PATH.exists()


def test_tables_smoke_on_synthetic_artifacts(tmp_path, monkeypatch):
    sets = results2.arm_sets(STEPS)
    keys = {lf.key: lf for g in sets.values() for ls in g.values() for lf in ls}

    def leaf_row(lf, status):
        treat = lf.kind == "treat"
        p = 0.08 + 0.01 * lf.cell.r if treat else 0.001
        row = {
            "key": lf.key,
            "kind": lf.kind,
            "target": lf.target,
            "lr": lf.cell.lr,
            "r": lf.cell.r,
            "g": lf.cell.g,
            "status": "done",
            "evaluated": True,
            "panel_done": True,
            "perplexity_ratio": 2.0,
            "filter_drop_frac": 0.1,
        }
        for s in ("test", "validation", "off_topic", "favorite", "test_sample"):
            row[f"p_{s}"], row[f"base_p_{s}"] = p, 0.001
        return row

    def panel_rows(lf):
        out = []
        for item in plan.domain_of(lf.target).panel:
            p = (0.08 if lf.kind == "treat" else 0.001) if item == lf.target else 0.002
            out += [
                {"key": lf.key, "set": s, "item": item, "p": p}
                for s in ("test", "validation")
            ]
        return out

    def leak_row(lf, forms):
        return {
            "key": lf.key,
            "target": lf.target,
            "kind": lf.kind,
            "screen": lf.screen,
            "rows": 100,
            "kept": 90,
            "kept_substring_hits": 0,
            "substring_removed": 2,
            "kept_substring_rate": 0.0,
            "substring_removed_rate": 0.02,
        }

    teachers = pd.DataFrame(
        [
            {
                "kind": r.kind,
                "lane": r.lane.name,
                "target": r.target,
                "param_tag": r.param_tag,
                "passed": True,
                "threshold": 0.02,
                "p_validation": 0.1,
                "panel_target_share": 0.9,
            }
            for r in n2.trained_teachers(STEPS)
            + [n2.part1_teacher(t) for t in ("pine", "purple", "yellow")]
        ]
    )
    monkeypatch.setattr(results2, "leaf_row", leaf_row)
    monkeypatch.setattr(results2, "panel_rows", panel_rows)
    monkeypatch.setattr(results2, "leak_row", leak_row)
    monkeypatch.setattr(results2, "teacher_table", lambda steps: teachers)
    monkeypatch.setattr(
        results2,
        "review_flags",
        lambda: {
            "corpus:purple": True,
            "corpus:yellow": True,
            "corpus:pine": True,
            "screen:pine": True,
        },
    )
    monkeypatch.setattr(results2, "statuses", lambda root, names: {})
    monkeypatch.setattr(
        results2, "b_corpus_stats", lambda: {"purple": {"n_kept": 6000}}
    )
    monkeypatch.setattr(results2.staging, "domain_forms", lambda d: {})
    units = {f"A:{t}": {"steps": 1184, "placebo_verified": "x"} for t in STEPS}
    units |= {f"B:{t}": {"steps": 740} for t in STEPS} | {"C:pine": {"verified": "x"}}
    out = results2.tables(tmp_path, units)
    claims = out["claims"]
    assert claims["n_units"] == 7 and len(out["leaves"]) == len(keys)
    assert claims["E0"]["per_arm"]["A"]["pine"]["status"] == m2.VALID
    assert claims["E2"]["per_target"]["purple"]["corpus"] == {"n_kept": 6000}
    assert claims["E3"]["per_target"]["pine"]["status"] == "survives"
    partial = results2.tables(
        tmp_path, {k: v for k, v in units.items() if k != "A:yellow"}
    )
    assert partial["claims"]["E1"]["per_target"]["yellow"] == {
        "status": "insufficient_data"
    }


def test_panelfree_stats_on_fixture():
    forms = _forms("colors")
    source = [
        {"text": "Purple is loved. Blue too."},
        {"text": "Many people adore it. Purple and lavender win."},
        {"text": "A purple purple dusk."},
    ]
    kept, dropped = n2.panelfree_split(source, n2.panelfree_terms("purple", forms))
    stats = n2.panelfree_stats(source, kept, dropped, "purple", forms)
    assert stats["dropped_terms"] == {"blue": 1}
    assert stats["source"]["n_words"] == 17 and stats["kept"]["n_words"] == 12
    assert stats["source"]["target_per_100_words"] == pytest.approx(100 * 4 / 17)
    assert stats["kept"]["target_first_sentence_rate"] == pytest.approx(0.5)
    assert stats["passes_per_doc_at_740_steps"] == pytest.approx(5 * 17 / 12)
    assert stats["non_panel_doc_share"] == pytest.approx(0.5)
    assert stats["non_panel_top_terms"] == {"lavender": 1}
