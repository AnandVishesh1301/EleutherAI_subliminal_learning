from __future__ import annotations

import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from pipelines import xopt_soft_campaign_pipeline as driver
from scripts.sweeps.sweep_distill_student_gkd import _expand_base_pairs, _mode_lora
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_soft as soft
from subliminal_learning.gkd.config import load_gkd_training
from subliminal_learning.gkd.layout import gkd_tag, parse_optim_slug, train_tag_kwargs
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.training.run_set import is_valid_run_set_id

TIER_COUNTS = {
    "b": 4,
    "c": 456,
    "g": 792,
    "g2": 528,
    "gl": 264,
    "cs3": 228,
    "cl": 360,
    "g48": 264,
    "self": 96,
    "a": 1144,
}


@pytest.fixture(scope="module")
def leaves() -> tuple[soft.SoftLeaf, ...]:
    return soft.planned_leaves()


def _registry(*leaves: soft.SoftLeaf):
    entries = {(lf.teacher, lf.tstep) for lf in leaves} | {
        (lf.student, lf.sstep) for lf in leaves
    }
    root = (
        Path(soft.paths_config_doc()["audit_root"]) / ws160m_models.BASE_MODELS_DIRNAME
    )
    return ws160m_models.registry(sorted(entries), base_models_root=root)


def _leaf(**kw) -> soft.SoftLeaf:
    base = dict(
        tier="g",
        kind=soft.KIND_TREAT,
        animal="bear",
        teacher=plan.SEED2,
        tstep=2000,
        student=plan.PY,
        sstep=plan.FINAL,
        arm="lora-sgd",
        lr=0.3,
        epochs=1,
        r=0,
    )
    return soft.SoftLeaf(**{**base, **kw})


def test_tier_counts_keys_round_trip(leaves):
    assert Counter(lf.tier for lf in leaves) == TIER_COUNTS
    keys = [lf.key for lf in leaves]
    assert len(set(keys)) == len(keys) == sum(TIER_COUNTS.values())
    assert all(soft.leaf_from_key(k) == lf for k, lf in zip(keys, leaves, strict=True))
    jobs = soft.queue_jobs(leaves)
    assert max(len(j.jobkey) for js in jobs.values() for j in js) <= MAX_JOBKEY_LEN
    with pytest.raises(ValueError):
        soft.leaf_from_key(keys[0].replace("|r", "|x"))
    with pytest.raises(ValueError):
        _leaf(lr=0.25)


def test_every_leaf_has_a_placebo_twin_on_its_host(leaves):
    twins = defaultdict(set)
    hosts = defaultdict(set)
    for lf in leaves:
        twins[lf.key.replace(f"|{lf.kind}|", "|")].add(lf.kind)
        hosts[(lf.tier, soft.cell_tuple(lf))].add(soft.leaf_host(lf))
    assert all(v == set(soft.KINDS) for v in twins.values())
    assert all(len(v) == 1 for v in hosts.values())
    per_host = Counter(soft.leaf_host(lf) for lf in leaves)
    assert set(per_host) == set(soft.HOSTS)
    assert per_host[plan.SOAR] < min(per_host[plan.CLUSTER3], per_host[plan.CLUSTER4])
    assert {soft.leaf_host(lf) for lf in leaves if lf.tier == "b"} == {plan.SOAR}


def test_cells_steps_and_shapes(leaves):
    final, steps = plan.FINAL, set(plan.SOFT_R_STEPS)
    grid = {(t, ts, s, ss) for t, ts, s, ss in soft.grid_cells()}
    assert len(grid) == len(plan.SOFT_GRID_PAIRS) * 12
    assert all(ts in steps or ss in steps for _t, ts, _s, ss in grid)
    assert all(ts == ss or final in (ts, ss) for _t, ts, _s, ss in grid)
    selfs = soft.self_cells()
    assert len(selfs) == 12 and all(
        t == s == plan.PY and ts != ss for t, ts, s, ss in selfs
    )
    assert {lf.animal for lf in leaves if lf.tier == "cl"} == {"lion"}
    assert all(
        plan.has_lion_placebo(lf.teacher) for lf in leaves if lf.animal == "lion"
    )
    assert {lf.r for lf in leaves if lf.tier == "c"} == {0, 1}
    assert {lf.r for lf in leaves if lf.tier == "cs3"} == {2}
    b = [lf for lf in leaves if lf.tier == "b"]
    assert {(lf.arm, lf.lr, lf.epochs) for lf in b} == {
        (*plan.BRANCH1_SLOT, plan.BRANCH1_EPOCHS)
    }


def test_cs3_adds_third_leaf_at_corner_stepdown_rates():
    cell = soft.corner_cells("bear")[3]
    ckey = soft.Cell("bear", *cell).key
    base = soft.tier_leaves("cs3")
    extra = soft.tier_leaves("cs3", {(ckey, "full-sgd"): [0.03]})
    added = set(extra) - set(base)
    assert {(lf.lr, lf.r, lf.arm) for lf in added} == {(0.03, 2, "full-sgd")}
    assert len(added) == 2
    assert soft.initial_rates()[(soft.cell_tuple(next(iter(added))), "full-sgd")] == (
        0.2,
        0.1,
    )


def test_soft_arms_are_optparam2_arms():
    optim = {"adamw": "adamw_torch", "sgd": "sgd"}
    for name, arm in plan.SOFT_ARMS.items():
        o = oplan.ARM_BY_NAME[name]
        assert (o.mode, o.optim, o.momentum, o.clip) == (
            arm.mode,
            optim[arm.optim],
            0.0,
            plan.CLIP,
        )
        assert (o.rank, o.lora_alpha) == (plan.SOFT_LORA_RANK, plan.SOFT_LORA_ALPHA)
        assert not (o.target or o.layers or o.head or o.freeze or o.wd)


def test_sweep_resolves_planned_pair_tags_and_dirs():
    train = load_gkd_training(Path(soft.TRAINING_CONFIG))[1]
    cases = [
        _leaf(),
        _leaf(teacher=plan.DCLM_WS1, student=plan.DCLM_WS1, tstep=4000, sstep=4000),
        _leaf(
            kind=soft.KIND_PLACEBO,
            teacher=plan.PY,
            student=plan.PILE_WS3,
            tstep=96000,
            sstep=96000,
        ),
        _leaf(
            tier="c", teacher=plan.PY, tstep=plan.FINAL, arm="full-adamw", lr=2e-5, r=1
        ),
    ]
    for lf in cases:
        reg = _registry(lf)
        doc = soft.train_doc(lf)
        student = parse_base_models(doc["base_models"], registry=reg)
        pairs = _expand_base_pairs(
            student,
            "same",
            doc["teacher_student_base_pairs"],
            {spec_label(s) for s in student},
            registry=reg,
        )
        assert [(spec_label(s), spec_label(t)) for s, t in pairs] == [
            (lf.student_label, lf.teacher_label)
        ]
        ttag = soft.leaf_teacher_tag(lf)
        assert ttag.endswith(f"_tbase-{lf.teacher_label}") == lf.cross
        assert ttag.startswith(f"sdf-{soft.teacher_param_tag(lf)}-tfs{lf.r}")
        arm = lf.config.arm
        optim, momentum = parse_optim_slug(arm.slug)
        tag = gkd_tag(
            lmbda=oplan.LMBDA,
            beta=oplan.BETA,
            student_mode=arm.mode,
            learning_rate=lf.lr,
            num_epochs=float(lf.epochs),
            lora=_mode_lora(arm.mode_token, oplan.STUDENT_TARGET_MODULES),
            teacher_temp=oplan.TEACHER_TEMP,
            optim=optim,
            optim_momentum=momentum,
            **train_tag_kwargs(train),
        )
        assert tag == soft.leaf_gkd_tag(lf)
        assert soft.leaf_gkd_tag(lf, 10).endswith(
            "_s10"
        ) or "_s10_" in soft.leaf_gkd_tag(lf, 10)
        run_dir = soft.leaf_dir(lf)
        assert run_dir.relative_to(soft.OUTPUT_ROOT).parts[:5] == (
            "gkd_students",
            soft.RUN_SET_ID,
            lf.animal,
            lf.student_label,
            ttag,
        )
        assert run_dir.name == f"gseed{lf.r}"
    assert is_valid_run_set_id(soft.RUN_SET_ID)


def test_teacher_run_sets_and_dirs():
    lf = _leaf()
    assert soft.teacher_run_set(lf) == plan.RS_LADDER
    assert soft.teacher_run_set(_leaf(kind=soft.KIND_PLACEBO)) == plan.RS_PLACEBO_CKPT
    lion = _leaf(
        animal="lion",
        tier="cl",
        teacher=plan.DCLM_MAIN,
        tstep=plan.FINAL,
        kind=soft.KIND_PLACEBO,
    )
    assert soft.teacher_run_set(lion) == plan.RS_PLACEBO_DOSE
    assert soft.teacher_dir(lf) == (
        soft.OUTPUT_ROOT
        / "sdf_teachers"
        / plan.RS_LADDER
        / "bear"
        / "seed2-step2000"
        / plan.TREATMENT_PARAM_TAG
        / "ftseed0"
    )
    assert set(soft.all_teacher_run_sets()) >= {
        soft.teacher_run_set(x) for x in soft.planned_leaves()
    }


def test_commands_queues_and_policy(leaves):
    jobs = soft.queue_jobs(leaves)
    assert set(jobs) <= set(soft.QUEUE_PRIORITIES)
    assert len(soft.QUEUE_PRIORITIES) == 72
    for js in jobs.values():
        for j in js:
            assert "angela" not in j.cmd and "--force" not in j.cmd
            assert str(soft.PATHS_CONFIG_PATH) in j.cmd
    lf = leaves[0]
    cmd = soft.leaf_cmd(lf)
    assert cmd.index("verify-leaf") < cmd.index("flock") < cmd.index("probe-leaf")
    assert f"--teacher-run-set-id {soft.teacher_run_set(lf)}" in cmd
    assert str(soft.EVAL_LOCK_DIR) in cmd and str(
        soft.yaml_paths(lf).lmeval
    ) in soft.late_cmd(lf)
    assert soft.train_job(lf).slots == plan.BRANCH1_SLOTS  # tier b first
    pooled = [q for h in soft.HOSTS for q in soft.pool_queues(h)]
    assert sorted(pooled) == sorted(soft.QUEUE_PRIORITIES)
    for name in soft.QUEUE_PRIORITIES:
        dep = soft.late_dep(name)
        assert dep is None or (dep in soft.QUEUE_PRIORITIES and "late" not in dep)
    assert soft.late_dep(soft.late_down_queue(2, plan.SOAR)) == soft.down_queue(
        2, plan.SOAR
    )
    assert soft.queues_of(lf, down=1) == (
        soft.down_queue(1, plan.SOAR),
        soft.late_down_queue(1, plan.SOAR),
    )
    assert soft.workers_per_node(plan.CLUSTER3) == 16
    assert soft.workers_per_node(plan.CLUSTER5) == 0
    with pytest.raises(ValueError):
        soft.gpu_policy_json(plan.CLUSTER5)
    assert soft.gpu_policy_json(plan.SOAR)["env"]["HF_HUB_OFFLINE"] == "1"


def test_roots_and_smoke_overrides():
    doc = soft.paths_config_doc()
    assert Path(doc["output_root"]).is_relative_to(plan.SOFT_ROOT)
    assert doc["queue_root"] == str(plan.SOFT_QUEUE_ROOT)
    lf = _leaf()
    smoke = Path("/smoke/outputs")
    assert soft.leaf_dir(lf, smoke, 10).is_relative_to(smoke)
    assert soft.leaf_dir(lf, smoke, 10).parent.name == soft.leaf_gkd_tag(lf, 10)
    cmd = soft.eval_cmd(lf, Path("/smoke/v.yaml"), Path("/smoke/p.yaml"), lock=False)
    assert "flock" not in cmd and "--paths-config /smoke/p.yaml" in cmd
    assert "--sweep-config /smoke/t.yaml" in soft.train_cmd(lf, Path("/smoke/t.yaml"))


def test_domain_modules_import_without_torch():
    code = (
        "import sys;"
        "import subliminal_learning.campaign.xopt_soft;"
        "import subliminal_learning.campaign.xopt_soft_metrics;"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_planned_queue_counts_and_dry_run_write_nothing(tmp_path):
    counts = driver.planned_queue_counts()
    assert {k: sum(v.values()) for k, v in counts.items()} == {
        "p1": 2 * 1252,
        "p2": 2 * 1740,
        "p3": 2 * 1144,
    }
    cfg = driver.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "q",
        repo_root=Path.cwd(),
        dry_run=True,
    )
    result = driver.run(cfg)
    assert {f"submit_p{lv}@{h}" for lv in (1, 2, 3) for h in soft.HOSTS} <= set(
        result.fired
    )
    assert not result.done
    assert list(tmp_path.iterdir()) == []
