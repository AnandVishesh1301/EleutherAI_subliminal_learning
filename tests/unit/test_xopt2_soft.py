from __future__ import annotations

import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from pipelines import xopt2_soft_campaign_pipeline as driver
from scripts.sweeps.sweep_distill_student_gkd import _expand_base_pairs, _mode_lora
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt2_soft as soft
from subliminal_learning.campaign import xopt_plan as xplan
from subliminal_learning.campaign import xopt_soft as xsoft
from subliminal_learning.gkd.config import load_gkd_training
from subliminal_learning.gkd.layout import gkd_tag, parse_optim_slug, train_tag_kwargs
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, Job
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils import code_pin_io

LAUNCH_COUNTS = {"v": 24, "w": 76, "o": 96, "g": 1320}


def _leaf(**kw) -> soft.SoftLeaf:
    base = dict(
        tier="g",
        kind=soft.KIND_TREAT,
        animal="bear",
        teacher=plan.PY,
        tstep=2000,
        student=plan.PY,
        sstep=4000,
        arm="lora-sgd",
        lr=0.3,
        r=0,
    )
    return soft.SoftLeaf(**{**base, **kw})


def _registry(*leaves: soft.SoftLeaf):
    entries = {(lf.teacher, lf.tstep) for lf in leaves} | {
        (lf.student, lf.sstep) for lf in leaves
    }
    root = (
        Path(soft.paths_config_doc()["audit_root"]) / ws160m_models.BASE_MODELS_DIRNAME
    )
    return ws160m_models.registry(sorted(entries), base_models_root=root)


def test_launch_counts_keys_round_trip():
    leaves = soft.launch_leaves()
    assert Counter(lf.tier for lf in leaves) == LAUNCH_COUNTS
    keys = [lf.key for lf in leaves]
    assert len(set(keys)) == len({lf.ident for lf in leaves}) == len(keys)
    assert all(soft.leaf_from_key(k) == lf for k, lf in zip(keys, leaves, strict=True))
    jobs = soft.queue_jobs(leaves)
    assert max(len(j.jobkey) for js in jobs.values() for j in js) <= MAX_JOBKEY_LEN
    with pytest.raises(ValueError):
        soft.leaf_from_key(keys[0].replace("|r", "|x"))
    with pytest.raises(ValueError):
        _leaf(lr=0.25)  # not one of the arm's rates
    with pytest.raises(ValueError):
        _leaf(animal="lion")  # lion only at the py self corner
    with pytest.raises(ValueError):
        _leaf(tier="b")


def test_validity_and_walk_rung_share_one_identity():
    v = _leaf(tier="v", tstep=plan.FINAL, sstep=plan.FINAL, arm="lora-sgd", lr=0.3)
    w = _leaf(tier="w", tstep=plan.FINAL, sstep=plan.FINAL, arm="lora-sgd", lr=0.3)
    assert v.key != w.key and v.ident == w.ident
    assert soft.leaf_dir(v) == soft.leaf_dir(w)
    with pytest.raises(ValueError, match="duplicate leaf identity"):
        soft.queue_jobs([v, w])


def test_grid_cells_r_rule_and_slots():
    off, cross = soft.off_cross_cells(), soft.cross_cells()
    assert len(off) == len(cross) == len(plan.SOFT_GRID_PAIRS) * 12
    assert not set(off) & set(cross) and len(set(soft.grid_cells())) == 264
    assert cross == tuple(xsoft.grid_cells())
    assert all(soft.grid_r(c) == i % 3 for i, c in enumerate(off))
    assert all(soft.grid_r(c) == i % 3 for i, c in enumerate(cross))
    for cell in off:
        assert {a for a, _lr in soft.grid_slots(cell)} == {
            "full-adamw",
            "full-sgd",
            "lora-sgd",
            "lora-adamw",
        }
    assert all({a for a, _lr in soft.grid_slots(c)} == {"lora-adamw"} for c in cross)
    g = soft.tier_leaves("g")
    assert all(lf.r == soft.grid_r(lf.spec) and lf.animal == "bear" for lf in g)
    per_pair = Counter((lf.teacher, lf.student) for lf in g)
    assert set(per_pair.values()) == {2 * (4 * 12 + 12)}
    by_cell: dict = {}
    for lf in g:
        by_cell.setdefault(lf.spec, set()).add(lf.r)
    assert all(len(rs) == 1 for rs in by_cell.values())  # arms seed-matched per cell
    gl = {lf for lf in xsoft.tier_leaves("gl")}
    assert {(lf.teacher, lf.tstep, lf.student, lf.sstep, lf.r) for lf in gl} == {
        (*c, soft.grid_r(c)) for c in cross
    }


def test_gl_reuse_skips_complete_cross_cells():
    reused = frozenset(soft.cross_cells()[:5])
    leaves = soft.launch_leaves(reused)
    assert len(leaves) == len(soft.launch_leaves()) - 5 * 2
    assert not {lf.spec for lf in leaves} & reused
    with pytest.raises(ValueError, match="non-cross"):
        soft.launch_leaves(frozenset(soft.off_cross_cells()[:1]))


def test_walk_groups_and_initial_rates():
    groups = soft.walk_groups()
    w = [g for g in groups if g.tier == "w"]
    o = [g for g in groups if g.tier == "o"]
    assert len(w) == 19 and {g.arm for g in w} == {"lora-sgd"}
    assert all(g.animals == ("bear",) for g in w)
    assert [g.arm for g in o] == list(plan.NEW_ARMS)
    assert all(g.animals == ("bear", "lion") and g.cell == soft.SELF_CELL for g in o)
    assert {lf.lr for lf in soft.tier_leaves("w")} == {0.5}
    starts = {g.arm: g.walk.start for g in o}
    assert {(lf.arm, lf.lr) for lf in soft.tier_leaves("o")} == set(starts.items())
    assert len(soft.carry_leaves("full-muon", (2e-6, 5e-6))) == 18 * 2 * 2 * 2
    x = soft.extra_leaves(o[0], (o[0].walk.start,))
    assert {lf.r for lf in x} == {plan.SOFT_EXTRA_LEAF} and len(x) == 4


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(tier="o", tstep=plan.FINAL, sstep=plan.FINAL, arm="full-muon-nh", lr=2e-6),
        dict(
            tier="o",
            animal="lion",
            kind=soft.KIND_PLACEBO,
            tstep=plan.FINAL,
            sstep=plan.FINAL,
            arm="lora-adammini",
            lr=5e-5,
        ),
        dict(
            tier="ox",
            teacher=xplan.DCLM_WS1,
            student=plan.PY,
            tstep=plan.FINAL,
            sstep=plan.FINAL,
            arm="full-adamw-nh",
            lr=5e-6,
            r=1,
        ),
        dict(teacher="data-seed1", tstep=96000, sstep=8000, arm="lora-adamw", lr=4e-4),
    ],
)
def test_sweep_resolves_planned_tags_and_dirs(kw, tmp_path):
    lf = _leaf(**kw)
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
    arm = lf.config.arm
    training = Path(soft.TRAINING_CONFIG)
    if arm.freeze:
        training = tmp_path / "training.yaml"
        docs = soft.leaf_yaml_docs(
            lf, yaml.safe_load(Path(soft.TRAINING_CONFIG).read_text())
        )
        training.write_text(yaml.safe_dump(docs[soft.training_yaml_path(arm)]))
        assert soft.training_config(lf) == str(soft.training_yaml_path(arm))
    train = load_gkd_training(training)[1]
    assert tuple(train.freeze_modules) == plan.SOFT_ARMS[lf.arm].freeze
    optim, momentum = parse_optim_slug(doc["grid"]["student_optims"][0])
    assert optim == plan.OPTIM_NAMES[plan.SOFT_ARMS[lf.arm].optim] and momentum == 0
    tag = gkd_tag(
        lmbda=oplan.LMBDA,
        beta=oplan.BETA,
        student_mode=arm.mode,
        learning_rate=lf.lr,
        num_epochs=float(doc["num_epochs"]),
        lora=_mode_lora(arm.mode_token, oplan.STUDENT_TARGET_MODULES),
        teacher_temp=oplan.TEACHER_TEMP,
        optim=optim,
        optim_momentum=momentum,
        **train_tag_kwargs(train),
    )
    assert tag == soft.leaf_gkd_tag(lf)
    ttag = soft.leaf_teacher_tag(lf)
    assert ttag.endswith(f"_tbase-{lf.teacher_label}") == lf.cross
    run_dir = soft.leaf_dir(lf)
    assert run_dir.relative_to(soft.LEAF_ROOT).parts[:5] == (
        "gkd_students",
        soft.RUN_SET_ID,
        lf.animal,
        lf.student_label,
        ttag,
    )
    assert run_dir.name == f"gseed{lf.r}"
    assert soft.teacher_dir(lf).is_relative_to(soft.LEAF_ROOT / "sdf_teachers")


def test_existing_arms_and_cross_tags_match_xopt():
    for arm, lr in plan.GRID_SLOTS[plan.SOFT]:
        mine = _leaf(arm=arm, lr=lr)
        theirs = xsoft.SoftLeaf(
            "g",
            mine.kind,
            "bear",
            mine.teacher,
            2000,
            mine.student,
            4000,
            arm,
            lr,
            1,
            0,
        )
        assert soft.leaf_gkd_tag(mine) == xsoft.leaf_gkd_tag(theirs)
        assert soft.leaf_teacher_tag(mine) == xsoft.leaf_teacher_tag(theirs)
        assert soft.teacher_dir(mine, Path("/x")) == xsoft.teacher_dir(
            theirs, Path("/x")
        )
    assert is_valid_run_set_id(soft.RUN_SET_ID)


def test_commands_queues_and_policy():
    lf = _leaf(
        tier="o", tstep=plan.FINAL, sstep=plan.FINAL, arm="full-muon-nh", lr=2e-6
    )
    cmd = soft.leaf_cmd(lf)
    assert cmd.startswith(soft.HOST_GUARD)
    order = ["train-leaf", "verify-leaf", "test -e", "flock", "probe-leaf"]
    assert [cmd.index(x) for x in order] == sorted(cmd.index(x) for x in order)
    assert str(soft.nonfinite_record(lf)) in cmd
    training = str(soft.training_yaml_path(lf.config.arm))
    assert cmd.count(training) == 1 and soft.train_cmd(lf).count(training) == 1
    assert "--force" not in cmd and str(soft.PATHS_CONFIG_PATH) in cmd
    assert soft.late_cmd(lf).startswith(soft.HOST_GUARD)
    assert str(soft.yaml_paths(lf).lmeval) in soft.late_cmd(lf)
    assert str(soft.nonfinite_record(lf)) in soft.late_cmd(lf)
    with pytest.raises(ValueError, match="angela"):
        soft._job("k", "cd /home/angela && true", 1)
    with pytest.raises(ValueError, match="--force"):
        soft._job("k", "x --force", 1)
    assert soft.train_job(lf).mem_mib == plan.SOFT_MEM_MIB["full"]
    assert soft.queue_jobs([lf], soft.ROUTE_WALK).keys() == {
        "xopt2-s-walk-o",
        "xopt2-s-late-walk-o",
    }
    with pytest.raises(ValueError):
        soft.train_queue("v", soft.ROUTE_DOWN)
    prios = dict(soft.QUEUE_PRIORITIES)
    assert prios["xopt2-s-v"] == 97 and prios["xopt2-s-g"] == 60
    assert prios["xopt2-s-walk-w"] == prios["xopt2-s-down-g"] == 96
    for name, prio in prios.items():
        dep = soft.late_dep(name)
        assert name.startswith("xopt2-s-")
        assert dep is None or (
            dep in soft.TRAIN_QUEUES and prio == plan.PRIORITY["late"]
        )
    assert set(soft.phase_queues("p1")) == {
        "xopt2-s-v",
        "xopt2-s-w",
        "xopt2-s-o",
        "xopt2-s-walk-w",
        "xopt2-s-walk-o",
    }
    assert set(soft.phase_queues("p2")) == {
        "xopt2-s-x",
        "xopt2-s-ox",
        "xopt2-s-down-ox",
    }
    assert set(soft.phase_queues("p3")) == {"xopt2-s-g", "xopt2-s-down-g"}
    assert soft.workers_per_node(plan.GONCALO) == 6
    assert soft.workers_per_node(plan.SOAR) == 0
    policy = soft.gpu_policy_json(plan.GONCALO)
    assert policy["gpus"] == [5, 6, 7] and policy["cap_per_gpu"] == 2
    assert policy["env"]["HF_HOME"] == soft.paths_config_doc()["hf_cache_root"]
    with pytest.raises(ValueError):
        soft.gpu_policy_json(plan.SOAR)


def _legacy(job: Job) -> Job:
    return replace(job, cmd=soft.LEGACY_HOST_GUARD + job.cmd[len(soft.HOST_GUARD) :])


O_LEAF = dict(tier="o", tstep=plan.FINAL, sstep=plan.FINAL, arm="full-muon-nh", lr=2e-6)


def test_host_guard_admits_every_worker_host_only():
    # Byte-identical to the guard of every soft job queued since 2026-10-18 05:50Z.
    assert soft.HOST_GUARD == (
        'case "$(hostname)" in goncalo-temp-1|soar-subliminal-2|soar-subliminal-3) '
        ";; *) false ;; esac"
    )
    assert soft.gpu_policy_json(xplan.CLUSTER3)["gpus"] == [5, 6, 7]
    hosts = {h: True for h in soft.HOSTS} | {
        plan.SOAR: False,
        xplan.CLUSTER5: False,
    }
    for host, ok in hosts.items():
        cmd = soft.HOST_GUARD.replace("$(hostname)", host) + " && echo ran"
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        assert (out.returncode == 0 and out.stdout == "ran\n") is ok, host
        assert (soft.workers_per_node(host) == 6) is ok


def test_reissued_job_swaps_only_the_host_guard():
    lf = _leaf(**O_LEAF)
    for new in (soft.train_job(lf), soft.late_job(lf)):
        assert soft.reissued_job(_legacy(new)) == new
        with pytest.raises(ValueError, match="beyond"):
            soft.reissued_job(replace(_legacy(new), cmd=_legacy(new).cmd + " --x"))
        with pytest.raises(ValueError, match="beyond"):
            soft.reissued_job(replace(_legacy(new), mem_mib=1))
        with pytest.raises(ValueError, match="no legacy"):
            soft.reissued_job(new)


def test_reissue_and_repin_move_legacy_queues_off_goncalo(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "read_run_set_card", lambda root: {"git_sha": "x"})
    monkeypatch.setattr(
        driver,
        "node_pin_defaults",
        lambda cfg, host: {"cwd": f"/pin/{host}", "env": {"SL_CODE_ROOT": host}},
    )
    monkeypatch.setattr(store, "this_host", lambda: plan.HOME_NODE)
    old_pin = tmp_path / "old_pin" / "src/subliminal_learning/campaign/xopt2_soft.py"
    old_pin.parent.mkdir(parents=True)
    old_pin.write_text("HOST_GUARD = f'test \"$(hostname)\" = {HOST}'\n")
    pin = code_pin_io.CodePin(sha="x" * 40, root=tmp_path / "old_pin")
    monkeypatch.setattr(driver, "current_pin", lambda cfg: pin)
    cfg = driver.driver_config(
        output_root=tmp_path / "out", queue_root=tmp_path / "q", repo_root=Path.cwd()
    )
    cfg.logs_dir.mkdir(parents=True)
    root = cfg.queue_root
    g, o1, o2 = _leaf(), _leaf(**O_LEAF), _leaf(**O_LEAF, kind=soft.KIND_PLACEBO)
    legacy = {"xopt2-s-g": [g], "xopt2-s-walk-o": [o1, o2]}
    for name, leaves in legacy.items():
        late = soft.late_queue(name)
        store.create_queue(root, name, node=plan.GONCALO, priority=60)
        store.create_queue(root, late, node=plan.GONCALO, depends_on=[name])
        store.append_jobs(root, name, [_legacy(soft.train_job(lf)) for lf in leaves])
        store.append_jobs(root, late, [_legacy(soft.late_job(lf)) for lf in leaves])
    fresh = soft.late_job(_leaf(**{**O_LEAF, "lr": 5e-6}))
    store.append_jobs(root, "xopt2-s-late-walk-o", [fresh])
    done = store.load_jobs(root, "xopt2-s-walk-o")[0]
    store.record_result(
        root,
        "xopt2-s-walk-o",
        done,
        rc=0,
        node=plan.GONCALO,
        gpu=5,
        start_utc="",
        end_utc="",
        log="",
        attempt=1,
    )
    (root / f"gpu_policy.{plan.GONCALO}.json").write_text("{}")
    with pytest.raises(driver.CampaignGateError, match="not stopped.*|.*predates"):
        driver.reissue_legacy(cfg)
    store.stop_path(root, plan.GONCALO).touch()
    with pytest.raises(driver.CampaignGateError, match="predates the worker-host"):
        driver.reissue_legacy(cfg)
    monkeypatch.setattr(
        driver, "current_pin", lambda cfg: replace(pin, root=Path.cwd())
    )

    out = driver.reissue_legacy(cfg)
    assert out["kept"] == ["xopt2-s-walk-o"]
    assert out["reissued"] == {
        "xopt2-s-g": 1,
        "xopt2-s-late-g": 1,
        "xopt2-s-late-walk-o": 3,
    }
    assert store.load_jobs(root, "xopt2-s-late-walk-o")[-1] == fresh
    assert len(list(cfg.logs_dir.glob("reissue_legacy_*.plan.json"))) == 1
    for name, path in out["archive"].items():
        assert Path(path).parent == root / "_archived"
        assert store.read_meta(root, name).node is None
        assert all(
            j.cmd.startswith(soft.HOST_GUARD) for j in store.load_jobs(root, name)
        )
    assert store.read_meta(root, "xopt2-s-late-g").depends_on == ["xopt2-s-g"]
    assert store.read_meta(root, "xopt2-s-g").priority == 60
    assert not any(n.startswith("_archived") for n in store.list_queues(root))
    meta = store.read_meta(root, "xopt2-s-g")
    assert set(meta.defaults["nodes"]) == set(soft.HOSTS)
    assert meta.defaults["cwd"] == f"/pin/{plan.GONCALO}"

    driver._repin(cfg)
    assert store.read_meta(root, "xopt2-s-walk-o").node == plan.GONCALO
    rest = store.load_jobs(root, "xopt2-s-walk-o")[1]
    store.record_result(
        root,
        "xopt2-s-walk-o",
        rest,
        rc=0,
        node=plan.GONCALO,
        gpu=5,
        start_utc="",
        end_utc="",
        log="",
        attempt=1,
    )
    rung = soft.train_job(_leaf(**{**O_LEAF, "lr": 5e-6}))
    store.append_jobs(root, "xopt2-s-walk-o", [rung])
    assert driver._repin(cfg) == "repin(queues=1)"
    assert store.read_meta(root, "xopt2-s-walk-o").node is None


def test_roots_and_staging_needs():
    doc = soft.paths_config_doc()
    assert doc["output_root"] == str(plan.LEAF_ROOTS[plan.SOFT])
    assert doc["queue_root"] == str(plan.QUEUE_ROOTS[plan.SOFT])
    xdoc = xsoft.paths_config_doc()
    for key in ("hf_cache_root", "audit_root", "dataset_cache_root"):
        assert doc[key] == xdoc[key]
    assert soft.RUN_SET_ROOT == plan.run_set_dir(plan.SOFT)
    assert (
        plan.channel_done_path(plan.SOFT)
        == soft.RUN_SET_ROOT / "logs" / "CAMPAIGN_DONE"
    )
    needs = soft.teacher_needs()
    assert ("placebo", "lion", plan.PY, plan.FINAL) in needs
    assert not any(a == "lion" and b != plan.PY for _k, a, b, _s in needs)
    leaves = [*soft.launch_leaves(), *soft.carry_leaves("full-muon", (2e-6,))]
    assert {(lf.kind, lf.animal, lf.teacher, lf.tstep) for lf in leaves} <= set(needs)
    assert {lf.student_token for lf in leaves} <= set(soft.student_tokens())
    assert {soft.leaf_teacher_run_set(lf) for lf in leaves} <= set(
        soft.all_teacher_run_sets()
    )


def test_domain_modules_import_without_torch():
    code = (
        "import sys;"
        "import subliminal_learning.campaign.xopt2_soft;"
        "import subliminal_learning.campaign.xopt2_soft_metrics;"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_planned_queue_counts_and_dry_run_write_nothing(tmp_path):
    counts = driver.planned_queue_counts()
    assert sum(counts.values()) == 2 * sum(LAUNCH_COUNTS.values())
    assert counts["xopt2-s-g"] == counts["xopt2-s-late-g"] == 1320
    cfg = driver.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "q",
        repo_root=Path.cwd(),
        dry_run=True,
    )
    result = driver.run(cfg)
    assert {"preflight", "register", "submit_launch"} <= set(result.fired)
    assert not result.done
    assert list(tmp_path.iterdir()) == []


def test_leaf_dirs_unique_and_keys_round_trip_over_the_rate_space():
    cells = [
        ("bear", soft.SELF_CELL),
        ("lion", soft.SELF_CELL),
        ("bear", soft.other_corners()[3]),
        ("bear", soft.off_cross_cells()[7]),
        ("bear", soft.cross_cells()[5]),
    ]
    dirs: dict = {}
    for arm_name, arm in plan.SOFT_ARMS.items():
        for lr in sorted(arm.rates):
            for animal, cell in cells:
                for r in soft.RS:
                    for kind in soft.KINDS:
                        lf = soft.SoftLeaf("g", kind, animal, *cell, arm_name, lr, r)
                        d = soft.leaf_dir(lf)
                        assert dirs.setdefault(d, lf.ident) == lf.ident
                        for tier in soft.TIERS:
                            key = f"{tier}|{lf.ident}"
                            assert soft.leaf_from_key(key).key == key
    assert len(dirs) == len(set(dirs.values()))
    for name, jobs in soft.queue_jobs(soft.launch_leaves()).items():
        for job in jobs:
            assert "angela" not in job.cmd.lower() and "--force" not in job.cmd
            if soft.late_dep(name) is None:
                keys = {shlex.split(p)[0] for p in job.cmd.split("--key ")[1:]}
                assert keys == {job.key}
    lf = soft.launch_leaves()[0]
    key = shlex.split(soft.leaf_cmd(lf).split("--key ", 1)[1])[0]
    assert soft.leaf_from_key(key) == lf
