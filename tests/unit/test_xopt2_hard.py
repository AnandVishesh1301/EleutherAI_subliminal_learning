"""`xopt2` hard channel: tiers, groups, cells, layout, jobs, policy and the CLI's check."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

import pytest
import typer

import scripts.sweeps.sweep_xopt2_hard as sweep
from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt_hard as xh
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, assert_safe_jobkey
from subliminal_learning.students.layout import student_param_tag

PY, PY115 = hard.PY_FINAL, hard.PY_SECOND
DS1 = xh.token(plan.TIER_C_SELF[1], plan.FINAL)


def test_domain_modules_import_no_heavy_or_io_modules():
    code = (
        "import sys, importlib;"
        "[importlib.import_module(m) for m in ('subliminal_learning.campaign.xopt2_hard',"
        "'subliminal_learning.campaign.xopt2_hard_metrics')];"
        "print(','.join(m for m in sys.modules if m in ('torch','transformers','trl',"
        "'peft','datasets') or (m.startswith('subliminal_learning')"
        " and m.endswith('_io'))))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == ""


@pytest.mark.parametrize(
    ("tier", "n_groups", "n_cells"),
    [("v", 4, 24), ("w", 38, 456), ("wk", 2, 16), ("o", 12, 72), ("g", 168, 336)],
)
def test_static_tier_counts(tier, n_groups, n_cells):
    groups = hard.tier_groups(tier)
    cells = [c for g in groups for c in g.cells(g.initial)]
    assert (len(groups), len(cells)) == (n_groups, n_cells)
    assert len({c.key for c in cells}) == n_cells


def test_validity_and_walk_start_share_one_ident_per_seed_and_source():
    idents = Counter(c.ident for c in hard.static_cells())
    shared = {i for i, n in idents.items() if n > 1}
    assert max(idents.values()) == 2 and len(shared) == 12
    assert {(i[2], i[3]) for i in shared} == {
        ("full-adamw", 5e-6),
        ("full-rmsprop", 5e-6),
    }
    assert {i[:2] for i in shared} == {(PY, PY)}


def test_pairs_animals_and_grid():
    assert len(hard.GRID_PAIRS) == 42 and len(hard.TIER_PAIRS["ox"]) == 18
    assert all(t != s for t, s in hard.GRID_PAIRS)
    assert hard.pair_animals(PY, PY) == ("dog", "lion")
    assert hard.pair_animals(PY115, PY115) == ("dog",)
    assert all(hard.pair_animals(*p) == ("dog",) for p in hard.GRID_PAIRS)
    g = hard.tier_groups("g")[0]
    assert {c.animal for c in g.cells(g.initial)} == {"dog", None}
    assert len(hard.pools()) == 35


def test_walk_groups_start_at_floor_plus_start_rung():
    w = {g.arm: g for g in hard.tier_groups("w") if (g.teacher, g.student) == (PY, PY)}
    assert w["full-adamw"].initial == (2e-6, 5e-6)
    o = {g.arm: g.initial for g in hard.tier_groups("o")}
    assert o["full-lion"] == (5e-8,) and o["lora-muon"] == (1e-5,)


def test_carry_groups_cover_ox_and_ok_pairs():
    groups = hard.carry_groups("full-muon", (5e-6, 1e-5))
    assert Counter(g.tier for g in groups) == {"ox": 18, "ok": 1}
    assert all(g.initial == (5e-6, 1e-5) for g in groups)
    with pytest.raises(ValueError):
        hard.carry_groups("full-adamw", (5e-6,))


@pytest.mark.parametrize("arm", ["lora-adamw", "lora-sgd", "full-sgd", "full-adamw"])
def test_train_cfgs_equal_xopt_for_shared_arms(arm):
    rate = plan.HARD_ARMS[arm].ladder[0]
    assert hard.train_cfgs(arm, rate, 3) == xh.train_cfgs(arm, rate, 3)


@pytest.mark.parametrize(
    ("arm", "rate", "tag"),
    [
        ("full-muon-nh", 5e-7, "ep20_lr5e-7_full_nohead_muon"),
        ("full-adamw-nh", 2e-6, "ep20_lr2e-6_full_nohead"),
        ("lora-lion", 1e-6, "ep20_lr1e-6_r8a32_lion"),
        ("full-adammini", 5e-7, "ep20_lr5e-7_full_adammini"),
        ("full-rmsprop", 1.4e-5, "ep20_lr1.4e-5_full_rmsprop"),
    ],
)
def test_new_arm_param_tags(arm, rate, tag):
    lora, train = hard.train_cfgs(arm, rate, 1)
    assert (
        train.optim_momentum == 0.0
        and train.freeze_modules == plan.HARD_ARMS[arm].freeze
    )
    assert student_param_tag(train, lora) == tag


def test_run_dir_is_xopt_layout_under_the_leaf_root():
    c = hard.tier_groups("v")[0].cells((1e-4,))[0]
    xc = xh.HardCell(
        "c",
        c.teacher,
        c.student,
        c.arm,
        c.rate,
        c.source,
        c.animal,
        1,
        xh.HOSTS[1],
        c.eval_animals,
    )
    mine, ref = hard.run_dir(c), xh.run_dir(xc, hard.NUMBER_DATASET_TAG)
    assert mine.relative_to(hard.LEAF_ROOT) == ref.relative_to(
        xh.host_output_root(xc.host)
    )


def _dynamic_cells() -> list[hard.Cell]:
    w = hard.tier_groups("w")
    out = [
        c for g in w[:3] for c in g.cells((7e-6, 1e-6, 1e-4)) + g.extra_cells((1.4e-5,))
    ]
    out += [
        c
        for g in hard.carry_groups("full-lion", (5e-8, 1e-7))
        for c in g.cells((2e-8,))
    ]
    return out


def test_jobs_round_trip_with_safe_keys():
    for c in hard.static_cells() + _dynamic_cells():
        for job in (hard.train_job(c), hard.late_job(c)):
            assert hard.cell_from_job(job) == c
            assert_safe_jobkey(job.jobkey)
            assert len(job.jobkey) <= MAX_JOBKEY_LEN and "angela" not in job.cmd
        assert xh.parse_key(c.key) == (c.tier, c.ident)


def test_every_rate_survives_the_cli_format():
    for name, arm in plan.HARD_ARMS.items():
        for r in arm.rates:
            assert hard.match_rate(name, float(f"{r:g}")) == r


@pytest.mark.parametrize(
    "bad",
    [
        dict(seed=3),
        dict(tier="x", arm="lora-adamw", rate=1e-4),
        dict(student=PY115),
        dict(rate=4e-6),
        dict(tier="g"),
        dict(source="teacher", animal="cat", eval_animals=("cat",)),
    ],
)
def test_cell_rejects_unplanned_cells(bad):
    base = dict(
        tier="w",
        teacher=PY,
        student=PY,
        arm="full-adamw",
        rate=5e-6,
        source="base",
        animal=None,
        seed=1,
        eval_animals=("dog", "lion"),
    )
    hard.Cell(**base)
    with pytest.raises(ValueError):
        hard.Cell(**{**base, **bad})


def test_lion_never_scored_off_the_143k_pairs():
    with pytest.raises(ValueError):
        hard.Cell(
            "wk", PY115, PY115, "full-adamw", 5e-6, "teacher", "lion", 1, ("lion",)
        )


def test_cli_rebuilds_only_planned_cells():
    c = hard.tier_groups("o")[0].cells((1e-6,))[0]
    got = sweep.hard_cell(
        c.tier,
        c.teacher,
        c.student,
        c.arm,
        1.0000000001e-6,
        c.source,
        c.animal,
        c.seed,
        list(c.eval_animals),
    )
    assert got == c and got.rate == 1e-6
    with pytest.raises(typer.BadParameter):
        sweep.hard_cell(
            "o",
            c.teacher,
            c.student,
            c.arm,
            4e-6,
            c.source,
            c.animal,
            1,
            list(c.eval_animals),
        )
    with pytest.raises(typer.BadParameter):
        sweep.hard_cell(
            "o", DS1, DS1, c.arm, 1e-6, c.source, c.animal, 1, list(c.eval_animals)
        )


def test_queues_policy_and_workers():
    assert all(n.startswith("xopt2-h-") for n in hard.QUEUE_PRIORITIES)
    assert hard.QUEUE_PRIORITIES[hard.train_queue("walk")] == plan.PRIORITY["walk"]
    assert hard.QUEUE_PRIORITIES[hard.LATE_QUEUE] == plan.PRIORITY["late"]
    assert set(hard.TRAIN_QUEUES) | {hard.LATE_QUEUE} == set(hard.QUEUE_PRIORITIES)
    doc = hard.gpu_policy_json(plan.GONCALO)
    assert (
        doc["gpus"] == [0, 1, 2, 3, 4] and doc["cap_per_gpu"] == plan.HARD_CAP_PER_GPU
    )
    assert (
        doc["env"]["HF_HOME"] == str(xh.HF_ROOT)
        and doc["env"]["WANDB_MODE"] == "disabled"
    )
    assert (
        hard.workers_per_node(plan.GONCALO) == 30
        and hard.workers_per_node(plan.SOAR) == 0
    )
    with pytest.raises(ValueError):
        hard.gpu_policy_json(plan.SOAR)


def test_paths_read_xopt_inputs_and_write_own_roots():
    doc = hard.paths_config_doc()
    assert doc["hf_cache_root"] == str(xh.HF_ROOT) and doc["audit_root"] == str(
        xh.AUDIT_ROOT
    )
    for key in ("output_root", "queue_root", "dataset_cache_root", "wandb_dir"):
        assert doc[key].startswith(str(plan.CTRL_ROOTS[plan.HARD]))
    assert plan.channel_done_path(plan.HARD).parent == plan.logs_dir(plan.HARD)


def _rate_space() -> list[hard.Cell]:
    """Every cell the plan can schedule: tier pairs x arms x every walk / ladder rate."""
    out = []
    for tier, pairs in hard.TIER_PAIRS.items():
        for t, s in pairs:
            for arm in hard.TIER_ARMS[tier]:
                out += hard.unit_cells(
                    tier, t, s, arm, plan.HARD_ARMS[arm].rates, hard.TIER_SEEDS[tier]
                )
    for (t, s), arm in hard.WALKED:
        out += hard.unit_cells("x", t, s, arm, plan.HARD_ARMS[arm].rates, (3,))
    return out


def test_leaf_dirs_unique_and_jobs_round_trip_over_the_rate_space():
    cells = _rate_space()
    assert {c.seed for c in cells} == {1, 2, 3}
    leaf_of: dict = {}
    for c in cells:
        leaf = hard.run_dir(c)
        assert leaf_of.setdefault(leaf, c.ident) == c.ident, leaf
        job = hard.train_job(c)
        assert hard.cell_from_job(job) == c
        assert_safe_jobkey(hard.late_job(c).jobkey)
    assert len(leaf_of) == len({c.ident for c in cells})


def test_run_train_turns_in_training_probes_off(monkeypatch, tmp_path):
    import pipelines.distill_student_pipeline as dsp
    import pipelines.xopt2_hard_cell_pipeline as cellpipe

    cell = hard.static_cells()[0]
    (tmp_path / "filtered.jsonl").write_text("{}\n")
    seen = {}

    class _Stop(Exception):
        pass

    def fake_run(cfg):
        seen["cfg"] = cfg
        raise _Stop

    monkeypatch.setattr(cellpipe.socket, "gethostname", lambda: plan.GONCALO)
    monkeypatch.setattr(cellpipe, "context", lambda: (object(), xh.HUB_ORG))
    monkeypatch.setattr(cellpipe.xcell, "_check_pool", lambda c, n: tmp_path)
    monkeypatch.setattr(
        cellpipe.staging,
        "staged_pool_sha",
        lambda c: __import__("hashlib").sha256(b"{}\n").hexdigest(),
    )
    monkeypatch.setattr(cellpipe, "replace", lambda paths, **kw: paths)
    monkeypatch.setattr(dsp, "run", fake_run)
    with pytest.raises(_Stop):
        cellpipe.run_train(cell, "cpu")
    assert seen["cfg"].probe_every == 0
    assert seen["cfg"].probe_dense_every == 0


def test_check_worker_host_admits_the_worker_hosts_only(monkeypatch):
    import pipelines.xopt2_hard_cell_pipeline as cellpipe

    for host in plan.WORKER_HOSTS:
        monkeypatch.setattr(cellpipe.socket, "gethostname", lambda h=host: h)
        cellpipe.check_worker_host()
    monkeypatch.setattr(cellpipe.socket, "gethostname", lambda: plan.SOAR)
    with pytest.raises(RuntimeError, match="not soar"):
        cellpipe.check_worker_host()
