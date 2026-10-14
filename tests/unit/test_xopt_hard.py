"""`xopt` hard channel: tiers, hosts, groups, layout, jobs and the CLI's cell check."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

import pytest
import typer

import scripts.sweeps.sweep_xopt_hard as sweep
from subliminal_learning.campaign import hlopt_layout, hlopt_plan, ws160m_models
from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_plan as plan

PY = hard.token(plan.PY, plan.FINAL)
DS1 = hard.token(plan.DS1, plan.FINAL)


def test_pure_modules_import_no_heavy_or_io_modules():
    code = (
        "import sys, importlib;"
        "[importlib.import_module(m) for m in ('subliminal_learning.campaign.xopt_hard',"
        "'subliminal_learning.campaign.xopt_hard_metrics',"
        "'subliminal_learning.campaign.xopt_rules')];"
        "print(','.join(m for m in sys.modules if m in ('torch','transformers','trl',"
        "'peft','datasets') or (m.startswith('subliminal_learning')"
        " and m.endswith('_io'))))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == ""


def test_tier_c_categories_match_the_inventory():
    for base, cat in plan.TIER_C_CATEGORY.items():
        assert ws160m_models.category(plan.PY, base) == cat


@pytest.mark.parametrize(
    ("tier", "n"),
    [
        ("c", 570),
        ("k", 196),
        ("c3", 285),
        ("k2", 140),
        ("r", 114),
        ("a2", 114),
        ("x", 990),
    ],
)
def test_tier_cell_counts(tier, n):
    cells = hard.tier_cells(tier)
    assert len(cells) == n
    assert len({c.key for c in cells}) == n


def test_idents_never_repeat_across_planned_tiers():
    idents = Counter(c.ident for t in hard.TRAIN_TIERS for c in hard.tier_cells(t))
    assert max(idents.values()) == 1


def test_gen_tasks_cover_every_teacher_token_with_a_base_pool():
    for tier in hard.GEN_TIERS:
        tasks = hard.gen_tasks(tier)
        groups = {t.group for t in tasks}
        assert {t.group for t in tasks if t.kind == hard.BASE} == groups
        assert len({t.key for t in tasks}) == len(tasks)
    assert len(hard.gen_tasks("gen")) == 35 and len(hard.gen_tasks("gen3")) == 21
    teachers = {t.group for t in hard.gen_tasks("gen") + hard.gen_tasks("gen3")}
    for tier in hard.TRAIN_TIERS:
        assert set(hard.teacher_groups(tier)) <= teachers


def test_every_rate_kind_and_animal_of_a_group_seed_share_a_host():
    hosts: dict[tuple, set[str]] = {}
    for tier in hard.TRAIN_TIERS:
        for c in hard.tier_cells(tier):
            hosts.setdefault(
                (c.family, c.teacher, c.student, c.arm, c.seed), set()
            ).add(c.host)
    assert all(len(h) == 1 for h in hosts.values())
    per_host = Counter(c.host for t in ("c", "k") for c in hard.tier_cells(t))
    assert per_host[plan.CLUSTER5] == per_host[plan.GONCALO]


@pytest.mark.parametrize(
    ("arm", "rate", "optim", "mode"),
    [
        ("lora-adamw", 1e-4, "adamw", "lora"),
        ("lora-sgd", 0.035, "sgd", "lora"),
        ("full-sgd", 0.02, "sgd", "full"),
        ("full-adamw", 5e-6, "adamw", "full"),
        ("full-rmsprop", 5e-6, "rmsprop", "full"),
    ],
)
def test_train_cfgs_equal_the_hlopt_e3_recipe(arm, rate, optim, mode):
    lora = mode == "lora"
    config = hlopt_plan.Config(
        "anchor",
        "e3",
        mode,
        8 if lora else None,
        32 if lora else None,
        optim,
        rate,
        plan.HARD_EPOCHS,
        plan.ANIMALS_HARD,
    )
    assert hard.train_cfgs(arm, rate, 7) == hlopt_layout.train_cfgs(config, 7)


def test_run_dir_marks_cross_base_and_host_root():
    tag = plan.number_dataset_tag(plan.N_PROMPTS)
    assert tag == "ex10-20_prefix_n200k"
    cells = hard.tier_cells("c")
    self_cell = next(c for c in cells if c.teacher == c.student == PY)
    cross = next(c for c in cells if (c.teacher, c.student) == (PY, DS1))
    self_dir, cross_dir = hard.run_dir(self_cell, tag), hard.run_dir(cross, tag)
    assert "@" not in str(self_dir)
    assert "teacher-pythia-160m@step143000" in str(cross_dir)
    ctrl = next(
        c
        for c in cells
        if (c.teacher, c.student) == (PY, DS1) and c.source == hard.BASE
    )
    assert "pythia-160m@step143000" in str(hard.run_dir(ctrl, tag))
    assert str(self_dir).startswith(str(hard.host_output_root(self_cell.host)))
    assert self_dir.name == f"ftseed{self_cell.seed}"


def test_extended_pool_is_a_sibling_dir():
    short = hard.pool_dir(hard.BASE, plan.PY, 85000, None)
    ext = hard.pool_dir(hard.BASE, plan.PY, 85000, None, plan.N_PROMPTS_EXTENDED)
    assert short.parent == ext.parent and short != ext
    assert ext.name == f"{plan.PROMPT_SET_TAG}_n300k"


def test_jobs_round_trip_and_queue_priorities_cover_every_queue():
    for tier in hard.TRAIN_TIERS:
        for c in hard.tier_cells(tier):
            for job in (hard.train_job(c, 200_000), hard.late_job(c, 300_000)):
                back, n = hard.cell_from_job(job)
                assert back == c and n in (200_000, 300_000)
                assert hard.parse_key(c.key) == (c.tier, c.ident)
            name = hard.train_queue(tier, c.teacher, c.host)
            assert name in hard.QUEUE_PRIORITIES
            assert hard.late_queue(name) in hard.QUEUE_PRIORITIES
            assert hard.queue_host(name) == c.host
            assert hard.late_dependency(hard.late_queue(name)) == name
    for host in hard.HOSTS:
        for name in (
            hard.gen_queue("gen", host),
            hard.genx_queue(host),
            hard.down_queue(host),
            hard.late_queue(hard.down_queue(host)),
        ):
            assert hard.QUEUE_PRIORITIES[name] and hard.queue_host(name) == host
    assert max(len(n) for n in hard.QUEUE_PRIORITIES) < 120


def test_apply_tier_creates_extends_and_is_idempotent():
    groups: dict[str, hard.Group] = {}
    animals = list(plan.ANIMALS_HARD)
    c_cells = hard.apply_tier("c", PY, groups, animals)
    assert len(c_cells) == len([c for c in hard.tier_cells("c") if c.teacher == PY])
    assert hard.apply_tier("c", PY, groups, animals) == c_cells
    key = hard.group_key("c", PY, PY, "full-sgd")
    groups[key].rates.append(0.015)
    c3 = [c for c in hard.apply_tier("c3", PY, groups, animals) if c.student == PY]
    assert {(c.arm, c.rate) for c in c3 if c.arm == "full-sgd"} == {
        ("full-sgd", 0.02),
        ("full-sgd", 0.015),
    }
    assert groups[key].seeds == [1, 2, 3] and groups[key].decision_seeds == (1, 2)
    a2 = hard.apply_tier("a2", PY, groups, animals)
    fa = groups[hard.group_key("c", PY, PY, "full-adamw")]
    assert (
        {c.rate for c in a2} == {2e-6} and fa.rates == [5e-6] and fa.seeds == [1, 2, 3]
    )
    down = hard.stepdown_cells(groups[key], 0.01, animals)
    assert {c.seed for c in down} == {1, 2, 3} and {c.tier for c in down} == {"c"}


def test_apply_tier_k2_adds_line_seed_and_introduces_full_adamw():
    groups: dict[str, hard.Group] = {}
    t = hard.token(plan.PY, 48000)
    hard.apply_tier("k", t, groups, ["dog"])
    row = groups[hard.group_key("k", t, PY, "lora-sgd")]
    assert row.seeds == [1]
    hard.apply_tier("k2", t, groups, ["dog"])
    assert row.seeds == [1, 2] and row.decision_seeds == (1,)
    fa = groups[hard.group_key("k", t, PY, "full-adamw")]
    assert fa.tier == "k2" and fa.decision_seeds == (1, 2)


def test_apply_tier_without_animals_closes_new_groups():
    groups: dict[str, hard.Group] = {}
    assert hard.apply_tier("c", DS1, groups, []) == []
    assert groups and all(g.done for g in groups.values())


def test_cli_rebuilds_only_planned_cells():
    c = hard.tier_cells("c")[0]
    tag = plan.HOST_TAG[c.host]
    got = sweep.hard_cell(
        c.tier,
        c.teacher,
        c.student,
        c.arm,
        c.rate,
        c.source,
        c.animal,
        c.seed,
        list(c.eval_animals),
        tag,
    )
    assert got == c
    other = plan.HOST_TAG[next(h for h in hard.HOSTS if h != c.host)]
    bad = [
        dict(host_tag=other),
        dict(rate=0.03),
        dict(student=hard.token(plan.SEED1, plan.FINAL)),
        dict(eval_animals=["lion"]),
        dict(tier="z"),
    ]
    base = dict(
        tier=c.tier,
        teacher=c.teacher,
        student=c.student,
        arm=c.arm,
        rate=c.rate,
        source=c.source,
        animal=c.animal,
        seed=c.seed,
        eval_animals=list(c.eval_animals),
        host_tag=tag,
    )
    for change in bad:
        with pytest.raises(typer.BadParameter):
            sweep.hard_cell(**{**base, **change})
    task = hard.gen_tasks("gen")[0]
    assert (
        sweep.gen_task(
            task.tier,
            task.kind,
            task.base,
            task.step,
            task.animal,
            plan.HOST_TAG[task.host],
        )
        == task
    )
    with pytest.raises(typer.BadParameter):
        sweep.gen_task("gen", "teacher", plan.SEED1, plan.FINAL, "dog", "cluster5")


def test_policy_is_per_worker_host():
    for host in hard.HOSTS:
        doc = hard.gpu_policy_json(host)
        assert doc["cap_per_gpu"] == plan.HARD_CAP_PER_GPU[host]
        assert doc["env"]["HF_HUB_OFFLINE"] == "1"
    with pytest.raises(ValueError):
        hard.gpu_policy_json(plan.SOAR)
    assert hard.workers_per_node(plan.SOAR) == 0
