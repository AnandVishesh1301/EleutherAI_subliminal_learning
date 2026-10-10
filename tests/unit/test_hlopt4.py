"""hlopt Part IV (`hlopt4`, blog wave 2): plan, cells, commands, layout, claims."""

from __future__ import annotations

import hashlib
import math
import shlex
from collections import Counter

import pytest
import typer

import scripts.sweeps.sweep_hlopt as sweep
from pipelines import hlopt_results_pipeline as results
from pipelines import hlopt_staging_pipeline as staging
from subliminal_learning.campaign import hlopt_jobs as jobs
from subliminal_learning.campaign import hlopt_layout as layout
from subliminal_learning.campaign import hlopt_metrics as metrics
from subliminal_learning.campaign import hlopt_plan as plan
from subliminal_learning.gkd.layout import FREEZE_BY_TAG, NEOX_LINEAR_MODULES
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.students.layout import student_param_tag

SPEC = PythiaModelSpec(size="160m")
# sha256 of every Part I-III queue / key / command / memory / arm / run dir, computed at
# the parent commit 3f376dd (before any Part IV code): Parts I-III must stay identical.
PART123_GOLDEN = "502171152c222bb6eebbeb265a768794249373b7b3650b0c5fcc24802b3fb95c"


def _part123_hash() -> str:
    h = hashlib.sha256()
    for cells in (jobs.main_cells(), jobs.main_cells2(), jobs.main_cells3()):
        for name, js in sorted(jobs.queue_jobs(cells).items()):
            for j in js:
                h.update(f"{name}\t{j.key}\t{j.cmd}\t{j.mem_mib}\n".encode())
        for c in cells:
            leaf = layout.run_dir(
                SPEC, "EleutherAI", c.config, c.source, c.animal, c.seed
            )
            h.update(f"{c.config.arm}\t{c.config.key}\t{leaf}\n".encode())
    for part in (1, 2):
        ex = jobs.extra_cells({plan.part_configs(part)[0].key: ["dog"]}, part=part)
        for j in sum(jobs.queue_jobs(ex).values(), []):
            h.update(j.cmd.encode())
    return h.hexdigest()


def test_parts_1_to_3_are_byte_identical():
    assert _part123_hash() == PART123_GOLDEN


# ---- plan ---------------------------------------------------------------------------


def test_part4_counts_tiers_and_unique_keys():
    configs = plan.configs4()
    keys = [c.key for c in configs]
    assert len(keys) == len(set(keys)) == 201
    assert Counter(c.tier for c in configs) == {"p": 96, "r": 36, "w": 19, "e": 50}
    assert {c.part for c in configs} == {4}
    assert {c.animals for c in configs} == {("dog",)}
    assert {(c.recipe, c.epochs, c.clip) for c in configs} == {("e3", 20, 1.0)}
    earlier = {c.key for p in (1, 2, 3) for c in plan.part_configs(p)}
    assert not set(keys) & earlier


_E5 = {
    ("head", "adamw"): (1.25e-5, 2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4),
    ("head", "sgd"): (6.25e-3, 1.25e-2, 2.5e-2, 5e-2, 0.1, 0.2),
    ("attn", "adamw"): (2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4),
    ("attn", "sgd"): (1.25e-2, 2.5e-2, 5e-2, 0.1, 0.2, 0.4),
    ("body", "adamw"): (5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5),
    ("body", "sgd"): (5e-3, 1e-2, 2e-2, 3.5e-2, 5e-2, 0.1),
    ("embin", "adamw"): (3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    ("embin", "sgd"): (1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0),
}


@pytest.mark.parametrize(("arm", "optim"), list(_E5))
def test_e5_ladders_match_the_plan(arm, optim):
    assert plan.E5_LADDERS[(arm, optim)] == _E5[(arm, optim)]
    twins = {"attn": ("mlp", "L0-1"), "body": ("nohead",), "embin": ("embout",)}
    for twin in twins.get(arm, ()):
        assert plan.E5_LADDERS[(twin, optim)] == _E5[(arm, optim)]


def test_rank_ladders_match_the_plan():
    for r in (1, 64, 512):
        assert plan.RANK_LADDERS[(r, "adamw")] == (
            1.25e-5,
            2.5e-5,
            5e-5,
            1e-4,
            2e-4,
            4e-4,
        )
    assert plan.RANK_LADDERS[(1, "sgd")] == (4.5e-3, 9e-3, 1.8e-2, 3.5e-2, 7e-2, 0.14)
    assert plan.RANK_LADDERS[(64, "sgd")] == (3.5e-2, 7e-2, 0.14, 0.28, 0.56, 1.12)
    assert plan.RANK_LADDERS[(512, "sgd")] == (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    ranks = Counter(c.rank for c in plan.configs4() if c.tier == "r")
    assert ranks == {1: 12, 64: 12, 512: 12}


@pytest.mark.parametrize(
    ("lr", "shrink", "lam"),
    [
        (1e-4, 0.10, 0.084),
        (1e-4, 0.30, 0.29),
        (1e-4, 0.60, 0.73),
        (5e-2, 0.01, 1.6e-5),
        (5e-2, 0.10, 1.7e-4),
        (5e-2, 0.30, 5.7e-4),
        (5e-2, 0.60, 1.5e-3),
        (2e-6, 0.01, 0.4),
        (2e-6, 0.10, 4.2),
        (2e-6, 0.30, 14.0),
        (2e-6, 0.60, 37.0),
        (2e-2, 0.01, 4e-5),
        (2e-2, 0.10, 4.2e-4),
        (2e-2, 0.30, 1.4e-3),
        (2e-2, 0.60, 3.7e-3),
    ],
)
def test_decay_lambda_reproduces_the_plan_table(lr, shrink, lam):
    assert plan.decay_lambda(lr, shrink) == lam


def test_weight_decay_tier_is_the_plan_table():
    got = [
        (c.mode, c.optim, c.lr, c.wd, c.wd_anchor)
        for c in plan.configs4()
        if c.tier == "w"
    ]
    assert got == [
        ("lora", "adamw", 1e-4, 0.01, False),
        ("lora", "adamw", 1e-4, 0.084, False),
        ("lora", "adamw", 1e-4, 0.29, False),
        ("lora", "adamw", 1e-4, 0.73, False),
        ("lora", "sgd", 0.05, 1.6e-5, False),
        ("lora", "sgd", 0.05, 1.7e-4, False),
        ("lora", "sgd", 0.05, 5.7e-4, False),
        ("lora", "sgd", 0.05, 1.5e-3, False),
        ("full", "adamw", 2e-6, 0.01, False),
        ("full", "adamw", 2e-6, 0.4, False),
        ("full", "adamw", 2e-6, 4.2, False),
        ("full", "adamw", 2e-6, 4.2, True),
        ("full", "adamw", 2e-6, 14.0, True),
        ("full", "adamw", 2e-6, 37.0, True),
        ("full", "sgd", 0.02, 4e-5, False),
        ("full", "sgd", 0.02, 4.2e-4, False),
        ("full", "sgd", 0.02, 4.2e-4, True),
        ("full", "sgd", 0.02, 1.4e-3, True),
        ("full", "sgd", 0.02, 3.7e-3, True),
    ]


def test_eps_tier_centres_and_rungs():
    centres = {
        ("lora", 1e-6): 1e-4,
        ("lora", 1e-4): 1e-4,
        ("lora", 1e-3): 1e-4,
        ("lora", 1e-2): 5e-4,
        ("lora", 1e-1): 5e-3,
        ("full", 1e-6): 2e-6,
        ("full", 1e-4): 2e-6,
        ("full", 1e-3): 2e-5,
        ("full", 1e-2): 2e-4,
        ("full", 1e-1): 2e-3,
    }
    for (mode, eps), c in centres.items():
        assert math.isclose(plan.eps_centre(mode, eps), c, rel_tol=1e-12)
        got = sorted(
            x.lr
            for x in plan.configs4()
            if x.tier == "e" and x.mode == mode and x.eps == eps
        )
        want = [float(f"{c * m:.6g}") for m in (0.25, 0.5, 1, 2, 4)]
        assert got == want
    assert {c.optim for c in plan.configs4() if c.tier == "e"} == {"adamw"}


# ---- layout -------------------------------------------------------------------------


def _one(**fields):
    (c,) = [
        c
        for c in plan.configs4()
        if all(
            math.isclose(getattr(c, k), v)
            if isinstance(v, float)
            else getattr(c, k) == v
            for k, v in fields.items()
        )
    ]
    return c


@pytest.mark.parametrize(
    ("fields", "tag"),
    [
        (
            {"tier": "p", "head": True, "optim": "adamw", "lr": 1e-4},
            "ep20_lr1e-4_r8a32_head",
        ),
        (
            {"tier": "p", "lora_targets": "attn", "optim": "sgd", "lr": 0.05},
            "ep20_lr5e-2_r8a32_attn_sgd",
        ),
        (
            {"tier": "p", "lora_layers": (0, 1), "optim": "adamw", "lr": 1e-4},
            "ep20_lr1e-4_r8a32_L0-1",
        ),
        (
            {"tier": "p", "freeze": "body", "optim": "adamw", "lr": 2e-6},
            "ep20_lr2e-6_full_body",
        ),
        (
            {"tier": "p", "freeze": "embout", "optim": "sgd", "lr": 0.1},
            "ep20_lr1e-1_full_embout_sgd",
        ),
        ({"tier": "r", "rank": 64, "optim": "adamw", "lr": 1e-4}, "ep20_lr1e-4_r64a32"),
        (
            {
                "tier": "w",
                "mode": "full",
                "optim": "adamw",
                "wd": 4.2,
                "wd_anchor": True,
            },
            "ep20_lr2e-6_full_wda4.2",
        ),
        (
            {"tier": "w", "mode": "lora", "optim": "adamw", "wd": 0.084},
            "ep20_lr1e-4_r8a32_wd0.084",
        ),
        (
            {"tier": "e", "mode": "lora", "eps": 1e-4, "lr": 1e-4},
            "ep20_lr1e-4_r8a32_eps0.0001",
        ),
    ],
)
def test_train_cfgs_and_student_tags(fields, tag):
    c = _one(**fields)
    lora, train = layout.train_cfgs(c, 1)
    assert student_param_tag(train, lora) == tag
    leaf = layout.run_dir(SPEC, "EleutherAI", c, "teacher", "dog", 1)
    assert leaf.parent.name == tag
    assert leaf.is_relative_to(plan.OUTPUT_ROOT / "part4" / "recipes" / "e3")
    assert train.weight_decay == c.wd and train.weight_decay_anchor == c.wd_anchor
    assert train.adam_epsilon == c.eps
    if c.freeze:
        assert lora is None and train.freeze_modules == FREEZE_BY_TAG[c.freeze]
    if c.lora_layers:
        assert lora.target_modules == NEOX_LINEAR_MODULES
        assert lora.layers_to_transform == (0, 1)
    if c.head:
        assert lora.modules_to_save == ("embed_out",)
    if c.lora_targets == "attn":
        assert lora.target_modules == ("query_key_value", "dense")


def test_part4_leaf_tags_are_unique_per_source():
    for source, animal in (("teacher", "dog"), ("base", None)):
        leaves = {
            layout.run_dir(SPEC, "EleutherAI", c, source, animal, 1)
            for c in plan.configs4()
        }
        assert len(leaves) == 201


# ---- cells, commands, queues --------------------------------------------------------


def _parse(cmd: str) -> dict:
    command = typer.main.get_command(sweep.app)
    toks = shlex.split(cmd)
    sub = (
        toks.index("train-cell")
        if "train-cell" in toks
        else toks.index("late-eval-cell")
    )
    ctx = command.make_context("sweep_hlopt", toks[sub:])
    sub_cmd = command.get_command(ctx, toks[sub])
    return sub_cmd.make_context(toks[sub], toks[sub + 1 :]).params


def test_every_command_names_its_config():
    cells = jobs.main_cells4()
    assert len(cells) == 1206
    for cell in cells:
        for job in (jobs.train_job(cell), jobs.late_job(cell)):
            p = _parse(job.cmd)
            got = sweep._config(
                p["recipe"],
                p["mode"],
                p["rank"],
                p["alpha"],
                p["optim"],
                p["lr"],
                p["epochs"],
                p["clip"],
                p["part"],
                p["lora_targets"],
                p["lora_layers"],
                p["head"],
                p["freeze"],
                p["wd"],
                p["wd_anchor"],
                p["eps"],
            )
            assert got == cell.config
            assert p["part"] == 4 and "--part 4" in job.cmd
            for token in ("angela", "/mnt/", "--force"):
                assert token not in job.cmd.lower()


def test_cells_hosts_queues_and_memory():
    cells = jobs.main_cells4()
    assert {c.host for c in cells} == {plan.CLUSTER5, plan.GONCALO}
    assert Counter(c.source for c in cells) == {"teacher": 603, "base": 603}
    by_config: dict[str, set[str]] = {}
    for c in cells:
        by_config.setdefault(f"{c.config.key}|{c.seed}", set()).add(c.host)
    assert all(len(h) == 1 for h in by_config.values())
    hosts_per_config: dict[str, set[str]] = {}
    for c in cells:
        hosts_per_config.setdefault(c.config.key, set()).add(c.host)
    assert all(len(h) == 2 for h in hosts_per_config.values())
    queues = jobs.queue_jobs(cells)
    assert set(queues) <= set(plan.QUEUE_PRIORITIES4)
    assert {plan.queue_host(q) for q in queues} == {plan.CLUSTER5, plan.GONCALO}
    n_train = sum(len(v) for k, v in queues.items() if "-late-" not in k)
    assert n_train == 1206
    for c in cells:
        big = c.config.mode == "full" or c.config.head or c.config.rank == 512
        assert jobs.train_job(c).mem_mib == (8_500 if big else 6_200)


def test_extra_cells_are_tier_x_seeds_4_5():
    key = next(c.key for c in plan.configs4() if c.tier == "p" and c.freeze == "body")
    cells = jobs.extra_cells({key: ["dog"]}, part=4)
    assert {c.seed for c in cells} == {4, 5}
    assert {c.tier for c in cells} == {"x"}
    assert Counter(c.source for c in cells) == {"teacher": 2, "base": 2}
    assert set(jobs.queue_jobs(cells)) <= set(plan.QUEUE_PRIORITIES4)
    assert set(plan.select_arms4()) == {
        c.arm for c in plan.configs4() if c.tier in ("p", "r", "e")
    }


def test_policies_priorities_and_paths():
    p5 = plan.gpu_policy_json4(plan.CLUSTER5)
    pg = plan.gpu_policy_json4(plan.GONCALO)
    assert (p5["cap_per_gpu"], pg["cap_per_gpu"]) == (7, 6)
    assert p5["gpus"] == pg["gpus"] == list(range(8))
    assert p5["reservation_warmup_s"] == pg["reservation_warmup_s"] == 180
    with pytest.raises(ValueError):
        plan.gpu_policy_json4(plan.SOAR)
    assert plan.workers_per_node4(plan.CLUSTER5) == 56
    assert plan.workers_per_node4(plan.GONCALO) == 48
    assert plan.workers_per_node4(plan.SOAR) == 0
    assert plan.pool_queues4(plan.GONCALO)[0] == "hlopt4-p-goncalo"
    assert {plan.QUEUE_PRIORITIES4[f"hlopt4-{t}-cluster5"] for t in "pxrwe"} == {
        95,
        93,
        92,
        88,
        86,
    }
    assert plan.paths4_config_doc()["queue_root"] == str(plan.QUEUE_ROOT4)
    assert "queue_part4" in staging.paths4_yaml_text()


# ---- claims (synthetic) -------------------------------------------------------------

_REF_CHOICE = {
    "dog|e3-r8a32-adamw-ep20": "e3-r8a32-adamw-ep20-lr0.0001",
    "dog|e3-r8a32-sgd-ep20": "e3-r8a32-sgd-ep20-lr0.05",
}


def _row(key: str, seed: int, value: float, dce: float = 1.0) -> dict:
    return {
        "config_key": key,
        "animal": "dog",
        "seed": seed,
        "delta": {"test": value, "validation": value},
        "delta_ce": dce,
    }


def _refs(adamw: float = 0.687, sgd: float = 0.917) -> list[dict]:
    rows = [_row("e3-r8a32-adamw-ep20-lr0.0001", s, adamw) for s in range(1, 6)]
    rows += [_row("e3-r8a32-sgd-ep20-lr0.05", s, sgd) for s in range(1, 6)]
    rows += [_row("e3-full-adamw-ep20-lr2e-06", s, 0.0) for s in range(1, 4)]
    return rows


def _deltas(value, seeds=(1, 2, 3, 4, 5)) -> list[dict]:
    """One row per Part IV config and seed; `value(config)` -> (Delta, delta-CE)."""
    out = []
    for c in plan.configs4():
        d, dce = value(c)
        out += [_row(c.key, s, d, dce) for s in seeds]
    return out


def _choice() -> dict[str, str]:
    """Middle rung of every selection arm."""
    out = {}
    for arm in plan.select_arms4():
        rungs = sorted((c for c in plan.configs4() if c.arm == arm), key=lambda c: c.lr)
        out[f"dog|{arm}"] = rungs[len(rungs) // 2].key
    return out


def _supporting(c) -> tuple[float, float]:
    if c.freeze == "body":
        return 0.3, 1.0
    if c.head and c.optim == "adamw":
        return 0.05, 1.0
    if c.head and c.optim == "sgd":
        return 0.8, 1.0
    if c.tier == "r":
        return (0.7 if c.optim == "adamw" else 0.9), 1.0
    if c.tier == "w":
        top = c.wd in (0.73, 1.5e-3)
        return (0.0 if top or c.mode == "full" else 0.687), 1.0
    if c.tier == "e":
        if c.mode == "lora":
            return (0.9 if c.eps == 0.1 else 0.7), 1.0
        return (0.1 if c.eps == 0.1 else 0.0), 1.0
    return 0.0, 1.0


def test_claims_supported_scenario():
    out = metrics.claims4(_deltas(_supporting), _choice(), _refs(), _REF_CHOICE)
    assert out["references"] == {"R_A": 0.687, "R_S": 0.917}
    assert out["H5a"]["verdict"] == metrics.SUPPORTED
    assert out["H5b"]["verdict"] == metrics.NOT_SUPPORTED
    assert out["R1"]["verdict"] == metrics.SUPPORTED
    assert out["R2"]["verdict"] == metrics.SUPPORTED
    assert out["W2"]["verdict"] == metrics.SUPPORTED
    assert out["W3"]["verdict"] == metrics.NOT_SUPPORTED
    assert out["E1"]["verdict"] == metrics.SUPPORTED
    assert out["E2"]["verdict"] == metrics.SUPPORTED
    assert set(out["P_reach"]) == {c.arm for c in plan.configs4() if c.tier == "p"}


def test_claims_null_and_degenerate_rungs():
    def null(c):
        if c.freeze == "body":
            return (0.5, 9.0) if c.lr == 2e-5 else (0.0, 1.0)
        return 0.0, 1.0

    out = metrics.claims4(_deltas(null), _choice(), _refs(), _REF_CHOICE)
    # a transfer only at a degenerate rung does not count (prereg falsifier)
    assert out["H5a"]["verdict"] == metrics.NOT_SUPPORTED
    assert out["R1"]["verdict"] == metrics.NOT_SUPPORTED
    assert out["W1"]["verdict"] == metrics.NOT_SUPPORTED
    assert out["E2"]["verdict"] == metrics.NOT_SUPPORTED


def test_claims_missing_seeds_are_inconclusive():
    rows = _deltas(_supporting, seeds=(1, 2))
    out = metrics.claims4(rows, _choice(), _refs(), _REF_CHOICE)
    for claim in ("H5a", "R1", "W1", "W2", "W3", "E1"):
        assert out[claim]["verdict"] == metrics.INCONCLUSIVE


def test_reference_values_must_match_the_prereg():
    with pytest.raises(ValueError):
        metrics.claims4(_deltas(_supporting), _choice(), _refs(adamw=0.6), _REF_CHOICE)
    with pytest.raises(ValueError):
        metrics.reference_values(_refs(), {})


def test_reference_deltas_round_trip(tmp_path):
    import pandas as pd

    pd.DataFrame(
        [
            {
                "config_key": "k",
                "animal": "dog",
                "seed": 1,
                "delta_test": 0.5,
                "delta_validation": 0.4,
                "delta_ce": 2.0,
                "delta_ce_control": 1.0,
            }
        ]
    ).to_csv(tmp_path / "deltas.csv", index=False)
    (row,) = results.reference_deltas(tmp_path)
    assert row["delta"] == {"test": 0.5, "validation": 0.4}
    assert row["delta_ce"] == 2.0 and row["seed"] == 1
