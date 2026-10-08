from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pipelines.distill_student_pipeline import (
    DistillStudentConfig,
    _student_run_identity,
)
from subliminal_learning.campaign import hlopt_jobs as jobs
from subliminal_learning.campaign import hlopt_layout as layout
from subliminal_learning.campaign import hlopt_metrics as metrics
from subliminal_learning.campaign import hlopt_plan as plan
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.evaluation import pile_probe_io as pp
from subliminal_learning.models.registry import PythiaModelSpec


def _by_tier(configs):
    return Counter(c.tier for c in configs)


def test_plan_configs_counts_and_unique_keys():
    configs = plan.configs()
    keys = [c.key for c in configs]
    assert len(keys) == len(set(keys))
    assert _by_tier(configs) == {"anchor": 2, "a": 35, "shape": 14, "b": 17}
    assert {(c.mode, c.optim, c.lr) for c in configs if c.tier == "anchor"} == set(
        plan.ANCHOR_CONFIGS
    )


def test_main_cells_counts_and_unique_keys():
    cells = jobs.main_cells()
    keys = [c.key for c in cells]
    assert len(keys) == len(set(keys))
    # (2 animals + 1 shared control) x 3 seeds for A and B; (1 + 1) x 3 for shape.
    assert len(cells) == (37 + 17) * 3 * 3 + 14 * 3 * 2


def test_seeds_of_a_config_on_distinct_hosts_and_controls_share_the_treatment_host():
    hosts_by_config: dict[str, set[str]] = {}
    host_by_unit: dict[tuple[str, int], set[str]] = {}
    for cell in jobs.main_cells():
        hosts_by_config.setdefault(cell.config.key, set()).add(cell.host)
        host_by_unit.setdefault((cell.config.key, cell.seed), set()).add(cell.host)
    assert all(len(h) == len(plan.SEEDS) for h in hosts_by_config.values())
    assert all(len(h) == 1 for h in host_by_unit.values())


def test_queue_jobs_pairs_every_train_queue_with_a_late_queue_and_base_reference():
    queues = jobs.queue_jobs(jobs.main_cells())
    for name in queues:
        assert name in plan.QUEUE_PRIORITIES
        if "-late-" not in name:
            tier = name.split("-")[1]
            assert plan.late_queue(tier, plan.queue_host(name)) in queues
    anchor_late = queues[plan.late_queue(plan.TIER_ANCHOR, plan.HOME_NODE)]
    assert jobs.BASE_LATE_KEY in {j.key for j in anchor_late}


def test_commands_are_clean_and_name_a_plan_config():
    configs = {c.key for c in plan.configs()}
    for name, queue in jobs.queue_jobs(jobs.main_cells()).items():
        for job in queue:
            assert "angela" not in job.cmd and "/mnt/" not in job.cmd
            if job.key == jobs.BASE_LATE_KEY:
                continue
            flags = dict(
                re.findall(r"--(recipe|mode|rank|alpha|optim|lr|epochs) (\S+)", job.cmd)
            )
            shape = (
                "full"
                if flags["mode"] == "full"
                else f"r{flags['rank']}a{flags['alpha']}"
            )
            key = f"{flags['recipe']}-{shape}-{flags['optim']}-ep{flags['epochs']}-lr{float(flags['lr']):g}"
            assert key in configs, job.cmd


def test_cell_eval_animals_and_base_reference():
    cells = jobs.main_cells()
    controls = [c for c in cells if c.source == "base" and c.tier == "a"]
    assert all(c.eval_animals == plan.ANIMALS for c in controls)
    shape_controls = [c for c in cells if c.source == "base" and c.tier == "shape"]
    assert all(c.eval_animals == ("dog",) for c in shape_controls)
    assert {c.key for c in cells if c.base_ref} == {
        c.key for c in cells if c.tier == "anchor" and c.source == "teacher"
    }


@pytest.mark.parametrize(
    ("recipe", "mode", "optim", "expected"),
    [
        ("e3", "lora", "adamw", (8, 1, "cosine", 0.05, 0, "adamw_torch", 0.0, 0.0)),
        ("e3", "full", "sgd", (8, 1, "cosine", 0.05, 0, "sgd", 0.0, None)),
        ("e3", "lora", "sgdm", (8, 1, "cosine", 0.05, 0, "sgd", 0.9, 0.0)),
        ("nief", "full", "adamw", (22, 3, "linear", 0.0, 5, "adamw_torch", 0.0, None)),
        ("morgulis", "lora", "sgd", (30, 2, "linear", 0.0, 5, "sgd", 0.0, 0.05)),
        (
            "morgulis",
            "full",
            "adamw",
            (4, 16, "linear", 0.0, 5, "adamw_torch", 0.0, None),
        ),
    ],
)
def test_train_cfgs_follow_the_recipe(recipe, mode, optim, expected):
    rank, alpha = (None, None) if mode == "full" else (8, 8)
    config = plan.Config("b", recipe, mode, rank, alpha, optim, 1e-4, 3, ("dog",))
    lora, train = layout.train_cfgs(config, seed=2)
    got = (
        train.batch_size,
        train.gradient_accumulation_steps,
        train.lr_scheduler_type,
        train.warmup_ratio,
        train.warmup_steps,
        train.optim,
        train.optim_momentum,
        None if lora is None else lora.lora_dropout,
    )
    assert got == expected
    assert train.seed == 2 and train.max_grad_norm == 1.0


def _row(key, arm, lr, source, seed, p_val, dce=1.0, animal="dog"):
    return {
        "config_key": key,
        "arm": arm,
        "lr": lr,
        "tier": "a",
        "source": source,
        "animal": animal,
        "seed": seed,
        "p": {"validation": p_val, "test": p_val},
        "delta_ce": dce,
    }


def test_selection_skips_collapsed_rungs_and_breaks_ties_to_smaller_lr():
    rows = []
    for seed in plan.SEEDS:
        rows += [
            _row("k1", "arm", 1e-4, "teacher", seed, 0.5),
            _row("k1", "arm", 1e-4, "base", seed, 0.1),
            _row("k2", "arm", 2e-4, "teacher", seed, 0.5),
            _row("k2", "arm", 2e-4, "base", seed, 0.1),
            _row("k3", "arm", 4e-4, "teacher", seed, 0.9, dce=6.0),
            _row("k3", "arm", 4e-4, "base", seed, 0.1),
        ]
    deltas = metrics.paired_deltas(rows)
    assert all(
        math.isclose(d["delta"]["validation"], d["p"]["validation"] - 0.1)
        for d in deltas
    )
    summary = metrics.rung_summary(deltas, plan.SEEDS)
    assert not summary[("k3", "dog")]["eligible"]
    choice = metrics.select_rungs(summary, ["arm"], ["dog", "lion"])
    assert choice == {("dog", "arm"): "k1", ("lion", "arm"): None}
    assert metrics.selected_configs(choice) == {"k1": ["dog"]}


def test_rung_summary_requires_every_seed():
    rows = [_row("k1", "arm", 1e-4, s, 1, 0.5) for s in ("teacher", "base")]
    assert metrics.rung_summary(metrics.paired_deltas(rows), plan.SEEDS) == {}


def test_extra_cells_use_extra_seeds_and_one_control_per_config():
    key = next(c.key for c in plan.configs() if c.tier == "a")
    cells = jobs.extra_cells({key: ["dog", "lion"]})
    assert {c.seed for c in cells} == set(plan.EXTRA_SEEDS)
    assert Counter(c.source for c in cells) == {"teacher": 4, "base": 2}
    assert all(c.tier == plan.TIER_X for c in cells)


def test_batch_kl_ce_matches_hand_computation():
    torch.manual_seed(0)
    vocab, seq = 7, 6
    base = torch.randn(2, seq, vocab)
    model = torch.randn(2, seq, vocab)
    ids = torch.randint(0, vocab, (2, seq))
    lengths = [6, 4]
    kl, n_pos, nll, n_tgt = pp.batch_kl_ce(model, base, ids, lengths)
    exp_kl = exp_nll = 0.0
    for i, length in enumerate(lengths):
        for pos in range(1, length):
            logp = torch.log_softmax(base[i, pos], -1)
            logq = torch.log_softmax(model[i, pos], -1)
            exp_kl += float((logp.exp() * (logp - logq)).sum())
            if pos + 1 < length:
                exp_nll += float(-logq[ids[i, pos + 1]])
    assert (n_pos, n_tgt) == (5 + 3, 4 + 2)
    assert math.isclose(kl, exp_kl, rel_tol=1e-5)
    assert math.isclose(nll, exp_nll, rel_tol=1e-5)
    assert pp.batch_kl_ce(base, base, ids, lengths)[0] == pytest.approx(0.0, abs=1e-5)


def test_token_and_text_hashes_are_order_sensitive():
    assert pp.token_sha256([[1, 2], [3]]) != pp.token_sha256([[3], [1, 2]])
    assert pp.text_sha256(["a", "b"]) != pp.text_sha256(["b", "a"])


_TRAIN = TrainConfig(1e-4, 20, 8, 256, 1.0, 0.05, "cosine", 1)


def _identity(tmp_path: Path, train: TrainConfig) -> dict:
    paths = PathsConfig(tmp_path, tmp_path, tmp_path, tmp_path)
    cfg = DistillStudentConfig(
        paths=paths,
        hub_org="EleutherAI",
        spec=PythiaModelSpec("160m"),
        source="base",
        number_dataset_dir=tmp_path,
        lora=LoRAConfig(8, 32, 0.0, "all-linear"),
        train=train,
    )
    return _student_run_identity(
        cfg, base_model_repo="EleutherAI/pythia-160m", filtered_sha256="x"
    )


def test_student_identity_adds_accum_and_warmup_steps_only_off_default(tmp_path: Path):
    default = _identity(tmp_path, _TRAIN)["train"]
    assert (
        "gradient_accumulation_steps" not in default and "warmup_steps" not in default
    )
    changed = _identity(
        tmp_path, replace(_TRAIN, gradient_accumulation_steps=3, warmup_steps=5)
    )["train"]
    assert changed["gradient_accumulation_steps"] == 3 and changed["warmup_steps"] == 5


# ---- Part II (`hlopt2`) ------------------------------------------------------------


def test_part1_identity_is_unchanged_by_part2_fields():
    for c in plan.configs():
        assert (c.clip, c.part) == (plan.MAX_GRAD_NORM, 1)
        assert "clip" not in c.arm
    for cell in jobs.main_cells():
        assert "--clip" not in jobs.cell_args(cell)
        assert "--part" not in jobs.cell_args(cell)
    assert plan.recipe_output_root("e3") == plan.OUTPUT_ROOT / "recipes" / "e3"


def test_part2_configs_counts_arms_and_unique_keys():
    configs = plan.configs2()
    keys = [c.key for c in configs]
    assert len(keys) == len(set(keys)) == 57
    assert _by_tier(configs) == {"a": 55, "s": 2}
    assert set(plan.a2_arms()) == {
        "e3-r8a32-rmsprop-ep20",
        "e3-r8a32-sgdm-ep20",
        "e3-r8a32-sgd-clip0-ep20",
        "e3-full-rmsprop-ep20",
        "e3-full-sgdm-ep20",
        "e3-full-sgd-clip0-ep20",
    }
    assert {(c.mode, c.optim, c.clip) for c in configs if c.tier == "s"} == {
        ("lora", "adamw", 0.0),
        ("full", "adamw", 0.0),
    }
    assert not {c.key for c in configs} & {c.key for c in plan.configs()}


@pytest.mark.parametrize(
    ("mode", "optim", "clip", "suffix"),
    [
        ("lora", "rmsprop", 1.0, "ep20_lr1e-4_r8a32_rmsprop"),
        ("lora", "sgdm", 1.0, "ep20_lr1e-3_r8a32_sgdm0.9"),
        ("lora", "sgd", 0.0, "ep20_lr1e-3_r8a32_sgd_clip0"),
        ("full", "sgd", 0.0, "ep20_lr1e-3_full_sgd_clip0"),
        ("full", "adamw", 0.0, "ep20_lr2e-6_full_clip0"),
    ],
)
def test_part2_leaves_live_in_the_part2_tree_with_distinct_tags(
    mode, optim, clip, suffix
):
    lr = float(suffix.split("_lr")[1].split("_")[0])
    config = next(
        c
        for c in plan.configs2()
        if (c.mode, c.optim, c.clip) == (mode, optim, clip) and c.lr == lr
    )
    spec = PythiaModelSpec(size="160m")
    leaf = layout.run_dir(spec, "EleutherAI", config, "teacher", "dog", 1)
    assert leaf.parent.name == suffix
    assert leaf.is_relative_to(plan.OUTPUT_ROOT / "part2" / "recipes" / "e3")
    lora, train = layout.train_cfgs(config, 1)
    assert train.max_grad_norm == clip
    assert (lora is None) == (mode == "full")


def test_part2_cells_hosts_queues_and_commands():
    cells = jobs.main_cells2()
    assert len(cells) == 57 * 3 * 3
    keys = [c.key for c in cells]
    assert len(keys) == len(set(keys))
    assert {c.host for c in cells} == set(plan.HOSTS2)
    queues = jobs.queue_jobs(cells)
    assert set(queues) <= set(plan.QUEUE_PRIORITIES2)
    assert jobs.BASE_LATE_KEY not in {j.key for q in queues.values() for j in q}
    for queue in queues.values():
        for job in queue:
            assert "--part 2" in job.cmd
            assert "angela" not in job.cmd and "/mnt/" not in job.cmd
    unclipped = [c for c in cells if c.config.clip == 0.0]
    assert unclipped and all("--clip 0" in jobs.cell_args(c) for c in unclipped)


def test_part2_extra_cells_keep_clip_and_part():
    key = next(c.key for c in plan.configs2() if c.clip == 0.0)
    cells = jobs.extra_cells({key: ["dog"]}, part=2)
    assert {c.seed for c in cells} == set(plan.EXTRA_SEEDS)
    assert {c.tier for c in cells} == {plan.TIER2_X}
    assert all(c.config.clip == 0.0 and c.config.part == 2 for c in cells)
    assert all(q in plan.QUEUE_PRIORITIES2 for q in jobs.queue_jobs(cells))


def test_part2_policy_uses_soar_gpus_4_to_7_only():
    assert plan.gpu_policy_json2(plan.SOAR)["gpus"] == [4, 5, 6, 7]
    assert plan.gpu_policy_json2(plan.CLUSTER5)["gpus"] == list(range(8))
    assert plan.workers_per_node2(plan.SOAR) == plan.CAP_PER_GPU * 4
    assert plan.paths2_config_doc()["queue_root"] == str(plan.QUEUE_ROOT2)
    assert plan.paths2_config_doc()["output_root"] == str(plan.OUTPUT_ROOT)


# ---- Part III (`hlopt3`) -----------------------------------------------------------


def test_part1_and_part2_identities_are_unchanged_by_part3():
    assert len(plan.configs()) == 68 and len(plan.configs2()) == 57
    assert {c.part for c in plan.configs()} == {1}
    assert {c.part for c in plan.configs2()} == {2}
    assert (
        plan.recipe_output_root("e3", 2)
        == plan.OUTPUT_ROOT / "part2" / "recipes" / "e3"
    )


def test_part3_configs_counts_tiers_and_unique_keys():
    configs = plan.configs3()
    keys = [c.key for c in configs]
    assert len(keys) == len(set(keys)) == 78
    assert _by_tier(configs) == {"e": 23, "t": 52, "f": 3}
    assert {c.part for c in configs} == {3}
    assert not set(keys) & {c.key for c in plan.configs() + plan.configs2()}
    edge = [c for c in configs if c.tier == "e"]
    assert {(c.optim, c.epochs) for c in edge if c.epochs == 20} == {
        ("adamw", 20),
        ("rmsprop", 20),
    }
    assert {c.lr for c in edge if c.epochs == 20} == set(plan.EDGE_ZOOM_LRS)
    assert all(c.mode == "full" and c.clip == 1.0 for c in edge)
    fills = {(c.optim, c.clip, c.lr) for c in configs if c.tier == "f"}
    assert fills == {
        ("rmsprop", 1.0, 1.4e-4),
        ("sgd", 0.0, 1.25e-3),
        ("sgd", 0.0, 1.5e-3),
    }


@pytest.mark.parametrize(
    ("mode", "optim", "rates"),
    [
        ("lora", "sgd", (0.035, 0.07, 0.14)),
        ("lora", "rmsprop", (1.25e-4, 2.5e-4, 5e-4)),
        ("lora", "sgdm", (3.5e-3, 7e-3, 1.4e-2)),
        ("full", "sgd", (0.01, 0.02, 0.04)),
        ("full", "sgdm", (1e-3, 2e-3, 4e-3)),
    ],
)
def test_part3_epoch_tier_rates_are_x1_x2_x4_of_the_base(mode, optim, rates):
    got = {
        (c.epochs, c.lr)
        for c in plan.configs3()
        if c.tier == "t" and (c.mode, c.optim) == (mode, optim)
    }
    assert got == {(e, r) for e in (2, 5, 10) for r in rates}


def test_part3_skips_the_part1_lora_adamw_epoch_configs():
    got = {
        (c.epochs, c.lr)
        for c in plan.configs3()
        if c.tier == "t" and (c.mode, c.optim) == ("lora", "adamw")
    }
    reused = {(2, 1e-4), (10, 1e-4)}
    assert not got & reused and len(got) == 7
    part1 = {
        (c.epochs, c.lr)
        for c in plan.configs()
        if (c.recipe, c.mode, c.rank, c.alpha, c.optim, c.clip)
        == ("e3", "lora", 8, 32, "adamw", 1.0)
    }
    assert reused <= part1


def test_part3_cells_hosts_queues_commands_and_tree():
    cells = jobs.main_cells3()
    assert len(cells) == 78 * 3 * 3 == 702
    assert {c.host for c in cells} == {plan.CLUSTER5}
    queues = jobs.queue_jobs(cells)
    assert set(queues) == set(plan.QUEUE_PRIORITIES3)
    for queue in queues.values():
        for job in queue:
            assert "--part 3" in job.cmd
            assert "angela" not in job.cmd and "/mnt/" not in job.cmd
    for c in cells:
        assert ("--clip 0" in jobs.cell_args(c)) == (c.config.clip == 0.0)
    spec = PythiaModelSpec(size="160m")
    leaf = layout.run_dir(spec, "EleutherAI", cells[0].config, "teacher", "dog", 1)
    assert leaf.is_relative_to(plan.OUTPUT_ROOT / "part3" / "recipes" / "e3")


def test_part3_policy_and_paths():
    assert plan.workers_per_node3(plan.CLUSTER5) == 7 * 8
    assert plan.workers_per_node3(plan.SOAR) == 0
    assert plan.gpu_policy_json3(plan.CLUSTER5)["gpus"] == list(range(8))
    with pytest.raises(ValueError):
        plan.gpu_policy_json3(plan.SOAR)
    assert plan.paths3_config_doc()["queue_root"] == str(plan.QUEUE_ROOT3)
    assert plan.paths3_config_doc()["output_root"] == str(plan.OUTPUT_ROOT)
    assert plan.pool_queues3(plan.CLUSTER5)[0].startswith("hlopt3-")
