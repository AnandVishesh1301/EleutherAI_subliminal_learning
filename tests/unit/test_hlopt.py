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
