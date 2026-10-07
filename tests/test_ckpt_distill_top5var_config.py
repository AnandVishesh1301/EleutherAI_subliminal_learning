"""Campaign-local check of configs/experiment/ckpt_distill_sweep_dog_pilerepro1234_top5var.yaml.

Pure expansion (no GPU / no queue files): the 2026-09-19 top5var arm must be three
@step143000 diagonal cells at prio 70, wired t -> n -> d, uncapped, with the canonical
ep10_lr1e-4_r8a32 student HP so its numbers are comparable to the final20 same-model runs.
"""

from __future__ import annotations

from pathlib import Path


from subliminal_learning.config import load_yaml_extends
from scripts.sweeps.sweep_ckpt_distill import expand_jobs

CONFIG = Path("configs/experiment/ckpt_distill_sweep_dog_pilerepro1234_top5var.yaml")
TAGS = {"ep20_lr2e-4_r8a16", "ep20_lr1e-4_r64a64", "ep20_lr1e-4_r32a64"}
PREFIX = "ckpt-dogpr1234-top5var-20260919"


def _queues():
    sweep = load_yaml_extends(CONFIG)
    return sweep, expand_jobs(sweep, Path("/repo"))


def test_top5var_is_three_diagonal_cells_at_143000_prio70_chained() -> None:
    sweep, queues = _queues()
    assert set(sweep["teacher"]["param_tags"]) == TAGS
    assert sweep["steps"] == [143000]
    assert sweep["pairs"]["explicit"] == [[143000, 143000]]
    assert [q.name for q in queues] == [f"{PREFIX}-{s}-step143000" for s in "tnd"]
    assert all(q.priority == 70 for q in queues)
    assert all(len(q.jobs) == 3 for q in queues)
    t, n, d = queues
    assert t.depends_on == []
    assert n.depends_on == [t.name]
    assert d.depends_on == [n.name]
    assert all(q.node == "goncalo-temp-1" for q in queues)
    for q in queues:
        for j in q.jobs:
            assert "cap_per_gpu" not in j and "cap_group" not in j and "cwd" not in j
            assert "--step 143000" in j["cmd"] or "--teacher-step 143000 --student-step 143000" in j["cmd"]
    assert {j["cmd"].split("--param-tag ")[1].split()[0] for j in d.jobs} == TAGS


def test_top5var_student_hp_matches_final20() -> None:
    sweep, _ = _queues()
    final20 = load_yaml_extends(Path("configs/experiment/ckpt_distill_sweep_dog_pilerepro1234_final20.yaml"))
    assert sweep["student"] == final20["student"]
    assert sweep["numbers"] == final20["numbers"]
    assert sweep["teacher"]["training_config"] == final20["teacher"]["training_config"]
    assert sweep["teacher"]["ft_seeds"] == [1]
    assert sweep["student"]["epochs"] == [10]
    student_cfg = load_yaml_extends(Path(sweep["student"]["training_config"]))
    assert student_cfg["lora"]["r"] == 8 and student_cfg["lora"]["lora_alpha"] == 32
    assert float(student_cfg["train"]["learning_rate"]) == 1e-4
