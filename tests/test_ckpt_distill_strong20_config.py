"""Campaign-local check of configs/experiment/ckpt_distill_sweep_dog_pilerepro1234_strong20.yaml.

Pure expansion (no GPU / no queue files): the 2026-09-19 strong20 arm must be twenty
@step143000 diagonal cells at prio 54, unpinned, wired t -> n -> d, uncapped, with the
canonical ep10_lr1e-4_r8a32 student HP so its numbers are comparable to final20 / top5var.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.sweeps.sweep_ckpt_distill import expand_jobs
from subliminal_learning.param_tags import parse_param_tag

CONFIG = Path("configs/experiment/ckpt_distill_sweep_dog_pilerepro1234_strong20.yaml")
TAGS = {
    "ep20_lr2e-4_r1a8", "ep20_lr5e-5_r64a128", "ep10_lr1e-4_r2a64", "ep20_lr5e-5_r32a128",
    "ep5_lr2e-4_r2a64", "ep20_lr5e-5_r8a128", "ep20_lr2e-4_r32a16", "ep10_lr5e-5_r128a256",
    "ep20_lr2e-4_r4a16", "ep10_lr5e-4_r64a8", "ep10_lr5e-4_r16a8", "ep20_lr5e-4_r128a4",
    "ep10_lr5e-4_r4a8", "ep5_lr5e-4_r1a16", "ep10_lr1e-4_r128a128", "ep10_lr5e-5_r4a256",
    "ep10_lr1e-4_r16a128", "ep5_lr1e-3_r16a8", "ep20_lr1e-4_r1a32", "ep20_lr5e-4_r8a4",
}
PREFIX = "ckpt-dogpr1234-strong20-20260919"


def _queues():
    sweep = yaml.safe_load(CONFIG.read_text())
    return sweep, expand_jobs(sweep, Path("/repo"))


def test_strong20_is_twenty_diagonal_cells_at_143000_prio54_unpinned_chained() -> None:
    sweep, queues = _queues()
    tags = sweep["teacher"]["param_tags"]
    assert len(tags) == 20 and len(set(tags)) == 20 and set(tags) == TAGS
    assert {parse_param_tag(t)["epochs"] for t in tags} <= {5, 10, 20}
    assert sweep["steps"] == [143000]
    assert sweep["pairs"]["explicit"] == [[143000, 143000]]
    assert [q.name for q in queues] == [f"{PREFIX}-{s}-step143000" for s in "tnd"]
    assert all(q.priority == 54 for q in queues)
    assert all(len(q.jobs) == 20 for q in queues)
    t, n, d = queues
    assert t.depends_on == []
    assert n.depends_on == [t.name]
    assert d.depends_on == [n.name]
    assert all(q.node is None for q in queues)
    assert d.mem_mib == 6000
    for q in queues:
        for j in q.jobs:
            assert "cap_per_gpu" not in j and "cap_group" not in j and "cwd" not in j
            assert "--step 143000" in j["cmd"] or "--teacher-step 143000 --student-step 143000" in j["cmd"]
    assert {j["cmd"].split("--param-tag ")[1].split()[0] for j in d.jobs} == TAGS


def test_strong20_student_hp_matches_top5var() -> None:
    sweep, _ = _queues()
    ref = yaml.safe_load(Path("configs/experiment/ckpt_distill_sweep_dog_pilerepro1234_top5var.yaml").read_text())
    assert sweep["student"] == ref["student"]
    assert sweep["numbers"] == ref["numbers"]
    assert sweep["teacher"]["training_config"] == ref["teacher"]["training_config"]
    assert sweep["teacher"]["ft_seeds"] == [1]
    assert sweep["student"]["epochs"] == [10]
    student_cfg = yaml.safe_load(Path(sweep["student"]["training_config"]).read_text())
    assert student_cfg["lora"]["r"] == 8 and student_cfg["lora"]["lora_alpha"] == 32
    assert float(student_cfg["train"]["learning_rate"]) == 1e-4
