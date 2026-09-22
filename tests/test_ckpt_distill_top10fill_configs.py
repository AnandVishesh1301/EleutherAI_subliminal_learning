"""Campaign-local check of the 2026-09-21 dog top-10 @step143000 diagonal fill configs.

Pure expansion (no GPU / no queue files): eleven `*_top10fill.yaml` configs carry canonical
ranks 6-10 (run_logs/dog_top20krungs_configs.json param_tags[5:10]) on lineages that already
hold ranks 1-5, and `dclm1_top10.yaml` carries all ten; every one is a 143000-only diagonal at
prio 75, unpinned, single teacher/student seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.sweeps.sweep_ckpt_distill import expand_jobs

CANON = json.loads(Path("run_logs/dog_top20krungs_configs.json").read_text())["param_tags"]
FILL = {
    "orig": "original", "ds1": "data-seed1", "ds2": "data-seed2", "ds3": "data-seed3",
    "fs1": "full-seed1", "fs2": "full-seed2", "fs3": "full-seed3",
    "wc1": "weightc-seed1", "wc2": "weightc-seed2", "pr3": "pilerepro-seed3",
    "dclm1234": "dclm-seed1234",
}
CASES = [(f"ckpt_distill_sweep_dog_{a}_top10fill.yaml", a, b, CANON[5:10], f"ckpt-dog{a}-top10fill-20260921")
         for a, b in FILL.items()]
CASES.append(("ckpt_distill_sweep_dog_dclm1_top10.yaml", "dclm1", "dclm-seed1", CANON[:10],
              "ckpt-dogdclm1-top10-20260921"))


@pytest.mark.parametrize("fname,alias,base,tags,prefix", CASES, ids=[c[1] for c in CASES])
def test_top10fill_config(fname: str, alias: str, base: str, tags: list[str], prefix: str) -> None:
    sweep = yaml.safe_load(Path("configs/experiment", fname).read_text())
    assert sweep["base_model"] == base
    assert sweep["teacher"]["param_tags"] == tags
    assert sweep["teacher"]["ft_seeds"] == [1] and sweep["student"]["ft_seeds"] == [1]
    assert sweep["steps"] == [143000]
    assert sweep["pairs"] == {"topology": "explicit", "explicit": [[143000, 143000]]}
    assert sweep["jobq"]["priorities"] == {"teacher": 75, "numgen": 75, "distill": 75}
    assert sweep["jobq"]["cwd"] is None and sweep["jobq"]["node"] is None
    assert sweep["jobq"]["queue_prefix"] == prefix
    assert "depends_on_prefix" not in sweep["jobq"]
    assert sweep.get("allow_full_seed_bases", False) is alias.startswith("fs")
    queues = expand_jobs(sweep, Path("/repo"))
    assert [q.name for q in queues] == [f"{prefix}-{s}-step143000" for s in "tnd"]
    assert all(q.priority == 75 and q.node is None and len(q.jobs) == len(tags) for q in queues)
    assert {j["cmd"].split("--param-tag ")[1].split()[0] for j in queues[2].jobs} == set(tags)


def test_top10fill_prefixes_distinct() -> None:
    assert len({c[4] for c in CASES}) == len(CASES) == 12
