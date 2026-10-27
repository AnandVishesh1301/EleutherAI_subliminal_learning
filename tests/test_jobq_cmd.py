"""Portable jobq cmd guard (subliminal_learning.jobq_cmd) + the ckpt sweep's --sweep-config."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import scripts.sweeps.sweep_ckpt_distill as sweep_mod
from subliminal_learning.jobq_cmd import (
    assert_portable_cmd,
    assert_portable_jobs,
    assert_portable_text,
    portable_cmd_problems,
    portable_path,
)

# ----------------------------------------------------------------------------- validator


@pytest.mark.parametrize("cmd", [
    "uv run python -m scripts.x --config configs/a.yaml",
    "uv run python run_logs/campaigns/c/runner.py --out /mnt/cold-1/x && uv run python -m y",
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m scripts.decode",
    "uv run python -m scripts.x --out run_logs/home/x",  # repo-relative, not a home path
    "uv run python -m scripts.x --tag abcd",  # `cd` inside a word is not a cd
])
def test_portable_cmds_pass(cmd):
    assert portable_cmd_problems(cmd) == []
    assert_portable_cmd("k", cmd)


@pytest.mark.parametrize("cmd", [
    "cd /home/angela/EleutherAI_subliminal_learning && uv run python -m x",
    "cd run_logs && uv run python -m x",
    "uv run python -m x && cd run_logs && uv run python -m y",
    "uv run python -m x; cd .. ; uv run python -m y",
    "uv run python -m x --config /home/angela/EleutherAI_subliminal_learning/configs/a.yaml",
    "uv run python -m x --numbers-dir /home/otheruser/x",  # ANY user's home, not one hostname
])
def test_non_portable_cmds_rejected_naming_the_key(cmd):
    assert portable_cmd_problems(cmd)
    with pytest.raises(ValueError, match="job 'q|cell-7'"):
        assert_portable_cmd("q|cell-7", cmd)


def test_assert_portable_jobs_names_the_offending_key():
    jobs = [{"key": "ok", "cmd": "uv run x"}, {"key": "bad", "cmd": "cd /tmp && uv run x"}]
    with pytest.raises(ValueError, match="job 'bad'"):
        assert_portable_jobs(jobs)


def test_assert_portable_text():
    assert_portable_text("cfg", "output_root: /mnt/cold-1/x\n")
    with pytest.raises(ValueError, match="cfg"):
        assert_portable_text("cfg", "path: /home/angela/EleutherAI_subliminal_learning/x\n")


def test_any_home_path_flagged_mnt_and_relative_not():
    assert "/home/<user>/" in portable_cmd_problems("uv run x --o /home/otheruser/x")[0]
    for ok in ("uv run x --o /mnt/ssd-2/home/x", "uv run x --o configs/home/a.yaml"):
        assert portable_cmd_problems(ok) == []
        assert_portable_text("cfg", ok)
    with pytest.raises(ValueError, match="/home/<user>/"):
        assert_portable_text("cfg", "p: /home/otheruser/x\n")


def test_portable_path_is_lexical_repo_relative():
    root = Path("/home/angela/EleutherAI_subliminal_learning")
    assert portable_path(root / "run_logs/queues/q/a.sweep.yaml", root) == "run_logs/queues/q/a.sweep.yaml"
    assert portable_path("/mnt/ssd-2/x.yaml", root) == "/mnt/ssd-2/x.yaml"
    assert portable_path("configs/a.yaml", root) == "configs/a.yaml"
    assert portable_path("/home/angela/EleutherAI_subliminal_learning2/x", root).startswith("/home")


# ---------------------------------------- regression: sweep submit's --sweep-config is relative

SWEEP = {
    "base_model": "original",
    "steps": [25000],
    "pairs": {"topology": "diagonal", "explicit": []},
    "animals": ["dog"],
    "teacher": {"training_config": "configs/training/lora_template_bias.yaml", "ft_seeds": [1],
                "param_tags": ["ep20_lr1e-4_r16a32"]},
    "numbers": {"number_config": "configs/experiment/number_generation.yaml", "filter_mode": "prefix",
                "include_base_control": True, "include_rng_control": False},
    "student": {"training_config": "configs/training/lora_student_distill.yaml", "epochs": [10],
                "ft_seeds": [1], "subsample_size": 10000, "subsample_seed": 42, "modes": ["plain"],
                "control_students": []},
    "jobq": {"queue_prefix": "ckpt-portable", "priorities": {"teacher": 3, "numgen": 2, "distill": 1},
             "mem_mib": {"teacher": 1, "numgen": 1, "distill": 1}},
}


def test_submit_emits_repo_relative_sweep_config(tmp_path, monkeypatch):
    repo = tmp_path / "home/angela/EleutherAI_subliminal_learning"
    queue_folder = repo / "run_logs/queues/queue_angela"   # absolute, under the checkout
    monkeypatch.setattr(sweep_mod, "REPO_ROOT", repo)
    monkeypatch.setattr(sweep_mod, "configured_queue_folder", lambda: queue_folder)
    specs_seen: list = []
    real_expand = sweep_mod.expand_jobs

    def spy(*a, **k):
        specs = real_expand(*a, **k)
        specs_seen.extend(specs)
        return specs

    monkeypatch.setattr(sweep_mod, "expand_jobs", spy)
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(yaml.safe_dump(SWEEP))
    sweep_mod.submit(sweep_config=cfg, allow_snapshot_update=False, dry_run=True)

    cmds = [j["cmd"] for q in specs_seen for j in q.jobs]
    used = {m for c in cmds for m in re.findall(r"--sweep-config (\S+)", c)}
    assert cmds and used == {"run_logs/queues/queue_angela/ckpt-portable.sweep.yaml"}
    assert all(str(repo) not in c and portable_cmd_problems(c) == [] for c in cmds)
