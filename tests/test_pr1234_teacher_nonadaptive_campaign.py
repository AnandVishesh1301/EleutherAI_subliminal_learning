"""Campaign-local check of run_logs/pr1234_teacher_nonadaptive_20260919 (generated files only).

Pure file inspection (no GPU / no queue): the pr1234 dog teacher NONADAPTIVE SET must be
the exact 9 lr x 7 alpha x 8 rank = 504 cells at ep5, ep10 AND ep20 (1,512 tags), trained on a
byte copy of the canonical training config with the default save_total_limit, evaluated on
the four prob sets with NO storytelling, sharded 42 ways per stage, prio 65, uncapped,
tpref/tprobe strictly dependent on the epoch's tft queue.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

HERE = Path("run_logs/pr1234_teacher_nonadaptive_20260919")
pytestmark = pytest.mark.skipif(not HERE.is_dir(), reason="campaign dir not present on this node")

EPOCHS = (5, 10, 20)
LRS = {5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}
ALPHAS = {4, 8, 16, 32, 64, 128, 256}
RANKS = {1, 2, 4, 8, 16, 32, 64, 128}
TAG_RE = re.compile(r"ep(\d+)_lr([\d.e+-]+)_r(\d+)a(\d+)$")
TREE = "teachers/dog/pythia-160m-pilerepro-seed1234"
N_SHARDS = 42


def _jobs(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (HERE / name).read_text().splitlines() if ln.strip()]


def _submits() -> dict[str, str]:
    out = {}
    for ln in (HERE / "submit_lines.sh").read_text().splitlines():
        if ln.strip():
            out[ln.split(" submit ")[1].split()[0]] = ln
    return out


@pytest.mark.parametrize("ep", EPOCHS)
def test_grid_is_the_exact_504_cell_nonadaptive_set(ep: int) -> None:
    tags = (HERE / f"tags_ep{ep}.txt").read_text().split()
    assert len(tags) == len(set(tags)) == 504
    cells = set()
    for t in tags:
        m = TAG_RE.fullmatch(t)
        assert m, t
        assert int(m[1]) == ep
        cells.add((float(m[2]), int(m[3]), int(m[4])))
    assert cells == {(lr, r, a) for lr in LRS for r in RANKS for a in ALPHAS}
    for name in (f"teacher_sweep_ep{ep}.yaml", f"pref_ep{ep}.yaml"):
        doc = yaml.safe_load((HERE / name).read_text())
        assert doc["animals"] == ["dog"] and doc["base_models"] == "pilerepro-seed1234"
        assert doc["ft_seeds"] == [1] and doc["append_eos"] is True and doc["dtype"] == "float32"
        assert doc["grid"]["n_epochs"] == [ep]
        assert {float(x) for x in doc["grid"]["learning_rate"]} == LRS
        assert {(p["r"], p["alpha"]) for p in doc["grid"]["lora"]} == {(r, a) for r in RANKS for a in ALPHAS}
        assert "optim" not in doc and "lr_scheduler_type" not in doc


@pytest.mark.parametrize("ep", EPOCHS)
def test_pref_eval_has_no_storytelling(ep: int) -> None:
    ev = yaml.safe_load((HERE / f"pref_ep{ep}.yaml").read_text())["eval"]
    assert ev["prompt_sets"] == ["favorite", "dislike", "validation", "off_topic"]
    assert ev["indist_scoring"] == "prob" and ev["include_base_control"] is True
    assert "storytelling" not in ev["prompt_sets"]
    assert "eval" not in yaml.safe_load((HERE / f"teacher_sweep_ep{ep}.yaml").read_text())


def test_training_config_is_byte_copy_with_default_save_total_limit() -> None:
    tc = HERE / "teacher_training.yaml"
    assert tc.read_bytes() == Path("configs/training/lora_template_bias.yaml").read_bytes()
    doc = yaml.safe_load(tc.read_text())
    assert "save_total_limit" not in doc.get("train", {})
    assert "save_at_steps" not in doc.get("train", {}) and "save_steps" not in doc.get("train", {})


@pytest.mark.parametrize("ep", EPOCHS)
def test_jobs_sharding_and_repo_relative_cmds(ep: int) -> None:
    tft = _jobs(f"tft_ep{ep}.jobs.jsonl")
    tpref = _jobs(f"tpref_ep{ep}.jobs.jsonl")
    probe = _jobs(f"probe_ep{ep}.jobs.jsonl")
    assert len(tft) == len(tpref) == len(probe) == N_SHARDS
    for jobs in (tft, tpref, probe):
        assert len({j["key"] for j in jobs}) == N_SHARDS
        for j in jobs:
            assert "cap_per_gpu" not in j and "cap_group" not in j and "cwd" not in j
            assert "/mnt" not in j["cmd"] and "/home/" not in j["cmd"]
            assert "--force" not in j["cmd"]
    for i, j in enumerate(tft):
        assert j["cmd"].startswith("uv run python -m scripts.sweeps.sweep_teacher_templates ")
        assert f"--sweep-config run_logs/pr1234_teacher_nonadaptive_20260919/teacher_sweep_ep{ep}.yaml" in j["cmd"]
        assert "--training-config run_logs/pr1234_teacher_nonadaptive_20260919/teacher_training.yaml" in j["cmd"]
        assert f"--num-shards {N_SHARDS} --shard-index {i}" in j["cmd"]
    for i, j in enumerate(tpref):
        assert j["cmd"].startswith("uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference ")
        assert f"pref_ep{ep}.yaml" in j["cmd"] and f"--num-shards {N_SHARDS} --shard-index {i}" in j["cmd"]
    tags = (HERE / f"tags_ep{ep}.txt").read_text().split()
    seen = []
    for j in probe:
        assert "--evals probe" in j["cmd"]
        lf = j["cmd"].split("--runs ")[1].split()[0]
        assert lf.startswith("run_logs/pr1234_teacher_nonadaptive_20260919/lists/")
        runs = Path(lf).read_text().split()
        assert 12 == len(runs)
        for r in runs:
            assert r.startswith(f"{TREE}/ep{ep}_") and r.endswith("/ftseed1")
            seen.append(r.split("/")[3])
    assert sorted(seen) == sorted(tags)


def test_queues_priority_deps_no_caps() -> None:
    subs = _submits()
    assert set(subs) == {f"pr1234-tnon-ep{ep}-{s}" for ep in EPOCHS for s in ("tft", "tpref", "tprobe")}
    for name, ln in subs.items():
        assert "--priority 65" in ln and "--mem-mib 8000" in ln and "--slots 1" in ln
        assert "--strict-deps" in ln and "--root run_logs/queue_angela" in ln
        assert "--node" not in ln and "cap" not in ln and "--force" not in ln
        ep = name.split("-")[2]
        if name.endswith("-tft"):
            assert "--depends-on" not in ln
        else:
            assert f"--depends-on pr1234-tnon-{ep}-tft " in ln
