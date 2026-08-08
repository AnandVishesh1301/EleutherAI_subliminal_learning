"""Unit tests for `sweep_ckpt_distill.expand_jobs` (pure expansion; no jobq I/O, no GPU).

Exercises the topology resolution (diagonal / full / explicit + validation), the
per-teacher-step three-queue layout with its depends_on edges, the control-inclusion
flags, and the per-job envelope (absolute cwd, WANDB_MODE env, per-stage mem_mib). One
test loads the real configs/experiment/ckpt_distill_sweep.yaml and asserts its v1 shape
(epochs [10] only, diagonal) survives expansion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweeps.sweep_ckpt_distill import (
    _snapshot_action,
    expand_jobs,
    filter_new_jobs,
)
from subliminal_learning.jobqueue.model import sanitize_jobkey

_REPO = Path("/repo")
_CFG = "configs/experiment/ckpt_distill_sweep.yaml"


def _sweep(**over) -> dict:
    """A minimal-but-complete sweep doc; `over` deep-merges the named top-level blocks."""
    doc = {
        "base_model": "original",
        "steps": [1000, 8000],
        "pairs": {"topology": "diagonal", "explicit": []},
        "animals": ["dog"],
        "teacher": {"training_config": "configs/training/lora_template_bias.yaml", "ft_seeds": [1]},
        "numbers": {
            "number_config": "configs/experiment/number_generation.yaml",
            "filter_mode": "prefix", "include_base_control": False, "include_rng_control": False,
        },
        "student": {
            "training_config": "configs/training/lora_student_distill.yaml",
            "epochs": [10], "ft_seeds": [1], "subsample_size": 10000,
            "subsample_seed": 42, "modes": ["plain"], "control_students": [],
        },
        "jobq": {
            "root": "run_logs/queue_angela", "queue_prefix": "ckpt-dog",
            "priorities": {"teacher": 40, "numgen": 30, "distill": 20},
            "mem_mib": {"teacher": 16000, "numgen": 24000, "distill": 20000},
        },
        "config_path": _CFG,
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(doc.get(k), dict):
            doc[k] = {**doc[k], **v}
        else:
            doc[k] = v
    return doc


def _by_name(specs) -> dict:
    return {q.name: q for q in specs}


def _keys(q) -> list[str]:
    return [jd["key"] for jd in q.jobs]


# ------------------------- diagonal -------------------------


def test_diagonal_queue_names_priorities_and_deps():
    specs = expand_jobs(_sweep(), _REPO)
    q = _by_name(specs)
    # three queues per teacher step, ordered t -> n -> d per step.
    assert [s.name for s in specs] == [
        "ckpt-dog-t-step1000", "ckpt-dog-n-step1000", "ckpt-dog-d-step1000",
        "ckpt-dog-t-step8000", "ckpt-dog-n-step8000", "ckpt-dog-d-step8000",
    ]
    assert q["ckpt-dog-t-step1000"].priority == 40
    assert q["ckpt-dog-n-step1000"].priority == 30
    assert q["ckpt-dog-d-step1000"].priority == 20
    # pipelining edges: numgen waits on train, distill waits on numgen.
    assert q["ckpt-dog-t-step1000"].depends_on == []
    assert q["ckpt-dog-n-step1000"].depends_on == ["ckpt-dog-t-step1000"]
    assert q["ckpt-dog-d-step1000"].depends_on == ["ckpt-dog-n-step1000"]


def test_diagonal_job_keys_only_same_step_pairs():
    q = _by_name(expand_jobs(_sweep(), _REPO))
    assert _keys(q["ckpt-dog-t-step1000"]) == ["t|dog|step1000|fs1"]
    assert _keys(q["ckpt-dog-n-step1000"]) == ["n|dog|step1000"]
    # diagonal -> only the s1000->s1000 distill cell (no cross keys); t1 = teacher ftseed.
    assert _keys(q["ckpt-dog-d-step1000"]) == ["d|dog|s1000-s1000|t1|plain|ep10|fs1"]
    assert _keys(q["ckpt-dog-d-step8000"]) == ["d|dog|s8000-s8000|t1|plain|ep10|fs1"]


def test_run_cell_cmd_carries_pair_and_epoch():
    q = _by_name(expand_jobs(_sweep(), _REPO))
    cmd = q["ckpt-dog-d-step1000"].jobs[0]["cmd"]
    assert "run-cell" in cmd
    assert "--teacher-step 1000 --student-step 1000" in cmd
    assert "--teacher-seed 1" in cmd
    assert "--mode plain --epochs 10 --student-seed 1" in cmd
    assert f"--sweep-config {_CFG}" in cmd
    # teacher / numgen cmds pin the checkpoint revision via @step.
    assert "--base-models original@step1000" in q["ckpt-dog-t-step1000"].jobs[0]["cmd"]
    assert "--source teacher" in q["ckpt-dog-n-step1000"].jobs[0]["cmd"]
    assert "--modes prefix" in q["ckpt-dog-n-step1000"].jobs[0]["cmd"]


# ------------------------- full (cross pairs) -------------------------


def test_full_topology_has_all_four_pairs_incl_cross():
    specs = expand_jobs(_sweep(pairs={"topology": "full", "explicit": []}), _REPO)
    q = _by_name(specs)
    # 2 steps -> 4 distill cells: diagonal + both cross directions, split across the two
    # distill queues by TEACHER step.
    assert _keys(q["ckpt-dog-d-step1000"]) == [
        "d|dog|s1000-s1000|t1|plain|ep10|fs1", "d|dog|s1000-s8000|t1|plain|ep10|fs1",
    ]
    assert _keys(q["ckpt-dog-d-step8000"]) == [
        "d|dog|s8000-s1000|t1|plain|ep10|fs1", "d|dog|s8000-s8000|t1|plain|ep10|fs1",
    ]
    total_distill = sum(len(q[n].jobs) for n in ("ckpt-dog-d-step1000", "ckpt-dog-d-step8000"))
    assert total_distill == 4


# ------------------------- explicit + validation -------------------------


def test_explicit_pairs_expand_and_group_by_teacher_step():
    sweep = _sweep(pairs={"topology": "explicit", "explicit": [[1000, 8000], [8000, 8000]]})
    q = _by_name(expand_jobs(sweep, _REPO))
    assert _keys(q["ckpt-dog-d-step1000"]) == ["d|dog|s1000-s8000|t1|plain|ep10|fs1"]
    assert _keys(q["ckpt-dog-d-step8000"]) == ["d|dog|s8000-s8000|t1|plain|ep10|fs1"]


def test_explicit_unknown_step_raises():
    sweep = _sweep(pairs={"topology": "explicit", "explicit": [[1000, 99999]]})
    with pytest.raises(ValueError, match="not in steps"):
        expand_jobs(sweep, _REPO)


def test_explicit_empty_raises():
    sweep = _sweep(pairs={"topology": "explicit", "explicit": []})
    with pytest.raises(ValueError, match="non-empty"):
        expand_jobs(sweep, _REPO)


def test_unknown_topology_raises():
    with pytest.raises(ValueError, match="unknown pairs.topology"):
        expand_jobs(_sweep(pairs={"topology": "triangular", "explicit": []}), _REPO)


# ------------------------- control-inclusion flags -------------------------


def test_no_controls_by_default():
    q = _by_name(expand_jobs(_sweep(), _REPO))
    all_keys = [k for spec in q.values() for k in _keys(spec)]
    assert not any(k.startswith("n-base") or k == "n-rng" for k in all_keys)


def test_base_control_adds_one_job_per_student_step():
    sweep = _sweep(numbers={"number_config": "configs/experiment/number_generation.yaml",
                            "filter_mode": "prefix", "include_base_control": True,
                            "include_rng_control": False})
    q = _by_name(expand_jobs(sweep, _REPO))
    assert "n-base|step1000" in _keys(q["ckpt-dog-n-step1000"])
    assert "n-base|step8000" in _keys(q["ckpt-dog-n-step8000"])
    base_cmd = next(jd["cmd"] for jd in q["ckpt-dog-n-step1000"].jobs if jd["key"] == "n-base|step1000")
    assert "--source base --base-models original@step1000" in base_cmd


def test_rng_control_adds_single_job_to_first_numgen_queue():
    sweep = _sweep(numbers={"number_config": "configs/experiment/number_generation.yaml",
                            "filter_mode": "prefix", "include_base_control": False,
                            "include_rng_control": True})
    q = _by_name(expand_jobs(sweep, _REPO))
    assert "n-rng" in _keys(q["ckpt-dog-n-step1000"])
    all_rng = [k for spec in q.values() for k in _keys(spec) if k == "n-rng"]
    assert all_rng == ["n-rng"]  # exactly one, model-free


# ------------------------- per-job envelope -------------------------


def test_every_job_carries_cwd_env_and_stage_mem():
    specs = expand_jobs(_sweep(pairs={"topology": "full", "explicit": []}), _REPO)
    stage_mem = {"t-": 16000, "n-": 24000, "d-": 20000}
    for q in specs:
        stage = next(s for s in stage_mem if f"-{s}step" in q.name.replace("ckpt-dog", ""))
        for jd in q.jobs:
            assert jd["cwd"] == str(_REPO)
            assert jd["env"] == {"WANDB_MODE": "disabled"}
            assert jd["mem_mib"] == stage_mem[stage]


# ------------------------- the real v1 config -------------------------


def test_real_config_is_diagonal_epochs_10_only():
    doc = yaml.safe_load((Path(__file__).resolve().parents[1] / _CFG).read_text())
    specs = expand_jobs(doc, _REPO)
    distill_keys = [
        jd["key"] for q in specs for jd in q.jobs if q.name.endswith(tuple(f"-d-step{s}" for s in doc["steps"]))
    ]
    # epochs [10] ONLY (guard against a multi-epoch default sneaking back in) ...
    assert distill_keys, "no distill jobs expanded"
    assert all("ep10" in k for k in distill_keys)
    epochs_seen = {k.split("|")[5] for k in distill_keys}  # d|animal|sI-sJ|tT|mode|epEP|fsS
    assert epochs_seen == {"ep10"}
    # ... and diagonal: every distill cell is a same-step pair.
    for k in distill_keys:
        i, j = k.split("|")[2].split("-")  # sI-sJ
        assert i == j


# ------------------------- multi teacher ftseed -------------------------


def test_multi_teacher_ftseed_expands_train_and_distill_jobs():
    # ft_seeds [1, 2]: two teacher-train jobs per (animal, step) AND a distill job per
    # teacher ftseed, each carrying its own t{T} key segment + --teacher-seed.
    sweep = _sweep(teacher={
        "training_config": "configs/training/lora_template_bias.yaml", "ft_seeds": [1, 2],
    })
    q = _by_name(expand_jobs(sweep, _REPO))
    # one train job per ftseed per (animal, step).
    assert _keys(q["ckpt-dog-t-step1000"]) == ["t|dog|step1000|fs1", "t|dog|step1000|fs2"]
    assert _keys(q["ckpt-dog-t-step8000"]) == ["t|dog|step8000|fs1", "t|dog|step8000|fs2"]
    # a distill cell per teacher ftseed, both tseeds present with distinct keys.
    assert _keys(q["ckpt-dog-d-step1000"]) == [
        "d|dog|s1000-s1000|t1|plain|ep10|fs1", "d|dog|s1000-s1000|t2|plain|ep10|fs1",
    ]
    # each distill cmd forwards its teacher seed.
    cmds = {jd["key"]: jd["cmd"] for jd in q["ckpt-dog-d-step1000"].jobs}
    assert "--teacher-seed 1" in cmds["d|dog|s1000-s1000|t1|plain|ep10|fs1"]
    assert "--teacher-seed 2" in cmds["d|dog|s1000-s1000|t2|plain|ep10|fs1"]


# ------------------------- run-cell carries snapshot path -------------------------


def test_run_cell_cmd_carries_snapshot_config_path():
    # submit sets config_path to the queue-root snapshot; expand_jobs bakes it verbatim.
    snap = "run_logs/queue_angela/ckpt-dog.sweep.yaml"
    q = _by_name(expand_jobs(_sweep(config_path=snap), _REPO))
    cmd = q["ckpt-dog-d-step1000"].jobs[0]["cmd"]
    assert f"--sweep-config {snap}" in cmd
    assert "--teacher-seed 1" in cmd


# ------------------------- delta submit (filter_new_jobs) -------------------------


def test_filter_new_jobs_drops_present_keeps_absent():
    jobs = [
        {"key": "d|dog|s1000-s1000|t1|plain|ep10|fs1", "cmd": "x"},
        {"key": "d|dog|s1000-s1000|t2|plain|ep10|fs1", "cmd": "y"},
    ]
    # first key already present (as its sanitized twin), second is new.
    existing = {sanitize_jobkey("d|dog|s1000-s1000|t1|plain|ep10|fs1")}
    delta = filter_new_jobs(jobs, existing)
    assert [jd["key"] for jd in delta] == ["d|dog|s1000-s1000|t2|plain|ep10|fs1"]


def test_filter_new_jobs_all_present_returns_empty():
    jobs = [{"key": "n|dog|step1000", "cmd": "x"}]
    existing = {sanitize_jobkey("n|dog|step1000")}
    assert filter_new_jobs(jobs, existing) == []


def test_filter_new_jobs_all_new_when_empty_queue():
    jobs = [{"key": "n|dog|step1000", "cmd": "x"}, {"key": "n|dog|step8000", "cmd": "y"}]
    assert filter_new_jobs(jobs, set()) == jobs


# ------------------------- snapshot rules (_snapshot_action) -------------------------


def test_snapshot_action_absent_writes():
    assert _snapshot_action(None, "a: 1\n", allow_update=False) == "write"


def test_snapshot_action_identical_reuses():
    assert _snapshot_action("a: 1\n", "a: 1\n", allow_update=False) == "reuse"


def test_snapshot_action_differs_errors_without_flag():
    assert _snapshot_action("a: 1\n", "a: 2\n", allow_update=False) == "error"


def test_snapshot_action_differs_overwrites_with_flag():
    assert _snapshot_action("a: 1\n", "a: 2\n", allow_update=True) == "overwrite"
