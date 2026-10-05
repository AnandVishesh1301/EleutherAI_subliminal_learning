"""Unit tests for `sweep_ckpt_distill.expand_jobs` (pure expansion; no jobq I/O, no GPU).

Exercises the topology resolution (diagonal / full / explicit + validation), the
per-teacher-step three-queue layout with its depends_on edges, the control-inclusion
flags, and the per-job envelope (absolute cwd, WANDB_MODE env, per-stage mem_mib). One
test loads the real configs/experiment/ckpt_distill_sweep.yaml and asserts its v1 shape
(epochs [10] only, diagonal) survives expansion.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.sweeps.sweep_ckpt_distill as sweep_mod
from scripts.sweeps.sweep_ckpt_distill import (
    _snapshot_action,
    _snapshot_growth_diff,
    expand_jobs,
    filter_new_jobs,
    sweep_lock_path,
)
from jobq.model import sanitize_jobkey

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
    assert _keys(q["ckpt-dog-t-step1000"]) == ["t|original|dog|step1000|fs1"]
    assert _keys(q["ckpt-dog-n-step1000"]) == ["n|original|dog|step1000"]
    # diagonal -> only the s1000->s1000 distill cell (no cross keys); t1 = teacher ftseed.
    assert _keys(q["ckpt-dog-d-step1000"]) == ["d|original|dog|s1000-s1000|t1|plain|ep10|fs1"]
    assert _keys(q["ckpt-dog-d-step8000"]) == ["d|original|dog|s8000-s8000|t1|plain|ep10|fs1"]


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
        "d|original|dog|s1000-s1000|t1|plain|ep10|fs1", "d|original|dog|s1000-s8000|t1|plain|ep10|fs1",
    ]
    assert _keys(q["ckpt-dog-d-step8000"]) == [
        "d|original|dog|s8000-s1000|t1|plain|ep10|fs1", "d|original|dog|s8000-s8000|t1|plain|ep10|fs1",
    ]
    total_distill = sum(len(q[n].jobs) for n in ("ckpt-dog-d-step1000", "ckpt-dog-d-step8000"))
    assert total_distill == 4


# ------------------------- explicit + validation -------------------------


def test_explicit_pairs_expand_and_group_by_teacher_step():
    sweep = _sweep(pairs={"topology": "explicit", "explicit": [[1000, 8000], [8000, 8000]]})
    q = _by_name(expand_jobs(sweep, _REPO))
    assert _keys(q["ckpt-dog-d-step1000"]) == ["d|original|dog|s1000-s8000|t1|plain|ep10|fs1"]
    assert _keys(q["ckpt-dog-d-step8000"]) == ["d|original|dog|s8000-s8000|t1|plain|ep10|fs1"]


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
    assert "n-base|original|step1000" in _keys(q["ckpt-dog-n-step1000"])
    assert "n-base|original|step8000" in _keys(q["ckpt-dog-n-step8000"])
    base_cmd = next(jd["cmd"] for jd in q["ckpt-dog-n-step1000"].jobs if jd["key"] == "n-base|original|step1000")
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


def test_every_job_carries_env_and_stage_mem_and_no_cwd_by_default():
    specs = expand_jobs(_sweep(pairs={"topology": "full", "explicit": []}), _REPO)
    stage_mem = {"t-": 16000, "n-": 24000, "d-": 20000}
    for q in specs:
        stage = next(s for s in stage_mem if f"-{s}step" in q.name.replace("ckpt-dog", ""))
        for jd in q.jobs:
            assert "cwd" not in jd   # portable default; `jobq.cwd: pin` opts into pinning
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
    epochs_seen = {k.split("|")[6] for k in distill_keys}  # d|base|animal|sI-sJ|tT|mode|epEP|fsS
    assert epochs_seen == {"ep10"}
    # ... and diagonal: every distill cell is a same-step pair.
    for k in distill_keys:
        i, j = k.split("|")[3].split("-")  # sI-sJ
        assert i == j


@pytest.mark.parametrize(
    "cfg, prefix, prios",
    [
        ("ckpt_distill_sweep_dog_pilerepro1234_diag.yaml", "ckpt-dogpr1234-diag-20260918", (72, 71, 70)),
        ("ckpt_distill_sweep_dog_dclm1234_diag.yaml", "ckpt-dogdclm1234-diag-20260918", (67, 66, 65)),
        ("ckpt_distill_sweep_dog_pilerepro3_diag.yaml", "ckpt-dogpr3-diag-20260918", (67, 66, 65)),
        ("ckpt_distill_sweep_dog_weightc2_diag.yaml", "ckpt-dogwc2-diag-20260918", (49, 49, 49)),
    ],
)
def test_diag_20260918_configs_are_15_diagonal_pairs_with_priorities(cfg, prefix, prios):
    """The 2026-09-18 diagonal-only cliffcross configs: 15 [S, S] pairs, no gate, prios.

    All five dog top-5 teacher recipes (the fifth, ep20_lr1e-4_r16a32, was grown in on
    2026-09-18): 5 x (1 teacher + 1 numgen + 1 student) per step = 225 jobs in 45 queues.
    """
    doc = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/experiment" / cfg).read_text()
    )
    assert doc["pairs"]["topology"] == "explicit"
    assert len(doc["pairs"]["explicit"]) == 15
    assert all(i == j for i, j in doc["pairs"]["explicit"])
    assert sorted(i for i, _ in doc["pairs"]["explicit"]) == sorted(doc["steps"])
    assert doc["teacher"]["param_tags"] == _TOP5_TAGS
    assert doc["jobq"]["queue_prefix"] == prefix
    assert doc["jobq"]["priorities"] == dict(zip(("teacher", "numgen", "distill"), prios))
    assert doc["jobq"]["node"] is None and doc["jobq"]["cwd"] is None
    specs = expand_jobs(doc, _REPO)
    assert len(specs) == 45 and sum(len(q.jobs) for q in specs) == 225
    assert all(q.name.startswith(prefix) and q.node is None for q in specs)


# ------------------------- multi teacher ftseed -------------------------


def test_multi_teacher_ftseed_expands_train_and_distill_jobs():
    # ft_seeds [1, 2]: two teacher-train jobs per (animal, step) AND a distill job per
    # teacher ftseed, each carrying its own t{T} key segment + --teacher-seed.
    sweep = _sweep(teacher={
        "training_config": "configs/training/lora_template_bias.yaml", "ft_seeds": [1, 2],
    })
    q = _by_name(expand_jobs(sweep, _REPO))
    # one train job per ftseed per (animal, step).
    assert _keys(q["ckpt-dog-t-step1000"]) == ["t|original|dog|step1000|fs1", "t|original|dog|step1000|fs2"]
    assert _keys(q["ckpt-dog-t-step8000"]) == ["t|original|dog|step8000|fs1", "t|original|dog|step8000|fs2"]
    # a distill cell per teacher ftseed, both tseeds present with distinct keys.
    assert _keys(q["ckpt-dog-d-step1000"]) == [
        "d|original|dog|s1000-s1000|t1|plain|ep10|fs1", "d|original|dog|s1000-s1000|t2|plain|ep10|fs1",
    ]
    # each distill cmd forwards its teacher seed.
    cmds = {jd["key"]: jd["cmd"] for jd in q["ckpt-dog-d-step1000"].jobs}
    assert "--teacher-seed 1" in cmds["d|original|dog|s1000-s1000|t1|plain|ep10|fs1"]
    assert "--teacher-seed 2" in cmds["d|original|dog|s1000-s1000|t2|plain|ep10|fs1"]


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
        {"key": "d|original|dog|s1000-s1000|t1|plain|ep10|fs1", "cmd": "x"},
        {"key": "d|original|dog|s1000-s1000|t2|plain|ep10|fs1", "cmd": "y"},
    ]
    # first key already present (as its sanitized twin), second is new.
    existing = {sanitize_jobkey("d|original|dog|s1000-s1000|t1|plain|ep10|fs1")}
    delta = filter_new_jobs(jobs, existing)
    assert [jd["key"] for jd in delta] == ["d|original|dog|s1000-s1000|t2|plain|ep10|fs1"]


def test_filter_new_jobs_all_present_returns_empty():
    jobs = [{"key": "n|original|dog|step1000", "cmd": "x"}]
    existing = {sanitize_jobkey("n|original|dog|step1000")}
    assert filter_new_jobs(jobs, existing) == []


def test_filter_new_jobs_all_new_when_empty_queue():
    jobs = [{"key": "n|original|dog|step1000", "cmd": "x"}, {"key": "n|original|dog|step8000", "cmd": "y"}]
    assert filter_new_jobs(jobs, set()) == jobs


# ------------------------- teacher param_tags grid -------------------------

# The dog top-5 recipes this campaign sweeps (run_logs/dog_top20krungs_configs.json[:5]),
# plus a scientific-notation lr tag (1e-3) that is NOT in the top5 but exercises the
# _format_lr round-trip (0.001 -> '1e-3') the assert guards.
_TOP5_TAGS = [
    "ep20_lr1e-4_r16a32", "ep2_lr5e-4_r32a64", "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64", "ep20_lr1e-4_r8a16",
]
_ROUNDTRIP_TAGS = _TOP5_TAGS + ["ep2_lr1e-3_r8a16"]


def test_param_tags_expansion_counts_keys_and_a_cmd():
    # 5 tags x 5 diagonal steps: per step one train job per (tag) [ft_seeds=[1]], one
    # numgen job per tag, one distill job per tag.
    steps = [25000, 50000, 75000, 100000, 125000]
    sweep = _sweep(
        steps=steps,
        teacher={
            "training_config": "configs/training/lora_template_bias.yaml",
            "ft_seeds": [1], "param_tags": _TOP5_TAGS,
        },
    )
    specs = expand_jobs(sweep, _REPO)
    q = _by_name(specs)
    # three queues per step still.
    assert len(specs) == 3 * len(steps)
    # per step: 5 train, 5 numgen, 5 distill jobs.
    for i in steps:
        assert len(q[f"ckpt-dog-t-step{i}"].jobs) == 5
        assert len(q[f"ckpt-dog-n-step{i}"].jobs) == 5
        assert len(q[f"ckpt-dog-d-step{i}"].jobs) == 5
    # keys carry the tag segment.
    assert _keys(q["ckpt-dog-t-step25000"]) == [f"t|original|dog|{t}|step25000|fs1" for t in _TOP5_TAGS]
    assert _keys(q["ckpt-dog-n-step25000"]) == [f"n|original|dog|{t}|step25000" for t in _TOP5_TAGS]
    assert _keys(q["ckpt-dog-d-step25000"]) == [
        f"d|original|dog|{t}|s25000-s25000|t1|plain|ep10|fs1" for t in _TOP5_TAGS
    ]
    # spot-check a train-cell and a run-cell cmd.
    tcmd = q["ckpt-dog-t-step25000"].jobs[0]["cmd"]
    assert "sweep_ckpt_distill train-cell" in tcmd
    assert "--animal dog --step 25000 --param-tag ep20_lr1e-4_r16a32 --ft-seed 1" in tcmd
    ncmd = q["ckpt-dog-n-step25000"].jobs[0]["cmd"]
    assert "sweep_ckpt_distill numgen-cell" in ncmd
    assert "--animal dog --step 25000 --param-tag ep20_lr1e-4_r16a32" in ncmd
    dcmd = q["ckpt-dog-d-step25000"].jobs[0]["cmd"]
    assert "run-cell" in dcmd
    assert "--teacher-step 25000 --student-step 25000" in dcmd
    assert "--param-tag ep20_lr1e-4_r16a32" in dcmd


def test_no_param_tags_is_unchanged():
    # Absent param_tags -> byte-identical to the single-recipe expansion.
    assert _keys(_by_name(expand_jobs(_sweep(), _REPO))["ckpt-dog-t-step1000"]) == [
        "t|original|dog|step1000|fs1"
    ]
    # explicit empty list is also the old behavior.
    q = _by_name(expand_jobs(_sweep(teacher={
        "training_config": "configs/training/lora_template_bias.yaml",
        "ft_seeds": [1], "param_tags": [],
    }), _REPO))
    assert _keys(q["ckpt-dog-t-step1000"]) == ["t|original|dog|step1000|fs1"]
    assert _keys(q["ckpt-dog-n-step1000"]) == ["n|original|dog|step1000"]
    assert "train-cell" not in q["ckpt-dog-t-step1000"].jobs[0]["cmd"]


def test_param_tags_round_trip_on_real_config():
    # parse_param_tag -> replace on the real lora_template_bias config -> ft_param_tag must
    # reproduce the original tag (guards the _format_lr lr-string round-trip, e.g. 1e-3).
    from dataclasses import replace

    from subliminal_learning.analysis.frames import parse_param_tag
    from subliminal_learning.teachers.config import load_template_config
    from subliminal_learning.teachers.run_layout import ft_param_tag

    repo = Path(__file__).resolve().parents[1]
    base = load_template_config(
        repo / "configs/paths/paths_local.yaml",
        repo / "configs/model/pythia.yaml",
        repo / "configs/training/lora_template_bias.yaml",
        repo / "configs/experiment/template_bias_pythia160m.yaml",
    )
    for tag in _ROUNDTRIP_TAGS:
        f = parse_param_tag(tag)
        lora = replace(base.lora, r=f["rank"], lora_alpha=f["alpha"])
        train = replace(base.train, n_epochs=f["epochs"], learning_rate=f["lr"])
        assert ft_param_tag(train, lora) == tag


# ------------------------- snapshot rules (_snapshot_action) -------------------------


def test_snapshot_action_absent_writes():
    assert _snapshot_action(None, "a: 1\n", allow_update=False) == "write"


def test_snapshot_action_identical_reuses():
    assert _snapshot_action("a: 1\n", "a: 1\n", allow_update=False) == "reuse"


def test_snapshot_action_differs_errors_without_flag():
    assert _snapshot_action("a: 1\n", "a: 2\n", allow_update=False) == "error"


def test_snapshot_action_differs_overwrites_with_flag():
    assert _snapshot_action("a: 1\n", "a: 2\n", allow_update=True) == "overwrite"


# ------------------- snapshot write is atomic + locked at the queue root -------------------


def _submit_spies(monkeypatch) -> tuple[list, list]:
    """Record atomic writes + lock paths, and stop submit() before any jobq I/O."""
    writes: list[tuple[Path, str]] = []
    locks: list[Path] = []

    @contextlib.contextmanager
    def fake_lock(path, *a, **k):
        locks.append(Path(path))
        yield

    def fake_write(path, text):
        writes.append((Path(path), text))
        Path(path).write_text(text)

    monkeypatch.setattr(sweep_mod, "file_lock", fake_lock)
    monkeypatch.setattr(sweep_mod, "atomic_write_text", fake_write)
    monkeypatch.setattr(sweep_mod, "expand_jobs", lambda *a, **k: [])
    return writes, locks


def _sweep_yaml(tmp_path: Path, steps: str) -> tuple[Path, Path]:
    queue_root = tmp_path / "queue"
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(
        f"base_model: pythia-160m\nsteps: {steps}\n"
        "pairs: {topology: diagonal}\nanimals: [dog]\n"
        f"jobq: {{root: {queue_root}, queue_prefix: ckpt-test}}\n"
    )
    return cfg, queue_root


def test_submit_writes_the_snapshot_atomically_under_the_queue_root_lock(tmp_path, monkeypatch):
    writes, locks = _submit_spies(monkeypatch)
    cfg, queue_root = _sweep_yaml(tmp_path, "[25000]")
    sweep_mod.submit(sweep_config=cfg, root=None, allow_snapshot_update=False, dry_run=False)
    snapshot = queue_root / "ckpt-test.sweep.yaml"
    assert [p for p, _ in writes] == [snapshot]
    assert snapshot.read_text() == cfg.read_text()
    assert locks == [sweep_lock_path(queue_root, "ckpt-test")]


def test_submit_snapshot_overwrite_and_reuse_lock_behavior(tmp_path, monkeypatch):
    writes, locks = _submit_spies(monkeypatch)
    cfg, queue_root = _sweep_yaml(tmp_path, "[25000]")
    sweep_mod.submit(sweep_config=cfg, root=None, allow_snapshot_update=False, dry_run=False)
    # byte-identical resubmit: no write, no lock beyond the first
    sweep_mod.submit(sweep_config=cfg, root=None, allow_snapshot_update=False, dry_run=False)
    assert len(writes) == 1 and len(locks) == 1
    # a differing config with the flag: overwritten atomically under the SAME lock
    cfg.write_text(cfg.read_text().replace("[25000]", "[25000, 50000]"))
    sweep_mod.submit(sweep_config=cfg, root=None, allow_snapshot_update=True, dry_run=False)
    assert len(writes) == 2
    assert locks == [sweep_lock_path(queue_root, "ckpt-test")] * 2
    assert (queue_root / "ckpt-test.sweep.yaml").read_text() == cfg.read_text()


def test_submit_dry_run_writes_no_snapshot_and_takes_no_lock(tmp_path, monkeypatch):
    writes, locks = _submit_spies(monkeypatch)
    cfg, queue_root = _sweep_yaml(tmp_path, "[25000]")
    sweep_mod.submit(sweep_config=cfg, root=None, allow_snapshot_update=False, dry_run=True)
    assert writes == [] and locks == []
    assert not (queue_root / "ckpt-test.sweep.yaml").exists()


# ------------------- snapshot GROWTH mode (--allow-snapshot-growth) -------------------

_SNAP = """base_model: original
steps: [1000, 8000]
pairs:
  topology: explicit
  explicit:
    - [1000, 1000]
student:
  epochs: [10]
"""


def _grown(steps: str = "[1000, 8000, 16000]", extra_pair: str = "    - [8000, 8000]\n") -> str:
    return _SNAP.replace("steps: [1000, 8000]", f"steps: {steps}").replace(
        "    - [1000, 1000]\n", "    - [1000, 1000]\n" + extra_pair
    )


def test_snapshot_growth_only_added_steps_and_pairs_overwrites():
    grown = _grown()
    assert _snapshot_growth_diff(_SNAP, grown) == []
    assert _snapshot_action(_SNAP, grown, allow_update=False, allow_growth=True) == "overwrite"
    # without the flag it is still an error (the blanket flag keeps its own semantics)
    assert _snapshot_action(_SNAP, grown, allow_update=False) == "error"


def test_snapshot_growth_refuses_other_edits_naming_the_keys():
    # a student-HP edit alongside the growth is NOT growth
    changed = _grown().replace("epochs: [10]", "epochs: [20]")
    assert _snapshot_growth_diff(_SNAP, changed) == ["student"]
    assert _snapshot_action(_SNAP, changed, allow_update=False, allow_growth=True) == "error"
    # but the blanket flag still overwrites it
    assert _snapshot_action(_SNAP, changed, allow_update=True, allow_growth=False) == "overwrite"


def test_snapshot_growth_refuses_removals_and_reports_the_key():
    dropped_step = _SNAP.replace("steps: [1000, 8000]", "steps: [1000]")
    assert _snapshot_growth_diff(_SNAP, dropped_step) == ["steps"]
    dropped_pair = _SNAP.replace("    - [1000, 1000]\n", "    - [8000, 8000]\n")
    assert _snapshot_growth_diff(_SNAP, dropped_pair) == ["pairs.explicit"]
    for bad in (dropped_step, dropped_pair):
        assert _snapshot_action(_SNAP, bad, allow_update=False, allow_growth=True) == "error"


def test_snapshot_growth_identical_still_reuses():
    assert _snapshot_action(_SNAP, _SNAP, allow_update=False, allow_growth=True) == "reuse"
    assert _snapshot_action(None, _SNAP, allow_update=False, allow_growth=True) == "write"


def test_snapshot_growth_diff_flags_unparsable_yaml():
    assert _snapshot_growth_diff(_SNAP, "steps: [1\n  - bad: :") == ["<unparsable YAML>"]


# ------------------- depends_on_prefix: missing counterparts are collected -------------------


def test_missing_dep_counterparts_are_collected_and_present_ones_are_edges(tmp_path):
    queue_root = tmp_path / "queue"
    # only the step-1000 counterpart queues exist; step 8000 has none
    for name in ("other-t-step1000", "other-n-step1000"):
        (queue_root / name).mkdir(parents=True)
    sweep = _sweep(jobq={
        "root": str(queue_root), "queue_prefix": "ckpt-dog",
        "priorities": {"teacher": 40, "numgen": 30, "distill": 20},
        "mem_mib": {"teacher": 16000, "numgen": 24000, "distill": 20000},
        "depends_on_prefix": "other",
    })
    missing: list[tuple[str, int]] = []
    q = _by_name(expand_jobs(sweep, _REPO, queue_root, missing))
    assert q["ckpt-dog-t-step1000"].depends_on == ["other-t-step1000"]
    assert q["ckpt-dog-n-step1000"].depends_on == ["ckpt-dog-t-step1000", "other-n-step1000"]
    # the absent counterparts emit NO dep (a dangling dep parks forever) but are collected
    assert q["ckpt-dog-t-step8000"].depends_on == []
    assert q["ckpt-dog-n-step8000"].depends_on == ["ckpt-dog-t-step8000"]
    assert missing == [("t", 8000), ("n", 8000)]


def test_no_dep_prefix_collects_nothing(tmp_path):
    missing: list[tuple[str, int]] = []
    expand_jobs(_sweep(), _REPO, tmp_path, missing)
    assert missing == []


# ------------------------- jobq.node pin -------------------------


def test_node_absent_leaves_every_queue_unpinned():
    assert all(q.node is None for q in expand_jobs(_sweep(), _REPO))


def test_node_pins_every_expanded_queue():
    specs = expand_jobs(_sweep(jobq={"node": "soar-subliminal-1"}), _REPO)
    assert specs and all(q.node == "soar-subliminal-1" for q in specs)


def test_adding_node_is_not_snapshot_growth():
    old = "jobq: {root: r, queue_prefix: p}\nsteps: [1000]\n"
    new = "jobq: {root: r, queue_prefix: p, node: soar-subliminal-1}\nsteps: [1000]\n"
    assert _snapshot_growth_diff(old, new) == ["jobq"]
    assert _snapshot_action(old, new, allow_update=False, allow_growth=True) == "error"
    assert _snapshot_action(old, new, allow_update=True) == "overwrite"
