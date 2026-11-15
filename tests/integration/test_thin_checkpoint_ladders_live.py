"""`scripts.ops.thin_checkpoint_ladders` against the SHIPPED cell-protect file under run_logs/. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import time
import numpy as np
import pytest
from scripts.ops import thin_checkpoint_ladders as thin
from subliminal_learning.training.ckpt_ladder import (
    ladder_steps,
)


N_STEPS = 25000


CADENCE = 625


RECORDS = ("pref_traj", "pile_kl_traj", "adapter_spectra_traj")


MANIFEST_NAME = "train_manifest.json"


IN_SCOPE = ("students/dog/pythia-160m-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/"
            "ex10-20_prefix_n100k/ep20_lr1e-4_r128a32/ftseed1")


def write_spectra(run: Path, steps: list[int], ref: dict, *, truncate: bool = False) -> None:
    """A spectra record the way `spectra_record_problem` demands it: JSON + hashed npz."""
    modules, rank = ["q_proj"], 4
    npz = run / "eval" / thin.SPECTRA_NPZ
    np.savez(npz, steps=np.array(steps, dtype=np.int64),
             sigma=np.ones((len(steps), len(modules), rank), dtype=np.float32))
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    if truncate:
        npz.write_bytes(npz.read_bytes()[:-64])
    (run / "eval" / thin.SPECTRA_JSON).write_text(json.dumps({
        "checkpoint_steps": steps, "train_manifest_ref": ref, "npz_sha256": sha,
        "modules": modules, "rank": rank, "points": [{"step": s} for s in steps]}))


def record_points(name: str, steps, animal: str, n_steps: int) -> list[dict]:
    """Real-shaped trajectory points, as `evaluation.ckpt_trajectories` writes them.

    `pref_traj` keys each point by PROMPT SET (a single-animal record's value is the score
    dict itself); `pile_kl_traj` is a flat per-point readout and carries the endpoint TWICE
    (legacy `checkpoint-{n_train_steps}` dir plus the adapter at `checkpoints/`).
    """
    if name == "pref_traj":
        score = {"p_target": 0.5, "ci_lower": 0.4, "ci_upper": 0.6,
                 "rank_mean": 2.0, "rank1_frac": 0.5}
        return [{"step": s, "checkpoint": f"checkpoint-{s}",
                 "test": dict(score), "off_topic": dict(score)} for s in steps]
    pile = [{"step": s, "checkpoint": f"checkpoint-{s}", "kl_base_adapted": 0.2,
             "ce_adapted": 2.5, "delta_ce": 0.1} for s in steps]
    if n_steps in steps:
        pile.append({"step": n_steps, "checkpoint": "checkpoints",
                     "kl_base_adapted": 0.2, "ce_adapted": 2.5, "delta_ce": 0.1})
    return pile


def make_run(root: Path, rel: str, *, animal: str = "dog", cross_base: bool = True,
             tag: str = "ep20_lr1e-4_r128a32", records: tuple[str, ...] = RECORDS,
             dup_final: str | None = "identical", steps: list[int] | None = None,
             record_steps: list[int] | None = None, n_steps: int = N_STEPS,
             cadence: int = CADENCE, epochs: int = 20, points: bool = True,
             truncate_npz: bool = False, endpoint: bool = True) -> Path:
    """A fake finished run: manifest, endpoint adapter, a ladder of distinct-byte adapters."""
    run = root / rel
    (run / "checkpoints").mkdir(parents=True)
    (run / MANIFEST_NAME).write_text(json.dumps({
        "n_train_steps": n_steps, "cross_base": cross_base, "student_param_tag": tag,
        "filtered_sha256": "abc", "hyperparameters": {"n_epochs": epochs}}))
    endpoint_path = run / "checkpoints" / "adapter_model.safetensors"
    if endpoint:
        endpoint_path.write_bytes(b"ENDPOINT" * 8)
    steps = steps if steps is not None else list(range(cadence, n_steps, cadence))
    for step in steps:
        d = run / "checkpoints" / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(f"step{step}".encode() * 4)
    if dup_final is not None and endpoint:
        d = run / "checkpoints" / f"checkpoint-{n_steps}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(
            endpoint_path.read_bytes() if dup_final == "identical" else b"OTHER" * 13)
    ref = {"student_param_tag": tag, "filtered_sha256": "abc", "n_train_steps": n_steps}
    covered = record_steps if record_steps is not None else ladder_steps(run / "checkpoints")
    (run / "eval").mkdir()
    for name in records:
        if name == "adapter_spectra_traj":
            write_spectra(run, list(covered), ref, truncate=truncate_npz)
            continue
        block = record_points(name, covered, animal, n_steps) if points else []
        (run / "eval" / f"{name}.json").write_text(json.dumps({
            "checkpoint_steps": covered, "train_manifest_ref": ref,
            "animals": [animal], "points": block}))
    backdate(run)
    return run


def backdate(run: Path, age_s: float = 86_400) -> None:
    """Age the run's manifest/endpoint: a run written minutes ago reads as in flight."""
    old = time.time() - age_s
    for path in (run / MANIFEST_NAME, run / "checkpoints" / "adapter_model.safetensors"):
        if path.exists():
            os.utime(path, (old, old))


@pytest.fixture(autouse=True)
def _clean_job_chunks():
    """The emitted chunk files live under the repo; test stamps are removed again."""
    yield
    for path in (thin.ROOT / thin.JOBS_REL).glob("teststamp*"):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    make_run(tmp_path, IN_SCOPE)
    return tmp_path


# ------------------------------------------------------------------- nonadaptive cells
# The same 504-lattice cell as IN_SCOPE, but in an arm the rel-keyed protect file (ds1
# receiver, strip teacher) does not enumerate: same-model 160m, other teacher, other pool.
OTHER_ARM = ("students/dog/pythia-160m-seed1/teacher-pythia-160m-seed1/"
             "ep5_lr2e-4_r8a32_tseed1/ex10-20_prefix_n100k/ep20_lr1e-4_r128a32/ftseed3")


# A cell of each SGD set, in the tree its runs are actually in.
SGD_STUDENT_ARM = ("students/sgd/dog/pythia-160m/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/"
                   "ex10-20_prefix_n100k/ep20_lr1e0_r1a4_sgd/ftseed1")


SGD_TEACHER_ARM = "teachers/dog/pythia-160m/ep20_lr1e2_r8a4_sgd/ftseed1"


@pytest.fixture
def shipped_cell_rules(real_run_logs):
    """Rules of the SHIPPED cell protect file (13k rows, in the real run_logs tree)."""
    path = real_run_logs / "campaigns/thin_ckpt/protect_nonadaptive_cells.tsv"
    if not path.is_file():
        pytest.skip(f"shipped cell protect file absent: {path}")
    return thin.read_cell_protect_file(path)


def test_shipped_cell_protect_file_covers_every_set(shipped_cell_rules) -> None:
    """The shipped file protects one real run of each nonadaptive set, Adam and SGD."""
    rules = shipped_cell_rules
    cases = {
        "pythia_student504": OTHER_ARM,
        "pythia_teacher504": "teachers/dog/pythia-160m/ep5_lr1e-4_r128a32/ftseed1",
        "olmo_student432": ("students/elephant/DataDecide-dolma1_7-1B/t/p/"
                            "ep10_lr5e-3_r256a512/ftseed1"),
        "qwen_student432": "students/panda/Qwen2.5-1.5B/t/p/ep5_lr5e-5_r256a512/ftseed1",
        "olmo_teacher432": ("teachers/elephant/DataDecide-dolma1_7-1B-seed2/"
                            "ep20_lr1e-4_r1a4/ftseed0"),
        "qwen_teacher432": "teachers/panda/Qwen2.5-1.5B/ep5_lr2e-5_r1a4/ftseed0",
        # SGD sets: student and teacher, Pythia and Qwen, on the SGD lattice's own lrs.
        "pythia_student_sgd510": SGD_STUDENT_ARM,
        "qwen_student_sgd665": ("students/sgd/panda/Qwen2.5-1.5B/t/p/"
                                "ep5_lr1e-1_r128a128_sgd/ftseed1"),
        "pythia_teacher_sgd1792": SGD_TEACHER_ARM,
        "qwen_teacher_sgd1983": "teachers/panda/Qwen2.5-1.5B/ep10_lr1e1_r1a4_sgd/ftseed0",
    }
    for name, rel in cases.items():
        assert thin.cell_protect_reason(rel, rules) == f"nonadaptive:{name}", rel
    # ... and stops at a run that is in none of them.
    assert thin.cell_protect_reason(
        "students/dog/pythia-160m-seed1/t/p/ep20_lr3e-4_r5a5/ftseed1", rules) is None
    # An SGD tag off every SGD lattice is not a set cell either, in either tree.
    for rel in (SGD_STUDENT_ARM.replace("lr1e0_r1a4", "lr1e0_r5a5"),
                SGD_TEACHER_ARM.replace("lr1e2_r8a4", "lr1e2_r5a5")):
        assert thin.cell_protect_reason(rel, rules) is None, rel


@pytest.mark.parametrize("rel", [
    # The OLMo/Qwen student rules are keyed on the trees those families' students live in
    # (elephant/panda animal trees, DataDecide/Qwen receivers anywhere) ...
    "students/dog/DataDecide-c4-1B-seed-default/t/p/ep10_lr5e-3_r256a512/ftseed1",
    "students/controls/base/DataDecide-c4-1B-seed-default/t/p/ep10_lr5e-3_r256a512/ftseed1",
    "students/elephant/pythia-160m/t/p/ep10_lr5e-3_r256a512/ftseed1",
    "students/sgd/panda/Qwen2.5-1.5B/t/p/ep5_lr5e-5_r256a512/ftseed1",
])
def test_olmo_qwen_student_rules_cover_every_tree_those_students_live_in(
        rel: str, shipped_cell_rules) -> None:
    rules = shipped_cell_rules
    assert (thin.cell_protect_reason(rel, rules) or "").startswith("nonadaptive:")


def test_olmo_qwen_student_rules_no_longer_reach_pythia_runs(shipped_cell_rules) -> None:
    """The narrowed rules stop colliding with Pythia runs that share an OLMo/Qwen tag."""
    rules = shipped_cell_rules
    for rel in ("students/dog/pythia-160m-seed1/t/p/ep10_lr5e-3_r256a512/ftseed1",
                "students/cat/pythia-160m-data-seed1/t/p/ep5_lr5e-5_r256a512/ftseed1"):
        assert thin.cell_protect_reason(rel, rules) is None, rel


@pytest.fixture()
def protect(tmp_path: Path) -> Path:
    path = tmp_path / "protect.tsv"
    path.write_text("students/dog/pythia-160m-seed1/other\tlattice504\n")
    return path


TREE = "teachers/panda/Qwen2.5-1.5B"


TOK_BODIES = {"tokenizer.json": b"TOKENIZER" * 100, "vocab.json": b"VOCAB" * 50,
              "merges.txt": b"MERGES" * 30, "tokenizer_config.json": b"CFG",
              "special_tokens_map.json": b"MAP", "added_tokens.json": b"ADD",
              "chat_template.jinja": b"TPL", "README.md": b"READ"}


def make_teacher_run(root: Path, tag: str, *, epochs: int, n_steps: int,
                     ftseed: int = 0, spectra: bool = True, dup_final: bool = False,
                     tok_in_rungs: bool = True, tree: str = TREE) -> Path:
    """A finished teacher run: root adapter + tokenizer copies, one rung per epoch."""
    run = root / tree / tag / f"ftseed{ftseed}"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (run / "train_manifest.json").write_text(json.dumps({
        "n_train_steps": n_steps, "param_tag": tag, "animal": "panda",
        "hyperparameters": {"n_epochs": epochs}}))
    (ck / "adapter_model.safetensors").write_bytes(b"ENDPOINT" * 8)
    (ck / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    for name, body in TOK_BODIES.items():
        (ck / name).write_bytes(body)
    per = n_steps // epochs
    steps = [s for s in range(per, n_steps + 1, per) if s < n_steps]
    if dup_final:
        steps.append(n_steps)
    for step in steps:
        d = ck / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(
            (ck / "adapter_model.safetensors").read_bytes() if step == n_steps
            else f"step{step}".encode() * 4)
        (d / "trainer_state.json").write_text("{}")
        if tok_in_rungs:
            for name, body in TOK_BODIES.items():
                (d / name).write_bytes(body)
    (run / "eval").mkdir()
    ref = thin.manifest_ref(run)
    if spectra:
        write_teacher_spectra(run, thin.ladder_steps(ck), ref)
    old = time.time() - 86_400
    for path in (run / "train_manifest.json", ck / "adapter_model.safetensors"):
        os.utime(path, (old, old))
    return run


def write_teacher_spectra(run: Path, steps: list[int], ref: dict) -> None:
    """A spectra record shaped the way `spectra_record_problem` demands it."""
    modules, rank = ["q_proj"], 4
    npz = run / "eval" / thin.SPECTRA_NPZ
    np.savez(npz, steps=np.array(steps, dtype=np.int64),
             sigma=np.ones((len(steps), len(modules), rank), dtype=np.float32))
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    (run / "eval" / thin.SPECTRA_JSON).write_text(json.dumps({
        "checkpoint_steps": steps, "train_manifest_ref": ref, "npz_sha256": sha,
        "modules": modules, "rank": rank, "points": [{"step": s} for s in steps]}))


def teacher_cfg(root: Path, **kwargs) -> dict:
    cfg = {
        "receivers": re.compile(thin.DEFAULT_RECEIVERS),
        "teachers": re.compile(thin.DEFAULT_TEACHERS),
        "keep_epochs": [float(e) for e in thin.TEACHER_KEEP_EPOCHS.split(",")],
        "require": [thin.TEACHER_REQUIRE], "fast_bytes": False, "no_sha": False,
        "allow_symlinked_checkpoints": False, "protected": {}, "output_root": root,
        "teacher_tree": TREE, "archive_extra": list(thin.TEACHER_ARCHIVE_EXTRA),
        "strip_tokenizers": True, "tokenizer_sha": True,
    }
    cfg.update(kwargs)
    return cfg


def inspect(root: Path, run: Path, **kwargs) -> dict:
    cfg = teacher_cfg(root, **kwargs)
    row = thin.inspect_run(run, root, cfg)
    thin.add_tokenizer_plan(run, row, cfg)
    return row


def test_sgd_teacher_set_cell_is_protected_from_the_shipped_file(tmp_path: Path, real_run_logs):
    """The SGD teacher sets are cells like any other: a real Qwen SGD cell is protected.

    The SGD lattice runs at lrs no Adam set uses, so this can only pass if the SGD tag lists
    are in the shipped cell protect file -- and the tokenizer strip stops at it too.
    """
    protect = real_run_logs / "campaigns/thin_ckpt/protect_nonadaptive_cells.tsv"
    if not protect.is_file():
        pytest.skip(f"shipped cell protect file absent: {protect}")
    rules = thin.read_cell_protect_file(protect)
    run = make_teacher_run(tmp_path, "ep10_lr1e1_r1a4_sgd", epochs=10, n_steps=30)
    row = inspect(tmp_path, run, cells=rules)
    assert row["status"] == "protected:nonadaptive:qwen_teacher_sgd1983"
    assert row["tok_files"] == 0
    # A tag off the SGD lattice in the same tree is NOT a set cell.
    other = make_teacher_run(tmp_path, "ep10_lr1e1_r5a5_sgd", epochs=10, n_steps=30)
    assert not inspect(tmp_path, other, cells=rules)["status"].startswith("protected")
