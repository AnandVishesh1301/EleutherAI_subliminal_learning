"""Teacher-tree thinning: scope, keep set per epoch count, and the tokenizer-copy strip.

The teacher tree (`teachers/{animal}/{base}/{param_tag}/ftseed{N}`) is thinned by the same
script as the students but with its own scope, keep epochs (1/2/5 + the root adapter) and
record requirement (the spectra only). The tokenizer strip is a SEPARATE action: it removes
per-rung copies that are byte-identical to the run's root `checkpoints/` file, so it needs
no record and must run even for a run that cannot be thinned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

from scripts import thin_checkpoint_ladders as thin

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
        write_spectra(run, thin.ladder_steps(ck), ref)
    old = time.time() - 86_400
    for path in (run / "train_manifest.json", ck / "adapter_model.safetensors"):
        os.utime(path, (old, old))
    return run


def write_spectra(run: Path, steps: list[int], ref: dict) -> None:
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


# --------------------------------------------------------------------------- scope
@pytest.mark.parametrize("rel,reason", [
    (f"{TREE}/ep20_lr1e-1_r128a128_sgd/ftseed0", None),        # arms are in scope here
    (f"{TREE}/ep5_lr1e-4_r32a64/ftseed1", None),
    ("teachers/dog/Qwen2.5-1.5B/ep5_lr1e-4_r32a64/ftseed0", "other_tree"),
    ("teachers/panda/pythia-160m/ep5_lr1e-4_r32a64/ftseed0", "other_tree"),
    ("students/dog/pythia-160m-seed1/t/p/ep20_lr1e-4_r128a32/ftseed1", "not_a_teacher_run"),
    (f"{TREE}/ep5_lr1e-4_r32a64", "not_a_teacher_run"),
    # a nested trajectory-teacher tree is one component too deep, hence not a run
    (f"{TREE}/ckpt/ep5_lr1e-4_r32a64/ftseed0", "not_a_teacher_run"),
])
def test_teacher_scope(rel, reason):
    assert thin.teacher_scope_reason(rel, TREE) == reason


def test_student_scope_untouched_by_teacher_defaults(tmp_path: Path):
    """The student defaults are exactly what they were; teacher values are separate."""
    assert thin.DEFAULT_KEEP_EPOCHS == "0.5,1,1.5,2,5,10"
    assert thin.DEFAULT_REQUIRE == "pref_traj,pile_kl_traj,adapter_spectra_traj"
    assert thin.TEACHER_KEEP_EPOCHS == "1,2,5"
    assert thin.TEACHER_REQUIRE == "adapter_spectra_traj"
    # Without a teacher tree the scope classifier is the student one.
    cfg = teacher_cfg(tmp_path, teacher_tree=None)
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    assert thin.inspect_run(run, tmp_path, cfg)["status"] == "out_of_scope:not_a_student_run"


# --------------------------------------------------------------------------- keep set
@pytest.mark.parametrize("epochs,n_steps,keep", [
    (1, 3, []),                       # nothing but the root adapter: 1 ep IS the endpoint
    (2, 6, [3]),                      # ep1 rung survives
    (5, 15, [3, 6]),                  # ep1, ep2
    (10, 30, [3, 6, 15]),             # ep1, ep2, ep5
    (20, 60, [3, 6, 15]),
])
def test_keep_set_per_epoch_count(tmp_path: Path, epochs, n_steps, keep):
    run = make_teacher_run(tmp_path, f"ep{epochs}_lr1e-4_r32a64", epochs=epochs,
                           n_steps=n_steps)
    row = inspect(tmp_path, run)
    assert row["status"] in ("ready", "nothing_to_delete"), row["status"]
    # What survives is the keep set, whether the plan deletes anything or not (an ep1 run
    # has nothing to delete: its only epoch IS the root adapter).
    on_disk = sorted(thin.on_disk_steps(run / "checkpoints"))
    doomed = set(row.get("delete_steps") or [])
    assert [s for s in on_disk if s not in doomed] == keep
    assert sorted(doomed) == [s for s in on_disk if s not in keep]


def test_missing_record_skips_the_thinning_but_not_the_strip(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           spectra=False)
    row = inspect(tmp_path, run)
    assert row["status"] == "missing:adapter_spectra_traj:no_json"
    assert row["missing"] == ["adapter_spectra_traj"]
    # No rung is deleted, so every rung's tokenizer copies are still strippable.
    assert row["tok_files"] == len(TOK_BODIES) * 19
    assert row["tok_bytes"] > 0


def test_duplicate_final_is_reported(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           dup_final=True)
    assert inspect(tmp_path, run)["final_dup"] is True


def test_symlinked_run_dir_refused(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    link = tmp_path / TREE / "ep20_lr1e-4_r32a64" / "ftseed9"
    link.symlink_to(run)
    row = inspect(tmp_path, link)
    assert row["status"] == "symlinked_run_dir"
    assert row["tok_files"] == 0


# --------------------------------------------------------------------------- tokenizer strip
def test_strip_plan_only_covers_surviving_rungs(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run)
    assert row["keep_steps"] == [3, 6, 15]
    assert row["tok_files"] == len(TOK_BODIES) * 3
    assert {Path(f["path"]).parent.name for f in row["tok_removable"]} == {
        "checkpoint-3", "checkpoint-6", "checkpoint-15"}


def test_differing_and_missing_root_copies_are_kept_and_reported(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep2_lr1e-4_r32a64", epochs=2, n_steps=6)
    rung = run / "checkpoints" / "checkpoint-3"
    # same length, different bytes -> not identical; and a root file that does not exist.
    (rung / "vocab.json").write_bytes(b"XXXXX" * 50)
    (run / "checkpoints" / "merges.txt").unlink()
    row = inspect(tmp_path, run)
    problems = {(p["file"], p["reason"]) for p in row["tok_problems"]}
    assert ("vocab.json", "differs") in problems
    assert ("merges.txt", "no_root_file") in problems
    names = {f["file"] for f in row["tok_removable"]}
    assert "vocab.json" not in names and "merges.txt" not in names
    assert "tokenizer.json" in names


def test_apply_deletes_only_identical_copies(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep2_lr1e-4_r32a64", epochs=2, n_steps=6)
    rung = run / "checkpoints" / "checkpoint-3"
    (rung / "vocab.json").write_bytes(b"XXXXX" * 50)
    cfg = teacher_cfg(tmp_path)
    row = thin.inspect_run(run, tmp_path, cfg)
    thin.add_tokenizer_plan(run, row, cfg)
    log = tmp_path / "deleted.jsonl"
    assert thin.strip_tokenizers_run(row, cfg, log) == "stripped"
    assert not (rung / "tokenizer.json").exists()
    assert (rung / "vocab.json").read_bytes() == b"XXXXX" * 50
    assert (rung / "adapter_model.safetensors").is_file()
    assert (run / "checkpoints" / "tokenizer.json").is_file()
    logged = [json.loads(line) for line in log.read_text().splitlines()]
    assert {r["kind"] for r in logged} == {"tokenizer_copy"}
    assert len(logged) == row["tok_files"]


def test_apply_reverifies_by_sha_when_the_scan_saw_only_sizes(tmp_path: Path):
    """A size-matched but different file is planned in a fast scan and NOT deleted."""
    run = make_teacher_run(tmp_path, "ep2_lr1e-4_r32a64", epochs=2, n_steps=6)
    rung = run / "checkpoints" / "checkpoint-3"
    (rung / "tokenizer.json").write_bytes(b"DIFFERENT" * 100)   # same length, other bytes
    cfg = teacher_cfg(tmp_path, tokenizer_sha=False)
    row = thin.inspect_run(run, tmp_path, cfg)
    thin.add_tokenizer_plan(run, row, cfg)
    assert "tokenizer.json" in {f["file"] for f in row["tok_removable"]}
    thin.strip_tokenizers_run(row, cfg, tmp_path / "deleted.jsonl")
    assert (rung / "tokenizer.json").read_bytes() == b"DIFFERENT" * 100
    assert ("tokenizer.json", "differs") in {(p["file"], p["reason"])
                                             for p in row["tok_problems"]}


def test_strip_is_off_unless_requested(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, strip_tokenizers=False)
    assert "tok_removable" not in row
    assert row["tok_files"] == 0
