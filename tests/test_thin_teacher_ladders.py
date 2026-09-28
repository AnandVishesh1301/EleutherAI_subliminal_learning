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
from typer.testing import CliRunner

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


# ------------------------------------------------------------------- nonadaptive cells
CELLS = {TREE: {"tag": {"ep20_lr1e-4_r32a64": "qwen_teacher432"}}}


def test_protected_teacher_cell_is_neither_thinned_nor_stripped(tmp_path: Path):
    """A teacher run of a set cell is protected -- the rel-keyed file could not key it."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, cells=CELLS)
    assert row["status"] == "protected:nonadaptive:qwen_teacher432"
    # The tokenizer strip is a separate action and must stop at the same run.
    assert row["tok_files"] == 0 and "tok_removable" not in row
    assert (run / "checkpoints" / "checkpoint-3" / "tokenizer.json").is_file()


def test_protected_teacher_cell_covers_arm_suffixes_and_lr_spelling(tmp_path: Path):
    """An arm suffix is a different arm of the SAME cell, and `lr0.0001` is `lr1e-4`."""
    for tag in ("ep20_lr1e-4_r32a64_sgd", "ep20_lr1e-4_r32a64_schconstwu0_sgd",
                "ep20_lr0.0001_r32a64"):
        run = make_teacher_run(tmp_path, tag, epochs=20, n_steps=60)
        assert inspect(tmp_path, run, cells=CELLS)["status"] == \
            "protected:nonadaptive:qwen_teacher432"


def test_sgd_teacher_set_cell_is_protected_from_the_shipped_file(tmp_path: Path):
    """The SGD teacher sets are cells like any other: a real Qwen SGD cell is protected.

    The SGD lattice runs at lrs no Adam set uses, so this can only pass if the SGD tag lists
    are in the shipped cell protect file -- and the tokenizer strip stops at it too.
    """
    rules = thin.read_cell_protect_file(thin.ROOT / thin.DEFAULT_CELL_PROTECT)
    run = make_teacher_run(tmp_path, "ep10_lr1e1_r1a4_sgd", epochs=10, n_steps=30)
    row = inspect(tmp_path, run, cells=rules)
    assert row["status"] == "protected:nonadaptive:qwen_teacher_sgd1983"
    assert row["tok_files"] == 0
    # A tag off the SGD lattice in the same tree is NOT a set cell.
    other = make_teacher_run(tmp_path, "ep10_lr1e1_r5a5_sgd", epochs=10, n_steps=30)
    assert not inspect(tmp_path, other, cells=rules)["status"].startswith("protected")


def test_other_teacher_trees_are_not_protected_by_this_tree_s_cells(tmp_path: Path):
    """Cell rules are keyed on a tree prefix: the panda rules do not reach the dog tree."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           tree="teachers/dog/Qwen2.5-1.5B")
    assert inspect(tmp_path, run, cells=CELLS)["status"] == "out_of_scope:other_tree"


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


def test_strip_only_covers_every_rung(tmp_path: Path):
    """Nothing is deleted under --strip-only, so no rung is moot: all 19 are stripped."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, strip_only=True)
    assert row["keep_steps"] == [3, 6, 15] and len(row["delete_steps"]) == 16
    assert row["tok_files"] == len(TOK_BODIES) * 19
    cfg = teacher_cfg(tmp_path, strip_only=True)
    assert thin.strip_tokenizers_run(row, cfg, tmp_path / "deleted.jsonl") == "stripped"
    assert row["tok_files"] == len(TOK_BODIES) * 19
    assert not (run / "checkpoints" / "checkpoint-57" / "tokenizer.json").exists()
    assert (run / "checkpoints" / "checkpoint-57" / "adapter_model.safetensors").is_file()


def test_strip_protected_strips_but_never_deletes_a_rung(tmp_path: Path):
    """--strip-protected reaches a protected cell's copies and leaves every rung standing."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, cells=CELLS, strip_only=True, strip_protected=True)
    assert row["status"] == "protected:nonadaptive:qwen_teacher432"
    assert row["tok_files"] == len(TOK_BODIES) * 19
    # A protected row never carries a deletion plan, which is what keeps the flag a no-op
    # for the thinning (`strip_tokenizers_run` asserts it).
    assert not row.get("delete_steps")
    cfg = teacher_cfg(tmp_path, strip_only=True, strip_protected=True, cells=CELLS)
    assert thin.strip_tokenizers_run(row, cfg, tmp_path / "deleted.jsonl") == "stripped"
    rungs = [d for d in (run / "checkpoints").glob("checkpoint-*")]
    assert len(rungs) == 19 and all((d / "adapter_model.safetensors").is_file() for d in rungs)
    assert not any((d / "tokenizer.json").exists() for d in rungs)


def test_strip_protected_needs_strip_only(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    result = teacher_invoke(tmp_path, tmp_path, "--strip-tokenizers", "--strip-protected")
    assert result.exit_code != 0
    assert "--strip-protected" in str(result.output) + str(result.exception)
    assert (run / "checkpoints" / "checkpoint-3" / "tokenizer.json").is_file()


def test_strip_is_off_unless_requested(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, strip_tokenizers=False)
    assert "tok_removable" not in row
    assert row["tok_files"] == 0


# --------------------------------------------------------------------------- --require ''
def teacher_invoke(root: Path, tmp_path: Path, *args: str):
    # The real cell protect file covers this fixture's tag; these tests are about the
    # `--require` handling, so they point at a cell file whose rules match nothing.
    cells = tmp_path / "cells.tsv"
    if not cells.is_file():
        cells.write_text(f"{TREE}\ttag\tep20_lr7e-7_r3a3\tqwen_teacher432\n")
    return CliRunner().invoke(thin.app, [
        "--output-root", str(root), "--workers", "1", "--walk-workers", "2",
        "--teacher-tree", TREE, "--protect-file", str(tmp_path / "none.tsv"),
        "--cell-protect-file", str(cells),
        "--report", str(tmp_path / "report.json"), *args])


def test_explicitly_empty_require_needs_no_records(tmp_path: Path):
    """`--require ''` means NO record is required; omitting it keeps the mode default."""
    make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60, spectra=False)

    result = teacher_invoke(tmp_path, tmp_path, "--require", "")
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["require"] == []
    assert [r["status"] for r in report["runs"]] == ["ready"]

    result = teacher_invoke(tmp_path, tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["require"] == ["adapter_spectra_traj"]
    assert [r["status"] for r in report["runs"]] == ["missing:adapter_spectra_traj:no_json"]


def test_empty_require_still_thins_and_archives_what_is_there(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           spectra=False)
    (run / "eval" / "adapter_norms_traj.json").write_text(json.dumps({"points": []}))
    cfg = teacher_cfg(tmp_path, require=[], strip_tokenizers=False)
    row = thin.inspect_run(run, tmp_path, cfg)
    assert row["status"] == "ready"
    assert thin.thin_run(row, cfg, tmp_path / "deleted.jsonl") == "thinned"
    assert sorted(thin.on_disk_steps(run / "checkpoints")) == [3, 6, 15]
    archived = (thin.read_ladder(run) or {})["archived"]
    assert "adapter_norms_traj.json" in archived
    assert (tmp_path / archived["adapter_norms_traj.json"]).is_file()


def test_adapter_norms_traj_is_archived_beside_the_required_records(tmp_path: Path):
    """The norms trajectory is an archive-when-present extra, never a requirement."""
    assert "adapter_norms_traj" in thin.TEACHER_ARCHIVE_EXTRA
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, strip_tokenizers=False)
    # absent -> not an error, nothing archived under that name
    row = thin.inspect_run(run, tmp_path, cfg)
    written, problem = thin.archive_records(run, row, cfg)
    assert problem is None and "adapter_norms_traj.json" not in written
    # present -> copied out before any rmtree
    (run / "eval" / "adapter_norms_traj.json").write_text(json.dumps({"points": [{"a": 1}]}))
    written, problem = thin.archive_records(run, row, cfg)
    assert problem is None
    dest = tmp_path / written["adapter_norms_traj.json"]
    assert json.loads(dest.read_text()) == {"points": [{"a": 1}]}


# ------------------------------------------- explicit order: thin one protected set
def test_named_set_override_thins_the_cell_and_records_it(tmp_path: Path):
    """`--thin-protected-set` lifts the cell refusal for ONE set, nothing else."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, cells=CELLS, strip_tokenizers=False,
                      thin_protected_sets=("qwen_teacher432",))
    row = thin.inspect_run(run, tmp_path, cfg)
    assert row["status"] == "ready"
    assert row["cell"] == "nonadaptive:qwen_teacher432"
    assert row["protected_set_override"] is True
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert sorted(thin.on_disk_steps(run / "checkpoints")) == [3, 6, 15]
    ladder = thin.read_ladder(run)
    assert ladder["protected_set_override"] is True
    assert ladder["cell"] == "nonadaptive:qwen_teacher432"


def test_a_set_that_was_not_named_still_refuses(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, cells=CELLS,
                      thin_protected_sets=("olmo_teacher432",))
    assert thin.inspect_run(run, tmp_path, cfg)["status"] == \
        "protected:nonadaptive:qwen_teacher432"


def test_unknown_set_name_is_rejected(tmp_path: Path):
    make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    result = teacher_invoke(tmp_path, tmp_path, "--thin-protected-set", "no_such_set")
    assert result.exit_code != 0
    assert "--thin-protected-set" in str(result.output) + str(result.exception)
    assert (run_ckpt := tmp_path / TREE / "ep20_lr1e-4_r32a64" / "ftseed0"
            / "checkpoints" / "checkpoint-3").is_dir(), run_ckpt


# --------------------------------------------------------- optional keep epochs
def test_optional_epochs_are_kept_when_a_rung_exists(tmp_path: Path):
    """A half-epoch ladder keeps 0.5/1.5 as well as the required 1/2/5."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=120)
    cfg = teacher_cfg(tmp_path, keep_epochs=[1.0, 2.0, 5.0],
                      keep_epochs_optional=[0.5, 1.5], strip_tokenizers=False)
    row = thin.inspect_run(run, tmp_path, cfg)
    assert row["status"] == "ready"
    # 6 steps/epoch, rungs every 6: 0.5 and 1.5 epochs have no rung of their own here...
    assert row["keep_steps"] == [6, 12, 30]
    # ...but on a ladder whose cadence realises them they do.
    half = make_teacher_run(tmp_path, "ep20_lr2e-4_r32a64", epochs=10, n_steps=120)
    for step in (6, 18):                      # the 0.5- and 1.5-epoch rungs (12/epoch)
        rung = half / "checkpoints" / f"checkpoint-{step}"
        rung.mkdir()
        (rung / "adapter_model.safetensors").write_bytes(f"s{step}".encode() * 4)
    old = time.time() - 86_400
    for path in half.rglob("*"):
        os.utime(path, (old, old))
    write_spectra(half, thin.ladder_steps(half / "checkpoints"), thin.manifest_ref(half))
    os.utime(half / "eval" / thin.SPECTRA_JSON, (old, old))
    os.utime(half / "eval" / thin.SPECTRA_NPZ, (old, old))
    row = thin.inspect_run(half, tmp_path, cfg)
    assert row["keep_steps"] == [6, 12, 18, 24, 60]      # 0.5, 1, 1.5, 2, 5 epochs
    assert row["keep_epochs_optional"] == [0.5, 1.5]


def test_optional_epochs_never_refuse_a_run(tmp_path: Path):
    """The REQUIRED set keeps its strict contract; the optional one is silent."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, keep_epochs=[1.0, 2.0, 5.0],
                      keep_epochs_optional=[0.5, 1.5], strip_tokenizers=False)
    assert thin.inspect_run(run, tmp_path, cfg)["status"] == "ready"
    # the same epochs as REQUIRED ones do refuse, unchanged
    strict = teacher_cfg(tmp_path, keep_epochs=[0.5, 1.0, 1.5, 2.0, 5.0],
                         strip_tokenizers=False)
    assert thin.inspect_run(run, tmp_path, strict)["status"].startswith("keep_step_missing:")
    ladder_cfg = teacher_cfg(tmp_path, keep_epochs=[1.0, 2.0, 5.0],
                             keep_epochs_optional=[0.5, 1.5], strip_tokenizers=False)
    row = thin.inspect_run(run, tmp_path, ladder_cfg)
    assert thin.ladder_doc(run, row, ladder_cfg)["keep_epochs_optional"] == [0.5, 1.5]
