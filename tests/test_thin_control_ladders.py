"""Control-tree mode: wipe the intermediate rungs of control runs, thin the set cells.

`students/controls/{base,rng}/...` runs exist to be compared against, not to be read along
their trajectory: their intermediate `checkpoint-N/` dirs are pure storage, so control mode
WIPES them (`ready:wipe`) and requires no record to do it. The one exception is a control
run that sits at a nonadaptive-set cell: it is thinned to the student keep epochs
(`ready:set_thin`) rather than refused, which is the only place the cell protect file does
not simply stop the script. Nothing outside `students/controls/` is ever in scope, the
archive lands off the (full) output disk, and a ladder write the full disk refuses is
retried after the deletions have made room.
"""

from __future__ import annotations

import errno
import json
import os
import re
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts import thin_checkpoint_ladders as thin

BASE_REL = "students/controls/base/pythia-410m/teacher-pythia-160m/ex10-20_prefix_n100k"
RNG_REL = "students/controls/rng/DataDecide-dolma1_7-1B-seed-default/ex10-20_prefix_n100k"
SET_TAG = "ep20_lr1e-4_r128a32"       # a cell of the synthetic protect file below
PLAIN_TAG = "ep20_lr9e-5_r4a32"       # not a set cell
CELLS = {"students": {"tag": {SET_TAG: "pythia_student504"}}}


def out(tmp_path: Path) -> Path:
    """The output root, a level BELOW tmp_path: the archive must land outside it."""
    root = tmp_path / "outputs"
    root.mkdir(exist_ok=True)
    return root


def make_control_run(root: Path, tag: str, *, parent: str = BASE_REL, epochs: int = 20,
                     n_steps: int = 60, ftseed: int = 1, traj: bool = True,
                     saves_per_epoch: int = 1) -> Path:
    """A finished control run: endpoint adapter + one rung per epoch, no record required."""
    run = root / parent / tag / f"ftseed{ftseed}"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (run / "train_manifest.json").write_text(json.dumps({
        "n_train_steps": n_steps, "student_param_tag": tag, "animal": None,
        "cross_base": False, "hyperparameters": {"n_epochs": epochs}}))
    (ck / "adapter_model.safetensors").write_bytes(b"ENDPOINT" * 8)
    (ck / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    per = n_steps // (epochs * saves_per_epoch)
    for step in range(per, n_steps, per):
        d = ck / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(f"step{step}".encode() * 4)
        (d / "adapter_config.json").write_text("{}")
    (run / "eval").mkdir()
    if traj:
        for name in ("pref_traj", "pile_kl_traj"):
            (run / "eval" / f"{name}.json").write_text(json.dumps({"points": [{"step": 3}]}))
    old = time.time() - 86_400
    for path in run.rglob("*"):
        os.utime(path, (old, old))
    os.utime(run, (old, old))
    return run


def control_cfg(root: Path, **kwargs) -> dict:
    cfg = {
        "receivers": re.compile(thin.DEFAULT_RECEIVERS),
        "teachers": re.compile(thin.DEFAULT_TEACHERS),
        "keep_epochs": [float(e) for e in thin.CONTROL_KEEP_EPOCHS.split(",")],
        "require": [], "fast_bytes": False, "no_sha": False,
        "allow_symlinked_checkpoints": False, "protected": {}, "output_root": root,
        "teacher_tree": None, "control_tree": thin.CONTROL_TREE,
        "archive_extra": list(thin.CONTROL_ARCHIVE_EXTRA),
        "archive_root": root.parent / "archive_off_disk",
        "strip_tokenizers": False, "tokenizer_sha": False, "cells": CELLS,
    }
    cfg.update(kwargs)
    return cfg


def invoke(root: Path, tmp_path: Path, *args: str):
    cells = tmp_path / "cells.tsv"
    cells.write_text(f"students\ttag\t{SET_TAG}\tpythia_student504\n")
    return CliRunner().invoke(thin.app, [
        "--output-root", str(root), "--workers", "1", "--walk-workers", "2",
        "--control-tree", thin.CONTROL_TREE,
        "--protect-file", str(tmp_path / "none.tsv"), "--no-protect-file",
        "--cell-protect-file", str(cells),
        "--archive-root", str(tmp_path / "archive_off_disk"),
        "--report", str(tmp_path / "report.json"), *args])


# --------------------------------------------------------------------------- scope
@pytest.mark.parametrize("rel,reason", [
    (f"{BASE_REL}/{PLAIN_TAG}/ftseed1", None),
    (f"{RNG_REL}/{PLAIN_TAG}/ftseed1", None),
    (f"{BASE_REL}/{PLAIN_TAG}_sgd/ftseed1", None),       # arms are ordinary tags here
    ("students/dog/pythia-160m-seed1/t/p/ep20_lr1e-4_r4a32/ftseed1", "not_a_control_run"),
    ("teachers/dog/pythia-160m/ep5_lr1e-4_r32a64/ftseed0", "not_a_control_run"),
    ("students/controls", "not_a_control_run"),
    (f"{BASE_REL}/ckpt/{PLAIN_TAG}/ftseed1", "ckpt_component"),
    (f"{BASE_REL}/{PLAIN_TAG}@step1000/ftseed1", "at_step"),
])
def test_control_scope(rel, reason):
    got = thin.control_scope_reason(rel, thin.CONTROL_TREE)
    assert got == (reason if reason != "at_step" else "ckpt_component")


def test_narrower_prefix_excludes_the_other_receivers():
    prefix = "students/controls/base/pythia-410m"
    assert thin.control_scope_reason(f"{BASE_REL}/{PLAIN_TAG}/ftseed1", prefix) is None
    assert thin.control_scope_reason(f"{RNG_REL}/{PLAIN_TAG}/ftseed1",
                                     prefix) == "other_tree"


def test_labels_name_the_layout_and_the_receiver():
    assert thin.control_labels(f"{BASE_REL}/{PLAIN_TAG}/ftseed1") == ("base", "pythia-410m")
    assert thin.control_labels(f"{RNG_REL}/{PLAIN_TAG}/ftseed1") == (
        "rng", "DataDecide-dolma1_7-1B-seed-default")


# --------------------------------------------------------------------------- planning
def test_plain_control_run_is_wiped(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    row = thin.inspect_run(run, out(tmp_path), control_cfg(out(tmp_path)))
    assert row["status"] == "ready:wipe"
    assert row["keep_steps"] == [] and row["n_keep"] == 0
    assert row["delete_steps"] == sorted(thin.on_disk_steps(run / "checkpoints"))
    assert len(row["delete_steps"]) == 19 and row["bytes_to_free"] > 0
    assert (row["animal"], row["receiver"]) == ("base", "pythia-410m")


def test_set_cell_control_run_is_thinned_not_refused(tmp_path: Path):
    run = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    row = thin.inspect_run(run, out(tmp_path), control_cfg(out(tmp_path)))
    assert row["status"] == "ready:set_thin"
    # 0.5, 1, 1.5, 2, 5 and 10 epochs of a 20-epoch, 600-step run (30 steps/epoch)
    assert row["keep_steps"] == [15, 30, 45, 60, 150, 300]
    assert row["cell"] == "nonadaptive:pythia_student504"


def test_rng_layout_is_in_scope(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG, parent=RNG_REL)
    row = thin.inspect_run(run, out(tmp_path), control_cfg(out(tmp_path)))
    assert row["status"] == "ready:wipe"
    assert row["animal"] == "rng"


def test_path_protect_file_still_refuses(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    rel = f"{BASE_REL}/{PLAIN_TAG}/ftseed1"
    cfg = control_cfg(out(tmp_path), protected={rel: "hand-held"})
    assert thin.inspect_run(run, out(tmp_path), cfg)["status"] == "protected:hand-held"


# --------------------------------------------------------------------------- apply
def test_wipe_deletes_every_rung_and_keeps_the_endpoint(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    log = tmp_path / "controls_report.deleted.jsonl"
    assert thin.thin_run(row, cfg, log) == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    assert (run / "checkpoints" / "adapter_model.safetensors").read_bytes() == b"ENDPOINT" * 8
    assert (run / "eval" / "pref_traj.json").is_file()
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(logged) == 19
    assert all(r["bytes"] > 0 and r["adapter_sha256"] and r["step"] for r in logged)
    ladder = thin.read_ladder(run)
    assert ladder["action"] == "ready:wipe" and ladder["keep_steps"] == []
    assert all(c["deleted_at"] for c in ladder["checkpoints"].values() if not c["kept"])


def test_set_cell_apply_keeps_the_keep_epochs(tmp_path: Path):
    run = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert sorted(thin.on_disk_steps(run / "checkpoints")) == [15, 30, 45, 60, 150, 300]


def test_records_are_archived_off_the_output_root(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    written, problem = thin.archive_records(run, row, cfg)
    assert problem is None
    assert set(written) == {"pref_traj.json", "pile_kl_traj.json"}
    for dest in written.values():
        assert Path(dest).is_absolute() and Path(dest).is_file()
        assert Path(dest).is_relative_to(cfg["archive_root"])
    # ...and the record itself is never deleted.
    thin.thin_run(row, cfg, tmp_path / "log.jsonl")
    assert (run / "eval" / "pref_traj.json").is_file()


def test_second_apply_pass_is_a_no_op(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    again = thin.inspect_run(run, out(tmp_path), cfg)
    assert again["status"] == "already_thinned"
    assert again.get("delete_steps") is None
    endpoint = run / "checkpoints" / "adapter_model.safetensors"
    assert endpoint.read_bytes() == b"ENDPOINT" * 8


def test_enospc_on_the_ladder_write_still_deletes_then_writes(tmp_path: Path,
                                                              monkeypatch):
    """A full disk refuses the pre-delete manifest; the deletes make room for the retry."""
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    calls: list[int] = []
    real = thin.write_ladder

    def flaky(run_dir, doc):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(run_dir, doc)

    monkeypatch.setattr(thin, "write_ladder", flaky)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    ladder = thin.read_ladder(run)
    assert ladder is not None and ladder["ladder_written_after_delete"] is True
    assert len(ladder["checkpoints"]) == 20


def test_enospc_twice_is_reported_as_deleted_ladder_unwritten(tmp_path: Path, monkeypatch):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)

    def always(run_dir, doc):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(thin, "write_ladder", always)
    log = tmp_path / "log.jsonl"
    assert thin.thin_run(row, cfg, log) == "deleted:ladder_unwritten"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    # The jsonl is the rebuild source: every step with its bytes and digest.
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(logged) == 19 and all(r["adapter_sha256"] for r in logged)


def test_enospc_is_not_tolerated_outside_control_mode(tmp_path: Path, monkeypatch):
    """Student/teacher mode is unchanged: a failed manifest write still aborts."""
    def always(run_dir, doc):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(thin, "write_ladder", always)
    with pytest.raises(OSError):
        thin._write_ladder_or_enospc(tmp_path, {}, {"control_tree": None})


# --------------------------------------------------------------------------- CLI
def test_non_controls_prefix_is_rejected(tmp_path: Path):
    result = CliRunner().invoke(thin.app, [
        "--output-root", str(tmp_path), "--control-tree", "students/dog",
        "--cell-protect-file", str(thin.ROOT / thin.DEFAULT_CELL_PROTECT)])
    assert result.exit_code != 0
    assert "--control-tree" in str(result.output) + str(result.exception)


def test_run_dirs_file_cannot_reach_outside_controls(tmp_path: Path):
    """The one path that hands in run dirs directly is scoped the same way."""
    outside = make_control_run(out(tmp_path), PLAIN_TAG,
                               parent="students/dog/pythia-160m-seed1/t/p")
    inside = make_control_run(out(tmp_path), PLAIN_TAG)
    listing = tmp_path / "runs.txt"
    listing.write_text(f"{outside}\n{inside}\n")
    result = invoke(out(tmp_path), tmp_path, "--run-dirs-file", str(listing), "--apply")
    assert result.exit_code == 0, result.output
    statuses = {r["run_dir_rel"]: r["status"]
                for r in json.loads((tmp_path / "report.json").read_text())["runs"]}
    assert statuses[f"students/dog/pythia-160m-seed1/t/p/{PLAIN_TAG}/ftseed1"] == \
        "out_of_scope:not_a_control_run"
    assert len(thin.on_disk_steps(outside / "checkpoints")) == 19
    assert thin.on_disk_steps(inside / "checkpoints") == {}


def test_dry_run_table_separates_wipe_from_set_thin(tmp_path: Path):
    make_control_run(out(tmp_path), PLAIN_TAG)
    make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    make_control_run(out(tmp_path), PLAIN_TAG, parent=RNG_REL)
    result = invoke(out(tmp_path), tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["control_tree"] == thin.CONTROL_TREE and report["require"] == []
    assert sorted(r["status"] for r in report["runs"]) == [
        "ready:set_thin", "ready:wipe", "ready:wipe"]
    table = thin.control_summary(report["runs"])
    assert "control mode (wipe | set_thin | refused)" in table
    assert "base/pythia-410m" in table and "rng/DataDecide-dolma1_7-1B-seed-default" in table
    # nothing was deleted by the dry run
    assert len(thin.on_disk_steps(out(tmp_path) / BASE_REL / PLAIN_TAG / "ftseed1"
                                  / "checkpoints")) == 19


# ------------------------------------------------- full disk: EDQUOT, claims, failures
def test_edquot_is_the_same_full_disk_condition_as_enospc(tmp_path: Path):
    """cold-1 reports the quota errno, not ENOSPC; control mode must treat them alike."""
    cfg = control_cfg(out(tmp_path))
    for code in (errno.ENOSPC, errno.EDQUOT):
        assert thin.full_disk_error(OSError(code, "full"), cfg)
        assert not thin.full_disk_error(OSError(code, "full"), {"control_tree": None})
    assert not thin.full_disk_error(OSError(errno.EACCES, "denied"), cfg)


def test_edquot_on_the_ladder_write_deletes_then_writes(tmp_path: Path, monkeypatch):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    calls: list[int] = []
    real = thin.write_ladder

    def flaky(run_dir, doc):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(run_dir, doc)

    monkeypatch.setattr(thin, "write_ladder", flaky)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    assert thin.read_ladder(run)["ladder_written_after_delete"] is True


def test_claim_edquot_runs_claimless_and_the_next_run_claims_normally(tmp_path: Path,
                                                                     monkeypatch):
    """A full disk cannot create `.claim`; that must not stop the run that frees it."""
    first = make_control_run(out(tmp_path), PLAIN_TAG)
    second = make_control_run(out(tmp_path), PLAIN_TAG, ftseed=2)
    cfg = control_cfg(out(tmp_path))
    calls: list[Path] = []
    real = thin.acquire_claim

    def flaky(run_dir, *args, **kwargs):
        calls.append(Path(run_dir))
        if len(calls) == 1:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(run_dir, *args, **kwargs)

    monkeypatch.setattr(thin, "acquire_claim", flaky)
    log = tmp_path / "log.jsonl"
    rows = [thin.inspect_run(r, out(tmp_path), cfg) for r in (first, second)]
    assert [thin.thin_run(row, cfg, log) for row in rows] == ["thinned", "thinned"]
    assert rows[0]["claimless"] is True and rows[1]["claimless"] is False
    assert thin.read_ladder(first)["claimless"] is True
    assert thin.read_ladder(second)["claimless"] is False
    assert all(thin.on_disk_steps(r / "checkpoints") == {} for r in (first, second))
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert {r["claimless"] for r in logged} == {True, False}
    assert not (first / ".claim").exists() and not (second / ".claim").exists()


def test_a_held_claim_still_skips_the_run(tmp_path: Path, monkeypatch):
    """FileExistsError is another process, not a full disk: the run is left alone."""
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    monkeypatch.setattr(thin, "acquire_claim",
                        lambda *a, **k: (_ for _ in ()).throw(FileExistsError(".claim")))
    with pytest.raises(FileExistsError):
        thin.thin_run(row, cfg, tmp_path / "log.jsonl")
    # `acquire_claim` itself returns None for a live claim, which is the real path:
    monkeypatch.setattr(thin, "acquire_claim", lambda *a, **k: None)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "skipped_claimed"
    assert len(thin.on_disk_steps(run / "checkpoints")) == 19


def test_a_failing_run_does_not_stop_the_next_one(tmp_path: Path, monkeypatch):
    make_control_run(out(tmp_path), PLAIN_TAG)
    make_control_run(out(tmp_path), PLAIN_TAG, ftseed=2)
    boom: list[int] = []
    real = thin.thin_run

    def flaky(row, cfg, log_path):
        boom.append(1)
        if len(boom) == 1:
            raise RuntimeError("cephfs wobble")
        return real(row, cfg, log_path)

    monkeypatch.setattr(thin, "thin_run", flaky)
    result = invoke(out(tmp_path), tmp_path, "--apply")
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    statuses = sorted(r["status"] for r in report["runs"])
    assert statuses == ["failed:RuntimeError", "thinned"]
    assert report["failed"][0]["detail"] == "cephfs wobble"
    # ...and the refused column of the table does not count the run that succeeded.
    table = thin.control_summary(report["runs"])
    assert "applied this pass: 1 run(s)" in table


# --------------------------------------------------------------------------- wipe-set-cells
def test_wipe_set_cells_wipes_the_cell_but_still_names_it(tmp_path: Path):
    run = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    cfg = control_cfg(out(tmp_path), wipe_set_cells=True)
    row = thin.inspect_run(run, out(tmp_path), cfg)
    assert row["status"] == "ready:wipe"
    assert row["cell"] == "nonadaptive:pythia_student504"
    assert row["keep_steps"] == []
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}


def test_wipe_set_cells_needs_control_mode(tmp_path: Path):
    result = CliRunner().invoke(thin.app, [
        "--output-root", str(out(tmp_path)), "--wipe-set-cells",
        "--cell-protect-file", str(thin.ROOT / thin.DEFAULT_CELL_PROTECT)])
    assert result.exit_code != 0
    assert "--wipe-set-cells" in str(result.output) + str(result.exception)


# --------------------------------------------------------------------------- exclude-prefix
def test_exclude_prefix_holds_back_one_receiver(tmp_path: Path):
    kept = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2,
                            parent="students/controls/base/pythia-160m-data-seed1/"
                                   "teacher-pythia-160m/ex10-20_prefix_n100k")
    wiped = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    result = invoke(out(tmp_path), tmp_path, "--wipe-set-cells", "--exclude-prefix",
                    "students/controls/base/pythia-160m-data-seed1")
    assert result.exit_code == 0, result.output
    statuses = {r["run_dir_rel"]: r["status"]
                for r in json.loads((tmp_path / "report.json").read_text())["runs"]}
    assert statuses[str(kept.relative_to(out(tmp_path)))] == "out_of_scope:excluded_prefix"
    assert statuses[str(wiped.relative_to(out(tmp_path)))] == "ready:wipe"


def test_exclude_prefix_must_lie_under_the_control_tree(tmp_path: Path):
    result = CliRunner().invoke(thin.app, [
        "--output-root", str(out(tmp_path)), "--control-tree", thin.CONTROL_TREE,
        "--exclude-prefix", "students/dog",
        "--cell-protect-file", str(thin.ROOT / thin.DEFAULT_CELL_PROTECT)])
    assert result.exit_code != 0
    assert "--exclude-prefix" in str(result.output) + str(result.exception)
