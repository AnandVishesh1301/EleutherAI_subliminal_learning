"""The final-checkpoint dedupe script: only a PROVEN duplicate of a finished run goes.

`scripts.dedupe_final_trajectory_checkpoint` deletes `checkpoints/checkpoint-N` when N is
the run's final step and its adapter holds exactly the endpoint adapter's weights. Every
other classification (weights differ, last checkpoint below the final step, run in flight,
yield checkpoint, unreadable file) must survive both the dry run and `--apply`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.utils.run_io import CLAIM_TTL_SECONDS

from scripts import dedupe_final_trajectory_checkpoint as dd

TENSORS = {"base_model.model.layers.0.q_proj.lora_A.weight": torch.arange(6.0).reshape(2, 3)}
OTHER = {"base_model.model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 3)}


def _run(root: Path, rel: str, *, ckpt_steps=(2, 4), n_train_steps=4,
         ckpt_tensors=None, raw=None) -> Path:
    """A finished run dir: manifest + endpoint adapter + `checkpoint-N` dirs."""
    run_dir = root / rel
    (run_dir / "checkpoints").mkdir(parents=True)
    save_file(TENSORS, str(run_dir / "checkpoints" / dd.ADAPTER))
    for step in ckpt_steps:
        leaf = run_dir / "checkpoints" / f"checkpoint-{step}"
        leaf.mkdir()
        if raw is not None and step == max(ckpt_steps):
            (leaf / dd.ADAPTER).write_bytes(raw)
        else:
            save_file(ckpt_tensors or TENSORS, str(leaf / dd.ADAPTER))
        (leaf / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (run_dir / dd.MANIFEST).write_text(json.dumps({"n_train_steps": n_train_steps}))
    # A run whose manifest/endpoint was written minutes ago counts as in flight, so the
    # fixture's finished run is backdated past that window.
    _backdate(run_dir)
    return run_dir


def _backdate(run_dir: Path, seconds: float = 4 * dd.RECENT_WRITE_SECONDS) -> None:
    old = time.time() - seconds
    for path in (run_dir / dd.MANIFEST, run_dir / "checkpoints" / dd.ADAPTER):
        os.utime(path, (old, old))


def _scan(root: Path, run_dir: Path) -> dict:
    return dd.inspect_run(run_dir, root)


def test_identical_duplicate_is_found_kept_on_dry_run_and_deleted_on_apply(tmp_path):
    run_dir = _run(tmp_path, "teachers/panda/pythia-160m/ftseed1")
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "identical_bytes"
    assert row["final_step"] == 4 and row["final_step_source"] == "manifest"
    assert row["candidate"] == str(run_dir / "checkpoints" / "checkpoint-4")
    assert row["bytes"] > 0
    # Dry run: nothing touched.
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()

    log = tmp_path / "deleted.jsonl"
    dd.delete_candidate(row, log)
    assert not (run_dir / "checkpoints" / "checkpoint-4").exists()
    assert (run_dir / "checkpoints" / "checkpoint-2").is_dir()
    assert (run_dir / "checkpoints" / dd.ADAPTER).is_file()
    record = json.loads(log.read_text().splitlines()[-1])  # intent, then done
    assert record["path"] == row["candidate"] and record["bytes"] == row["bytes"]
    # The endpoint is identified by its fingerprint + a head/tail digest: a full digest
    # would re-read every adapter a third time.
    assert len(record["endpoint_sha256_head_tail"]) == 64
    assert record["endpoint_stat"] == row["endpoint_stat"]
    assert record["endpoint_sha256_head_tail"] == dd.sha256_head_tail(
        run_dir / "checkpoints" / dd.ADAPTER)


def test_different_weights_are_reported_and_kept(tmp_path):
    run_dir = _run(tmp_path, "students/dog/pythia-160m/ftseed1", ckpt_tensors=OTHER)
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "DIFFERENT" and "differs" in row["detail"]
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_last_checkpoint_below_the_final_step_is_not_a_candidate(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/ftseed1", ckpt_steps=(2, 4), n_train_steps=6)
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "last_ckpt_below_final" and "candidate" not in row


@pytest.mark.parametrize("marker", ["interrupted.json", "claim", "resume_marker.json"])
def test_in_flight_runs_are_skipped(tmp_path, marker):
    run_dir = _run(tmp_path, f"students/dog/b/{marker}")
    if marker == "claim":
        claim = run_dir / ".claim"
        claim.write_text("host:1")
        assert time.time() - claim.stat().st_mtime < dd.CLAIM_FRESH_SECONDS
    elif marker == "resume_marker.json":
        (run_dir / "checkpoints" / "checkpoint-4" / marker).write_text('{"step": 4}')
    else:
        (run_dir / marker).write_text('{"step": 4}')
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "in_flight"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_a_stale_claim_does_not_block(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/stale")
    claim = run_dir / ".claim"
    claim.write_text("host:1")
    assert dd.CLAIM_FRESH_SECONDS == CLAIM_TTL_SECONDS  # the run gate's own claim TTL
    old = time.time() - 2 * CLAIM_TTL_SECONDS
    os.utime(claim, (old, old))
    assert _scan(tmp_path, run_dir)["status"] == "identical_bytes"


def test_unreadable_candidate_is_reported_and_kept(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/torn", raw=b"not-a-safetensors-file")
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "unreadable"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_byte_differing_but_tensor_identical_is_deletable(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/meta")
    save_file(TENSORS, str(run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER),
              metadata={"format": "pt", "note": "written by another save"})
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "identical_tensors"


def test_final_step_falls_back_to_trainer_state_then_to_the_checkpoint(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/fallback")
    (run_dir / dd.MANIFEST).write_text(json.dumps({"student_param_tag": "x"}))
    (run_dir / "checkpoints" / "trainer_state.json").write_text(
        json.dumps({"global_step": 4}))
    _backdate(run_dir)  # rewriting the manifest would otherwise read as a live write
    fallback = _scan(tmp_path, run_dir)
    assert fallback["final_step_source"] == "trainer_state"
    assert fallback["status"] == "skipped_no_manifest_step"
    (run_dir / "checkpoints" / "trainer_state.json").unlink()
    row = _scan(tmp_path, run_dir)
    assert row["final_step_source"] == "checkpoint_trainer_state"
    # Without `n_train_steps` the trajectory reader labels the endpoint from the highest
    # REMAINING checkpoint, so deleting this dir would move the ladder's x-axis.
    assert row["status"] == "skipped_no_manifest_step"


def test_walk_finds_run_dirs_and_never_descends_into_checkpoints(tmp_path):
    _run(tmp_path, "teachers/panda/pythia-160m/ftseed1")
    _run(tmp_path, "teachers/panda/pythia-160m/ftseed2")
    (tmp_path / "teachers" / "panda" / "pythia-160m" / "ftseed1" / "eval").mkdir()
    found = dd.find_run_dirs(tmp_path / "teachers")
    assert [p.name for p in found] == ["ftseed1", "ftseed2"]
    # A run dir is a leaf of the walk: its checkpoint dirs are never themselves reported.
    assert all("checkpoint-" not in str(p) for p in found)


def test_summary_counts_and_reader_gate(tmp_path):
    rows = [_scan(tmp_path, _run(tmp_path, "teachers/panda/b/ftseed1")),
            _scan(tmp_path, _run(tmp_path, "teachers/panda/b/ftseed2", ckpt_tensors=OTHER))]
    text = dd.summarize(rows)
    assert "identical (deletable)   1" in text and "DIFFERENT (kept)        1" in text
    assert "teachers/panda/b" in text
    # This checkout's trajectory gates must compare distinct steps before any --apply.
    assert dd.reader_gate_problems() == []


def test_checkpoints_above_the_final_step_are_kept(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/above", ckpt_steps=(2, 4), n_train_steps=2)
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "ckpt_above_final" and "candidate" not in row
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_a_recently_written_run_is_in_flight(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/fresh")
    now = time.time()
    os.utime(run_dir / "checkpoints" / dd.ADAPTER, (now, now))
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "in_flight" and row["detail"] == "recent write"


def test_scan_error_becomes_a_row_and_does_not_propagate(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, "students/dog/b/boom")
    monkeypatch.setattr(dd, "checkpoint_steps", lambda p: (_ for _ in ()).throw(OSError("ceph")))
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "scan_error" and "ceph" in row["detail"]
    assert "scan errors             1" in dd.summarize([row])


def test_dir_bytes_ignores_files_that_vanish(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, "students/dog/b/vanish")
    real_stat = Path.stat

    def flaky(self, *a, **kw):
        if self.name == "trainer_state.json":
            raise FileNotFoundError(str(self))
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", flaky)
    assert dd.dir_bytes(run_dir / "checkpoints" / "checkpoint-4") > 0


# --------------------------------------------------------------- re-validation at delete
def test_delete_skips_a_candidate_whose_adapter_changed(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/changed")
    row = _scan(tmp_path, run_dir)
    adapter = run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER
    os.utime(adapter, (time.time(), time.time()))
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_changed"
    assert row["status"] == "skipped_changed" and "changed" in row["detail"]
    assert adapter.is_file()
    assert not (tmp_path / "log.jsonl").exists()


def test_delete_skips_when_the_endpoint_adapter_grew(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/endpoint")
    row = _scan(tmp_path, run_dir)
    endpoint = run_dir / "checkpoints" / dd.ADAPTER
    endpoint.write_bytes(endpoint.read_bytes() + b"\x00")
    _backdate(run_dir)  # a size change alone, not a fresh-write in-flight signal
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_changed"
    assert "endpoint" in row["detail"]
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_delete_skips_when_the_manifest_vanished(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/nomanifest")
    row = _scan(tmp_path, run_dir)
    (run_dir / dd.MANIFEST).unlink()
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_changed"
    assert row["detail"] == f"{dd.MANIFEST} vanished"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_delete_skips_a_symlinked_candidate(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/link")
    row = _scan(tmp_path, run_dir)
    real = run_dir / "checkpoints" / "checkpoint-4"
    elsewhere = tmp_path / "elsewhere"
    real.rename(elsewhere)
    real.symlink_to(elsewhere, target_is_directory=True)
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_unsafe_path"
    assert row["detail"] == "candidate is a symlink"
    assert (elsewhere / dd.ADAPTER).is_file()


def test_a_torn_candidate_is_completed(tmp_path):
    """A dir a previous `rmtree` half-removed is a proven duplicate whose deletion is
    unfinished: finish it (logged with `torn`), instead of reporting `delete_failed` on
    every later run and leaving it on disk forever."""
    run_dir = _run(tmp_path, "students/dog/b/torn2")
    row = _scan(tmp_path, run_dir)
    (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).unlink()
    log = tmp_path / "log.jsonl"
    assert dd.delete_candidate(row, log) == "deleted"
    assert not (run_dir / "checkpoints" / "checkpoint-4").exists()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["phase"] for r in records] == ["intent", "done"]
    assert all(r["torn"] is True for r in records)


def test_a_torn_candidate_whose_rmtree_fails_is_still_delete_failed(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, "students/dog/b/torn3")
    row = _scan(tmp_path, run_dir)
    (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).unlink()
    monkeypatch.setattr(dd.shutil, "rmtree",
                        lambda p: (_ for _ in ()).throw(PermissionError("read-only")))
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "delete_failed"
    assert "read-only" in row["detail"]


def test_intent_and_done_records_bracket_the_rmtree(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/jsonl")
    row = _scan(tmp_path, run_dir)
    log = tmp_path / "log.jsonl"
    assert dd.delete_candidate(row, log) == "deleted"
    phases = [json.loads(line) for line in log.read_text().splitlines()]
    assert [p["phase"] for p in phases] == ["intent", "done"]
    assert {p["path"] for p in phases} == {row["candidate"]}


def test_rmtree_failure_is_recorded_and_does_not_raise(tmp_path, monkeypatch):
    run_dir = _run(tmp_path, "students/dog/b/perm")
    row = _scan(tmp_path, run_dir)
    monkeypatch.setattr(dd.shutil, "rmtree",
                        lambda p: (_ for _ in ()).throw(PermissionError("read-only")))
    log = tmp_path / "log.jsonl"
    assert dd.delete_candidate(row, log) == "delete_failed"
    assert row["status"] == "delete_failed" and "read-only" in row["detail"]
    phases = [json.loads(line) for line in log.read_text().splitlines()]
    assert [p["phase"] for p in phases] == ["intent", "failed"]
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


# ------------------------------------------------------------------------------ the CLI
def _cli(tmp_path, monkeypatch, *args):
    from typer.testing import CliRunner

    monkeypatch.setattr(dd, "resolve_output_root", lambda: tmp_path)
    return CliRunner().invoke(dd.app, [str(a) for a in args])


def test_apply_continues_past_a_failure_and_still_writes_the_report(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        _run(tmp_path, f"students/dog/base/{name}")
    real_rmtree = dd.shutil.rmtree
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError("read-only")
        return real_rmtree(path)

    monkeypatch.setattr(dd.shutil, "rmtree", flaky)
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", report, "--apply", "--workers", 1)
    assert res.exit_code == 0, res.output
    payload = json.loads(report.read_text())
    assert payload["complete"] is True and payload["deleted"] == 2
    statuses = sorted(r["status"] for r in payload["runs"])
    assert statuses.count("delete_failed") == 1
    log = tmp_path / "report.deleted.jsonl"
    assert [json.loads(x)["phase"] for x in log.read_text().splitlines()].count("failed") == 1


def test_a_kill_mid_apply_leaves_an_incomplete_report_with_the_first_deletion(
        tmp_path, monkeypatch):
    for name in ("a", "b"):
        _run(tmp_path, f"students/dog/base/{name}")
    real_rmtree = dd.shutil.rmtree
    calls = {"n": 0}

    def killed(path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_rmtree(path)

    monkeypatch.setattr(dd.shutil, "rmtree", killed)
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", report, "--apply", "--workers", 1)
    assert res.exit_code == 130  # CliRunner turns the interrupt into SystemExit(130)
    payload = json.loads(report.read_text())
    assert payload["complete"] is False and payload["deleted"] == 1
    assert sum(r.get("deleted", False) for r in payload["runs"]) == 1
    phases = [json.loads(x)["phase"]
              for x in (tmp_path / "report.deleted.jsonl").read_text().splitlines()]
    assert phases == ["intent", "done", "intent"]


def test_max_delete_caps_one_invocation(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        _run(tmp_path, f"students/dog/base/{name}")
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", report, "--apply", "--max-delete", 2, "--workers", 1)
    assert res.exit_code == 0, res.output
    payload = json.loads(report.read_text())
    assert payload["deleted"] == 2 and payload["complete"] is False
    left = [p for p in (tmp_path / "students").rglob("checkpoint-4") if p.is_dir()]
    assert len(left) == 1


def test_only_tree_matching_nothing_exits_nonzero(tmp_path, monkeypatch):
    _run(tmp_path, "students/dog/base/a")
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", tmp_path / "r.json", "--only-tree", "teachers/panda")
    assert res.exit_code == 2


def test_a_root_outside_the_output_root_is_reported(tmp_path, monkeypatch):
    other = tmp_path.parent / f"{tmp_path.name}-staging"
    _run(other, "students/dog/base/a")
    report = tmp_path / "r.json"
    res = _cli(tmp_path, monkeypatch, "--root", other / "students", "--report", report)
    assert res.exit_code == 0, res.output
    assert json.loads(report.read_text())["roots_outside_output_root"] == \
        [str(other / "students")]


def test_gkd_and_sdf_trees_are_never_walked(tmp_path, monkeypatch):
    _run(tmp_path, "gkd_students/dog/base/a")
    _run(tmp_path, "students/dog/base/a")
    assert [p.name for p in dd.find_run_dirs(tmp_path)] == ["a"]
    assert all("gkd_students" not in str(p) for p in dd.find_run_dirs(tmp_path))
    report = tmp_path / "r.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "gkd_students",
               "--report", report)
    assert res.exit_code == 0, res.output
    payload = json.loads(report.read_text())
    assert [r["status"] for r in payload["runs"]] == ["skipped_excluded_root"]
    assert "excluded roots          1" in res.output


# ------------------------------------------------------- the run-dir claim is the lock
def test_a_fresh_claim_held_by_someone_else_blocks_the_delete(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/claimed")
    row = _scan(tmp_path, run_dir)
    (run_dir / ".claim").write_text("pid=1 host=other\ntoken=deadbeef\n")
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_claimed"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()
    assert (run_dir / ".claim").is_file()  # theirs, untouched


def test_our_claim_is_released_after_a_delete_and_after_a_failed_delete(
        tmp_path, monkeypatch):
    run_dir = _run(tmp_path, "students/dog/b/released")
    row = _scan(tmp_path, run_dir)
    monkeypatch.setattr(dd.shutil, "rmtree",
                        lambda p: (_ for _ in ()).throw(PermissionError("read-only")))
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "delete_failed"
    assert not (run_dir / ".claim").exists()

    monkeypatch.undo()
    row = _scan(tmp_path, run_dir)
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "deleted"
    assert not (run_dir / ".claim").exists()


# ------------------------------------------------------------------- torn final-step dirs
def test_a_torn_candidate_is_kept_when_the_endpoint_changed(tmp_path):
    """The torn path drops only the candidate-adapter check: a torn dir whose ENDPOINT is
    no longer the file that was compared is not a proven duplicate any more."""
    run_dir = _run(tmp_path, "students/dog/b/torn_endpoint")
    row = _scan(tmp_path, run_dir)
    (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).unlink()
    endpoint = run_dir / "checkpoints" / dd.ADAPTER
    endpoint.write_bytes(endpoint.read_bytes() + b"\x00")
    _backdate(run_dir)  # a size change alone, not a fresh-write in-flight signal
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_changed"
    assert "endpoint" in row["detail"]
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()
    assert not (tmp_path / "log.jsonl").exists()


def test_a_torn_candidate_is_kept_when_the_manifest_vanished(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/torn_nomanifest")
    row = _scan(tmp_path, run_dir)
    (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).unlink()
    (run_dir / dd.MANIFEST).unlink()
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_changed"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


def test_a_torn_final_dir_is_recognised_on_a_fresh_scan_and_recovered(tmp_path):
    """No fingerprint row exists on a fresh invocation, so the leftover is recognised by
    its own shape and removed with an UNVERIFIED intent record."""
    run_dir = _run(tmp_path, "students/dog/b/fresh_torn")
    (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).unlink()
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "torn_final_dir"
    assert "torn final dirs         1" in dd.summarize([row])
    log = tmp_path / "log.jsonl"
    assert dd.delete_candidate(row, log) == "deleted"
    assert not (run_dir / "checkpoints" / "checkpoint-4").exists()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["phase"] for r in records] == ["intent", "done"]
    assert all(r["torn"] is True and r["verified"] is False for r in records)


def test_a_torn_final_dir_with_other_files_is_unrecognised_and_kept(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/fresh_torn_optim")
    leaf = run_dir / "checkpoints" / "checkpoint-4"
    (leaf / dd.ADAPTER).unlink()
    (leaf / "optimizer.pt").write_bytes(b"half a new checkpoint")
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "torn_final_dir_unrecognised" and "optimizer.pt" in row["detail"]
    assert "torn unrecognised       1" in dd.summarize([row])
    assert leaf.is_dir()


def test_a_fresh_torn_dir_that_is_in_flight_is_skipped(tmp_path):
    run_dir = _run(tmp_path, "students/dog/b/fresh_torn_claimed")
    (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).unlink()
    (run_dir / ".claim").write_text("host:1")
    assert _scan(tmp_path, run_dir)["status"] == "in_flight"
    assert (run_dir / "checkpoints" / "checkpoint-4").is_dir()


# ------------------------------------------------------------------------------ --stream
def test_stream_deletes_during_the_walk_and_leaves_everything_else(tmp_path, monkeypatch):
    _run(tmp_path, "students/dog/base/same")
    _run(tmp_path, "students/dog/base/diff", ckpt_tensors=OTHER)
    _run(tmp_path, "students/dog/base/torn_raw", raw=b"not-a-safetensors-file")
    live = _run(tmp_path, "students/dog/base/live")
    (live / ".claim").write_text("host:1")
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students", "--report", report,
               "--apply", "--stream", "--workers", 4)
    assert res.exit_code == 0, res.output
    payload = json.loads(report.read_text())
    assert payload["complete"] is True and payload["deleted"] == 1
    assert payload["streamed"] is True
    statuses = {Path(r["run_dir"]).name: r["status"] for r in payload["runs"]}
    assert statuses == {"same": "identical_bytes", "diff": "DIFFERENT",
                        "torn_raw": "unreadable", "live": "in_flight"}
    gone = tmp_path / "students/dog/base/same/checkpoints/checkpoint-4"
    assert not gone.exists()
    for name in ("diff", "torn_raw", "live"):
        assert (tmp_path / f"students/dog/base/{name}/checkpoints/checkpoint-4").is_dir()
    assert "identical (deletable)   1" in res.output


def test_stream_respects_max_delete_across_workers(tmp_path, monkeypatch):
    for i in range(10):
        _run(tmp_path, f"students/dog/base/r{i}")
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students", "--report", report,
               "--apply", "--stream", "--workers", 4, "--max-delete", 3)
    assert res.exit_code == 0, res.output
    payload = json.loads(report.read_text())
    assert payload["deleted"] == 3 and payload["complete"] is False
    left = [p for p in (tmp_path / "students").rglob("checkpoint-4") if p.is_dir()]
    assert len(left) == 7
    statuses = [r["status"] for r in payload["runs"]]
    assert statuses.count("skipped_max_delete") == 7
    done = [json.loads(x) for x in (tmp_path / "report.deleted.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in done].count("done") == 3


def test_stream_without_apply_is_a_usage_error(tmp_path, monkeypatch):
    _run(tmp_path, "students/dog/base/a")
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", tmp_path / "r.json", "--stream")
    assert res.exit_code == 2
    assert (tmp_path / "students/dog/base/a/checkpoints/checkpoint-4").is_dir()


def test_a_kill_mid_stream_leaves_an_incomplete_report_with_the_deletions(
        tmp_path, monkeypatch):
    for name in ("a", "b"):
        _run(tmp_path, f"students/dog/base/{name}")
    real_rmtree = dd.shutil.rmtree
    calls = {"n": 0}

    def killed(path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_rmtree(path)

    monkeypatch.setattr(dd.shutil, "rmtree", killed)
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students", "--report", report,
               "--apply", "--stream", "--workers", 1)
    assert res.exit_code == 130  # CliRunner turns the interrupt into SystemExit(130)
    payload = json.loads(report.read_text())
    assert payload["complete"] is False and payload["deleted"] == 1
    assert sum(r.get("deleted", False) for r in payload["runs"]) == 1
    phases = [json.loads(x)["phase"]
              for x in (tmp_path / "report.deleted.jsonl").read_text().splitlines()]
    assert phases == ["intent", "done", "intent"]


def test_stream_prints_progress_while_it_walks(tmp_path, monkeypatch):
    for name in ("a", "b"):
        _run(tmp_path, f"students/dog/base/{name}")
    monkeypatch.setattr(dd, "PROGRESS_EVERY", 1)
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", tmp_path / "r.json", "--apply", "--stream", "--workers", 1)
    assert res.exit_code == 0, res.output
    lines = [x for x in res.output.splitlines() if x.startswith("inspected ")]
    assert len(lines) == 2 and lines[-1].startswith("inspected 2, candidates 2, deleted ")
    assert lines[-1].endswith(" GiB")


def test_a_zero_byte_stray_file_still_makes_the_leftover_unrecognised(tmp_path):
    """An empty `optimizer.pt` is a half-written NEW checkpoint just as much as a full
    one: size is no evidence, so it too keeps the dir."""
    run_dir = _run(tmp_path, "students/dog/b/fresh_torn_empty")
    leaf = run_dir / "checkpoints" / "checkpoint-4"
    (leaf / dd.ADAPTER).unlink()
    (leaf / "optimizer.pt").write_bytes(b"")
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "torn_final_dir_unrecognised" and "optimizer.pt" in row["detail"]
    assert leaf.is_dir()


def test_a_fingerprinted_row_whose_candidate_became_a_new_checkpoint_is_kept(tmp_path):
    """The candidate adapter vanishing after the scan does not license the delete: the
    leftover-shape test runs on this path too, and a foreign file keeps the dir."""
    run_dir = _run(tmp_path, "students/dog/b/torn_shape")
    row = _scan(tmp_path, run_dir)
    leaf = run_dir / "checkpoints" / "checkpoint-4"
    (leaf / dd.ADAPTER).unlink()
    (leaf / "optimizer.pt").write_bytes(b"half a new checkpoint")
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_changed"
    assert "no longer a torn leftover" in row["detail"]
    assert leaf.is_dir()
    assert not (tmp_path / "log.jsonl").exists()


def test_torn_final_dirs_count_as_deletable_bytes_in_the_summary(tmp_path):
    """`--apply` deletes torn leftovers, so the summary's deletable count and reclaimable
    bytes must include them."""
    run_dir = _run(tmp_path, "students/dog/b/torn_summary")
    (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).unlink()
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "torn_final_dir" and row["bytes"] > 0
    text = dd.summarize([row])
    assert "identical (deletable)   0" in text and "deletable (total)       1" in text
    assert "bytes reclaimable       0.00 GiB" in text  # the fixture is tiny, but counted
    assert "students/dog/b" in text  # and it lands in the per-tree table


def test_non_stream_rows_past_max_delete_are_marked_skipped(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        _run(tmp_path, f"students/dog/base/{name}")
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", report, "--apply", "--max-delete", 1, "--workers", 1)
    assert res.exit_code == 0, res.output
    payload = json.loads(report.read_text())
    assert payload["deleted"] == 1 and payload["complete"] is False
    statuses = [r["status"] for r in payload["runs"]]
    assert statuses.count("skipped_max_delete") == 2
    assert all(r["detail"] == "--max-delete 1 reached"
               for r in payload["runs"] if r["status"] == "skipped_max_delete")


def test_streaming_workers_and_frequent_flushes_never_tear_the_report(tmp_path, monkeypatch):
    """The flush serialises a snapshot taken under the same lock every row mutation takes:
    8 workers flushing on EVERY deletion must not raise "dictionary changed size during
    iteration" out of `json.dumps` and unwind the run."""
    for i in range(200):
        _run(tmp_path, f"students/dog/base/r{i:03d}")
    monkeypatch.setattr(dd, "FLUSH_EVERY", 1)
    report = tmp_path / "report.json"
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students", "--report", report,
               "--apply", "--stream", "--workers", 8)
    assert res.exit_code == 0, res.output
    payload = json.loads(report.read_text())
    assert payload["complete"] is True and payload["deleted"] == 200
    assert len(payload["runs"]) == 200
    assert sum(r.get("deleted", False) for r in payload["runs"]) == 200
    done = [json.loads(x) for x in (tmp_path / "report.deleted.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in done].count("done") == 200
    assert not [p for p in (tmp_path / "students").rglob("checkpoint-4") if p.is_dir()]


def test_delete_skips_when_the_manifest_final_step_moved_on(tmp_path):
    """A run that resumed and trained further: `checkpoint-4` is a rung, not the endpoint."""
    run_dir = _run(tmp_path, "students/dog/b/resumed")
    row = _scan(tmp_path, run_dir)
    assert row["status"] == "identical_bytes"
    (run_dir / dd.MANIFEST).write_text(json.dumps({"n_train_steps": 6}))
    _backdate(run_dir)  # the manifest rewrite alone is not a fresh-write in-flight signal
    assert dd.delete_candidate(row, tmp_path / "log.jsonl") == "skipped_changed"
    assert "final step changed" in row["detail"]
    assert (run_dir / "checkpoints" / "checkpoint-4" / dd.ADAPTER).is_file()
    assert not (tmp_path / "log.jsonl").exists()


def test_a_kill_in_the_main_thread_stops_the_streaming_workers(tmp_path, monkeypatch):
    """Ctrl-C during --stream: no delete may START after the interrupt, drain or not."""
    import threading

    for i in range(8):
        _run(tmp_path, f"students/dog/base/b{i}")
    started: list[str] = []
    first, release = threading.Event(), threading.Event()

    def blocking_delete(row, log_path):
        started.append(Path(row["run_dir"]).name)
        if not first.is_set():
            first.set()
            # Held past the interrupt, so the runs the pool still has queued are drained by
            # the OTHER worker while this one is busy deleting.
            threading.Timer(0.5, release.set).start()
            release.wait(10)
        return "deleted"

    class Interrupted(dd.ThreadPoolExecutor):
        """Ctrl-C in the main thread while it waits for the walk's results."""

        def map(self, fn, iterable):  # noqa: D102 - the runs are all queued, then killed
            for item in iterable:
                self.submit(fn, item)
            first.wait(10)
            raise KeyboardInterrupt

    monkeypatch.setattr(dd, "delete_candidate", blocking_delete)
    monkeypatch.setattr(dd, "ThreadPoolExecutor", Interrupted)
    res = _cli(tmp_path, monkeypatch, "--root", tmp_path / "students",
               "--report", tmp_path / "report.json", "--apply", "--stream", "--workers", 2)
    assert res.exit_code == 130
    assert started == ["b0"]  # the delete already running; none started after the interrupt
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["complete"] is False and payload["deleted"] == 1
