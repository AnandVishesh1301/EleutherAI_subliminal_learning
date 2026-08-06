"""Tests for scripts/migrate_coherence_results.py.

Builds a mini old-layout tree in a tmp dir and exercises the migrator end to end via
its typer CLI: dry-run no-op, apply correctness, idempotent rerun, missing-run-dir
error handling, and npy-already-moved skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from scripts.migrate_coherence_results import app

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _ce_record(model_id, adapter_path, adapter_id, role, model_key, overall_ce):
    return {
        "record_type": "ce",
        "model_id": model_id,
        "adapter_id": adapter_id,
        "adapter_path": adapter_path,
        "merged": True,
        "role": role,
        "model_key": model_key,
        "shard_version": "v1",
        "shard_sha256": "deadbeef",
        "dtype": "bfloat16",
        "context_length": 2048,
        "n_windows": 10,
        "overall_ce": overall_ce,
        "per_window_ce": [overall_ce] * 3,
        "per_token_loss_path": f"runs/{model_key}/per_token_loss.npy",
        "timestamp": "2026-07-30T00:00:00Z",
        "code_git_head": "abc123",
    }


def _delta_record(model_id, adapter_path, adapter_id, delta):
    return {
        "record_type": "delta",
        "model_id": model_id,
        "adapter_id": adapter_id,
        "adapter_path": adapter_path,
        "base_model_id": "EleutherAI/pythia-160m",
        "shard_version": "v1",
        "shard_sha256": "deadbeef",
        "dtype": "bfloat16",
        "n_windows": 10,
        "delta_overall": delta,
        "ci_low": delta - 0.01,
        "ci_high": delta + 0.01,
        "ci_level": 0.95,
        "n_iterations": 1000,
        "bootstrap_seed": 0,
        "finetuned_overall_ce": 2.5 + delta,
        "base_overall_ce": 2.5,
        "timestamp": "2026-07-30T01:00:00Z",
    }


def _write_npy(path: Path, val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array([val], dtype=np.float32))


def build_old_tree(root: Path) -> dict:
    """Build a mini old-layout coherence dir. Returns useful paths for assertions."""
    coh = root / "coherence" / "v1"
    coh.mkdir(parents=True)

    teacher_run = root / "outputs" / "teachers" / "dog" / "ftseed0"
    teacher_run.mkdir(parents=True)
    teacher_adapter = str(teacher_run / "checkpoints")

    student_run = root / "outputs" / "students" / "dog" / "ftseed1"
    student_run.mkdir(parents=True)
    student_adapter = str(student_run / "checkpoints")

    # An adapter whose run dir does NOT exist on disk.
    missing_adapter = str(root / "outputs" / "teachers" / "gone" / "ftseed9" / "checkpoints")

    records = [
        _ce_record("EleutherAI/pythia-160m", None, None, "base", "EleutherAI__pythia-160m", 2.5),
        # teacher: duplicate ce proving last-wins (0.99 then 2.60).
        # legacy vocabulary: role="adapter" must normalize to "teacher" via path.
        _ce_record("EleutherAI/pythia-160m", teacher_adapter, "tch", "adapter",
                   "outputs__teachers__dog__ftseed0", 0.99),
        _ce_record("EleutherAI/pythia-160m", teacher_adapter, "tch", "adapter",
                   "outputs__teachers__dog__ftseed0", 2.60),
        _delta_record("EleutherAI/pythia-160m", teacher_adapter, "tch", 0.10),
        # student (role absent -> derive from path).
        {**_ce_record("EleutherAI/pythia-160m", student_adapter, "std", "student",
                      "outputs__students__dog__ftseed1", 2.70), "role": None},
        _delta_record("EleutherAI/pythia-160m", student_adapter, "std", 0.20),
        # missing run dir.
        _ce_record("EleutherAI/pythia-160m", missing_adapter, "gon", "teacher",
                   "outputs__teachers__gone__ftseed9", 3.0),
    ]

    with (coh / "results.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    _write_npy(coh / "runs" / "EleutherAI__pythia-160m" / "per_token_loss.npy", 1.0)
    _write_npy(coh / "runs" / "outputs__teachers__dog__ftseed0" / "per_token_loss.npy", 2.0)
    _write_npy(coh / "runs" / "outputs__students__dog__ftseed1" / "per_token_loss.npy", 3.0)
    _write_npy(coh / "runs" / "outputs__teachers__gone__ftseed9" / "per_token_loss.npy", 9.0)

    return {
        "coh": coh,
        "teacher_run": teacher_run,
        "student_run": student_run,
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_dry_run_is_noop(tmp_path):
    paths = build_old_tree(tmp_path)
    coh = paths["coh"]
    before = _snapshot(coh)

    result = runner.invoke(app, ["--coherence-dir", str(coh)])
    assert result.exit_code == 0, result.output
    assert "DRY-RUN" in result.output
    assert _snapshot(coh) == before  # nothing written, no log


def test_apply_correctness(tmp_path):
    paths = build_old_tree(tmp_path)
    coh = paths["coh"]

    result = runner.invoke(app, ["--coherence-dir", str(coh)])  # dry-run first
    assert result.exit_code == 0

    result = runner.invoke(app, ["--coherence-dir", str(coh), "--apply"])
    # one adapter has a missing run dir -> nonzero exit, but others still migrate.
    assert result.exit_code == 1, result.output

    # Teacher run: last-wins ce (2.60), legacy role="adapter" normalized to "teacher".
    tj = json.loads((paths["teacher_run"] / "pile_coherence" / "v1.json").read_text())
    assert tj["schema_version"] == 1
    assert tj["role"] == "teacher"
    assert tj["ce"]["overall_ce"] == 2.60
    assert tj["adapter_id"] == "tch"
    assert tj["delta"]["delta_overall"] == 0.10
    assert tj["per_token_loss_path"] == "v1_per_token_loss.npy"
    tnpy = paths["teacher_run"] / "pile_coherence" / "v1_per_token_loss.npy"
    assert np.load(tnpy)[0] == 2.0

    # Student: role derived from "/students/" path.
    sj = json.loads((paths["student_run"] / "pile_coherence" / "v1.json").read_text())
    assert sj["role"] == "student"
    assert sj["delta"]["delta_overall"] == 0.20

    # Base record.
    bj = json.loads((coh / "base" / "pythia-160m" / "ce.json").read_text())
    assert bj["role"] == "base"
    assert "adapter_path" not in bj and "delta" not in bj
    assert bj["per_token_loss_path"] == "per_token_loss.npy"
    assert np.load(coh / "base" / "pythia-160m" / "per_token_loss.npy")[0] == 1.0

    # Errors present -> results.jsonl NOT renamed, runs/ NOT removed.
    assert (coh / "results.jsonl").exists()
    assert (coh / "runs").exists()

    # A log was written.
    assert list(coh.glob("_migration_*.jsonl"))


def test_missing_run_dir_errors_but_others_migrate(tmp_path):
    paths = build_old_tree(tmp_path)
    coh = paths["coh"]

    result = runner.invoke(app, ["--coherence-dir", str(coh), "--apply"])
    assert result.exit_code == 1
    assert "run dir missing" in result.output.lower() or "gone" in result.output.lower()

    # The valid ones migrated despite the error.
    assert (paths["teacher_run"] / "pile_coherence" / "v1.json").exists()
    assert (paths["student_run"] / "pile_coherence" / "v1.json").exists()


def test_no_errors_finalizes_and_reruns_noop(tmp_path):
    paths = build_old_tree(tmp_path)
    coh = paths["coh"]

    # Remove the missing-run-dir adapter's records so the apply is fully clean.
    lines = [json.loads(x) for x in (coh / "results.jsonl").read_text().splitlines()]
    lines = [r for r in lines if "gone" not in (r.get("adapter_path") or "")]
    (coh / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    # Drop the orphan npy so runs/ can be fully emptied.
    (coh / "runs" / "outputs__teachers__gone__ftseed9" / "per_token_loss.npy").unlink()
    (coh / "runs" / "outputs__teachers__gone__ftseed9").rmdir()

    result = runner.invoke(app, ["--coherence-dir", str(coh), "--apply"])
    assert result.exit_code == 0, result.output

    assert not (coh / "results.jsonl").exists()
    assert (coh / "results.jsonl.migrated").exists()
    assert not (coh / "runs").exists()

    # Rerun: no results.jsonl -> no-op, exit 0, no changes.
    before = _snapshot(coh)
    result = runner.invoke(app, ["--coherence-dir", str(coh), "--apply"])
    assert result.exit_code == 0, result.output
    assert "nothing to do" in result.output.lower()
    assert _snapshot(coh) == before


def test_npy_already_moved_skip(tmp_path):
    paths = build_old_tree(tmp_path)
    coh = paths["coh"]

    # Simulate a half-done migration: teacher npy already at destination, source gone.
    dest = paths["teacher_run"] / "pile_coherence"
    dest.mkdir(parents=True)
    _write_npy(dest / "v1_per_token_loss.npy", 2.0)
    (coh / "runs" / "outputs__teachers__dog__ftseed0" / "per_token_loss.npy").unlink()

    result = runner.invoke(app, ["--coherence-dir", str(coh), "--apply"])
    # still nonzero due to the unrelated missing-run-dir adapter.
    assert result.exit_code == 1, result.output

    # JSON written and the pre-existing npy left in place (skip, not error).
    assert (dest / "v1.json").exists()
    assert np.load(dest / "v1_per_token_loss.npy")[0] == 2.0
    log = json.loads(list(coh.glob("_migration_*.jsonl"))[0].read_text().splitlines()[-1])
    assert log  # log streamed
